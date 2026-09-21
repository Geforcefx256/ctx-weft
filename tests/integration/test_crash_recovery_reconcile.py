"""崩溃恢复 → reconcile 端到端:mid-tool 重启后重跑未完成的工具,而非裸重发 LLM（spec/07 §6/§9）。

构造一个"崩在工具批次中途"的持久态:事件流有 session+task,memory 里有一条带 dangling tool_call
的 assistant LLM_RESPONSE（无对应 TOOL_RESULT）。recover_agent 后,_resolve 的 dangling 检测应把
initial_step 路由到 reconcile,由 ReconcileStep 经 gateway 重跑该工具(写出真实 TOOL_RESULT),再进 act。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest import mock

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols import (
    MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext,
)
from ctx_weft.protocols.capability import (
    CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider,
)
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import MemoryScope as _MS
from ctx_weft.protocols.memory import MemoryKind as _MK
_MEM_TASK_SCOPE = _MS.TASK
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio


class _RecordingTool(ToolCapabilityProvider):
    name = "test"

    def __init__(self) -> None:
        self.invoked: list[dict] = []

    async def list(self, ctx) -> list[ToolCapability]:
        # wp6（spec: tool-operations）：声明 idempotent —— 崩溃后同 op_id 重跑恰好一次。
        # 默认 reviewed 的行为由 test_tool_outcome_not_rerun.py 钉（作结、不重跑）。
        return [ToolCapability(id="test:web", name="web", description="fetch a page",
                               recovery_policy="idempotent")]

    async def retrieve(self, ctx) -> list[ToolCapability]:
        return await self.list(ctx)        # 让 CapabilityResolver 绑定 web 进 cache

    async def describe(self, ctx) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, cid, args, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run(args)

    async def _run(self, args) -> AsyncIterator[CapabilityEvent]:
        self.invoked.append(args)
        yield CapabilityEvent(kind="result", payload={"content": "PAGE BODY"})

    async def cancel(self, iid, ctx) -> None:
        return None


async def test_crash_mid_tool_reinvokes_dangling_via_reconcile() -> None:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="done"), MockResponse(text="done")])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    mem = InMemoryMemoryProvider()
    runtime.providers.register_memory(mem)
    tool = _RecordingTool()
    runtime.providers.register_capability(tool)

    from ctx_weft.core.utils.ids import mint_call_id
    # tool_call 的 id 是摄入点铸造的内部标识——它同时就是账本键。
    tcid = mint_call_id(anchor="rec_r", ordinal=0, raw_id="tc1", turn_seq=0)
    sid, tid, aid = "ses_r", "tsk_r", "agt_root"
    ts = datetime(2026, 6, 13, tzinfo=timezone.utc)

    def ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=sid,
                     type=type_, timestamp=ts, task_id=task_id, payload=payload)

    seed = [
        ev(1, EventType.SESSION_CREATED, user_prompt="fetch it", template_id="agent:tpl_echo",
           root_agent_id=aid),
        ev(2, EventType.RUN_STARTED),
        ev(3, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid}),
        ev(4, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
    ]
    for e in seed:
        await append_one(runtime.event_store, e)

    # memory:崩在工具批次中途 —— assistant turn 含 dangling tool_call "web",无 TOOL_RESULT。
    scope = MemoryAddress(session_id=sid, task_id=tid, agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id=tid, agent_id=aid)
    await mem.ingest(MemoryEvent(type=MemoryEventType.USER_PROMPT, address=scope, content="fetch it",
        timestamp=ts, role="user", metadata={"task_id": tid}), pctx)
    await mem.ingest(MemoryEvent(type=MemoryEventType.LLM_RESPONSE, address=scope, content="",
        timestamp=ts, role="assistant",
        metadata={"tool_calls": [{"id": tcid, "name": "test__web", "input": {"url": "x"}}]}), pctx)

    # 注入「崩溃前调用过」的事实：一条 CapabilityInvoked、没有 Finished
    # ——这正是账本时代 `status=started` 的等价物，idempotent 分支的输入。
    from ctx_weft.core.utils.clock import now_utc as _now
    from ctx_weft.protocols.events import Event as _Ev, EventType as _ET

    await append_one(runtime.event_store, _Ev(
        id="evt_seed_invoked", type=_ET.CAPABILITY_INVOKED, session_id=sid,
        run_id="r_seed", sequence=1, timestamp=_now(),
        payload={"tool_call_id": tcid, "invocation_id": "inv_first",
                 "capability_name": "test__web", "capability_id": "test:web"}))

    # 装填是调用方的责任（2026-09-21：`recover_agent` 对 registry miss 直接抛
    # `AgentNotLoaded`，按 agent 扫全库的 sweep 已删）。只喂内存，不建 TM、不跑。
    await runtime.rebuild_session(sid)
    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await runtime.recover_agent(aid)
        for _ in range(50):
            if tool.invoked:
                break
            await asyncio.sleep(0.02)

    # reconcile 重跑了 dangling 工具(而非裸重发 LLM)
    assert tool.invoked, "dangling tool 'web' should be re-invoked by ReconcileStep on crash recovery"
    results = await mem.recall_recent(scope=scope, types=[MemoryEventType.TOOL_RESULT], limit=10, ctx=pctx)
    assert any(r.metadata.get("tool_call_id") == tcid and "PAGE BODY" in r.content for r in results), \
        f"expected real TOOL_RESULT for {tcid}; got {[(r.metadata.get('tool_call_id'), r.content) for r in results]}"
