"""崩溃恢复：task 已开始、但提问没进 memory——要从事件日志把它补回来。

新 task 的提问在第一轮算数之前只暂存在未提交窗口里（spec 2026-09-09）。提交时
`TaskStarted` 等事件先落盘、暂存的提问后落盘，崩在两者之间就是这里构造的形状：日志说
task 是 ACTIVE，memory 里没有提问。`task_from_projection` 按状态推断「提问已在 memory」，
不核对的话 driver 不会再写它，这个 task 从此缺了原始提问。

提问在 `TaskCreated` 的 payload 里，恢复时还原得出来——核对后补写即可。反过来，提问已经
在 memory 里的（常规情形）不得被写第二遍。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest import mock

import pytest

from ctx_weft.core.utils.content import content_to_text
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.llm.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _ActRouterLLM, _all_request_text, _poll
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio

PROMPT = "summarize the quarterly report"


@pytest.mark.parametrize("prompt_in_memory", [False, True], ids=["missing", "present"])
async def test_recovery_restores_a_started_tasks_prompt_from_the_event_log(prompt_in_memory) -> None:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    mem = InMemoryMemoryProvider()
    runtime.providers.register_memory(mem)

    sid, tid, aid = "ses_p", "tsk_p", "agt_root"
    ts = datetime(2026, 9, 17, tzinfo=timezone.utc)

    def ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=sid,
                     type=type_, timestamp=ts, task_id=task_id, payload=payload)

    for e in [
        ev(1, EventType.SESSION_CREATED, user_prompt=PROMPT, template_id="agent:tpl_echo",
           root_agent_id=aid),
        ev(2, EventType.RUN_STARTED),
        ev(3, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid, "user_prompt": PROMPT}),
        ev(4, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
    ]:
        await runtime.event_store.append(e)

    scope = MemoryAddress(session_id=sid, task_id=tid, agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id=tid, agent_id=aid)
    if prompt_in_memory:
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
            content=PROMPT, timestamp=ts, role="user", metadata={"task_id": tid}), pctx)

    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await runtime.recover_agent(aid)
        await _poll(lambda: llm.act_requests or None)
        await asyncio.sleep(0.2)

    assert PROMPT in _all_request_text(llm.act_requests[0]), (
        "恢复后第一轮 LLM 请求必须带着原始提问")
    view = await mem.load_view(scope, MemoryScope.TASK, pctx, kinds=[MemoryKind.CONVERSATION_TURN])
    prompts = [r for r in view if r.role == "user" and PROMPT in content_to_text(r.content)]
    assert len(prompts) == 1, (
        f"提问在 memory 里必须恰好一条（缺了要补、有了不得重复）：{[r.content for r in prompts]}")
