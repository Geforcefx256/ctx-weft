"""委派回填：origin_tool_call_id 透传到 child（spec/06 §5）。

锁定：
- delegate_task / delegate_plan 把发起调用的 tool_call_id 写到每个 child.origin_tool_call_id
- 经 ControlCapabilityProvider 调用时，tool_call_id 经 ProviderContext.extra → ControlContext 透传
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import (
    PROVIDER_NAME,
    ControlCapabilityProvider,
    ControlContext,
    delegate_plan,
    delegate_task,
)
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


class _FakeTM:
    def __init__(self) -> None:
        self.staged: list[Task] = []

    def stage_task(self, child: Task, **kwargs) -> None:
        self.staged.append(child)

    def get_task(self, tid: str):
        return None


def _parent() -> Task:
    return Task(id="p1", session_id="s1", status="ACTIVE", title="P")


def _ctx(tm: _FakeTM, tool_call_id: str) -> ControlContext:
    return ControlContext(
        session_id="s1", task_id="p1", agent_id="a1", task=_parent(),
        task_manager=tm, session=None, tool_call_id=tool_call_id,
    )


def test_delegate_task_records_origin_tool_call_id() -> None:
    tm = _FakeTM()
    delegate_task(title="child", task_prompt="do it", ctx=_ctx(tm, "tc_42"))
    assert len(tm.staged) == 1
    assert tm.staged[0].origin_tool_call_id == "tc_42"
    assert tm.staged[0].parent_task_id == "p1"


def test_delegate_plan_records_distinct_origin_per_child() -> None:
    tm = _FakeTM()
    delegate_plan(tasks=[{"title": "a"}, {"title": "b"}], ctx=_ctx(tm, "tc_99"))
    ids = [c.origin_tool_call_id for c in tm.staged]
    assert len(set(ids)) == 2, f"each plan child needs its own origin_tool_call_id, got {ids}"
    assert "tc_99" not in ids, "plan children must NOT reuse the delegate_plan call id"


@pytest.mark.asyncio
async def test_tool_call_id_threads_through_provider_extra() -> None:
    """gateway 把 tool_call_id 放进 ProviderContext.extra → _handle → ControlContext。"""
    tm = _FakeTM()
    parent = _parent()
    tm.get_task = lambda tid: parent  # type: ignore[assignment]
    session = Session(id="s1", tenant_id="default", user_prompt="x", status="RUNNING")
    provider = ControlCapabilityProvider()
    provider.register_session("s1", tm, session)

    ctx = ProviderContext(
        session_id="s1", tenant_id="default", task_id="p1", agent_id="a1",
        extra={"tool_call_id": "tc_thread"},
    )
    async for _ev in provider.invoke(f"{PROVIDER_NAME}:delegate_task", {"title": "c", "task_prompt": "p"}, ctx):
        pass

    assert len(tm.staged) == 1
    assert tm.staged[0].origin_tool_call_id == "tc_thread"


# ── gateway 派发路由：TASK_DISPATCH(agent 层) 写入、即时 TOOL_RESULT 暂挂 ──────────────


class _DispatchProvider(ToolCapabilityProvider):
    name = "control"

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="control:delegate_task", name="delegate_task", description="dispatch")

    async def list(self, ctx):
        return [self._cap()]

    async def retrieve(self, ctx):
        return [self._cap()]

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            yield CapabilityEvent(kind="result", payload={"content": "Sub-task scheduled."})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None:
        return None


class _PlanDispatchProvider(ToolCapabilityProvider):
    name = "control"

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="control:delegate_plan", name="delegate_plan", description="plan")

    async def list(self, ctx):
        return [self._cap()]

    async def retrieve(self, ctx):
        return [self._cap()]

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            yield CapabilityEvent(kind="result", payload={"content": "Plan scheduled."})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None:
        return None


@pytest.mark.asyncio
async def test_gateway_delegate_plan_still_eager_writes_envelope() -> None:
    """delegate_plan 的 envelope 框 + 配对 ack 仍由 gateway eager 写（per-child 框才走 finalize 铸）——
    2026-07-03 只移除了 delegate_task 的 eager 写，plan envelope 不受影响。"""
    from ctx_weft.core.capabilities.control_tools import _PLAN_DISPATCH_ACK
    mem = InMemoryMemoryProvider()
    cache = CapabilityCache()
    cap = ToolCapability(id="control:delegate_plan", name="delegate_plan", description="plan")
    cache.put("agt_1", [cap])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[_PlanDispatchProvider()],
        memory=mem, event_bus=InProcessEventBus(),
    )
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="run_1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1", parent_task_id=None),
        agent=SimpleNamespace(id="agt_1", template_id="t"), scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )

    await gw.invoke("control__delegate_plan", {"tasks": []}, state, ctx, tool_call_id="tc_plan")

    pctx = ProviderContext(session_id="s1", tenant_id="default")
    turns = await mem.recall_recent(scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 10, pctx)
    frame = [r for r in turns if r.role == "assistant"
             and any(tc.get("id") == "tc_plan" for tc in (r.metadata.get("tool_calls") or []))]
    ack = [r for r in turns if r.role == "tool" and r.metadata.get("tool_call_id") == "tc_plan"]
    assert len(frame) == 1, "plan envelope 框须 eager 写"
    assert ack and ack[0].content == _PLAN_DISPATCH_ACK, "plan envelope 配对 ack 须 eager 写"


@pytest.mark.asyncio
async def test_gateway_delegate_task_defers_frame_to_finalize() -> None:
    """§2.3(2026-07-03 修订)：gateway 对 delegate_task **不再 eager 写框**——框由 child finalize 铸、
    与 result 同锚 task.started_at。故 gateway invoke 后 parent scope 无任何 memory 写入
    （无 AGENT_CONVERSATION_TURN 框、无 TASK_DISPATCH、无即时 TOOL_RESULT）。"""
    mem = InMemoryMemoryProvider()
    cache = CapabilityCache()
    cap = ToolCapability(id="control:delegate_task", name="delegate_task", description="dispatch")
    cache.put("agt_1", [cap])
    gw = CapabilityGateway(
        capability_cache=cache,
        capability_providers=[_DispatchProvider()],
        memory=mem,
        event_bus=InProcessEventBus(),
    )
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="run_1",
        session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1", parent_task_id=None),
        agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )

    await gw.invoke("control__delegate_task", {"title": "c"}, state, ctx, tool_call_id="tc_d")

    pctx = ProviderContext(session_id="s1", tenant_id="default")
    # delegate_task 不 eager 写框（改由 finalize 铸）；也不写 legacy TASK_DISPATCH / 即时 TOOL_RESULT
    assert await mem.recall_recent(scope, [MemoryEventType.AGENT_CONVERSATION_TURN], 10, pctx) == []
    assert await mem.recall_recent(scope, [MemoryEventType.TASK_DISPATCH], 10, pctx) == []
    assert await mem.recall_recent(scope, [MemoryEventType.TOOL_RESULT], 10, pctx) == []


class _AssessProvider(ToolCapabilityProvider):
    name = "control"

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="control:report_task_outcome",
                              name="report_task_outcome", description="verdict")

    async def list(self, ctx):
        return [self._cap()]

    async def retrieve(self, ctx):
        return [self._cap()]

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            yield CapabilityEvent(kind="result", payload={"content": "human reply"})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None:
        return None


@pytest.mark.asyncio
async def test_delegate_plan_returns_envelope_ack() -> None:
    from ctx_weft.core.capabilities.control_tools import delegate_plan, _PLAN_DISPATCH_ACK
    tm = _FakeTM()
    res = delegate_plan(tasks=[{"title": "a"}, {"title": "b"}], ctx=_ctx(tm, "tc_plan"))
    # spec: task-handoff——ack 前缀保持信封语义，尾部追加与 spec 顺序对应的 id 列表
    assert res.content.startswith(_PLAN_DISPATCH_ACK.rstrip("via start_task.").rstrip())
    ids = [c.id for c in tm.staged]
    assert ", ".join(ids) in res.content


@pytest.mark.asyncio
async def test_gateway_silent_tool_writes_nothing_to_task_layer() -> None:
    """report_task_outcome（编排/裁决）不入 task 对话——其 HITL 回复改由 finalize 注入。"""
    mem = InMemoryMemoryProvider()
    cache = CapabilityCache()
    cap = ToolCapability(id="control:report_task_outcome",
                         name="report_task_outcome", description="verdict")
    cache.put("agt_1", [cap])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[_AssessProvider()],
        memory=mem, event_bus=InProcessEventBus(),
    )
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="r1",
        session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1"),
        agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )

    await gw.invoke("control__report_task_outcome", {"x": 1}, state, ctx, tool_call_id="tc_a")

    pctx = ProviderContext(session_id="s1", tenant_id="default")
    inv = await mem.recall_recent(scope, [MemoryEventType.TOOL_INVOCATION], 10, pctx)
    res = await mem.recall_recent(scope, [MemoryEventType.TOOL_RESULT], 10, pctx)
    assert inv == [] and res == []  # task 对话零污染
