"""capability 事件流的两条完整性不变式（spec: tool-operations / event-commit）。

事件流是宿主可见的审计面，也是恢复判据的候选来源。两条：

1. **不留孤立的 INVOKED** —— 发了「调用了」就必须真的调用，或者压根别发。
2. **调了工具就关窗** —— 未提交窗口的前提是「还什么不可逆的事都没发生」；工具一旦
   跑过，`discard_round` 再说「当没发生过」就是撒谎。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.utils.ids import mint_call_id
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    Authorizer,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

CAP_TYPES = (EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED)
TC = mint_call_id(anchor="asst_ei", ordinal=0, raw_id="call_1", turn_seq=1)


class _Tool(ToolCapabilityProvider):
    name = "fx"

    def __init__(self) -> None:
        self.calls = 0

    def _cap(self):
        return ToolCapability(id="fx:act", name="act", description="d", side_effects=True)

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)
    async def cancel(self, i, ctx): return None

    def invoke(self, cid, args, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _r():
            self.calls += 1
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return _r()


class _CountingAllow(Authorizer):
    def __init__(self) -> None:
        self.calls = 0

    async def authorize(self, capability, ctx, arguments=None, *, tool_call_id=""):
        self.calls += 1
        return AuthorizationDecision(allowed=True)


class _TM:
    """与真 TaskManager 同口径的最小桩（manager.py: commit_round → commit_provisional）。"""

    def __init__(self, bus) -> None:
        self._bus, self.committed = bus, []

    async def commit_round(self, task_id):
        self.committed.append(task_id)
        await self._bus.commit_provisional(task_id)


def _harness():
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))        # required 语义
    mem, tool = InMemoryMemoryProvider(), _Tool()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")
    state = LoopState(
        run_id="run_1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE", parent_task_id=None),
        agent=SimpleNamespace(id="a1", template_id="tpl"), scope=scope,
        resolved_model=SimpleNamespace(model="m", account=""), sequence_counter=0)
    authz = _CountingAllow()
    async def _read(session_id, types):
        return await store.read_session_events_of_types(session_id, types)

    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=bus,
                      provider_ctx=pctx, task_manager=_TM(bus),
                      read_events_of_types=_read)
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[tool], memory=mem, event_bus=bus,
 provider_authorizers={"fx": authz})
    ctx.capability_gateway = gw
    return gw, store, tool, authz, state, ctx


@pytest.mark.asyncio
async def test_completed_reentry_leaves_no_orphan_invoked():
    """同逻辑调用重入：短路必须发生在发 INVOKED **之前**。

    放在之后（改造前如此）会先发一条 `CapabilityInvoked`，再从短路 return——不发
    `CapabilityFinished`，于是事件流里留下一条孤立的「调用了」而 provider 根本没被
    调用。宿主按 Invoked/Finished 配对做 UI 会看到一次永远悬着的调用。
    """
    gw, store, tool, authz, state, ctx = _harness()
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)
    # 重入由调用侧告知（reconcile / HITL 冷续跑都知道自己是重入）——热路径一次调用
    # 只发生一次，不该为重入白折一遍事件流。
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC, reentry=True)

    evs = await store.read_session_events_of_types("s1", CAP_TYPES)
    kinds = [e.type for e in evs]
    assert tool.calls == 1, "重入不得再打 provider"
    assert kinds == [EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED], kinds
    # 连带：重放不再走事前授权（重放什么都不执行，而 authorize 问的是「能不能跑」）。
    assert authz.calls == 1, "重放不该再问一次事前授权"


@pytest.mark.asyncio
async def test_invoking_a_tool_closes_the_provisional_window():
    """调了工具就关窗——此后 discard 不能把这次调用从日志里抹掉。

    `begin_round` 的三个调用点里两个是消息注入 / HITL 冷续跑，而 ReconcileStep 跑在
    任何 LLM chunk 之前：它的 invoke 落在开着的窗口里。不关窗的话 `discard_round`
    之后事件流会否认一次已经发生了副作用的调用存在过。
    """
    gw, store, tool, authz, state, ctx = _harness()
    ctx.event_bus.begin_provisional("t1")

    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)
    assert tool.calls == 1, "副作用确实发生了"
    assert ctx.task_manager.committed == ["t1"], "调用工具应触发关窗"

    ctx.event_bus.discard_provisional("t1")
    evs = await store.read_session_events_of_types("s1", CAP_TYPES)
    assert [e.type for e in evs] == [
        EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED], \
        "已发生的副作用不得被 discard 抹掉"


@pytest.mark.asyncio
async def test_invoked_is_durable_before_the_provider_runs():
    """`CapabilityInvoked` 在 provider 真正被调用**之前**已获存储确认。

    这是「崩溃后能判出副作用是否已发生」的全部依据——提交门让 emit 拿到确认才返回。
    做法：工具在自己的 invoke() 里回查事件库找自己的 INVOKED。
    """
    gw, store, tool, authz, state, ctx = _harness()
    seen: list[bool] = []

    def _probing_invoke(cid, args, pctx) -> AsyncIterator[CapabilityEvent]:
        async def _r():
            evs = await store.read_session_events_of_types("s1", CAP_TYPES)
            seen.append(any(e.type == EventType.CAPABILITY_INVOKED for e in evs))
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return _r()

    tool.invoke = _probing_invoke
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)
    assert seen == [True], "provider 执行时自己的 INVOKED 必须已经落库"
