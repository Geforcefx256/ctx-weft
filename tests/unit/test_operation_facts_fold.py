"""恢复判据：capability 事件流的折叠（spec: tool-operations）。

一次工具调用的执行记录**就是它在事件流里留下的痕迹**。本文件钉住折叠的语义——
它取代了此前那张 `operations` 表（`OperationStore` / CAS revision）。对照关系：

    账本 prepared      → 没有 CapabilityInvoked（授权拒绝时事件流一条都没有）
    账本 started       → 有 INVOKED、无 FINISHED
    账本 completed     → 有 FINISHED，payload 里那份正是进对话的收敛版
    账本 waiting_human → HITL 自己的账（registry 未决请求 + HitlOpened）
    账本 revision CAS  → 唯一消费方（宿主并发处置 API）已删，无并发写者
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.control.reducers import (
    CAP_FOLD_EVENT_TYPES,
    OperationFacts,
    fold_operations,
)
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

TC = mint_call_id(anchor="asst_fold", ordinal=0, raw_id="call_1", turn_seq=1)


# ── 夹具 ──────────────────────────────────────────────────────────────────────


class _Tool(ToolCapabilityProvider):
    name = "fx"

    def __init__(self, *, content="ok", hang=False, spillable=True,
                 policy="reviewed") -> None:
        self.calls = 0
        self._content, self._hang = content, hang
        self._spillable, self._policy = spillable, policy

    def _cap(self):
        return ToolCapability(id="fx:act", name="act", description="d",
                              side_effects=True, spillable=self._spillable,
                              recovery_policy=self._policy)

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)
    async def cancel(self, i, ctx): return None

    def invoke(self, cid, args, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _r():
            self.calls += 1
            if self._hang:
                await asyncio.sleep(30)     # 进程在这一刻消失：FINISHED 永不发出
            yield CapabilityEvent(kind="result", payload={"content": self._content})
        return _r()


class _Deny(Authorizer):
    async def authorize(self, capability, ctx, arguments=None, *, tool_call_id=""):
        return AuthorizationDecision(allowed=False, message="policy: blocked")


class _TM:
    """与真 TaskManager 同口径的最小桩（commit_round → commit_provisional）。"""

    def __init__(self, bus) -> None:
        self._bus = bus

    async def commit_round(self, task_id) -> None:
        await self._bus.commit_provisional(task_id)


def _harness(tool, *, authorizer=None, sink=False):
    store, bus = InMemoryEventStore(), InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")
    state = LoopState(
        run_id="run_1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE", parent_task_id=None),
        agent=SimpleNamespace(id="a1", template_id="tpl"), scope=scope,
        resolved_model=SimpleNamespace(model="m", account=""), sequence_counter=0)


    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=bus,
                      provider_ctx=pctx, task_manager=_TM(bus),
                      event_store=store)
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    providers = [tool]
    if sink:
        from ctx_weft.providers.capability_results import ResultsCapabilityProvider
        providers.append(ResultsCapabilityProvider())
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=providers, memory=mem, event_bus=bus,
        provider_authorizers={"fx": authorizer} if authorizer else None)
    ctx.capability_gateway = gw
    return gw, store, state, ctx


async def _facts(store) -> OperationFacts:
    evs = await store.read_session_events_of_types("s1", CAP_FOLD_EVENT_TYPES)
    return fold_operations(evs).get(TC, OperationFacts())


# ── 状态机的四档，逐个折出来 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finished_call_folds_to_invoked_and_finished():
    tool = _Tool()
    gw, store, state, ctx = _harness(tool)
    res = await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts = await _facts(store)
    assert (facts.invoked, facts.finished) == (True, True)
    assert len(facts.attempts) == 1
    assert facts.outcome == "success"
    assert facts.result == str(res.content), "事件里那份 == 进对话那份"


@pytest.mark.asyncio
async def test_started_then_process_gone_folds_to_invoked_not_finished():
    """**最关键的一条**：进程在 provider 执行中消失 → invoked 而未 finished。

    这正是账本 `status=started` 曾经表达的事实，也是「副作用可能已发生」的唯一判据。
    `CapabilityInvoked` 经提交门 emit——调 provider 之前就已获存储确认，所以这条判据
    不依赖任何额外的持久化。
    """
    tool = _Tool(hang=True)
    gw, store, state, ctx = _harness(tool)
    running = asyncio.create_task(gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC))
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    facts = await _facts(store)
    assert tool.calls == 1, "provider 真的动过手了"
    assert (facts.invoked, facts.finished) == (True, False)
    assert len(facts.attempts) == 1


@pytest.mark.asyncio
async def test_authz_denied_leaves_no_capability_events():
    """授权拒绝 → 事件流一条都没有 ⟹「没有 INVOKED = provider 从未被调用」硬成立。

    `_record_invocation` 排在授权与参数校验之后、provider 之前，所以拒绝与校验失败都
    不会留下 INVOKED。账本的 `prepared` 那一档因此失去存在理由。
    """
    tool = _Tool()
    gw, store, state, ctx = _harness(tool, authorizer=_Deny())
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts = await _facts(store)
    assert tool.calls == 0
    assert (facts.invoked, facts.finished) == (False, False)


@pytest.mark.asyncio
async def test_converged_result_rides_in_the_event():
    """超长输出：事件里带的是**收敛版**（进对话的那份），不是被截断的原文。

    原方案写账本的直接动机是「完整结果不得仅依赖审计事件中被截断的 8000 字符文本」。
    收敛之后那份实测 ~2.3K，远在 8000 之内，且自带取回说明——全文在 sink。
    """
    tool = _Tool(content="z" * 50_000 + "TAIL_EVIDENCE")
    gw, store, state, ctx = _harness(tool, sink=True)
    res = await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts = await _facts(store)
    assert facts.result == str(res.content), "事件里那份 == 进对话那份"
    assert len(facts.result) < 8000, "收敛版放得进事件 payload"
    assert "TAIL_EVIDENCE" in facts.result, "尾部证据在"
    assert not facts.truncated


@pytest.mark.asyncio
async def test_unspillable_output_is_marked_truncated():
    """`spillable=False` 的输出**不收敛**，超 8000 会在事件里被截断——已知边界。

    这类工具全是只读、可重新派生的（`fs__read_file` / `results__read_tool_output` /
    skill 的 `list_files`，清一色 `side_effects=False` 且声明 `idempotent`），正确的恢复
    动作是**重跑而非重放**。`truncated` 让 gateway 判得出「事件里这份不能拿来重放」。
    """
    tool = _Tool(content="q" * 30_000, spillable=False)
    gw, store, state, ctx = _harness(tool, sink=True)
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts = await _facts(store)
    assert facts.result_length == 30_000
    assert facts.truncated, "事件里这份是截断的，重放要据此避开"


@pytest.mark.asyncio
async def test_two_attempts_fold_in_order():
    """一次逻辑调用可以有多次执行尝试；attempts 按事件序折出来。

    用 `idempotent`：`reviewed` 在重入时会交给重跑授权，没注册就作结不重跑——那是
    另一条路（见 `test_operation_recovery_policy.py`），这里要的是真的跑第二次。
    """
    tool = _Tool(hang=True, policy="idempotent")
    gw, store, state, ctx = _harness(tool)
    for _ in range(2):
        running = asyncio.create_task(
            gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC, reentry=True))
        await asyncio.sleep(0.05)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    facts = await _facts(store)
    assert tool.calls == 2
    assert len(facts.attempts) == 2
    assert len(set(facts.attempts)) == 2, "每次尝试有自己的 invocation_id"


def test_bare_wire_ids_are_folded_but_not_a_judgement_key():
    """裸 wire id 同样进折叠结果，但跨回合会互相覆盖——调用方按 `is_internal_call_id` 取舍。"""
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.protocols.events import Event

    def _ev(t, tcid, inv):
        return Event(id=f"evt_{inv}", type=t, session_id="s1", run_id="run_1",
                     sequence=0, timestamp=now_utc(),
                     payload={"tool_call_id": tcid, "invocation_id": inv})

    folded = fold_operations([
        _ev(EventType.CAPABILITY_INVOKED, "call_1", "inv_a"),
        _ev(EventType.CAPABILITY_FINISHED, "call_1", "inv_a"),
        _ev(EventType.CAPABILITY_INVOKED, "call_1", "inv_b"),   # 另一个回合，同 wire id
    ])
    assert folded["call_1"].attempts == ("inv_a", "inv_b"), "两个回合被折成了一条——歧义"
