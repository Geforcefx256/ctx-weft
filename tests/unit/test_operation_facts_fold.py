"""恢复判据能不能只从 capability 事件流折出来（阶段 2 验收）。

账本存在的理由是「崩溃后判得出副作用是否已发生」。本文件逐场景对照两个来源——
只读事件流折出的事实 vs 账本行——证明前者足够，为账本退场提供依据。

`fold_operations` 是候选实现，落地时移进 `core/control/reducers.py`（与
`fold_hitl_snapshot` / `fold_pending_task_recap` 并列，那个文件的职责就是
「把 event 序列重建成 state」）。
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import AsyncIterator
from datetime import datetime
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
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.operations import InMemoryOperationStore

CAP_FOLD_EVENT_TYPES = (EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED)
TC = mint_call_id(anchor="asst_fold", ordinal=0, raw_id="call_1", turn_seq=1)


@dataclasses.dataclass(frozen=True)
class OperationFacts:
    """一次逻辑调用在事件流里留下的痕迹。**折出来的，不存。**"""

    invoked: bool = False        # 有 INVOKED = provider 被调用过（无 = 确定没跑过）
    finished: bool = False
    attempts: tuple[str, ...] = ()
    result: str | None = None    # FINISHED 里那份收敛版——重放直接用
    outcome: str | None = None
    result_length: int | None = None   # 原始长度；> len(result) 即事件里这份被截断
    last_attempt_at: datetime | None = None


def fold_operations(events: list[Event]) -> dict[str, OperationFacts]:
    """按 tool_call_id 折 CAPABILITY_INVOKED / FINISHED。"""
    out: dict[str, OperationFacts] = {}
    for ev in events:
        p = ev.payload or {}
        tcid = p.get("tool_call_id") or ""
        if not tcid:
            continue
        f = out.get(tcid, OperationFacts())
        if ev.type == EventType.CAPABILITY_INVOKED:
            f = dataclasses.replace(
                f, invoked=True, attempts=(*f.attempts, p.get("invocation_id", "")),
                last_attempt_at=ev.timestamp)
        elif ev.type == EventType.CAPABILITY_FINISHED:
            f = dataclasses.replace(
                f, finished=True, result=p.get("result"), outcome=p.get("outcome"),
                result_length=p.get("result_length"))
        out[tcid] = f
    return out


# ── 夹具 ──────────────────────────────────────────────────────────────────────


class _Tool(ToolCapabilityProvider):
    name = "fx"

    def __init__(self, *, content="ok", hang=False, spillable=True) -> None:
        self.calls = 0
        self._content, self._hang, self._spillable = content, hang, spillable

    def _cap(self):
        return ToolCapability(id="fx:act", name="act", description="d",
                              side_effects=True, spillable=self._spillable)

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
    mem, ledger = InMemoryMemoryProvider(), InMemoryOperationStore()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")
    state = LoopState(
        run_id="run_1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE", parent_task_id=None),
        agent=SimpleNamespace(id="a1", template_id="tpl"), scope=scope,
        resolved_model=SimpleNamespace(model="m", account=""), sequence_counter=0)
    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=bus,
                      provider_ctx=pctx, task_manager=_TM(bus))
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    providers = [tool]
    if sink:
        from ctx_weft.providers.capability_results import ResultsCapabilityProvider
        providers.append(ResultsCapabilityProvider())
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=providers, memory=mem, event_bus=bus,
        operation_store=ledger,
        provider_authorizers={"fx": authorizer} if authorizer else None)
    ctx.capability_gateway = gw
    return gw, store, ledger, state, ctx


async def _facts_and_record(store, ledger, ctx):
    evs = await store.read_session_events_of_types("s1", CAP_FOLD_EVENT_TYPES)
    facts = fold_operations(evs).get(TC, OperationFacts())
    return facts, await ledger.get(TC, ctx.provider_ctx)


# ── 等价性 ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_completed_matches_ledger():
    tool = _Tool()
    gw, store, ledger, state, ctx = _harness(tool)
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts, rec = await _facts_and_record(store, ledger, ctx)
    assert (facts.invoked, facts.finished) == (True, True)
    assert facts.attempts == tuple(rec.attempts)
    assert facts.outcome == "success"


@pytest.mark.asyncio
async def test_started_then_process_gone_is_foldable():
    """**最关键的一条**：进程在 provider 执行中消失 → invoked 而未 finished。

    这正是账本 `status=started` 想表达的事实，也是「副作用可能已发生」的唯一判据。
    """
    tool = _Tool(hang=True)
    gw, store, ledger, state, ctx = _harness(tool)
    running = asyncio.create_task(
        gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC))
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    facts, rec = await _facts_and_record(store, ledger, ctx)
    assert tool.calls == 1, "provider 真的动过手了"
    assert (facts.invoked, facts.finished) == (True, False)
    assert rec.status.value == "started", "与账本同一判断"
    assert facts.attempts == tuple(rec.attempts)


@pytest.mark.asyncio
async def test_authz_denied_leaves_no_capability_events():
    """授权拒绝 → 事件流一条都没有 ⟹「没有 INVOKED = provider 从未被调用」硬成立。

    这让账本的 `prepared` 那一档失去存在理由：没跑过就是没有 INVOKED。
    """
    tool = _Tool()
    gw, store, ledger, state, ctx = _harness(tool, authorizer=_Deny())
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts, rec = await _facts_and_record(store, ledger, ctx)
    assert tool.calls == 0
    assert (facts.invoked, facts.finished) == (False, False)
    assert rec is None, "账本同样没有记录"


@pytest.mark.asyncio
async def test_converged_result_rides_in_the_event():
    """超长输出：事件里带的是**收敛版**（进对话的那份），不是被截断的原文。

    原方案写账本的直接动机是「完整结果不得仅依赖审计事件中被截断的 8000 字符文本」。
    收敛之后那份实测 ~2.3K，远在 8000 之内，且自带取回说明——全文在 sink。
    """
    tool = _Tool(content="z" * 50_000 + "TAIL_EVIDENCE")
    gw, store, ledger, state, ctx = _harness(tool, sink=True)
    res = await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts, _ = await _facts_and_record(store, ledger, ctx)
    assert facts.result == str(res.content), "事件里那份 == 进对话那份"
    assert len(facts.result) < 8000, "收敛版放得进事件 payload"
    assert "TAIL_EVIDENCE" in facts.result, "尾部证据在"
    assert facts.result_length == len(facts.result), "未被 8000 截断"


@pytest.mark.asyncio
async def test_unspillable_output_is_truncated_in_the_event():
    """`spillable=False` 的输出**不收敛**，超 8000 就在事件里被截断——已知边界。

    但这类工具全是只读、可重新派生的（`fs__read_file` / `results__read_tool_output` /
    skill 的 `list_files`），正确的恢复动作是**重跑而非重放**，所以截断不构成数据丢失。
    `result_length` 让调用方判得出「事件里这份是截断的」。
    """
    tool = _Tool(content="q" * 30_000, spillable=False)
    gw, store, ledger, state, ctx = _harness(tool, sink=True)
    await gw.invoke("fx__act", {}, state, ctx, tool_call_id=TC)

    facts, _ = await _facts_and_record(store, ledger, ctx)
    assert facts.result_length == 30_000
    assert len(facts.result) < facts.result_length, "被截断了，而 result_length 说得出原长"
