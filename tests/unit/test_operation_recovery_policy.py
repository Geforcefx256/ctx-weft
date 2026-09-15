"""恢复策略表与重跑授权链（spec: tool-operations）。

组件级：真 reconcile + 真 gateway + 真事件库（内存）+ 提交门，policy 经 capability 声明。

判据来自**事件流**：`CapabilityInvoked` 在 provider 之前经提交门落库，所以
「有没有 INVOKED」就是「provider 有没有被调用过」。夹具因此靠**播事件**而不是写账本
来构造崩溃前的状态——那张 `operations` 表已经不存在了。

分派矩阵：reviewed 交重跑授权不自行重跑 / idempotent 同 id 恰一次 / 折不出事实时作结
「无从查证」/ call_1 复用串扰根治 / 重跑授权两分支（放行 · 作结）+ 未注册的默认形态。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.loop.capability_gateway import CapabilityGateway, tool_result_record_id
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.steps.reconcile import ReconcileStep
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import mint_call_id
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    CapabilityEvent,
    CapabilityProviderInfo,
    RerunAuthorizer,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.memory import MemoryEvent
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

#: 摄入点铸造的内部标识——既是消息里的 tool_call id，也是事件 payload 的配对键。
OP = mint_call_id(anchor="rec_fx", ordinal=0, raw_id="call_1", turn_seq=0)


class _EffectTool(ToolCapabilityProvider):
    name = "fx"

    def __init__(self, policy: str = "reviewed") -> None:
        self.policy = policy
        self.executions = 0

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="fx:act", name="act", description="d",
                              side_effects=True, recovery_policy=self.policy)

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)

    def invoke(self, cid, args, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _r():
            self.executions += 1
            yield CapabilityEvent(kind="result", payload={"content": f"eff{self.executions}"})
        return _r()

    async def cancel(self, i, ctx): return None


class _TM:
    def __init__(self, bus) -> None:
        self._bus = bus

    async def commit_round(self, task_id) -> None:
        await self._bus.commit_provisional(task_id)


async def _seed(store, *, tool_call_id: str, invoked: bool, finished: bool = False,
                invocation_id: str = "inv_first", result: str = "old result") -> None:
    """播出「崩溃前」的 capability 事件——夹具构造状态的唯一手段。

    `invoked and not finished` = 账本时代的 `status=started`；两个都 False = 没跑过。
    """
    from ctx_weft.protocols.events import Event

    seq = 0

    def _ev(etype, payload):
        nonlocal seq
        seq += 1
        return Event(id=f"evt_seed_{seq}", type=etype, session_id="s1", run_id="r0",
                     sequence=seq, timestamp=now_utc(), payload=payload)

    batch = []
    if invoked:
        batch.append(_ev(EventType.CAPABILITY_INVOKED, {
            "tool_call_id": tool_call_id, "invocation_id": invocation_id,
            "capability_name": "fx__act", "capability_id": "fx:act"}))
    if finished:
        batch.append(_ev(EventType.CAPABILITY_FINISHED, {
            "tool_call_id": tool_call_id, "invocation_id": invocation_id,
            "capability_name": "fx__act", "outcome": "success",
            "result": result, "result_length": len(result)}))
    for ev in batch:
        await store.append(ev)


async def _mk_fixture(policy="reviewed", *, invoked=True, finished=False, tool=None):
    tool = tool if tool is not None else _EffectTool(policy)
    mem = InMemoryMemoryProvider()
    event_store, bus = InMemoryEventStore(), InProcessEventBus()
    bus.attach_commit_gate(CommitGate(event_store))
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")

    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE"),
        agent=SimpleNamespace(id="a1", template_id="t"),
        scope=scope, resolved_model=SimpleNamespace(model="m", account=""),
        sequence_counter=0,
    )

    async def _read(session_id, types):
        return await event_store.read_session_events_of_types(session_id, types)

    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=bus,
                      provider_ctx=pctx, task_manager=_TM(bus),
                      read_events_of_types=_read)
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    gw = CapabilityGateway(capability_cache=cache, capability_providers=[tool],
                           memory=mem, event_bus=bus)
    ctx.capability_gateway = gw

    await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="", timestamp=now_utc(), role="assistant",
        metadata={"tool_calls": [{"id": OP, "name": "fx__act", "input": {"n": 1}}]}),
        pctx)

    await _seed(event_store, tool_call_id=OP, invoked=invoked, finished=finished)
    return tool, event_store, bus, state, ctx, OP


async def _tool_records(ctx, state):
    recs = await ctx.memory.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    return [r for r in recs if r.role == "tool"]


# ── 分派矩阵 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reviewed_started_concludes_without_rerun():
    """H3 的核心性质：不重跑。但**不停机**——作结写成工具结果，循环继续。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    outcome = await ReconcileStep().execute(state, ctx)

    assert tool.executions == 0, "reviewed + 已调用未完成 MUST NOT re-execute"
    assert outcome.next_step == "prepare", "不确定是工具结果，不是控制流——不停机"
    recs = await _tool_records(ctx, state)
    assert len(recs) == 1 and "unverified" in str(recs[0].content)


@pytest.mark.asyncio
async def test_idempotent_started_reruns_exactly_once():
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent")
    outcome = await ReconcileStep().execute(state, ctx)

    assert tool.executions == 1
    assert outcome.next_step == "prepare"
    evs = await store.read_session_events_of_types(
        "s1", (EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED))
    assert sum(1 for e in evs if e.type == EventType.CAPABILITY_INVOKED) == 2, \
        "播的那条 + 重跑这条"


@pytest.mark.asyncio
async def test_never_invoked_is_first_execution():
    """折得出事实而里面没有 INVOKED = **确定还没跑过**，首执安全（即使 reviewed）。

    比账本时代的判据更硬：那时要先判账本跨不跨进程（内存账本重启后一片空白，「没跑过」
    与「跑过但记录没了」长得一样）；事件库在 required 模式下本来就必须持久。
    """
    tool, store, bus, state, ctx, op_id = await _mk_fixture(
        policy="reviewed", invoked=False)
    outcome = await ReconcileStep().execute(state, ctx)

    assert tool.executions == 1, "确定未启动 → 首执，即使策略是 reviewed"
    assert outcome.next_step == "prepare"


@pytest.mark.asyncio
async def test_no_event_query_at_all_concludes_without_rerun():
    """折不出事实（宿主直构 / 测试替身没接查询）——无从判断，不重跑。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(
        policy="idempotent", invoked=False)
    ctx.read_events_of_types = None
    outcome = await ReconcileStep().execute(state, ctx)

    assert tool.executions == 0, "无从判断的副作用工具不得自动重跑"
    assert outcome.next_step == "prepare"


@pytest.mark.asyncio
async def test_finished_is_not_dangling():
    """已完成 → reconcile 跳过，并从事件里那份补写 memory（不再收敛一遍）。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(
        policy="reviewed", invoked=True, finished=True)
    outcome = await ReconcileStep().execute(state, ctx)

    assert tool.executions == 0
    assert outcome.next_step == "prepare"
    recs = await _tool_records(ctx, state)
    assert len(recs) == 1
    assert recs[0].id == tool_result_record_id(op_id), "确定性记录 id"
    assert recs[0].content == "old result", "照抄事件里那份"


@pytest.mark.asyncio
async def test_call1_reuse_no_cross_talk():
    """call_1 复用串扰根治：旧 tool 记录的 wire id 不使新调用被误判完成。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent")
    from ctx_weft.core.loop.steps.reconcile import _dangling_tool_calls

    # 上一回合的 tool 记录（wire id 同为 call_1，但属于别的 record）
    await ctx.memory.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=state.scope, content="old result", timestamp=now_utc(),
        role="tool", metadata={"tool_call_id": "call_1"}), ctx.provider_ctx)

    dangling, _ = await _dangling_tool_calls(ctx.memory, state.scope, ctx.provider_ctx)
    assert len(dangling) == 1, "按内部标识判定：复用的 call_1 不得让新调用误判为已完成"
    await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1


# ── 两值策略与重跑授权链 ──────────────────────────────────────────────────────


def test_policy_values_validated_not_silently_defaulted():
    from ctx_weft.protocols.capability import normalize_recovery_policy as N

    for bad in ("retry_safe", "queryable", "manual", "idempotant", "Idempotent", "auto"):
        with pytest.raises(ValueError):
            N(bad)


def test_only_two_policies_exist():
    """策略面就是两个值——core 一视同仁的分类不该分成多个值。"""
    from ctx_weft.protocols.capability import RecoveryPolicy as P

    assert [p.value for p in P] == ["idempotent", "reviewed"]


class _Rerun(RerunAuthorizer):
    """重跑授权由**宿主注册**，不由 provider 实现——它甚至不必认识那个 provider。"""

    def __init__(self, verdict) -> None:
        self.verdict = verdict
        self.asked = 0
        self.seen: object = None

    async def authorize_rerun(self, capability, context, ctx, arguments=None, *,
                              tool_call_id=""):
        self.asked += 1
        self.seen = context
        return self.verdict


def _with_rerun(ctx, authorizer, key="fx"):
    """把重跑授权挂到 gateway（等价于 registry 的 rerun_authorizer= 注册位）。"""
    ctx.capability_gateway._rerun_authorizers[key] = authorizer
    return authorizer


@pytest.mark.asyncio
async def test_rerun_denied_concludes_without_executing():
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    rr = _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="外部已完成：流水号 TX-9981")))

    await ReconcileStep().execute(state, ctx)
    assert rr.asked == 1, "reviewed 必须先问重跑授权"
    assert tool.executions == 0, "allowed=False → 绝不执行"
    recs = await _tool_records(ctx, state)
    assert "TX-9981" in str(recs[0].content)


@pytest.mark.asyncio
async def test_rerun_allowed_reexecutes():
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    rr = _with_rerun(ctx, _Rerun(AuthorizationDecision(allowed=True)))

    await ReconcileStep().execute(state, ctx)
    assert rr.asked == 1
    assert tool.executions == 1, "权威判定没跑成 → 重跑"


@pytest.mark.asyncio
async def test_rerun_says_it_cannot_tell():
    """「我不知道」用文本表达——core 一视同仁：不重跑、写结果、继续。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False,
        message="查不到这笔交易的记录；网关侧索引可能延迟，副作用可能已发生。不要重试。")))

    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 0, "判不了就不许跑"
    assert outcome.next_step == "prepare", "不停机，交给 agent"
    recs = await _tool_records(ctx, state)
    assert "不要重试" in str(recs[0].content), "授权方的措辞原样成为工具结果"


@pytest.mark.asyncio
async def test_rerun_context_carries_the_attempts_not_the_credentials():
    """`RerunContext` 只给身份与尝试记录——**不给执行参数**。

    早先版本递过 `effective_args`（未脱敏、含凭据），理由是「要按参数去外部查」。撤销
    了：重跑授权由宿主注册、与 provider 同属宿主，而 provider 为了事后查得到本来就必须
    在执行时把 `ctx.extra["tool_call_id"]` 当幂等键存进自己的系统。
    """
    from ctx_weft.protocols.capability import RerunContext

    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    rr = _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="查过了")))
    await ReconcileStep().execute(state, ctx)

    assert isinstance(rr.seen, RerunContext)
    assert rr.seen.tool_call_id == op_id
    assert rr.seen.attempts == ("inv_first",)
    assert not hasattr(rr.seen, "effective_args"), "核心不替宿主保管凭据明文"


@pytest.mark.asyncio
async def test_rerun_authorizer_is_per_tool_not_per_provider_object():
    """逐工具粒度：给自己不控制的 provider 挂，同 provider 其它工具不受影响。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(allowed=True)), key="fx:act")
    await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1, "精确 capability_id 优先于 provider 前缀"


@pytest.mark.asyncio
async def test_human_decision_default_mapping():
    """基类自带的默认：人批准 = 重跑，人拒绝 = 用人写的那句话作结。"""
    from ctx_weft.protocols.capability import RerunContext
    from ctx_weft.protocols.hitl import HITL_OUTCOME_ACCEPTED, HitlDecision

    rr = _Rerun(AuthorizationDecision(allowed=True))
    cap = ToolCapability(id="fx:act", name="act", description="d")
    rc = RerunContext(tool_call_id="tc", attempts=())
    approved = await rr.on_human_decision(
        cap, rc, None, None, "tc", HitlDecision(outcome=HITL_OUTCOME_ACCEPTED))
    assert approved.allowed is True
    rejected = await rr.on_human_decision(
        cap, rc, None, None, "tc",
        HitlDecision(outcome="rejected", message="别重试，我手工处理了"))
    assert rejected.allowed is False
    assert rejected.result_is_error is False, "人工作结不是错误"
    assert "手工处理" in str(rejected.message)


@pytest.mark.asyncio
async def test_reviewed_without_rerun_authorizer_concludes_not_errors():
    """没注册重跑授权是**默认形态**而非配置错误：core 代为作结「无从查证」。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    outcome = await ReconcileStep().execute(state, ctx)

    assert tool.executions == 0
    assert outcome.next_step == "prepare"
    recs = await _tool_records(ctx, state)
    assert "unverified" in str(recs[0].content)


def test_no_uncertainty_control_plane():
    """「不确定」不再有专属控制面：错误码 / 事件类型 / 宿主处置 API 全部不存在。"""
    import ctx_weft.protocols.capability as _cap
    from ctx_weft.core.models.discriminators import TaskErrorCode
    from ctx_weft.core.runtime import CtxWeftRuntime

    assert not hasattr(TaskErrorCode, "TOOL_OUTCOME_UNKNOWN")
    assert not hasattr(EventType, "OPERATION_UNCERTAIN")
    assert not hasattr(CtxWeftRuntime, "resolve_operation")
    for gone in ("Adjudication", "OperationAdjudicator", "OperationStore",
                 "OperationRecord", "OperationStatus", "OperationUpdate",
                 "RevisionConflict"):
        assert not hasattr(_cap, gone), gone


@pytest.mark.asyncio
async def test_conclude_closes_the_dangling_invoked():
    """作结要补发 `CapabilityFinished`——它闭合的正是上一轮那条悬着的 INVOKED。

    不发的话那次调用永远停在「已调用未完成」，下一次恢复又要重问一遍重跑授权。
    """
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="查过了，不必重试")))
    await ReconcileStep().execute(state, ctx)

    from ctx_weft.core.control.reducers import CAP_FOLD_EVENT_TYPES, fold_operations
    facts = fold_operations(
        await store.read_session_events_of_types("s1", CAP_FOLD_EVENT_TYPES))[op_id]
    assert facts.finished, "作结之后这次调用不再是「已调用未完成」"
    assert facts.attempts == ("inv_first",), "没有多出一次从未发生的尝试"


@pytest.mark.asyncio
async def test_conclude_writes_the_deterministic_result_id():
    """作结写的 TOOL_RESULT 必须带**确定性** id —— 它是 dangling 判定的 memory 通道。"""
    tool, store, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="外部已完成：TX-1")))
    await ReconcileStep().execute(state, ctx)

    recs = await _tool_records(ctx, state)
    assert [r.id for r in recs] == [tool_result_record_id(op_id)]

    await ReconcileStep().execute(state, ctx)      # 再跑一轮：认得出已完成
    assert len(await _tool_records(ctx, state)) == 1, "重跑 reconcile 不得写出第二份"
