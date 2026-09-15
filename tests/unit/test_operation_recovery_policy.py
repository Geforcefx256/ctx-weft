"""恢复策略表全分支（spec: tool-operations；wp6-2.2，design D2）。

组件级：真 reconcile + 真 gateway + 真账本（内存），policy 经 capability 声明。
分派矩阵：reviewed 交裁决链不自行重跑 / idempotent 同 op_id 恰一次 / 无账本记录作结
「无从查证」/ call_1 复用串扰根治 / 裁决链两分支（rerun / conclude）+ 无裁决者默认形态。
账本执行序见 test_gateway_operation_ledger，真子进程强退见
tests/integration/test_operation_crash_matrix.py。
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.steps.reconcile import ReconcileStep
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.memory import MemoryEvent
from ctx_weft.core.utils.ids import mint_call_id

#: 摄入点铸造的内部标识——既是消息里的 tool_call id，也是账本键。
OP = mint_call_id(anchor="rec_fx", ordinal=0, raw_id="call_1", turn_seq=0)

from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    OperationRecord,
    OperationStatus,
    OperationUpdate,
    RerunAuthorizer,
)
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.operations import InMemoryOperationStore
from ctx_weft.core.utils.clock import now_utc

from ctx_weft.core.models.discriminators import TaskErrorCode


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


class _Bus:
    def __init__(self): self.events = []
    async def emit(self, e): self.events.append(e)


async def _mk_fixture(policy="reviewed", ledger_status: OperationStatus | None = OperationStatus.STARTED,
                      tool=None):
    tool = tool if tool is not None else _EffectTool(policy)
    mem = InMemoryMemoryProvider()
    ops = InMemoryOperationStore()
    bus = _Bus()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="a1")

    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="t1", status="ACTIVE"),
        agent=SimpleNamespace(id="a1", template_id="t"),
        scope=scope, resolved_model=SimpleNamespace(model="m", account=""),
        sequence_counter=0,
    )
    ctx = LoopContext(assembler=None, llm=None, memory=mem, event_bus=bus,
                      provider_ctx=pctx)
    cache = CapabilityCache()
    cache.put("a1", [tool._cap()])
    gw = CapabilityGateway(capability_cache=cache, capability_providers=[tool],
                           memory=mem, event_bus=InProcessEventBus(), operation_store=ops)
    ctx.capability_gateway = gw

    rid = await mem.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK, address=scope,
        content="", timestamp=now_utc(), role="assistant",
        metadata={"tool_calls": [{"id": OP, "name": "fx__act", "input": {"n": 1}}]}),
        pctx)

    op_id = OP        # 账本键就是 tool_call 的内部标识
    if ledger_status is not None:
        await ops.prepare(OperationRecord(
            tool_call_id=op_id, tenant_id="default", session_id="s1", agent_id="a1",
            assistant_record_id=rid, tool_ordinal=0, tool_name="fx__act",
            status=ledger_status, revision=2, attempts=["inv_first"],
        ), pctx)
    return tool, ops, bus, state, ctx, op_id


async def test_reviewed_started_concludes_without_rerun():
    """H3 的核心性质：不重跑。但**不停机**——作结写成工具结果，循环继续。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 0, "reviewed + started MUST NOT re-execute"
    assert outcome.next_step == "prepare", "不确定是工具结果，不是控制流——不停机"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED, "已作结（结论内容在 result 里）"
    assert "unverified" in str(rec.result), rec.result


async def test_idempotent_started_reruns_exactly_once():
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent")
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1
    assert outcome.next_step == "prepare"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED
    assert rec.attempts == ["inv_first", rec.attempts[-1]] or len(rec.attempts) == 2


async def test_durable_ledger_without_record_is_first_execution():
    """**持久**账本 + 内部标识 + 查不到记录 = 确定还没跑过，首执安全。

    gateway 的 prepare 在授权与参数校验**之后**，所以这三个条件同时成立只能是崩在
    prepare 之前——provider 确定没被调用过。此前这里和「真·无从判断」走同一个出口，
    给 agent 写一句「无法确定副作用是否已发生」——对这条路那是不实的。
    """
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed",
                                                          ledger_status=None)
    ops.durable = True        # 扮演 SqlOperationStore：跨进程活着
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1, "确定未启动 → 首执，即使策略是 reviewed"
    assert outcome.next_step == "prepare"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED


async def test_non_durable_ledger_without_record_concludes_without_rerun():
    """内存账本查不到记录**不**证明没跑过——重启后它一片空白，两种情形一模一样。

    O-T05 的真子进程强退就踩在这里：宿主没注册持久账本，恢复进程拿到空的内存默认
    账本，若按「没记录 = 没跑过」处理，副作用会被执行两次。
    """
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent",
                                                          ledger_status=None)
    assert ops.durable is False
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 0, "不持久 → 无从判断 → 不重跑"
    assert outcome.next_step == "prepare"


async def test_no_ledger_at_all_concludes_without_rerun():
    """账本根本没接线（宿主直构 gateway / 存量数据）——同样不重跑。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent",
                                                          ledger_status=None)
    ctx.capability_gateway._operation_store = None
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 0, "无账本身份的副作用工具不得自动重跑"
    assert outcome.next_step == "prepare"


async def test_prepared_runs_first_execution():
    """prepared 且从未 started：首执（此前无副作用）——即使 reviewed。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed",
                                                          ledger_status=OperationStatus.PREPARED)
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1
    assert outcome.next_step == "prepare"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED


async def test_call1_reuse_no_cross_talk():
    """call_1 复用串扰根治：旧 tool 记录的 wire id 不使新调用被误判完成。

    场景：上一回合 call_1 已有 tool 记录（wire 通道 done）；本回合复用 call_1——
    双通道判据按内部标识（与 wire id 无关）→ 仍为 dangling，正确进入策略分派。
    """
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent")
    # 上一回合的 tool 记录（wire id 同为 call_1，但属于别的 record）
    from ctx_weft.protocols.memory import MemoryEvent as ME
    prev_rid = await ctx.memory.ingest(ME(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=state.scope, content="old result", timestamp=now_utc(),
        role="tool", metadata={"tool_call_id": "call_1"}), ctx.provider_ctx)
    from ctx_weft.core.loop.steps.reconcile import _dangling_tool_calls
    dangling, _ = await _dangling_tool_calls(ctx.memory, state.scope, ctx.provider_ctx)
    assert len(dangling) == 1, "按内部标识判定：复用的 call_1 不得让新调用误判为已完成"
    await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1


# ── 两值策略与裁决链（spec: tool-operations，wp6 重构）──────────────────────
#
# 策略只有两值，判据是「core 要不要做决定」。「谁来裁决」不是策略值而是一条链：
# provider 实现 OperationAdjudicator → 它裁；否则落到人。裁决能力靠 isinstance
# 发现，不靠 capability 声明。


def test_policy_values_validated_not_silently_defaulted():
    """非法取值响亮失败；缺省按需审核。**不静默降级**是这条的全部意义。"""
    from ctx_weft.protocols.capability import RecoveryPolicy as P, normalize_recovery_policy as N

    assert N("idempotent") is P.IDEMPOTENT
    assert N("reviewed") is P.REVIEWED
    assert N(P.IDEMPOTENT) is P.IDEMPOTENT
    assert N(None) is P.REVIEWED and N("") is P.REVIEWED   # Provider 没表态 → 需审核
    for bad in ("retry_safe", "queryable", "manual", "idempotant", "Idempotent", "auto"):
        with pytest.raises(ValueError, match="unknown recovery_policy"):
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
        self.seen_args: dict | None = None

    async def authorize_rerun(self, capability, record, ctx, arguments=None, *, tool_call_id=""):
        self.asked += 1
        self.seen_args = record.effective_args
        return self.verdict


def _with_rerun(ctx, authorizer, key="fx"):
    """把重跑授权挂到 gateway（等价于 registry 的 rerun_authorizer= 注册位）。"""
    ctx.capability_gateway._rerun_authorizers[key] = authorizer
    return authorizer


async def test_rerun_denied_concludes_without_executing():
    """重跑授权说不 → 它给的 message 成为这次调用的结果，provider 不被执行。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    rr = _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="外部已完成：流水号 TX-9981")))

    await ReconcileStep().execute(state, ctx)
    assert rr.asked == 1, "reviewed 必须先问重跑授权"
    assert tool.executions == 0, "allowed=False → 绝不执行"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED
    assert "TX-9981" in str(rec.result)


async def test_rerun_allowed_reexecutes():
    """重跑授权放行 → 同 tool_call_id 重跑。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    rr = _with_rerun(ctx, _Rerun(AuthorizationDecision(allowed=True)))

    await ReconcileStep().execute(state, ctx)
    assert rr.asked == 1
    assert tool.executions == 1, "权威判定没跑成 → 重跑"


async def test_rerun_says_it_cannot_tell():
    """「我不知道」用文本表达——core 一视同仁：不重跑、写结果、继续。

    这正是砍掉三态枚举的理由：查到真结果和查不到，对 core 是同一条路。
    """
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False,
        message="查不到这笔交易的记录；网关侧索引可能延迟，副作用可能已发生。不要重试。")))

    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 0, "判不了就不许跑"
    assert outcome.next_step == "prepare", "不停机，交给 agent"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert "不要重试" in str(rec.result), "授权方的措辞原样成为工具结果"


async def test_rerun_sees_effective_args_not_the_raw_ones():
    """账本持有的是**上一次真实执行**的参数（授权 + HITL 改写之后）。

    没有这一条，任何走过 HITL 改参的调用，重跑授权都查不对东西。
    """
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    rec0 = await ops.get(op_id, ctx.provider_ctx)
    await ops.compare_and_set(op_id, rec0.revision, OperationUpdate(), ctx.provider_ctx)
    # 模拟首次执行落库的 effective_args（与对话里那份 {"n": 1} 不同）
    ops._records[op_id].effective_args = {"n": 1, "order_id": "ORD-7", "approved_by": "ops"}
    rr = _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="查过了，没跑成但也别重试")))

    await ReconcileStep().execute(state, ctx)
    assert rr.seen_args == {"n": 1, "order_id": "ORD-7", "approved_by": "ops"}


async def test_rerun_authorizer_is_per_tool_not_per_provider_object():
    """逐工具粒度：给自己不控制的 provider 挂，同 provider 其它工具不受影响。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(allowed=True)), key="fx:act")
    await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1, "精确 capability_id 优先于 provider 前缀"


async def test_human_decision_default_mapping():
    """基类自带的默认：人批准 = 重跑，人拒绝 = 用人写的那句话作结。"""
    from ctx_weft.protocols.hitl import HITL_OUTCOME_ACCEPTED, HitlDecision

    rr = _Rerun(AuthorizationDecision(allowed=True))
    cap = ToolCapability(id="fx:act", name="act", description="d")
    approved = await rr.on_human_decision(
        cap, None, None, None, "tc", HitlDecision(outcome=HITL_OUTCOME_ACCEPTED))
    assert approved.allowed is True
    rejected = await rr.on_human_decision(
        cap, None, None, None, "tc", HitlDecision(outcome="rejected", message="别重试，我手工处理了"))
    assert rejected.allowed is False
    assert rejected.result_is_error is False, "人工作结不是错误"
    assert "手工处理" in str(rejected.message)


async def test_reviewed_without_rerun_authorizer_concludes_not_errors():
    """没注册重跑授权是**默认形态**而非配置错误：core 代为作结「无从查证」，不报错、不重跑。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    outcome = await ReconcileStep().execute(state, ctx)
    assert tool.executions == 0
    assert outcome.next_step == "prepare"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED
    assert "unverified" in str(rec.result)


def test_no_uncertainty_control_plane():
    """「不确定」不再有专属控制面：错误码 / 事件类型 / 宿主处置 API 全部不存在。"""
    from ctx_weft.core.models.discriminators import TaskErrorCode
    from ctx_weft.core.runtime import CtxWeftRuntime
    from ctx_weft.protocols.events import EventType
    from ctx_weft.protocols.capability import OperationStatus

    assert not hasattr(TaskErrorCode, "TOOL_OUTCOME_UNKNOWN")
    assert not hasattr(EventType, "OPERATION_UNCERTAIN")
    assert not hasattr(CtxWeftRuntime, "resolve_operation")
    import ctx_weft.protocols.capability as _cap
    assert not hasattr(_cap, "Adjudication")
    assert not hasattr(_cap, "OperationAdjudicator")
    assert not hasattr(OperationStatus, "UNKNOWN")


async def test_conclude_writes_the_deterministic_result_id():
    """作结写的 TOOL_RESULT 必须带**确定性** id —— 它是 dangling 判定的 memory 通道。

    拿自动 id 写进去，这次调用在下一轮 reconcile 眼里仍是 dangling，而账本已 COMPLETED，
    于是「账本完成而 memory 缺失」的补写分支会再写一条：对话里两份结果。
    """
    from ctx_weft.protocols.capability import tool_result_record_id

    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(
        allowed=False, result_is_error=False, message="外部已完成：TX-1")))
    await ReconcileStep().execute(state, ctx)

    recs = await ctx.memory.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    tool_recs = [r for r in recs if r.role == "tool"]
    assert [r.id for r in tool_recs] == [tool_result_record_id(op_id)]

    # 再跑一轮 reconcile：memory 通道认得出来，不再产生第二份结果。
    await ReconcileStep().execute(state, ctx)
    recs2 = await ctx.memory.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    assert len([r for r in recs2 if r.role == "tool"]) == 1, "重跑 reconcile 不得写出第二份"


async def test_conclude_emits_no_orphan_finished_event():
    """作结发生在 `_record_invocation` 之前 —— 发 CapabilityFinished 就是一条没有配对
    Invoked 的孤立事件，而这次 invoke 本来也没真的调用 provider。改前同样只写 memory。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="reviewed")
    _with_rerun(ctx, _Rerun(AuthorizationDecision(allowed=False, message="不重试")))
    await ReconcileStep().execute(state, ctx)

    kinds = [getattr(e, "type", None) for e in bus.events]
    assert EventType.CAPABILITY_FINISHED not in kinds, kinds
