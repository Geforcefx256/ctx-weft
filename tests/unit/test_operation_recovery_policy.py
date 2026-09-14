"""恢复策略表全分支（spec: tool-operations；wp6-2.2，design D2）。

组件级：真 reconcile + 真 gateway + 真账本（内存），policy 经 capability 声明。
分派矩阵：reviewed 不重跑 / idempotent 同 op_id 恰一次 / 存量无身份 unknown /
call_1 复用串扰根治。裁决链三态与 cancel 闭环见下方与 test_gateway_operation_ledger
与 test_tool_outcome_unknown。
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
from ctx_weft.protocols.operations import (
    OperationRecord,
    OperationStatus,
    operation_id_for,
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
        metadata={"tool_calls": [{"id": "call_1", "name": "fx__act", "input": {"n": 1}}]}),
        pctx)

    op_id = operation_id_for("default", "s1", "a1", rid, 0)
    if ledger_status is not None:
        await ops.prepare(OperationRecord(
            operation_id=op_id, tenant_id="default", session_id="s1", agent_id="a1",
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


async def test_legacy_no_ledger_record_concludes_without_rerun():
    """存量无身份（WP5 前的数据）：不以随机 id 执行副作用，作结后继续。"""
    tool, ops, bus, state, ctx, op_id = await _mk_fixture(policy="idempotent",
                                                          ledger_status=None)
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
    双通道判据按 op_id（record 不同）→ 仍为 dangling，正确进入策略分派。
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
    assert len(dangling) == 1, "op-id judged: reused call_1 must still dangle for the new call"
    await ReconcileStep().execute(state, ctx)
    assert tool.executions == 1


# ── 两值策略与裁决链（spec: tool-operations，wp6 重构）──────────────────────
#
# 策略只有两值，判据是「core 要不要做决定」。「谁来裁决」不是策略值而是一条链：
# provider 实现 OperationAdjudicator → 它裁；否则落到人。裁决能力靠 isinstance
# 发现，不靠 capability 声明。


def test_policy_values_validated_not_silently_defaulted():
    """非法取值响亮失败；缺省按需审核。**不静默降级**是这条的全部意义。"""
    from ctx_weft.protocols.operations import RecoveryPolicy as P, normalize_recovery_policy as N

    assert N("idempotent") is P.IDEMPOTENT
    assert N("reviewed") is P.REVIEWED
    assert N(P.IDEMPOTENT) is P.IDEMPOTENT
    assert N(None) is P.REVIEWED and N("") is P.REVIEWED   # Provider 没表态 → 需审核
    for bad in ("retry_safe", "queryable", "manual", "idempotant", "Idempotent", "auto"):
        with pytest.raises(ValueError, match="unknown recovery_policy"):
            N(bad)


def test_only_two_policies_exist():
    """策略面就是两个值——core 一视同仁的分类不该分成多个值。"""
    from ctx_weft.protocols.operations import RecoveryPolicy as P

    assert [p.value for p in P] == ["idempotent", "reviewed"]


class _AdjudicatingTool(_EffectTool):
    """实现裁决接口的 provider——**不需要在 capability 上声明任何东西**。"""

    def __init__(self, verdict, policy: str = "reviewed") -> None:
        super().__init__(policy)
        self.verdict = verdict
        self.adjudications = 0

    async def adjudicate(self, operation_id, ctx):
        self.adjudications += 1
        return self.verdict


async def test_adjudicator_conclusion_backfills_without_executing():
    """裁决者说不重跑 → 它给的 result 成为这次调用的结果，provider 不被执行。"""
    from ctx_weft.protocols.operations import Adjudication

    adj = _AdjudicatingTool(Adjudication.conclude("外部已完成：流水号 TX-9981"))
    _, ops, bus, state, ctx, op_id = await _mk_fixture(tool=adj)

    await ReconcileStep().execute(state, ctx)
    assert adj.adjudications == 1, "reviewed 必须先问裁决者"
    assert adj.executions == 0, "rerun=False → 绝不执行"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED
    assert rec.result == "外部已完成：流水号 TX-9981"


async def test_adjudicator_rerun_safe_reexecutes():
    """裁决者说重跑安全 → 同 op_id 重跑。"""
    from ctx_weft.protocols.operations import Adjudication

    adj = _AdjudicatingTool(Adjudication.rerun_safe())
    _, ops, bus, state, ctx, op_id = await _mk_fixture(tool=adj)

    await ReconcileStep().execute(state, ctx)
    assert adj.adjudications == 1
    assert adj.executions == 1, "权威判定没跑成 → 重跑"


async def test_adjudicator_says_it_cannot_tell():
    """「我不知道」由裁决者用文本表达——core 一视同仁：不重跑、写结果、继续。

    这正是砍掉三态枚举的理由：查到真结果和查不到，对 core 是同一条路。
    """
    from ctx_weft.protocols.operations import Adjudication

    adj = _AdjudicatingTool(Adjudication.conclude(
        "查不到这笔交易的记录；网关侧索引可能延迟，副作用可能已发生。不要重试。"))
    _, ops, bus, state, ctx, op_id = await _mk_fixture(tool=adj)

    outcome = await ReconcileStep().execute(state, ctx)
    assert adj.executions == 0, "判不了就不许跑"
    assert outcome.next_step == "prepare", "不停机，交给 agent"
    rec = await ops.get(op_id, ctx.provider_ctx)
    assert rec.status == OperationStatus.COMPLETED
    assert "不要重试" in str(rec.result), "裁决者的措辞原样成为工具结果"


async def test_reviewed_without_adjudicator_concludes_not_errors():
    """没有裁决者是**默认形态**而非配置错误：core 代为作结「无从查证」，不报错、不重跑。"""
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
    from ctx_weft.protocols.operations import OperationStatus

    assert not hasattr(TaskErrorCode, "TOOL_OUTCOME_UNKNOWN")
    assert not hasattr(EventType, "OPERATION_UNCERTAIN")
    assert not hasattr(CtxWeftRuntime, "resolve_operation")
    assert not hasattr(OperationStatus, "UNKNOWN")
