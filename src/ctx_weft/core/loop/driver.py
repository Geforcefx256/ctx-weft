"""Step 抽象 + StepDriver。

设计文档 §6.2 / §6.4。
"""

from __future__ import annotations

import dataclasses
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from ctx_weft.core.assembler import AssembledPrompt, ContextAssembler
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import Event, EventBus, EventType
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task

from ctx_weft.protocols import (
    LLMClient, MemoryEvent, MemoryKind, MemoryScope, MemoryProvider, MemoryAddress, ProviderContext,
)
from ctx_weft.protocols.events import EventOrigin

if TYPE_CHECKING:
    from ctx_weft.core.loop.steps.observe import Verdict
    from ctx_weft.core.loop.steps.act import TurnRecord
    from ctx_weft.core.control.tokens import CancelToken, PauseToken
    from ctx_weft.core.loop.capability_gateway import CapabilityGateway
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.orchestrator import TaskManager
    from ctx_weft.core.orchestrator.model import ResolvedModel
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome
    from ctx_weft.core.hitl.service import HitlService
    from ctx_weft.core.loop.hitl_waiter import HitlWaiter
    from ctx_weft.protocols.capability import CapabilityProvider

logger = logging.getLogger(__name__)


# ── Step / StepOutcome ────────────────────────────────────────────────────────


@dataclass
class StepOutcome:
    """每步执行的统一返回（设计文档 §6.2）。"""

    next_step: str | None
    state_patch: dict[str, Any] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    request_pause: bool = False


@runtime_checkable
class Step(Protocol):
    name: str

    @abstractmethod
    async def execute(
        self,
        state: "LoopState",
        ctx: "LoopContext",
    ) -> StepOutcome: ...


# ── LoopState ─────────────────────────────────────────────────────────────────


@dataclass
class LoopState:
    """Loop 跨 Step 共享的可变状态。Driver 维护，按 state_patch 增量更新。"""

    run_id: str
    session: Session
    task: Task
    agent: Agent
    scope: MemoryAddress
    sequence_counter: int = 0  # 每发一个事件 +1

    # 由 PrepareStep 写入
    assembled_prompt: AssembledPrompt | None = None
    # 由 ActStep 写入
    transcript: list[TurnRecord] = field(default_factory=list)
    act_exit_reason: str = ""
    # 由 ObserveStep 写入
    verdict: Verdict | None = None

    #: 本次 run 的结局，由 loop 在结束前填好、交给 TaskManager 决定 task 处置。
    #: loop 报「发生了什么」，不报「task 该变成什么」——后者是 TM 的活
    #: （docs/superpowers/plans/2026-09-02-task-status-ownership.md 的处置表）。
    run_outcome: "RunOutcome | None" = None

    #: 本次 run 实际用的 (client, account, model, 窗口)——由派发方在构造 LoopState
    #: 之前解出并塞入（AgentLifecycleManager.resolve_model / materialize 的产物）。
    #: resolve_llm_identity 的唯一真值来源，不再读 session.llm_model 兜底。
    resolved_model: "ResolvedModel | None" = None

    # 其他扩展字段
    extra: dict[str, Any] = field(default_factory=dict)

    origin: str = ""  # driver 每步开始前写入，make_event 默认从这里取（events-v2.md §4）

    def apply_patch(self, patch: dict[str, Any]) -> "LoopState":
        """应用 state_patch 返回新 LoopState（浅拷贝）。"""
        if not patch:
            return self
        return dataclasses.replace(self, **patch)


# ── LoopContext ───────────────────────────────────────────────────────────────


@dataclass
class RunPhase:
    """Per-run loop-progress flags (set by ActStep), used to pick the interrupt phase.

    produced      — 本 run 是否吐过 token（区分①未出 token / ②已出 token）。
    in_tool_loop  — 是否已进入工具调用循环（③）。
    """

    produced: bool = False
    in_tool_loop: bool = False


@dataclass
class LoopContext:
    """每次 loop run 一个，包装所有跨 step 的依赖。"""

    assembler: ContextAssembler
    llm: LLMClient
    memory: MemoryProvider
    event_bus: EventBus
    provider_ctx: ProviderContext
    # capability 解析（Phase 3+）
    capability_cache: CapabilityCache|None = None
    capability_providers: list[CapabilityProvider]|None = None
    capability_gateway: CapabilityGateway|None = None
    authorizer: Any = None
    skill_provider_index: dict = None  # provider_name → SkillCapabilityProvider；PrepareStep 用于加载 Level2
    # 控制令牌（Phase 6）
    cancel_token: CancelToken|None = None
    pause_token: PauseToken|None = None
    # run 级阶段标记（ActStep 维护；曾挂在 CancelToken 上）
    run_phase: RunPhase = field(default_factory=RunPhase)
    # 配置
    config: Any = None
    # TaskManager 引用（Phase 5+）；PrepareStep compact dispatch 用；None 时退化为 inline compact
    task_manager: TaskManager|None = None
    # HITL：管账的 service 与管栈的 waiter 分开持有——旧实现把两者塞进一个对象，
    # 于是编排层被迫认识协程栈（spec §3）。
    hitl: "HitlService | None" = None
    waiter: "HitlWaiter | None" = None
    # blob store：出网前把 ref 还原成 base64 用（Phase 3b）。默认 None → 不 rehydrate，
    # 既有构造点与既有测试行为逐字节不变。
    blob_store: "Any" = None


#: `state.extra` 键：本轮是否已过提交点（`act._commit_round` 的幂等标志）。
ROUND_COMMITTED_KEY = "_round_committed"
#: `state.extra` 键：`PrepareStep` 判定该跑 recognize_intent，但要等提交点才起飞。
#: 判定留在 prepare 是因为只有那里手握 `bound_capabilities`；起飞在提交点是因为
#: 旁路只该为**真的发生过**的那一轮花一次 LLM 调用（见 `act._commit_round`）。
RECOGNIZE_INTENT_PENDING_KEY = "_recognize_intent_pending"


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(
    state: LoopState,
    type: str,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    causation_id: str | None = None,
    *,
    origin: str | None = None,
) -> Event:
    """构造一个 run 级 Event：从 LoopState 抽字段 + 自增 sequence。

    封套本身与 `EVENT_TYPES` 白名单校验交 `core.util.new_event`——那是
    全仓唯一一份。本函数只保留 run 域真正属于自己的两件事：LoopState 的字段抽取，
    与 `sequence_counter` 自增。

    **先算后提交**：新值先算出来交给 `new_event`，它校验通过、真的造出事件之后才写回
    counter。这样「坏类型不改 counter」这条改造前的行为逐字保留——`llm_gateway` 与
    `act` 都在 emit 之前读 `state.sequence_counter` 拼 request_id，不该因为一次校验
    失败就跳号。
    """
    seq = state.sequence_counter + 1
    ev = new_event(
        type,
        session_id=state.session.id,
        tenant_id=state.session.tenant_id,
        origin=origin if origin is not None else getattr(state, "origin", ""),
        run_id=state.run_id,
        sequence=seq,
        task_id=state.task.id,
        agent_id=state.agent.id,
        payload=payload,
        metadata=metadata,
        causation_id=causation_id,
    )
    state.sequence_counter = seq
    return ev


async def _persist_user_prompt(state, ctx) -> None:
    """task 启动时持久化 raw user_prompt（呈现态框架由 composer 渲染期生成，不落库）。

    记下这条记录的 id（`task.user_prompt_memory_id`）：用户在 LLM 开口之前按暂停时，
    act 的丢弃路径要靠它把这一轮的用户消息 `fold` 掉（spec 2026-09-09）。**必须记 id
    而不是事后「取视图里最后一条 user」**——用户连发两条、或上一条是 HITL 应答时，
    那种取法会撤错人。
    """
    task = state.task
    if not task.user_prompt or task.user_prompt_in_memory:
        return
    from ctx_weft.core.utils.clock import now_utc
    record_id = await ctx.memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            # 原样落库（含多模态）：这是图片在改造前第一次消失的地方。
            # 装配期是否拍扁由框架决定（Phase 2），落库必须无损。
            content=task.user_prompt,
            timestamp=now_utc(),
            role="user",
            metadata={"task_id": task.id},
        ),
        ctx.provider_ctx,
    )
    task.user_prompt_memory_id = record_id
    task.user_prompt_in_memory = True


# ── StepDriver ────────────────────────────────────────────────────────────────


_STEP_ORIGIN: dict[str, str] = {
    "prepare": EventOrigin.LOOP_PREPARE,
    "act": EventOrigin.LOOP_ACT,
    "observe": EventOrigin.LOOP_OBSERVE,
    "recognize_intent": EventOrigin.LOOP_RECOGNIZE_INTENT,
    "compact": EventOrigin.LOOP_COMPACT,
    "finalize": EventOrigin.LOOP_FINALIZE,
    "suspend": EventOrigin.LOOP_SUSPEND,
    "reconcile": EventOrigin.LOOP_RECONCILE,
}


@dataclass
class StepDriver:
    """驱动 Step 链。从 initial_step 开始，按 outcome.next_step 顺序执行。"""

    steps: dict[str, Step]
    initial_step: str = "prepare"

    async def _ensure_blackboard_subscriptions(self, state: LoopState, ctx: LoopContext) -> None:
        """No-op since Phase 3 (2026-06-30).

        Predecessor results now reach a task via memory recall (Phase 2 inherit/recall), and the
        observer's own-children review affordance is surfaced in the observe cue from task_manager
        (see ObserveStep). The blackboard mechanism (subscribe_topic/recall_topic/BlackboardSource/
        BLACKBOARD_PUBLISH) and `tracking_task_ids` are intentionally kept; only the subscription
        wiring is removed.
        """
        return

    async def run(
        self,
        initial_state: LoopState,
        ctx: LoopContext,
    ) -> AsyncIterator[StepOutcome]:
        state = initial_state

        # 子任务真正开始执行 → 在派发方 scope 铸派发框 + running ack（同锚 started_at）。
        # 须在 _persist_user_prompt 之前概念上成立（框 @ started_at < 子 body @ now），实际由
        # 时间戳排序保证，与写入先后无关。刻意不在派发时刻铸——那时子任务生死未定，弃子/staged
        # 丢弃会留下永远 pending 的孤儿框（详见 steps.finalize.ensure_dispatch_frame_at_start）。
        # 函数级 import：steps.finalize 在模块级 import 本模块，反向模块级 import 会成环。
        from ctx_weft.core.loop.steps.finalize import ensure_dispatch_frame_at_start
        await ensure_dispatch_frame_at_start(state, ctx)

        # 任务启动时立即持久化 raw user_prompt，保证 resume 时对话上下文完整可重建
        # （呈现态框架 ## Current Task/Message 由 composer 渲染期生成，不落库）。
        #
        # ⚠ spec 2026-09-09 推迟的是**事件**，不是这一次 memory 写入——两者的推迟代价
        # 完全不同。落库若推迟到 act 的提交点，PrepareStep 的预算折叠（L0.5 图片降级、
        # L1/L3 折叠）在**每一轮的首次装配**时都看不见这条记录：带图的第一条消息因此
        # 一张都降不了，直接顶着满额图片去撞窗口。这条已由
        # `tests/integration/test_media_fold_replay_e2e.py` 实测钉住。
        #
        # 所以这一份照旧立刻落库；它的「撤销」由 act 的丢弃路径用 `memory.fold([id], [])`
        # 完成（纯遗忘，标 superseded，`load_view` 自然滤掉）——那是 provider 早就有的
        # 原语，compact / finalize / background_observe 都在用，不是为此新造的东西。
        await _persist_user_prompt(state, ctx)

        # Blackboard 订阅：本 task 订阅相关任务的结果 topic，下一次 reason 即可感知。
        # 幂等，每次 run 都执行：① 同 plan 前序（tracking_task_ids）② 已派生的子任务。
        await self._ensure_blackboard_subscriptions(state, ctx)

        next_step_name: str | None = self.initial_step

        while next_step_name is not None:
            # 软打断（interrupt）由 act 的 checkpoint 负责 park，这里不硬取消（否则 step 间命中会误终态）；
            # 仅硬取消（cancel 模式）在 step 边界抛 CancelledError。
            # 曾经 CancelToken 上挂过 mode 字段区分 pause/cancel 两种模式，`mode == "cancel"`
            # 这个条件是那段历史的残留；pause 模式早已退役、CancelToken 现在没有 mode 属性，
            # getattr 恒回落默认值 "cancel"，条件恒真——删掉，不是漏判。
            tok = ctx.cancel_token
            if tok is not None and tok.is_cancelled:
                tok.raise_if_cancelled()

            step = self.steps.get(next_step_name)
            if step is None:
                raise ValueError(f"Step '{next_step_name}' not registered")

            # 每步开始前写 state.origin（driver 发的三条 STEP_* 事件覆盖为 LOOP_DRIVER）
            state.origin = _STEP_ORIGIN.get(step.name, EventOrigin.LOOP_DRIVER)

            # emit StepStarted
            start_ev = make_event(
                state, EventType.STEP_STARTED, {"step_name": step.name},
                origin=EventOrigin.LOOP_DRIVER,
            )
            await ctx.event_bus.emit(start_ev)

            try:
                outcome = await step.execute(state, ctx)
            except Exception as e:
                error_dict = {
                    "step_name": step.name,
                    "error_code": type(e).__name__,
                    "error_message": str(e),
                }
                fail_ev = make_event(
                    state, EventType.STEP_FAILED,
                    error_dict,
                    origin=EventOrigin.LOOP_DRIVER,
                )
                await ctx.event_bus.emit(fail_ev)
                raise

            # apply state patch
            if outcome.state_patch:
                state = state.apply_patch(outcome.state_patch)

            # emit collected events
            for ev in outcome.events:
                await ctx.event_bus.emit(ev)

            # emit StepCompleted
            done_ev = make_event(
                state, EventType.STEP_COMPLETED,
                {"step_name": step.name, "next_step": outcome.next_step},
                origin=EventOrigin.LOOP_DRIVER,
            )
            await ctx.event_bus.emit(done_ev)

            yield outcome

            next_step_name = outcome.next_step
