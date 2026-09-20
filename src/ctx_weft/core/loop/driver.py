"""Step 抽象 + StepDriver。

设计文档 §6.2 / §6.4。
"""

from __future__ import annotations

import dataclasses
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from ctx_weft.core.assembler import AssembledPrompt, ContextAssembler
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import Event, EventBus, EventStore, EventType
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
    #: 事件流的**读侧**（spec: tool-operations）。`event_bus` 只有 emit / subscribe，
    #: 而恢复判据要折 capability 事件——读取一律经 `reducers.load_events_of_types`
    #: （按类型轻查询，store 不支持时降级为全量读 + 内存过滤）。
    #: None（宿主直构 / 测试替身）→ 折不出事实，调用方按「无从判断」保守处理。
    event_store: "EventStore | None" = None
    # HITL：管账的 service 与管栈的 waiter 分开持有——旧实现把两者塞进一个对象，
    # 于是编排层被迫认识协程栈（spec §3）。
    hitl: "HitlService | None" = None
    waiter: "HitlWaiter | None" = None
    # blob store：出网前把 ref 还原成 base64 用（Phase 3b）。默认 None → 不 rehydrate，
    # 既有构造点与既有测试行为逐字节不变。
    blob_store: "Any" = None


#: `state.extra` 键：本轮是否已过提交点（`act._commit_round` 的幂等标志）。
ROUND_COMMITTED_KEY = "_round_committed"
#: `state.extra` 键：本轮是由一条**热应答**开出来的（值 = 那个 hitl_id）。由 gateway 在
#: 协程被叫醒时写（`_rearm_commit_point_after_hot_reply`）。act 据此把「LLM 开口前的
#: 暂停」走成热撤销而不是整轮丢弃，见 `act._discard_round_if_uncommitted`。
HOT_REPLY_ROUND_KEY = "_hot_reply_round"
#: `state.extra` 键：`PrepareStep` 判定该跑 recognize_intent，但要等提交点才起飞。
#: 判定留在 prepare 是因为只有那里手握 `bound_capabilities`；起飞在提交点是因为
#: 旁路只该为**真的发生过**的那一轮花一次 LLM 调用（见 `act._commit_round`）。
RECOGNIZE_INTENT_PENDING_KEY = "_recognize_intent_pending"
#: `state.extra` 键：本 run 已用掉的「上下文恢复」次数（act 越过停机线 → 回 prepare
#: 压缩续跑）。配额见 `LoopConfig.max_context_recoveries`；耗尽后退回 observe/retry。
CONTEXT_RECOVERY_COUNT_KEY = "_context_recovery_count"
#: `state.extra` 键：本 run 已用掉的 act 轮数（跨恢复累计）。`max_turns_per_act` 的语义是
#: 「一次执行最多几轮」，恢复不该把它重置——否则每次恢复白送一整份轮数预算。
ACT_TURNS_USED_KEY = "_act_turns_used"
#: `state.extra` 键：上一次 compact「门开了但一条 MEMORY_COMPACTED 都没产」。各级都有
#: 可折性 guard，全 noop 说明这个 scope 已经压不动了——再恢复一次只是白烧一轮 LLM，
#: 故 act 据此立即放弃恢复，不等配额慢慢耗尽。
COMPACT_NOOP_KEY = "_compact_noop"


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


async def ingest_or_stage(
    memory: Any, event: "MemoryEvent", provider_ctx: Any, *,
    task_manager: Any, task_id: str,
) -> None:
    """task 层 memory 写入的唯一分流点：这个 task 开着未提交窗口 → 暂存进窗口；否则直接写。

    窗口开着 = 这一轮还没算数（LLM 还没开口）。这期间写进来的东西——用户刚发的消息、
    人刚给的答复、由答复回灌出来的工具结果——在这一轮算数之前都不该出现在 memory 里：
    否则它会跑在 `HitlResolved` 之前，撤销这一轮也只能靠补偿写（fold）去抹。暂存的写入
    由 `TaskManager.commit_round` 的钩子在终局事实之后按序落盘，丢弃时随窗口一起扔掉。
    同一轮里要读到它们的地方（prepare 装配）走 `TaskManager.staged_memory` 叠加。
    """
    stage = getattr(task_manager, "stage_memory", None)   # 宿主/测试的 TM 替身可能没有
    if stage is not None and task_id and stage(task_id, memory, event, provider_ctx):
        return
    await memory.ingest(event, provider_ctx)


def task_prompt_record_id(task_id: str, content: "str | list[Any]") -> str:
    """task 提问的**确定性** memory 记录 id。全仓唯一派生点（落库侧与恢复核对侧共用）。

    确定性的用处是让「这条提问写过没有」不必再靠猜：同一条提问重复写命中同一个 id，
    memory 的 id 契约（已存在的 id——含已 superseded——= no-op）直接兜住；提问被改写过
    （存量 reopen 数据）则哈希不同，照常写进去。被 L3 坍缩掉的旧提问仍占着它的旧 id，
    不会被复活。

    指纹取自**拍平的文本 + 各 part 的种类计数**：跨重启时提问从事件还原，文本形态稳定，
    而图片的引用形态在 event / memory 两个命名空间下并不相同，不能进指纹。纯图片提问
    因此靠 part 计数区分（文本为空时仅剩它）。
    """
    import hashlib
    from collections import Counter

    from ctx_weft.core.utils.content import content_to_text

    text = content if isinstance(content, str) else content_to_text(content)
    kinds = Counter(getattr(p, "type", "?") for p in content) if isinstance(content, list) else {}
    blob = f"{text}\u0000{sorted(kinds.items())}"
    digest = hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()[:16]
    return f"uprompt:{task_id}:{digest}"


async def _persist_user_prompt(state, ctx) -> None:
    """task 启动时持久化 raw user_prompt（呈现态框架由 composer 渲染期生成，不落库）。

    新开的 task 在第一轮算数之前开着未提交窗口（spec 2026-09-09），所以这条提问是**暂存**
    的（`ingest_or_stage`）：LLM 开口时随提交落盘，开口前被暂停就随窗口一起扔掉。
    `user_prompt_in_memory` 在暂存时就置真——它回答的是「装配时能不能从历史里找到这条
    提问」，而暂存的记录装配时看得见；窗口被丢弃时由 `TaskManager.discard_round` 还原。
    """
    task = state.task
    if not task.user_prompt or task.user_prompt_in_memory:
        return
    from ctx_weft.core.utils.clock import now_utc
    await ingest_or_stage(
        ctx.memory,
        MemoryEvent(
            id=task_prompt_record_id(task.id, task.user_prompt),
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
        task_manager=getattr(ctx, "task_manager", None),
        task_id=task.id,
    )
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
        BLACKBOARD_PUBLISH) is intentionally kept; only the subscription wiring is removed.
        (`tracking_task_ids`, the other thing this used to key off, was deleted with reopen
        on 2026-09-19.)
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
        # 窗口开着时它进**暂存区**（`ingest_or_stage`），这一轮算数才落盘、被撤销就随
        # 窗口扔掉——memory 里不出现「还没算数」的记录，撤销因此不需要任何补偿写。
        # 装配看得见暂存区（PrepareStep 叠加），所以模型不会看不到这条提问。
        #
        # 压缩（L0.5 图片降级 / L1 / L3）只折 memory、够不着暂存区，所以 PrepareStep
        # 判定要压缩时会**先提交再压缩**——带图的第一条消息照样降得了
        # （`tests/integration/test_media_fold_replay_e2e.py` 钉住这条）。
        await _persist_user_prompt(state, ctx)

        # Blackboard 订阅：Phase 3 起是 no-op（见该方法 docstring），保留调用点。
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
