"""CtxWeftRuntime：顶层 API。

Phase 4 版本：完整 SessionRegistry + TaskManager + AgentLifecycleManager 支持；
同时保留 run_single_task() 兼容 Phase 1 测试。
"""

from __future__ import annotations

from ctx_weft.core.control.converters import agent_detail, task_detail
from ctx_weft.core.control.session_queries import public_task_detail

import asyncio
import contextlib
import functools
import inspect
import logging
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart

from ctx_weft.core.assembler import ContextAssembler
from ctx_weft.core.control.reducers import read_task_view, rebuild_view, reduce_events
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.protocols.runtime import PromptDiagnostic, QueryResult, RuntimeClosedError, RuntimeShutdownError, SessionDetail, ShutdownReport, SessionInitializationResult, SessionForkResult
from ctx_weft.protocols.task import TaskDetail
from ctx_weft.core.assembler.assembler import AssemblerDeps
from ctx_weft.core.assembler.prompts import runtime_prompt
from ctx_weft.core.assembler.sources import (
    AgentRecallSource,
    BlackboardSource,
    CapabilitySource,
    GuidanceSource,
    IdentitySource,
    KnowledgeRetrievalSource,
    SemanticRecallSource,
    TaskSpecSource,
)
from ctx_weft.core.control.tokens import CancelToken, PauseToken, RunTokens
from ctx_weft.core.models.discriminators import CancelReason, InterruptReason
from ctx_weft.protocols.events import Event, EventOrigin, EventType
from ctx_weft.core.utils.event import emit_event
from ctx_weft.core.hitl.registry import (
    HITL_STAGE_AUTHZ,
    HITL_STAGE_RERUN,
    HitlRegistry,
    PendingHitl,
)
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import ExecutionRegistry, LoopContext, LoopState, StepDriver, make_event
from ctx_weft.core.loop.hitl_waiter import HitlWaiter
from ctx_weft.core.hitl.registry import reply_memory_id
from ctx_weft.core.loop.signals import HitlPark, RoundDiscarded
from ctx_weft.core.loop.steps import (
    ActStep,
    FinalizeStep,
    ObserveStep,
    PrepareStep,
    RecognizeIntentStep,
)
from ctx_weft.core.loop.steps.compact import CompactStep
from ctx_weft.core.loop.steps.reconcile import ReconcileStep
from ctx_weft.core.loop.steps.suspend import SuspendStep
from ctx_weft.core.capabilities.cache import CapabilityCachePool
from ctx_weft.core.capabilities.control_tools import ControlCapabilityProvider
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.model import ModelChoice, ResolvedModel
from ctx_weft.protocols.llm import ModelCallConfig
from ctx_weft.core.orchestrator.lifecycle.agent_state import AgentInput
from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.orchestrator.task.runner import AgentBinding, effective_agent_id
from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES
from ctx_weft.core.registry import ProviderRegistry
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import PORT_NONE, NormalTaskSettings, Task, default_port_for
from ctx_weft.core.models.errors import (
    AgentNotFound,
    AgentNotLoaded,
    AgentNotRunningError,
    SessionAlreadyExistsError,
    SessionNotFound,
    TaskNotFound,
    crash_error_code,
    crash_run_outcome,
)
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.core.utils.task_ref import task_ref
from ctx_weft.protocols import (
    AgentTemplate,
    LLMClient,
    LLMOutageError,
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryProvider,
    MemoryScope,
    ProviderContext,
    load_records,
)
from ctx_weft.protocols.agent import AgentDetail, AgentSummary, CompactReceipt
from ctx_weft.protocols.capability import (
    AgentCapabilityProvider,
    SessionScopedCapabilityProvider,
)
from ctx_weft.protocols.events import EventBus, PersistenceUnavailableError
from ctx_weft.protocols.hitl import (
    HITL_OUTCOME_ACCEPTED,
    HITL_OUTCOME_CANCELLED,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    HitlReply,
    HitlRequestView,
    NoResumeDelivery,
    ToolResultDelivery,
    UserTurnDelivery,
)

logger = logging.getLogger(__name__)

#: `resume_agent` 冷续跑一个暂停气泡时喂给 `_write_hitl_reply_turn` 的 message——
#: 操作者的"继续跑"没有新指示可言，空串会落到那边的 `"(no response)"` 兜底文案
#: （读起来像"问了没人答"，语义不对：这不是一次没人回答的提问）。与
#: `act.INTERRUPTED_MARK` / `act.CANCELLED_MARK` 同一方括号风格的合成标记。
_RESUME_MARK = "[resumed by operator]"


@dataclass
class _PendingAssistantMemory:
    """Retain the original envelopes until both durable effects are confirmed."""

    finished: Event
    memory: MemoryEvent
    provider_ctx: ProviderContext
    after_position: int
    committed: bool = False
    provisional: bool = False


def _latest_prior_root_task(task_manager: "TaskManager", t: "Task") -> "Task | None":
    """The most recent prior root task (no parent) in the session — the inherit source
    for a root user-turn dispatched straight to a sub-agent.

    Such a task has ``use_subagent=True`` + ``inherit_memory=True`` but no
    ``parent_task_id`` (root turns have none), so the parent-based copy in ``_resolve``
    has nothing to copy from. We fall back to the previous root task: because
    ``_copy_memory_for_inherit`` recalls by ``agent_id``, sourcing from it pulls the
    prior root agent's whole conversation so far, restoring cross-turn continuity.
    """
    prior = [
        x for x in task_manager.all_tasks()
        if not x.parent_task_id and x.id != t.id
        and x.created_at is not None and t.created_at is not None
        and x.created_at < t.created_at
    ]
    return max(prior, key=lambda x: x.created_at, default=None)


async def _copy_memory_for_inherit(
    *,
    source_agent_id: str,
    child_task: "Task",
    sub_agent: "Agent",
    memory: "MemoryProvider",
    session_id: str,
    tenant_id: str,
    source_task_id: str | None = None,
) -> None:
    """spawn 时把 source agent 的当前召回视图复制进 child agent scope（spec Phase 2 2026-06-30）。

    **第一参数是 agent 不是 task**（spec/09 §6）：这个函数从第一天起就是按 agent_id 取
    记忆的（下面两次 `load_records` 都只给 `agent_id`，不给 `task_id`），从前收 `parent_task`
    只是为了从它身上推出那个 agent id。改收 agent id 之后，「继承谁的记忆」可以由调用方
    直接锚定——血缘（`parent_agent_id`）与记忆来源就此成为两个正交的轴，
    `dispatch_task` 的「不从任何已有 agent 派生、但要它的上下文」才表达得出来。

    ``source_task_id``：仅用于元数据标签，可空（显式指定 agent 时调用方常常并不关心
    是哪条 task）。真正的来源标识是 `inherited_from_agent_id`，恒写。

    **全 keyword-only**：第一个位置参数从 `Task` 变成了 `str`，而 Python 不会为此报错
    ——旧的位置调用会把一个 Task 对象静默绑到 `source_agent_id` 上，`load_records` 拿它
    当 agent_id 查，查空，于是「继承了个寂寞」而测试只在断言处才隐约红。加一道 `*`
    让这类陈旧调用在调用点就 TypeError。

    镜像父此刻 AgentRecallSource 的两路召回：task 层 body（父自身 + 同 agent 兄弟，按 agent_id 跨 task）
    + agent 层对话回合（Phase 1 写的 start_task 框 / 跨 agent bubble / 同 agent finish 对）。二者按
    (timestamp, seq_no) 归并后写入 child scope，作 child 的起始记忆；之后两边各自演进。
    同 agent 兄弟 body 因此带框（不再裸泄漏），跨 agent 兄弟以 bubble 呈现。
    """
    # AGENT_COMPACT_SUMMARY（父的黑盒折叠派发日志）仍排除——对子无用（沿用 2026-06-23 的窄化意图，
    # 只是现在改为 mirror 而非「仅 OPEN-task body」）。
    from ctx_weft.protocols import MemoryAddress, MemoryEvent

    ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id)

    # Mirror the parent agent's current recall view (spec Phase 2, 2026-06-30):
    #   (a) task-layer body — parent's own turns + same-agent siblings' bodies (by agent_id), and
    #   (b) agent-layer dispatch turns — the start_task frames, cross-agent bubbles, and same-agent
    #       finish pairs that Phase 1 writes into the parent agent scope.
    # Merging both by (timestamp, seq_no) means inherited same-agent sibling bodies arrive FRAMED
    # (their start_task frame precedes them, so no naked leak) and cross-agent siblings arrive as
    # bubbles. AGENT_COMPACT_SUMMARY is still excluded — the parent's folded black-box dispatch log
    # is of little use to a child.
    from ctx_weft.protocols import MemoryKind, MemoryScope
    # v2 P3a：body = TASK 视图默认 kinds（对话+段摘要，跨 task 半址）；frames = AGENT 视图
    # conversation turn（AGENT_COMPACT_SUMMARY = SUMMARY kind，天然排除）。
    body_records = await load_records(memory,
        MemoryAddress(session_id=session_id, agent_id=source_agent_id),
        MemoryScope.TASK, ctx,
    )
    frame_records = await load_records(memory,
        MemoryAddress(session_id=session_id, agent_id=source_agent_id),
        MemoryScope.AGENT, ctx, kinds=[MemoryKind.CONVERSATION_TURN],
    )
    from ctx_weft.core.assembler.builder import history_records
    combined = history_records([*body_records, *frame_records])
    child_scope = MemoryAddress(session_id=session_id, task_id=child_task.id, agent_id=sub_agent.id)
    for r in combined:  # chronological → re-ingest preserves order via fresh per-scope seq_no
        # `inherited_from_agent_id` 恒写（真正的来源标识）；`inherited_from_task_id`
        # 只在调用方给了源 task 时附上——显式锚定 agent 的路径没有「那条 task」可言。
        md: dict = {"inherited_from_agent_id": source_agent_id}
        if source_task_id:
            md["inherited_from_task_id"] = source_task_id
        if r.role == "assistant" and r.metadata.get("tool_calls"):
            md["tool_calls"] = r.metadata["tool_calls"]
        if r.role == "tool" and r.metadata.get("tool_call_id"):
            md["tool_call_id"] = r.metadata["tool_call_id"]
        await memory.ingest(
            MemoryEvent(
                # 确定性 id：同一份源记录复制给同一个子 scope 恒是同一条。丢弃一轮会把
                # `user_prompt_in_memory` 还原（`discard_round`），而这份复制的门就挂在
                # 那个标志上——随机 id 的话，撤销之后重跑会把整段继承记忆再抄一遍。
                id=f"inherit:{child_task.id}:{r.id}" if r.id else None,
                kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
                address=child_scope,
                content=r.content,
                timestamp=r.timestamp,
                role=r.role,
                metadata=md,
                blob_refs=r.blob_refs,
            ),
            ctx,
        )


# ── ProviderRegistry ──────────────────────────────────────────────────────────


# ── SessionStartParams ────────────────────────────────────────────────────────


@dataclass
class SessionStartParams:
    """Parameters for start_session(). Construct via SessionStartParams.create().

    resume=False → create a new session. session_id=None lets runtime generate the ID;
                   session_id=<id> creates a NEW session with a host-provided ID
                   (so the host can pre-register session-scoped resources, e.g. the
                   filesystem provider's workspace, before execution starts).
    resume=True  → resume an existing session (root_agent_id recovered from event store);
                   session_id must be set.
    """

    template_id: str
    user_prompt: "str | list[ContentPart]"
    initial_task_settings: NormalTaskSettings
    session_id: str | None = None
    tenant_id: str = "default"
    llm_account: str | None = None
    llm_model: str | None = None
    llm_config: ModelCallConfig | None = None
    resume: bool = False
    # 这一轮没有人看顾（后台自治作业）：透传到 root task 的 `Task.unattended`，并强制
    # 它不 park。见 `Task.unattended` / `HitlService.open`。
    unattended: bool = False
    # 这一轮接在 session 的哪个交互口上（见 `Task.port_key`）。None = 未声明，按
    # `unattended` 回落；显式给口才会产出「接口但自治」那一格。
    port_key: str | None = None

    @classmethod
    def create(
        cls,
        template_id: str,
        user_prompt: "str | list[ContentPart]",
        *,
        session_id: str | None = None,
        initial_task: dict | None = None,
        tenant_id: str = "default",
        llm_account: str | None = None,
        llm_model: str | None = None,
        llm_config: ModelCallConfig | None = None,
        resume: bool = False,
        unattended: bool = False,
        port_key: str | None = None,
    ) -> "SessionStartParams":
        from ctx_weft.core.models.task import deserialize_settings
        return cls(
            template_id=template_id,
            user_prompt=user_prompt,
            session_id=session_id,
            initial_task_settings=deserialize_settings(initial_task),
            tenant_id=tenant_id,
            llm_account=llm_account,
            llm_model=llm_model,
            llm_config=llm_config,
            resume=resume,
            unattended=unattended,
            port_key=port_key,
        )


# ── TurnHandle ────────────────────────────────────────────────────────────────


@dataclass
class TurnHandle:
    """一次外部交互的句柄：**agent + task 两个轴**（2026-09-04 spec §3）。

    四个身份字段恒非空：

    - ``agent_id`` —— 被寻址的 agent。`start_session` 给的是该 session 的 **root
      agent**（`session.root_agent_id`，在 `SESSION_CREATED` 之前铸好，见
      `SessionRegistry.create_session` / `resume_session`）；`send_message` 给的是
      调用方点名的那个；`run_single_task` 给的是执行那一条 task 的 agent。
      直接传给 `send_message(agent_id, ...)` 或 `get_agent(agent_id)`。
    - ``task_id`` —— 这次交互落到的 task。

    **不含 ``run_id``。** run 是引擎内部一轮循环的相关性 id，句柄不需要它：
    `events()` 按 agent + task 订阅，`wait_for_finish()` 等 task 终态。host 若要按轮
    聚合，每条事件的信封里都带 `run_id`，直接读。放进句柄反而会多一个无法诚实填写的
    字段——`send_message` 的「注入且不重排」分支在返回那一刻确实还没有新一轮。
    """

    session_id: str
    agent_id: str
    task_id: str
    template_id: str
    event_bus: EventBus
    _state: LoopState | None = None
    # 存储不可用健康查询（spec: event-commit）：runtime 构造时注入；None（测试替身）
    # 表示无健康面，wait_for_finish 按旧行为等超时。
    _storage_health: "Callable[[str], str | None] | None" = None
    _storage_recovery: Any = None
    _pending_observe: "Callable[[str, str], str | None] | None" = None
    _is_finished: Any = None

    async def events(self) -> AsyncIterator[Event]:
        from ctx_weft.protocols.events import EventFilter
        async for ev in self.event_bus.stream(
            EventFilter(agent_id=self.agent_id, task_id=self.task_id)
        ):
            yield ev

    async def wait_for_finish(self, timeout: float = 300.0) -> LoopState | None:
        """Wait for task termination and memory settlement; raise TimeoutError on expiry.

        LoopState remains a compatibility result. Use runtime.get_task for stable
        results on handles returned by send_message or dispatch_task.
        """
        from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT

        terminal = {et for et, status in TASK_STATUS_BY_EVENT.items()
                    if status in TERMINAL_TASK_STATUSES}
        queue = asyncio.Queue()

        async def receive(event):
            if (event.agent_id == self.agent_id and event.task_id == self.task_id
                    and (event.type in terminal or event.type == EventType.OBSERVE_APPLIED)):
                queue.put_nowait(event)

        # Subscribe synchronously before the first await; no late-subscription gap.
        subscription = self.event_bus.subscribe(None, receive)
        terminal_seen = False
        seen_recaps = set()
        try:
            async with asyncio.timeout(timeout):
                while True:
                    if self._storage_health is not None:
                        reason = self._storage_health(self.session_id)
                        if reason is not None:
                            if self._storage_recovery is not None:
                                await self._storage_recovery(self.session_id)
                            else:
                                raise PersistenceUnavailableError(reason)
                    if self._is_finished is not None and await self._is_finished(
                            self.session_id, self.task_id):
                        # A commit can fail during the awaited state query.
                        # Durable terminal rows alone do not clear unknown receipts.
                        if self._storage_health is not None and self._storage_health(self.session_id):
                            if self._storage_recovery is not None:
                                await self._storage_recovery(self.session_id)
                                continue
                            raise PersistenceUnavailableError(self._storage_health(self.session_id))
                        return self._state
                    if terminal_seen and self._is_finished is None:
                        recap = (self._pending_observe(self.session_id, self.task_id)
                                 if self._pending_observe else None)
                        if recap is None or recap in seen_recaps:
                            return self._state
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.25)
                    except TimeoutError:
                        continue
                    if event.type in terminal:
                        terminal_seen = True
                    elif event.type == EventType.OBSERVE_APPLIED:
                        seen_recaps.add(event.run_id)
        finally:
            await subscription.unsubscribe()


async def _task_has_dangling_tool_call(memory, scope, provider_ctx) -> bool:
    """该 scope 最近一个 assistant turn 是否存在「有 tool_call、无 TOOL_RESULT」（spec/07 §6）。"""
    from ctx_weft.core.loop.steps.reconcile import _dangling_tool_calls
    dangling, _ = await _dangling_tool_calls(memory, scope, provider_ctx)
    return bool(dangling)


# ── SessionHandle ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SessionHandle:
    """`create_session` 的返回：一条**容器会话**的两个身份（spec/09 §11）。

    **纯值对象，刻意没有 `wait_for_finish()` / `events()`。** `TurnHandle` 是「一次
    交互」的句柄——它有一条 task 可等、有一条事件流可订。容器会话建出来时里面一件活
    都没有，没有可等的东西；要等就等 `dispatch_task` 返回的那个 `TurnHandle`。

    ``root_agent_id``：这条会话的 root agent，已实例化且 `idle`。它存在但手上没有对话
    ——恢复链上「`root_agent_id` 非空」那条假设因此不破（`resume_session` 会拒 root 为
    空的投影）。想让活直接落在它头上就 `dispatch_task(..., agent_id=root_agent_id)`；
    想另起一棵树就不传 `agent_id`。
    """

    session_id: str
    root_agent_id: str
    template_id: str


# ── CtxWeftRuntime ─────────────────────────────────────────────────────────────


def _reply_turn_agent_id(req: "PendingHitl", target: "Task") -> str:
    """人的答复该写进**哪个 agent scope**。写入（`_write_hitl_reply_turn`）与「查它写没写过」
    （`_injected_reply_ids`）共用这一份，两边不得各推一次。

    必须是本 task 对话真正所在的 agent scope——`AgentRecallSource` 用
    `recall_recent_by_agent` 按 `scope.agent_id` 过滤召回 task body。冷重启从旧事件日志
    重建的 HITL 丢了 agent_id（legacy `HITL_REQUIRED` 投影未持久化它），`req.agent_id=""`
    会把回复写进空 agent scope → 对 actor 装配不可见 → 重启后「第一句」丢失。故回退到
    task 的真实 agent（assigned/creator），与首条 USER_PROMPT 落库时同 scope。
    """
    return (req.agent_id
            or getattr(target, "assigned_agent_id", "")
            or getattr(target, "creator_agent_id", "")
            or "")


def _suspended_on_live_children(task_manager: "TaskManager", task: "Task") -> bool:
    """该 task 是否「SUSPENDED 且尚有未终态的子任务」——即等着 `_try_resume_parent` 唤醒。

    这是 `restore` 唯一**刻意不重排**的形状（见 `TaskManager.restore` 的注释）：它必须
    保持 SUSPENDED，否则 `_try_resume_parent` 的 `status == "SUSPENDED"` 门失效，子任务
    收尾时那次唤醒静默消失，父任务永久停摆。
    """
    if task.status != "SUSPENDED":
        return False
    for cid in task_manager.children_of(task.id):
        child = task_manager.get_task(cid)
        if child is None or child.status not in TERMINAL_TASK_STATUSES:
            return True
    return False


def runtime_command(method):
    @functools.wraps(method)
    async def command(self, *args, **kwargs):
        current = asyncio.current_task()
        nested = current in self._commands
        if self._closing and not nested:
            raise RuntimeClosedError("Runtime is closing")
        self._commands.add(current)
        try:
            return await method(self, *args, **kwargs)
        finally:
            if not nested:
                self._commands.discard(current)
    return command



def _session_command(
    *,
    load_session: Literal["required", "optional", "skip"],
    allow_missing_session: bool = False,
    missing_session_as_agent: bool = False,
    skip_no_resume_reply: bool = False,
    skip_finalized_stop: bool = False,
    recover_storage_first: bool = False,
):
    """Serialize addressing, optional materialization and command delivery.

    required raises on a missing session; optional uses non-strict rebuilding;
    skip operates only on loaded state. allow_missing_session delegates unresolved
    addresses to the command. missing_session_as_agent preserves agent-facing
    lookup errors. skip_no_resume_reply avoids loading for notification replies.
    skip_finalized_stop rejects protected stops before initialization and locking;
    stop commands still recheck admission after taking the session lock.
    """
    if load_session not in {"required", "optional", "skip"}:
        raise ValueError(f"unknown session loading policy: {load_session!r}")

    def decorate(method):
        signature = inspect.signature(method)

        @functools.wraps(method)
        async def command(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs).arguments
            sid = bound.get("session_id")
            aid = bound.get("agent_id")
            if aid:
                rec = self._agent_lifecycle_manager.record_of(aid)
                if rec is not None:
                    if sid is not None and sid != rec.session_id:
                        raise ValueError("agent_id does not belong to session_id")
                    sid = rec.session_id
            reply = bound.get("reply")
            if reply is not None:
                pending = self.hitl_registry.get(reply.hitl_id)
                if pending is not None:
                    if sid is not None and sid != pending.session_id:
                        raise ValueError("HITL request does not belong to session_id")
                    sid = pending.session_id
            if sid is None:
                if allow_missing_session:
                    return await method(self, *args, **kwargs)
                raise AgentNotLoaded(
                    "pass session_id for a cold command, or call rebuild_session(session_id) first")

            def stop_is_protected():
                return skip_finalized_stop and (
                    self._is_agent_writing_task_finalized(aid) if aid else
                    self._is_writing_task_finalized(sid))

            # A stop received during TaskFinalized belongs to this write, not a
            # future run. Do not let initialization or serialization queue it.
            if stop_is_protected():
                return [] if aid else False
            if (recover_storage_first and self.storage_health(sid) is not None
                    and self._session_lock_owners.get(sid) is not asyncio.current_task()):
                # Storage reconciliation may wait for emitters whose subscribers
                # query this session. Complete it before owning the command guard.
                if not await self.recover_storage(sid):
                    raise PersistenceUnavailableError(self.storage_health(sid))
            # Nested commands already run under the outer command's calibration
            # check. In particular, an optional missing-session purge must not
            # spawn an initialization task that waits for its own caller's lock.
            if self._session_lock_owners.get(sid) is not asyncio.current_task():
                try:
                    await self.initialize(sid)
                except SessionNotFound:
                    if load_session not in {"optional", "skip"}:
                        if missing_session_as_agent:
                            raise AgentNotFound(f"unknown agent {aid!r} in session {sid!r}") from None
                        raise
            if stop_is_protected():
                return [] if aid else False
            async with self._session_guard(sid):
                if load_session == "skip":
                    return await method(self, *args, **kwargs)
                if (skip_no_resume_reply and reply is not None and pending is not None
                        and isinstance(pending.delivery, NoResumeDelivery)):
                    return await method(self, *args, **kwargs)
                if load_session == "optional":
                    await self._rebuild_session_locked(sid, strict=False)
                else:
                    try:
                        await self._rebuild_session_locked(sid, strict=True)
                    except SessionNotFound:
                        if missing_session_as_agent:
                            raise AgentNotFound(f"unknown agent {aid!r} in session {sid!r}") from None
                        raise
                return await method(self, *args, **kwargs)

        return runtime_command(command)

    return decorate


@dataclass
class _ActorRunCommit:
    """Original actor envelopes retained until their commits are confirmed."""

    started: Event
    start_confirmed: bool = False
    finished: Event | None = None


class CtxWeftRuntime:
    """Top-level runtime.

    Supports two modes:
    - run_single_task(): single-task testing convenience entry point
    - start_session(): Phase 4 orchestration; session_id=None creates, session_id=<id> resumes
    """

    def __init__(
        self,
        *,
        providers: ProviderRegistry | None = None,
        llm: LLMClient | None = None,
        event_bus: EventBus | None = None,
        event_store: "Any | None" = None,
        config: "RuntimeConfig | None" = None,
        snapshot_every_n: int = 0,
    ) -> None:
        from ctx_weft.core.models.config import RuntimeConfig
        self._closing = False
        self._close_task = None
        self._commands = set()
        self._owned_tasks = set()
        self._config = config or RuntimeConfig()
        self._llm = llm  # fallback for backward compat / tests
        self.providers = providers or ProviderRegistry()
        # 默认实现只在 host 没给时才解析——避免「默认」从运行期选择退化成 import 期耦合。
        if event_bus is None:
            from ctx_weft.providers.events import InProcessEventBus
            event_bus = InProcessEventBus()
        self._event_bus: EventBus = event_bus
        if self._config.event_commit_policy not in ("required", "best_effort"):
            raise ValueError("event_commit_policy must be 'required' or 'best_effort'")
        if (self._config.event_commit_policy == "required"
                and not callable(getattr(event_bus, "attach_commit_gate", None))):
            raise ValueError("required event bus must implement attach_commit_gate")
        # HITL：构造期一次性接线，**没有 setter、没有半成品窗口**。裸构造即生产形态
        # （spec 2026-09-01 重设计 §3）。`_normalize_hitl_content` 是与 start_session /
        # run_single_task 共用的内容校验 + 外部化方法（Phase 3c Task A），此处原样接进
        # `ReplyIntake` 作为其 `ContentNormalizer`。
        self.hitl_registry = HitlRegistry(max_resolved=self._config.hitl_max_resolved)
        self.hitl = HitlService(
            registry=self.hitl_registry,
            event_bus=self._event_bus,
            reply_intake=ReplyIntake(self._normalize_hitl_content),
        )
        self._hitl_timeout_sec = self._config.hitl_timeout_sec
        if event_store is None:
            from ctx_weft.providers.events import InMemoryEventStore
            event_store = InMemoryEventStore()
        self.event_store = event_store
        # 有序提交是 EventStore 的必需部分（spec: event-log），**与提交策略无关**：
        # 恢复路径（rebuild_view / SnapshotWriter）无条件走 position 一致切面，没有
        # 按 ID 排序的回落分支。这里做的是契约校验而非能力协商——Protocol 的
        # @abstractmethod 只拦得住显式继承的实现，鸭子类型 store 缺方法要到第一次
        # 提交才炸，那时错误已经离现场很远。
        from ctx_weft.protocols.events import supports_ordered_commit
        if not supports_ordered_commit(self.event_store):
            raise ValueError(
                f"EventStore {type(self.event_store).__name__} 未实现有序提交："
                "append_batch / read_range / committed_head 三者必须齐全（spec: event-log）。"
                "按事件 ID 排序的 store 不是「功能少一点」，而是恢复语义错误——延迟提交的"
                "事件会永久落在快照游标之外，两条恢复路径给出不同的世界且不报错"
                "（可靠性方案 H2）。内置 InMemoryEventStore / SqlEventStore 均已实现；"
                "自定义 store 请参照 tests/unit/test_ordered_event_store_conformance.py。")
        # 这里从前还有一道 `supports_replay` 门：协议上的 `replay` 是「能用的默认实现」，
        # 鸭子类型 store 继承不到，所以要在构造期兜一把。2026-09-21 `replay` 搬进 core
        # （`reducers.replay_session`，只用 read_range + committed_head 两个必需原语）之后，
        # 那道门没有存在的理由了——上面那道校验已经保证这两个原语齐全。
        # 存储不可用健康表（spec: event-commit）：session_id → 原因。CommitGate 失败时
        # **先标记后抛**；公开查询走 storage_health()。内存态——崩溃后由持久日志重建。
        self._storage_unavailable: dict[str, str] = {}
        self._storage_recovery_tasks: dict[str, asyncio.Task] = {}
        self._storage_retryable: set[str] = set()
        self._pending_tool_results: dict = {}
        self._unstarted_tools: dict = {}
        self._storage_interrupted_tools: set = set()
        self._commit_gate = None
        self._snapshot_every_n = snapshot_every_n
        # Auto-register 内置 providers（与用户注册的 providers 无关）
        control_provider = ControlCapabilityProvider()
        self.providers.register_capability(control_provider)

        from ctx_weft.core.capabilities.skill_executor import (
            SkillExecutorCapabilityProvider,
        )
        skill_executor = SkillExecutorCapabilityProvider(self.providers)
        self.providers.register_capability(skill_executor)

        # media:get_image —— 取回被 L0.5 降级掉的图（子设计 §9）。core 侧 provider：
        # 工具体要读 task 视图算「原本挂在第几条 user 回合」，普通 capability provider
        # 拿不到 MemoryProvider。memory / blob store 由它在调用时经 registry 解析，
        # 故此处不要求它们已注册。
        from ctx_weft.core.media.capability import MediaCapabilityProvider
        self.providers.register_capability(MediaCapabilityProvider(self.providers))

        # 模板通道硬校验（spec 2026-07-22）：模板进入 core 的唯一通道是
        # AgentCapabilityProvider；缺失则 root agent 都无法实例化，构造即失败。
        agent_provider_names = [
            p.name for p in self.providers.get_capability_providers()
            if isinstance(p, AgentCapabilityProvider)
        ]
        if not agent_provider_names:
            raise ValueError(
                "CtxWeftRuntime requires at least one AgentCapabilityProvider in the "
                "ProviderRegistry — register one before constructing, e.g. "
                "providers.register_capability(LocalAgentTemplateProvider(templates_dir))"
            )
        from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
        self._template_lookup = TemplateLookup(
            self.providers, self._config.fallback_template_ref)
        # Agent 注册表：runtime 级长生命周期组件，_agents 是 agent 身份与配置的唯一住所。
        # 从前 AgentLifecycleManager 是每次调用 new 一个的临时对象，见
        # docs/events-v2.md §2.1.1（与 SessionRegistry 同形的那次晋升）。
        # model_resolver=self._resolve_llm：registry 现解不缓存（那是 LLMClientResolver
        # 的职责），构造期注入、无默认值——单测跑的和生产跑的必须是同一个东西
        # （core/hitl/reply_intake.py docstring 的既有立场，Task 4 因默认 bus 判过一次
        # Critical，这里不重蹈）。self._resolve_llm 已带好「无 provider 时回落 self._llm」
        # 那条分支，无需在此重复。
        self._agent_lifecycle_manager = AgentLifecycleManager(
            template_lookup=self._template_lookup, event_bus=self._event_bus,
            model_resolver=self._resolve_llm)
        # ALM 的输入端：只认 TASK_*（_INPUT_BY_EVENT），发 AGENT_*。与
        # SessionRegistry.attach_to_bus 之间无顺序依赖——两者各订各的事件类型，互不
        # 消费对方发出的事件（docs/events-v2.md §2.1.1，Task 12）。

        # Capability cache (per-session, shared across all agents in runtime)
        self._capability_cache = CapabilityCachePool()

        # Per-run 控制信号 registry：session_id → {task_id → RunTokens}。随派发登记、随 run
        # 注销（_SessionTaskRunner.execute）——pause/cancel 经 registry 必达全部在途 run
        # （spec 2026-07-05）。
        self._run_tokens: dict[str, dict[str, RunTokens]] = {}
        # Execution lifecycle only: never background runs or cancellation intents.
        self._pending_actor_runs: dict[tuple[str, str], _ActorRunCommit] = {}
        # pause 弃子进行中的 session：新派发 run 的 PauseToken 出生即 paused
        # （root agent 任务被重排后，新 run 在 act 首个 checkpoint 立即 park）。
        self._pausing: set[str] = set()
        # 本轮暂停的续跑点名额（一次性）：pause_session pause 到在途 root run、或闩锁窗口内
        # 第一个 root run born-pause 时认领；此后窗口内再派发的 root run 一律 born-cancel。
        # 与 _pausing 同生命周期（_on_idle / _release_round / pause_session 兜底一起清）。
        self._pause_claimed: set[str] = set()
        # compact 等一次性操作的忙位（原先借 _cancel_tokens dict 占位）。
        self._busy_sessions: set[str] = set()
        self._task_managers: dict[str, TaskManager] = {}
        self._step_executions: dict[str, ExecutionRegistry] = {}
        # Cold owners are loaded without a queue. Recovery work starts only after
        # the caller has delivered its input; this is not an agent/task status.
        self._prepared_recoveries: dict[str, tuple[TaskManager, int, dict]] = {}
        # 会话成员登记表。生命周期事实属于 task / agent；会话展示状态由宿主聚合。
        self._session_registry = SessionRegistry(
            agent_lifecycle_manager=self._agent_lifecycle_manager,
            event_bus=self._event_bus,
            task_max_concurrent=self._config.task_max_concurrent,
            task_max_retries=self._config.task_max_retries,
            autonomous_requeue_max=self._config.autonomous_requeue_max,
            autonomous_requeue_backoff_base_sec=(
                self._config.autonomous_requeue_backoff_base_sec),
        )
        # SM 的输入端：只认 TaskManager 的四类事件（_INPUT_BY_EVENT），本 task
        # 之后 TM 还没开始发这三条信号，运行时行为不变（docs/events-v2.md §2.1.1）。
        # Per-session resume 锁：串行化同一 session 的 recover_session，避免重叠的冷 HITL 应答 /
        # /resume 并发建出两个 TaskManager、两套 drain 竞争派发（spec/07 §9）。惰性建、不回收
        # （体量微小、按 session 数有界）。
        self._resume_locks: dict[str, asyncio.Lock] = {}
        self._assistant_memory_locks: dict[str, asyncio.Lock] = {}
        self._assistant_agent_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        self._assistant_pending: dict[tuple[str, str, str], dict[str, _PendingAssistantMemory]] = {}
        self._assistant_memory_ready: dict[tuple[str, str], tuple[MemoryProvider, int]] = {}
        self._memory_seed_ready: dict[tuple[str, str], MemoryProvider] = {}
        # Frozen envelopes survive cancellation/unknown receipts in this process.
        self._fork_creations: dict[str, tuple[str, tuple[Event, ...]]] = {}
        self._session_lock_owners: dict[str, asyncio.Task] = {}
        self._loaded_sessions: set[str] = set()
        self._initialized_sessions: set[str] = set()
        self._initializing_sessions: dict[str, asyncio.Task] = {}
        # Wiring is synchronous and does no history I/O. Recovery is per session.
        self._attach_runtime()

    def _spawn(self, operation, **kwargs):
        task = asyncio.create_task(operation, **kwargs)
        self._owned_tasks.add(task)

        def finished(done):
            self._owned_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(finished)
        return task

    async def close(self, *, timeout: float = 30.0) -> ShutdownReport:
        """Stop this instance, retaining committed tasks/HITL for a fresh runtime.

        Concurrent callers share shutdown. Timeout or caller cancellation never
        abandons cleanup. A timeout report does not authorize closing providers;
        await close again until execution is quiescent. Queries remain available.
        """
        current = asyncio.current_task()
        if (current in self._commands or current in self._owned_tasks
                or any(current in tm.owned_executions() for tm in self._task_managers.values())):
            raise RuntimeError("close must be called outside runtime execution")
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_instance())
            self._close_task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        done, _ = await asyncio.wait({self._close_task}, timeout=timeout)
        if not done:
            raise RuntimeShutdownError(ShutdownReport(
                quiescent=False, durable=False, errors=("Shutdown timed out; cleanup is still running",)))
        return self._close_task.result()

    async def _close_instance(self) -> ShutdownReport:
        from ctx_weft.core.control.startup import calibrate_session
        from ctx_weft.core.models.errors import SessionNotFound

        # Snapshot every owner before cancellation callbacks can remove entries.
        managers = dict(self._task_managers)
        sessions = set(managers) | set(self._initializing_sessions)
        sessions.update(self._storage_unavailable)
        errors = []
        while True:
            managers.update(self._task_managers)
            sessions.update(managers)
            for manager in managers.values():
                manager.begin_shutdown()
            tasks = set(self._owned_tasks) | set(self._commands)
            for manager in managers.values():
                tasks.update(manager.owned_executions())
            for registry in self._step_executions.values():
                tasks.update(registry.tasks.values())
            tasks = {task for task in tasks if not task.done()}
            if not tasks:
                break
            for task in tasks:
                task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors.extend(repr(r) for r in results
                          if isinstance(r, BaseException) and not isinstance(r, asyncio.CancelledError))
        # All producers have exited. Confirm submissions whose receipt was lost
        # before dropping provisional buffers; never turn an uncommitted round
        # into a committed one merely because the process is shutting down.
        for sid in sorted(sessions):
            try:
                await self._confirm_storage_facts(sid)
                await self.hitl.confirm_opens(sid)
                await self.hitl.confirm_resolutions(sid)
                manager = managers.get(sid)
                if manager is not None:
                    for tid in list(manager.open_round_task_ids):
                        if manager._round_commit_started(tid):
                            await manager.commit_round(tid)
                    await self._confirm_assistant_events(manager.session)
                    await self._settle_storage_assistant_memory(manager.session)
                registry = self._step_executions.get(sid)
                if registry is not None:
                    await registry.confirm_metadata_commits(self._event_bus)
                self._evict_session_memory(sid)
                await calibrate_session(self.event_store, sid, reason="runtime_shutdown",
                                        publish=self._event_bus.emit_batch)
                # Cancellation may interrupt an append and temporarily isolate the
                # session. All receipts and shutdown facts are now confirmed;
                # read-only queries after close must see the recovered facts.
                if not self.clear_storage_isolation(sid):
                    raise RuntimeError("Shutdown left unconfirmed storage submissions")
            except SessionNotFound:
                pass  # A create command can be stopped before its first commit.
            except Exception as exc:
                errors.append(f"{sid}: {exc!r}")
        for detach in (self.persistence.detach, self._agent_lifecycle_manager.detach_from_bus,
                       self._session_registry.detach_from_bus):
            try:
                await detach()
            except Exception as exc:
                errors.append(f"detach: {exc!r}")
        if sessions and self._config.event_commit_policy != "required":
            errors.append("best_effort event commits cannot guarantee recoverable shutdown")
        report = ShutdownReport(True, not errors, tuple(sorted(sessions)), tuple(errors))
        if errors:
            raise RuntimeShutdownError(report)
        return report

    @runtime_command
    async def initialize(self, session_id: str) -> SessionInitializationResult | None:
        """Autonomously recover one session before execution, without loading owners.

        Concurrent callers share one task, shielded from waiter cancellation.
        Return the recovery report to those callers; subsequent calls return None.
        Only successful session IDs survive completion, never tasks or reports.
        Read-only queries do not require initialization or trigger crash repairs.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if session_id in self._initialized_sessions:
            return None
        task = self._initializing_sessions.get(session_id)
        if task is None:
            task = self._spawn(self._initialize_session_once(session_id))
            self._initializing_sessions[session_id] = task

            def completed(done):
                if self._initializing_sessions.get(session_id) is done:
                    self._initializing_sessions.pop(session_id, None)
                # Retrieve failures even when every waiter disconnected.
                if not done.cancelled():
                    done.exception()

            task.add_done_callback(completed)
        result = await asyncio.shield(task)
        return deepcopy(result)

    async def _initialize_session_once(self, session_id: str) -> SessionInitializationResult | None:
        from ctx_weft.core.control.startup import calibrate_session

        async with self._session_guard(session_id):
            # A concurrent creation may have initialized this session while we waited.
            if session_id in self._initialized_sessions:
                return None
            result = await calibrate_session(self.event_store, session_id)
            self._initialized_sessions.add(session_id)
            return result

    def _attach_runtime(self) -> None:
        # 提交策略分岔（spec: event-commit，change reliability-wp3）：
        # - required（默认）：CommitGate 接进 emit 路径（提交确认先于通知）；persister
        #   不再接线；SnapshotWriter（若启用）单独接——它消费的已是确认提交流。
        # - best_effort：旧 attach_snapshotting 路径（吞存储错误），启动告警、不可靠恢复。
        policy = self._config.event_commit_policy
        if policy not in ("required", "best_effort"):
            raise ValueError(
                f"event_commit_policy must be 'required' or 'best_effort', got {policy!r}")
        if policy == "required":
            attacher = getattr(self._event_bus, "attach_commit_gate", None)
            if not callable(attacher):
                raise ValueError(
                    "event_commit_policy='required' 需要支持提交门的事件总线（实现 "
                    "EventBus.attach_commit_gate：emit/commit_provisional 在 fanout 前 "
                    "先经 gate 确认存储提交）。InProcessEventBus 已支持；自定义总线请实现"
                    "该扩展，或显式配置 event_commit_policy='best_effort' 并接受丢事件风险。")
            from ctx_weft.core.events.commit_gate import CommitGate
            self._commit_gate = CommitGate(self.event_store,
                on_unavailable=self._mark_storage_unavailable,
                max_attempts=self._config.storage_commit_max_attempts,
                base_delay=self._config.storage_recovery_base_delay_sec)
            self._event_bus.attach_commit_gate(self._commit_gate)
            attach_recovery = getattr(self._event_bus, "attach_storage_recovery", None)
            if attach_recovery is not None:
                attach_recovery(self._wait_for_commit)
            from ctx_weft.providers.events import PersistenceHandle
            writer = None
            if self._snapshot_every_n > 0:
                from ctx_weft.core.control.snapshot_writer import SnapshotWriter
                writer = SnapshotWriter(self.event_store, self._event_bus,
                                        every_n_events=self._snapshot_every_n,
                                        memory_settled=self._memory_settled,
                                        get_memory=self.providers.get_memory)
            # handle 统一暴露：required 模式 persister=None（提交走 gate）、writer 可 detach。
            self.persistence = PersistenceHandle(None, writer)
        else:
            logger.warning(
                "event_commit_policy='best_effort'：存储失败将被吞掉（不可靠恢复）；"
                "仅建议显式接受丢事件的观测用途。")
            # 旧路径（spec 2026-08-29 §6.4 + final review R15）：persister 必须先于
            # snapshot writer 订阅；handle 存公开属性 persistence 供宿主 detach。
            from ctx_weft.core.control.snapshot_writer import attach_snapshotting
            self.persistence = attach_snapshotting(
                self._event_bus, self.event_store, every_n=self._snapshot_every_n,
                memory_settled=self._memory_settled, get_memory=self.providers.get_memory)

        self._agent_lifecycle_manager.attach_to_bus()
        self._session_registry.attach_to_bus()

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        return self._resume_locks.setdefault(session_id, asyncio.Lock())

    @contextlib.asynccontextmanager
    async def _session_guard(self, session_id: str):
        # Reentry is allowed only for this asyncio task, never an inherited context.
        current = asyncio.current_task()
        if self._session_lock_owners.get(session_id) is current:
            yield
            return
        async with self._session_lock(session_id):
            self._session_lock_owners[session_id] = current
            try:
                yield
            finally:
                self._session_lock_owners.pop(session_id, None)

    def _session_is_loaded(self, session_id: str) -> bool:
        tm = self._task_managers.get(session_id)
        return (session_id in self._loaded_sessions and tm is not None
                and tm.session is not None
                and bool(self._agent_lifecycle_manager.record_of(tm.session.root_agent_id or "")))

    @property
    def event_bus(self) -> EventBus:
        return self._event_bus

    def _resolve_llm(
        self,
        llm_account: str | None = None,
        llm_model: str | None = None,
        *,
        ctx: ProviderContext,
    ) -> LLMClient:
        """Resolve LLMClient: registry first, then fallback to self._llm."""
        if self.providers.has_llm_provider():
            return self.providers.get_llm_provider().get_client(llm_account, llm_model, ctx=ctx)
        if self._llm is not None:
            return self._llm
        raise RuntimeError(
            "No LLM available. Register an LLMProvider via providers.register_llm_provider() "
            "or pass llm= to CtxWeftRuntime."
        )

    async def _validate_and_normalize_content(
        self,
        content: "str | list[ContentPart]",
        session_id: str,
        *,
        tenant_id: str = "default",
    ) -> "tuple[str | list[ContentPart], str | list[dict] | None]":
        """入口内容校验 + **双侧外部化**的单一真源（三个入口共用）。

        返回 `(memory 侧归一化内容, event 侧 jsonable)`。两个产物都从**同一份原始
        content** 派生，各自只碰一个 store：memory 侧走 `normalize_content`，event
        侧走 `content_to_event_jsonable`。两个 ref 不必相同，core 也不比较它们——
        两个契约独立、ref 命名空间互不相通（2026-08-28 解耦方案）。

        **event 侧必须先做**：它要的是归一化**之前**的原始字节。等 `normalize_content`
        把图片 part 改写成 memory ref 之后再喂给 event 侧，拿到的只会是一个 event
        store 永远打不开的引用（`content_to_event_jsonable` 会把它降级成文本占位并
        告警——图片就在事件流里丢了）。

        顺序恒为 `validate_content` → 两侧外部化，理由两条（spec §6.1）：
        (a) 被拒的内容不该在任何 blob store 里留下垃圾——校验失败必须发生在任何 `put`
        之前；(b) `normalize_content` 里的 `b64decode(..., validate=True)` 刻意不加
        try/except，靠 validate 先行把畸形 base64 拦成 `InvalidContentError`。抽成
        这一个方法之后，三处调用点的顺序不会再各自漂移。

        **(a) 有一个诚实的例外**：event 侧先 put、memory 侧后 put，故 memory 侧
        `normalize_content` 失败（blob store 报错等）时，本方法带着异常返回，而字节
        **已经**落进了 event blob store——留下一份无人引用的孤儿。这个顺序是被「event
        侧要的是归一化之前的原始字节」硬性决定的（见上），不为此改序、也不加补偿删除
        （删除本身会失败、且 `EventBlobStore` 协议里没有 delete）。孤儿的回收归宿主的
        event blob 保留策略，与「被拒的内容不在 blob store 里留垃圾」相比，这里的口径
        准确说法是：**校验（validate）失败恒不留垃圾；外部化中途失败可能留下 event 侧
        孤儿字节**。

        **不解析 LLM、不判模型能力**（spec 2026-08-28）：模态处置归 `LLMClient`
        实现方，core 全程透传。这也让「纯文本不提前解析 LLM」这条不变量自动成立
        ——本方法根本不碰 LLM。

        memory 侧不能外部化（`NullMemoryBlobStore`）时第一个产物是原样的同一对象；
        纯文本 content 两侧都是零 IO 直通（两个函数对 `str` 都原样返回）。
        """
        from ctx_weft.protocols.content import content_to_event_jsonable, normalize_content, validate_content

        event_blob_store = self.providers.get_event_blob_store()
        validate_content(content, event_blob_store=event_blob_store)
        ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id)
        event_jsonable = await content_to_event_jsonable(
            content, event_blob_store=event_blob_store, ctx=ctx,
        )
        blob_store = self.providers.get_memory_blob_store()
        if not blob_store.can_externalize:
            return content, event_jsonable
        normalized = await normalize_content(content, blob_store=blob_store, ctx=ctx)
        return normalized, event_jsonable

    #: tenant 的缺省值——**core 不推断 tenant**（见 `_load_agents_of` 的说明）。
    _DEFAULT_TENANT = "default"

    async def _normalize_hitl_content(
        self, content: "str | list[ContentPart]", session_id: str, tenant_id: str,
    ) -> "tuple[str | list[ContentPart], str | list[dict] | None]":
        """HITL 应答内容的校验 + 双侧外部化（`ReplyIntake` 的 `ContentNormalizer` 回调）。

        自己什么都不解：校验/外部化由三入口共用的 `_validate_and_normalize_content` 完成
        （顺序恒为 validate → 两侧外部化，不在此重写一遍），返回值原样透传它的二元组
        `(memory 侧内容, event 侧载荷)`——`HitlService` 拿后者直接发 HITL_* 事件。

        `tenant_id` 由 `ReplyIntake` 从 `PendingHitl.tenant_id` 递进来（见
        `ContentNormalizer` 的契约）。改造前这里是拿 `session_id` 去事件日志里**反查**
        tenant 的，而那个字段从请求被开出来的那一刻就一直在手里——反查不只是多一次 IO，
        它还会在「手工构造 / host 直接喂的请求」上解出与请求自带值不同的答案。

        event 侧与 memory 侧同用这一个 tenant：两边的 blob 落在同一个锚点上。

        """
        return await self._validate_and_normalize_content(
            content, session_id, tenant_id=tenant_id or self._DEFAULT_TENANT,
        )

    def _register_run_tokens(
        self, session_id: str, task_id: str, *, root_run: bool = True,
    ) -> RunTokens:
        """为一次派发发放控制信号对；pause 弃子窗口内按执行 agent 分流出生信号：

        - root agent 的 run 出生即 paused → act 首检查点 park 成唯一续跑点。名额一次性
          （_pause_claimed）：root agent 同时有多个任务（后继任务/多条消息）时，第一个
          root run 认领后，窗口内再派发的 root run（如子任务死光被重排的 SUSPENDED root
          任务）一律 born-cancel——"一次暂停恰一个续跑点"。park 中的 run 非终态，其祖先
          不会被 _try_resume_parent 重排，合法等待中的父任务不受此规则误伤。
        - 非 root run 出生即 cancelled（M-1）——检查点 pause 先于 cancel，若两信号齐置，
          多级委派中被 _try_resume_parent 重排的中间父任务会 park 出气泡抢走续跑点；
          born-cancel 使其协作取消，再经 _try_resume_parent 逐级级联到 root agent 的任务。
        """
        tokens = RunTokens(cancel=CancelToken(), pause=PauseToken())
        if session_id in self._pausing:
            if root_run and session_id not in self._pause_claimed:
                self._pause_claimed.add(session_id)
                tokens.pause.pause()
            else:
                tokens.cancel.cancel()
        self._run_tokens.setdefault(session_id, {})[task_id] = tokens
        return tokens

    def _deregister_run_tokens(self, session_id: str, task_id: str, *, expected=None) -> None:
        per = self._run_tokens.get(session_id)
        if per is None:
            return
        if expected is not None and per.get(task_id) is not expected:
            return
        per.pop(task_id, None)
        if not per:
            self._run_tokens.pop(session_id, None)

    # ── 存储不可用健康（spec: event-commit）────────────────────────────────────

    def _mark_storage_unavailable(self, session_id: str, cause: BaseException) -> None:
        """CommitGate 失败回调：先标记后抛（并发路径读到的健康状态与异常一致）。"""
        prev = self._storage_unavailable.get(session_id)
        if prev is None:
            self._storage_unavailable[session_id] = f"{type(cause).__name__}: {cause}"
            logger.error(
                "session %s enters storage_unavailable isolation: %r", session_id, cause)
        from ctx_weft.core.events.commit_gate import retryable_storage_error
        if self._config.storage_recovery_max_attempts and retryable_storage_error(cause):
            self._storage_retryable.add(session_id)
            self._start_storage_recovery(session_id)

    def _start_storage_recovery(self, session_id):
        if self._closing:
            raise RuntimeClosedError("Runtime is closing")
        task = self._storage_recovery_tasks.get(session_id)
        if task is None or task.done():
            task = self._spawn(self._recover_storage_session(session_id),
                                       name=f"storage-recovery:{session_id}")
            self._storage_recovery_tasks[session_id] = task
        return task

    @runtime_command
    async def recover_storage(self, session_id: str) -> bool:
        """Confirm original submissions and resume safely; concurrent callers share one worker.

        False leaves isolation and all original envelopes intact for a later
        explicit retry. Cancellation of a waiter does not cancel shared recovery.
        """
        task = self._storage_recovery_tasks.get(session_id)
        if (self.storage_health(session_id) is None and not self._has_pending_storage(session_id)
                and (task is None or task.done())):
            return True
        return await asyncio.shield(self._start_storage_recovery(session_id))

    async def _wait_for_commit(self, session_id, batch_id):
        task = self._storage_recovery_tasks.get(session_id)
        if task is None:
            if not self._config.storage_recovery_max_attempts or session_id not in self._storage_retryable:
                raise PersistenceUnavailableError(self.storage_health(session_id))
            task = self._start_storage_recovery(session_id)
        receipt = self._spawn(self._commit_gate.wait_confirmation(session_id, batch_id))
        try:
            await asyncio.wait({receipt, task}, return_when=asyncio.FIRST_COMPLETED)
            if receipt.done():
                receipt.result()
                return
            if task.done() and not task.cancelled() and task.result():
                # Full recovery confirmed every receipt before this waiter ran.
                return
            raise PersistenceUnavailableError(self.storage_health(session_id) or "Storage recovery failed")
        finally:
            if not receipt.done():
                receipt.cancel()
            await asyncio.gather(receipt, return_exceptions=True)

    async def _wait_for_storage(self, session_id):
        reason = self.storage_health(session_id)
        if reason is None:
            return
        task = self._storage_recovery_tasks.get(session_id)
        if task is asyncio.current_task():
            raise PersistenceUnavailableError(reason)
        if task is None:
            if not self._config.storage_recovery_max_attempts or session_id not in self._storage_retryable:
                raise PersistenceUnavailableError(reason)
            task = self._start_storage_recovery(session_id)
        if not await asyncio.shield(task) or self.storage_health(session_id) is not None:
            raise PersistenceUnavailableError(self.storage_health(session_id) or reason)

    async def _confirm_storage_facts(self, session_id):
        if self._commit_gate is not None:
            await self._commit_gate.confirm_pending(session_id)
        confirm = getattr(self._event_bus, "confirm_pending_storage", None)
        if confirm is not None:
            await confirm(session_id)
        from ctx_weft.core.loop.capability_gateway import confirm_tool_results
        await confirm_tool_results(self._pending_tool_results, session_id)

    def _has_pending_storage(self, session_id):
        pending = getattr(self._event_bus, "has_pending_storage", None)
        return ((self._commit_gate is not None and self._commit_gate.has_pending(session_id))
                or (pending is not None and pending(session_id))
                or any(entry.session_id == session_id for entry in self._pending_tool_results.values())
                or any(key[1] == session_id and any(not entry.provisional for entry in entries.values())
                       for key, entries in self._assistant_pending.items()))

    async def _recover_storage_session(self, session_id):
        try:
            async with asyncio.timeout(self._config.storage_recovery_max_duration_sec):
                return await self._recover_storage_attempts(session_id)
        except TimeoutError:
            logger.error("Storage recovery timed out for %s; original submissions retained", session_id)
            return False

    async def _recover_storage_attempts(self, session_id):
        for attempt in range(max(1, self._config.storage_recovery_max_attempts)):
            await asyncio.sleep(min(self._config.storage_recovery_base_delay_sec * 2 ** attempt, 5.0))
            try:
                # Do not own the session guard while waiting for a live emitter:
                # its subscribers may query runtime state under that same guard.
                await self._confirm_storage_facts(session_id)
                tm = self._task_managers.get(session_id)
                if tm is not None:
                    await self.hitl.confirm_opens(session_id)
                    await self.hitl.confirm_resolutions(session_id)
                    await self._confirm_assistant_events(tm.session)
                    await self._restore_assistant_messages(tm.session)
                    await self._settle_storage_assistant_memory(tm.session)
                async with self._session_guard(session_id):
                    if tm is not None:
                        await self._recover_in_existing_tm(tm, user_reply=None,
                            resumed_task_id=None, storage_ready=True)
                    elif not self.clear_storage_isolation(session_id):
                        continue
                    return self.storage_health(session_id) is None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Storage recovery attempt %s failed for %s: %r",
                               attempt + 1, session_id, exc, exc_info=True)
        logger.error("Storage recovery exhausted for %s; session remains isolated", session_id)
        return False

    def storage_health(self, session_id: str) -> str | None:
        """None = 健康；否则返回隔离原因（storage_unavailable）。"""
        return self._storage_unavailable.get(session_id)

    def clear_storage_isolation(self, session_id: str) -> bool:
        """宿主确认存储恢复后解除隔离（恢复流程须先按 batch_id 收口未知提交）。"""
        if self._has_pending_storage(session_id):
            return False
        if self._storage_unavailable.pop(session_id, None) is not None:
            self._storage_retryable.discard(session_id)
            for key in list(self._assistant_memory_ready):
                if key[1] == session_id:
                    self._assistant_memory_ready.pop(key, None)
            logger.info("session %s storage isolation cleared", session_id)
        return True

    @_session_command(load_session="skip", skip_finalized_stop=True)
    @runtime_command
    async def pause_session(self, session_id: str) -> bool:
        """软打断（spec 2026-07-05）：放弃其余在途/排队任务，只留 root agent 当前那一轮。
        - 置 _pausing 闩锁：其间新派发的 root agent run 出生即 paused（被 _try_resume_parent
          重排后在 act 首检查点 park，不烧 LLM）——但续跑点名额一次性（_pause_claimed）：
          在途 root run 被 pause 或首个 root run born-pause 即认领，窗口内其后的 root run
          一律 born-cancel；非 root run 出生即 cancelled（多级委派的中间父任务协作取消后
          逐级级联，气泡最终必落 root agent，M-1）。
        - 排队任务全部放弃（abandon_pending：标 CANCELED，不动 session 状态、不封 drain）。
        - 在途 run 按真实执行 agent 划分：== root agent 的那一轮（同 agent 串行 ≤1）pause →
          park 一个 wait 气泡；其余（含被顶替旧 TM 的 inflight）cancel → 协作取消终态。
        - 闩锁由 _on_idle（root park 后会话空闲）或 _release_round 清除。

        不是 `pause_agent` 的广播版：`pause_agent` 级联暂停一个 agent 子树、不杀任何
        东西、对非 running 的子孙静默跳过；这里对非 root agent 做的是取消它们的**在途
        run**（`_cancel_run_token`），run 收尾后 agent 落回 idle、仍然活着——不是把它们
        挨个 `pause_agent` 一遍（2026-09-04 spec §7.1）。
        """
        per = self._run_tokens.get(session_id, {})
        tm = self._task_managers.get(session_id)
        if self._is_writing_task_finalized(session_id):
            return False
        if not per and (tm is None or tm.is_done()):
            return False
        self._pausing.add(session_id)
        root_agent = ""
        # 有轮窗口开着 → **跳过排队弃子**（spec 2026-09-09）。
        #
        # `abandon_pending` 会把队列里每一个 task 标 CANCELED、发 TaskCanceled、再逐个
        # `_try_resume_parent`。那些 task 有自己的 task_id，**不在本轮窗口里**，事件照常
        # 落盘。而本轮马上就要被整体撤销——一次「当作没发生过」的撤销顺手永久杀掉别人
        # 排着的工作，是说不通的。这个窗口只有 TTFT 那么宽，少弃一次子的代价是那些排队
        # 任务在本轮续跑时照常被派发，与用户按暂停之前的预期一致。
        #
        # 窗口没开（普通的运行中暂停）时行为逐字节不变。
        rounds_open = bool(tm.open_round_task_ids) if tm is not None else False
        if tm is not None:
            tm.set_pause_abandon(True)
            root_agent = (tm.session.root_agent_id or "") if tm.session is not None else ""
            if not rounds_open:
                # 保留 root agent 已入队未派发的那一条（keep_agent），其余排队任务弃子。
                await tm.abandon_pending(
                    reason=CancelReason.PAUSE_ABANDON, keep_agent=root_agent or None
                )
        for task_id in list(per):
            if root_agent and tm is not None and tm.running_agent_of(task_id) == root_agent:
                delivered = self._pause_task(session_id, task_id)
                # 在途 root run 即唯一续跑点：认领名额，闩锁窗口内此后派发的 root scope
                # 任务（如被 _try_resume_parent 重排的 SUSPENDED root 任务）born-cancel。
                if delivered:
                    self._pause_claimed.add(session_id)
            else:
                # 非 root：取消它的**在途 run**，不是取消这个 agent——它经 TASK_CANCELED
                # → AgentInput.SETTLED 落回 idle，仍然活着、仍可被 send_message 寻址。
                # 刻意不用 cancel_agent：那会把它推到 terminated，是语义变更
                # （2026-09-04 spec §7.1）。
                self._cancel_run_token(session_id, task_id)
        # 补 drain：把保留的 root 排队条目派发出去——它出生即 paused → act 首检查点 park 出唯一
        # 气泡，成为本次暂停的续跑点。对空队列 / 满并发是安全 no-op。放在信号循环后、兜底前。
        if tm is not None:
            await tm.drain()
        # 竞态兜底：信号发完会话已静止（root 恰好收尾、无可 park 对象）→ 立即清闩锁防残留。
        if tm is not None and tm.is_done():
            self._pausing.discard(session_id)
            self._pause_claimed.discard(session_id)
            tm.set_pause_abandon(False)
        return True

    def _is_writing_task_finalized(self, session_id: str, task_id: str | None = None) -> bool:
        tm = self._task_managers.get(session_id)
        check = getattr(tm, "is_writing_finalized", None)
        return bool(check(task_id)) if callable(check) else False

    def _is_agent_writing_task_finalized(self, agent_id: str) -> bool:
        rec = self._agent_lifecycle_manager.record_of(agent_id)
        if rec is None:
            return False
        tm = self._task_managers.get(rec.session_id)
        current = tm.current_task_for_agent(agent_id, rec.current_task_id) if tm is not None else rec.current_task_id
        return bool(current and self._is_writing_task_finalized(rec.session_id, current))

    def _pause_task(self, session_id: str, task_id: str) -> bool:
        """内部原语：定向暂停指定在途 task 的 run（spec 2026-07-05 §2.3）。`pause_agent`
        与 `pause_session` 是它仅有的两个调用方（2026-09-04 spec §8）。

        pause 指定在途 task 的 run → 它在检查点 park 自己的 wait 气泡，经多 pending
        面板回复续跑。不在跑（无本 run 令牌）→ False。只停该 task 本身的 run，不涉及
        其子任务。"""
        if self._is_writing_task_finalized(session_id, task_id):
            return False
        tokens = self._run_tokens.get(session_id, {}).get(task_id)
        if tokens is None:
            return False
        tokens.pause.pause()
        return True

    @_session_command(load_session="skip", skip_finalized_stop=True)
    @runtime_command
    async def cancel_session(self, session_id: str) -> bool:
        """硬取消：清队列 + 该 session 下**每个 agent** 经 `cancel_agent` 显式转
        `terminated`（2026-09-04 spec §7.2）。

        memory 保留。开新对话由调用方另起（新 /messages → 同 session_id 的 new run）。

        先关闭派发、向在途 run 送达停止信号，等待开窗回合的执行者完成收尾再撤销回合。
        随后终局未决 HITL、清队列；每个 agent 仍经 `cancel_agent` 进入终态。
        """
        # 「有没有东西要取消」的判据看三处，不是只看 TaskManager 在不在内存里：
        # ALM 里还登记着 agent（跑完的会话 record 常驻，旧入口也可能只装填 ALM），
        # 或者还挂着未决 HITL——这两种情形下 TM 都可能不在内存，而
        # 该做的事（把 agent 转 `terminated`、把提问收口）一件都没少。
        #
        # 只看 TM 的旧判据会让这些会话**静默早退**：`/cancel` 什么都没做就返回 False，
        # 未决提问留成「有 HitlOpened 无终局事件」的孤儿，重启后 `rebuild_session` 又把它
        # 当未决恢复出来（总账 A10 要防的正是这个）。
        per = self._run_tokens.get(session_id, {})
        task_manager = self._task_managers.get(session_id)
        if self._is_writing_task_finalized(session_id):
            return False
        if (self.storage_health(session_id) is not None and task_manager is not None
                and not any(task_manager._dispatch_is_live(tid)
                            for tid in task_manager.running_task_ids())):
            # Stopped owners can hold an unconfirmed terminal disposition.
            # Confirm it before cancellation; never replace it with CANCELED.
            await self._confirm_storage_facts(session_id)
            await task_manager.reconcile_stopped_tasks()
            await self._repair_observe_handoffs(task_manager)
            if self.clear_storage_isolation(session_id):
                await self._activate_prepared_tasks_locked(task_manager)
        has_agents = bool(self._agent_lifecycle_manager.agent_ids_of_session(session_id))
        has_pending_hitl = bool(self.hitl_registry.list_pending(session_id=session_id))
        if not per and task_manager is None and not has_agents and not has_pending_hitl:
            return False
        # Storage repair above can await. Admit this request only while no
        # publication is protected, immediately before closing dispatch below.
        if self._is_writing_task_finalized(session_id):
            return False
        # 取消前判定会话是否已空闲挂起（无在跑任务）。RUNNING：在途 task 经 CancelToken→checkpoint
        # 协作取消→on_task_finished→is_done→_fire_session_done→_on_done 自行回收，故此处不抢着回收。
        idle = task_manager is not None and task_manager.is_done()
        # Deliver stop signals while execution ownership still exists. Closing a
        # provisional window first loses both the task and its cancellation route.
        if task_manager is not None:
            task_manager.begin_cancel()
        for task_id in list(per):
            self._cancel_run_token(session_id, task_id)
        executions = self._step_executions.get(session_id)
        if executions is not None:
            executions.stop_metadata()
        if task_manager is not None:
            await task_manager.wait_for_round_dispatches()
        if executions is not None:
            await executions.finish_metadata(self._event_bus)
        # ① 未决的 ask_user 一并终局，且**先于**下面 cancel_all 触发的会话终态——
        # 与熔断 trip 序列同一条纪律（HITL 终局须先于会话终态）。不终局的代价在重启后：
        # rebuild_session 按「有 HitlOpened 无终局事件」折 pending，会把已取消会话的
        # 提问当未决恢复出来（总账 A10）。
        # **先把开着的未提交窗口收掉**：窗口靠「那一轮的提交点」来关，而这条会话就此不再
        # 有下一轮。不收的代价有三：那一轮里发出的 `TASK_CANCELED` 等事实随缓冲一起消失、
        # 总线的缓冲没人清（逐出 TM 之后仍留在那里）、被这一轮收下的答复停在待终局——
        # 它不在 `list_pending` 里，下面那一句收口遍历不到它，重启后又成了「有 HitlOpened
        # 无终局事件」的未决提问（总账 A10）。
        #
        # 用丢弃而不是提交：用户取消了会话，这一轮本就不该算数。丢弃会把待终局的答复退回
        # 待答，于是下面那一句正好把它们以 `cancelled` 收口、事实落盘。
        # `getattr`：宿主 / 单测的 TM 替身可能没有窗口这套（与本文件其余几处同姿态）。
        for _tid in list(getattr(task_manager, "open_round_task_ids", None) or ()):
            try:
                await task_manager.discard_round(_tid)
            except Exception:
                logger.exception(
                    "cancel_session: discard_round failed for task %s", _tid)
        await self._cancel_session_hitl(session_id, message=CancelReason.USER_CANCEL)
        # 已终局但还没被消费的决定：会话取消之后没人会再消费它们了（任务不会再跑，
        # `_inject_resolved_user_turns` 对终态 task 本就跳过），所以一并盖 `HitlClosed`。
        # 不盖就永远留在折叠的补注入清单里——而这条会话的事件日志还在（core 不删日志）。
        # 上面那步收口的是**未决**的，它们终局成 cancelled，折叠本就不收录，两件事不重叠。
        try:
            await self.hitl.close_resolved(session_id)
        except Exception:
            logger.exception("cancel_session: close_resolved failed for %s", session_id)
        # Persist each agent's existing session-canceled terminal fact before its
        # task effects. Recovery can derive the full owned task set, not just the
        # routing hint for its most recently queued task.
        for aid in list(self._agent_lifecycle_manager.agent_ids_of_session(session_id)):
            await self.cancel_agent(aid, reason="session_canceled", _session_cancel_admitted=True)
        if task_manager is not None:
            await task_manager.cancel_all(reason=CancelReason.USER_CANCEL, _admitted=True)
        if idle:
            # 已暂停/中断（无在跑 task）的会话被取消：cancel_all 不经 _fire_session_done，
            # _on_done 不会触发，故在此显式清掉这一轮的控制信号残余。
            #
            # **不逐出**（2026-09-08 生命周期改造）：取消 = 这一轮不跑了，不等于这条
            # 会话不要了——用户多半还要看它的历史，甚至接着聊。逐出由持有方显式调
            # `forget_session`。从前这里调的是 `_release_session`，会连 agent record
            # 一起摘掉，于是「取消后再发消息」和「跑完后再发消息」撞同一个
            # `AgentNotFound`。
            self._release_round(session_id)
        return True

    # ── Agent 级取消（Task 19）───────────────────────────────────────────────

    async def _read_persisted_task(self, session_id, task_id):
        # rebuild_view uses a valid snapshot plus its event delta, including a commit
        # whose receipt was lost before EventBus could notify its subscribers.
        head = await self.event_store.committed_head(session_id)
        task = (await rebuild_view(self.event_store, session_id, through_position=head)).tasks.get(task_id)
        if task is None or not task.details_complete:
            task = (await self._historical_task_view(session_id, through_position=head)).tasks.get(task_id)
        return task

    async def _confirm_task_settled(self, task):
        agent_id = task.assigned_agent_id or task.creator_agent_id
        if agent_id:
            await self._agent_lifecycle_manager.confirm_settled(agent_id, task.id)
        registry = self._step_executions.get(task.session_id)
        if registry is not None:
            registry.notify_finish(task.id)

    async def _finalize_canceled_tasks(self, session, tasks, reason):
        await self._finalize_cancel_memory(session, tasks, reason, strict=True)
        tids = {task.id for task in tasks}
        for req in list(self.hitl_registry.list_pending(session_id=session.id)):
            if req.task_id in tids:
                await self.hitl.cancel(req.id, message=CancelReason.USER_CANCEL)
        for tid in tids:
            await self.hitl.close_resolved(session.id, task_id=tid)
        registry = self._step_executions.get(session.id)
        if registry is not None:
            for tid in tids:
                registry.notify_finish(tid)

    @runtime_command
    async def cancel_agent(self, agent_id: str, *, reason: str | None = None,
                           _session_cancel_admitted: bool = False) -> list[str]:
        """Terminate the agent and cancel its current task subtree through TaskManager.

        AgentTerminated is the durable state fact preceding task cancellation. Repeating
        the call repairs incomplete task effects without emitting another termination.
        Descendant agents retain their own lifecycles even when they execute canceled tasks.
        Unrelated tasks are not selected by agent ancestry or ownership. The
        session_canceled reason used by cancel_session covers all owned tasks;
        cold recovery derives that same scope from the existing terminal fact.
        The private admission argument is only for cancel_session's existing
        request; independent public calls retain normal write-window rejection.
        """
        from ctx_weft.core.orchestrator.lifecycle.agent_state import AGENT_CANCEL_REASON
        lm = self._agent_lifecycle_manager
        rec = lm.record_of(agent_id)
        if rec is None:
            return []
        if rec.status == "terminated" and not rec.status_reason.startswith(AGENT_CANCEL_REASON):
            return []
        if rec.session_id not in self._task_managers:
            await self.rebuild_session(rec.session_id, tenant_id=rec.tenant_id, strict=True)
            rec = lm.record_of(agent_id)
        tm = self._task_managers.get(rec.session_id)
        current = (tm.current_task_for_agent(agent_id, rec.current_task_id)
                   if tm is not None and rec.status != "terminated" else rec.current_task_id)
        cause = (rec.status_reason if rec.status == "terminated" else
                 AGENT_CANCEL_REASON + (reason or "canceled"))
        if not current:
            await self._cancel_pending_hitl_of(agent_id, session_id=rec.session_id)
        killed = []
        async def terminate():
            nonlocal current
            if tm is not None and rec.status != "terminated":
                current = tm.current_task_for_agent(agent_id, current)
            if await lm.apply_input(agent_id, AgentInput.CANCEL, task_id=current, reason=cause):
                killed.append(agent_id)
        if tm is not None and cause == AGENT_CANCEL_REASON + "session_canceled":
            await tm.cancel_session_agent_tasks(agent_id, cause, before_cancel=terminate,
                                                _admitted=_session_cancel_admitted)
        elif tm is not None and current:
            # Serialize the agent state commit with observe verdict acceptance.
            # On persistence failure no task changes or run signals have occurred.
            await tm.cancel_task_tree(current, cause, before_cancel=terminate,
                current_agent_id=agent_id if rec.status != "terminated" else None)
        else:
            await terminate()
        return killed

    async def _cancel_pending_hitl_of(
        self, agent_id: str, *, session_id: str, defer: bool = False,
        defer_only_task_id: str | None = None,
    ) -> None:
        """终局**该 agent 名下**全部未决 HITL（`cancel_agent` 专用）。

        与 `_cancel_session_hitl` 同一模式（best-effort，单条失败不阻断其余），区别
        在粒度：`_cancel_session_hitl` 按 session 收口全部未决请求，这里额外按
        `agent_id` 过滤（`HitlRequestView.agent_id`）——`cancel_agent` 只该终局这一个
        agent 名下的未决提问，不能误伤同一 session 里其他 agent 仍然合法在等的提问。

        ``defer``：两个调用方口径不同。`cancel_agent` 用默认的 False——那是真终结，
        没有可撤销的一轮。`_inject_user_turn` 传 True——旧气泡的收口跟着新消息那一轮走，
        用户在 LLM 开口之前按暂停时它要回到 pending（spec 2026-09-09）。
        """
        for v in self._list_pending_hitl_memory(session_id=session_id):
            if v.agent_id != agent_id or v.resolved:
                continue
            # `defer_only_task_id`：只有这个 task 的气泡跟着这一轮走，其余立即收口
            # （提交 / 撤销钩子都按 task 查待终局请求，别的 task 的会卡住，见调用点）。
            this_defer = defer and (
                defer_only_task_id is None or v.task_id == defer_only_task_id)
            try:
                await self.hitl.cancel(
                    v.id, message=CancelReason.USER_CANCEL, defer=this_defer)
            except Exception:
                logger.exception(
                    "_cancel_pending_hitl_of: cancel failed for agent=%s hitl=%s", agent_id, v.id,
                )

    def _hitl_answer_ready(self, session_id: str, task_id: str, hitl_id: str) -> bool:
        """`human_answer_ready` 钩子：这条请求属于该 task，且人已经给出了决定。

        「给出了决定」含**待终局**那一份（`effective_decision`）：冷应答先登记待终局、
        `HitlResolved` 要等这一轮的提交点才发，而这里判的正是「答复到过没有」，不是
        「事实落盘没有」。取消也算——那同样是一个决定，重排后由 reconcile 把它变成
        一条工具结果继续跑。
        """
        req = self.hitl_registry.get(hitl_id)
        return (
            req is not None
            and req.session_id == session_id
            and req.task_id == task_id
            and req.effective_decision is not None
        )

    async def _commit_round_writes(
        self, session_id: str, task_id: str, staged: "list[tuple[Any, Any, Any]]",
    ) -> None:
        """`commit_round` 钩子：这一轮算数了——先终局答复，再让答复进 memory。

        1. 该 task 上待终局的 HITL 答复逐条终局（发 `HitlResolved`）。判据与 `_revert_round`
           同一份（按 task 全量查）。
        2. 这一轮暂存的 memory 写入按写入顺序落盘（`TaskManager.stage_memory`）。
        3. 给刚终局的那些答复盖 `HitlClosed`——见那一步的注释：这是「决定已终局」与「效果
           已持久」同时成立的唯一位置。

        **顺序就是这个钩子的全部要点**：终局事实先落盘，答复才进 memory。反过来，崩在
        两步之间就会出现「memory 里有答复、日志里问题还悬着」——人再答一次，模型却
        早已拿着旧答复往下跑。按现在的顺序，崩在中间只会是「已终局、memory 里还没有」，
        那正是既有恢复路径会补的形状（`UserTurn` 由 `_inject_resolved_user_turns` 补写，
        工具结果由 reconcile 按事件里的 `CapabilityFinished` 补写）。

        **失败即上抛，不吞**：任何一条终局失败就不再写 memory（否则 memory 跑到日志前面，
        正是这个顺序要防的形状）；写 memory 失败同样上抛。`TaskManager.commit_round` 据此
        保留快照，下一个提交点重试——两步都幂等（已终局的 HITL 再 commit 是 no-op，暂存
        记录带 id、重复 ingest 是 no-op）。
        """
        committed: list[PendingHitl] = []
        for req in self.hitl_registry.claim_pending_for_task(session_id, task_id):
            done = await self.hitl.commit(req.id)
            if done is not None:
                committed.append(done)
        for memory, event, pctx in staged:
            key = self._assistant_key(pctx, event.address)
            pending = self._assistant_pending.get(key, {})
            if event.id not in pending:
                await memory.ingest(event, pctx)
                continue
            async with self._assistant_agent_lock(key):
                entry = self._assistant_pending.get(key, {}).get(event.id)
                if entry is not None:
                    # commit_provisional has already confirmed the original batch.
                    entry.committed = self._config.event_commit_policy == "required"
                    entry.provisional = False
                await memory.ingest(event, pctx)
                self._assistant_pending.get(key, {}).pop(event.id, None)
        # 3. 给这一轮终局掉的答复盖「已了结」章（`HitlClosed`）。
        #
        # **这里是那枚章唯一正确的位置**，因为两个条件到这一行才同时成立：决定已终局
        # （第 1 步刚发 `HitlResolved`）、效果已持久（第 2 步刚把暂存的写入落盘）。放在
        # 更早的任何地方都不行：
        #
        # - 放在写对话那一步（`_write_hitl_reply_turn`）—— 那时请求还是待终局，而且写入
        #   还在暂存区；`close()` 因此会安静跳过，章落不下来（那正是它的用意）。
        # - 放在窗口里任何位置 —— 事件会进这一轮的缓冲，而缓冲在**钩子之前**就补投完了
        #   （`TaskManager.commit_round` 先 `commit_provisional` 再调本钩子），于是这枚章
        #   会排在 `HitlResolved` **之前**。折叠会先摘掉 `opened`、再撞上 `HitlResolved`
        #   找不到请求 → 那条终局整个丢掉。
        #
        # 这一轮被丢弃时走的是 `_revert_round`：答复 `release` 回 pending，本方法压根不跑,
        # 章自然也不会发——正是要的那个形状。
        #
        # 盖章失败不上抛（`close()` 自己不抛）：前两步才是这个钩子的承重步骤，一枚事后的
        # 章不该把一次已经落定的提交变成失败。漏掉的后果只是那条记录多留一轮，下次恢复
        # 幂等补一遍。
        for done in committed:
            await self.hitl.close(done)

    async def _revert_round(self, session_id: str, task_id: str) -> None:
        """撤销一轮里**不属于 TaskManager** 的那两样（`revert_round` 钩子，spec 2026-09-09）。

        1. **HITL**：把这一轮收下的答复（HITL 应答 / 被新消息收口的旧气泡）`release` 回
           pending。它从来没发过
           `HitlResolved`，`HitlOpened` 还在原地，于是日志描述的正是撤销之前的世界，
           会话状态折叠自然回到 `PAUSED`。

        2. **暂停闩锁**：见下方第 3 步注释。

        memory 不在其列：这一轮的用户消息、答复、由答复回灌的工具结果都只在窗口的暂存区
        里（`TaskManager.stage_memory`），关窗时随快照扔掉，memory 从来没见过它们。

        best-effort：一次撤销失败不该把「用户按了暂停」变成 run 崩溃。最坏结果是一个气泡
        停在待终局，且会记一条 exception 日志。
        """
        tm = self._task_managers.get(session_id)

        # TaskManager calls revert only before an irreversible/uncertain commit.
        for key, pending in list(self._assistant_pending.items()):
            if key[1] != session_id or not any(
                    entry.provisional and entry.finished.task_id == task_id
                    for entry in pending.values()):
                continue
            async with self._assistant_agent_lock(key):
                for record_id, entry in list(pending.items()):
                    if entry.provisional and entry.finished.task_id == task_id:
                        pending.pop(record_id, None)

        # 按 task 全量退回，而不是记一个 id：`_cancel_pending_hitl_of` 收口的是该 agent
        # 名下**全部**未决请求，可能不止一条。
        for req in self.hitl_registry.claim_pending_for_task(session_id, task_id):
            try:
                await self.hitl.release(req.id)
            except Exception:
                logger.exception(
                    "_revert_round: release hitl %s of task %s failed", req.id, task_id)

        # 3. **清暂停闩锁。** 这一步不清，下一轮会出生即取消。
        #
        #    `pause_session` 置 `_pausing`（窗口内新派发的 run 按闩锁分流）并让本轮的 run
        #    认领了 `_pause_claimed`（"一次暂停恰一个续跑点"）。正常路径上这两个由
        #    `_on_idle` 在会话静止时清——但丢弃之后 task 退回 `AWAITING_HUMAN`（D/B 类）
        #    而不是终态，`tm.is_done()` 不成立，`_on_idle` 根本不触发，闩锁就留在那里：
        #    用户重答一次，新 run 一出生就被 born-cancel，会话看起来"答了没反应"。
        #
        #    实测钉在 `test_round_discarded_hitl_reply_e2e` 的第 ⑥ 步（重答后必须真的
        #    跑起来）。本轮已经整个撤销，闩锁的两个目的（弃子、留一个续跑点）都不再成立。
        self._pausing.discard(session_id)
        self._pause_claimed.discard(session_id)
        if tm is not None:
            tm.set_pause_abandon(False)

    # ── Agent 级暂停 / 恢复（Task 20）────────────────────────────────────────

    @_session_command(load_session="skip", allow_missing_session=True, skip_finalized_stop=True)
    @runtime_command
    async def pause_agent(self, agent_id: str, *, reason: str | None = None) -> list[str]:
        """暂停 agent 及其全部**当前 running** 的子孙（spec 7）。

        只对 `running` 生效——`agent_id` 本身非 running 直接报错（`AgentNotRunningError`）；
        级联展开到的子孙里非 running 的静默跳过（暂停不该殃及本就 idle/等待中的子孙）。

        建在既有的**定向暂停原语** `_pause_task`（spec 2026-07-05 §2.3）之上，不是把
        `cancel_agent` 的「立即拍状态」搬过来抄一份——`_pause_task` 只对准这一个 task
        的 run 发一次软打断信号，被暂停的 run 在自己的下一个检查点自行 park 出一个
        wait 气泡（`ActStep._interrupt_checkpoint` / `_run_llm_turn` /
        `_execute_tool_calls` 命中 `pause_token.is_paused` 后统一走
        `act._park_for_interrupt(...)`），agent 状态由那次
        **真实**的 `TASK_AWAITING_HUMAN` 事件经 ALM 转成 `waiting_human`——不是本方法
        直接拍的。

        R24（本任务的核实结论，见 task-20-report.md）：spec/brief 写的
        `running --pause--> interrupted` 与实现不符——`interrupted` 只由
        `TaskManager._suspend_task_interrupted`（宿主 outage/崩溃）驱动，操作者暂停
        经检查点 park，落的是 `waiting_human`。本方法因此**不**调用
        `AgentLifecycleManager.apply_input`，全部转移留给真实的 `TASK_AWAITING_HUMAN` 事件
        走 ALM 唯一入口——这里若手动拍一个 `AgentInput.INTERRUPTED`，就会在
        `waiting_human` 之外多出一条假的 `interrupted` 分支，且与实际状态不符
        （run 还没被暂停完，agent 已经被判定"暂停完成"）。

        返回值是**已成功递送暂停信号**（`_pause_task` 命中一个在途 run token）的
        agent id 列表——暂停本身是异步生效的，返回时这些 agent 多半仍是 `running`，
        真正落 `waiting_human` 要等它们各自跑到下一个检查点。
        """
        reg = self._agent_lifecycle_manager
        rec = reg.record_of(agent_id)
        if rec is None:
            raise AgentNotFound(f"unknown agent: {agent_id}")
        if self._is_agent_writing_task_finalized(agent_id):
            return []
        if rec.status != "running":
            raise AgentNotRunningError(
                f"agent {agent_id} is {rec.status}, not running; nothing to pause"
            )

        paused: list[str] = []
        for aid in [agent_id, *reg.descendants_of(agent_id)]:
            r = reg.record_of(aid)
            if r is None or r.status != "running" or not r.current_task_id:
                continue
            if self._pause_task(r.session_id, r.current_task_id):
                paused.append(aid)
        return paused

    @_session_command(load_session="required")
    @runtime_command
    async def resume_agent(self, agent_id: str, *, session_id: str | None = None) -> list[str]:
        """恢复 agent 及其全部**由暂停产生**的等待气泡（spec 7；R24）。

        `resume_agent` **不能**无差别恢复 `waiting_human` 的子孙——那个状态同时是
        「被 `pause_agent` 暂停、park 在检查点」和「agent 主动 `ask_user` 正在等
        用户真答」两种截然不同情形的落点，无差别放行等于替用户回答了那个真问题。

        判据（实测确认，见 task-20-report.md）是 `PendingHitl.delivery`：
        - `pause_agent`/`pause_session` 产生的气泡恒为
          `UserTurnDelivery(preface ∈ {PREFACE_AFTER_INTERRUPT, PREFACE_AFTER_INTERRUPT_EDIT})`
          （`act._park_for_interrupt(...)` 的唯一产物）；
        - `ask_user` 的真实结构化提问用的是 `ToolResultDelivery`
          （`reply_as_result=True`，见 `control_capability.py` 的 `ask_user`
          构造），与前者的类型本身就不同，天然互斥；
        - act 纯文本收尾的软待命（`act._park_await_user(...)` 的产物）虽然**同样**是
          `UserTurnDelivery`，但 `preface == PREFACE_NORMAL`——那不是暂停产生的，
          是正常一轮说完话后的自然等待，`resume_agent` 若把它也放行，等于没有
          任何新用户输入就凭空续了一轮，同样不对。

        三者叠在一起，只有 `UserTurnDelivery` 且 `preface` 落在
        `{PREFACE_AFTER_INTERRUPT, PREFACE_AFTER_INTERRUPT_EDIT}` 才是本方法该碰的。

        续跑走既有的 HITL 冷续跑通路（`reply_to_hitl` → `_resume_after_hitl` →
        `recover_session`），不直接 `apply_input(AgentInput.RESUMED)`——那样会把
        agent 状态拍成 `running`，但驱动它真正再跑起来的 task 这时其实还没有被
        重排/派发，状态与现实不符；`apply_input` 的唯一入口纪律也要求转移经由
        真事件发生，不由调用方越过 TaskManager 直接拍。真正的 `running` 由续跑
        起来之后的那次**真实** `TASK_STARTED` 事件驱动。
        """
        reg = self._agent_lifecycle_manager
        if not reg.has(agent_id):
            raise AgentNotFound(f"unknown agent: {agent_id}")

        resumed: list[str] = []
        for aid in [agent_id, *reg.descendants_of(agent_id)]:
            r = reg.record_of(aid)
            if r is None or r.status != "waiting_human":
                continue
            req = self._pause_bubble_of(aid, session_id=r.session_id)
            if req is None:
                continue
            await self.reply_to_hitl(
                HitlReply(hitl_id=req.id, outcome=HITL_OUTCOME_ACCEPTED,
                          agent_id=req.agent_id, message=_RESUME_MARK)
            )
            resumed.append(aid)
        return resumed

    def _pause_bubble_of(self, agent_id: str, *, session_id: str) -> "PendingHitl | None":
        """该 agent 名下**由暂停产生**的未决 wait 气泡（`resume_agent` 的判据，R24）；
        没有则 `None`。判据见 `resume_agent` 文档字符串。
        """
        for req in self.hitl_registry.list_pending(session_id=session_id):
            if (
                req.agent_id == agent_id
                and isinstance(req.delivery, UserTurnDelivery)
                and req.delivery.preface in (PREFACE_AFTER_INTERRUPT, PREFACE_AFTER_INTERRUPT_EDIT)
            ):
                return req
        return None

    # ── 换模型：两条命令（批次 B）──────────────────────────────────────────────
    #
    # llm_* 此后只出现在这两条命令的入参里（以及「建一个 session」的入参里）。
    # 两条都是纯赋值——不入队、不改任何 task 状态、不触发调度，对一个所有 task
    # 都在等人的 agent 调用它完全安全。派发时 `AgentLifecycleManager.resolve_model` 现读
    # record，因此换模型对**尚未派发**的 run 立即生效；已经在跑的 run 手上的
    # `ResolvedModel` 是那次 assemble() 时现解的快照，不会被这两条命令追改。

    @runtime_command
    async def set_agent_llm(
        self, agent_id: str, *, llm_account: str = "", llm_model: str = "",
        llm_config: ModelCallConfig | None = None,
        reason: str = "user_selected",
    ) -> bool:
        """host 入口：把 `(llm_account, llm_model)` 包成 `ModelChoice`，转发给 registry。"""
        return await self._agent_lifecycle_manager.set_agent_llm(
            agent_id, ModelChoice(account=llm_account, model=llm_model,
                                  config=llm_config if llm_config is not None else ModelCallConfig()), reason=reason,
        )

    @runtime_command
    async def set_session_llm(
        self, session_id: str, *, llm_account: str = "", llm_model: str = "",
        llm_config: ModelCallConfig | None = None,
        reason: str = "user_selected",
    ) -> int:
        """host 入口：作用于该 session 下 registry 持有的全部 agent record。

        返回值 = 真正改动了 record 的 agent 数（幂等 no-op 不计数）。
        """
        return await self._agent_lifecycle_manager.set_session_llm(
            session_id, ModelChoice(account=llm_account, model=llm_model,
                                  config=llm_config if llm_config is not None else ModelCallConfig()), reason=reason,
        )

    # ── Phase 1 compat ───────────────────────────────────────────────────────

    @runtime_command
    async def run_single_task(
        self,
        *,
        session_id: str | None = None,
        template_id: str,
        user_prompt: "str | list[ContentPart]",
        tenant_id: str = "default",
        llm_account: str | None = None,
        llm_model: str | None = None,
        llm_config: ModelCallConfig | None = None,
    ) -> tuple[TurnHandle, LoopState]:
        """单任务测试便利入口：起一个 task 端到端跑完并等它结束（2026-09-04 spec §2）。

        60 处测试在用，返回的句柄已带 `agent_id`，与 agent-centric 不冲突——保留它是
        纯收益、删它是纯成本。"""
        from ctx_weft.protocols.content import content_to_text

        sid = session_id or generate_id("ses")
        # 单例：这条路径自己建一个 TM 跑到底——session 若已有主，就会与既有 TM 并存。
        # 入口即拒，先于下面任何持久化。
        async with self._session_guard(sid):
            if self._session_in_memory(sid) or await self.event_store.committed_head(sid):
                raise SessionAlreadyExistsError(sid)
            ctx = ProviderContext(session_id=sid, tenant_id=tenant_id)
            # 入口即拒、不落库：格式/blob 门控须在任何持久化（Session/Task/事件）之前完成
            # （spec 2026-08-24 Phase 3a）。_resolve_llm 是纯查表，此处先行调用安全——
            # 且必须先于 instantiate()：那一步会发 AgentInstantiated 事件，是这条路径上
            # 第一个「persist」，llm 不可解析须在它之前失败，而非之后（此调用只为早失败，
            # 结果不留用——agent 的窗口由下面 instantiate(llm=...) 内部按同一份 choice
            # 重新解出，registry 不缓存 client，这里的返回值即弃是设计的一部分）。
            self._resolve_llm(llm_account, llm_model, ctx=ctx)
            # validate → normalize 的顺序与另外两个入口共用同一个方法，不再各写一遍。
            # event 侧产物在这条路径上没有**入口事件**载得下它（run_single_task 自己
            # register_task、不经 push_task，也不发 SESSION_CREATED），但它必须挂到 Task 上：
            # 这条路径产出的 task 与另外两个入口共用同一份 `user_prompt_event_jsonable`
            # 契约（`Task` 的字段注释），入口不同不该让 task 上的字段形态不同。
            # 代价是携图时会在 event blob store 里留一份暂时无人引用的字节；不为此加分支，
            # 是因为「三入口共用同一个真源」这条不变量比省掉一次 compat 路径上的 put 更值钱
            # （宿主的 event blob 回收本就按自己的保留策略走，见 EventBlobStore 协议）。
            user_prompt, user_prompt_event_jsonable = await self._validate_and_normalize_content(
                user_prompt, sid, tenant_id=tenant_id,
            )
            lm = self._agent_lifecycle_manager

            # llm= 传选择（可空）——不是身份：这条创建路径唯一发 AgentInstantiated 的地方，
            # 选择须住进 agent record，事件层才有真值可报（批次 B）。
            agent, template = await lm.instantiate(
                template_id=template_id, session_id=sid, tenant_id=tenant_id, ctx=ctx,
                llm=ModelChoice(account=llm_account or "", model=llm_model or "",
                                config=llm_config if llm_config is not None else ModelCallConfig()),
            )
            # Bind the model and immutable token budget for this run.
            resolved_model = lm.resolve_model(agent.id)

            session = Session(
                id=sid,
                user_prompt=content_to_text(user_prompt),
                tenant_id=tenant_id,
                root_agent_id=agent.id,
                llm_provider=llm_account or "",
                llm_model=llm_model or "",
                created_at=now_utc(),
            )
            task = Task(
                id=generate_id("tsk"),
                session_id=sid,
                status="ACTIVE",
                tenant_id=tenant_id,
                assigned_agent_id=agent.id,
                creator_agent_id=agent.id,
                title="User Request",
                description=content_to_text(user_prompt)[:200],
                user_prompt=user_prompt,
                user_prompt_event_jsonable=user_prompt_event_jsonable,
                # 无人值守（2026-09-22）：这条入口的语义就是「跑到完成再返回」，没有人会
                # 发下一条消息。不标的话，纯文本收尾会 park 等一个永远不来的人——判据自
                # `interaction_mode` 退场后只剩「有没有人在」，而这里答案明确是没有。
                unattended=True,
                # 同理不接任何交互口：没有对端可往返，产出由本方法同步返回给调用方。
                port_key=PORT_NONE,
                created_at=now_utc(),
            )

            task_manager = self._session_registry._new_task_manager(session)
            task_manager.set_session(session)
            task_manager.register_task(task)
            # compat 路径不经 _register_and_drain（没有队列、没有 drain），但 TM 一样要登记为
            # 这个 session 的 owner：跑的这段时间里若有别的入口（send_message 冷启动的
            # recover_session 等）来找这个 session 的 TM，必须撞上这一个，而不是再建一个与它
            # 并存（单例，见 `_bind_task_manager`）。跑完在下面的 finally 里摘掉——这个 TM
            # 没有 runner、没挂回调，留在册里会让之后的续跑拿到一个派发不了的 TM；摘掉之后
            # 续跑照旧从事件日志重建，与改动前一致。
            self._bind_task_manager(sid, task_manager)
            # 同理登记 session，使 /hitl 等入口在这条路径上也查得到该 session。
            self._session_registry.register_session(sid, tenant_id=tenant_id)
            self._initialized_sessions.add(sid)

            for p in self.providers.get_capability_providers():
                if isinstance(p, ControlCapabilityProvider):
                    p.register_session(sid, task_manager, session)
                    break
        try:
            try:
                state, handle = await self._execute_task(
                    session=session,
                    task=task,
                    agent=agent,
                    template=template,
                    run_id=generate_id("run"),
                    memory=self.providers.get_memory(),
                    resolved_model=resolved_model,
                    task_manager=task_manager,
                )
            except Exception as e:
                # 崩溃入口（与 TaskManager._run_task 的那条同形、同一个工厂）：_run_loop
                # 重抛，这里交给 TM 落状态发事件，再原样抛给调用方。
                await task_manager.apply_run_outcome(task.id, crash_run_outcome(e))
                raise
            # compat 路径没有队列、不经 _run_task，但一样要有人消费 run 的结局——
            # 否则 outage / park / 取消在这条路径上没人写 task 状态、没人发 task 事件
            # （Task 4 之前是 _run_loop 自己写自己发）。兜底同 _run_task。
            run_outcome = state.run_outcome or RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="success")
            await task_manager.apply_run_outcome(task.id, run_outcome)
            await task_manager.finish_observe(run_outcome, task.id)
        finally:
            # **不 forget_session**：2026-09-04（Task 12）起这里不再代 TM 报队列状态
            # （`announce_queue_state` 已停发，events-v2 §5），但仍然不 forget——
            # `forget_session` 会连同该 session 已登记的成员 agent 集合一起清空
            # （`_SessionState.agent_ids`），冷 HITL 应答期 `register_session` 是
            # `setdefault`（重入保留原状态），一旦这里先 forget 就等于把成员集合
            # 归零，把已经跑过的 agent 从这个 session 里凭空摘掉。
            for p in self.providers.get_capability_providers():
                if isinstance(p, SessionScopedCapabilityProvider):
                    p.deregister_session(sid)
            # compare-and-pop：只摘自己登记的那个（见上面 _bind_task_manager 处的注释）。
            if self._task_managers.get(sid) is task_manager:
                self._task_managers.pop(sid, None)
                self._prepared_recoveries.pop(sid, None)
        return handle, state

    # ── Phase 4 full session ─────────────────────────────────────────────────

    @runtime_command
    async def fork_session_from_response(
        self, *, source_session_id: str, response_event_id: str, session_id: str,
        template_id: str | None = None, llm_account: str | None = None,
        llm_model: str | None = None, llm_config: ModelCallConfig | None = None,
    ) -> "SessionForkResult":
        """Create an idle root from the reply's committed AGENT-memory checkpoint.

        session_id is the caller's durable idempotency key. No source execution is
        loaded; retries use target-owned materials even after source deletion.
        Raises ForkCheckpointUnavailableError until ObserveApplied is committed.
        """
        from ctx_weft.core.memory_seed import digest, identity, prepare_seed, fork_result
        from ctx_weft.core.control.reducers import rebuild_view
        from ctx_weft.protocols.llm import serialize_call_config
        from ctx_weft.protocols.runtime import MemorySeedConflictError

        if not session_id or session_id == source_session_id:
            raise ValueError("A distinct target session_id is required")
        if self._config.event_commit_policy != "required":
            raise ValueError("Memory seeding requires confirmed event persistence")
        llm_config = deepcopy(llm_config)
        request_identity = {"source_session_id": source_session_id,
            "response_event_id": response_event_id, "template_id": template_id,
            "llm_account": llm_account, "llm_model": llm_model,
            "llm_config": serialize_call_config(llm_config) if llm_config is not None else None}
        fingerprint = digest({**request_identity, "format_version": 2})
        legacy_fingerprint = digest({**request_identity, "format_version": 1})
        if self.storage_health(session_id) is not None:
            if not await self.recover_storage(session_id):
                raise PersistenceUnavailableError(self.storage_health(session_id))
        async with self._session_guard(session_id):
            pending = self._fork_creations.get(session_id)
            if pending is not None:
                if pending[0] != fingerprint:
                    raise MemorySeedConflictError("Target session has another fork request")
                await self.event_bus.emit_batch(pending[1])
            if not await self.event_store.committed_head(session_id):
                events = await prepare_seed(self, source_session_id=source_session_id,
                    response_event_id=response_event_id, session_id=session_id,
                    agent_id=identity(session_id, "root-agent"), fingerprint=fingerprint,
                    template_id=template_id, llm_account=llm_account,
                    llm_model=llm_model, llm_config=llm_config)
                self._fork_creations[session_id] = (fingerprint, events)
                await self.event_bus.emit_batch(events)
            view = await rebuild_view(self.event_store, session_id)
            session = view.sessions.get(session_id)
            if (session is None or session.context_seed is None
                    or session.context_seed["request_fingerprint"] not in (fingerprint, legacy_fingerprint)):
                raise MemorySeedConflictError("Target session has another creation request")
            await self._ensure_memory_seed(session)
            await self._rebuild_session_locked(session_id, strict=True)
            self._fork_creations.pop(session_id, None)
            return fork_result(session)

    async def _ensure_memory_seed(self, session_view) -> None:
        from ctx_weft.core.memory_seed import restore_seed

        await restore_seed(self, session_view)

    @runtime_command
    async def create_session(
        self,
        *,
        template_id: str,
        session_id: str | None = None,
        tenant_id: str = "default",
        llm_account: str | None = None,
        llm_model: str | None = None,
        llm_config: ModelCallConfig | None = None,
    ) -> SessionHandle:
        """建一条**容器会话**：实例化 root agent、发 `SESSION_CREATED`，但**不推 root
        task**（spec/09 §11）。会话里的活全部由后续 `dispatch_task` 派进来。

        与 `start_session` 的分工只有一条：**这一轮有没有「用户的话」**。
        `start_session` 收 `user_prompt` 并立刻拿它开一条对话式 root task；本方法没有
        prompt 可收，建出来的会话是空的、静止的、随时可以被派活。所以本方法**不收**
        `user_prompt` / `initial_task` / `unattended`——那三个描述的都是 root task，
        而这里根本没有 root task。

        **返回 `SessionHandle` 而非 `TurnHandle`**：后者的四个身份字段恒非空、其中
        `task_id` 是「这次交互落到的 task」，空会话里没有这样一条 task，硬造一个
        字段填不出来。要句柄就去 `dispatch_task` 拿。

        **root agent 照常存在且 `idle`。** 它只是手上没有对话——「一个 session 恰有一个
        非空 `root_agent_id`」这条恢复链上的假设不破（`resume_session` 会拒 root 为空
        的投影）。之后 `dispatch_task(..., agent_id=handle.root_agent_id)` 把活落在它
        头上，或者不传 `agent_id` 另起一棵树，两条都成立。

        **TM 连 runner/回调一起接好**（走与 `start_session` 同一个 `_register_and_drain`）：
        不接的话，`dispatch_task` 末尾那次 `drain()` 会撞上
        `RuntimeError("No task runner registered")`——而它是 fire-and-forget 出去的，
        撞了也只在日志里，任务就此静默搁浅。空队列上的那次 `drain()` 本身只是空转：
        会话终结信号只从 `on_task_finished` / `finalize_idle_session` / `cancel_all`
        发出，不会因为「一件活都没有」就自己宣告结束。
        """
        # 单例（`_bind_task_manager`）：给一个本进程里已经有主的 session_id 再建一次，
        # 会造出第二个 TM、第二条 SESSION_CREATED。入口即拒，先于任何持久化。
        session_id = session_id or generate_id("ses")
        if session_id and self._session_in_memory(session_id):
            raise SessionAlreadyExistsError(session_id)

        memory = self.providers.get_memory()
        sm = self._session_registry

        # 与 `start_session` 同一把 per-session 锁：从「查有没有活 owner」到「新 TM 登记
        # 上」之间有 await，同一 session 上的并发入口插进来会各建一个。session_id 交给
        # 下游现铸时不可能撞车，不必拿锁。
        lock = (self._session_guard(session_id) if session_id
                else contextlib.nullcontext())
        async with lock:
            if session_id and (self._session_in_memory(session_id) or await self.event_store.committed_head(session_id)):
                raise SessionAlreadyExistsError(session_id)   # 锁内复查
            session, _no_root_task, task_manager = await sm.create_session(
                template_id=template_id,
                user_prompt="",              # 没有 root task 就没有「这一轮的用户输入」
                tenant_id=tenant_id,
                llm_model=llm_model,
                llm_account=llm_account,
                session_id=session_id,
                llm_config=llm_config,
                with_root_task=False,
            )
            # 模板在 create_session 内部已解析并校验过一次（入口即拒、不落库），这里
            # 不再取第二份：runner 不持有模板，装配时按执行 agent 现解
            # （`_SessionTaskRunner._template_of`）。
            task_manager.set_unhealthy_check(
                lambda sid: self.storage_health(sid) is not None)
            task_manager.set_runner(self._make_task_runner(
                session=session,
                lm=sm.agent_lifecycle_manager,
                memory=memory,
                task_manager=task_manager,
            ))
            self._register_and_drain(session, task_manager)

        return SessionHandle(
            session_id=session.id,
            root_agent_id=session.root_agent_id or "",
            template_id=template_id,
        )

    @runtime_command
    async def start_session(self, params: SessionStartParams) -> TurnHandle:
        """Create or resume a session and start execution.

        params.resume is False → new session (session_id=None → runtime generates it;
                                  session_id=<id> → new session with that host-provided ID).
        params.resume is True  → resume existing session (root_agent_id recovered from events).
        """
        import dataclasses as _dc

        # 单例（`_bind_task_manager`）：「新建」一个本进程里已经有主的 session_id 会造出
        # 第二个 TM、第二条 SESSION_CREATED。入口即拒，先于下面任何持久化。
        if not params.resume and params.session_id and self._session_in_memory(params.session_id):
            raise SessionAlreadyExistsError(params.session_id)

        memory = self.providers.get_memory()
        # 入口即拒、不落库：sm.create_session / sm.resume_session 会立即持久化
        # （instantiate + SESSION_CREATED/RESUMED 事件），所以格式校验与
        # EventBlobStore 门控必须在它们之前。
        # **不做模型能力判断**（spec 2026-08-28-multimodal-adapter-dispatch）：
        # 模态处置归 LLMClient 实现方，core 全程透传。原先为了视觉门控要在这里
        # 惰性解析 LLM，现在整段不碰 LLM，「纯文本不提前解析 LLM」自动成立。
        # 顺序关键：validate 先于 normalize——被拒的内容不该在 blob store 留垃圾。
        if params.resume and not params.session_id:
            raise ValueError("session_id is required when resume=True")
        # Mint before any persisted creation event so every opening uses the same lock.
        sid = params.session_id or generate_id("ses")
        if params.resume:
            await self.initialize(sid)
        normalized, user_prompt_event_jsonable = await self._validate_and_normalize_content(
            params.user_prompt, sid, tenant_id=params.tenant_id,
        )
        params = _dc.replace(params, session_id=sid, user_prompt=normalized)
        async with self._session_guard(sid):
            return await self._start_session_locked(
                params, memory=memory, user_prompt_event_jsonable=user_prompt_event_jsonable,
            )

    async def _start_session_locked(
        self, params: SessionStartParams, *, memory: MemoryProvider,
        user_prompt_event_jsonable: "str | list[dict] | None",
    ) -> TurnHandle:
        """`start_session` 在 per-session 锁内的那一半：建/续 session、接线、登记 TM。"""
        sm = self._session_registry
        lm = sm.agent_lifecycle_manager

        if not params.resume:
            if params.session_id and await self.event_store.committed_head(params.session_id):
                raise SessionAlreadyExistsError(params.session_id)
            # 锁内复查：入口那道判重之后、拿到锁之前，并发的同 id 调用可能已经建好了。
            if params.session_id and self._session_in_memory(params.session_id):
                raise SessionAlreadyExistsError(params.session_id)
            session, root_task, task_manager = await sm.create_session(
                template_id=params.template_id,
                user_prompt=params.user_prompt,
                tenant_id=params.tenant_id,
                llm_model=params.llm_model,
                llm_account=params.llm_account,
                llm_config=params.llm_config,
                initial_task_settings=params.initial_task_settings,
                session_id=params.session_id,
                user_prompt_event_jsonable=user_prompt_event_jsonable,
                unattended=params.unattended,
                port_key=params.port_key,
            )
        else:
            if params.llm_config is not None:
                raise ValueError("Set agent config with set_agent_llm before resuming")
            # 有活 owner 就把新 root task 推进它（单例）；没有才由 resume_session 新建。
            if not params.session_id:
                raise ValueError("session_id is required when resume=True")
            await self._rebuild_session_locked(params.session_id, tenant_id=params.tenant_id, strict=True)
            existing = self._task_managers[params.session_id]
            session, root_task, task_manager = await sm.start_next_turn(
                session_id=params.session_id,
                event_store=self.event_store,
                user_prompt=params.user_prompt,
                tenant_id=params.tenant_id,
                llm_model=params.llm_model,
                llm_account=params.llm_account,
                initial_task_settings=params.initial_task_settings,
                user_prompt_event_jsonable=user_prompt_event_jsonable,
                unattended=params.unattended,
                port_key=params.port_key,
                task_manager=existing,
            )

        # `or ""` is defensive only: `Session.root_agent_id` is typed `str | None` for
        # Session's general use, but on this path it is always non-empty — create_session
        # mints it via generate_id("agt") before SESSION_CREATED, and resume_session raises
        # RuntimeError up front if the recovered projection has no root_agent_id (verified
        # 2026-09-03, Task 22). handle.agent_id is therefore always the addressable root
        # agent (see TurnHandle docstring), never "".
        handle = TurnHandle(
            session_id=session.id,
            agent_id=session.root_agent_id or "",
            task_id=root_task.id,
            template_id=params.template_id,
            event_bus=self._event_bus,
            _storage_health=self.storage_health,
            _storage_recovery=self._wait_for_storage,
            _pending_observe=self._pending_observe_run_id,
            _is_finished=self._task_is_settled,
        )

        # 这一段到 `set_runner` **不留 await**：resume 复用活 owner 时，新 root task 一进队
        # 就可能被别处的 drain（并发的 pause / send_message）派发出去，若那一刻 runner 还是
        # 上一轮的，这一轮的 run 结果就写进上一轮的 TurnHandle。从前为此要把模板解析特意
        # 提到推 task 之前（两条分支各解一次、彼此点名）；runner 不再持有模板之后，这里
        # 已经没有任何需要 await 的准备工作，那个约束自然成立。
        task_manager.set_unhealthy_check(
            lambda sid: self.storage_health(sid) is not None)  # spec: event-commit
        task_manager.set_runner(self._make_task_runner(
            session=session,
            lm=lm,
            memory=memory,
            task_manager=task_manager,
            handle=handle,
        ))
        task_manager.set_session(session)

        self._register_and_drain(session, task_manager)
        return handle

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _register_and_drain(
        self,
        session: "Session",
        task_manager: "TaskManager",
        *, start_tasks: bool = True,
    ) -> None:
        """Wire up ControlCapabilityProvider, set done callback, launch drain.

        对同一个 TM 重入是幂等的（重挂回调 + 补一次 drain）——复用活 owner 的几条路径
        都靠这一点。换一个 TM 进来则由 `_bind_task_manager` 拒绝（单例）。
        """
        # 先登记、后接线：换 TM 的非法调用必须在碰任何注册表之前就被拒绝。
        self._bind_task_manager(session.id, task_manager)

        for p in self.providers.get_capability_providers():
            if isinstance(p, ControlCapabilityProvider):
                p.register_session(session.id, task_manager, session)
                break

        # 纳入 SM 管理。setdefault 语义，重入安全：多轮对话/恢复重建都会走到这里，
        # 已有状态不被重置（新一轮的显式 RUNNING 由 resume_session 负责）。
        self._session_registry.register_session(session.id, tenant_id=session.tenant_id)

        async def _on_done() -> None:
            # compare-and-clear：仅当本 TM 仍是当前 owner 才清，避免顶替它的新 TM 被误清。
            #
            # **这里不再拆 TaskManager、也不再摘 agent record**（2026-09-08 生命周期
            # 改造）：一轮跑完不等于这条会话不要了——agent 只是回到 `idle`，随时可以
            # 接下一条消息，TM 也照旧能接新 task（`drain` 只被 `_cancelled` 与
            # `is_current` 挡，`_fire_session_done` 不设任何闩）。真正的回收是显式的
            # `forget_session`，由持有方按它自己的策略调用。
            if self._task_managers.get(session.id) is task_manager:
                self._release_round(session.id)

        async def _on_idle() -> None:
            # per-run token 生命周期已随 run 对齐（execute finally 注销），无需在此回收。
            # 只清 pause 弃子闩锁；compare-and-check 防被顶替旧 TM 的迟到 idle 误清新一轮闩锁。
            if self._task_managers.get(session.id) is task_manager:
                self._pausing.discard(session.id)
                self._pause_claimed.discard(session.id)
                task_manager.set_pause_abandon(False)

        # 一次性接线：8 个回调整体装好，装不出半接线的中间态（见 orchestrator/hooks.py）。
        task_manager.set_hooks(TaskManagerHooks(
            # 归属权谓词：session 的 TM 是单例，活 owner 不会被顶替；但它可能被逐出
            # （forget/purge）后又有新 TM 为这个 session 建起来。旧 TM 的收尾若迟到（被其慢
            # 的 background observe 拖住），必须认出自己已不是 owner、变 no-op，否则会冲掉
            # 新 TM 的会话状态。
            is_current=lambda tm=task_manager: self._task_managers.get(session.id) is tm,
            # 熔断真终结的三处注入：cancel 挂起 HITL / 协作取消在途 run / memory 闭合。
            # 均 best-effort——trip 序列本身不因缺失或异常而崩溃（TaskManager 侧已兜底）。
            cancel_pending_hitl=lambda sid=session.id: self._cancel_session_hitl(
                sid, message=CancelReason.FAILURE_THRESHOLD),
            cancel_inflight=lambda tid, sid=session.id: self._cancel_run_token(sid, tid),
            review_switched_task=lambda task, sess=session, tm=task_manager: self._relaunch_observe(
                session=sess, task_manager=tm, task=task,
                agent_id=task.assigned_agent_id or task.creator_agent_id or sess.root_agent_id or "", boundary="task_switch"),
            retire_switched_input=self._retire_switched_input,
            # 丢弃一轮时把 TM 够不到的两样东西撤回来：memory 里那条用户消息、
            # 被这条消息收口的旧 HITL 气泡。见 `_revert_round`。
            revert_round=lambda tid, sid=session.id: self._revert_round(sid, tid),
            # run 收尾时的自检：它等的那个问题是不是已经有答复了（见钩子定义处）。
            human_answer_ready=lambda tid, hid, sid=session.id: self._hitl_answer_ready(
                sid, tid, hid),
            # 提交一轮：待终局的 HITL 答复终局，再落盘暂存的 memory 写入。见 `_commit_round_writes`。
            prepare_round_commit=lambda tid, sid=session.id: self.hitl.stage_commits(sid, tid),
            commit_round=lambda tid, staged, sid=session.id: self._commit_round_writes(
                sid, tid, staged),
            threshold_finalizer=lambda root, ack_tasks, failures, sess=session: (
                self._finalize_threshold_memory(sess, root, ack_tasks, failures)),
            # 统一取消胶囊闭合：cancel_all / 熔断清场（已启动挂起排队）/ 在途协作取消
            # funnel 三处调用点共用同一注入点。
            cancel_finalizer=lambda tasks, reason, sess=session: (
                self._finalize_cancel_memory(sess, tasks, reason)),
            read_task_state=lambda tid, sid=session.id: self._read_persisted_task(sid, tid),
            cancel_guard=lambda tid, name="finish_report", sid=session.id: self._step_executions.setdefault(
                sid, ExecutionRegistry()).task_guard(tid, name),
            tree_cancel_finalizer=lambda tasks, reason, sess=session: self._finalize_canceled_tasks(
                sess, tasks, reason),
            confirm_task_settled=self._confirm_task_settled,
            on_session_done=_on_done,
            on_session_idle=_on_idle,
            # task 落终态 → 两件事，见 `_on_task_terminal`。挂在终态而非 run 收尾，因为同一个
            # task 可以跑多个 run（retry/resume）：pin 要跨得过重试（run 收尾的 evict 已明确
            # 不碰 pin 区，见 CapabilityCache.evict），HITL 决定同理——重试时它还要用。
            on_task_terminal=lambda tid, sid=session.id, tenant=session.tenant_id: self._on_task_terminal(sid, tid, tenant),
        ))

        self._initialized_sessions.add(session.id)
        self._loaded_sessions.add(session.id)
        if start_tasks:
            self._spawn(task_manager.drain())

    async def _on_task_terminal(self, session_id: str, task_id: str, tenant_id: str) -> None:
        """`on_task_terminal` 钩子：task 落终态时回收挂在它身上的东西。

        1. 清掉该 task 运行期 pin 进来的能力（`CapabilityCache`）。
        2. 给它名下**已终局却再也不会被消费**的 HITL 决定盖 `HitlClosed`。

        第 2 条是那套「了结」机制里最通用的退役判据：终态 task 不会再跑，
        `_inject_resolved_user_turns` 对它本就直接跳过（`target.status in
        TERMINAL_TASK_STATUSES`），gateway 也不会再为它求批。挂在这一个点上，一次覆盖
        cancel / fail / finish 三种——比逐条去堵取消路径干净，也不会漏掉正常结束那一类
        （那类同样可能留下已终局未消费的决定，只是比取消罕见）。

        **不会误伤重试**：`_settle` 判 retry 时先 return，根本走不到这个钩子（与 pin 跨重试
        保住是同一个理由），所以「还要再跑一轮」的 task 的决定不会被提前销掉。

        best-effort：钩子在 `TaskManager.on_task_finished` 里已被 try/except 包住，这里不再
        自己吞——盖章本身也不抛（见 `HitlService.close`）。
        """
        self._capability_cache.for_context(ProviderContext(
            session_id=session_id, tenant_id=tenant_id)).clear_pins(task_id)
        await self.hitl.close_resolved(session_id, task_id=task_id)

    def _memory_settled(self, session_id: str) -> bool:
        """输入暂存效果是否落地——快照写入前置条件。assistant 另有独立核对水位。

        对输入暂存区的不变式：**有快照 ⟹ 到切面为止，输入 memory 效果都已落地。** 恢复路径
        「以快照为准」全靠它；一张领先的快照是静默的数据丢失（它说某件事做完了，所以
        `pending_observe` / HITL `resolved` 里没有它，而 memory 里其实没有）。

        恢复尚未补齐必需输入时不能写快照；正常执行再检查有没有开着的未提交窗口。
        开着窗口就意味着这一轮的 memory
        写入还在暂存区（`ingest_or_stage`），而缓冲里的事件已经先于它们被补投——
        `TaskManager.commit_round` 的顺序是「补投 → 钩子（落盘暂存）→ 关窗」，而
        `SnapshotWriter` 是 rest 订阅者，在补投那一步就被叫醒了。实测见
        `tests/unit/test_snapshot_not_ahead_of_memory.py`。

        恢复完成后，窗口之外的输入写入由调用方同步等待。「恢复完成且没有开着的窗」
        保护的是这条输入边界。assistant 写入失败仍可能进入 RunFinished 的 finally；
        那类缺口必须由 assistant_memory_verified_position 保留，不能由本谓词宣称完成。

        **这个判断为什么必须在 core**：它读的是 `TaskManager` 的窗口状态，而 provider 只看得
        见 EventBus。把它留在 provider 手里，provider 只能拿「收到某个事件」当代理，而那个
        代理恰好是错的——缓冲里的每一条事件都排在它自己的 memory 效果之前。

        没有 TM（会话还没起 / 已逐出）→ True：没有在跑的轮次，也就没有暂存的写入。
        """
        tm = self._task_managers.get(session_id)
        if tm is None:
            return True
        if getattr(tm, "recovery_pending", False):
            return False
        checker = getattr(tm, "any_round_open", None)
        if not callable(checker):
            return True          # 鸭子类型的 TM 替身（单测）——不因此挡住快照
        return not checker()

    def _bind_task_manager(self, session_id: str, task_manager: "TaskManager") -> None:
        """把 TM 登记为该 session 的 owner——**唯一写 `_task_managers` 的地方**（逐出除外）。

        **一个 session 同一时刻只有一个 TaskManager**。该 session 已有别的 TM 在册时直接
        抛，绝不覆盖：被覆盖的旧 TM 已派发出去的 run 会照样跑完，而新 TM 的
        `busy_agents` 只看得见自己的 `_running_tasks`——同 agent 串行这条保证就此失效，
        同一个 agent_id 上并发出两个 run（它们共用 `CapabilityCache` 里按 agent_id 存的
        那份工具面，先结束的那个 `evict` 会把另一个的工具清掉）。

        所以要「接着跑」的入口一律复用活 owner，而不是另建一个：`send_message` 的
        `_start_task_for_agent`、`start_session(resume=True)`、冷 HITL 应答与 `/resume`
        （`_recover_session_locked`）。只有 session 在内存里没有 TM 时（从未建过、进程
        重启、已被 forget/purge 逐出）才新建。撞上这里的异常 = 某个入口漏了复用，是 bug。
        对同一个 TM 重复登记是幂等的。
        """
        current = self._task_managers.get(session_id)
        if current is not None and current is not task_manager:
            raise RuntimeError(
                f"session {session_id!r} already has a live TaskManager; a session has "
                f"exactly one — reuse it instead of creating another"
            )
        self._task_managers[session_id] = task_manager

    def _session_in_memory(self, session_id: str) -> bool:
        """这个 session_id 在本进程里是否已经有主（有 TM，或已登记过 agent）。

        「新建 session」前的判重用：两处任一命中，再建一次都会与既有的那份冲突
        （第二个 TM / 第二条 SESSION_CREATED）。只看内存——冷的、只在事件日志里的
        session 不在此列。
        """
        return (
            session_id in self._task_managers
            or bool(self._agent_lifecycle_manager.agent_ids_of_session(session_id))
        )

    def _release_round(self, session_id: str) -> None:
        """一轮跑完之后的**轻量**清理：只清这一轮的控制信号残余。幂等。

        **不碰 TaskManager、不碰 agent record、不碰 scoped provider**（2026-09-08 生命
        周期改造）。从前这里是 `_release_session`，会话一跑完就把 TM 和该 session 下
        全部 ALM record 一起拆掉——而 agent 在概念上只是回到 `idle`（spec 3.1），
        `send_message` / `list_agents` / `get_agent` 却随即全部瞎掉（一条正常结束的
        会话再发消息 → `AgentNotFound`）。现在内存里的东西是**缓存**：只增不删，
        什么时候收由持有方显式调 `forget_session` / `forget_agent` 决定。

        留在这里的三样都是**per-run 控制信号**，跨轮留着有害而非有用：
        `_run_tokens` 的空壳（真正的注销在 `_SessionTaskRunner.execute` 的 finally）、
        以及 pause 弃子的两个闩锁——`_on_idle` 只在 park 那条路上清它们，done 这条路
        不清的话下一轮开跑会撞上一个上一轮遗留的"正在暂停"标志。
        """
        self._run_tokens.pop(session_id, None)
        self._pausing.discard(session_id)
        self._pause_claimed.discard(session_id)

    #: 「安静」的 agent 状态：不在跑、不等人、不等 /resume。只有这两态的 agent 可以被
    #: 忘掉——其余三态各自还有人/事在等它，record 是那件事的载体。
    _QUIESCENT_AGENT_STATUSES = frozenset({"idle", "terminated"})

    def session_is_quiescent(self, session_id: str) -> bool:
        """这条会话此刻安静吗？= 没有在跑的活、没有人在等它。`forget_*` 的准入判据。

        三条都要满足，缺一条就还不能忘：

        - **TaskManager 队列已空、无 task 在跑**（`is_done()`）。光看 agent 状态不够：
          一个 `PENDING` 还没被派发的 task，其 agent 在 ALM 里仍是 `idle`——此时逐出
          会把 TM 连同那个排着队的 task 一起丢掉，它永远不会跑。TM 不在内存里视为满足
          （压根没有队列可言）。
        - **没有未决 HITL**：有人正等着回答，`hitl_registry` 里那条记录还要用。
        - **没有待终局的答复、没有开着的未提交窗口**：这两样都表示「有一轮正开着」——答复
          刚收下还没落定、事件还攒在缓冲里。此刻逐出 TM 会把那一轮连同它的暂存一起丢掉，
          而它们在 `list_pending` 里是看不见的（待终局的请求刻意不出现在那份列表）。
        - **每个 agent 都处于 `idle` / `terminated`**：`running` 在跑；`waiting_human`
          在等人；`interrupted` 在等 `/resume`，而续跑要用那份内存状态。

        判据放在 runtime 而不是 ALM：这三样分属 `_task_managers` / `hitl_registry` /
        `AgentLifecycleManager`，只有组合根同时认识它们。
        """
        tm = self._task_managers.get(session_id)
        if tm is not None and not tm.is_done():
            return False
        if self.hitl_registry.list_pending(session_id=session_id):
            return False
        if self.hitl_registry.claim_pending_for_session(session_id):
            return False
        if tm is not None and tm.open_round_task_ids:
            return False
        reg = self._agent_lifecycle_manager
        for aid in reg.agent_ids_of_session(session_id):
            rec = reg.record_of(aid)
            if rec is not None and rec.status not in self._QUIESCENT_AGENT_STATUSES:
                return False
        return True

    def forget_session(self, session_id: str) -> bool:
        """忘掉一条**已经安静下来**的会话，释放它占的内存。返回是否真的忘了。

        这是**缓存逐出**，不是删除：不终结任何东西、不发任何事件、不碰事件日志。会话
        本身一点没少——真相源始终是事件日志，任何入口撞上 miss 都能用 `rebuild_session`
        装填回来（`send_message` 自己就会走这条自愈）。所以调它是安全的，随便调。

        **还在跑 / 还有人等着 → 拒绝，返回 False，什么都不做**（判据见
        `session_is_quiescent`）。不是"由调用方保证"，是这里自己把关：一个安全的逐出
        接口不该要求每个调用方都先背一遍不变量。硬要销毁一条还活着的会话是**另一件
        事**，走 `purge_session`。

        逐出什么：TaskManager、该 session 的全部 agent record、成员登记、per-run 控制
        信号残余、scoped provider（fs workspace、control 的 TM 注册等）。

        **`_resume_locks[session_id]` 刻意不收**：它可能正被一个在
        `_recover_session_locked` 里的协程持有着。把字典项弹掉不会让那个协程放手，只会
        让下一个 `recover_session` `setdefault` 出**第二把锁**并直接进去——同一 session 上
        两条重建路径并行，正是这把锁存在的理由。它按 session 数有界、体量微小，不回收
        是构造期就定下的（见 `__init__` 里那行注释）。
        """
        if self._session_lock(session_id).locked() or not self.session_is_quiescent(session_id):
            logger.info("forget_session: %s 还没安静下来（在跑 / 有人在等），不逐出", session_id)
            return False
        self._evict_session_memory(session_id)
        return True

    def forget_agent(self, agent_id: str) -> bool:
        """忘掉**单个**已经安静下来的 agent record。返回是否真的忘了。

        `forget_session` 的单点版本，用于"一棵委派子树跑完、父会话还开着"这类精确回收。
        同样是缓存逐出、同样自己把关：`running` / `waiting_human` / `interrupted` 一律
        拒绝——record 是五态机的载体，而 `ALM.handle_event` 对未登记的 agent_id 是**静默
        return**，删掉一个还活着的 agent 等于让它后续的 `TASK_*` 全部落进黑洞，状态机
        停在原地再也不动。要终结一个在跑的 agent 用 `cancel_agent`（唯一的 agent 终态
        入口，会发 `AgentTerminated`），终结之后它就是 `terminated`，可以忘了。

        **不看同 session 其他 agent，也不看 task 队列**：那是 `forget_session` 的粒度。
        这里只对这一个 agent 负责。
        """
        rec = self._agent_lifecycle_manager.record_of(agent_id)
        if rec is None:
            return False
        if self._session_lock(rec.session_id).locked() or rec.status not in self._QUIESCENT_AGENT_STATUSES:
            logger.info("forget_agent: agent %s 处于 %s，不逐出", agent_id, rec.status)
            return False
        self._loaded_sessions.discard(rec.session_id)
        return self._agent_lifecycle_manager.forget_agent(agent_id)

    @_session_command(load_session="optional")
    @runtime_command
    async def purge_session(self, session_id: str) -> None:
        """**销毁**一条会话的运行时存在：终结还在跑的一切，然后把内存清干净。

        与 `forget_session` 的分野就是"要不要动这条会话本身"：

        - `forget_session` 只是**忘记**——不终结任何东西、不发任何事件，会话一点没少，
          随时能装填回来。所以它对还活着的会话直接拒绝。
        - `purge_session` 是**销毁**——先 `cancel_session`（收口未决 HITL → 清队列 →
          逐个 `cancel_agent` 到 `terminated`，顺序纪律在那里），再无条件逐出内存。
          调用方拿它来实现"删除这条会话"这类不可逆操作。

        逐出的范围也比 `forget_session` 宽一档：它额外摘掉 `hitl_registry` 里这条会话的
        全部记录。`forget_session` 刻意留着那些已终局记录（`resolved_for_session` 的崩溃
        窗口兜底要用，而那条路上会话随时能装填回来）；purge 之后会话不会回来了，留着
        就是孤儿。

        ⚠ 它**不删事件日志**。core 不管持久化：会话的事实住在 event store 里，那是
        调用方（host）自己的删除步骤，本方法只负责运行时这一半。所以严格说它是"停掉
        并遗忘"，不是"抹掉存在过"。

        逐出这一步**不再过 `session_is_quiescent`**：`cancel_agent` 是协作取消，在途 run
        要跑到下一个检查点才真正收尾，`tm.is_done()` 在这一刻很可能还是 False。既然
        整条会话都要销毁了，等它把这一轮跑完没有意义——那些 run 的结局不会有人再看。

        **终结之前先确保它在内存里**：`cancel_session` 读的是 `hitl_registry` 与 ALM
        的内存现状——一条已经被 `forget_session` 逐出（或进程刚起来还没 `recover` 到）
        的会话，那两处都是空的，于是「收口未决 HITL」「逐个 `cancel_agent`」全都一次
        不进，静默跳过。表现是删掉的会话在重启后又冒出一条未决提问：`rebuild_session` 按
        「有 `HitlOpened` 无终局事件」把它当未决恢复了出来。装填一次只是读事件日志喂
        内存，不跑任何东西；会话本就不在事件日志里则装填出 0 条，后面几步自然全是
        no-op。
        """
        try:
            if not self._agent_lifecycle_manager.agent_ids_of_session(session_id):
                await self.rebuild_session(session_id)
            await self.cancel_session(session_id)
        except Exception:
            # 终结失败不该让内存永远留着（调用方多半正在删这条会话，没有第二次机会）。
            logger.exception("purge_session: cancel_session 失败，仍继续逐出内存 (%s)", session_id)
        # HITL 记录也摘掉。**只有这条路摘**，`forget_session` 那条不摘——那边会话随时能
        # 从事件日志装填回来，已终局记录还要给 `resolved_for_session` 的崩溃窗口兜底用；
        # 这边会话要没了，留着就是指向一个再也重建不出来的 session 的孤儿，而 `gc()` 只按
        # `max_resolved` 裁剪最旧的，要等它被后来的挤出去才消失。
        # 放在 `cancel_session` **之后**：那一步刚把未决的逐个终局掉，此刻摘的应当全是
        # 已终局项（真摘到未决的，registry 会记一条 WARNING）。
        #
        # **摘之前先盖章**：摘掉之后就没有请求对象可盖了，而事件日志还在（core 不删日志），
        # 那些已终局未消费的决定会永远留在折叠的清单里。上面的 `cancel_session` 正常路径
        # 已经盖过一轮，这里是它抛异常时的兜底——`close()` 按内存 `closed` 标记幂等，正常
        # 路径下这一次是 no-op，不会发重复的章。
        try:
            await self.hitl.close_resolved(session_id)
        except Exception:
            logger.exception("purge_session: close_resolved failed for %s", session_id)
        self.hitl_registry.forget_session(session_id)
        self._evict_session_memory(session_id)
        self._initialized_sessions.discard(session_id)
        self._fork_creations.pop(session_id, None)

    def _evict_session_memory(self, session_id: str) -> None:
        """无条件把该 session 的运行时内存摘干净。`forget_session`（过判据之后）与
        `purge_session`（销毁路径）共用——摘什么、摘的顺序只写一份。幂等。

        **不收 `hitl_registry` 里该 session 的条目**：`HitlRegistry` 没有按 session 清的
        口子（`gc()` 是全局的，resolved 那半由 `max_resolved` 自己封顶）。两条调用路径
        都不需要它：`forget_session` 的判据本就要求无未决 HITL；`purge_session` 之前的
        `cancel_session` 已经把它们逐个终局。要真按 session 清，得先给 registry 加那个
        口，不该在这里伸手进它的内部结构。
        """
        # 逐出之前先把残留的未提交窗口丢掉：窗口的两半分别住在 TM（快照 + 暂存）与总线
        # （事件缓冲）里，只摘 TM 会让总线那半留在原地——此后该 task 的事件会一直往一个
        # 没人关的缓冲里堆。正常路径上到这里已经没有开着的窗（`cancel_session` 先收过，
        # `forget_session` 的判据也不放行），这一步是兜底。
        tm = self._task_managers.get(session_id)
        for _tid in list(getattr(tm, "open_round_task_ids", None) or ()):
            logger.warning(
                "_evict_session_memory: dropping the still-open round of task %s", _tid)
            tm.drop_round_buffer(_tid)
        self._release_round(session_id)
        self._task_managers.pop(session_id, None)
        self._capability_cache.forget_session(session_id)
        self._loaded_sessions.discard(session_id)
        self._prepared_recoveries.pop(session_id, None)
        for key in list(self._memory_seed_ready):
            if key[0] == session_id:
                self._memory_seed_ready.pop(key, None)
        self._agent_lifecycle_manager.forget_session(session_id)
        self._step_executions.pop(session_id, None)
        self._session_registry.forget_session(session_id)
        for key in list(self._assistant_memory_ready):
            if key[1] == session_id:
                self._assistant_memory_ready.pop(key, None)
        for key in list(self._assistant_pending):
            if key[1] == session_id:
                self._assistant_pending.pop(key, None)
        for _p in self.providers.get_capability_providers():
            if isinstance(_p, SessionScopedCapabilityProvider):
                _p.deregister_session(session_id)

    @runtime_command
    async def rebuild_session(
        self, session_id: str, *, tenant_id: str | None = None, strict: bool = False,
    ) -> int:
        """Ensure one complete owner, without execution. Return its agent count.

        Crash calibration is performed by initialize(session_id) before materialization.
        An existing owner is reused; missing members are repaired without replacing
        live records. strict=True raises on absence or failure; the compatibility
        non-strict path returns zero after rolling back an incomplete cold load."""
        if self._session_lock_owners.get(session_id) is not asyncio.current_task():
            try:
                await self.initialize(session_id)
            except SessionNotFound:
                if strict:
                    raise
                return 0
        lock = self._session_guard(session_id)
        async with lock:
            return await self._rebuild_session_locked(
                session_id, tenant_id=tenant_id, strict=strict)

    async def _rebuild_session_locked(
        self, session_id: str, *, tenant_id: str | None = None, strict: bool = False,
    ) -> int:
        from ctx_weft.core.control.reducers import rebuild_view
        from ctx_weft.core.models.errors import SessionNotFound

        existing = self._task_managers.get(session_id)
        # Loaded owners also pass this guard: a failed hot seed must not be
        # bypassed by the normal already-loaded fast return.
        if session_id in self._fork_creations:
            seed_view = await rebuild_view(self.event_store, session_id)
            if session_id in seed_view.sessions:
                await self._ensure_memory_seed(seed_view.sessions[session_id])
        if self._session_is_loaded(session_id):
            return len(self._agent_lifecycle_manager.agent_ids_of_session(session_id))
        view = await rebuild_view(self.event_store, session_id)
        if session_id not in view.sessions:
            if strict:
                raise SessionNotFound(session_id)
            return 0
        await self._ensure_memory_seed(view.sessions[session_id])
        if existing is not None:
            # An owner is never replaced. Repair only missing members, not live records.
            if not existing.session.root_agent_id:
                existing.session.root_agent_id = view.sessions[session_id].root_agent_id
            missing = {aid: av for aid, av in view.agents.items()
                       if self._agent_lifecycle_manager.record_of(aid) is None}
            await self._agent_lifecycle_manager.load(
                missing, session_id=session_id,
                tenant_id=existing.session.tenant_id)
            if existing._runner is None:
                existing.set_runner(self._make_task_runner(
                    session=existing.session, lm=self._agent_lifecycle_manager,
                    memory=self.providers.get_memory(), task_manager=existing))
            self._register_and_drain(existing.session, existing, start_tasks=False)
            return len(self._agent_lifecycle_manager.agent_ids_of_session(session_id))
        try:
            await self._load_hitl_view(session_id, view)
            n, actual_tenant = await self._load_agents_view(session_id, view, tenant_id=tenant_id)
            await self._load_task_manager_view(session_id, view, tenant_id=actual_tenant)
            return n
        except BaseException as exc:
            self._evict_session_memory(session_id)
            self.hitl_registry.forget_session(session_id)
            if strict or not isinstance(exc, Exception):
                raise
            logger.exception("rebuild_session: incomplete load rolled back (%s)", session_id)
            return 0

    async def _load_task_manager_view(self, session_id: str, view, *, tenant_id: str) -> None:
        """Construct and publish a fully loaded owner, with no execution queue."""
        from ctx_weft.core.control.converters import (
            session_from_projection, task_from_projection,
        )

        sess_proj = view.sessions[session_id]
        session = session_from_projection(sess_proj)
        session.tenant_id = tenant_id
        await self._restore_assistant_messages(session)
        recap_task_ids = {info.get("task_id", key) for key, info in view.pending_observe.items()}
        missing_recaps = {tid for tid in recap_task_ids
                          if tid not in view.tasks or not view.tasks[tid].details_complete}
        if missing_recaps:
            history = await self._historical_task_view(session_id)
            for tid in missing_recaps:
                if tid not in history.tasks:
                    raise TaskNotFound(f"missing task for pending recap: {tid}")
                view.tasks[tid] = history.tasks[tid]
        all_tasks = [task_from_projection(tp) for tp in view.tasks.values()]
        await self._restore_task_prompts(all_tasks, session_id, session.tenant_id)
        resumable = [t for t in all_tasks if t.status not in TERMINAL_TASK_STATUSES]
        await self._verify_task_prompts_in_memory(resumable, session_id, session.tenant_id)
        lm = self._agent_lifecycle_manager
        lm.register_session(session.id, tenant_id=session.tenant_id)
        task_manager = self._session_registry._new_task_manager(session)
        task_manager.set_session(session)
        terminal_ids = {t.id for t in all_tasks if t.status in TERMINAL_TASK_STATUSES}
        task_manager.restore(
            all_tasks, terminal_ids, enqueue=False,
            threshold_tripped=sess_proj.threshold_tripped,
        )
        task_manager.set_recovery_pending(True)
        task_manager.set_unhealthy_check(lambda sid: self.storage_health(sid) is not None)
        task_manager.set_runner(self._make_task_runner(
            session=session, lm=lm, memory=self.providers.get_memory(),
            task_manager=task_manager,
        ))
        await self._restore_appended_messages(session)
        # Publish last: a failed load must not leave an apparently ready owner.
        try:
            self._register_and_drain(session, task_manager, start_tasks=False)
        except Exception:
            if self._task_managers.get(session_id) is task_manager:
                self._task_managers.pop(session_id, None)
                self._loaded_sessions.discard(session_id)
            raise
        # Retain only deferred recovery work, not the replayed session view.
        self._prepared_recoveries[session_id] = (
            task_manager, view.tasks_total, dict(view.pending_observe))

    async def _activate_prepared_tasks_locked(
        self, tm: "TaskManager", *, skip_hitl_id: str = "",
    ) -> None:
        """Finish cold recovery after input delivery, before the first drain.

        Caller holds the session's resume lock. Hot owners have no deferred work.
        Recaps can call a model, so they belong here, never in rebuild_session.
        """
        from ctx_weft.core.loop.steps.observe import blocks_actor_recovery
        session = tm.session
        if session is None:
            raise RuntimeError("TaskManager has no session")
        prepared = self._prepared_recoveries.get(session.id)
        if prepared is None or prepared[0] is not tm:
            return
        await tm.restore_observe_decisions(prepared[2])
        from ctx_weft.core.orchestrator.lifecycle.agent_state import AGENT_CANCEL_REASON
        lm = self._agent_lifecycle_manager
        for aid in lm.agent_ids_of_session(session.id):
            rec = lm.record_of(aid)
            if rec.status == "terminated" and rec.status_reason.startswith(AGENT_CANCEL_REASON):
                if rec.status_reason == AGENT_CANCEL_REASON + "session_canceled":
                    await tm.cancel_session_agent_tasks(aid, rec.status_reason)
                elif rec.current_task_id:
                    await tm.cancel_task_tree(rec.current_task_id, rec.status_reason)
        await tm.recover_canceled_tasks()
        await self._inject_resolved_user_turns(session, tm, skip_hitl_id=skip_hitl_id)
        parked = {
            r.task_id for r in self.hitl_registry.list_pending(session_id=session.id)
            if r.task_id
        }
        parked.update(info.get("task_id", key) for key, info in prepared[2].items()
                      if blocks_actor_recovery(info))
        tm.requeue_resumable(parked)
        await tm.dispose_blocked_dependents()
        pending_observes = dict(prepared[2])
        for task in tm.all_tasks():
            if task.status == "REVIEWING" and not any(
                    info.get("task_id", key) == task.id for key, info in pending_observes.items()):
                task.switch_requested = True
                pending_observes[task.id] = {
                    "boundary": "task_switch",
                    "agent_id": task.assigned_agent_id or task.creator_agent_id or session.root_agent_id,
                }
        prepared[2].update(pending_observes)
        for key, info in pending_observes.items():
            tid = info.get("task_id", key)
            task = tm.get_task(tid)
            if task is not None:
                started = await self._relaunch_observe(
                    session=session, task_manager=tm, task=task,
                    agent_id=info.get("agent_id") or task.assigned_agent_id or "",
                    boundary=info.get("boundary") or "finish",
                    recovery_info=info if info.get("observe_id") else {"observe_id": tid},
                )
                if started:
                    prepared[2].pop(key, None)
        if not prepared[2]:
            self._prepared_recoveries.pop(session.id, None)
        tm.set_recovery_pending(False)

    async def _retire_switched_input(self, task: Task) -> None:
        tm = self._task_managers.get(task.session_id)
        if (task.session_id in self._pausing and task.switched_to_task_id
                and tm is not None and tm.session is not None
                and tm.running_agent_of(task.id) == tm.session.root_agent_id):
            # The replaced root run cannot park after its switch. Transfer its
            # pause claim before releasing its slot or dispatching the successor.
            self._pause_claimed.discard(task.session_id)
        for req in list(self.hitl_registry.list_pending(session_id=task.session_id)):
            if req.task_id == task.id:
                try:
                    await self.hitl.cancel(req.id, message="Task goal replaced")
                except PersistenceUnavailableError:
                    raise
                except Exception:
                    logger.exception("Failed to retire replaced task input %s", req.id)

    async def _cancel_session_hitl(self, session_id: str, *, message: CancelReason) -> None:
        """取消该 session 全部未决 pending HITL（best-effort，逐个 cancel）。两个调用方：
        熔断 trip 序列第 3 步注入（`message=FAILURE_THRESHOLD`）、`cancel_session` 用户
        主动取消会话（`message=USER_CANCEL`）——各传各的判别值，不再硬编码熔断专属。

        单条取消失败不阻断其余——HitlResolved 需尽量全部先于会话终态发出，但这不是硬要求。
        """
        for req in list(self.hitl_registry.list_pending(session_id=session_id)):
            try:
                await self.hitl.cancel(req.id, message=message)
            except Exception:
                logger.exception(
                    "_cancel_session_hitl: cancel failed for session=%s hitl=%s", session_id, req.id,
                )

    def _cancel_run_token(self, session_id: str, task_id: str) -> bool:
        """trip 序列第 5/6 步注入：对指定在途 task 发协作取消信号（查 `_run_tokens`）。

        只发信号，不代表任务立即终结：本 run 收尾时是否发 TASK_CANCELED 由 `_run_loop`
        finally 的 was_cancelled 守卫按 task.status 判定（熔断已先手标 FAILED 的不发）。
        找不到 token（该 task 此刻并不在跑）→ False。
        """
        if self._is_writing_task_finalized(session_id, task_id):
            return False
        executions = self._step_executions.get(session_id)
        if executions is not None:
            executions.stop_metadata(task_id)
        tokens = self._run_tokens.get(session_id, {}).get(task_id)
        if tokens is None:
            return False
        tokens.cancel.cancel()
        return True

    async def _finalize_threshold_memory(
        self,
        session: Session,
        root_task: "Task | None",
        ack_tasks: list[Task],
        failures: list[tuple[str, str]],
    ) -> None:
        """trip 序列第 7 步注入：memory 闭合（内联 await，落在 SESSION_FINISHED/SSE 关闭之前）。

        - ack_tasks（已启动带框、被熔断打断的子任务）：把它们的派发对 tool 槽替换为「取消
          文案」——若某条在途任务恰好赶在取消信号前正常收尾，FinalizeStep 的 replace=True
          会再次替换为真实终态，自愈为真相，本次替换不是最后写者也无妨。
        - root finish 对：仅当熔断亲手把 root 判 FAILED 且它已经 started_at（有胶囊可闭）时才写；
          root 已经是别的路径判的终态（比如 FinalizeStep 已闭合）时 caller 传 None，此处直接跳过。
        best-effort per item：单条异常记日志、不阻断其余写入（TaskManager 侧对整个回调的异常
        也已兜底，这里的粒度是"一个坏任务不能拖累其它任务"）。
        """
        from ctx_weft.core.loop.steps.finalize import (
            _ensure_dispatch_frame,
            _put_dispatch_result,
            _synthesize_dispatch_pair,
        )

        memory = self.providers.get_memory()

        for t in ack_tasks:
            try:
                parent_scope = MemoryAddress(
                    session_id=session.id, task_id=t.parent_task_id, agent_id=t.creator_agent_id,
                )
                provider_ctx = ProviderContext(
                    session_id=session.id, tenant_id=session.tenant_id,
                    task_id=t.parent_task_id, agent_id=t.creator_agent_id,
                )
                ts, tool_call_id = await _ensure_dispatch_frame(
                    memory, parent_scope, t, provider_ctx)
                await _put_dispatch_result(
                    memory, parent_scope, t,
                    runtime_prompt('threshold_subtask_cancelled', ref=task_ref(t)),
                    ts, provider_ctx, replace=True, tool_call_id=tool_call_id,
                )
            except Exception:
                logger.exception(
                    "_finalize_threshold_memory: ack replace failed for task %s", t.id,
                )

        if root_task is not None and root_task.started_at:
            try:
                scope = MemoryAddress(
                    session_id=session.id, task_id=root_task.id, agent_id=session.root_agent_id,
                )
                provider_ctx = ProviderContext(
                    session_id=session.id, tenant_id=session.tenant_id,
                    task_id=root_task.id, agent_id=session.root_agent_id,
                )
                summary = "Failure threshold hit — consecutive failures: " + "; ".join(
                    f"{i}) {ref}: {reason}" for i, (ref, reason) in enumerate(failures, start=1)
                )
                await _synthesize_dispatch_pair(
                    memory, scope, root_task,
                    act_recap=runtime_prompt('threshold_tripped_recap'),
                    task_summary=summary,
                    outcome="fail",
                    provider_ctx=provider_ctx,
                )
            except Exception:
                logger.exception(
                    "_finalize_threshold_memory: root finish pair failed for task %s", root_task.id,
                )

    async def _finalize_cancel_memory(
        self, session: Session, tasks: list[Task], reason: str, *, strict: bool = False,
    ) -> None:
        """统一取消胶囊闭合（Task 14）：cancel_all / 熔断清场（已启动挂起排队） / 在途协作取消
        funnel（`on_task_finished(CANCELED)`）三处调用点的公共落点，逐任务调用
        `synthesize_cancel_closure`（ack 终态化 + `[outcome=cancelled]` finish 对，find-only
        对 born-cancel 未铸框的子任务整体跳过）。

        默认逐任务 best-effort；停驻取消及恢复使用 strict，失败时保留 observe 待办，
        不能先宣布 Applied 再丢掉未完成的取消收尾。
        """
        from ctx_weft.core.loop.steps.finalize import synthesize_cancel_closure

        memory = self.providers.get_memory()
        for t in tasks:
            try:
                provider_ctx = ProviderContext(
                    session_id=session.id, tenant_id=session.tenant_id,
                    task_id=t.id, agent_id=t.assigned_agent_id or t.creator_agent_id,
                )
                registry = self._step_executions.setdefault(session.id, ExecutionRegistry())
                async with registry.task_guard(t.id, "finish_report"):
                    await synthesize_cancel_closure(memory, session.id, t, provider_ctx, reason)
            except Exception:
                if strict:
                    raise
                logger.exception("_finalize_cancel_memory: closure failed for task %s", t.id)

    def _make_task_runner(
        self,
        *,
        session: Session,
        lm: AgentLifecycleManager,
        memory: MemoryProvider,
        task_manager: TaskManager,
        handle: "TurnHandle | None" = None,
    ) -> "_SessionTaskRunner":
        """构造本 session/run 的两阶段 runner（原闭包工厂的显式化）。

        不再收 llm_account/llm_model：这次要用的模型由 assemble() 经
        `AgentLifecycleManager.materialize`/`resolve_model` 按 agent record 的
        `ModelChoice` 现解，不再是 runner 构造期就定死的会话级值。

        **也不再收 template/template_id**（2026-09-30，同一条理由的第二次应用）：模板
        同样由 assemble() 按执行 agent 的 record 现解，见 `_SessionTaskRunner._template_of`。
        """
        return _SessionTaskRunner(
            runtime=self, session=session,
            lm=lm, memory=memory,
            task_manager=task_manager,
            handle=handle,
        )

    # ── Crash recovery ───────────────────────────────────────────────────────

    @_session_command(load_session="required", recover_storage_first=True)
    async def recover_session(self, session_id: str) -> None:
        """Continue recoverable work in this session, without creating a new task.

        Repeated calls reuse the same owner and never answer pending HITL requests.
        Cold initialization and storage confirmation precede the guarded recovery.
        """
        await self._recover_session_locked(session_id)

    async def _recover_session_locked(
        self,
        session_id: str,
        *,
        user_reply: "PendingHitl | None" = None,
        resumed_task_id: str | None = None,
        hitl_id: str = "",
        keep_alive: bool = False,
    ) -> None:
        """Load through rebuild_session, then deliver input and resume one owner."""
        tm = self._task_managers.get(session_id)
        if tm is None or not tm.is_alive():
            await self._rebuild_session_locked(session_id, strict=True)
            tm = self._task_managers.get(session_id)
        if tm is None:
            raise RuntimeError(f"Session {session_id!r} not found in event store")
        if keep_alive:
            return
        prepared = self._prepared_recoveries.get(session_id)
        cold = prepared is not None and prepared[0] is tm
        resumable = any(t.status not in TERMINAL_TASK_STATUSES for t in tm.all_tasks())
        if cold and not resumable and not prepared[1]:
            # A valid container has no tasks. Cold and hot recovery are both no-ops.
            self._prepared_recoveries.pop(session_id, None)
            tm.set_recovery_pending(False)
            return
        try:
            await self._recover_in_existing_tm(
                tm, user_reply=user_reply, resumed_task_id=resumed_task_id, hitl_id=hitl_id,
            )
        finally:
            # Hot failures retain the affected owner's slot/round. Do not leave
            # unrelated queued tasks behind a session-wide preparation barrier.
            # A cold owner still awaiting required input remains blocked.
            if session_id not in self._prepared_recoveries:
                tm.set_recovery_pending(False)
        if cold and not resumable:
            await tm.finalize_idle_session()

    async def _verify_task_prompts_in_memory(
        self, tasks: "list[Task]", session_id: str, tenant_id: str,
    ) -> None:
        """恢复期核对：投影判定「提问已在 memory」的 task，memory 里是否真有那条提问。

        `task_from_projection` 只能按状态**猜**：ACTIVE / SUSPENDED 猜「已写进 memory」，
        其余猜「没写」。两个方向都会错，所以这里对着 memory 核实，两边都修正：

        - 猜「已写」却没有：新 task 的提问在第一轮算数前只在暂存区里，提交时事件**先**落盘、
          暂存**后**落盘，崩在中间就是这一形状。信了猜测，driver 不再写它，这个 task 从此
          缺了原始提问。
        - 猜「没写」其实有（`AWAITING_HUMAN` / `INTERRUPTED` / 恢复后的 `PENDING`）：driver
          会**再写一条**，时间戳是恢复时刻、排在对话末尾。实测复现过。

        提问本身不会丢：它在 `TaskCreated`（重排过的话是 `TaskRequeued`）的 payload 里，
        `_restore_task_prompts` 刚把它还原到 `task.user_prompt` 上。

        「这条提问在不在」的判据有两道，命中一道即算在：
        1. 确定性 id（`task_prompt_record_id`）已在视图里——新数据走这条，提问被改写过
           （存量 reopen 数据）时哈希不同，于是如实判「不在」，修订版照常写入；
        2. 存量数据（自动 id）：该 task 的 TASK 层 user 回合、不带 `metadata["source"]`
           （那是 HITL 应答 / 注入消息）、且拍平文本与当前提问一致。L3 坍缩物以原文开头，
           因此用前缀匹配。

        读写失败由 rebuild_session 的 strict 策略处理，不能宣布补写成功。
        """
        from ctx_weft.core.loop.driver import task_prompt_record_id
        from ctx_weft.protocols.content import content_to_text
        from ctx_weft.protocols import MemoryKind as _MK
        memory = self.providers.get_memory()
        ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id)
        for task in tasks:
            if not task.user_prompt:
                continue
            view = await load_records(memory,
                MemoryAddress(session_id=session_id, task_id=task.id),
                MemoryScope.TASK, ctx, kinds=[_MK.CONVERSATION_TURN],
            )
            want_id = task_prompt_record_id(task.id, task.user_prompt)
            want_text = (task.user_prompt if isinstance(task.user_prompt, str)
                         else content_to_text(task.user_prompt)).strip()
            present = any(
                r.id == want_id or (
                    r.role == "user" and ("source" not in (r.metadata or {}) or (
                        r.id == f"userprompt:{task.id}"
                        and (r.metadata or {}).get("source") == "resume_heal"))
                    and (r.metadata or {}).get("task_id", task.id) == task.id
                    and want_text
                    and (r.content if isinstance(r.content, str)
                         else content_to_text(r.content)).strip().startswith(want_text)
                )
                for r in view
            )
            if present == task.user_prompt_in_memory and (present or task.status == "PENDING"):
                continue
            if present:
                logger.info(
                    "_verify_task_prompts_in_memory: task %s (%s) already has its prompt in "
                    "memory — not writing it a second time", task.id, task.status)
            else:
                logger.warning(
                    "_verify_task_prompts_in_memory: task %s is %s but its prompt is not in "
                    "memory (crashed between commit and flush) — re-ingesting from the event log",
                    task.id, task.status)
            task.user_prompt_in_memory = present
            if not present and task.status != "PENDING":
                # A committed run can be canceled immediately after materialization,
                # before a driver ever gets a chance to flush its missing input.
                from ctx_weft.core.loop.driver import persist_task_prompt
                await persist_task_prompt(task,
                    MemoryAddress(session_id=session_id, task_id=task.id,
                                  agent_id=task.assigned_agent_id or task.creator_agent_id),
                    memory, ctx)

    async def _restore_task_prompts(
        self, tasks: "list[Task]", session_id: str, tenant_id: str,
    ) -> None:
        """恢复态的 prompt 从 event ref 转回 memory ref。**逐 task 独立降级：任一 task
        转换失败只降级它自己，不牵连其他 task、更不中断整场恢复。**

        转换前先把事件侧的原样形态快照到 `user_prompt_event_jsonable`（见 Task 5）：
        它就是从事件里读来的那一份，零成本、且与首次发射逐字节相同。

        本函数刻意不走 `validate_content`——它的输入直接来自事件重放，不是入口，
        套不上「先 validate 后 normalize」那条不变量（`_validate_and_normalize_content`，
        `runtime.py:573` 附近）。事件日志可能损坏、`EventBlobStore` 实现也可能有 bug，
        `hydrate_event_content` / `normalize_content` 因此可能抛出——`normalize_content`
        对 base64 解码刻意不做 try/except（`content.py` 该函数 docstring 原话），
        正是假定调用方已经过 `validate_content` 筛过一轮，而这里明确没有这层保证。
        抛出时把该字段整体降级成 `downgrade_images_to_text` 产出的确定性文本占位：
        **不允许任何解不开的 ref（event 侧或 memory 侧）流进 task 字段**，并用
        `logger.error` 记下 task id / field name，让问题看得见，而不是被这层降级
        悄悄吞掉。崩溃恢复是最不能再崩一次的地方——一个 task 的坏数据不该拖垮
        整场会话恢复。
        """
        from ctx_weft.protocols.content import content_to_jsonable, downgrade_images_to_text, hydrate_event_content, normalize_content

        event_blob_store = self.providers.get_event_blob_store()
        blob_store = self.providers.get_memory_blob_store()
        ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id)
        for task in tasks:
            content = task.user_prompt
            if not content:
                continue
            # 快照恒先于「要不要转换」的判断：纯文本 prompt 也必须落这一份
            # （str 经 content_to_jsonable 往返即自身、零成本）。
            task.user_prompt_event_jsonable = content_to_jsonable(content)
            if isinstance(content, str):
                continue  # 纯文本无 ref 可转，零 blob IO 直通
            try:
                hydrated = await hydrate_event_content(
                    content, event_blob_store=event_blob_store, ctx=ctx)
                if blob_store.can_externalize:
                    hydrated = await normalize_content(
                        hydrated, blob_store=blob_store, ctx=ctx)
            except Exception:
                logger.error(
                    "_restore_task_prompts: task_id=%s user_prompt 转换失败，"
                    "降级为纯文本占位（不让解不开的 ref 混进 memory）",
                    task.id, exc_info=True,
                )
                hydrated = downgrade_images_to_text(content)
            task.user_prompt = hydrated

    async def _restore_observe_outputs(self, session_id, task, recovery_info):
        """Rebuild the actor's output at the original scheduling event, before retrying observe.

        Use log order and the intent's foreground run, so later replies, previous
        attempts and observer responses cannot become the recovered answer.
        """
        from ctx_weft.core.loop.steps.act import TurnRecord, _synthesize_final_outputs
        from ctx_weft.protocols import ToolCall

        observe_id = recovery_info.get("observe_id")
        if not observe_id:
            return {}  # Legacy recap events have no durable scheduling intent.
        from ctx_weft.core.control.reducers import OBSERVE_REQUEST_EVENT_TYPES
        rows = await self.event_store.read_range(session_id, include_types=(
            EventType.LLM_RESPONSE_FINISHED, *OBSERVE_REQUEST_EVENT_TYPES))
        responses = []
        for row in rows:
            ev = row.event
            intent = (ev.payload.get("observe_request", {})
                      if ev.type == EventType.HITL_OPENED else ev.payload)
            if ev.type != EventType.LLM_RESPONSE_FINISHED:
                if intent.get("observe_id") == observe_id:
                    original = (await self._historical_task_view(
                        session_id, through_position=row.position)).tasks.get(task.id)
                    if original is not None:
                        task.title, task.description = original.title, original.description
                        task.user_prompt = original.user_prompt
                        await self._restore_task_prompts([task], session_id, task.tenant_id)
                    actor_run_id = ev.run_id or (responses[-1].run_id if responses else None)
                    transcript = [TurnRecord(
                        turn=response.payload.get("turn", 0), messages_sent=[],
                        assistant_text=response.payload.get("content", ""),
                        tool_calls=[ToolCall(**call)
                                    for call in response.payload.get("tool_calls", [])],
                    ) for response in responses if response.run_id == actor_run_id]
                    from types import SimpleNamespace
                    state = SimpleNamespace(task=task, extra={})
                    _synthesize_final_outputs(state, transcript)
                    state.extra["restored_transcript"] = transcript
                    return state.extra
            elif ev.task_id == task.id and ev.origin == EventOrigin.LOOP_ACT:
                responses.append(ev)
        raise RuntimeError(f"Missing observe scheduling intent: {observe_id}")

    async def _relaunch_observe(
        self, *, session: Session,
        task_manager: TaskManager, task: Task, agent_id: str, boundary: str,
        recovery_info: dict | None = None,
    ) -> bool:
        """恢复：重建 LoopState 重跑一个被崩溃打断的段 recap，登记到传入 TM（track_background）。

        close 边界由 observe 从完整持久化 finish 对恢复原文保留策略；runtime 不重新推断。

        **模板按 `agent_id` 自己的出身解**（2026-09-30）：从前它由调用方连同会话模板一起
        传进来，于是一个子 agent 被打断的段 recap 会用 root 的 identity 重跑——而 recap
        的产物是要写回那个 agent 记忆的。同 `_SessionTaskRunner._template_of` 的口径。
        """

        from ctx_weft.core.loop.steps.observe import blocks_actor_recovery
        try:
            await self._ensure_assistant_memory(session)
            if task.status == "CANCELED":
                # Repair cancellation before the obsolete observation can publish
                # Applied and release the task retained for recovery.
                await self._finalize_cancel_memory(session, [task], task.error or "canceled", strict=True)
            from copy import copy
            context_task = copy(task)
            recap_extra = {}
            # Legacy Started-only recaps have no scheduling event; their task
            # terminal event already carries outputs.
            if recovery_info and recovery_info.get("task_id"):
                recap_extra = await self._restore_observe_outputs(session.id, context_task, recovery_info)
            recap_extra["observe_context_task"] = context_task
            lm = self._agent_lifecycle_manager
            # 水合，不新建：这是重跑一个已存在 agent 打断的段 recap。register_session
            # 保证跨重启后 registry 为空时 materialize 的回落有正确的 session 语境
            # （幂等：session 已登记则不覆盖）。
            lm.register_session(session.id, tenant_id=session.tenant_id)
            agent, rm = lm.materialize(
                agent_id, session_id=session.id, tenant_id=session.tenant_id)
            template = await self._template_lookup.get_template(
                agent.template_id, None,
                ctx=ProviderContext(session_id=session.id, tenant_id=session.tenant_id),
            )
            memory = self.providers.get_memory()
            scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=agent.id)
            provider_ctx = self._build_provider_ctx(session, task, agent)
            assembler = self._build_assembler(
                memory, provider_ctx, task_manager=task_manager)
            gateway = self._build_gateway(memory, provider_ctx)
            llm = rm.client
            loop_ctx = self._build_loop_ctx(
                assembler, llm, memory, provider_ctx, gateway, None, task_manager,
            )
            state = LoopState(
                run_id=generate_id("run"), session=session, task=task, agent=agent,
                transcript=recap_extra.pop("restored_transcript", []), act_exit_reason=boundary,
                scope=scope, extra={"template": template, **recap_extra,
                    "observe_recovery": dict(recovery_info or {})}, resolved_model=rm,
                # 孤儿 run：不走 StepDriver.run，构造时显式钉住 origin，否则
                # 后台 ObserveStep 内部经 make_event(state, ...) 发的
                # 事件 origin 都会是空串。
                origin=EventOrigin.LOOP_BACKGROUND_OBSERVE,
            )
            # A restarted observer has no preceding PrepareStep to bind its tools.
            from ctx_weft.core.loop.steps._capabilities import resolve_and_bind
            state.extra["bound_capabilities"] = await resolve_and_bind(state, loop_ctx)
            if recovery_info and "context_record_ids" not in recovery_info:
                from ctx_weft.core.control.reducers import OBSERVE_REQUEST_EVENT_TYPES
                rows = await self.event_store.read_range(session.id, include_types=OBSERVE_REQUEST_EVENT_TYPES)
                for row in rows:
                    payload = (row.event.payload.get("observe_request", {})
                               if row.event.type == EventType.HITL_OPENED else row.event.payload)
                    if payload.get("observe_id") == recovery_info.get("observe_id") and "context_record_ids" in payload:
                        recovery_info = {**recovery_info, "context_record_ids": payload["context_record_ids"]}
                        break
            info = dict(recovery_info or {})
            mechanical = bool(info.get("verdict"))
            settled = info.get("settled") or info.get("abandoned") or task.status == "CANCELED"
            enqueue_pending = (info.get("settled") and task.status == "PENDING"
                               and info.get("disposition", {}).get("status") == "PENDING")
            if mechanical and settled and not enqueue_pending:
                launched = await loop_ctx.steps.start_durable("observe", state, loop_ctx, options={
                    **info, "boundary": info["report_boundary"], "judge_task": False, "settled": True})
            elif mechanical or info.get("execution_mode") == "foreground":
                # A saved verdict still owns foreground work until settlement:
                # its fold/finalize must have the same slot and stop tokens as
                # fresh judgement. Already settled replay must stay idempotent.
                needs_foreground = (not mechanical and info.get("judge_task", True)
                                    and not settled
                                    and task.status not in TERMINAL_TASK_STATUSES)
                if needs_foreground:
                    launched = task_manager.start_foreground_recovery(task.id, agent_id,
                        lambda: self._resume_observe_finalization(state, loop_ctx, info, foreground=True))
                else:
                    launched = self._spawn(self._resume_observe_finalization(state, loop_ctx, info))
                loop_ctx.steps.executions.register(state.run_id, task.id, "observe", launched,
                    agent_id=agent_id, observe_id=info.get("observe_id", ""))
                task_manager.track_background(launched)
            else:
                launched = await loop_ctx.steps.start_durable(
                    "observe", state, loop_ctx, options={"boundary": boundary, **info})
            if blocks_actor_recovery(recovery_info or {}):
                registry = loop_ctx.steps.executions
                registry.recovering_observe_tasks.add(task.id)
                launched.add_done_callback(lambda _: registry.recovering_observe_tasks.discard(task.id))
            return True
        except Exception:
            logger.exception("recover: failed to relaunch task recap for task=%s", task.id)
            return False

    async def _resume_observe_finalization(self, state, ctx, info, *, foreground=False):
        """Recover foreground settlement here; observation never drives a step chain."""
        from ctx_weft.core.loop.driver import ExecutionMode, StepExecution
        from ctx_weft.core.loop.steps.observe import (
            ObservationResult, Verdict, apply_foreground_assessment, capture_context, emit_effect,
        )
        options = dict(info)
        ctx.execution = StepExecution(ExecutionMode.FOREGROUND, options)
        await capture_context(state, ctx, options)
        state.extra.update(observe_id=options["observe_id"],
            observe_effects=list(options.get("effects", [])))
        state.extra["observe_fork_options"] = options
        if options.get("completion"):
            state.extra["observe_completion"] = options["completion"]
        mechanical = bool(options.get("verdict"))
        commit = None
        tokens = None
        if foreground:
            state.origin = EventOrigin.LOOP_OBSERVE
            tokens = self._register_run_tokens(state.session.id, state.task.id,
                root_run=state.agent.id == state.session.root_agent_id)
            ctx.cancel_token, ctx.pause_token = tokens.cancel, tokens.pause
            if ctx.task_manager._cancelled or ctx.task_manager.is_task_tree_canceled(state.task.id):
                tokens.cancel.cancel()
        started = make_event(state, EventType.RUN_STARTED, {
            "run_id": state.run_id, "initial_step": "observe_finalization_recovery"},
            origin=EventOrigin.RUNTIME if foreground else state.origin)
        if foreground:
            commit = _ActorRunCommit(started)
            self._pending_actor_runs[(state.session.id, state.task.id)] = commit
        error = None
        cancelled = False
        try:
            await ctx.event_bus.emit(started)
            if commit is not None:
                commit.start_confirmed = True
            # Recap-only execution acquires this guard inside ObserveStep. Keep
            # recovery's outer guard only for judgement/finalization paths that
            # do not acquire it themselves; asyncio.Lock is not reentrant.
            guard = (ctx.steps.executions.task_guard(state.task.id, "observe")
                     if mechanical or options.get("judge_task", True) else contextlib.nullcontext())
            async with guard:
                if mechanical:
                    result = ObservationResult(verdict=Verdict(**options["verdict"]))
                    state.extra["observe_decision"] = options
                    apply_foreground_assessment(state, result, state.act_exit_reason)
                    state = state.apply_patch({"observation": result, "verdict": result.verdict})
                    decision = options["disposition"]
                    finalize = not (state.task.status in {"FINISHED", "FAILED", "CANCELED"}
                                    and state.task.status != decision["status"])
                    if options.get("settled"):
                        finalize = False
                        state.run_outcome = RunOutcome(
                            kind=RunOutcomeKind.COMPLETED, verdict=result.verdict.task_outcome,
                            outputs=state.task.outputs, observe_id=options["observe_id"],
                            observe_disposition=decision, observe_report_pending=True)
                else:
                    if foreground:
                        state, observed = await ctx.steps._execute_step(ObserveStep(), state, ctx, foreground=True)
                    else:
                        observed = await ObserveStep().execute(state, ctx)
                        state = state.apply_patch(observed.state_patch)
                        for event in observed.events:
                            await emit_effect(state, ctx, event)
                    finalize = observed.next_step == "finalize"
                if finalize:
                    if foreground:
                        state.origin = EventOrigin.LOOP_FINALIZE
                        state, finalized = await ctx.steps._execute_step(FinalizeStep(), state, ctx, foreground=True)
                    else:
                        finalized = await FinalizeStep().execute(state, ctx)
                        state = state.apply_patch(finalized.state_patch)
                        for event in finalized.events:
                            await emit_effect(state, ctx, event)
        except HitlPark as park:
            state.run_outcome = RunOutcome(kind=RunOutcomeKind.AWAITING_HUMAN, hitl_id=park.hitl_id)
        except asyncio.CancelledError:
            cancelled = True
            if not foreground:
                raise
            state.run_outcome = RunOutcome(kind=RunOutcomeKind.CANCELED,
                observe_id=state.extra.get("observe_id", ""))
        except Exception as exc:
            error = exc
            raise
        finally:
            finished = make_event(state, EventType.RUN_FINISHED, {
                "outcome": ("canceled" if cancelled else "interrupted" if error else
                            state.run_outcome.kind.value if foreground and state.run_outcome else "completed"),
                "final_status": state.task.status, "will_retry": False,
                "total_events": state.sequence_counter, "total_turns": len(state.transcript),
                "error": str(error) if error else None,
                "error_type": type(error).__name__ if error else None,
            }, origin=EventOrigin.RUNTIME if foreground else state.origin)
            if commit is not None:
                commit.finished = finished
            try:
                # As on the ordinary actor path, isolation retains both original
                # envelopes. Confirmation must publish Started before Finished.
                if commit is None or not isinstance(error, PersistenceUnavailableError):
                    await ctx.event_bus.emit(finished)
                    if commit is not None:
                        self._pending_actor_runs.pop((state.session.id, state.task.id), None)
            finally:
                if tokens is not None:
                    self._deregister_run_tokens(state.session.id, state.task.id, expected=tokens)
        outcome = state.run_outcome
        if foreground:
            return outcome
        if outcome is not None:
            if not options.get("settled"):
                status = await ctx.task_manager.apply_run_outcome(state.task.id, outcome)
                await ctx.task_manager._settle(state.task.id, status)
            elif outcome.observe_disposition and outcome.observe_disposition["status"] == "PENDING":
                await ctx.task_manager._settle(state.task.id, "PENDING")
            await ctx.task_manager.finish_observe(outcome, state.task.id)
        if mechanical:
            await ctx.steps.start_durable("observe", state, ctx, options={
                **options, "boundary": options["report_boundary"], "judge_task": False, "settled": True})

    async def _repair_observe_handoffs(self, tm):
        """Restore stopped opening/observe work on a live owner from the same facts as cold load."""
        sid = tm.session.id
        view = await rebuild_view(self.event_store, sid)
        await tm.reconcile_handoffs(self.hitl_registry.list_pending(session_id=sid), view.tasks)
        registry = self._step_executions.get(sid)
        pending = {key: info for key, info in view.pending_observe.items()
                   if not (registry and (registry.observe_is_live(info["observe_id"])
                           if info.get("observe_id") else
                           registry.pending_run_id(info.get("task_id", key), "observe")))
                   and not tm._dispatch_is_live(info.get("task_id", key))}
        prepared = self._prepared_recoveries.get(sid)
        if pending:
            if prepared is None:
                self._prepared_recoveries[sid] = (tm, view.tasks_total, pending)
            else:
                prepared[2].update(pending)

    async def _recover_in_existing_tm(
        self,
        tm: "TaskManager",
        *,
        user_reply: "PendingHitl | None",
        resumed_task_id: str | None,
        hitl_id: str = "",
        storage_ready: bool = False,
    ) -> None:
        """活 owner 在世时的续跑——`_recover_session_locked` 在单例下唯一的一条复用路径。

        - 带 ``resumed_task_id``（冷 HITL 应答）：交给 `_resume_in_existing_tm`，重排
          被应答的那一个 task。活 owner 必然持有本 session 的全部 task（所有 task 都
          经它创建或由它 restore 进来）；不持有 = 这条不变量破了，抛出而不是另建一个
          TM 去兜——另建正是单例要消灭的东西。
        - 不带（``/resume``）：把 owner 内存里可续跑的 task 就地重排
          （`TaskManager.requeue_resumable`，判据与 `restore()` 同源，读的是内存而非事件
          重放），再补一次 drain。正在跑的、已在队列里的、还挂着未决 HITL 的都不动，
          所以对同一个 session 连按几次 ``/resume`` 是幂等的。

        冷装填 owner 的补写、依赖清理和 recap 在输入交付后、首次 drain 前完成。
        """
        from ctx_weft.core.loop.steps.observe import blocks_actor_recovery
        tm.set_recovery_pending(True)
        if not storage_ready:
            await self._confirm_storage_facts(tm.session.id)
            await self.hitl.confirm_opens(tm.session.id)
            await self.hitl.confirm_resolutions(tm.session.id)
            await self._confirm_assistant_events(tm.session)
            await self._restore_assistant_messages(tm.session)
            await self._settle_storage_assistant_memory(tm.session)
        await tm.reconcile_stopped_tasks()
        await self._repair_observe_handoffs(tm)
        if self.storage_health(tm.session.id) is not None and not self.clear_storage_isolation(tm.session.id):
            raise PersistenceUnavailableError("Unconfirmed submissions still block session recovery")
        if resumed_task_id is not None:
            if tm.get_task(resumed_task_id) is None:
                sid = tm.session.id if tm.session is not None else "?"
                raise RuntimeError(
                    f"live TaskManager of session {sid!r} does not own task "
                    f"{resumed_task_id!r}; the owner holds every task of its session, so "
                    f"this is a bug — refusing to build a second TaskManager"
                )
            await self._resume_in_existing_tm(
                tm, user_reply=user_reply,
                resumed_task_id=resumed_task_id, hitl_id=hitl_id,
            )
            return
        session = tm.session
        if session is None:  # 防御：经 _register_and_drain 登记的 owner 一定注入过 session
            raise RuntimeError("live TaskManager has no session — cannot resume in place")
        if user_reply is not None:
            await self._inject_user_reply(user_reply, session, tm)
        parked = {
            r.task_id for r in self.hitl_registry.list_pending(session_id=session.id)
            if r.task_id
        }
        prepared = self._prepared_recoveries.get(session.id)
        if prepared is not None:
            parked.update(info.get("task_id", key) for key, info in prepared[2].items()
                          if blocks_actor_recovery(info))
        registry = self._step_executions.get(session.id)
        if registry is not None:
            parked.update(registry.recovering_observe_tasks)
        requeued = tm.requeue_resumable(parked)
        if requeued:
            logger.info("Recovery: session %s resumed in place on its live TaskManager: %s",
                        session.id, requeued)
        await self._activate_prepared_tasks_locked(
            tm, skip_hitl_id=getattr(user_reply, "id", "") if user_reply is not None else "")
        tm.set_recovery_pending(False)
        self._register_and_drain(session, tm)

    async def _resume_in_existing_tm(
        self,
        tm: "TaskManager",
        *,
        user_reply: "PendingHitl | None",
        resumed_task_id: str,
        hitl_id: str = "",
    ) -> None:
        """把冷 HITL 应答作为消息投递给**存活的 owner TM**，就地重驱——不重建 TM（单 owner 架构）。

        - 不碰模型：换模型走 `set_agent_llm`/`set_session_llm`，registry 现读现解，
          续跑只管把已经存在的选择重新派发出去（批次 B）。
        - 控制令牌随 run 在派发时发放（per-run registry），无需在此重建。
        - wait_for_user 冷应答注入用户回复到 task 层（`_inject_user_reply` 委托
          `TaskManager.mark_human_resolved` 发 `TaskHumanResolved`）；approval 走 reconcile，由本方法直接调 `resume_task`
          发那条事件——两条路径合起来正好各发一次，不重不漏（D4）。
        - 重排被应答的 task 并重新 drain（``_register_and_drain`` 对同一 TM 幂等：重挂回调 + 派发）。

        ``hitl_id``：仅 approval 分支（``user_reply is None``）用得到，直接传给
        `resume_task`。wait_for_user 分支不需要它——`_inject_user_reply` 用的是自己
        收到的 `user_reply.id`。省略时默认空串，仅供无真实 HITL 语境的既有测试兼容。
        """
        session = tm.session
        if session is None:  # 防御：存活 owner 一定注入过 session
            raise RuntimeError("live TaskManager has no session — cannot resume in place")
        if user_reply is not None:
            await self._inject_user_reply(user_reply, session, tm)
        # 崩溃窗口兜底也放一份在这条路上：重建路径跑过一次之后，若那时某个 task 正在跑而
        # 被跳过，它的那条已终局答复只能等下一次续跑来补——就是这里（审查文档 M8）。
        await self._inject_resolved_user_turns(session, tm, skip_hitl_id=hitl_id)
        await tm.resume_task(resumed_task_id, hitl_id=hitl_id)
        await self._activate_prepared_tasks_locked(tm, skip_hitl_id=hitl_id)
        tm.set_recovery_pending(False)
        self._register_and_drain(session, tm)

    @_session_command(load_session="skip", allow_missing_session=True)
    @runtime_command
    async def compact_agent(
        self,
        agent_id: str,
        *,
        task_id: str = "",
    ) -> CompactReceipt:
        """Run a one-shot, compact-only operation over an IDLE agent's memory.

        Folds the agent layer (dispatch log) of ``agent_id``. Pass a real ``task_id``
        to also make that task's task layer eligible. Raises ``SessionBusyError`` if
        the agent's session is currently running.

        ``session_id`` is looked up from the agent record (2026-09-04 spec §7.3:
        compact was always an agent-grained operation — hanging it off the session
        had it backwards). An unregistered ``agent_id`` raises ``AgentNotFound``.

        Executes a foreground CompactStep through the step driver and emits a
        matching RunStarted/RunFinished pair around it (总账 C5: an orphan run_id
        with no start/finish confused hosts) — the session's projection status is
        still untouched, since RunStarted/RunFinished are reducer no-ops just like
        MemoryCompactStarted / MemoryCompacted.

        Returns a ``CompactReceipt``. When ``task_id`` is not supplied, the receipt's
        ``task_id`` is a transient in-memory carrier id with no event-store record (it
        only scopes the fold) and ``task_id_is_transient`` is ``True``; callers should
        not try to look it up.

        Note: a concurrent ``pause_session`` while a compact is in flight is not
        honoured mid-compact — ``CompactStep`` does not poll the pause token — but the
        idle-guard still prevents a new compact/drain from starting on this session.
        """
        import dataclasses as _dc

        from ctx_weft.core.control.converters import session_from_projection
        from ctx_weft.core.control.reducers import rebuild_view
        from ctx_weft.core.models.errors import SessionBusyError
        from ctx_weft.core.models.task import NormalTaskSettings, Task
        from ctx_weft.protocols import MemoryAddress, ProviderContext

        lm = self._agent_lifecycle_manager
        rec = lm.record_of(agent_id)
        if rec is None:
            raise AgentNotFound(f"unknown agent: {agent_id}")
        session_id = rec.session_id
        target_agent_id = agent_id
        transient = not task_id

        # ── idle-guard: claim the slot synchronously (no await before the claim) ──
        #
        # 「忙」不止看在跑的 run：开着的未提交窗口、已收下还没终局的答复，都表示有一轮
        # 正在进行，而它的暂存 memory 压缩根本看不见（压缩只折 memory）。此刻压缩等于对着
        # 一份缺了这一轮的视图折叠。判据与 `session_is_quiescent` 同源，只是这里不看 agent 态。
        _tm = self._task_managers.get(session_id)
        if (session_id in self._busy_sessions or self._run_tokens.get(session_id)
                or self.hitl_registry.claim_pending_for_session(session_id)
                or (_tm is not None and _tm.open_round_task_ids)):
            raise SessionBusyError(session_id)
        self._busy_sessions.add(session_id)
        token = CancelToken()
        try:
            view = await rebuild_view(self.event_store, session_id)
            proj = view.sessions.get(session_id)
            if proj is None:
                raise RuntimeError(f"Session {session_id!r} not found in event store")

            session = session_from_projection(proj)
            await self._ensure_assistant_memory(session)
            pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id)
            # 水合，不新建：target_agent_id 是已存在 agent。register_session 保证跨
            # 重启后 registry 为空时 materialize 的回落有正确的 session 语境（幂等）。
            lm.register_session(session.id, tenant_id=session.tenant_id)
            agent, rm = lm.materialize(
                target_agent_id, session_id=session.id, tenant_id=session.tenant_id)
            # materialize 不返回 template（它只读 record，不碰 TemplateLookup）——
            # state.extra 仍需要它（CompactStep 经 extra["template"] 读），单独取一次。
            #
            # 按 `agent.template_id` 解，不是会话那份（2026-09-30）：压的是这一个 agent 的
            # 上下文，折叠用的 identity 必须是它自己的。同时也让上面那道
            # 「session has no template_id — cannot compact」的守卫失去依据——压缩一个
            # 子 agent 从来不需要会话的模板，那道门删掉了。
            template = await self._template_lookup.get_template(
                agent.template_id, None, ctx=pctx,
            )
            # Manual compaction explicitly bypasses the budget trigger; usage stays factual.
            agent = _dc.replace(
                agent,
                # 跨重启 registry 为空时 materialize 的回落只给得出默认
                # memory_config/loop_config；用刚取到的真实 template 覆盖，
                # 与旧 instantiate_agent(existing_agent_id=...) 每次按 template 现算一致。
                memory_config=template.memory_config,
                loop_config=template.loop_config,
                runtime={"llm_model": rm.model},
            )

            task = Task(
                id=task_id or generate_id("tsk"),
                session_id=session.id,
                status="ACTIVE",
                tenant_id=session.tenant_id,
                assigned_agent_id=agent.id,
                creator_agent_id=agent.id,
                settings=NormalTaskSettings(),
                user_prompt_in_memory=True,  # nothing to ingest
                created_at=now_utc(),
            )

            memory = self.providers.get_memory()
            provider_ctx = self._build_provider_ctx(session, task, agent)
            assembler = self._build_assembler(memory, provider_ctx)
            gateway = self._build_gateway(memory, provider_ctx)
            llm = rm.client
            loop_ctx = self._build_loop_ctx(
                assembler, llm, memory, provider_ctx, gateway, token, None,
            )

            scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=agent.id)
            run_id = generate_id("run")
            state = LoopState(
                run_id=run_id,
                session=session,
                task=task,
                agent=agent,
                scope=scope,
                extra={"template": template},
                resolved_model=rm,
            )
            # 总账 C5：这段独立跑了一个 run_id 却从不发起止——host 会看到凭空出现又
            # 凭空消失的 run。补齐起止事件（payload 结构照抄 `_run_loop` 的实际发射点，
            # 见 task-5-report）；这条路径没有 StepDriver，也没有 RunOutcome，
            # `initial_step` 用它实际跑的那个 step 名 "compact"，`outcome` 按「跑完即
            # 完成 / 崩溃即 interrupted」的既有口径取值。
            await self._event_bus.emit(make_event(state, EventType.RUN_STARTED, payload={
                "run_id": run_id,
                "initial_step": "compact",
            }, origin=EventOrigin.RUNTIME))
            run_error: Exception | None = None
            was_cancelled = False
            try:
                state, outcome = await loop_ctx.steps._execute_foreground(
                    "compact", state, loop_ctx, options={
                        "trigger": "manual", "token_estimate": agent.loop_guard.context_tokens,
                        "force": True, "after": None,
                    })
            except asyncio.CancelledError:
                # F2：CancelledError 不是 Exception 子类，下面的 except Exception 接不住
                # ——接不住则 run_error 仍是 None，RunFinished 会谎报 completed。记下来，
                # 供 finally 里的 RunFinished.outcome 用（在异常继续传播之前发出，见下）。
                #
                # 这里**要重新抛出**——与本文件下方 `_run_loop` 的 except asyncio.CancelledError
                # 刻意不同，不是疏漏：`_run_loop` 吞是因为它把取消结果转成了 RunOutcome{kind=
                # CANCELED} 这个**返回值契约**塞回调用方（run 层报结局靠返回值）。
                # `compact_session` 没有这种契约——它返回的是 `{"session_id", "agent_id",
                # "task_id"}` 一个 id 字典，吞掉 CancelledError 会让调用方
                # （`task.cancel()` / `asyncio.wait_for(...)`）拿到一个看似正常的返回值、
                # 取消信息凭空消失——比事件层面的谎报更糟的语义谎报，必须重新抛出。
                was_cancelled = True
                raise
            except Exception as exc:
                run_error = exc
                raise
            finally:
                await self._event_bus.emit(make_event(state, EventType.RUN_FINISHED, payload={
                    "outcome": (
                        RunOutcomeKind.CANCELED.value if was_cancelled
                        else RunOutcomeKind.COMPLETED.value if run_error is None
                        else RunOutcomeKind.INTERRUPTED.value
                    ),
                    "final_status": task.status,
                    "will_retry": False,
                    "total_events": state.sequence_counter,
                    "total_turns": len(state.transcript),
                    "error": str(run_error) if run_error else None,
                    "error_type": type(run_error).__name__ if run_error else None,
                }, origin=EventOrigin.RUNTIME))

            return CompactReceipt(
                session_id=session.id, agent_id=agent.id, task_id=task.id,
                task_id_is_transient=transient,
            )
        finally:
            self._busy_sessions.discard(session_id)

    # ── Read-only host queries ──────────────────────────────────────────────

    def _query_result(self, data, source, loaded, position=None):
        return QueryResult(deepcopy(data), source, loaded, now_utc(), position)

    async def _query_view(self, session_id):
        head = await self.event_store.committed_head(session_id)
        view = await rebuild_view(self.event_store, session_id, through_position=head)
        if session_id not in view.sessions and session_id not in self._task_managers:
            raise SessionNotFound(session_id)
        return view, head

    async def _historical_task_view(self, session_id, *, through_position=None):
        # TaskReviewing can create a successor under the predecessor's task_id.
        # Replay task facts across the partition, independently of pruned snapshots.
        from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT
        head = (await self.event_store.committed_head(session_id)
                if through_position is None else through_position)
        types = tuple({EventType.TASK_CREATED, EventType.TASK_FINALIZED,
                       EventType.RECOGNIZE_INTENT_TOOL_CALL, EventType.OBSERVE_REQUESTED,
                       EventType.HITL_OPENED, *TASK_STATUS_BY_EVENT})
        rows = await self.event_store.read_range(
            session_id, through_position=head, include_types=types)
        return reduce_events([row.event for row in rows], session_id)

    async def inspect_prompt(
        self, session_id: str, event_id: str, *, trim: bool = True,
    ) -> QueryResult[PromptDiagnostic]:
        """Recompile redacted saved inputs without loading execution owners or invoking a model.

        The returned prompt is diagnostic, not a guarantee of historical wire bytes.
        Provider/template/memory reads use the session's persisted tenant identity.
        """
        from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
        from ctx_weft.core.prompt_diagnostics import inspect_prompt

        diagnostic, position = await inspect_prompt(
            self.event_store, session_id, event_id,
            memory=self.providers.get_memory(),
            template_lookup=TemplateLookup(self.providers),
            capability_provider_index={p.name: p for p in self.providers.get_capability_providers()},
            trim=trim,
        )
        return self._query_result(diagnostic, "events", self._session_is_loaded(session_id), position)

    async def get_session(self, session_id: str) -> QueryResult[SessionDetail]:
        async with self._session_guard(session_id):
            tm = self._task_managers.get(session_id)
            loaded = tm is not None and tm.session is not None
            if loaded:
                session = self._task_managers[session_id].session
                position = None
            else:
                view, position = await self._query_view(session_id)
                session = view.sessions[session_id]
            from ctx_weft.core.control.session_queries import session_state_summary
            if loaded:
                from types import SimpleNamespace
                lm = self._agent_lifecycle_manager
                state_view = SimpleNamespace(
                    agents={aid: lm.record_of(aid) for aid in lm.agent_ids_of_session(session_id)},
                    tasks={t.id: t for t in tm.all_tasks()},
                    hitl=SimpleNamespace(pending={r.id: r for r in self.hitl_registry.list_pending(session_id=session_id)}),
                )
                head = await self.event_store.committed_head(session_id)
            else:
                state_view, head = view, position
            state = await session_state_summary(
                self.event_store, state_view, session, head,
                threshold_tripped=tm._threshold_tripped if loaded else None,
            )
            detail = SessionDetail(
                session.id, session.root_agent_id or "",
                session.created_at, state,
            )
            return self._query_result(detail, "memory" if loaded else "events", loaded, position)

    async def get_task(self, session_id: str, task_id: str) -> QueryResult[TaskDetail]:
        async with self._session_guard(session_id):
            tm = self._task_managers.get(session_id)
            task = tm.get_task(task_id) if tm is not None else None
            if task is not None and task.details_complete:
                return self._query_result(task_detail(task), "memory", True)
            head = await self.event_store.committed_head(session_id)
            task = await read_task_view(self.event_store, session_id, task_id, through_position=head)
            if task is None:
                created = await self.event_store.read_last_of_type(session_id, EventType.SESSION_CREATED)
                if tm is None and (created is None or created.position > head):
                    raise SessionNotFound(session_id)
                raise TaskNotFound(task_id)
            return self._query_result(task_detail(task), "events", False, head)

    async def list_tasks(
        self, session_id: str, *, agent_id: str | None = None,
        statuses: list[str] | None = None, limit: int = 100,
    ) -> list[QueryResult[TaskDetail]]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        async with self._session_guard(session_id):
            _, head = await self._query_view(session_id)
            view = await self._historical_task_view(session_id, through_position=head)
            results = {tid: self._query_result(task_detail(t), "events", False, head)
                       for tid, t in view.tasks.items()}
            if session_id in self._task_managers:
                for task in self._task_managers[session_id].all_tasks():
                    if task.details_complete:
                        results[task.id] = self._query_result(task_detail(task), "memory", True)
            items = [r for r in results.values()
                     if (agent_id is None or r.data.assigned_agent_id == agent_id)
                     and (statuses is None or r.data.status in statuses)]
            items.sort(key=lambda r: (r.data.created_at.isoformat() if r.data.created_at else "",
                                      r.data.task_id), reverse=True)
            return items[:limit]

    async def get_agent(
        self, agent_id: str, *, session_id: str | None = None,
    ) -> QueryResult[AgentDetail]:
        rec = self._agent_lifecycle_manager.record_of(agent_id)
        if session_id is None:
            if rec is None:
                raise AgentNotLoaded("session_id is required for a cold agent query")
            session_id = rec.session_id
        async with self._session_guard(session_id):
            rec = self._agent_lifecycle_manager.record_of(agent_id)
            if rec is not None and rec.session_id != session_id:
                raise ValueError("agent_id does not belong to session_id")
            loaded = rec is not None and session_id in self._task_managers
            if loaded:
                agent = rec
                head = await self.event_store.committed_head(session_id)
                task = self._task_managers[session_id].get_task(agent.current_task_id)
            else:
                view, head = await self._query_view(session_id)
                agent = view.agents.get(agent_id)
                if agent is None:
                    raise AgentNotFound(agent_id)
                task = view.tasks.get(agent.current_task_id)
            current = await public_task_detail(self.event_store, session_id, agent.current_task_id, task, head)
            detail = agent_detail(agent_id, agent, session_id, current)
            return self._query_result(detail, "memory" if loaded else "events", loaded, None if loaded else head)

    async def list_agents(
        self, *, session_id: str | None = None, parent_agent_id: str | None = None,
        include_terminated: bool = False,
    ) -> list[QueryResult[AgentSummary]]:
        if session_id is None:
            # No implicit cross-partition scan; the unscoped API lists loaded members.
            return [self._query_result(a, "memory", True) for a in self._list_agents_memory(
                parent_agent_id=parent_agent_id, include_terminated=include_terminated)
                if self._session_is_loaded(
                    self._agent_lifecycle_manager.record_of(a.agent_id).session_id)]
        if self._session_is_loaded(session_id):
            return [self._query_result(a, "memory", True) for a in self._list_agents_memory(
                session_id=session_id, parent_agent_id=parent_agent_id,
                include_terminated=include_terminated)]
        view, head = await self._query_view(session_id)
        results = {}
        for aid, agent in view.agents.items():
            results[aid] = self._query_result(AgentSummary(
                agent_id=aid, parent_agent_id=agent.parent_agent_id, status=agent.status,
                current_task_id=agent.current_task_id, spawn_depth=agent.spawn_depth,
            ), "events", False, head)
        if session_id in self._task_managers:
            for agent in self._list_agents_memory(session_id=session_id, include_terminated=True):
                results[agent.agent_id] = self._query_result(agent, "memory", True)
        return [r for r in results.values()
                if (parent_agent_id is None or r.data.parent_agent_id == parent_agent_id)
                and (include_terminated or r.data.status != "terminated")]

    async def list_pending_hitl(
        self, *, session_id: str | None = None, agent_id: str | None = None,
    ):
        if session_id is None and agent_id is not None:
            rec = self._agent_lifecycle_manager.record_of(agent_id)
            if rec is None:
                raise AgentNotLoaded("session_id is required for a cold HITL query")
            session_id = rec.session_id
        if session_id is None or self._session_is_loaded(session_id):
            return [self._query_result(r, "memory", True) for r in self._list_pending_hitl_memory(
                session_id=session_id, agent_id=agent_id)]
        view, head = await self._query_view(session_id)
        return [self._query_result(r.to_view(), "events", False, head)
                for r in view.hitl.pending.values() if agent_id is None or r.agent_id == agent_id]

    def _list_pending_hitl_memory(
        self, *, session_id: str | None = None, agent_id: str | None = None,
    ) -> "list[HitlRequestView]":
        """未决 HITL 的**只读视图**列表。两个过滤都可选、可叠加，都不传 = 全部。

        host 面向 HITL 的读入口。刻意不暴露 `HitlRegistry`：`PendingHitl` 是 core 的活
        记录（带等待槽、stage、invocation_key 这些内部键），它自己的 docstring 就写着
        「不出 core」。经由本方法拿到的 `HitlRequestView` 才是契约层类型。

        `agent_id` 是 agent-centric 下的主用过滤轴（2026-09-04 spec §5.2）：HITL 自
        09-03 起已彻底 agent 化，只有这个查询入口此前停在 session 维度。

        **只读内存**：注意重启之后 registry 要先被装填（`rebuild_session()`）
        才有内容——「恢复是喂进来、不是查回去」（spec §3.1）。
        """
        return [
            r.to_view()
            for r in self.hitl_registry.list_pending(session_id=session_id, agent_id=agent_id)
        ]

    def _list_agents_memory(
        self,
        *,
        session_id: str | None = None,
        parent_agent_id: str | None = None,
        include_terminated: bool = False,
    ) -> "list[AgentSummary]":
        """列出 agent（spec 5；2026-09-04 spec §8 放宽 session_id）——host 的发现入口。

        三个过滤都可选、可叠加。`session_id` 不传 = 跨 session 列出全部登记 agent
        （`agent_id` 全局唯一，按 session 分片只是历史惯性）；传了则只列该 session。
        `parent_agent_id` 只返回其**直接**子 agent（不展开子孙——层级关系不在接口层
        嵌套，调用方按 `parent_agent_id` 自行还原成树）。`include_terminated` 默认
        False，避免列表随时间无限膨胀。

        数据源用 `AgentLifecycleManager` 自己的记录（经 `record_of`）：以 ALM 的内存
        现实为准，不会把已经不在内存里的 agent 报告出去。

        ⚠ **内存现实 ≠ 全部事实**：2026-09-08 起 ALM 是只增不删的缓存，但持有方可以
        显式 `forget_session` 把一条会话逐出。逐出之后这里返回空列表——那不表示这条
        会话没有 agent，只表示它此刻不在内存里。要「不管在不在内存里都列出来」，先
        `rebuild_session(session_id)` 装填再列。
        """
        reg = self._agent_lifecycle_manager
        ids = (reg.agent_ids_of_session(session_id) if session_id is not None
               else reg.all_agent_ids())
        out: list[AgentSummary] = []
        for aid in ids:
            rec = reg.record_of(aid)
            if rec is None:
                continue
            if parent_agent_id is not None and rec.parent_agent_id != parent_agent_id:
                continue
            if not include_terminated and rec.status == "terminated":
                continue
            out.append(AgentSummary(
                agent_id=aid,
                parent_agent_id=rec.parent_agent_id,
                status=rec.status,
                current_task_id=rec.current_task_id,
                spawn_depth=rec.spawn_depth,
                created_at=rec.created_at,
            ))
        return out

    @_session_command(load_session="required", missing_session_as_agent=True)
    @runtime_command
    async def send_message(
        self,
        agent_id: str,
        content: "str | list[ContentPart]",
        *,
        session_id: str | None = None,
        unattended: bool = False,
        port_key: str | None = None,
    ) -> TurnHandle:
        """向指定 agent 发一条外部消息，返回这次交互的 `TurnHandle`（spec §4.1；
        2026-09-04 spec §3.3）——agent-centric 的核心入口：外部消息按 agent 显式寻址，
        不再隐式挂「当前唯一活跃 task」。

        守卫：不存在 / `terminated` / `running` 一律抛错，**不排队**
        （`AgentLifecycleManager.assert_can_receive`）；调用方自行重试，或先
        pause/cancel。但「不存在」的判据是**事件日志里也没有**，不是「内存 registry 里
        没有」——registry 的一次 miss 只说明还没喂进来（见 `_hydrate_agent_for_send`），
        故守卫排在自愈之后。

        `session_id` 不参与路由（`agent_id` 全局唯一，路由永远只看 `current_task_id`），
        但有两个用途：提前发现「这个 agent 不属于该 session」这类误用；以及给上面那次
        自愈当定址索引。

        **后一个用途自 2026-09-21 起是硬要求**：registry miss 且没给 `session_id` →
        抛 `AgentNotLoaded`（`AgentNotFound` 的子类）。从前那条「没 session 语境就扫全部
        active session」的 sweep 已删，理由见 `rebuild_all_pending_hitl` 之后的墓碑注释。
        热路径不受影响——registry 命中时本参数仍然只用于防呆，路由永远只看 `agent_id`
        （spec §4.2：`agent_id` 全局唯一，足以路由）。

        路由三条路径（spec §4.2）：

        - `current_task_id` 已终态或为空 —— 新建 task 挂给该 agent
          （`_start_task_for_agent`，走既有 `push_task` 通路）。
        - 未终态、且不是「SUSPENDED 等子任务」—— 注入并重排该活 task。
        - 未终态、且 `_suspended_on_live_children` —— 只把消息写进对话，
          `_try_resume_parent` 在子任务收尾时自然唤醒它。

        三条都返回句柄，`task_id` 恒非空。调用方要区分「开了新一轮」还是「并进旧的」，
        比对返回的 `task_id` 与调用前 `get_agent(agent_id).data.state.current_task.task_id` 即可——
        句柄里不放这个布尔，也不放 run_id（第三条路径在返回那一刻还没有新一轮，
        见 2026-09-04 spec §3.2）。

        ``unattended``：这条消息**开出的新 task** 无人看顾（后台自治作业）——只对上面
        第一条路径（新建 task）生效，注入既有 task 的两条路径沿用那个 task 自己的标记
        （改写一个已在跑的 task 的「有没有人在」不属于本入口的职责）。见 `Task.unattended`。

        ``port_key``：同上，只对新建分支生效——这条消息开出的新 task 接在哪个交互口上
        （见 `Task.port_key`）。注入分支沿用那个 task 自己的口：一条消息不该把一个正在
        跟某个对端往返的 task 改接到别处。
        """
        reg = self._agent_lifecycle_manager
        rec = reg.record_of(agent_id)
        if rec is None:
            await self._hydrate_agent_for_send(agent_id, session_id)
            rec = reg.record_of(agent_id)
            if rec is None:
                raise AgentNotFound(f"unknown agent: {agent_id}")
        if session_id is not None and session_id != rec.session_id:
            raise ValueError(
                f"agent {agent_id} belongs to session {rec.session_id!r}, not {session_id!r}"
            )
        # 守卫在装填**之后**：装填前 registry 的 miss 只说明「还没喂进来」，不说明
        # 这个 agent 不能收消息。放在前面会让上面那次自愈永远执行不到（`status_of`
        # 对未登记的 id 抛 `AgentNotFound`，而它正是本方法要修的那个误判）。
        tm = self._task_managers.get(rec.session_id)
        if tm is None or not tm.is_alive():
            await self.rebuild_session(rec.session_id, tenant_id=rec.tenant_id, strict=True)
            rec = reg.record_of(agent_id)
        reg.assert_can_receive(agent_id)

        current = rec.current_task_id
        if current and not self._task_is_terminal(rec.session_id, current):
            task_id = await self._inject_user_turn(
                current, content, session_id=rec.session_id)
        else:
            task_id = await self._start_task_for_agent(
                agent_id, content, unattended=unattended, port_key=port_key)

        # Input is installed before cold tasks or background model work can run.
        tm = self._task_managers.get(rec.session_id)
        if tm is not None and tm.is_alive():
            lock = self._session_guard(rec.session_id)
            async with lock:
                await self._activate_prepared_tasks_locked(tm)
                self._spawn(tm.drain())

        return TurnHandle(
            session_id=rec.session_id,
            agent_id=agent_id,
            task_id=task_id,
            template_id=rec.template_id,
            event_bus=self._event_bus,
            _storage_health=self.storage_health,
            _storage_recovery=self._wait_for_storage,
            _pending_observe=self._pending_observe_run_id,
            _is_finished=self._task_is_settled,
        )

    async def _hydrate_agent_for_send(self, agent_id: str, session_id: str | None) -> None:
        """`send_message` 的 registry-miss 自愈：把这个 agent 装填回 ALM。

        **为什么 miss 是常态而不是错误**：2026-09-08 起 ALM 是**只增不删的缓存**，
        回收由持有方显式发起（`forget_session` / `forget_agent`）——host 会按自己的
        策略把久未使用的会话逐出内存。此外进程刚起来时 registry 整个是空的（恢复已改成
        用户驱动的按需装填，启动不再预热）。三种情形都不是"这个 agent
        不存在"，只是"还没喂进来"。事实一直在事件日志里。

        （2026-09-08 之前 miss 还有第四个、也是最常见的来源：会话一跑完
        `_fire_session_done` → `_release_session` 就把该 session 全部 record 摘掉，
        于是「正常结束的会话再发一条消息」必然撞 `AgentNotFound`。那条已经不再发生，
        但本方法仍是必要的——上面三种来源都还在。）

        与 `_hydrate_agent_for_cold_resume` 同一形状、同一理由（那边是冷 HITL 应答，
        这边是终态后续聊）：手握 `session_id` 就精确装填这一个 session，一次定址、
        成本有界。

        **没有 `session_id` 就抛 `AgentNotLoaded`，不去找**（2026-09-21）。从前这条路
        落到 `rebuild_agent` 的全量 sweep——扫遍全部 active session 直到撞见那个 agent。
        删它的理由见下面那段墓碑注释（`rebuild_all_pending_hitl` 之后）：装填是调用方的
        责任，而 core 替它猜要付 O(会话数)，换的是一个调用方本来就知道的值。
        `AgentNotLoaded` 是 `AgentNotFound` 的子类，既有的 `except AgentNotFound` 照样接住。

        **走 `rebuild_session` 而不是只调 `_load_agents_of`**：后者只喂 agent record，
        而 `send_message` 的注入分支会 `_cancel_pending_hitl_of`（`waiting_human` 的
        agent 收到外部消息 ⟹ 旧提问不会再有人答了）——那一步读的是 `hitl_registry`
        的内存。registry 冷着的话它一条都找不到，旧提问就成了「有 HitlOpened、无终局
        事件」的孤儿，重启后 `rebuild_session` 会把它当未决恢复出来，还会被 `resume_agent`
        的 `_pause_bubble_of` 误当成暂停气泡放行一次冷续跑。装填要装齐。

        **统一装填 ALM、HITL 和 TaskManager**：重建保留任务的持久化状态，不入队、
        不执行。`send_message` 必须先完成准备，再判断旧任务是否终态，最后交付输入
        并激活任务，不能把缺 owner 当作旧任务已经结束。

        严格装填失败直接抛出；不能把准备失败当作 agent 或旧任务不存在。
        """
        if session_id is None:
            # HITL 那半在这条路上本来也装不了（`rebuild_session` 按 session 定址），所以
            # 从前那条 sweep 连自己的目标都只完成一半。
            raise AgentNotLoaded(
                f"agent {agent_id!r} is not loaded and no session_id was given — "
                f"call rebuild_session(session_id) first, or pass session_id to send_message"
            )
        await self.rebuild_session(session_id, strict=True)

    def _task_is_terminal(self, session_id: str, task_id: str) -> bool:
        """`current_task_id` 是否已终态——`send_message` 路由的唯一判据。

        必须先装填 TM；缺 owner 是准备失败，不能当作旧任务已结束。
        已装填 owner 中缺 task 可由终态快照裁剪产生，沿用终态判据。
        """
        tm = self._task_managers.get(session_id)
        if tm is None:
            raise RuntimeError(f"Session {session_id!r} is not prepared for message routing")
        task = tm.get_task(task_id)
        if task is None:
            return True
        return task.status in TERMINAL_TASK_STATUSES or task.status == "REVIEWING"

    async def _inject_user_turn(
        self, task_id: str, content: "str | list[ContentPart]", *, session_id: str,
    ) -> str:
        """`send_message` 的注入分支：`current_task` 未终态 -> 把新消息当一轮用户
        发言写进该 task 的对话——复用 HITL 回复已经在用的落盘通路
        （`_ingest_user_turn`，从 `_write_hitl_reply_turn` 抽出，两边共用同一次
        memory ingest），不另起一套。

        与 HITL 回复的分野只在"content 怎么来"：那边要从 `PendingHitl.decision`
        派生（拒绝措辞 / 打断续接前缀），这里 content 就是调用方给的原样消息，未经
        任何 HITL 专属加工。

        是否要把 task 重新排进队列执行：
        - `_suspended_on_live_children`（spec §4.2：agent 因 `delegate_task` 处于
          `idle`，当前 task `SUSPENDED` 且仍有未终态子任务）—— 只写记忆、不碰状态：
          `_try_resume_parent` 会在子任务收尾时按 `status == "SUSPENDED"` 这道门
          自然唤醒它，那时它进 act 就看得见这里写下的这一轮（与 `_inject_user_reply`
          完全同一判据、同一处理）。
        - 其余非终态（`PENDING` / `AWAITING_HUMAN` / `INTERRUPTED` / 无子任务的
          `SUSPENDED`）—— `TaskManager.requeue_for_message` 重排：已排队 / 已在跑 /
          已终态它自身 no-op；真正被挡住的会置 `PENDING` 入队并发 `TaskRequeued`
          （**不是** `TaskHumanResolved`——那是 `TaskAwaitingHuman{hitl_id}` 的一对一
          配对解除事件，只属于 HITL 应答路径，发生在这里会留一个配不上对的孤儿事件，
          还会经 ALM 把 agent 状态提前翻成 `running`——task 明明还没真正开跑，见
          `requeue_for_message` 自己的 docstring）。调用方在输入交付完成后启动 `drain()`——
          `requeue_for_message` 只管入队、不 drain，与 `resume_task` 同一分工。

        非 `_suspended_on_live_children` 分支还会在 `requeue_for_message` 之前先
        终局该 agent 名下全部未决 HITL（`_cancel_pending_hitl_of`，`cancel_agent`
        专用方法，这里直接复用）——`waiting_human` 的 agent 收到外部消息本就意味着
        旧提问不会再有人去应答了，具体理由见下方内联注释。
        """
        tm = self._task_managers.get(session_id)
        if tm is None or tm.session is None:
            raise ValueError(f"no active TaskManager for session {session_id!r}")
        target = tm.get_task(task_id)
        if target is None:
            raise ValueError(f"task {task_id!r} not found in session {session_id!r}")
        session = tm.session
        # agent_id 必须是本 task 对话真正所在的 agent scope，与 `_write_hitl_reply_turn`
        # 走 `_reply_turn_agent_id` 同一回退口径（assigned 优先、creator 兜底）——这里
        # 没有 `PendingHitl.agent_id` 可用（不是 HITL 应答），直接从 task 上取。
        agent_id = target.assigned_agent_id or target.creator_agent_id or ""
        scope = MemoryAddress(session_id=session.id, task_id=target.id, agent_id=agent_id)
        pctx = ProviderContext(
            session_id=session.id, tenant_id=session.tenant_id,
            task_id=target.id, agent_id=agent_id,
        )
        bubble = self._pause_bubble_of(agent_id, session_id=session_id)
        if bubble is not None and bubble.delivery.preface == PREFACE_AFTER_INTERRUPT_EDIT:
            from ctx_weft.core.loop.steps.act import _interrupt_edit_prefix
            from ctx_weft.protocols.content import content_with_prefix
            content = content_with_prefix(content,
                _interrupt_edit_prefix(await self._last_user_prompt(scope, pctx)))
        normalized, _event_jsonable = await self._validate_and_normalize_content(
            content, session.id, tenant_id=session.tenant_id,
        )
        mem_id = generate_id("mem")
        msg_ts = now_utc()
        if _suspended_on_live_children(tm, target):
            # 没有「这一轮」（不开窗，见下方开窗处注释）→ `_ingest_user_turn` 直接落盘。
            await self._ingest_user_turn(
                scope, pctx, normalized, event_id=mem_id,
                task_id=target.id, source="send_message", timestamp=msg_ts,
            )
            await self._emit_message_appended(
                session, target.id, agent_id, mem_id, _event_jsonable, msg_ts)
            logger.info(
                "_inject_user_turn: task %s is SUSPENDED on live children — message "
                "written, state left alone so _try_resume_parent still wakes it",
                target.id)
            return target.id
        # 收口该 agent 名下未决 HITL（若有，典型是 `waiting_human` 的 agent 被
        # send_message 打断——旧提问再没人会去回答了）。顺序纪律与 `cancel_agent`/
        # `_cancel_session_hitl` 一致：必须先于下面 `requeue_for_message` 可能触发的
        # 状态推进（TaskRequeued -> ALM 让 agent 离开 waiting_human）；不然重启后
        # `rebuild_session` 会把这条「有 HitlOpened 无终局事件」的陈旧提问当未决恢复
        # 出来，且它会一直挂在 `list_pending_hitl` 里，被 `resume_agent` 的
        # `_pause_bubble_of` 误当作暂停气泡命中、放行一次冷续跑（那正是 R24 那条
        # 止损点要防的场景：真问题悬而未决时替用户放行）。
        #
        # message 复用 `CancelReason.USER_CANCEL`（`_cancel_pending_hitl_of` 内部
        # 硬编码这个值，直接复用不新写一套）。语义上不算精确——这里不是用户显式取消
        # 了那个提问，而是用户发了条新消息、使旧提问失去意义——但 `CancelReason` 目前
        # 只有 USER_CANCEL / FAILURE_THRESHOLD / PAUSE_ABANDON 三个值，后两个分别专属
        # 熔断跳闸与暂停弃子链路，语义上更不贴切；USER_CANCEL 是三者里最接近的近似值
        # （旧提问的作废终究是由用户的动作触发的），故不为此新增枚举值。
        # 开窗要先于收口与重排：两者都改内存态，而这一轮在 LLM 开口之前随时可能被撤销。
        # **只在这里开**，不在函数入口——上面那条 `_suspended_on_live_children` 早退分支
        # 没有为这条消息新开的 run（消息只是写进对话，等 `_try_resume_parent` 自然唤醒），
        # 没有「这一轮」可言，也就没有提交点会来关窗（spec 2026-09-09 §6）。
        stale_hitl = next(
            (v.id for v in self._list_pending_hitl_memory(session_id=session_id)
             if v.agent_id == agent_id and not v.resolved),
            "",
        )
        # 记下窗是不是本次开的：下面「重排没排上 → 就地提交」只该提交自己开的窗。
        opened_here = not tm.is_round_open(target.id)
        tm.begin_round(target.id, owns_task=False, hitl_id=stale_hitl)
        # 窗口开着 → 这条消息暂存进窗口（`_ingest_user_turn` 经 `ingest_or_stage` 分流），
        # 这一轮算数时才落盘，撤销时随窗口扔掉。
        await self._ingest_user_turn(
            scope, pctx, normalized, event_id=mem_id,
            task_id=target.id, source="send_message", timestamp=msg_ts,
        )
        # 消息正文的事件侧一份（带 task_id → 在窗口里缓冲、随这一轮提交**先于** memory 落盘）。
        await self._emit_message_appended(
            session, target.id, agent_id, mem_id, _event_jsonable, msg_ts)
        # `defer=True`：旧气泡的收口跟着这一轮走——撤销时它要回到 pending。
        #
        # **只有同一个 task 的气泡才跟着这一轮**：提交与撤销两个钩子都按 task 查待终局
        # 请求，而这里是按 **agent** 找旧气泡。该 agent 名下别的 task 上的气泡若也以
        # `defer` 收下，就没有任何一轮会去终局或退回它——永远卡在待终局，且从待答列表里
        # 消失。那些立即收口。
        await self._cancel_pending_hitl_of(
            agent_id, session_id=session_id, defer=True, defer_only_task_id=target.id)
        # 清旧进展，同 `_inject_user_reply`：新消息意味着有新工作要做，陈旧的
        # outputs/process_report 留着会让 success-guardrail 误判"已经产出过"。
        target.outputs = None
        target.process_report = None
        target.process_report_at = None
        requeued = await tm.requeue_for_message(task_id)
        if not requeued:
            # 重排是 no-op（task 已在队列 / 已在跑 / 已终态）→ **没有为这条消息新开的
            # run**，也就没有提交点会来关窗。窗口留着就是永久缓冲：这条消息以及它收口
            # 掉的气泡再也不会落盘。就地提交，让它与改造前同样立刻算数。
            #
            # 这是 spec 2026-09-09 §6 说的那个例外：消息被并进一个已经在跑/将跑的
            # task，没有「这一轮」可言，撤销也就无从谈起。
            #
            # **窗若不是本次开的就不提交**：该 task 已有一轮开着（热应答醒来还没等到 LLM
            # 开口、冷续跑还在 reconcile、新 task 首 chunk 前）。那一轮有自己的提交点与撤销
            # 路径，这条消息并进去、跟着它一起算数或一起撤回；在这里提交就替它提前关了窗——
            # 它的答复在 LLM 开口前被终局、暂存提前落盘，撤销从此无从谈起。
            if opened_here:
                await tm.commit_round(task_id)      # 钩子会一并终局被收口的旧气泡
        return target.id

    async def _start_task_for_agent(
        self, agent_id: str, content: "str | list[ContentPart]", *,
        unattended: bool = False, port_key: str | None = None, **_kw: Any,
    ) -> str:
        """`send_message` 的新建分支：`current_task` 已终态（或压根没有）-> 起一个
        新 task 挂给该 agent，走既有的 `push_task` 通路——与
        `SessionRegistry._make_root_task_manager` 起 root task 同一套写法，不新造
        一条派发路径。

        复用同一个正在跑的 TaskManager（`self._task_managers[session_id]`）：会话
        建立时 `_register_and_drain` 已经给它 `set_runner` / `set_is_current` /
        挂好 done/idle 回调，这里只 push 新 task；调用方在输入就绪后激活并 drain。

        正常入口已在判断旧任务前通过 `rebuild_session` 完成统一准备。这里仍保留
        缺 owner 的防御性兼容路径，处理只装填过 ALM 的旧入口：
        `rebuild_session(session_id, strict=True)` 只准备 owner，不提前执行或宣告会话结束。
        已有活 owner 时直接复用，不再次重建。
        """
        reg = self._agent_lifecycle_manager
        rec = reg.record_of(agent_id)
        if rec is None:
            # 从前这里靠 `send_message` 的 `assert_can_receive` 兜底（未登记必先抛
            # `AgentNotFound`），所以敢直接 `rec.session_id`。守卫现在排在自愈之后、
            # 且自愈可能在这两步之间被并发的 `forget_session` 冲掉，那条隐含保证
            # 没了——不补这一句就是 `AttributeError: 'NoneType' has no 'session_id'`，
            # 一个比原错误更难查的形状。
            raise AgentNotFound(
                f"agent {agent_id!r} disappeared before its task could be started — "
                f"the session may have been released concurrently; retry send_message"
            )
        tm = self._task_managers.get(rec.session_id)
        if tm is None or not tm.is_alive():
            await self.rebuild_session(rec.session_id, strict=True)
            rec = reg.record_of(agent_id)
            if rec is None:
                # 上面那次 `recover_session` 返回了，记录却不在——只可能是并发的
                # `forget_session` / `forget_agent` 在这两步之间把它摘掉了（`recover_session`
                # 自己对 miss 是直接抛 `AgentNotLoaded` 的，走不到这里；"session 在事件
                # 日志里压根不存在"那条 RuntimeError 同样从它里面就抛出去了）。防御性地
                # 给一个可诊断的类型化错误，而不是让下面的 `rec.session_id` 撞
                # AttributeError。
                raise AgentNotFound(
                    f"agent {agent_id!r} disappeared during cold recovery — "
                    f"the session may have been released concurrently; retry send_message"
                )
            tm = self._task_managers.get(rec.session_id)
            if tm is None:
                from ctx_weft.core.models.errors import SessionNotFound
                raise SessionNotFound(
                    f"agent {agent_id!r} (session {rec.session_id!r}) has no live "
                    f"TaskManager even after rebuild_session(strict=True) — this is "
                    f"a bug in the recovery path, not a transient condition; do not retry "
                    f"blindly, file it"
                )
        # 用户开口 ⟹ 该 agent 名下的旧未决 HITL 不会再有人答了——收口它们。
        #
        # 注入分支（`_inject_user_turn`）一直在做这件事，新建分支从前漏了。以前漏得起：
        # 走到这里说明 `current_task` 已终态，而终态 task 的气泡从前总是在终结时就被
        # 一并收掉。**纯文本 park 之后不再如此**（2026-09-24）：后台 observe 判 success
        # 会把 task 终结，而那个 `wait_for_user` 气泡要**留着**——它是「会话在等你说话」
        # 这个事实的载体，宿主按未决 HITL 折会话状态（PAUSED），判决替用户把它收掉就等于
        # 替用户宣布「不用说了」，会话会当场跳成已完成。于是收口的时机从「判决落定」挪到
        # 了这里：**用户真的开口，那个入口才算被用掉**。
        #
        # `defer=False`（与注入分支相反）：走到这条路的气泡都挂在一个已经终态的 task 上，
        # 没有「回到 pending 等重来」的意义——新一轮若被撤销，它也不该复活。
        await self._cancel_pending_hitl_of(agent_id, session_id=rec.session_id)
        normalized, event_jsonable = await self._validate_and_normalize_content(
            content, rec.session_id, tenant_id=rec.tenant_id,
        )
        from ctx_weft.protocols.content import content_to_text
        task = Task(
            id=generate_id("tsk"),
            session_id=rec.session_id,
            status="ACTIVE",
            tenant_id=rec.tenant_id,
            assigned_agent_id=agent_id,
            creator_agent_id=agent_id,
            # Leave the title empty so this new top-level task gets its own
            # recognize_intent pass after the round commits, like the first task.
            description=content_to_text(normalized)[:200],
            user_prompt=normalized,
            # 外部消息 = 用户对话，与 root task 同一口径：纯文本回合的归宿由
            # `unattended` 单独决定（见 `Task.unattended`）。既然是后台投喂的一条消息、
            # 没有人守着，就不会有下一条消息来解 park，那时纯文本让位等于永久挂起。
            unattended=unattended,
            # 交互口：调用方显式给的那个；未声明（None）则按 `unattended` 回落。
            port_key=(port_key if port_key is not None else default_port_for(unattended)),
            created_at=now_utc(),
        )
        # `provisional=True`：一条用户消息开出的新一轮，在 LLM 真的开口之前不算发生
        # （spec 2026-09-09）。这个 task 的创建/启动事件先只到达进程内状态机，act 收到
        # 首个 chunk 才提交、用户在那之前按暂停则整轮丢弃。委派子任务不走这条。
        await tm.push_task(
            task, user_prompt_event_jsonable=event_jsonable, provisional=True,
        )
        reg.set_current_task(agent_id, task.id)
        return task.id

    # ── 宿主侧任务派发（spec/09）────────────────────────────────────────────────

    @_session_command(load_session="required")
    @runtime_command
    async def dispatch_task(
        self,
        session_id: str,
        content: "str | list[ContentPart]",
        *,
        agent_id: str | None = None,
        settings: "NormalTaskSettings | dict | None" = None,
        title: str = "",
        description: str = "",
        unattended: bool = False,
        port_key: str | None = None,
        llm_account: str | None = None,
        llm_model: str | None = None,
        llm_config: ModelCallConfig | None = None,
        tenant_id: str | None = None,
    ) -> TurnHandle:
        """在已有 session 上派发一条**全新的顶层 task**（spec/09）——`delegate_task` 的
        host 侧对等物。

        ``tenant_id``：同 `rebuild_session`——host 给什么就是什么，不给则 ``"default"``。

        ``port_key``：这条 task 接在 session 的哪个交互口上（见 `Task.port_key`）。
        **旁支交互线由这里开出来**——给一个新口名，它就与主线并行，各自等自己的对端：

            dispatch_task(sid, "...", agent_id=None, port_key="btw",
                          settings=NormalTaskSettings(
                              use_subagent=True, subagent_template="agent:btw",
                              inherit_from_agent_id=root_agent_id))

        口是**涌现的**：没有 open/close API，给一个没见过的名字就等于开了一条，那个口
        上最后一个 task 终态就等于关了。``None`` = 未声明，按 `unattended` 回落。

        与 `send_message` 的分工：那个是**对话**（有活 task 就并进去，路由归 core），
        这个是**派发**（永不注入既有 task，开不开新 agent、挂谁、继承谁全由调用方声明）。
        与 `start_session(resume=True)` 的分工：那个开的是新一轮 root task，受
        `UnfinishedTasksError`「弃轮禁止」约束；这个只是往活着的会话里加一条 task，
        会话里有别的任务在跑时照样可用，也不发 `SessionResumed`。

        `agent_id` × `settings.use_subagent` 两根轴决定 Task 上两个字段
        （`creator_agent_id` / `assigned_agent_id`），后者才是真正驱动一切的东西：

        | agent_id | use_subagent | creator | assigned | 效果 |
        |----------|--------------|---------|----------|------|
        | A        | False        | A       | A        | 挂 A 上跑，延续它的对话 scope |
        | A        | True         | A       | 新铸     | 从 A 派生，`spawn_depth = A + 1` |
        | None     | True         | ""      | 新铸     | 全新 agent 树，`parent=None`、depth 0 |
        | None     | False        | —       | —        | `ValueError` |

        第四格拒绝而不默认挂 root：「不指定 agent」与「要 root 跑」是两个意图，不替
        调用方猜。要 root 就显式传 `agent_id=<root_agent_id>`。

        **`Session.root_agent_id` 一个字节都不改**——新树只是 ALM 里一条
        `parent_agent_id=None` 的 record，一个 session 下因此可以有多棵 agent 树。
        `agent_ids_of_session` / `cancel_session` / `forget_session` 按 session 收全部
        成员，森林与单树对它们没有区别。

        **重放一致性**（spec/09 §3.1，本入口的地基）：`instantiate(parent_agent_id=X)`
        的 `X` 恒等于 `task.creator_agent_id or None`，因为 `_rebuild_agents` 重放时正是
        按 `creator or None` 推 parent/depth。两边字字对齐，崩溃重建折出同一棵森林。

        **派生不污染 A 的对话**：这条 task 的 `parent_task_id` 是 `None`，`finalize` 的
        派发框与 bubble 两处准入判据都不成立——A 是血缘上的父，不是对话上的父，它从
        自己的视角看不到这次派发发生过。host 的派发不该凭空往一个正在对话的 agent 的
        上下文里插入它没做过的工具调用。

        守卫因此按「这个 agent 要不要亲自执行」分岔（`assert_can_receive` /
        `assert_can_parent`，判据都收敛在 ALM 里）：要它执行则 `running` 拒；只当血缘父
        则 `running` 放行——`delegate_task` 一直就是在父 agent 正跑的时候 spawn 子 agent 的。

        返回的 `TurnHandle.agent_id` 是**真正执行这条 task 的 agent**（新建的情形下就是
        本方法刚铸出来的那个 id，拿到即可用于后续 `send_message` / `get_agent` 寻址）。
        要同步语义就 `await handle.wait_for_finish()`——**它等得准，但返回 `None`**：
        `_state` 只由 runner 回填给它构造时拿到的那一个会话级句柄，本方法另铸的这个
        （与 `send_message` 同一形状）不在回填链上。等待本身如实——判据是事件流上的
        task 终态事件 + 那次 close 边界 recap，与会话级句柄同一条；拿不到的只是
        `LoopState`。要终局信息就读 task 投影或订 `handle.events()`。
        """
        import dataclasses as _dc

        from ctx_weft.core.models.errors import SessionNotFound
        from ctx_weft.core.models.task import deserialize_settings
        from ctx_weft.protocols.content import content_to_text

        # ── 1/2. tenant + 参数校验（零副作用、零 IO）──────────────────────────
        tenant_id = tenant_id or self._DEFAULT_TENANT
        s = settings if isinstance(settings, NormalTaskSettings) else (
            deserialize_settings(settings) if settings is not None else NormalTaskSettings()
        )
        if not isinstance(s, NormalTaskSettings):
            raise ValueError(
                f"dispatch_task: settings must be NormalTaskSettings, got {type(s).__name__}"
            )
        if agent_id is None and not s.use_subagent:
            raise ValueError(
                "dispatch_task: agent_id=None requires settings.use_subagent=True "
                "(that pair means 'start a fresh agent tree'). To run on the session's "
                "root agent, pass agent_id=<root_agent_id> explicitly."
            )
        if s.use_subagent and not s.subagent_template:
            # 入口自己 instantiate（见下），故必须手握**规范形式**的模板 id。没有可靠的
            # 默认可回落——「会话的模板」这个概念已经删干净了（2026-09-30），而 host 配的
            # `fallback_template_ref` 是给「解析不出来」兜底的母版，不是「没说要造什么」的
            # 答案。与其猜，不如要求调用方说清楚要造一个什么型号的 agent。
            raise ValueError(
                "dispatch_task: settings.subagent_template is required when "
                "use_subagent=True (qualified form, e.g. 'agent:researcher')"
            )
        if s.inherit_from_agent_id and not s.use_subagent:
            # 分寸：`inherit_memory` 默认就是 True，`use_subagent=False` 时它静默无效是
            # 既有行为（装配链只在 subagent 分支消费它），不动；但 `inherit_from_agent_id`
            # 没有默认值，写了就一定有意图，无效必须说出来。
            raise ValueError(
                "dispatch_task: settings.inherit_from_agent_id only applies when "
                "use_subagent=True (without a new agent there is nothing to copy into)"
            )
        # 瞬态累加器不接受外部输入（由 delegate_* 写、SuspendStep 清）。
        s = _dc.replace(s, spawn_titles=[])

        # ── 3. 活 owner TM（只读，不写任何注册表）─────────────────────────────
        tm = self._task_managers.get(session_id)
        if tm is None or not tm.is_alive():
            # 冷复活走与 `_start_task_for_agent` 同一条事件重建路径，不另写一套。
            # `keep_alive=True`：这个 TM 刚建好就要被塞新 task，不能让它对「没有可恢复
            # task」得出「空历史报错」或「立刻 finalize 收尾」两个结论（见
            # `recover_session` docstring）。session 在事件日志里不存在 → 这里抛。
            lock = self._session_guard(session_id)
            async with lock:
                await self._recover_session_locked(session_id, keep_alive=True)
            tm = self._task_managers.get(session_id)
            if tm is None:
                raise SessionNotFound(
                    f"session {session_id!r} has no live TaskManager even after cold "
                    f"recovery — this is a bug in the recovery path, not a transient "
                    f"condition; do not retry blindly, file it"
                )

        # ── 4. agent 守卫（判据按「要不要亲自执行」分岔，spec/09 §7.2）─────────
        reg = self._agent_lifecycle_manager
        if agent_id is not None:
            if reg.record_of(agent_id) is None:
                # 与 `send_message` 同一条自愈：registry 的一次 miss 只说明「还没喂进来」
                # （ALM 是只增不删的缓存、进程刚起来时整个是空的），不说明这个 agent
                # 不存在。手握 session_id 就精确装填这一个 session。
                await self._hydrate_agent_for_send(agent_id, session_id)
            rec = reg.record_of(agent_id)
            if rec is None:
                raise AgentNotFound(f"unknown agent: {agent_id}")
            if rec.session_id != session_id:
                raise ValueError(
                    f"agent {agent_id} belongs to session {rec.session_id!r}, "
                    f"not {session_id!r}"
                )
            if s.use_subagent:
                reg.assert_can_parent(agent_id)   # 只当血缘父：running 放行
            else:
                reg.assert_can_receive(agent_id)  # 要它亲自执行：running 拒

        # ── 5. 内容门控（必须先于第 7 步的第一次 emit）────────────────────────
        normalized, event_jsonable = await self._validate_and_normalize_content(
            content, session_id, tenant_id=tenant_id,
        )

        # ── 6/7. 铸 task id，需要新 agent 则就地 instantiate ──────────────────
        # 为什么入口自己建、不交给 `assemble`：`TurnHandle.agent_id` 恒非空要求返回时
        # 就知道执行者是谁，而 `assemble` 要到派发那一刻才跑。预建之后 assemble 走
        # `materialize` 分支（`assigned_agent_id` 已非空），与「子 agent 重派发 / 恢复」
        # 同一条路，不新增分支。副作用是 `SpawnDepthExceeded` 提前到这里同步抛出——
        # 对调用方更好：派发失败当场知道，而不是事后从事件流里发现。
        task_id = generate_id("tsk")
        if s.use_subagent:
            ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id)
            sub_tmpl_id = await self._template_lookup.resolve_qualified(
                s.subagent_template, ctx)
            new_agent, _ = await reg.instantiate(
                template_id=sub_tmpl_id, session_id=session_id, tenant_id=tenant_id,
                # 见 docstring 的重放一致性一段：恒等于 `creator_agent_id or None`。
                parent_agent_id=agent_id,
                task_id=task_id, ctx=ctx,
                # 显式给了才传；省略则沿用 instantiate 既有语义（派生继承父的选择，
                # 无父落 ModelChoice() 跟随账号默认）。
                llm=(ModelChoice(account=llm_account or "", model=llm_model or "",
                                config=llm_config if llm_config is not None else ModelCallConfig())
                     if (llm_account or llm_model or llm_config is not None) else None),
            )
            exec_agent_id, template_id = new_agent.id, new_agent.template_id
        else:
            exec_agent_id = agent_id or ""
            template_id = reg.record_of(exec_agent_id).template_id  # type: ignore[union-attr]

        # ── 8/9. 建 Task 并入队 ───────────────────────────────────────────────
        task = Task(
            id=task_id,
            session_id=session_id,
            status="ACTIVE",
            tenant_id=tenant_id,
            # 顶层 task：与 root task 平级，不参与 `_try_resume_parent` 的父子唤醒，
            # 也不会让 finalize 往创建者的 scope 铸派发框（见 docstring）。
            parent_task_id=None,
            creator_agent_id=agent_id or "",
            assigned_agent_id=exec_agent_id,
            title=title,
            description=description or content_to_text(normalized)[:200],
            user_prompt=normalized,
            user_prompt_event_jsonable=event_jsonable,
            settings=s,
            unattended=unattended,
            # 交互口：调用方显式给的那个；未声明（None）则按 `unattended` 回落。
            port_key=(port_key if port_key is not None else default_port_for(unattended)),
            created_at=now_utc(),
        )
        # `provisional=False`（对比 `_start_task_for_agent` 的 True）：未提交窗口是给
        # 「一条用户消息开出的一轮」准备的——LLM 没开口前不算发生、用户按暂停可整轮
        # 丢弃。host 的一次显式派发不是对话轮，**创建即事实**，与 `_flush_staged` 推
        # 委派子任务同口径。
        await tm.push_task(task, user_prompt_event_jsonable=event_jsonable)

        # ── 10/11/12 ─────────────────────────────────────────────────────────
        # 路由的 current_task 记在**执行者**头上，不是血缘父——派生场景下 A 并不跑它。
        reg.set_current_task(exec_agent_id, task.id)
        lock = self._session_guard(session_id)
        async with lock:
            await self._activate_prepared_tasks_locked(tm)
            self._spawn(tm.drain())
        return TurnHandle(
            session_id=session_id,
            agent_id=exec_agent_id,
            task_id=task.id,
            template_id=template_id,
            event_bus=self._event_bus,
            _storage_health=self.storage_health,
            _storage_recovery=self._wait_for_storage,
            _pending_observe=self._pending_observe_run_id,
            _is_finished=self._task_is_settled,
        )

    @_session_command(load_session="required", skip_no_resume_reply=True)
    @runtime_command
    async def reply_to_hitl(
        self, reply: "HitlReply", *, session_id: str | None = None,
    ) -> "HitlRequestView | None":
        """host 应答的唯一入口。返回已终局请求的视图；已终局再答 → `None`。

        **冷续跑由本返回值驱动，不挂总线订阅**（spec §7.3 订正）：该总线的 handler
        订阅者在 `emit()` 内部同步 drain，且背压下丢事件——把控制流关键信号挂上去，
        「人答了但会话永不续跑」就成了可能。

        **claimed 分流**：`resolved.claimed` 由 `HitlService._commit` 在取走等待槽的
        同一原子段里判定——True 说明一个活协程正在等这个 hitl_id，投递已把它就地叫醒，
        再触发一次冷续跑就是同一个 task 被驱动两次。

        **`agent_id` 防呆**（spec 4.3）：路由仍全靠 `hitl_id`（全局唯一），这一步不是
        路由必需——它要求调用方显式声明「我以为在回复哪个 agent」，与系统记录
        （`PendingHitl.agent_id`）不符则拒绝，而不是静默按 `hitl_id` 走掉。必须在任何
        副作用（消息外部化、状态终局、发事实）之前做，因此在这里、在调
        `self.hitl.resolve()` 之前完成。**严格相等**，空串也不例外：装填期占位项与
        折叠自旧事件的记录可能确无 `agent_id`（`""`），但 host 从 `HitlRequestView`
        读到的正是同一个空串，如实回填即可对上——没有理由放行「我不知道/不声明」。
        未知 `hitl_id` 时 `pending` 为 `None`，这一步不拦（未知 id 的报错留给
        `HitlService.resolve` 的 `KeyError`，与改动前行为一致）。
        """
        pending = self.hitl_registry.get(reply.hitl_id)
        sid = session_id or (pending.session_id if pending is not None else None)
        if sid is None:
            raise AgentNotLoaded("session_id is required for a cold HITL reply")
        if pending is None or not isinstance(pending.delivery, NoResumeDelivery):
            await self.rebuild_session(sid, strict=True)
        pending = self.hitl_registry.get(reply.hitl_id)
        if pending is not None and reply.agent_id != pending.agent_id:
            raise ValueError(
                f"agent_id mismatch: reply says {reply.agent_id!r}, "
                f"hitl {reply.hitl_id} belongs to {pending.agent_id!r}"
            )
        # 开窗必须先于 `resolve`：那一步会取走等待槽、改内存态，而这一轮在 LLM 真的
        # 开口之前随时可能被整体撤销（spec 2026-09-09）。`owns_task=False` —— 这条应答
        # 唤醒的是一个**既有** task，撤销只把它退回开窗前的样子，不摘掉它。
        #
        # **开不出窗就不推迟**：没有 TaskManager（冷启动、会话已被逐出内存）时这条应答
        # 不会经由本进程的某个 run 走到提交点，推迟等于让它永远停在待终局——那正是
        # `reply_to_hitl` 一向要避免的「人答了但会话不动」。开不出窗就照旧一步终局。
        #
        # **已终局的不开窗**：`resolve` 对它是幂等 no-op，这里开出去的窗没有任何一轮来关，
        # 该 task 此后的事件就全被挡在缓冲里（双击 / 陈旧页签重放就能触发）。
        #
        # **已收下过答复（`claim_pending`）→ 直接幂等返回 None**，什么都不碰：它的那一轮
        # 正开着（或正在提交），这次是并发的重复应答。不能落到下面去——不开窗就会以
        # `defer=False` 走一步终局，而 `registry.resolve` 只拒「已终局」，会把待终局的那条
        # 答复原地盖成终局、绕过它自己那一轮的提交与撤销。
        if pending is not None and pending.claim_pending:
            return None
        #
        # 窗若是**本次调用开的**（此前该 task 没有开着的窗），这件事没成时要自己撤掉：
        # `resolve` 抛异常（应答校验失败）或返回 None（并发的重复应答）。不撤，这扇窗没有
        # 任何一轮来关，该 task 此后的事件全被挡在缓冲里。别人开的窗不碰——它有自己的主人。
        round_task_id = ""
        opened_here = False
        _tm = None
        if (pending is not None and pending.task_id
                and not pending.resolved and not pending.claim_pending):
            _tm = self._task_managers.get(pending.session_id)
            # **只有真会续跑的应答才开窗**：窗口靠「那一轮的提交点」来关，没有那一轮就没人
            # 关它，该 task 此后的事件全被挡在缓冲里。两类没有下一轮：
            #   · `NoResumeDelivery`（纯通知 / 取消）——`_resume_after_hitl` 对它什么都不做；
            #   · 目标 task 已终态 / 已不在这个 TM 名下——重排必然 no-op。
            # 不开窗就走一步终局（`defer=False`），`HitlResolved` 当场发出，与「这条应答
            # 没有可撤销的一轮」这个事实一致。
            _target = _tm.get_task(pending.task_id) if _tm is not None else None
            # 第三类不开窗：**授权类应答（`authz` / `rerun`）不可撤销**。
            #
            # 批准一个危险工具之后再「撤回」语义上就不成立——工具可能已经跑了，撤掉的只是
            # 记录，不是世界。今天那条路更糟：`abandon_round` 明文丢掉「那条答复带出来的
            # `CapabilityFinished` 之类」与暂存的工具结果，连 `CapabilityInvoked` 一起丢，于是
            # 重入时 `facts` 为空、gateway 以为它没跑过，**连「准不准重跑」都不问就再跑一遍**。
            #
            # 不开窗 → `defer=False` → `HitlResolved` 当场落盘，工具是拿着一份**已终局**的
            # 批准在跑，于是那枚「已了结」的章能盖在 `CapabilityInvoked` 之后（见
            # `_record_invocation`）——门一过就花掉，这才是审批该有的生命周期。
            #
            # 判据用 **stage 而非 form**：stage 是 gateway 开气泡时显式传的授权语境，form 只是
            # 展示形态（host 完全可以用 approval form 问一个非授权问题）。`ask_user`
            # （`stage=tool`）**保持可撤销**——那是人打的字，2026-09-09 那个撤销窗口正是为它引入的。
            _is_authz = pending.stage in (HITL_STAGE_AUTHZ, HITL_STAGE_RERUN)
            _resumable = (
                not isinstance(pending.delivery, NoResumeDelivery)
                and not _is_authz
                and _target is not None
                and _target.status not in TERMINAL_TASK_STATUSES
            )
            if _is_authz and _tm is not None and _tm.is_round_open(pending.task_id):
                # 不该发生：一个 task 只有一个 run，而开窗的那几条路（消息注入）都会先收口
                # 既有气泡，加上本方法开头 `claim_pending` 那道早返回，未被 claim 的授权气泡
                # 不可能和一扇开着的窗共存。真撞上了要知道——那道缓冲闸按 task_id 定，不按
                # 窗口主人，于是这条 `HitlResolved` 会进别人那扇窗的缓冲；那一轮被丢弃时
                # 内存已终局、日志却丢了终局事实，正是两阶段当初要消灭的那种分歧。
                logger.warning(
                    "reply_to_hitl: 授权类应答 %s 撞上 task %s 上开着的未提交窗口——"
                    "终局事实可能被那一轮的缓冲吞掉，这条不变式要查",
                    pending.id, pending.task_id)
            if _tm is not None and _resumable:
                opened_here = not _tm.is_round_open(pending.task_id)
                _tm.begin_round(pending.task_id, owns_task=False, hitl_id=pending.id)
                round_task_id = pending.task_id
        # `defer`：冷热都只登记待终局，`HitlResolved` 留到 act 的提交点才发。热投递的
        # 协程醒来后由 gateway 调用 commit.rearm_after_hot_reply 重新武装提交点；
        # LLM 开口前被暂停则走热撤销（commit.discard_uncommitted_round）。
        try:
            resolved = await self.hitl.resolve(reply, defer=bool(round_task_id))
        except BaseException:
            if opened_here and _tm is not None:
                await _tm.drop_round(round_task_id)
            raise
        if resolved is None:
            if opened_here and _tm is not None:
                await _tm.drop_round(round_task_id)
            return None                       # 幂等：已终局，不重复续跑
        if resolved.claimed:
            return resolved.to_view()         # 热投递已就地续跑，不得双投
        await self._resume_after_hitl(resolved)
        return resolved.to_view()

    async def _resume_after_hitl(self, req: "PendingHitl") -> None:
        """按 **delivery** 分流续跑——不看 form，不看 capability_id（spec §5）。

        由 `reply_to_hitl` 调用时，`req` 已经**不可逆地终局**（`registry.resolve()`
        已提交、事实已发）——这是「唯一驱动方」路径本身，不是一个可以撤销重试的准备
        阶段。若这里 `recover_session` 抛出，重试 `reply_to_hitl` 只会撞见
        `hitl.resolve()` 对已终局请求的幂等 `None`（不重发事实、不重新触发续跑），
        会话就此永久卡住、且没有第二次机会补上——这正是「既没热投递、也没冷续跑」的
        那个「都没有」路径。异常仍然原样传给调用方（host 需要知道这次应答的续跑没
        成），但**先**用 `hitl_id` 记一条响亮的 exception 日志，让运维不必去反查
        「host 报的这次失败对应哪个已经提交但没跑起来的 HITL」。

        **冷路径必须先精确装填这一个 session**：`req` 本来就同时带着 `agent_id` **和**
        `session_id`（`PendingHitl` 两个字段都有），而 `recover_session` 对 registry miss
        是直接抛 `AgentNotLoaded` 的——它没有 session 语境，无从装填（2026-09-21 起；
        从前那条「扫全部 active session」的 sweep 已删）。所以这一步不是优化，是**这条
        冷应答路径能走通的前提**：漏掉它，人刚答完问题的那次续跑就会摔在
        `AgentNotLoaded` 上，而 `reply_to_hitl` 已经把 HITL 判成终局、没有第二次机会。
        `record_of` 命中就直接跳过——热路径（registry 已装填）里 `_load_agents_of`
        每次都要付一次 `rebuild_view` 的折叠代价，不能让它变成每次应答都白付一遍。

        **只在真会续跑的两个分支里做**，不是方法入口的无条件前置步骤：
        `NoResumeDelivery`（纯通知/取消）本来就不碰事件日志、不解 tenant——这是
        `test_hitl_multimodal_validation.py` 锁死的既有不变式（「纯文本应答不得为
        了解 tenant 去读事件日志」），预装填若挪到方法顶部会在这条路径上凭空引入
        一次从未需要过的事件日志读取，连带把该文件另一条「事件日志故障必须回落、
        不得抛出」的用例也带炸——那条用例期待的失败面是 tenant 解析（已有 best-
        effort 回落），不是本次新增的这次读。
        """
        try:
            if isinstance(req.delivery, ToolResultDelivery):
                await self._hydrate_agent_for_cold_resume(req)
                await self._recover_after_cold_hitl(
                    req.agent_id, req.session_id,
                    resumed_task_id=req.task_id, hitl_id=req.id,
                )
            elif isinstance(req.delivery, UserTurnDelivery):
                await self._hydrate_agent_for_cold_resume(req)
                await self._recover_after_cold_hitl(
                    req.agent_id, req.session_id,
                    user_reply=req, resumed_task_id=req.delivery.task_id,
                    hitl_id=req.id,
                )
            # NoResumeDelivery：纯通知 / 取消，无动作——连预装填都不做。
        except Exception:
            logger.exception(
                "_resume_after_hitl: cold resume failed after the HITL was already "
                "committed (hitl_id=%s, session_id=%s, task_id=%s, delivery=%s) — "
                "the session will not wake up on its own; a retry of reply_to_hitl "
                "won't help (resolve() is idempotent), this needs manual recovery",
                req.id, req.session_id, req.task_id, type(req.delivery).__name__,
            )
            raise

    async def _recover_after_cold_hitl(
        self, agent_id: str, session_id: str, **recover_kwargs: Any,
    ) -> None:
        """Resume only the answered task, independent of legacy agent addressing."""
        async with self._session_guard(session_id):
            await self._recover_session_locked(session_id, **recover_kwargs)

    async def _hydrate_agent_for_cold_resume(self, req: PendingHitl) -> None:
        """Ensure the answered session is fully materialized through rebuild_session."""
        if self._agent_lifecycle_manager.record_of(req.agent_id) is None:
            # tenant **就在手里**：`PendingHitl.tenant_id` 由开请求的调用方从其上下文传入
            # （`HitlService.open` 的契约：「本类自己不持有、也不去解」），冷折出来的那些
            # 则由 `fold_hitl_snapshot` 从 `Event.tenant_id` 填。所以这里不必再去日志反查
            # 一个已经随请求带过来的字段。
            await self.rebuild_session(req.session_id, tenant_id=req.tenant_id, strict=True)

    async def _inject_user_reply(
        self, req: "PendingHitl", session: Session, task_manager: TaskManager,
    ) -> None:
        """把 act 纯文本暂停（wait_for_user）的用户回复注入对话 **并把 task 掰回可跑状态**。

        = `_write_hitl_reply_turn`（只写记忆，幂等）+ 尾部的 task 状态重置（**不幂等**）。
        只有「正在驱动一次续跑」的调用方才该走这个组合——恢复期的补写用
        `_write_hitl_reply_turn`，见 `_inject_resolved_user_turns`（复审 Critical）。

        **状态重置是有条件的**（复审 I7）：一个「SUSPENDED 且尚有活子任务」的父任务，
        `restore` 刻意不把它重排（它要等子任务收尾时由 `_try_resume_parent` 唤醒，那道门
        正是 `status == "SUSPENDED"`）。在这里无条件翻成 PENDING **却没有人入队**，那次
        合法唤醒就被静默吞掉，父任务永久停摆——与恢复路径上早已拆掉的正是同一个形状。
        `_resume_in_existing_tm` 那条分支不受影响：它随后调 `resume_task()`，会真的入队。

        状态重置与 `TaskHumanResolved` 发射都委托给 `TaskManager.mark_human_resolved`
        （D4；Task 6 收拢），而不是 `resume_task`：重建路径上 `restore()` 早于本方法跑，
        且已把该 task 的状态从 `AWAITING_HUMAN` 翻成了 `PENDING`——等本方法执行到这里时
        状态已经"看不出"曾经被人挡住过，`resume_task` 那种"状态仍是
        AWAITING_HUMAN/SUSPENDED 才发"的判据在这条路径上必然落空。
        `mark_human_resolved` 不看当前状态，只要没走上面 SUSPENDED-on-children 的
        提前返回、且非终态，就发一次——这是 `TaskAwaitingHuman{hitl_id}` 在这条路径上
        唯一的解除点。
        """
        target = task_manager.get_task(req.task_id)
        if target is None:
            logger.warning("wait_for_user cold resume: task %s not found for HITL %s",
                           req.task_id, req.id)
            return
        await self._write_hitl_reply_turn(req, session, target)
        if _suspended_on_live_children(task_manager, target):
            logger.info(
                "_inject_user_reply: task %s is SUSPENDED on live children — reply written, "
                "state left alone so _try_resume_parent still wakes it (hitl=%s)",
                target.id, req.id)
            return
        # 清旧进展（restore 已重排,这里保证进展字段正确）；状态置回与事件发射
        # 交给 TaskManager.mark_human_resolved——它不看当前状态，正对得上
        # restore() 早于本方法跑、已把状态从 AWAITING_HUMAN 翻成 PENDING 的场景
        # （resume_task 的 was_blocked 判据在这里必然落空，见其 docstring）。
        target.outputs = None
        target.process_report = None
        target.process_report_at = None
        await task_manager.mark_human_resolved(target.id, hitl_id=req.id)

    async def _write_hitl_reply_turn(
        self, req: "PendingHitl", session: Session, target: "Task",
        *, timestamp: "datetime | None" = None,
    ) -> None:
        """**只把人的答复写进对话，绝不碰 task 状态。**

        拆出来的理由（复审 Critical）：记忆写入靠 `MemoryEvent.id = f"hitlreply:{hitl_id}"`
        幂等，可以在恢复期反复重放；而原先跟在它后面的 task 状态重置
        （`outputs = None` / `status = "PENDING"`）**不幂等**，重放会造成两种实打实的损坏：
        ① 一个 SUSPENDED 在活子任务上的父任务被翻成 PENDING **却没人入队**，
        `_try_resume_parent` 的 `status == "SUSPENDED"` 门随之失效 → 永久停摆；
        ② 早已消费过该答复的 task，其 `outputs` / `process_report` 在此后每次会话恢复时
        被清空 —— 与要关的那个窗口毫无关系的进度损失。

        原 `_inject_user_reply` 的注释与行为在这里逐字保留，仅去掉尾部的状态重置。

        `req` 是新契约的 `PendingHitl`：内容经 `req.decision` 取，打断续接判据经
        `req.delivery.preface`（不再 sniff legacy 的 `context` 字符串）。
        """
        # 这里曾有一道强一致屏障：等上一轮 park 甩出的后台 observe 落库，再注入本轮
        # USER_PROMPT，否则迟到的摘要会越到新消息之后、令下一轮装配误判「续跑」并埋掉
        # 新输入。2026-09-22 拆除——问题的根不在时间戳（`segment_fold` 的锚点早就取
        # `following[0].timestamp - 1μs`、不用 `now_utc()`），在段界是动态查找的：新
        # USER_PROMPT 一落库就成了「最后一条 user 回合」，折叠池随之变空。改由
        # `StepDriver.start` 钉住段界水位线，迟到的折叠自己落回原位，这里不必
        # 再等。人的回复因此不为任何后台 LLM 往返买单。

        # agent_id 必须是本 task 对话真正所在的 agent scope——AgentRecallSource 用
        # recall_recent_by_agent 按 scope.agent_id 过滤召回 task body（≠ recall_recent 的 task_id 键）。
        # 冷重启从事件日志重建的 HITL 丢了 agent_id（HITL_REQUIRED 投影未持久化它，见 reducers），
        # req.agent_id="" 会把回复写进空 agent scope → 对 actor 装配不可见 → 续跑 cue → 空白回复
        # （重启后「第一句」丢失）。回退到 task 的真实 agent（assigned/creator），与首条 USER_PROMPT
        # 落库时同 scope。
        agent_id = _reply_turn_agent_id(req, target)
        scope = MemoryAddress(session_id=session.id, task_id=target.id, agent_id=agent_id)
        pctx = ProviderContext(
            session_id=session.id, tenant_id=session.tenant_id,
            task_id=target.id, agent_id=agent_id,
        )
        from ctx_weft.protocols.content import content_with_prefix
        from ctx_weft.protocols.hitl import HITL_OUTCOME_REJECTED

        # `effective_decision` 而不是 `decision`：两阶段之下（spec 2026-09-09）冷应答
        # 在提交点之前只落成 `pending_decision`，而本方法正是跑在提交点之前的——读
        # `decision` 会拿到 None，用户说的那句话会静默变成空串注进对话。
        eff = req.effective_decision
        message = eff.message if eff else ""
        outcome = eff.outcome if eff else ""
        is_edit_interrupt = (
            isinstance(req.delivery, UserTurnDelivery)
            and req.delivery.preface == PREFACE_AFTER_INTERRUPT_EDIT
        )

        if outcome == HITL_OUTCOME_REJECTED:
            declined_prefix = runtime_prompt("human_declined", message="")
            content = (content_with_prefix(message, declined_prefix)
                       if message else runtime_prompt('human_rejected'))
        else:
            content = message or runtime_prompt('no_response')
            # ① 中途打断（未吐 token）续接：补「上一条请求已取消」说明
            # （preface = PREFACE_AFTER_INTERRUPT_EDIT）。
            if is_edit_interrupt:
                from ctx_weft.core.loop.steps.act import _interrupt_edit_prefix
                prev = await self._last_user_prompt(scope, pctx)
                prefix = _interrupt_edit_prefix(prev)
                content = content_with_prefix(content, prefix)
        # 应答可能被重试（host 超时重发 / 用户连点）：`resolve()` 对已终局请求已幂等
        # no-op（不会二次调用本方法），但这里再加一道幂等键——`id` 是 memory 层的幂等键
        # （provider 已实现），确定性地由 hitl_id 派生（spec §7.3/§12.2）。
        #
        # 键上那一维「第几次应答」（`reply_memory_id`）已是历史包袱：被撤销的答复现在只在
        # 未提交窗口的暂存区里、从没进过 memory，也就没有占着的旧键要躲。见那个函数的说明。
        # ``timestamp``：恢复期补写传那条答复**真正终局的时刻**（`resolved_at`），而不是
        # 「补写的此刻」——它发生得更早，用当前时刻会把它排到本次应答之后，对话顺序颠倒。
        reply_mem_id = reply_memory_id(req)
        await self._ingest_user_turn(
            scope, pctx, content, event_id=reply_mem_id,
            task_id=target.id, source="hitl_reply", timestamp=timestamp,
        )
        # 盖「已了结」章——**必须在 ingest 之后**，顺序纪律见 `EventType.HITL_CLOSED`。
        # 这条请求的持久效果就是上面那次 ingest：答复已经落进对话，它再也不会被补注入。
        # `close()` 自己不抛（一枚事后的章不该回滚已经成功的消费）。
        await self.hitl.close(req)

    async def _emit_message_appended(
        self, session: "Session", task_id: str, agent_id: str, memory_id: str,
        content_jsonable: "str | list[dict] | None", timestamp: "datetime",
    ) -> None:
        """发 `TaskMessageAppended`：注入消息的正文在事件日志里的那一份。见事件定义处的注释。"""
        await emit_event(
            self._event_bus, EventType.TASK_MESSAGE_APPENDED,
            session_id=session.id, tenant_id=session.tenant_id,
            origin=EventOrigin.RUNTIME, task_id=task_id, agent_id=agent_id or None,
            payload={
                "memory_id": memory_id,
                "agent_id": agent_id,
                "content": content_jsonable,
                "source": "send_message",
                "timestamp": timestamp.isoformat(),
            },
        )

    @staticmethod
    def _assistant_key(provider_ctx, address) -> tuple[str, str, str]:
        if not address.agent_id:
            raise ValueError("Assistant memory requires an owning agent")
        return (provider_ctx.tenant_id or "default", address.session_id, address.agent_id)

    def _assistant_agent_lock(self, key):
        # Keep identities stable across eviction while owners/waiters may exist.
        return self._assistant_agent_locks.setdefault(key, asyncio.Lock())

    async def _ensure_assistant_memory(self, session) -> None:
        key = (session.tenant_id or "default", session.id)
        ready = self._assistant_memory_ready.get(key)
        if ready is None or ready[0] is not self.providers.get_memory():
            await self._restore_assistant_messages(session, only_if_uninitialized=True)

    async def _persist_assistant_turn(self, state, ctx, finished, event) -> None:
        """Track before emitting; serialize only this memory owner's final write."""
        from ctx_weft.core.loop.driver import ingest_or_stage

        await self._ensure_assistant_memory(state.session)
        if ctx.memory is not self.providers.get_memory():
            raise RuntimeError("Assistant memory provider changed during the run")
        key = self._assistant_key(ctx.provider_ctx, event.address)
        after_position = self._assistant_memory_ready.get(key[:2], (ctx.memory, 0))[1]
        async with self._assistant_agent_lock(key):
            tm = ctx.task_manager
            provisional = bool(tm is not None and tm.is_round_open(state.task.id))
            entry = _PendingAssistantMemory(
                finished, event, ctx.provider_ctx,
                after_position, provisional=provisional)
            self._assistant_pending.setdefault(key, {})[event.id] = entry
            # Cancellation and uncertain receipts deliberately leave the entry.
            await ctx.event_bus.emit(finished)
            entry.committed = not provisional and self._config.event_commit_policy == "required"
            await ingest_or_stage(ctx.memory, event, ctx.provider_ctx,
                                  task_manager=tm, task_id=state.task.id)
            if not provisional:
                self._assistant_pending[key].pop(event.id, None)

    async def _settle_storage_assistant_memory(self, session):
        for key, entries in list(self._assistant_pending.items()):
            if key[1] == session.id and any(not entry.provisional for entry in entries.values()):
                async with self._assistant_agent_lock(key):
                    await self._repair_assistant_pending(key, skip_provisional=True)

    async def _repair_assistant_pending(self, key, *, skip_provisional=False) -> None:
        """Called with the agent lock held. Empty hot checks perform no storage I/O."""
        from ctx_weft.protocols.events import EventReadProjection
        from ctx_weft.core.control.reducers import REPLAY_BATCH

        pending = self._assistant_pending.get(key, {})
        for record_id, entry in list(pending.items()):
            if skip_provisional and entry.provisional:
                continue
            if not entry.committed:
                # emit may have committed and lost its acknowledgement, or the
                # event may still be provisional. Only durable facts authorize repair.
                head = await self.event_store.committed_head(key[1])
                confirmed = False
                cursor = entry.after_position
                while cursor < head and not confirmed:
                    upper = min(cursor + REPLAY_BATCH, head)
                    rows = await self.event_store.read_range(
                        key[1], after_position=cursor, through_position=upper,
                        include_types=(EventType.LLM_RESPONSE_FINISHED,),
                        projection=EventReadProjection({EventType.LLM_RESPONSE_FINISHED: (("memory_id",),)}))
                    confirmed = any(row.event.id == entry.finished.id
                                    and row.event.payload.get("memory_id") == record_id
                                    and (row.event.tenant_id or "default") == key[0]
                                    and row.event.agent_id == entry.finished.agent_id for row in rows)
                    cursor = upper
                if not confirmed:
                    raise RuntimeError(f"Assistant finished commit not confirmed: {entry.finished.id}")
                entry.committed = True
                entry.provisional = False
            memory = self.providers.get_memory()
            if not await memory.read_records([record_id], entry.provider_ctx):
                await self._ingest_recovered_assistant(memory, entry.memory, entry.provider_ctx)
            pending.pop(record_id, None)

    async def _confirm_assistant_events(self, session) -> None:
        """Explicit recovery retries uncertain single-event commits using original IDs.

        Provisional events must be confirmed as their original round batch by
        TaskManager; replaying one of those as a standalone event is forbidden.
        """
        for key, pending in list(self._assistant_pending.items()):
            if key[:2] != (session.tenant_id or "default", session.id):
                continue
            for record_id, entry in list(pending.items()):
                if entry.committed or entry.provisional:
                    continue
                async with self._assistant_agent_lock(key):
                    if pending.get(record_id) is not entry or entry.committed:
                        continue
                    await self._event_bus.emit(entry.finished)
                    entry.committed = self._config.event_commit_policy == "required"

    @contextlib.asynccontextmanager
    async def _assistant_memory_checkpoint(self, session, address):
        # Never acquire the session initialization lock while owning an agent lock.
        await self._ensure_assistant_memory(session)
        pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id)
        key = self._assistant_key(pctx, address)
        async with self._assistant_agent_lock(key):
            await self._repair_assistant_pending(key)
            yield

    @staticmethod
    async def _ingest_recovered_assistant(memory, event, pctx) -> None:
        try:
            await memory.ingest(event, pctx)
        except Exception:
            # Lost commit acknowledgements and immutable superseded rows are safe.
            if not await memory.read_records([event.id], pctx):
                raise
        if not await memory.read_records([event.id], pctx):
            raise RuntimeError(f"Assistant memory write not confirmed: {event.id}")

    async def _restore_assistant_messages(self, session: "Session", *, only_if_uninitialized=False) -> None:
        """Repair only finished references after the independently verified memory cut.

        State snapshots may be ahead of assistant writes; never use their state
        cut (or their pruned task set) to skip this scan. No model/tool calls.
        """
        from ctx_weft.core.control.reducers import assistant_memory_floor, missing_assistant_memories
        from ctx_weft.core.loop.steps.act import _assistant_memory_event
        from ctx_weft.protocols import LLMUsage, ToolCall

        lock = self._assistant_memory_locks.setdefault(session.id, asyncio.Lock())
        async with lock:
            ready_key = (session.tenant_id or "default", session.id)
            memory = self.providers.get_memory()
            ready = self._assistant_memory_ready.get(ready_key)
            if only_if_uninitialized and ready is not None and ready[0] is memory:
                return
            self._assistant_memory_ready.pop(ready_key, None)
            head = await self.event_store.committed_head(session.id)
            floor = await assistant_memory_floor(self.event_store, session.id, head=head)
            async for missing in missing_assistant_memories(
                    self.event_store, memory, session.id, session.tenant_id,
                    after_position=floor, through_position=head):
                row, = await self.event_store.read_range(
                    session.id, after_position=missing.position - 1,
                    through_position=missing.position,
                    include_types=(EventType.LLM_RESPONSE_FINISHED,))
                ev, p = row.event, row.event.payload
                pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id,
                                       task_id=ev.task_id, agent_id=ev.agent_id)
                event, _ = _assistant_memory_event(
                    MemoryAddress(session_id=session.id, task_id=ev.task_id, agent_id=ev.agent_id),
                    p["content"], [ToolCall(**call) for call in p.get("tool_calls", [])],
                    LLMUsage(**p.get("usage", {})), p.get("turn", 0),
                    anchor=p["memory_id"], timestamp=ev.timestamp)
                key = self._assistant_key(pctx, event.address)
                async with self._assistant_agent_lock(key):
                    if not await memory.read_records([event.id], pctx):
                        await self._ingest_recovered_assistant(memory, event, pctx)
                logger.warning("Restored assistant memory %s from finished event %s", event.id, ev.id)
            self._assistant_memory_ready[ready_key] = (memory, head)

    async def _restore_appended_messages(self, session: "Session") -> None:
        """恢复期补写：`TaskMessageAppended` 记着、memory 里却没有的注入消息，按原 id / 原时间戳写回。

        补的是暂存机制留下的崩溃窗口：注入消息在这一轮提交时**先**随事件补投落盘
        （`TaskMessageAppended`、`TaskRequeued`……），**后**才从暂存区写进 memory。崩在两者
        之间，日志说这条消息来过、host 也已经把它显示出来，memory 里却没有——模型永远看不见。

        幂等：按记录 id 判重（视图里已有即跳过）；已被压缩 supersede 的记录不在视图里，但
        memory 的 id 契约是「已存在的 id（含 superseded）= no-op」，再写一次也不会复活它。
        必需输入补写失败向上传播；装填和派发不得越过尚未恢复的输入。

        **只读快照切面之后的那一段**（`settled_memory_floor`），不读全会话。理由是那条
        「快照不得领先于 memory」的不变式反过来用：存在一张切面为 P 的可用快照 ⟹ position
        ≤ P 的事件其 memory 效果已落盘，所以 P 之前的每一条 `TaskMessageAppended` 必然撞上
        下面那个「视图里已有」而跳过——纯浪费，且随会话长度线性增长（实测 600 条 43ms /
        1.6MB，一条真实长会话上是几万条）。收窄之后上界是「一个快照间隔」（宿主配
        `snapshot_every_n=50`）。

        不变式在**每条**发 `TaskMessageAppended` 的路径上都成立，核对过三处：
        `_inject_user_turn` 的两个分支都**先 ingest 再 emit**（直写分支根本没有窗口；开窗
        分支整段时间窗都开着），而 `TaskManager.commit_round` 在**钩子之后**才 `pop` 掉那扇
        窗，所以暂存落盘期间 `any_round_open()` 恒为真、快照写不下去。

        取不到可用快照 → floor=0 → 退回全量，与从前行为一致。那只发生在首张快照之前或
        `projection_version` 刚 bump 之后。
        """
        from ctx_weft.core.control.reducers import (
            REPLAY_EXCLUDE_TYPES,
            replay_session,
            settled_memory_floor,
        )
        from ctx_weft.protocols.content import content_from_jsonable, hydrate_event_content, normalize_content
        from ctx_weft.protocols import MemoryEvent

        floor = await settled_memory_floor(self.event_store, session.id)
        head = await self.event_store.committed_head(session.id)
        # **分批读**：floor 通常离 head 只有一个快照间隔，但 floor==0（首张快照之前 /
        # 存量库 / projection_version 刚 bump）时区间就是整条会话。一次性物化那一段的代价
        # 与 SnapshotWriter 重锚同源（实测 20 万事件 604.9MB），所以这里同样按区间切。
        # 收集的只是命中类型的那几条——注入消息的条数与会话长度无关。
        events: "list[Event]" = []
        async for batch in replay_session(
                self.event_store, session.id, after_position=floor,
                through_position=head, exclude_types=REPLAY_EXCLUDE_TYPES):
            events.extend(e for e in batch
                          if e.type == EventType.TASK_MESSAGE_APPENDED)
        if not events:
            return
        memory = self.providers.get_memory()
        event_blob_store = self.providers.get_event_blob_store()
        blob_store = self.providers.get_memory_blob_store()
        seen: "dict[str, set[str]]" = {}
        for ev in events:
            p = ev.payload or {}
            task_id, memory_id = ev.task_id or "", p.get("memory_id") or ""
            if not task_id or not memory_id:
                continue
            agent_id = p.get("agent_id") or ev.agent_id or ""
            pctx = ProviderContext(session_id=session.id, tenant_id=session.tenant_id,
                                   task_id=task_id, agent_id=agent_id)
            try:
                if task_id not in seen:
                    view = await load_records(memory,
                        MemoryAddress(session_id=session.id, task_id=task_id),
                        MemoryScope.TASK, pctx, kinds=[MemoryKind.CONVERSATION_TURN])
                    seen[task_id] = {r.id for r in view}
                if memory_id in seen[task_id]:
                    continue
                content = content_from_jsonable(p.get("content") or "")
                if not isinstance(content, str):
                    content = await hydrate_event_content(
                        content, event_blob_store=event_blob_store, ctx=pctx)
                    if blob_store.can_externalize:
                        content = await normalize_content(
                            content, blob_store=blob_store, ctx=pctx)
                raw_ts = p.get("timestamp")
                ts = datetime.fromisoformat(raw_ts) if raw_ts else ev.timestamp
                await memory.ingest(MemoryEvent(
                    id=memory_id,
                    kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                    address=MemoryAddress(session_id=session.id, task_id=task_id,
                                          agent_id=agent_id),
                    content=content, timestamp=ts, role="user",
                    metadata={"task_id": task_id, "source": p.get("source") or "send_message"},
                ), pctx)
                seen[task_id].add(memory_id)
                logger.warning(
                    "_restore_appended_messages: re-ingested injected message %s of task %s "
                    "(crashed between round commit and staged-memory flush)", memory_id, task_id)
            except Exception:
                logger.exception(
                    "_restore_appended_messages: failed to restore %s of task %s",
                    memory_id, task_id)
                raise

    async def _ingest_user_turn(
        self, scope: "MemoryAddress", pctx: ProviderContext,
        content: "str | list[ContentPart]", *, event_id: str, task_id: str, source: str,
        timestamp: "datetime | None" = None,
    ) -> None:
        """把一条**已经算好**的用户侧内容，作为一轮 `CONVERSATION_TURN`（role=user）
        写进 `scope`（TASK 视图）——纯落盘这一步，不判断内容该怎么来、也不碰 task 状态。

        `_write_hitl_reply_turn`（HITL 应答，上面）与 `_inject_user_turn`
        （`send_message` 的注入分支，Task 18）共用同一次 `ingest`：两边的差别只在
        content 怎么派生（HITL 要拒绝措辞/打断续接前缀，`send_message` 就是调用方给的
        原样消息）与幂等键怎么起（`hitlreply:{hitl_id}` vs. 一个新生成的 id）——那部分
        差异留在各自调用方，这里不重复实现第二套 ingest。

        经 `ingest_or_stage` 分流：该 task 开着未提交窗口（HITL 冷应答 / 消息注入开的那一轮）
        → 暂存，这一轮算数时在 `HitlResolved` 之后落盘；没开窗（恢复期补写、挂起等子任务
        的注入）→ 直接写。
        """
        from ctx_weft.core.loop.driver import ingest_or_stage
        from ctx_weft.protocols import MemoryEvent
        await ingest_or_stage(
            self.providers.get_memory(),
            MemoryEvent(
                id=event_id,
                kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                address=scope,
                content=content,
                timestamp=timestamp or now_utc(),
                role="user",
                metadata={"task_id": task_id, "source": source},
            ),
            pctx,
            task_manager=self._task_managers.get(scope.session_id), task_id=task_id,
        )

    async def _inject_resolved_user_turns(
        self, session: Session, task_manager: TaskManager, *,
        skip_hitl_id: str = "",
    ) -> None:
        """恢复期补注入：把该 session 里**已终局的 `UserTurn` 请求**的答复写进对话。

        补的是一个 **`restore` 覆盖不到的洞**（复审 Finding 2）。`ToolResult` 那一类的
        续跑是「补一条 TOOL_RESULT」，由 reconcile 在任务重跑时自己完成；`UserTurn` 的
        续跑却是「把人说的话注入对话」——`restore` 只负责让任务重新入队，注入没人做。
        于是「人答了 → 决定落盘 → 进程在续跑之前崩了」这个窗口里，任务恢复后会重新进
        act，而**人的那句话彻底不见了**。这正是本次重设计要消灭的故障类。

        **只写记忆，绝不碰 task 状态**（`_write_hitl_reply_turn`，复审 Critical）。写入靠
        `MemoryEvent.id = f"hitlreply:{hitl_id}"` 幂等（spec §7.3/§12.2），那个键的存在
        就是为了让这一步可重放：已经注入过 → memory 层 no-op；没注入过 → 这是唯一一次
        把答复救回来的机会。故这里**不需要**记「注入跑没跑过」，那笔账要跨重启，又得多
        一份持久状态（绕回 §3.1）。

        **为什么不碰状态是正确的、而不只是更安全**：parked 真相源改成 registry 之后，
        HITL 已终局的 task 本来就会被 `restore` 重排（ACTIVE → else 分支；SUSPENDED +
        子任务全终态 → children 闸门）。唯一没被重排的形状是「SUSPENDED 在活子任务上」，
        而它恰恰**必须**保持 SUSPENDED——`_try_resume_parent` 以此为门，在子任务收尾时
        唤醒它，那时它进 act 就看得见这里写下的那一轮。三种形状都落对。

        跳过的几类：
        - `skip_hitl_id`：本次调用已由 `user_reply` 显式注入过的那条。
        - `legacy_origin`：本方法关的是**新模型**的崩溃窗口。升级前由 legacy 路径注入过的
          答复，其记忆记录是随机 id、去重不了，补写会凭空多一轮用户发言。
        - `parked_or_inflight_task_ids`：该 task 还挂着别的未决 HITL，或仍在被上一个活
          TaskManager 跑着（与 `restore` 用**同一个**集合——`:1439` 与本处必须一致）。
        - 无 `task_id` / 无 decision 的记录（装填占位项等），以及已终态的 task——
          它不再需要、也不该收到新输入。

        **best-effort**：与 hydration 同一姿态，单条失败只记账、不中断整场恢复。

        ── 代价与它的界（复审 I6）────────────────────────────────────────────
        交互式会话里每一条用户消息都是一次 `UserTurn` HITL，所以「已终局的 UserTurn」
        随会话轮数线性增长，而本方法在**每次冷应答**都会跑一遍。原实现对其中每一条都
        走一次 `_write_hitl_reply_turn`（= 一次 memory ingest，多数是 `hitlreply:` 幂等
        键下的 no-op，外加可能的一次 `_last_user_prompt` 视图读）——O(N) 次写 I/O。

        **界**：先按 task 各读**一次**对话视图，凡是 `hitlreply:{hitl_id}` 已在其中的
        请求直接跳过。这不是启发式：那条记录只可能由 `_write_hitl_reply_turn` 自己写下，
        「在视图里」**就是**「已经注入过」，不存在漏掉一条真正没被续跑的答复的可能。
        视图读不到（provider 不回显 ingest id、记录已被压缩折走）时集合为空 → 退回逐条
        写、行为与加界之前逐字节一致——失败方向朝「多做一次幂等写」，不朝「少救一条答复」。
        于是每次冷应答的代价从 O(N) 次写降到 O(该 session 里有已终局 UserTurn 的 task 数)
        次读。**仍未被界住**的是 `rebuild_session` 的整流折叠，见那里的说明。
        """
        # 鸭子类型的 TM 替身可能没有这个方法（单测）——取不到就当作「没有在跑的 task」。
        _running = getattr(task_manager, "running_task_ids", None)
        running: "set[str]" = set(_running()) if callable(_running) else set()
        candidates: list[tuple[PendingHitl, Any]] = []
        for req in self.hitl_registry.resolved_for_session(session.id):
            # **不看「这个 task 还挂着别的未决问题」**：那是重排要不要放行的判据，与
            # 「这条答复进没进过对话」无关。曾经借用它，代价是同一个 task 上前一条已终局的
            # 答复被跳过，而人答掉那个未决问题时走的是活 TM 那条路（它不做补写）——那条
            # 答复就此永久缺失（审查文档 M8）。
            #
            # 只避开**正在跑**的 task：往一个正在装配 prompt 的 task 的对话里插写是并发风险，
            # 而它正跑着就说明有人在驱动它，这条答复自会在它自己的路径上被消费。
            if (req.id == skip_hitl_id
                    or req.legacy_origin
                    or not isinstance(req.delivery, UserTurnDelivery)
                    or req.effective_decision is None
                    or req.effective_decision.outcome == HITL_OUTCOME_CANCELLED
                    or not req.task_id
                    or req.task_id in running):
                continue
            target = task_manager.get_task(req.task_id)
            if target is None or target.status in TERMINAL_TASK_STATUSES:
                continue                      # 已终态的 task 不再需要（也不该收到）新输入
            candidates.append((req, target))
        if not candidates:
            return

        already: set[str] = set()
        for scope_key in {(t.id, _reply_turn_agent_id(r, t)) for r, t in candidates}:
            already |= await self._injected_reply_ids(session, *scope_key)

        for req, target in candidates:
            if reply_memory_id(req) in already:
                continue                      # 已经注入过（见上「界」）
            try:
                await self._write_hitl_reply_turn(
                    req, session, target, timestamp=req.resolved_at)
            except Exception:
                logger.exception(
                    "_inject_resolved_user_turns: 注入失败 session=%s hitl=%s",
                    session.id, req.id)
                raise

    async def _injected_reply_ids(
        self, session: Session, task_id: str, agent_id: str,
    ) -> "set[str]":
        """该 task 对话里已经存在的 `hitlreply:*` 记忆记录 id。读不出来 → 空集（退回逐条写）。

        scope 必须与 `_write_hitl_reply_turn` 写入时**完全一致**（同一个 agent_id，见
        `_reply_turn_agent_id`）——查错 scope 只会查空，退回逐条幂等写，不会误判成
        「已注入」。

        **best-effort，绝不抛**：这是一层纯优化，失败只能让恢复多做几次幂等写，不能
        让整场恢复停下来。
        """
        try:
            view = await load_records(self.providers.get_memory(),
                MemoryAddress(session_id=session.id, task_id=task_id, agent_id=agent_id),
                MemoryScope.TASK,
                ProviderContext(session_id=session.id, tenant_id=session.tenant_id,
                                task_id=task_id, agent_id=agent_id),
                kinds=[MemoryKind.CONVERSATION_TURN],
            )
        except Exception:
            logger.debug("_injected_reply_ids: 视图读失败 task=%s，退回逐条幂等写",
                         task_id, exc_info=True)
            return set()
        return {r.id for r in view if isinstance(getattr(r, "id", None), str)
                and r.id.startswith("hitlreply:")}

    async def _last_user_prompt(self, scope: MemoryAddress, pctx: ProviderContext) -> str:
        """取 scope 内最近一条 USER_PROMPT 内容（供 ① 打断续接的「上一条取消」说明）。"""
        from ctx_weft.protocols import MemoryKind, MemoryScope
        try:
            view = await load_records(self.providers.get_memory(),
                scope, MemoryScope.TASK, pctx, kinds=[MemoryKind.CONVERSATION_TURN],
            )
        except Exception:
            return ""
        from ctx_weft.protocols.content import content_to_text
        ups = [r for r in view if r.role == "user"]
        return content_to_text(ups[-1].content) if ups else ""

    async def _load_agents_of(self, session_id: str, *, tenant_id: str | None = None):
        """Compatibility helper; all materialization is owned by rebuild_session."""
        n = await self.rebuild_session(session_id, tenant_id=tenant_id, strict=True)
        return n, self._task_managers[session_id].session.tenant_id

    async def _load_agents_view(
        self, session_id: str, view, *, tenant_id: str | None = None,
    ) -> "tuple[int, str]":
        """Shared loader: preserve identity, live tasks and cold-only broadcasts."""
        sess_proj = view.sessions.get(session_id)
        if tenant_id is None:
            tenant_id = (
                sess_proj.tenant_id if sess_proj is not None else self._DEFAULT_TENANT
            ) or self._DEFAULT_TENANT
        # 从前这里还要从投影取一份 `sess_proj.template_id` 当 `load()` 的
        # `fallback_template_id`（agent 自己的模板缺失时借会话那份）。已删：借来的结果是
        # 一个身份不明的 agent 冒充 root，兜底归 `RuntimeConfig.fallback_template_ref`，
        # 在装配解析模板那一刻才生效（见 `TemplateLookup.get_template`）。
        # 未提交窗口里的 task（spec 2026-09-09）折不进 `view`——它们的 `TASK_CREATED`
        # 还没落盘。把它们的 id 交给 `load()`，让它别拿日志折出来的旧值把路由判据倒回去。
        tm = self._task_managers.get(session_id)
        protected = set(tm.open_round_task_ids) if tm is not None else set()
        loaded = await self._agent_lifecycle_manager.load(
            view.agents, session_id=session_id,
            tenant_id=tenant_id,
            protected_current_tasks=protected,
        )
        return loaded, tenant_id

    async def _load_hitl_view(self, session_id: str, view) -> int:
        """Shared loader, including blob hydration and active-pending protection."""
        snapshot = view.hitl
        await self._hydrate_snapshot_messages(snapshot, session_id)
        return self.hitl_registry.load_snapshot(snapshot)

    # 这里从前有个 `_read_session_events_of_types(session_id, types)`——按类型取整条会话的
    # 事件，**不带 task 收窄、不带位置下界**。2026-09-20 删：src 里零调用者（唯一的引用是
    # 一条负向断言测试给它装计数器）。留着的代价不是那几行代码，是它是本轮一直在清的那个
    # 形状的现成模板——下一个要「按类型查一下」的人会照它写，而那条读随会话长度线性增长。
    # 真要按类型读：`reducers.load_events_of_types(store, sid, types, task_id=...)` 带 task
    # 收窄，或者像 `_restore_appended_messages` 那样按 position 区间分批。

    async def _hydrate_snapshot_messages(self, snapshot, session_id: str) -> None:
        """把 `decisions_for` 里的 **event 侧** 内容还原成 memory 侧可用的形态。

        覆盖 `decisions_for` **与** `resolved` 两张表里的全部决定（后者含没有 tool_call_id
        的 `UserTurn` park——它的答复同样要被注入 memory，同样不能带着 event 侧 ref 进去）。

        spec §12.3.3：折叠出来的 message 仍是事件形态（可能是 event blob store 的 ref）。
        直接喂进 memory 会写一个那个 store 永远打不开的引用——图就此静默消失。本方法是
        `fold_hitl_snapshot`（同步、纯函数，结构上做不了 I/O）之后的必经一步，口径与
        现行 `_cold_hitl_decision` 完全一致：hydrate（event → 字节）→ normalize（字节 →
        memory ref），纯文本零成本直通。

        **best-effort，绝不抛**：抛错会卡住整条恢复路径（`recover` 的每个 session、
        `recover_session` 的每次续跑都过这里）。失败一律降级为文本占位——降级本身
        （`downgrade_images_to_text`，纯函数）也在 try 里兜一道，宁可留着原内容也不让
        恢复崩掉。
        """
        # `decisions_for` 与 `resolved` 常指向**同一个** HitlDecision 对象（折叠时同源），
        # 按 id() 去重，避免同一条内容被 hydrate 两遍（第二遍拿到的已是 memory 侧内容，
        # 再喂 hydrate_event_content 会解不开 → 白白降级成占位）。
        targets: dict[int, tuple[str, Any]] = {}
        for key, (decision, _resume_state) in snapshot.decisions_for.items():
            targets.setdefault(id(decision), (str(key), decision))
        for hitl_id, req in snapshot.resolved.items():
            if req.decision is not None:
                targets.setdefault(id(req.decision), (hitl_id, req.decision))
        if not targets:
            return
        from ctx_weft.protocols.content import downgrade_images_to_text, hydrate_event_content, normalize_content
        # tenant 取自快照里的**请求本体**（`pending` / `resolved` 的 `PendingHitl`——它们
        # 由 `fold_hitl_snapshot` 从 `Event.tenant_id` 填），不去日志反查。同一会话的请求
        # 同属一个 tenant，故取到的第一个即可。
        # `decisions_for` 刻意不作来源：`load_snapshot` 给「只有决定、没有请求本体」的快照
        # 造的是 `tenant_id` 缺省为 "default" 的占位项（registry.py 的 ② 分支），拿它当
        # 锚点会把多租户宿主的图落错地方。一条都取不到（空快照不会走到这里）→ default。
        tenant_id = next(
            (r.tenant_id for r in (*snapshot.pending.values(), *snapshot.resolved.values())
             if r.tenant_id),
            self._DEFAULT_TENANT,
        )
        ctx: ProviderContext | None = None
        for key, decision in targets.values():
            message = decision.message
            if not message or isinstance(message, str):
                continue                       # 纯文本无 ref 可转，零 blob IO
            try:
                if ctx is None:
                    ctx = ProviderContext(
                        session_id=session_id,
                        tenant_id=tenant_id)
                hydrated = await hydrate_event_content(
                    message, event_blob_store=self.providers.get_event_blob_store(), ctx=ctx)
                blob_store = self.providers.get_memory_blob_store()
                decision.message = (
                    await normalize_content(hydrated, blob_store=blob_store, ctx=ctx)
                    if blob_store.can_externalize else hydrated)
            except Exception:
                logger.warning(
                    "HITL 恢复：event ref 还原失败，降级为文本占位 (key=%s)", key, exc_info=True)
                try:
                    decision.message = downgrade_images_to_text(message)
                except Exception:                       # pragma: no cover — 纯函数，防御性
                    logger.exception("HITL 恢复：降级占位也失败 (key=%s)", key)

    # 这里从前有 `rebuild_agent(agent_id)` 与 `rebuild_all_agents()`——「只有 agent_id、
    # 不知道 session」时扫**全部** active session 逐个装填，直到撞见那个 agent。
    # 2026-09-21 删，两个理由：
    #
    # ① **它是旧恢复模型的残留。** `recover()` 于 2026-09-09 删除后，装填已整体改成用户
    #    驱动（`rebuild_session` 的 docstring：「这是唯一的按需装填入口……用到哪条装哪条」）。
    #    调用方先装填、再按 agent 操作，是这套模型下的正确用法；替它猜是在补一件它本来就
    #    该做、而且做得到的事。
    # ② **它拿 O(会话数) 换一个调用方本来就知道的值。** 事件按 session 分区存
    #    （spec §6.1），`agent_id` 全局唯一却没有反向索引，所以「只有 agent_id」时找法
    #    只剩枚举。更糟的是它枚举的那个集合是坏的：`list_active_session_ids` 的两条
    #    discard 依据（`SessionFinished` / `SessionStatusChanged`）在 src 下没有任何 emit
    #    调用点，判据因此恒真——「active 集」= 这台机器历史上跑过的**全部**会话。这正是
    #    当初删 `recover()` 的同一个理由，只是它藏在一个按需自愈里、没被一起清掉。
    #
    # registry miss 且调用方没给 `session_id` → 抛 `AgentNotLoaded`（`AgentNotFound` 的
    # 子类，宿主既有的 except 照样接住）。手握 session_id 的 miss 仍然精确自愈，那是
    # 一次定址装填、成本有界，见 `_hydrate_agent_for_send` / `_hydrate_agent_for_cold_resume`。
    #
    # 同日删掉的还有 `rebuild_all_pending_hitl()`——「把所有 active session 的未决 HITL 都
    # 装填一遍」，供只带 hitl_id 的应答入口（`/hitl/{id}/*`）自愈。它枚举会话用的也是
    # `EventStore.list_active_session_ids`（该方法连同 `providers/events/_lifecycle` 那台
    # 状态机一并删除，见协议里的墓碑）。
    #
    # **删它的理由跟上面两条不同，是职责划界**：「有哪些会话」是 host 自己的数据——会话是
    # 它建的，它有自己的会话表和状态列。core 从事件流把这份清单重新推一遍是职责倒置，而且
    # 推得更差：那台状态机的两条 discard 依据（`SessionFinished` / `SessionStatusChanged`）
    # 在 src 下没有任何 emit 调用点，判据恒真，返回的实际是「这个库里出现过的全部会话」。
    # host 侧同样的问题是一条带索引的 status 查询（`PAUSED_HITL` / `PAUSED` 正是「等着人
    # 回话」那一档），既精确又便宜。
    #
    # 顺带解掉一个陷阱：既然判据恒真，谁去「修」它、让 discard 真的生效，
    # `rebuild_all_pending_hitl` 就会开始**漏** HITL——`TERMINAL_STATUSES` 含 `INTERRUPTED`，
    # 而一条被打断的会话完全可能正停在那儿等人回答。一个看起来在修 bug 的改动会静默制造
    # 一个更坏的。判据本身没了，这个反向依赖也就不存在了。
    #
    # host 要按会话装填，拿自己的清单逐条调 `rebuild_session(session_id)` /
    # `rebuild_session(session_id)`——两条都走 `rebuild_view` 的快照 + 增量，O(delta)。

    # ── Internal execution ───────────────────────────────────────────────────

    def _build_provider_ctx(self, session: Session, task: Task, agent: Agent) -> ProviderContext:
        return ProviderContext(
            session_id=session.id,
            tenant_id=session.tenant_id,
            task_id=task.id,
            agent_id=agent.id,
            agent_template_id=agent.template_id,
            timestamp=now_utc(),
            skill_name=task.settings.skill_name if isinstance(task.settings, NormalTaskSettings) else "",
        )

    def _build_assembler(
        self,
        memory: MemoryProvider,
        provider_ctx: ProviderContext,
        task_manager: "TaskManager | None" = None,
    ) -> ContextAssembler:
        deps = AssemblerDeps(
            memory=memory,
            knowledge_providers=self.providers.get_knowledge_providers(),
            template_lookup=self._template_lookup,
            provider_ctx=provider_ctx,
            capability_cache=self._capability_cache.for_context(provider_ctx),
            task_manager=task_manager,
            capability_provider_index={
                p.name: p for p in self.providers.get_capability_providers()
            },
        )
        return ContextAssembler(
            sources=[
                IdentitySource(),
                CapabilitySource(),
                TaskSpecSource(),
                AgentRecallSource(),
                BlackboardSource(),
                SemanticRecallSource(),
                KnowledgeRetrievalSource(),
                GuidanceSource(),
            ],
            deps=deps,
        )

    def _build_gateway(self, memory: MemoryProvider, provider_ctx: ProviderContext) -> CapabilityGateway:
        """本方法只在**派发时**被调（`_execute_task` / 恢复路径），不在 `__init__`——
        故此处现取 blob store 是安全的：宿主 `register_memory_blob_store()` 无论排在
        构造之前还是之后，跑到这里时都已经接上（同 `media/capability.py` 记的那个
        「先取后注册」坑，那边为此改成了调用时解析）。`get_memory_blob_store()`
        文档化为从不抛，未注册回落 `NullMemoryBlobStore`（`can_externalize` 恒 False）。
        """
        return CapabilityGateway(
            capability_cache=self._capability_cache.for_context(provider_ctx),
            capability_providers=self.providers.get_capability_providers(),
            memory=memory,
            event_bus=self._event_bus,
            provider_authorizers=self.providers.get_capability_authorizers(),
            rerun_authorizers=self.providers.get_capability_rerun_authorizers(),
            spill_threshold=self._config.spill_threshold,
            spill_preview_chars=self._config.spill_preview_chars,
            spill_tail_chars=self._config.spill_tail_chars,
            memory_blob_store=self.providers.get_memory_blob_store(),
            pending_results=self._pending_tool_results,
            wait_storage=self._wait_for_storage,
            unstarted_tools=self._unstarted_tools,
            interrupted_tools=self._storage_interrupted_tools,
        )

    async def _task_is_settled(self, session_id: str, task_id: str) -> bool:
        tm = self._task_managers.get(session_id)
        task = tm.get_task(task_id) if tm is not None else None
        if task is None:
            view = await self._historical_task_view(session_id)
            task = view.tasks.get(task_id)
        if task is None or task.status not in TERMINAL_TASK_STATUSES:
            return False
        prepared = self._prepared_recoveries.get(session_id)
        if prepared is not None and any(
                info.get("task_id", key) == task_id for key, info in prepared[2].items()):
            return False
        if self._pending_observe_run_id(session_id, task_id) is not None:
            return False
        # An empty execution registry also describes cold or failed-to-relaunch
        # recaps. Only durable Done discharges the recovery obligation.
        view = await rebuild_view(self.event_store, session_id)
        return not any(info.get("task_id", key) == task_id
                       for key, info in view.pending_observe.items())

    def _pending_observe_run_id(self, session_id: str, task_id: str) -> str | None:
        registry = self._step_executions.get(session_id)
        return registry.pending_run_id(task_id, "observe") if registry else None

    def _build_loop_ctx(
        self,
        assembler: ContextAssembler,
        llm: LLMClient,
        memory: MemoryProvider,
        provider_ctx: ProviderContext,
        gateway: CapabilityGateway,
        cancel_token: CancelToken | None,
        task_manager: "TaskManager | None",
        pause_token: "PauseToken | None" = None,
    ) -> LoopContext:
        return LoopContext(
            steps=self._build_step_driver("prepare", executions=self._step_executions.setdefault(
                provider_ctx.session_id, ExecutionRegistry())),
            assembler=assembler,
            llm=llm,
            memory=memory,
            event_bus=self._event_bus,
            provider_ctx=provider_ctx,
            capability_cache=self._capability_cache.for_context(provider_ctx),
            capability_providers=self.providers.get_capability_providers(),
            capability_gateway=gateway,
            cancel_token=cancel_token,
            task_manager=task_manager,
            event_store=self.event_store,
            hitl=self.hitl,
            waiter=HitlWaiter(self.hitl_registry, timeout_sec=self._hitl_timeout_sec),
            pause_token=pause_token,
            config=self._config,
            # 出网前 rehydrate ref→base64 用（Phase 3b）；未注册时是 NullMemoryBlobStore，
            # rehydrate_content 据其 can_externalize=False 原样返回、零开销。
            blob_store=self.providers.get_memory_blob_store(),
            event_blob_store=self.providers.get_event_blob_store(),
            persist_assistant_turn=self._persist_assistant_turn,
            assistant_memory_checkpoint=self._assistant_memory_checkpoint,
        )

    @staticmethod
    def _build_step_driver(initial_step: str, *, executions=None) -> StepDriver:
        return StepDriver(
            steps={
                "prepare": PrepareStep(),
                "act": ActStep(),
                "observe": ObserveStep(),
                "finalize": FinalizeStep(),
                "suspend": SuspendStep(),
                "compact": CompactStep(),
                "recognize_intent": RecognizeIntentStep(),
                "reconcile": ReconcileStep(),
            },
            initial_step=initial_step,
            executions=executions if executions is not None else ExecutionRegistry(),
        )

    async def _run_loop(
        self,
        state: LoopState,
        loop_ctx: LoopContext,
        driver: StepDriver,
        run_id: str,
        initial_step: str,
        task: Task,
        agent: Agent,
    ) -> LoopState:
        """Execute the step driver loop; emit lifecycle events; return final state.

        Raises on non-retriable errors (after emitting RunFinished).
        """
        # 段 recap 强一致（spec 2026-07-16 §2）：**本 task 或本 agent** 若有在途后台
        # recap（dispatch/interrupt/plain_text/close 边界），先等它折完再开跑——run 的
        # 一切 memory 读写都落在折叠结果之上。无 pending 零开销直通。
        #
        # 两个轴都要等，缺一个都有洞：
        #   task 轴  —— 同一个 task 的下一轮 run（retry / resume / reconcile 重放）。
        #   agent 轴 —— 同一个 agent 的**下一个 task**。`send_message` 打到已终态的
        #               agent 上会走 `_start_task_for_agent` 建新 task，新 task 的
        #               task 轴是空的，够不着上一轮那次仍在飞的 close 边界折叠。
        #               而 close 边界的 launch 恒在 `TaskFinished` **之前**登记（同协程、
        #               无 await 间隔），所以「上一轮刚结束、用户立刻发下一条」这个窗口
        #               里折叠必然在飞——不等的话，新 task 的首次装配会经
        #               `recall_recent_by_agent` 读到上一轮未被 supersede 的 raw
        #               而非胶囊，prompt 白胀一轮的量。
        #
        # recap 的护栏区（幂等护栏/短段门/事件 emit）在其自吞 try 之外、可能以异常终结，
        # 故此处防御吞掉（降级 = 不等待、段保 raw）；shield 保证 run 被取消时不牵连 recap。
        # **正确性那一半已由段界水位线接管**（`StepDriver.start` 钉住段界，
        # `segment_fold` 按它算折叠池）：迟到的折叠不再抢走段界、也不再排到新消息之后。
        #
        # **剩下的是性能**：不等的话首次装配可能读到上一轮尚未被 supersede 的 raw，拿
        # 原文而非胶囊——prompt 白胀一轮的量。这条代价已确认接受（2026-09-22），换来的
        # 是人回复后不必为一次后台 LLM 往返买单。
        #
        # 已知的边缘情况：一个 run 通常只在 `prepare` 装配一次，中途 fold 落地不影响它；
        # 但若这一轮触发了 context recovery 的 `prepare` 重入，第二次装配会读到折叠后的
        # 结果，同一个 run 内上下文前后不一致（变小）。不是错误，代价是 cache 前缀打穿。

        started = make_event(state, EventType.RUN_STARTED, payload={
            "run_id": run_id,
            "initial_step": initial_step,
        }, origin=EventOrigin.RUNTIME)
        key = (state.session.id, task.id)
        commit = _ActorRunCommit(started)
        self._pending_actor_runs[key] = commit
        if loop_ctx.task_manager is not None:
            loop_ctx.task_manager.bind_round_run(task.id, run_id)
        run_error: BaseException | None = None
        was_cancelled = False
        cancel_takes_effect = False
        cancel_source = ""
        try:
            await self._event_bus.emit(started)
            commit.start_confirmed = True
            async for outcome in driver.run(state, loop_ctx):
                if outcome.state_patch:
                    state = state.apply_patch(outcome.state_patch)
        except RoundDiscarded:
            # 这一轮在 LLM 开口之前被用户中止 → 当作没发生过（spec 2026-09-09）。
            #
            # 这里**不需要**任何抑制：窗口在整条 unwind 路径上都还开着（丢弃是
            # `_run_task` 在最后一步做的），所以 `finally` 照常发的那条 RUN_FINISHED
            # 同样落进缓冲、同样被一起丢掉。抛出点已经在窗口里发过 `TASK_CANCELED`
            # 把 agent 送回 `idle`，那条也一样。原样重抛，交给 `_run_task` 收尾。
            #
            # 反过来说：**丢弃之后**再发任何事件都会直接落盘。所以顺序不能动——
            # 「先在窗口里把状态摆平，最后一步才关窗丢弃」是这套设计的全部纪律。
            logger.info("_run_loop: task %s discarded before first chunk", task.id)
            raise
        except HitlPark as park:
            # run 的结局：这次执行停在「等人答一句」。**task 变成什么不在这里决定**
            # （Task 4）：TaskManager 据本 outcome 走处置表落 AWAITING_HUMAN 并发
            # TaskAwaitingHuman——挡住这个 task 的那一个请求就是 park.hitl_id（act 的
            # tool call 循环是串行的，第一个 park 就 unwind 整个 run，唯一确定）。
            # run_error 保持 None → finally 发 RUN_FINISHED(awaiting_human,
            # will_retry=False)，与委派挂起同形。
            state = state.apply_patch({"run_outcome": RunOutcome(
                kind=RunOutcomeKind.AWAITING_HUMAN, hitl_id=park.hitl_id,
            )})
            logger.info("_run_loop: task %s parked on HITL", task.id)
        except asyncio.CancelledError:
            if self._closing:
                raise
            was_cancelled = True
            # Record cancellation of this execution, without changing task status.
            # After RunFinished, TaskManager applies any durable threshold decision
            # as FAILED rather than turning that root task into CANCELED.
            cancel_takes_effect = task.status not in TERMINAL_TASK_STATUSES and not task.switched_to_task_id
            # 取消来源（总账 A3 剩下的一条）：token 是本 runtime 的协作取消；否则是外部
            # asyncio 取消（进程 shutdown / `wait_for` 超时）。二者产生同一个
            # `CancelledError`，只有 token 自己能区分。`cancel_token` 不是 `_run_loop`
            # 的形参，实测活在 `loop_ctx.cancel_token`。
            # 裁定（task-3-report.md「裁定后的实现」）：来源标签只进 RUN_CANCELED 的
            # payload，**不进 `RunOutcome.reason`**——后者会流进 `TaskCanceled.payload`，
            # 撞上刻意定下的 R5（CANCELED 不编造 reason，`disposition_for` 里 reason
            # 非空才放键）。RunOutcome.reason 保持空串，与之前行为等价。
            by_token = loop_ctx.cancel_token is not None and loop_ctx.cancel_token.is_cancelled
            cancel_source = "token" if by_token else "external"
            state = state.apply_patch({"run_outcome": RunOutcome(kind=RunOutcomeKind.CANCELED,
                observe_id=state.extra.get("observe_id", ""))})
            if loop_ctx.task_manager is not None:
                try:
                    await loop_ctx.task_manager.retract_canceled_run(task.id, run_id)
                except PersistenceUnavailableError as exc:
                    run_error = exc
                    raise
        except LLMOutageError as exc:
            # Task 2（loop 产出 RunOutcome，尚无消费者）：outage 硬编码 retriable=False——
            # **不得**转发 exc.retriable（LLMOutageError.retriable 恒为 True，见
            # protocols/llm.py；今天"outage 从不原地重试"靠的是路径隔离，不是这个标志位，
            # 转发会让 outage 在预算充足时被错误地原地重试。见 task_disposition.py 顶部契约。
            state = state.apply_patch({"run_outcome": RunOutcome(
                kind=RunOutcomeKind.INTERRUPTED, reason=InterruptReason.LLM_OUTAGE,
                error_code=InterruptReason.LLM_OUTAGE, error=str(exc), retriable=False,
            )})
            # 瞬时 LLM 故障自愈耗尽 / 中途断流 → 可恢复中断，**不是** task 失败。
            # task 置 INTERRUPTED（非终态，与 HitlPark 同形）→ _run_task 走挂起分支不判 FINISHED，
            # restore() 在 /resume 时据非终态重排；不发 TASK_FAILED；不增 failure_counter；
            # run_error 保持 None → finally 不再抛出（不经 _handle_task_failure）。
            # task.error / task.error_code 不在这里写：上面构造的 RunOutcome 已带着
            # error=str(exc)、error_code=InterruptReason.LLM_OUTAGE，`_run_task` 拿到
            # 后交 `apply_run_outcome`（task_manager.py）按同样的值写回 task——写两遍是
            # 纯冗余（Task 2 死代码清理）。`apply_run_outcome` 在 `_settle` 之前把这份
            # error_code 落到 task 对象上，`get_task(task_id)` 等内存态查询读到的
            # 已经是它写的那份，时序上稳（见 tests/unit/test_outage_interrupt_reason.py）。
            logger.warning("_run_loop: task %s interrupted by LLM outage: %s", task.id, exc)
            # run 级事实：这次执行死了。**无条件发**，与 task 后续怎么处置无关。
            # 会话状态由 TM 聚合后交给 SM 判定——这里不宣布会话怎么了。
            await self._event_bus.emit(make_event(state, EventType.RUN_INTERRUPTED, payload={
                "reason": InterruptReason.LLM_OUTAGE, "error_message": str(exc)},
                origin=EventOrigin.RUNTIME))
            # task 级事实（TaskInterrupted）不在这里发：outage 的 RunOutcome 带着
            # retriable=False 交给 TaskManager，由处置表判成 INTERRUPTED 并发出——
            # 「outage 从不原地重试」的判据从路径隔离变成了这个显式标志位（Task 4）。
        except PersistenceUnavailableError as exc:
            # spec: event-commit——存储不可用：不进通用重试语义。task 状态不动、
            # 不发 RUN_INTERRUPTED/终态事件（emit 同样要过提交门、会再撞同一故障）；
            # 会话已由 CommitGate 先标记 storage_unavailable，drain 停止派发。
            # re-raise 交 TaskManager 的同名分支收尾（那边同样不再发任何事件）。
            run_error = exc
            logger.error("_run_loop: task %s halted — session storage unavailable", task.id)
            raise
        except Exception as exc:
            run_error = exc
            # 这份 outcome **不是**给 TaskManager 的（本分支下面 `raise run_error`，
            # `state` 根本不返回给调用方；处置用的那份由 `TaskManager._run_task` 的
            # `except Exception` 就地构造）。它是给下面 finally 里 `RUN_FINISHED.outcome`
            # 用的——run 得说出自己是怎么结束的，否则崩溃支会兜底报成 "completed"。
            # retriable 取 `getattr(exc, "retriable", True)`，与 outage 支硬编码的 False
            # **不同源**，不许合并成一份（见 task_disposition.py 顶部契约）。
            state = state.apply_patch({"run_outcome": crash_run_outcome(exc)})
            # 运行层崩溃 = 可恢复中断的临时标记（非终态）：re-raise 交 _handle_task_failure
            # 定夺——原地重试（翻回 PENDING）或挂起等 /resume（保持 SUSPENDED + 发事件）。
            # 真失败只有 observer 判 fail 一条路（FinalizeStep 闭合胶囊、回传父亲）。
            # ContextOverflowError 不再特判终态：retriable=False 使其跳过重试直接挂起，
            # 溢出文案随 task.error / RUN_INTERRUPTED.error_message 抵达 host，错误码
            # 随 TASK_INTERRUPTED.error_code 直接上浮（提示换大窗口模型）——不再绕经
            # 已停发的会话级队列聚合信号。
            # 崩溃发生时 task 是否已是终态（如 observer 已判 FAILED、随后 FinalizeStep 又
            # 抛异常那条窄路径）——是的话下面 RUN_INTERRUPTED 也不发：那次执行的终局
            # 已经由 TaskFailed/RunFinished{FAILED} 宣布过，再发一条 RunInterrupted 会
            # 让「靠类型存在与否判断这次执行是否非正常终止」的 host（docs/events-v2.md
            # §2.4）误报一次「非正常终止」（M1）。与上面 A1 守卫、outage 支的
            # was_interrupted 同一个判据（M3 的教训：别让两处判断各写一份）。
            was_interrupted = task.status not in TERMINAL_TASK_STATUSES
            if was_interrupted:
                task.error = str(exc)
            if getattr(exc, "retriable", False):
                logger.warning("_run_loop: task %s failed (retriable): %s", task.id, exc)
            else:
                logger.exception("_run_loop: run failed for task %s", task.id)
            # run 级事实：这次执行确实死了，与 task 接下来是原地重试（TaskRequeued）
            # 还是挂起等 /resume（TaskInterrupted）无关——那个决定归
            # TaskManager._handle_task_failure，在 run 外面、判完才知道；run 域的四条
            # 事实（Started/Canceled/Finished/Interrupted）一律从这里发，别处不发。
            # 但 task 已是终态时不发（见上面 was_interrupted 的注释）。
            if was_interrupted:
                await self._event_bus.emit(make_event(state, EventType.RUN_INTERRUPTED, payload={
                    "reason": InterruptReason.RUN_CRASH,
                    "error_code": crash_error_code(exc),
                    "error_message": str(exc),
                }, origin=EventOrigin.RUNTIME))
        finally:
            if loop_ctx.capability_cache is not None:
                loop_ctx.capability_cache.evict(agent.id)
            canceled_event = None
            # RunCanceled describes the execution, not the task's terminal outcome.
            # Existing terminal tasks are sticky. A pending threshold failure is
            # applied by TaskManager only after this RunFinished has committed.
            if was_cancelled and cancel_takes_effect:
                # 取消来源标签（总账 A3）落在这里，不落进 RunOutcome/TaskCanceled：
                # RUN_CANCELED 没有任何 reducer 分支，全仓只有测试查它「发没发」，
                # 对无人消费的事件做纯增量不必顾虑 R5（CANCELED payload 不编造 reason）。
                canceled_event = make_event(state, EventType.RUN_CANCELED, payload={
                    "run_id": run_id, "source": cancel_source,
                }, origin=EventOrigin.RUNTIME)
            # will_retry=True suppresses SSE close on the host side.
            # cancelled → False; retriable=False → TaskManager won't retry anyway.
            will_retry = (
                run_error is not None
                and not isinstance(run_error, PersistenceUnavailableError)
                and task.retry_count < task.max_retries
                and getattr(run_error, "retriable", True)
            )
            # Task 3：run 自己的结局，run 词表（RunOutcomeKind）五值之一。正常跑完那条路
            # 不经任何 except 分支，`state.run_outcome` 已由 FinalizeStep/SuspendStep 的
            # state_patch 挂好（driver.run 循环里 `state = state.apply_patch(...)`）；
            # 仍为 None 是理论上不可达的兜底（例如驱动没有产出任何 StepOutcome 就正常退出）
            # ——按“没炸、没挂起、没取消”兜底为 COMPLETED，而不是让 host 拿到空 outcome。
            outcome_kind = (
                state.run_outcome.kind if state.run_outcome is not None
                else RunOutcomeKind.COMPLETED
            )
            # Keep the original envelope until confirmed, including unknown receipts.
            # Storage isolation delays publication; it does not close the run.
            finish = make_event(state, EventType.RUN_FINISHED, payload={
                "outcome": outcome_kind.value,
                # `final_status` 已废弃，下个周期删除——host 应改读上面的 `outcome`
                # （run 词表）。Task 4 起 run 不再写 task 状态，故这里只是**发 RUN_FINISHED
                # 那一刻** task 的状态（多半仍是 ACTIVE）：真正的终态由随后 TaskManager 的
                # 处置写定，靠它推 task 状态的 host 一定要改。
                "final_status": task.status,
                "will_retry": will_retry,
                "total_events": state.sequence_counter,
                "total_turns": len(state.transcript),
                "error": str(run_error) if run_error else None,
                "error_type": type(run_error).__name__ if run_error else None,
            }, origin=EventOrigin.RUNTIME)
            if isinstance(run_error, PersistenceUnavailableError):
                finish.payload["outcome"] = RunOutcomeKind.INTERRUPTED.value
            commit.finished = finish
            if not self._closing and not isinstance(run_error, PersistenceUnavailableError):
                if canceled_event is not None:
                    await self._event_bus.emit(canceled_event)
                await self._event_bus.emit(finish)
                self._pending_actor_runs.pop(key, None)

        if run_error is not None:
            raise run_error

        return state

    async def _execute_task(
        self,
        session: Session,
        task: Task,
        agent: Agent,
        template: AgentTemplate,
        run_id: str,
        memory: MemoryProvider,
        resolved_model: ResolvedModel,
        cancel_token: CancelToken | None = None,
        pause_token: "PauseToken | None" = None,
        initial_step: str = "prepare",
        task_manager: "TaskManager | None" = None,
        scope_agent_id: str | None = None,
    ) -> tuple[LoopState, TurnHandle]:
        provider_ctx = self._build_provider_ctx(session, task, agent)
        assembler = self._build_assembler(
            memory, provider_ctx, task_manager=task_manager)
        gateway = self._build_gateway(memory, provider_ctx)
        # 这次 run 实际用的 client——调用方（AgentBinding.model / run_single_task 的
        # resolved_model）已经解好，这里不再自己解析。三样东西各归各位：选择住
        # agent record，身份/窗口住这份 ResolvedModel，都不回填进 session
        # （回填冻结账号默认的问题见 agent_lifecycle_manager.py ModelChoice 的 docstring）。
        llm = resolved_model.client
        loop_ctx = self._build_loop_ctx(
            assembler,
            llm,
            memory,
            provider_ctx,
            gateway,
            cancel_token,
            task_manager,
            pause_token=pause_token,
        )

        scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=scope_agent_id or agent.id)
        state = LoopState(
            run_id=run_id,
            session=session,
            task=task,
            agent=agent,
            scope=scope,
            extra={"template": template},
            resolved_model=resolved_model,
            # 兜底：StepDriver.run 从 initial_step 开始的每一步都会用 Task 2 的
            # `_STEP_ORIGIN` 表覆盖这个值，这里的 LOOP_DRIVER 只在「驱动器还没跑
            # 第一步就已经有事件要发」这个理论缝隙里生效，不代表任何具体子步骤。
            origin=EventOrigin.LOOP_DRIVER,
        )
        driver = loop_ctx.steps
        driver.initial_step = initial_step
        state = await self._run_loop(state, loop_ctx, driver, run_id, initial_step, task, agent)
        handle = TurnHandle(
            session_id=session.id,
            agent_id=agent.id,
            task_id=task.id,
            template_id=template.id,
            event_bus=self._event_bus,
            _state=state,
            _storage_health=self.storage_health,
            _storage_recovery=self._wait_for_storage,
            _pending_observe=self._pending_observe_run_id,
            _is_finished=self._task_is_settled,
        )
        return state, handle


class _SessionTaskRunner:
    """两阶段 TaskRunner（每个 owner-TM 一个实例）：assemble 装配执行 agent，execute 驱动 step loop。

    原 _make_task_runner 闭包的显式化：闭包捕获 → 实例字段。恢复播种不再靠
    per-runner 缓存——agent 身份/配置的唯一住所是 runtime 级 `AgentLifecycleManager`
    registry（`lm`），恢复路径由 `recover_session`/`rebuild_session` 经共用的
    `_load_agents_of` 显式调 `lm.load()` 装填。
    assigned_agent_id 回填 / started_at / TASK_STARTED 均归 TaskManager（两阶段契约）。

    **不持有模板**（2026-09-30）：从前这里有一对 `_template`/`_template_id` 字段，
    装的是建 runner 那一刻按「会话的模板」解出来的那份，`assemble` 的非 subagent
    分支直接把它配给执行 agent。那是错的——runner 是 per-session 的，而执行 agent
    不是：`send_message` 可以把新一轮开在任意一个 park 中的子 agent 上
    （`_start_task_for_agent` 建的 task 是 `use_subagent=False`，正好落那个分支），
    于是子 agent 跑的是 root 的人格与 root 的 capability 授权。模板的唯一真相是
    agent record 的 `template_id`（`AgentInstantiated` 钉死的出身），所以现在每次
    `assemble` 按执行 agent 现解，见 `_template_of`。
    """

    def __init__(
        self,
        *,
        runtime: "CtxWeftRuntime",
        session: Session,
        lm: AgentLifecycleManager,
        memory: MemoryProvider,
        task_manager: TaskManager,
        handle: "TurnHandle | None" = None,
    ) -> None:
        self._runtime = runtime
        self._session = session
        self._registry = lm
        self._memory = memory
        self._task_manager = task_manager
        self._handle = handle

    async def confirm_finished(self, task_id: str) -> None:
        """Confirm the original actor start and finish without executing steps."""
        key = (self._session.id, task_id)
        await self._runtime.hitl.confirm_opens(self._session.id, task_id=task_id)
        commit = self._runtime._pending_actor_runs.get(key)
        if commit is not None:
            if commit.finished is None:
                raise RuntimeError(f"Actor run has not stopped: {commit.started.run_id}")
            # A buffered emit is not a persistence receipt. Confirm the owning
            # round before retrying the original finish, retaining it on failure.
            bus = self._runtime._event_bus
            if bus.is_run_provisional(task_id, commit.started.run_id):
                await self._task_manager.commit_round(task_id)
            if not commit.start_confirmed:
                await bus.emit(commit.started)
                commit.start_confirmed = True
            await bus.emit(commit.finished)
            self._runtime._pending_actor_runs.pop(key, None)

    # ── 阶段 1：装配 ─────────────────────────────────────────────────────────

    async def assemble(self, task_id: str) -> "AgentBinding | None":
        """决定并实例化执行 agent + memory 预备 + reconcile 探测（原 _resolve）。"""

        t = self._task_manager.get_task(task_id)
        if t is None:
            return None
        sess_id = self._session.id
        tenant_id = self._session.tenant_id
        await self._runtime._ensure_assistant_memory(self._session)

        # Cython 3.2 does not support structural pattern matching.
        s = t.settings
        if isinstance(s, NormalTaskSettings) and s.use_subagent is True:
            ctx = ProviderContext(session_id=sess_id, tenant_id=tenant_id)
            # 只有真新建才需要它（要造一个什么型号的 agent）；已有 assigned 的那条路
            # 读 agent record 自己的出身，见下。从前这里以 `or self._template_id`
            # 回落「会话的模板」，那个回落连带把重派发/恢复也拖进了 root 的模板——
            # 兜底的唯一住所现在是 `RuntimeConfig.fallback_template_ref`（host 声明的
            # 母版），由 `TemplateLookup.get_template` 内部消化，装配层不再自备一份。
            sub_tmpl_id = (
                await self._runtime._template_lookup.resolve_qualified(s.subagent_template, ctx)
                if s.subagent_template else ""
            )
            # 分支判据：t.assigned_agent_id 是否为空。真新建走 instantiate——
            # Registry 内部会按因果顺序发 AgentSpawned/SpawnRejected/AgentInstantiated；
            # 已有值时只是按同一 id 重新水合对象（重派发 / 恢复），走 materialize
            # （零事件、只读 record，不会把 spawn_depth 重算成 0）。SpawnDepthExceeded
            # 由 instantiate 内部发完 SpawnRejected 后原样上抛，此处不再捕获。
            if not t.assigned_agent_id:
                agent, tmpl = await self._registry.instantiate(
                    template_id=sub_tmpl_id, session_id=sess_id, tenant_id=tenant_id,
                    # `or None`：`instantiate` 的判据是 `is not None`，空串会被当成
                    # 一个真父落进 `_register_fallback("")`，给一个不存在的 agent 建
                    # record。创建者为空 = 无父 = 森林里的一棵新树（spec/09 §3.1），
                    # 这正是 `_rebuild_agents` 重放时 `creator or None` 的同一口径——
                    # 两边必须字字对齐，否则内存态与重放态的 parent/depth 会分叉。
                    parent_agent_id=t.creator_agent_id or None,
                    task_id=t.id, ctx=ctx,
                )
                # instantiate() 刻意不解模型（惰性不变量）；这里现解一次供
                # AgentBinding.model——不缓存，现解现弃是设计的一部分
                # （LLMClientResolver 才是那层缓存）。
                rm = self._registry.resolve_model(agent.id)
            else:
                agent, rm = self._registry.materialize(
                    t.assigned_agent_id, session_id=sess_id, tenant_id=tenant_id)
                # materialize 不返回 template（它只读 record，不碰 TemplateLookup）
                # → 按这个 agent **自己**的出身解一份。从前这里解的是 `sub_tmpl_id`
                # （settings 里那个 subagent_template），两者在重派发/恢复时并不总是
                # 同一个值：settings 可能没带模板（那时旧代码回落会话模板），而 agent
                # 的出身早已由 `AgentInstantiated` 钉死——record 才是真相。
                tmpl = await self._template_of(agent, ctx)
            t.assigned_agent_id = agent.id
            async with self._runtime._assistant_memory_checkpoint(
                    self._session, MemoryAddress(session_id=sess_id, agent_id=agent.id)):
                pass
            if s.inherit_memory and not t.user_prompt_in_memory:
                # 继承源三级，**显式优先**（spec/09 §6）：
                #   ① settings.inherit_from_agent_id —— host 直接锚定，绕过推导。
                #      血缘与记忆来源是正交的两个轴：跨树继承（新树 agent 拿 root
                #      的上下文）只能由这一级表达。
                #   ② 父任务的 agent —— 委派子任务的既有行为。
                #   ③ 上一条 root task 的 agent —— 直派 sub-agent 的 root 回合没有
                #      parent_task_id，不回落它的话子 agent 会空手起跑。
                # ②③ 逐字保留，存量行为零变化。
                src_aid, src_tid = s.inherit_from_agent_id, None
                if not src_aid:
                    src_t = (
                        self._task_manager.get_task(t.parent_task_id) if t.parent_task_id
                        else _latest_prior_root_task(self._task_manager, t)
                    )
                    if src_t:
                        src_aid = src_t.assigned_agent_id or src_t.creator_agent_id or ""
                        src_tid = src_t.id
                if src_aid:
                    async with self._runtime._assistant_memory_checkpoint(
                            self._session, MemoryAddress(session_id=sess_id, agent_id=src_aid)):
                        await _copy_memory_for_inherit(
                            source_agent_id=src_aid, child_task=t, sub_agent=agent,
                            memory=self._memory, session_id=sess_id, tenant_id=tenant_id,
                            source_task_id=src_tid,
                        )
            initial = await self._reconcile_or(t, agent, "prepare")
            return AgentBinding(agent_id=agent.id, agent=agent, template=tmpl,
                                initial_step=initial, run_id=generate_id("run"), model=rm)

        else:
            # 非 subagent 任务在**创建者**的 agent scope 上跑（延续创建者对话），
            # 而非一律 root——否则 subagent 派生的非 subagent 子会跑进 root scope、丢失
            # 创建者上下文并污染 root。scope 键与调度串行判定共用 effective_agent_id 单一真相。
            agent, rm = self._registry.materialize(
                effective_agent_id(t, self._session.root_agent_id or ""),
                session_id=sess_id, tenant_id=tenant_id,
            )
            # **模板跟着执行 agent，不跟着 session**（2026-09-30 修）。从前这里是
            # `template=self._template`（建 runner 时按会话模板解出的那份）：root task
            # 恰好对，而 `send_message` 投给一个 park 中的子 agent 时，开出的新 task
            # 正是 `use_subagent=False`、落这条分支——于是那个子 agent 跑 root 的
            # identity 与 root 的 capability 授权（它自己 `capability_refs` 里的
            # forbidden 一并失效），而同一次 run 的
            # `provider_ctx.agent_template_id` 报的又是它自己的模板，两者自相矛盾。
            tmpl = await self._template_of(
                agent, ProviderContext(session_id=sess_id, tenant_id=tenant_id))
            async with self._runtime._assistant_memory_checkpoint(
                    self._session, MemoryAddress(session_id=sess_id, agent_id=agent.id)):
                pass
            initial = await self._reconcile_or(t, agent, "prepare")
            return AgentBinding(agent_id=agent.id, agent=agent, template=tmpl,
                                initial_step=initial, run_id=generate_id("run"), model=rm)

    # ── 阶段 2：执行 ─────────────────────────────────────────────────────────

    async def execute(self, binding: "AgentBinding", task_id: str) -> "RunOutcome | None":
        """驱动 step loop，把 run 的结局交回 TaskManager（Task 4）。

        返回值取自 `_run_loop` **最终返回的那个 state**（`apply_patch` 会换对象，
        入口那个 state 上没有 run_outcome）。返回 None = 这次执行没留下结局（task 已
        不存在），由 TM 按 COMPLETED/success 兜底。崩溃不从这里回：`_run_loop` 重抛，
        TM 的 `except Exception` 就地构造 outcome。
        """
        t = self._task_manager.get_task(task_id)
        if t is None:
            return None
        # 单 owner 架构 seam：model 在派发时从可变 session 读取；控制令牌 per-run 发放——
        # 随本次派发登记进 runtime registry、run 结束注销，pause/cancel 经 registry 必达在途 run。
        # 出生信号在登记点按执行 agent 分流：pause 弃子窗口内 root run born-pause、
        # 非 root run born-cancel（一次暂停恰一个续跑点，且气泡必落 root agent）。
        tokens = self._runtime._register_run_tokens(
            self._session.id, task_id,
            root_run=binding.agent_id == (self._session.root_agent_id or ""),
        )
        # assemble 窗口补偿：TM 已整体取消（cancel_all 在 assemble 期间到达）→ 本 run 出生即取消。
        if self._task_manager.is_cancelled():
            tokens.cancel.cancel()
        try:
            if self._task_manager.is_task_tree_canceled(task_id):
                tokens.cancel.cancel()
                return RunOutcome(kind=RunOutcomeKind.CANCELED)
            s, _ = await self._runtime._execute_task(
                session=self._session,
                task=t,
                agent=binding.agent,
                template=binding.template,
                run_id=binding.run_id,
                memory=self._memory,
                resolved_model=binding.model,
                initial_step=binding.initial_step,
                task_manager=self._task_manager,
                cancel_token=tokens.cancel,
                pause_token=tokens.pause,
            )
        finally:
            self._runtime._deregister_run_tokens(self._session.id, task_id, expected=tokens)
        if s is None:
            return None
        if self._handle is not None:
            self._handle._state = s
        return s.run_outcome

    # ── helpers（原闭包内嵌函数）───────────────────────────────────────────────

    async def _template_of(
        self, agent: "Agent", ctx: "ProviderContext",
    ) -> "AgentTemplate":
        """这个 agent 的模板——每次装配按它自己的出身现解。

        真相源是 agent record 的 `template_id`（`AgentInstantiated` 钉死、经
        `materialize` 带出来的那个），**不是** session、也不是 task settings：一个
        session 里同时跑着不同型号的 agent 是常态（委派出来的子 agent、`send_message`
        直接投给某个 park 中的子 agent），把会话那份配给它们就是让它们冒充 root。

        代价是每次装配付一次 `get_template`，而 `LocalAgentTemplateProvider` 每次都
        重新扫盘（热更新友好是它有意的设计）。**不做「与会话模板相同就复用」的短路**：
        那等于把「用谁的模板」这个判断又交回给一个 per-session 的字段，而它正是本次
        要删掉的东西。

        解不出来不在这里兜底：`TemplateLookup.get_template` 自带
        `RuntimeConfig.fallback_template_ref`（host 声明的母版）那一层，连母版都解不出
        才抛——那时已经不是「某个模板没装好」，而是部署坏了，该大声报。
        """
        return await self._runtime._template_lookup.get_template(
            agent.template_id, None, ctx=ctx,
        )

    async def _reconcile_or(self, t: "Task", agent: "Agent", base: str) -> str:
        """base initial_step；若该 task 最近 assistant turn 有 dangling tool_call → reconcile。"""
        from ctx_weft.protocols.context import ProviderContext as _PCtx
        from ctx_weft.protocols.memory import MemoryAddress as _Scope
        sess_id = self._session.id
        scope = _Scope(session_id=sess_id, task_id=t.id, agent_id=agent.id)
        pctx = _PCtx(session_id=sess_id, tenant_id=self._session.tenant_id,
                     task_id=t.id, agent_id=agent.id)
        if await _task_has_dangling_tool_call(self._memory, scope, pctx):
            return "reconcile"
        return base
