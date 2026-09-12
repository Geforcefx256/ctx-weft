"""TaskManager：监听事件 + 调度 TaskQueue + parent resume 逻辑。

Phase 4 §4.2 + §4.7。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ctx_weft.core.models.discriminators import CancelReason, InterruptReason, TaskErrorCode
from ctx_weft.core.models.status import PARKED_TASK_STATUSES, TERMINAL_TASK_STATUSES
from ctx_weft.core.utils.event import emit_event
from ctx_weft.core.orchestrator.task.failure_threshold import plan_threshold_trip
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.orchestrator.task.reopen import build_reopen_prompt
from ctx_weft.core.models.errors import crash_error_code, crash_run_outcome
from ctx_weft.core.loop.park import RoundDiscarded
from ctx_weft.core.orchestrator.task.disposition import (
    RunOutcome,
    RunOutcomeKind,
    disposition_for,
)
from ctx_weft.core.orchestrator.task.queue import QueueEntry, TaskQueue, split_deps
from ctx_weft.core.orchestrator.task.runner import AgentBinding, TaskRunner, effective_agent_id
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.status import TaskStatus
from ctx_weft.core.models.task import CompactTaskSettings, MetadataFillerTaskSettings, NormalTaskSettings, Task
from ctx_weft.core.utils.clock import as_utc, now_utc
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.events import EventOrigin, EventType

if TYPE_CHECKING:
    from ctx_weft.protocols.events import EventBus

logger = logging.getLogger(__name__)

_ORIGIN = EventOrigin.ORCHESTRATOR_TASK_MANAGER

#: 词表住在 `core.domain.status`（会话/task/agent 三套并排，见该模块 docstring）。
#: 这里的两个别名只为不改动本文件里的既有引用点。
_PARKED_STATUSES = PARKED_TASK_STATUSES

#: 这条守卫此前长在 `_run_loop` 的三个 except 支里，随发射一起搬来。
_TERMINAL_STATUSES = TERMINAL_TASK_STATUSES

# 默认值；实际值由 host 经 RuntimeConfig → TaskManager 构造参数注入。
_DEFAULT_MAX_RETRIES    = 3
_DEFAULT_MAX_CONCURRENT = 4


class TaskManager:
    """Manages a per-session TaskQueue and drives task execution.

    The caller (SessionRegistry / CtxWeftRuntime) must:
    1. Register a task_runner callback (runs a single task).
    2. Call push_task() to add tasks.
    3. Call tick() or rely on event-driven draining.
    """

    def __init__(
        self,
        session_id: str,
        max_concurrent: int | None = None,
        task_max_retries: int | None = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._session_id = session_id
        self._max_concurrent = max_concurrent if max_concurrent is not None else _DEFAULT_MAX_CONCURRENT
        self._task_max_retries = task_max_retries if task_max_retries is not None else _DEFAULT_MAX_RETRIES
        self._queue: TaskQueue = TaskQueue()
        self._cancelled: bool = False
        self._tasks: dict[str, Task] = {}
        # 同一轮（一次 task run）内 delegate_task / delegate_plan 先投这里，
        # runner 正常返回后由 _flush_staged 统一入队，实现「同批次 FIFO」。
        # key = 正在运行的 task_id（即被 spawn 子任务的 parent_task_id）。
        self._staged: dict[str, list[tuple[Task, list[str] | None, str | None]]] = {}
        self._parent_map: dict[str, str] = {}  # child_task_id → parent_task_id
        self._children_of: dict[str, set[str]] = {}  # parent_task_id → set[child_task_ids]
        self._runner: TaskRunner | None = None
        self._running_tasks: set[str] = set()
        # 派发后登记的「真实执行 agent id」（task_id → binding.agent_id）——
        # 同 agent 串行判定对在跑任务用真值，只有队列候选才走 effective_agent_id 预测。
        self._running_agents: dict[str, str] = {}
        #: task_id → 这一轮的开窗快照（spec 2026-09-09）。键存在即「窗口开着」：
        #: 这个 task 的事件只到达进程内状态机，不落盘、不到 host，直到 act 收到本轮
        #: 第一个 chunk 才提交。详见 `EventBus` 的类 docstring 与 `begin_round`。
        self._rounds: dict[str, RoundSnapshot] = {}
        self._lock = asyncio.Lock()
        self._session: Session | None = None  # 注入后供 failure_counter 维护使用
        self._event_bus: "EventBus | None" = event_bus
        self._background_asyncio_tasks: set[asyncio.Task] = set()
        #: 一次性接线的 7 个回调（见 hooks.py）。整体替换，不逐字段合并。
        self._hooks = TaskManagerHooks()
        # spec: event-commit：会话存储健康检查（runtime 注入；None = 无健康面）。
        # drain 据此跳过隔离会话的派发——存储不可用时不再开新副作用。
        self._unhealthy_check = None
        # 归属权谓词：runtime 注入，返回本 TM 是否仍是该 session 的当前 owner。
        # None = 不受管（永远视为 current，保持旧行为）。被同 session 上更新的 TM
        # 顶替后返回 False → 迟到的收尾变 no-op（不发 SessionFinished、不清新一轮的控制信号）。
        # pause 弃子窗口标记（runtime.pause_session 置位、_on_idle/_release 复位）：
        # 置位期间任务取消不改 session 状态、run 收尾 staged 直接丢弃。
        self._pause_abandon = False
        # ── 熔断真终结（failure threshold trip）状态 ──────────────────────────────
        # 幂等闩：trip 后在途任务再失败会重进 FAILED 分支，没有它会重复清场 + 重复发事件。
        self._threshold_tripped: bool = False
        # spec: task-handoff——永久阻塞扫描的重入闩：处置里的 on_task_finished 会再进
        # 扫描入口，闩住后由外层不动点循环统一覆盖。
        self._in_blocked_dispose: bool = False
        # (title, reason) 随 failure_counter 同步积累（FAILED 追加、FINISHED 清空）；
        # 供 FAILURE_THRESHOLD_HIT payload 与 threshold_finalizer 引用。跨崩溃恢复不重建，接受。
        self._recent_failures: list[tuple[str, str]] = []
        # 三个 trip 序列的注入点（接线方式镜像 set_is_current）：None = 该副作用跳过，
        # trip 序列本身永远不因缺注入而崩溃。runtime 侧实现见 Task 10。
        # 统一取消胶囊闭合（Task 14）：cancel_all / 熔断清场（已启动挂起排队） / 在途协作取消 funnel
        # 三处调用点共用同一注入点。None-tolerant：缺注入时三处调用点自身各自跳过、不崩溃。
        # 事件侧 blob store 的注入点（set_event_blob_store / _event_blob_store /
        # _event_ctx）已删除：TASK_CREATED 自 Task 3 起、TASK_REQUEUED 自本任务
        # （blob-store 解耦 Task 5）起都改由调用方/task 上携带的现成 event jsonable
        # 供给，TaskManager 不再需要持有 event blob store——reopen_task 零 blob IO
        # 是**结构性**保证：这个类根本没有能力发起一次 blob 调用。

    def track_background(self, t: "asyncio.Task") -> None:
        """Track a fire-and-forget background coroutine so the session awaits it before close."""
        self._background_asyncio_tasks.add(t)
        t.add_done_callback(self._background_asyncio_tasks.discard)

    def set_unhealthy_check(self, check) -> None:
        """注入会话健康检查（spec: event-commit）。drain 跳过隔离会话的派发。"""
        self._unhealthy_check = check

    def set_runner(self, runner: TaskRunner) -> None:
        self._runner = runner

    def set_hooks(self, hooks: TaskManagerHooks) -> None:
        """一次性装好全部回调。**整体替换**，不逐字段合并（见 TaskManagerHooks）。"""
        self._hooks = hooks

    def set_session(self, session: Session) -> None:
        """注入 Session 对象，供 failure_counter 维护使用。"""
        self._session = session

    @property
    def session(self) -> "Session | None":
        """注入的 Session 对象（复用路径据它读/写本轮 llm 参数——model=会话状态）。"""
        return self._session

    def register_task(self, task: Task) -> None:
        self._tasks[task.id] = task

    def restore(
        self,
        all_tasks: list[Task],
        terminal_ids: set[str],
        parked_task_ids: set[str] | None = None,
    ) -> None:
        """Rebuild task registry and re-queue resumable tasks after a crash.

        ``parked_task_ids``: tasks with an unresolved pending HITL (spec/07 §9.1) —
        SUSPENDED but must STAY parked (not re-queued) until ``/answer`` resumes them.
        Empty/None → legacy behavior.

        挂在**已终局** HITL 上的 task 不需要在这里另开一条重排口子：ACTIVE 的走下面的
        ``else`` 分支、SUSPENDED 且子任务都终态的走 children 闸门、SUSPENDED 且尚有活
        子任务的由 ``_try_resume_parent`` 在子任务收尾时唤醒（那些子任务本身也被这一趟
        restore 重排了）。**反而不能**在这里把它提前置 PENDING——`_try_resume_parent`
        以 ``status == "SUSPENDED"`` 为门，提前改状态会把那次合法的唤醒静默吞掉，父任务
        就只跑了「早的那一次」、拿不到子任务的产出（Task 9 复审 Finding 1）。

        compact / recognize_intent are no longer scheduled as Tasks; any such obsolete
        task found in a replayed event stream is skipped (never re-queued). Recovery is
        condition-based.
        """
        parked = parked_task_ids or set()

        for t in all_tasks:
            self._tasks[t.id] = t
            if t.parent_task_id:
                self._parent_map[t.id] = t.parent_task_id
                self._children_of.setdefault(t.parent_task_id, set()).add(t.id)

        self._queue.seed_completed(terminal_ids)
        # spec: task-handoff——成功集与终态集分开装填：FINISHED 的才释放 on_success 依赖。
        succeeded_ids = {t.id for t in all_tasks if t.status == "FINISHED"}
        self._queue.seed_succeeded(succeeded_ids)

        for t in all_tasks:
            if t.status in TERMINAL_TASK_STATUSES:
                continue
            if isinstance(t.settings, (CompactTaskSettings, MetadataFillerTaskSettings)):
                continue  # obsolete ephemeral helpers — never re-scheduled (recovery is condition-based)
            # HITL-park：有未决 HITL → 保持挂起、不入队，**不论 ACTIVE 还是 SUSPENDED**。
            # 审批热等的任务恒为 ACTIVE（不会走 SUSPENDED 分支）；park 判据是"有无未决 HITL"
            # （parked_task_ids，源自 HitlRegistry.list_pending），而非 task.status（spec/07 §9.1，缺陷 A）。
            if t.id in parked:
                continue
            if t.status == "SUSPENDED":
                children = self._children_of.get(t.id, set())
                if all(cid in terminal_ids for cid in children):
                    t.status = "PENDING"
                    t.retry_count = 0  # 崩溃挂起带着耗尽的计数；恢复重跑从零重计
                    self._queue.push(QueueEntry(
                        task_id=t.id, session_id=self._session_id, priority=t.priority,
                    ))
            else:
                t.status = "PENDING"
                t.retry_count = 0
                # spec: task-handoff——按条件重建两集：any 依赖对任意终态释放、
                # success 依赖只对 FINISHED 释放（terminal-but-not-succeeded 的依赖
                # 保持阻塞，交给恢复期的永久阻塞扫描处置——A 已 FAILED 落盘、
                # B 的级联取消未落盘的崩溃窗口正是靠那个扫描兜底）。
                any_deps, success_deps = split_deps(t.dag_deps, t.dep_conditions)
                any_deps -= terminal_ids
                success_deps -= succeeded_ids
                self._queue.push(QueueEntry(
                    task_id=t.id, session_id=self._session_id,
                    priority=t.priority,
                    blocked_by=any_deps, blocked_success=success_deps,
                ))

    async def push_task(
        self,
        task: Task,
        blocked_by: list[str] | None = None,
        parent_task_id: str | None = None,
        *,
        user_prompt_event_jsonable: "str | list[dict] | None" = None,
        provisional: bool = False,
    ) -> None:
        """入队一个新任务并发 TASK_CREATED。

        ``provisional=True``：这个 task 由一条**用户消息**开出（`send_message` 的新建
        分支），在 LLM 真的开口之前它不算发生——开一道未提交窗口（`begin_round`，
        `owns_task=True`），本方法发出的 `TASK_CREATED` 连同随后的 `TASK_STARTED` /
        `RUN_STARTED` / `LLM_PROMPT_SENT` 一并只到达进程内状态机，不落盘、不到 host。
        act 收到本轮第一个 chunk 时 `commit_round` 补投，用户在那之前按暂停则
        `discard_round` 整轮丢弃。
        委派子任务（`_flush_staged`）**不走这条**：它们不是一轮对话的开端，且
        「创建即落盘」对它们仍然必要（见下方注释）。

        ``user_prompt_event_jsonable``：TASK_CREATED 里 user_prompt 的 event 侧载荷，
        由**入口**（`SessionRegistry.create_session` ← `CtxWeftRuntime.start_session`）
        从**归一化之前的原始** content 算好传进来。本方法不自己算：`task.user_prompt`
        到这里已是 memory 侧归一化过的内容，图片 part 是 memory ref，再算一次只会把
        一个 event store 打不开的引用写进事件（blob-store 解耦 Task 3）。

        不传（agent 派发的子任务经 `stage_task` → `_flush_staged` 走这条）时退回
        ``task.user_prompt`` 本身：那条路径的 prompt 恒是工具参数里的纯文本
        （见 `control_capability` 的 delegate_task / plan_tasks），jsonable 形态就是
        它自己，没有字节、也无处取原始字节。真出现 part 列表则**响亮拒绝**——静默
        塞进 payload 会直接击穿「事件库恒不含字节」。
        """
        # 统一用 TaskManager 级别的 max_retries，覆盖 Task 模型的硬编码默认值
        task.max_retries = self._task_max_retries
        # 开窗**必须先于**下面那条 TASK_CREATED——晚一步它就已经落盘了。
        if provisional:
            self.begin_round(task.id, owns_task=True)
        self._tasks[task.id] = task
        if parent_task_id:
            self._parent_map[task.id] = parent_task_id
            self._children_of.setdefault(parent_task_id, set()).add(task.id)
        if blocked_by:
            task.dag_deps = list(blocked_by)
            # spec: task-handoff——写入时物化缺省：新派发的依赖缺省 on_success，
            # 显式声明（delegate_plan 的 run_if）优先。读取侧（restore/queue）因此
            # 永远不需要猜缺省——存量事件没有 dep_conditions 才按 any 解释。
            conds = dict(task.dep_conditions or {})
            for dep in blocked_by:
                conds.setdefault(dep, "success")
            task.dep_conditions = conds

        any_deps, success_deps = split_deps(task.dag_deps, task.dep_conditions)
        entry = QueueEntry(
            task_id=task.id,
            session_id=self._session_id,
            priority=task.priority,
            blocked_by=any_deps,
            blocked_success=success_deps,
        )
        self._queue.push(entry)
        logger.debug(
            "TaskManager.push_task: %s blocked_any=%s blocked_success=%s",
            task.id, any_deps, success_deps,
        )
        # 创建即落盘：未跑过的排队 / blocked task 也进事件日志，
        # 崩溃恢复时 restore() 可由 dag_deps 重建依赖链，无需父任务重新 spawn。
        # push_task 是唯一的「新建」路径（retry / resume / active 重排都走 _queue.push），
        # 故此处恰好 emit 一次。
        user_prompt_jsonable = user_prompt_event_jsonable
        if user_prompt_jsonable is None:
            if isinstance(task.user_prompt, list):
                raise ValueError(
                    f"push_task({task.id}): 非纯文本 user_prompt 必须由调用方传 "
                    "user_prompt_event_jsonable（由归一化之前的原始 content 算出）——"
                    "事件库恒不含字节，这里没有原始字节可用"
                )
            user_prompt_jsonable = task.user_prompt   # str | None，本身即 jsonable
        # 挂在 task 上供 reopen_task 直接复用（零 blob IO——reopen 不引入新图，见
        # Task.user_prompt_event_jsonable 的字段注释）。
        task.user_prompt_event_jsonable = user_prompt_jsonable
        await self._emit(
            EventType.TASK_CREATED, task_id=task.id,
            payload=task_payload(task, user_prompt_jsonable=user_prompt_jsonable),
        )

    # ── 未提交窗口（spec 2026-09-09）────────────────────────────────────────
    #
    # 窗口的所有权在 TaskManager：它是 task 生命周期的主人，「这一轮算不算数」是它的
    # 判断。总线只负责执行分流（谁收得到），act 只负责在正确的时刻喊一声提交或丢弃。

    def begin_round(
        self, task_id: str, *, owns_task: bool, hitl_id: str = "",
    ) -> None:
        """开窗 + 拍这一轮的回滚快照。

        ``owns_task``：这个 task 是不是**这条消息开出来的**。
          · True（`send_message` 的新建分支）—— 丢弃要连 task 一起摘掉，agent 回 `idle`。
          · False（消息注入一个既有 task：HITL 冷续跑 / `/messages` 注入）—— task 早就
            存在、是一段正在进行的对话，丢弃只把它退回开窗前的样子，agent 回
            `waiting_human`。

        ``hitl_id``：被这条消息收口的那个旧气泡（有的话）。丢弃时经 `revert_round`
        钩子 `release` 回 pending —— 那正是「撤销前的世界」。

        幂等：重复开窗不覆盖已拍的快照、也不清空已攒的缓冲（retry 重排会重进同一条路径）。
        """
        if not task_id or task_id in self._rounds:
            return
        task = self._tasks.get(task_id)
        self._rounds[task_id] = RoundSnapshot(
            owns_task=owns_task,
            hitl_id=hitl_id,
            task_status=getattr(task, "status", ""),
            retry_count=getattr(task, "retry_count", 0),
            outputs=getattr(task, "outputs", None),
            process_report=getattr(task, "process_report", None),
            process_report_at=getattr(task, "process_report_at", None),
        )
        bus = self._event_bus
        if bus is not None:
            bus.begin_provisional(task_id)

    def is_round_open(self, task_id: str) -> bool:
        return task_id in self._rounds

    def round_hitl_id(self, task_id: str) -> str:
        """这一轮收口了哪个旧气泡（没有则空串）。供 `revert_round` 钩子 release 用。"""
        snap = self._rounds.get(task_id)
        return snap.hitl_id if snap is not None else ""

    @property
    def open_round_task_ids(self) -> "frozenset[str]":
        """当前开着窗口的 task id。

        两个读者：`Runtime._load_agents_of` 用它保住路由判据（这些 task 的
        `TASK_CREATED` 还没落盘，折不进 `AgentView`，热重装若照 view 覆盖
        `current_task_id` 就会把它倒回上一个已终态的 task）；`Runtime.pause_session`
        用它决定要不要跳过排队弃子（见那里）。
        """
        return frozenset(self._rounds)

    async def commit_round(self, task_id: str) -> None:
        """这一轮算数了：按序补投缓冲里的事件，关窗，丢掉快照。

        触发点有两个，**都不在本类**：act 收到本轮第一个 chunk（正常路径），以及
        `_run_task` 的收尾兜底（见那里——失败、outage 耗尽、装配炸掉一律提交，
        只有用户主动中止才丢弃）。
        """
        if self._rounds.pop(task_id, None) is None:
            return
        # `ROUND_COMMITTED` **先于**补投，且**不带 task_id**（那道闸按 task_id 定，带上
        # 就会被自己挡住）。顺序是给 host 看的：它据此把攒着的用户消息帧 flush 出去，
        # 那一帧必须排在这一轮的 task/run 帧**之前**——用户先说话，agent 才开跑。
        await self._emit(
            EventType.ROUND_COMMITTED, task_id=None, payload={"task_id": task_id},
        )
        bus = self._event_bus
        if bus is not None:
            await bus.commit_provisional(task_id)

    async def discard_round(self, task_id: str) -> None:
        """这一轮当作没发生过。**只用于用户主动中止**（首 chunk 之前按暂停 / 取消）。

        五步，顺序就是这段代码的全部要点——**每一步都必须发生在关窗之前**，窗口一关，
        此后发的任何事件都直接落盘：

        1. `revert_round` 钩子：memory 里那条用户消息 `fold` 掉、被收口的旧气泡
           `release` 回 pending（两样都在 orchestrator 之下，TM 够不到，见 hooks.py）。
        2. 还原快照：`status` / `retry_count` / `outputs` / `process_report*`。后三样
           是 `_inject_user_turn` 就地清掉的，没有别处存过旧值。

           **`title` / `description` 不在其列**：唯一会就地改它们的是
           `update_task_metadata`（`recognize_intent` 调），而旁路的起飞点在
           `act._commit_round` —— 那时窗口已经关了，它跑在窗口之外，改不到窗口里的状态。
        3. 发一条把 agent 拨回原位的 task 事件——**这是关键的一步**，且必须先于第 4 步
           （事件的 agent_id 从 `_running_agents` 反查，先清就发不出去了）：
           · owns_task → `TASK_CANCELED`（ALM 映射成 `SETTLED` → `idle`）
           · 否则     → `TASK_AWAITING_HUMAN{hitl_id}`（ALM → `waiting_human`）
           两者都是 ALM 本来就有的转移，不需要给它引入任何可回滚状态；而它们自己也在
           窗口里，随缓冲一起被丢掉 —— **内存状态正确，日志干净**。
        4. 队列与登记复位：新建的连 task 一起摘；既有的把重排塞进去的那个条目撤掉。
        5. 关窗丢缓冲。
        """
        snap = self._rounds.get(task_id)
        if snap is None:
            return

        # ① TM 够不到的那部分
        if self._hooks.revert_round is not None:
            try:
                await self._hooks.revert_round(task_id)
            except Exception:
                logger.exception(
                    "discard_round: revert_round hook failed for task %s "
                    "(继续丢弃——一次撤销失败不该把暂停变成 run 崩溃)", task_id)

        task = self._tasks.get(task_id)
        if task is not None and not snap.owns_task:
            # ② 还原被这条消息就地改掉的字段
            task.status = snap.task_status
            task.retry_count = snap.retry_count
            task.outputs = snap.outputs
            task.process_report = snap.process_report
            task.process_report_at = snap.process_report_at

        # ③ 把 agent 拨回原位（事件在窗口里，只到 ALM，不落盘）。
        #
        #    **必须先于下面的清理**：`_emit` 的 agent_id 是从 `_running_agents` /
        #    `Task.assigned_agent_id` 反查的，先清后发会发出一条没有 agent_id 的事件，
        #    而 ALM 的 `handle_event` 对没有 agent_id 的事件直接返回——agent 就永远停在
        #    `running` 了。（这条踩过一次，别再把顺序调回去。）
        if snap.owns_task:
            await self._emit(
                EventType.TASK_CANCELED, task_id=task_id,
                payload={"reason": "discarded_before_first_chunk"},
            )
        else:
            await self._emit(
                EventType.TASK_AWAITING_HUMAN, task_id=task_id,
                payload={"hitl_id": snap.hitl_id,
                         "reason": "discarded_before_first_chunk"},
            )

        # ④ 队列与登记复位。用 `cancel` + `unmark_running` 而**不是** `mark_complete` /
        #    `mark_failed`：后两者会把 task_id 塞进 `_completed`（给依赖解阻塞用的
        #    「已了结」集合）。一轮没发生过的 task 不该出现在那里。
        self._queue.cancel(task_id)
        self._queue.unmark_running(task_id)
        self._running_agents.pop(task_id, None)
        self._running_tasks.discard(task_id)
        self._staged.pop(task_id, None)
        if snap.owns_task:
            self._tasks.pop(task_id, None)

        # ⑤ 关窗——必须最后
        self._rounds.pop(task_id, None)
        bus = self._event_bus
        if bus is not None:
            bus.discard_provisional(task_id)

        # 关窗之后才发这条：它**不带 task_id**（绕开那道闸），host 据此把攒着的用户
        # 消息帧丢掉、把文本退回输入框当草稿。放在最后是为了「host 看到它的时候，这一轮
        # 已经确定不会再有任何事件出来了」。
        await self._emit(
            EventType.ROUND_DISCARDED, task_id=None,
            payload={"task_id": task_id, "reason": "discarded_before_first_chunk"},
        )

    def stage_task(
        self,
        task: Task,
        blocked_by: list[str] | None = None,
        parent_task_id: str | None = None,
    ) -> None:
        """把子任务投入当前 run 的缓冲区，延迟到 _flush_staged 才真正入队。

        与 push_task 同签名；调度顺序问题（同批次 FIFO）由 _flush_staged 统一处理。
        缓冲按 parent_task_id 分桶，避免 max_concurrent>1 时并发 run 的 spawn 串台。
        """
        key = parent_task_id or ""
        self._staged.setdefault(key, []).append((task, blocked_by, parent_task_id))
        logger.debug("TaskManager.stage_task: %s under parent=%s", task.id, key)

    def detach_staged(self, from_parent_id: str, to_parent_id: str | None) -> None:
        """把 from_parent_id 本轮 staged 的子任务改投到 to_parent_id 名下（独立后续）。

        用于 finish_task 与 delegate_* 同批出现时：当前 task 收尾，被派发任务
        不再做它的阻塞子任务，而改挂到它的 parent（当前是 root 则为 None→顶层）独立调度。

        只改写 tuple 的 parent_task_id（驱动 _flush_staged 时 push_task 的归属）与
        task.parent_task_id，**不搬桶**——桶 key 仍是当前运行 task 的 id，_flush_staged
        才能在本 run 结束时正常弹桶入队。plan 内部兄弟间的 blocked_by 顺序链原样保留。
        """
        bucket = self._staged.get(from_parent_id)
        if not bucket:
            return
        rewritten: list[tuple[Task, list[str] | None, str | None]] = []
        for task, blocked_by, _ in bucket:
            task.parent_task_id = to_parent_id
            rewritten.append((task, blocked_by, to_parent_id))
        self._staged[from_parent_id] = rewritten
        logger.debug(
            "TaskManager.detach_staged: %d staged task(s) re-parented %s → %s",
            len(rewritten), from_parent_id, to_parent_id,
        )

    async def _flush_staged(self, task_id: str) -> None:
        """run 正常结束时把缓冲区的子任务入队。

        队列是 LIFO（pop 取末尾），因此按投入顺序 reversed 后再 push，
        使同一批次内的子任务 pop 出来是 FIFO（投入顺序）；整批位于栈顶，
        相对更早批次仍是 LIFO（深度优先）。
        """
        staged = self._staged.pop(task_id, None)
        if not staged:
            return
        if self._pause_abandon or self._cancelled:
            # pause 弃子窗口 / 硬取消（含熔断 trip 封闸）：本轮 staged 的子任务直接丢弃
            # （push 时才发 TASK_CREATED，无投影残留），防止清队后又有漏网新任务入队被派发——
            # _cancelled 分支专堵在途 run 收尾迟到的 staged 子任务变成新的搁浅任务。
            logger.info("staged tasks of %s dropped (pause_abandon=%s cancelled=%s): %d",
                        task_id, self._pause_abandon, self._cancelled, len(staged))
            return
        for task, blocked_by, parent_task_id in reversed(staged):
            await self.push_task(task, blocked_by=blocked_by, parent_task_id=parent_task_id)

    def _effective_agent(self, task: "Task | None") -> str:
        """任务实际执行所在的 agent id——同 agent 串行判定的键（单一真相见 effective_agent_id）。"""
        root = self._session.root_agent_id if self._session else ""
        return effective_agent_id(task, root)

    def _session_unhealthy(self, task_id: str) -> bool:
        """该 task 所属会话是否处于存储隔离（spec: event-commit）。无健康面时恒 False。"""
        if self._unhealthy_check is None:
            return False
        task = self._tasks.get(task_id)
        if task is None:
            return False
        return bool(self._unhealthy_check(task.session_id))

    async def drain(self) -> None:
        """Pop and run tasks until queue is empty or max_concurrent reached.

        同 agent 不并发：跳过"目标 agent 正忙（已有同 agent 任务在跑）"的队列条目，
        它们留在队列里，等该 agent 空闲（某任务完成 → on_task_finished → 再 drain）时被选中。
        """
        if self._runner is None:
            raise RuntimeError("No task runner registered")

        if self._cancelled:
            return

        while True:
            # 被同一 session 上更新的 TM 顶替（recover_agent 覆盖了 _task_managers 映射）→
            # 立即停止派发，无声（不发事件、不改状态）。避免重叠 resume 下两套 drain 并行派发
            # 同一批任务；在跑协程照旧靠 _fire_session_done 处的 _is_current 收敛（spec/07 §9）。
            if self._hooks.is_current is not None and not self._hooks.is_current():
                return
            async with self._lock:
                if len(self._running_tasks) >= self._max_concurrent:
                    break
                busy_agents = {
                    self._running_agents.get(tid) or self._effective_agent(self._tasks.get(tid))
                    for tid in self._running_tasks
                }
                entry = self._queue.pop(
                    skip=lambda e: (
                        self._effective_agent(self._tasks.get(e.task_id)) in busy_agents
                        or self._session_unhealthy(e.task_id)
                    )
                )
                if entry is None:
                    break
                self._running_tasks.add(entry.task_id)

            asyncio.create_task(self._run_task(entry.task_id))

    async def _run_task(self, task_id: str) -> None:
        assert self._runner is not None
        task = self._tasks.get(task_id)
        if task is not None:
            task.status = "ACTIVE"
            task.actor_done = False
            # 上一轮的挂起意图不得带进这一轮：父任务被子任务唤醒后重跑，若 delegate 置的
            # suspend_requested 还留着，ActStep 会再路由去 SuspendStep、父永远醒不过来。
            task.suspend_requested = False

        # ── 阶段 1：装配（assemble）────────────────────────────────────────
        # 失败发生在 TASK_STARTED 之前 → 投影不会出现幽灵 ACTIVE；与执行失败
        # 共用重试路径，但 TASK_REQUEUED.reason=assembly_failure 可区分。
        try:
            binding = await self._runner.assemble(task_id)
        except PersistenceUnavailableError:
            # spec: event-commit——存储不可用：不 commit_round（会再撞同一故障）、
            # 不走失败处置（不发终态事件——emit 同样要过提交门）、不重排。会话已被
            # CommitGate 先标记 storage_unavailable；本 run 就此停住。
            logger.error("Task %s halted: session storage unavailable", task_id)
            return
        except Exception as e:
            logger.exception("Task %s assembly failed: %s", task_id, e)
            # 装配就炸了也算数（同下方收尾兜底的理由）：先提交，再发失败事件。
            await self.commit_round(task_id)
            await self._handle_task_failure(
                task_id, error=str(e), exc=e, reason=InterruptReason.ASSEMBLY_FAILURE,
            )
            return
        if binding is None:
            # task 已不存在（装配空转）：镜像旧行为（runner 首行 get_task None 即返回）
            await self.on_task_finished(task_id, status="FINISHED")
            return

        # 回填「真正用于执行的 agent id」+ 真实启动时刻；TASK_STARTED 由 TM 发，
        # 每次派发（含 retry/resume）恰好一条（reducer 只落非空 id；见 spec/07）。
        if task is not None:
            task.assigned_agent_id = binding.agent_id
            task.started_at = now_utc()
        self._running_agents[task_id] = binding.agent_id
        await self._emit(EventType.TASK_STARTED, task_id=task_id,
                         payload={"assigned_agent_id": binding.agent_id})

        # ── 阶段 2：执行（execute）────────────────────────────────────────
        # 消费 RunOutcome 有**两条**入口：正常返回（completed / awaiting_human /
        # suspended_on_children / canceled 四种，run 自己 return 回来）与崩溃
        # （`_run_loop` 重抛 → 下面的 `except Exception` 就地构造）。两条喂**同一张**
        # 处置表，「还能不能原地重试」的判断因此只剩 disposition_for 一处。
        try:
            try:
                outcome = await self._runner.execute(binding, task_id)
            except RoundDiscarded:
                # 这一轮在 LLM 开口之前被用户中止（spec 2026-09-09）：整轮抹掉。
                #
                # **不走 `apply_run_outcome`**——那会发一条 task 状态事件，而窗口这时
                # 已经要关了，那条事件会直接落盘，等于白丢。agent 回 `idle` 由抛出点
                # 在窗口内发的那条 `TASK_CANCELED` 负责（ALM 映射成 `SETTLED`），到这里
                # 内存状态已经摆平，只剩「关窗 + 抹痕迹」。
                #
                # `discard_round` 必须是本分支的**最后一步**，理由同上：窗口一关，
                # 此后任何事件都直接落盘。
                self._staged.pop(task_id, None)
                await self.discard_round(task_id)
                return
            except BaseException:
                # cancel(CancelledError) / 异常退出：丢弃未 flush 的缓冲，
                # 防止泄漏或日后 resume 时被误入队。再交回外层原有处理。
                self._staged.pop(task_id, None)
                raise
            if outcome is None:
                # 这次执行没留下结局（task 已不存在等）→ 按「没炸、没挂起、没取消」兜底。
                # **与旧路径并不完全同形**：旧路径走 `on_task_finished(FINISHED)`，一条 task
                # 事件都不发；这里会经处置表发一条
                # `TaskFinished{"outcome": "success", "summary": "", "outputs": None}`。
                # 当前接线下本分支不可达（`_tasks` 从不删条目；`_SessionTaskRunner.execute`
                # 只在 `get_task(task_id) is None` 时返回 None，而 `_execute_task` 恒返回
                # 非 None 的 state），故不是行为差异；留着是给未来的 runner 实现兜底。
                outcome = RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="success")
            # 处置**先于** _flush_staged：子任务一入队就可能跑完、回头唤醒父亲，而
            # `_try_resume_parent` 的门是 `parent.status == "SUSPENDED"`——父亲必须在
            # 子任务入队前落到 SUSPENDED（旧路径由 control tool 在 run 内写，同一时序）。
            # 未提交窗口的收尾兜底：这一轮跑到了有结局的地步（正常收尾 / park /
            # 挂起 / 失败都算），那它就**算数**——哪怕一个 chunk 都没吐出来（outage
            # 重试耗尽、provider 永久错、装配炸掉）。补一次提交，让那条用户消息和这个
            # task 如实进日志，`/resume` 才有东西可续。
            #
            # 只有用户主动中止走 `discard_round`，那条路由 act 抛 `RoundDiscarded`
            # 触发、发生在这之前，到这里 `_rounds` 已经没有它了 → 本句 no-op。
            #
            # 位置要在 `apply_run_outcome` **之前**：那一句会发 task 状态事件，得先让
            # 窗口里攒的 `TASK_CREATED` 出去，不然状态事件会先于创建事件落盘。
            await self.commit_round(task_id)
            status = await self.apply_run_outcome(task_id, outcome)
            # runner 正常跑完才把本轮 spawn 的子任务入队（“一轮跑完之后 push”）
            await self._flush_staged(task_id)
            await self._settle(task_id, status)
        except PersistenceUnavailableError:
            # spec: event-commit——同 assemble 分支：不提交、不处置、不重排。
            # 会话隔离已由 CommitGate 标记；drain 的健康检查挡住后续派发。
            logger.error("Task %s halted: session storage unavailable", task_id)
            return
        except Exception as e:
            if getattr(e, "retriable", False):
                logger.warning("Task %s failed (retriable): %s", task_id, e)
            else:
                logger.exception("Task %s failed: %s", task_id, e)
            # 同上：崩溃也算数，先提交再发终态事件。
            await self.commit_round(task_id)
            # 崩溃入口：结局的构造在 `crash_run_outcome` 一处（retriable 的取法是崩溃
            # 专用的，与 outage 支硬编码的 False 不同源——契约见那个工厂的 docstring）。
            status = await self.apply_run_outcome(task_id, crash_run_outcome(e))
            await self._settle(task_id, status)

    async def apply_run_outcome(self, task_id: str, outcome: RunOutcome) -> str:
        """run 的结局 → task 的处置：写状态 + 发那一条 task 状态事件。**唯一入口**。

        loop 只报「发生了什么」（RunOutcome），这里回答「那么 task 变成什么」——判据
        全在 `disposition_for` 那张纯函数表里，本方法只负责执行：写 status、写回
        retry_count、发事件。返回落定的状态，供调用方选出口。公开是因为`run_single_task`
        那条 compat 路径（没有队列、不经 `_run_task`）也要消费同一份结局。
        """
        task = self._tasks.get(task_id)
        if (task is not None and outcome.kind is not RunOutcomeKind.COMPLETED
                and task.status in _TERMINAL_STATUSES):
            # 终态守卫（见 _TERMINAL_STATUSES）：熔断 trip 先把 root 判 FAILED、再对在跑
            # 的 root 发协作取消，那次取消的 CANCELED 不得盖回已写定的 FAILED。run 侧的
            # 同一守卫是 `_run_loop` 的 cancel_takes_effect（它据此决定发不发 RUN_CANCELED）。
            return task.status
        disp = disposition_for(
            outcome,
            retry_count=task.retry_count if task is not None else 0,
            max_retries=task.max_retries if task is not None else self._task_max_retries,
        )
        if task is not None:
            task.status = disp.status
            if disp.event_type == "TaskRequeued":
                # **处置表不 mutate**：新的 retry_count 只在 payload 里，必须在这里写回，
                # 否则重试预算永不消耗、同一个 task 无限重排（旧路径是先 `+= 1` 再写 payload）。
                task.retry_count = disp.payload["retry_count"]
            if outcome.kind is RunOutcomeKind.INTERRUPTED:
                # 成因随 task 走：`disp.payload` 里的 error_code 只随这一条事件走一次，
                # `get_task(task_id)` 等后续内存态查询要看的是 task 对象上的这份存档
                # （注意 TaskView 不带该字段、跨重启不还原，见总账 A9）。
                task.error = outcome.error or task.error
                if outcome.error_code:
                    task.error_code = outcome.error_code
        await self._emit(EventType(disp.event_type), task_id=task_id, payload=disp.payload)
        return disp.status

    async def _settle(self, task_id: str, status: str) -> None:
        """处置落定之后的队列收尾：挂起出口 / 重排出口 / 终态收尾，三选一。

        出口逻辑与旧路径逐字相同，只是判据从「loop 写进 task.status 的值」换成了
        「处置表算出来的状态」。
        """
        if status in _PARKED_STATUSES:
            # 子任务已由 control tool 推入队列；parent 等待所有子任务完成后
            # 由 _try_resume_parent 重新入队，此处只需移出 running set 并 drain。
            # 三种非终态停顿共用这一条出口：SUSPENDED（等子任务）、AWAITING_HUMAN
            # （HitlPark，等人应答）、INTERRUPTED（LLM 故障 / run 崩溃，待 /resume 由 restore 重排）。
            async with self._lock:
                self._release_slot(task_id)
            await self.drain()
            # 整个会话因 park/suspend 进入空闲（无在跑任务、无待派子任务）→ 通知 runtime 回收
            # 按 run 计的控制信号。注意是 is_done（而非"本 task 挂起"）：父等子时子仍在跑，
            # is_done 为 False、不触发，待子完成 resume 父；只有全会话静止才算空闲挂起。
            if self.is_done():
                await self._fire_session_idle()
            return
        if status == "PENDING":
            # observer 判 retry（本轮未完成，含机械退出）或崩溃后可重试：重新入队。
            async with self._lock:
                self._release_slot(task_id)
                self._queue.push(QueueEntry(task_id=task_id, session_id=self._session_id))
            await self.drain()
            return
        # 终态：用处置算出的那个，避免把 FAILED/CANCELED 覆盖成 FINISHED
        final_status: TaskStatus = (  # type: ignore[assignment]
            status if status in _TERMINAL_STATUSES else "FINISHED"
        )
        await self.on_task_finished(task_id, status=final_status)

    async def reopen_chain(self, head_id: str, reason: str = "") -> bool:
        """Reopen a FINISHED sub-task **and force-reopen its plan successors**.

        When a plan step is reopened, every later step in the same plan chain
        (tasks that carry `head_id` in their `tracking_task_ids`) is invalidated too,
        because they were produced against the now-rejected result. This re-queues the
        whole chain in plan order, re-establishing `blocked_by` so each step waits for
        its predecessor's corrected result (delivered via the blackboard subscription).

        - head: reopened with the direct revision `reason`.
        - successors: reopened with an "upstream revised" note instead, pointing them at
          head's updated result.

        Returns True if the head was reopened.
        """
        head = self._tasks.get(head_id)
        if head is None or head.status != "FINISHED":
            return False

        _epoch = datetime.min.replace(tzinfo=timezone.utc)
        successors = sorted(
            (
                t for t in self._tasks.values()
                if t.session_id == self._session_id
                and t.status == "FINISHED"
                and head_id in (t.tracking_task_ids or [])
            ),
            key=lambda t: as_utc(t.created_at) if t.created_at else _epoch,
        )
        head_title = head.title or head_id

        prev_id: str | None = None
        for t in [head, *successors]:
            blocked = None if prev_id is None else [prev_id]
            upstream = None if t.id == head_id else (head_title, reason)
            await self.reopen_task(t.id, reason, blocked_by=blocked, upstream=upstream)
            prev_id = t.id
        return True

    async def reopen_task(
        self,
        task_id: str,
        reason: str = "",
        *,
        blocked_by: list[str] | None = None,
        upstream: tuple[str, str] | None = None,
    ) -> bool:
        """Re-queue a previously FINISHED task (observer review 'reopen').

        Resets the task to a clean PENDING state and rewrites `user_prompt` =
        original prompt + previous output + revision reason, so the re-run's prompt
        (task_spec block + re-ingested USER_PROMPT memory) tells the actor what was
        produced before and what to fix. The rewrite is always based on
        `original_user_prompt` so repeated reopens don't accumulate. Clearing
        `outputs` re-arms the success-guardrail so a no-op cannot be re-marked
        FINISHED with stale output.

        `blocked_by` gates re-execution until the given tasks complete (used by
        reopen_chain to keep plan order). `upstream=(head_title, head_reason)` marks a
        cascade-reopened successor: instead of a direct revision note it gets an
        "upstream task revised" instruction pointing at the predecessor's updated result
        (delivered via the blackboard subscription once the predecessor re-finishes).

        Emits TASK_REQUEUED (projections/SSE + restart-safe). Scheduling happens on
        the next drain (after the current run returns); does not drain re-entrantly.
        Returns True if the task was re-queued.
        """
        task = self._tasks.get(task_id)
        if task is None or task.status != "FINISHED":
            return False

        prompt = build_reopen_prompt(task, reason, upstream)

        async with self._lock:
            self._queue.unmark_completed(task_id)
            self._queue.unmark_running(task_id)
            task.status = "PENDING"
            task.actor_done = False
            task.retry_count = 0
            task.outputs = None
            task.finished_at = None
            task.user_prompt = prompt.user_prompt
            # 无条件写回（幂等）：非首次 reopen 时 build_reopen_prompt 返回的就是
            # task 上已有的那份 base，不会把快照冲掉。
            task.original_user_prompt = prompt.original_user_prompt
            task.original_user_prompt_event_jsonable = (
                prompt.original_user_prompt_event_jsonable
            )
            task.user_prompt_event_jsonable = prompt.user_prompt_event_jsonable
            task.user_prompt_in_memory = False  # let the driver re-ingest the revised prompt
            if blocked_by is not None:
                task.dag_deps = list(blocked_by)  # restart 时由 dag_deps 重建依赖链
                # spec: task-handoff——条件存活：链序不变（同 plan 后继按原顺序重排），
                # 沿用任务原有的条件声明；存量任务本就无条件（None）→ 继续按 any。
                old_conds = task.dep_conditions or {}
                task.dep_conditions = (
                    {dep: old_conds[dep] for dep in blocked_by if dep in old_conds} or None
                )
            any_deps, success_deps = split_deps(task.dag_deps, task.dep_conditions)
            self._queue.push(QueueEntry(
                task_id=task_id, session_id=self._session_id, priority=task.priority,
                blocked_by=any_deps, blocked_success=success_deps,
            ))
        # 把改写后的 prompt 一并落进事件，使崩溃恢复（event replay）能重建修订后的
        # user_prompt。reopen 只在 prompt 尾部追加**文本** section，不可能引入事件流
        # 没见过的图。故事件形态直接由首次发射那份 + 文本拼出，零 blob IO，且同一张图
        # 的 event ref 跨 reopen 逐字节相同（重放确定性）。
        await self._emit(
            EventType.TASK_REQUEUED,
            task_id=task_id,
            payload={
                "reason": "observer_review_reopen",
                "user_prompt": prompt.user_prompt_event_jsonable,
                "original_user_prompt": prompt.original_user_prompt_event_jsonable,
            },
        )
        logger.info("TaskManager.reopen_task: re-queued %s", task_id)
        return True

    async def _handle_task_failure(
        self, task_id: str, *, reason: str, error: str = "",
        exc: BaseException | None = None,
    ) -> None:
        """**装配失败**专用：先尝试自动 retry，耗尽或不可重试 → 挂起等恢复（绝不落终态 FAILED）。

        Task 4 起，**执行**（execute）崩溃不再走这里：它由 `_run_task` 的 `except
        Exception` 就地构造 `RunOutcome(INTERRUPTED, reason=InterruptReason.RUN_CRASH)`，喂
        `disposition_for` 那张表（重试判断因此只剩一处）。本方法保留是因为装配阶段
        （assemble）没有 run、也就没有 RunOutcome，且它的 TASK_REQUEUED
        `reason=InterruptReason.ASSEMBLY_FAILURE` 是对外可区分的契约。`reason` 因此改为
        必传：旧的默认值 `"run_failure_retry"` 随执行崩溃那条路径一起退场，留着只会是个
        再也不会出现的字面量。


        运行层崩溃（异常退出，未经 observer/FinalizeStep）是**可恢复中断**，不是任务失败：
        真失败只有 observer 判 fail 一条路（FinalizeStep 闭合胶囊、发 TaskFailed、回传父亲）。
        exc.retriable=False（如 LLMCallError 401 认证失败、CONTEXT_OVERFLOW）时跳过重试直接挂起，
        避免对确定性错误做无效重试。
        """
        task = self._tasks.get(task_id)
        if task is not None and task.status in _TERMINAL_STATUSES:
            # 终态不复活（见 apply_run_outcome 的同源守卫）：熔断 trip 可能已把这个
            # task 判 FAILED，此后的装配失败属内部清场，不该把写定的终态盖回非终态、
            # 也不再发任何事件。
            #
            # 但队列簿记的清场不能省：本方法只在 `_run_task` 的 assemble 异常分支被
            # 调用，而 `_run_task` 是 `drain()` 把 task_id 加进 `_running_tasks`（并
            # mark_running 进队列）之后才 `asyncio.create_task` 出来的，所以此刻
            # task_id 必然还占着一个 running 槽位——不摘掉它会话就会永久少一个可派
            # 发的并发槽位（tests/unit/test_terminal_guard_on_assembly_failure.py 的
            # test_terminal_guard_still_clears_running_bookkeeping 测的正是这个）。
            # drain() 之后让别的排队任务照常派发；trip 场景下 `_cancelled` 已置位，
            # drain() 本就空转，调用是安全的。
            #
            # 到此为止——不再跟 `_suspend_task_interrupted` 尾段一样调 is_done() /
            # `_fire_session_idle()`。这个 task 走到本分支时已经是终态：它的终态要么
            # 是熔断 trip 判的（`_trip_failure_threshold` 第 8 步已经同步跑完
            # `_fire_session_done()`，会话早已收尾），要么是 observer 判 fail 经
            # `on_task_finished` 判的（那条路自己就会在同一次调用里做完 is_done() /
            # 收尾判断）。两条路都已经把「会话是不是完了」回答过一次；此处再报一次
            # "idle"（暗示"挂起等恢复"）对一个已经终态收场的会话是错的信号，且
            # `_on_session_idle` 回调不是幂等收尾专用的 `_on_session_done`，不该在
            # 这里替它多按一次。
            async with self._lock:
                self._release_slot(task_id)
            await self.drain()
            return
        # 不可重试的错误（如认证失败 / 上下文溢出），不重试、直接挂起等恢复
        if exc is not None and not getattr(exc, "retriable", True):
            logger.warning(
                "Task %s non-retriable error (%s), suspending for recovery: %s",
                task_id, type(exc).__name__, error,
            )
            await self._suspend_task_interrupted(task_id, error, exc, reason=reason)
            return

        if task is not None and task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = "PENDING"
            task.error = error
            logger.info(
                "Task %s retrying (%d/%d): %s",
                task_id, task.retry_count, task.max_retries, error,
            )
            async with self._lock:
                self._release_slot(task_id)
                entry = QueueEntry(task_id=task_id, session_id=self._session_id)
                self._queue.push(entry)
            # 重排落事件：使投影从 ACTIVE 回到 PENDING；进程在重试间隙崩溃时
            # restore 据 PENDING 重排（而非把停留 ACTIVE 的任务误当成可恢复后重跑）。
            await self._emit(EventType.TASK_REQUEUED, task_id=task_id, payload={
                "reason": reason,
                "retry_count": task.retry_count,
            })
            await self.drain()
        else:
            await self._suspend_task_interrupted(task_id, error, exc, reason=reason)

    async def _suspend_task_interrupted(
        self, task_id: str, error: str, exc: BaseException | None, *, reason: str,
    ) -> None:
        """运行层崩溃的终局：挂起等 /resume，**不是失败**。

        置 INTERRUPTED（非终态）+ 发 TASK_INTERRUPTED——投影只认事件类型
        （TASK_STATUS_BY_EVENT），不发则任务停留 ACTIVE、restore 语义错位。这是一条
        **task 级事实**：TM 在 run 外面、不拥有 run（连 run_id 都拿不到），run 域的
        `RunInterrupted` 由 `runtime._run_loop` 自己发。会话/agent 怎么了也不在这里
        宣布——本方法发的这条 TASK_INTERRUPTED 由 `AgentLifecycleManager` 的五态机
        （`_INPUT_BY_EVENT`）接手，折成对应 agent 的 AGENT_INTERRUPTED（判据是事件
        类型，不是 reason 字面量）。2026-09-04（Task 12）前这里还会额外经
        `announce_queue_state` 聚合出一条会话级队列信号，现已停发、其事件类型也已于
        2026-09-05 删除（那条信号唯一的消费者早已降格）。
        **发射时机是重试判定之后**：`_handle_task_failure` 决定原地重试的那一支发的是
        TASK_REQUEUED（→ PENDING），只有落到本方法才是「停在 INTERRUPTED 等 /resume」。
        不发 TASK_FAILED、不增 failure_counter、不闭合胶囊：真失败只有 observer 判 fail
        一条路。恢复由 /resume → restore() 据非终态重排（重排时 retry_count 归零）。
        """
        task = self._tasks.get(task_id)
        error_code = crash_error_code(exc)
        if task is not None:
            task.status = "INTERRUPTED"
            task.error = error
            task.error_code = error_code
        async with self._lock:
            self._release_slot(task_id)
        # reason 只作**溯源**，不作路由——判据是 TASK_INTERRUPTED 这个类型本身。
        # 值本身是对外契约的一部分（host 升级须知的映射表按 reason 分流展示文案），
        # 故按调用方透传的 `reason` 原样发出，不再在这里硬编码单一来源——
        # `_handle_task_failure` 现在服务装配失败（ASSEMBLY_FAILURE）一条路，
        # 执行崩溃改走 `_run_task` 的 `except Exception`（RunOutcome 路径），两者的
        # reason 必须同源透传，不得在挂起出口各写各的
        # （tests/unit/test_assembly_failure_reason.py 检测的正是这一点；
        # tests/unit/test_layered_signals.py 检测的是「分流」而非「出现」）。
        await self._emit(EventType.TASK_INTERRUPTED, task_id=task_id, payload={
            "reason": reason,
            "error_code": error_code,
            "error_message": error,
            "retry_count": task.retry_count if task else 0,
        })
        # 其它 agent 的排队任务照常派发；全会话静止则通知 runtime 回收 per-run 控制信号
        await self.drain()
        if self.is_done():
            await self._fire_session_idle()

    async def on_task_finished(self, task_id: str, status: TaskStatus) -> None:
        # task 落终态 → 通知外部回收挂在 task 上的运行期状态（当前是 CapabilityCache 的 pin）。
        # **这里就是「终态」那条路**：`_settle` 判 retry（status == "PENDING"）时先 return，
        # 根本走不到这个函数，所以重试自动保住 pin，不需要额外判据。best-effort：清理失败
        # 不该拦住任务收尾。
        if self._hooks.on_task_terminal is not None:
            try:
                self._hooks.on_task_terminal(task_id)
            except Exception:
                logger.exception("TaskManager: on_task_terminal callback failed for %s", task_id)

        async with self._lock:
            self._clear_running(task_id)
            if status == "FAILED":
                self._queue.mark_failed(task_id)
            elif status == "CANCELED":
                # spec: task-handoff——取消也是「终态但非成功」：只释放 on_any 依赖，
                # on_success 后继保持阻塞（其善后由永久阻塞扫描处置）。对存量
                # （全部 any 依赖）行为不变——两种收尾都进 _completed。
                self._queue.mark_failed(task_id)
            else:
                self._queue.mark_complete(task_id)

            # Update task object
            task = self._tasks.get(task_id)
            if task:
                task.status = status
                task.finished_at = now_utc()

        # ── 取消胶囊闭合 funnel（Task 14）───────────────────────────────────────
        # 在途协作取消的任务：发信号时（cancel_all 的 CancelToken / 熔断的 cancel_inflight）
        # 只做了 ack 替换（幂等自愈），finish 对要等这里——终态真正坐实（`apply_run_outcome`
        # 据处置表把 task.status 写成 CANCELED；Task 4 之前是 _run_loop 自己写）——才补写。
        # 正常收尾的任务走 FinalizeStep，永不落到这个分支
        # （status 只会是 FINISHED/FAILED，与本 if 互斥）。未真正 start 过的任务（started_at 为
        # 空）不会有派发框/own scope 可闭，交由 synthesize_cancel_closure 的 find-only 兜底判定
        # 即可，这里额外用 started_at 提前短路只是省一次无意义调用。
        if status == "CANCELED" and task is not None and task.started_at \
                and self._hooks.cancel_finalizer is not None:
            try:
                await self._hooks.cancel_finalizer([task], task.error or "cancelled")
            except Exception:
                logger.exception("TaskManager: cancel_finalizer callback failed for %s", task_id)

        # ── failure_counter 维护 ──────────────────────────────────────────────
        if self._session is not None:
            if status == "FAILED":
                self._session.failure_counter += 1
                self._recent_failures.append((
                    (task.title if task and task.title else task_id),
                    ((task.error or task.process_report or "") if task else "")[:200],
                ))
                threshold = self._session.failure_threshold
                if (
                    threshold > 0
                    and self._session.failure_counter >= threshold
                    and not self._threshold_tripped
                ):
                    await self._trip_failure_threshold()
                    return
            elif status == "FINISHED":
                self._session.failure_counter = 0  # 成功时重置
                self._recent_failures.clear()  # 随 counter 同步清空
            elif status == "CANCELED":
                # 用户主动中断：标记 session 为 CANCELED，防止 is_done() 误判为 SUCCEEDED。
                # pause 弃子（_pause_abandon）除外：连带取消不定会话去向，由 root park 决定。
                # _threshold_tripped 除外：熔断已判会话 FAILED，在途任务迟到的协作取消收尾
                # 绝不能把终态盖回 CANCELED（trip 序列已先手，见 _trip_failure_threshold）。
                # spec: task-handoff——依赖阻塞取消除外：那是计划内部的失败传播，不是
                # 用户叫停；子任务被阻塞取消后父任务照常唤醒、观察并决定补救/收尾，
                # 会话终态交给正常收敛路径（同 _pause_abandon 的豁免形态）。
                if (
                    not self._pause_abandon
                    and not self._threshold_tripped
                    and not (task is not None and task.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP)
                ):
                    self._session.status = "CANCELED"

        # spec: task-handoff——永久阻塞善后的运行期入口：前序 FAILED/CANCELED 落定后，
        # on_success 后继若已判明永不可满足 → 立即处置（级联到不动点）。
        await self.dispose_blocked_dependents()

        # Try to resume parent
        await self._try_resume_parent(task_id)
        # Drain next
        await self.drain()

        # 若 queue 已空且无任务在运行，通知 session 真正结束
        # （有重试时 drain() 会把重试任务入队，is_done() 为 False，不触发）
        if self.is_done():
            # 被顶替旧 TM 的迟到收尾不得代表会话发信号——新 owner 的状态才是真相。
            # runtime 侧回调本就 compare-and-check，这里连事件也一并静默，避免污染
            # 事件流的 host 显示与重放。
            if self._hooks.is_current is not None and not self._hooks.is_current():
                return
            # 「还有人在等」不再靠注入的 pending-HITL 谓词判断，而是由 AWAITING_HUMAN 的
            # 任务表达；「断了」由 INTERRUPTED 的任务表达。这里只区分「要不要走终结
            # 收尾」——后者要 gather 后台协程 + 回调（2026-09-04 Task 12 起两者都不再
            # 额外经 `announce_queue_state` 聚合成会话级信号，那条信号已停发）。
            if self._blocked_or_interrupted():
                await self._fire_session_idle()
            else:
                if self._session is not None and self._session.status not in ("FAILED", "CANCELED"):
                    # failure_counter > 0 表示本轮有任务失败（成功时会被重置为 0）
                    self._session.status = self._final_status()
                await self._fire_session_done()

    def _find_blocked_forever(self) -> "tuple[str, str] | None":
        """定位一个 on_success 依赖已判明永不可满足的排队任务 → (task_id, 阻塞源 dep_id)。

        判据是**当前状态**而非回调：任何时刻某 success 依赖已 FAILED/CANCELED，
        该条目就永不可能被释放（没有再让它成功的路径——reopen 会重建条件，那是
        观察者的决定，不是调度的）。幂等：已终态任务跳过。
        """
        for entry in self._queue.peek_all():
            task = self._tasks.get(entry.task_id)
            if task is None or task.status in TERMINAL_TASK_STATUSES:
                continue
            for dep in entry.blocked_success:
                dep_task = self._tasks.get(dep)
                if dep_task is not None and dep_task.status in ("FAILED", "CANCELED"):
                    return entry.task_id, dep
        return None

    async def dispose_blocked_dependents(self) -> None:
        """spec: task-handoff——永久阻塞善后（运行期 + 恢复期共用，幂等）。

        定点迭代至不动点：处置一个被阻塞取消的任务后它自己也成了 CANCELED 终态，
        依赖它的 on_success 后继随之满足处置条件，重扫直到无新受害者。每个受害者：
        落 CANCELED + ``BLOCKED_BY_FAILED_DEP`` + 指明阻塞源，发 TASK_CANCELED
        （payload 带 error_code 与 blocked_by_task_id，投影据此可恢复解释），出队，
        然后走 on_task_finished 的正规收尾（队列记账 / 父任务唤醒 / drain / 会话判定
        ——其中会话终态改写被 error_code 豁免，见该分支注释）。

        崩溃窗口兜底：A 的 FAILED 已落盘、B 的级联取消未落盘时进程崩溃，恢复重建
        依赖后由 runtime 在首次 drain 前调本方法补扫（同一实现、同一幂等）。
        """
        if self._in_blocked_dispose:
            return  # 处置内部的 on_task_finished 再入；外层循环已覆盖其级联
        self._in_blocked_dispose = True
        try:
            while True:
                found = self._find_blocked_forever()
                if found is None:
                    return
                task_id, dep_id = found
                task = self._tasks.get(task_id)
                if task is None:
                    continue
                task.status = "CANCELED"
                task.error_code = TaskErrorCode.BLOCKED_BY_FAILED_DEP
                task.error = f"blocked by failed/canceled predecessor {dep_id}"
                task.finished_at = now_utc()
                self._queue.cancel(task_id)
                await self._emit(
                    EventType.TASK_CANCELED, task_id=task_id,
                    payload={
                        "error_code": TaskErrorCode.BLOCKED_BY_FAILED_DEP,
                        "blocked_by_task_id": dep_id,
                        "reason": f"dependency_failed:{dep_id}",
                    },
                )
                await self.on_task_finished(task_id, status="CANCELED")
        finally:
            self._in_blocked_dispose = False

    async def _trip_failure_threshold(self) -> None:
        """连败达阈值：真终结——清场 + root 判死 + 会话终态，而非裸终结。

        从 on_task_finished 的 FAILED 分支进入（其 `not _threshold_tripped` 前置保证只进一次）。
        步骤对应机制设计「trip 序列」1-8；顺序是定案，改动前请对照 task-9-brief.md。
        """
        # 1) 幂等闩置位；_cancelled 封闸——drain 守卫白拿，_flush_staged 的
        #    `_pause_abandon or _cancelled` 丢弃条件也据此堵住在途 run 迟到的 staged 子任务。
        self._threshold_tripped = True
        self._cancelled = True
        threshold = self._session.failure_threshold if self._session else 0
        counter = self._session.failure_counter if self._session else 0
        logger.warning(
            "Session %s failure_counter=%d reached threshold=%d → 熔断真终结",
            self._session_id, counter, threshold,
        )

        # 2) FAILURE_THRESHOLD_HIT，payload 带本轮已知连败清单
        await self._emit(EventType.FAILURE_THRESHOLD_HIT, payload={
            "failure_counter": counter,
            "threshold": threshold,
            "failures": [{"title": title, "reason": reason}
                         for title, reason in self._recent_failures],
        })

        # 3) 取消该 session 所有未决 pending HITL（best-effort）：HitlCancelled 须全部
        #    先于会话终态发出，防止 host 投影翻态早于 hitl 侧收尾。
        if self._hooks.cancel_pending_hitl is not None:
            try:
                await self._hooks.cancel_pending_hitl()
            except Exception:
                logger.exception("TaskManager: cancel_pending_hitl callback failed")

        # 清场**分类**交纯函数（见 failure_threshold.py）；**顺序**留在这里。
        async with self._lock:
            pending = self._queue.drain_pending()
        plan = plan_threshold_trip(
            self._tasks, pending_ids=pending, running_ids=self._running_tasks,
        )

        # 4) 清队 + 5) 取消挂起：非 root 直接标 CANCELED + 发事件。
        #    发事件的顺序仍是「先清队条目、后挂起条目」，与改造前一致。
        for tid in (*plan.cancel_queued, *plan.cancel_suspended):
            t = self._tasks[tid]
            t.status = "CANCELED"
            t.finished_at = now_utc()
            await self._emit(
                EventType.TASK_CANCELED, task_id=tid,
                payload={"reason": CancelReason.FAILURE_THRESHOLD},
            )

        # 5b) 在途非 root：只发协作取消信号，不发事件（其 TASK_CANCELED 由 run 结束后的
        #     `apply_run_outcome` 发；finish 对交由 on_task_finished 的取消胶囊闭合 funnel；
        #     已启动带框者进 plan.ack_task_ids，供 threshold_finalizer 做 eager ack 替换）。
        for tid in plan.signal_inflight:
            self._signal_cancel(tid)

        # 5.5) 立即整对闭合已终态的取消任务（清队 + 挂起，均已启动）——它已经是终态，
        #      没有后续 on_task_finished 会来补 finish 对。在途任务保持 eager ack + funnel。
        if plan.cancel_now_ids and self._hooks.cancel_finalizer is not None:
            try:
                await self._hooks.cancel_finalizer(
                    [self._tasks[tid] for tid in plan.cancel_now_ids],
                    CancelReason.FAILURE_THRESHOLD,
                )
            except Exception:
                logger.exception(
                    "TaskManager: cancel_finalizer callback failed (threshold cleanup)")

        # 6) root 判 FAILED。**先标 FAILED 再**对在跑的 root 发 cancel_inflight——顺序
        #    保证两道守卫都接得住：task 侧是 `apply_run_outcome` 的终态守卫（不把 FAILED
        #    盖回 CANCELED）；run 侧是 `_run_loop` 的 cancel_takes_effect（不发 RUN_CANCELED）。
        root_we_failed_and_started: Task | None = None
        for tid in plan.fail_roots:
            t = self._tasks[tid]
            t.status = "FAILED"
            t.error_code = TaskErrorCode.BY_THRESHOLD
            t.error = (f"Session failure threshold reached "
                       f"({counter} consecutive sub-task failures).")
            t.finished_at = now_utc()
            await self._emit(EventType.TASK_FAILED, task_id=tid, payload={
                "error_code": TaskErrorCode.BY_THRESHOLD,
                "error_message": t.error,
            })
            if t.started_at is not None:
                root_we_failed_and_started = t
        for tid in plan.signal_roots:
            self._signal_cancel(tid)

        # 7) finalizer：内联 await（不是后台甩），保证 memory 落盘先于 SESSION_FINISHED
        #    （SSE 关闭）；异常只记日志不阻断终结。
        if self._hooks.threshold_finalizer is not None:
            try:
                await self._hooks.threshold_finalizer(
                    root_we_failed_and_started,
                    [self._tasks[tid] for tid in plan.ack_task_ids],
                    list(self._recent_failures),
                )
            except Exception:
                logger.exception("TaskManager: threshold_finalizer callback failed")

        # 8) 会话终态 + 收尾事件。终态直接写定为 FAILED（2026-09-04 Task 12 起不再
        #    经由队列聚合信号/会话状态机——熔断的前提就是 failure_counter 已达阈值，
        #    `_final_status()` 此刻本就恒为 FAILED，无需绕一圈再落定）。
        if self._session is not None:
            self._session.status = "FAILED"
        await self._fire_session_done()

    async def cancel_all(self, *, reason: str = "") -> None:
        """硬取消整条 session 链：清空 pending 队列并标 CANCELED，会话置 CANCELED。

        在途 task 不在此处理——由 CancelToken → act checkpoint → CancelledError →
        `_run_loop` 交回 `RunOutcome(CANCELED)` → `apply_run_outcome` 置该 task CANCELED
        → on_task_finished（其 drain() 被 _cancelled 守卫挡住，
        finish 对经 on_task_finished 的取消胶囊闭合 funnel 补写，见 Task 14）。

        标态后立即对已启动的任务（`started_at` 非空——含 root：root 没有 origin_tool_call_id，
        但 synthesize_cancel_closure 的 own-root 形态不需要框，仍要闭合自己的 scope）经
        `_cancel_finalizer` 闭合胶囊（ack 终态化 + `[outcome=cancelled]` finish 对）。未启动
        的任务从未铸框/写过任何 memory，跳过——零 memory 写。
        """
        self._cancelled = True
        async with self._lock:
            pending = self._queue.drain_pending()
        to_close: list[Task] = []
        for tid in pending:
            t = self._tasks.get(tid)
            if t is not None:
                t.status = "CANCELED"
                t.finished_at = now_utc()
                if t.started_at:
                    to_close.append(t)
            await self._emit(EventType.TASK_CANCELED, task_id=tid, payload={"reason": reason})
        if to_close and self._hooks.cancel_finalizer is not None:
            try:
                await self._hooks.cancel_finalizer(to_close, reason or CancelReason.USER_CANCEL)
            except Exception:
                logger.exception("TaskManager: cancel_finalizer callback failed (cancel_all)")
        if self._session is not None:
            self._session.status = "CANCELED"

    def _signal_cancel(self, task_id: str) -> None:
        """对在途 task 发协作取消信号（best-effort：缺注入或异常都不阻断 trip 序列）。"""
        if self._hooks.cancel_inflight is None:
            return
        try:
            self._hooks.cancel_inflight(task_id)
        except Exception:
            logger.exception("TaskManager: cancel_inflight callback failed for %s", task_id)

    def _clear_running(self, task_id: str) -> None:
        """从「在跑」登记里摘掉这个 task。**调用方须已持 `self._lock`。**"""
        self._running_tasks.discard(task_id)
        self._running_agents.pop(task_id, None)

    def _release_slot(self, task_id: str) -> None:
        """归还一个派发槽位：清在跑登记 + 解除队列的 running 标记。

        **调用方须已持 `self._lock`。** 五个非终态出口（park / 重排 / 装配失败的终态
        守卫 / 装配失败的重试 / INTERRUPTED 挂起）逐字共用这三行——不摘干净会话就永久
        少一个并发槽位。终态出口（`on_task_finished`）不走这里：它的队列侧动作是
        `mark_complete` / `mark_failed`（两者内部已 `_running.discard`），只需
        `_clear_running`。
        """
        self._clear_running(task_id)
        self._queue.unmark_running(task_id)

    def _agent_id_of(self, task_id: str) -> str | None:
        """先看正在跑的登记，再回落到 task 自己的 assigned_agent_id。"""
        running = self._running_agents.get(task_id)
        if running:
            return running
        task = self._tasks.get(task_id)
        return task.assigned_agent_id if task is not None else None

    async def _emit(
        self,
        event_type: EventType,
        task_id: str | None = None,
        payload: dict | None = None,
        *,
        agent_id: str | None = None,
    ) -> None:
        """构造并发出一个 session/task 级别的事件（无 LoopState）。

        V2 §0：envelope 管身份。``agent_id`` 未显式传且有 ``task_id`` 时自动解析
        （先查在跑登记 ``_running_agents``，查不到再回落到 ``Task.assigned_agent_id``）；
        无 ``task_id``（如 ``FAILURE_THRESHOLD_HIT`` 这类会话级聚合信号）则 agent_id
        保持 None，不乱填。
        """
        await emit_event(
            self._event_bus,
            event_type,
            session_id=self._session_id,
            tenant_id=self._session.tenant_id if self._session else "default",
            origin=_ORIGIN,
            task_id=task_id,
            agent_id=(
                agent_id if agent_id is not None
                else (self._agent_id_of(task_id) if task_id is not None else None)
            ),
            payload=payload,
        )

    def _final_status(self) -> str:
        """全部终态时的会话结论。failure_counter > 0 表示本轮有任务失败。

        2026-09-04（Task 12）起 `announce_queue_state` 及其队列聚合事件已停发（事件
        类型于 2026-09-05 删除），本方法不再是谁的数据源——唯一读者是本文件
        `_settle` 里的 ``self._session.status = self._final_status()``（终态收尾
        直接写 session 状态，不再经由已退役的会话状态机）。**只返回终态值**：
        空串或别的东西会把 `_session.status` 写成一个不存在的状态。
        """
        if self._session is not None and self._session.failure_counter > 0:
            return "FAILED"
        return "SUCCEEDED"

    def _blocked_or_interrupted(self) -> bool:
        """还有非终态停顿的任务在等人/等 /resume——会话不算走完。"""
        return any(t.status in _PARKED_STATUSES for t in self._tasks.values())

    async def _fire_session_idle(self) -> None:
        """会话空闲挂起（park/suspend，非终结）：通知 runtime 回收 per-run 控制信号。

        2026-09-04（Task 12）起不再先报队列状态——`announce_queue_state` 已停发
        （其消费者会话状态机早已降格，见 events-v2 §5），本方法只剩通知这一件事。
        """
        if self._hooks.on_session_idle is not None:
            try:
                await self._hooks.on_session_idle()
            except Exception:
                logger.exception("TaskManager: session_idle callback failed")

    async def _fire_session_done(self) -> None:
        """会话可能已经走完：先等后台协程，再走可选回调。

        2026-09-04（Task 12）起不再报队列状态——`announce_queue_state` 已停发
        （events-v2 §5），幂等因此全靠 `_on_session_done` 侧的回收本身幂等。
        """
        # 等待所有后台协程完成，确保 RecognizeIntent 等事件全部 emit 后再关闭 SSE 流
        if self._background_asyncio_tasks:
            await asyncio.gather(*list(self._background_asyncio_tasks), return_exceptions=True)
        # 归属权判定放在 gather **之后**：顶替可能发生在等待后台任务期间（旧 TM 的
        # background observe 拖久了，用户已开启下一轮、新 TM 接管了 session）。此时本 TM
        # 已非 owner → 收尾变 no-op，绝不发 SessionFinished、绝不触发 on_session_done，
        # 否则会冲掉新一轮的 HITL 挂起态、把任务卡在 ACTIVE。
        if self._hooks.is_current is not None and not self._hooks.is_current():
            logger.info("TaskManager(%s): superseded during session-done; skip callback",
                        self._session_id)
            return
        if self._hooks.on_session_done is not None:
            try:
                await self._hooks.on_session_done()
            except Exception:
                logger.exception("TaskManager: session_done callback failed")

    async def _try_resume_parent(self, finished_task_id: str) -> None:
        parent_id = self._parent_map.get(finished_task_id)
        if parent_id is None:
            return

        # 判定 all_done → 翻转 SUSPENDED→PENDING → push 三步必须在同一临界区内完成：
        # 否则两个（同 agent）子任务并发完成时会各自读到 all_done=True + status==SUSPENDED，
        # 双双 push/resume 父任务（父在同一 scope 上并发跑两遍，污染 memory）。drain 留到锁外。
        #
        # PENDING 不是 ACTIVE（D5）：ACTIVE 的正主是 TASK_STARTED（TM 派发时发、回填
        # assigned_agent_id）。这里只是「解除阻塞、入队」，真正开始跑要等 drain 派发到它。
        # 此前置 ACTIVE 是入队前的抢跑，投影因此在派发前的窗口就误报「在跑」。
        resumed = False
        async with self._lock:
            siblings = self._children_of.get(parent_id, set())
            # 空集守卫：无已登记子任务时绝不 resume（all([]) 恒为 True 的 vacuous-truth 防御）。
            all_done = bool(siblings) and all(
                (self._tasks[tid].status if tid in self._tasks else "PENDING")
                in TERMINAL_TASK_STATUSES
                for tid in siblings
            )
            if all_done:
                parent_task = self._tasks.get(parent_id)
                if parent_task and parent_task.status == "SUSPENDED":
                    parent_task.status = "PENDING"
                    self._queue.push(QueueEntry(
                        task_id=parent_id,
                        session_id=self._session_id,
                    ))
                    resumed = True

        if resumed:
            logger.info("All children of %s done, resuming parent", parent_id)
            # 先发事件、再 drain（评审 Important）：drain() 派发走 asyncio.create_task，
            # 不等子协程跑完就把控制权交还——旧顺序「drain 在前」曾经无害，是因为
            # TaskResumed/TaskStarted 都折叠成 ACTIVE，谁先谁后结果一样。D5 把
            # TaskResumed 改成 PENDING 后，若 TaskStarted（→ACTIVE）先落进事件流，
            # 重放会把一个真正在跑的任务钉成 PENDING。「解除阻塞」先于「开始执行」
            # ——这正是本次改动的核心语义，事件顺序也要跟着这条排。
            await self._emit(EventType.TASK_RESUMED, task_id=parent_id)
            await self.drain()

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def children_of(self, task_id: str) -> set[str]:
        """该 task 已派生的子任务 id 集合（用于 blackboard 订阅）。"""
        return set(self._children_of.get(task_id, set()))

    def all_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def is_done(self) -> bool:
        """True when queue is empty and nothing is running."""
        return not self._queue.has_pending() and not self._running_tasks

    async def finalize_idle_session(self, status: str) -> None:
        """恢复专用：会话所有 task 已终态但 session 因崩溃未落终态 —— 设终态并复用
        _fire_session_done（先 gather 重跑的后台 recap，再发 SESSION_FINISHED + 回调）。

        镜像 on_task_finished 的会话收尾：走同一条 `_fire_session_done` 链，终态由调用方
        传入的 `status` 直接写定（2026-09-04 Task 12 起不再经由队列聚合信号/会话状态机）。幂等由 `_fire_session_done` 内的归属权判定 + `_on_session_done`
        自身幂等承担。
        """
        if self._session is not None:
            self._session.status = status
        await self._fire_session_done()

    async def resume_task(self, task_id: str, *, hitl_id: str) -> None:
        """重排一个被 HITL 应答唤醒的 task：置 PENDING 并入队，供**复用活 owner**的就地续跑路径。

        不重建 TM——直接把该 task 塞回本 owner 的队列。已终结/在跑/已在队列的任务不重复入队。
        wait_for_user 冷应答已由 `_inject_user_reply` 置 PENDING（并自行发 `TaskHumanResolved`，
        见其 docstring）；approval 走此路径重排后由 reconcile 重放 dangling tool_call——这里
        才是 approval 分支唯一发 `TaskHumanResolved` 的地方，与 `TaskAwaitingHuman{hitl_id}`
        配对（D4）。

        **只在真的解除了「被人挡住」时才发那条事件**（`was_blocked` 判据）：`_inject_user_reply`
        走 wait_for_user 分支时已经委托 `mark_human_resolved` 把状态改成 PENDING 并发过一次；若这里不做这个判断，
        紧随其后的这次调用会对同一个 hitl_id 重复发 `TaskHumanResolved`——配对就不再是一对一。
        """
        t = self._tasks.get(task_id)
        if t is None or t.status in TERMINAL_TASK_STATUSES:
            return
        if task_id in self._running_tasks:
            return
        if any(e.task_id == task_id for e in self._queue.peek_all()):
            return
        was_blocked = t.status in ("AWAITING_HUMAN", "SUSPENDED")
        t.status = "PENDING"
        t.retry_count = 0  # 挂起期间的旧计数不带入新一轮 attempt
        self._queue.push(QueueEntry(
            task_id=task_id, session_id=self._session_id, priority=t.priority,
        ))
        if was_blocked:
            await self._emit(EventType.TASK_HUMAN_RESOLVED, task_id=task_id,
                              payload={"hitl_id": hitl_id})

    async def requeue_for_message(self, task_id: str, *, reason: str = "send_message") -> bool:
        """把一个被**外部消息**（`CtxWeftRuntime.send_message`，Task 18）重新激活的
        task 放回队列——发 `TaskRequeued`，**不是** `TaskHumanResolved`。

        `TaskHumanResolved` 是 `TaskAwaitingHuman{hitl_id}` 的**一对一配对解除事件**
        （见 `resume_task` docstring："唯一发 TaskHumanResolved 的地方…配对就不再是
        一对一"；`reducers.py` 同一断言）——它只属于 HITL 应答路径。外部消息注入
        没有对应的 `TaskAwaitingHuman`，硬发它会在事件流里留一个配不上对的孤儿，还
        会经 ALM（`_INPUT_BY_EVENT` 把它译成 `AgentInput.HUMAN_RESOLVED`）把 agent
        状态**提前**翻成 `running`——task 这时只是入了队，真正的 `TASK_STARTED`
        要等 `drain()` 真派发才发，这个窗口里 agent 报 running 但其实没在跑，会让
        紧接着的下一次 `send_message` 被 `assert_can_receive` 误判成 `AgentBusyError`
        拒收。

        `TASK_REQUEUED` 才是语义对的事实：`reducers.py` 自己的注释说它和
        `TaskHumanResolved` "效果相同（判据是类型不是 payload）"，区别只在后者多背
        了一层 HITL 配对；ALM 把它译成 `AgentInput.SETTLED` → agent 回 `idle`，
        正确反映"已入队、尚未开跑"，`drain()` 真正派发时 `TASK_STARTED` 才会把它
        翻成 `running`——时序对得上。

        与 `resume_task` 同样的三道幂等闸（已终态 / 已在跑 / 已在队列 -> no-op，
        不重复入队不重发事实）；不看 `was_blocked`——不管当前是哪种非终态挡着，
        只要真的把它塞回了队列就发一次。返回是否真的发生了重排（供调用方决定要不要
        补一次 `drain()`——本方法本身不 drain，与 `resume_task` 同一分工，drain
        交给调用方，见 `_resume_in_existing_tm` / `_try_resume_parent` 的既有先例）。
        """
        t = self._tasks.get(task_id)
        if t is None or t.status in _TERMINAL_STATUSES:
            return False
        if task_id in self._running_tasks:
            return False
        if any(e.task_id == task_id for e in self._queue.peek_all()):
            return False
        t.status = "PENDING"
        t.retry_count = 0  # 挂起期间的旧计数不带入新一轮 attempt
        self._queue.push(QueueEntry(
            task_id=task_id, session_id=self._session_id, priority=t.priority,
        ))
        await self._emit(EventType.TASK_REQUEUED, task_id=task_id, payload={"reason": reason})
        return True

    async def mark_human_resolved(self, task_id: str, *, hitl_id: str) -> None:
        """HITL 应答落地后把 task 置回 PENDING 并发事实——**不看当前状态**。

        与 `resume_task` 的区别：后者有 `was_blocked` 门（只对
        AWAITING_HUMAN/SUSPENDED 生效），而 wait_for_user 冷应答的重建路径上
        `restore()` 先跑、已把该 task 的状态从 AWAITING_HUMAN 翻成了 PENDING，
        那道门必然落空——`_inject_user_reply` 走的就是这条入口。

        不入队（与旧行为一致）：wait_for_user 场景下重排交给调用方后续的驱动，
        本方法只落状态 + 发事实。

        终态不复活：已 FINISHED/FAILED/CANCELED 的 task 原样返回、不发事件。
        """
        task = self._tasks.get(task_id)
        if task is None or task.status in _TERMINAL_STATUSES:
            return
        task.status = "PENDING"
        await self._emit(EventType.TASK_HUMAN_RESOLVED, task_id=task_id,
                          payload={"hitl_id": hitl_id})

    def is_cancelled(self) -> bool:
        """本 TM 是否已被硬取消（cancel_all 置 _cancelled）——供派发点补投 born-cancel 判定。"""
        return self._cancelled

    def is_alive(self) -> bool:
        """本 TM 是否仍是该 session 的当前 owner。

        供 recover_agent 判断"是否有活 TM"，以决定复用还是重建、以及新 TM 要跳过哪些
        在跑任务。

        ⚠ **它只表达「有没有被顶替」，不表达「这一轮跑完没有」**（2026-09-08 生命周期
        改造后的口径收窄）。从前 `_fire_session_done` -> `_release_session` 会把终结的
        TM 从 `_task_managers` 摘掉，`is_current` 随即为 False，于是这个方法顺带兼了
        「未终结」的意思；现在 TM 常驻到显式 `forget_session`，一个已经发过
        `SessionFinished` 的 TM 在这里照样是 alive——**这是有意的**：它确实还能接新
        task（`drain` 只被 `_cancelled` 与 `is_current` 挡，`_fire_session_done` 不设
        任何闩），`_start_task_for_agent` 因此可以直接复用它，不必重建一份。
        要判断「这一轮是不是走完了」请用 `is_done()`。
        """
        return self._hooks.is_current is None or self._hooks.is_current()

    def running_task_ids(self) -> set[str]:
        """当前正在执行（已派发、_run_task 未返回）的 task id 集合。"""
        return set(self._running_tasks)

    def running_agent_of(self, task_id: str) -> str | None:
        """在跑 run 的真实执行 agent id（无此在跑任务 → None）。pause 弃子划分的真相源。"""
        return self._running_agents.get(task_id)

    def set_pause_abandon(self, flag: bool) -> None:
        """置位/复位 pause 弃子窗口标志。

        置位期间产生两个效果：① CANCELED 任务不再连带把 session 状态改成 CANCELED（会话去向留给
        root park 决定，见 _on_task_finished 里的判断）；② run 结束时本轮新 staged 出的子任务直接丢弃、
        不入队派发（见 _flush_staged），防止弃子清队后又漏网新任务被派发。
        """
        self._pause_abandon = flag

    async def abandon_pending(
        self, *, reason: str = CancelReason.PAUSE_ABANDON, keep_agent: str | None = None,
    ) -> list[str]:
        """放弃排队中任务（标 CANCELED、发 TASK_CANCELED），不触碰 session 状态。

        ``keep_agent`` 非 None **且该 agent 当前没有在途 run** 时：effective agent 等于它的
        排队条目保留在队列（相对顺序不变）、不放弃——pause 弃子须保留 root agent 已入队未派发
        的那一条（root run 尚未派发、无可 pause 的对象），交由调用方补 drain 派发成为唯一续跑点；
        否则它随全清被误取消，会话既无在途 run 又无排队条目、无气泡 → 永久滞留 RUNNING。
        keep_agent 已有在途 run（那一轮已被 pause→park 成续跑点）或为 None 时，root-scope 排队
        条目照旧全清——保证「一次暂停恰一个续跑点」，与全清行为一致。

        与 cancel_all 的差异：不置 _cancelled（弃子后仍需 drain 派发保留/重排的任务）、
        不把 session 置 CANCELED（pause 弃子不是用户取消）。
        """
        async with self._lock:
            entries = self._queue.peek_all()
            self._queue.drain_pending()
            busy_agents = set(self._running_agents.values())
            keep = keep_agent is not None and keep_agent not in busy_agents
            cancelled: list[str] = []
            for e in entries:
                if keep and self._effective_agent(self._tasks.get(e.task_id)) == keep_agent:
                    self._queue.push(e)  # 保留：相对顺序按原队列顺序回填
                else:
                    cancelled.append(e.task_id)
        for tid in cancelled:
            t = self._tasks.get(tid)
            if t is not None:
                t.status = "CANCELED"
                t.finished_at = now_utc()
            await self._emit(EventType.TASK_CANCELED, task_id=tid, payload={"reason": reason})
        # 队列弃子不经 on_task_finished，不会自动触发父任务重排：若某 SUSPENDED 父任务的
        # 子任务此刻**全部**还在排队（无一在途），无人调用 _try_resume_parent → 父任务永不
        # 重排、会话滞留 RUNNING 且无续跑点。此处对每个被弃子任务补触发重排检查（幂等：
        # 兄弟仍在途时 all_done 不成立、由其 on_task_finished 接力；父已 ACTIVE 不二次入队）。
        for tid in cancelled:
            await self._try_resume_parent(tid)
        return cancelled


@dataclass
class RoundSnapshot:
    """开窗那一刻的可回滚状态（spec 2026-09-09）。

    只装 **TaskManager 自己拥有** 的东西。memory 记录 id 挂在 `Task.user_prompt_memory_id`
    上（落库那一刻记下的），HITL 与 ALM 归 runtime —— 那两样经 `revert_round` 钩子回退。

    后三个字段是这份快照存在的主要理由：`_inject_user_turn` 会就地把 `outputs` /
    `process_report` / `process_report_at` 清空（"新消息意味着有新工作要做"），而旧值
    **没有任何别处存过**。不拍这一张，撤销就只能撤一半：消息没了，上一轮的进展也没了。
    """

    owns_task: bool
    hitl_id: str = ""
    task_status: str = ""
    retry_count: int = 0
    outputs: Any | None = None
    process_report: str | None = None
    process_report_at: "datetime | None" = None


def task_payload(task: Task, *, user_prompt_jsonable: "str | list[dict] | None") -> dict:
    """TaskCreated 事件的 payload，供 sessions.py translate_event 构建前端 task 对象。

    ``user_prompt_jsonable``：`push_task` 在 `await self._emit(...)` 之前备好的
    event-jsonable 载荷（保 ref、绝不落字节；源头是入口从原始 content 算出的那一份，
    见 `push_task` 的 docstring）——**不**在本函数内部算，因为本函数是同步的、还负责
    十余个与内容无关的字段，async 化会让所有调用方等一次 IO（spec §6）。

    刻意做成**必传的 keyword-only 参数**（无默认值）：曾经有一条「默认 None 时退回
    同步 `content_to_jsonable`」的分支，效果是把 inline base64 直接塞回 TaskCreated
    payload——今天零调用方用它（`push_task` 是唯一调用点，且必传），留着只是把「事件
    库恒不含字节」这条不变量的破口焊死在函数签名的默认参数上，静静等下一个复用本函数
    的人在没有 await 的地方顺手调用。去掉默认值后，下一个复用者会在签名处就被挡住，
    而不是无声地把字节写回事件。
    """
    import dataclasses
    settings_d = dataclasses.asdict(task.settings)
    settings_d["_type"] = type(task.settings).__name__
    ts = task.created_at.isoformat() if task.created_at else ""
    prompt = user_prompt_jsonable
    return {
        "task": {
            "id": task.id,
            "session_id": task.session_id,
            "status": task.status,
            "title": task.title or "",
            "description": task.description or "",
            "creator_agent_id": task.creator_agent_id or "",
            "assigned_agent_id": task.assigned_agent_id or "",
            "parent_task_id": task.parent_task_id or "",
            # 空 part 列表被 `or ""` 降级成 ""：今日安全，因为本仓处处把 [] 与 "" 当等价的
            # "无内容"（没有生成合法的空/短 part 列表的路径）。Phase 3 若出现这样的合法列表
            # （例如一张裁掉了文字的纯图片 prompt 被上游误判为"空"），这里就会把它错误吞掉。
            "user_prompt": prompt or "",
            "priority": task.priority,
            "max_retries": task.max_retries,
            "dag_deps": task.dag_deps,
            # spec: task-handoff——依赖条件与显式输入随创建事件落盘（None 原样落：
            # 存量读侧按「未声明」解释，不虚构）。
            "dep_conditions": task.dep_conditions,
            "inputs": task.inputs,
            "interaction_mode": task.interaction_mode,
            "unattended": task.unattended,
            "origin_tool_call_id": task.origin_tool_call_id or "",
            "origin_tool_name": task.origin_tool_name or "",
            "settings": settings_d,
            "result": None,
            "outputs": {},
            "error": None,
            "created_at": ts,
            "updated_at": ts,
        }
    }
