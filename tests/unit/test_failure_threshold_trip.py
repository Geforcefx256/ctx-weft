"""熔断真终结：连败达阈值 → 清场 + root 判死 + 会话终态（trip 序列）。

覆盖 task-9-brief.md 「机制设计（全量）」的 _trip_failure_threshold 步骤 1-8：
- 事件序：FAILURE_THRESHOLD_HIT → TASK_CANCELED*（非 root）→ TASK_FAILED(root) →
  TASK_QUEUE_DRAINED(final_status=FAILED)。Task 6 起会话终态由 SessionRegistry 据这条
  聚合信号发 SessionFinished，TM 不再自己发 SESSION_STATUS_CHANGED/SESSION_FINISHED。
- 队列/挂起任务全 CANCELED，root FAILED（error_code=TASK_FAILED_BY_THRESHOLD）。
- 幂等闩：第 4 败不重进 trip、不重发事件。
- 封闸后 _flush_staged 丢弃本轮 staged 子任务。
- CANCELED 迟到（在途协作取消收尾）不盖 FAILED。
- ack_tasks / failures 正确传给 threshold_finalizer stub。
- root 已终态（自己是第 N 败或已 FINISHED）→ 判死跳过。
- cancel_inflight 对在途 task id 被调用（非 root 与 root 分别验证）。
"""

from __future__ import annotations

import asyncio

from ctx_weft.protocols.events import EventType
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.clock import now_utc
from tests.unit._stub_runner import StubRunner


class _CapturingBus:
    async def emit_batch(self, events, *, on_committed=None):
        if on_committed is not None:
            on_committed()
        for event in events:
            await self.emit(event)

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


def _types(bus: _CapturingBus) -> list:
    return [e.type for e in bus.events]


def _tm(bus: _CapturingBus, threshold: int = 3) -> tuple[TaskManager, Session]:
    tm = TaskManager(session_id="s1", event_bus=bus)
    tm._max_concurrent = 0  # drain 空转：不真正派发
    session = Session(id="s1", user_prompt="",  failure_threshold=threshold)
    tm.set_session(session)
    tm.set_runner(StubRunner(tm))
    return tm, session


def _root(tid: str = "root", status: str = "SUSPENDED", **kw) -> Task:
    """root 默认 SUSPENDED——镜像生产中"父等子"模式（delegate 后父挂起等子完成）。

    is_done() 只看 queue/_running_tasks，不看已登记任务状态；若 root 是 ACTIVE 但既不在
    队列也不在 _running_tasks（测试里常见的"只登记不真跑"简化），子任务逐个 on_task_finished
    后 is_done() 会立即真、误触发与本测试无关的会话终态收尾分支。SUSPENDED 触发 is_done() 块里
    既有的"父等子"守卫（保持空闲而非终结），与生产语义一致。
    """
    return Task(id=tid, session_id="s1", status=status, parent_task_id=None, **kw)


def _child(tid: str, parent: str = "root", status: str = "ACTIVE", **kw) -> Task:
    return Task(id=tid, session_id="s1", status=status, parent_task_id=parent, **kw)


async def _fail_n_times(tm: TaskManager, task_ids: list[str]) -> None:
    """对给定的一串（不同）task_id 各触发一次 FAILED，驱动 failure_counter 递增到阈值。"""
    for tid in task_ids:
        await tm.on_task_finished(tid, status="FAILED")


async def test_trip_event_sequence_on_third_failure() -> None:
    """3 连败：THRESHOLD_HIT → TASK_CANCELED*（非 root）→ TASK_FAILED(root) → session.status=FAILED。

    2026-09-04（Task 12，events-v2 §5）起第 8 步不再额外发一条会话级 TaskQueueDrained
    落定终态——`_trip_failure_threshold` 直接把 `session.status` 写成 "FAILED"（trip
    序列 8 号步骤，紧接在 6 号步骤 TASK_FAILED 之后、同一次同步调用内），已经没有
    独立事件可用来做「在 TASK_FAILED 之后」的顺序断言，改成直接断言这个值。
    """
    bus = _CapturingBus()
    tm, session = _tm(bus)
    root = _root()
    tm.register_task(root)
    for tid in ("c1", "c2", "c3"):
        tm.register_task(_child(tid))

    await _fail_n_times(tm, ["c1", "c2", "c3"])

    types = _types(bus)
    assert EventType.FAILURE_THRESHOLD_HIT in types
    hit_idx = types.index(EventType.FAILURE_THRESHOLD_HIT)
    failed_idx = types.index(EventType.TASK_FAILED)
    assert hit_idx < failed_idx
    assert EventType.SESSION_STATUS_CHANGED not in types
    assert "TaskQueueDrained" not in [str(t) for t in types]
    assert not hasattr(session, "status")


async def test_queue_and_suspended_tasks_canceled_root_failed() -> None:
    """在队/挂起的非 root 任务全部 CANCELED；root（ACTIVE，非终态）判 FAILED。"""
    bus = _CapturingBus()
    tm, session = _tm(bus)
    root = _root()
    tm.register_task(root)
    tm.register_task(_child("c1"))
    tm.register_task(_child("c2"))
    queued = _child("queued1", status="PENDING")
    tm.register_task(queued)
    tm._queue.push(QueueEntry(task_id="queued1", session_id="s1"))
    suspended = _child("susp1", status="SUSPENDED")
    tm.register_task(suspended)

    await _fail_n_times(tm, ["c1", "c2"])
    await tm.on_task_finished("queued1", status="FAILED")  # 第 3 败——由某个非 root 任务触发

    assert queued.status == "CANCELED"
    assert suspended.status == "CANCELED"
    assert root.status == "FAILED"
    assert root.error_code == "TASK_FAILED_BY_THRESHOLD"
    assert not hasattr(session, "status")

    canceled_ids = {e.task_id for e in bus.events if e.type == EventType.TASK_CANCELED}
    assert "susp1" in canceled_ids
    # queued1 是触发第 3 败的任务本身：已在 on_task_finished 顶部被标 FAILED（不是 trip 序列的清队对象）；
    # trip 序列真正处理的是"它之外"仍排队/挂起的非 root 条目。


async def test_idempotent_no_retrip_on_fourth_failure() -> None:
    """幂等闩：trip 之后再有非 root 任务失败，不重发 THRESHOLD_HIT / 不重复判 root。"""
    bus = _CapturingBus()
    tm, session = _tm(bus)
    root = _root()
    tm.register_task(root)
    for tid in ("c1", "c2", "c3", "c4"):
        tm.register_task(_child(tid))

    await _fail_n_times(tm, ["c1", "c2", "c3"])
    hit_count_before = _types(bus).count(EventType.FAILURE_THRESHOLD_HIT)
    failed_count_before = _types(bus).count(EventType.TASK_FAILED)

    # 第 4 败：迟到的在途协作取消收尾也可能走到这里；不应重进 trip
    await tm.on_task_finished("c4", status="FAILED")

    assert _types(bus).count(EventType.FAILURE_THRESHOLD_HIT) == hit_count_before
    assert _types(bus).count(EventType.TASK_FAILED) == failed_count_before
    assert not hasattr(session, "status")


async def test_flush_staged_dropped_after_trip() -> None:
    """封闸（_cancelled=True）后，run 收尾时 _flush_staged 丢弃本轮 staged 的子任务。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    root = _root()
    tm.register_task(root)
    for tid in ("c1", "c2", "c3"):
        tm.register_task(_child(tid))
    await _fail_n_times(tm, ["c1", "c2", "c3"])
    assert tm._cancelled is True

    late_task = Task(id="late_child", session_id="s1", status="PENDING")
    tm.stage_task(late_task, parent_task_id="c1")
    await tm._flush_staged("c1")

    assert "late_child" not in tm._tasks
    assert EventType.TASK_CREATED not in [e.type for e in bus.events if e.task_id == "late_child"]


async def test_late_canceled_does_not_override_failed_session_status() -> None:
    """在途非 root run 的协作取消迟到收尾（status="CANCELED"）不得把 session 从 FAILED 盖回 CANCELED。"""
    bus = _CapturingBus()
    tm, session = _tm(bus)
    root = _root()
    tm.register_task(root)
    for tid in ("c1", "c2", "c3"):
        tm.register_task(_child(tid))
    inflight = _child("inflight1")
    tm.register_task(inflight)
    tm._running_tasks.add("inflight1")

    await _fail_n_times(tm, ["c1", "c2", "c3"])
    assert not hasattr(session, "status")

    # inflight1 的 run 退出路径迟到收尾：走 on_task_finished(status="CANCELED")
    await tm.on_task_finished("inflight1", status="CANCELED")

    assert not hasattr(session, "status")  # 不被盖成 CANCELED


async def test_ack_tasks_and_failures_passed_to_finalizer() -> None:
    """ack_tasks（Task 14 起：只含在途、终态未坐实的已启动带框任务）与 failures 清单正确传给
    threshold_finalizer。已经直接标 CANCELED 的挂起任务（framed_susp）改走 cancel_finalizer
    整对闭合（见 test_suspended_started_task_routed_to_cancel_finalizer），不再进 ack_tasks——
    threshold_finalizer 的 ack_tasks 现在专指「发了信号但还没收尾」的在途任务。
    """
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    root = _root(started_at=now_utc())
    tm.register_task(root)

    framed_susp = _child(
        "framed_susp", status="SUSPENDED",
        started_at=now_utc(),
        origin_tool_call_id="call-1",
    )
    tm.register_task(framed_susp)
    unframed_susp = _child("unframed_susp", status="SUSPENDED")  # 未启动/无框——不应进任何闭合路径
    tm.register_task(unframed_susp)

    framed_inflight = _child(
        "framed_inflight",
        started_at=now_utc(),
        origin_tool_call_id="call-2",
    )
    tm.register_task(framed_inflight)
    tm._running_tasks.add("framed_inflight")

    for tid in ("d1", "d2", "d3"):
        tm.register_task(_child(tid))

    captured: dict = {}

    async def _finalizer(root_task, ack_tasks, failures):
        captured["root"] = root_task
        captured["ack_ids"] = sorted(t.id for t in ack_tasks)
        captured["failures"] = failures

    tm.set_hooks(TaskManagerHooks(threshold_finalizer=_finalizer))
    await _fail_n_times(tm, ["d1", "d2", "d3"])

    assert captured["root"] is root
    # framed_susp 已终态坐实（直接 CANCELED）→ 不再经 threshold_finalizer 的 ack_tasks；
    # framed_inflight 仍在途（只发了协作取消信号）→ 保留在 ack_tasks 做 eager ack 替换。
    assert captured["ack_ids"] == ["framed_inflight"]
    assert len(captured["failures"]) == 3
    assert all(isinstance(f, tuple) and len(f) == 2 for f in captured["failures"])


async def test_suspended_started_task_routed_to_cancel_finalizer() -> None:
    """Task 14：挂起中已启动的任务被熔断清场标 CANCELED 后，立即经 cancel_finalizer 整对闭合
    （不再等 threshold_finalizer 的 ack-only 半闭合——它已经是终态，没有后续收尾会补 finish 对）。
    """
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    root = _root(started_at=now_utc())
    tm.register_task(root)

    framed_susp = _child(
        "framed_susp", status="SUSPENDED",
        started_at=now_utc(),
        origin_tool_call_id="call-1",
    )
    tm.register_task(framed_susp)
    for tid in ("d1", "d2", "d3"):
        tm.register_task(_child(tid))

    captured: dict = {}

    async def _cancel_finalizer(tasks, reason):
        captured["ids"] = sorted(t.id for t in tasks)
        captured["reason"] = reason

    tm.set_hooks(TaskManagerHooks(cancel_finalizer=_cancel_finalizer))
    await _fail_n_times(tm, ["d1", "d2", "d3"])

    assert captured["ids"] == ["framed_susp"]
    assert captured["reason"] == "failure_threshold"


async def test_cancel_inflight_called_for_inflight_non_root_and_root() -> None:
    """cancel_inflight 对在途非 root task 与在途 root task 都被调用（收到对应 task_id）。"""
    bus = _CapturingBus()
    tm, _session = _tm(bus)
    root = _root(status="ACTIVE")  # root 自己在跑（无待完成子任务），而非"父等子"
    root.started_at = now_utc()
    tm.register_task(root)
    tm._running_tasks.add("root")
    # 真正"在途且不是失败触发者本身"的非 root task：三个触发计数的任务另有其人（c1/c2/c3）。
    inflight_child = _child("inflight_child")
    tm.register_task(inflight_child)
    tm._running_tasks.add("inflight_child")
    for tid in ("c1", "c2", "c3"):
        tm.register_task(_child(tid))

    called: list[str] = []

    def _cancel_inflight(tid: str) -> bool:
        called.append(tid)
        return True

    tm.set_hooks(TaskManagerHooks(cancel_inflight=_cancel_inflight))

    await _fail_n_times(tm, ["c1", "c2", "c3"])

    assert "inflight_child" in called
    assert "root" in called
    assert root.status == "ACTIVE"  # Terminal settlement waits for the root's execution exit.
    assert "root" in tm._pending_threshold_roots


async def test_root_exit_waits_for_trip_commit_and_settles_only_once():
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind

    entered = asyncio.Event()
    release = asyncio.Event()
    class DelayedBus(_CapturingBus):
        async def emit(self, event):
            if event.type == EventType.FAILURE_THRESHOLD_HIT:
                entered.set()
                await release.wait()
            await super().emit(event)

    bus = DelayedBus()
    tm, session = _tm(bus, threshold=1)
    root = _root(status="ACTIVE", started_at=now_utc())
    tm.register_task(root)
    tm._running_tasks.add(root.id)
    tm.register_task(_child("failed"))
    trip = asyncio.create_task(_fail_n_times(tm, ["failed"]))
    await entered.wait()
    completion = asyncio.create_task(tm.apply_run_outcome(
        root.id, RunOutcome(kind=RunOutcomeKind.CANCELED)))
    await asyncio.sleep(0)
    assert not completion.done() and root.status == "ACTIVE"
    # Root exits while trip cleanup is yielding, before its root loop executes.
    async def await_root_exit():
        await completion
    tm.set_hooks(TaskManagerHooks(cancel_pending_hitl=await_root_exit))
    release.set()
    await trip
    assert await completion == "FAILED"
    assert session.failure_counter == 1  # The aggregate root failure is not another failure.
    root_failed = [event for event in bus.events
                   if event.type == EventType.TASK_FAILED and event.task_id == root.id]
    assert len(root_failed) == 1
    assert _types(bus).index(EventType.FAILURE_THRESHOLD_HIT) < bus.events.index(root_failed[0])


async def test_root_already_terminal_skips_finalization() -> None:
    """root 自己是触发第 N 败的任务（FinalizeStep 已闭合为 FAILED）→ trip 判死跳过，不重复改状态/发事件。"""
    bus = _CapturingBus()
    tm, session = _tm(bus)
    root = _root()
    tm.register_task(root)
    tm.register_task(_child("c1"))
    tm.register_task(_child("c2"))

    await _fail_n_times(tm, ["c1", "c2"])
    # 第 3 败由 root 自己触发：观察者已把它判 fail，FinalizeStep 已经把 status 置 FAILED
    root.status = "FAILED"
    root.error_code = "TASK_FAILED_BY_OBSERVER"
    root.finished_at = now_utc()

    await tm.on_task_finished("root", status="FAILED")

    # root 的 error_code 不应被 trip 序列覆盖成 TASK_FAILED_BY_THRESHOLD
    assert root.error_code == "TASK_FAILED_BY_OBSERVER"
    assert not hasattr(session, "status")
    # trip 序列没有为 root 再发一条 TASK_FAILED（root 已终态、判死跳过）
    root_failed_events = [
        e for e in bus.events if e.type == EventType.TASK_FAILED and e.task_id == "root"
    ]
    assert root_failed_events == []


async def test_cancel_pending_hitl_invoked_before_terminal_status() -> None:
    """cancel_pending_hitl 在终态落定（→ 收尾）之前被 await 调用。

    2026-09-04（Task 12，events-v2 §5）起 trip 序列第 8 步不再发 TASK_QUEUE_DRAINED
    （`announce_queue_state` 已停发）——`session.status = "FAILED"` 直接写定,紧接着
    调 `_fire_session_done()`。没有事件可拦截了,改拦 `_fire_session_done` 本身的调用
    时机（它是终态写定之后**唯一**还在的、可挂钩的收尾点）。
    """
    bus = _CapturingBus()
    tm, session = _tm(bus)
    root = _root()
    tm.register_task(root)
    for tid in ("c1", "c2", "c3"):
        tm.register_task(_child(tid))

    call_order: list[str] = []

    async def _cancel_hitl():
        call_order.append("cancel_hitl")

    tm.set_hooks(TaskManagerHooks(cancel_pending_hitl=_cancel_hitl))
    orig_fire_session_done = tm._fire_session_done

    async def _tracking_fire_session_done():
        if tm._threshold_tripped and root.status == "FAILED":
            call_order.append("session_failed")
        await orig_fire_session_done()

    tm._fire_session_done = _tracking_fire_session_done

    await _fail_n_times(tm, ["c1", "c2", "c3"])

    assert call_order == ["cancel_hitl", "session_failed"]
    assert not hasattr(session, "status")
