"""S4.1 带外判决入口：一条不属于任何活跃 run 的结局。

park 之后 run 就地结束（`_cold_park` 抛 `HitlPark` 释放协程），后台 observe 判完时这个
task 没有任何 run 在跑——`_close_report` 那条「同 run 内由 finalize 取用」的路走不通。

**仲裁是这道入口的要害**：`_inject_user_turn` 写 USER_PROMPT 时不等后台 observe，所以
「人先开口」是常态而非边角。判决到达时若 task 已不在 `AWAITING_HUMAN`，它就过时了。
"""

from __future__ import annotations

from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType


class _CapturingBus:
    async def emit_batch(self, events, *, on_committed=None):
        self.events.extend(events)
        if on_committed is not None:
            on_committed()

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


def _tm() -> tuple[TaskManager, _CapturingBus]:
    bus = _CapturingBus()
    tm = TaskManager(session_id="s1", event_bus=bus)
    tm.set_session(Session(id="s1", user_prompt="",  root_agent_id="root"))
    # `_settle` 末尾恒调 `drain()`：给一个哑 runner 过掉「未注册」守卫，再把并发置 0
    # 让 drain 空转——本文件测的是仲裁与状态转移，不是派发。
    tm.set_runner(object())
    tm._max_concurrent = 0
    return tm, bus


def _parked(tm: TaskManager, tid: str = "A", **kw) -> Task:
    task = Task(id=tid, session_id="s1", status="AWAITING_HUMAN",
                assigned_agent_id="root", creator_agent_id="root", **kw)
    tm.register_task(task)
    return task


def _success(**kw) -> RunOutcome:
    return RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="success", **kw)


def _types(bus: _CapturingBus) -> list:
    return [e.type for e in bus.events]


# ── 被接受 ────────────────────────────────────────────────────────────────────

async def test_success_verdict_finishes_the_parked_task() -> None:
    tm, bus = _tm()
    task = _parked(tm, outputs="交付物")

    accepted = await tm.apply_out_of_band_verdict(
        "A", _success(summary="小结", outputs="交付物"),
        process_report="做了甲乙丙", task_summary="小结",
    )

    assert accepted is True
    assert task.status == "FINISHED"
    assert task.observer_outcome == "success"
    assert task.process_report == "做了甲乙丙"
    assert task.process_report_at is not None
    assert task.task_summary == "小结"
    assert EventType.TASK_FINISHED in _types(bus)


async def test_hint_is_cleared_when_not_supplied() -> None:
    """不给 hint 就清掉，不留上一轮的过期指令。"""
    tm, _ = _tm()
    task = _parked(tm, outputs="x")
    task.next_step_hint = "Next Step Hint: 上一轮的陈货"

    await tm.apply_out_of_band_verdict("A", _success())

    assert task.next_step_hint is None


# ── 被拒绝 ────────────────────────────────────────────────────────────────────

async def test_verdict_rejected_when_user_spoke_first() -> None:
    """人先开口 → `_inject_user_turn` 把它重排成 PENDING → 这份判决已过时。"""
    tm, bus = _tm()
    task = _parked(tm, outputs="交付物")
    task.status = "PENDING"                      # 模拟已被重排

    accepted = await tm.apply_out_of_band_verdict(
        "A", _success(), process_report="来晚了的 recap",
    )

    assert accepted is False
    assert task.status == "PENDING"              # 一个字段都没动
    assert task.observer_outcome is None
    assert task.process_report is None
    assert bus.events == []                      # 连事件都不该发


async def test_verdict_rejected_when_task_already_terminal() -> None:
    tm, _ = _tm()
    task = _parked(tm)
    task.status = "FINISHED"

    assert await tm.apply_out_of_band_verdict("A", _success()) is False
    assert task.observer_outcome is None


async def test_verdict_rejected_for_unknown_task() -> None:
    tm, _ = _tm()
    assert await tm.apply_out_of_band_verdict("不存在", _success()) is False


async def test_verdict_rejected_when_suspended_on_children() -> None:
    """SUSPENDED 等子任务不是「停下来等人」，带外判决不该认领它。"""
    tm, _ = _tm()
    task = _parked(tm)
    task.status = "SUSPENDED"

    assert await tm.apply_out_of_band_verdict("A", _success()) is False
    assert task.status == "SUSPENDED"


# ── 幂等 ──────────────────────────────────────────────────────────────────────

async def test_second_verdict_is_rejected() -> None:
    """第一次落定后 task 已不在 AWAITING_HUMAN，重复提交自然被仲裁挡掉。"""
    tm, bus = _tm()
    _parked(tm, outputs="x")

    assert await tm.apply_out_of_band_verdict("A", _success()) is True
    before = len(bus.events)
    assert await tm.apply_out_of_band_verdict("A", _success()) is False
    assert len(bus.events) == before


# ── retry / fail 维持 park（S5）────────────────────────────────────────────────

async def test_waiting_verdict_keeps_it_parked():
    tm, bus = _tm()
    task = _parked(tm, retry_count=0, max_retries=3, error="previous blocker")
    accepted = await tm.apply_out_of_band_verdict(
        "A", RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="waiting"),
        process_report="asked user")
    assert accepted
    assert task.status == "AWAITING_HUMAN"
    assert task.observer_outcome == "waiting"
    assert task.retry_count == 0
    assert task.error is None
    assert bus.events == []


async def test_continue_always_requeues_and_preserves_exception_budget():
    tm, bus = _tm()
    task = _parked(tm, outputs="interim", retry_count=99, max_retries=0)
    seen = []
    async def resume():
        seen.append((task.status, list(bus.events)))
    assert await tm.apply_out_of_band_verdict("A",
        RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="continue"),
        next_step_hint="finish work", resume=resume)
    assert task.status == "PENDING"
    assert task.observer_outcome == "continue"
    assert task.retry_count == 99
    assert task.outputs is None
    assert task.next_step_hint == "finish work"
    assert seen == [("PENDING", [])]
    assert _types(bus) == [EventType.TASK_REQUEUED]
    assert [entry.task_id for entry in tm._queue.peek_all()] == ["A"]
    assert not await tm.apply_out_of_band_verdict("A",
        RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="continue"))


async def test_fail_verdict_closes_the_task():
    tm, bus = _tm()
    task = _parked(tm)
    assert await tm.apply_out_of_band_verdict("A",
        RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="fail", error="API unavailable"))
    assert task.status == "FAILED"
    assert task.observer_outcome == "fail"
    assert task.error == "API unavailable"
    assert EventType.TASK_FAILED in _types(bus)


# ── 带外收尾钩子（2026-09-27）──────────────────────────────────────────────────
#
# `finalize` 跑在「状态已写定」与「转移已宣布」之间那个窗口里。三条约束各钉一侧，下面
# 一条一测。第一个用户是收 park 气泡（子任务那一档），S-a 之后 bubble 给 parent /
# blackboard / TASK_FINALIZED 也挂在这里。

async def test_finalize_runs_after_the_status_is_written() -> None:
    """**在仲裁之后**：钩子被调到的那一刻，task 已经是终态了。

    这是竞态的关门点——status 一落终态，人的消息在 `send_message` 层就按
    `_task_is_terminal` 走新建分支，不会再 requeue 这个正在收尾的 task。所以带 IO 的
    收尾跑在锁外是安全的。
    """
    tm, _bus = _tm()
    task = _parked(tm, outputs="交付物")
    seen: list[str] = []

    async def _fin() -> None:
        seen.append(task.status)

    await tm.apply_out_of_band_verdict("A", _success(), finalize=_fin)
    assert seen == ["FINISHED"]


async def test_finalize_runs_before_the_transition_is_announced() -> None:
    """**在 `_settle` 之前**：`_settle` → `on_task_finished` → `_try_resume_parent` 会
    唤醒 parent，parent 不能在子任务的产出落地之前醒过来装配。

    用「钩子跑的时候 `TaskFinished` 还没发出去」来钉这个顺序。
    """
    tm, bus = _tm()
    _parked(tm)
    at_hook: list[list] = []

    async def _fin() -> None:
        at_hook.append(_types(bus))

    await tm.apply_out_of_band_verdict("A", _success(), finalize=_fin)
    assert at_hook == [[]], "钩子跑在任何事件之前"
    assert EventType.TASK_FINISHED in _types(bus), "之后照常宣布"


async def test_a_failing_finalize_does_not_swallow_the_transition() -> None:
    """收尾抛异常 → 记日志，照常往下发事件 + `_settle`。

    状态已经是终态了：不发事件宿主永远不知道这个 task 结束了，不 `_settle` 则槽位永不
    释放、单交互线永久占着。降级成「这个 task 少了收尾副作用」远好过卡住整个会话。
    """
    tm, bus = _tm()
    _parked(tm)

    async def _boom() -> None:
        raise RuntimeError("bubble service is down")

    accepted = await tm.apply_out_of_band_verdict("A", _success(), finalize=_boom)
    assert accepted is True
    assert tm.get_task("A").status == "FINISHED"
    assert EventType.TASK_FINISHED in _types(bus)


async def test_finalize_is_not_called_when_the_verdict_is_rejected() -> None:
    """人先开口 → 仲裁拒绝 → 没有转移，也就没有收尾。"""
    tm, _bus = _tm()
    _parked(tm)
    tm.get_task("A").status = "PENDING"        # 人的消息已把它重排
    called = []

    await tm.apply_out_of_band_verdict(
        "A", _success(), finalize=lambda: called.append(1) or _noop())
    assert called == []


async def test_terminal_finalize_is_not_called_for_continue() -> None:
    """retry 维持 park——task 没终结，没有收尾可言。"""
    tm, _bus = _tm()
    _parked(tm)
    called = []

    await tm.apply_out_of_band_verdict(
        "A", RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="continue"),
        finalize=lambda: called.append(1) or _noop())
    assert called == []
    assert tm.get_task("A").status == "PENDING"


async def _noop() -> None:
    return None


async def test_cancel_wins_against_late_verdict():
    tm, bus = _tm()
    task = _parked(tm, outputs="answer")
    assert await tm.cancel_stopped_task(task.id, "user canceled")
    assert not await tm.apply_out_of_band_verdict(task.id, _success(), process_report="obsolete")
    assert task.status == "CANCELED"
    assert task.process_report is None
    assert _types(bus) == [EventType.TASK_CANCELED]
    assert not await tm.cancel_stopped_task(task.id, "again")


async def test_committed_verdict_wins_before_cancel_even_before_task_finished():
    import asyncio
    tm, bus = _tm()
    task = _parked(tm, outputs="answer")
    committed = asyncio.Event()
    release = asyncio.Event()
    async def before_apply(accepted, disposition):
        assert accepted and disposition.status == "FINISHED"
    async def finalize():
        committed.set()
        await release.wait()
    producer = asyncio.create_task(tm.apply_out_of_band_verdict(task.id, _success(),
        before_apply=before_apply, finalize=finalize))
    await committed.wait()
    assert EventType.TASK_FINISHED not in _types(bus)
    assert not await tm.cancel_stopped_task(task.id, "late cancel")
    release.set()
    assert await producer
    assert task.status == "FINISHED"
    assert _types(bus) == [EventType.TASK_FINISHED]


async def test_failed_cancel_commit_does_not_revoke_observer():
    import pytest
    tm, bus = _tm()
    task = _parked(tm, outputs="answer")
    original = bus.emit
    async def fail(event):
        raise OSError("cancel commit failed")
    bus.emit = fail
    with pytest.raises(OSError, match="cancel commit failed"):
        await tm.cancel_stopped_task(task.id, "cancel")
    assert task.status == "AWAITING_HUMAN"
    bus.emit = original
    assert await tm.apply_out_of_band_verdict(task.id, _success())
    assert task.status == "FINISHED"
