"""spec: task-handoff——依赖放行：只有前序 FINISHED 解锁、永久阻塞善后、会话豁免。"""

from __future__ import annotations

import pytest

from ctx_weft.core.models.discriminators import TaskErrorCode
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry, TaskQueue
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InProcessEventBus


class _Bus:
    def __init__(self) -> None:
        self.bus = InProcessEventBus()
        self.events: list = []

        async def _sink(e) -> None:
            self.events.append(e)

        self.bus.subscribe(None, _sink)

    def of(self, t: EventType):
        return [e for e in self.events if e.type == t]


def _task(tid: str, status: str = "PENDING", **kw) -> Task:
    return Task(id=tid, session_id="s1", status=status, **kw)


class _NoopRunner:
    """哑 runner：drain 需要 runner 在册；max_concurrent=0 保证不真正派发。"""

    def set_task_manager(self, tm) -> None:  # pragma: no cover - 接口对齐
        pass

    async def assemble(self, task_id):
        return None

    async def execute(self, binding, task_id):
        return None


def _tm(bus: _Bus | None = None, session: Session | None = None) -> TaskManager:
    tm = TaskManager(session_id="s1", max_concurrent=0,
                     event_bus=bus.bus if bus else None)
    tm.set_session(session or Session(id="s1", user_prompt="", status="RUNNING"))
    tm.set_runner(_NoopRunner())
    return tm


# ── 队列：唯一解锁依据是 FINISHED ────────────────────────────────────────────


def test_finished_releases_dependents():
    q = TaskQueue()
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    q.mark_running("a")
    q.mark_complete("a")  # FINISHED
    assert q.pop().task_id == "b"


def test_failed_releases_nothing():
    """改造前这里是 `_completed.add(task_id)`（"treat failed as done for unblocking"），
    后继于是拿着不存在的产出照跑。"""
    q = TaskQueue()
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    q.mark_running("a")
    q.mark_failed("a")
    assert q.pop() is None


def test_canceled_terminal_releases_nothing():
    q = TaskQueue()
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    q.mark_running("a")
    q.mark_failed("a")  # 取消走同一「终态但非成功」记账
    assert q.pop() is None


def test_seed_succeeded_only_takes_finished():
    q = TaskQueue()
    q.seed_succeeded(set())            # x 终态但未成功 → 不入集
    q.push(QueueEntry(task_id="c", session_id="s", blocked_by={"x"}))
    assert q.pop() is None
    q.seed_succeeded({"x"})
    assert q.pop().task_id == "c"


def test_unmark_succeeded_reblocks_reopened_dep():
    q = TaskQueue()
    q.mark_complete("a")
    q.unmark_succeeded("a")            # a 被 reopen → 后继重新等它
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    assert q.pop() is None


# ── restore ──────────────────────────────────────────────────────────────────


def test_restore_failed_dep_stays_blocked():
    """前序 FAILED → 恢复后保持阻塞（等恢复期扫描处置），不放行。"""
    tm = _tm()
    a = _task("a", status="FAILED")
    b = _task("b", dag_deps=["a"])
    tm.restore([a, b], terminal_ids={"a"})
    entry = tm._queue.peek_all()[0]
    assert entry.task_id == "b" and entry.blocked_by == {"a"}


def test_restore_finished_dep_releases():
    tm = _tm()
    a = _task("a", status="FINISHED")
    b = _task("b", dag_deps=["a"])
    tm.restore([a, b], terminal_ids={"a"})
    assert tm._queue.peek_all()[0].blocked_by == set()


@pytest.mark.asyncio
async def test_push_task_blocks_on_declared_deps():
    bus = _Bus()
    tm = _tm(bus)
    t = _task("b")
    await tm.push_task(t, blocked_by=["a"])
    assert t.dag_deps == ["a"]
    entry = next(e for e in tm._queue.peek_all() if e.task_id == "b")
    assert entry.blocked_by == {"a"}
    assert bus.of(EventType.TASK_CREATED)[0].payload["task"]["dag_deps"] == ["a"]


# ── 永久阻塞善后 / 会话豁免 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_predecessor_disposes_dependent():
    bus = _Bus()
    tm = _tm(bus)
    parent = _task("P", status="SUSPENDED")
    a = _task("a", parent_task_id="P")
    b = _task("b", parent_task_id="P", dag_deps=["a"])
    for t in (parent, a, b):
        tm.register_task(t)
    await tm.push_task(a)
    await tm.push_task(b, blocked_by=["a"])
    tm._children_of["P"] = {"a", "b"}

    a.status = "FAILED"
    await tm.on_task_finished("a", status="FAILED")

    # b：不执行、CANCELED + BLOCKED_BY_FAILED_DEP、事件带阻塞源
    assert b.status == "CANCELED"
    assert b.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP
    ev = bus.of(EventType.TASK_CANCELED)[0]
    assert ev.task_id == "b"
    assert ev.payload["blocked_by_task_id"] == "a"
    assert ev.payload["error_code"] == "BLOCKED_BY_FAILED_DEP"
    # 会话不被改写为 CANCELED（依赖取消 ≠ 用户叫停）
    assert tm.session.status == "RUNNING"
    # 队列无滞留
    assert all(e.task_id != "b" for e in tm._queue.peek_all())


@pytest.mark.asyncio
async def test_blocked_disposal_cascades_to_fixpoint():
    bus = _Bus()
    tm = _tm(bus)
    a = _task("a")
    b = _task("b", dag_deps=["a"])
    c = _task("c", dag_deps=["b"])   # 级联受害者
    for t in (a, b, c):
        tm.register_task(t)
    await tm.push_task(a)
    await tm.push_task(b, blocked_by=["a"])
    await tm.push_task(c, blocked_by=["b"])

    a.status = "FAILED"
    await tm.on_task_finished("a", status="FAILED")

    assert b.status == "CANCELED" and b.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP
    assert c.status == "CANCELED" and c.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP
    assert c.error and "b" in c.error                      # 级联的阻塞源指向 b
    # 幂等：重复扫描无新受害者
    n_events = len(bus.events)
    await tm.dispose_blocked_dependents()
    assert len(bus.events) == n_events


@pytest.mark.asyncio
async def test_independent_siblings_unaffected_by_a_failure():
    """无依赖边的兄弟任务（多次 delegate_task 的形态）不受彼此成败影响。"""
    bus = _Bus()
    tm = _tm(bus)
    a = _task("a")
    other = _task("other")            # 与 a 无边
    for t in (a, other):
        tm.register_task(t)
    await tm.push_task(a)
    await tm.push_task(other)

    a.status = "FAILED"
    await tm.on_task_finished("a", status="FAILED")

    assert other.status == "PENDING"
    assert any(e.task_id == "other" for e in tm._queue.peek_all())


@pytest.mark.asyncio
async def test_recovery_scan_disposes_crash_window_leftover():
    """崩溃窗口：A FAILED 已落盘、B 级联取消未落盘 → 恢复期扫描补齐。"""
    bus = _Bus()
    tm = _tm(bus)
    a = _task("a", status="FAILED")
    b = _task("b", dag_deps=["a"])
    tm.restore([a, b], terminal_ids={"a"})

    await tm.dispose_blocked_dependents()   # runtime 在首次 drain 前调

    assert b.status == "CANCELED"
    assert b.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP
    assert bus.of(EventType.TASK_CANCELED)
    assert tm._queue.peek_all() == []       # 无滞留


@pytest.mark.asyncio
async def test_blocked_cancel_does_not_touch_failure_counter():
    bus = _Bus()
    tm = _tm(bus)
    a = _task("a")
    b = _task("b", dag_deps=["a"])
    tm.register_task(a)
    tm.register_task(b)
    await tm.push_task(a)
    await tm.push_task(b, blocked_by=["a"])
    before = tm.session.failure_counter

    a.status = "FAILED"
    await tm.on_task_finished("a", status="FAILED")

    # a 的 FAILED 计数 +1 是真失败；b 的 CANCELED 不再计入
    assert tm.session.failure_counter == before + 1


@pytest.mark.asyncio
async def test_blocked_cancel_still_wakes_parent():
    """依赖取消保留父任务唤醒：子任务全终态后 SUSPENDED 父回 PENDING。"""
    bus = _Bus()
    tm = _tm(bus)
    parent = _task("P", status="SUSPENDED")
    a = _task("a", parent_task_id="P")
    b = _task("b", parent_task_id="P", dag_deps=["a"])
    for t in (parent, a, b):
        tm.register_task(t)
    tm._children_of["P"] = {"a", "b"}
    tm._parent_map["a"] = "P"
    tm._parent_map["b"] = "P"
    await tm.push_task(a)
    await tm.push_task(b, blocked_by=["a"])

    a.status = "FAILED"
    await tm.on_task_finished("a", status="FAILED")   # 触发 b 的处置 → b 终态

    assert parent.status == "PENDING"                 # 被唤醒、重排
    assert tm.session.status != "CANCELED"
