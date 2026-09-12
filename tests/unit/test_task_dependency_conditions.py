"""spec: task-handoff——依赖条件：双完成集解锁、写入物化、永久阻塞善后、会话豁免。"""

from __future__ import annotations

import pytest

from ctx_weft.core.models.discriminators import TaskErrorCode
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry, TaskQueue, split_deps
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


# ── 4.1 队列双完成集 ─────────────────────────────────────────────────────────


def test_split_deps_interprets_missing_condition_as_any():
    any_d, succ_d = split_deps(["a", "b"], None)
    assert any_d == {"a", "b"} and succ_d == set()
    any_d, succ_d = split_deps(["a", "b"], {"a": "success"})
    assert any_d == {"b"} and succ_d == {"a"}


def test_finished_releases_both_kinds():
    q = TaskQueue()
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    q.push(QueueEntry(task_id="c", session_id="s", blocked_success={"a"}))
    q.mark_running("a")
    q.mark_complete("a")  # FINISHED
    assert q.pop() is not None and q.pop() is not None  # b、c 都放行


def test_failed_releases_only_any_deps():
    q = TaskQueue()
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    q.push(QueueEntry(task_id="c", session_id="s", blocked_success={"a"}))
    q.mark_running("a")
    q.mark_failed("a")  # FAILED
    assert q.pop().task_id == "b"      # on_any 放行
    assert q.pop() is None             # on_success 仍阻塞


def test_canceled_terminal_releases_only_any_deps():
    q = TaskQueue()
    q.push(QueueEntry(task_id="c", session_id="s", blocked_success={"a"}))
    q.push(QueueEntry(task_id="b", session_id="s", blocked_by={"a"}))
    q.mark_running("a")
    q.mark_failed("a")  # 取消走同一「终态但非成功」记账
    assert q.pop().task_id == "b"
    assert q.pop() is None


def test_seed_succeeded_only_releases_success_deps_on_finished():
    q = TaskQueue()
    q.seed_completed({"x"})
    q.seed_succeeded(set())            # x 终态但未成功
    q.push(QueueEntry(task_id="c", session_id="s", blocked_success={"x"}))
    assert q.pop() is None


# ── 4.2 写入物化 / restore / reopen ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_push_task_materializes_default_success():
    bus = _Bus()
    tm = _tm(bus)
    t = _task("b", dag_deps=[], dep_conditions=None)
    await tm.push_task(t, blocked_by=["a"])
    assert t.dep_conditions == {"a": "success"}          # 缺省物化为 success
    entry = next(e for e in tm._queue.peek_all() if e.task_id == "b")
    assert entry.blocked_success == {"a"} and entry.blocked_by == set()
    # 事件里全量显式落盘
    ev = bus.of(EventType.TASK_CREATED)[0]
    assert ev.payload["task"]["dep_conditions"] == {"a": "success"}


@pytest.mark.asyncio
async def test_push_task_keeps_explicit_any_condition():
    tm = _tm()
    t = _task("c", dep_conditions={"a": "any"})
    await tm.push_task(t, blocked_by=["a"])
    assert t.dep_conditions == {"a": "any"}              # 显式声明优先于缺省
    entry = tm._queue.peek_all()[-1]
    assert entry.blocked_by == {"a"} and entry.blocked_success == set()


def test_restore_legacy_deps_replay_as_any():
    """存量事件：dag_deps 无条件字段 → 回放按 any 解释（历史保真）。"""
    tm = _tm()
    a = _task("a", status="FAILED")
    b = _task("b", dag_deps=["a"], dep_conditions=None)
    tm.restore([a, b], terminal_ids={"a"})
    entry = tm._queue.peek_all()[0]
    assert entry.task_id == "b"
    assert entry.blocked_by == set() and entry.blocked_success == set()  # a 已终态，any 释放


def test_restore_new_style_success_dep_stays_blocked_on_failed():
    """新派发：success 条件 + 前序 FAILED → 恢复后保持阻塞（等扫描处置）。"""
    tm = _tm()
    a = _task("a", status="FAILED")
    b = _task("b", dag_deps=["a"], dep_conditions={"a": "success"})
    tm.restore([a, b], terminal_ids={"a"})
    entry = tm._queue.peek_all()[0]
    assert entry.blocked_success == {"a"}               # 未释放


def test_restore_finished_dep_releases_success_dep():
    tm = _tm()
    a = _task("a", status="FINISHED")
    b = _task("b", dag_deps=["a"], dep_conditions={"a": "success"})
    tm.restore([a, b], terminal_ids={"a"})
    entry = tm._queue.peek_all()[0]
    assert entry.blocked_success == set() and entry.blocked_by == set()


@pytest.mark.asyncio
async def test_reopen_chain_preserves_conditions():
    """重开链重建依赖后条件存活：success 仍 success、legacy 仍 any。"""
    from datetime import datetime, timezone

    def mk(tid, tracking, created):
        return Task(id=tid, session_id="s1", status="FINISHED",
                    dag_deps=[], dep_conditions=None, tracking_task_ids=tracking,
                    created_at=datetime(2026, 1, 1, 0, 0, created, tzinfo=timezone.utc))

    tm = _tm()
    c1 = mk("c1", [], 1)
    c2 = mk("c2", ["c1"], 2)
    c2.dag_deps = ["c1"]
    c2.dep_conditions = {"c1": "any"}                   # 显式 any
    c3 = mk("c3", ["c1", "c2"], 3)
    c3.dag_deps = ["c2"]
    c3.dep_conditions = {"c2": "success"}
    for t in (c1, c2, c3):
        tm.register_task(t)

    assert await tm.reopen_chain("c1", "fix") is True
    assert c2.dep_conditions == {"c1": "any"}
    assert c3.dep_conditions == {"c2": "success"}


# ── 4.3 + 4.6 永久阻塞善后 / 会话豁免 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_predecessor_disposes_success_dependent():
    bus = _Bus()
    tm = _tm(bus)
    parent = _task("P", status="SUSPENDED")
    a = _task("a", parent_task_id="P")
    b = _task("b", parent_task_id="P", dag_deps=["a"], dep_conditions={"a": "success"})
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
async def test_blocked_disposal_cascades_and_spares_any_dependents():
    bus = _Bus()
    tm = _tm(bus)
    a = _task("a")
    b = _task("b", dag_deps=["a"], dep_conditions={"a": "success"})
    c = _task("c", dag_deps=["b"], dep_conditions={"b": "success"})   # 级联受害者
    cleanup = _task("cl", dag_deps=["a"], dep_conditions={"a": "any"})
    for t in (a, b, c, cleanup):
        tm.register_task(t)
    await tm.push_task(a)
    await tm.push_task(b, blocked_by=["a"])
    await tm.push_task(c, blocked_by=["b"])
    await tm.push_task(cleanup, blocked_by=["a"])

    a.status = "FAILED"
    await tm.on_task_finished("a", status="FAILED")

    assert b.status == "CANCELED" and b.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP
    assert c.status == "CANCELED" and c.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP
    assert c.error and "b" in c.error                      # 级联的阻塞源指向 b
    # on_any 清理任务不被处置（保持 PENDING、可派发）
    assert cleanup.status == "PENDING"
    assert any(e.task_id == "cl" for e in tm._queue.peek_all())
    # 幂等：重复扫描无新受害者
    n_events = len(bus.events)
    await tm.dispose_blocked_dependents()
    assert len(bus.events) == n_events


@pytest.mark.asyncio
async def test_recovery_scan_disposes_crash_window_leftover():
    """崩溃窗口：A FAILED 已落盘、B 级联取消未落盘 → 恢复期扫描补齐。"""
    bus = _Bus()
    tm = _tm(bus)
    a = _task("a", status="FAILED")
    b = _task("b", dag_deps=["a"], dep_conditions={"a": "success"})
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
    b = _task("b", dag_deps=["a"], dep_conditions={"a": "success"})
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
    b = _task("b", parent_task_id="P", dag_deps=["a"], dep_conditions={"a": "success"})
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
