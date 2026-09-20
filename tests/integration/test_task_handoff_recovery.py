"""spec: task-handoff——集成路径：ack-id→review、计划断裂、阻塞原因可解释。

驱动方式：真 TaskManager + 真 InProcessEventBus 发射真事件 →（模拟崩溃）事件列表经
reduce_events → converters → 新 TaskManager.restore 重建——这正是 rebuild_view 的
内核链路；控制工具经真 ControlContext 调用。
"""

from __future__ import annotations

import pytest

from ctx_weft.core.capabilities.control_tools import (
    ControlContext,
    delegate_plan,
    delegate_task,
)
from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InProcessEventBus


class _CapturingBus:
    def __init__(self) -> None:
        self.bus = InProcessEventBus()
        self.events: list = []

        async def _sink(e) -> None:
            self.events.append(e)

        self.bus.subscribe(None, _sink)


class _NoopRunner:
    async def assemble(self, task_id):
        return None

    async def execute(self, binding, task_id):
        return None


def _tm(bus) -> TaskManager:
    tm = TaskManager(session_id="s1", max_concurrent=0, event_bus=bus.bus)
    tm.set_session(Session(id="s1", user_prompt="", status="RUNNING"))
    tm.set_runner(_NoopRunner())
    return tm


def _ctx(tm: TaskManager, task: Task) -> ControlContext:
    return ControlContext(
        session_id="s1", task_id=task.id, agent_id="a1", task=task,
        task_manager=tm, session=None, tool_call_id="tc_1",
    )


def _parent() -> Task:
    return Task(id="P", session_id="s1", status="ACTIVE", title="parent")


async def _delegate_plan_ordered(tm: TaskManager, parent: Task, specs: list[dict]):
    """delegate_plan + flush，按 stage 顺序返回 (ack, 子任务列表)——spec 顺序的唯一可靠读法
    （created_at 在微秒内并列，不能作排序键）。"""
    order: list[Task] = []
    orig_stage = tm.stage_task

    def _rec(child: Task, **kw) -> None:
        order.append(child)
        orig_stage(child, **kw)

    tm.stage_task = _rec
    try:
        from ctx_weft.core.capabilities.control_tools import delegate_plan as _dp
        ack = _dp(tasks=specs, ctx=_ctx(tm, parent))
        await tm._flush_staged(parent.id)
    finally:
        tm.stage_task = orig_stage
    return ack, order


# ── 3.3 端到端：派发 ack 的 id 就是 observer 看到的那个句柄 ──────────────────


@pytest.mark.asyncio
async def test_delegate_ack_id_matches_the_observer_subtask_list():
    """派发 ack 里的 id 与 observe 子任务清单里的 id 必须是同一个。

    这是「稳定句柄」这条链的端到端判据：actor 派发时从 ack 记下 id，observer 之后据
    同一个 id 在 `next_step_hint` 里指名哪个子任务的产出不合格（2026-09-19 起 observer
    不再有 `task_reviews`，也不动任何 task 状态；重派还是自己做由下一轮 actor 决定）。
    """
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = _parent()
    tm.register_task(parent)
    tm._children_of[parent.id] = set()

    res, children = await _delegate_plan_ordered(
        tm, parent, [{"title": "step one"}, {"title": "step two"}],
    )
    assert len(children) == 2
    # ack 逐条给「标题 + id」，且 id 就是真实子任务 id
    for i, c in enumerate(children, start=1):
        assert f"{i}. {c.title!r} ({c.id})" in res.content

    c1 = children[0].id
    c1_task = tm.get_task(c1)
    c1_task.status = "FINISHED"
    c1_task.outputs = "done"
    tm._queue.mark_complete(c1)

    # observe 侧的子任务清单（ObserveStep 就是这么构造 extra["subtasks"] 的）
    listed = {cid: tm.get_task(cid) for cid in tm.children_of(parent.id)}
    assert c1 in listed, "派发出来的子任务必须出现在 observer 的清单里"
    assert listed[c1].title == c1_task.title
    assert listed[c1].status == "FINISHED"      # 结局随清单一起给 observer


# ── 4.4 计划断裂：step2 失败 → 其后各步级联取消（带原因），会话不被误标 ──────


@pytest.mark.asyncio
async def test_plan_break_cancels_all_successors():
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = _parent()
    tm.register_task(parent)

    _, ordered = await _delegate_plan_ordered(tm, parent, [
        {"title": "produce"},
        {"title": "transform"},
        {"title": "final"},
        {"title": "cleanup"},
    ])
    produce, transform, final, cleanup = ordered
    # plan 是严格串行链：produce→transform→final→cleanup
    assert final.dag_deps == [transform.id]
    assert cleanup.dag_deps == [final.id]

    produce.status = "FINISHED"
    await tm.on_task_finished(produce.id, status="FINISHED")
    transform.status = "FAILED"
    await tm.on_task_finished(transform.id, status="FAILED")

    assert final.status == "CANCELED"
    assert final.error_code == "BLOCKED_BY_FAILED_DEP"
    assert final.error and transform.id in final.error
    # 级联到不动点：cleanup 依赖 final，final 被取消后它同样永不可满足
    assert cleanup.status == "CANCELED"
    assert cleanup.error_code == "BLOCKED_BY_FAILED_DEP"
    assert cleanup.error and final.id in cleanup.error
    assert tm.session.status != "CANCELED"            # 会话不被误标
    assert not any(e.task_id in (final.id, cleanup.id) for e in tm._queue.peek_all())


# ── 4.5 阻塞原因的持久化与恢复后可解释 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_blocked_reason_survives_replay_and_reaches_review_face():
    bus = _CapturingBus()
    tm = _tm(bus)
    parent = Task(id="P", session_id="s1", status="SUSPENDED", title="parent")
    tm.register_task(parent)
    _, ordered = await _delegate_plan_ordered(tm, parent, [
        {"title": "produce"},
        {"title": "final"},
    ])
    produce, final = ordered

    produce.status = "FAILED"
    await tm.on_task_finished(produce.id, status="FAILED")
    assert final.status == "CANCELED"

    # 模拟重启：回放全部事件（含阻塞取消的 TASK_CANCELED）→ 投影可解释
    view = reduce_events(bus.events, "run_1")
    fv = view.tasks[final.id]
    assert fv.status == "CANCELED"
    assert fv.error_code == "BLOCKED_BY_FAILED_DEP"
    assert fv.blocked_by_task_id == produce.id
    recovered_final = task_from_projection(fv)
    assert recovered_final.error_code == "BLOCKED_BY_FAILED_DEP"

    # 恢复后的 TM：终态与阻塞原因仍可读（父观察面的 note 数据源）
    restored_tm = _tm(_CapturingBus())
    all_tasks = [task_from_projection(v) for v in view.tasks.values()]
    terminal = {tid for tid, v in view.tasks.items()
                if v.status in ("FINISHED", "FAILED", "CANCELED")}
    restored_tm.restore(all_tasks, terminal)
    child = restored_tm.get_task(final.id)
    assert child.error_code == "BLOCKED_BY_FAILED_DEP"
    assert child.error and produce.id in child.error
