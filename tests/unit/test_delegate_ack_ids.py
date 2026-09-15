"""spec: task-handoff——delegate 工具的 ack 回传子任务 id（模型后续操作的稳定句柄）。"""

from __future__ import annotations

from ctx_weft.core.capabilities.control_tools import ControlContext, delegate_plan, delegate_task
from ctx_weft.core.models.task import Task


class _FakeTM:
    def __init__(self) -> None:
        self.staged: list[Task] = []

    def stage_task(self, child: Task, **kwargs) -> None:
        self.staged.append(child)

    def get_task(self, tid: str):
        return None


def _ctx(tm: _FakeTM) -> ControlContext:
    return ControlContext(
        session_id="s1", task_id="p1", agent_id="a1",
        task=Task(id="p1", session_id="s1", status="ACTIVE", title="P"),
        task_manager=tm, session=None, tool_call_id="tc_1",
    )


# ── delegate_task ────────────────────────────────────────────────────────────


def test_delegate_task_ack_carries_child_id():
    tm = _FakeTM()
    res = delegate_task(title="load", task_prompt="p", ctx=_ctx(tm))
    assert len(tm.staged) == 1
    # ack 同时给标题与 id：id 是后续 task_reviews 的稳定句柄，标题让模型对得上刚派的是哪个。
    child = tm.staged[0]
    assert f"{child.title!r} ({child.id})" in res.content


# ── delegate_plan ────────────────────────────────────────────────────────────


def test_delegate_plan_ack_carries_ids_in_spec_order():
    tm = _FakeTM()
    res = delegate_plan(tasks=[{"title": "a"}, {"title": "b"}, {"title": "c"}], ctx=_ctx(tm))
    assert len(tm.staged) == 3
    for i, c in enumerate(tm.staged, start=1):
        assert f"{i}. {c.title!r} ({c.id})" in res.content


def test_delegate_plan_chains_each_task_on_the_previous():
    """plan 的边即「按顺序做」：每一步 blocked_by 前一步（放行条件是前序成功）。"""
    tm = _FakeTM()
    delegate_plan(tasks=[{"title": "a"}, {"title": "b"}, {"title": "c"}], ctx=_ctx(tm))
    a, b, c = tm.staged
    assert a.tracking_task_ids == []
    assert b.tracking_task_ids == [a.id]
    assert c.tracking_task_ids == [a.id, b.id]
