"""spec: task-handoff——delegate 工具的 run_if 声明与 ack 回传子任务 id。"""

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
    # ack 里出现的 id 就是 staged 子任务的 id——模型后续 task_reviews 的稳定句柄。
    assert f"(task_id: {tm.staged[0].id})" in res.content


# ── delegate_plan ────────────────────────────────────────────────────────────


def test_delegate_plan_ack_carries_ids_in_spec_order():
    tm = _FakeTM()
    res = delegate_plan(tasks=[{"title": "a"}, {"title": "b"}, {"title": "c"}], ctx=_ctx(tm))
    assert len(tm.staged) == 3
    ids = [c.id for c in tm.staged]
    assert ", ".join(ids) in res.content


def test_delegate_plan_run_if_materialized_into_dep_conditions():
    tm = _FakeTM()
    delegate_plan(tasks=[
        {"title": "a"},
        {"title": "b"},                      # 缺省 success
        {"title": "cleanup", "run_if": "any"},
    ], ctx=_ctx(tm))
    a, b, c = tm.staged
    assert b.dep_conditions == {a.id: "success"}
    assert c.dep_conditions == {b.id: "any"}
    assert a.dep_conditions is None  # 首任务无前序


def test_delegate_plan_invalid_run_if_rejects_whole_plan():
    tm = _FakeTM()
    res = delegate_plan(tasks=[
        {"title": "a"},
        {"title": "b", "run_if": "whenever"},
    ], ctx=_ctx(tm))
    assert tm.staged == []
    assert "invalid run_if" in res.content
