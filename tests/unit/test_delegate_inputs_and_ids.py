"""spec: task-handoff——delegate 工具的 inputs 消费、run_if 声明与 ack 回传 id。"""

from __future__ import annotations

import pytest

from ctx_weft.core.capabilities.control_tools import ControlContext, delegate_plan, delegate_task
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InProcessEventBus


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


# ── 2.1 delegate_task ────────────────────────────────────────────────────────


def test_delegate_task_inputs_reach_child_and_ack_carries_id():
    tm = _FakeTM()
    res = delegate_task(title="load", task_prompt="p", inputs={"file": "a.csv"}, ctx=_ctx(tm))
    assert len(tm.staged) == 1
    child = tm.staged[0]
    assert child.inputs == {"file": "a.csv"}
    # 回执三处一致：ack 内出现的 id 就是 staged 子任务的 id
    assert f"(task_id: {child.id})" in res.content


def test_delegate_task_without_inputs_leaves_child_inputs_none():
    tm = _FakeTM()
    delegate_task(title="load", task_prompt="p", ctx=_ctx(tm))
    assert tm.staged[0].inputs is None


def test_delegate_task_invalid_inputs_rejects_without_dispatch():
    tm = _FakeTM()
    res = delegate_task(title="load", task_prompt="p", inputs={"cb": object()}, ctx=_ctx(tm))
    assert tm.staged == []
    assert res.content.startswith("Cannot delegate sub-task")
    assert "JSON" in res.content


def test_delegate_task_normalizes_oversized_inputs():
    tm = _FakeTM()
    delegate_task(title="load", task_prompt="p",
                  inputs={"values": list(range(10_000))}, ctx=_ctx(tm))
    child = tm.staged[0]
    assert "_truncated" in child.inputs  # 规整标记进了契约


# ── 2.2 delegate_plan ────────────────────────────────────────────────────────


def test_delegate_plan_per_task_inputs_and_ordered_id_ack():
    tm = _FakeTM()
    res = delegate_plan(tasks=[
        {"title": "a", "inputs": {"file": "a.csv"}},
        {"title": "b", "inputs": {"file": "b.csv"}},
        {"title": "c"},  # 无 inputs
    ], ctx=_ctx(tm))
    assert len(tm.staged) == 3
    assert tm.staged[0].inputs == {"file": "a.csv"}
    assert tm.staged[1].inputs == {"file": "b.csv"}
    assert tm.staged[2].inputs is None
    # ack 的 id 列表与 spec 顺序一致
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


def test_delegate_plan_invalid_inputs_rejects_whole_plan():
    tm = _FakeTM()
    res = delegate_plan(tasks=[
        {"title": "a"},
        {"title": "b", "inputs": {"cb": object()}},
    ], ctx=_ctx(tm))
    assert tm.staged == []
    assert res.content.startswith("Cannot delegate plan")


# ── 契约 → TASK_CREATED payload（真 TM）──────────────────────────────────────


async def _push_and_capture(task: Task) -> dict:
    bus = InProcessEventBus()
    events: list = []

    async def _sink(e) -> None:
        events.append(e)

    bus.subscribe(None, _sink)
    tm = TaskManager(session_id="s1", event_bus=bus)
    tm.set_session(Session(id="s1", user_prompt="", status="RUNNING"))
    await tm.push_task(task)
    ev = next(e for e in events if e.type == EventType.TASK_CREATED)
    return ev.payload["task"]


@pytest.mark.asyncio
async def test_push_task_emits_inputs_in_created_payload():
    payload = await _push_and_capture(Task(
        id="c1", session_id="s1", status="PENDING", title="x",
        inputs={"file": "a.csv"},
    ))
    assert payload["inputs"] == {"file": "a.csv"}


@pytest.mark.asyncio
async def test_push_task_emits_null_inputs_for_legacy_shape():
    """无输入任务显式落 None（读侧按「未声明」解释，不虚构）。"""
    payload = await _push_and_capture(Task(
        id="c1", session_id="s1", status="PENDING", title="x",
    ))
    assert payload["inputs"] is None
