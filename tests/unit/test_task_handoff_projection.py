"""spec: task-handoff——inputs / dep_conditions / 阻塞取消结局码的投影往返。

覆盖：TASK_CREATED 携带新字段折入 TaskView；TASK_CANCELED 的 error_code 与
blocked_by_task_id 回填；存量事件（无新键）不炸且语义缺省；converter 回填 Task。
"""

from __future__ import annotations

from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.events import Event, EventType


def _ev(t: EventType, payload: dict, *, task_id: str = "tsk_1") -> Event:
    return Event(
        id=generate_id("evt"), run_id=None, sequence=0,
        session_id="sess_1", type=t, timestamp=now_utc(),
        tenant_id="default", task_id=task_id, payload=payload,
    )


def _created(extra_task: dict | None = None) -> Event:
    task: dict = {
        "id": "tsk_1", "session_id": "sess_1", "status": "PENDING", "title": "t",
        "description": "", "creator_agent_id": "", "assigned_agent_id": "",
        "parent_task_id": None, "user_prompt": "p", "priority": 0, "max_retries": 3,
        "timeout_ms": 0, "dag_deps": [], "interaction_mode": "", "settings": {},
        "origin_tool_call_id": None, "origin_tool_name": None,
        "result": None, "outputs": {}, "error": None,
        "created_at": None, "updated_at": None,
    }
    if extra_task:
        task.update(extra_task)
    return _ev(EventType.TASK_CREATED, {"task": task})


# ── TASK_CREATED：inputs / dep_conditions 折入 ────────────────────────────────


def test_created_folds_inputs_and_dep_conditions():
    view = reduce_events([
        _created({
            "inputs": {"file": "a.csv", "mode": "strict"},
            "dag_deps": ["tsk_0"],
            "dep_conditions": {"tsk_0": "success"},
        }),
    ], "run_1")
    t = view.tasks["tsk_1"]
    assert t.inputs == {"file": "a.csv", "mode": "strict"}
    assert t.dep_conditions == {"tsk_0": "success"}
    # converter 回填 Task
    task = task_from_projection(t)
    assert task.inputs == {"file": "a.csv", "mode": "strict"}
    assert task.dep_conditions == {"tsk_0": "success"}


def test_created_without_new_keys_keeps_legacy_defaults():
    """存量事件形态：无 inputs / dep_conditions 键 → None（不炸、不虚构）。"""
    view = reduce_events([_created()], "run_1")
    t = view.tasks["tsk_1"]
    assert t.inputs is None
    assert t.dep_conditions is None
    task = task_from_projection(t)
    assert task.inputs is None
    assert task.dep_conditions is None


def test_created_with_null_values_keeps_legacy_defaults():
    """新代码显式落 None（无依赖/无输入的任务）同样回到缺省语义。"""
    view = reduce_events([
        _created({"inputs": None, "dep_conditions": None}),
    ], "run_1")
    t = view.tasks["tsk_1"]
    assert t.inputs is None
    assert t.dep_conditions is None


# ── TASK_CANCELED：阻塞取消的结局码与阻塞源 ──────────────────────────────────


def test_canceled_folds_blocked_reason_into_view():
    view = reduce_events([
        _created(),
        _ev(EventType.TASK_CANCELED, {
            "error_code": "BLOCKED_BY_FAILED_DEP",
            "blocked_by_task_id": "tsk_0",
            "reason": "dependency_failed",
        }),
    ], "run_1")
    t = view.tasks["tsk_1"]
    assert t.status == "CANCELED"
    assert t.error_code == "BLOCKED_BY_FAILED_DEP"
    assert t.blocked_by_task_id == "tsk_0"
    assert t.error == "dependency_failed"
    # converter 回填 Task（阻塞源留在投影，error/error_code 进 Task）
    task = task_from_projection(t)
    assert task.error_code == "BLOCKED_BY_FAILED_DEP"
    assert task.error == "dependency_failed"


def test_plain_canceled_folds_reason_without_error_code():
    """用户侧/弃子取消：无 error_code → 只折 reason，error_code 保持 None。"""
    view = reduce_events([
        _created(),
        _ev(EventType.TASK_CANCELED, {"reason": "discarded_before_first_chunk"}),
    ], "run_1")
    t = view.tasks["tsk_1"]
    assert t.error_code is None
    assert t.blocked_by_task_id is None
    assert t.error == "discarded_before_first_chunk"
