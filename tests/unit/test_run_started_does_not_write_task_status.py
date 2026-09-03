"""`RunStarted`（run 域）不得写 task 状态标量。

T5 为「每个 run_id 都有起止」这条不变量，给 recap / recognize_intent / compact 这些
观测型 run 也补了 `RunStarted`。但 `_apply` 里 `RUN_STARTED` 分支仍写
`view.task_status = "ACTIVE"`——纯文本冷 park 场景下事件序变成
`TaskAwaitingHuman → RunStarted(recognize_intent) → RunStarted(recap)`，
折叠结果 `view.task_status == "ACTIVE"` 而 `view.tasks[tid].status == "AWAITING_HUMAN"`，
标量说在跑、逐 task 说在等人，投影自相矛盾。

`ACTIVE` 的正主是 `TASK_STARTED`（`TASK_STATUS_BY_EVENT`，`reducers.py:50`，TM 派发时发）。
`RUN_STARTED` 这行是冗余，且在观测型 run 场景下是错的——对照
`test_run_interrupted_split.py::test_run_interrupted_alone_does_not_write_task_status`，
`RUN_INTERRUPTED` 早已被剥夺这个权力，`RUN_STARTED` 是漏网的那个。
"""

from __future__ import annotations

from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.core.utils import now_utc
from ctx_weft.protocols.events import Event, EventType


def _ev(t: str, payload: dict, seq: int) -> Event:
    return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id="s1",
                 type=t, timestamp=now_utc(), task_id="tsk_1", payload=payload)


def test_run_started_after_awaiting_human_does_not_flip_task_status_to_active() -> None:
    """观测型 run（recap / recognize_intent）在冷 park 场景里补发的 RunStarted，
    不能把已经 AWAITING_HUMAN 的 task 拉回 ACTIVE——标量与逐 task 视图必须一致。
    """
    events = [
        _ev(EventType.TASK_CREATED, {"task": {"id": "tsk_1", "session_id": "s1",
                                              "status": "PENDING"}}, 1),
        _ev(EventType.TASK_STARTED, {"assigned_agent_id": "agt_1"}, 2),
        _ev(EventType.TASK_AWAITING_HUMAN, {}, 3),
        _ev(EventType.RUN_STARTED, {}, 4),
    ]
    view = reduce_events(events, run_id="run_1")

    assert view.task_status == "AWAITING_HUMAN"
    assert view.tasks["tsk_1"].status == "AWAITING_HUMAN"


def test_run_started_alone_does_not_write_task_status() -> None:
    """run 域的事实不写 task 状态——对照
    `test_run_interrupted_alone_does_not_write_task_status` 的形状。
    """
    events = [
        _ev(EventType.TASK_CREATED, {"task": {"id": "tsk_1", "session_id": "s1",
                                              "status": "PENDING"}}, 1),
        _ev(EventType.RUN_STARTED, {}, 2),
    ]
    view = reduce_events(events, run_id="run_1")

    assert view.task_status == "UNKNOWN"
    assert view.tasks["tsk_1"].status == "PENDING"
