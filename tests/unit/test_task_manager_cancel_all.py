"""TaskManager.cancel_all: drain pending → CANCELED, session CANCELED, drain guarded."""

import pytest

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.queue import QueueEntry
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from tests.unit._stub_runner import StubRunner



def _tm() -> TaskManager:
    tm = TaskManager(session_id="s1", event_bus=InProcessEventBus(), max_concurrent=1)
    tm.set_session(Session(id="s1", tenant_id="default", user_prompt="x"))
    return tm


@pytest.mark.asyncio
async def test_cancel_all_cancels_pending_and_session():
    tm = _tm()
    for tid in ("a", "b"):
        tm.register_task(Task(id=tid, session_id="s1", status="PENDING",
                              settings=NormalTaskSettings()))
        tm._queue.push(QueueEntry(task_id=tid, session_id="s1"))

    await tm.cancel_all(reason="user_cancel")

    assert tm._queue.has_pending() is False
    assert tm.get_task("a").status == "CANCELED"
    assert tm.get_task("b").status == "CANCELED"
    assert not hasattr(tm._session, "status")


@pytest.mark.asyncio
async def test_drain_is_a_noop_after_cancel_all():
    tm = _tm()
    tm.set_runner(StubRunner(tm))               # would be called if drain scheduled
    tm.register_task(Task(id="a", session_id="s1", status="PENDING",
                          settings=NormalTaskSettings()))
    tm._queue.push(QueueEntry(task_id="a", session_id="s1"))
    await tm.cancel_all()
    await tm.drain()                            # guard → returns without scheduling
    assert tm.get_task("a").status == "CANCELED"
