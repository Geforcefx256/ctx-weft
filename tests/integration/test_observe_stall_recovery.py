
from tests._recovery import recover_fixture_session
"""Public recovery of a foreground report must preserve its parked task."""

import asyncio
from dataclasses import asdict, replace

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.loop.steps.observe import ObservationResult
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.events import EventType
from tests.integration.test_observe_waiting_continue import quiescent, start


@pytest.mark.parametrize("saved_completion", [False, True])
async def test_recover_foreground_report_preserves_waiting_task_and_actor_slot(saved_completion):
    rt, llm, handle = await start(["waiting", "waiting"])
    task = await quiescent(rt, handle)
    tm = rt._task_managers[handle.session_id]
    pending_hitl, = rt.hitl_registry.list_pending(handle.session_id)
    rows = await rt.event_store.read_range(handle.session_id)
    opened = next(r.event for r in rows if r.event.type == EventType.HITL_OPENED
                  and r.event.payload.get("observe_request"))
    intent = {**opened.payload["observe_request"], "observe_id": generate_id("obs"),
              "execution_mode": "foreground", "judge_task": False, "boundary": "interrupt"}
    requested = replace(opened, id=generate_id("evt"), run_id=generate_id("run"), sequence=1,
                        type=EventType.OBSERVE_REQUESTED, payload=intent)
    await rt._event_bus.emit(requested)
    if saved_completion:
        await rt._event_bus.emit(replace(requested, id=generate_id("evt"), sequence=2,
            type=EventType.OBSERVE_COMPLETED, payload={
                "task_id": task.id, "observe_id": intent["observe_id"],
                "accepted": True, "action": "recap", "short_segment": False,
                "result": asdict(ObservationResult(act_recap="saved report")),
                "completed_at": requested.timestamp.isoformat()}))

    try:
        await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
        async with asyncio.timeout(2):
            while tm._background_asyncio_tasks or tm._dispatch_tasks:
                await asyncio.sleep(0.01)
        assert task.status == "AWAITING_HUMAN"
        assert not tm.running_task_ids()
        assert llm.actor_count == 1
        assert rt.hitl_registry.list_pending(handle.session_id) == [pending_hitl]
        view = await rebuild_view(rt.event_store, handle.session_id)
        assert not view.pending_observe and not view.open_runs
        rows = await rt.event_store.read_range(handle.session_id)
        request_position = next(row.position for row in rows if row.event.id == requested.id)
        assert not any(r.event.type == EventType.TASK_STARTED and r.position > request_position
                       for r in rows)
    finally:
        jobs = list(tm._background_asyncio_tasks | set(tm._dispatch_tasks.values()))
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
