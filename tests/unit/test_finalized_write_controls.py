"""TaskFinalized stop admission has no deferred signal or failure-time residue."""

import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock

import pytest

from ctx_weft.core.loop.driver import make_event
from ctx_weft.core.loop.steps.observe import emit_effect
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError
from tests._initialization import initialize_all
from tests.unit.test_out_of_band_verdict import _tm
from tests.unit.test_runtime_agent_api import _plant, _rt


@pytest.mark.parametrize("failure", [PersistenceUnavailableError, asyncio.CancelledError])
@pytest.mark.parametrize("producer", ["loop", "manager"])
async def test_finalized_write_guard_releases_on_publication_failure(fake_state_ctx, failure, producer):
    state, ctx = fake_state_ctx
    tm = TaskManager(state.session.id)
    ctx.task_manager = tm
    event = make_event(state, EventType.TASK_FINALIZED, {})
    cause = failure("write interrupted")

    async def fail(current):
        assert current.type == EventType.TASK_FINALIZED and tm.is_writing_finalized(state.task.id)
        assert await tm.cancel_task_tree(state.task.id, "ignored") == []
        raise cause

    ctx.event_bus.emit = fail
    async def fail_batch(events, *, on_committed=None):
        assert len(events) == 1
        await fail(events[0])

    ctx.event_bus.emit_batch = fail_batch
    tm._event_bus = ctx.event_bus
    with pytest.raises(failure) as raised:
        if producer == "loop":
            await emit_effect(state, ctx, event)
        else:
            await tm._emit(EventType.TASK_FINALIZED, task_id=state.task.id)
    assert raised.value is cause
    assert not tm.is_writing_finalized()
    assert "TaskFinalized:" not in state.extra.get("observe_effects", [])


async def test_protected_descendant_prevents_ancestor_cancel_marker_and_callback():
    tm, bus = _tm()
    parent = Task(id="parent", session_id="s1", status="SUSPENDED")
    child = Task(id="child", session_id="s1", status="ACTIVE", parent_task_id="parent")
    tm.restore([parent, child], set(), enqueue=False)
    terminate = AsyncMock()
    with tm.finalized_write(child.id):
        with tm.finalized_write(child.id):
            async with tm._lock:
                assert await asyncio.wait_for(tm.cancel_task_tree(
                    parent.id, "ignored", before_cancel=terminate), 1) == []
            await tm.cancel_all(reason="ignored")
        assert tm.is_writing_finalized(child.id)
    terminate.assert_not_awaited()
    assert parent.status == "SUSPENDED" and child.status == "ACTIVE"
    assert parent.error is None and child.error is None
    assert not bus.events and not tm._cancelled and not tm.is_writing_finalized()
    assert set(await tm.cancel_task_tree(parent.id, "accepted", before_cancel=terminate)) == {parent.id, child.id}
    terminate.assert_awaited_once()
    if tm._background_asyncio_tasks:
        await asyncio.gather(*list(tm._background_asyncio_tasks))


@pytest.mark.parametrize("operation", ["pause_session", "cancel_session"])
async def test_session_stop_during_finalized_write_has_no_latch(operation):
    rt = _rt()
    _plant(rt, "agent", None, status="running")
    tm = rt._task_managers["s1"]
    task = Task(id="task", session_id="s1", status="ACTIVE", assigned_agent_id="agent")
    tm.register_task(task)
    rt._agent_lifecycle_manager._agents["agent"].current_task_id = task.id
    tokens = rt._register_run_tokens("s1", task.id)
    await initialize_all(rt)
    with tm.finalized_write(task.id):
        assert await getattr(rt, operation)("s1") is False
        assert not rt._cancel_run_token("s1", task.id)
    assert not tokens.pause.is_paused and not tokens.cancel.is_cancelled
    assert not tm._cancelled and "s1" not in rt._pausing and "s1" not in rt._pause_claimed
    assert rt._agent_lifecycle_manager.status_of("agent") == "running"
    # Leaving the narrow write phase restores control delivery on this run.
    assert await rt.pause_agent("agent") == ["agent"]
    assert tokens.pause.is_paused


@pytest.mark.parametrize("terminal", [False, True])
async def test_admitted_cancel_rechecks_writer_after_waiting_for_task_lock(terminal):
    tm, _ = _tm()
    task = Task(id="task", session_id="s1", status="FINISHED" if terminal else "PENDING",
                assigned_agent_id="root")
    tm.register_task(task)
    tm.begin_cancel()
    terminate = AsyncMock()
    cancel = None
    try:
        with ExitStack() as writers:
            async with tm._lock:
                cancel = asyncio.create_task(tm.cancel_session_agent_tasks(
                    "root", "cancel_agent:session_canceled", before_cancel=terminate, _admitted=True))
                await asyncio.sleep(0)
                writers.enter_context(tm.finalized_write(task.id))
            await asyncio.sleep(0)
            assert not cancel.done() and not tm._lock.locked()
            terminate.assert_not_awaited()
            assert await tm.cancel_task_tree(task.id, "new request") == []
            late_terminate = AsyncMock()
            assert await tm.cancel_session_agent_tasks("root", "new request", before_cancel=late_terminate) == []
            late_terminate.assert_not_awaited()
        assert await asyncio.wait_for(asyncio.shield(cancel), 1) == ([] if terminal else [task.id])
        terminate.assert_awaited_once()
        assert task.status == ("FINISHED" if terminal else "CANCELED")
        if tm._background_asyncio_tasks:
            await asyncio.gather(*list(tm._background_asyncio_tasks))
    finally:
        if cancel is not None:
            cancel.cancel()
            await asyncio.gather(cancel, return_exceptions=True)


async def test_publication_failure_wakes_all_joiners_after_last_nested_writer():
    tm = TaskManager("s1")
    jobs = []
    try:
        with pytest.raises(RuntimeError, match="publication failed"):
            with tm.finalized_write("second"):
                with tm.finalized_write("first"):
                    jobs = [asyncio.create_task(tm.wait_for_finalized_writes()) for _ in range(2)]
                    await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert all(not job.done() for job in jobs)
                raise RuntimeError("publication failed")
        async with tm._lock:
            await asyncio.wait_for(asyncio.gather(*jobs), 1)
        assert not tm.is_writing_finalized()
        # A later write must establish a fresh wait even though an earlier one
        # already signaled completion.
        with tm.finalized_write("next"):
            jobs.append(asyncio.create_task(tm.wait_for_finalized_writes()))
            await asyncio.sleep(0)
            assert not jobs[-1].done()
        await asyncio.wait_for(jobs[-1], 1)
    finally:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
