"""重开会话复用 owner。旧轮的熔断与派发状态不能带入新轮。"""

from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.errors import UnfinishedTasksError
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import EventPersister, InMemoryEventStore, InProcessEventBus
from tests._event_helpers import all_events
from tests.unit._stub_runner import StubRunner


async def _owner(threshold: int = 1):
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    EventPersister(store, bus)
    await bus.emit(new_event(
        EventType.SESSION_CREATED, session_id="s1", tenant_id="default", origin="test",
        payload={"user_prompt": "old", "root_agent_id": "agt"},
    ))
    registry = SessionRegistry(agent_lifecycle_manager=None, event_bus=bus)
    session = Session(id="s1", user_prompt="old", root_agent_id="agt",
                      failure_threshold=threshold)
    owner = registry._new_task_manager(session)
    owner.set_runner(StubRunner(owner))
    root = Task(id="old", session_id="s1", status="PENDING", title="old task",
                assigned_agent_id="agt")
    await owner.push_task(root)
    return registry, store, owner, root


async def _fail_without_dispatch(owner, root):
    owner._queue.cancel(root.id)
    root.error = "old failure"
    status = await owner.apply_run_outcome(
        root.id, RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="fail", error="old failure"))
    await owner.on_task_finished(root.id, status=status)


@pytest.mark.parametrize("threshold", [1, 5])
@pytest.mark.parametrize("recovery_pending", [False, True])
async def test_reopen_resets_failure_state_and_dispatches_on_the_same_owner(
    threshold, recovery_pending,
):
    registry, store, owner, old_root = await _owner(threshold)
    await _fail_without_dispatch(owner, old_root)
    owner.set_recovery_pending(recovery_pending)

    session, root, reopened = await registry.start_next_turn(
        "s1", store, user_prompt="new", task_manager=owner)

    assert reopened is owner
    assert owner.get_task(old_root.id) is old_root and old_root.status == "FAILED"
    assert session.failure_counter == 0
    assert not owner.is_cancelled() and not owner._threshold_tripped
    assert not owner.recovery_pending
    assert owner._pending_threshold_roots == set() and owner._threshold_commit is None
    assert owner._recent_failures == []
    projected = (await rebuild_view(store, "s1")).sessions["s1"]
    assert projected.failure_counter == 0 and not projected.threshold_tripped

    # Dispatch and trip again to verify both the new breaker and its failure list.
    ran = []

    async def fail_new_round(_sid, task_id):
        ran.append(task_id)
        owner.get_task(task_id).error = "new failure"
        return RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="fail", error="new failure")

    session.failure_threshold = 1
    owner.set_runner(StubRunner(owner, fail_new_round))
    await owner.drain()
    await asyncio.wait_for(owner._dispatch_tasks[root.id], timeout=2)

    assert ran == [root.id]
    assert root.status == "FAILED" and owner._threshold_tripped
    assert session.failure_counter == 1
    projected = (await rebuild_view(store, "s1")).sessions["s1"]
    assert projected.failure_counter == 1 and projected.threshold_tripped
    hits = [ev for ev in await all_events(store, "s1")
            if ev.type == EventType.FAILURE_THRESHOLD_HIT]
    assert len(hits) == (2 if threshold == 1 else 1)
    assert len(hits[-1].payload["failures"]) == 1
    assert hits[-1].payload["failures"][0]["reason"] == "new failure"


async def test_reopen_refusal_preserves_the_existing_breaker():
    registry, store, owner, old_root = await _owner()
    await _fail_without_dispatch(owner, old_root)
    owner.register_task(Task(id="unfinished", session_id="s1", status="PENDING"))
    old_session = owner.session
    failures = list(owner._recent_failures)

    with pytest.raises(UnfinishedTasksError):
        await registry.start_next_turn("s1", store, user_prompt="new", task_manager=owner)

    assert owner.session is old_session and old_session.failure_counter == 1
    assert owner.is_cancelled() and owner._threshold_tripped
    assert owner._recent_failures == failures
    assert not any(ev.type == EventType.SESSION_RESUMED
                   for ev in await all_events(store, "s1"))


async def test_reopen_waits_for_old_terminal_dispatch_before_resetting():
    registry, store, owner, old_root = await _owner()
    entered, release = asyncio.Event(), asyncio.Event()

    async def finish_old_threshold(_root, _acks, _failures):
        entered.set()
        await release.wait()

    async def fail_old_round(_sid, _tid):
        return RunOutcome(kind=RunOutcomeKind.COMPLETED, verdict="fail", error="old failure")

    owner.set_hooks(TaskManagerHooks(threshold_finalizer=finish_old_threshold))
    owner.set_runner(StubRunner(owner, fail_old_round))
    old_session = owner.session
    await owner.drain()
    await asyncio.wait_for(entered.wait(), timeout=2)
    assert old_root.status == "FAILED"  # Terminal fact published; settlement still running.

    reopening = asyncio.create_task(registry.start_next_turn(
        "s1", store, user_prompt="new", task_manager=owner))
    try:
        await asyncio.sleep(0)
        assert not reopening.done()
        assert owner.session is old_session and owner._threshold_tripped
    finally:
        release.set()
        session, root, reopened = await asyncio.wait_for(reopening, timeout=2)

    assert reopened is owner and session.failure_counter == 0
    assert not owner.is_cancelled() and not owner._threshold_tripped
    assert owner.get_task(root.id) is root
