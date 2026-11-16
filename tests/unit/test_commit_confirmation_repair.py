"""Unknown receipts preserve batch identity and sufficient cold-recovery facts."""

import asyncio

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError
from ctx_weft.providers.events.bus.in_process.bus import InProcessEventBus
from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.unit.test_runtime_initialization_queries import append, container, runtime



@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_lost", [False, True])
@pytest.mark.parametrize("retry_kind,expected_count", [("continue", 0), ("interrupted", 1)])
async def test_stopped_requeue_submission_preserves_committed_budget(receipt_lost, retry_kind, expected_count):
    from ctx_weft.core.models.task import Task
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
    from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    tm = TaskManager("s1", event_bus=bus, max_concurrent=0)
    task = Task(id="t1", session_id="s1", status="ACTIVE", max_retries=3)
    await tm.push_task(task)
    tm._queue.cancel(task.id)
    tm._running_tasks.add(task.id)
    tm.set_runner(object())
    async def read(tid):
        return (await rebuild_view(store, "s1")).tasks.get(tid)
    tm.set_hooks(TaskManagerHooks(read_task_state=read))
    notified = []
    async def observe_requeue(event):
        notified.append((task.status, task.retry_count))
    bus.subscribe(EventType.TASK_REQUEUED, observe_requeue)
    original = store.append_batch
    async def append_batch(sid, bid, events):
        if any(e.type == EventType.TASK_REQUEUED for e in events):
            if receipt_lost:
                await original(sid, bid, events)
            raise OSError("requeue receipt")
        return await original(sid, bid, events)
    store.append_batch = append_batch
    with pytest.raises(PersistenceUnavailableError):
        await tm.apply_run_outcome(task.id, RunOutcome(
            kind=RunOutcomeKind.COMPLETED if retry_kind == "continue" else RunOutcomeKind.INTERRUPTED,
            verdict="continue", retriable=True))
    assert task.retry_count == 0
    assert task.status == "ACTIVE" and not notified
    store.append_batch = original
    await tm.reconcile_stopped_tasks()
    assert task.retry_count == expected_count and task.status == "PENDING"
    assert notified == [("PENDING", expected_count)]
    assert not tm.running_task_ids()
    assert [e.task_id for e in tm._queue.peek_all()] == [task.id]
    head = await store.committed_head("s1")
    await tm.reconcile_stopped_tasks()
    assert task.retry_count == expected_count
    assert await store.committed_head("s1") == head


@pytest.mark.asyncio
async def test_uncertain_batch_is_immutable_and_concurrent_retries_notify_once():
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    seen = []
    async def collect(event):
        seen.append(event)
    bus.subscribe(None, collect)
    original = store.append_batch
    attempts = []
    async def commit(sid, bid, events):
        attempts.append((bid, [(e.id, dict(e.payload)) for e in events]))
        receipt = await original(sid, bid, events)
        if len(attempts) <= 2:
            raise OSError("lost receipt")
        return receipt
    store.append_batch = commit
    def event(kind, payload):
        return new_event(kind, session_id="s1", tenant_id="default", origin="test", task_id="t1", payload=payload)
    start = event(EventType.RUN_STARTED, {"value": "original"})
    bus.begin_provisional("t1", batch_id="fixed")
    await bus.emit(start)
    with pytest.raises(PersistenceUnavailableError):
        await bus.commit_provisional("t1")
    start.payload["value"] = "changed after commit"
    canceled = event(EventType.TASK_CANCELED, {})
    await bus.emit(canceled)
    with pytest.raises(PersistenceUnavailableError):
        await bus.commit_provisional("t1")
    await asyncio.gather(*(bus.commit_provisional("t1") for _ in range(3)))
    assert attempts[0] == attempts[1] == attempts[2]
    assert attempts[3][0] != "fixed"
    assert [(e.id, e.payload) for e in seen] == [(start.id, {"value": "original"}), (canceled.id, {})]
    assert len(await store.read_range("s1")) == 2
    assert not bus.has_pending_commit("t1") and "t1" not in bus._provisional


@pytest.mark.asyncio
async def test_discard_confirms_an_attempted_batch_instead_of_rolling_it_back():
    from ctx_weft.core.models.task import Task
    from ctx_weft.core.orchestrator.task.manager import TaskManager

    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    tm = TaskManager(session_id="s1", event_bus=bus)
    task = Task(id="t1", session_id="s1", status="ACTIVE", unattended=True)
    await tm.push_task(task, provisional=True)
    original = store.append_batch
    failed = False
    async def append_batch(sid, bid, events):
        nonlocal failed
        receipt = await original(sid, bid, events)
        if not failed and any(e.type == EventType.TASK_CREATED for e in events):
            failed = True
            raise OSError("unknown round receipt")
        return receipt
    store.append_batch = append_batch
    with pytest.raises(PersistenceUnavailableError):
        await tm.commit_round(task.id)
    await tm.discard_round(task.id)
    assert tm.get_task(task.id) is task
    assert not tm.is_round_open(task.id)
    rows = await store.read_range("s1")
    assert sum(r.event.type == EventType.TASK_CREATED for r in rows) == 1
    assert not any(r.event.type == EventType.ROUND_DISCARDED for r in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FINISHED", "CANCELED"])
@pytest.mark.parametrize("snapshot", [False, True])
async def test_startup_repairs_missing_idle_after_finalized_task(status, snapshot):
    store = InMemoryEventStore()
    old = runtime(store)
    handle = await container(old)
    kind = EventType.TASK_FINISHED if status == "FINISHED" else EventType.TASK_CANCELED
    await append(store, "s1",
        (EventType.TASK_CREATED, "done", handle.root_agent_id, {"task": {
            "id": "done", "assigned_agent_id": handle.root_agent_id}}),
        (EventType.TASK_STARTED, "done", handle.root_agent_id, {}),
        (EventType.AGENT_RUNNING, "done", handle.root_agent_id, {}),
        (kind, "done", handle.root_agent_id, {}),
        (EventType.TASK_FINALIZED, "done", handle.root_agent_id, {}))
    if snapshot:
        view = prune_view_for_snapshot(await rebuild_view(store, "s1"))
        assert view.tasks["done"].finalized
        await seed_snapshot(store, "s1", view, cut=await store.committed_head("s1"))
    fresh = runtime(store)
    await initialize_all(fresh)
    view = await rebuild_view(store, "s1")
    assert view.agents[handle.root_agent_id].status == "idle"
    head = await store.committed_head("s1")
    await initialize_all(runtime(store))
    assert await store.committed_head("s1") == head
