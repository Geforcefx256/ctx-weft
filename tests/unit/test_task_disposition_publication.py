"""Lifecycle notifications expose the committed task state to live queries."""

import pytest

from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus



def setup(*, retry_count=0):
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    tm = TaskManager("s1", event_bus=bus, max_concurrent=0)
    task = Task(id="t1", session_id="s1", status="ACTIVE", retry_count=retry_count)
    tm.register_task(task)
    return tm, task, bus, store


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,retries,status,count", [
    (RunOutcome(RunOutcomeKind.COMPLETED, verdict="success"), 0, "FINISHED", 0),
    (RunOutcome(RunOutcomeKind.COMPLETED, verdict="fail"), 0, "FAILED", 0),
    (RunOutcome(RunOutcomeKind.COMPLETED, verdict="continue"), 0, "PENDING", 0),
    (RunOutcome(RunOutcomeKind.COMPLETED, verdict="continue"), 3, "PENDING", 3),
    (RunOutcome(RunOutcomeKind.INTERRUPTED, error="connection lost", error_code="LLM_OUTAGE"),
     0, "INTERRUPTED", 0),
    (RunOutcome(RunOutcomeKind.AWAITING_HUMAN, hitl_id="h1"), 0, "AWAITING_HUMAN", 0),
    (RunOutcome(RunOutcomeKind.SUSPENDED_ON_CHILDREN), 0, "SUSPENDED", 0),
])
async def test_subscriber_queries_new_state_only_after_commit(outcome, retries, status, count):
    tm, task, bus, store = setup(retry_count=retries)
    before_commit, notified = [], []
    append = store.append_batch

    async def append_batch(sid, bid, events):
        before_commit.append((task.status, task.retry_count))
        return await append(sid, bid, events)

    store.append_batch = append_batch

    async def query(event):
        rows = await store.read_range("s1")
        assert any(row.event.id == event.id for row in rows)
        current = tm.get_task(event.task_id)
        notified.append((current.status, current.retry_count, current.error, current.error_code))

    bus.subscribe(None, query)
    assert await tm.apply_run_outcome(task.id, outcome) == status
    assert before_commit == [("ACTIVE", retries)]
    code = outcome.error_code or ("TASK_FAILED_BY_OBSERVER" if status == "FAILED" else None)
    assert notified == [(status, count, outcome.error or None, code)]


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_entry", ["apply", "reconcile"])
@pytest.mark.parametrize("retry_count", [0, 2])
@pytest.mark.parametrize("kind", [RunOutcomeKind.COMPLETED, RunOutcomeKind.INTERRUPTED])
async def test_notification_retry_preserves_the_committed_budget(retry_entry, retry_count, kind):
    tm, task, bus, store = setup(retry_count=retry_count)
    expected_count = retry_count + (kind is RunOutcomeKind.INTERRUPTED)
    attempts = []

    async def notify(event):
        attempts.append((event.id, event.payload["retry_count"], task.status, task.retry_count))
        if len(attempts) == 1:
            raise RuntimeError("required subscriber unavailable after commit")

    bus.subscribe(EventType.TASK_REQUEUED, notify, provisional=True, required=True)
    outcome = RunOutcome(kind, verdict="continue", retriable=True)
    with pytest.raises(RuntimeError, match="required subscriber unavailable"):
        await tm.apply_run_outcome(task.id, outcome)
    assert task.status == "PENDING" and task.retry_count == expected_count

    if retry_entry == "apply":
        await tm.apply_run_outcome(task.id, outcome)
    else:
        await tm.reconcile_stopped_tasks()

    assert attempts == [(attempts[0][0], expected_count, "PENDING", expected_count)] * 2
    assert len(await store.read_range("s1")) == 1
    assert task.retry_count == expected_count
    assert not tm._pending_emissions


@pytest.mark.asyncio
async def test_manager_without_bus_still_publishes_disposition():
    tm = TaskManager("s1")
    task = Task(id="t1", session_id="s1", status="ACTIVE")
    tm.register_task(task)
    await tm.apply_run_outcome(task.id, RunOutcome(RunOutcomeKind.COMPLETED, verdict="continue"))
    assert (task.status, task.retry_count) == ("PENDING", 0)


@pytest.mark.asyncio
async def test_failed_commit_does_not_publish_failure_metadata_and_retry_keeps_its_count():
    tm, task, bus, store = setup(retry_count=3)
    append = store.append_batch
    attempts = []

    async def fail_once(sid, bid, events):
        attempts.append((bid, events[0].id))
        if len(attempts) == 1:
            raise OSError("storage offline")
        return await append(sid, bid, events)

    store.append_batch = fail_once
    outcome = RunOutcome(RunOutcomeKind.COMPLETED, verdict="fail", error="actual cause")
    with pytest.raises(PersistenceUnavailableError):
        await tm.apply_run_outcome(task.id, outcome)
    assert (task.status, task.error_code, task.failure_retry_count) == ("ACTIVE", None, None)
    await tm.apply_run_outcome(task.id, outcome)
    assert task.error == "actual cause" and task.error_code == "TASK_FAILED_BY_OBSERVER"
    assert task.failure_retry_count == 3 and task.retry_count == 3
    assert attempts[0] == attempts[1] and len(await store.read_range("s1")) == 1
