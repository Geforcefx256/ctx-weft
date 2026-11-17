"""Judging review ordinals survive retries, snapshot cuts and existing logs."""
import asyncio
from copy import copy
from dataclasses import replace

import pytest

from ctx_weft.core.control.converters import task_from_projection
from ctx_weft.core.control.reducers import (
    apply_events,
    deserialize_view,
    read_task_view,
    rebuild_view,
    reduce_events,
    serialize_view,
)
from ctx_weft.core.control.startup_view import rebuild_startup_view
from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.loop.driver import ExecutionMode, StepDriver, StepExecution, make_event
from ctx_weft.core.loop.steps.observe import begin
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus
from tests._snapshot_helpers import seed_snapshot


async def setup_owner(state, ctx):
    store, bus = InMemoryEventStore(), InProcessEventBus()
    bus.attach_commit_gate(CommitGate(store))
    await bus.emit(make_event(state, EventType.SESSION_CREATED, {"root_agent_id": state.agent.id}))
    tm = TaskManager("s1", event_bus=bus, max_concurrent=0)
    state.task = Task(id="t1", session_id="s1", status="ACTIVE")
    await tm.push_task(state.task)
    ctx.event_bus, ctx.task_manager = bus, tm
    return store, bus, tm


@pytest.mark.parametrize("mode", list(ExecutionMode))
async def test_new_reviews_count_once_and_report_only_does_not(fake_state_ctx, mode):
    state, ctx = fake_state_ctx
    store, _, _ = await setup_owner(state, ctx)
    for judging, expected in [(True, 1), (False, 1), (True, 2)]:
        ctx.execution = StepExecution(mode, {"judge_task": judging})
        await begin(state, ctx, "plain_text", judging)
        head = await store.committed_head("s1")
        index = ctx.execution.options.get("observe_index")
        for _ in range(3):
            await begin(state, ctx, "plain_text", judging)
        assert state.task.observe_count == expected
        assert index == (expected if judging else None)
        assert await store.committed_head("s1") == head
    assert (await rebuild_view(store, "s1")).tasks["t1"].observe_count == 2


@pytest.mark.parametrize("receipt_lost", [False, True])
async def test_unknown_start_confirms_original_envelope_without_counting_twice(fake_state_ctx, receipt_lost):
    state, ctx = fake_state_ctx
    store, _, _ = await setup_owner(state, ctx)
    original, attempts = store.append_batch, []

    async def fail_start(sid, bid, events):
        if events[0].type == EventType.OBSERVE_STARTED:
            attempts.append((bid, events[0]))
            if len(attempts) == 1:
                if receipt_lost:
                    await original(sid, bid, events)
                raise OSError("unknown Started receipt")
        return await original(sid, bid, events)

    store.append_batch = fail_start
    with pytest.raises(PersistenceUnavailableError):
        await begin(state, ctx, "plain_text", True)
    assert state.task.observe_count == 0
    assert not ctx.execution.options.get("started")
    assert ctx.execution.options["observe_index"] == 1
    if receipt_lost:
        # Simulates a crash before the in-memory on_committed callback.
        assert (await rebuild_view(store, "s1")).tasks["t1"].observe_count == 1
    await begin(state, ctx, "plain_text", True)
    await begin(state, ctx, "plain_text", True)
    assert attempts[0] == attempts[1]
    assert state.task.observe_count == 1
    starts = await store.read_range("s1", include_types=(EventType.OBSERVE_STARTED,))
    assert len(starts) == 1 and starts[0].event.payload["observe_index"] == 1


async def test_overlapping_reviews_read_owner_count_instead_of_context_copy(fake_state_ctx):
    state, ctx = fake_state_ctx
    store, _, tm = await setup_owner(state, ctx)
    ctx.steps = StepDriver({})
    other_state = replace(state, task=copy(state.task), extra={})
    other_ctx = replace(ctx, execution=StepExecution(ExecutionMode.BACKGROUND))
    await asyncio.gather(begin(state, ctx, "plain_text", True),
                         begin(other_state, other_ctx, "plain_text", True))
    assert tm.get_task("t1").observe_count == 2
    assert {ctx.execution.options["observe_index"], other_ctx.execution.options["observe_index"]} == {1, 2}
    assert (await rebuild_view(store, "s1")).tasks["t1"].observe_count == 2


async def test_next_review_confirms_uncertain_predecessor_before_allocating(fake_state_ctx):
    state, ctx = fake_state_ctx
    store, _, _ = await setup_owner(state, ctx)
    ctx.steps = StepDriver({})
    original = store.append_batch
    attempts = []

    async def lose_receipt(sid, bid, events):
        receipt = await original(sid, bid, events)
        if events[0].type == EventType.OBSERVE_STARTED:
            attempts.append((bid, events[0]))
            if len(attempts) == 1:
                raise OSError("lost predecessor receipt")
        return receipt

    store.append_batch = lose_receipt
    with pytest.raises(PersistenceUnavailableError):
        await begin(state, ctx, "plain_text", True)
    other_state = replace(state, extra={})
    other_ctx = replace(ctx, execution=StepExecution())
    await begin(other_state, other_ctx, "plain_text", True)
    assert attempts[0] == attempts[1]
    assert other_ctx.execution.options["observe_index"] == state.task.observe_count == 2
    assert ctx.execution.options["observe_index"] == 1
    head = await store.committed_head("s1")
    await begin(state, ctx, "plain_text", True)
    assert await store.committed_head("s1") == head
    assert (await rebuild_view(store, "s1")).tasks["t1"].observe_count == 2


@pytest.mark.parametrize("snapshot", [False, True, "old"])
async def test_cold_count_and_pending_ordinal_survive_snapshot_and_retry(fake_state_ctx, snapshot):
    state, ctx = fake_state_ctx
    store, bus, _ = await setup_owner(state, ctx)
    for index in range(1, 4):
        ctx.execution = StepExecution()
        await begin(state, ctx, "plain_text", True)
        if index < 3:
            await bus.emit(make_event(state, EventType.OBSERVE_APPLIED, {
                "task_id": "t1", "observe_id": ctx.execution.options["observe_id"]}))
    pending_id = ctx.execution.options["observe_id"]
    if snapshot:
        view = await rebuild_view(store, "s1")
        overrides = None
        if snapshot == "old":
            blob = serialize_view(view)
            blob["tasks"]["t1"].pop("observe_count")
            overrides = {"projection_version": 11, "state_blob": blob}
        await seed_snapshot(store, "s1", view, cut=await store.committed_head("s1"), overrides=overrides)
    view = await rebuild_view(store, "s1")
    assert view.tasks["t1"].observe_count == 3
    assert view.pending_observe[pending_id]["observe_index"] == 3
    assert (await read_task_view(store, "s1", "t1", through_position=await store.committed_head("s1"))).observe_count == 3
    startup = await rebuild_startup_view(store, "s1", through_position=await store.committed_head("s1"))
    assert startup.tasks["t1"].observe_count == 3

    # A fresh owner loads the projection rather than retaining any hot object.
    cold_tm = TaskManager("s1", max_concurrent=0, event_bus=bus)
    cold_task = task_from_projection(view.tasks["t1"])
    cold_tm.restore([cold_task], terminal_ids=set(), enqueue=False)
    state.task, ctx.task_manager = cold_task, cold_tm
    ctx.execution = StepExecution(options=dict(view.pending_observe[pending_id]))
    head = await store.committed_head("s1")
    await begin(state, ctx, "plain_text", True)
    assert cold_task.observe_count == 3
    assert ctx.execution.options["observe_index"] == 3
    assert await store.committed_head("s1") == head
    ctx.execution = StepExecution()
    await begin(state, ctx, "plain_text", True)
    assert cold_task.observe_count == ctx.execution.options["observe_index"] == 4
    assert (await rebuild_view(store, "s1")).tasks["t1"].observe_count == 4


async def test_cold_task_count_reads_only_observe_metadata(fake_state_ctx):
    state, ctx = fake_state_ctx
    store, bus, _ = await setup_owner(state, ctx)
    payload = {"task_id": "t1", "observe_id": "old", "judge_task": True,
               "context_record_ids": ["large context"]}
    await bus.emit(make_event(state, EventType.HITL_OPENED, {
        "observe_request": payload, "form": {"body": "private historical form"}}))
    await bus.emit(make_event(state, EventType.OBSERVE_STARTED, {
        "task_id": "t1", "observe_id": "old"}))
    original, projected = store.read_range, []

    async def capture(*args, **kwargs):
        assert kwargs.get("projection") is not None
        rows = await original(*args, **kwargs)
        projected.extend(row.event.payload for row in rows)
        return rows

    store.read_range = capture
    task = await read_task_view(store, "s1", "t1", through_position=await store.committed_head("s1"))
    assert task.observe_count == 1
    assert "private historical form" not in str(projected)
    assert "large context" not in str(projected)


def test_existing_unnumbered_starts_use_request_judging_flag_and_deduplicate(fake_state_ctx):
    state, _ = fake_state_ctx
    events = [make_event(state, EventType.TASK_CREATED, {"task": {"id": "t1"}})]
    for index, judging in enumerate([True, False, True], 1):
        payload = {"task_id": "t1", "observe_id": f"old-{index}", "judge_task": judging}
        # Parked requests are embedded in HitlOpened rather than emitted separately.
        events.append(make_event(state, EventType.HITL_OPENED, {"observe_request": payload}))
        started = make_event(state, EventType.OBSERVE_STARTED,
                             {"task_id": "t1", "observe_id": payload["observe_id"]})
        events.extend([started, started])
        if index < 3:
            events.append(make_event(state, EventType.OBSERVE_APPLIED, payload))
    view = reduce_events(events, "s1")
    assert view.tasks["t1"].observe_count == 2
    assert view.pending_observe["old-3"]["observe_index"] == 2
    view = deserialize_view(serialize_view(view))
    apply_events([events[-1]], view)
    assert view.tasks["t1"].observe_count == 2


def test_indexed_starts_for_other_tasks_and_replays_do_not_inflate_count(fake_state_ctx):
    state, _ = fake_state_ctx
    created = [make_event(state, EventType.TASK_CREATED, {"task": {"id": tid}}) for tid in ("t1", "t2")]
    def started(tid, index):
        return make_event(state, EventType.OBSERVE_STARTED, {
            "task_id": tid, "observe_id": f"{tid}-{index}", "judge_task": True, "observe_index": index})
    third, first = started("t1", 3), started("t2", 1)
    view = reduce_events([*created, third, first, third, started("t1", 2)], "s1")
    assert view.tasks["t1"].observe_count == 3
    assert view.tasks["t2"].observe_count == 1
