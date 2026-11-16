
from tests._recovery import recover_fixture_session
"""Restart at durable event boundaries, with and without recovery snapshots."""

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.unit.test_runtime_initialization_queries import append, container, runtime



@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryEventStore()
    else:
        from ctx_weft.providers.events.store.sql import open_sqlite_event_store
        async with open_sqlite_event_store(tmp_path / "events.db") as event_store:
            yield event_store


async def snapshot_if_requested(store, sid, snapshot):
    if snapshot:
        view = prune_view_for_snapshot(await rebuild_view(store, sid))
        await seed_snapshot(store, sid, view, cut=await store.committed_head(sid))


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
@pytest.mark.parametrize("status,task_event,agent_event", [
    ("SUSPENDED", EventType.TASK_SUSPENDED, EventType.AGENT_IDLE),
    ("REVIEWING", EventType.TASK_REVIEWING, EventType.AGENT_IDLE),
    ("AWAITING_HUMAN", EventType.TASK_AWAITING_HUMAN, EventType.AGENT_WAITING_HUMAN),
    ("INTERRUPTED", EventType.TASK_INTERRUPTED, EventType.AGENT_IDLE),
])
async def test_restart_marks_parked_agent_interrupted_without_rewriting_task(
    store, snapshot, status, task_event, agent_event,
):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id,
         {"task": {"id": "t", "assigned_agent_id": h.root_agent_id}}),
        (task_event, "t", h.root_agent_id, {}),
        (agent_event, "t", h.root_agent_id, {}))
    await snapshot_if_requested(store, h.session_id, snapshot)

    fresh = runtime(store)
    result = await fresh.initialize(h.session_id)
    assert result.state.root_task.status == status
    assert result.state.root_agent.state.status == "interrupted"
    assert result.state.has_interrupted_agents and not result.state.has_running_agents
    event, = [row.event for row in result.startup_events]
    assert (event.type, event.agent_id, event.task_id, event.payload["reason"]) == (
        EventType.AGENT_INTERRUPTED, h.root_agent_id, "t", "crash_recovery")
    assert not fresh._task_managers and not fresh._agent_lifecycle_manager.all_agent_ids()

    head = await store.committed_head(h.session_id)
    again = await runtime(store).initialize(h.session_id)
    assert again.startup_events == [] and again.state == result.state
    assert await store.committed_head(h.session_id) == head
    await fresh.rebuild_session(h.session_id, strict=True)
    assert (await fresh.get_task(h.session_id, "t")).data.status == status
    assert (await fresh.get_agent(h.root_agent_id)).data.state.status == "interrupted"
    assert not fresh._run_tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
@pytest.mark.parametrize("delivery_kind", ["user", "tool", "notification", "other_agent"])
async def test_parked_agent_recovery_respects_matching_hitl(store, snapshot, delivery_kind):
    from ctx_weft.protocols.hitl import HitlAsk, NoResumeDelivery, ToolResultDelivery, UserTurnDelivery

    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id,
         {"task": {"id": "t", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_AWAITING_HUMAN, "t", h.root_agent_id, {}),
        (EventType.AGENT_WAITING_HUMAN, "t", h.root_agent_id, {}))
    delivery = (NoResumeDelivery() if delivery_kind == "notification" else
                ToolResultDelivery("call") if delivery_kind == "tool" else UserTurnDelivery("t"))
    req = await old.hitl.open(
        HitlAsk(form="question", delivery=delivery), session_id=h.session_id, task_id="t",
        agent_id="other" if delivery_kind == "other_agent" else h.root_agent_id,
        stage="tool", unattended=False)
    await snapshot_if_requested(store, h.session_id, snapshot)

    fresh = runtime(store)
    result = await fresh.initialize(h.session_id)
    expected = "waiting_human" if delivery_kind in {"user", "tool"} else "interrupted"
    assert result.state.root_agent.state.status == expected
    assert result.state.root_task.status == "AWAITING_HUMAN"
    assert [row.data.id for row in await fresh.list_pending_hitl(session_id=h.session_id)] == [req.id]
    assert bool(result.startup_events) == (expected == "interrupted")


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
async def test_recovered_suspended_parent_is_resumable_after_child_finished(store, snapshot):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "parent", h.root_agent_id,
         {"task": {"id": "parent", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_SUSPENDED, "parent", h.root_agent_id, {}),
        (EventType.AGENT_IDLE, "parent", h.root_agent_id, {}),
        (EventType.AGENT_INSTANTIATED, None, "child-agent", {"template_id": "agent:tpl_echo"}),
        (EventType.TASK_CREATED, "child", "child-agent", {"task": {
            "id": "child", "assigned_agent_id": "child-agent",
            "creator_agent_id": h.root_agent_id, "parent_task_id": "parent"}}),
        (EventType.TASK_FINISHED, "child", "child-agent", {}),
        (EventType.AGENT_IDLE, "child", "child-agent", {}))
    await snapshot_if_requested(store, h.session_id, snapshot)

    fresh = runtime(store)  # Hold dispatch so explicit recovery's queue can be inspected.
    result = await fresh.initialize(h.session_id)
    assert result.state.root_agent.state.status == "interrupted"
    assert result.state.root_task.status == "SUSPENDED"
    await fresh.rebuild_session(h.session_id, strict=True)
    tm = fresh._task_managers[h.session_id]
    assert not tm._queue.peek_all() and not fresh._run_tokens
    await recover_fixture_session(fresh, h.root_agent_id)
    assert [entry.task_id for entry in tm._queue.peek_all()] == ["parent"]
    assert tm.get_task("parent").status == "PENDING"
    assert tm.get_task("child").status == "FINISHED"


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
@pytest.mark.parametrize("status,event", [
    ("PENDING", None),
    ("ACTIVE", EventType.TASK_STARTED),
    ("REVIEWING", EventType.TASK_REVIEWING),
    ("AWAITING_HUMAN", EventType.TASK_AWAITING_HUMAN),
    ("SUSPENDED", EventType.TASK_SUSPENDED),
    ("INTERRUPTED", EventType.TASK_INTERRUPTED),
    ("FINISHED", EventType.TASK_FINISHED),
    ("FAILED", EventType.TASK_FAILED),
    ("CANCELED", EventType.TASK_CANCELED),
])
async def test_restart_preserves_durable_task_state(store, snapshot, status, event):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id,
         {"task": {"id": "t", "assigned_agent_id": h.root_agent_id, "title": "work"}}))
    if event is not None:
        await append(store, h.session_id,
                     (event, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}))
    await snapshot_if_requested(store, h.session_id, snapshot)
    restarted = runtime(store)
    await initialize_all(restarted)
    rows = await store.read_range(h.session_id)
    assert not any(row.event.type == EventType.SESSION_STATUS_CHANGED for row in rows)
    expected = "INTERRUPTED" if status in {"ACTIVE", "PENDING"} else status
    assert (await restarted.get_task(h.session_id, "t")).data.status == expected
    assert not restarted._task_managers
    assert restarted._agent_lifecycle_manager.all_agent_ids() == []
    head = await store.committed_head(h.session_id)
    await initialize_all(runtime(store))
    assert await store.committed_head(h.session_id) == head
    await restarted.rebuild_session(h.session_id, strict=True)
    assert (await restarted.get_task(h.session_id, "t")).data.status == expected
    assert await store.committed_head(h.session_id) == head


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
async def test_restart_before_first_dispatch_is_interrupted_and_explicitly_resumable(store, snapshot):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id, (EventType.TASK_CREATED, "pending", h.root_agent_id,
        {"task": {"id": "pending", "assigned_agent_id": h.root_agent_id,
                  "user_prompt": "saved request"}}))
    await snapshot_if_requested(store, h.session_id, snapshot)
    restarted = runtime(store)  # max_concurrent=0 keeps explicit requeue observable.
    result, = (await initialize_all(restarted)).sessions
    assert result.state.root_task.status == "INTERRUPTED"
    assert result.state.has_interrupted_agents and not result.state.has_running_agents
    assert not restarted._task_managers
    await restarted.rebuild_session(h.session_id, strict=True)
    tm = restarted._task_managers[h.session_id]
    assert tm.recovery_pending and tm.is_done()
    assert not restarted._run_tokens.get(h.session_id)
    await recover_fixture_session(restarted, h.root_agent_id)
    assert not tm.recovery_pending
    assert [entry.task_id for entry in tm._queue.peek_all()] == ["pending"]
    assert tm.get_task("pending").status == "PENDING"
    # Repeated resume cannot enqueue the same task twice.
    await recover_fixture_session(restarted, h.root_agent_id)
    assert [entry.task_id for entry in tm._queue.peek_all()] == ["pending"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_kind", ["tool", "user"])
async def test_pending_task_with_durable_hitl_remains_a_human_wait(store, delivery_kind):
    from ctx_weft.protocols.hitl import HitlAsk, ToolResultDelivery, UserTurnDelivery

    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id, (EventType.TASK_CREATED, "pending", h.root_agent_id,
        {"task": {"id": "pending", "assigned_agent_id": h.root_agent_id}}))
    delivery = ToolResultDelivery("call") if delivery_kind == "tool" else UserTurnDelivery("pending")
    req = await old.hitl.open(HitlAsk(form="approval", delivery=delivery),
        session_id=h.session_id, task_id="pending", agent_id=h.root_agent_id,
        stage="authz", unattended=False)
    restarted = runtime(store)
    report = await initialize_all(restarted)
    assert not report.interrupted_task_ids
    assert (await restarted.get_task(h.session_id, "pending")).data.status == "AWAITING_HUMAN"
    assert (await restarted.get_agent(h.root_agent_id, session_id=h.session_id)).data.state.status == "waiting_human"
    assert [row.data.id for row in await restarted.list_pending_hitl(session_id=h.session_id)] == [req.id]
    head = await store.committed_head(h.session_id)
    await initialize_all(runtime(store))
    assert await store.committed_head(h.session_id) == head


@pytest.mark.asyncio
async def test_pending_child_recovery_preserves_suspended_parent(store):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "parent", h.root_agent_id,
         {"task": {"id": "parent", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_SUSPENDED, "parent", h.root_agent_id, {}),
        (EventType.AGENT_INSTANTIATED, None, "child-agent",
         {"template_id": "agent:tpl_echo", "parent_agent_id": h.root_agent_id}),
        (EventType.TASK_CREATED, "child", "child-agent",
         {"task": {"id": "child", "parent_task_id": "parent", "assigned_agent_id": "child-agent"}}))
    restarted = runtime(store)
    result, = (await initialize_all(restarted)).sessions
    assert result.state.has_interrupted_agents
    assert (await restarted.get_task(h.session_id, "parent")).data.status == "SUSPENDED"
    assert (await restarted.get_task(h.session_id, "child")).data.status == "INTERRUPTED"
    await restarted.rebuild_session(h.session_id, strict=True)
    await recover_fixture_session(restarted, h.root_agent_id)
    tm = restarted._task_managers[h.session_id]
    assert tm.get_task("parent").status == "SUSPENDED"
    assert [entry.task_id for entry in tm._queue.peek_all()] == ["child"]


@pytest.mark.asyncio
async def test_pending_successor_recovery_preserves_reviewing_predecessor(store):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "old", h.root_agent_id,
         {"task": {"id": "old", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_REVIEWING, "old", h.root_agent_id,
         {"successor": {"task": {"id": "new", "assigned_agent_id": h.root_agent_id}},
          "replaced_task_ids": ["old"]}))
    restarted = runtime(store)
    result, = (await initialize_all(restarted)).sessions
    assert result.state.root_task.task_id == "new" and result.state.root_task.status == "INTERRUPTED"
    assert (await restarted.get_task(h.session_id, "old")).data.status == "REVIEWING"


@pytest.mark.asyncio
async def test_repeated_initialize_does_not_interrupt_live_queued_work(store):
    from ctx_weft.core.models.task import Task

    rt = runtime(store)
    h = await container(rt)
    await rt.rebuild_session(h.session_id, strict=True)
    # Hold dispatch, so this is a real queued task rather than an active run.
    tm = rt._task_managers[h.session_id]
    tm._max_concurrent = 0
    await tm.push_task(Task(id="queued", session_id=h.session_id, status="PENDING",
                           assigned_agent_id=h.root_agent_id, user_prompt="queued"))
    head = await store.committed_head(h.session_id)
    await initialize_all(rt)
    assert (await rt.get_task(h.session_id, "queued")).data.status == "PENDING"
    assert await store.committed_head(h.session_id) == head
    assert [entry.task_id for entry in tm._queue.peek_all()] == ["queued"]


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
@pytest.mark.parametrize("cut", ["run_started", "run_finished"])
async def test_restart_closes_unfinished_foreground_run(store, snapshot, cut):
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id,
         {"task": {"id": "t", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_STARTED, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "t", h.root_agent_id, {}))
    start = new_event(EventType.RUN_STARTED, session_id=h.session_id,
                      tenant_id="default", origin="test", run_id="lost-run",
                      task_id="t", agent_id=h.root_agent_id, sequence=1)
    await store.append_batch(h.session_id, start.id, [start])
    if cut == "run_finished":
        end = new_event(EventType.RUN_FINISHED, session_id=h.session_id,
                        tenant_id="default", origin="test", run_id="lost-run",
                        task_id="t", agent_id=h.root_agent_id, sequence=2)
        await store.append_batch(h.session_id, end.id, [end])
        await append(store, h.session_id, (EventType.TASK_FINISHED, "t", h.root_agent_id, {}))
    await snapshot_if_requested(store, h.session_id, snapshot)
    await initialize_all(runtime(store))
    rows = await store.read_range(h.session_id, include_types=(EventType.RUN_FINISHED,))
    assert [row.event.run_id for row in rows] == ["lost-run"], (
        f"restart left execution open after {cut}")
    head = await store.committed_head(h.session_id)
    await initialize_all(runtime(store))
    assert await store.committed_head(h.session_id) == head


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["cross_cut", "no_delta", "finished_after_cut", "started_after_cut",
                                      "finished_before_cut", "terminal_task_background"])
async def test_run_recovery_reads_only_snapshot_delta(store, monkeypatch, scenario):
    old = runtime(store)
    h = await container(old)
    sid, aid = h.session_id, h.root_agent_id
    await append(store, sid, (EventType.TASK_CREATED, "t", aid,
                             {"task": {"id": "t", "assigned_agent_id": aid}}))

    async def event(kind, sequence):
        ev = new_event(kind, session_id=sid, tenant_id="default", origin="test",
                       run_id="cross-snapshot-run", task_id="t", agent_id=aid,
                       sequence=sequence, payload={"initial_step": "background_observe"})
        await store.append_batch(sid, ev.id, [ev])

    if scenario != "started_after_cut":
        await event(EventType.RUN_STARTED, 2)
        await event(EventType.STEP_STARTED, 7)
    if scenario == "finished_before_cut":
        await event(EventType.RUN_FINISHED, 9)
    if scenario == "terminal_task_background":
        await append(store, sid, (EventType.TASK_FINISHED, "t", aid, {"outputs": "done"}))
    cut = await store.committed_head(sid)
    snapshot_view = prune_view_for_snapshot(await rebuild_view(store, sid))
    if scenario == "terminal_task_background":
        assert "t" not in snapshot_view.tasks
        assert snapshot_view.open_runs["cross-snapshot-run"]["task_id"] == "t"
    await seed_snapshot(store, sid, snapshot_view, cut=cut)
    if scenario == "started_after_cut":
        await event(EventType.RUN_STARTED, 2)
    if scenario in {"cross_cut", "started_after_cut", "terminal_task_background"}:
        await event(EventType.LLM_RESPONSE_FINISHED, 12)
        # A lower sequence committed later must not lower the recovery sequence.
        await event(EventType.STEP_COMPLETED, 10)
    elif scenario == "finished_after_cut":
        await event(EventType.RUN_FINISHED, 12)

    head = await store.committed_head(sid)
    reads = []
    original_read = store.read_range

    async def delta_only(session_id, **kwargs):
        reads.append(kwargs)
        if kwargs.get("related_task_id"):
            assert kwargs["related_task_id"] == "t"
            assert kwargs["after_position"] == (0 if scenario == "terminal_task_background" else cut)
            assert kwargs["through_position"] >= head
            return await original_read(session_id, **kwargs)
        assert kwargs.get("after_position") == cut, kwargs
        assert kwargs.get("through_position") == head, kwargs
        return await original_read(session_id, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(store, "read_range", delta_only)
        await initialize_all(runtime(store))
    assert len(reads) == 2  # Snapshot delta plus a bounded public task-body query.
    rows = await original_read(sid, after_position=head)
    run_events = [r.event for r in rows if r.event.run_id == "cross-snapshot-run"]
    if scenario in {"finished_before_cut", "finished_after_cut"}:
        assert not run_events
    else:
        assert [ev.type for ev in run_events] == [EventType.RUN_INTERRUPTED, EventType.RUN_FINISHED]
        assert [ev.sequence for ev in run_events] == ([8, 9] if scenario == "no_delta" else [13, 14])
        assert all(ev.task_id == "t" and ev.agent_id == aid for ev in run_events)
    assert not (await rebuild_view(store, sid)).open_runs
    settled_head = await store.committed_head(sid)
    await initialize_all(runtime(store))
    assert await store.committed_head(sid) == settled_head


@pytest.mark.asyncio
async def test_old_snapshot_without_run_state_falls_back_to_log(store):
    old = runtime(store)
    h = await container(old)
    ev = new_event(EventType.RUN_STARTED, session_id=h.session_id, tenant_id="default",
                   origin="test", run_id="old-open-run", task_id="pruned-task", sequence=5)
    await store.append_batch(h.session_id, ev.id, [ev])
    view = await rebuild_view(store, h.session_id)
    from ctx_weft.core.control.reducers import serialize_view
    blob = serialize_view(view)
    blob.pop("open_runs")
    await seed_snapshot(store, h.session_id, view,
                        cut=await store.committed_head(h.session_id),
                        overrides={"projection_version": 6, "state_blob": blob})
    await initialize_all(runtime(store))
    rows = await store.read_range(h.session_id, include_types=(EventType.RUN_FINISHED,))
    assert [(r.event.run_id, r.event.sequence) for r in rows] == [("old-open-run", 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["log", "snapshot"])
@pytest.mark.parametrize("delivery_kind", ["tool", "user", "notification"])
@pytest.mark.parametrize("cut", ["opened", "deferred", "retracted", "resolved", "closed"])
async def test_restart_at_hitl_event_boundary(store, snapshot, delivery_kind, cut):
    from ctx_weft.protocols.hitl import (
        HitlAsk,
        HitlReply,
        NoResumeDelivery,
        ToolResultDelivery,
        UserTurnDelivery,
    )

    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id,
         {"task": {"id": "t", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_STARTED, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "t", h.root_agent_id, {}))
    delivery = {"tool": ToolResultDelivery("call"), "user": UserTurnDelivery("t"),
                "notification": NoResumeDelivery()}[delivery_kind]
    req = await old.hitl.open(
        HitlAsk(form="approval", delivery=delivery), session_id=h.session_id,
        task_id="t", agent_id=h.root_agent_id, stage="authz", unattended=False)
    if cut != "opened":
        await old.hitl.resolve(HitlReply(hitl_id=req.id, outcome="accepted",
                                        agent_id=h.root_agent_id, message="yes"),
                               defer=cut in {"deferred", "retracted"})
        if cut == "retracted":
            await old.hitl.release(req.id)
        elif cut == "closed":
            await old.hitl.close(req)
    await snapshot_if_requested(store, h.session_id, snapshot)
    rt = runtime(store)
    await initialize_all(rt)
    pending = cut in {"opened", "deferred", "retracted"}
    expected = "AWAITING_HUMAN" if pending and delivery_kind != "notification" else "INTERRUPTED"
    assert (await rt.get_task(h.session_id, "t")).data.status == expected
    assert bool(await rt.list_pending_hitl(session_id=h.session_id)) == pending
    head = await store.committed_head(h.session_id)
    await initialize_all(runtime(store))
    assert await store.committed_head(h.session_id) == head
    await rt.rebuild_session(h.session_id, strict=True)
    assert bool(await rt.list_pending_hitl(session_id=h.session_id)) == pending
    assert (await rt.get_task(h.session_id, "t")).data.status == expected
