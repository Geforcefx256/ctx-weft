
from ctx_weft.core.control import session_queries as session_query_module
"""Light startup must produce the same durable repairs as full reconstruction."""

# ruff: noqa: F811 -- pytest fixtures are explicitly imported for parametrization.
from dataclasses import replace

import pytest

from ctx_weft.core.control import startup
from ctx_weft.core.control.reducers import (
    apply_events,
    prune_view_for_snapshot,
    read_task_view,
    rebuild_view,
)
from ctx_weft.protocols.events import EventReadProjection, EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.unit.test_read_primitives_conformance import _ev, store  # noqa: F401
from tests.unit.test_runtime_initialization_queries import runtime



async def seed_history(store, scenario):
    rows = [
        (EventType.SESSION_CREATED, None, {"root_agent_id": "root", "user_prompt": "big"}),
        (EventType.AGENT_INSTANTIATED, None, {"template_id": "agent:tpl_echo"}),
        (EventType.TASK_CREATED, "task", {"task": {
            "id": "task", "assigned_agent_id": "root", "user_prompt": "big"}}),
        (EventType.TASK_STARTED, "task", {"assigned_agent_id": "root"}),
        (EventType.AGENT_RUNNING, "task", {}),
        (EventType.RUN_STARTED, "task", {}),
        ("LargeUnrelatedEvent", "task", {"body": "big"}),
    ]
    if scenario in {"human", "no_resume", "threshold"}:
        delivery = {"kind": "user_turn", "task_id": "task"} if scenario != "no_resume" else {"kind": "unknown"}
        rows += [(EventType.HITL_OPENED, "task", {
            "hitl_id": "ask", "delivery": delivery, "prompt": "big", "detail": "big"})]
    elif scenario == "legacy_retracted":
        rows += [
            (EventType.HITL_REQUIRED, "task", {"hitl_id": "ask", "form": "wait"}),
            (EventType.HITL_ANSWERED, "task", {"hitl_id": "ask", "message": "answer"}),
        ]
    elif scenario == "review":
        rows += [(EventType.TASK_REVIEWING, "task", {
            "successor": {"task": {"id": "next", "assigned_agent_id": "root", "user_prompt": "big"}},
            "replaced_task_ids": ["task"], "rewired_dependencies": {"task": ["next"]}})]
    elif scenario == "terminated":
        rows += [(EventType.AGENT_TERMINATED, "task", {"reason": "user_cancel"})]
    elif scenario == "finished":
        rows += [(EventType.TASK_FINISHED, "task", {"outputs": "big"}),
                 (EventType.AGENT_IDLE, "task", {}), (EventType.RUN_FINISHED, "task", {})]
    if scenario == "threshold":
        rows += [(EventType.FAILURE_THRESHOLD_HIT, "task", {
            "root_task_ids": ["task"], "cancel_task_ids": []})]
    events = [replace(_ev(i, kind), agent_id="root", task_id=tid, payload=payload)
              for i, (kind, tid, payload) in enumerate(rows, 1)]
    await store.append_batch("s1", "history", events)


def repair_facts(report):
    return [(r.session_id, r.before_position, r.committed_position, r.state,
             [(e.event.type, e.event.run_id, e.event.sequence, e.event.task_id,
               e.event.agent_id, e.event.payload) for e in r.startup_events]) for r in report.sessions]


async def full_startup(monkeypatch, store):
    async def full_task(store, sid, tid, *, through_position):
        return await read_task_view(store, sid, tid, through_position=through_position, use_snapshot=False)
    with monkeypatch.context() as context:
        context.setattr(startup, "rebuild_startup_view", rebuild_view)
        context.setattr(session_query_module, "read_task_view", full_task)
        return await initialize_all(runtime(store))


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["running", "human", "no_resume", "threshold",
                                      "legacy_retracted", "review", "terminated", "finished"])
@pytest.mark.parametrize("snapshot", [None, 11, 10, 4])
async def test_light_startup_matches_full_repairs(store, monkeypatch, scenario, snapshot):
    await seed_history(store, scenario)
    if snapshot is not None:
        view = prune_view_for_snapshot(await rebuild_view(store, "s1"))
        await seed_snapshot(store, "s1", view, cut=await store.committed_head("s1"),
                            overrides={"projection_version": snapshot})
    # A late non-control event must still advance the interrupted run sequence.
    late = replace(_ev(999, "LargeUnrelatedEvent"), agent_id="root", task_id="task",
                   payload={"body": "big"})
    if scenario == "legacy_retracted":
        late = replace(late, type=EventType.HITL_REPLY_RETRACTED, payload={"hitl_id": "ask"})
    await store.append_batch("s1", "late", [late])
    reference = InMemoryEventStore()
    await reference.append_batch("s1", "copy", [r.event for r in await store.read_range("s1")])
    expected = await full_startup(monkeypatch, reference)
    rt = runtime(store)
    actual = await initialize_all(rt)
    assert repair_facts(actual) == repair_facts(expected)
    assert actual.interrupted_task_ids == expected.interrupted_task_ids
    assert actual.interrupted_session_ids == expected.interrupted_session_ids
    assert not rt._task_managers and not rt._agent_lifecycle_manager.all_agent_ids()
    second = await initialize_all(runtime(store))
    assert all(not result.startup_events for result in second.sessions)
    assert [r.state for r in second.sessions] == [r.state for r in actual.sessions]


@pytest.mark.asyncio
async def test_full_reducer_rejects_partial_events(store):
    from ctx_weft.core.control.types import RunStateView
    await store.append_batch("s1", "batch", [_ev(1, "TaskStarted")])
    row, = await store.read_range("s1", projection=EventReadProjection({}))
    with pytest.raises(TypeError, match="Partial events"):
        apply_events([row.event], RunStateView("", "", "", ""))


@pytest.mark.asyncio
@pytest.mark.parametrize("compact", ["absent", "corrupt", "unknown-version"])
async def test_old_or_invalid_compact_snapshot_uses_legacy_projection(store, monkeypatch, compact):
    await seed_history(store, "finished")
    view = prune_view_for_snapshot(await rebuild_view(store, "s1"))
    overrides = {"startup_state_blob": None}
    if compact == "corrupt":
        overrides["startup_state_blob"] = {}
    elif compact == "unknown-version":
        overrides = {"startup_projection_version": 999}
    await seed_snapshot(store, "s1", view, cut=await store.committed_head("s1"), overrides=overrides)
    reference = InMemoryEventStore()
    await reference.append_batch("s1", "copy", [r.event for r in await store.read_range("s1")])
    expected = await full_startup(monkeypatch, reference)
    assert repair_facts(await initialize_all(runtime(store))) == repair_facts(expected)


@pytest.mark.asyncio
async def test_account_only_agent_instantiation_preserves_running_state(store, monkeypatch):
    events = [
        replace(_ev(1, EventType.SESSION_CREATED), agent_id="root", payload={"root_agent_id": "root"}),
        replace(_ev(2, EventType.AGENT_INSTANTIATED), agent_id="root", payload={"llm_account": "account"}),
        replace(_ev(3, EventType.AGENT_RUNNING), agent_id="root", task_id=None),
    ]
    await store.append_batch("s1", "history", events)
    reference = InMemoryEventStore()
    await reference.append_batch("s1", "history", events)
    assert repair_facts(await initialize_all(runtime(store))) == repair_facts(await full_startup(monkeypatch, reference))


@pytest.mark.asyncio
async def test_initialization_never_loads_bodies_on_cold_execution(store):
    await seed_history(store, "finished")
    first = runtime(store)
    await initialize_all(first)
    # Lightweight startup is never used as an execution/details cache.
    detail = await first.get_task("s1", "task")
    assert detail.data.outputs == "big"
