"""Public session facts expose SDK-owned counters without loading cold execution."""

import pytest

from ctx_weft.core.control.reducers import rebuild_view, prune_view_for_snapshot
from ctx_weft.protocols.events import EventType
from tests._snapshot_helpers import seed_snapshot
from tests.unit.test_runtime_initialization_queries import append, container, runtime


async def test_hot_query_reads_current_task_manager_counters_and_new_round_reset():
    rt = runtime()
    h = await container(rt)
    manager = rt._task_managers[h.session_id]
    head = await rt.event_store.committed_head(h.session_id)
    # Owner values intentionally differ from the journal: hot queries expose live facts.
    manager.session.failure_counter = 3
    manager._threshold_tripped = True
    result = await rt.get_session(h.session_id)
    assert result.loaded and result.source == "memory"
    assert result.data.state.failure_counter == 3
    assert result.data.state.threshold_tripped is True
    assert rt._task_managers[h.session_id] is manager
    assert await rt.event_store.committed_head(h.session_id) == head
    manager.reset_for_new_round()
    state = (await rt.get_session(h.session_id)).data.state
    assert state.failure_counter == 0 and state.threshold_tripped is False


@pytest.mark.parametrize("snapshot", [False, True])
async def test_cold_query_and_initialization_use_replayed_counter_facts(snapshot):
    rt = runtime()
    h = await container(rt)
    sid, aid = h.session_id, h.root_agent_id
    await append(rt.event_store, sid,
        (EventType.TASK_CREATED, "child", aid, {"task": {"id": "child"}}),
        (EventType.TASK_FAILED, "child", aid, {"error_code": "ACTOR_ERROR"}),
        (EventType.FAILURE_THRESHOLD_HIT, None, None, {"failure_counter": 1, "threshold": 1}),
        (EventType.TASK_FAILED, "root", aid, {"error_code": "TASK_FAILED_BY_THRESHOLD"}),
    )
    cut = await rt.event_store.committed_head(sid)
    if snapshot:
        view = prune_view_for_snapshot(await rebuild_view(rt.event_store, sid))
        await seed_snapshot(rt.event_store, sid, view, cut=cut)
    assert rt.forget_session(sid)
    head = await rt.event_store.committed_head(sid)
    result = await rt.get_session(sid)
    assert not result.loaded and result.source == "events"
    assert result.data.state.failure_counter == 1
    assert result.data.state.threshold_tripped is True
    assert not rt._task_managers
    assert await rt.event_store.committed_head(sid) == head

    fresh = runtime(rt.event_store)
    initialized = await fresh.initialize(sid)
    assert initialized.state.failure_counter == 1
    assert initialized.state.threshold_tripped is True
    assert not fresh._task_managers
    await fresh.rebuild_session(sid)
    loaded = await fresh.get_session(sid)
    assert loaded.loaded and loaded.source == "memory"
    assert loaded.data.state.failure_counter == 1
    assert loaded.data.state.threshold_tripped is True
    assert fresh.forget_session(sid)
    await append(rt.event_store, sid,
        (EventType.SESSION_RESUMED, None, None, {}),
        (EventType.TASK_CREATED, "new", aid, {"task": {"id": "new"}}),
        (EventType.TASK_FAILED, "new", aid, {"error_code": "ACTOR_ERROR"}),
    )
    current = (await fresh.get_session(sid)).data.state
    assert current.failure_counter == 1 and current.threshold_tripped is False
    assert not fresh._task_managers
