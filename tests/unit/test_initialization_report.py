"""Initialization summaries are durable-cut facts, not live runtime owners."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
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
        async with open_sqlite_event_store(tmp_path / "report.db") as event_store:
            yield event_store


async def test_report_is_after_commit_and_survives_a_second_process_losing_the_report(store):
    h = await container(runtime(store))
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "t", h.root_agent_id, {}),
        (EventType.RUN_STARTED, "t", h.root_agent_id, {}))
    before = await store.committed_head(h.session_id)
    rt = runtime(store)
    rt.get_task = AsyncMock(side_effect=AssertionError("no public query during initialize"))
    report = await initialize_all(rt)
    result, = report.sessions
    assert result.before_position == before
    assert result.committed_position == await store.committed_head(h.session_id)
    assert result.committed_position > before
    assert result.startup_events == await store.read_range(h.session_id, after_position=before)
    assert result.state.has_interrupted_agents and not result.state.has_running_agents
    assert result.state.root_task.status == "INTERRUPTED"
    assert not rt._task_managers and not rt._agent_lifecycle_manager.all_agent_ids()

    # The host never received the first report. A fresh SDK emits no duplicate
    # repairs, but must still provide the fully calibrated state for the host.
    restarted = runtime(store)
    second, = (await initialize_all(restarted)).sessions
    assert second.startup_events == []
    assert second.before_position == second.committed_position == result.committed_position
    assert second.state == result.state
    report.sessions.clear()
    assert await rt.initialize(h.session_id) is None


@pytest.mark.parametrize("base_kind", ["full", "slim", "pruned"])
async def test_summary_reads_pruned_root_only_and_accepts_status_only_snapshot(store, monkeypatch, base_kind):
    h = await container(runtime(store))
    for tid in ("unrelated", "root-task"):
        await append(store, h.session_id,
            (EventType.TASK_CREATED, tid, h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
            (EventType.TASK_FINISHED, tid, h.root_agent_id, {"outputs": "done"}))
    await append(store, h.session_id, (EventType.AGENT_IDLE, "root-task", h.root_agent_id, {}))
    view = await rebuild_view(store, h.session_id)
    if base_kind == "pruned":
        view = prune_view_for_snapshot(view)
        assert "root-task" not in view.tasks
    elif base_kind == "slim":
        view.tasks["root-task"].details_complete = False
    cut = await store.committed_head(h.session_id)
    await seed_snapshot(store, h.session_id, view, cut=cut)
    calls = []
    original = store.read_range
    async def read(sid, **kw):
        calls.append(kw)
        return await original(sid, **kw)
    monkeypatch.setattr(store, "read_range", read)
    last = AsyncMock(wraps=store.read_last_of_type)
    monkeypatch.setattr(store, "read_last_of_type", last)
    report = await initialize_all(runtime(store))
    result, = report.sessions
    assert result.startup_events == [] and result.state.root_task.status == "FINISHED"
    assert result.state.root_task.task_id == "root-task"
    target = [kw for kw in calls if kw.get("related_task_id")]
    # Compact recovery facts need one bounded task-body query for the public DTO.
    assert len(target) == 1
    if target:
        assert target[0]["related_task_id"] == "root-task"
        assert target[0]["through_position"] == result.committed_position
    assert len(calls) == 1 + len(target)
    assert last.await_count <= 2  # Control snapshot plus the full task query.


async def test_lost_commit_confirmation_yields_no_success_report_then_recovers(store, monkeypatch):
    h = await container(runtime(store))
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "t", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "t", h.root_agent_id, {}))
    rt = runtime(store)
    original = store.append_batch
    async def lose_confirmation(*args, **kwargs):
        await original(*args, **kwargs)
        raise OSError("commit confirmation lost")
    monkeypatch.setattr(store, "append_batch", lose_confirmation)
    with pytest.raises(OSError, match="confirmation lost"):
        await initialize_all(rt)
    assert not rt._initialized_sessions and not rt._initializing_sessions
    committed = await store.committed_head(h.session_id)
    monkeypatch.setattr(store, "append_batch", original)
    result, = (await initialize_all(rt)).sessions
    assert result.startup_events == [] and result.committed_position == committed
    assert result.state.has_interrupted_agents and result.state.root_task.status == "INTERRUPTED"
