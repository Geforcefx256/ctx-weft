"""Existing cancel_agent persists agent termination and recovers task-tree cancellation."""
import pytest

from ctx_weft.core.control.reducers import rebuild_view, reduce_events
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_lifecycle_crash_boundaries import run_worker
from tests.integration.test_observe_application_recovery import recover_and_drain, assert_cold_recovery_is_noop
from tests.integration.test_single_observe_recovery import baseline, stored_content
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parked_interrupt_success", "parked_interrupt_retry"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["none", "agent_terminated", "task_canceled", "cancel_closed", "observe_completed", "recovery_cancel_closed"])
async def test_cancellation_survives_every_cancel_boundary(tmp_path, tmp_path_factory, mode, snapshot, cut):
    expected = await baseline(tmp_path_factory, mode)
    await run_worker(tmp_path, mode, "agent_terminated" if cut == "recovery_cancel_closed" else cut)
    if cut == "recovery_cancel_closed":
        await run_worker(tmp_path, mode, "cancel_closed", recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover_and_drain(store, memory, mode, snapshot)
            rows = await store.read_range("crash-session")
            view = await rebuild_view(store, "crash-session")
            task, = reduce_events([r.event for r in rows], "").tasks.values()
            assert task.status == "CANCELED"
            assert view.agents[root].status == "terminated"
            assert rt._agent_lifecycle_manager.status_of(root) == "terminated"
            assert not view.hitl.pending and not view.pending_observe and not view.open_runs
            interrupted, = [r for r in rows if r.event.type == "AgentTerminated"
                            and r.event.payload.get("reason", "").startswith("cancel_agent:")]
            canceled, = [r for r in rows if r.event.type == "TaskCanceled"]
            assert interrupted.position < canceled.position
            assert not any(r.event.type in ("TaskFinished",) for r in rows)
            assert_lifecycles(rows)
            assert await stored_content(store, memory) == expected
            await assert_cold_recovery_is_noop(store, memory, mode, root)
            assert await stored_content(store, memory) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["agent_terminated", "task_finalized"])
async def test_accepted_verdict_remains_authoritative_after_cancellation(tmp_path, tmp_path_factory, snapshot, cut):
    mode = "accepted_interrupt"
    expected = await baseline(tmp_path_factory, mode)
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover_and_drain(store, memory, mode, snapshot)
            rows = await store.read_range("crash-session")
            task, = reduce_events([r.event for r in rows], "").tasks.values()
            assert task.status == "FINISHED"
            assert not any(r.event.type in ("TaskCanceled",) for r in rows)
            assert sum(r.event.type == "TaskFinished" for r in rows) == 1
            assert (await rebuild_view(store, "crash-session")).agents[root].status == "terminated"
            assert await stored_content(store, memory) == expected
            assert rt._llm.actor_count == rt._llm.judgements == 0
            assert_lifecycles(rows)
            await assert_cold_recovery_is_noop(store, memory, mode, root)
