"""Cancellation ownership and exhausted mechanical reports on real runtime paths."""

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view, reduce_events
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_lifecycle_crash_boundaries import run_worker
from tests.integration.test_observe_application_recovery import (
    assert_cold_recovery_is_noop,
    recover_and_drain,
)
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles
from tests.integration.test_single_observe_recovery import baseline, stored_content



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parked_cancel_success", "parked_cancel_retry"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["none", "task_canceled", "cancel_closed", "observe_completed"])
async def test_cancel_parked_task_revokes_old_observer(tmp_path, tmp_path_factory, mode, snapshot, cut):
    expected = await baseline(tmp_path_factory, mode)
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover_and_drain(store, memory, mode, snapshot)
            view = await rebuild_view(store, "crash-session")
            rows = await store.read_range("crash-session")
            task, = reduce_events([r.event for r in rows], "").tasks.values()
            assert task.status == "CANCELED"
            assert not view.hitl.pending
            assert not view.pending_observe and not view.open_runs
            rows = await store.read_range("crash-session")
            assert sum(r.event.type == EventType.TASK_CANCELED for r in rows) == 1
            canceled = next(r for r in rows if r.event.type == EventType.TASK_CANCELED)
            started = next(r for r in rows if r.event.type == EventType.RUN_STARTED
                           and r.event.origin != "loop.background_observe")
            finished = next(r for r in rows if r.event.type == EventType.RUN_FINISHED
                            and r.event.run_id == started.event.run_id)
            assert finished.position < canceled.position
            applied = next(r for r in rows if r.event.type == EventType.OBSERVE_APPLIED)
            resolved = [r.position for r in rows if r.event.type == EventType.HITL_RESOLVED]
            assert resolved and max(resolved) < applied.position
            assert not any(r.event.type == EventType.TASK_FINISHED for r in rows)
            assert sum(r.event.type == EventType.TASK_FINALIZED for r in rows) == 1
            assert_lifecycles(rows)
            actual = await stored_content(store, memory)
            assert actual == expected
            assert "completed answer" in str(actual) if mode.endswith("retry") else "done" in str(actual)
            assert "reviewed" not in str(actual)
            await assert_cold_recovery_is_noop(store, memory, mode, root)
            assert await stored_content(store, memory) == actual


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("mode", ["retry", "mechanical_retry"])
@pytest.mark.parametrize("cut", ["observe_completed", "task_finalized", "task_requeued"])
async def test_continue_with_zero_exception_budget_recovers_without_failure(tmp_path, snapshot, mode, cut):
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover_and_drain(store, memory, "foreground", snapshot)
            tm = rt._task_managers["crash-session"]
            async with asyncio.timeout(10):
                while (tm._dispatch_tasks or tm._background_asyncio_tasks
                       or any(t.status in {"ACTIVE", "PENDING"} for t in tm._tasks.values())):
                    await asyncio.sleep(0.01)
            task, = tm._tasks.values()
            assert task.status == "FINISHED"
            assert task.max_retries == task.retry_count == 0
            rows = await store.read_range("crash-session")
            assert not any(r.event.type == EventType.TASK_FAILED for r in rows)
            requeues = [r for r in rows if r.event.type == EventType.TASK_REQUEUED
                        and r.event.payload.get("outcome") == "continue"]
            assert len(requeues) == 1
            view = await rebuild_view(store, "crash-session")
            assert not view.pending_observe and not view.open_runs
            assert_lifecycles(rows)
            await assert_cold_recovery_is_noop(store, memory, "foreground", root)


@pytest.mark.asyncio
async def test_public_cancel_does_not_overwrite_an_already_committed_verdict(tmp_path, monkeypatch):
    import asyncio

    import ctx_weft.core.loop.steps.observe as observe
    from ctx_weft import SessionStartParams
    from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime
    committed = asyncio.Event()
    release = asyncio.Event()
    original = observe._out_of_band_finalize
    async def hold_finalize(*args, **kwargs):
        committed.set()
        await release.wait()
        return await original(*args, **kwargs)
    monkeypatch.setattr(observe, "_out_of_band_finalize", hold_finalize)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, _ = make_lifecycle_runtime(store, memory, "finish_park")
            await initialize_all(rt)
            handle = await rt.start_session(SessionStartParams.create(
                session_id="crash-session", template_id="agent:tpl_echo", user_prompt="work"))
            await asyncio.wait_for(committed.wait(), 10)
            rows = await store.read_range("crash-session")
            assert any(r.event.type == EventType.OBSERVE_COMPLETED and r.event.payload["accepted"] for r in rows)
            assert not any(r.event.type == EventType.TASK_FINISHED for r in rows)
            await rt.cancel_agent(handle.agent_id, reason="cancel after decision")
            release.set()
            tm = rt._task_managers["crash-session"]
            while tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            rows = await store.read_range("crash-session")
            assert not any(r.event.type == EventType.TASK_CANCELED for r in rows)
            assert sum(r.event.type == EventType.TASK_FINISHED for r in rows) == 1
            assert_lifecycles(rows)
            content = await stored_content(store, memory)
            assert "reviewed" in str(content[1])
