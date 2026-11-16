
from tests._recovery import recover_fixture_session
"""Crash real scheduling and trip paths; assert durable ordering after restart."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.protocols import load_records
from ctx_weft.core.loop.task_memory_fold import task_record_kind
from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.core.models.discriminators import TaskErrorCode
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime

WORKER = Path(__file__).with_name("_lifecycle_crash_worker.py")


async def run_worker(path, mode, cut, *, recovering=False):
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, "-u", str(WORKER), str(path), mode, cut,
         *(["recover"] if recovering else [])],
        cwd=str(WORKER.parents[2]), capture_output=True, text=True, timeout=40)
    assert proc.returncode == (0 if cut == "none" else 73), proc.stdout + proc.stderr


async def restart(store, memory, mode, snapshot):
    if snapshot:
        view = prune_view_for_snapshot(await rebuild_view(store, "crash-session"))
        await seed_snapshot(store, "crash-session", view,
                            cut=await store.committed_head("crash-session"))
    rt, _ = make_lifecycle_runtime(store, memory, mode)
    await initialize_all(rt)
    head = await store.committed_head("crash-session")
    again, _ = make_lifecycle_runtime(store, memory, mode)
    await initialize_all(again)
    assert await store.committed_head("crash-session") == head
    return rt


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("mode", ["plain_text", "finish_park", "interrupt"])
@pytest.mark.parametrize("cut", ["hitl_opened", "background_run_started", "recap_started", "recap_done",
                                "observer_response", "task_finished", "task_finalized", "observe_completed",
                                "finish_memory", "blackboard_memory"])
async def test_observe_real_process_crash(tmp_path, snapshot, mode, cut):
    if mode == "interrupt" and cut in {"observer_response", "task_finished", "task_finalized", "finish_memory", "blackboard_memory"}:
        pytest.skip("This interrupted partial stream has no completed actor turn; observe skips judging")
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            before = await rebuild_view(store, "crash-session")
            if cut != "recap_done":
                assert len(before.pending_observe) == 1
                intent = next(iter(before.pending_observe.values()))
                assert intent["observe_id"] and intent["segment_end_at"]
            rt = await restart(store, memory, mode, snapshot)
            view = await rebuild_view(store, "crash-session")
            root = view.sessions["crash-session"].root_agent_id
            await recover_fixture_session(rt, root, session_id="crash-session")
            tm = rt._task_managers["crash-session"]
            if tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            final = await rebuild_view(store, "crash-session")
            assert not final.pending_observe
            tasks = await rt.list_tasks("crash-session")
            rows = await store.read_range("crash-session")
            for row in rows:
                if row.event.type == EventType.OBSERVE_REQUESTED:
                    assert "outputs" not in row.event.payload
                if row.event.type == EventType.HITL_OPENED and row.event.payload.get("observe_request"):
                    assert "outputs" not in row.event.payload["observe_request"]
            expected = "AWAITING_HUMAN" if mode == "interrupt" else "FINISHED"
            assert tasks and all(row.data.status == expected for row in tasks), {
                "tasks": [(row.data.status, row.data.outputs) for row in tasks],
                "tools": rt._llm.tool_names,
                "observation": [(row.event.type, row.event.payload) for row in rows
                    if row.event.type in {"ObserveCompleted", "CapabilityFinished"}],
            }
            if mode != "interrupt":
                expected_output = "completed answer" if mode == "plain_text" else "done"
                assert all(row.data.outputs == expected_output for row in tasks)
                assert sum(row.event.type == EventType.TASK_FINALIZED for row in rows) == 1
                from ctx_weft.protocols import (
                    MemoryAddress,
                    MemoryKind,
                    MemoryScope,
                    ProviderContext,
                )
                provider_ctx = ProviderContext(session_id="crash-session", tenant_id="default")
                publications = await load_records(memory, MemoryAddress(session_id="crash-session"),
                    MemoryScope.SESSION, provider_ctx, kinds=[MemoryKind.PUBLICATION])
                assert len(publications) == 1
                turns = await load_records(memory, MemoryAddress(session_id="crash-session", agent_id=root),
                    MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
                assert sum(task_record_kind(turn) == "finish" for turn in turns) == 1

            done = [row.event.payload.get("observe_id") for row in rows
                    if row.event.type == EventType.OBSERVE_APPLIED]
            assert len(done) == 1 and done[0]
            head = await store.committed_head("crash-session")
            await recover_fixture_session(rt, root)
            assert await store.committed_head("crash-session") == head
            again, _ = make_lifecycle_runtime(store, memory, mode)
            await initialize_all(again)
            await recover_fixture_session(again, root, session_id="crash-session")
            assert await store.committed_head("crash-session") == head


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["background_run_started", "recap_started", "observer_response",
                                "task_finished", "task_finalized", "recap_done"])
async def test_crash_again_during_observe_recovery(tmp_path, snapshot, cut):
    await run_worker(tmp_path, "finish_park", "hitl_opened")
    await run_worker(tmp_path, "finish_park", cut, recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = await restart(store, memory, "finish_park", snapshot)
            root = (await rt.get_session("crash-session")).data.root_agent_id
            await recover_fixture_session(rt, root, session_id="crash-session")
            tm = rt._task_managers.get("crash-session")
            if tm and tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            view = await rebuild_view(store, "crash-session")
            assert not view.pending_observe
            tasks = await rt.list_tasks("crash-session")
            assert tasks and all(t.data.status == "FINISHED" and t.data.outputs == "done" for t in tasks)
            rows = await store.read_range("crash-session")
            assert sum(r.event.type == EventType.TASK_FINISHED for r in rows) == 1
            assert sum(r.event.type == EventType.TASK_FINALIZED for r in rows) == 1
            starts = [r.event.run_id for r in rows if r.event.type == EventType.RUN_STARTED]
            ends = [r.event.run_id for r in rows if r.event.type == EventType.RUN_FINISHED]
            assert starts and all(ends.count(run_id) == 1 for run_id in starts)
            head = await store.committed_head("crash-session")
            again, _ = make_lifecycle_runtime(store, memory, "finish_park")
            await initialize_all(again)
            await recover_fixture_session(again, root, session_id="crash-session")
            assert await store.committed_head("crash-session") == head


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
async def test_finalized_success_survives_changed_observer_on_restart(tmp_path, snapshot):
    """A committed successful close must not become a fresh verdict after restart."""
    await run_worker(tmp_path, "finish_park", "task_finalized")
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            # The same actor output is now judged as fail by the mock observer.
            rt = await restart(store, memory, "threshold", snapshot)
            root = (await rt.get_session("crash-session")).data.root_agent_id
            await recover_fixture_session(rt, root, session_id="crash-session")
            tm = rt._task_managers["crash-session"]
            if tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            tasks = await rt.list_tasks("crash-session")
            assert [t.data.status for t in tasks] == ["FINISHED"]


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["trip_decision", "root_run_finished", "root_task_failed", "none"])
async def test_threshold_real_process_crash(tmp_path, snapshot, cut):
    await run_worker(tmp_path, "threshold", cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = await restart(store, memory, "threshold", snapshot)
            rows = await store.read_range("crash-session")
            trip = next(row for row in rows if row.event.type == EventType.FAILURE_THRESHOLD_HIT)
            root_tid, = trip.event.payload["root_task_ids"]
            root_runs = [row for row in rows if row.event.type == EventType.RUN_STARTED
                         and row.event.task_id == root_tid
                         and row.event.payload.get("initial_step") == "prepare"]
            assert root_runs
            root_failure = [row for row in rows if row.event.type == EventType.TASK_FAILED
                            and row.event.task_id == root_tid]
            assert len(root_failure) == 1
            for start in root_runs:
                ends = [row for row in rows if row.event.type == EventType.RUN_FINISHED
                        and row.event.run_id == start.event.run_id]
                assert len(ends) == 1
                assert trip.position < ends[0].position < root_failure[0].position
            task = await rt.get_task("crash-session", root_tid)
            assert task.data.status == "FAILED"
            assert task.data.error_code == TaskErrorCode.BY_THRESHOLD
            assert not any(row.event.type == EventType.SESSION_STATUS_CHANGED for row in rows)
            root_agent = (await rt.get_session("crash-session")).data.root_agent_id
            assert (await rt.get_agent(root_agent, session_id="crash-session")).data.state.status == "idle"
