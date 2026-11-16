
from tests._recovery import recover_fixture_session
"""A saved foreground decision must remain cancellable through public recovery."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.protocols import MemoryKind
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_max_turns_observe_stop import _drained, _FiftyTurnsLLM, _runtime


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("phase", ["immediate", "before_fold", "after_fold", "before_disposition"])
async def test_cancel_saved_foreground_decision_via_public_recovery(tmp_path, monkeypatch, snapshot, phase):
    worker = Path(__file__).with_name("_max_turns_observe_crash_worker.py")
    proc = await asyncio.to_thread(subprocess.run, [sys.executable, "-u", str(worker), str(tmp_path)],
        cwd=str(worker.parents[2]), capture_output=True, text=True, timeout=30)
    assert proc.returncode == 73, proc.stdout + proc.stderr
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            session_id = "crash-session"
            view = await rebuild_view(store, session_id)
            intent, = view.pending_observe.values()
            assert intent["act_turns"] == 50 and intent["boundary"] == "max_turns"
            if snapshot:
                await seed_snapshot(store, session_id, prune_view_for_snapshot(view),
                                    cut=await store.committed_head(session_id))
            agent_id = view.sessions[session_id].root_agent_id
            task_id, = view.tasks
            llm = _FiftyTurnsLLM()
            rt = _runtime(llm, memory, store)
            await initialize_all(rt)
            entered, release, interrupted = [asyncio.Event() for _ in range(3)]
            original_fold = memory._commit_organize
            recovery_runs = set()
            acks = []

            async def fold(ids, replacements, ctx):
                if not any(r.kind == MemoryKind.SUMMARY for r in replacements):
                    return await original_fold(ids, replacements, ctx)
                if phase == "after_fold":
                    result = await original_fold(ids, replacements, ctx)
                if phase in {"before_fold", "after_fold"}:
                    entered.set()
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        interrupted.set()
                        raise
                return result if phase == "after_fold" else await original_fold(ids, replacements, ctx)

            async def at_run_boundary(event):
                if (event.type == EventType.RUN_STARTED and
                        event.payload.get("initial_step") == "observe_finalization_recovery"):
                    recovery_runs.add(event.run_id)
                if event.type == EventType.RUN_FINISHED and event.run_id in recovery_runs:
                    if phase == "before_disposition":
                        acks.append(await rt.cancel_agent(agent_id))

            monkeypatch.setattr(memory, "_commit_organize", fold)
            rt.event_bus.subscribe(None, at_run_boundary)
            tm = None
            try:
                await recover_fixture_session(rt, agent_id, session_id=session_id)
                tm = rt._task_managers[session_id]
                if phase == "immediate":
                    acks.append(await rt.cancel_agent(agent_id))
                elif phase in {"before_fold", "after_fold"}:
                    await asyncio.wait_for(entered.wait(), 3)
                    acks.append(await rt.cancel_agent(agent_id))
                    # Do not release the fold: cancellation must interrupt it.
                    await asyncio.wait_for(interrupted.wait(), 2)
                await _drained(tm)
                assert acks == [[agent_id]]
                assert tm.get_task(task_id).status == "CANCELED"
                assert not tm.running_task_ids() and not tm._queue.peek_all()
                assert llm.actor_turns == llm.reviews == 0
                final = await rebuild_view(store, session_id)
                assert not final.pending_observe and not final.open_runs
                rows = await store.read_range(session_id)
                canceled, = [r for r in rows if r.event.type == EventType.TASK_CANCELED]
                assert not any(r.position > canceled.position and r.event.type in {
                    EventType.TASK_REQUEUED, EventType.TASK_STARTED} for r in rows)
                applied, = [r for r in rows if r.event.type == EventType.OBSERVE_APPLIED]
                finalized, = [r for r in rows if r.event.type == EventType.TASK_FINALIZED
                              and r.position > canceled.position]
                assert canceled.position < finalized.position < applied.position
                await rt.cancel_agent(agent_id)
                await recover_fixture_session(rt, agent_id, session_id=session_id)
                await _drained(tm)
                # Rebuild and recover once more to cover durable cancellation.
                again = _runtime(llm, memory, store)
                await initialize_all(again)
                await recover_fixture_session(again, agent_id, session_id=session_id)
                await _drained(again._task_managers[session_id])
                assert again._task_managers[session_id].get_task(task_id).status == "CANCELED"
                assert not (await rebuild_view(store, session_id)).pending_observe
            finally:
                release.set()
                jobs = list(tm._background_asyncio_tasks | set(tm._dispatch_tasks.values())) if tm else []
                for job in jobs:
                    job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
