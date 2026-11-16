
from tests._recovery import recover_fixture_session
"""Ignore public pause/cancel during the recovered TaskFinalized write."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_max_turns_observe_stop import _drained, _runtime
from tests.integration.test_observe_waiting_pause import _ContextLimitLLM


class _HeldSuccessLLM(_ContextLimitLLM):
    def __init__(self):
        super().__init__()
        self.finish = True
        self.actor_entered = asyncio.Event()
        self.actor_release = asyncio.Event()

    def complete(self, request, stream=True):
        chunks = super().complete(request, stream)
        if not any(t.name == "work__advance" for t in request.tools):
            return chunks

        async def held():
            self.actor_entered.set()
            await self.actor_release.wait()
            async for chunk in chunks:
                yield chunk
        return held()


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_skip_stop_during_recovered_finalized_write(tmp_path, monkeypatch, snapshot, after_commit, action):
    worker = Path(__file__).with_name("_max_turns_observe_crash_worker.py")
    proc = await asyncio.to_thread(subprocess.run, [sys.executable, "-u", str(worker), str(tmp_path)],
        cwd=str(worker.parents[2]), capture_output=True, text=True, timeout=35)
    assert proc.returncode == 73, proc.stdout + proc.stderr
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            sid = "crash-session"
            view = await rebuild_view(store, sid)
            intent, = view.pending_observe.values()
            assert intent["act_turns"] == 50 and intent["boundary"] == "max_turns"
            if snapshot:
                await seed_snapshot(store, sid, prune_view_for_snapshot(view),
                                    cut=await store.committed_head(sid))
            agent_id = view.sessions[sid].root_agent_id
            task_id, = view.tasks
            llm = _HeldSuccessLLM()
            rt = _runtime(llm, memory, store)
            await initialize_all(rt)
            entered, release = asyncio.Event(), asyncio.Event()
            runs = set()
            append = store.append_batch

            async def started(event):
                if (event.type == EventType.RUN_STARTED and
                        event.payload.get("initial_step") == "observe_finalization_recovery"):
                    runs.add(event.run_id)

            async def held_append(session_id, batch_id, events, *args, **kwargs):
                matching = any(e.type == EventType.TASK_FINALIZED and e.run_id in runs for e in events)
                if matching and after_commit:
                    receipt = await append(session_id, batch_id, events, *args, **kwargs)
                if matching:
                    entered.set()
                    await release.wait()
                return (receipt if matching and after_commit else
                        await append(session_id, batch_id, events, *args, **kwargs))

            monkeypatch.setattr(store, "append_batch", held_append)
            rt.event_bus.subscribe(None, started)
            tm = None
            try:
                await recover_fixture_session(rt, agent_id, session_id=sid)
                tm = rt._task_managers[sid]
                await asyncio.wait_for(entered.wait(), 5)
                signal = rt.pause_agent if action == "pause" else rt.cancel_agent
                assert await asyncio.wait_for(signal(agent_id), 2) == []
                tokens = rt._run_tokens[sid][task_id]
                assert not tokens.pause.is_paused and not tokens.cancel.is_cancelled
                assert tm.get_task(task_id).status == "ACTIVE"
                release.set()
                await asyncio.wait_for(llm.actor_entered.wait(), 5)
                # A skipped signal belongs to neither the old run nor its successor.
                successor = rt._run_tokens[sid][task_id]
                assert successor is not tokens
                assert not successor.pause.is_paused and not successor.cancel.is_cancelled
                assert not rt.hitl_registry.list_pending(sid)
                before_reply = await store.read_range(sid)
                assert not any(r.event.type in {EventType.HITL_OPENED, EventType.TASK_AWAITING_HUMAN}
                               for r in before_reply)
                llm.actor_release.set()
                await _drained(tm, timeout=8)
                assert tm.get_task(task_id).status == "FINISHED" and llm.actor_turns == 1
                rows = await store.read_range(sid)
                assert not any(r.event.type in {EventType.TASK_CANCELED, EventType.AGENT_TERMINATED}
                               for r in rows)
                assert sum(r.event.type == EventType.TASK_REQUEUED for r in rows) == 1
                final = await rebuild_view(store, sid)
                assert not final.pending_observe and not final.open_runs
                again = _runtime(llm, memory, store)
                await initialize_all(again)
                await recover_fixture_session(again, agent_id, session_id=sid)
                await _drained(again._task_managers[sid])
                assert again._task_managers[sid].get_task(task_id).status == "FINISHED"
                assert llm.actor_turns == 1
            finally:
                release.set()
                llm.actor_release.set()
                jobs = list(tm._background_asyncio_tasks | set(tm._dispatch_tasks.values())) if tm else []
                for job in jobs:
                    job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
