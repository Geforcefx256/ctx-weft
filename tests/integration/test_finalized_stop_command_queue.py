
from tests._recovery import recover_fixture_session
"""A stop issued during finalization must never queue behind a public command."""

import asyncio
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_max_turns_observe_stop import _drained, _runtime
from tests.integration.test_observe_finalized_stop_skip import _HeldSuccessLLM


@pytest.mark.parametrize("operation", ["pause_agent", "pause_session", "cancel_session"])
@pytest.mark.parametrize("after_commit", [False, True])
async def test_stop_during_finalized_write_skips_public_command_queue(tmp_path, monkeypatch, operation, after_commit):
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
            agent = view.sessions[sid].root_agent_id
            task_id, = view.tasks
            llm = _HeldSuccessLLM()
            rt = _runtime(llm, memory, store)
            await initialize_all(rt)
            write_entered, write_release, command_entered, command_release = [asyncio.Event() for _ in range(4)]
            append, normalize = store.append_batch, rt._validate_and_normalize_content
            runs = set()
            dispatch_job = signal = None

            async def started(event):
                if (event.type == EventType.RUN_STARTED and
                        event.payload.get("initial_step") == "observe_finalization_recovery"):
                    runs.add(event.run_id)

            async def held_append(session_id, batch_id, events, *args, **kwargs):
                matching = any(e.type == EventType.TASK_FINALIZED and e.run_id in runs for e in events)
                if matching and after_commit:
                    receipt = await append(session_id, batch_id, events, *args, **kwargs)
                if matching:
                    write_entered.set()
                    await write_release.wait()
                return (receipt if matching and after_commit else
                        await append(session_id, batch_id, events, *args, **kwargs))

            async def held_normalize(*args, **kwargs):
                if asyncio.current_task() is dispatch_job:
                    command_entered.set()
                    await command_release.wait()
                return await normalize(*args, **kwargs)

            monkeypatch.setattr(store, "append_batch", held_append)
            monkeypatch.setattr(rt, "_validate_and_normalize_content", held_normalize)
            rt.event_bus.subscribe(None, started)
            await recover_fixture_session(rt, agent, session_id=sid)
            tm = rt._task_managers[sid]
            try:
                await asyncio.wait_for(write_entered.wait(), 5)
                old_tokens = rt._run_tokens[sid][task_id]
                dispatch_job = asyncio.create_task(rt.dispatch_task(
                    sid, "Additional work", agent_id=agent,
                    settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"}))
                await asyncio.wait_for(command_entered.wait(), 5)
                assert rt._session_lock_owners[sid] is dispatch_job
                assert tm.is_writing_finalized(task_id)
                initialize = AsyncMock(wraps=rt.initialize)
                monkeypatch.setattr(rt, "initialize", initialize)
                target = agent if operation == "pause_agent" else sid
                signal = asyncio.create_task(getattr(rt, operation)(target))
                # Neither gate is released: the stop must complete immediately,
                # before initialization or waiting behind the public dispatch.
                result = await asyncio.wait_for(asyncio.shield(signal), 1)
                assert result == ([] if operation == "pause_agent" else False)
                initialize.assert_not_awaited()
                assert not old_tokens.pause.is_paused and not old_tokens.cancel.is_cancelled
                assert not rt._pausing and not rt._pause_claimed and not tm._cancelled
                write_release.set()
                await asyncio.wait_for(llm.actor_entered.wait(), 5)
                new_tokens = rt._run_tokens[sid][task_id]
                assert new_tokens is not old_tokens
                assert rt._session_lock_owners[sid] is dispatch_job
                assert not new_tokens.pause.is_paused and not new_tokens.cancel.is_cancelled
                command_release.set()
                auxiliary = await asyncio.wait_for(dispatch_job, 5)
                assert not new_tokens.pause.is_paused and not new_tokens.cancel.is_cancelled
                llm.actor_release.set()
                await _drained(tm, timeout=8)
                assert tm.get_task(task_id).status == tm.get_task(auxiliary.task_id).status == "FINISHED"
                rows = await store.read_range(sid)
                assert not any(r.event.type in {EventType.TASK_CANCELED, EventType.AGENT_TERMINATED} for r in rows)
                assert sum(r.event.type == EventType.TASK_REQUEUED and r.event.task_id == task_id for r in rows) == 1
                final = await rebuild_view(store, sid)
                assert not final.pending_observe and not final.open_runs and not rt.hitl_registry.list_pending(sid)
            finally:
                write_release.set()
                command_release.set()
                llm.actor_release.set()
                jobs = list(tm._background_asyncio_tasks | set(tm._dispatch_tasks.values()))
                jobs += [job for job in (dispatch_job, signal) if job is not None]
                for job in jobs:
                    job.cancel()
                await asyncio.gather(*jobs, return_exceptions=True)
