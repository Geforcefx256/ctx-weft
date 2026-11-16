"""User cancellation closes actual streams before retracting their event windows."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import rebuild_view, reduce_events
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext, load_records
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._session_cancel_worker import RETRACTED, SID, runtime, wait_until
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM

WORKER = Path(__file__).with_name("_session_cancel_worker.py")


async def run_worker(path, mode, cut):
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), mode, cut],
        capture_output=True, text=True, cwd=WORKER.parents[2], timeout=45)
    assert proc.returncode == (0 if cut == "none" else 73), proc.stdout + proc.stderr


def assert_order(events):
    view = reduce_events(events, "")
    assert not view.open_runs and not view.hitl.pending
    for start in [e for e in events if e.type == "RunStarted" and e.origin == "runtime"]:
        finish, = [e for e in events if e.type == "RunFinished" and e.run_id == start.run_id]
        assert events.index(start) < events.index(finish)
        for canceled in [e for e in events if e.type == "TaskCanceled" and e.task_id == start.task_id]:
            assert events.index(finish) < events.index(canceled)
    for canceled in [e for e in events if e.type == "TaskCanceled"]:
        finalized = [e for e in events if e.type == "TaskFinalized" and e.task_id == canceled.task_id]
        assert len(finalized) == 1
        assert events.index(canceled) < events.index(finalized[0])
    return view


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["before", "arrival", "after", "hot", "unknown"])
async def test_user_cancel_releases_stream_dispatch_and_slots(tmp_path, mode):
    await run_worker(tmp_path, mode, "none")
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            events = [r.event for r in await store.read_range(SID)]
            view = assert_order(events)
            if mode in ("before", "arrival", "hot"):
                for aid in view.agents:
                    records = await load_records(memory, MemoryAddress(session_id=SID, agent_id=aid),
                        MemoryScope.AGENT, ProviderContext(session_id=SID, tenant_id="default"))
                    assert not any(RETRACTED in str(r.content) for r in records)
            head = await store.committed_head(SID)
            await initialize_all(runtime(store, memory, _FinishEveryTaskLLM(), recovering=True))
            assert await store.committed_head(SID) == head


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,cut", [
    ("before", "stream_closed"), ("before", "round_discarded"),
    ("before", "agent_terminated"), ("hot", "stream_closed"),
    ("hot", "round_discarded"), ("hot", "agent_terminated"),
    ("unknown", "unknown_committed"), ("after", "stream_closed"),
    ("unknown", "recover_prompt"),
])
async def test_user_cancel_process_crash_recovery(tmp_path, mode, cut):
    await run_worker(tmp_path, mode, "unknown_committed" if cut == "recover_prompt" else cut)
    if cut == "recover_prompt":
        await run_worker(tmp_path, "recover", "recover_prompt")
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = runtime(store, memory, _FinishEveryTaskLLM(), recovering=True)
            await initialize_all(rt)
            await rt.rebuild_session(SID, strict=True)
            # A crash before any durable cancellation fact leaves the previous
            # conversation recoverable. The user may issue Cancel again.
            await rt.cancel_session(SID)
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm.running_task_ids() and not tm._background_asyncio_tasks)
            assert not rt._run_tokens.get(SID) and not tm._dispatch_tasks
            events = [r.event for r in await store.read_range(SID)]
            view = assert_order(events)
            raw = []
            for task in view.tasks.values():
                raw.extend(await load_records(memory,
                    MemoryAddress(session_id=SID, task_id=task.id, agent_id=task.assigned_agent_id),
                    MemoryScope.TASK, ProviderContext(session_id=SID, tenant_id="default")))
            assert any(RETRACTED in str(r.content) for r in raw) == (mode in ("after", "unknown"))
            if mode in ("after", "unknown"):
                assert sum(RETRACTED in str(r.content) for r in raw) == 1
            head = await store.committed_head(SID)
            await rt.cancel_session(SID)
            assert await store.committed_head(SID) == head
            rt2 = runtime(store, memory, _FinishEveryTaskLLM(), recovering=True)
            await initialize_all(rt2)
            assert await store.committed_head(SID) == head
            assert not (await rebuild_view(store, SID)).open_runs
