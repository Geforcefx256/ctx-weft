
from tests._recovery import recover_fixture_session
"""Queued cancellation retries survive missing receipts and repeated process crashes."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.protocols import load_records
from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration._queued_session_cancel_worker import SID, drain, runtime

WORKER = Path(__file__).with_name("_queued_session_cancel_worker.py")


async def run_worker(path, fault, recovering=False):
    result = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), fault, *(["recover"] if recovering else [])],
        cwd=WORKER.parents[2], capture_output=True, text=True, timeout=40)
    assert result.returncode == (0 if fault in {"none", "failed", "unknown", "unknown_retries"} else 73), result.stdout + result.stderr


async def content(memory, tasks):
    """Include raw task history and agent finish/report records in the comparison."""
    from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext
    result = []
    view_ctx = ProviderContext(session_id=SID, tenant_id="default")
    scopes = [(MemoryScope.TASK, MemoryAddress(session_id=SID, task_id=t.id,
                agent_id=t.assigned_agent_id)) for t in tasks]
    scopes.extend((MemoryScope.AGENT, MemoryAddress(session_id=SID, agent_id=aid))
                  for aid in {t.assigned_agent_id for t in tasks})
    for scope, address in scopes:
        records = await load_records(memory, address, scope, view_ctx)
        result.extend((str(scope), address.task_id or address.agent_id, r.id, str(r.content)) for r in records)
    return sorted(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("fault", ["none", "failed", "unknown", "unknown_retries", "failed_crash", "unknown_crash",
                                    "terminated", "canceled", "finalized", "recovery_canceled", "recovery_finalized"])
async def test_queued_session_cancel_recovers(tmp_path, snapshot, fault):
    if fault.startswith("recovery_"):
        await run_worker(tmp_path, "terminated")
        await run_worker(tmp_path, fault.removeprefix("recovery_"), recovering=True)
    else:
        await run_worker(tmp_path, fault)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            history = reduce_events([r.event for r in await store.read_range(SID)], "")
            owner = next(t.assigned_agent_id for t in history.tasks.values() if t.user_prompt == "QUEUED ONE")
            if snapshot:
                await seed_snapshot(store, SID, prune_view_for_snapshot(await rebuild_view(store, SID)),
                                    cut=await store.committed_head(SID))
            rt, llm = runtime(store, memory, recovering=True)
            await initialize_all(rt)
            await rt.rebuild_session(SID, strict=True)
            # Recover the terminated agent, without reissuing cancel_session.
            await recover_fixture_session(rt, owner, session_id=SID)
            await drain(rt)
            view = await rebuild_view(store, SID)
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert llm.actors == 0
            rows = await store.read_range(SID)
            history = reduce_events([r.event for r in rows], "")
            assert len(history.tasks) == 5
            assert all(t.status == "CANCELED" and t.finalized for t in history.tasks.values())
            for task in history.tasks.values():
                canceled, = [r for r in rows if r.event.type == "TaskCanceled" and r.event.task_id == task.id]
                finalized, = [r for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == task.id]
                assert canceled.position < finalized.position
                if str(task.user_prompt).startswith("QUEUED"):
                    assert not any(r.event.type == "RunStarted" and r.event.task_id == task.id for r in rows)
                    terminated, = [r for r in rows if r.event.type == "AgentTerminated" and r.event.agent_id == task.assigned_agent_id]
                    assert terminated.position < canceled.position
                for start in [r for r in rows if r.event.type == "RunStarted" and r.event.task_id == task.id
                              and r.event.origin == "runtime"]:
                    finish, = [r for r in rows if r.event.type == "RunFinished" and r.event.run_id == start.event.run_id]
                    assert start.position < finish.position < canceled.position
            head = await store.committed_head(SID)
            before = await content(memory, history.tasks.values())
            await rt.cancel_session(SID)
            await drain(rt)
            rt2, llm2 = runtime(store, memory, recovering=True)
            await initialize_all(rt2)
            await rt2.rebuild_session(SID, strict=True)
            await recover_fixture_session(rt2, owner, session_id=SID)
            await drain(rt2)
            assert await store.committed_head(SID) == head
            assert await content(memory, history.tasks.values()) == before
            assert llm2.actors == 0
