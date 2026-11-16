"""Frozen provisional commits and agent settlement survive retries and repeated crashes."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext, load_records
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_cancel_task_tree_recovery import recover

WORKER = Path(__file__).with_name("_commit_repair_worker.py")
BASELINES = {}


async def run_worker(path, stage, fault, cut):
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), stage, fault, cut],
        capture_output=True, text=True, cwd=WORKER.parents[2], timeout=45)
    assert proc.returncode == (0 if cut == "hot" else 73), proc.stdout + proc.stderr


async def reports(store, memory):
    view = reduce_events([r.event for r in await store.read_range("tree-session")], "")
    names = {t.id: str(t.user_prompt) for t in view.tasks.values()}
    result = []
    for aid in view.agents:
        records = await load_records(memory, MemoryAddress(session_id="tree-session", agent_id=aid),
            MemoryScope.AGENT, ProviderContext(session_id="tree-session", tenant_id="default"),
            kinds=[MemoryKind.CONVERSATION_TURN])
        for record in records:
            if record.metadata.get("finish_policy", {}).get("outcome") != "cancelled":
                continue
            content = str(record.content)
            for tid, name in names.items():
                content = content.replace(tid, name)
            result.append((names[record.metadata["origin_task_id"]], record.role, content))
    return sorted(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["batch", "idle"])
@pytest.mark.parametrize("fault", ["before", "unknown"])
@pytest.mark.parametrize("cut", ["hot", "cold", "recover_run", "recover_finalized"])
@pytest.mark.parametrize("snapshot", [False, True])
async def test_commit_repair_recovery(tmp_path, tmp_path_factory, stage, fault, cut, snapshot):
    if stage not in BASELINES:
        baseline = tmp_path_factory.mktemp("normal_commit_repair_" + stage)
        await run_worker(baseline, stage, "none", "hot")
        async with open_sqlite_event_store(baseline / "events.db") as store:
            async with open_sqlite_memory(baseline / "memory.db") as memory:
                BASELINES[stage] = await reports(store, memory)
                assert BASELINES[stage]
    await run_worker(tmp_path, stage, fault, "hot" if cut == "hot" else "cold")
    if snapshot:
        async with open_sqlite_event_store(tmp_path / "events.db") as store:
            await seed_snapshot(store, "tree-session", prune_view_for_snapshot(await rebuild_view(store, "tree-session")),
                                cut=await store.committed_head("tree-session"))
    if cut.startswith("recover_"):
        await run_worker(tmp_path, stage, fault, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover(store, memory, False)
            rows = await store.read_range("tree-session")
            view = reduce_events([r.event for r in rows], "")
            canceled = [t for t in view.tasks.values() if t.status == "CANCELED"]
            assert len(canceled) == (1 if stage == "batch" else 2)
            assert all(t.finalized for t in canceled)
            assert not rt._task_managers["tree-session"].running_task_ids()
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert [r.event.agent_id for r in rows if r.event.type == "AgentTerminated"] == [root]
            for task in canceled:
                event, = [r for r in rows if r.event.type == "TaskCanceled" and r.event.task_id == task.id]
                finalized, = [r for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == task.id]
                assert event.position < finalized.position
                for start in [r for r in rows if r.event.type == "RunStarted" and r.event.task_id == task.id
                              and r.event.origin == "runtime"]:
                    finish, = [r for r in rows if r.event.type == "RunFinished" and r.event.run_id == start.event.run_id]
                    assert start.position < finish.position < event.position
                if task.assigned_agent_id != root:
                    agent = view.agents[task.assigned_agent_id]
                    if agent.current_task_id == task.id:
                        assert agent.status == "idle"
            assert all(t.status != "CANCELED" for t in view.tasks.values()
                       if str(t.user_prompt).startswith("UNRELATED"))
            assert await reports(store, memory) == BASELINES[stage]
            head = await store.committed_head("tree-session")
            await recover(store, memory, False)
            assert await store.committed_head("tree-session") == head
            assert await reports(store, memory) == BASELINES[stage]
