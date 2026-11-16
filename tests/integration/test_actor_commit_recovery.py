"""Uncertain actor starts and finalized receipts preserve ordering and dispatch ownership."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_cancel_dispatch_windows import run_window
from tests.integration.test_cancel_task_tree_recovery import cancel_content, recover

WORKER = Path(__file__).with_name("_actor_commit_worker.py")
BASELINES = {}
CASES = [(stage, fault, mode) for stage in ("start", "active", "observe")
         for fault in ("before", "unknown") for mode in ("hot", "cold", "recovery")]
CASES += [("start", "unknown", "recovery_actor_finished"),
          ("start", "unknown", "recovery_canceled"),
          ("active", "before", "recovery_finalized"),
          ("observe", "before", "recovery_finalized")]
CASES += [(stage, "final_only_unknown", "hot") for stage in ("active", "observe")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,fault,mode", CASES)
@pytest.mark.parametrize("snapshot", [False, True])
async def test_actor_commit_and_cancellation_recovery(tmp_path, tmp_path_factory, stage, fault, mode, snapshot):
    if stage not in BASELINES:
        baseline = tmp_path_factory.mktemp("actor_commit_normal_" + stage)
        await run_window(baseline, "registration" if stage == "start" else stage, "none")
        async with open_sqlite_event_store(baseline / "events.db") as store:
            async with open_sqlite_memory(baseline / "memory.db") as memory:
                BASELINES[stage] = await cancel_content(store, memory)
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(tmp_path), stage, fault, mode],
        capture_output=True, text=True, cwd=WORKER.parents[2], timeout=40)
    assert proc.returncode == (0 if mode == "hot" else 73), proc.stdout + proc.stderr
    if snapshot:
        async with open_sqlite_event_store(tmp_path / "events.db") as store:
            await seed_snapshot(store, "tree-session", prune_view_for_snapshot(await rebuild_view(store, "tree-session")),
                                cut=await store.committed_head("tree-session"))
    if mode.startswith("recovery"):
        cut = mode.removeprefix("recovery_") if mode != "recovery" else "startup"
        await run_window(tmp_path, "active", cut, recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover(store, memory, False)
            rows = await store.read_range("tree-session")
            view = reduce_events([r.event for r in rows], "")
            tree = [t for t in view.tasks.values() if str(t.user_prompt).startswith("TREE")]
            assert len(tree) == 3 and all(t.status == "CANCELED" and t.finalized for t in tree)
            tm = rt._task_managers["tree-session"]
            assert not tm.running_task_ids()
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert [r.event.agent_id for r in rows if r.event.type == "AgentTerminated"] == [root]
            assert all(a.status != "terminated" for aid, a in view.agents.items() if aid != root)
            assert all(t.status != "CANCELED" for t in view.tasks.values()
                       if str(t.user_prompt).startswith("UNRELATED"))
            for task in tree:
                canceled, = [r for r in rows if r.event.task_id == task.id and r.event.type == "TaskCanceled"]
                finalized, = [r for r in rows if r.event.task_id == task.id and r.event.type == "TaskFinalized"]
                assert canceled.position < finalized.position
                for started in [r for r in rows if r.event.task_id == task.id and r.event.type == "RunStarted"]:
                    finished, = [r for r in rows if r.event.run_id == started.event.run_id
                                 and r.event.type == "RunFinished"]
                    if started.event.origin == "runtime":
                        assert started.position < finished.position < canceled.position
            assert await cancel_content(store, memory) == BASELINES[stage]
            head = await store.committed_head("tree-session")
            await recover(store, memory, False)
            assert await store.committed_head("tree-session") == head
            assert await cancel_content(store, memory) == BASELINES[stage]
