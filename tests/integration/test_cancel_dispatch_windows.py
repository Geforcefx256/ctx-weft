"""Cancellation survives dispatch gaps and lost commit receipts without an actor restart."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import reduce_events
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_cancel_task_tree_recovery import cancel_content, recover

WORKER = Path(__file__).with_name("_cancel_window_worker.py")
BASELINES = {}


async def run_window(path, stage, fault, recovering=False):
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), stage, fault, *(["recover"] if recovering else [])],
        capture_output=True, text=True, cwd=WORKER.parents[2], timeout=35)
    success = fault in {"none", "commit_failed", "commit_unknown", "assembly_error"} or (
        fault.startswith("run_") and not fault.endswith("_crash"))
    assert proc.returncode == (0 if success else 73), proc.stdout + proc.stderr


CASES = [(stage, fault, snapshot) for stage in ("assembly", "registration", "active")
         for fault in ("none", "commit_failed", "commit_unknown", "commit_failed_crash",
                       "commit_unknown_crash", "canceled", "finalized", "recovery_canceled", "recovery_finalized")
         for snapshot in (False, True)]
CASES += [("active", "memory", snapshot) for snapshot in (False, True)]
CASES += [(stage, fault, snapshot) for stage in ("cross_assembly", "cross_registration")
          for fault in ("none", "commit_unknown", "commit_failed_crash", "recovery_finalized")
          for snapshot in (False, True)]
CASES += [("observe", fault, snapshot) for fault in
          ("none", "commit_failed", "commit_unknown", "commit_unknown_crash", "recovery_finalized")
          for snapshot in (False, True)]
CASES += [(stage, "assembly_error", snapshot) for stage in ("assembly", "cross_assembly")
          for snapshot in (False, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,fault,snapshot", CASES)
async def test_cancel_dispatch_and_commit_windows(tmp_path, tmp_path_factory, stage, fault, snapshot):
    if stage not in BASELINES:
        normal = tmp_path_factory.mktemp("normal_" + stage)
        await run_window(normal, stage, "none")
        async with open_sqlite_event_store(normal / "events.db") as store:
            async with open_sqlite_memory(normal / "memory.db") as memory:
                BASELINES[stage] = await cancel_content(store, memory)
    if fault.startswith("recovery_"):
        await run_window(tmp_path, stage, "source")
        await run_window(tmp_path, stage, fault.removeprefix("recovery_"), recovering=True)
    else:
        await run_window(tmp_path, stage, fault)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover(store, memory, snapshot)
            rows = await store.read_range("tree-session")
            view = reduce_events([r.event for r in rows], "")
            tree = [t for t in view.tasks.values() if str(t.user_prompt).startswith("TREE")]
            assert len(tree) == (2 if stage.startswith("cross_") else 3)
            assert all(t.status == "CANCELED" and t.finalized for t in tree)
            target_name = "TREE CHILD" if stage.startswith("cross_") else "TREE GRANDCHILD"
            grand = next(t for t in tree if str(t.user_prompt) == target_name)
            assert [r.event.agent_id for r in rows if r.event.type == "AgentTerminated"] == [root]
            assert all(a.status != "terminated" for aid, a in view.agents.items() if aid != root)
            assert not view.open_runs and not view.hitl.pending and not view.pending_observe
            unrelated = [t for t in view.tasks.values() if str(t.user_prompt).startswith("UNRELATED")]
            assert len(unrelated) == 2 and all(t.status != "CANCELED" for t in unrelated)
            for task in tree:
                canceled, = [r for r in rows if r.event.type == "TaskCanceled" and r.event.task_id == task.id]
                finalized, = [r for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == task.id]
                assert canceled.position < finalized.position
                runs = [r.event.run_id for r in rows if r.event.type == "RunStarted" and r.event.task_id == task.id
                        and r.event.origin != "loop.background_observe"]
                for run in runs:
                    finished, = [r for r in rows if r.event.type == "RunFinished" and r.event.run_id == run]
                    assert finished.position < canceled.position
            if stage not in {"active", "observe"}:
                assert not any(r.event.task_id == grand.id and r.event.type in ("RunStarted", "LLMResponseFinished") for r in rows)
            assert await cancel_content(store, memory) == BASELINES[stage]
            head = await store.committed_head("tree-session")
            await recover(store, memory, False)
            assert await store.committed_head("tree-session") == head
            assert await cancel_content(store, memory) == BASELINES[stage]
