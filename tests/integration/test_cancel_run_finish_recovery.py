"""Actor closure confirmation precedes cancellation; background observe stays independent."""
import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_cancel_dispatch_windows import run_window
from tests.integration.test_cancel_task_tree_recovery import cancel_content, recover

BASELINES = {}
CASES = [(fault, None) for fault in (
    "run_failed", "run_unknown", "run_failed_twice", "run_unknown_twice",
    "run_failed_crash", "run_unknown_crash")]
CASES += [(fault, cut) for fault in ("run_failed_crash", "run_unknown_crash")
          for cut in ("canceled", "finalized")]
CASES += [("run_failed_crash", "actor_finished")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["active", "observe"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("fault,second_cut", CASES)
async def test_run_finish_confirmation(tmp_path, tmp_path_factory, stage, snapshot, fault, second_cut):
    if stage not in BASELINES:
        baseline = tmp_path_factory.mktemp("run_finish_normal_" + stage)
        await run_window(baseline, stage, "none")
        async with open_sqlite_event_store(baseline / "events.db") as store:
            async with open_sqlite_memory(baseline / "memory.db") as memory:
                BASELINES[stage] = await cancel_content(store, memory)
    await run_window(tmp_path, stage, fault)
    if snapshot:
        async with open_sqlite_event_store(tmp_path / "events.db") as store:
            await seed_snapshot(store, "tree-session", prune_view_for_snapshot(await rebuild_view(store, "tree-session")),
                                cut=await store.committed_head("tree-session"))
    if second_cut:
        # Crash again during startup run repair or subsequent task cancellation.
        await run_window(tmp_path, stage, second_cut, recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            _, root = await recover(store, memory, False)
            rows = await store.read_range("tree-session")
            view = reduce_events([r.event for r in rows], "")
            tree = [t for t in view.tasks.values() if str(t.user_prompt).startswith("TREE")]
            assert len(tree) == 3 and all(t.status == "CANCELED" and t.finalized for t in tree)
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert [r.event.agent_id for r in rows if r.event.type == "AgentTerminated"] == [root]
            assert all(a.status != "terminated" for aid, a in view.agents.items() if aid != root)
            unrelated = [t for t in view.tasks.values() if str(t.user_prompt).startswith("UNRELATED")]
            assert len(unrelated) == 2 and all(t.status != "CANCELED" for t in unrelated)
            for task in tree:
                canceled, = [r for r in rows if r.event.task_id == task.id and r.event.type == "TaskCanceled"]
                finalized, = [r for r in rows if r.event.task_id == task.id and r.event.type == "TaskFinalized"]
                assert canceled.position < finalized.position
                for started in [r for r in rows if r.event.task_id == task.id and r.event.type == "RunStarted"]:
                    finished, = [r for r in rows if r.event.run_id == started.event.run_id
                                 and r.event.type == "RunFinished"]
                    if started.event.origin == "runtime":
                        assert finished.position < canceled.position
            assert await cancel_content(store, memory) == BASELINES[stage]
            head = await store.committed_head("tree-session")
            await recover(store, memory, False)
            assert await store.committed_head("tree-session") == head
            assert await cancel_content(store, memory) == BASELINES[stage]
