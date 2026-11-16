
from tests._recovery import recover_fixture_session
"""cancel_agent cancels the current task tree without selecting unrelated tasks."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext, load_records
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration._cancel_tree_worker import make_tree_runtime

WORKER = Path(__file__).with_name("_cancel_tree_worker.py")
BASELINE = None


async def run_tree(path, cut, *, recovering=False):
    result = await asyncio.to_thread(subprocess.run, [sys.executable, str(WORKER), str(path), cut, *(["recover"] if recovering else [])],
        capture_output=True, text=True, cwd=WORKER.parents[2], timeout=35)
    assert result.returncode == (0 if cut == "none" or cut.endswith("commit_failed") else 73), result.stdout + result.stderr


async def recover(store, memory, snapshot):
    if snapshot:
        await seed_snapshot(store, "tree-session", prune_view_for_snapshot(await rebuild_view(store, "tree-session")),
                            cut=await store.committed_head("tree-session"))
    rt, llm = make_tree_runtime(store, memory, recovering=True)
    await initialize_all(rt)
    root = (await rt.get_session("tree-session")).data.root_agent_id
    await recover_fixture_session(rt, root, session_id="tree-session")
    tm = rt._task_managers["tree-session"]
    while tm._background_asyncio_tasks:
        await asyncio.gather(*list(tm._background_asyncio_tasks))
    assert llm.actors == 0
    return rt, root


async def cancel_content(store, memory):
    rows = await store.read_range("tree-session")
    view = reduce_events([r.event for r in rows], "")
    tree = [t for t in view.tasks.values() if str(t.user_prompt).startswith("TREE")]
    names = {t.id: str(t.user_prompt) for t in tree}
    result = []
    for aid in {t.assigned_agent_id or t.creator_agent_id for t in tree}:
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
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["none", "agent_terminated", "root_canceled", "descendant_canceled", "cancel_slot", "finalized",
    "agent_commit_failed", "task_cancel_commit_failed", "task_cancel_commit_failed_crash",
    "child_cancel_commit_failed", "child_cancel_commit_failed_crash",
    "finalized_commit_failed", "finalized_commit_failed_crash", "recovery_cancel_slot", "recovery_finalized"])
async def test_task_tree_cancellation_recovers(tmp_path, tmp_path_factory, snapshot, cut):
    global BASELINE
    if BASELINE is None:
        baseline_path = tmp_path_factory.mktemp("normal_cancel_tree")
        await run_tree(baseline_path, "none")
        async with open_sqlite_event_store(baseline_path / "events.db") as store:
            async with open_sqlite_memory(baseline_path / "memory.db") as memory:
                BASELINE = await cancel_content(store, memory)
    expected = BASELINE
    if cut.startswith("recovery_"):
        await run_tree(tmp_path, "agent_terminated")
        await run_tree(tmp_path, cut.removeprefix("recovery_"), recovering=True)
    else:
        await run_tree(tmp_path, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, root = await recover(store, memory, snapshot)
            rows = await store.read_range("tree-session")
            view = reduce_events([r.event for r in rows], "")
            tree = [t for t in view.tasks.values() if str(t.user_prompt).startswith("TREE")]
            unrelated = [t for t in view.tasks.values() if str(t.user_prompt).startswith("UNRELATED")]
            assert len(tree) == 3 and all(t.status == "CANCELED" and t.finalized for t in tree)
            assert len(unrelated) == 2 and all(t.status != "CANCELED" for t in unrelated)
            assert view.agents[root].status == "terminated"
            terminated, = [r for r in rows if r.event.type == "AgentTerminated"]
            assert terminated.event.agent_id == root
            assert all(a.status != "terminated" for aid, a in view.agents.items() if aid != root)
            assert all(rt._agent_lifecycle_manager.status_of(aid) != "terminated"
                       for aid in view.agents if aid != root)
            assert all(view.agents[t.assigned_agent_id].status != "terminated" for t in unrelated)
            assert not view.pending_observe and not view.hitl.pending and not view.open_runs
            for task in tree:
                canceled, = [r for r in rows if r.event.type == "TaskCanceled" and r.event.task_id == task.id]
                finalized, = [r for r in rows if r.event.type == "TaskFinalized"
                              and r.event.task_id == task.id and r.event.payload.get("outcome") == "cancelled"]
                assert terminated.position < canceled.position < finalized.position
                runs = [r.event.run_id for r in rows if r.event.type == "RunStarted"
                        and r.event.origin != "loop.background_observe" and r.event.task_id == task.id]
                for run in runs:
                    finished = next(r for r in rows if r.event.type == "RunFinished" and r.event.run_id == run)
                    assert finished.position < canceled.position
            assert await cancel_content(store, memory) == expected
            head = await store.committed_head("tree-session")
            await recover(store, memory, False)
            assert await store.committed_head("tree-session") == head
            assert await cancel_content(store, memory) == expected
