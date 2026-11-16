
from tests._recovery import recover_fixture_session
"""Same-agent child reports survive every handoff cut without losing the answer."""

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
from tests.integration._same_agent_observe_worker import make_child_runtime
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles

WORKER = Path(__file__).with_name("_same_agent_observe_worker.py")
BASELINES = {}


async def run_child(path, order, size, cut, recovering=False):
    result = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), order, size, cut, *(["recover"] if recovering else [])],
        capture_output=True, text=True, cwd=WORKER.parents[2], timeout=40)
    assert result.returncode == (0 if cut == "none" else 73), result.stdout + result.stderr


async def content(store, memory):
    rows = await store.read_range("child-session")
    view = reduce_events([r.event for r in rows], "")
    child, = [t for t in view.tasks.values() if t.parent_task_id]
    root = view.sessions["child-session"].root_agent_id
    ctx = ProviderContext(session_id="child-session", tenant_id="default")
    scope = MemoryAddress(session_id="child-session", agent_id=root, task_id=child.id)
    raw = await load_records(memory, scope, MemoryScope.TASK, ctx,
                                kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY])
    turns = await load_records(memory, MemoryAddress(session_id="child-session", agent_id=root),
                                  MemoryScope.AGENT, ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    finish = [r for r in turns if r.metadata.get("origin_task_id") == child.id]
    assert child.parent_task_id in view.tasks
    normalize = lambda value: str(value).replace(child.id, "<child>").replace(child.parent_task_id, "<parent>")
    return [[(r.role, normalize(r.content)) for r in records] for records in (raw, finish)]


async def recover(store, memory, size, snapshot=False):
    if snapshot:
        await seed_snapshot(store, "child-session", prune_view_for_snapshot(await rebuild_view(store, "child-session")),
                            cut=await store.committed_head("child-session"))
    rt, llm = make_child_runtime(store, memory, size, recovering=True)
    await initialize_all(rt)
    root = (await rt.get_session("child-session")).data.root_agent_id
    await recover_fixture_session(rt, root, session_id="child-session")
    tm = rt._task_managers["child-session"]
    async with asyncio.timeout(20):
        while tm.running_task_ids() or tm._background_asyncio_tasks:
            if tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            await asyncio.sleep(.01)
    return rt, llm


CASES = [(order, size, snapshot, cut) for order in ("early", "late") for size in ("short", "long")
         for snapshot in (False, True) for cut in
         ("completed", "finish_slot", "task_finalized", "report_replaced", "before_applied")]
CASES += [(order, "long", snapshot, "raw_folded") for order in ("early", "late") for snapshot in (False, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("order,size,snapshot,cut", CASES)
async def test_child_handoff_matches_normal(tmp_path, tmp_path_factory, order, size, snapshot, cut):
    if (order, size) not in BASELINES:
        path = tmp_path_factory.mktemp(f"child_{order}_{size}")
        await run_child(path, order, size, "none")
        async with open_sqlite_event_store(path / "events.db") as store:
            async with open_sqlite_memory(path / "memory.db") as memory:
                BASELINES[order, size] = await content(store, memory)
    await run_child(tmp_path, order, size, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, llm = await recover(store, memory, size, snapshot)
            actual = await content(store, memory)
            assert actual == BASELINES[order, size]
            assert "FORMAL REPORT" in str(actual[1])
            assert "no process recap" not in str(actual[1])
            assert sum("CHILD FINAL ANSWER" in text for _, text in actual[0] + actual[1]) == 1
            assert ("CHILD FINAL ANSWER" in str(actual[0])) == (size == "short")
            rows = await store.read_range("child-session")
            view = reduce_events([r.event for r in rows], "")
            child, = [t for t in view.tasks.values() if t.parent_task_id]
            requested, = [r for r in rows if r.event.type == "ObserveRequested" and r.event.task_id == child.id]
            assert requested.event.payload["report_boundary"] in ("finish", "normal")
            completed = next(r for r in rows if r.event.type == "ObserveCompleted" and r.event.task_id == child.id)
            finalized = next(r for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == child.id)
            if cut not in ("finish_slot",):
                assert (completed.position < finalized.position) == (order == "early")
            finished = next(r for r in rows if r.event.type == "TaskFinished" and r.event.task_id == child.id)
            applied = next(r for r in rows if r.event.type == "ObserveApplied" and r.event.task_id == child.id)
            assert finished.position < applied.position
            actor_responses = [r for r in rows if r.event.type == "LLMResponseFinished"
                               and r.event.origin == "loop.act" and r.event.task_id == child.id]
            assert len(actor_responses) == 1
            assert_lifecycles(rows)
            assert not (await rebuild_view(store, "child-session")).pending_observe
            head = await store.committed_head("child-session")
            again, replay_llm = await recover(store, memory, size)
            assert replay_llm.new_actor_calls == 0
            assert await store.committed_head("child-session") == head
            assert await content(store, memory) == actual


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["early", "late"])
@pytest.mark.parametrize("size", ["short", "long"])
@pytest.mark.parametrize("snapshot", [False, True])
async def test_child_handoff_survives_second_crash(tmp_path, tmp_path_factory, order, size, snapshot):
    if (order, size) not in BASELINES:
        path = tmp_path_factory.mktemp(f"normal_double_{order}_{size}")
        await run_child(path, order, size, "none")
        async with open_sqlite_event_store(path / "events.db") as store:
            async with open_sqlite_memory(path / "memory.db") as memory:
                BASELINES[order, size] = await content(store, memory)
    await run_child(tmp_path, order, size, "completed")
    await run_child(tmp_path, order, size, "report_replaced", recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            await recover(store, memory, size, snapshot)
            actual = await content(store, memory)
            assert actual == BASELINES[order, size]
            assert_lifecycles(await store.read_range("child-session"))
            head = await store.committed_head("child-session")
            await recover(store, memory, size)
            assert await store.committed_head("child-session") == head
            assert await content(store, memory) == actual
