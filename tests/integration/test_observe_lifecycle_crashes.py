
from tests._recovery import recover_fixture_session
"""All four observe stages on actual foreground and recap execution paths."""

import asyncio

import pytest

from ctx_weft.protocols import load_records
from ctx_weft.core.loop.task_memory_fold import task_record_kind
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime
from tests.integration.test_lifecycle_crash_boundaries import restart, run_worker



def assert_lifecycles(rows):
    requested, started, completed, applied = {}, {}, {}, {}
    mechanical = set()
    for row in rows:
        ev, payload = row.event, row.event.payload
        if ev.type == EventType.HITL_OPENED:
            payload = payload.get("observe_request", {})
        identity = payload.get("observe_id")
        if not identity:
            continue
        if ev.type in {EventType.OBSERVE_REQUESTED, EventType.HITL_OPENED}:
            requested[identity] = row.position
            if payload.get("verdict"):
                mechanical.add(identity)
        elif ev.type == EventType.OBSERVE_STARTED:
            started.setdefault(identity, []).append(row.position)
        elif ev.type == EventType.OBSERVE_COMPLETED:
            completed.setdefault(identity, []).append(row.position)
        elif ev.type == EventType.OBSERVE_APPLIED:
            applied.setdefault(identity, []).append(row.position)
    assert requested
    assert requested.keys() == completed.keys() == applied.keys() == started.keys()
    for identity, position in requested.items():
        assert len(started[identity]) == len(completed[identity]) == len(applied[identity]) == 1
        assert position < min(started[identity]) < completed[identity][0] < applied[identity][0]
    for row in rows:
        identity = row.event.causation_id or row.event.payload.get("observe_id")
        if identity in completed and row.event.type in {
                EventType.TASK_FINALIZED, EventType.BLACKBOARD_PUBLISHED,
                EventType.MEMORY_INGESTED, EventType.TASK_FINISHED, EventType.TASK_FAILED,
                EventType.TASK_REQUEUED}:
            prerequisite = requested[identity] if identity in mechanical else completed[identity][0]
            assert prerequisite < row.position < applied[identity][0]
    assert not any(r.event.type in {"TaskRecapRequested", "TaskRecapStarted", "TaskRecapDone"} for r in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["foreground", "mechanical"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["observe_requested", "observe_started", "observe_completed",
                                "task_finalized", "task_finished", "observe_applied", "finish_memory",
                                "blackboard_memory", "none"])
async def test_foreground_and_recap_real_crash(tmp_path, mode, snapshot, cut):
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = await restart(store, memory, mode, snapshot)
            root = (await rt.get_session("crash-session")).data.root_agent_id
            await recover_fixture_session(rt, root, session_id="crash-session")
            tm = rt._task_managers.get("crash-session")
            while tm and tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            tasks = await rt.list_tasks("crash-session")
            assert [(t.data.status, t.data.outputs) for t in tasks] == [("FINISHED", "done")]
            view = await rebuild_view(store, "crash-session")
            assert not view.pending_observe and not view.open_runs
            rows = await store.read_range("crash-session")
            assert_lifecycles(rows)
            assert sum(r.event.type == EventType.TASK_FINALIZED for r in rows) == 1
            assert sum(r.event.type == EventType.TASK_FINISHED for r in rows) == 1
            finished = next(r for r in rows if r.event.type == EventType.TASK_FINISHED)
            actor_runs = [r.event.run_id for r in rows if r.event.type == EventType.RUN_STARTED
                          and r.event.payload.get("initial_step") == "prepare"]
            assert actor_runs
            for run_id in actor_runs:
                ends = [r for r in rows if r.event.type == EventType.RUN_FINISHED and r.event.run_id == run_id]
                assert len(ends) == 1 and ends[0].position < finished.position
            from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext
            scope = MemoryAddress(session_id="crash-session", agent_id=root)
            provider_ctx = ProviderContext(session_id="crash-session", tenant_id="default")
            publications = await load_records(memory, MemoryAddress(session_id="crash-session"), MemoryScope.SESSION, provider_ctx,
                                                  kinds=[MemoryKind.PUBLICATION])
            assert len(publications) == 1
            turns = await load_records(memory, scope, MemoryScope.AGENT, provider_ctx,
                                           kinds=[MemoryKind.CONVERSATION_TURN])
            assert sum(task_record_kind(turn) == "finish" for turn in turns) == 1

            head = await store.committed_head("crash-session")
            again, _ = make_lifecycle_runtime(store, memory, mode)
            await initialize_all(again)
            await recover_fixture_session(again, root, session_id="crash-session")
            assert await store.committed_head("crash-session") == head


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["summary_requested", "summary_started", "summary_completed", "summary_applied"])
async def test_recap_only_real_crash(tmp_path, snapshot, cut):
    from tests.integration.test_lifecycle_crash_boundaries import test_observe_real_process_crash
    boundary = {"summary_requested": "hitl_opened", "summary_started": "recap_started",
                "summary_completed": "observe_completed", "summary_applied": "recap_done"}[cut]
    await test_observe_real_process_crash(tmp_path, snapshot, "interrupt", boundary)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["foreground", "mechanical"])
@pytest.mark.parametrize("snapshot", [False, True])
async def test_crash_again_while_applying_foreground_result(tmp_path, mode, snapshot):
    # Mechanical Completed is now the asynchronous report: it may follow finalize.
    # Stop before finalize, then crash once more while applying the saved decision.
    await run_worker(tmp_path, mode, "observe_requested" if mode == "mechanical" else "observe_completed")
    await run_worker(tmp_path, mode, "task_finalized", recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = await restart(store, memory, mode, snapshot)
            root = (await rt.get_session("crash-session")).data.root_agent_id
            await recover_fixture_session(rt, root, session_id="crash-session")
            tm = rt._task_managers.get("crash-session")
            while tm and tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            tasks = await rt.list_tasks("crash-session")
            assert [(t.data.status, t.data.outputs) for t in tasks] == [("FINISHED", "done")]
            rows = await store.read_range("crash-session")
            assert_lifecycles(rows)
            assert sum(r.event.type == EventType.TASK_FINALIZED for r in rows) == 1
            assert sum(r.event.type == EventType.TASK_FINISHED for r in rows) == 1
            head = await store.committed_head("crash-session")
            again, _ = make_lifecycle_runtime(store, memory, mode)
            await initialize_all(again)
            await recover_fixture_session(again, root, session_id="crash-session")
            assert await store.committed_head("crash-session") == head


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("mode", ["retry", "mechanical_retry"])
async def test_requeued_foreground_decision_enqueues_after_crash(tmp_path, snapshot, mode):
    await run_worker(tmp_path, mode, "task_requeued")
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = await restart(store, memory, "foreground", snapshot)
            root = (await rt.get_session("crash-session")).data.root_agent_id
            await recover_fixture_session(rt, root, session_id="crash-session")
            tm = rt._task_managers["crash-session"]
            async with asyncio.timeout(20):
                while (tm.running_task_ids() or tm._background_asyncio_tasks or
                       (await rebuild_view(store, "crash-session")).pending_observe):
                    if tm._background_asyncio_tasks:
                        await asyncio.gather(*list(tm._background_asyncio_tasks))
                    else:
                        await asyncio.sleep(0.01)
            tasks = await rt.list_tasks("crash-session")
            assert [t.data.status for t in tasks] == ["FINISHED"]
            rows = await store.read_range("crash-session")
            assert sum(r.event.type == EventType.TASK_REQUEUED for r in rows) == 1
            assert rt._llm.actor_count == 1
            assert rt._llm.judgements == 1  # Only the next attempt; never rejudge the saved decision.
            assert_lifecycles(rows)
            assert not (await rebuild_view(store, "crash-session")).pending_observe
