
from tests._recovery import recover_fixture_session
"""One logical observation survives producer/report races on the public runtime path."""

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view, reduce_events
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime
from tests.integration.test_lifecycle_crash_boundaries import restart, run_worker
from tests.integration.test_observe_application_recovery import assert_cold_recovery_is_noop
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles



_BASELINES = {}


async def stored_content(store, memory):
    rows = await store.read_range("crash-session")
    task, = reduce_events([r.event for r in rows], "").tasks.values()
    root = reduce_events([r.event for r in rows], "").sessions["crash-session"].root_agent_id
    ctx = ProviderContext(session_id="crash-session", tenant_id="default")
    raw = await load_records(memory, MemoryAddress(session_id="crash-session", agent_id=root, task_id=task.id),
        MemoryScope.TASK, ctx, kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.TOOL_AUDIT])
    finish = await load_records(memory, MemoryAddress(session_id="crash-session", agent_id=root),
        MemoryScope.AGENT, ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    finish = [r for r in finish if r.metadata.get("origin_task_id") == task.id]
    return [[(r.kind, r.role, str(r.content).replace(task.id, "<task>")) for r in records]
            for records in (raw, finish)]


async def baseline(tmp_path_factory, mode):
    if mode not in _BASELINES:
        path = tmp_path_factory.mktemp(mode + "_normal")
        await run_worker(path, mode, "none")
        async with open_sqlite_event_store(path / "events.db") as store:
            async with open_sqlite_memory(path / "memory.db") as memory:
                _BASELINES[mode] = await stored_content(store, memory)
    return _BASELINES[mode]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", ["short", "long"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("scenario,cut", [
    ("cancel", "cancel_closed"), ("cancel", "before_summary_applied"),
    ("cancel", "summary_applied"), ("cancel", "none"),
    ("late_subscribe", "before_summary_applied"), ("late_subscribe", "summary_applied"),
    ("late_subscribe", "none"),
])
async def test_original_producer_ends_or_is_cancelled(tmp_path, tmp_path_factory, size, snapshot, scenario, cut):
    mode = f"handoff_{scenario}_{size}"
    expected = await baseline(tmp_path_factory, mode)
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = await restart(store, memory, mode, snapshot)
            root = (await rt.get_session("crash-session")).data.root_agent_id
            pending = (await rebuild_view(store, "crash-session")).pending_observe
            await asyncio.wait_for(recover_fixture_session(rt, root, session_id="crash-session",
                keep_alive=not pending and scenario == "late_subscribe"), 15)
            tm = rt._task_managers["crash-session"]
            while tm._background_asyncio_tasks:
                await asyncio.wait_for(asyncio.gather(*list(tm._background_asyncio_tasks)), 15)
            view = await rebuild_view(store, "crash-session")
            rows = await store.read_range("crash-session")
            task, = reduce_events([r.event for r in rows], "").tasks.values()
            assert task.status == ("CANCELED" if scenario == "cancel" else "INTERRUPTED")
            assert not view.pending_observe and not view.open_runs
            rows = await store.read_range("crash-session")
            assert_lifecycles(rows)
            actors = [r.event.run_id for r in rows if r.event.type == EventType.RUN_STARTED
                      and r.event.payload.get("initial_step") == "prepare"]
            terminal = next(r.position for r in rows if r.event.type == (
                EventType.TASK_CANCELED if scenario == "cancel" else EventType.TASK_INTERRUPTED)
                and r.event.payload.get("reason") != "crash_recovery")
            assert actors
            assert all(any(r.event.type == EventType.RUN_FINISHED and r.event.run_id == run_id
                           and r.position < terminal for r in rows) for run_id in actors)
            requests = [r.event.payload for r in rows if r.event.type == EventType.OBSERVE_REQUESTED]
            assert len(requests) == 1
            assert requests[0]["verdict"]["task_outcome"] == "success"
            assert requests[0]["disposition"]["status"] == "FINISHED"
            assert "outputs" not in requests[0]["disposition"]["payload"]
            assert not any("parent_observe_id" in r.event.payload for r in rows)
            ctx = ProviderContext(session_id="crash-session", tenant_id="default")
            records = await load_records(memory, MemoryAddress(session_id="crash-session", agent_id=root),
                MemoryScope.AGENT, ctx, kinds=[MemoryKind.CONVERSATION_TURN])
            own = [r for r in records if r.metadata.get("origin_task_id") == task.id]
            if scenario == "cancel":
                assert any(r.metadata.get("finish_policy", {}).get("outcome") == "cancelled" for r in own)
                assert not any("reviewed" in str(r.content) for r in own)
            else:
                assert not own
            raw = await load_records(memory, MemoryAddress(session_id="crash-session", agent_id=root, task_id=task.id),
                MemoryScope.TASK, ctx, kinds=[MemoryKind.CONVERSATION_TURN])
            assert any(r.role == "assistant" for r in raw)
            assert await stored_content(store, memory) == expected
            if scenario == "cancel":
                await assert_cold_recovery_is_noop(store, memory, mode, root)
            else:
                # Explicitly resuming INTERRUPTED tasks starts new actor work;
                # rebuilding their completed observation must itself be a no-op.
                head = await store.committed_head("crash-session")
                again, _ = make_lifecycle_runtime(store, memory, mode)
                await initialize_all(again)
                await recover_fixture_session(again, root, session_id="crash-session", keep_alive=True)
                assert await store.committed_head("crash-session") == head
