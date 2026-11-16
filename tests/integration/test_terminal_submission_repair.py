
from tests._recovery import recover_fixture_session
"""Real runtime/SQLite regressions for submission, pruning and settlement repair."""

import asyncio
import dataclasses

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration._cancel_tree_worker import TreeLLM
from tests.integration._queued_session_cancel_worker import SID, drain, runtime
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_queued_session_cancel_recovery import content



async def setup_queue(store, memory, *, snapshots=False):
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.loop_config = dataclasses.replace(template.loop_config, short_segment_token_threshold=0)
    provider.register(template)
    llm = TreeLLM()
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store,
                      snapshot_every_n=1 if snapshots else 0,
                      config=RuntimeConfig(task_max_concurrent=1))
    rt.providers.register_memory(memory)
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(session_id=SID,
        template_id="agent:tpl_echo", user_prompt="TREE ROOT",  unattended=True))
    await asyncio.wait_for(llm.grandchild_started.wait(), 15)
    tm = rt._task_managers[SID]
    await wait_until(lambda: not tm.open_round_task_ids)
    first = await rt.dispatch_task(SID, "QUEUED ONE", agent_id=handle.agent_id, unattended=True,
        settings=NormalTaskSettings(use_subagent=True, subagent_template="agent:tpl_echo"))
    second = await rt.dispatch_task(SID, "QUEUED TWO", agent_id=first.agent_id, unattended=True)
    return rt, llm, first, second


async def full_view(store):
    return reduce_events([r.event for r in await store.read_range(SID)], SID)


@pytest.mark.asyncio
async def test_automatic_snapshot_does_not_duplicate_pruned_cancellation(tmp_path):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, llm, first, _ = await setup_queue(store, memory, snapshots=True)
            await rt.cancel_session(SID)
            llm.release_reports.set()
            await drain(rt)
            assert await store.read_last_of_type(SID, "StateSnapshot") is not None
            assert first.task_id not in (await rebuild_view(store, SID)).tasks
            head = await store.committed_head(SID)
            before = await content(memory, (await full_view(store)).tasks.values())
            for _ in range(3):
                await rt.cancel_session(SID)
                await drain(rt)
            assert await store.committed_head(SID) == head
            assert await content(memory, (await full_view(store)).tasks.values()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("receipt_lost", [False, True])
async def test_generic_cancel_finalization_recovers_across_old_snapshot(tmp_path, legacy, receipt_lost):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, llm, first, second = await setup_queue(store, memory)
            llm.release_reports.set()
            await rt.cancel_agent(first.agent_id)
            assert rt._task_managers[SID].get_task(first.task_id).status == "ACTIVE"
            assert rt._task_managers[SID].get_task(second.task_id).status == "CANCELED"
            original = store.append_batch
            failed = False

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal failed
                if not failed and any(e.type == "TaskFinalized" and e.task_id == first.task_id for e in events):
                    failed = True
                    if receipt_lost:
                        await original(sid, bid, events, *args, **kwargs)
                    raise OSError("generic cancellation finalization receipt")
                return await original(sid, bid, events, *args, **kwargs)

            store.append_batch = append
            with pytest.raises(PersistenceUnavailableError):
                await rt.cancel_session(SID)
            store.append_batch = original
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            view = prune_view_for_snapshot(await full_view(store))
            if legacy:
                view.tasks.pop(first.task_id, None)  # Actual version-10 pruning loss.
            await seed_snapshot(store, SID, view, cut=await store.committed_head(SID),
                                overrides={"projection_version": 10 if legacy else 11})
            cold, _ = runtime(store, memory, recovering=True)
            cut = await store.committed_head(SID) - 1
            original_read = store.read_range
            async def bounded_read(*args, **kwargs):
                if kwargs.get("after_position", 0) < cut:
                    types = kwargs.get("include_types")
                    assert types and not {"RunStarted", "RunFinished", "RunInterrupted"}.intersection(types), kwargs
                return await original_read(*args, **kwargs)
            store.read_range = bounded_read
            try:
                await initialize_all(cold)
            finally:
                store.read_range = original_read
            await cold.rebuild_session(SID, strict=True)
            # Crash recovery itself can lose another finalization receipt.
            failed = False
            store.append_batch = append
            if not receipt_lost:
                with pytest.raises(PersistenceUnavailableError):
                    await recover_fixture_session(cold, first.agent_id, session_id=SID)
                cold.clear_storage_isolation(SID)
            store.append_batch = original
            cold2, _ = runtime(store, memory, recovering=True)
            await initialize_all(cold2)
            await cold2.rebuild_session(SID, strict=True)
            await recover_fixture_session(cold2, first.agent_id, session_id=SID)
            await drain(cold2)
            rows = await store.read_range(SID)
            for task in (await full_view(store)).tasks.values():
                assert task.status == "CANCELED" and task.finalized
                canceled, = [r for r in rows if r.event.type == "TaskCanceled" and r.event.task_id == task.id]
                finalized, = [r for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == task.id]
                assert canceled.position < finalized.position
            head = await store.committed_head(SID)
            await cold2.cancel_session(SID)
            await drain(cold2)
            assert await store.committed_head(SID) == head


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_lost", [False, True])
async def test_pause_commits_selected_queue_atomically_before_removal(tmp_path, receipt_lost):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, llm, first, second = await setup_queue(store, memory)
            llm.release_reports.set()
            original = store.append_batch
            attempts, batches = 0, []

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal attempts
                if any(e.type == "TaskCanceled" and e.task_id == first.task_id for e in events):
                    attempts += 1
                    batches.append((bid, [(e.id, e.payload) for e in events]))
                    assert {first.task_id, second.task_id} <= {e.task_id for e in events}
                    if attempts <= 3:
                        if receipt_lost:
                            await original(sid, bid, events, *args, **kwargs)
                        raise OSError("pause cancellation receipt")
                return await original(sid, bid, events, *args, **kwargs)

            store.append_batch = append
            for _ in range(3):
                with pytest.raises(PersistenceUnavailableError):
                    await rt.pause_session(SID)
                tm = rt._task_managers[SID]
                assert {first.task_id, second.task_id} <= {e.task_id for e in tm._queue.peek_all()}
                assert all(tm.get_task(tid).status == "ACTIVE" for tid in (first.task_id, second.task_id))
                rt.clear_storage_isolation(SID)
            await rt.pause_session(SID)
            assert all(batch == batches[0] for batch in batches)
            store.append_batch = original
            history = await full_view(store)
            assert all(history.tasks[tid].status == "CANCELED" for tid in (first.task_id, second.task_id))
            await rt.cancel_session(SID)
            await drain(rt)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FINISHED", "FAILED"])
@pytest.mark.parametrize("receipt_lost", [False, True])
@pytest.mark.parametrize("entry", ["recover", "cancel"])
async def test_terminal_receipt_repair_releases_slot_and_wakes_report(tmp_path, monkeypatch, status, receipt_lost, entry):
    from ctx_weft.core.loop.steps.observe import ObserveStep
    from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime
    if status == "FAILED":
        original_verdict = ObserveStep._mechanical_verdict
        monkeypatch.setattr(ObserveStep, "_mechanical_verdict", lambda self, *args:
            dataclasses.replace(original_verdict(self, *args), task_outcome="fail"))
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt, _ = make_lifecycle_runtime(store, memory, "mechanical")
            await initialize_all(rt)
            original = store.append_batch
            event_type = "TaskFinished" if status == "FINISHED" else "TaskFailed"
            attempts = 0
            signatures = []

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal attempts
                if any(e.type == event_type for e in events):
                    # Inject at the durable disposition boundary. A report can
                    # subscribe later; recovery must still wake/finish it.
                    attempts += 1
                    signatures.append((bid, [(e.id, e.payload) for e in events]))
                    if attempts <= 3:
                        if receipt_lost:
                            await original(sid, bid, events, *args, **kwargs)
                        raise OSError("terminal receipt lost")
                return await original(sid, bid, events, *args, **kwargs)

            store.append_batch = append
            handle = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="finish",  unattended=True))
            tm = rt._task_managers[SID]
            await wait_until(lambda: rt.storage_health(SID) is not None and not tm._dispatch_tasks)
            assert handle.task_id in tm.running_task_ids()
            for _ in range(3):
                rt.clear_storage_isolation(SID)
                try:
                    if entry == "cancel":
                        await rt.cancel_session(SID)
                    else:
                        await recover_fixture_session(rt, handle.agent_id, session_id=SID)
                except PersistenceUnavailableError:
                    continue
                break
            store.append_batch = original
            await drain(rt)
            assert all(item == signatures[0] for item in signatures)
            assert not tm.running_task_ids()
            view = await rebuild_view(store, SID)
            assert view.tasks[handle.task_id].status == status
            assert not view.pending_observe and not view.open_runs
            rows = await store.read_range(SID)
            terminal, = [r for r in rows if r.event.type == event_type]
            actor_finish, = [r for r in rows if r.event.type == "RunFinished" and r.event.origin == "runtime"]
            assert actor_finish.position < terminal.position
            before = await content(memory, view.tasks.values())
            head = await store.committed_head(SID)
            cold, _ = make_lifecycle_runtime(store, memory, "mechanical")
            await initialize_all(cold)
            await cold.rebuild_session(SID, strict=True)
            await recover_fixture_session(cold, handle.agent_id, session_id=SID)
            await drain(cold)
            assert await store.committed_head(SID) == head
            assert await content(memory, view.tasks.values()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["before", "committed", "recover_finalized"])
async def test_pause_batch_process_crash_preserves_entire_selection(tmp_path, cut):
    import subprocess
    import sys
    from pathlib import Path
    worker = Path(__file__).with_name("_terminal_submission_worker.py")
    for stage in (["committed", "recover_finalized"] if cut == "recover_finalized" else [cut]):
        result = await asyncio.to_thread(subprocess.run,
            [sys.executable, str(worker), str(tmp_path), stage], cwd=worker.parents[2],
            capture_output=True, text=True, timeout=30)
        assert result.returncode == 73, result.stdout + result.stderr
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            cold, _ = runtime(store, memory, recovering=True)
            await initialize_all(cold)
            await cold.rebuild_session(SID, strict=True)
            history = await full_view(store)
            queued = [t for t in history.tasks.values() if str(t.user_prompt).startswith("QUEUED")]
            owner = queued[0].assigned_agent_id
            await recover_fixture_session(cold, owner, session_id=SID)
            tm = cold._task_managers[SID]
            if cut == "before":
                assert all(tm.get_task(t.id).status not in {"CANCELED", "FINISHED", "FAILED"} for t in queued)
                assert {t.id for t in queued} <= {e.task_id for e in tm._queue.peek_all()}
            else:
                assert all(tm.get_task(t.id).status == "CANCELED" and tm.get_task(t.id).finalized for t in queued)
                assert not {t.id for t in queued} & {e.task_id for e in tm._queue.peek_all()}
                assert not hasattr(tm.session, "status")
            assert all(t.status != "CANCELED" for t in tm.all_tasks() if t.id not in {q.id for q in queued})
            assert not any(r.event.type == "AgentTerminated" for r in await store.read_range(SID))
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            head = await store.committed_head(SID)
            await recover_fixture_session(cold, owner, session_id=SID)
            assert await store.committed_head(SID) == head
