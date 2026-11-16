
"""Crash real task dispatch and cancellation commits, retaining unrelated live work."""

import asyncio
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

import ctx_weft.core.loop.steps.observe as observe
from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT, rebuild_view, reduce_events
from ctx_weft.core.runtime import _SessionTaskRunner
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._cancel_tree_worker import make_tree_runtime


async def worker(path, stage, fault, recovering=False):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            rt, llm = make_tree_runtime(store, memory, recovering=recovering, mechanical=stage == "observe")
            root = grand = None
            target_name = "TREE CHILD" if stage.startswith("cross_") else "TREE GRANDCHILD"
            entered, release, failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
            report_waiting = asyncio.Event()
            llm.finish_grandchild = stage == "observe"
            original_read = store.read_range
            async def read(*args, **kwargs):
                rows = await original_read(*args, **kwargs)
                if (stage == "observe" and not fault.startswith("run_") and not recovering
                        and set(kwargs.get("include_types") or ()) ==
                        {"ObserveRequested", *TASK_STATUS_BY_EVENT} and not report_waiting.is_set()
                        and any(r.event.type == "ObserveRequested" and r.event.task_id == grand for r in rows)):
                    assert not any(r.event.type == "TaskCanceled" and r.event.task_id == grand for r in rows)
                    report_waiting.set()
                return rows
            store.read_range = read
            notified = []
            run_fault = fault.startswith("run_")
            failed_envelopes = []
            original_append = store.append_batch

            async def append(sid, bid, events, *args, **kwargs):
                target = any(e.type == "TaskCanceled" and e.task_id == grand for e in events)
                if stage == "observe" and target and not recovering and not run_fault:
                    await asyncio.wait_for(report_waiting.wait(), 10)
                closing_actor = any(e.type == "RunFinished" and e.task_id == grand
                                    and e.origin == "runtime" for e in events)
                if run_fault and closing_actor:
                    limit = 2 if fault.endswith("_twice") else 1
                    if len(failed_envelopes) < limit:
                        failed_envelopes.append([(e.id, e.run_id, e.sequence, e.payload) for e in events])
                        if fault.startswith("run_unknown"):
                            await original_append(sid, bid, events, *args, **kwargs)
                        failed.set()
                        raise OSError("injected actor RunFinished receipt failure")
                    assert [(e.id, e.run_id, e.sequence, e.payload) for e in events] == failed_envelopes[0]
                inject = target and fault.startswith("commit_") and not failed.is_set()
                if inject and fault.startswith("commit_failed"):
                    failed.set()
                    raise OSError("injected before TaskCanceled commit")
                receipt = await original_append(sid, bid, events, *args, **kwargs)
                if inject:
                    failed.set()
                    raise OSError("injected lost TaskCanceled receipt after commit")
                for event in events:
                    if recovering and fault == "startup" and event.type == "RunFinished":
                        os._exit(73)
                    if event.type == "AgentTerminated":
                        assert event.agent_id == root, "descendant agent was terminated"
                        if fault == "source":
                            os._exit(73)
                    if event.task_id == grand and (
                        fault == "canceled" and event.type == "TaskCanceled"
                        or fault == "finalized" and event.type == "TaskFinalized"
                        or fault == "actor_finished" and event.type == "RunFinished" and event.origin == "runtime"
                    ):
                        os._exit(73)
                return receipt

            store.append_batch = append
            original_ingest = memory.ingest

            async def ingest(event, ctx):
                result = await original_ingest(event, ctx)
                if (fault == "memory" and event.metadata.get("origin_task_id") == grand
                        and event.metadata.get("finish_policy", {}).get("outcome") == "cancelled"):
                    os._exit(73)
                return result

            memory.ingest = ingest

            async def notice(event):
                if event.type == "TaskCanceled":
                    notified.append(event.task_id)

            rt.event_bus.subscribe(None, notice)
            if recovering:
                view = await rebuild_view(store, "tree-session")
                root = view.sessions["tree-session"].root_agent_id
                grand = next((t.id for t in view.tasks.values() if str(t.user_prompt) == target_name), None)
                if grand is None:
                    # Fault targeting needs historical IDs even when the valid
                    # snapshot has correctly pruned an already-finalized task.
                    history = reduce_events([r.event for r in await store.read_range("tree-session")], "")
                    grand = next(t.id for t in history.tasks.values() if str(t.user_prompt) == target_name)
                await initialize_all(rt)
                await recover_fixture_session(rt, root, session_id="tree-session")
                raise AssertionError("recovery did not reach crash boundary")

            original_assemble, original_execute = _SessionTaskRunner.assemble, _SessionTaskRunner.execute

            async def assemble(owner, tid):
                binding = await original_assemble(owner, tid)
                if stage.endswith("assembly") and str(owner._task_manager.get_task(tid).user_prompt) == target_name:
                    entered.set()
                    await release.wait()
                    if fault == "assembly_error":
                        raise RuntimeError("assembly failed while cancellation was pending")
                return binding

            async def execute(owner, binding, tid):
                if stage.endswith("registration") and str(owner._task_manager.get_task(tid).user_prompt) == target_name:
                    entered.set()
                    await release.wait()
                return await original_execute(owner, binding, tid)

            _SessionTaskRunner.assemble, _SessionTaskRunner.execute = assemble, execute
            original_prepare = observe.prepare_foreground_observe
            async def prepare(state, ctx, *args, **kwargs):
                result = await original_prepare(state, ctx, *args, **kwargs)
                if stage == "observe" and str(state.task.user_prompt) == target_name:
                    entered.set()
                    await release.wait()
                return result
            observe.prepare_foreground_observe = prepare
            if stage == "observe":
                from tests.runtime_helpers import install_mechanical_report_fixture
                async def before_handoff(state, ctx):
                    if str(state.task.user_prompt) == "TREE GRANDCHILD":
                        entered.set()
                        await release.wait()
                install_mechanical_report_fixture(
                    lambda state: str(state.task.user_prompt) == "TREE GRANDCHILD",
                    before=before_handoff)

            await initialize_all(rt)
            handle = await rt.start_session(SessionStartParams.create(session_id="tree-session",
                template_id="agent:tpl_echo", user_prompt="TREE ROOT",  unattended=True))
            root = handle.agent_id
            await asyncio.wait_for(llm.grandchild_started.wait() if stage == "active" else entered.wait(), 10)
            tm = rt._task_managers["tree-session"]
            grand_task = next(t for t in tm.all_tasks() if str(t.user_prompt) == target_name)
            grand = grand_task.id
            async with asyncio.timeout(10):
                while tm.open_round_task_ids or tm.get_task(handle.task_id).status != "SUSPENDED":
                    await asyncio.sleep(.01)
            other = await rt.dispatch_task("tree-session", "UNRELATED RUN", agent_id=root,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                port_key="other", unattended=True)
            await asyncio.wait_for(llm.unrelated_started.wait(), 10)
            async with asyncio.timeout(10):
                while tm.open_round_task_ids:
                    await asyncio.sleep(.01)
            tm._max_concurrent = 0
            queued = await rt.dispatch_task("tree-session", "UNRELATED QUEUED", agent_id=root,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                port_key="queued", unattended=True)
            actors_before = llm.actors
            await rt.cancel_agent(root, reason="cancel tree")
            if stage == "observe" and not run_fault:
                llm.release_reports.set()
            release.set()
            if fault.startswith("commit_") or run_fault:
                await asyncio.wait_for(failed.wait(), 10)
                async with asyncio.timeout(10):
                    while tm._dispatch_is_live(grand):
                        await asyncio.sleep(.01)
                assert grand_task.status == "ACTIVE", "failed receipt mutated the in-memory terminal state"
                assert rt.storage_health("tree-session") is not None
                rows = await store.read_range("tree-session")
                canceled = [r for r in rows if r.event.task_id == grand and r.event.type == "TaskCanceled"]
                assert len(canceled) == (0 if run_fault or fault.startswith("commit_failed") else 1)
                if run_fault:
                    finishes = [r for r in rows if r.event.task_id == grand and r.event.type == "RunFinished"
                                and r.event.origin == "runtime"]
                    assert len(finishes) == int(fault.startswith("run_unknown"))
                    assert ("tree-session", grand) in rt._pending_actor_runs
                assert grand not in notified
                assert not any(r.event.task_id == grand and r.event.type == "TaskFinalized" for r in rows)
                if fault.endswith("_crash"):
                    os._exit(73)
                rt.clear_storage_isolation("tree-session")
                if fault.endswith("_twice"):
                    try:
                        await rt.cancel_agent(root, reason="cancel tree")
                    except Exception as exc:
                        from ctx_weft.protocols.events import PersistenceUnavailableError
                        assert isinstance(exc, PersistenceUnavailableError)
                    else:
                        raise AssertionError("second RunFinished failure did not block cancellation")
                    assert grand_task.status == "ACTIVE" and not grand_task.finalized
                    assert failed_envelopes[0] == failed_envelopes[1]
                    rows = await store.read_range("tree-session")
                    assert not any(r.event.task_id == grand and r.event.type in
                                   ("TaskCanceled", "TaskFinalized") for r in rows)
                    rt.clear_storage_isolation("tree-session")
                await rt.cancel_agent(root, reason="cancel tree")
                if run_fault:
                    assert ("tree-session", grand) not in rt._pending_actor_runs
            llm.release_reports.set()
            async with asyncio.timeout(10):
                while (any(t.status != "CANCELED" or not t.finalized for t in tm.all_tasks()
                           if str(t.user_prompt).startswith("TREE")) or tm._background_asyncio_tasks
                       or tm._dispatch_is_live(grand) or grand in tm.running_task_ids()):
                    await asyncio.sleep(.01)
            if stage not in {"active", "observe"}:
                assert not llm.grandchild_started.is_set() and llm.actors == actors_before
            assert not tm._dispatch_is_live(grand) and grand not in tm.running_task_ids()
            before = llm.unrelated_tokens
            await asyncio.sleep(.03)
            assert llm.unrelated_tokens > before
            assert tm.get_task(other.task_id).status == "ACTIVE"
            assert rt._agent_lifecycle_manager.status_of(other.agent_id) == "running"
            assert queued.task_id not in tm.running_task_ids()
            assert rt._agent_lifecycle_manager.status_of(queued.agent_id) == "idle"
            assert rt._agent_lifecycle_manager.status_of(grand_task.assigned_agent_id or grand_task.creator_agent_id) == "idle"
            rows = await store.read_range("tree-session")
            assert [r.event.agent_id for r in rows if r.event.type == "AgentTerminated"] == [root]
            view = await rebuild_view(store, "tree-session")
            assert view.tasks[grand].status == "CANCELED" and view.tasks[grand].finalized
            assert sum(r.event.type == "TaskCanceled" and r.event.task_id == grand for r in rows) == 1
            assert sum(r.event.type == "TaskFinalized" and r.event.task_id == grand for r in rows) == 1
            os._exit(0 if fault in {"none", "commit_failed", "commit_unknown", "assembly_error"}
                     or run_fault else 74)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3], "recover" in sys.argv[4:]))
