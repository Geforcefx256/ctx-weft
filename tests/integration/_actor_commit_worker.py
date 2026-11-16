"""Exercise actor start uncertainty and cancellation settlement through public APIs."""

import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests._initialization import initialize_all

import ctx_weft.core.loop.steps.observe as observe
from ctx_weft import SessionStartParams
from ctx_weft.core.runtime import _SessionTaskRunner
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._cancel_tree_worker import make_tree_runtime


async def worker(path, stage, fault, mode):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            rt, llm = make_tree_runtime(store, memory, mechanical=stage == "observe")
            entered, release, failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
            llm.finish_grandchild = stage == "observe"
            original_execute = _SessionTaskRunner.execute
            original_prepare = observe.prepare_foreground_observe

            async def execute(owner, binding, tid):
                if stage == "start" and str(owner._task_manager.get_task(tid).user_prompt) == "TREE GRANDCHILD":
                    entered.set()
                    await release.wait()
                return await original_execute(owner, binding, tid)

            async def prepare(state, ctx, *args, **kwargs):
                result = await original_prepare(state, ctx, *args, **kwargs)
                if stage == "observe" and str(state.task.user_prompt) == "TREE GRANDCHILD":
                    entered.set()
                    await release.wait()
                return result

            _SessionTaskRunner.execute = execute
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
            await asyncio.wait_for(llm.grandchild_started.wait() if stage == "active" else entered.wait(), 10)
            tm = rt._task_managers["tree-session"]
            grand = next(t for t in tm.all_tasks() if str(t.user_prompt) == "TREE GRANDCHILD")
            async with asyncio.timeout(10):
                while tm.open_round_task_ids:
                    await asyncio.sleep(.01)
            other = await rt.dispatch_task("tree-session", "UNRELATED RUN", agent_id=handle.agent_id,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"}, unattended=True)
            await asyncio.wait_for(llm.unrelated_started.wait(), 10)
            async with asyncio.timeout(10):
                while tm.open_round_task_ids:
                    await asyncio.sleep(.01)
            original_append = store.append_batch
            direct_finalization = fault == "final_only_unknown"
            starts, finishes, finalizations = [], [], []
            counts = {"start": 0, "finish": 0, "finalized": 0}
            terminals = []
            original_terminal = tm._hooks.on_task_terminal

            async def terminal(tid):
                terminals.append(tid)
                if original_terminal:
                    await original_terminal(tid)
            tm.set_hooks(replace(tm._hooks, on_task_terminal=terminal))

            async def append(sid, bid, events, *args, **kwargs):
                target = next((e for e in events if e.task_id == grand.id and e.type in
                               ("RunStarted", "RunFinished", "TaskFinalized") and
                               (e.type == "TaskFinalized" or e.origin == "runtime")), None)
                boundary = None
                if target is not None:
                    boundary = {"RunStarted": "start", "RunFinished": "finish", "TaskFinalized": "finalized"}[target.type]
                    {"start": starts, "finish": finishes, "finalized": finalizations}[boundary].append(target)
                inject = fault != "none" and (
                    stage == "start" and boundary == "start" and counts["start"] < 2 or
                    stage != "start" and not direct_finalization and boundary == "finish" and counts["finish"] == 0 or
                    stage != "start" and boundary == "finalized" and counts["finalized"] < (2 if fault == "before" else 1))
                if inject:
                    counts[boundary] += 1
                    # The first actor close fails before commit; the next boundary
                    # independently loses a receipt or fails before its commit.
                    unknown = fault in ("unknown", "final_only_unknown") and boundary != "finish"
                    if unknown:
                        await original_append(sid, bid, events, *args, **kwargs)
                    failed.set()
                    raise OSError("injected " + boundary + (" lost receipt" if unknown else " precommit failure"))
                return await original_append(sid, bid, events, *args, **kwargs)

            store.append_batch = append
            if stage == "start":
                release.set()
                await asyncio.wait_for(failed.wait(), 10)
            else:
                await rt.cancel_agent(handle.agent_id, reason="cancel tree")
                release.set()
                await asyncio.wait_for(failed.wait(), 10)
            async with asyncio.timeout(10):
                while tm._dispatch_is_live(grand.id):
                    await asyncio.sleep(.01)
            assert grand.status == ("CANCELED" if direct_finalization else "ACTIVE")
            assert grand.id in tm.running_task_ids()
            assert (("tree-session", grand.id) in rt._pending_actor_runs) != direct_finalization
            actors = llm.actors
            failures = 0
            for _ in range(5):
                rt.clear_storage_isolation("tree-session")
                try:
                    await rt.cancel_agent(handle.agent_id, reason="cancel tree")
                except PersistenceUnavailableError:
                    failures += 1
                    assert grand.id in tm.running_task_ids()
                    if mode != "hot" and (stage != "start" or counts["start"] == 2):
                        os._exit(73)
                else:
                    if not rt.storage_health("tree-session"):
                        break
            assert failures >= 1 or direct_finalization
            llm.release_reports.set()
            async with asyncio.timeout(15):
                while (any(t.status != "CANCELED" or not t.finalized or t.id in tm.running_task_ids()
                           for t in tm.all_tasks() if str(t.user_prompt).startswith("TREE"))
                       or tm._background_asyncio_tasks):
                    await asyncio.sleep(.01)
            assert llm.actors == actors, "confirmation restarted the actor"
            assert ("tree-session", grand.id) not in rt._pending_actor_runs
            assert len({e.id for e in starts}) == int(stage == "start")
            assert len({e.id for e in finishes}) == 1
            if stage == "start" and fault != "none":
                assert counts["start"] == 2 and not llm.grandchild_started.is_set()
            assert terminals.count(grand.id) == 1
            rows = await store.read_range("tree-session")
            head = await store.committed_head("tree-session")
            await asyncio.gather(*(rt.cancel_agent(handle.agent_id, reason="cancel tree") for _ in range(3)))
            assert await store.committed_head("tree-session") == head
            assert terminals.count(grand.id) == 1
            # A new standalone task on the same surviving agent must really execute.
            assert await rt.recover_storage("tree-session")
            next_task = await rt.dispatch_task("tree-session", "UNRELATED RUN FOLLOWUP",
                agent_id=grand.assigned_agent_id, settings={"use_subagent": False}, unattended=True)
            async with asyncio.timeout(10):
                while llm.actors == actors:
                    await asyncio.sleep(.01)
            assert next_task.task_id in tm.running_task_ids()
            assert tm.get_task(other.task_id).status == "ACTIVE"
            assert rt._agent_lifecycle_manager.status_of(other.agent_id) == "running"
            assert sum(r.event.type == "TaskCanceled" and r.event.task_id == grand.id for r in rows) == 1
            assert sum(r.event.type == "TaskFinalized" and r.event.task_id == grand.id for r in rows) == 1
            os._exit(0)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), *sys.argv[2:5]))
