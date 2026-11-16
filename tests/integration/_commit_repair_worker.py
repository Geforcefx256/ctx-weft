
"""Real provisional batch and derived AgentIdle failures, including late delivery."""

import asyncio
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.protocols import LLMChunk
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._cancel_tree_worker import make_tree_runtime
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM


async def worker(path, stage, fault, cut):
    other_started = asyncio.Event()
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            if cut.startswith("recover_"):
                from ctx_weft.core.control.reducers import rebuild_view
                view = await rebuild_view(store, "tree-session")
                root = view.sessions["tree-session"].root_agent_id
                rt, _ = make_tree_runtime(store, memory, recovering=True)
                original = store.append_batch
                async def recover_append(sid, bid, events, *args, **kwargs):
                    receipt = await original(sid, bid, events, *args, **kwargs)
                    kind = "RunFinished" if cut == "recover_run" else "TaskFinalized"
                    if any(e.type == kind for e in events):
                        os._exit(73)
                    return receipt
                store.append_batch = recover_append
                await initialize_all(rt)
                await recover_fixture_session(rt, root, session_id="tree-session")
                raise AssertionError("second recovery crash was not reached")

            if stage == "batch":
                provider = InlineAgentTemplateProvider()
                provider.register(make_echo_template())
                llm = _FinishEveryTaskLLM()
                original_complete = llm.complete
                def complete(request, stream=True):
                    if (not {t.name for t in request.tools}.intersection(
                            {"control__update_task_metadata", "control__report_observation"})
                            and any("UNRELATED RUN" in str(getattr(m, "content", "")) for m in request.messages)):
                        async def hold():
                            other_started.set()
                            while True:
                                yield LLMChunk(kind="token", text="working ")
                                await asyncio.sleep(.01)
                        return hold()
                    return original_complete(request, stream)
                llm.complete = complete
                rt = make_runtime(llm=llm, agent_provider=provider, event_store=store)
                rt.providers.register_memory(memory)
            else:
                rt, llm = make_tree_runtime(store, memory)
                original_complete = llm.complete
                def complete(request, stream=True):
                    names = {t.name for t in request.tools}
                    if not names.intersection({"control__update_task_metadata", "control__report_observation"}) and llm.tree_actors == 1:
                        llm.tree_actors = 2  # The delegated child streams instead of delegating again.
                    return original_complete(request, stream)
                llm.complete = complete
            await initialize_all(rt)
            root_handle = await rt.start_session(SessionStartParams.create(session_id="tree-session",
                template_id="agent:tpl_echo", user_prompt="FIRST" if stage == "batch" else "TREE ROOT",
                 unattended=True))
            tm = rt._task_managers["tree-session"]
            if stage == "batch":
                await root_handle.wait_for_finish(timeout=15)
                async with asyncio.timeout(15):
                    while tm.running_task_ids() or tm._background_asyncio_tasks:
                        await asyncio.sleep(.01)
            else:
                await asyncio.wait_for(llm.grandchild_started.wait(), 10)
            other = await rt.dispatch_task("tree-session", "UNRELATED RUN", agent_id=root_handle.agent_id,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"}, unattended=True)
            await asyncio.wait_for(other_started.wait() if stage == "batch" else llm.unrelated_started.wait(), 10)
            async with asyncio.timeout(10):
                while other.task_id not in tm.running_task_ids() or tm.open_round_task_ids:
                    await asyncio.sleep(.01)
            target = None if stage == "batch" else next(t.id for t in tm.all_tasks() if str(t.user_prompt) == "TREE CHILD")
            original_append = store.append_batch
            entered, release = asyncio.Event(), asyncio.Event()
            attempts, frozen_events, notices = [], None, []
            async def notice(event):
                notices.append(event.id)
            rt.event_bus.subscribe(None, notice)

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal frozen_events
                relevant = (stage == "batch" and any(e.type == "TaskCreated" and e.task_id == target for e in events)
                            or stage == "idle" and any(e.type == "AgentIdle" and e.task_id == target for e in events))
                if relevant:
                    signature = (bid, [(e.id, e.type, e.run_id, e.sequence, e.payload) for e in events])
                    if frozen_events is None:
                        from copy import deepcopy
                        frozen_events = deepcopy(signature)
                    assert signature == frozen_events, "uncertain batch or transition changed on retry"
                    attempts.append(signature)
                    if fault == "unknown" or fault == "none":
                        receipt = await original_append(sid, bid, events, *args, **kwargs)
                    if stage == "batch" and len(attempts) == 1:
                        entered.set()
                        await release.wait()
                    if fault != "none" and len(attempts) <= 2:
                        raise OSError("injected " + stage + " " + fault + " receipt failure")
                    if fault == "before":
                        receipt = await original_append(sid, bid, events, *args, **kwargs)
                    return receipt
                receipt = await original_append(sid, bid, events, *args, **kwargs)
                if stage == "batch" and cut != "hot" and any(e.type == "AgentTerminated" for e in events):
                    os._exit(73)
                return receipt

            store.append_batch = append
            if stage == "batch":
                handle = await rt.send_message(root_handle.agent_id, "SECOND", unattended=True)
                target = handle.task_id
                await asyncio.wait_for(entered.wait(), 10)
                cancel = asyncio.create_task(rt.cancel_agent(root_handle.agent_id, reason="cancel repair"))
                async with asyncio.timeout(10):
                    while not any(e.type == "AgentTerminated" for e in rt.event_bus._provisional[target].events):
                        await asyncio.sleep(.01)
                release.set()
                await cancel
            else:
                await rt.cancel_agent(root_handle.agent_id, reason="cancel repair")
            async with asyncio.timeout(10):
                while tm._dispatch_is_live(target):
                    await asyncio.sleep(.01)
            if fault != "none":
                assert attempts
                assert not any(eid in notices for eid, *_ in frozen_events[1])
            for _ in range(6):
                rt.clear_storage_isolation("tree-session")
                try:
                    await rt.cancel_agent(root_handle.agent_id, reason="cancel repair")
                except PersistenceUnavailableError:
                    if stage == "idle" and cut != "hot":
                        os._exit(73)
                else:
                    break
            if stage == "idle":
                llm.release_reports.set()
            async with asyncio.timeout(15):
                while (target in tm.running_task_ids() or tm._background_asyncio_tasks
                       or not tm.get_task(target).finalized):
                    await asyncio.sleep(.01)
            assert tm.get_task(target).status == "CANCELED"
            assert not tm.open_round_task_ids
            rows = await store.read_range("tree-session")
            ids = [r.event.id for r in rows]
            assert len(ids) == len(set(ids))
            assert sum(r.event.type == "TaskCanceled" and r.event.task_id == target for r in rows) == 1
            assert sum(r.event.type == "TaskFinalized" and r.event.task_id == target for r in rows) == 1
            if fault != "none":
                assert len(attempts) == 3
            if stage == "idle":
                agent = tm.get_task(target).assigned_agent_id
                assert rt._agent_lifecycle_manager.status_of(agent) == "idle"
                assert await rt.recover_storage("tree-session")
                followup = await rt.dispatch_task("tree-session", "UNRELATED RUN FOLLOWUP", agent_id=agent,
                    settings={"use_subagent": False}, unattended=True)
                async with asyncio.timeout(10):
                    while followup.task_id not in tm.running_task_ids() or rt._agent_lifecycle_manager.status_of(agent) != "running":
                        await asyncio.sleep(.01)
                old = next(r.event for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == target)
                await rt.event_bus.emit(old)  # Real late delivery of an already-committed terminal fact.
                await rt._confirm_task_settled(tm.get_task(target))
                await rt.cancel_agent(root_handle.agent_id, reason="cancel repair")
                assert rt._agent_lifecycle_manager.status_of(agent) == "running"
                assert rt._agent_lifecycle_manager.record_of(agent).current_task_id == followup.task_id
            assert tm.get_task(other.task_id).status == "ACTIVE"
            assert rt._agent_lifecycle_manager.status_of(other.agent_id) == "running"
            os._exit(0)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), *sys.argv[2:5]))
