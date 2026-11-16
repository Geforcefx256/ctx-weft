"""Follow-up task metadata survives SQLite reopen and snapshot recovery."""

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view, reduce_events
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._event_helpers import all_events
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration._session_cancel_worker import runtime, wait_until
from tests.integration.test_pause_on_first_message_e2e import _NamesTasksLLM



@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_phase", ["none", "before_followup", "after_followup"])
async def test_followup_metadata_survives_cold_recovery(tmp_path, snapshot_phase):
    sid = "followup-metadata"
    llm = _NamesTasksLLM(
        responses=[MockResponse(text="First answer"), MockResponse(text="Second answer")],
        observer_verdict="success", context_limit=100_000,
    )
    llm.stalling = False

    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = runtime(store, memory, llm)
            await initialize_all(rt)
            first = await rt.start_session(SessionStartParams.create(
                session_id=sid, template_id="agent:tpl_echo", user_prompt="First request",
                 unattended=True,
            ))
            await first.wait_for_finish(timeout=10)
            tm = rt._task_managers[sid]
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            assert tm.get_task(first.task_id).title == "Task 1"

            async def snapshot():
                head = await store.committed_head(sid)
                view = reduce_events(await all_events(store, sid), sid)
                await seed_snapshot(store, sid, prune_view_for_snapshot(view), cut=head)

            if snapshot_phase == "before_followup":
                await snapshot()

            # Keep the follow-up waiting for the user so cold owner materialization
            # must restore a complete, nonterminal task, including its metadata.
            llm._observer_verdict = "waiting"
            second = await rt.send_message(first.agent_id, "Second request", session_id=sid)
            task = tm.get_task(second.task_id)
            await wait_until(lambda: task.title == "Task 2" and task.status == "AWAITING_HUMAN"
                             and not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            assert llm.sidecar_calls == 2
            hot = (await rt.get_task(sid, second.task_id)).data
            assert hot.title == "Task 2" and hot.description == "Goal 2"
            assert (await rt.get_session(sid)).data.state.goal == "Session goal 2"

            events = await all_events(store, sid)
            created = next(e for e in events
                           if e.type == EventType.TASK_CREATED and e.task_id == second.task_id)
            metadata = [e for e in events
                        if e.type == EventType.RECOGNIZE_INTENT_TOOL_CALL
                        and e.task_id == second.task_id]
            assert created.payload["task"]["title"] == ""
            assert len(metadata) == 1 and metadata[0].payload["title"] == "Task 2"
            assert events.index(created) < events.index(metadata[0])
            if snapshot_phase == "after_followup":
                await snapshot()

    # Reopen both databases and use a fresh runtime, with no objects from the old
    # owner. Read before and after materialization to cover both query paths.
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            cold_llm = _NamesTasksLLM(responses=[], context_limit=100_000)
            rt = runtime(store, memory, cold_llm, recovering=True)
            cold = (await rt.get_task(sid, second.task_id)).data
            assert cold.title == hot.title and cold.description == hot.description
            assert (await rt.get_session(sid)).data.state.goal == "Session goal 2"
            await initialize_all(rt)
            await rt.rebuild_session(sid, strict=True)
            restored = rt._task_managers[sid].get_task(second.task_id)
            assert restored.title == "Task 2" and restored.description == "Goal 2"
            assert restored.user_prompt == "Second request"
            assert restored.status == "AWAITING_HUMAN"
            assert (await rt.get_task(sid, second.task_id)).data.title == hot.title
            assert (await rt.get_session(sid)).data.state.goal == "Session goal 2"
            assert (await rt.get_task(sid, first.task_id)).data.title == "Task 1"
            replayed = reduce_events(await all_events(store, sid), sid)
            rebuilt = await rebuild_view(store, sid)
            assert rebuilt.tasks[second.task_id].title == replayed.tasks[second.task_id].title
            assert cold_llm.sidecar_calls == 0
