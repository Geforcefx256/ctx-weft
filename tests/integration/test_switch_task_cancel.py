
from tests._recovery import recover_fixture_session
"""Public goal replacement and cancellation must settle on either side of commit."""

import asyncio
from contextlib import AsyncExitStack

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_max_turns_observe_stop import _drained, _runtime
from tests.integration.test_observe_waiting_continue import quiescent


class _UserGoalSwitchLLM(MockLLMAdapter):
    def __init__(self):
        super().__init__([], context_limit=100_000)
        self.actors = 0

    def complete(self, request, stream=True):
        names = {tool.name for tool in request.tools}
        if "control__update_task_metadata" in names:
            response = MockResponse(tool_calls=[ToolCall(
                id="metadata", name="control__update_task_metadata",
                arguments={"title": "Work", "description": "Complete the requested goal"})])
        elif "control__report_observation" in names:
            schema = next(t.input_schema for t in request.tools if t.name == "control__report_observation")
            terminal_only = schema["properties"]["task_status"].get("enum") == ["success", "fail"]
            response = MockResponse(tool_calls=[ToolCall(
                id="review", name="control__report_observation", arguments={
                    "task_status": "fail" if terminal_only else "waiting" if self.actors == 1 else "success",
                    "act_recap": "Reviewed the goal", "task_summary": "Reviewed requested work",
                    "task_failure_reason": "The user replaced the unfinished old goal" if terminal_only else ""})])
        else:
            self.actors += 1
            if self.actors == 2:
                assert "control__switch_task" in names
                assert "Forget the old goal" in "\n".join(str(m.content) for m in request.messages)
                response = MockResponse(tool_calls=[ToolCall(
                    id="switch", name="control__switch_task", arguments={
                        "title": "New goal", "description": "Complete the new requested goal",
                        "task_prompt": "Answer the new question", "reason": "The user changed the goal"})])
            else:
                response = MockResponse(text=("I need more details about the old goal" if self.actors == 1
                                              else "Finished the new goal"))
        return self._stream(response, request)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase", ["normal", "before_decision", "during_commit", "after_commit"])
async def test_user_changes_goal_then_cancels_around_switch_commit(tmp_path, monkeypatch, backend, phase):
    async with AsyncExitStack() as stack:
        memory = (await stack.enter_async_context(open_sqlite_memory(tmp_path / "memory.db"))
                  if backend == "sqlite" else InMemoryMemoryProvider())
        store = (await stack.enter_async_context(open_sqlite_event_store(tmp_path / "events.db"))
                 if backend == "sqlite" else None)
        llm = _UserGoalSwitchLLM()
        rt = _runtime(llm, memory, store)
        await initialize_all(rt)
        entered, release, cancel_started = asyncio.Event(), asyncio.Event(), asyncio.Event()
        runs, acks = set(), []
        handle = await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="Work on the old goal and ask me for missing details"))
        tm = rt._task_managers[handle.session_id]
        await quiescent(rt, handle)
        assert tm.get_task(handle.task_id).status == "AWAITING_HUMAN"
        assert len(rt.hitl_registry.list_pending(handle.session_id)) == 1
        append = rt.event_store.append_batch

        async def cancel():
            cancel_started.set()
            acks.append(await rt.cancel_agent(handle.agent_id))

        async def on_event(event):
            if event.type == EventType.OBSERVE_REQUESTED and event.payload.get("boundary") == "task_switch":
                runs.add(event.run_id)
            if (phase == "after_commit" and event.type == EventType.TASK_REVIEWING
                    and event.task_id == handle.task_id):
                # A normal event consumer may cancel immediately after learning
                # that the switch committed. Notification must not own TM's lock.
                await cancel()

        async def held_append(sid, bid, events, *args, **kwargs):
            before_decision = phase == "before_decision" and any(
                e.type == EventType.RUN_FINISHED and e.run_id in runs for e in events)
            during_commit = phase == "during_commit" and any(e.type == EventType.TASK_REVIEWING for e in events)
            if before_decision or during_commit:
                entered.set()
                await release.wait()
            return await append(sid, bid, events, *args, **kwargs)

        monkeypatch.setattr(rt.event_store, "append_batch", held_append)
        rt.event_bus.subscribe(None, on_event, provisional=True, required=True)
        cancellation = None
        try:
            resumed = await rt.send_message(handle.agent_id,
                "Forget the old goal. Switch to the new goal instead.", session_id=handle.session_id)
            assert resumed.task_id == handle.task_id
            if phase in {"before_decision", "during_commit"}:
                await asyncio.wait_for(entered.wait(), 5)
                cancellation = asyncio.create_task(cancel())
                await asyncio.wait_for(cancel_started.wait(), 1)
                if phase == "before_decision":
                    await asyncio.wait_for(asyncio.shield(cancellation), 2)
                else:
                    assert not cancellation.done(), "switch commit owns the arbitration boundary"
                release.set()
                await asyncio.wait_for(asyncio.shield(cancellation), 5)
            async with asyncio.timeout(5):
                expected_count = 1 if phase == "before_decision" else 2
                while (len(tm.all_tasks()) != expected_count or any(
                        task.status not in {"FINISHED", "FAILED", "CANCELED"} for task in tm.all_tasks())):
                    await asyncio.sleep(0.01)
            await _drained(tm, timeout=5)
            await handle.wait_for_finish(timeout=2)
            tasks = tm.all_tasks()
            old = tm.get_task(handle.task_id)
            if phase == "before_decision":
                assert tasks == [old] and old.status == "CANCELED" and llm.actors == 2
            else:
                successor, = [task for task in tasks if task.id != old.id]
                assert old.status == "FAILED"
                assert successor.status == ("FINISHED" if phase == "normal" else "CANCELED")
                assert llm.actors == (3 if phase == "normal" else 2)
            assert acks == ([] if phase == "normal" else [[handle.agent_id]])
            assert not tm.running_task_ids() and not tm._queue.peek_all()
            assert not tm._staged and not tm._pending_switches
            assert not rt.hitl_registry.list_pending(handle.session_id)
            view = await rebuild_view(rt.event_store, handle.session_id)
            assert not view.pending_observe and not view.open_runs
            assert {tid: task.status for tid, task in view.tasks.items()} == {task.id: task.status for task in tasks}
            rows = await rt.event_store.read_range(handle.session_id)
            if phase == "before_decision":
                assert not any(r.event.type == EventType.TASK_REVIEWING for r in rows)
                canceled, = [r for r in rows if r.event.type == EventType.TASK_CANCELED]
                finalized, = [r for r in rows if r.event.type == EventType.TASK_FINALIZED
                              and r.position > canceled.position]
                switch_intent, = [r for r in rows if r.event.type == EventType.OBSERVE_REQUESTED
                                  and r.event.payload.get("boundary") == "task_switch"]
                applied, = [r for r in rows if r.event.type == EventType.OBSERVE_APPLIED
                            and r.event.payload.get("observe_id") == switch_intent.event.payload["observe_id"]]
                assert canceled.position < finalized.position < applied.position
            if phase != "normal":
                await rt.cancel_agent(handle.agent_id)
                await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
                await _drained(tm, timeout=5)
                if backend == "sqlite":
                    rt = _runtime(llm, memory, rt.event_store)
                    await initialize_all(rt)
                    await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
                    await _drained(rt._task_managers[handle.session_id], timeout=5)
                assert llm.actors == 2
                final = await rebuild_view(rt.event_store, handle.session_id)
                assert not final.pending_observe and not final.open_runs
                assert all(task.status in {"FINISHED", "FAILED", "CANCELED"} for task in final.tasks.values())
        finally:
            release.set()
            jobs = list(tm._background_asyncio_tasks | set(tm._dispatch_tasks.values()))
            if cancellation is not None:
                jobs.append(cancellation)
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
