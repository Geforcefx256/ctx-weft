
from tests._recovery import recover_fixture_session
"""Whole-session cancellation must settle either side of a real goal switch."""

import asyncio
from contextlib import AsyncExitStack

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_max_turns_observe_stop import _drained, _runtime
from tests.integration.test_observe_waiting_continue import quiescent
from tests.integration.test_switch_task_cancel import _UserGoalSwitchLLM


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase", ["before_decision", "during_commit", "after_commit", "after_enqueue"])
async def test_cancel_session_during_goal_switch_settles_before_hot_and_cold_recovery(
    tmp_path, monkeypatch, backend, phase,
):
    async with AsyncExitStack() as stack:
        memory = (await stack.enter_async_context(open_sqlite_memory(tmp_path / "memory.db"))
                  if backend == "sqlite" else InMemoryMemoryProvider())
        store = (await stack.enter_async_context(open_sqlite_event_store(tmp_path / "events.db"))
                 if backend == "sqlite" else None)
        llm = _UserGoalSwitchLLM()
        rt = _runtime(llm, memory, store)
        await initialize_all(rt)
        handle = await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="Work on the old goal and ask me for missing details"))
        tm = rt._task_managers[handle.session_id]
        await quiescent(rt, handle)
        assert tm.get_task(handle.task_id).status == "AWAITING_HUMAN"
        assert len(rt.hitl_registry.list_pending(handle.session_id)) == 1
        entered, release, cancel_started, canceled = [asyncio.Event() for _ in range(4)]
        runs, acks = set(), []
        append = rt.event_store.append_batch

        async def cancel():
            cancel_started.set()
            acks.append(await rt.cancel_session(handle.session_id))
            canceled.set()

        async def on_event(event):
            if event.type == EventType.OBSERVE_REQUESTED and event.payload.get("boundary") == "task_switch":
                runs.add(event.run_id)
            if ((phase == "after_commit" and event.type == EventType.TASK_REVIEWING
                 and event.task_id == handle.task_id)
                    or (phase == "after_enqueue" and event.type == EventType.TASK_CREATED
                        and event.payload.get("task", {}).get("replaces_task_id") == handle.task_id)):
                # Required consumers may cancel as soon as a committed handoff
                # becomes visible. This must not join the emitting actor itself.
                await cancel()

        async def held_append(sid, bid, events, *args, **kwargs):
            before = phase == "before_decision" and any(
                e.type == EventType.RUN_FINISHED and e.run_id in runs for e in events)
            during = phase == "during_commit" and any(e.type == EventType.TASK_REVIEWING for e in events)
            if before or during:
                entered.set()
                await release.wait()
            return await append(sid, bid, events, *args, **kwargs)

        monkeypatch.setattr(rt.event_store, "append_batch", held_append)
        subscription = rt.event_bus.subscribe(None, on_event, provisional=True, required=True)
        stopping = None
        owners = [tm]
        try:
            routed = await rt.send_message(handle.agent_id,
                "Forget the old goal. Switch to the new goal instead.", session_id=handle.session_id)
            assert routed.task_id == handle.task_id
            if phase in {"before_decision", "during_commit"}:
                await asyncio.wait_for(entered.wait(), 5)
                stopping = asyncio.create_task(cancel())
                await asyncio.wait_for(cancel_started.wait(), 1)
                if phase == "before_decision":
                    await asyncio.wait_for(asyncio.shield(stopping), 3)
                    # The final generic queue sweep must preserve the durable
                    # agent-cancellation scope for the still-exiting actor.
                    assert tm.get_task(handle.task_id).error == "cancel_agent:session_canceled"
                else:
                    assert not stopping.done(), "the switch commit owns task arbitration"
                release.set()
            await asyncio.wait_for(canceled.wait(), 5)
            async with asyncio.timeout(5):
                expected_count = 1 if phase == "before_decision" else 2
                while (len(tm.all_tasks()) != expected_count
                       or any(t.status != "CANCELED" for t in tm.all_tasks())
                       or tm._dispatch_tasks or tm._background_asyncio_tasks):
                    await asyncio.sleep(0.01)
            await handle.wait_for_finish(timeout=2)
            assert acks == [True] and llm.actors == 2
            assert not tm.running_task_ids() and not tm._queue.peek_all()
            assert not tm._staged and not tm._pending_switches
            assert not rt.hitl_registry.list_pending(handle.session_id)
            view = await rebuild_view(rt.event_store, handle.session_id)
            assert not view.pending_observe and not view.open_runs
            expected = {tid: "CANCELED" for tid in view.tasks}
            assert len(expected) == expected_count and all(t.status == "CANCELED" for t in view.tasks.values())
            rows = await rt.event_store.read_range(handle.session_id)
            terminated, = [r for r in rows if r.event.type == EventType.AGENT_TERMINATED]
            assert terminated.event.payload["reason"] == "cancel_agent:session_canceled"
            if phase == "before_decision":
                assert not any(r.event.type == EventType.TASK_REVIEWING for r in rows)
                finished, = [r for r in rows if r.event.type == EventType.RUN_FINISHED and r.event.run_id in runs]
                canceled_row, = [r for r in rows if r.event.type == EventType.TASK_CANCELED]
                finalized, = [r for r in rows if r.event.type == EventType.TASK_FINALIZED
                              and r.position > canceled_row.position]
                applied, = [r for r in rows if r.event.type == EventType.OBSERVE_APPLIED
                            and r.position > canceled_row.position]
                assert terminated.position < finished.position < canceled_row.position < finalized.position < applied.position
            await subscription.unsubscribe()

            # Recovery alone must preserve the completed cancellation. Do not
            # hide a stranded successor by canceling the whole session again.
            await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
            await _drained(tm, timeout=5)
            hot = await rebuild_view(rt.event_store, handle.session_id)
            assert {tid: task.status for tid, task in hot.tasks.items()} == expected
            assert not hot.pending_observe and not hot.open_runs
            await rt.cancel_session(handle.session_id)
            if backend == "sqlite":
                rt = _runtime(llm, memory, rt.event_store)
                await initialize_all(rt)
                await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
                tm = rt._task_managers[handle.session_id]
                owners.append(tm)
                await _drained(tm, timeout=5)
                await rt.cancel_session(handle.session_id)
                await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
                await _drained(tm, timeout=5)
            final = await rebuild_view(rt.event_store, handle.session_id)
            assert {tid: task.status for tid, task in final.tasks.items()} == expected
            assert not final.pending_observe and not final.open_runs
            assert not tm.running_task_ids() and not tm._queue.peek_all()
            assert llm.actors == 2
            rows = await rt.event_store.read_range(handle.session_id)
            assert sum(r.event.type == EventType.AGENT_TERMINATED for r in rows) == 1
            assert sum(r.event.type == EventType.TASK_CANCELED for r in rows) == expected_count
            assert not any(r.position > terminated.position and r.event.type in {
                EventType.TASK_REQUEUED, EventType.TASK_STARTED} for r in rows)
        finally:
            release.set()
            await subscription.unsubscribe()
            jobs = [job for owner in owners
                    for job in owner._background_asyncio_tasks | set(owner._dispatch_tasks.values())]
            if stopping is not None:
                jobs.append(stopping)
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
