
from tests._recovery import recover_fixture_session
"""An accepted session cancel survives a later observation publication window."""

import asyncio
from contextlib import AsyncExitStack

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import LLMChunk, MemoryKind, ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_max_turns_observe_stop import _drained, _runtime


class _SlowMetadataLLM(MockLLMAdapter):
    def __init__(self):
        super().__init__([], context_limit=100_000)
        self.metadata_started, self.review_started, self.allow_review = [asyncio.Event() for _ in range(3)]
        self.actors, self.finish = 0, False

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        if "control__update_task_metadata" in names:
            if self.finish:
                return self._stream(MockResponse(tool_calls=[ToolCall(id="meta", name="control__update_task_metadata",
                    arguments={"title": "Work", "description": "Complete requested work"})]), request)

            async def metadata():
                self.metadata_started.set()
                yield LLMChunk(kind="token", text="Naming the task")
                await asyncio.Event().wait()
            return metadata()
        if "control__report_observation" in names:
            async def review():
                self.review_started.set()
                await self.allow_review.wait()
                response = MockResponse(tool_calls=[ToolCall(id="review", name="control__report_observation",
                    arguments={"task_status": "success" if self.finish else "continue",
                               "act_recap": "Completed work" if self.finish else "Another turn is needed"})])
                async for chunk in self._stream(response, request):
                    yield chunk
            return review()
        self.actors += 1
        return self._stream(MockResponse(text="Completed work" if self.finish else "Made progress on the work"), request)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase,after_commit", [
    ("before_decision", False), ("during_fold", False), ("during_fold", True),
])
async def test_accepted_session_cancel_survives_later_finalized_write_and_recovery(
    tmp_path, monkeypatch, backend, phase, after_commit,
):
    async with AsyncExitStack() as stack:
        memory = (await stack.enter_async_context(open_sqlite_memory(tmp_path / "memory.db"))
                  if backend == "sqlite" else InMemoryMemoryProvider())
        store = (await stack.enter_async_context(open_sqlite_event_store(tmp_path / "events.db"))
                 if backend == "sqlite" else None)
        llm = _SlowMetadataLLM()
        rt = _runtime(llm, memory, store)
        await initialize_all(rt)
        meta_finished, release_meta, meta_notified = [asyncio.Event() for _ in range(3)]
        fold_entered, release_fold, finalized, release_finalized = [asyncio.Event() for _ in range(4)]
        append, original_fold = rt.event_store.append_batch, memory._commit_organize

        async def held_append(sid, bid, events, *args, **kwargs):
            if any(e.type == EventType.RUN_FINISHED and e.origin == "loop.recognize_intent" for e in events):
                meta_finished.set()
                await release_meta.wait()
            matching = any(e.type == EventType.TASK_FINALIZED and e.payload.get("outcome") == "continue"
                           for e in events)
            if matching and after_commit:
                receipt = await append(sid, bid, events, *args, **kwargs)
            if matching:
                finalized.set()
                await release_finalized.wait()
            return (receipt if matching and after_commit else await append(sid, bid, events, *args, **kwargs))

        async def held_fold(ids, replacements, ctx):
            if phase == "during_fold" and any(r.kind == MemoryKind.SUMMARY for r in replacements):
                fold_entered.set()
                await release_fold.wait()
            return await original_fold(ids, replacements, ctx)

        async def notice(event):
            if event.type == EventType.RUN_FINISHED and event.origin == "loop.recognize_intent":
                meta_notified.set()

        monkeypatch.setattr(rt.event_store, "append_batch", held_append)
        monkeypatch.setattr(memory, "_commit_organize", held_fold)
        subscription = rt.event_bus.subscribe(None, notice, required=True, provisional=True)
        handle = await rt.start_session(SessionStartParams.create(template_id="agent:tpl_echo",
            user_prompt="Complete the work across as many turns as needed"))
        tm = rt._task_managers[handle.session_id]
        owners, cancel = [tm], None
        try:
            await asyncio.wait_for(llm.metadata_started.wait(), 5)
            await asyncio.wait_for(llm.review_started.wait(), 5)
            async with asyncio.timeout(5):
                while tm._dispatch_tasks:
                    await asyncio.sleep(0.01)
            assert tm.get_task(handle.task_id).status == "AWAITING_HUMAN"
            if phase == "during_fold":
                llm.allow_review.set()
                await asyncio.wait_for(fold_entered.wait(), 5)
            cancel = asyncio.create_task(rt.cancel_session(handle.session_id))
            await asyncio.wait_for(meta_finished.wait(), 5)
            assert tm._cancelled and not tm.is_writing_finalized()
            if phase == "during_fold":
                release_fold.set()
                await asyncio.wait_for(finalized.wait(), 5)
                release_meta.set()
                await asyncio.wait_for(meta_notified.wait(), 5)
                done, _ = await asyncio.wait({cancel}, timeout=0.05)
                assert not done, "an accepted cancel must wait for publication instead of skipping its cleanup"
                assert not tm._lock.locked(), "publication wait must not own task arbitration"
                # These are new requests, including a reason identical to the
                # admitted request. None may inherit that earlier admission.
                assert await asyncio.wait_for(rt.pause_agent(handle.agent_id), 1) == []
                assert await asyncio.wait_for(rt.pause_session(handle.session_id), 1) is False
                assert await asyncio.wait_for(rt.cancel_agent(handle.agent_id), 1) == []
                assert await asyncio.wait_for(rt.cancel_agent(handle.agent_id, reason="session_canceled"), 1) == []
                assert await asyncio.wait_for(rt.cancel_session(handle.session_id), 1) is False
                assert tm.get_task(handle.task_id).error is None
                release_finalized.set()
            else:
                llm.allow_review.set()
                registry = rt._step_executions[handle.session_id]
                async with asyncio.timeout(5):
                    while registry.pending_run_id(handle.task_id, "observe"):
                        await asyncio.sleep(0.01)
                assert not finalized.is_set(), "a verdict arriving after session cancellation must be rejected"
                release_meta.set()
            assert await asyncio.wait_for(asyncio.shield(cancel), 5) is True
            await _drained(tm, timeout=5)
            await handle.wait_for_finish(timeout=2)
            assert tm.get_task(handle.task_id).status == "CANCELED" and llm.actors == 1
            assert not tm.running_task_ids() and not tm._queue.peek_all() and not tm._applying_observe_tasks
            assert not rt.hitl_registry.list_pending(handle.session_id)
            assert not tm.is_writing_finalized()
            view = await rebuild_view(rt.event_store, handle.session_id)
            assert view.tasks[handle.task_id].status == "CANCELED"
            assert view.agents[handle.agent_id].status == "terminated"
            assert not view.pending_observe and not view.open_runs
            rows = await rt.event_store.read_range(handle.session_id)
            terminated, = [r for r in rows if r.event.type == EventType.AGENT_TERMINATED]
            canceled, = [r for r in rows if r.event.type == EventType.TASK_CANCELED]
            closure, = [r for r in rows if r.event.type == EventType.TASK_FINALIZED
                        and r.event.payload.get("outcome") == "cancelled"]
            assert terminated.position < canceled.position < closure.position
            if phase == "during_fold":
                published, = [r for r in rows if r.event.type == EventType.TASK_FINALIZED
                              and r.event.payload.get("outcome") == "continue"]
                assert published.position < terminated.position
            await subscription.unsubscribe()
            # If recovery accidentally revives the work, the model can finish;
            # actor count and cancellation facts will expose the extra execution.
            llm.finish = True
            await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
            await _drained(tm, timeout=5)
            assert tm.get_task(handle.task_id).status == "CANCELED" and llm.actors == 1
            if backend == "sqlite":
                rt = _runtime(llm, memory, rt.event_store)
                await initialize_all(rt)
                await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
                tm = rt._task_managers[handle.session_id]
                owners.append(tm)
                await _drained(tm, timeout=5)
            final = await rebuild_view(rt.event_store, handle.session_id)
            assert final.tasks[handle.task_id].status == "CANCELED" and llm.actors == 1
            assert final.agents[handle.agent_id].status == "terminated"
            assert not final.pending_observe and not final.open_runs
            assert not tm.running_task_ids() and not tm._queue.peek_all()
            rows = await rt.event_store.read_range(handle.session_id)
            assert sum(r.event.type == EventType.AGENT_TERMINATED for r in rows) == 1
            assert sum(r.event.type == EventType.TASK_CANCELED for r in rows) == 1
            assert not any(r.position > terminated.position and r.event.type == EventType.TASK_STARTED for r in rows)
        finally:
            release_meta.set()
            release_fold.set()
            release_finalized.set()
            llm.allow_review.set()
            await subscription.unsubscribe()
            jobs = [job for owner in owners
                    for job in owner._background_asyncio_tasks | set(owner._dispatch_tasks.values())]
            if cancel is not None:
                jobs.append(cancel)
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
