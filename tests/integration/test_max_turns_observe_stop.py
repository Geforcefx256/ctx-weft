
from tests._recovery import recover_fixture_session
"""Stop a real 50-turn foreground observation during and after its memory fold."""

import asyncio
from contextlib import AsyncExitStack
from dataclasses import replace

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.loop.steps.observe import ObserveStep
from ctx_weft.protocols import CapabilityRef, MemoryKind, MemoryScope, ToolCall, load_records
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_observe_turn_limit_status import _WorkTool


class _FiftyTurnsLLM(MockLLMAdapter):
    def __init__(self):
        super().__init__([], context_limit=100_000)
        self.actor_turns = 0
        self.reviews = 0
        self.finish = False

    def complete(self, request, stream=True):
        names = {tool.name for tool in request.tools}
        if "control__update_task_metadata" in names:
            response = MockResponse(tool_calls=[ToolCall(
                id="metadata", name="control__update_task_metadata",
                arguments={"title": "Work", "description": "Complete the requested work"})])
        elif "control__report_observation" in names:
            self.reviews += 1
            if self.reviews == 1:
                assert self.actor_turns == 50
                text = request.system + "\n" + "\n".join(str(m.content) for m in request.messages)
                assert "50 actor turns" in text
            response = MockResponse(tool_calls=[ToolCall(
                id="review", name="control__report_observation",
                arguments={"task_status": "success" if self.finish else "continue",
                           "act_recap": "Checked the saved progress"})])
        else:
            self.actor_turns += 1
            response = (MockResponse(text="Completed work") if self.finish else
                        MockResponse(tool_calls=[ToolCall(
                            id=f"actor-{self.actor_turns}", name="work__advance", arguments={})]))
        return self._stream(response, request)


def _runtime(llm, memory, store=None):
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    assert template.loop_config.max_turns_per_act == 50
    template.capability_refs = [CapabilityRef(capability_id="work:advance", mode="required")]
    template.loop_config = replace(template.loop_config, short_segment_token_threshold=0)
    provider.register(template)
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store)
    rt.providers.register_memory(memory)
    rt.providers.register_capability(_WorkTool())
    return rt


async def _drained(tm, timeout=2):
    async with asyncio.timeout(timeout):
        while tm._dispatch_tasks or tm._background_asyncio_tasks:
            await asyncio.sleep(0.01)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase", ["before_commit", "after_commit", "after_return", "before_finalize"])
@pytest.mark.parametrize("action", ["pause", "cancel"])
async def test_stop_after_fifty_turns_at_observe_fold(tmp_path, monkeypatch, backend, phase, action):
    async with AsyncExitStack() as stack:
        memory = (await stack.enter_async_context(open_sqlite_memory(tmp_path / "memory.db"))
                  if backend == "sqlite" else InMemoryMemoryProvider())
        llm = _FiftyTurnsLLM()
        rt = _runtime(llm, memory)
        await initialize_all(rt)
        entered, release, interrupted = [asyncio.Event() for _ in range(3)]
        held = False
        stop_acks, canceled_views = [], []
        original_fold = memory._commit_organize

        async def stop(agent_id):
            ack = await (rt.pause_agent(agent_id) if action == "pause" else rt.cancel_agent(agent_id))
            stop_acks.append(ack)

        async def hold():
            entered.set()
            await release.wait()

        # In SQLite, stop *inside* the real transaction after superseding raw
        # rows, to verify rollback rather than only canceling a pre-call wait.
        if backend == "sqlite" and phase == "before_commit":
            original_ingest = memory._ingest_in_tx

            async def ingest(db, event, ctx):
                if event.kind == MemoryKind.SUMMARY and held and not entered.is_set():
                    await hold()
                return await original_ingest(db, event, ctx)

            monkeypatch.setattr(memory, "_ingest_in_tx", ingest)

        async def fold(ids, replacements, ctx):
            nonlocal held
            if held or not any(r.kind == MemoryKind.SUMMARY for r in replacements):
                return await original_fold(ids, replacements, ctx)
            held = True
            if phase not in {"before_commit", "after_commit"}:
                return await original_fold(ids, replacements, ctx)
            try:
                if phase == "before_commit" and backend == "memory":
                    await hold()
                result = await original_fold(ids, replacements, ctx)
                if phase == "after_commit":
                    await hold()
                return result
            except asyncio.CancelledError:
                # The original SQL transaction has unwound before this read.
                canceled_views.append(await load_records(
                    memory, replacements[0].address, MemoryScope.TASK, ctx))
                interrupted.set()
                raise

        monkeypatch.setattr(memory, "_commit_organize", fold)
        if phase == "after_return":
            original_retry = ObserveStep._fold_retry_segment

            async def retry(step, state, ctx, verdict, events):
                await original_retry(step, state, ctx, verdict, events)
                if not entered.is_set():
                    await stop(state.agent.id)
                    entered.set()

            monkeypatch.setattr(ObserveStep, "_fold_retry_segment", retry)
        if phase == "before_finalize":
            async def on_event(event):
                if (event.type == EventType.STEP_STARTED
                        and event.payload.get("step_name") == "finalize" and not entered.is_set()):
                    await stop(event.agent_id)
                    entered.set()

            subscription = rt.event_bus.subscribe(None, on_event)
            stack.push_async_callback(subscription.unsubscribe)

        handle = await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="Complete the work"))
        tm = rt._task_managers[handle.session_id]
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert llm.actor_turns == 50
            if phase in {"before_commit", "after_commit"}:
                await stop(handle.agent_id)
                # Keep release closed: stopping must interrupt the fold itself.
                await asyncio.wait_for(interrupted.wait(), 2)
                snapshot, = canceled_views
                assert sum(r.kind == MemoryKind.SUMMARY for r in snapshot) == (
                    1 if phase == "after_commit" else 0)
                assert sum(r.kind == MemoryKind.CONVERSATION_TURN and r.role == "assistant"
                           for r in snapshot) == (0 if phase == "after_commit" else 50)
            assert stop_acks == [[handle.agent_id]]
            await _drained(tm)
            task = tm.get_task(handle.task_id)
            assert task.status == ("AWAITING_HUMAN" if action == "pause" else "CANCELED")
            assert llm.actor_turns == 50
            assert not tm.running_task_ids()
            view = await rebuild_view(rt.event_store, handle.session_id)
            assert not view.pending_observe and not view.open_runs
            rows = await rt.event_store.read_range(handle.session_id)
            requested, = [r for r in rows if r.event.type == EventType.OBSERVE_REQUESTED
                          and r.event.payload.get("boundary") == "max_turns"]
            identity = requested.event.payload["observe_id"]
            assert requested.event.payload["act_turns"] == 50
            assert requested.event.payload["judge_task"] is True
            completed, = [r for r in rows if r.event.type == EventType.OBSERVE_COMPLETED
                          and r.event.payload.get("observe_id") == identity]
            applied, = [r for r in rows if r.event.type == EventType.OBSERVE_APPLIED
                        and r.event.payload.get("observe_id") == identity]
            assert requested.position < completed.position < applied.position
            assert not any(r.event.type == EventType.TASK_REQUEUED for r in rows)
            if action == "cancel":
                finalized, = [r for r in rows if r.event.type == EventType.TASK_FINALIZED]
                assert finalized.position < applied.position
                assert not rt.hitl_registry.list_pending(handle.session_id)
                await handle.wait_for_finish(timeout=2)
                head = await rt.event_store.committed_head(handle.session_id)
                await rt.cancel_agent(handle.agent_id)
                assert await rt.event_store.committed_head(handle.session_id) == head
            else:
                pending, = rt.hitl_registry.list_pending(handle.session_id)
                assert pending.task_id == handle.task_id
                # A restart must preserve the pause rather than replay the old
                # accepted continue decision. SQLite also exercises persisted folds.
                if backend == "sqlite":
                    rt = _runtime(llm, memory, rt.event_store)
                    await initialize_all(rt)
                await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
                assert llm.actor_turns == 50
                llm.finish = True
                resumed = await rt.send_message(handle.agent_id, "Continue the work",
                                                session_id=handle.session_id)
                await resumed.wait_for_finish(timeout=5)
                tm = rt._task_managers[handle.session_id]
                await _drained(tm)
                assert tm.get_task(handle.task_id).status == "FINISHED"
                assert llm.actor_turns == 51
                view = await rebuild_view(rt.event_store, handle.session_id)
                assert not view.pending_observe and not view.open_runs
        finally:
            release.set()
            jobs = list(tm._background_asyncio_tasks | set(tm._dispatch_tasks.values()))
            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
