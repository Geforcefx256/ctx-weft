
from tests._recovery import recover_fixture_session
"""Reach foreground waiting through actual act context exhaustion, then pause/resume."""

from dataclasses import replace

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import CapabilityRef, LLMChunk, LLMUsage, ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import PREFACE_AFTER_INTERRUPT_EDIT
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_observe_turn_limit_status import _WorkTool
from tests.integration.test_observe_waiting_continue import quiescent


class _ContextLimitLLM(MockLLMAdapter):
    def __init__(self):
        super().__init__([], context_limit=100_000)
        self.actor_turns = 0
        self.finish = False

    def complete(self, request, stream=True):
        names = {tool.name for tool in request.tools}
        if "control__update_task_metadata" in names:
            response = MockResponse(tool_calls=[ToolCall(
                id="metadata", name="control__update_task_metadata",
                arguments={"title": "Work", "description": "Complete the work"})])
        elif "control__report_observation" in names:
            response = MockResponse(tool_calls=[ToolCall(
                id="review", name="control__report_observation",
                arguments={"task_status": "success" if self.finish else "waiting",
                           "act_recap": "Need the user's input"})])
        elif "work__advance" in names:
            self.actor_turns += 1
            response = (MockResponse(text="Completed work") if self.finish else
                        MockResponse(tool_calls=[ToolCall(
                            id=f"actor-{self.actor_turns}", name="work__advance", arguments={})]))
        else:
            response = MockResponse(text="Condensed progress")

        async def chunks():
            async for chunk in self._stream(response, request):
                if chunk.kind == "usage" and "work__advance" in names and not self.finish:
                    yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=100_000))
                else:
                    yield chunk
        return chunks()


@pytest.mark.parametrize("recoveries", [0, 2])
@pytest.mark.parametrize("cold", [False, True])
async def test_context_exhaustion_waiting_pause_resume_public_chain(recoveries, cold):
    llm = _ContextLimitLLM()
    template = make_echo_template()
    assert template.loop_config.max_context_recoveries == 2
    template.loop_config = replace(template.loop_config, max_context_recoveries=recoveries,
                                   short_segment_token_threshold=0)
    template.capability_refs = [CapabilityRef(capability_id="work:advance", mode="required")]
    provider = InlineAgentTemplateProvider()
    provider.register(template)
    memory = InMemoryMemoryProvider()

    def runtime(store=None):
        rt = make_runtime(llm=llm, agent_provider=provider, event_store=store)
        rt.providers.register_memory(memory)
        rt.providers.register_capability(_WorkTool())
        return rt

    rt = runtime()
    await initialize_all(rt)
    acks = []

    async def pause_at_wait(event):
        if event.type == EventType.HITL_OPENED and event.payload.get("tool_call_id") and not acks:
            acks.append(await rt.pause_agent(event.agent_id))

    rt.event_bus.subscribe(None, pause_at_wait)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="Complete the work"))
    tm = rt._task_managers[handle.session_id]
    await quiescent(rt, handle)
    assert acks == [[handle.agent_id]]
    assert tm.get_task(handle.task_id).status == "AWAITING_HUMAN"
    turns = llm.actor_turns
    assert turns == (1 if recoveries == 0 else 2)
    rows = await rt.event_store.read_range(handle.session_id)
    requested, = [r for r in rows if r.event.type == EventType.OBSERVE_REQUESTED
                  and r.event.payload.get("boundary") == "context_limit"]
    assert requested.event.payload["execution_mode"] == "foreground"
    assert requested.event.payload["judge_task"] is True
    pending = rt.hitl_registry.list_pending(handle.session_id)
    assert len(pending) == 1
    assert pending[0].delivery.preface == PREFACE_AFTER_INTERRUPT_EDIT
    view = await rebuild_view(rt.event_store, handle.session_id)
    assert not view.pending_observe and not view.open_runs
    if cold:
        rt = runtime(rt.event_store)
        await initialize_all(rt)
        await recover_fixture_session(rt, handle.agent_id, session_id=handle.session_id)
        assert llm.actor_turns == turns
        assert len(rt.hitl_registry.list_pending(handle.session_id)) == 1
    llm.finish = True
    assert await rt.resume_agent(handle.agent_id, session_id=handle.session_id) == [handle.agent_id]
    tm = rt._task_managers[handle.session_id]
    await quiescent(rt, handle)
    assert tm.get_task(handle.task_id).status == "FINISHED"
    assert llm.actor_turns == turns + 1
    assert not rt.hitl_registry.list_pending(handle.session_id)
    view = await rebuild_view(rt.event_store, handle.session_id)
    assert not view.pending_observe and not view.open_runs
