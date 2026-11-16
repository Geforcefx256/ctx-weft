"""At each 50-turn boundary, continue work or fail for a capability limitation."""

import asyncio

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.protocols import CapabilityRef, ToolCall
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse, observer_success_with
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


class _WorkTool(ToolCapabilityProvider):
    name = "work"

    async def list(self, ctx):
        return [ToolCapability(id="work:advance", name="advance", description="Advance the work",
                               purposes=["act"], side_effects=False)]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    async def invoke(self, capability_id, arguments, ctx):
        yield CapabilityEvent(kind="result", payload={"content": "Progress saved; the goal remains unfinished."})

    async def cancel(self, invocation_id, ctx):
        pass


class _LimitLLM(MockLLMAdapter):
    def __init__(self):
        responses = []
        for attempt in range(5):
            responses.extend(
                MockResponse(tool_calls=[ToolCall(
                    id=f"work_{attempt}_{i}", name="work__advance", arguments={})])
                for i in range(50)
            )
            responses.append(observer_success_with(
                ["waiting", "success", "continue", "waiting", "fail"][attempt]))
        super().__init__(responses)

    def complete(self, request, stream=True):
        if any(t.name == "control__report_observation" for t in request.tools or []):
            text = request.system + "\n" + "\n".join(str(m.content) for m in request.messages)
            assert "50 actor turns" in text
        if any(t.name == "control__update_task_metadata" for t in request.tools or []):
            return self._stream(MockResponse(tool_calls=[ToolCall(
                id="metadata", name="control__update_task_metadata",
                arguments={"title": "Complete work", "description": "Complete the requested work"},
            )]), request)
        return super().complete(request, stream)


@pytest.mark.asyncio
async def test_turn_limits_continue_until_an_explicit_capability_failure():
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.capability_refs = [CapabilityRef(capability_id="work:advance", mode="required")]
    provider.register(template)
    store = InMemoryEventStore()
    runtime = make_runtime(llm=_LimitLLM(), agent_provider=provider, event_store=store)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    runtime.providers.register_capability(_WorkTool())
    await initialize_all(runtime)
    queried = []
    turns = 0

    async def query_on_notification(event):
        nonlocal turns
        if event.type == EventType.ACT_TURN_STARTED:
            turns += 1
        if event.type in (EventType.TASK_REQUEUED, EventType.TASK_FAILED):
            task = (await runtime.get_task(event.session_id, event.task_id)).data
            queried.append((turns, event.type, task.status, event.payload["retry_count"]))

    subscription = runtime.event_bus.subscribe(None, query_on_notification)
    handle = await runtime.start_session(SessionStartParams.create(
        session_id="turn-limit", template_id="agent:tpl_echo", user_prompt="Complete the work",

    ))
    await handle.wait_for_finish(timeout=30)
    tm = runtime._task_managers[handle.session_id]
    while tm._background_asyncio_tasks:
        await asyncio.gather(*list(tm._background_asyncio_tasks))

    assert queried == [
        (50, EventType.TASK_REQUEUED, "PENDING", 0),
        (100, EventType.TASK_REQUEUED, "PENDING", 0),
        (150, EventType.TASK_REQUEUED, "PENDING", 0),
        (200, EventType.TASK_REQUEUED, "PENDING", 0),
        (250, EventType.TASK_FAILED, "FAILED", 0),
    ]
    task = tm.get_task(handle.task_id)
    assert not task.unattended
    rows = await store.read_range(handle.session_id)
    requests = [r for r in rows if r.event.type == EventType.OBSERVE_REQUESTED]
    assert len(requests) == 5
    assert all(r.event.payload["act_turns"] == 50 for r in requests)
    failed, = [r.event for r in rows if r.event.type == EventType.TASK_FAILED]
    assert failed.payload["error_code"] == "TASK_FAILED_BY_OBSERVER"
    assert task.retry_count == 0 and tm.session.failure_counter == 1
    assert tm.is_done() and not tm.running_task_ids()
    assert (await runtime.get_agent(handle.agent_id)).data.state.status == "idle"
    await subscription.unsubscribe()
