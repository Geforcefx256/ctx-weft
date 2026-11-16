"""Real gateways and runtime recovery must never repeat an external tool effect."""
from __future__ import annotations

import asyncio
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest

from ctx_weft import RuntimeConfig
from ctx_weft.core.loop.capability_gateway import tool_result_record_id
from ctx_weft.protocols import CapabilityRef, ToolCall
from ctx_weft.protocols.capability import CapabilityEvent
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit.test_capability_event_integrity import TC, _harness, _Tool
from tests.unit.test_storage_recovery import runtime



class _TaskLLM(MockLLMAdapter):
    def complete(self, request, stream=True):
        if "control__update_task_metadata" in {tool.name for tool in request.tools}:
            return self._stream(MockResponse(text=""), request)
        return super().complete(request, stream=stream)


def harness(**config):
    old, store, tool, authz, state, ctx = _harness()
    rt = runtime(store, **config)
    rt.providers.register_memory(ctx.memory)
    rt.providers.register_capability(tool)
    rt._capability_cache = old._cache
    ctx.event_bus = rt.event_bus
    ctx.task_manager._bus = rt.event_bus
    gateway = rt._build_gateway(ctx.memory)
    gateway._spill_threshold = 0
    ctx.capability_gateway = gateway
    return rt, gateway, store, tool, state, ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED])
@pytest.mark.parametrize("receipt_lost", [False, True])
async def test_automatic_recovery_preserves_call_stack_and_executes_tool_once(kind, receipt_lost):
    rt, gateway, store, tool, state, ctx = harness()
    original = store.append_batch
    attempts = []
    notifications = []
    full_content = "complete-result-" * 1000

    def invoke(cid, args, pctx):
        async def run():
            tool.calls += 1
            yield CapabilityEvent(kind="result", payload={"content": full_content})
        return run()

    tool.invoke = invoke

    async def append(sid, bid, events):
        if any(item.type == kind for item in events):
            attempts.append((bid, deepcopy(events)))
            if len(attempts) == 1:
                assert tool.calls == (1 if kind == EventType.CAPABILITY_FINISHED else 0)
                if receipt_lost:
                    await original(sid, bid, events)
                raise ConnectionError("temporary commit/receipt failure")
        return await original(sid, bid, events)

    async def observe(item):
        notifications.append(item.id)

    store.append_batch = append
    rt.event_bus.subscribe(None, observe)
    async with asyncio.timeout(5):
        result = await gateway.invoke("fx__act", {}, state, ctx, tool_call_id=TC)
        assert await rt.recover_storage("s1")
    assert result.content == full_content
    assert tool.calls == 1
    assert attempts[0] == attempts[1]
    assert all(count == 1 for count in Counter(notifications).values())
    rows = await ctx.memory.read_records([tool_result_record_id(TC)], ctx.provider_ctx)
    assert len(rows) == 1 and rows[0].content == full_content
    replayed = await gateway.invoke("fx__act", {}, state, ctx, tool_call_id=TC, reentry=True)
    assert replayed.content == full_content and tool.calls == 1
    events = await store.read_range("s1")
    assert sum(row.event.type == kind for row in events) == 1
    assert rt.storage_health("s1") is None


@pytest.mark.asyncio
async def test_exhausted_recovery_retains_full_result_for_later_confirmation_without_tool_repeat():
    rt, gateway, store, tool, state, ctx = harness(storage_recovery_max_attempts=2)
    original = store.append_batch
    down = True
    attempts = []

    async def append(sid, bid, events):
        if any(item.type == EventType.CAPABILITY_FINISHED for item in events):
            attempts.append(bid)
            if down:
                raise ConnectionError("storage still down")
        return await original(sid, bid, events)

    store.append_batch = append
    with pytest.raises(PersistenceUnavailableError):
        await asyncio.wait_for(gateway.invoke("fx__act", {}, state, ctx, tool_call_id=TC), 5)
    assert tool.calls == 1
    assert rt.storage_health("s1") is not None
    assert not rt.clear_storage_isolation("s1")
    down = False
    assert await asyncio.wait_for(rt.recover_storage("s1"), 5)
    # A reconciled invocation reuses its original result, not the provider.
    result = await gateway.invoke("fx__act", {}, state, ctx, tool_call_id=TC, reentry=True)
    assert result.content == "ok" and tool.calls == 1
    assert len(set(attempts)) == 1
    assert not rt._pending_tool_results


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [EventType.LLM_RESPONSE_FINISHED,
    EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_PROGRESS, EventType.CAPABILITY_FINISHED])
@pytest.mark.parametrize("receipt_lost", [False, True])
async def test_running_session_auto_recovers_and_finishes_without_reexecuting_tool(kind, receipt_lost):
    agents = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.capability_refs = [CapabilityRef(capability_id="fx:act", mode="required")]
    agents.register(template)
    store = InMemoryEventStore()
    llm = _TaskLLM(responses=[
        MockResponse(tool_calls=[ToolCall(id="call", name="fx__act", arguments={})]),
        MockResponse(text="done"),
    ], observer_verdict="success")
    rt = make_runtime(agent_provider=agents, llm=llm, event_store=store,
        config=RuntimeConfig(storage_commit_max_attempts=1, storage_recovery_base_delay_sec=0.001))
    tool = _Tool()

    def invoke(cid, args, pctx):
        async def run():
            tool.calls += 1
            yield CapabilityEvent(kind="progress", payload={"message": "working"})
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return run()

    tool.invoke = invoke
    rt.providers.register_capability(tool)
    rt.providers.register_memory(InMemoryMemoryProvider())
    await initialize_all(rt)
    container = await rt.create_session(template_id="agent:tpl_echo", session_id="s1")
    original = store.append_batch
    attempts = []

    async def append(sid, bid, events):
        if any(item.type == kind and item.origin.startswith("loop.") for item in events):
            attempts.append((bid, deepcopy(events)))
            if len(attempts) == 1:
                if receipt_lost:
                    await original(sid, bid, events)
                raise ConnectionError("one temporary storage interruption")
        return await original(sid, bid, events)

    # A host subscriber querying runtime state must not deadlock with recovery.
    async def query_after_commit(item):
        await rt.get_session(item.session_id)

    rt.event_bus.subscribe(kind, query_after_commit)
    store.append_batch = append
    handle = await rt.dispatch_task("s1", "do the operation once",
                                   agent_id=container.root_agent_id, unattended=True)
    await handle.wait_for_finish(timeout=8)
    assert attempts[0] == attempts[1]
    assert tool.calls == 1
    assert (await rt.get_task("s1", handle.task_id)).data.status == "FINISHED"
    assert rt.storage_health("s1") is None


@pytest.mark.asyncio
async def test_receipt_recovery_does_not_repeat_external_process_side_effect(tmp_path):
    from tests.integration.test_tool_outcome_not_rerun import _fixture, _read_effects

    path = tmp_path / "external-effect.db"
    memory, state, ctx, tool = _fixture(path)
    store = InMemoryEventStore()
    rt = runtime(store)
    rt.providers.register_memory(memory)
    rt.providers.register_capability(tool)
    rt._capability_cache = ctx.capability_gateway._cache
    gateway = rt._build_gateway(memory)
    ctx.event_bus = rt.event_bus
    ctx.event_store = store
    original = store.append_batch
    failed = False

    async def append(sid, bid, events):
        nonlocal failed
        receipt = await original(sid, bid, events)
        if not failed and any(item.type == EventType.CAPABILITY_FINISHED for item in events):
            failed = True
            raise ConnectionError("committed tool result; acknowledgement lost")
        return receipt

    store.append_batch = append
    await gateway.invoke("probe__record", {}, state, ctx, tool_call_id=TC)
    assert await rt.recover_storage("s1")
    await gateway.invoke("probe__record", {}, state, ctx, tool_call_id=TC, reentry=True)
    assert await rt.recover_storage("s1")
    assert _read_effects(path) == 1 and len(tool.invocations) == 1


@pytest.mark.asyncio
async def test_stopped_real_task_can_recover_after_budget_exhaustion_without_tool_repeat():
    agents = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.capability_refs = [CapabilityRef(capability_id="fx:act", mode="required")]
    agents.register(template)
    store = InMemoryEventStore()
    rt = make_runtime(agent_provider=agents, event_store=store,
        llm=_TaskLLM(responses=[MockResponse(tool_calls=[ToolCall(
            id="call", name="fx__act", arguments={})]), MockResponse(text="done")],
            observer_verdict="success"),
        config=RuntimeConfig(storage_commit_max_attempts=1, storage_recovery_max_attempts=2,
                             storage_recovery_base_delay_sec=0.001))
    tool = _Tool()
    rt.providers.register_capability(tool)
    rt.providers.register_memory(InMemoryMemoryProvider())
    await initialize_all(rt)
    container = await rt.create_session(template_id="agent:tpl_echo", session_id="s1")
    original = store.append_batch
    down = True

    async def append(sid, bid, events):
        if down and any(item.type == EventType.CAPABILITY_FINISHED for item in events):
            raise ConnectionError("outage beyond automatic recovery budget")
        return await original(sid, bid, events)

    store.append_batch = append
    handle = await rt.dispatch_task("s1", "do once", agent_id=container.root_agent_id, unattended=True)
    with pytest.raises(PersistenceUnavailableError):
        await handle.wait_for_finish(timeout=5)
    tm = rt._task_managers["s1"]
    async with asyncio.timeout(3):
        while tm._dispatch_is_live(handle.task_id):
            await asyncio.sleep(0)
    assert tool.calls == 1 and rt.storage_health("s1") is not None
    down = False
    assert await asyncio.wait_for(rt.recover_storage("s1"), 5)
    await handle.wait_for_finish(timeout=5)
    assert tool.calls == 1
    assert (await rt.get_task("s1", handle.task_id)).data.status == "FINISHED"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_PROGRESS])
async def test_exhausted_recovery_distinguishes_never_started_from_interrupted_idempotent_tool(kind):
    rt, gateway, store, tool, state, ctx = harness(storage_recovery_max_attempts=1)
    cap = replace(tool._cap(), recovery_policy="idempotent")
    gateway._cache.put("a1", [cap])
    original = store.append_batch
    down = True

    def invoke(cid, args, pctx):
        async def run():
            tool.calls += 1
            yield CapabilityEvent(kind="progress", payload={"message": "started"})
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return run()

    tool.invoke = invoke

    async def append(sid, bid, events):
        if down and any(item.type == kind for item in events):
            raise ConnectionError("long outage")
        return await original(sid, bid, events)

    store.append_batch = append
    with pytest.raises(PersistenceUnavailableError):
        await gateway.invoke("fx__act", {}, state, ctx, tool_call_id=TC)
    assert tool.calls == (0 if kind == EventType.CAPABILITY_INVOKED else 1)
    down = False
    assert await rt.recover_storage("s1")
    result = await gateway.invoke("fx__act", {}, state, ctx, tool_call_id=TC, reentry=True)
    assert tool.calls == 1
    if kind == EventType.CAPABILITY_PROGRESS:
        assert "outcome unverified" in result.content.lower()
    else:
        assert result.content == "ok"
    rows = await store.read_range("s1", include_types=(EventType.CAPABILITY_INVOKED,))
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_cancel_during_storage_recovery_does_not_restart_cancelled_task():
    agents = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.capability_refs = [CapabilityRef(capability_id="fx:act", mode="required")]
    agents.register(template)
    store = InMemoryEventStore()
    rt = make_runtime(agent_provider=agents, event_store=store,
        llm=_TaskLLM(responses=[MockResponse(tool_calls=[ToolCall(
            id="call", name="fx__act", arguments={})])], observer_verdict="success"),
        config=RuntimeConfig(storage_commit_max_attempts=1, storage_recovery_base_delay_sec=0.01))
    tool = _Tool()
    rt.providers.register_capability(tool)
    rt.providers.register_memory(InMemoryMemoryProvider())
    await initialize_all(rt)
    container = await rt.create_session(template_id="agent:tpl_echo", session_id="s1")
    original = store.append_batch
    failed, release = asyncio.Event(), asyncio.Event()

    async def append(sid, bid, events):
        if any(item.type == EventType.CAPABILITY_FINISHED for item in events):
            if not failed.is_set():
                failed.set()
                raise ConnectionError("tool finished, database temporarily unavailable")
            await release.wait()
        return await original(sid, bid, events)

    store.append_batch = append
    handle = await rt.dispatch_task("s1", "do once", agent_id=container.root_agent_id, unattended=True)
    try:
        await asyncio.wait_for(failed.wait(), 3)
        await asyncio.wait_for(rt.cancel_session("s1"), 3)
    finally:
        release.set()
    assert await asyncio.wait_for(rt.recover_storage("s1"), 5)
    tm = rt._task_managers["s1"]
    async with asyncio.timeout(3):
        while tm._dispatch_is_live(handle.task_id):
            await asyncio.sleep(0)
    assert (await rt.get_task("s1", handle.task_id)).data.status == "CANCELED"
    assert tool.calls == 1
