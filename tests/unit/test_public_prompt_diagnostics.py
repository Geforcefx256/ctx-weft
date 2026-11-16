"""Public diagnostics validate ownership and never materialize execution state."""
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ctx_weft.core.loop.llm_gateway import prompt_event_inputs
from ctx_weft.protocols import MemoryEvent, ProviderContext
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.template import IdentityFacet
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime
from tests.unit.test_prompt_input_snapshot import assembled, prompt_providers


async def example(purpose="act", *, tenant="default", scope=None, version=1):
    memory = InMemoryMemoryProvider()
    prompt, _, _ = await assembled(memory, purpose)
    templates = InlineAgentTemplateProvider()
    templates.register(replace(make_echo_template(), id="example", identity={
        name: IdentityFacet(f"role for {name}")
        for name in ("act", "observe", "compact", "recognize_intent")}))
    rt = make_runtime(agent_provider=templates)
    rt.providers.register_memory(memory)
    for name, provider in prompt_providers().items():
        provider.name = name
        rt.providers.register_capability(provider)
    payload = prompt_event_inputs(prompt, prompt.messages)
    payload["version"] = version
    if scope is not None:
        payload["scope"].update(scope)
    for event in [
        Event(id="session", run_id="", sequence=0, session_id="s", tenant_id=tenant,
              type=EventType.SESSION_CREATED, timestamp=datetime.now(UTC), payload={}),
        Event(id="prompt", run_id="r", sequence=1, session_id="s", tenant_id=tenant,
              task_id="t", agent_id="a", type=EventType.LLM_PROMPT_SENT,
              timestamp=datetime.now(UTC), payload=payload),
    ]:
        await rt.event_store.append_batch("s", event.id, [event])
    return rt, memory, prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("purpose", ["act", "observe", "compact", "recognize_intent"])
@pytest.mark.parametrize("trim", [True, False])
async def test_public_inspection_recompiles_without_execution_or_writes(purpose, trim):
    rt, memory, prompt = await example(purpose)
    try:
        head = await rt.event_store.committed_head("s")
        result = await rt.inspect_prompt("s", "prompt", trim=trim)
        assert result.source == "events" and not result.loaded
        assert result.committed_position == head
        assert result.data.source == "recompiled" and result.data.trim is trim
        assert result.data.system_prompt == prompt.system
        assert result.data.messages == [{"role": m.role, "content": m.content} for m in prompt.messages]
        assert result.data.tool_names == [t.name for t in prompt.tools]
        assert not rt._task_managers and not rt._initialized_sessions
        assert not rt._agent_lifecycle_manager._agents
        assert await rt.event_store.committed_head("s") == head
        # DTO mutation must not change persisted inputs or the next result.
        result.data.messages.clear()
        assert (await rt.inspect_prompt("s", "prompt")).data.messages
    finally:
        await rt.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope,version", [
    ({"session_id": "other"}, 1), ({"tenant_id": "other"}, 1),
    ({"task_id": "other"}, 1), ({"agent_id": "other"}, 1), ({}, 99)])
async def test_public_inspection_rejects_foreign_scope_and_unsupported_versions(scope, version):
    rt, _, _ = await example(scope=scope, version=version)
    try:
        with pytest.raises(ValueError):
            await rt.inspect_prompt("s", "prompt")
        with pytest.raises(ValueError, match="unavailable"):
            await rt.inspect_prompt("other", "prompt")
        with pytest.raises(ValueError, match="unavailable"):
            await rt.inspect_prompt("s", "session")
    finally:
        await rt.close()


@pytest.mark.asyncio
async def test_diagnostics_use_persisted_tenant_and_redact_image_bytes(monkeypatch):
    import json
    from copy import deepcopy

    rt, memory, _ = await example(tenant="tenant-a")
    record = (await memory.read_records(["input"], ProviderContext(session_id="s")))[0]
    memory = InMemoryMemoryProvider()
    rt.providers.register_memory(memory)
    await memory.ingest(MemoryEvent(
        id=record.id, kind=record.kind, scope=record.scope, address=record.address,
        content=record.content, timestamp=record.timestamp, role=record.role,
        metadata=record.metadata), ProviderContext(session_id="s", tenant_id="tenant-a"))
    read = memory.read_records
    contexts = []

    async def scoped(ids, ctx):
        contexts.append(ctx)
        return await read(ids, ctx)

    monkeypatch.setattr(memory, "read_records", scoped)
    saved = await rt.event_store.read_range("s", include_types=(EventType.LLM_PROMPT_SENT,))
    event = deepcopy(saved[0].event)
    event.id = "image-prompt"
    event.payload["appended_messages"] = [{"role": "user", "tool_calls": None, "tool_call_id": None, "content": [
        {"type": "image", "source_type": "base64", "media_type": "image/png", "data": "c2VjcmV0" * 100}]}]
    await rt.event_store.append_batch("s", event.id, [event])
    try:
        result = await rt.inspect_prompt("s", event.id)
        assert contexts and all(c.tenant_id == "tenant-a" and c.session_id == "s" for c in contexts)
        assert "c2VjcmV0" * 100 not in json.dumps(result.data.messages)
    finally:
        await rt.close()


@pytest.mark.asyncio
async def test_standalone_diagnostics_need_only_read_dependencies_and_one_cut(monkeypatch):
    from unittest.mock import AsyncMock

    from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
    from ctx_weft.core.prompt_diagnostics import inspect_prompt

    rt, memory, prompt = await example()
    store = rt.event_store
    position = await store.committed_head("s")
    head = AsyncMock(wraps=store.committed_head)
    read = AsyncMock(wraps=store.read_range)
    monkeypatch.setattr(store, "committed_head", head)
    monkeypatch.setattr(store, "read_range", read)
    try:
        diagnostic, cut = await inspect_prompt(
            store, "s", "prompt", memory=memory,
            template_lookup=TemplateLookup(rt.providers),
            capability_provider_index={p.name: p for p in rt.providers.get_capability_providers()},
            trim=False,
        )
        assert diagnostic.system_prompt == prompt.system
        assert diagnostic.source == "recompiled" and diagnostic.trim is False
        assert cut == position
        head.assert_awaited_once_with("s")
        assert read.await_args_list
        assert all(call.kwargs["through_position"] == cut for call in read.await_args_list)
        assert not rt._task_managers and not rt._initialized_sessions
    finally:
        await rt.close()
