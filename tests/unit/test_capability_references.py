"""Capability inputs are references; compilation validates current provider listings."""

import json
from dataclasses import asdict
from unittest.mock import AsyncMock

import pytest

from ctx_weft.core.assembler import ContextAssembler, PromptContext
from ctx_weft.core.assembler.assembler import AssemblerDeps
from ctx_weft.core.loop.llm_gateway import build_llm_request, stream_llm_resilient
from ctx_weft.core.loop.llm_stream import ensure_collected_stream
from ctx_weft.protocols import LLMChunk
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.protocols.events import EventType
from tests.unit.test_gateway_prompt_compile import setup


def ref(cid):
    return {"provider": cid.rsplit(":", 1)[0], "id": cid}


def text(prompt):
    return prompt.system + str([m.content for m in prompt.messages])


async def test_event_inputs_contain_only_capability_references():
    inputs, state, ctx, _ = await setup()
    request = await build_llm_request(ctx, state, inputs)
    payload = inputs.for_request([], request.tool_refs).to_payload()
    assert payload["capabilities"] == [ref("test:read"), ref("test:skill")]
    assert payload["tools"] == [ref("test:read")]
    for serialized in (json.dumps(payload), json.dumps(asdict(inputs))):
        for body in ("old tool description", "old provider", "old skill instructions",
                     "input_schema", "skill_definitions", "capability_errors"):
            assert body not in serialized


async def test_compile_lists_each_provider_once_and_uses_saved_order():
    inputs, state, ctx, _ = await setup()
    provider = ctx.assembler.deps.capability_provider_index["test"]
    extra = ToolCapability(id="test:extra", name="extra", purposes=["act"])
    provider.capabilities.insert(0, extra)
    provider.retrieve = AsyncMock(side_effect=AssertionError("must not reselect capabilities"))
    prompt = await ctx.assembler.compile(inputs, template=state.extra["template"],
                                         tool_refs=[ref(extra.id), ref("test:read")])
    assert provider.calls == 1
    assert [tool.name for tool in prompt.tools] == ["test__extra", "test__read"]
    assert "test__extra" not in text(prompt)  # pin affects tools, not the base recall selection
    provider.retrieve.assert_not_called()


async def test_provider_description_uses_context_and_invalidates_compiled_base():
    from ctx_weft.protocols.capability import CapabilityProviderInfo

    inputs, state, ctx, _ = await setup()
    provider = ctx.assembler.deps.capability_provider_index["test"]
    provider.description = "shared description must not be used"
    version = ["first"]
    calls = []

    async def describe(pctx):
        calls.append((pctx.tenant_id, pctx.session_id))
        return CapabilityProviderInfo(name="test", description=f"instructions:{pctx.tenant_id}:{version[0]}")

    provider.describe = describe
    first = await build_llm_request(ctx, state, inputs)
    assert f"instructions:{ctx.provider_ctx.tenant_id}:first" in text(first)
    assert provider.description not in text(first)
    version[0] = "second"
    second = await build_llm_request(ctx, state, inputs)
    assert f"instructions:{ctx.provider_ctx.tenant_id}:second" in text(second)
    assert f"instructions:{ctx.provider_ctx.tenant_id}:first" not in text(second)
    assert calls == [(ctx.provider_ctx.tenant_id, ctx.provider_ctx.session_id)] * 2


async def test_historical_compile_reads_current_definitions_and_skill_body():
    inputs, state, ctx, _ = await setup()
    payload = inputs.for_request([], [ref("test:read")]).to_payload()
    provider = ctx.assembler.deps.capability_provider_index["test"]
    provider.capabilities[0].description = "current tool description"
    provider.capabilities[0].input_schema = {"type": "object", "properties": {"new": {"type": "number"}}}
    provider.definitions["skill"].instructions = "current skill instructions"
    other = ContextAssembler([], AssemblerDeps(
        ctx.assembler.deps.memory, [], ctx.provider_ctx,
        capability_provider_index={"test": provider}))
    prompt = await other.compile(PromptContext.from_payload(payload), template=state.extra["template"])
    assert prompt.tools[0].description == "current tool description"
    assert "new" in prompt.tools[0].input_schema["properties"]
    assert "current skill instructions" in text(prompt)
    assert "old skill instructions" not in text(prompt)
    assert provider.calls == 1


@pytest.mark.parametrize("failure", ["missing_id", "missing_provider", "list_error"])
async def test_unavailable_capability_is_skipped_warned_and_reported_at_runtime(failure, caplog):
    inputs, state, ctx, _ = await setup()
    provider = ctx.assembler.deps.capability_provider_index["test"]
    if failure == "missing_id":
        provider.capabilities = [cap for cap in provider.capabilities if cap.id != "test:read"]
    elif failure == "missing_provider":
        ctx.assembler.deps.capability_provider_index.clear()
    else:
        provider.list = AsyncMock(side_effect=RuntimeError("offline"))
    prompt = await ctx.assembler.compile(inputs, template=state.extra["template"],
                                         tool_refs=[ref("test:read")])
    assert prompt.tools == []
    assert "test__read" not in text(prompt)
    assert any(error["id"] == "test:read" and error["error"] for error in prompt.capability_errors)
    assert prompt.tool_refs == [ref("test:read")]
    assert any(r.levelname == "WARNING" and "provider=test" in r.message
               and "capability_id=test:read" in r.message for r in caplog.records)
    assert inputs.capabilities[0] == ref("test:read")
    assert "capability_errors" not in inputs.to_payload()


async def test_cached_base_recompiles_after_a_capability_disappears(caplog):
    inputs, state, ctx, _ = await setup()
    first = await build_llm_request(ctx, state, inputs)
    assert "test__read" in text(first)
    provider = ctx.assembler.deps.capability_provider_index["test"]
    provider.capabilities = [cap for cap in provider.capabilities if cap.id != "test:read"]
    second = await build_llm_request(ctx, state, inputs)
    assert "test__read" not in text(second)
    assert second.tools == []
    assert second.capability_errors[0]["id"] == "test:read"
    assert provider.calls == 2
    assert "capability_id=test:read" in caplog.text


async def test_gateway_preserves_unlisted_pin_ref_but_sends_no_definition(caplog):
    from ctx_weft.core.capabilities.cache import CapabilityCache

    inputs, state, ctx, events = await setup()
    ctx.capability_cache = CapabilityCache()
    ctx.capability_cache.put("a", [ToolCapability(id="test:read", name="read", purposes=["act"])])
    ctx.capability_cache.pin("a", "t", [
        ToolCapability(id="test:expired", name="expired", purposes=["act"])])
    request = await build_llm_request(ctx, state, inputs)
    assert [t.name for t in request.tools] == ["test__read"]
    assert request.capability_errors[0]["id"] == "test:expired"
    calls = []

    async def complete(request, stream=True):
        calls.append(request)
        yield LLMChunk(kind="token", text="done")

    ctx.llm.complete = complete
    stream = ensure_collected_stream(stream_llm_resilient(ctx, state, request, inputs=inputs))
    async for _ in stream:
        stream.accept_current()
    payload = next(e.payload for e in events if e.type == EventType.LLM_PROMPT_SENT)
    assert payload["tools"] == [ref("test:read"), ref("test:expired")]
    assert payload["tool_names"] == ["test__read"]
    assert "capability_errors" not in payload
    assert calls[0].tool_refs is None and calls[0].capability_errors == []
    assert "capability_id=test:expired" in caplog.text


async def test_skill_definition_failure_is_a_runtime_warning(caplog):
    inputs, state, ctx, _ = await setup()
    provider = ctx.assembler.deps.capability_provider_index["test"]
    provider.load_definition = AsyncMock(side_effect=RuntimeError("skill removed"))
    request = await build_llm_request(ctx, state, inputs)
    assert "old skill instructions" not in request.system + str(request.messages)
    assert request.capability_errors == [{**ref("test:skill"),
                                          "error": "provider.load_definition() failed: skill removed"}]
    assert "capability_id=test:skill" in caplog.text
    assert "capability_errors" not in inputs.for_request([], request.tool_refs).to_payload()
