"""Exercise the actual OpenAI SDK against local HTTP mocks, never live accounts."""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
from ctx_weft.protocols import ModelCallConfig

import json

import httpx
import pytest

from ctx_weft.core.loop.llm_stream import CollectedLLMStream
from ctx_weft.protocols import ImagePart, LLMCallError, LLMClient, LLMMessage, LLMRequest, LLMTool
from ctx_weft.providers.llm import (
    LLMAccount,
    LLMProvider,
    ModelCapabilities,
    ModelConfig,
    ModelLimits,
)
from ctx_weft.providers.llm.adapters.openai import OpenAIAdapter
from ctx_weft.providers.llm.adapters.openai_compatible import OpenAICompatibleAdapter


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")


def request(**kwargs):
    return LLMRequest("configured-model", "system", [LLMMessage("user", "hello")], **kwargs)


def event(delta, finish_reason=None):
    return {
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0, "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


USAGE = {
    "prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
    "prompt_tokens_details": {"cached_tokens": 40},
    "completion_tokens_details": {"reasoning_tokens": 10},
}


def sse(*events):
    text = "".join("data: " + json.dumps(value) + "\n\n" for value in events) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


def adapter(handler, **kwargs):
    return OpenAIAdapter("test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kwargs)


async def collect(client, req=None, **kwargs):
    try:
        return [chunk async for chunk in client.complete(req or request(), **kwargs)]
    finally:
        await client.aclose()


@pytest.mark.parametrize("base", ["https://g.test", "https://g.test/v1/", "https://g.test/v1/chat/completions"])
async def test_sdk_request_and_trailing_usage(base):
    seen = []

    def handler(req):
        seen.append(req)
        tail = {**event({}), "choices": [], "usage": USAGE}
        return sse(event({"content": "hello"}), event({}, "stop"), tail)

    client = adapter(handler, base_url=base)
    assert isinstance(client, LLMClient)
    assert not isinstance(client, OpenAICompatibleAdapter)
    assert client._sdk.max_retries == 0
    chunks = await collect(client, request(config=ModelCallConfig(max_tokens=777)))
    wire = json.loads(seen[0].content)
    assert str(seen[0].url) == "https://g.test/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert wire["max_completion_tokens"] == 777
    assert "max_tokens" not in wire and "max_output_tokens" not in wire
    assert wire["model"] == "configured-model"
    assert wire["stream_options"] == {"include_usage": True}
    assert "".join(chunk.text for chunk in chunks if chunk.kind == "token") == "hello"
    usage = next(chunk.usage for chunk in chunks if chunk.kind == "usage")
    assert (usage.prompt_tokens, usage.input_tokens, usage.cache_read_tokens) == (100, 60, 40)
    assert (usage.completion_tokens, usage.reasoning_tokens, usage.total_tokens) == (30, 10, 130)
    assert [chunk.kind for chunk in chunks][-2:] == ["usage", "done"]
    assert client._sdk.is_closed()


async def test_parallel_tool_arguments_and_history():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return sse(
            event({"tool_calls": [
                {"index": 0, "id": "call-a", "type": "function", "function": {"name": "lookup", "arguments": '{"q":'}},
                {"index": 1, "id": "call-b", "type": "function", "function": {"name": "lookup", "arguments": '{"q":'}},
            ]}),
            event({"tool_calls": [
                {"index": 1, "function": {"arguments": '"b"}'}},
                {"index": 0, "function": {"arguments": '"a"}'}},
            ]}),
            event({}, "tool_calls"),
        )

    req = request(tools=[LLMTool("lookup", "find", {"type": "object"})])
    req.messages.extend([
        LLMMessage("assistant", "", tool_calls=[{"id": "previous", "name": "lookup", "arguments": {"q": "old"}}], reasoning_content="private"),
        LLMMessage("tool", "found", tool_call_id="previous"),
    ])
    chunks = await collect(adapter(handler), req)
    calls = [chunk.tool_call for chunk in chunks if chunk.kind == "tool_call"]
    assert [(call.id, call.name, call.arguments) for call in calls] == [
        ("call-a", "lookup", {"q": "a"}), ("call-b", "lookup", {"q": "b"}),
    ]
    assert any(chunk.kind == "tool_call_partial" for chunk in chunks)
    assert seen[0]["tools"][0]["function"]["name"] == "lookup"
    assert "reasoning_content" not in seen[0]["messages"][-2]
    assert seen[0]["messages"][-1]["tool_call_id"] == "previous"


async def test_nonstreaming_uses_sdk_completion():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
            "usage": USAGE,
        })

    chunks = await collect(adapter(handler), stream=False)
    assert seen[0]["stream"] is False and "stream_options" not in seen[0]
    assert "max_completion_tokens" not in seen[0]
    assert chunks[0].text == "answer"
    assert chunks[-1].kind == "done"


@pytest.mark.parametrize("image_input", [False, True])
async def test_image_capability_and_tool_image_relocation(image_input):
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return sse(event({"content": "ok"}), event({}, "stop"))

    req = request()
    req.messages = [
        LLMMessage("user", [ImagePart(data="aGVsbG8=", media_type="image/png")]),
        LLMMessage("assistant", "", tool_calls=[{"id": "c", "name": "screenshot", "arguments": {}}]),
        LLMMessage("tool", [ImagePart(data="aGVsbG8=", media_type="image/png")], tool_call_id="c"),
    ]
    await collect(adapter(handler, image_input=image_input), req)
    wire = seen[0]["messages"]
    images = [part for message in wire if isinstance(message.get("content"), list)
              for part in message["content"] if part["type"] == "image_url"]
    assert len(images) == (2 if image_input else 0)
    assert all(isinstance(message["content"], str) for message in wire if message["role"] == "tool")


@pytest.mark.parametrize("status,retriable", [(400, False), (401, False), (403, False), (429, True), (503, True)])
async def test_sdk_http_errors_preserve_message_and_are_not_retried(status, retriable):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(status, headers={"retry-after": "2"}, json={"error": {"message": "secret-upstream-body"}})

    with pytest.raises(LLMCallError) as caught:
        await collect(adapter(handler))
    assert len(seen) == 1
    assert caught.value.status_code == status
    assert caught.value.retriable is retriable and caught.value.outage is retriable
    assert caught.value.retry_after_sec == 2
    assert "secret-upstream-body" in str(caught.value)


async def test_transport_error_retains_outage_classification():
    def handler(req):
        raise httpx.ConnectError("secret-network-address", request=req)

    with pytest.raises(LLMCallError) as caught:
        await collect(adapter(handler))
    assert caught.value.retriable and caught.value.outage
    assert "secret-network-address" in str(caught.value)


async def test_stream_without_terminal_is_not_success():
    with pytest.raises(LLMCallError, match="terminal"):
        await collect(adapter(lambda req: sse(event({"content": "partial"}))))


async def test_official_deltas_preserve_repeated_text_and_literal_markup():
    chunks = await collect(adapter(lambda req: sse(
        event({"content": "ha"}), event({"content": "ha"}),
        event({"content": " <tool_call>example</tool_call>"}), event({}, "stop"),
    )))
    assert "".join(chunk.text for chunk in chunks if chunk.kind == "token") == "haha <tool_call>example</tool_call>"
    assert not any(chunk.kind == "tool_call" for chunk in chunks)


async def test_collector_preserves_sdk_text_and_usage_without_dialect_parsing():
    client = adapter(lambda req: sse(
        event({"content": "ha"}), event({"content": "ha <tool_call>example</tool_call>"}),
        event({}, "stop"), {**event({}), "choices": [], "usage": USAGE},
    ))
    stream = CollectedLLMStream(client.complete(request()))
    try:
        async for _ in stream:
            stream.accept_current()
        result = stream.get_final_result()
        assert result.text == "haha <tool_call>example</tool_call>"
        assert result.usage.total_tokens == 130
        assert result.provider_state == {} and result.tool_calls == ()
    finally:
        await stream.aclose()
        await client.aclose()


async def test_canceling_read_closes_the_sdk_stream():
    closed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield ("data: " + json.dumps(event({"content": "first"})) + "\n\n").encode()
            yield ("data: " + json.dumps(event({}, "stop")) + "\n\n").encode()

        async def aclose(self):
            closed.append(True)

    client = adapter(lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body()))
    source = client.complete(request())
    try:
        assert (await anext(source)).text == "first"
        await source.aclose()
        assert closed == [True]
    finally:
        await client.aclose()


async def test_sdk_models_and_close():
    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, json={"object": "list", "data": [{"id": "available", "object": "model", "created": 0, "owned_by": "openai"}]})

    client = adapter(handler)
    try:
        assert await client.list_models() == ["available"]
    finally:
        await client.aclose()
    assert seen == ["https://api.openai.com/v1/models"]
    assert client._sdk.is_closed()


async def test_protocols_coexist_and_legacy_config_stays_compatible():
    class Store:
        def save(self, account):
            pass

    provider = LLMProvider(Store())
    provider.register_account(LLMAccount("account", "key", "https://g.test", [
        ModelConfig("legacy", "openai", ModelLimits(128_000)),
        ModelConfig("compatible", "openai-compatible", ModelLimits(128_000)),
        ModelConfig("official", "openai-official", ModelLimits(128_000), ModelCapabilities(["text", "image"])),
    ]))
    legacy, compatible, official = [provider.get_client("account", name, ctx=_PROVIDER_CONTEXT) for name in ("legacy", "compatible", "official")]
    try:
        assert isinstance(legacy._adapter, OpenAICompatibleAdapter)
        assert isinstance(compatible._adapter, OpenAICompatibleAdapter)
        assert type(official._adapter) is OpenAIAdapter and official._adapter._image_input
        assert provider.get_client(ctx=_PROVIDER_CONTEXT).model == "legacy"
        assert [model.protocol for model in provider.get_account("account").models] == ["openai", "openai-compatible", "openai-official"]
    finally:
        await legacy._adapter._client.aclose()
        await compatible._adapter._client.aclose()
        await official._adapter.aclose()
