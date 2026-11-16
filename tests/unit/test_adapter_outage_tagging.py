# ctx-weft/tests/unit/test_adapter_outage_tagging.py
import json

import httpx
import pytest

from ctx_weft.protocols import LLMCallError, LLMRequest
from ctx_weft.providers.llm.adapters.anthropic import AnthropicAdapter
from ctx_weft.providers.llm.adapters.openai_compatible import OpenAICompatibleAdapter



def _anthropic_adapter(handler):
    a = AnthropicAdapter(api_key="k")
    a._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return a


def _openai_adapter(handler):
    a = OpenAICompatibleAdapter(api_key="k")
    a._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return a


def _req():
    return LLMRequest(model="m", system="s", messages=[])


async def _drain(adapter):
    out = []
    async for c in adapter.complete(_req()):
        out.append(c)
    return out


# ---------- Fake clients for mid-flight transport errors ----------

class _FakeStreamResponse:
    """Fake httpx response that yields one valid SSE line then raises ReadError."""

    def __init__(self, lines):
        self.status_code = 200
        self.headers = {}
        self._lines = lines

    async def aread(self):
        return b""

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        raise httpx.ReadError("connection reset mid-stream", request=None)


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass


class _AnthropicMidFlightClient:
    """Fake _client for AnthropicAdapter that returns a 200 stream then dies."""

    def stream(self, method, url, **kwargs):
        # One content_block_delta with text_delta — sets produced=True in anthropic.py
        sse_line = "data: " + json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        })
        return _FakeStreamCtx(_FakeStreamResponse([sse_line]))


class _OpenAIMidFlightClient:
    """Fake _client for OpenAICompatibleAdapter that returns a 200 stream then dies."""

    def stream(self, method, url, **kwargs):
        # One choices delta with content — sets produced=True in openai.py
        sse_line = "data: " + json.dumps({
            "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
        })
        return _FakeStreamCtx(_FakeStreamResponse([sse_line]))


# ---------- Anthropic ----------

@pytest.mark.asyncio
async def test_503_is_outage():
    def handler(request):
        return httpx.Response(503, headers={"retry-after": "7"}, text="overloaded")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_anthropic_adapter(handler))
    assert ei.value.retriable is True
    assert ei.value.outage is True
    assert ei.value.retry_after_sec == 7.0


@pytest.mark.asyncio
async def test_429_is_outage():
    def handler(request):
        return httpx.Response(429, text="rate limited")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_anthropic_adapter(handler))
    assert ei.value.outage is True
    assert ei.value.retriable is True


@pytest.mark.asyncio
async def test_401_is_not_outage_and_not_retriable():
    def handler(request):
        return httpx.Response(401, text="bad key")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_anthropic_adapter(handler))
    assert ei.value.retriable is False
    assert ei.value.outage is False


@pytest.mark.asyncio
async def test_anthropic_mid_flight_transport_is_outage():
    """Transport error after first chunk is emitted → retriable outage (no inline retry)."""
    a = AnthropicAdapter(api_key="k")
    a._client = _AnthropicMidFlightClient()
    got = []
    with pytest.raises(LLMCallError) as ei:
        async for c in a.complete(_req()):
            got.append(c)
    assert ei.value.retriable is True
    assert ei.value.outage is True


@pytest.mark.asyncio
async def test_bad_retry_after_header_gives_none():
    """Malformed retry-after header on 503 → retry_after_sec is None, outage is True."""
    def handler(request):
        return httpx.Response(503, headers={"retry-after": "soon"}, text="overloaded")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_anthropic_adapter(handler))
    assert ei.value.outage is True
    assert ei.value.retry_after_sec is None


# ---------- OpenAI (mirror) ----------

@pytest.mark.asyncio
async def test_openai_503_is_outage():
    def handler(request):
        return httpx.Response(503, headers={"retry-after": "5"}, text="overloaded")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_openai_adapter(handler))
    assert ei.value.retriable is True
    assert ei.value.outage is True
    assert ei.value.retry_after_sec == 5.0


@pytest.mark.asyncio
async def test_openai_429_is_outage():
    """OpenAI 429 → retriable outage, mirrors Anthropic 429 test."""
    def handler(request):
        return httpx.Response(429, text="rate limited")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_openai_adapter(handler))
    assert ei.value.outage is True
    assert ei.value.retriable is True


@pytest.mark.asyncio
async def test_openai_401_is_not_outage_and_not_retriable():
    def handler(request):
        return httpx.Response(401, text="bad key")
    with pytest.raises(LLMCallError) as ei:
        await _drain(_openai_adapter(handler))
    assert ei.value.retriable is False
    assert ei.value.outage is False


@pytest.mark.asyncio
async def test_openai_mid_flight_transport_is_outage():
    """OpenAI transport error after first chunk is emitted → retriable outage."""
    a = OpenAICompatibleAdapter(api_key="k")
    a._client = _OpenAIMidFlightClient()
    got = []
    with pytest.raises(LLMCallError) as ei:
        async for c in a.complete(_req()):
            got.append(c)
    assert ei.value.retriable is True
    assert ei.value.outage is True
