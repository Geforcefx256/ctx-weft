"""A mid-stream network stall must raise an outage error, not hang forever.

Reproduces the real bug: the LLM starts responding, then the connection silently
goes dead (no bytes, no RST). With the streaming read timeout disabled the adapter
awaits the next byte forever. With a finite idle (read) timeout, httpx raises
ReadTimeout (a TransportError); since chunks were already produced, the adapter
raises LLMCallError(outage=True) -> the runtime routes it to INTERRUPTED.

The test uses a tiny local HTTP server that streams one SSE token then stalls,
keeping the socket open. We bound the call with asyncio.wait_for so that a
regression (the hang) fails loudly instead of blocking the suite.
"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from ctx_weft.protocols import LLMCallError, LLMMessage, LLMRequest
from ctx_weft.providers.llm.adapters.anthropic import AnthropicAdapter
from ctx_weft.providers.llm.adapters.openai_compatible import OpenAICompatibleAdapter


# One SSE event that yields a visible token (-> produced=True) in each provider's
# wire shape, then the handler stalls without sending anything else or closing.
_ANTHROPIC_TOKEN = (
    b'data: {"type":"content_block_delta","index":0,'
    b'"delta":{"type":"text_delta","text":"hi"}}\n\n'
)
_OPENAI_TOKEN = b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'


async def _stalling_server(token: bytes):
    """Serve HTTP/1.1: one SSE token, then hold the connection open in silence."""
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            await reader.read(65536)  # consume the request (headers + small body)
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"\r\n"
                + token
            )
            await writer.drain()
            # Stall until the client's read timeout closes the socket.  Waiting
            # for EOF avoids leaving 30-second handler tasks behind in the full suite.
            await reader.read()
        except (ConnectionError, asyncio.CancelledError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _assert_stall_raises_outage(adapter):
    req = LLMRequest(model="m", system="", messages=[LLMMessage(role="user", content="hi")])

    # This is a loopback transport test.  Do not let machine-level proxy
    # settings route localhost through a proxy during the full test suite.
    await adapter._client.aclose()
    adapter._client = httpx.AsyncClient(timeout=adapter._timeout, trust_env=False)

    async def _drain():
        return [c async for c in adapter.complete(req, stream=True)]

    # 8s budget >> the 1s idle timeout; a hang (regression) trips wait_for instead.
    try:
        with pytest.raises(LLMCallError) as exc:
            await asyncio.wait_for(_drain(), timeout=8)
        assert exc.value.outage is True, "mid-stream stall must be tagged outage"
    finally:
        await adapter._client.aclose()


@pytest.mark.asyncio
async def test_anthropic_midstream_stall_raises_outage():
    server, port = await _stalling_server(_ANTHROPIC_TOKEN)
    try:
        adapter = AnthropicAdapter(api_key="k", base_url=f"http://127.0.0.1:{port}", timeout_sec=1)
        await _assert_stall_raises_outage(adapter)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_openai_midstream_stall_raises_outage():
    server, port = await _stalling_server(_OPENAI_TOKEN)
    try:
        adapter = OpenAICompatibleAdapter(api_key="k", base_url=f"http://127.0.0.1:{port}", timeout_sec=1)
        await _assert_stall_raises_outage(adapter)
    finally:
        server.close()
        await server.wait_closed()
