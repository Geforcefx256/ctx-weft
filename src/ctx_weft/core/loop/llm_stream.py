"""Per-request collection shared by adapters and every model-stream consumer.

Native adapters keep raw text here and only emit its visible deltas. Legacy
clients supply already-visible chunks. Neither gateways nor business callers
accumulate a second copy. Accepted positions are lengths, never text snapshots.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from ctx_weft.protocols.llm import (
    LLMChunk,
    LLMStream,
    LLMStreamBuffers,
    LLMStreamParser,
    LLMStreamResult,
    LLMUsage,
    ToolCall,
    stream_writer_context,
)


@dataclass(frozen=True)
class _Position:
    text: int = 0
    reasoning: int = 0
    tools: int = 0
    finish_reason: str | None = None


class StreamState:
    """The only mutable full-content owner for one underlying request attempt."""

    def __init__(self) -> None:
        self.buffers = LLMStreamBuffers()
        self._parser: LLMStreamParser | None = None
        self._extensions: dict[str, Any] = {}
        self._reasoning: list[tuple[int | None, str]] = []
        self._tools: list[ToolCall] = []
        self._usage = LLMUsage()
        self._delivered = _Position()
        self._accepted = _Position()
        self._final: LLMStreamResult | None = None
        self._closed_result: LLMStreamResult | None = None

    @property
    def content_text(self) -> str:
        return self.buffers.content_text

    @property
    def tool_buffers(self) -> dict[int, dict[str, Any]]:
        return self.buffers.tool_buffers

    def record(self, chunk: LLMChunk) -> None:
        p = self._delivered
        text, reasoning, tools, finish = (
            p.text,
            p.reasoning,
            p.tools,
            p.finish_reason,
        )
        if chunk.kind == "token":
            if self._parser is None:
                self.buffers.content_text += chunk.text
            text += len(chunk.text)
        elif chunk.kind == "reasoning":
            index = getattr(chunk, "reasoning_index", None)
            if self._reasoning and self._reasoning[-1][0] == index:
                self._reasoning[-1] = (index, self._reasoning[-1][1] + chunk.text)
            else:
                self._reasoning.append((index, chunk.text))
            reasoning += len(chunk.text)
        elif chunk.kind == "tool_call" and chunk.tool_call is not None:
            self._tools.append(chunk.tool_call)
            tools += 1
        elif chunk.kind == "usage" and chunk.usage is not None:
            # Usage is an already-received request fact, independent of whether
            # the business consumer accepts the current output position.
            reported = {name: value for name, value in asdict(chunk.usage).items()
                        if value is not None}
            values = asdict(self._usage)
            # Counters are cumulative within an attempt: replace known values,
            # never add snapshots or erase a known value with an absent field.
            # Re-derive dependent counts only when their operands change and
            # are all known; LLMUsage remains the owner of the arithmetic.
            for target, operands in (
                ("total_tokens", ("prompt_tokens", "completion_tokens")),
                ("input_tokens", ("prompt_tokens", "cache_read_tokens", "cache_write_tokens")),
            ):
                if (target not in reported
                        and any(name in reported and reported[name] != values[name] for name in operands)
                        and all(reported.get(name, values[name]) is not None for name in operands)):
                    values[target] = None
            values.update(reported)
            self._usage = LLMUsage(**values)
        if getattr(chunk, "finish_reason", None) is not None:
            finish = chunk.finish_reason
        self._delivered = _Position(text, reasoning, tools, finish)

    def accept_current(self) -> None:
        self._accepted = self._delivered

    def _result(self, position: _Position) -> LLMStreamResult:
        visible = self._parser.visible_text() if self._parser is not None else self.content_text
        remaining = position.reasoning
        parts: list[str] = []
        block_parts: dict[int, list[str]] = {}
        incomplete: set[int] = set()
        for index, text in self._reasoning:
            take = text[:remaining]
            remaining -= len(take)
            parts.append(take)
            if index is not None:
                block_parts.setdefault(index, []).append(take)
                if len(take) != len(text):
                    incomplete.add(index)
        reasoning_text = "".join(parts)
        single_block = len(block_parts) == 1 and all(
            index is not None for index, _ in self._reasoning
        )
        reasoning_blocks = {
            index: reasoning_text if single_block else "".join(parts)
            for index, parts in block_parts.items()
        }
        tools = tuple(self._tools[: position.tools])
        provider_state = (
            self._parser.provider_state(reasoning_blocks, incomplete, tools)
            if self._parser is not None else {}
        )
        return LLMStreamResult(
            text=visible[: position.text],
            reasoning=reasoning_text,
            tool_calls=tools,
            usage=replace(self._usage),
            finish_reason=position.finish_reason,
            provider_state=provider_state,
        )

    def partial_result(self) -> LLMStreamResult:
        if self._closed_result is not None:
            return self._closed_result
        if self._final is not None:
            if self._accepted == self._delivered:
                return self._final
            # A stream can finish before a business consumer accepts its last chunk.
            p, full = self._accepted, self._final
            return LLMStreamResult(
                text=full.text[: p.text],
                reasoning=full.reasoning[: p.reasoning],
                tool_calls=full.tool_calls[: p.tools],
                usage=full.usage,
                finish_reason=p.finish_reason,
            )
        return self._result(self._accepted)

    def finish(self) -> None:
        if self._final is None:
            self._final = self._result(self._delivered)
            self._release_buffers()

    def close(self) -> None:
        if self._final is None and self._closed_result is None:
            self._closed_result = self._result(self._accepted)
            self._release_buffers()

    def _release_buffers(self) -> None:
        self.buffers.content_text = ""
        self.tool_buffers.clear()
        self._reasoning.clear()
        self._tools.clear()
        self._extensions.clear()
        if self._parser is not None:
            self._parser.release()
            self._parser = None

    def get_final_result(self) -> LLMStreamResult:
        if self._final is None:
            raise RuntimeError("Model stream has not completed; use partial_result().")
        return self._final


class CollectedLLMStream:
    def __init__(
        self,
        source: AsyncIterator[LLMChunk] | Callable[[CollectedLLMStream], AsyncIterator[LLMChunk]],
    ) -> None:
        self.state = StreamState()
        self._source = source(self) if callable(source) else source
        self._finished = False
        self._closed = False
        self._reading = False
        self._read_task: asyncio.Task[Any] | None = None

    def begin_attempt(self) -> LLMStreamBuffers:
        self.state = StreamState()
        return self.state.buffers

    def attach_parser(self, parser: LLMStreamParser) -> None:
        self.state._parser = parser

    def extension(self, key: str, factory: Callable[[], Any]) -> Any:
        if key not in self.state._extensions:
            self.state._extensions[key] = factory()
        return self.state._extensions[key]

    def __aiter__(self) -> CollectedLLMStream:
        return self

    async def __anext__(self) -> LLMChunk:
        if self._closed or self._finished:
            raise StopAsyncIteration
        if self._reading:
            raise RuntimeError("A model stream supports only one reader.")
        self._reading = True
        self._read_task = asyncio.current_task()
        with stream_writer_context(self):
            try:
                chunk = await self._source.__anext__()
                self.state.record(chunk)
                return chunk
            except StopAsyncIteration:
                self._finished = True
                self.state.finish()
                raise
            finally:
                self._reading = False
                self._read_task = None

    def accept_current(self) -> None:
        self.state.accept_current()

    def partial_result(self) -> LLMStreamResult:
        return self.state.partial_result()

    def get_final_result(self) -> LLMStreamResult:
        return self.state.get_final_result()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                reader = self._read_task
                if reader is not None and reader is not asyncio.current_task():
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                close = getattr(self._source, "aclose", None)
                if close is not None:
                    await close()
            finally:
                self.state.close()

    async def __aenter__(self) -> CollectedLLMStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


def ensure_collected_stream(source: AsyncIterator[LLMChunk]) -> LLMStream:
    """Reuse native stream ownership; collect only complete-only legacy clients."""
    return source if isinstance(source, LLMStream) else CollectedLLMStream(source)


async def stream_until_cancel(source, cancel_token):
    """Close a request before propagating cancellation, including before first byte."""
    stopped = asyncio.create_task(cancel_token.wait()) if cancel_token is not None else None
    reader = None
    try:
        while True:
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            reader = asyncio.ensure_future(source.__anext__())
            await asyncio.wait({reader, stopped} if stopped is not None else {reader},
                               return_when=asyncio.FIRST_COMPLETED)
            if cancel_token is not None:
                cancel_token.raise_if_cancelled()
            try:
                yield reader.result()
            except StopAsyncIteration:
                return
    finally:
        pending = [t for t in (reader, stopped) if t is not None]
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await source.aclose()


class DelegatingLLMStream:
    """Gateway lifecycle/retry wrapper. It owns no body or reasoning buffers."""

    def __init__(self, source: Callable[[DelegatingLLMStream], AsyncIterator[LLMChunk]]) -> None:
        self._source = source(self)
        self._inner: LLMStream | None = None
        self._finished = False
        self._closed = False
        self._read_task: asyncio.Task[Any] | None = None
        self.request_started = False

    def bind(self, source: AsyncIterator[LLMChunk]) -> LLMStream:
        self._inner = ensure_collected_stream(source)
        return self._inner

    def discard_result(self) -> None:
        """Detach a closed failed stream, including during retry backoff/cancel."""
        self._inner = None

    def __aiter__(self) -> DelegatingLLMStream:
        return self

    async def __anext__(self) -> LLMChunk:
        if self._closed or self._finished:
            raise StopAsyncIteration
        if self._read_task is not None:
            raise RuntimeError("A model stream supports only one reader.")
        self._read_task = asyncio.current_task()
        try:
            return await self._source.__anext__()
        except StopAsyncIteration:
            self._finished = True
            raise
        finally:
            self._read_task = None

    def accept_current(self) -> None:
        if self._inner is not None:
            self._inner.accept_current()

    def partial_result(self) -> LLMStreamResult:
        return self._inner.partial_result() if self._inner is not None else LLMStreamResult()

    def get_final_result(self) -> LLMStreamResult:
        if not self._finished or self._inner is None:
            raise RuntimeError("Model stream has not completed; use partial_result().")
        return self._inner.get_final_result()

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                reader = self._read_task
                if reader is not None and reader is not asyncio.current_task():
                    reader.cancel()
                    await asyncio.gather(reader, return_exceptions=True)
                await self._source.aclose()
            finally:
                if self._inner is not None:
                    await self._inner.aclose()

    async def __aenter__(self) -> DelegatingLLMStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
