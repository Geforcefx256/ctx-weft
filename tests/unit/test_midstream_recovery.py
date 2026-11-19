
from tests.model_testkit import model_snapshot
"""A failed generation is revoked before retry; consumers see only fresh state."""

import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.core.control.tokens import CancelToken
from ctx_weft.core.loop import llm_gateway
from ctx_weft.core.loop.llm_gateway import stream_llm_resilient
from ctx_weft.protocols import (
    LLMCallError,
    LLMChunk,
    LLMMessage,
    LLMOutageError,
    LLMRequest,
    LLMUsage,
    ToolCall,
)
from ctx_weft.protocols.events import EventType
from tests.assembler_testkit import scripted_inputs


class Bus:
    def __init__(self):
        self.events = []

    async def emit(self, event):
        self.events.append(event)

    async def emit_batch(self, events):
        self.events.extend(events)


def context(client, **config):
    return SimpleNamespace(
        llm=client, event_bus=Bus(), cancel_token=CancelToken(),
        config=SimpleNamespace(llm_self_heal_max_attempts=3,
                               llm_self_heal_base_delay_sec=0,
                               llm_self_heal_max_interval_sec=0, **config),
    )


def state():
    return SimpleNamespace(agent=SimpleNamespace(id="a", loop_guard=None),
                           session=SimpleNamespace(id="s", tenant_id="t", token_used=0),
                           task=SimpleNamespace(id="task"), run_id="run",
                           sequence_counter=0, resolved_model=model_snapshot(model="m", account="acc"))


def request():
    return LLMRequest(model="m", system="s", messages=[LLMMessage(role="user", content="input")])


class Client:
    def __init__(self, failed_chunks, fail_for=1, advance=None):
        self.failed_chunks = failed_chunks
        self.fail_for = fail_for
        self.calls = 0
        self.closed = []
        self.inputs = []
        self.advance = advance

    def complete(self, req, stream=True):
        self.calls += 1
        call = self.calls
        assert self.closed == list(range(1, call)), "old request must close before the next starts"
        self.inputs.append(req.messages[0].content)
        req.messages[0].content = "adapter mutation"

        async def run():
            try:
                if call <= self.fail_for:
                    for chunk in self.failed_chunks:
                        yield chunk
                    if self.advance:
                        self.advance(call)
                    raise LLMCallError("wire lost", retriable=True, outage=True)
                yield LLMChunk(kind="token", text="fresh")
                yield LLMChunk(kind="done", finish_reason="stop")
            finally:
                self.closed.append(call)
        return run()


@pytest.mark.parametrize("chunks", [
    [LLMChunk(kind="token", text="old")],
    [LLMChunk(kind="reasoning", text="old thought")],
    [LLMChunk(kind="tool_call_partial")],
    [LLMChunk(kind="tool_call", tool_call=ToolCall(id="old", name="write", arguments={}))],
])
async def test_retries_from_clean_state_with_one_request_identity(chunks):
    client = Client(chunks)
    ctx = context(client)
    req = request()
    stream = stream_llm_resilient(ctx, state(), req)
    async with stream:
        async for _ in stream:
            stream.accept_current()
    result = stream.get_final_result()
    assert (result.text, result.reasoning, result.tool_calls, result.provider_state) == ("fresh", "", (), {})
    assert client.inputs == ["input", "input"]
    assert req.messages[0].content == "input"
    events = ctx.event_bus.events
    discarded = [e for e in events if e.type == EventType.LLM_RESPONSE_DISCARDED]
    assert len(discarded) == 1 and discarded[0].payload["will_retry"]
    assert not discarded[0].payload["usage_known"]
    assert len([e for e in events if e.type == EventType.LLM_REQUEST_STARTED]) == 1
    assert {e.payload["request_id"] for e in events} == {"req_run_a_0"}
    assert all("attempt_id" not in e.payload for e in events)
    types = [e.type for e in events]
    assert types.index(EventType.LLM_RESPONSE_DISCARDED) < types.index(EventType.LLM_RETRY_TRIGGERED)


async def test_exhaustion_also_revokes_the_last_partial_output(caplog):
    client = Client([LLMChunk(kind="token", text="old")], fail_for=10)
    ctx = context(client)
    stream = stream_llm_resilient(ctx, state(), request())
    with pytest.raises(LLMOutageError, match="^wire lost$") as error:
        async with stream:
            async for _ in stream:
                stream.accept_current()
    assert stream.partial_result().text == ""
    discarded = [e for e in ctx.event_bus.events if e.type == EventType.LLM_RESPONSE_DISCARDED]
    assert [e.payload["will_retry"] for e in discarded] == [True, True, False]
    assert client.closed == [1, 2, 3]
    assert client.calls == 3
    assert isinstance(error.value.__cause__, LLMCallError)
    assert error.value.outage and error.value.retriable
    assert "retry budget exhausted after 3 attempt(s)" in caplog.text


async def test_cancel_during_backoff_cannot_save_discarded_content(monkeypatch):
    client = Client([LLMChunk(kind="token", text="old")])
    ctx = context(client)
    stream = stream_llm_resilient(ctx, state(), request())

    async def cancel(_delay, token):
        assert stream.partial_result().text == ""
        token.cancel()
        token.raise_if_cancelled()

    monkeypatch.setattr(llm_gateway, "_sleep_cancellable", cancel)
    with pytest.raises(asyncio.CancelledError):
        async with stream:
            async for _ in stream:
                stream.accept_current()
    assert client.calls == 1 and stream.partial_result().text == ""


async def test_long_output_gets_one_nonrenewable_midstream_window(monkeypatch, caplog):
    now = [0.0]
    monkeypatch.setattr(llm_gateway, "monotonic", lambda: now[0])
    client = Client([LLMChunk(kind="token", text="old")], fail_for=10,
                    advance=lambda call: now.__setitem__(0, 400.0 if call == 1 else 411.0))
    ctx = context(client, llm_self_heal_max_duration_sec=300,
                  llm_midstream_recovery_max_duration_sec=10)
    stream = stream_llm_resilient(ctx, state(), request())
    with pytest.raises(LLMOutageError, match="^wire lost$") as error:
        async with stream:
            async for _ in stream:
                stream.accept_current()
    assert client.calls == 2  # initial 300s passed; second failure does not renew the window
    assert client.closed == [1, 2]
    assert stream.partial_result().text == ""
    discarded = [e for e in ctx.event_bus.events if e.type == EventType.LLM_RESPONSE_DISCARDED]
    assert [e.payload["will_retry"] for e in discarded] == [True, False]
    assert isinstance(error.value.__cause__, LLMCallError)
    assert error.value.outage and error.value.retriable
    assert "retry budget exhausted after 2 attempt(s)" in caplog.text


async def test_discard_reports_observed_usage_without_a_finished_response():
    usage = LLMUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12)
    ctx = context(Client([LLMChunk(kind="usage", usage=usage)]))
    loop_state = state()
    stream = stream_llm_resilient(ctx, loop_state, request())
    async with stream:
        async for _ in stream:
            stream.accept_current()
    event = next(e for e in ctx.event_bus.events if e.type == EventType.LLM_RESPONSE_DISCARDED)
    assert event.payload["usage_known"] and event.payload["usage"]["total_tokens"] == 12
    assert stream.get_final_result().usage.total_tokens is None
    assert loop_state.session.token_used == 12


async def test_failed_tool_declaration_never_executes_or_enters_memory():
    from ctx_weft.core.loop.steps.act import ActStep
    from ctx_weft.core.models.config import RuntimeConfig
    from ctx_weft.protocols import MemoryEventType
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from tests.integration.test_interactive_task import _act_state_ctx

    class Flaky(MockLLMAdapter):
        calls = 0

        def complete(self, req, stream=True):
            self.calls += 1
            if self.calls == 1:
                async def failed():
                    yield LLMChunk(kind="token", text="discarded")
                    yield LLMChunk(kind="tool_call", tool_call=ToolCall(
                        id="bad", name="write", arguments={"target": "bad"}))
                    raise LLMCallError("wire lost", retriable=True, outage=True)
                return failed()
            return super().complete(req, stream)

    executed = []

    async def invoke(**kwargs):
        executed.append((kwargs["tool_name"], kwargs["arguments"]))
        return SimpleNamespace(content="result", is_error=False)

    llm = Flaky(responses=[
        MockResponse(text="fresh", tool_calls=[ToolCall(id="ok", name="lookup", arguments={"q": "good"})]),
        MockResponse(text="done"),
    ])
    loop_state, ctx, _, _, memory = _act_state_ctx(True, llm, gateway=SimpleNamespace(invoke=invoke))
    ctx.config = RuntimeConfig(llm_self_heal_base_delay_sec=0, llm_self_heal_max_interval_sec=0)
    await ActStep().execute(loop_state, ctx)
    assert executed == [("lookup", {"q": "good"})]
    records = await memory.recall_recent(loop_state.scope, [MemoryEventType.LLM_RESPONSE], 10, ctx.provider_ctx)
    assert {record.content for record in records} == {"fresh", "done"}
    assert llm.calls == 3


async def test_pause_during_backoff_does_not_persist_the_revoked_reply(monkeypatch):
    from ctx_weft.core.control.tokens import PauseToken
    from ctx_weft.core.loop.signals import HitlPark
    from ctx_weft.core.loop.steps.act import _run_llm_turn
    from ctx_weft.protocols import MemoryEventType
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from tests.integration.test_interactive_task import _act_state_ctx

    class Flaky(MockLLMAdapter):
        calls = 0

        def complete(self, req, stream=True):
            self.calls += 1
            async def failed():
                yield LLMChunk(kind="token", text="revoked reply")
                raise LLMCallError("wire lost", retriable=True, outage=True)
            return failed()

    llm = Flaky(responses=[MockResponse(text="unused")])
    loop_state, ctx, _, _, memory = _act_state_ctx(False, llm)
    ctx.pause_token = PauseToken()

    async def pause(_delay, _token):
        ctx.pause_token.pause()
        await asyncio.Event().wait()  # Act must stop this pending read itself.

    monkeypatch.setattr(llm_gateway, "_sleep_cancellable", pause)
    prompt = loop_state.prompt_inputs
    with pytest.raises(HitlPark):
        await asyncio.wait_for(_run_llm_turn(loop_state, ctx, scripted_inputs(prompt), [], 1), timeout=2)
    records = await memory.recall_recent(loop_state.scope, [MemoryEventType.LLM_RESPONSE], 10, ctx.provider_ctx)
    assert records == []
    assert llm.calls == 1
