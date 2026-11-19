"""Usage survives stream exits without accepting output or counting a retry twice."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ctx_weft.core.control.tokens import CancelToken, PauseToken
from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.loop.signals import HitlPark, RoundDiscarded
from ctx_weft.core.loop.steps.act import _run_llm_turn
from ctx_weft.core.loop.steps.compact import _summarize
from ctx_weft.core.loop.steps.observe import ObserveStep
from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.protocols import LLMCallError, LLMChunk, LLMUsage, MemoryScope, load_records
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.protocols.events import EventOrigin, EventType, PersistenceUnavailableError
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from tests._event_helpers import all_events
from tests.assembler_testkit import scripted_inputs
from tests.integration.test_interactive_task import _act_state_ctx

ENTRIES = ("act", "observe", "compact", "recognize_intent")


class UsageClient(MockLLMAdapter):
    def __init__(self, *, error=None, stall=False, before_usage=None):
        super().__init__(responses=[])
        self.error = error
        self.stall = stall
        self.before_usage = before_usage
        self.waiting = asyncio.Event()
        self.closed = False

    async def complete(self, request, stream=True):
        try:
            if self.before_usage:
                self.before_usage()
            yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=0, completion_tokens=3))
            if self.stall:
                self.waiting.set()
                await asyncio.Event().wait()
            if self.error:
                raise self.error
            yield LLMChunk(kind="token", text="summary or answer")
        finally:
            self.closed = True


def setup(entry, client):
    state, ctx, task, _, _ = _act_state_ctx(False, client)
    state.origin = getattr(EventOrigin, f"LOOP_{entry.upper()}")
    ctx.pause_token, ctx.cancel_token = PauseToken(), CancelToken()
    ctx.config = RuntimeConfig(llm_self_heal_base_delay_sec=0, llm_self_heal_max_interval_sec=0)
    inputs = scripted_inputs(system="", messages=[], tools=[], purpose=entry)

    async def assemble(request):
        return inputs

    ctx.assembler.assemble = assemble
    task.title = ""
    state.extra["bound_capabilities"] = [ToolCapability(
        id="metadata", name="metadata", purposes=["recognize_intent"])]
    store = InMemoryEventStore()
    ctx.event_bus.attach_commit_gate(CommitGate(store))

    async def run():
        if entry == "act":
            return await _run_llm_turn(state, ctx, inputs, [], 1)
        if entry == "observe":
            return await ObserveStep()._run_react(
                state, ctx, inputs=inputs, max_rounds=1, terminal_tool_name="unused")
        if entry == "compact":
            return await _summarize(state, ctx, records=[])
        return await RecognizeIntentStep().execute(state, ctx)

    return state, ctx, store, run


def cost_events(events):
    return [event for event in events
            if event.type in {EventType.LLM_RESPONSE_FINISHED, EventType.LLM_RESPONSE_DISCARDED}
            and any(value is not None for value in event.payload["usage"].values())]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("error", [None, LLMCallError("invalid", retriable=False),
                                  LLMCallError("truncated", retriable=True), TimeoutError("wire timeout"),
                                  RuntimeError("stream decode failed")])
async def test_completed_or_failed_usage_has_one_existing_response_event(entry, error):
    client = UsageClient(error=error)
    state, _, store, run = setup(entry, client)
    if error and entry != "recognize_intent":
        with pytest.raises(type(error), match=str(error)):
            await run()
    else:
        await run()
    events = await all_events(store, state.session.id)
    cost, = cost_events(events)
    assert cost.type == (EventType.LLM_RESPONSE_DISCARDED if error else EventType.LLM_RESPONSE_FINISHED)
    assert cost.payload["usage"]["prompt_tokens"] == 0
    assert cost.payload["usage"]["total_tokens"] == 3
    assert cost.payload["usage"]["cache_read_tokens"] is None
    if error:
        assert cost.payload["usage_known"] is True
        assert cost.payload["will_retry"] is False
    assert client.closed


@pytest.mark.parametrize("entry", ENTRIES)
async def test_cancel_after_usage_keeps_cost_and_does_not_create_assistant_memory(entry):
    client = UsageClient(stall=True)
    state, ctx, store, run = setup(entry, client)
    pending = asyncio.create_task(run())
    await asyncio.wait_for(client.waiting.wait(), timeout=1)
    ctx.cancel_token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=1)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.type == EventType.LLM_RESPONSE_FINISHED
    assert cost.payload["finish_reason"] == "cancelled"
    assert cost.payload["usage"]["total_tokens"] == 3
    assert not cost.payload.get("memory_id")
    assert cost.payload["content"] == "" and cost.payload["tool_calls"] == []
    assert await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx) == []
    assert client.closed


@pytest.mark.parametrize("entry", ["act", "observe", "compact"])
async def test_pause_keeps_usage_with_existing_finished_event(entry):
    class PartialUsage(UsageClient):
        async def complete(self, request, stream=True):
            try:
                yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=0))
                yield LLMChunk(kind="usage", usage=LLMUsage(completion_tokens=3))
                self.waiting.set()
                await asyncio.Event().wait()
            finally:
                self.closed = True

    client = PartialUsage()
    state, ctx, store, run = setup(entry, client)
    pending = asyncio.create_task(run())
    await asyncio.wait_for(client.waiting.wait(), timeout=1)
    ctx.pause_token.pause()
    if entry != "act":
        # Foreground observe/compact stop their cancellable work this way.
        pending.cancel()
    with pytest.raises(HitlPark if entry == "act" else asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=1)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.type == EventType.LLM_RESPONSE_FINISHED
    assert cost.payload["finish_reason"] == "paused"
    assert cost.payload["usage"]["total_tokens"] == 3
    assert not cost.payload.get("memory_id")
    assert client.closed


async def test_observe_idle_timeout_keeps_usage_as_discarded():
    client = UsageClient(stall=True)
    state, ctx, store, run = setup("observe", client)
    ctx.config = replace(ctx.config, observe_llm_idle_timeout_sec=0.02)
    with pytest.raises(TimeoutError, match="no progress"):
        await asyncio.wait_for(run(), timeout=1)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.type == EventType.LLM_RESPONSE_DISCARDED
    assert cost.payload["reason"] == "timeout"
    assert cost.payload["usage_known"] is True
    assert cost.payload["usage"]["total_tokens"] == 3
    assert client.closed


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("exhausted", [False, True])
async def test_retry_keeps_each_attempt_usage_once(entry, exhausted):
    class Retrying(UsageClient):
        calls = 0

        async def complete(self, request, stream=True):
            self.calls += 1
            if self.calls == 1 or exhausted:
                yield LLMChunk(kind="usage", usage=LLMUsage(cache_read_tokens=0))
                raise LLMCallError("wire lost", outage=True)
            yield LLMChunk(kind="token", text="fresh summary or answer")
            yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=2, completion_tokens=3))

    client = Retrying()
    state, ctx, store, run = setup(entry, client)
    ctx.config = replace(ctx.config, llm_self_heal_max_attempts=2)
    if exhausted and entry != "recognize_intent":
        with pytest.raises(LLMCallError, match="wire lost"):
            await run()
    else:
        await run()
    costs = cost_events(await all_events(store, state.session.id))
    assert client.calls == 2 and len(costs) == 2
    assert len({event.payload["request_id"] for event in costs}) == 1
    discarded = [event for event in costs if event.type == EventType.LLM_RESPONSE_DISCARDED]
    assert [event.payload["will_retry"] for event in discarded] == ([True, False] if exhausted else [True])
    assert all(event.payload["usage"]["cache_read_tokens"] == 0 for event in discarded)
    assert all(event.payload["usage"]["total_tokens"] is None for event in discarded)
    if not exhausted:
        assert costs[-1].type == EventType.LLM_RESPONSE_FINISHED
        assert costs[-1].payload["usage"]["total_tokens"] == 5
        assert costs[-1].payload["usage"]["cache_read_tokens"] is None


@pytest.mark.parametrize("stop", ["pause", "cancel"])
async def test_unaccepted_usage_survives_discarding_a_provisional_round(stop):
    client = UsageClient()
    state, ctx, store, run = setup("act", client)
    client.before_usage = ctx.pause_token.pause if stop == "pause" else ctx.cancel_token.cancel
    ctx.task_manager = SimpleNamespace(is_round_open=lambda _: True)
    ctx.event_bus.begin_provisional(state.task.id)
    with pytest.raises(RoundDiscarded if stop == "pause" else asyncio.CancelledError):
        await run()
    ctx.event_bus.discard_provisional(state.task.id)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.payload["usage"]["total_tokens"] == 3
    assert cost.payload["finish_reason"] == ("paused" if stop == "pause" else "cancelled")
    assert not cost.payload.get("memory_id")
    assert client.closed


async def test_uncertain_interrupted_usage_commit_confirms_the_original_event_once():
    class LostReceipt(InMemoryEventStore):
        def __init__(self):
            super().__init__()
            self.batches = []

        async def append_batch(self, session_id, batch_id, events):
            self.batches.append((batch_id, tuple(event.id for event in events)))
            receipt = await super().append_batch(session_id, batch_id, events)
            if len(self.batches) == 1:
                raise ConnectionError("receipt lost after commit")
            return receipt

    client = UsageClient()
    state, ctx, _, run = setup("act", client)
    store = LostReceipt()
    ctx.event_bus.attach_commit_gate(CommitGate(store))
    client.before_usage = ctx.pause_token.pause
    ctx.event_bus.begin_provisional(state.task.id)
    seen = []

    async def collect(event):
        seen.append(event)

    ctx.event_bus.subscribe(None, collect)
    with pytest.raises(PersistenceUnavailableError):
        await run()
    assert seen == []
    ctx.event_bus.discard_provisional(state.task.id)
    await ctx.event_bus.confirm_pending_storage(state.session.id)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.payload["usage"]["total_tokens"] == 3
    assert store.batches == [(cost.id, (cost.id,))] * 2
    assert [event.id for event in seen] == [cost.id]


async def test_metadata_discard_commit_failure_does_not_submit_another_completion():
    class LostDiscardReceipt(InMemoryEventStore):
        def __init__(self):
            super().__init__()
            self.responses = []

        async def append_batch(self, session_id, batch_id, events):
            receipt = await super().append_batch(session_id, batch_id, events)
            for event in events:
                if event.type in {EventType.LLM_RESPONSE_DISCARDED, EventType.LLM_RESPONSE_FINISHED}:
                    self.responses.append(event)
                    if len(self.responses) == 1:
                        raise ConnectionError("discard receipt lost")
            return receipt

    client = UsageClient(error=LLMCallError("wire lost", outage=True))
    state, ctx, _, run = setup("recognize_intent", client)
    store = LostDiscardReceipt()
    ctx.event_bus.attach_commit_gate(CommitGate(store))
    with pytest.raises(PersistenceUnavailableError):
        await run()
    assert len(store.responses) == 1
    original = store.responses[0]
    assert original.type == EventType.LLM_RESPONSE_DISCARDED
    await ctx.event_bus.confirm_pending_storage(state.session.id)
    assert [event.id for event in store.responses] == [original.id, original.id]
    assert [event.id for event in cost_events(await all_events(store, state.session.id))] == [original.id]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("exit_kind", ["normal", "retry", "cancel", "timeout"])
async def test_partial_usage_updates_are_retained_on_each_exit(entry, exit_kind):
    class PartialUsage(UsageClient):
        calls = 0

        async def complete(self, request, stream=True):
            self.calls += 1
            if self.calls > 1:
                yield LLMChunk(kind="token", text="fresh response")
                yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=2, completion_tokens=3))
                return
            yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=11))
            yield LLMChunk(kind="usage", usage=LLMUsage(completion_tokens=2))
            if exit_kind == "cancel":
                ctx.cancel_token.cancel()
            yield LLMChunk(kind="usage", usage=LLMUsage(completion_tokens=5))
            yield LLMChunk(kind="usage", usage=LLMUsage())
            if exit_kind == "retry":
                raise LLMCallError("wire lost", outage=True)
            if exit_kind == "timeout":
                raise TimeoutError("wire timeout")
            yield LLMChunk(kind="token", text="completed response")

    state, ctx, store, run = setup(entry, PartialUsage())
    if exit_kind == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run()
    elif exit_kind == "timeout" and entry != "recognize_intent":
        with pytest.raises(TimeoutError):
            await run()
    else:
        await run()
    costs = cost_events(await all_events(store, state.session.id))
    assert len(costs) == (2 if exit_kind == "retry" else 1)
    assert costs[0].payload["usage"]["prompt_tokens"] == 11
    assert costs[0].payload["usage"]["completion_tokens"] == 5
    assert costs[0].payload["usage"]["total_tokens"] == 16
    if exit_kind == "retry":
        assert costs[0].type == EventType.LLM_RESPONSE_DISCARDED
        assert costs[1].payload["usage"]["total_tokens"] == 5
    elif exit_kind == "cancel":
        assert costs[0].payload["finish_reason"] == "cancelled"


@pytest.mark.parametrize("entry,stop", [(entry, "cancel") for entry in ENTRIES] + [("act", "pause")])
async def test_stop_during_retry_backoff_does_not_repeat_discarded_usage(monkeypatch, entry, stop):
    from ctx_weft.core.loop import llm_gateway

    client = UsageClient(error=LLMCallError("wire lost", outage=True))
    state, ctx, store, run = setup(entry, client)

    async def stop_backoff(delay, token):
        if stop == "pause":
            ctx.pause_token.pause()
            await asyncio.Event().wait()
        token.cancel()
        token.raise_if_cancelled()

    monkeypatch.setattr(llm_gateway, "_sleep_cancellable", stop_backoff)
    with pytest.raises(HitlPark if stop == "pause" else asyncio.CancelledError):
        await asyncio.wait_for(run(), 1)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.type == EventType.LLM_RESPONSE_DISCARDED
    assert cost.payload["usage"]["total_tokens"] == 3
    assert client.closed


@pytest.mark.parametrize("entry", ENTRIES)
async def test_cancel_during_normal_finished_commit_keeps_original_usage_once(entry):
    from ctx_weft.core.loop.driver import StepDriver

    class DelayedReceipt(InMemoryEventStore):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.responses = []

        async def append_batch(self, session_id, batch_id, events):
            receipt = await super().append_batch(session_id, batch_id, events)
            for event in events:
                if event.type == EventType.LLM_RESPONSE_FINISHED:
                    self.responses.append(event.id)
                    if len(self.responses) == 1:
                        self.entered.set()
                        await asyncio.Event().wait()
            return receipt

    state, ctx, _, run = setup(entry, UsageClient())
    store = DelayedReceipt()
    ctx.event_bus.attach_commit_gate(CommitGate(store))
    pending = asyncio.create_task(run())
    await asyncio.wait_for(store.entered.wait(), 1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, 1)
    await ctx.event_bus.confirm_pending_storage(state.session.id)
    if entry == "recognize_intent":
        await StepDriver.for_context(ctx).executions.confirm_metadata_commits(ctx.event_bus)
    cost, = cost_events(await all_events(store, state.session.id))
    assert cost.payload["usage"]["total_tokens"] == 3
    assert store.responses == [cost.id, cost.id]
