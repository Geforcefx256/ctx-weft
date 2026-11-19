"""Compaction barriers track failed/in-flight writes without hot history scans."""

import asyncio
from datetime import timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.driver import StepExecution
from ctx_weft.core.loop.steps.act import _assistant_memory_event
from ctx_weft.core.loop.steps.observe import ObserveStep, Verdict, capture_context
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import LLMUsage, MemoryAddress, MemoryScope, ProviderContext
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_assistant_memory_recovery import (
    NOW,
    SESSION,
    InlineAgentTemplateProvider,
    make_runtime,
)


def setup():
    runtime = make_runtime(agent_provider=InlineAgentTemplateProvider())
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)
    return runtime, memory


def turn(runtime, memory, agent="a", record_id=None, task_manager=None):
    address = MemoryAddress(session_id=SESSION.id, agent_id=agent, task_id=f"task-{agent}")
    state = SimpleNamespace(session=SESSION, scope=address, agent=SimpleNamespace(id=agent),
                            task=SimpleNamespace(id=address.task_id), extra={},
                            sequence_counter=0, origin="loop.act", run_id=f"run-{agent}")
    ctx = SimpleNamespace(memory=memory, event_bus=runtime.event_bus, task_manager=task_manager,
                          provider_ctx=ProviderContext(session_id=SESSION.id, tenant_id=SESSION.tenant_id,
                                                       task_id=address.task_id, agent_id=agent),
                          assistant_memory_checkpoint=runtime._assistant_memory_checkpoint,
                          execution=StepExecution())
    event, _ = _assistant_memory_event(address, "answer", [], LLMUsage(), 1,
                                        anchor=record_id or generate_id("asst"), timestamp=NOW)
    finished = Event(id=generate_id("evt"), run_id=state.run_id, sequence=1,
                     session_id=SESSION.id, tenant_id=SESSION.tenant_id,
                     task_id=address.task_id, agent_id=agent, timestamp=NOW,
                     type=EventType.LLM_RESPONSE_FINISHED, origin="loop.act",
                     payload={"memory_id": event.id, "content": "answer", "tool_calls": [],
                              "usage": {}, "turn": 1})
    return state, ctx, finished, event


def options(**overrides):
    return {"observe_id": generate_id("obs"),
            "segment_end_at": (NOW + timedelta(seconds=1)).isoformat(), **overrides}


async def fail_write(runtime, memory, monkeypatch, *args, **kwargs):
    values = turn(runtime, memory, *args, **kwargs)
    real = memory.ingest

    async def fail(event, ctx):
        raise OSError("memory unavailable")

    monkeypatch.setattr(memory, "ingest", fail)
    with pytest.raises(OSError):
        await runtime._persist_assistant_turn(*values)
    monkeypatch.setattr(memory, "ingest", real)
    return values


@pytest.mark.asyncio
async def test_successful_hot_writes_and_empty_checkpoints_do_no_read_io(monkeypatch):
    runtime, memory = setup()
    await runtime._ensure_assistant_memory(SESSION)

    async def unexpected(*args, **kwargs):
        raise AssertionError("hot assistant checks must not scan/read storage")

    monkeypatch.setattr(runtime.event_store, "read_range", unexpected)
    monkeypatch.setattr(runtime.event_store, "read_last_of_type", unexpected)
    monkeypatch.setattr(runtime.event_store, "committed_head", unexpected)
    monkeypatch.setattr(memory, "read_records", unexpected)
    for agent in ("a", "b", "a"):
        state, ctx, finished, event = turn(runtime, memory, agent)
        await runtime._persist_assistant_turn(state, ctx, finished, event)
        async with runtime._assistant_memory_checkpoint(SESSION, state.scope):
            pass
    assert not any(runtime._assistant_pending.values())


@pytest.mark.asyncio
async def test_observer_pins_repaired_reply_and_soft_delete_is_not_revived(monkeypatch):
    runtime, memory = setup()
    state, ctx, _, event = await fail_write(runtime, memory, monkeypatch)
    opts = options()
    await capture_context(state, ctx, opts)
    assert event.id in opts["context_record_ids"]
    assert not any(runtime._assistant_pending.values())
    await memory.fold([event.id], [], ctx.provider_ctx)
    async with runtime._assistant_memory_checkpoint(SESSION, state.scope):
        pass
    assert await memory.read_records([event.id], ctx.provider_ctx)
    assert not await memory.load_view_ids(state.scope, MemoryScope.TASK, ctx.provider_ctx)


@pytest.mark.asyncio
async def test_unrepaired_gap_blocks_observer_input_capture(monkeypatch):
    runtime, memory = setup()
    state, ctx, _, _ = await fail_write(runtime, memory, monkeypatch)

    async def unavailable(*args, **kwargs):
        raise OSError("still offline")

    monkeypatch.setattr(memory, "ingest", unavailable)
    monkeypatch.setattr(memory, "load_view_ids", lambda *a, **k: pytest.fail("read incomplete view"))
    opts = options()
    with pytest.raises(OSError):
        await capture_context(state, ctx, opts)
    assert "context_record_ids" not in opts
    assert any(runtime._assistant_pending.values())


@pytest.mark.asyncio
async def test_other_agent_finishes_and_captures_while_one_agent_write_is_blocked(monkeypatch):
    runtime, memory = setup()
    await runtime._ensure_assistant_memory(SESSION)
    a = turn(runtime, memory, "a")
    b = turn(runtime, memory, "b")
    entered, release = asyncio.Event(), asyncio.Event()
    real = memory.ingest

    async def blocked(event, ctx):
        if event.id == a[3].id:
            entered.set()
            await release.wait()
        return await real(event, ctx)

    monkeypatch.setattr(memory, "ingest", blocked)
    writer = asyncio.create_task(runtime._persist_assistant_turn(*a))
    waiting = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        # A's finished fact is durable, but its memory write has not completed.
        assert await runtime.event_store.committed_head(SESSION.id) > 0
        waiting = asyncio.create_task(capture_context(a[0], a[1], options()))
        await asyncio.sleep(0)
        assert not waiting.done()
        await asyncio.wait_for(runtime._persist_assistant_turn(*b), 2)
        selected = options()
        await asyncio.wait_for(capture_context(b[0], b[1], selected), 2)
        assert b[3].id in selected["context_record_ids"]
        assert not writer.done() and not waiting.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(writer, waiting), 2)
    finally:
        release.set()
        await asyncio.gather(writer, *([waiting] if waiting else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_writer_leaves_pending_for_the_next_checkpoint(monkeypatch):
    runtime, memory = setup()
    await runtime._ensure_assistant_memory(SESSION)
    values = turn(runtime, memory)
    entered = asyncio.Event()
    real = memory.ingest
    values[2].payload["usage"] = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}

    async def blocked(*args):
        entered.set()
        await asyncio.Future()

    monkeypatch.setattr(memory, "ingest", blocked)
    writer = asyncio.create_task(runtime._persist_assistant_turn(*values))
    await asyncio.wait_for(entered.wait(), 2)
    writer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writer
    assert any(runtime._assistant_pending.values())
    monkeypatch.setattr(memory, "ingest", real)
    async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
        assert await memory.read_records([values[3].id], values[1].provider_ctx)
    responses = [row.event for row in await runtime.event_store.read_range(SESSION.id)
                 if row.event.type == EventType.LLM_RESPONSE_FINISHED]
    assert len(responses) == 1 and responses[0].id == values[2].id
    assert responses[0].payload["usage"]["total_tokens"] == 5


@pytest.mark.parametrize("stage", ["initialization", "agent_lock"])
async def test_cancel_before_assistant_event_registration_keeps_only_usage(monkeypatch, stage):
    runtime, memory = setup()
    await runtime._ensure_assistant_memory(SESSION)
    state, ctx, finished, event = turn(runtime, memory)
    finished.payload["usage"] = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    entered = asyncio.Event()
    if stage == "initialization":
        async def blocked(*args):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(runtime, "_ensure_assistant_memory", blocked)
    lock = runtime._assistant_agent_lock(runtime._assistant_key(ctx.provider_ctx, event.address))
    if stage == "agent_lock":
        await lock.acquire()
    try:
        writer = asyncio.create_task(runtime._persist_assistant_turn(state, ctx, finished, event))
        if stage == "initialization":
            await asyncio.wait_for(entered.wait(), 1)
        else:
            await asyncio.sleep(0)
        assert not writer.done()
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(writer, 1)
        responses = [row.event for row in await runtime.event_store.read_range(SESSION.id)
                     if row.event.type == EventType.LLM_RESPONSE_FINISHED]
        response, = responses
        assert response.id == finished.id
        assert response.payload["usage"]["total_tokens"] == 5
        assert response.payload["finish_reason"] == "cancelled"
        assert response.payload["content"] == "" and not response.payload.get("memory_id")
        assert await memory.read_records([event.id], ctx.provider_ctx) == []
        assert not any(runtime._assistant_pending.values())
    finally:
        if stage == "agent_lock":
            lock.release()


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [False, True])
async def test_uncertain_event_commit_is_confirmed_before_any_repair(monkeypatch, committed):
    runtime, memory = setup()
    await runtime._ensure_assistant_memory(SESSION)
    values = turn(runtime, memory)
    real = runtime.event_bus.emit

    async def uncertain(event):
        if committed:
            await real(event)
        raise OSError("event commit receipt lost")

    monkeypatch.setattr(runtime.event_bus, "emit", uncertain)
    with pytest.raises(OSError):
        await runtime._persist_assistant_turn(*values)
    monkeypatch.setattr(runtime.event_bus, "emit", real)
    if committed:
        async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
            assert await memory.read_records([values[3].id], values[1].provider_ctx)
    else:
        with pytest.raises(RuntimeError, match="commit not confirmed"):
            async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
                pytest.fail("uncommitted fact must not authorize a compaction")
        assert not await memory.read_records([values[3].id], values[1].provider_ctx)
    # Explicit recovery confirms the same immutable single-event batch, including
    # the case where the first attempt never reached storage at all.
    await runtime._confirm_assistant_events(SESSION)
    async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
        assert await memory.read_records([values[3].id], values[1].provider_ctx)
    rows = await runtime.event_store.read_range(SESSION.id, include_types=(EventType.LLM_RESPONSE_FINISHED,))
    assert [row.event.id for row in rows] == [values[2].id]


@pytest.mark.asyncio
@pytest.mark.parametrize("discard", [False, True])
async def test_provisional_memory_is_pending_until_round_commit_or_discard(discard):
    runtime, memory = setup()
    await runtime._ensure_assistant_memory(SESSION)
    staged = []
    tm = SimpleNamespace(is_round_open=lambda task_id: True)

    def stage(task_id, provider, event, ctx):
        staged.append((provider, event, ctx))
        return True

    tm.stage_memory = stage
    values = turn(runtime, memory, task_manager=tm)
    task_id = values[0].task.id
    runtime.event_bus.begin_provisional(task_id)
    await runtime._persist_assistant_turn(*values)
    assert staged and any(runtime._assistant_pending.values())
    assert not await memory.read_records([values[3].id], values[1].provider_ctx)
    if discard:
        await runtime._revert_round(SESSION.id, task_id)
        runtime.event_bus.discard_provisional(task_id)
        assert not await runtime.event_store.committed_head(SESSION.id)
    else:
        with pytest.raises(RuntimeError, match="commit not confirmed"):
            async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
                pytest.fail("must not repair uncommitted staged memory")
        await runtime.event_bus.commit_provisional(task_id)
        await runtime._commit_round_writes(SESSION.id, task_id, staged)
        assert await memory.read_records([values[3].id], values[1].provider_ctx)
    assert not any(runtime._assistant_pending.values())


@pytest.mark.asyncio
async def test_old_observer_cannot_fold_a_reply_missing_from_its_pinned_input(monkeypatch):
    runtime, memory = setup()
    state, ctx, _, event = await fail_write(runtime, memory, monkeypatch)
    opts = options(context_record_ids=[])
    await capture_context(state, ctx, opts)
    assert opts["context_record_ids"] == []
    assert state.extra["observe_keep_raw"] is True
    await ObserveStep()._fold_retry_segment(
        state, ctx, Verdict(task_outcome="continue", act_recap="old incomplete recap"), [])
    assert event.id in await memory.load_view_ids(state.scope, MemoryScope.TASK, ctx.provider_ctx)


@pytest.mark.asyncio
async def test_compact_repairs_before_its_first_memory_read_and_stops_on_failure(monkeypatch):
    from tests.compaction_testkit import escalating_compact

    runtime, memory = setup()
    state, ctx, _, event = await fail_write(runtime, memory, monkeypatch)
    state.agent.loop_config = SimpleNamespace(compact_target_ratio=0.5, compact_token_ratio=0.8,
                                              compact_keep_last=100, collapse_keep_last=100)
    state.resolved_model = SimpleNamespace(effective_input_limit=1000)
    real = memory.load_view_ids
    reads = []

    async def read(*args, **kwargs):
        assert await memory.read_records([event.id], ctx.provider_ctx)
        reads.append(True)
        return await real(*args, **kwargs)

    monkeypatch.setattr(memory, "load_view_ids", read)
    await escalating_compact(state, ctx, token_estimate=1000, force=True)
    assert reads
    await fail_write(runtime, memory, monkeypatch)
    reads.clear()

    async def fail(*args):
        raise OSError("cannot repair")

    monkeypatch.setattr(memory, "ingest", fail)
    with pytest.raises(OSError):
        await escalating_compact(state, ctx, token_estimate=1000, force=True)
    assert not reads


@pytest.mark.asyncio
async def test_actual_compact_summary_includes_repaired_reply(monkeypatch):
    from tests import compaction_testkit as compact
    from ctx_weft.protocols import load_records

    runtime, memory = setup()
    state, ctx, _, event = await fail_write(runtime, memory, monkeypatch)
    state.agent.loop_config = SimpleNamespace(compact_target_ratio=0.5, compact_token_ratio=0.8,
                                              compact_keep_last=0, collapse_keep_last=0)
    state.resolved_model = SimpleNamespace(effective_input_limit=1000)
    seen = []
    ctx.llm = SimpleNamespace(tokenizer=SimpleNamespace(count=len))

    async def summary(state, ctx, scope):
        records = await load_records(memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
        assert any(r.id == event.id and r.content == "answer" for r in records)
        seen.append(True)
        return "summary including answer"

    compact.patch_summary(monkeypatch, summary)
    await compact.escalating_compact(state, ctx, token_estimate=1000, force=True)
    assert seen
    active = await memory.load_view_ids(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    assert active and event.id not in active
    async with runtime._assistant_memory_checkpoint(SESSION, state.scope):
        pass
    assert event.id not in await memory.load_view_ids(state.scope, MemoryScope.TASK, ctx.provider_ctx)


@pytest.mark.asyncio
async def test_pending_lost_memory_ack_is_confirmed_without_repeating_the_write(monkeypatch):
    runtime, memory = setup()
    values = turn(runtime, memory)
    real = memory.ingest

    async def lost_ack(event, ctx):
        await real(event, ctx)
        raise OSError("memory commit acknowledgement lost")

    monkeypatch.setattr(memory, "ingest", lost_ack)
    with pytest.raises(OSError):
        await runtime._persist_assistant_turn(*values)

    async def unexpected(*args):
        pytest.fail("already persisted record must not be written again")

    monkeypatch.setattr(memory, "ingest", unexpected)
    async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
        pass
    assert not any(runtime._assistant_pending.values())


@pytest.mark.asyncio
async def test_failed_initialization_never_enables_empty_pending_fast_path(monkeypatch):
    from tests.integration.test_assistant_memory_recovery import finished

    runtime, memory = setup()
    await finished(runtime.event_store)
    state, _, _, _ = turn(runtime, memory)
    real = memory.ingest

    async def fail(*args):
        raise OSError("bootstrap repair unavailable")

    monkeypatch.setattr(memory, "ingest", fail)
    with pytest.raises(OSError):
        async with runtime._assistant_memory_checkpoint(SESSION, state.scope):
            pytest.fail("uninitialized empty pending table is not proof")
    assert not runtime._assistant_memory_ready
    monkeypatch.setattr(memory, "ingest", real)
    async with runtime._assistant_memory_checkpoint(SESSION, state.scope):
        assert await memory.read_records(["reply"], ProviderContext(session_id="s", tenant_id="acme"))


@pytest.mark.asyncio
async def test_snapshot_proof_stops_at_pending_gap_and_advances_after_repair(monkeypatch):
    from tests.integration.test_assistant_memory_recovery import snapshot

    runtime, memory = setup()
    state, _, _, _ = await fail_write(runtime, memory, monkeypatch)
    before = await snapshot(runtime.event_store, memory)
    assert before.assistant_memory_verified_position == 0
    async with runtime._assistant_memory_checkpoint(SESSION, state.scope):
        pass
    after = await snapshot(runtime.event_store, memory)
    assert after.assistant_memory_verified_position == after.last_commit_position


@pytest.mark.asyncio
async def test_changing_memory_provider_requires_a_new_history_baseline():
    runtime, memory = setup()
    values = turn(runtime, memory)
    await runtime._persist_assistant_turn(*values)
    replacement = InMemoryMemoryProvider()
    runtime.providers.register_memory(replacement)
    async with runtime._assistant_memory_checkpoint(SESSION, values[0].scope):
        assert await replacement.read_records([values[3].id], values[1].provider_ctx)


@pytest.mark.asyncio
async def test_staged_reply_commits_before_park_observer_capture():
    from ctx_weft.core.loop.steps.act import _cold_park

    runtime, memory = setup()
    staged = []
    opened = True
    tm = SimpleNamespace(is_round_open=lambda task_id: opened)

    def stage(task_id, provider, event, ctx):
        staged.append((provider, event, ctx))
        return True

    async def commit(task_id):
        nonlocal opened
        await runtime.event_bus.commit_provisional(task_id)
        await runtime._commit_round_writes(SESSION.id, task_id, staged)
        opened = False

    class ParkReached(Exception):
        pass

    tm.stage_memory, tm.commit_round = stage, commit
    state, ctx, finished, event = turn(runtime, memory, task_manager=tm)
    runtime.event_bus.begin_provisional(state.task.id)
    await runtime._persist_assistant_turn(state, ctx, finished, event)

    async def open_hitl(ask, **kwargs):
        assert not opened
        assert event.id in kwargs["observe_request"]["context_record_ids"]
        assert await memory.read_records([event.id], ctx.provider_ctx)
        raise ParkReached

    ctx.hitl = SimpleNamespace(open=open_hitl)
    with pytest.raises(ParkReached):
        await asyncio.wait_for(
            _cold_park(state, ctx, "wait", unattended=False, observe_boundary="plain_text"), 2)
    assert not any(runtime._assistant_pending.values())
