"""Real Act/Observe boundaries publish typed, source-independent fork snapshots."""

import asyncio
from datetime import UTC, datetime

import pytest

from ctx_weft import ForkCheckpointUnavailableError, RuntimeShutdownError
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.loop.task_memory_fold import build_finish_slots
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    ToolCall,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit.test_session_memory_seed import seed_records


class ReplyLLM(MockLLMAdapter):
    def __init__(self, verdicts=("success",)):
        super().__init__(responses=[])
        self.verdicts = verdicts
        self.observations = 0
        self.actor_requests = []

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__report_observation" in names:
            self.observations += 1
            verdict = self.verdicts[min(self.observations - 1, len(self.verdicts) - 1)]
            return self._stream(
                MockResponse(
                    tool_calls=[
                        ToolCall(
                            id=f"observe-{self.observations}",
                            name="control__report_observation",
                            arguments={
                                "task_status": verdict,
                                "act_recap": f"recap {self.observations}",
                                "task_summary": f"summary {self.observations}",
                            },
                        )
                    ]
                ),
                request,
            )
        self.actor_requests.append(request)
        return self._stream(MockResponse(text=f"answer {len(self.actor_requests)}"), request)


def core_with(llm, store=None, memory=None):
    agents = InlineAgentTemplateProvider()
    agents.register(make_echo_template())
    core = make_runtime(agent_provider=agents, llm=llm, event_store=store)
    core.providers.register_memory(memory or InMemoryMemoryProvider())
    return core


async def wait_for(predicate):
    async with asyncio.timeout(6):
        while True:
            result = await predicate()
            if result:
                return result
            await asyncio.sleep(0.01)


async def applied_rows(core, sid):
    return [
        r
        for r in await core.event_store.read_range(sid, include_types=(EventType.OBSERVE_APPLIED,))
        if r.event.payload.get("fork_checkpoint")
    ]


async def start(core, *, unattended=False):
    container = await core.create_session(template_id="agent:tpl_echo", session_id="source")
    ctx = ProviderContext(session_id="source", agent_id=container.root_agent_id)
    address = MemoryAddress(session_id="source", agent_id=container.root_agent_id)
    memory = core.providers.get_memory()
    await memory.ingest(
        MemoryEvent(
            id="experience",
            kind=MemoryKind.SUMMARY,
            scope=MemoryScope.AGENT,
            address=address,
            content="prior experience",
            role="user",
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        ctx,
    )
    slots = build_finish_slots(
        scope=address,
        task_id="past-task",
        parent_task_id=None,
        title="past",
        outcome="success",
        act_recap="prior recap",
        task_summary="prior summary",
        final_reply="prior reply",
        base=datetime(2026, 1, 2, tzinfo=UTC),
        tool_call_id="past-group",
    )
    for index, slot in enumerate(slots):
        slot.id = f"past-{index}"
        slot.metadata["finish_policy"] = {
            "slot_count": len(slots),
            "capsule_state": "full",
            "outcome": "success",
            "preserve_raw": True,
            "has_summary": True,
        }
    await memory.fold([], slots, ctx)
    turn = await core.dispatch_task(
        "source",
        "private current task input",
        agent_id=container.root_agent_id,
        unattended=unattended,
    )
    return turn


@pytest.mark.asyncio
@pytest.mark.parametrize("unattended", [False, True])
async def test_reply_binding_and_applied_gate_preserve_agent_types_groups(monkeypatch, unattended):
    llm = ReplyLLM()
    core = core_with(llm)
    reached, release = asyncio.Event(), asyncio.Event()
    original = core.event_store.append_batch

    async def pause(sid, bid, events):
        if events[0].type == EventType.OBSERVE_APPLIED and events[0].payload.get("fork_checkpoint"):
            reached.set()
            await release.wait()
        return await original(sid, bid, events)

    monkeypatch.setattr(core.event_store, "append_batch", pause)
    try:
        turn = await start(core, unattended=unattended)
        await asyncio.wait_for(reached.wait(), 6)
        prepared = (
            await core.event_store.read_last_of_type(
                "source", EventType.REPLY_FORK_CHECKPOINT_PREPARED
            )
        ).event
        anchor = prepared.payload["fork_checkpoint"]["reply_anchor"]
        pending = (await rebuild_view(core.event_store, "source")).pending_observe[
            prepared.payload["observe_id"]
        ]
        assert pending["reply_anchor"] == anchor
        with pytest.raises(ForkCheckpointUnavailableError):
            await core.fork_session_from_response(
                source_session_id="source",
                response_event_id=anchor["response_event_id"],
                session_id="branch",
            )
        assert await core.event_store.committed_head("branch") == 0
        # This AGENT write is later than the frozen boundary and must stay out.
        await core.providers.get_memory().ingest(
            MemoryEvent(
                id="future",
                kind=MemoryKind.SUMMARY,
                scope=MemoryScope.AGENT,
                address=MemoryAddress(session_id="source", agent_id=turn.agent_id),
                role="user",
                content="future secret",
                timestamp=datetime.now(UTC),
            ),
            ProviderContext(session_id="source"),
        )
        release.set()
        await wait_for(lambda: applied_rows(core, "source"))
        result = await core.fork_session_from_response(
            source_session_id="source",
            response_event_id=anchor["response_event_id"],
            session_id="branch",
        )
        records = await seed_records(core, result)
        assert any(
            r.kind == MemoryKind.SUMMARY and r.content == "prior experience" for r in records
        )
        assert not any(
            "private current task input" in str(r.content) or "future secret" in str(r.content)
            for r in records
        )
        assert all(r.scope == MemoryScope.AGENT for r in records)
        historical = [r for r in records if r.metadata.get("origin_task_id") == "past-task"]
        assert {r.metadata["task_memory"]["kind"] for r in historical} == {
            "recap",
            "finish",
            "final_reply",
        }
        assert len({r.metadata["task_memory"]["group_id"] for r in historical}) == 1
        assert historical[0].metadata["task_memory"]["group_id"] != "past-group"
        assert sum(r.role == "assistant" and r.content.strip() == "answer 1" for r in records) == 1
    finally:
        release.set()
        await core.close()


@pytest.mark.asyncio
async def test_continue_publishes_the_first_reply_checkpoint_without_later_reply():
    llm = ReplyLLM(("continue", "waiting"))
    core = core_with(llm)
    try:
        await start(core)

        async def two():
            rows = await applied_rows(core, "source")
            return rows if len(rows) == 2 else None

        rows = await wait_for(two)
        assert (
            rows[0].event.payload["fork_checkpoint"]["observe_id"]
            != rows[1].event.payload["fork_checkpoint"]["observe_id"]
        )
        first = rows[0].event.payload["fork_checkpoint"]["reply_anchor"]["response_event_id"]
        result = await core.fork_session_from_response(
            source_session_id="source", response_event_id=first, session_id="branch"
        )
        text = "\n".join(str(r.content) for r in await seed_records(core, result))
        assert "answer 1" in text and "answer 2" not in text
        assert len(llm.actor_requests) == 2
    finally:
        await core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unattended", [False, True])
@pytest.mark.parametrize(
    "failed_type", [EventType.REPLY_FORK_CHECKPOINT_PREPARED, EventType.OBSERVE_APPLIED]
)
async def test_checkpoint_survives_cold_recovery_without_reasking(
    monkeypatch, unattended, failed_type
):
    from ctx_weft.core.control.startup_view import rebuild_startup_view
    from tests._snapshot_helpers import seed_snapshot

    llm = ReplyLLM()
    core = core_with(llm)
    failed = asyncio.Event()
    attempted = []
    original_append = core.event_store.append_batch

    def denied(event):
        if event.type == failed_type and event.payload.get("fork_checkpoint"):
            attempted.append(event)
            return True
        return False

    async def reject_batch(sid, bid, events):
        if denied(events[0]):
            failed.set()
            raise OSError("Checkpoint event was not submitted")
        return await original_append(sid, bid, events)

    monkeypatch.setattr(core.event_store, "append_batch", reject_batch)
    await start(core, unattended=unattended)
    await asyncio.wait_for(failed.wait(), 6)

    async def no_dispatch():
        return (
            not core._task_managers["source"]._dispatch_tasks
            and not core._task_managers["source"]._background_asyncio_tasks
        )

    await wait_for(no_dispatch)
    view = await rebuild_view(core.event_store, "source")
    (info,) = view.pending_observe.values()
    checkpoint = attempted[0].payload["fork_checkpoint"]
    assert (info.get("fork_checkpoint") == checkpoint) == (failed_type == EventType.OBSERVE_APPLIED)
    assert info["reply_anchor"] == checkpoint["reply_anchor"]
    await seed_snapshot(
        core.event_store, "source", view, cut=await core.event_store.committed_head("source")
    )
    startup = await rebuild_startup_view(
        core.event_store, "source", through_position=await core.event_store.committed_head("source")
    )
    assert startup.pending_observe[info["observe_id"]].get("fork_checkpoint") == info.get(
        "fork_checkpoint"
    )
    with pytest.raises(RuntimeShutdownError):
        await core.close()
    monkeypatch.undo()
    # A new process must use Completed and the prepared fixed IDs, even after
    # unrelated AGENT memory appears. No observation or actor is asked again.
    memory = core.providers.get_memory()
    await memory.ingest(
        MemoryEvent(
            id="late",
            kind=MemoryKind.SUMMARY,
            scope=MemoryScope.AGENT,
            address=MemoryAddress(session_id="source", agent_id=checkpoint["agent_id"]),
            role="user",
            content="late source data",
            timestamp=datetime.now(UTC),
        ),
        ProviderContext(session_id="source"),
    )
    cold_llm = ReplyLLM()
    cold = core_with(cold_llm, core.event_store, memory)
    try:
        with pytest.raises(ForkCheckpointUnavailableError):
            await cold.fork_session_from_response(
                source_session_id="source",
                response_event_id=checkpoint["reply_anchor"]["response_event_id"],
                session_id="branch",
            )
        assert not cold._task_managers
        await cold.recover_session("source")
        rows = await wait_for(lambda: applied_rows(cold, "source"))
        assert rows[0].event.payload["fork_checkpoint"] == checkpoint
        assert not cold_llm.actor_requests and cold_llm.observations == 0
        result = await cold.fork_session_from_response(
            source_session_id="source",
            response_event_id=checkpoint["reply_anchor"]["response_event_id"],
            session_id="branch",
        )
        assert all(
            "late source data" not in str(r.content) for r in await seed_records(cold, result)
        )
    finally:
        await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unattended", [False, True])
@pytest.mark.parametrize(
    "failed_type", [EventType.REPLY_FORK_CHECKPOINT_PREPARED, EventType.OBSERVE_APPLIED]
)
async def test_unknown_checkpoint_receipt_confirms_original_batch(
    monkeypatch, unattended, failed_type
):
    from copy import deepcopy

    llm = ReplyLLM()
    core = core_with(llm)
    original = core.event_store.append_batch
    lost = asyncio.Event()
    attempts = []

    async def ambiguous(sid, bid, events):
        receipt = await original(sid, bid, events)
        if events[0].type == failed_type and (
            failed_type == EventType.REPLY_FORK_CHECKPOINT_PREPARED
            or events[0].payload.get("fork_checkpoint")
        ):
            attempts.append((bid, deepcopy(events)))
            if len(attempts) == 1:
                lost.set()
                raise RuntimeError("checkpoint receipt lost after commit")
        return receipt

    monkeypatch.setattr(core.event_store, "append_batch", ambiguous)
    try:
        await start(core, unattended=unattended)
        await asyncio.wait_for(lost.wait(), 6)
        async with asyncio.timeout(6):
            await core.recover_storage("source")
        rows = await wait_for(lambda: applied_rows(core, "source"))
        assert len(rows) == 1
        assert all(attempt == attempts[0] for attempt in attempts)
        assert len(llm.actor_requests) == llm.observations == 1
        cp = rows[0].event.payload["fork_checkpoint"]
        result = await core.fork_session_from_response(
            source_session_id="source",
            response_event_id=cp["reply_anchor"]["response_event_id"],
            session_id="branch",
        )
        assert (
            sum(str(r.content).strip() == "answer 1" for r in await seed_records(core, result)) == 1
        )
    finally:
        monkeypatch.undo()
        await core.close()


@pytest.mark.asyncio
async def test_old_checkpoint_survives_agent_memory_folding_and_target_is_organizable():
    llm = ReplyLLM()
    core = core_with(llm)
    try:
        turn = await start(core)
        (row,) = await wait_for(lambda: applied_rows(core, "source"))
        cp = row.event.payload["fork_checkpoint"]
        memory = core.providers.get_memory()
        ctx = ProviderContext(session_id="source", agent_id=turn.agent_id)
        await memory.fold(
            cp["agent_memory_ids"],
            [
                MemoryEvent(
                    id="later-summary",
                    kind=MemoryKind.SUMMARY,
                    scope=MemoryScope.AGENT,
                    address=MemoryAddress(session_id="source", agent_id=turn.agent_id),
                    role="user",
                    content="newer agent summary",
                    timestamp=datetime.now(UTC),
                )
            ],
            ctx,
        )
        result = await core.fork_session_from_response(
            source_session_id="source",
            response_event_id=cp["reply_anchor"]["response_event_id"],
            session_id="branch",
        )
        records = await seed_records(core, result)
        assert any(
            r.content == "prior experience" and r.kind == MemoryKind.SUMMARY for r in records
        )
        assert not any(r.content == "newer agent summary" for r in records)
        candidates = await memory.recall_for_organize(
            ProviderContext(session_id="branch", agent_id=result.root_agent_id),
            operation="fold_history",
            keep_last=0,
        )
        assert candidates
        assert any(r.metadata.get("origin_task_id") == "past-task" for r in candidates)
    finally:
        await core.close()
