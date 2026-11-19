"""Forked containers own their frozen memory and repair it without executing work."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft import MemorySeedConflictError, SessionForkResult
from ctx_weft.core.assembler.context import PromptContext
from ctx_weft.core.control.reducers import deserialize_view, rebuild_view, serialize_view
from ctx_weft.core.control.startup_view import rebuild_startup_view
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.llm import ModelCallConfig, serialize_call_config
from tests._snapshot_helpers import seed_snapshot
from tests.unit.test_runtime_initialization_queries import container, runtime


async def source(core, *, tools=False, inline=False):
    handle = await container(core)
    ctx = ProviderContext(session_id=handle.session_id, agent_id=handle.root_agent_id)
    address = MemoryAddress(
        session_id=handle.session_id, agent_id=handle.root_agent_id, task_id="task"
    )
    start = datetime(2026, 11, 18, tzinfo=UTC)
    memories = [
        MemoryEvent(
            id="source-user",
            kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.AGENT,
            address=address,
            content="old user input",
            role="user",
            timestamp=start,
        )
    ]
    if tools:
        memories.extend(
            [
                MemoryEvent(
                    id="source-call",
                    kind=MemoryKind.CONVERSATION_TURN,
                    scope=MemoryScope.AGENT,
                    address=address,
                    content="looking it up",
                    role="assistant",
                    timestamp=start + timedelta(seconds=1),
                    metadata={
                        "tool_calls": [
                            {"id": "call-source", "name": "search", "arguments": {"q": "x"}}
                        ]
                    },
                ),
                MemoryEvent(
                    id="source-result",
                    kind=MemoryKind.CONVERSATION_TURN,
                    scope=MemoryScope.AGENT,
                    address=address,
                    content="old tool result",
                    role="tool",
                    timestamp=start + timedelta(seconds=2),
                    metadata={"tool_call_id": "call-source"},
                ),
            ]
        )
    for memory in memories:
        await core.providers.get_memory().ingest(memory, ctx)
    inputs = PromptContext(
        template_id="agent:tpl_echo",
        scope={
            "session_id": handle.session_id,
            "agent_id": handle.root_agent_id,
            "task_id": "task",
        },
        task={"id": "task", "user_prompt_in_memory": True},
        context_limit=1,
        memory_blocks=[{"kind": "history", "memory_id": m.id} for m in memories],
    )
    if inline:
        inputs.appended_messages.append(
            {"role": "user", "content": "inline user input", "tool_calls": [], "tool_call_id": None}
        )
    prompt = new_event(
        EventType.LLM_PROMPT_SENT,
        session_id=handle.session_id,
        tenant_id="default",
        agent_id=handle.root_agent_id,
        task_id="task",
        run_id="run",
        origin="loop.act",
        payload={
            **inputs.to_payload(),
            "request_id": "request",
            "llm_config": serialize_call_config(ModelCallConfig(reasoning_effort="high")),
        },
    )
    response = new_event(
        EventType.LLM_RESPONSE_FINISHED,
        session_id=handle.session_id,
        tenant_id="default",
        agent_id=handle.root_agent_id,
        task_id="task",
        run_id="run",
        origin="loop.act",
        payload={
            "request_id": "request",
            "content": "selected answer",
            "memory_id": "source-answer",
            "tool_calls": [],
            "llm_account": "historic-account",
            "llm_model": "historic-model",
        },
    )
    await core.event_bus.emit_batch((prompt, response))
    await publish_checkpoint(core, response, [m.id for m in memories])
    return handle, prompt, response


async def publish_checkpoint(core, response, ids, *, applied=True):
    """Commit the Observe protocol for storage-focused seed tests."""
    observe_id = "obs:" + response.id
    anchor = {"response_event_id": response.id, "response_memory_id": response.payload["memory_id"]}
    checkpoint = {
        "version": 1,
        "observe_id": observe_id,
        "reply_anchor": anchor,
        "agent_id": response.agent_id,
        "task_id": response.task_id,
        "agent_memory_ids": list(ids),
    }

    def event(kind, payload):
        return new_event(
            kind,
            session_id=response.session_id,
            tenant_id=response.tenant_id,
            agent_id=response.agent_id,
            task_id=response.task_id,
            origin="loop.observe",
            payload={
                "task_id": response.task_id,
                "agent_id": response.agent_id,
                "observe_id": observe_id,
                **payload,
            },
        )

    events = (
        event(EventType.OBSERVE_REQUESTED, {"reply_anchor": anchor, "fork_base_record_ids": ids}),
        event(EventType.OBSERVE_COMPLETED, {"action": "recap", "accepted": True}),
        event(EventType.REPLY_FORK_CHECKPOINT_PREPARED, {"fork_checkpoint": checkpoint}),
    )
    if applied:
        events += (event(EventType.OBSERVE_APPLIED, {"fork_checkpoint": checkpoint}),)
    await core.event_bus.emit_batch(events)
    return checkpoint


async def fork(core, response, **kwargs):
    return await core.fork_session_from_response(
        source_session_id=response.session_id,
        response_event_id=response.id,
        session_id="branch",
        **kwargs,
    )


async def seed_records(core, result):
    memory = core.providers.get_memory()
    ctx = ProviderContext(session_id=result.session_id, agent_id=result.root_agent_id)
    ids = await memory.load_view_ids(
        MemoryAddress(session_id=ctx.session_id, agent_id=ctx.agent_id), MemoryScope.AGENT, ctx
    )
    return await memory.read_records(ids, ctx)


@pytest.mark.asyncio
async def test_copies_saved_candidates_superseded_and_inline_without_running(monkeypatch):
    core = runtime()
    _, prompt, response = await source(core, inline=True)
    memory = core.providers.get_memory()
    await memory.fold(["source-user"], [], ProviderContext(session_id="s1"))

    async def no_recall(*args, **kwargs):
        pytest.fail("fork reselected the current memory view")

    monkeypatch.setattr(memory, "load_view_ids", no_recall)
    result = await fork(core, response)
    assert isinstance(result, SessionForkResult)
    assert result.prompt_event_id == prompt.id and result.record_count == 2
    monkeypatch.undo()
    records = await seed_records(core, result)
    assert [r.content for r in records] == [
        "old user input",
        "selected answer",
    ]
    assert all(r.scope == MemoryScope.AGENT and r.address.task_id is None for r in records)
    assert all(r.id != "source-user" for r in records)
    state = (await core.get_session(result.session_id)).data.state
    assert state.root_agent.state.status == "idle" and state.root_task is None
    assert not (await core.list_tasks(result.session_id))
    detail = (await core.get_agent(result.root_agent_id, session_id=result.session_id)).data
    assert detail.llm_model == "historic-model" and detail.llm_config.reasoning_effort == "high"
    rows = await core.event_store.read_range(result.session_id)
    assert [r.event.type for r in rows] == [
        EventType.SESSION_CREATED,
        EventType.AGENT_INSTANTIATED,
        EventType.AGENT_MEMORY_SEED_REQUESTED,
        EventType.AGENT_MEMORY_SEED_APPLIED,
    ]
    assert await fork(core, response) == result
    assert len(await seed_records(core, result)) == 2
    await core.close()


@pytest.mark.asyncio
async def test_tool_pair_remains_history_with_new_link_identity():
    core = runtime()
    _, _, response = await source(core, tools=True)
    result = await fork(core, response)
    rows = await seed_records(core, result)
    calls = [r for r in rows if r.metadata.get("tool_calls")]
    results = [r for r in rows if r.role == "tool"]
    assert len(calls) == len(results) == 1
    assert calls[0].metadata["tool_calls"][0]["id"] == results[0].metadata["tool_call_id"]
    assert results[0].metadata["tool_call_id"] != "call-source"
    assert not await core.list_tasks("branch")
    await core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["before_fold", "after_fold", "before_applied", "after_applied"]
)
async def test_failure_and_cold_retry_use_target_owned_plan(monkeypatch, failure):
    core = runtime()
    _, _, response = await source(core)
    memory = core.providers.get_memory()
    if failure in {"before_fold", "after_fold"}:
        original = memory.fold

        async def broken(*args, **kwargs):
            if failure == "after_fold":
                await original(*args, **kwargs)
            raise OSError("lost memory receipt")

        monkeypatch.setattr(memory, "fold", broken)
    else:
        original = core.event_bus.emit_batch

        async def broken(events, **kwargs):
            if events[0].type == EventType.AGENT_MEMORY_SEED_APPLIED:
                if failure == "after_applied":
                    await original(events, **kwargs)
                raise OSError("lost applied receipt")
            return await original(events, **kwargs)

        monkeypatch.setattr(core.event_bus, "emit_batch", broken)
    if failure == "after_fold":
        # A successful readback converts an uncertain write into confirmed success.
        await fork(core, response)
    else:
        with pytest.raises(OSError):
            await fork(core, response)
    monkeypatch.undo()
    await core.close()
    cold = runtime(core.event_store)
    # New memory instance models restoring from the target events alone.
    original_read = cold.event_store.read_range

    async def target_only(sid, **kwargs):
        assert sid != "s1", "retry read the deleted source session"
        return await original_read(sid, **kwargs)

    monkeypatch.setattr(cold.event_store, "read_range", target_only)
    result = await fork(cold, response)
    assert [r.content for r in await seed_records(cold, result)] == [
        "old user input",
        "selected answer",
    ]
    rows = await cold.event_store.read_range("branch")
    assert sum(r.event.type == EventType.AGENT_MEMORY_SEED_APPLIED for r in rows) == 1
    await cold.close()


@pytest.mark.asyncio
async def test_pending_seed_survives_snapshot_and_queries_do_not_repair(monkeypatch):
    core = runtime()
    _, _, response = await source(core)

    async def broken(*args, **kwargs):
        raise OSError("memory offline")

    monkeypatch.setattr(core.providers.get_memory(), "fold", broken)
    with pytest.raises(OSError):
        await fork(core, response)
    monkeypatch.undo()
    view = await rebuild_view(core.event_store, "branch")
    assert view.sessions["branch"].context_seed["applied_event_id"] is None
    assert (
        deserialize_view(serialize_view(view)).sessions["branch"].context_seed
        == view.sessions["branch"].context_seed
    )
    await seed_snapshot(
        core.event_store, "branch", view, cut=await core.event_store.committed_head("branch")
    )
    startup = await rebuild_startup_view(
        core.event_store, "branch", through_position=await core.event_store.committed_head("branch")
    )
    assert startup.sessions["branch"].context_seed == view.sessions["branch"].context_seed
    cold = runtime(core.event_store)
    ctx = ProviderContext(session_id="branch")
    ids = [
        item["id"]
        for item in (
            await core.event_store.read_last_of_type(
                "branch", EventType.AGENT_MEMORY_SEED_REQUESTED
            )
        ).event.payload["records"]
    ]
    assert (await cold.get_session("branch")).loaded is False
    await cold.initialize("branch")
    assert await cold.providers.get_memory().read_records(ids, ctx) == []
    await cold.rebuild_session("branch", strict=True)
    assert len(await cold.providers.get_memory().read_records(ids, ctx)) == 2
    await cold.close()
    await core.close()


@pytest.mark.asyncio
async def test_target_request_conflict_and_concurrent_retries():
    import asyncio

    core = runtime()
    _, _, response = await source(core)
    a, b = await asyncio.gather(fork(core, response), fork(core, response))
    assert a == b
    with pytest.raises(MemorySeedConflictError):
        await fork(core, response, llm_model="different")
    await container(core, "ordinary")
    with pytest.raises(MemorySeedConflictError):
        await core.fork_session_from_response(
            source_session_id="s1", response_event_id=response.id, session_id="ordinary"
        )
    await core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["origin", "tools", "missing", "scope"])
async def test_invalid_source_does_not_create_target(bad):
    core = runtime()
    _, prompt, response = await source(core)
    # Build a second committed anchor rather than mutating the persisted event.
    bad_prompt = deepcopy(prompt)
    bad_prompt.id += "bad"
    bad_prompt.payload["request_id"] = "bad"
    bad_response = deepcopy(response)
    bad_response.id += "bad"
    bad_response.payload["request_id"] = "bad"
    if bad == "origin":
        bad_response.origin = "loop.observe"
    elif bad == "tools":
        bad_response.payload["tool_calls"] = [{"id": "pending"}]
    elif bad == "missing":
        bad_prompt.payload["memory_blocks"][0]["memory_id"] = "gone"
    else:
        bad_prompt.payload["scope"]["session_id"] = "other"
    await core.event_bus.emit_batch((bad_prompt, bad_response))
    with pytest.raises(ValueError):
        await fork(core, bad_response)
    assert await core.event_store.committed_head("branch") == 0
    await core.close()


@pytest.mark.asyncio
async def test_unknown_creation_receipt_reuses_original_batch(monkeypatch):
    from ctx_weft.protocols.events import PersistenceUnavailableError

    core = runtime()
    _, _, response = await source(core)
    original = core.event_store.append_batch
    seen = []

    async def lost_receipt(sid, bid, events):
        if sid == "branch":
            seen.append((bid, deepcopy(events)))
            await original(sid, bid, events)
            raise RuntimeError("receipt lost")
        return await original(sid, bid, events)

    monkeypatch.setattr(core.event_store, "append_batch", lost_receipt)
    with pytest.raises(PersistenceUnavailableError):
        await fork(core, response)
    assert not core._agent_lifecycle_manager.agent_ids_of_session("branch")
    monkeypatch.undo()
    result = await fork(core, response)
    rows = await core.event_store.read_range("branch")
    assert len(rows) == 4
    assert [r.event.id for r in rows[:3]] == [ev.id for ev in seen[0][1]]
    assert [r.content for r in await seed_records(core, result)] == [
        "old user input",
        "selected answer",
    ]
    await core.close()


@pytest.mark.asyncio
async def test_record_conflict_blocks_seed_and_does_not_overwrite(monkeypatch):
    core = runtime()
    _, _, response = await source(core)
    memory = core.providers.get_memory()

    async def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(memory, "fold", offline)
    with pytest.raises(OSError):
        await fork(core, response)
    monkeypatch.undo()
    requested = (
        await core.event_store.read_last_of_type("branch", EventType.AGENT_MEMORY_SEED_REQUESTED)
    ).event
    item = requested.payload["records"][0]
    ctx = ProviderContext(session_id="branch", agent_id=requested.agent_id)
    await memory.ingest(
        MemoryEvent(
            id=item["id"],
            kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.AGENT,
            address=MemoryAddress(session_id="branch", agent_id=requested.agent_id),
            role="user",
            content="impostor",
            timestamp=datetime.now(UTC),
        ),
        ctx,
    )
    with pytest.raises(MemorySeedConflictError):
        await fork(core, response)
    assert (await memory.read_records([item["id"]], ctx))[0].content == "impostor"
    assert (
        await core.event_store.read_last_of_type("branch", EventType.AGENT_MEMORY_SEED_APPLIED)
        is None
    )
    await core.close()


@pytest.mark.asyncio
async def test_sql_seed_cold_verification_does_not_revive_superseded_records(tmp_path):
    from ctx_weft.providers.memory.sql import open_sqlite_memory

    path = tmp_path / "seed.db"
    core = runtime()
    async with open_sqlite_memory(path) as memory:
        core.providers.register_memory(memory)
        _, _, response = await source(core)
        result = await fork(core, response)
        rows = await seed_records(core, result)
        ctx = ProviderContext(session_id="branch", agent_id=result.root_agent_id)
        await memory.fold([rows[0].id], [], ctx)
        await core.close()
    cold = runtime(core.event_store)
    async with open_sqlite_memory(path) as memory:
        cold.providers.register_memory(memory)
        # Source-independent retry verifies even already-superseded immutable records.
        assert await fork(cold, response) == result
        assert [r.content for r in await seed_records(cold, result)] == ["selected answer"]
        await cold.close()


class ScopedBlobs:
    """Distinct event/memory namespaces with enforced session ownership."""

    can_externalize = True

    def __init__(self, namespace, opaque=False):
        self.namespace = namespace
        self.values = {}
        self.opaque = opaque
        self.put_count = 0

    async def put(self, data, media_type, ctx):
        from hashlib import sha256

        self.put_count += 1
        suffix = str(self.put_count).encode() if self.opaque else b""
        ref = "blob:" + sha256(self.namespace.encode() + data + suffix).hexdigest()
        self.values[(ctx.session_id, ref)] = (data, media_type)
        return ref

    async def get(self, ref, ctx):
        return self.values.get((ctx.session_id, ref))


@pytest.mark.asyncio
@pytest.mark.parametrize("opaque", [False, True])
async def test_images_are_owned_by_target_in_both_blob_namespaces(opaque):
    from ctx_weft.protocols.content import ImagePart, TextPart

    core = runtime()
    _, prompt, response = await source(core)
    memory_blobs, event_blobs = ScopedBlobs("memory", opaque), ScopedBlobs("events", opaque)
    core.providers.register_memory_blob_store(memory_blobs)
    core.providers.register_event_blob_store(event_blobs)
    ctx = ProviderContext(session_id="s1")
    raw = b"\x89PNG\r\n\x1a\nimage"
    ref = await memory_blobs.put(raw, "image/png", ctx)
    record = MemoryEvent(
        id="source-image",
        kind=MemoryKind.CONVERSATION_TURN,
        scope=MemoryScope.AGENT,
        address=MemoryAddress(session_id="s1", agent_id=response.agent_id, task_id="task"),
        role="user",
        content=[
            ImagePart(source_type="ref", data=ref, media_type="image/png"),
            TextPart(text=f"Retained image: {ref}"),
        ],
        blob_refs=[ref],
        timestamp=prompt.timestamp,
    )
    await core.providers.get_memory().ingest(record, ctx)
    prompt = deepcopy(prompt)
    prompt.id += "image"
    prompt.payload["request_id"] = "image"
    prompt.payload["memory_blocks"].append({"kind": "history", "memory_id": record.id})
    response = deepcopy(response)
    response.id += "image"
    response.payload["request_id"] = "image"
    await core.event_bus.emit_batch((prompt, response))
    await publish_checkpoint(core, response, ["source-user", record.id])
    result = await fork(core, response)
    original_memory = core.providers.get_memory()
    await core.close()
    memory_blobs.values = {
        key: value for key, value in memory_blobs.values.items() if key[0] != "s1"
    }
    event_blobs.values = {key: value for key, value in event_blobs.values.items() if key[0] != "s1"}
    cold = runtime(core.event_store)
    cold.providers.register_memory(original_memory)
    cold.providers.register_memory_blob_store(memory_blobs)
    cold.providers.register_event_blob_store(event_blobs)
    puts = memory_blobs.put_count
    await cold.rebuild_session("branch", strict=True)
    assert memory_blobs.put_count == puts  # Verification needs no new opaque refs.
    rows = await seed_records(cold, result)
    image = next(
        part
        for row in rows
        if isinstance(row.content, list)
        for part in row.content
        if isinstance(part, ImagePart)
    )
    assert await memory_blobs.get(image.data, ProviderContext(session_id="branch")) == (
        raw,
        "image/png",
    )
    await cold.close()
    fresh = runtime(core.event_store)
    fresh.providers.register_memory_blob_store(memory_blobs)
    fresh.providers.register_event_blob_store(event_blobs)
    await fresh.rebuild_session("branch", strict=True)
    assert len(await seed_records(fresh, result)) == result.record_count
    await fresh.close()


@pytest.mark.asyncio
async def test_followup_prompt_uses_seed_without_replaying_source_execution():
    import asyncio

    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    agents = InlineAgentTemplateProvider()
    agents.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="branch continuation")])
    core = make_runtime(agent_provider=agents, llm=llm)
    core.providers.register_memory(InMemoryMemoryProvider())
    _, _, response = await source(core)
    result = await fork(core, response)
    assert llm.last_act_request is None
    await core.send_message(result.root_agent_id, "new question", session_id=result.session_id)
    async with asyncio.timeout(5):
        while llm.last_act_request is None:
            await asyncio.sleep(0.01)
    text = "\n".join(str(m.content) for m in llm.last_act_request.messages)
    assert "old user input" in text
    assert "selected answer" in text
    assert "new question" in text
    await core.close()


@pytest.mark.asyncio
async def test_cancelled_write_is_repaired_with_same_record_ids(monkeypatch):
    import asyncio

    core = runtime()
    _, _, response = await source(core)
    memory = core.providers.get_memory()
    original = memory.fold

    async def cancelled(*args, **kwargs):
        await original(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(memory, "fold", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await fork(core, response)
    monkeypatch.undo()
    result = await fork(core, response)
    assert len(await seed_records(core, result)) == 2
    await core.close()


@pytest.mark.asyncio
async def test_seed_body_is_not_repeated_in_snapshot_and_reducer_rejects_mismatched_applied():
    from ctx_weft.core.control.reducers import apply_events

    core = runtime()
    _, _, response = await source(core)
    await fork(core, response)
    view = await rebuild_view(core.event_store, "branch")
    assert "old user input" not in str(serialize_view(view))
    applied = (
        await core.event_store.read_last_of_type("branch", EventType.AGENT_MEMORY_SEED_APPLIED)
    ).event
    apply_events([applied], view)
    bad = deepcopy(applied)
    bad.payload["manifest_hash"] = "different"
    with pytest.raises(MemorySeedConflictError):
        apply_events([bad], view)
    await core.close()


@pytest.mark.asyncio
async def test_first_fork_from_cold_source_does_not_load_source_execution():
    core = runtime()
    _, _, response = await source(core)
    memory = core.providers.get_memory()
    await core.close()
    cold = runtime(core.event_store)
    cold.providers.register_memory(memory)
    result = await fork(cold, response)
    assert "s1" not in cold._task_managers
    assert not cold._agent_lifecycle_manager.agent_ids_of_session("s1")
    assert len(await seed_records(cold, result)) == 2
    await cold.close()


@pytest.mark.asyncio
async def test_inline_task_record_is_not_copied():
    from ctx_weft.core.assembler.context import json_value

    core = runtime()
    _, prompt, response = await source(core)
    (record,) = await core.providers.get_memory().read_records(
        ["source-user"], ProviderContext(session_id="s1")
    )
    inline = json_value(record)
    inline.update(id="", content="staged input", scope="task")
    prompt = deepcopy(prompt)
    prompt.id += "inline"
    prompt.payload["request_id"] = "inline"
    prompt.payload["memory_blocks"] = [{"kind": "history", "memory_id": "", "record": inline}]
    response = deepcopy(response)
    response.id += "inline"
    response.payload["request_id"] = "inline"
    await core.event_bus.emit_batch((prompt, response))
    await publish_checkpoint(core, response, [])
    result = await fork(core, response)
    assert [r.content for r in await seed_records(core, result)] == [
        "selected answer",
    ]
    await core.close()


@pytest.mark.asyncio
async def test_sql_events_and_memory_survive_full_close_and_reopen(tmp_path, monkeypatch):
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store
    from ctx_weft.providers.memory.sql import open_sqlite_memory

    event_path, memory_path = tmp_path / "events.db", tmp_path / "memory.db"
    async with (
        open_sqlite_event_store(event_path) as store,
        open_sqlite_memory(memory_path) as memory,
    ):
        core = runtime(store)
        core.providers.register_memory(memory)
        _, _, response = await source(core, tools=True, inline=True)
        result = await fork(core, response)
        view = await rebuild_view(store, "branch")
        await seed_snapshot(store, "branch", view, cut=await store.committed_head("branch"))
        await core.close()
    async with (
        open_sqlite_event_store(event_path) as store,
        open_sqlite_memory(memory_path) as memory,
    ):
        core = runtime(store)
        core.providers.register_memory(memory)
        original = store.read_range

        async def target_only(sid, **kwargs):
            assert sid == "branch"
            return await original(sid, **kwargs)

        monkeypatch.setattr(store, "read_range", target_only)
        assert await fork(core, response) == result
        assert len(await seed_records(core, result)) == result.record_count
        await core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("prepared", [False, True])
async def test_unapplied_reply_is_not_forkable_and_does_not_load_source(prepared):
    from ctx_weft import ForkCheckpointUnavailableError

    core = runtime()
    _, prompt, response = await source(core)
    prompt = deepcopy(prompt)
    prompt.id += "pending"
    prompt.payload["request_id"] = "pending"
    response = deepcopy(response)
    response.id += "pending"
    response.payload["request_id"] = "pending"
    await core.event_bus.emit_batch((prompt, response))
    if prepared:
        await publish_checkpoint(core, response, ["source-user"], applied=False)
    cold = runtime(core.event_store)
    cold.providers.register_memory(core.providers.get_memory())
    with pytest.raises(ForkCheckpointUnavailableError):
        await fork(cold, response)
    assert not cold._task_managers
    assert await core.event_store.committed_head("branch") == 0
    await cold.close()
    await core.close()


@pytest.mark.asyncio
async def test_checkpoint_cannot_import_task_scope_even_if_referenced():
    core = runtime()
    _, prompt, response = await source(core)
    await core.providers.get_memory().ingest(
        MemoryEvent(
            id="private-task",
            kind=MemoryKind.SUMMARY,
            scope=MemoryScope.TASK,
            address=MemoryAddress(session_id="s1", agent_id=response.agent_id, task_id="task"),
            role="assistant",
            content="private task recap",
            timestamp=datetime.now(UTC),
        ),
        ProviderContext(session_id="s1"),
    )
    prompt, response = deepcopy(prompt), deepcopy(response)
    prompt.id += "private"
    prompt.payload["request_id"] = "private"
    response.id += "private"
    response.payload["request_id"] = "private"
    await core.event_bus.emit_batch((prompt, response))
    await publish_checkpoint(core, response, ["private-task"])
    with pytest.raises(MemorySeedConflictError):
        await fork(core, response)
    assert await core.event_store.committed_head("branch") == 0
    await core.close()


@pytest.mark.asyncio
async def test_v1_seed_remains_recoverable_with_original_fingerprint(monkeypatch):
    from ctx_weft.core.memory_seed import digest, identity, prepare_seed

    core = runtime()
    _, _, response = await source(core)
    fingerprint = digest(
        dict(
            source_session_id="s1",
            response_event_id=response.id,
            template_id=None,
            llm_account=None,
            llm_model=None,
            llm_config=None,
            format_version=1,
        )
    )
    events = await prepare_seed(
        core,
        source_session_id="s1",
        response_event_id=response.id,
        session_id="legacy",
        agent_id=identity("legacy", "root-agent"),
        fingerprint=fingerprint,
        template_id=None,
        llm_account=None,
        llm_model=None,
        llm_config=None,
    )
    request = events[-1]
    request.payload["format_version"] = 1
    for item in request.payload["records"]:
        item.pop("kind")
    request.payload["manifest_hash"] = digest(request.payload["records"])
    await core.event_bus.emit_batch(events)
    cold = runtime(core.event_store)
    original = cold.event_store.read_range

    async def target_only(sid, **kwargs):
        assert sid == "legacy"
        return await original(sid, **kwargs)

    monkeypatch.setattr(cold.event_store, "read_range", target_only)
    result = await cold.fork_session_from_response(
        source_session_id="s1", response_event_id=response.id, session_id="legacy"
    )
    assert len(await seed_records(cold, result)) == 2
    await cold.close()
    monkeypatch.undo()
    await core.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption", ["binding", "agent", "duplicate_ids", "invalid_id", "applied"]
)
async def test_reducer_rejects_changed_checkpoint_materials(corruption):
    from ctx_weft.core.control.reducers import reduce_events

    core = runtime()
    try:
        await source(core)
        events = [deepcopy(row.event) for row in await core.event_store.read_range("s1")]
        kind = (
            EventType.OBSERVE_APPLIED
            if corruption == "applied"
            else EventType.REPLY_FORK_CHECKPOINT_PREPARED
        )
        event = next(event for event in events if event.type == kind)
        checkpoint = event.payload["fork_checkpoint"]
        if corruption == "binding":
            checkpoint["reply_anchor"]["response_memory_id"] = "another-reply"
        elif corruption == "agent":
            checkpoint["agent_id"] = "another-agent"
        elif corruption == "duplicate_ids":
            checkpoint["agent_memory_ids"] *= 2
        elif corruption == "invalid_id":
            checkpoint["agent_memory_ids"] = [{}]
        else:
            checkpoint["agent_memory_ids"] = []
        with pytest.raises(MemorySeedConflictError):
            reduce_events(events, "")
    finally:
        await core.close()
