"""Unknown commits retain identity; health never clears while any receipt is missing."""
from __future__ import annotations

import asyncio
import sqlite3
from collections import Counter
from copy import deepcopy
from dataclasses import replace

import pytest
from sqlalchemy import text

from ctx_weft import RuntimeConfig
from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import (
    EventConflictError,
    EventType,
    PersistenceUnavailableError,
    TransientPersistenceError,
)
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



def runtime(store, **config):
    agents = InlineAgentTemplateProvider()
    agents.register(make_echo_template())
    rt = make_runtime(agent_provider=agents, event_store=store,
        llm=MockLLMAdapter(responses=[]), config=RuntimeConfig(
            storage_commit_max_attempts=1, storage_recovery_base_delay_sec=0.001, **config))
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


def event(identifier, **payload):
    return replace(new_event(EventType.CAPABILITY_FINISHED, session_id="s1",
                   tenant_id="default", origin="test", payload=payload), id=identifier)


@pytest.mark.asyncio
@pytest.mark.parametrize("receipt_lost", [False, True])
async def test_inline_confirmation_uses_original_batch_and_not_new_events(receipt_lost):
    store = InMemoryEventStore()
    original = store.append_batch
    attempts = []

    async def append(sid, bid, events):
        attempts.append((sid, bid, deepcopy(events)))
        if len(attempts) == 1:
            if receipt_lost:
                await original(sid, bid, events)
            raise ConnectionError("temporary lost connection/receipt")
        return await original(sid, bid, events)

    store.append_batch = append
    gate = CommitGate(store, max_attempts=3, base_delay=0)
    records = await gate.commit([event("e1", value="original")], batch_id="fixed-batch")
    assert attempts[0] == attempts[1]
    assert len(records) == 1 and records[0].position == 1
    assert len(await store.read_range("s1")) == 1
    assert not gate.has_pending("s1")


@pytest.mark.asyncio
async def test_unknown_envelopes_cannot_be_replaced_and_survive_caller_mutation():
    store = InMemoryEventStore()
    original = store.append_batch

    async def fail(*args):
        raise ConnectionError("unknown")

    store.append_batch = fail
    gate = CommitGate(store)
    item = event("e1", value=["original"])
    with pytest.raises(PersistenceUnavailableError):
        await gate.commit([item], batch_id="fixed")
    item.payload["value"].append("changed")
    with pytest.raises(EventConflictError):
        await gate.commit([item], batch_id="fixed")
    store.append_batch = original
    await gate.confirm_pending("s1")
    rows = await store.read_range("s1")
    assert rows[0].event.payload == {"value": ["original"]}


@pytest.mark.asyncio
async def test_all_unknown_batches_must_confirm_before_isolation_clears():
    store = InMemoryEventStore()
    rt = runtime(store, storage_recovery_max_attempts=0)
    original = store.append_batch
    unavailable = {"a", "b"}
    seen = []

    async def append(sid, bid, events):
        if bid in unavailable:
            raise ConnectionError(bid)
        assert rt.storage_health(sid) is not None
        return await original(sid, bid, events)

    async def observe(item):
        seen.append(item.id)

    store.append_batch = append
    rt.event_bus.subscribe(None, observe)
    outcomes = await asyncio.gather(*(rt.event_bus.emit(event(bid)) for bid in ("a", "b")),
                                    return_exceptions=True)
    assert all(isinstance(outcome, PersistenceUnavailableError) for outcome in outcomes)
    assert not rt.clear_storage_isolation("s1")
    unavailable.remove("a")
    assert not await rt.recover_storage("s1")
    assert rt.storage_health("s1") is not None
    assert not rt.clear_storage_isolation("s1")
    unavailable.clear()
    # Concurrent manual requests share the worker, and each delivery notifies once.
    assert all(await asyncio.gather(*(rt.recover_storage("s1") for _ in range(4))))
    assert rt.storage_health("s1") is None
    assert Counter(seen) == {"a": 1, "b": 1}
    assert len(await store.read_range("s1")) == 2


@pytest.mark.asyncio
async def test_inflight_submission_also_prevents_manual_clear():
    store = InMemoryEventStore()
    rt = runtime(store, storage_recovery_max_attempts=0)
    original = store.append_batch
    entered, release = asyncio.Event(), asyncio.Event()

    async def append(sid, bid, events):
        entered.set()
        await release.wait()
        return await original(sid, bid, events)

    rt._mark_storage_unavailable("s1", OSError("isolated"))
    store.append_batch = append
    submitting = asyncio.create_task(rt.event_bus.emit(event("active")))
    try:
        await entered.wait()
        assert not rt.clear_storage_isolation("s1")
        assert rt.storage_health("s1") is not None
    finally:
        release.set()
        await submitting
    assert rt.clear_storage_isolation("s1")


@pytest.mark.asyncio
async def test_cancellation_after_real_commit_keeps_original_unknown_receipt():
    store = InMemoryEventStore()
    original = store.append_batch
    gate = CommitGate(store)

    async def append(sid, bid, events):
        await original(sid, bid, events)
        raise asyncio.CancelledError()

    store.append_batch = append
    with pytest.raises(asyncio.CancelledError):
        await gate.commit([event("cancelled")], batch_id="original")
    assert gate.has_pending("s1")
    store.append_batch = original
    await gate.confirm_pending("s1")
    assert len(await store.read_range("s1")) == 1


@pytest.mark.asyncio
async def test_real_sqlite_busy_is_classified_and_same_batch_is_retried(tmp_path):
    path = tmp_path / "locked.db"
    async with open_sqlite_event_store(path) as store:
        async with store._factory() as db:
            await db.execute(text("PRAGMA busy_timeout=20"))
        locker = sqlite3.connect(path)
        locker.execute("BEGIN IMMEDIATE")
        failed = asyncio.Event()
        original = store.append_batch
        attempts = []

        async def append(sid, bid, events):
            attempts.append(bid)
            try:
                return await original(sid, bid, events)
            except TransientPersistenceError as exc:
                assert isinstance(exc.__cause__.orig, sqlite3.OperationalError)
                failed.set()
                raise

        store.append_batch = append
        gate = CommitGate(store, max_attempts=3, base_delay=0.05)
        submitting = asyncio.create_task(gate.commit([event("real-busy")], batch_id="same-batch"))
        try:
            await asyncio.wait_for(failed.wait(), 3)
        finally:
            locker.rollback()
            locker.close()
        records = await asyncio.wait_for(submitting, 3)
        assert len(records) == 1 and records[0].position == 1
        assert len(attempts) >= 2 and set(attempts) == {"same-batch"}
        assert len(await store.read_range("s1")) == 1


@pytest.mark.asyncio
async def test_recovery_deadline_keeps_unknown_batch_and_does_not_unlock_session():
    store = InMemoryEventStore()
    # Leave room for Windows scheduling when the healthy retry runs under load.
    # The blocked append still proves that the configured deadline is enforced.
    rt = runtime(store, storage_recovery_max_attempts=0, storage_recovery_max_duration_sec=0.5)
    original = store.append_batch

    async def fail(*args):
        raise ConnectionError("unavailable")

    store.append_batch = fail
    with pytest.raises(PersistenceUnavailableError):
        await rt.event_bus.emit(event("deadline"))

    async def hang(*args):
        await asyncio.Event().wait()

    store.append_batch = hang
    assert not await asyncio.wait_for(rt.recover_storage("s1"), 3)
    assert rt.storage_health("s1") is not None and not rt.clear_storage_isolation("s1")
    store.append_batch = original
    assert await rt.recover_storage("s1")
    assert len(await store.read_range("s1")) == 1


@pytest.mark.asyncio
async def test_healthy_recovery_does_not_schedule_or_resume_business_work():
    rt = runtime(InMemoryEventStore())

    async def unexpected_recovery(session_id):
        pytest.fail("healthy recovery must not requeue business work")

    rt._recover_storage_session = unexpected_recovery
    assert await rt.recover_storage("s1")
    assert not rt._storage_recovery_tasks
