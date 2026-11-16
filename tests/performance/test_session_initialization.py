"""Initialization retains IDs only; repeat checks require no history I/O."""
import asyncio
import gc
import tracemalloc
import weakref
from time import perf_counter
from unittest.mock import AsyncMock

import pytest
from tests.unit.test_runtime_initialization_queries import runtime

from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import EventType



@pytest.mark.asyncio
async def test_many_initialized_sessions_release_reports_and_repeat_without_io(monkeypatch, record_property):
    rt = runtime()
    count = 300
    for i in range(count):
        sid = f"session-{i}"
        await rt.event_store.append_batch(sid, sid, [new_event(
            EventType.SESSION_CREATED, session_id=sid, tenant_id="default", origin="test",
            payload={"user_prompt": "x" * 32768})])
    references = []
    tracemalloc.start()
    started = perf_counter()
    for i in range(count):
        report = await rt.initialize(f"session-{i}")
        references.append(weakref.ref(report))
    del report
    await asyncio.sleep(0)
    gc.collect()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert all(ref() is None for ref in references)
    assert len(rt._initialized_sessions) == count and not rt._initializing_sessions
    assert not rt._task_managers and not rt._loaded_sessions
    assert not rt._agent_lifecycle_manager.all_agent_ids()
    record_property("initialize_seconds", perf_counter() - started)
    record_property("retained_bytes", retained)
    record_property("peak_bytes", peak)
    for method in ("list_session_ids", "read_range", "read_last_of_type", "committed_head"):
        monkeypatch.setattr(rt.event_store, method, AsyncMock(side_effect=AssertionError("repeat I/O")))
    started = perf_counter()
    for _ in range(30):
        for i in range(count):
            assert await rt.initialize(f"session-{i}") is None
    elapsed = perf_counter() - started
    record_property("repeat_seconds", elapsed)
    print(f"initialized={count} retained={retained} peak={peak} repeated_checks={count * 30} seconds={elapsed:.4f}")
