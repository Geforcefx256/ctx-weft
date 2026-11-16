"""Measure real SQLite transfer/decoding and Python peak allocation at startup.

Elapsed time is reported, not used as a flaky machine-speed assertion. Wire bytes
and peak Python allocations enforce that the optimization actually discards bodies.
"""

import tracemalloc
from dataclasses import replace
from statistics import median
from time import perf_counter

import pytest
from tests._initialization import initialize_all
from tests.unit.test_read_primitives_conformance import _ev
from tests.unit.test_runtime_initialization_queries import runtime
from tests.unit.test_startup_field_projection import full_startup, repair_facts

from ctx_weft.core.control.snapshot_writer import SnapshotWriter
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.events.store.sql import store as sql_store



@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True], ids=["full-replay", "compact-snapshot"])
async def test_sqlite_startup_discards_large_bodies(tmp_path, monkeypatch, record_property, snapshot):
    async with open_sqlite_event_store(tmp_path / "large.db") as store:
        body = "x" * (1024 * 1024)
        facts = [
            (EventType.SESSION_CREATED, {"root_agent_id": "root", "user_prompt": body}),
            (EventType.AGENT_INSTANTIATED, {"template_id": "agent:tpl_echo"}),
            (EventType.TASK_CREATED, {"task": {"id": "task", "assigned_agent_id": "root", "user_prompt": body}}),
            (EventType.TASK_FINISHED, {"outputs": body}),
            (EventType.AGENT_IDLE, {}),
        ]
        events = [replace(_ev(i, kind), agent_id="root", task_id="task", payload=payload)
                  for i, (kind, payload) in enumerate(facts, 1)]
        events += [replace(_ev(i, "LargeUnrelatedEvent"), agent_id="root", task_id="task",
                           payload={"body": body[:32768]}, metadata={"body": body[:32768]})
                   for i in range(6, 134)]
        await store.append_batch("s1", "large", events)
        if snapshot:
            await SnapshotWriter(store, None)._write("s1", events[-1], reason="test")

        decoded = 0
        full_row, projected_row = sql_store._row_to_event, sql_store.row_to_projected
        def count_full(row):
            nonlocal decoded
            decoded += len(row.payload_json.encode()) + len(row.metadata_json.encode())
            return full_row(row)
        def count_projected(row, **kw):
            nonlocal decoded
            decoded += len(row["payload_json"].encode()) + len(row["metadata_json"].encode())
            return projected_row(row, **kw)
        monkeypatch.setattr(sql_store, "_row_to_event", count_full)
        monkeypatch.setattr(sql_store, "row_to_projected", count_projected)

        async def measure(full):
            nonlocal decoded
            decoded = 0
            tracemalloc.start()
            started = perf_counter()
            try:
                report = await full_startup(monkeypatch, store) if full else await initialize_all(runtime(store))
                elapsed = perf_counter() - started
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return report, {"seconds": elapsed, "decoded_bytes": decoded, "peak_bytes": peak}

        expected, full = await measure(True)
        actual, light = await measure(False)
        # Allocation tracing disproportionately taxes query construction. Measure
        # elapsed time separately, with cold SQL compilation for both paths.
        for is_full, values in ((True, full), (False, light)):
            samples = []
            for _ in range(3):
                store._factory.kw["bind"].sync_engine.clear_compiled_cache()
                started = perf_counter()
                if is_full:
                    await full_startup(monkeypatch, store)
                else:
                    await initialize_all(runtime(store))
                samples.append(perf_counter() - started)
            values["seconds"] = median(samples)
        assert repair_facts(actual) == repair_facts(expected)
        assert light["decoded_bytes"] < full["decoded_bytes"] / 20
        assert light["peak_bytes"] < full["peak_bytes"] / 2
        for name, values in (("full", full), ("light", light)):
            for key, value in values.items():
                record_property(f"{name}_{key}", value)
        print(f"\nstartup snapshot={snapshot}: full={full}; light={light}")
