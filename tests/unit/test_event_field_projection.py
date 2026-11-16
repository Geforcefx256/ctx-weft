"""Field projection preserves row selection, JSON types, and storage isolation."""
# ruff: noqa: F811 -- pytest fixtures are explicitly imported for parametrization.
from dataclasses import replace

import pytest

from ctx_weft.protocols.events import (
    Event,
    EventReadProjection,
    ProjectedEvent,
    ProjectedStoredEvent,
)
from tests.unit.test_read_primitives_conformance import _ev, store  # noqa: F401


@pytest.mark.asyncio
async def test_metadata_survives_commit_reads_and_caller_mutation(store):
    original = {"host": {"sources": [{"url": "https://example.com/"}]}}
    event = replace(_ev(1, "Test"), metadata=original)
    receipt = await store.append_batch("s1", "metadata-batch", [event])
    original["host"]["sources"].clear()
    event.metadata.clear()
    receipt.records[0].event.metadata.clear()
    row, = await store.read_range("s1")
    expected = {"host": {"sources": [{"url": "https://example.com/"}]}}
    assert row.event.metadata == expected
    row.event.metadata.clear()
    last = await store.read_last_of_type("s1", "Test")
    assert last.event.metadata == expected
    projection = EventReadProjection({"Test": ()}, include_metadata=True)
    projected, = await store.read_range("s1", projection=projection)
    assert projected.event.metadata == expected
    replay = replace(_ev(1, "Test"), metadata=expected)
    assert (await store.append_batch("s1", "metadata-batch", [replay])).records[0].event.metadata == expected



@pytest.mark.asyncio
async def test_projection_preserves_json_shape_and_envelopes(store):
    payload = {"keep": {"null": None, "false": False, "true": True, "zero": 0, "empty": "",
                         "array": [], "object": {}, "unicode": "中文", "drop": "large"},
               "maps": {"a.b": {"id": 1, "body": "large"}, 'quote"': {"id": 2}},
               "items": [{"id": 3, "body": "large"}, {"id": 4}], "drop": "large"}
    event = replace(_ev(1, "Test"), payload=payload, metadata={"secret": "omitted"})
    await store.append_batch("s1", "batch", [event])
    projection = EventReadProjection({"Test": tuple(
        ("keep", key) for key in payload["keep"] if key != "drop") + (
            ("maps", "*", "id"), ("items", "*", "id"), ("absent",))})
    row, = await store.read_range("s1", projection=projection)
    assert isinstance(row, ProjectedStoredEvent) and isinstance(row.event, ProjectedEvent)
    assert not isinstance(row.event, Event)
    assert row.event.payload == {
        "keep": {k: v for k, v in payload["keep"].items() if k != "drop"},
        "maps": {"a.b": {"id": 1}, 'quote"': {"id": 2}}, "items": [{"id": 3}, {"id": 4}]}
    assert row.event.metadata == {}
    assert row.event.run_id == event.run_id and row.event.sequence == event.sequence
    assert row.position == 1
    last = await store.read_last_of_type("s1", "Test", projection=projection)
    assert last == row
    row.event.payload["maps"]["a.b"]["id"] = 99
    full, = await store.read_range("s1")
    assert full.event.payload == payload and full.event.metadata == event.metadata


@pytest.mark.asyncio
async def test_projection_does_not_filter_envelopes_or_change_related_matching(store):
    await store.append_batch("s1", "batch", [
        replace(_ev(1, "TaskCreated"), payload={"task": {"id": "target", "body": "large"}}),
        replace(_ev(2, "Noise"), payload={"body": "large"}),
    ])
    projection = EventReadProjection({"TaskCreated": (("task", "status"),)})
    rows = await store.read_range("s1", projection=projection)
    assert [r.position for r in rows] == [1, 2]
    assert rows[1].event.payload == {}
    related = await store.read_range("s1", related_task_id="target", projection=projection)
    assert [r.position for r in related] == [1]
    assert related[0].event.payload == {"task": {}}
    assert await store.read_range("s1", after_position=1, through_position=2,
                                 exclude_types=("Noise",), projection=projection) == []
    assert await store.read_last_of_type("s1", "Absent", projection=projection) is None
    with pytest.raises(TypeError, match="complete Event"):
        await store.append_batch("s1", "invalid", [rows[0].event])


@pytest.mark.asyncio
async def test_whole_subtree_metadata_and_null_intermediate(store):
    event = replace(_ev(1, "Test"), payload={"a": None, "b": False, "c": {"nested": [1, None]}},
                    metadata={"kept": True})
    await store.append_batch("s1", "batch", [event])
    projection = EventReadProjection({"Test": (("a", "child"), ("b", "child"), ("c",))}, True)
    row = await store.read_last_of_type("s1", "Test", projection=projection)
    assert row.event.payload == event.payload and row.event.metadata == event.metadata


@pytest.mark.asyncio
async def test_projection_cache_distinguishes_field_sets_and_whole_payload(store):
    event = replace(_ev(1, "Test"), payload={"a": False, "b": None, 'quote".key': "中文"})
    await store.append_batch("s1", "batch", [event])
    for paths, expected in [
        ((("a",),), {"a": False}), ((("b",),), {"b": None}),
        ((( 'quote".key',),), {'quote".key': "中文"}), (((),), event.payload),
    ]:
        projection = EventReadProjection({"Test": paths})
        rows = await store.read_range("s1", projection=projection)
        assert rows[0].event.payload == expected
        assert (await store.read_last_of_type("s1", "Test", projection=projection)).event.payload == expected


@pytest.mark.asyncio
async def test_postgres_projection_compiles_without_full_orm_selection():
    from sqlalchemy import select
    from sqlalchemy.dialects.postgresql import dialect

    from ctx_weft.providers.events.store.sql.projection import projected_columns

    query = select(*projected_columns(EventReadProjection({"Test": (("map", "*", "id"),)})))
    statement = str(query.compile(dialect=dialect()))
    assert "WITH RECURSIVE projected" in statement
    assert "jsonb_each" in statement and "jsonb_array_elements" in statement
    assert "metadata_json AS metadata_json" not in statement
