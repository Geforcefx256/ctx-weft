"""recall_recent_by_agent：按 agent_id 跨 task 召回 task 层对话（spec 2026-06-23 重构）。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_TASK_TYPES = [T.USER_PROMPT, T.LLM_RESPONSE, T.TOOL_RESULT, T.TASK_COMPACT_SUMMARY]


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _sc(task_id: str, agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent_id)


def _ev(type_, scope, content, t) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t))


@pytest.mark.asyncio
async def test_recalls_task_conversation_across_tasks_of_same_agent() -> None:
    m = InMemoryMemoryProvider()
    await m.ingest(_ev(T.USER_PROMPT, _sc("tA", "ag1"), "promptA", 0), _ctx())
    await m.ingest(_ev(T.USER_PROMPT, _sc("tB", "ag1"), "promptB", 1), _ctx())
    await m.ingest(_ev(T.USER_PROMPT, _sc("tC", "ag2"), "promptC", 2), _ctx())  # other agent

    recs = await m.recall_recent_by_agent(_sc("ignored", "ag1"), _TASK_TYPES, 100, _ctx())
    contents = {r.content for r in recs}
    assert contents == {"promptA", "promptB"}        # ag1's tasks only, not ag2
    assert all(r.metadata.get("task_id") in {"tA", "tB"} for r in recs)


@pytest.mark.asyncio
async def test_excludes_superseded() -> None:
    m = InMemoryMemoryProvider()
    eid = await m.ingest(_ev(T.LLM_RESPONSE, _sc("tA", "ag1"), "live", 0), _ctx())
    await m.ingest(_ev(T.LLM_RESPONSE, _sc("tA", "ag1"), "dead", 1), _ctx())
    # supersede the second
    recs_all = await m.recall_recent_by_agent(_sc("x", "ag1"), _TASK_TYPES, 100, _ctx())
    dead_id = next(r.id for r in recs_all if r.content == "dead")
    await m.supersede([dead_id], _ctx())

    recs = await m.recall_recent_by_agent(_sc("x", "ag1"), _TASK_TYPES, 100, _ctx())
    assert [r.content for r in recs] == ["live"]
    assert recs[0].id == eid  # the surviving "live" record is the first-ingested one


@pytest.mark.asyncio
async def test_limit_truncates_to_most_recent() -> None:
    m = InMemoryMemoryProvider()
    for i in range(3):
        await m.ingest(_ev(T.LLM_RESPONSE, _sc("tA", "ag1"), f"r{i}", i), _ctx())
    recs = await m.recall_recent_by_agent(_sc("x", "ag1"), _TASK_TYPES, 1, _ctx())
    assert [r.content for r in recs] == ["r2"]  # newest-first, limit=1


@pytest.mark.asyncio
async def test_empty_types_returns_nothing() -> None:
    m = InMemoryMemoryProvider()
    await m.ingest(_ev(T.USER_PROMPT, _sc("tA", "ag1"), "x", 0), _ctx())
    assert await m.recall_recent_by_agent(_sc("x", "ag1"), [], 100, _ctx()) == []
