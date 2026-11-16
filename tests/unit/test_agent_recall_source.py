"""AgentRecallSource：单一装配路径——OPEN task 全对话 + CLOSED 残留（spec 2026-06-23）。"""

from __future__ import annotations

from tests.model_testkit import model_snapshot


from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextRequest
from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import history_messages


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _sc(task_id: str, agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent_id)


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


async def _collect(mem, scope) -> list:
    deps = AssemblerDeps(memory=mem, knowledge_providers=[], provider_ctx=_ctx())
    req = ContextRequest(resolved_model=model_snapshot(100_000, 0), purpose="act", scope=scope, task=None, agent=None, session=None,
                         template=None, bound_capabilities=[])
    blocks = [b async for b in AgentRecallSource().fetch(req, deps)]
    records = {r.id: r for r in await mem.read_records([b.memory_id for b in blocks], _ctx())}
    for block in blocks:
        block.value = records[block.memory_id]
    return blocks


@pytest.mark.asyncio
async def test_open_tasks_of_agent_all_visible() -> None:
    mem = InMemoryMemoryProvider()
    # task tA (open) and sibling tB (open), same agent ag1
    await mem.ingest(_ev(T.USER_PROMPT, _sc("tA"), "do A", 0, role="user"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, _sc("tA"), "A reply", 1, role="assistant"), _ctx())
    await mem.ingest(_ev(T.USER_PROMPT, _sc("tB"), "do B", 2, role="user"), _ctx())

    blocks = await _collect(mem, _sc("tB"))  # assembling for active task tB
    contents = {b.value.content for b in blocks}
    assert {"do A", "A reply", "do B"} <= contents  # sibling tA conversation visible


@pytest.mark.asyncio
async def test_closed_task_shows_dispatch_pair_and_hides_unpaired() -> None:
    mem = InMemoryMemoryProvider()
    # paired dispatch (closed sub-task residue)
    await mem.ingest(_ev(T.TASK_DISPATCH, _sc("t1"), "", 3, role="assistant",
                         tool_call_id="d1", tool_name="delegate_task", arguments={"title": "x"}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, _sc("t1"), "sub out", 4, role="tool",
                         tool_call_id="d1"), _ctx())
    # unpaired dispatch (open sub-task in progress) — must be hidden
    await mem.ingest(_ev(T.TASK_DISPATCH, _sc("t1"), "", 5, role="assistant",
                         tool_call_id="d2", tool_name="delegate_task", arguments={"title": "y"}), _ctx())

    blocks = await _collect(mem, _sc("t1"))
    tool_call_ids = [
        tc["id"]
        for b in blocks if b.value.role == "assistant"
        for tc in b.value.metadata.get("tool_calls", [])
    ]
    assert "d1" in tool_call_ids       # paired dispatch rendered
    assert "d2" not in tool_call_ids   # unpaired dispatch hidden


@pytest.mark.asyncio
async def test_composer_merges_open_conv_and_residue_by_timestamp() -> None:
    mem = InMemoryMemoryProvider()
    await mem.ingest(_ev(T.USER_PROMPT, _sc("t1"), "root prompt", 0, role="user"), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH, _sc("t1"), "", 1, role="assistant",
                         tool_call_id="d1", tool_name="delegate_task", arguments={}), _ctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, _sc("t1"), "child out", 2, role="tool",
                         tool_call_id="d1"), _ctx())

    blocks = await _collect(mem, _sc("t1"))
    messages = history_messages(blocks, task_id="t1")
    roles = [m.role for m in messages]
    assert roles == ["user", "user", "user"]
    assert "<task_start" in messages[1].content
    assert "<task_result" in messages[2].content and "child out" in messages[2].content
    assert all(not message.tool_calls for message in messages)
