"""_copy_memory_for_inherit mirrors the parent's recall view (spec Phase 2 2026-06-30).

Same-agent sibling → framed full capsule (start_task frame precedes its body); cross-agent
sibling → frame + bubble. No naked sibling body without its dispatch frame.
"""
import pytest
from datetime import datetime, timezone, timedelta

from ctx_weft.core.runtime import _copy_memory_for_inherit
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryEvent, MemoryEventType as T, MemoryAddress
from ctx_weft.protocols.context import ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

_BASE = datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc)


def _ctx():
    return ProviderContext(session_id="s1", tenant_id="default")


def _ev(typ, scope, content, *, role, ts, md=None):
    return MemoryEvent(type=typ, address=scope, content=content, timestamp=ts, role=role,
                       metadata=md or {})


@pytest.mark.asyncio
async def test_inherit_mirrors_frames_and_bubbles():
    mem = InMemoryMemoryProvider()
    sess, parent_agent = "s1", "agR"
    parent_task_id = "p1"
    parent_scope = MemoryAddress(session_id=sess, task_id=parent_task_id, agent_id=parent_agent)

    # parent's own body (plan task) — task layer
    await mem.ingest(_ev(T.USER_PROMPT, parent_scope, "提交一个 plan", role="user", ts=_BASE), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, parent_scope, "好的", role="assistant", ts=_BASE + timedelta(seconds=1)), _ctx())

    # same-agent sibling Lily: start_task frame (back-dated) + her body (task layer, later) — agent+task layers
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, parent_scope, "", role="assistant",
                         ts=_BASE + timedelta(seconds=2),
                         md={"origin_task_id": parent_task_id,
                             "tool_calls": [{"id": "tc_lily", "name": "control:start_task", "input": {}}]}), _ctx())
    lily_scope = MemoryAddress(session_id=sess, task_id="lily", agent_id=parent_agent)  # same agent
    await mem.ingest(_ev(T.USER_PROMPT, lily_scope, "请向 Lily 问好", role="user",
                         ts=_BASE + timedelta(seconds=5)), _ctx())

    # cross-agent sibling Amy: start_task frame + bubble (both agent layer, parent scope)
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, parent_scope, "", role="assistant",
                         ts=_BASE + timedelta(seconds=3),
                         md={"origin_task_id": parent_task_id,
                             "tool_calls": [{"id": "tc_amy", "name": "control:start_task", "input": {}}]}), _ctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, parent_scope, "你好 Amy！👋", role="tool",
                         ts=_BASE + timedelta(seconds=3),
                         md={"origin_task_id": parent_task_id, "tool_call_id": "tc_amy"}), _ctx())

    parent_task = Task(id=parent_task_id, session_id=sess, status="SUSPENDED", tenant_id="default",
                       assigned_agent_id=parent_agent, creator_agent_id=parent_agent,
                       settings=NormalTaskSettings())
    child_task = Task(id="andy", session_id=sess, status="PENDING", tenant_id="default",
                      creator_agent_id=parent_agent, parent_task_id=parent_task_id,
                      settings=NormalTaskSettings())
    sub_agent = type("A", (), {"id": "agB"})()

    # spec/09 §6：第一参数改收 agent id（血缘与记忆来源正交）；源 task 只作元数据标签。
    await _copy_memory_for_inherit(
        source_agent_id=parent_task.assigned_agent_id, child_task=child_task,
        sub_agent=sub_agent, memory=mem, session_id=sess, tenant_id="default",
        source_task_id=parent_task.id)

    child_scope = MemoryAddress(session_id=sess, task_id="andy", agent_id="agB")
    turns = await mem.recall_recent(child_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    # recall_recent returns newest-first; flip to chronological for ordering assertions
    chrono = list(reversed(turns))
    contents = [(t.role, t.content) for t in chrono]

    # cross-agent Amy bubble is present (was previously MISSING)
    assert any(role == "tool" and "你好 Amy" in c for role, c in contents), \
        f"cross-agent bubble must be inherited; got {contents}"

    # same-agent Lily body present AND preceded by a start_task frame (was previously a NAKED leak)
    lily_body_idx = next(i for i, (role, c) in enumerate(contents) if "请向 Lily 问好" in c)
    preceding_frames = [t for t in chrono[:lily_body_idx]
                        if t.role == "assistant"
                        and any(tc.get("id") == "tc_lily" for tc in (t.metadata.get("tool_calls") or []))]
    assert preceding_frames, "Lily's body must be preceded by its start_task frame (no naked leak)"

    # everything carries both provenance tags (agent id is the real one, task id is optional)
    assert all(t.metadata.get("inherited_from_agent_id") == parent_agent for t in chrono)
    assert all(t.metadata.get("inherited_from_task_id") == parent_task_id for t in chrono)
