"""inherit_memory：spawn 时把 parent 当前召回快照复制进 child scope（spec 2026-06-23）。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.core.runtime import _copy_memory_for_inherit, _latest_prior_root_task
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _ev(type_, scope, content, t, role=None) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role)


async def test_inherit_copies_parent_recall_into_child_scope() -> None:
    mem = InMemoryMemoryProvider()
    parent_scope = MemoryAddress(session_id="s1", task_id="p1", agent_id="ag1")
    await mem.ingest(_ev(T.USER_PROMPT, parent_scope, "parent ask", 0, role="user"), _ctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, parent_scope, "parent reply", 1, role="assistant"), _ctx())

    parent_task = Task(id="p1", session_id="s1", status="SUSPENDED", assigned_agent_id="ag1",
                       creator_agent_id="ag1", title="P", settings=NormalTaskSettings())
    child_task = Task(id="c1", session_id="s1", status="PENDING", assigned_agent_id="ag2",
                      creator_agent_id="ag1", parent_task_id="p1", title="C",
                      settings=NormalTaskSettings(inherit_memory=True))
    sub_agent = Agent(id="ag2", session_id="s1", template_id="t", parent_agent_id="ag1")

    # spec/09 §6：第一参数改收 agent id（血缘与记忆来源正交），且全 keyword-only。
    await _copy_memory_for_inherit(
        source_agent_id=parent_task.assigned_agent_id, child_task=child_task,
        sub_agent=sub_agent, memory=mem, session_id="s1", tenant_id="default",
        source_task_id=parent_task.id)

    child_scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="ag2")
    turns = await mem.recall_recent(child_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    contents = {t.content for t in turns}
    assert contents == {"parent ask", "parent reply"}
    # original parent events untouched
    parent_recs = await mem.recall_recent(parent_scope, [T.USER_PROMPT, T.LLM_RESPONSE], 100, _ctx())
    assert len(parent_recs) == 2


class _FakeTM:
    """Minimal stand-in exposing only all_tasks(), which is all the helper needs."""

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    def all_tasks(self) -> list[Task]:
        return list(self._tasks)


def _root(id_: str, t: int) -> Task:
    return Task(id=id_, session_id="s1", status="FINISHED", title=id_,
                created_at=_BASE + timedelta(seconds=t), settings=NormalTaskSettings())


def _child(id_: str, parent: str, t: int) -> Task:
    return Task(id=id_, session_id="s1", status="FINISHED", title=id_, parent_task_id=parent,
                created_at=_BASE + timedelta(seconds=t), settings=NormalTaskSettings())


async def test_latest_prior_root_picks_most_recent_earlier_root() -> None:
    """无 parent 的根子 agent 任务，回退继承源=created_at 最近的前序根任务。"""
    root1 = _root("r1", 0)
    root2 = _root("r2", 10)          # the most recent prior root
    sub = _child("sub", "r1", 5)     # not a root — must be ignored
    current = _root("cur", 20)       # the task being dispatched (a root subagent turn)
    later = _root("later", 30)       # created after current — must be ignored
    tm = _FakeTM([root1, root2, sub, current, later])

    picked = _latest_prior_root_task(tm, current)

    assert picked is root2


async def test_latest_prior_root_returns_none_when_no_prior_root() -> None:
    """首轮即 subagent 根任务，无前序根 → None（跳过继承，行为同现状）。"""
    current = _root("cur", 20)
    sub = _child("sub", "cur", 25)
    tm = _FakeTM([current, sub])

    assert _latest_prior_root_task(tm, current) is None


async def test_inherit_preserves_assistant_segment_summary() -> None:
    """parent 段摘要（role=assistant）inherit 后透传为 child 的 assistant 回合，
    且不误加 tool_calls；user 锚点恒在其前（段摘要非首条，R1 守护）。"""
    mem = InMemoryMemoryProvider()
    parent_scope = MemoryAddress(session_id="s1", task_id="p1", agent_id="ag1")
    await mem.ingest(_ev(T.USER_PROMPT, parent_scope, "原始诉求", 0, role="user"), _ctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, parent_scope, "段①摘要", 1, role="assistant"), _ctx())

    parent_task = Task(id="p1", session_id="s1", status="SUSPENDED", assigned_agent_id="ag1",
                       creator_agent_id="ag1", title="P", settings=NormalTaskSettings())
    child_task = Task(id="c1", session_id="s1", status="PENDING", assigned_agent_id="ag2",
                      creator_agent_id="ag1", parent_task_id="p1", title="C",
                      settings=NormalTaskSettings(inherit_memory=True))
    sub_agent = Agent(id="ag2", session_id="s1", template_id="t", parent_agent_id="ag1")

    # spec/09 §6：第一参数改收 agent id（血缘与记忆来源正交），且全 keyword-only。
    await _copy_memory_for_inherit(
        source_agent_id=parent_task.assigned_agent_id, child_task=child_task,
        sub_agent=sub_agent, memory=mem, session_id="s1", tenant_id="default",
        source_task_id=parent_task.id)

    child_scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="ag2")
    turns = list(reversed(await mem.recall_recent(child_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())))
    summary = next(t for t in turns if t.content == "段①摘要")
    assert summary.role == "assistant"
    assert not summary.metadata.get("tool_calls"), "段摘要无 tool_calls，不应误加"
    # 段摘要非首条：其前有 user 锚点
    assert turns[0].role == "user" and turns[0].content == "原始诉求"
