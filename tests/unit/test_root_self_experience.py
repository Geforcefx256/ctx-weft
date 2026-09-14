"""root 自经验胶囊（task-resident，spec 2026-06-28 §3.1）：close 只写 finish 对、不镜像 body。

行为翻转（Task 1）：
- `_synthesize_dispatch_pair` **不再镜像** task 层 USER_PROMPT/TASK_COMPACT_SUMMARY/
  LLM_RESPONSE/TOOL_RESULT 进 agent 层——body 留各自 task 层（即胶囊）。
- agent 层每 task 只写 **finish 对**（assistant finish_task tool_call + tool Process Report）。
- 召回时 task 层 body 与 agent 层 finish 对按 (timestamp, seq_no) 归并（见 assembler 测试）。

注：原「交错时间线镜像」相关断言（UP/段摘要镜像、镜像顺序）随 mirror 删除而移除——其验证的
机制已不存在；body-保留改由 task 层召回验证（见 test_close_task / test_capsule_golden）。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.core.loop.steps.finalize import _synthesize_dispatch_pair
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio
T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _task_sc(task_id="t1", agent="ag1") -> MemoryAddress:
    """task 层 scope（含 task_id）——写入 task 层事件用。"""
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent)


def _agent_sc(agent="ag1") -> MemoryAddress:
    """agent 层 scope（task_id=None）——_synthesize_dispatch_pair 写入目标。"""
    return MemoryAddress(session_id="s1", task_id=None, agent_id=agent)


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


def _task(task_id="t1", agent="ag1", prompt="帮我把这个ppt写成pdf") -> Task:
    return Task(id=task_id, session_id="s1", status="FINISHED", tenant_id="default",
                assigned_agent_id=agent, creator_agent_id=agent, parent_task_id=None,
                title="PPTX转PDF", description="转换为 PDF", user_prompt=prompt,
                settings=NormalTaskSettings())


async def _agent_turns(mem, agent_scope):
    """返回 agent scope 的 AGENT_CONVERSATION_TURN 记录（chronological）。"""
    recs = await mem.recall_recent(
        agent_scope, [T.AGENT_CONVERSATION_TURN], 2000, _ctx())
    return list(reversed(recs))


async def test_synthesize_writes_only_finish_pair_no_body_mirror():
    """task-resident：task 层有 UP + 段摘要时，agent 层仍只写 finish 对（不镜像 body）。
    body（UP/段摘要）留 task 层、原样保留。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_sc()
    asc = _agent_sc()
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "帮我转", 0, role="user"), _ctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, tsc, "### 会话目标\n转 PDF", 1, role="assistant"), _ctx())
    task = _task()

    await _synthesize_dispatch_pair(mem, asc, task, "PDF转换执行过程：成功转换了文件", "PDF成功完成转换", "success", _ctx())

    turns = await _agent_turns(mem, asc)
    # 只有 finish 对（assistant finish_task + tool task_summary）
    roles = [r.role for r in turns]
    assert roles == ["assistant", "tool"], f"task-resident: only finish pair expected; got roles={roles}"
    assert all(r.type == T.AGENT_CONVERSATION_TURN for r in turns)
    assert turns[-2].metadata.get("tool_calls", [{}])[0].get("name", "").endswith("finish_task")
    # 反转契约：finish 对 assistant.content = act_recap（过程复述，≠ 答复）；tool.content = task_summary
    assert turns[-2].content.startswith("PDF转换执行过程：成功转换了文件")
    assert turns[-1].content == "[task: 'PPTX转PDF' (t1)] PDF成功完成转换"  # tool 槽带归属前缀
    assert all(r.metadata.get("origin_task_id") == "t1" for r in turns)

    # body 留 task 层（UP + 段摘要原样保留，未被镜像/supersede）
    body = await mem.recall_recent(tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY], 100, _ctx())
    assert {r.content for r in body} == {"帮我转", "### 会话目标\n转 PDF"}


async def test_no_task_layer_body_still_writes_finish_pair():
    """task 层无任何 body → agent 层仍写 finish 对（2 条）。"""
    mem = InMemoryMemoryProvider()
    asc = _agent_sc()
    task = _task(prompt="")

    await _synthesize_dispatch_pair(mem, asc, task, "out", "", "success", _ctx())

    turns = await _agent_turns(mem, asc)
    assert len(turns) == 2
    assert turns[-2].role == "assistant"
    assert turns[-1].role == "tool"
    # 无 user 镜像（无 body）
    assert not any(r.role == "user" for r in turns)


async def test_finish_pair_marker_and_report():
    """finish 对（反转契约）：assistant.content = act_recap（过程复述，≠ 答复）、finish_task 无参标记；
    tool content = task_summary（过程报告）。答复由内联 body / blackboard 承载，不在 finish 对。"""
    mem = InMemoryMemoryProvider()
    tsc = _task_sc()
    asc = _agent_sc()
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "帮我转", 0, role="user"), _ctx())
    task = _task()
    task.outputs = "## PDF 已完成"  # 答复走 blackboard / 内联 body，不进 finish 对

    await _synthesize_dispatch_pair(mem, asc, task, "PDF转换执行过程摘要", "成功转换", "success", _ctx())

    turns = await _agent_turns(mem, asc)
    assert [r.role for r in turns] == ["assistant", "tool"]
    tcs = turns[-2].metadata.get("tool_calls", [])
    assert len(tcs) == 1 and tcs[0]["name"].endswith("finish_task")
    assert tcs[0]["input"] == {}   # finish_task 无参收尾标记
    # 反转契约：assistant.content = act_recap（过程复述），tool.content = task_summary
    assert turns[-2].content.startswith("PDF转换执行过程摘要")
    assert turns[-1].content == "[task: 'PPTX转PDF' (t1)] 成功转换"  # tool 槽带归属前缀
    # tool_call 配对
    assert turns[-1].metadata.get("tool_call_id") == tcs[0]["id"]
