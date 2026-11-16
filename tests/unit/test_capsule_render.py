"""Task-6 胶囊渲染 + 孤儿配对回归（spec 2026-06-26）。

验证 _synthesize_dispatch_pair 写入的 finish 对
（assistant tool_calls=[control__finish_task] + tool tool_call_id 配对）：

1. AgentRecallSource.fetch 能正确渲染为 assistant+tool 两块（不丢弃）。
2. PromptBuilder 的历史投影产出 assistant(tool_calls=[finish_task]) + tool(tool_call_id)。
3. drop_orphan_tool_results 不把 tool 块判为孤儿丢弃（配对完整时）。
4. 孤立（无前序 assistant）的 tool 块确实被 drop_orphan_tool_results 丢弃（边界保护）。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot


from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextRequest
from ctx_weft.core.loop.llm_gateway import drop_orphan_tool_results
from ctx_weft.protocols import (
    LLMMessage,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
)
from ctx_weft.protocols.capability import qualify
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import history_messages, recalled_blocks


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)

_FINISH_TASK_NAME = qualify("control:finish_task")  # → "control__finish_task"


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _sc(task_id: str = "t1", agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent_id)


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(
        type=type_,
        address=scope,
        content=content,
        timestamp=_BASE + timedelta(seconds=t),
        role=role,
        metadata=meta,
    )


async def _collect(mem, scope) -> list:
    deps = AssemblerDeps(memory=mem, knowledge_providers=[], provider_ctx=_ctx())
    req = ContextRequest(resolved_model=model_snapshot(100_000, 0),
        purpose="act",
        scope=scope,
        task=None,
        agent=None,
        session=None,
        template=None,
        bound_capabilities=[],
    )
    return await recalled_blocks(req, deps)


# ─────────────────────────────────────────────────────────────────────────────
# 1. AgentRecallSource.fetch 渲染 finish 对
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_recall_renders_finish_pair() -> None:
    """AgentRecallSource 能把 AGENT_CONVERSATION_TURN finish 对渲染为两块。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    tool_call_id = "ftc-001"
    outputs_text = "final answer"

    # assistant 块：finish_task tool_call（Task-6 写入格式）
    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope, "",
            t=10, role="assistant",
            origin_task_id="t1",
            tool_calls=[{
                "id": tool_call_id,
                "name": _FINISH_TASK_NAME,
                "input": {"result": outputs_text},
            }],
        ),
        _ctx(),
    )
    # tool 块：Process Report（Task-6 写入格式）
    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope,
            "Process Report: all done",
            t=10, role="tool",
            origin_task_id="t1",
            tool_call_id=tool_call_id,
        ),
        _ctx(),
    )

    blocks = await _collect(mem, scope)

    asst_blocks = [b for b in blocks if b.value.role == "assistant"]
    tool_blocks = [b for b in blocks if b.value.role == "tool"]

    assert asst_blocks, "assistant block (finish_task tool_call) must be rendered"
    assert tool_blocks, "tool block (Process Report) must be rendered"

    # assistant 块携带正确的 tool_calls
    finish_tool_calls = [
        tc for b in asst_blocks
        for tc in b.value.metadata.get("tool_calls", [])
        if tc.get("name") == _FINISH_TASK_NAME
    ]
    assert finish_tool_calls, f"finish_task tool_call not found in assistant blocks; got {asst_blocks}"
    assert finish_tool_calls[0]["id"] == tool_call_id
    assert finish_tool_calls[0]["input"]["result"] == outputs_text

    # tool 块携带匹配的 tool_call_id
    matched_tool = [
        b for b in tool_blocks
        if b.value.metadata.get("tool_call_id") == tool_call_id
    ]
    assert matched_tool, "tool block must carry matching tool_call_id"
    assert "Process Report" in matched_tool[0].value.content


# ─────────────────────────────────────────────────────────────────────────────
# 2. Composer 把两块渲染为 LLMMessage 对
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composer_renders_finish_pair_as_llm_messages() -> None:
    """PromptBuilder 把 finish 对投影成 assistant(tool_calls) + tool(tool_call_id)。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    tool_call_id = "ftc-002"

    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope, "",
            t=10, role="assistant",
            origin_task_id="t1",
            tool_calls=[{"id": tool_call_id, "name": _FINISH_TASK_NAME, "input": {"result": "ok"}}],
        ),
        _ctx(),
    )
    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope,
            "Process Report: completed",
            t=10, role="tool",
            origin_task_id="t1",
            tool_call_id=tool_call_id,
        ),
        _ctx(),
    )

    blocks = await _collect(mem, scope)
    messages = history_messages(blocks, task_id="t1")

    assert all(m.role != "tool" and not m.tool_calls for m in messages)
    assert any(m.role == "assistant" and m.content == "ok" for m in messages)
    assert sum(m.role == "user" and "<task_finish" in str(m.content) for m in messages) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. drop_orphan_tool_results 保留完整 finish 对
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_drop_orphan_preserves_finish_pair() -> None:
    """配对完整时，drop_orphan_tool_results 不应丢弃 tool 块。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    tool_call_id = "ftc-003"

    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope, "",
            t=10, role="assistant",
            origin_task_id="t1",
            tool_calls=[{"id": tool_call_id, "name": _FINISH_TASK_NAME, "input": {"result": "done"}}],
        ),
        _ctx(),
    )
    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope,
            "Process Report: OK",
            t=10, role="tool",
            origin_task_id="t1",
            tool_call_id=tool_call_id,
        ),
        _ctx(),
    )

    blocks = await _collect(mem, scope)
    messages = history_messages(blocks, task_id="t1")

    # 确认消息列表合法（含 assistant + tool 配对）
    assert all(m.role != "tool" for m in messages)
    after = drop_orphan_tool_results(messages)
    assert after == messages
    assert any(m.role == "assistant" and m.content == "done" for m in after)
    assert sum(m.role == "user" and "<task_finish" in str(m.content) for m in after) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. 孤立 tool 块确实被 drop_orphan_tool_results 丢弃（边界）
# ─────────────────────────────────────────────────────────────────────────────


def test_drop_orphan_removes_unmatched_tool_message() -> None:
    """孤立 tool 消息（无前序 assistant tool_call）必须被 drop_orphan_tool_results 丢弃。"""
    messages = [
        LLMMessage(role="user", content="hello"),
        # tool 消息无前序 assistant tool_call → 孤儿
        LLMMessage(role="tool", content="orphan result", tool_call_id="orphan-id"),
    ]
    after = drop_orphan_tool_results(messages)
    assert all(m.role != "tool" for m in after), (
        "orphan tool message should have been dropped, but survived"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. finish 对 + 其他对话混合时，所有配对均保留
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finish_pair_survives_alongside_conversation() -> None:
    """finish 对与普通对话混合时，finish 对不被孤儿清理误删。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    tool_call_id = "ftc-004"

    # 普通 task 层对话（OPEN task）
    await mem.ingest(
        _ev(T.USER_PROMPT, scope, "start task", t=0, role="user"),
        _ctx(),
    )
    await mem.ingest(
        _ev(T.LLM_RESPONSE, scope, "working on it", t=1, role="assistant", tool_calls=[]),
        _ctx(),
    )

    # finish 对（agent 层 AGENT_CONVERSATION_TURN）
    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope, "",
            t=10, role="assistant",
            origin_task_id="t1",
            tool_calls=[{"id": tool_call_id, "name": _FINISH_TASK_NAME, "input": {"result": "result"}}],
        ),
        _ctx(),
    )
    await mem.ingest(
        _ev(
            T.AGENT_CONVERSATION_TURN, scope,
            "Process Report: success",
            t=10, role="tool",
            origin_task_id="t1",
            tool_call_id=tool_call_id,
        ),
        _ctx(),
    )

    blocks = await _collect(mem, scope)
    messages = history_messages(blocks, task_id="t1")
    after = drop_orphan_tool_results(messages)

    # finish 对的 tool 消息必须存活
    assert any(m.role == "assistant" and m.content == "result" for m in after)
    assert sum(m.role == "user" and "<task_finish" in str(m.content) for m in after) == 1
    assert any("working on it" in str(m.content) for m in after)
    assert all(not m.tool_calls and m.role != "tool" for m in after)
