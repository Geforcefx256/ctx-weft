"""G7 向后兼容：旧式（镜像型）胶囊召回顺序验证。

pre-refactor 胶囊将所有回合直接作为 AGENT_CONVERSATION_TURN 写入 agent scope
（"mirrored" 格式）：user 锚点、assistant finish_task、tool Process Report
均带 origin_task_id，时间戳升序。

验证：AgentRecallSource + composer 能正确召回这些旧记录，
组合后的消息列表按 (timestamp, seq_no) 升序排列。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.utils.estimate import estimate_tokens
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import recalled_blocks


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
SESSION = "s_legacy"


def _pctx() -> ProviderContext:
    return ProviderContext(session_id=SESSION, tenant_id="default")


def _agent_scope(agent_id: str = "ag_old") -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=None, agent_id=agent_id)


def _mirrored(type_: MemoryEventType, scope: MemoryAddress, content: str, t_offset: int,
               role: str, origin_task_id: str, **meta) -> MemoryEvent:
    """旧式镜像记录：AGENT_CONVERSATION_TURN 写入 agent scope，带 origin_task_id。"""
    return MemoryEvent(
        type=type_,
        address=scope,
        content=content,
        timestamp=_BASE + timedelta(seconds=t_offset),
        role=role,
        metadata={"origin_task_id": origin_task_id, **meta},
    )


async def _recall_blocks(mem: InMemoryMemoryProvider, agent_scope: MemoryAddress) -> list:
    """Drive AgentRecallSource.fetch and return blocks sorted by (timestamp, seq_no)."""
    deps = SimpleNamespace(memory=mem, provider_ctx=_pctx())
    req = SimpleNamespace(scope=agent_scope, token_counter=estimate_tokens)
    blocks = await recalled_blocks(req, deps)
    blocks.sort(key=lambda b: (b.value.timestamp, b.metadata.get("sequence", 0)))
    return blocks


@pytest.mark.asyncio
async def test_legacy_mirrored_capsule_recalled_in_order() -> None:
    """G7：旧式镜像胶囊（AGENT_CONVERSATION_TURN 写入 agent scope）仍能被召回，
    且按 (timestamp, seq_no) 升序排列。

    模拟 pre-refactor 胶囊结构：
      t=1  AGENT_CONVERSATION_TURN  role=user    (用户提问 / 锚点)
      t=2  AGENT_CONVERSATION_TURN  role=assistant  (finish_task tool call)
      t=3  AGENT_CONVERSATION_TURN  role=tool    (Process Report)
    三条均带 origin_task_id，无新字段依赖。
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope("ag_old")
    task_id = "task_legacy_001"

    # 旧式镜像 3 条：user 锚点、assistant finish_task call、tool Process Report
    await mem.ingest(
        _mirrored(T.AGENT_CONVERSATION_TURN, asc,
                  "请分析这份报告", 1, "user", task_id),
        _pctx(),
    )
    await mem.ingest(
        _mirrored(T.AGENT_CONVERSATION_TURN, asc,
                  "", 2, "assistant", task_id,
                  tool_calls=[{
                      "id": "tc_legacy",
                      "name": "control__finish_task",
                      "input": {"result": "分析完毕"},
                  }]),
        _pctx(),
    )
    await mem.ingest(
        _mirrored(T.AGENT_CONVERSATION_TURN, asc,
                  "Process Report: 分析完毕，发现 3 处关键问题。", 3, "tool", task_id,
                  tool_call_id="tc_legacy"),
        _pctx(),
    )

    blocks = await _recall_blocks(mem, asc)

    # 应召回到 3 条 AGENT_CONVERSATION_TURN（旧式镜像）
    act_blocks = [b for b in blocks if b.value.type == T.AGENT_CONVERSATION_TURN]
    assert len(act_blocks) == 3, (
        f"G7: expected 3 legacy AGENT_CONVERSATION_TURN blocks, got {len(act_blocks)}: "
        f"{[(b.value.role, b.value.content[:40]) for b in act_blocks]}"
    )

    # 按 (timestamp, seq_no) 升序：user → assistant → tool
    roles = [b.value.role for b in act_blocks]
    assert roles == ["user", "assistant", "tool"], (
        f"G7: legacy blocks must appear in ascending timestamp order (user→assistant→tool); "
        f"got roles={roles}"
    )

    # 内容验证
    assert "请分析这份报告" in act_blocks[0].value.content, (
        f"G7: user block content mismatch: {act_blocks[0].value.content!r}"
    )
    assert "Process Report:" in act_blocks[2].value.content, (
        f"G7: tool block must contain Process Report: {act_blocks[2].value.content!r}"
    )


@pytest.mark.asyncio
async def test_legacy_multi_task_mirrored_capsule_order() -> None:
    """G7 变体：多个旧式镜像任务（多 origin_task_id），按时间戳全局升序召回。

    任务 A（t=1,2,3）在任务 B（t=4,5,6）之前 → 召回顺序应保留此全局时序。
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope("ag_old2")

    # 任务 A：t=1,2,3
    for t_off, role, content in [
        (1, "user", "任务A：整理数据"),
        (2, "assistant", ""),
        (3, "tool", "Process Report: 数据整理完成。"),
    ]:
        await mem.ingest(
            _mirrored(T.AGENT_CONVERSATION_TURN, asc, content, t_off, role, "task_A"),
            _pctx(),
        )

    # 任务 B：t=4,5,6
    for t_off, role, content in [
        (4, "user", "任务B：生成摘要"),
        (5, "assistant", ""),
        (6, "tool", "Process Report: 摘要生成完成。"),
    ]:
        await mem.ingest(
            _mirrored(T.AGENT_CONVERSATION_TURN, asc, content, t_off, role, "task_B"),
            _pctx(),
        )

    blocks = await _recall_blocks(mem, asc)
    act_blocks = [b for b in blocks if b.value.type == T.AGENT_CONVERSATION_TURN]
    assert len(act_blocks) == 6, (
        f"G7 multi-task: expected 6 legacy blocks, got {len(act_blocks)}"
    )

    # 全局时序：任务 A 的 3 条（t=1,2,3）在任务 B 的 3 条（t=4,5,6）之前
    timestamps = [b.value.timestamp for b in act_blocks]
    assert timestamps == sorted(timestamps), (
        f"G7 multi-task: legacy blocks must be in ascending timestamp order; got {timestamps}"
    )

    # 前 3 属于任务 A，后 3 属于任务 B
    contents_a = [b.value.content for b in act_blocks[:3]]
    contents_b = [b.value.content for b in act_blocks[3:]]
    assert any("任务A" in c for c in contents_a), "G7: first 3 blocks should be task A"
    assert any("任务B" in c for c in contents_b), "G7: last 3 blocks should be task B"
