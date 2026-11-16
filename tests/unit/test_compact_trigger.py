"""PrepareStep._should_compact 纯预算触发（spec 2026-07-01 §3.6）。

历史注记：本文件曾覆盖「agent 层 / task 层可折叠 active 条数达 compact_message_delta
即触发」的累积增长修复（spec/06 §7）——HITL 密集叶子任务因 park→SUSPENDED 绕过 observe
的 max_turns 压缩、无界增长却点不着的问题。该消息条数门控在 2026-07-01 随 compact 改为
纯预算驱动而被移除（见 prepare.py _should_compact / escalating_compact）；growth-trigger
用例随之删除，只保留 token 比率契约。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.prepare import PrepareStep
from ctx_weft.protocols import MemoryEventType
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

T = MemoryEventType


def _scope():
    from ctx_weft.protocols import MemoryAddress
    return MemoryAddress(session_id="s1", task_id="t1", agent_id="a1")


def _pctx():
    from ctx_weft.protocols import ProviderContext
    return ProviderContext(session_id="s1", tenant_id="default")


def _ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_pctx(), task_manager=None)


def _state(*, ratio=0.8, context_limit=100000):
    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(compact_token_ratio=ratio),
        loop_guard=SimpleNamespace(context_limit=context_limit),
    )
    return SimpleNamespace(agent=agent, scope=_scope(), resolved_model=model_snapshot(context_limit=context_limit))


@pytest.mark.asyncio
async def test_below_ratio_no_trigger() -> None:
    """token 比率不足 → 不触发（消息条数已废，无论条数多少都不参与判断）。"""
    mem = InMemoryMemoryProvider()
    assert await PrepareStep()._should_compact(_state(), _ctx(mem), token_estimate=10) is False


@pytest.mark.asyncio
async def test_token_ratio_still_triggers() -> None:
    """token 比例达阈值 → 触发。"""
    mem = InMemoryMemoryProvider()
    state = _state(ratio=0.8, context_limit=1000)
    assert await PrepareStep()._should_compact(state, _ctx(mem), token_estimate=900) is True
