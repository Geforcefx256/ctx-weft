"""上下文恢复的两个判定函数的单元覆盖：`_can_recover_context` 与 `_account_tokens` 的阈值。

集成层（tests/integration/test_context_recovery_e2e.py）钉的是「整条恢复路径跑得通」，
每跑一条要起一个真 runtime；判定本身的决策表放在这里逐格钉，便宜且能覆盖集成层够不着的
组合（配额恰好耗尽、第二次恢复、字段缺失的回落）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.driver import COMPACT_NOOP_KEY, CONTEXT_RECOVERY_COUNT_KEY
from ctx_weft.core.loop.steps.act import _account_tokens, _can_recover_context
from ctx_weft.core.models.agent import LoopGuard
from ctx_weft.core.utils.estimate import effective_limit
from ctx_weft.protocols import LLMUsage, MemoryAddress
from ctx_weft.protocols.template import LoopConfig


def _state(*, limit: int = 2, used: int = 0, noop=None, loop_config=...):
    """最小 state：`_can_recover_context` 只读 agent.loop_config 与 extra 两处。"""
    extra: dict = {}
    if used:
        extra[CONTEXT_RECOVERY_COUNT_KEY] = used
    if noop is not None:
        extra[COMPACT_NOOP_KEY] = noop
    lc = LoopConfig(max_context_recoveries=limit) if loop_config is ... else loop_config
    return SimpleNamespace(agent=SimpleNamespace(loop_config=lc), extra=extra)


# ── _can_recover_context 决策表 ───────────────────────────────────────────────


def test_disabled_by_zero_quota():
    """`max_context_recoveries=0` = 关闭恢复：行为回到改造前（直接退 observe 吃 retry）。"""
    assert _can_recover_context(_state(limit=0)) is False


def test_negative_quota_is_treated_as_disabled():
    """负数配置不该绕过配额（`<= 0` 而非 `== 0`）。"""
    assert _can_recover_context(_state(limit=-1)) is False


def test_first_recovery_ignores_a_stale_noop_flag():
    """**回归钉**：`used == 0` 时不采信「压不动」旗。

    本 run 开头那次 prepare 的 compact 也会写这面旗，而那一次几乎必然 noop——run 刚起，
    task 层往往只有一条 USER_PROMPT，各级 guard 全不满足。曾经不作区分，于是旗在 act 还没
    发出第一个请求时就立起来，整个 run 的恢复资格被一笔勾销（实测 prepare→act→observe，
    恢复从不发生）。想知道压得动压不动，总得先真压一次。
    """
    assert _can_recover_context(_state(used=0, noop=True)) is True


def test_first_recovery_allowed_without_flag():
    assert _can_recover_context(_state(used=0, noop=False)) is True


def test_second_recovery_vetoed_when_compaction_freed_nothing():
    """已恢复过一次、那次 compact 零折叠 → 立刻放弃，不把配额耗在注定无果的往返上。"""
    assert _can_recover_context(_state(limit=2, used=1, noop=True)) is False


def test_second_recovery_allowed_when_compaction_worked():
    """压得动就继续：配额没用完、旗是 False → 允许第二次恢复。"""
    assert _can_recover_context(_state(limit=2, used=1, noop=False)) is True


def test_quota_exhausted_stops_recovery():
    """`used == limit` 即耗尽（而非 `>`）——第三次越线退回 observe/retry 的老路。"""
    assert _can_recover_context(_state(limit=2, used=2, noop=False)) is False


def test_quota_exhausted_even_without_flag_or_overshoot():
    """`used > limit`（配置被调小等）同样耗尽，不得因越界而放行。"""
    assert _can_recover_context(_state(limit=1, used=5, noop=False)) is False


def test_missing_config_field_falls_back_to_disabled():
    """`loop_config` 没有这个字段（旧快照 / 精简 stub）→ 保守地当作关闭，而不是崩。"""
    assert _can_recover_context(_state(loop_config=SimpleNamespace())) is False


# ── _account_tokens 的停机阈值边界 ────────────────────────────────────────────


class _FakeMemory:
    async def load_view(self, *a, **kw):
        return []


def _acct_state(*, context_limit=100_000, reserve=0, loop_config=...):
    lc = LoopConfig() if loop_config is ... else loop_config
    guard = LoopGuard(context_limit=context_limit, reserved_output_tokens=reserve)
    return SimpleNamespace(
        agent=SimpleNamespace(id="a1", loop_guard=guard, loop_config=lc),
        session=SimpleNamespace(token_used=0),
        scope=MemoryAddress(session_id="s1", task_id="t1", agent_id="a1"),
    )


def _ctx():
    return SimpleNamespace(memory=_FakeMemory(), provider_ctx=SimpleNamespace())


@pytest.mark.asyncio
async def test_no_stop_when_effective_limit_is_zero():
    """预留吃光整个窗口（eff <= 0）→ 永不停机。

    停机意味着「结束本段 act」，而 eff<=0 时**每一轮**都会越线，等于任务寸步难行。
    这种配置的正路是在装配期响亮失败（ContextOverflowError），不是在这里静默绞杀。
    """
    state = _acct_state(context_limit=1000, reserve=1000)
    assert effective_limit(1000, 1000) == 0
    hit = await _account_tokens(state, _ctx(), LLMUsage(prompt_tokens=999_999))
    assert hit is False


@pytest.mark.asyncio
async def test_no_stop_when_usage_is_unavailable():
    """provider 没回 usage（prompt_tokens==0）→ 不停机。

    0 是「不知道」，不是「很小」。拿它去比阈值会恒不命中，但显式写出来是为了挡住反向的
    改法（比如哪天改成 `<=` 或拿估算值兜底），那会让无 usage 的 provider 每轮都被判越线。
    """
    state = _acct_state(context_limit=1000)
    assert await _account_tokens(state, _ctx(), LLMUsage(prompt_tokens=0)) is False


@pytest.mark.asyncio
async def test_stop_ratio_falls_back_to_legacy_value_when_field_missing():
    """`loop_config` 缺 `context_limit_stop_ratio` → 回落 **0.8**（= 改造前的硬编码）。

    刻意不回落到新默认值 0.9：字段缺失意味着这是一份旧配置，行为该与旧代码一致。
    """
    state = _acct_state(context_limit=1000, loop_config=SimpleNamespace())
    assert await _account_tokens(state, _ctx(), LLMUsage(prompt_tokens=800)) is True
    state = _acct_state(context_limit=1000, loop_config=SimpleNamespace())
    assert await _account_tokens(state, _ctx(), LLMUsage(prompt_tokens=799)) is False


@pytest.mark.asyncio
async def test_stop_ratio_is_read_from_config():
    """比率真的来自配置（不是又一个写死的数）。"""
    state = _acct_state(context_limit=1000, loop_config=LoopConfig(context_limit_stop_ratio=0.5))
    assert await _account_tokens(state, _ctx(), LLMUsage(prompt_tokens=500)) is True
    state = _acct_state(context_limit=1000, loop_config=LoopConfig(context_limit_stop_ratio=0.5))
    assert await _account_tokens(state, _ctx(), LLMUsage(prompt_tokens=499)) is False
