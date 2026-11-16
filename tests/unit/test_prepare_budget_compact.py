from tests.model_testkit import model_snapshot
from types import SimpleNamespace
import pytest

from ctx_weft.core.loop.steps.prepare import PrepareStep



@pytest.mark.asyncio
async def test_should_compact_pure_budget_true():
    state = SimpleNamespace(resolved_model=model_snapshot(context_limit=1000), agent=SimpleNamespace(
        loop_config=SimpleNamespace(compact_token_ratio=0.8),
        loop_guard=SimpleNamespace(context_limit=1000)))
    ctx = SimpleNamespace()
    assert await PrepareStep()._should_compact(state, ctx, token_estimate=850) is True


@pytest.mark.asyncio
async def test_should_compact_below_ratio_false_even_with_many_messages():
    # 消息条数门控已废：token 低就不压，无论消息多少
    state = SimpleNamespace(resolved_model=model_snapshot(context_limit=1000), agent=SimpleNamespace(
        loop_config=SimpleNamespace(compact_token_ratio=0.8),
        loop_guard=SimpleNamespace(context_limit=1000)))
    ctx = SimpleNamespace()
    assert await PrepareStep()._should_compact(state, ctx, token_estimate=200) is False


@pytest.mark.asyncio
async def test_should_compact_reserve_lowers_trigger_to_effective_limit():
    """spec §6：触发基线是 effective_limit = context_limit - reserved_output_tokens，
    而非原始 context_limit。

    context_limit=100_000, reserved_output_tokens=20_000 → effective_limit=80_000；
    compact_token_ratio=0.8 → 有效阈值 int(80_000*0.8)=64_000，
    而按旧的「直接用 context_limit」逻辑阈值会是 int(100_000*0.8)=80_000。
    token_estimate=64_000（≥ 有效阈值，但 < 旧阈值 80_000）：只有基于 effective_limit
    判定才会在此处触发 —— 这正是本用例要区分的行为。
    """
    state = SimpleNamespace(resolved_model=model_snapshot(context_limit=100_000, output_reserve=20_000), agent=SimpleNamespace(
        loop_config=SimpleNamespace(compact_token_ratio=0.8),
        loop_guard=SimpleNamespace(context_limit=100_000, reserved_output_tokens=20_000)))
    ctx = SimpleNamespace()
    assert await PrepareStep()._should_compact(state, ctx, token_estimate=64_000) is True


@pytest.mark.asyncio
async def test_should_compact_reserve_effective_limit_control_just_below_threshold():
    """同上参数，token_estimate=63_999（刚低于有效阈值 64_000）→ 不触发。"""
    state = SimpleNamespace(resolved_model=model_snapshot(context_limit=100_000, output_reserve=20_000), agent=SimpleNamespace(
        loop_config=SimpleNamespace(compact_token_ratio=0.8),
        loop_guard=SimpleNamespace(context_limit=100_000, reserved_output_tokens=20_000)))
    ctx = SimpleNamespace()
    assert await PrepareStep()._should_compact(state, ctx, token_estimate=63_999) is False
