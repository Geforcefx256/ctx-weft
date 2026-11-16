
from ctx_weft.protocols import ModelCallConfig
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.context import PromptContext
from ctx_weft.core.loop.llm_gateway import apply_dynamic_max_tokens
from ctx_weft.core.loop.steps.prepare import PrepareStep
from ctx_weft.core.models.errors import ContextOverflowError
from ctx_weft.core.orchestrator.model import ModelChoice
from ctx_weft.protocols import LLMRequest, OutputPolicy
from tests.assembler_testkit import prompt_request
from tests.model_testkit import model_snapshot
from tests.unit.test_model_resolution import TPL, _Client, _reg


@pytest.mark.parametrize(
    "context,reserve,ceiling",
    [(0, 0, None), (100, 100, None), (100, -1, None), (100, 0, 101), (100, 0, 0)],
)
def test_invalid_budget_rejected(context, reserve, ceiling):
    with pytest.raises(ValueError):
        model_snapshot(context, reserve, ceiling)


def test_resolved_model_is_immutable_and_input_limit_is_derived():
    resolved = model_snapshot(128_000, 8000, 16_000)
    assert resolved.effective_input_limit == 120_000
    with pytest.raises(FrozenInstanceError):
        resolved.context_limit = 64_000
    inputs = PromptContext.from_request(prompt_request(resolved_model=resolved))
    assert (inputs.context_limit, inputs.reserved_output_tokens) == (128_000, 8000)


@pytest.mark.asyncio
async def test_switch_updates_next_dispatch_budget_without_mutating_running_snapshot():
    clients = {
        "large": _Client(context_limit=200_000, output_reserve=8192),
        "small": _Client(context_limit=16_000, output_reserve=2000),
    }
    registry = _reg(lambda account, model, **_context: clients[model])
    agent, _ = await registry.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default", llm=ModelChoice(model="large")
    )
    _, running = registry.materialize(agent.id, session_id="s1", tenant_id="default")
    await registry.set_agent_llm(agent.id, ModelChoice(model="small"), reason="user_selected")
    next_agent, next_run = registry.materialize(agent.id, session_id="s1", tenant_id="default")
    assert running.client is clients["large"]
    assert running.effective_input_limit == 191_808
    assert next_run.client is clients["small"]
    state = SimpleNamespace(agent=next_agent, resolved_model=next_run)
    assert await PrepareStep()._should_compact(state, None, 12_000)
    state.resolved_model = running
    assert not await PrepareStep()._should_compact(state, None, 12_000)
    clients["small"].context_limit = 32_000
    assert registry.resolve_model(agent.id).context_limit == 32_000
    assert next_run.context_limit == 16_000


@pytest.mark.asyncio
async def test_agents_in_same_session_have_independent_budgets():
    registry = _reg(
        lambda account, model, **_context: _Client(
            context_limit=100_000 if model == "a" else 20_000, output_reserve=1000
        )
    )
    agents = [
        await registry.instantiate(
            template_id=TPL, session_id="s1", tenant_id="default", llm=ModelChoice(model=model)
        )
        for model in ("a", "b")
    ]
    assert [registry.resolve_model(agent.id).context_limit for agent, _ in agents] == [
        100_000,
        20_000,
    ]


def test_output_floor_never_exceeds_hard_ceiling_or_remaining_window():
    resolved = model_snapshot(1000, 100, 50)
    request = LLMRequest(model="m", system="", messages=[], prompt_token_estimate=990)
    apply_dynamic_max_tokens(None, request, resolved)
    assert request.config.max_tokens == 10
    request = replace(request, config=ModelCallConfig(max_tokens=999))
    apply_dynamic_max_tokens(None, request, resolved)
    assert request.config.max_tokens == 10
    with pytest.raises(ContextOverflowError):
        apply_dynamic_max_tokens(None, replace(request, prompt_token_estimate=1000), resolved)


def test_output_policy_comes_from_budget_even_if_runtime_config_disagrees():
    request = LLMRequest(model="m", system="", messages=[], prompt_token_estimate=100)
    resolved = model_snapshot(100_000, 1000, output_policy=OutputPolicy(output_ratio=0.1))
    ctx = SimpleNamespace(config=SimpleNamespace(dynamic_max_tokens_output_ratio=0.9))
    apply_dynamic_max_tokens(ctx, request, resolved)
    assert request.config.max_tokens == 10_000


def test_zero_policy_minima_still_allocate_positive_bounded_output():
    request = LLMRequest(model="m", system="", messages=[], prompt_token_estimate=99)
    resolved = model_snapshot(
        100, 0, output_policy=OutputPolicy(floor=0, output_ratio=0, output_min=0)
    )
    apply_dynamic_max_tokens(None, request, resolved)
    assert request.config.max_tokens == 1
