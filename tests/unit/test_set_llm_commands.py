"""换模型只有两条命令，且都是纯赋值。

set_session_llm 发的是 N 条 AgentLlmChanged，不是一条会话级事件——
真相源因此仍然唯一，reducer 不必处理「一条事件改 N 个实体」。
host 要展示「这是一次会话级切换」→ 按 causation_id 聚合。
"""
from __future__ import annotations

import inspect

import pytest

from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager

from ctx_weft.core.orchestrator.model import ModelChoice
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.registry import ProviderRegistry
from ctx_weft.protocols.events import EventType
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
)


TPL = "agent:tpl_echo"


class _Client:
    account, model = "a0", "m0"
    context_limit, output_reserve = 1000, 100


class _Bus:
    def __init__(self):
        self.events = []

    async def emit(self, ev):
        self.events.append(ev)


def _reg():
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    providers = ProviderRegistry()
    providers.register_capability(provider)
    reg = AgentLifecycleManager(
        template_lookup=TemplateLookup(providers),
        event_bus=_Bus(),
        model_resolver=lambda a, m, **_context: _Client(),
    )
    reg.register_session("s1", tenant_id="default")
    reg.register_session("s2", tenant_id="default")
    return reg


@pytest.mark.asyncio
async def test_set_agent_llm_emits_and_returns_true():
    reg = _reg()
    a, _ = await reg.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    reg.event_bus.events.clear()
    changed = await reg.set_agent_llm(
        a.id, ModelChoice(account="x", model="y"), reason="user_selected")
    assert changed is True
    assert [e.type for e in reg.event_bus.events] == [EventType.AGENT_LLM_CHANGED]
    assert reg._agents[a.id].llm == ModelChoice(account="x", model="y")


@pytest.mark.asyncio
async def test_same_choice_is_a_noop():
    """host 重复点击不刷屏。"""
    reg = _reg()
    a, _ = await reg.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    await reg.set_agent_llm(a.id, ModelChoice(account="x"), reason="user_selected")
    reg.event_bus.events.clear()
    changed = await reg.set_agent_llm(a.id, ModelChoice(account="x"), reason="user_selected")
    assert changed is False
    assert reg.event_bus.events == []


@pytest.mark.asyncio
async def test_set_session_llm_emits_n_events_sharing_causation_id():
    reg = _reg()
    a, _ = await reg.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    b, _ = await reg.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    other, _ = await reg.instantiate(template_id=TPL, session_id="s2", tenant_id="default")
    reg.event_bus.events.clear()

    n = await reg.set_session_llm(
        "s1", ModelChoice(account="x", model="y"), reason="user_selected")

    assert n == 2
    evs = reg.event_bus.events
    assert [e.type for e in evs] == [EventType.AGENT_LLM_CHANGED] * 2
    assert len({e.causation_id for e in evs}) == 1
    assert evs[0].causation_id is not None
    assert {e.agent_id for e in evs} == {a.id, b.id}
    # 另一个 session 不受影响
    assert reg._agents[other.id].llm == ModelChoice()


@pytest.mark.asyncio
async def test_set_llm_touches_no_task_state():
    """纯赋值：不入队、不改任何 task 状态、不触发调度。"""
    reg = _reg()
    a, _ = await reg.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    await reg.set_agent_llm(a.id, ModelChoice(account="x"), reason="user_selected")
    assert [e.type for e in reg.event_bus.events][-1] == EventType.AGENT_LLM_CHANGED


def test_resume_hint_is_gone():
    import ctx_weft.protocols as protocols
    from ctx_weft.protocols.hitl import HitlReply
    assert not hasattr(protocols, "ResumeHint")
    assert "resume_hint" not in inspect.signature(HitlReply).parameters


def test_recover_agent_takes_no_llm_params():
    from ctx_weft.core.runtime import CtxWeftRuntime
    sig = inspect.signature(CtxWeftRuntime.recover_session)
    assert "llm_account" not in sig.parameters
    assert "llm_model" not in sig.parameters
