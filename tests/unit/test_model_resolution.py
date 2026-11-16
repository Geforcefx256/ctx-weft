"""「用哪个 client」和「窗口多大」是同一次解析的两面。

LLMClient 协议本来就把 context_limit / output_reserve 定义成抽象属性
（protocols/llm.py:270-279），不是 duck-type 的额外物——所以窗口永远
从 client 现读，一次都不必存。
"""
from __future__ import annotations

import pytest

from ctx_weft.core.orchestrator.lifecycle.agent_manager import (
    AgentLifecycleManager,
    ModelChoice,
)
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
)


TPL = "agent:tpl_echo"


class _Client:
    def __init__(self, account="acct_default", model="mdl_default",
                 context_limit=200_000, output_reserve=8192):
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = context_limit, output_reserve


class _Bus:
    def __init__(self):
        self.events = []

    async def emit(self, ev):
        self.events.append(ev)


def _reg(resolver=None):
    from ctx_weft.core.registry import ProviderRegistry

    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    providers = ProviderRegistry()
    providers.register_capability(provider)
    reg = AgentLifecycleManager(
        template_lookup=TemplateLookup(providers=providers),
        event_bus=_Bus(),
        model_resolver=resolver or (lambda a, m, **_context: _Client()),
    )
    reg.register_session("s1", tenant_id="default")
    return reg


@pytest.mark.asyncio
async def test_empty_choice_takes_identity_from_client():
    """("", "") 的含义是「跟随账号默认」——身份由 client 报，不回写 choice。"""
    reg = _reg()
    agent, _ = await reg.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    rm = reg.resolve_model(agent.id)
    assert (rm.account, rm.model) == ("acct_default", "mdl_default")
    assert reg._agents[agent.id].llm == ModelChoice()   # 选择仍是空，未被冻结


@pytest.mark.asyncio
async def test_explicit_choice_wins():
    reg = _reg()
    agent, _ = await reg.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default",
        llm=ModelChoice(account="acct_x", model="mdl_x"))
    rm = reg.resolve_model(agent.id)
    assert (rm.account, rm.model) == ("acct_x", "mdl_x")


@pytest.mark.asyncio
async def test_window_comes_from_client_in_immutable_snapshot():
    reg = _reg(resolver=lambda a, m, **_context: _Client(context_limit=42, output_reserve=7))
    agent, _ = await reg.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    got, rm = reg.materialize(
        agent.id, session_id=agent.session_id, tenant_id=agent.tenant_id)
    assert rm.context_limit == 42
    assert not hasattr(got.loop_guard, "context_limit")
    assert rm.reserved_output_tokens == 7


@pytest.mark.asyncio
async def test_child_inherits_parent_choice():
    """决定①：继承**派生它的那个 agent**，不是 root。"""
    reg = _reg()
    root, _ = await reg.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default",
        llm=ModelChoice(account="a1", model="m1"))
    child, _ = await reg.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default",
        parent_agent_id=root.id, task_id="tsk_1")
    assert reg._agents[child.id].llm == ModelChoice(account="a1", model="m1")


@pytest.mark.asyncio
async def test_registry_does_not_cache_the_client():
    """缓存客户端是 LLMClientResolver 的职责；Registry 存第二份就有第二个失效问题。"""
    calls = []

    def resolver(a, m, *, ctx):
        calls.append((a, m))
        return _Client()

    reg = _reg(resolver=resolver)
    agent, _ = await reg.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    calls.clear()
    reg.resolve_model(agent.id)
    reg.resolve_model(agent.id)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_resolution_context_belongs_to_each_agent_and_survives_load():
    seen = []

    def resolve(account, model, *, ctx):
        seen.append(ctx)
        return _Client()

    reg = _reg(resolve)
    agents = []
    for sid, tenant in [("sa", "user-a:cowork-a"), ("sb", "user-b:cowork-b")]:
        reg.register_session(sid, tenant_id=tenant)
        agent, _ = await reg.instantiate(template_id=TPL, session_id=sid, tenant_id=tenant)
        agents.append(agent)
    for agent in [agents[0], agents[1], agents[0]]:
        reg.resolve_model(agent.id)
        ctx = seen[-1]
        assert (ctx.session_id, ctx.tenant_id, ctx.agent_id, ctx.agent_template_id) == (
            agent.session_id, agent.tenant_id, agent.id, agent.template_id,
        )
    parent = agents[0]
    child, _ = await reg.instantiate(
        template_id=TPL, session_id=parent.session_id, tenant_id=parent.tenant_id,
        parent_agent_id=parent.id,
    )
    reg.resolve_model(child.id)
    assert seen[-1].agent_id == child.id
    assert seen[-1].tenant_id == parent.tenant_id
    from ctx_weft.core.control.reducers import reduce_events

    view = reduce_events(reg.event_bus.events, "run")
    restored = _reg(resolve)
    await restored.load(view.agents, session_id=parent.session_id, tenant_id=parent.tenant_id)
    restored.resolve_model(parent.id)
    assert seen[-1].session_id == parent.session_id
    assert seen[-1].tenant_id == parent.tenant_id
