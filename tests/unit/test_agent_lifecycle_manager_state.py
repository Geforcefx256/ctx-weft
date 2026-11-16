"""AgentLifecycleManager 持有 agent 的身份与配置——注册表，不是状态机。

镜像 SessionRegistry 的 _SessionState / _states / register_session 形状
（docs/events-v2.md §2.1.1 的那次晋升）。区别：它零订阅。
"""
from __future__ import annotations

import pytest

from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
)



class _Bus:
    """本文件只测 registry 的状态语义，不关心事件——显式哨兵替身，不靠默认值。"""

    async def emit(self, ev) -> None:
        return None


class _Client:
    """model_resolver 的桩返回值——构造期注入、无默认值，测试替身须显式给出。"""

    def __init__(self, account="acct_default", model="mdl_default",
                 context_limit=200_000, output_reserve=8192):
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = context_limit, output_reserve


def _lm() -> AgentLifecycleManager:
    from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
    from ctx_weft.core.registry import ProviderRegistry
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    providers = ProviderRegistry()
    providers.register_capability(provider)
    return AgentLifecycleManager(
        template_lookup=TemplateLookup(providers=providers), event_bus=_Bus(),
        model_resolver=lambda a, m, **_context: _Client(),
    )


@pytest.mark.asyncio
async def test_instantiate_registers_a_record():
    lm = _lm()
    agent, _tmpl = await lm.instantiate(
        template_id="agent:tpl_echo", session_id="s1", tenant_id="default",
    )
    assert lm.has(agent.id)
    assert lm.template_id_of(agent.id) == agent.template_id


@pytest.mark.asyncio
async def test_register_session_is_reentrant():
    lm = _lm()
    lm.register_session("s1", tenant_id="t1")
    lm.register_session("s1", tenant_id="OTHER")
    # 已存在则保留原状态——与 SessionRegistry.register_session 同口径
    assert lm._sessions["s1"].tenant_id == "t1"


@pytest.mark.asyncio
async def test_forget_session_drops_only_that_sessions_agents():
    lm = _lm()
    a, _ = await lm.instantiate(
        template_id="agent:tpl_echo", session_id="s1", tenant_id="default")
    b, _ = await lm.instantiate(
        template_id="agent:tpl_echo", session_id="s2", tenant_id="default")
    lm.forget_session("s1")
    assert not lm.has(a.id)
    assert lm.has(b.id)


@pytest.mark.asyncio
async def test_instantiate_parent_fallback_uses_this_calls_session_not_most_recent():
    """多 session 并发恢复：unregistered 父 agent 的回落语境该取*本次 instantiate
    调用自己*的 session/tenant，不是「最近一次 register_session 的会话」——否则
    在 s2 之后才给 s1 spawn 子 agent 会把 s1 的孩子错记成 s2 的 tenant。
    """
    lm = _lm()
    lm.register_session("s1", tenant_id="tenant_1")
    lm.register_session("s2", tenant_id="tenant_2")
    child, _ = await lm.instantiate(
        template_id="agent:tpl_echo", session_id="s1", tenant_id="tenant_1",
        parent_agent_id="agt_ghost_parent_never_registered",
    )
    assert lm.has("agt_ghost_parent_never_registered")
    assert lm._agents["agt_ghost_parent_never_registered"].session_id == "s1"
    assert lm._agents["agt_ghost_parent_never_registered"].tenant_id == "tenant_1"
    assert lm._agents[child.id].tenant_id == "tenant_1"
