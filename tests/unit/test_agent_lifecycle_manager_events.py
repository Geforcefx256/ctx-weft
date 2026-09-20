"""agent 域四条事件的唯一发射点是 AgentLifecycleManager。

搬迁之前：AgentInstantiated 由 session_registry.py:233（root）与
runtime.py:2713（子 agent）两处发，AgentSpawned/SpawnRejected 在
runtime.py —— 而深度判定（SpawnDepthExceeded）本来就在 LM 里。
判定在这边、事件在那边。
"""
from __future__ import annotations

import pytest

from ctx_weft.core.orchestrator.lifecycle.agent_manager import (
    AgentLifecycleManager,
    SpawnDepthExceeded,
)
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.protocols.events import EventType
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
)

pytestmark = pytest.mark.asyncio

TPL = "agent:tpl_echo"


class _Bus:
    def __init__(self):
        self.events = []

    async def emit(self, ev):
        self.events.append(ev)

    def types(self):
        return [e.type for e in self.events]


class _Client:
    """model_resolver 的桩返回值——构造期注入、无默认值，测试替身须显式给出。"""

    def __init__(self, account="acct_default", model="mdl_default",
                 context_limit=200_000, output_reserve=8192):
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = context_limit, output_reserve


def _lm(bus, *, max_depth=3):
    import dataclasses as _dc

    from ctx_weft.core.registry import ProviderRegistry

    tmpl = make_echo_template()
    tmpl = _dc.replace(tmpl, loop_config=_dc.replace(tmpl.loop_config, max_spawn_depth=max_depth))
    provider = InlineAgentTemplateProvider()
    provider.register(tmpl)
    providers = ProviderRegistry()
    providers.register_capability(provider)
    lm = AgentLifecycleManager(
        template_lookup=TemplateLookup(providers=providers), event_bus=bus,
        model_resolver=lambda a, m: _Client(),
    )
    lm.register_session("s1", tenant_id="default", fallback_template_id=TPL)
    return lm


async def test_root_emits_only_instantiated():
    bus = _Bus()
    lm = _lm(bus)
    await lm.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    assert bus.types() == [EventType.AGENT_INSTANTIATED]


async def test_child_emits_spawned_then_instantiated():
    """因果顺序：先记「这次 spawn 被准了」，再记「诞生的 agent 长这样」。"""
    bus = _Bus()
    lm = _lm(bus)
    root, _ = await lm.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    bus.events.clear()
    await lm.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default",
        parent_agent_id=root.id, task_id="tsk_1")
    assert bus.types() == [EventType.AGENT_SPAWNED, EventType.AGENT_INSTANTIATED]


async def test_depth_exceeded_emits_spawn_rejected_and_raises():
    """SpawnRejected 的 envelope agent_id 填**父**——子 agent 没诞生，没有 id 可填。"""
    bus = _Bus()
    lm = _lm(bus, max_depth=0)
    root, _ = await lm.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    bus.events.clear()
    with pytest.raises(SpawnDepthExceeded):
        await lm.instantiate(
            template_id=TPL, session_id="s1", tenant_id="default",
            parent_agent_id=root.id, task_id="tsk_1")
    assert bus.types() == [EventType.SPAWN_REJECTED]
    assert bus.events[0].agent_id == root.id


async def test_materialize_emits_nothing():
    bus = _Bus()
    lm = _lm(bus)
    agent, _ = await lm.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    bus.events.clear()
    lm.materialize(agent.id, session_id=agent.session_id, tenant_id=agent.tenant_id)
    assert bus.events == []
