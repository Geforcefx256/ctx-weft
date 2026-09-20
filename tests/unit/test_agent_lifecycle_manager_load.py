"""恢复是「喂进来」——recover_agent 折出 AgentView，显式装填进 Registry。

替代 tests/unit/test_agents_from_projection.py：那个函数已并入 load()。
行为差异（有意的）：装填出来的 record 带**真正的 template 配置**，
而不是 agents_from_projection 留下的 dataclass 默认值。
"""
from __future__ import annotations

import pytest

from ctx_weft.core.control.types import AgentView
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.registry import ProviderRegistry
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
)

pytestmark = pytest.mark.asyncio

TPL = "agent:tpl_echo"


class _Bus:
    """本文件测的是 load() 的装填结果，不关心事件。"""

    async def emit(self, ev) -> None:
        return None


class _Client:
    """model_resolver 的桩返回值——构造期注入、无默认值，测试替身须显式给出。"""

    def __init__(self, account="acct_default", model="mdl_default",
                 context_limit=200_000, output_reserve=8192):
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = context_limit, output_reserve


def _lm() -> AgentLifecycleManager:
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    providers = ProviderRegistry()
    providers.register_capability(provider)
    return AgentLifecycleManager(
        template_lookup=TemplateLookup(providers=providers), event_bus=_Bus(),
        model_resolver=lambda a, m: _Client(),
    )


async def test_load_fills_records_from_views():
    lm = _lm()
    views = {
        "agt_a": AgentView(id="agt_a", spawn_depth=0, parent_agent_id=None,
                           template_id=TPL),
        "agt_b": AgentView(id="agt_b", spawn_depth=1, parent_agent_id="agt_a",
                           template_id=TPL),
    }
    n = await lm.load(views, session_id="s1", tenant_id="default",
                      fallback_template_id=TPL)
    assert n == 2
    assert lm.has("agt_a") and lm.has("agt_b")
    assert lm._agents["agt_b"].spawn_depth == 1
    assert lm._agents["agt_b"].parent_agent_id == "agt_a"


async def test_load_resolves_template_config_not_dataclass_defaults():
    """行为变化，且是变正确了：agents_from_projection 留的是 dataclass 默认值。"""
    lm = _lm()
    tmpl = make_echo_template()
    views = {"agt_a": AgentView(id="agt_a", template_id=TPL)}
    await lm.load(views, session_id="s1", tenant_id="default",
                  fallback_template_id=TPL)
    got, _rm = lm.materialize("agt_a", session_id="s1", tenant_id="default")
    assert got.memory_config == tmpl.memory_config
    assert got.loop_config == tmpl.loop_config


async def test_empty_template_id_falls_back():
    """存量事件流里子 agent 没发过 AgentInstantiated → template_id 为空。

    回落而非报错是刻意的：授权按模板做策略，重启后把未知模板判成
    「无权限」会让老会话直接跑不动。
    """
    lm = _lm()
    views = {"agt_a": AgentView(id="agt_a", template_id="")}
    await lm.load(views, session_id="s1", tenant_id="default",
                  fallback_template_id=TPL)
    assert lm.template_id_of("agt_a")


async def test_load_is_idempotent():
    lm = _lm()
    views = {"agt_a": AgentView(id="agt_a", template_id=TPL)}
    await lm.load(views, session_id="s1", tenant_id="default",
                  fallback_template_id=TPL)
    await lm.load(views, session_id="s1", tenant_id="default",
                  fallback_template_id=TPL)
    assert len(lm._agents) == 1


async def test_unresolvable_template_id_falls_back_to_default_config_not_raises():
    """恢复期缺口降级，不抛——模板查不到时用 MemoryConfig()/LoopConfig() 默认值继续。"""
    lm = _lm()
    views = {"agt_a": AgentView(id="agt_a", template_id="agent:does_not_exist")}
    n = await lm.load(views, session_id="s1", tenant_id="default",
                      fallback_template_id=TPL)
    assert n == 1
    assert lm.has("agt_a")
