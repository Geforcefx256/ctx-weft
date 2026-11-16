"""D1 焊死：换模型在重放里必须存在（Task 9 必办 2）。

Task 8 评审指出这条链逐行读码是通的，但没有测试焊死；Task 8 结束时
`AGENT_LLM_CHANGED` 还没有真实发射点，只能手工捏造事件。Task 9 给了真实发射点
（`AgentLifecycleManager.set_agent_llm`），本文件走真实路径钉住整条链：

    set_agent_llm(...)  →  AgentLlmChanged 事件  →  reduce_events 折叠成 AgentView
                        →  load()  →  _AgentRecord.llm  →  resolve_model() 拿到新 choice
"""
from __future__ import annotations

import pytest

from ctx_weft.core.control.reducers import reduce_events
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
    def __init__(self, account: str, model: str) -> None:
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = 200_000, 8192


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, ev) -> None:
        self.events.append(ev)


def _make_registry(bus) -> AgentLifecycleManager:
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    providers = ProviderRegistry()
    providers.register_capability(provider)
    return AgentLifecycleManager(
        template_lookup=TemplateLookup(providers),
        event_bus=bus,
        # 拿账号/模型现造一个 client——校验 resolve_model 真的把新 choice 传下去。
        model_resolver=lambda account, model, **_context: _Client(account, model),
    )


@pytest.mark.asyncio
async def test_llm_switch_survives_a_full_replay_and_reload():
    """真实路径，不是手工捏造事件：instantiate → set_agent_llm → 重放 → load → resolve_model。"""
    origin_bus = _RecordingBus()
    origin_registry = _make_registry(origin_bus)
    origin_registry.register_session("s1", tenant_id="default")

    agent, _ = await origin_registry.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default",
    )
    # 起初跟随账号默认（instantiate 不传 llm → ModelChoice()）。
    assert origin_registry.resolve_model(agent.id).account == ""

    changed = await origin_registry.set_agent_llm(
        agent.id, ModelChoice(account="acct-new", model="model-new"),
        reason="user_selected",
    )
    assert changed is True

    # ── 重放：只用 event_bus 上真实产生的事件，不手工捏造 ──────────────────────
    view = reduce_events(origin_bus.events, run_id="run_replay")
    assert view.agents[agent.id].llm_account == "acct-new"
    assert view.agents[agent.id].llm_model == "model-new"

    # ── 新进程视角：一个全新的、空的 registry 只靠 load() 喂入 ──────────────────
    fresh_bus = _RecordingBus()
    fresh_registry = _make_registry(fresh_bus)
    n = await fresh_registry.load(
        view.agents, session_id="s1", tenant_id="default",
    )
    assert n == 1

    resolved = fresh_registry.resolve_model(agent.id)
    assert (resolved.account, resolved.model) == ("acct-new", "model-new")
    # Task 11（2026-09-04 spec §6.4）起：load() 装填一个此前 registry 里没有的
    # agent（真正的冷装填，这里正是——`fresh_registry` 全新、`_agents` 是空的）
    # 之后会按折出来的现状广播一条 AGENT_*，取代旧断言「load() 纯读、不广播任何
    # 事件」——那条断言钉的正是本 task 要修的缺口（重启后 host 投影冻在崩溃前
    # 状态,直到某条冷路径偶然把 ALM 重新装填）。这里没设置 llm 之外的任何状态
    # 事件,AgentView.status 因此是字段默认值 "idle" → 广播 AgentIdle。
    assert fresh_bus.events == []  # Materialization must not replay durable facts onto the bus.


@pytest.mark.asyncio
async def test_session_llm_switch_also_survives_replay_for_every_agent():
    """`set_session_llm` 的 N 条事件在重放折叠后，每个 agent 都拿到新 choice。"""
    bus = _RecordingBus()
    registry = _make_registry(bus)
    registry.register_session("s1", tenant_id="default")
    a, _ = await registry.instantiate(template_id=TPL, session_id="s1", tenant_id="default")
    b, _ = await registry.instantiate(template_id=TPL, session_id="s1", tenant_id="default")

    n = await registry.set_session_llm(
        "s1", ModelChoice(account="acct-batch", model="model-batch"), reason="user_selected",
    )
    assert n == 2

    view = reduce_events(bus.events, run_id="run_replay")
    fresh_registry = _make_registry(_RecordingBus())
    await fresh_registry.load(
        view.agents, session_id="s1", tenant_id="default",
    )
    for aid in (a.id, b.id):
        resolved = fresh_registry.resolve_model(aid)
        assert (resolved.account, resolved.model) == ("acct-batch", "model-batch")
