"""instantiate（真新建）与 materialize（水合）是两件事。

拆开之前它们是同一个方法的两种模式，靠 existing_agent_id 是否为 None 区分，
而区分的结果只有调用方知道 —— 那正是 agent 域事件散落在 runtime 里的原因。
"""
from __future__ import annotations

import pytest

from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
)

pytestmark = pytest.mark.asyncio

TPL = "agent:tpl_echo"


class _Bus:
    """本文件测的是 instantiate/materialize 的返回值语义，不关心事件。"""

    async def emit(self, ev) -> None:
        return None


class _Client:
    def __init__(self, account="acct_default", model="mdl_default",
                 context_limit=200_000, output_reserve=8192):
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = context_limit, output_reserve


def _lm() -> AgentLifecycleManager:
    from ctx_weft.core.registry import ProviderRegistry

    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    providers = ProviderRegistry()
    providers.register_capability(provider)
    lm = AgentLifecycleManager(
        template_lookup=TemplateLookup(providers=providers),
        event_bus=_Bus(),
        model_resolver=lambda a, m: _Client(),
    )
    lm.register_session("s1", tenant_id="default", fallback_template_id=TPL)
    return lm


async def test_instantiate_has_no_existing_agent_id_param():
    import inspect
    sig = inspect.signature(AgentLifecycleManager.instantiate)
    assert "existing_agent_id" not in sig.parameters


async def test_materialize_carries_template_config():
    lm = _lm()
    agent, tmpl = await lm.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    got, rm = lm.materialize(          # 不再收窗口参数
        agent.id, session_id=agent.session_id, tenant_id=agent.tenant_id)
    assert got.id == agent.id
    assert got.template_id == tmpl.id
    # 从 template 来，不是 dataclass 默认 —— 这修掉了 agents_from_projection 的旧行为
    assert got.memory_config == tmpl.memory_config
    assert got.loop_config == tmpl.loop_config
    # 窗口从 client 派生（批次 B）
    assert got.loop_guard.context_limit == rm.context_limit
    assert got.loop_guard.reserved_output_tokens == rm.reserved_output_tokens


async def test_materialize_is_a_fresh_object_each_call():
    """Agent 带一次 run 的可变量（loop_guard.context_tokens 由 act.py 改写），
    所以每次派发产出新实例是正确的，不是浪费。"""
    lm = _lm()
    agent, _ = await lm.instantiate(
        template_id=TPL, session_id="s1", tenant_id="default")
    a, _ = lm.materialize(
        agent.id, session_id=agent.session_id, tenant_id=agent.tenant_id)
    b, _ = lm.materialize(
        agent.id, session_id=agent.session_id, tenant_id=agent.tenant_id)
    assert a is not b


async def test_materialize_unknown_id_falls_back_and_never_raises(caplog):
    """恢复期缺口降级，不抛 —— 与 agents_from_projection 的既有口径一致。

    `session_id` / `tenant_id` 是**调用方给的**（2026-09-19 起必传）：补出来的占位记录按
    它们归属，registry 不再去猜「最近一次 register_session 的会话」。`template_id` 仍取
    该 session 登记项里的 `fallback_template_id`——那是 registry 自己的账。
    """
    lm = _lm()
    got, _rm = lm.materialize("agt_never_seen", session_id="s1", tenant_id="default")
    assert got.id == "agt_never_seen"
    assert got.session_id == "s1"     # 按调用方给的归属，不是猜的
    assert got.template_id  # 回落到 session 的 fallback_template_id
    assert lm.has("agt_never_seen")  # 就地补登记，第二次不再警告


async def test_materialize_honours_the_given_session_not_the_most_recent_one():
    """撞上未登记 id 时按**调用方给的** session 归属，不是「最近一次 register_session」。

    2026-09-19 之前 `_register_fallback` 在无语境时回落
    `next(reversed(self._sessions.items()))`。那条回落在同一进程处理过多个会话时会把 agent
    连同它的 tenant 归到**别的** session 上——而且 `register_session` 用 `setdefault`，已
    登记的会话不会被移到末尾，所以「最近一次」跟「当前正在处理的这次」并不是一回事：
    先处理 A、再处理 B、然后回头对 A 做 compact，猜出来的就是 B。

    后果不止 tenant：`_AgentRecord.session_id` 一起被猜错，于是这一次派发/装配拿到的
    `Agent` 对象归属错会话，用 `rec.tenant_id` 发出的 AGENT_* 事件也带着错 tenant 落库。
    """
    lm = _lm()                                  # 已登记 s1
    lm.register_session("s_older", tenant_id="tenant-older", fallback_template_id=TPL)
    lm.register_session("s_newer", tenant_id="tenant-newer", fallback_template_id=TPL)
    # 回头处理较早那个（register_session 是 setdefault，不会把 s_older 移到末尾）
    lm.register_session("s_older", tenant_id="tenant-older", fallback_template_id=TPL)

    got, _rm = lm.materialize(
        "agt_unregistered", session_id="s_older", tenant_id="tenant-older")

    assert got.session_id == "s_older", "必须按调用方给的会话归属"
    assert got.tenant_id == "tenant-older", "tenant 同理——不得落到最近登记的那个会话上"
    rec = lm.record_of("agt_unregistered")
    assert (rec.session_id, rec.tenant_id) == ("s_older", "tenant-older")


async def test_materialize_requires_the_session_context():
    """`session_id` / `tenant_id` 必传——漏传要在签名层被挡住，而不是让 registry 去猜。

    做成可选参数的代价就是上面那条用例描述的猜法会悄悄回来；用 inspect 钉死它。
    """
    import inspect

    params = inspect.signature(AgentLifecycleManager.materialize).parameters
    for name in ("session_id", "tenant_id"):
        assert name in params, f"materialize 必须收 {name}"
        assert params[name].default is inspect.Parameter.empty, (
            f"{name} 不得有默认值——省略它只能靠猜")
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY


async def test_unregistered_session_still_honours_the_given_tenant():
    """连 session 都没登记过（更深的恢复缺口）：tenant 仍用调用方给的，模板留空。"""
    lm = _lm()

    got, _rm = lm.materialize(
        "agt_x", session_id="s_never_registered", tenant_id="tenant-from-caller")

    assert got.session_id == "s_never_registered"
    assert got.tenant_id == "tenant-from-caller"
    assert got.template_id == "", "session 没登记 → 没有 fallback 模板可用，留空而不是猜"
