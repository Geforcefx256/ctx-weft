"""create_session must emit SESSION_CREATED before TASK_CREATED.

Regression: the host projection inserts the task row with a FK to sessions.id. If
TASK_CREATED is projected before SESSION_CREATED, the FK fails and the root task row
is lost from the read model (sqlite3.IntegrityError: FOREIGN KEY constraint failed).
The causal order must be Session → Agent → Task.
"""

from __future__ import annotations

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.protocols.events import EventType
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.registry import ProviderRegistry
from ctx_weft.protocols import (
    AgentCapabilityProvider,
    AgentTemplate,
    CapabilityProviderInfo,
    IdentityFacet,
    LoopConfig,
    MemoryConfig,
)


class _Resolver(AgentCapabilityProvider):
    name = "agent"

    def __init__(self, t: AgentTemplate) -> None:
        self._t = t

    async def list(self, ctx):
        return []

    async def get_template(self, template_id, version, ctx):
        return self._t

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name)


class _Client:
    """model_resolver 的桩返回值——构造期注入、无默认值，测试替身须显式给出。"""

    def __init__(self, account="acct_default", model="mdl_default",
                 context_limit=200_000, output_reserve=8192):
        self.account, self.model = account, model
        self.context_limit, self.output_reserve = context_limit, output_reserve


def _template() -> AgentTemplate:
    return AgentTemplate(
        id="tpl", name="t", version="1",
        identity={"act": IdentityFacet(text="soul")},
        capability_refs=[], memory_config=MemoryConfig(), loop_config=LoopConfig(),
    )


async def test_session_created_emitted_before_task_created() -> None:
    bus = InProcessEventBus()
    order: list[str] = []

    async def rec(ev):
        order.append(ev.type)

    bus.subscribe(None, rec)
    resolver = _Resolver(_template())
    _reg = ProviderRegistry()
    _reg.register_capability(resolver)
    sm = SessionRegistry(
        agent_lifecycle_manager=AgentLifecycleManager(
            template_lookup=TemplateLookup(_reg), event_bus=bus,
            model_resolver=lambda a, m, **_context: _Client(),
        ),
        event_bus=bus,
    )
    await sm.create_session(template_id="agent:tpl", user_prompt="你好")

    assert EventType.SESSION_CREATED in order
    assert EventType.AGENT_INSTANTIATED in order
    assert EventType.TASK_CREATED in order
    # 完整因果序：Session → Agent → Task（docstring 顶部所述）。
    assert (
        order.index(EventType.SESSION_CREATED)
        < order.index(EventType.AGENT_INSTANTIATED)
        < order.index(EventType.TASK_CREATED)
    )
