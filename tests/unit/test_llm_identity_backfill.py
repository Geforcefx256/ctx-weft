"""全新会话经真实创建路径跑任务，LLM 请求/响应事件须携带实际解析出的 model/account，
而非 "mock" 兜底。

背景（批次 B 之前）：resolve_llm_identity 以 session.llm_model 为真值，但三条创建路径
（run_single_task / SessionRegistry.create_session / resume_session）从不写该字段——
host 即便显式传了 model，事件仍恒报 "mock"；依赖账号 default_model 时更无处可读。
当时的修法是 _execute_task 执行前把解析出的 client 身份**回填**进 session 空缺字段。

批次 B 把回填删了：真值改住 `AgentLifecycleManager._AgentRecord.llm`（`ModelChoice`，可空 =
跟随账号默认），派发时经 `resolve_model` 现解出 `ResolvedModel`（client + 身份 + 窗口），
直接进 `LoopState.resolved_model`，不再经过 session 这一站——回填的副作用是把
`("", "")`「跟随账号默认」这个含义钉成具体模型名，账号默认从此对该会话失效。
本文件不再断言 session.llm_model/llm_provider 被回填；只断言事件本身携带的身份正确。
"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.protocols import (
    AgentCapability,
    AgentCapabilityProvider,
    AgentTemplate,
    CapabilityProviderInfo,
    IdentityFacet,
    LoopConfig,
    MemoryConfig,
    ProviderContext,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm import ModelCapabilities, ModelLimits
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.llm.provider import LLMAccount, LLMProvider, ModelConfig, _FixedModelClient
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import make_runtime


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")

# ── minimal fakes（自包含，避免跨测试文件 import）─────────────────────────────


class _InlineAgentTemplateProvider(AgentCapabilityProvider):
    name = "agent"

    def __init__(self) -> None:
        self._templates: dict[str, AgentTemplate] = {}

    def register(self, template: AgentTemplate) -> None:
        self._templates[template.id] = template

    async def list(self, ctx: ProviderContext) -> list[AgentCapability]:
        return []

    async def get_template(
        self, template_id: str, version, ctx: ProviderContext
    ) -> AgentTemplate | None:
        return self._templates.get(template_id)

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=len(self._templates))


def _echo_template() -> AgentTemplate:
    return AgentTemplate(
        id="tpl_echo",
        name="echo_agent",
        version="0.1.0",
        identity={
            "act": IdentityFacet(text="Echo the user."),
            "observe": IdentityFacet(text="Evaluate the echo."),
        },
        description="identity backfill test agent",
        capability_refs=[],
        memory_config=MemoryConfig(),
        loop_config=LoopConfig(),
    )


class _FakeStore:
    def save(self, account) -> None: ...
    def delete(self, name) -> bool:
        return True

    def list_all(self) -> list:
        return []


class _OneAccountResolver:
    """镜像 LLMProvider.get_client 的默认解析：account/model 缺省时落到唯一账号+默认模型。"""

    def __init__(
        self, adapter, *, account: str = "acct-main", default_model: str = "real-model-x"
    ) -> None:
        self._adapter = adapter
        self._account = account
        self._default_model = default_model

    def get_client(self, account=None, model=None, *, ctx):
        return _FixedModelClient(
            self._adapter,
            model or self._default_model,
            128_000,
            8_192,
            account=account or self._account,
        )


def _make_runtime(resolver_llm: _OneAccountResolver) -> CtxWeftRuntime:
    templates = _InlineAgentTemplateProvider()
    templates.register(_echo_template())
    runtime = make_runtime(agent_provider=templates)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    runtime.providers.register_llm_provider(resolver_llm)
    return runtime


def _collect_llm_events(runtime: CtxWeftRuntime) -> list:
    events: list = []

    async def _cap(ev) -> None:
        if ev.type in (EventType.LLM_REQUEST_STARTED, EventType.LLM_RESPONSE_FINISHED):
            events.append(ev)

    runtime.event_bus.subscribe(None, _cap)
    return events


# ── _FixedModelClient / LLMProvider 暴露实际解析身份 ──────────────────────────


def test_fixed_model_client_exposes_identity():
    client = _FixedModelClient(
        MockLLMAdapter(responses=[]),
        "real-model-x",
        128_000,
        8_192,
        account="acct-main",
    )
    assert client.model == "real-model-x"
    assert client.account == "acct-main"


def test_provider_get_client_carries_resolved_identity():
    p = LLMProvider(_FakeStore())
    p.register_account(
        LLMAccount(
            name="acct-main",
            api_key="sk",
            base_url="https://x",
            models=[
                ModelConfig(
                    name="real-model-x",
                    limit=ModelLimits(128000, output=None),
                    protocol="openai",
                    capabilities=ModelCapabilities(),
                )
            ],
            default_model="real-model-x",
        ),
        persist=False,
    )
    client = p.get_client(None, None, ctx=_PROVIDER_CONTEXT)  # 双缺省 → 唯一账号 + default_model
    assert client.model == "real-model-x"
    assert client.account == p.get_account("acct-main").resource_id


# ── run_single_task：真实创建路径的事件真值 ───────────────────────────────────


@pytest.mark.asyncio
async def test_events_carry_resolved_default_model():
    """host 不显式传 model（依赖账号 default_model）→ 事件报实际解析出的模型，非 mock。"""
    # `observer_verdict="success"`：actor 回一段正文就收尾，而 2026-09-28 起纯文本不再被
    # 机械判 success（`_mechanical_verdict` 判 continue）——要让它终结得让 observer 去判。
    resolver = _OneAccountResolver(
        MockLLMAdapter(responses=[MockResponse(text="hi")], observer_verdict="success")
    )
    runtime = _make_runtime(resolver)
    events = _collect_llm_events(runtime)

    await initialize_all(runtime)
    _handle, state = await runtime.run_single_task(
        template_id="agent:tpl_echo",
        user_prompt="say hello",
    )

    assert state.task.status == "FINISHED"
    started = [e for e in events if e.type == EventType.LLM_REQUEST_STARTED]
    finished = [e for e in events if e.type == EventType.LLM_RESPONSE_FINISHED]
    assert started and finished
    for ev in started:
        assert ev.payload["model"] == "real-model-x", ev.payload
        assert ev.payload["llm_account"] == "acct-main", ev.payload
    for ev in finished:
        assert ev.payload["llm_model"] == "real-model-x", ev.payload
        assert ev.payload["llm_account"] == "acct-main", ev.payload


@pytest.mark.asyncio
async def test_events_carry_explicit_model():
    """host 显式传 model/account → 事件按传入值计账。"""
    # `observer_verdict="success"`：actor 回一段正文就收尾，而 2026-09-28 起纯文本不再被
    # 机械判 success（`_mechanical_verdict` 判 continue）——要让它终结得让 observer 去判。
    resolver = _OneAccountResolver(
        MockLLMAdapter(responses=[MockResponse(text="hi")], observer_verdict="success")
    )
    runtime = _make_runtime(resolver)
    events = _collect_llm_events(runtime)

    await initialize_all(runtime)
    await runtime.run_single_task(
        template_id="agent:tpl_echo",
        user_prompt="say hello",
        llm_account="acct-b",
        llm_model="explicit-y",
    )

    started = [e for e in events if e.type == EventType.LLM_REQUEST_STARTED]
    assert started
    for ev in started:
        assert ev.payload["model"] == "explicit-y", ev.payload
        assert ev.payload["llm_account"] == "acct-b", ev.payload


# ── 裸 adapter 直传（无 LLMProvider，host env 自举场景）───────────────────────


def test_bare_adapters_expose_model():
    """AnthropicAdapter/OpenAICompatibleAdapter 公开 model：host 把 env 模型构造进裸 adapter
    直传 llm= 时，回填才有真值可读（实际调用一直用 _model 替换 "mock"，账面须同源）。"""
    from ctx_weft.providers.llm.adapters.anthropic import AnthropicAdapter
    from ctx_weft.providers.llm.adapters.openai_compatible import OpenAICompatibleAdapter

    assert AnthropicAdapter(api_key="k", model="env-model-a").model == "env-model-a"
    assert OpenAICompatibleAdapter(api_key="k", model="env-model-b").model == "env-model-b"


@pytest.mark.asyncio
async def test_events_carry_bare_adapter_model():
    """host 不走 provider、直传带 model 的裸 adapter → 事件报 adapter 配置的模型。"""

    class _ModelMock(MockLLMAdapter):
        @property
        def model(self) -> str:
            return "env-model-a"

    templates = _InlineAgentTemplateProvider()
    templates.register(_echo_template())
    runtime = make_runtime(
        llm=_ModelMock(responses=[MockResponse(text="hi")]),
        agent_provider=templates,
    )
    runtime.providers.register_memory(InMemoryMemoryProvider())
    events = _collect_llm_events(runtime)

    await initialize_all(runtime)
    _handle, _state = await runtime.run_single_task(
        template_id="agent:tpl_echo",
        user_prompt="say hello",
    )

    started = [e for e in events if e.type == EventType.LLM_REQUEST_STARTED]
    assert started
    for ev in started:
        assert ev.payload["model"] == "env-model-a", ev.payload
