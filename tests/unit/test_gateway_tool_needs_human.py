"""工具 provider 侧的 needs_human 路径（段 2 · Task 5）。

覆盖：needs_human 是流的终点（其后事件不可见）、resume 拿到 resume_state 且不重算、
reply_as_result 短路（含多模态答复）、未实现 HumanResumable 是契约违例、驱逐→park。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL, HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.hitl_waiter import HitlWaiter
from ctx_weft.core.loop.park import HitlPark
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.hitl import HitlAsk, HitlReply, ToolResultDelivery
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

_AGENT_ID = "agt_1"


class _PassthroughNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        return content, content


class _AsksThenResumes(ToolCapabilityProvider):
    """实现了 HumanResumable：让出前算好 plan，重入时不重算。"""

    name = "deploy"

    def __init__(self) -> None:
        self.computed = 0
        self.applied_with = None

    async def list(self, ctx):
        return [ToolCapability(id="deploy:apply", name="apply", description="deploy")]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._invoke(capability_id, arguments, ctx)

    async def _invoke(self, capability_id, arguments, ctx):
        self.computed += 1
        yield CapabilityEvent(kind="progress", payload={"text": "planning"})
        yield CapabilityEvent(kind="needs_human", payload={"ask": HitlAsk(
            form="question",
            delivery=ToolResultDelivery(tool_call_id=ctx.extra["tool_call_id"]),
            prompt="确认部署？", resume_state={"plan": "deploy-7"})})
        yield CapabilityEvent(kind="result", payload={"content": "SHOULD NOT BE SEEN"})

    def resume(self, ask_id, decision, resume_state, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._resume(ask_id, decision, resume_state, ctx)

    async def _resume(self, ask_id, decision, resume_state, ctx):
        self.applied_with = (resume_state, decision.outcome)
        yield CapabilityEvent(kind="result", payload={"content": "deployed"})

    async def cancel(self, iid, ctx):
        return None


class _AsksAsResult(ToolCapabilityProvider):
    """reply_as_result：答复即结果，不实现 resume。"""

    name = "control"

    async def list(self, ctx):
        return [ToolCapability(id="control:ask_user", name="ask_user", description="ask")]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._invoke(capability_id, arguments, ctx)

    async def _invoke(self, capability_id, arguments, ctx):
        yield CapabilityEvent(kind="needs_human", payload={"ask": HitlAsk(
            form="question",
            delivery=ToolResultDelivery(tool_call_id=ctx.extra["tool_call_id"]),
            prompt="你的名字？", reply_as_result=True)})

    async def cancel(self, iid, ctx):
        return None


class _AsksButNotResumable(ToolCapabilityProvider):
    name = "broken"

    async def list(self, ctx):
        return [ToolCapability(id="broken:x", name="x", description="broken")]

    async def retrieve(self, ctx):
        return await self.list(ctx)

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._invoke(capability_id, arguments, ctx)

    async def _invoke(self, capability_id, arguments, ctx):
        yield CapabilityEvent(kind="needs_human", payload={"ask": HitlAsk(
            form="question",
            delivery=ToolResultDelivery(tool_call_id=ctx.extra["tool_call_id"]),
            prompt="?")})

    async def cancel(self, iid, ctx):
        return None


_CAPS_BY_PROVIDER: dict[type, ToolCapability] = {
    _AsksThenResumes: ToolCapability(id="deploy:apply", name="apply", description="deploy"),
    _AsksAsResult: ToolCapability(id="control:ask_user", name="ask_user", description="ask"),
    _AsksButNotResumable: ToolCapability(id="broken:x", name="x", description="broken"),
}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = getattr(part, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


_shared: dict = {}


def _make_gateway(tool_provider, timeout_sec=None):
    """搭一个 gateway + 一套真实的 HitlRegistry/HitlService/HitlWaiter，装配单个 tool
    provider，与 ``test_gateway_authz_hitl.py`` 的惯例一致（用 qualified 名做入口）。
    """
    registry = HitlRegistry()
    bus = InProcessEventBus()
    service = HitlService(
        registry=registry, event_bus=bus,
        reply_intake=ReplyIntake(_PassthroughNormalizer()),
    )
    waiter = HitlWaiter(registry, timeout_sec=timeout_sec)
    cache = CapabilityCache()
    cache.put(_AGENT_ID, [_CAPS_BY_PROVIDER[type(tool_provider)]])
    gw = CapabilityGateway(
        capability_cache=cache,
        capability_providers=[tool_provider],
        memory=InMemoryMemoryProvider(),
        event_bus=bus,
    )
    _shared["hitl"] = service
    _shared["waiter"] = waiter
    return gw, registry, service


def _state() -> LoopState:
    session = SimpleNamespace(id="s1", tenant_id="default")
    task = SimpleNamespace(id="tsk_1", unattended=False)
    agent = SimpleNamespace(id=_AGENT_ID, template_id="tmpl_a", session_id="s1")
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id=_AGENT_ID)
    return LoopState(run_id="r1", session=session, task=task, agent=agent, scope=scope, resolved_model=SimpleNamespace(model="mock", account=""))


def _ctx() -> LoopContext:
    return LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(),
        event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id=_AGENT_ID,
        ),
        hitl=_shared.get("hitl"),
        waiter=_shared.get("waiter"),
    )


# ── 测试 ────────────────────────────────────────────────────────────────────


async def test_needs_human_stops_stream_consumption_immediately():
    """该事件是流的终点——其后的 result 事件必须不可见。"""
    provider = _AsksThenResumes()
    gw, reg, svc = _make_gateway(provider)
    task = asyncio.create_task(gw.invoke("deploy__apply", {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="accepted",
                                agent_id=_AGENT_ID))
    result = await task
    assert "SHOULD NOT BE SEEN" not in _text_of(result.content)
    assert _text_of(result.content).strip().endswith("deployed")


async def test_resume_gets_the_resume_state_and_does_not_recompute():
    provider = _AsksThenResumes()
    gw, reg, svc = _make_gateway(provider)
    task = asyncio.create_task(gw.invoke("deploy__apply", {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="accepted",
                                agent_id=_AGENT_ID))
    await task
    assert provider.computed == 1                                  # 没重算
    assert provider.applied_with == ({"plan": "deploy-7"}, "accepted")


async def test_reply_as_result_short_circuits_without_reentry():
    gw, reg, svc = _make_gateway(_AsksAsResult())
    task = asyncio.create_task(gw.invoke("control__ask_user", {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="accepted",
                                agent_id=_AGENT_ID, message="小明"))
    result = await task
    assert "小明" in _text_of(result.content)


async def test_reply_as_result_carries_multimodal_answers_through():
    """人的答复带图时，图必须进最终 content，不能被拍成文本。

    ``data`` 用**真** base64：答复与 provider 交的 part 走同一条路（`_ToolStream.parts`），
    因此同样过 gateway 的 `legalize_tool_result_parts`（校验 + 外部化）。这是
    有意为之的兜底——宿主若没接 `set_content_normalizer`，人递进来的字节在入口一次都
    没被校验过。畸形 base64 在那里会被换成占位，那是**正确**行为，不该用假数据绕开。
    """
    from ctx_weft.protocols import ImagePart, TextPart

    gw, reg, svc = _make_gateway(_AsksAsResult())
    task = asyncio.create_task(gw.invoke("control__ask_user", {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(
        hitl_id=reg.list_pending()[0].id, outcome="accepted", agent_id=_AGENT_ID,
        message=[TextPart(text="就这张"),
                 ImagePart(data="QUJDREVG", media_type="image/png",
                           source_type="base64")]))
    result = await task
    assert any(isinstance(p, ImagePart) for p in result.content)


async def test_needs_human_without_resumable_is_a_contract_violation():
    gw, reg, svc = _make_gateway(_AsksButNotResumable())
    task = asyncio.create_task(gw.invoke("broken__x", {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    await svc.resolve(HitlReply(hitl_id=reg.list_pending()[0].id, outcome="accepted",
                                agent_id=_AGENT_ID))
    result = await task
    assert result.is_error is True and "does not implement" in _text_of(result.content)


async def test_eviction_in_the_tool_path_parks_without_a_result():
    gw, _reg, _svc = _make_gateway(_AsksThenResumes(), timeout_sec=0)
    with pytest.raises(HitlPark):
        await gw.invoke("deploy__apply", {}, _state(), _ctx(), tool_call_id="call_1")


async def test_the_ask_is_recorded_under_the_tool_stage():
    """确认走的是 HITL_STAGE_TOOL 而不是 authz 那条——两阶段键必须分开（安全修复的核心）。"""
    provider = _AsksThenResumes()
    gw, reg, svc = _make_gateway(provider)
    task = asyncio.create_task(gw.invoke("deploy__apply", {}, _state(), _ctx(),
                                         tool_call_id="call_1"))
    await asyncio.sleep(0)
    pending = reg.list_pending()[0]
    assert pending.stage == HITL_STAGE_TOOL
    await svc.resolve(HitlReply(hitl_id=pending.id, outcome="accepted", agent_id=_AGENT_ID))
    await task
