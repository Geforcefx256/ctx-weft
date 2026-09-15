"""Gateway 执行参数与审计参数分离（spec: capability-gateway；上游方案 H4 / WP1）。

三条通道（design D1）：
  original_arguments  调用方入参，全程不被修改（审批指纹 invocation_key 用它）
  effective_arguments 授权/HITL 修改 + schema 校验后的参数——不脱敏，传 Provider
  audit_arguments     对 effective 的脱敏副本——只进事件与 TOOL_AUDIT

基线（修复前）Provider 收到脱敏后的 '***'（探针 H4 实测）——①② 两组在本文件
落地时必须先红，是「现象成立」的回归证据；③④⑤ 钉住既有正确部分。
"""
from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from types import SimpleNamespace

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


class _HeaderEcho(ToolCapabilityProvider):
    """收下参数并深拷贝留证——provider 视角的参数快照。"""

    name = "mcp:web"

    def __init__(self) -> None:
        self.invoked = False
        self.received: dict | None = None

    def _cap(self) -> ToolCapability:
        return ToolCapability(
            id="mcp:web:fetch", name="fetch", description="d",
            input_schema={"type": "object",
                          "properties": {"url": {"type": "string"},
                                         "headers": {"type": "object"}}},
        )

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            self.invoked = True
            self.received = copy.deepcopy(arguments)
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None: return None


class _RewriteAuthorizer:
    """模拟人在 HITL 中批准并改写参数（modified_arguments = 改写后的完整有效参数）。"""

    def __init__(self, modified: dict) -> None:
        self._modified = modified

    async def authorize(self, cap, ctx, arguments, tool_call_id=""):
        return AuthorizationDecision(allowed=True, modified_arguments=copy.deepcopy(self._modified))


class _RecordingBus:
    def __init__(self):
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


def _state_ctx(bus=None):
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1"), agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=bus or InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )
    return mem, state, ctx


def _gw(provider, mem, bus=None, authorizers=None):
    cache = CapabilityCache()
    cache.put("agt_1", [provider._cap()])
    return CapabilityGateway(
        capability_cache=cache, capability_providers=[provider],
        memory=mem, event_bus=bus or InProcessEventBus(),
        provider_authorizers=authorizers or {},
    )


# ── ① 原始认证头到达 Provider（基线必红：现状收到 '***'）──────────────────────


async def test_provider_receives_original_authorization_header() -> None:
    p = _HeaderEcho()
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke(
        "mcp__web__fetch",
        {"url": "https://example.test",
         "headers": {"Authorization": "FAKE_TEST_TOKEN", "X-Custom": "plain"}},
        state, ctx,
    )
    assert res.is_error is False
    assert p.invoked is True
    assert p.received["headers"]["Authorization"] == "FAKE_TEST_TOKEN", (
        f"provider got redacted headers: {p.received!r}"
    )


# ── ② HITL 改写后的参数原样执行（基线必红）────────────────────────────────────


async def test_provider_receives_hitl_modified_header_unredacted() -> None:
    p = _HeaderEcho()
    mem, state, ctx = _state_ctx()
    gw = _gw(p, mem, authorizers={
        "mcp:web": _RewriteAuthorizer({
            "url": "https://approved.test",
            "headers": {"Authorization": "HUMAN_APPROVED_TOKEN"},
        }),
    })
    res = await gw.invoke(
        "mcp__web__fetch",
        {"url": "https://example.test",
         "headers": {"Authorization": "FAKE_TEST_TOKEN"}},
        state, ctx,
    )
    assert res.is_error is False
    assert p.received["url"] == "https://approved.test"          # 改写生效
    assert p.received["headers"]["Authorization"] == "HUMAN_APPROVED_TOKEN", (
        f"provider got redacted post-HITL headers: {p.received!r}"
    )


# ── ③ 非敏感字段不受脱敏影响 ───────────────────────────────────────────────────


async def test_non_sensitive_fields_pass_through() -> None:
    p = _HeaderEcho()
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke(
        "mcp__web__fetch",
        {"url": "https://example.test",
         "headers": {"X-Custom": "plain", "Authorization": "FAKE_TEST_TOKEN"}},
        state, ctx,
    )
    assert res.is_error is False
    assert p.received["url"] == "https://example.test"
    assert p.received["headers"]["X-Custom"] == "plain"


# ── ④ 调用方传入的原始参数字典不被修改 ────────────────────────────────────────


async def test_caller_arguments_dict_not_mutated() -> None:
    p = _HeaderEcho()
    mem, state, ctx = _state_ctx()
    args = {"url": "https://example.test",
            "headers": {"Authorization": "FAKE_TEST_TOKEN"}}
    snapshot = copy.deepcopy(args)
    await _gw(p, mem).invoke("mcp__web__fetch", args, state, ctx)
    assert args == snapshot, "original_arguments 通道被原地修改"


# ── ⑤ 审计副本脱敏（事件 payload + TOOL_AUDIT 只含 '***'）────────────────────


async def test_audit_records_redacted_copy() -> None:
    p = _HeaderEcho()
    bus = _RecordingBus()
    mem, state, ctx = _state_ctx(bus=bus)
    await _gw(p, mem, bus=bus).invoke(
        "mcp__web__fetch",
        {"url": "https://example.test",
         "headers": {"Authorization": "FAKE_TEST_TOKEN"}},
        state, ctx,
    )

    invoked = [e for e in bus.events if e.type == EventType.CAPABILITY_INVOKED]
    assert invoked, "no CapabilityInvoked event"
    audit_args = invoked[0].payload["arguments"]
    assert audit_args["headers"]["Authorization"] == "***", (
        f"event payload leaks plaintext: {audit_args!r}"
    )

    recs = await mem.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx,
                               kinds=[MemoryKind.TOOL_AUDIT])
    assert recs, "no TOOL_AUDIT record"
    audit_text = str(recs[0].content)
    assert "FAKE_TEST_TOKEN" not in audit_text and "***" in audit_text, (
        f"TOOL_AUDIT leaks plaintext: {audit_text!r}"
    )
