"""Gateway 参数校验（spec B）：只拦 required/type/enum，忽略 additionalProperties/format。

控制工具（spec: capability-gateway）例外收紧：未知顶层参数显式拒绝并回灌错误。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.core.loop.capability_gateway import CapabilityGateway, _validate_args
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import (
    _CONTROL_TOOLS,
    ControlCapabilityProvider,
)
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


class _Echo(ToolCapabilityProvider):
    name = "mcp:a"

    def __init__(self, schema: dict | None = None) -> None:
        self.invoked = False
        self.received: dict | None = None
        self._schema = schema or {}

    def _cap(self) -> ToolCapability:
        return ToolCapability(
            id="mcp:a:search", name="search", description="s",
            input_schema=self._schema,
        )

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name, capability_count=1)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            self.invoked = True
            self.received = dict(arguments)
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return _run()

    async def cancel(self, invocation_id, ctx) -> None: return None


def _state_ctx():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1"), agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1"),
    )
    return mem, state, ctx


def _gw(provider, mem):
    cache = CapabilityCache()
    cache.put("agt_1", [provider._cap()])
    return CapabilityGateway(
        capability_cache=cache, capability_providers=[provider],
        memory=mem, event_bus=InProcessEventBus(),
    )


# ── _validate_args 纯函数 ──────────────────────────────────────────────────────


def test_missing_required_reported() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    assert _validate_args({}, schema) is not None


def test_type_mismatch_reported() -> None:
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    assert _validate_args({"count": "abc"}, schema) is not None


def test_enum_out_of_range_reported() -> None:
    schema = {"type": "object", "properties": {"mode": {"enum": ["r", "w"]}}}
    assert _validate_args({"mode": "x"}, schema) is not None


def test_additional_properties_ignored() -> None:
    # additionalProperties:false 故意不拦 —— 多余 key 放行（spec B）
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}
    assert _validate_args({"a": "ok", "extra": 1}, schema) is None


def test_format_ignored() -> None:
    # format 不开 checker，天然不查
    schema = {"type": "object", "properties": {"email": {"type": "string", "format": "email"}}}
    assert _validate_args({"email": "not-an-email"}, schema) is None


def test_empty_schema_passes() -> None:
    assert _validate_args({"anything": 1}, {}) is None
    assert _validate_args({"anything": 1}, None) is None


def test_valid_args_pass() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    assert _validate_args({"q": "hi"}, schema) is None


def test_malformed_schema_fails_open() -> None:
    # 畸形 schema 不该让校验崩，fail-open 放行
    assert _validate_args({"a": 1}, {"properties": {"a": {"type": 123}}}) is None


# ── 经 gateway.invoke 的端到端接线 ─────────────────────────────────────────────


async def test_invoke_blocks_invalid_and_skips_provider() -> None:
    p = _Echo({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {}, state, ctx)
    assert res.is_error is True
    assert "invalid arguments" in res.content
    assert p.invoked is False  # 安全不变式：非法参数绝不下发到 provider


async def test_invoke_passes_valid_args() -> None:
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {"q": "x"}, state, ctx)
    assert res.is_error is False
    assert p.invoked is True


async def test_invoke_coerces_then_validates() -> None:
    # "3" 先被 _coerce_args 收敛成 int，再校验 → 通过，且 provider 收到的是 int
    p = _Echo({"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {"n": "3"}, state, ctx)
    assert res.is_error is False
    assert p.received == {"n": 3}


async def test_invoke_strips_unknown_keys_before_provider() -> None:
    # schema 未声明的键在下发前被剥掉，provider 只收到声明过的参数。
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {"q": "x", "junk": 99}, state, ctx)
    assert res.is_error is False
    assert p.received == {"q": "x"}  # junk 已剥


async def test_invoke_all_unknown_then_required_fails() -> None:
    # 只发了未知键（如救援抠出的错碎片 {"b": 2}）→ 剥成空 → required 校验失败 → 报错、不下发。
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {"b": 2}, state, ctx)
    assert res.is_error is True
    assert "invalid arguments" in res.content
    assert p.invoked is False


async def test_invoke_raw_wrapper_gets_clear_error_not_required_property() -> None:
    # adapter 兜底的 {"_raw": <无法解析文本>}（可解析的已在 finalize 解包）走到 gateway
    # → 给「参数不是合法 JSON」的直白报错，而不是误导性的 "'x' is a required property"
    # （后者会诱导模型照抄 _raw、陷入死循环）。且绝不下发到 provider。
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {"_raw": "{not json"}, state, ctx)
    assert res.is_error is True
    assert "not valid JSON" in res.content
    assert "required property" not in res.content
    assert p.invoked is False


async def test_invoke_raw_wrapper_error_echoes_malformed_text() -> None:
    # gateway 的 _raw 报错要带上畸形原文，模型下一轮读 tool_result 才看得到自己写错了什么、
    # 据此自纠（线上 arguments 那格已被降级成合法 "{}"，原文只能靠这条 error 传回）。
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke(
        "mcp__a__search", {"_raw": '{"q": "hi" "extra": 1}'}, state, ctx)
    assert res.is_error is True
    assert '{"q": "hi" "extra": 1}' in res.content
    assert p.invoked is False


async def test_invoke_raw_wrapper_error_truncates_huge_text() -> None:
    # 畸形原文可能是大 write_file 的几 KB 内容；报错里截断，别把整段回灌炸 context。
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]})
    mem, state, ctx = _state_ctx()
    huge = "x" * 5000
    res = await _gw(p, mem).invoke("mcp__a__search", {"_raw": huge}, state, ctx)
    assert res.is_error is True
    assert len(res.content) < 2000  # 截断，不整段回灌
    assert "truncated" in res.content


# ── 非控制工具的剥键语义（未知键静默剥除，fail-open）──────────────────────


async def test_non_control_unknown_key_still_stripped_silently() -> None:
    # 回归锚：收紧只限控制工具；外部工具保持剥键容错（与 test_invoke_strips_unknown_keys_
    # before_provider 同行为，并排在此处钉住对照契约）
    p = _Echo({"type": "object", "properties": {"q": {"type": "string"}}})
    mem, state, ctx = _state_ctx()
    res = await _gw(p, mem).invoke("mcp__a__search", {"q": "x", "junk": 99}, state, ctx)
    assert res.is_error is False
    assert not res.is_error, "未知键应被静默剥除，不产生错误反馈"
    assert p.received == {"q": "x"}
