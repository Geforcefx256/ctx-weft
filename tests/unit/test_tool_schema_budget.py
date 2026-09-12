"""tool-schema-budget 回归（spec: context-budget）。

覆盖：费率单一真源（与旧两处公式口径一致）、装配预留（大 schema 入账 / compact 空
面 / cache 缺失回退 / 溢出报错用真窗口）、指纹感知同名工具定义增长、循环增量追踪
（pin 大 schema 后估算立即增长 / 工具面不变行为不变）、发送前超限不硬拒 + WARNING、
与 context-evidence 的组合口径（预留先扣减、证据地板在剩余内生效）。
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextAssembler, ContextRequest
from ctx_weft.core.assembler.budget import PriorityBudgetStrategy
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.priority import slot_priority
from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.loop.llm_gateway import (
    apply_dynamic_max_tokens,
    request_prompt_estimate,
)
from ctx_weft.core.models.agent import LoopGuard
from ctx_weft.core.models.errors import ContextOverflowError
from ctx_weft.core.utils.estimate import estimate_tools_tokens, tools_signature
from ctx_weft.protocols import LLMMessage, LLMRequest, LLMTool, MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer


def _big_tool(schema_pad: int = 30_000) -> LLMTool:
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "blob": {"type": "string", "description": "d" * schema_pad},
        },
    }
    return LLMTool(name="t__big", description="big schema tool", input_schema=schema)


def _tiny_tool() -> LLMTool:
    return LLMTool(name="t__tiny", description="t", input_schema={"type": "object"})


# ── 费率单一真源 ─────────────────────────────────────────────────────────────


def test_estimate_tools_tokens_matches_legacy_formula():
    t = _big_tool(200)
    count = HeuristicTokenizer().count
    legacy = (
        count(t.name) + count(t.description or "")
        + count(json.dumps(t.input_schema, ensure_ascii=False))
    )
    ours = estimate_tools_tokens([t], count)
    # key 排序只改键序不改内容：与旧公式差异限于排序引起的字节序（<1 token 量级）。
    assert abs(ours - legacy) <= 1
    assert estimate_tools_tokens([], count) == 0
    assert estimate_tools_tokens(None, count) == 0


def test_signature_senses_definition_growth_same_name():
    small = [LLMTool(name="t__x", description="d", input_schema={"type": "object"})]
    big = [LLMTool(name="t__x", description="d", input_schema={
        "type": "object", "properties": {"blob": {"type": "string", "description": "p" * 5_000}}})]
    assert tools_signature(small) != tools_signature(big)   # 同名 schema 增长 → 指纹变
    assert tools_signature(small) == tools_signature([small[0]])
    assert tools_signature([]) == ""


# ── 装配预留 ─────────────────────────────────────────────────────────────────


def _cache_with(tool: ToolCapability):
    cache = CapabilityCache()
    cache.put("ag1", [tool])
    return cache


def _cap_tool(name="t:big") -> ToolCapability:
    return ToolCapability(
        id=name, name=name.rsplit(":", 1)[1], kind="tool", purposes=["act"],
        description="big schema tool",
        input_schema={"type": "object",
                      "properties": {"blob": {"type": "string", "description": "d" * 30_000}}})


def _assembler(cache, request_limit=200_000):
    return ContextAssembler(
        sources=[TaskSpecSource()],
        budget=PriorityBudgetStrategy(),
        composer=DefaultComposer(),
        deps=AssemblerDeps(
            memory=SimpleNamespace(), knowledge_providers=[],
            provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                         task_id="tsk_1", agent_id="ag1"),
            capability_cache=cache, agent_id="ag1",
        ),
    ), request_limit


def _req(purpose="act", session_limit=200_000):
    task = SimpleNamespace(
        id="tsk_1", title="T", description="", user_prompt="hi",
        user_prompt_in_memory=False, outputs=None)
    return ContextRequest(
        purpose=purpose,
        scope=MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="ag1"),
        task=task, agent=SimpleNamespace(id="ag1"),
        session=SimpleNamespace(context_limit=session_limit, reserved_output_tokens=0),
        template=None, bound_capabilities=[], extra={},
    )


class _OneBlockSource:
    name = "stub"

    def __init__(self, blk) -> None:
        self._blk = blk

    async def fetch(self, request, deps):
        yield self._blk


@pytest.mark.asyncio
async def test_assembly_reserves_tools_budget():
    asm, _ = _assembler(_cache_with(_cap_tool()))
    prompt = await asm.assemble(_req())
    assert prompt.metadata["tools_reserved_tokens"] > 1_000      # 大 schema 显著入账
    assert prompt.metadata["tools_signature"]


@pytest.mark.asyncio
async def test_assembly_compact_and_no_cache_zero_reservation():
    asm, _ = _assembler(_cache_with(_cap_tool()))
    prompt = await asm.assemble(_req(purpose="compact"))
    assert prompt.metadata["tools_reserved_tokens"] == 0         # compact 工具面恒空

    asm2, _ = _assembler(None)                                    # cache 缺失（测试直构）
    prompt2 = await asm2.assemble(_req())
    assert prompt2.metadata["tools_reserved_tokens"] == 0


@pytest.mark.asyncio
async def test_overflow_error_reports_real_window():
    """溢出报错的 effective_limit 用真窗口（非扣减后的内容预算）。"""
    from ctx_weft.core.assembler.assembler import ContextBlock

    floor = ContextBlock(
        id="spec", source="task_spec", kind="task_spec", target="messages",
        content="", priority=0, token_estimate=10,
        metadata={"title": "T", "description": "", "user_prompt": "hi"})
    asm = ContextAssembler(
        sources=[_OneBlockSource(floor)],
        budget=PriorityBudgetStrategy(),
        composer=DefaultComposer(),
        deps=AssemblerDeps(
            memory=SimpleNamespace(), knowledge_providers=[],
            provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                         task_id="tsk_1", agent_id="ag1"),
            capability_cache=_cache_with(_cap_tool()), agent_id="ag1"),
    )
    # 真窗口 300；预留 ~18k → 内容预算归零 → 地板(10) 仍超 → 报错。
    with pytest.raises(ContextOverflowError) as ei:
        await asm.assemble(_req(session_limit=300))
    assert ei.value.effective_limit == 300          # 真窗口，非 max(0, 300−reserved)


# ── 循环增量追踪 ─────────────────────────────────────────────────────────────


def _request(messages: list[LLMMessage], tools) -> LLMRequest:
    req = LLMRequest(model="mock", system="s", messages=messages, tools=tools)
    return req


def _tokenizer():
    return HeuristicTokenizer()


def test_incremental_tracks_tools_fingerprint_change():
    tok = _tokenizer()
    guard = LoopGuard(context_tokens=1_000, context_limit=100_000)
    m = [LLMMessage(role="user", content="q")]

    # 首轮（基线在场）：纯消息增量，工具面首记不加减值。
    est1 = request_prompt_estimate(tok, _request(m, [_tiny_tool()]), guard, baseline_msg_count=0)
    assert est1 == 1_000 + request_prompt_estimate.__globals__["_estimate_message_tokens"](
        m[0], tok.count)
    sig1, est_tools1 = guard.last_tools_signature, guard.last_tools_est

    # pin 大 schema：指纹变化 → 估算 +=（新面 − 旧面），立即增长。
    m2 = m + [LLMMessage(role="assistant", content="a")]
    est2 = request_prompt_estimate(tok, _request(m2, [_big_tool()]), guard, baseline_msg_count=1)
    msg_delta = request_prompt_estimate.__globals__["_estimate_message_tokens"](m2[1], tok.count)
    assert est2 == 1_000 + msg_delta + (guard.last_tools_est - est_tools1)
    assert guard.last_tools_signature != sig1
    assert est2 - est1 > 1_000                                    # 大 schema 显著增长

    # 模拟 usage 回来：真实基线刷新（含工具面实测）——下一轮增量不再含工具差值。
    guard.context_tokens = est2
    m3 = m2 + [LLMMessage(role="user", content="follow")]
    sig_before = guard.last_tools_signature
    est3 = request_prompt_estimate(tok, _request(m3, [_big_tool()]), guard, baseline_msg_count=2)
    msg_delta3 = request_prompt_estimate.__globals__["_estimate_message_tokens"](m3[2], tok.count)
    assert est3 == est2 + msg_delta3                              # 纯消息增量（差值被基线吸收）
    assert guard.last_tools_signature == sig_before


def test_incremental_same_name_schema_growth_detected():
    """同名工具 schema 显著增长：指纹必须变化且估算增长（审核 #7 场景）。"""
    tok = _tokenizer()
    guard = LoopGuard(context_tokens=1_000)
    small = [LLMTool(name="t__x", description="d", input_schema={"type": "object"})]
    big = [LLMTool(name="t__x", description="d", input_schema={
        "type": "object", "properties": {"blob": {"type": "string", "description": "p" * 8_000}}})]

    request_prompt_estimate(tok, _request([LLMMessage(role="user", content="q")], small),
                            guard, baseline_msg_count=0)
    est_small = guard.last_tools_est
    request_prompt_estimate(tok, _request([LLMMessage(role="user", content="q")], big),
                            guard, baseline_msg_count=0)
    assert guard.last_tools_est > est_small + 1_000               # schema 增长被入账


# ── 发送前：不硬拒 + WARNING ────────────────────────────────────────────────


def test_overlimit_does_not_reject_but_warns(caplog):
    tok = _tokenizer()
    guard = LoopGuard(context_tokens=10, context_limit=1_000, reserved_output_tokens=0)
    ctx = SimpleNamespace(llm=SimpleNamespace(
        context_limit=1_000, output_ceiling=None,
        config=None,
        tokenizer=tok))
    request = _request([LLMMessage(role="user", content="q" * 100)], [_big_tool()])
    request.prompt_token_estimate = 5_000                          # 远超有效窗口 1_000
    with caplog.at_level(logging.WARNING, logger="ctx_weft.core.loop.llm_gateway"):
        apply_dynamic_max_tokens(ctx, request, guard)
    # 不硬拒：max_tokens 被写入（收紧到 floor），且 WARNING 可检索。
    assert request.max_tokens == 1_024
    assert any("exceeds effective window" in r.getMessage() for r in caplog.records)


def test_dynamic_max_tokens_floor_when_fully_exceeded():
    """极端超限：remaining 跌破 floor → max_tokens 落到 floor（1024），请求照发。"""
    guard = LoopGuard(context_tokens=10, context_limit=1_000, reserved_output_tokens=0)
    ctx = SimpleNamespace(llm=SimpleNamespace(context_limit=1_000, output_ceiling=None))
    request = _request([LLMMessage(role="user", content="q")], [])
    request.prompt_token_estimate = 999_999
    apply_dynamic_max_tokens(ctx, request, guard)
    assert request.max_tokens == 1_024


    # ── 偏差留痕（act 回喂处，spec R4）─────────────────────────────────────────


@pytest.mark.asyncio
async def test_estimate_feedback_log_carries_tools_fields(caplog):
    """act 回喂处偏差留痕：est_seg / base / 实测 / tools 指纹与估算同现（可归因）。"""
    from ctx_weft.core.assembler.assembler import AssembledPrompt
    from ctx_weft.core.loop.driver import LoopContext, LoopState
    from ctx_weft.core.loop.steps.act import ActStep
    from ctx_weft.core.models.agent import Agent
    from ctx_weft.core.models.task import NormalTaskSettings, Task
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="ag1")
    state = LoopState(
        run_id="r1",
        session=SimpleNamespace(id="s1", tenant_id="default", token_used=0),
        task=Task(id="tsk_1", session_id="s1", status="ACTIVE", title="T",
                  interaction_mode="auto", settings=NormalTaskSettings()),
        agent=Agent(id="ag1", session_id="s1", template_id="t"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    llm = MockLLMAdapter(responses=[MockResponse(text="done")])
    prompt = AssembledPrompt(
        system="", messages=[LLMMessage(role="user", content="hi")],
        tools=[_tiny_tool()], token_count=1)
    ctx = LoopContext(
        assembler=None, llm=llm, memory=mem, event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="ag1"),
    )
    state.assembled_prompt = prompt
    with caplog.at_level(logging.INFO, logger="ctx_weft.core.loop.steps.act"):
        await ActStep().execute(state, ctx)
    assert any(
        "token estimate feedback" in r.getMessage()
        and "tools_signature=" in r.getMessage()
        and "tools_est=" in r.getMessage()
        and "actual_prompt=" in r.getMessage()
        for r in caplog.records), caplog.text


# ── 与 context-evidence 的组合口径 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_combined_schema_reservation_then_evidence_floor():
    """schema 预留先扣减；证据提级在其剩余内生效（两 change 组合语义）。

    构造：真窗口仅容 task_spec 地板 + 小证据；大 schema 预留把内容预算压到
    「无预留时本可存活的高分证据」之下——证据被裁、地板存活，且 metadata 可归因。
    """
    from ctx_weft.core.assembler.assembler import ContextBlock

    class _OneBlockSource:
        name = "stub"

        def __init__(self, blk) -> None:
            self._blk = blk

        async def fetch(self, request, deps):
            yield self._blk

    ev = ContextBlock(
        id="ev1", source="knowledge:kb", kind="reference", target="messages",
        content="answer evidence", priority=slot_priority("reference"),
        token_estimate=50, metadata={"score": 0.9, "timestamp": "2026-01-01T00:00:00"})
    spec = ContextBlock(
        id="spec", source="task_spec", kind="task_spec", target="messages",
        content="", priority=0, token_estimate=10,
        metadata={"title": "T", "description": "", "user_prompt": "hi"})

    def build(cache):
        return ContextAssembler(
            sources=[_OneBlockSource(spec), _OneBlockSource(ev)],
            budget=PriorityBudgetStrategy(evidence_top_k=3),
            composer=DefaultComposer(),
            deps=AssemblerDeps(
                memory=SimpleNamespace(), knowledge_providers=[],
                provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                             task_id="tsk_1", agent_id="ag1"),
                capability_cache=cache, agent_id="ag1"),
        )

    # 无 cache（零预留）：预算 200 容得下 spec(10)+ev(50) → 证据存活。
    prompt_plain = await build(None).assemble(_req(session_limit=200))
    body = "\n".join(m.content for m in prompt_plain.messages if isinstance(m.content, str))
    assert "answer evidence" in body

    # 带 cache：预留按实测动态取窗口——内容预算 = (reserved + 30)，容地板(10) 不容
    # 地板+证据(60)：证据被裁、地板存活、不溢出、可归因（schema 预留先扣减、证据
    # 地板在其剩余内生效——两 change 组合语义）。
    probe = await build(_cache_with(_cap_tool())).assemble(_req(session_limit=10_000_000))
    reserved = probe.metadata["tools_reserved_tokens"]
    assert reserved > 200
    prompt_reserved = await build(_cache_with(_cap_tool())).assemble(
        _req(session_limit=reserved + 30))
    body2 = "\n".join(
        m.content for m in prompt_reserved.messages if isinstance(m.content, str))
    assert "answer evidence" not in body2                       # 剩余预算容不下证据
    assert prompt_reserved.metadata["tools_reserved_tokens"] == reserved
