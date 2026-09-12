"""compact-fidelity 回归（spec: compact-fidelity）。

分层验收（固定 mock 的输出不随 cue 变化——语义存活断言不承担 cue 守卫）：

- 管道层（CI · mock）：输入材料进入压缩请求、digest 解析与多轮落库、降级路径。
- 请求层（CI · 请求断言）：compact 请求的 cue 含五小节契约——cue 被删改即失败。
- 回取贯通：digest Evidence 引用经 results:read_tool_output 实际取回（依赖
  tool-result-recovery 已落地）。
- 语义质量层（非 CI）：见 benchmarks/compact_fidelity_eval.py。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextAssembler, ContextRequest
from ctx_weft.core.assembler.budget import PriorityBudgetStrategy
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource
from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource
from ctx_weft.core.loop.steps import compact as cm
from ctx_weft.core.loop.steps.compact import (
    DIGEST_SECTIONS, REQUIRED_DIGEST_SECTIONS, ParsedDigest, parse_structured_digest,
    summarize_for_compact,
)
from ctx_weft.core.models.agent import Agent, LoopGuard
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import LLMChunk, LLMMessage, MemoryAddress, MemoryScope, ProviderContext
from ctx_weft.providers.capability_results import ResultsCapabilityProvider
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.results import InMemoryToolResultStore
from ctx_weft.protocols.results import READ_TOOL_QUALIFIED_NAME


CONSTRAINT_MARK = "[CONSTRAINT_TOKEN_X7]"
TODO_MARK = "[TODO_TOKEN_Y9]"
FAILURE_MARK = "[FAILURE_TOKEN_Z3]"

_SCRIPTED_DIGEST = f"""## Goal
Deliver the report.

## Constraints
Must honor {CONSTRAINT_MARK} (max 3 pages).

## Done
- gathered data (done-item-1)

## Remaining
- {TODO_MARK} still open
- root cause {FAILURE_MARK} unresolved

## Evidence
- {READ_TOOL_QUALIFIED_NAME}(invocation_id='inv_evidence_1', tail=200): big log tail
"""


# ── 解析器（纯函数） ─────────────────────────────────────────────────────────


def test_parse_all_sections():
    d = parse_structured_digest(_SCRIPTED_DIGEST)
    assert d.degraded is False
    assert set(d.sections) == set(DIGEST_SECTIONS)
    assert CONSTRAINT_MARK in d.sections["Constraints"]
    assert TODO_MARK in d.sections["Remaining"]


def test_parse_missing_required_section_degrades():
    broken = _SCRIPTED_DIGEST.replace("## Constraints\n", "## Limits\n")
    d = parse_structured_digest(broken)
    assert d.degraded is True
    assert d.text == broken                       # 整文作 digest（不丢内容）


def test_parse_plain_prose_and_evidence_optional():
    d = parse_structured_digest("just prose, no sections")
    assert d.degraded is True and d.sections == {}
    no_ev = _SCRIPTED_DIGEST.split("## Evidence")[0].rstrip() + "\n"
    d2 = parse_structured_digest(no_ev)
    assert d2.degraded is False and "Evidence" not in d2.sections


def test_required_sections_definition():
    assert REQUIRED_DIGEST_SECTIONS == ("Goal", "Constraints", "Done", "Remaining")


# ── harness ──────────────────────────────────────────────────────────────────


class _CaptureLLM:
    """记录压缩请求并回放预排 digest 的 mock（last_request 供请求层断言）。"""

    def __init__(self, digest: str) -> None:
        self.last_request = None
        self._digest = digest
        self.tokenizer = HeuristicTokenizer()

    async def complete(self, request, stream: bool = True):
        self.last_request = request
        for t in (self._digest[i:i + 16] for i in range(0, len(self._digest), 16)):
            yield LLMChunk(kind="token", text=t)


def _state_ctx(llm):
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="ag1")
    state = SimpleNamespace(
        agent=Agent(id="ag1", session_id="s1", template_id="t",
                    loop_guard=LoopGuard()),
        scope=scope,
        task=SimpleNamespace(id="tsk_1", title="T", description="",
                             user_prompt="do the report",
                             user_prompt_in_memory=True, outputs=None),
        session=SimpleNamespace(context_limit=200_000, reserved_output_tokens=0),
        extra={}, run_id="r1", origin=None,
        resolved_model=SimpleNamespace(model="mock", account=""),
        sequence_counter=1,
    )
    ctx = SimpleNamespace(
        assembler=ContextAssembler(
            sources=[TaskSpecSource(), AgentRecallSource()],
            budget=PriorityBudgetStrategy(),
            composer=DefaultComposer(),
            deps=AssemblerDeps(
                memory=mem, knowledge_providers=[],
                provider_ctx=ProviderContext(
                    session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="ag1"),
                capability_cache=None, agent_id="ag1"),
        ),
        llm=llm, memory=mem, cancel_token=None, event_bus=None, config=None,
        provider_ctx=ProviderContext(
            session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="ag1"),
    )
    return state, ctx, mem, scope


# ── 请求层（CI）：cue 契约完整性 ────────────────────────────────────────────


def _cue_of(llm) -> str:
    assert llm.last_request is not None
    return "\n".join(
        m.content for m in llm.last_request.messages
        if isinstance(m.content, str))


def _assert_cue_contract(cue: str) -> None:
    for name in DIGEST_SECTIONS:
        assert f"## {name}" in cue, f"cue missing section {name}"
    assert READ_TOOL_QUALIFIED_NAME in cue       # Evidence 引用格式示例


@pytest.mark.asyncio
async def test_request_layer_task_cue_carries_contract(monkeypatch):
    llm = _CaptureLLM(_SCRIPTED_DIGEST)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    state, ctx, mem, scope = _state_ctx(llm)
    await summarize_for_compact(state, ctx, scope="task")
    _assert_cue_contract(_cue_of(llm))


@pytest.mark.asyncio
async def test_request_layer_agent_cue_carries_contract(monkeypatch):
    llm = _CaptureLLM(_SCRIPTED_DIGEST)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    state, ctx, mem, scope = _state_ctx(llm)
    await summarize_for_compact(state, ctx, scope="agent")
    _assert_cue_contract(_cue_of(llm))


@pytest.mark.asyncio
async def test_guard_cue_mutation_fails_request_assertion(monkeypatch):
    """守门验证（3.4）：删掉 Constraints 指示 → 请求层断言失败。

    注意：不使用「固定 mock 摘要存活」证 cue（其输出不随 cue 变化）——cue 契约
    的唯一 CI 守卫是请求断言层。
    """
    from ctx_weft.core.assembler import composer as composer_mod
    llm = _CaptureLLM(_SCRIPTED_DIGEST)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    monkeypatch.setattr(
        composer_mod, "_COMPACTION_INSTRUCTION",
        composer_mod._COMPACTION_INSTRUCTION.replace(
            "## Constraints — user-imposed constraints and hard limits that must keep "
            "being honored (write \"none\" only if truly none).\n", ""))
    state, ctx, mem, scope = _state_ctx(llm)
    await summarize_for_compact(state, ctx, scope="task")
    with pytest.raises(AssertionError, match="Constraints"):
        _assert_cue_contract(_cue_of(llm))


# ── 管道层（CI · mock）：材料传递、解析、落库、降级 ──────────────────────────


def _ingest_task_material(mem, scope, pctx, *, converged_tool_line: str | None = None):
    """预排输入材料：约束/待办/失败原因 + （可选）收敛版大输出工具结果。"""
    import dataclasses as dc
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.protocols import MemoryEvent, MemoryKind

    async def _ing():
        await mem.ingest(MemoryEvent(
            id=generate_id("mev"), kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.TASK, address=scope, timestamp=now_utc(),
            role="user", content=f"do the report; {CONSTRAINT_MARK} max 3 pages"), pctx)
        await mem.ingest(MemoryEvent(
            id=generate_id("mev"), kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.TASK, address=scope, timestamp=now_utc(),
            role="assistant",
            content=f"attempted; {FAILURE_MARK} timeout in step 2; {TODO_MARK} retry left",
            metadata={"tool_calls": []}), pctx)
        if converged_tool_line:
            await mem.ingest(MemoryEvent(
                id=generate_id("mev"), kind=MemoryKind.CONVERSATION_TURN,
                scope=MemoryScope.TASK, address=scope, timestamp=now_utc(),
                role="tool", content=converged_tool_line,
                metadata={"tool_call_id": "tc_1_0_x", "tool_name": "mcp__t__log",
                          "invocation_id": "inv_evidence_1"}), pctx)
    return _ing()


_CONVERGED_LINE = (
    "[Tool output truncated: 9000 chars exceeded 4000-char limit; full text available "
    f"via {READ_TOOL_QUALIFIED_NAME}(invocation_id='inv_evidence_1', tail=N or offset=N, "
    "limit=N)]\n--- preview (first 100 chars) ---\nAAAA...\n--- tail (last 120 chars) "
    "---\n..." + "B" * 100)


@pytest.mark.asyncio
async def test_pipeline_materials_reach_compact_request(monkeypatch):
    """管道层：约束/待办/失败原因/大输出收敛版确实进入压缩请求的输入历史。"""
    llm = _CaptureLLM(_SCRIPTED_DIGEST)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    state, ctx, mem, scope = _state_ctx(llm)
    await _ingest_task_material(mem, scope, ctx.provider_ctx,
                                converged_tool_line=_CONVERGED_LINE)
    digest = await summarize_for_compact(state, ctx, scope="task")
    request_text = _cue_of(llm)
    for mark in (CONSTRAINT_MARK, TODO_MARK, FAILURE_MARK):
        assert mark in request_text, f"input material {mark} missing from compact request"
    assert "inv_evidence_1" in request_text       # 收敛版里的执行身份可见（2.1 可见性）
    assert digest.degraded is False


@pytest.mark.asyncio
async def test_pipeline_digest_lands_verbatim_with_markers(monkeypatch):
    """管道层：结构化 digest 落库逐字不丢节；Done/Remaining 归属不变。"""
    llm = _CaptureLLM(_SCRIPTED_DIGEST)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    state, ctx, mem, scope = _state_ctx(llm)
    await _ingest_task_material(mem, scope, ctx.provider_ctx)
    digest = await summarize_for_compact(state, ctx, scope="task")
    n = await cm.collapse_task_layer(state, ctx, keep_last=0, summary_text=digest)
    assert n >= 2                                    # 折叠了输入材料
    view = await mem.load_view(scope, MemoryScope.TASK, ctx.provider_ctx)
    collapsed = [r for r in view if r.metadata.get("collapsed")][0]
    body = collapsed.content
    assert CONSTRAINT_MARK in body and TODO_MARK in body and FAILURE_MARK in body
    assert "done-item-1" in body.split("## Done")[1].split("## Remaining")[0]   # 归属不变
    assert collapsed.metadata["digest_degraded"] is False
    assert len(body) <= len(_SCRIPTED_DIGEST) * 2   # 长度有界（原文 + digest 两节）


@pytest.mark.asyncio
async def test_pipeline_degraded_prose_still_completes(monkeypatch):
    """降级路径：散文响应 → 整文落库、degraded=True、压缩正常完成（不炸链路）。"""
    prose = f"did stuff, {TODO_MARK} remains"
    llm = _CaptureLLM(prose)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    state, ctx, mem, scope = _state_ctx(llm)
    await _ingest_task_material(mem, scope, ctx.provider_ctx)
    digest = await summarize_for_compact(state, ctx, scope="task")
    assert digest.degraded is True and digest.text == prose
    n = await cm.collapse_task_layer(state, ctx, keep_last=0, summary_text=digest)
    assert n >= 2
    view = await mem.load_view(scope, MemoryScope.TASK, ctx.provider_ctx)
    collapsed = [r for r in view if r.metadata.get("collapsed")][0]
    assert TODO_MARK in collapsed.content
    assert collapsed.metadata["digest_degraded"] is True


# ── 回取贯通（2.2）：digest 引用 → read_tool_output 实际取回 ─────────────────


@pytest.mark.asyncio
async def test_evidence_reference_readback_roundtrip(monkeypatch):
    llm = _CaptureLLM(_SCRIPTED_DIGEST)

    async def _stream(ctx, state, request):
        async for c in llm.complete(request):
            yield c
    monkeypatch.setattr(cm, "stream_llm_resilient", _stream)
    state, ctx, mem, scope = _state_ctx(llm)
    await _ingest_task_material(mem, scope, ctx.provider_ctx,
                                converged_tool_line=_CONVERGED_LINE)
    digest = await summarize_for_compact(state, ctx, scope="task")
    # store 里有该执行的全文（B 的收敛写入；这里直接布景同一键）。
    store = InMemoryToolResultStore()
    await store.put("inv_evidence_1", "HEAD..." + "TAIL_EVIDENCE_9876543210", None)
    reader = ResultsCapabilityProvider(lambda: store)

    # digest 的 Evidence 节引用了 inv_evidence_1 → 经回读工具实际取回内容。
    assert "inv_evidence_1" in digest.sections.get("Evidence", "")
    out = ""
    async for ev in reader.invoke(
            "results:read_tool_output",
            {"invocation_id": "inv_evidence_1", "tail": 30}, ctx.provider_ctx):
        out = ev.payload.get("content", out)
    assert "TAIL_EVIDENCE_9876543210" in out
