"""两类 compact 走同一 CompactStep，按层折叠（spec/06 §7）。

折叠 task 层写 TASK_COMPACT_SUMMARY、折叠 agent 层写 AGENT_COMPACT_SUMMARY。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.steps.compact import CompactStep
from tests.compaction_testkit import COLLAPSE_DELIM
from ctx_weft.core.loop.steps.finalize import FinalizeStep
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


class _FakeAssembler:
    async def compile(self, inputs, **kwargs):
        return await compiled_test_inputs(inputs, **kwargs)

    async def assemble(self, request):
        return scripted_inputs(SimpleNamespace(messages=[], system="", tools=[], token_count=0), purpose=request.purpose)


class _FakeLLM:
    context_limit = 1_000_000  # apply_dynamic_max_tokens ceiling fallback (Task 2 网关接线)
    tokenizer = HeuristicTokenizer()

    async def complete(self, req, stream=True):
        yield SimpleNamespace(kind="token", text="SUMMARY")


def _scope() -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id="T_target", agent_id="ag1")


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default", agent_id="ag1", task_id="T_target")


def _ev(type_, content, t, role=None) -> MemoryEvent:
    return MemoryEvent(type=type_, address=_scope(), content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role)


async def _run_compact(mem) -> None:
    task = SimpleNamespace(id="C1", status="ACTIVE")
    agent = SimpleNamespace(
        id="ag1", runtime={"llm_model": "mock"},
        loop_config=SimpleNamespace(
            compact_keep_last=1, collapse_keep_last=1,
            compact_token_ratio=0.1, compact_target_ratio=0.0),
        loop_guard=SimpleNamespace(context_limit=1000, context_tokens=1000),
    )
    session = SimpleNamespace(id="s1", tenant_id="default", goal="")
    state = LoopState(run_id="r1", session=session, task=task, agent=agent,
                      scope=_scope(), extra={"template": None, "bound_capabilities": []},
                      transcript=[], resolved_model=model_snapshot(model="mock", account="", context_limit=1000))
    ctx = LoopContext(assembler=_FakeAssembler(), llm=_FakeLLM(), memory=mem,
                      event_bus=InProcessEventBus(), provider_ctx=_pctx())
    await CompactStep().execute(state, ctx)


@pytest.mark.asyncio
async def test_task_compact_writes_task_summary() -> None:
    mem = InMemoryMemoryProvider()
    await mem.ingest(_ev(T.USER_PROMPT, "u", 0, "user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, "a1", 1, "assistant"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, "a2", 2, "assistant"), _pctx())

    await _run_compact(mem)

    # Task layer is now collapsed into a USER_PROMPT (not TASK_COMPACT_SUMMARY)
    up_recs = await mem.recall_recent(_scope(), [T.USER_PROMPT], 10, _pctx())
    collapsed = [r for r in up_recs if r.metadata.get("collapsed")]
    assert collapsed, "Expected a collapsed USER_PROMPT from collapse_task_layer"
    assert "SUMMARY" in collapsed[0].content   # LLM summary is embedded after COLLAPSE_DELIM
    assert COLLAPSE_DELIM in collapsed[0].content
    # keep_last=1: LLM "a2" survives as the kept record
    llm_recs = await mem.recall_recent(_scope(), [T.LLM_RESPONSE], 10, _pctx())
    assert "a2" in [r.content for r in llm_recs]  # keep_last=1


@pytest.mark.asyncio
async def test_agent_compact_writes_agent_summary() -> None:
    """新格式：用 AGENT_CONVERSATION_TURN（parent=None）作 root 胶囊触发 agent 层压缩。"""
    mem = InMemoryMemoryProvider()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    # 植入 2 个 L0 单元 = task 层 body（task_id=root{grp}）+ AGENT_CONVERSATION_TURN finish 对
    # （keep_last=keep_pair=1 → 折最旧 1 个到 L2）
    for grp, t0 in enumerate([0, 10]):
        oid = f"root{grp}"
        await mem.ingest(MemoryEvent(
            type=T.USER_PROMPT,
            address=MemoryAddress(session_id="s1", task_id=oid, agent_id=_scope().agent_id),
            content=f"body {oid}", timestamp=base + timedelta(seconds=t0), role="user",
        ), _pctx())
        for role, dt in [("user", 0), ("assistant", 1)]:
            await mem.ingest(MemoryEvent(
                type=T.AGENT_CONVERSATION_TURN, address=_scope(),
                content=f"turn {oid} {role}",
                timestamp=base + timedelta(seconds=t0 + dt), role=role,
                metadata={"origin_task_id": oid, "parent_task_id": None},
            ), _pctx())

    await _run_compact(mem)

    recs = await mem.recall_recent(
        _scope(), [T.AGENT_CONVERSATION_TURN, T.AGENT_COMPACT_SUMMARY], 10, _pctx()
    )
    assert any(r.type == T.AGENT_COMPACT_SUMMARY and r.content == "SUMMARY" for r in recs)


@pytest.mark.asyncio
async def test_finalize_retry_no_process_report_no_user_message() -> None:
    """retry：observe 分析结果已由折叠出的 TASK_COMPACT_SUMMARY 段摘要承载；finalize 不再写
    process_report/process_report_at，也不注入 user message（spec 2026-07-01 §3.1）。"""
    task = Task(id="T1", session_id="s1", status="PENDING", title="X")
    task.observer_outcome = "continue"
    mem = InMemoryMemoryProvider()
    state = LoopState(
        run_id="r1",
        session=SimpleNamespace(id="s1", tenant_id="default"),
        task=task,
        agent=SimpleNamespace(id="ag1", loop_config=SimpleNamespace(compact_keep_last=6)),
        scope=MemoryAddress(session_id="s1", task_id="T1", agent_id="ag1"),
        verdict=SimpleNamespace(task_outcome="continue", act_recap="missing X; do Y next",
                                task_summary=""),
        resolved_model=model_snapshot(model="mock", account="", context_limit=1000),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem,
        event_bus=InProcessEventBus(), provider_ctx=_pctx(), task_manager=None,
    )
    await FinalizeStep().execute(state, ctx)

    assert task.process_report is None  # 不再写 process_report（旧 Progress So Far 字段路径已废）
    assert task.process_report_at is None
    # retry_count 的 +1 归 TaskManager（Task 4：处置表算新值、TM 写回）——finalize 里
    # 再加一次会让重试预算一轮烧两格。这里断言 finalize **没动**它。
    assert task.retry_count == 0
    recs = await mem.recall_recent(
        MemoryAddress(session_id="s1", task_id="T1", agent_id="ag1"), [T.USER_PROMPT], 10, _pctx()
    )
    assert recs == []  # 不注入 user message


