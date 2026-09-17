from datetime import datetime, timedelta, UTC
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.compact import COLLAPSE_DELIM, collapse_task_layer
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider as InMemoryBlackboard
from ctx_weft.protocols import (
    MemoryEvent, MemoryEventType as T, MemoryAddress, ProviderContext,
)
from ctx_weft.protocols.events import EventType

pytestmark = pytest.mark.asyncio

_BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _ctx():
    return ProviderContext(session_id="s", tenant_id="tn")


def _state(mem, scope):
    return SimpleNamespace(
        scope=scope, task=SimpleNamespace(id="t1"),
        agent=SimpleNamespace(), session=SimpleNamespace())


async def _ingest(mem, scope, typ, content, i, role="user"):
    await mem.ingest(MemoryEvent(type=typ, address=scope, content=content,
                                 timestamp=_BASE + timedelta(seconds=i), role=role,
                                 metadata={"task_id": scope.task_id}), _ctx())


async def test_collapse_folds_early_keeps_recent():
    mem = InMemoryBlackboard()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "原始请求：做 X", 0)
    await _ingest(mem, scope, T.LLM_RESPONSE, "step1", 1, role="assistant")
    await _ingest(mem, scope, T.TOOL_RESULT, "r1", 2, role="tool")
    await _ingest(mem, scope, T.LLM_RESPONSE, "step2", 3, role="assistant")
    await _ingest(mem, scope, T.TOOL_RESULT, "r2", 4, role="tool")

    ctx = SimpleNamespace(memory=mem, provider_ctx=_ctx())
    n = await collapse_task_layer(_state(mem, scope), ctx, keep_last=2, summary_text="做了 step1/step2")

    assert n == 3  # 前 3 条被折
    recs = await mem.recall_recent(scope, [T.USER_PROMPT, T.LLM_RESPONSE, T.TOOL_RESULT], 100, _ctx())
    # newest-first；最老一条应是新坍缩 USER_PROMPT
    by_type = [(r.type, r.content) for r in reversed(recs)]
    assert by_type[0][0] == T.USER_PROMPT
    assert "原始请求：做 X" in by_type[0][1]       # 原始消息节
    assert COLLAPSE_DELIM in by_type[0][1]
    assert "做了 step1/step2" in by_type[0][1]     # 执行摘要节
    # 保留最近 2 条 raw
    assert by_type[1] == (T.LLM_RESPONSE, "step2")
    assert by_type[2] == (T.TOOL_RESULT, "r2")


async def test_collapse_noop_when_within_keep_last():
    mem = InMemoryBlackboard()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "orig", 0)
    await _ingest(mem, scope, T.LLM_RESPONSE, "step1", 1, role="assistant")
    ctx = SimpleNamespace(memory=mem, provider_ctx=_ctx())
    n = await collapse_task_layer(_state(mem, scope), ctx, keep_last=5, summary_text="x")
    assert n == 0


async def test_recollapse_keeps_original_bounded():
    mem = InMemoryBlackboard()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    # 已坍缩过一次的 USER_PROMPT
    await _ingest(mem, scope, T.USER_PROMPT, f"原始请求：做 X{COLLAPSE_DELIM}旧摘要", 0)
    await _ingest(mem, scope, T.LLM_RESPONSE, "step3", 1, role="assistant")
    await _ingest(mem, scope, T.TOOL_RESULT, "r3", 2, role="tool")
    ctx = SimpleNamespace(memory=mem, provider_ctx=_ctx())
    await collapse_task_layer(_state(mem, scope), ctx, keep_last=1, summary_text="新摘要含 step3")

    recs = await mem.recall_recent(scope, [T.USER_PROMPT], 100, _ctx())
    newest = recs[0].content
    assert newest.count("原始请求：做 X") == 1     # 原始节没有嵌套膨胀
    assert "旧摘要" not in newest                   # 旧摘要节被新摘要替掉
    assert "新摘要含 step3" in newest


async def test_escalating_compact_l3_uses_collapse_keep_last(monkeypatch):
    """迁移自旧 _compact_scope 用例（双阈值并行折已废）：只有 task 层可折（无 agent 层派发对）
    → L1/L2 天然跳过（无 root residue / 无 kept origin），落到 L3 用 collapse_keep_last(2)。"""
    from ctx_weft.core.loop.steps import compact as cm

    calls = []
    kept_arg = {}

    async def _fake_summ(state, ctx, *, scope="task"):
        calls.append(scope)
        return f"summary-{scope}"

    async def _fake_collapse(state, ctx, keep_last, summary_text):
        kept_arg["keep_last"] = keep_last
        # escalating_compact 交来的是「取摘要的函数」（惰性：真 collapse 在确定要折之后才求值）。
        # 替身照真实实现去求值，本用例钉的「L3 产出一次 task 摘要」才有意义。
        await cm._resolve_digest(summary_text)
        return 3

    monkeypatch.setattr(cm, "summarize_for_compact", _fake_summ)
    monkeypatch.setattr(cm, "collapse_task_layer", _fake_collapse)

    mem = InMemoryBlackboard()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    for i in range(6):  # 6 条 TASK_COMPACT_TYPES > collapse_keep_last(2)
        await _ingest(mem, scope, T.LLM_RESPONSE, f"turn{i}", i, role="assistant")

    agent = SimpleNamespace(
        loop_config=SimpleNamespace(
            compact_keep_last=6, collapse_keep_last=2,
            compact_token_ratio=0.1, compact_target_ratio=0.0),
        id="a", loop_guard=SimpleNamespace(context_limit=1000, context_tokens=1000))
    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"), agent=agent,
                            session=SimpleNamespace(id="s", tenant_id="tn"),
                            extra={}, run_id="run1", sequence_counter=0)
    ctx = SimpleNamespace(memory=mem, provider_ctx=_ctx(),
                          task_manager=None, event_bus=None,
                          llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))

    events = await cm.escalating_compact(state, ctx, token_estimate=1000, trigger="compact")

    # 无 agent 层派发对 → 无 root residue / 无 kept origin → L1/L2 天然跳过 → 只产 task 摘要
    assert calls == ["task"]
    assert kept_arg["keep_last"] == 2
    assert any(e.payload.get("source") == "collapse" for e in events)


async def test_escalating_l3_fires_on_segment_only_accumulation(monkeypatch):
    """回归 Fix 2：retry 累积后 task 层只有 USER_PROMPT + 段摘要（无 raw）。L3 守卫须用
    _TASK_LAYER_TYPES 计数（含 TASK_COMPACT_SUMMARY）才会触发；旧守卫用 TASK_COMPACT_TYPES
    只数到 1 条 USER_PROMPT → L3 永不触发 → 累积段摘要坍缩不了。"""
    from ctx_weft.core.loop.steps import compact as cm

    async def _fake_summ(state, ctx, *, scope="task"):
        return "坍缩摘要"
    monkeypatch.setattr(cm, "summarize_for_compact", _fake_summ)  # 免真实 LLM；collapse 用真的

    mem = InMemoryBlackboard()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "原始请求：做 X", 0)
    for i in range(3):  # 3 条段摘要（无任何 raw），collapse_keep_last=2 → 3+1 > 2 触发
        await _ingest(mem, scope, T.TASK_COMPACT_SUMMARY, f"段摘要{i}", i + 1, role="assistant")

    agent = SimpleNamespace(
        id="a", loop_config=SimpleNamespace(
            compact_keep_last=6, collapse_keep_last=2,
            compact_token_ratio=0.1, compact_target_ratio=0.01),
        loop_guard=SimpleNamespace(context_limit=1000, context_tokens=1000))
    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"), agent=agent,
                            session=SimpleNamespace(id="s", tenant_id="tn"),
                            extra={}, run_id="r1", sequence_counter=0)
    ctx = SimpleNamespace(memory=mem, provider_ctx=_ctx(), task_manager=None, event_bus=None,
                          llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))

    events = await cm.escalating_compact(state, ctx, token_estimate=1000, trigger="compact")

    # L3 真的坍缩了：产出带 COLLAPSE_DELIM 的 USER_PROMPT，段摘要收敛到 collapse_keep_last
    assert any(e.payload.get("source") == "collapse" for e in events), "L3 应触发"
    ups = await mem.recall_recent(scope, [T.USER_PROMPT], 100, _ctx())
    assert any(COLLAPSE_DELIM in u.content and "原始请求：做 X" in u.content for u in ups)
    segs = await mem.recall_recent(scope, [T.TASK_COMPACT_SUMMARY], 100, _ctx())
    assert len(segs) == 2, "坍缩后段摘要保留 collapse_keep_last=2 条"


async def test_collapsed_user_prompt_gets_current_task_frame():
    from ctx_weft.core.assembler.assembler import ContextBlock
    from ctx_weft.core.assembler.composer import DefaultComposer
    from ctx_weft.core.utils.content import content_to_text

    collapsed = f"原始请求：做 X{COLLAPSE_DELIM}已完成 step1/step2"
    blk = ContextBlock(id="u", source="agent_recall", kind="history", target="messages",
                       content=collapsed, priority=3, token_estimate=1,
                       metadata={"role": "user", "type": "user_prompt", "timestamp": "1",
                                 "task_id": "t1"})
    task = SimpleNamespace(id="t1", title="任务标题", description="", user_prompt="做 X",
                           user_prompt_in_memory=True, process_report=None,
                           process_report_at=None, outputs=None)
    req = SimpleNamespace(task=task, purpose="act")

    msgs = DefaultComposer()._build_actor_messages([blk], req)
    framed = content_to_text(msgs[-1].content) if msgs else ""
    joined = "\n".join(content_to_text(m.content) for m in msgs)
    assert "## Current Task" in joined and "任务标题" in joined
    assert "原始请求：做 X" in joined          # 原始节
    assert "已完成 step1/step2" in joined      # 摘要节
