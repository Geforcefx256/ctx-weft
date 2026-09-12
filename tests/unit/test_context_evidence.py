"""context-evidence-delivery 回归（spec: context-evidence）。

覆盖：贯通（provider → 召回 → 最终请求含证据与来源；裁掉无幽灵）、cache 前缀稳定
（检索变化不打穿锚回合）、提级边界（AND 语义硬上界 / K=0 / 多 source / 无 score 保护）、
对照（紧张预算下证据晚于陈旧胶囊）、地板不受侵蚀、丢弃留痕、未识别 kind 警告。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import (
    AssemblerDeps,
    ContextAssembler,
    ContextBlock,
    ContextRequest,
)
from ctx_weft.core.assembler.budget import PriorityBudgetStrategy
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.priority import slot_priority
from ctx_weft.core.assembler.sources.knowledge import KnowledgeRetrievalSource
from ctx_weft.core.assembler.sources.long_memory import SemanticRecallSource
from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.knowledge import KnowledgeDoc, KnowledgeQuery

# ── 贯通：provider → 召回 → 最终请求 ─────────────────────────────────────────


class _FakeKnowledge:
    name = "fakekb"

    def __init__(self, docs: list[KnowledgeDoc]) -> None:
        self._docs = docs

    async def retrieve(self, query: KnowledgeQuery, ctx):
        for d in self._docs:
            yield d


class _FakeMemory:
    """SemanticRecallSource 只消费 recall_semantic——最小假体。"""

    def __init__(self, records: list) -> None:
        self._records = records

    async def recall_semantic(self, *, query, scope, top_k, ctx):
        return self._records


def _request(session_limit=200_000):
    task = SimpleNamespace(
        id="tsk_1", title="T", description="", user_prompt="how to X?",
        user_prompt_in_memory=False, outputs=None)
    return ContextRequest(
        purpose="act",
        scope=MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="ag1"),
        task=task,
        agent=SimpleNamespace(id="ag1"),
        session=SimpleNamespace(
            context_limit=session_limit, reserved_output_tokens=0),
        template=None,
        bound_capabilities=[],
        extra={},
    )


def _pipeline(knowledge_docs, memory_records, *, session_limit=200_000,
              evidence_top_k=3, evidence_score_floor=0.0):
    assembler = ContextAssembler(
        sources=[
            TaskSpecSource(),
            SemanticRecallSource(),
            KnowledgeRetrievalSource(),
        ],
        budget=PriorityBudgetStrategy(
            evidence_top_k=evidence_top_k, evidence_score_floor=evidence_score_floor),
        composer=DefaultComposer(),
        deps=AssemblerDeps(
            memory=_FakeMemory(memory_records),
            knowledge_providers=[_FakeKnowledge(knowledge_docs)],
            provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                         task_id="tsk_1", agent_id="ag1"),
            capability_cache=None,
        ),
    )
    return assembler


@pytest.mark.asyncio
async def test_pipeline_evidence_reaches_final_request():
    doc = KnowledgeDoc(id="d1", content="Answer is 42.", score=0.9, source="fakekb")
    rec = SimpleNamespace(id="m1", content="prior session learned 42.", score=0.8)
    prompt = await _pipeline([doc], [rec]).assemble(_request())
    body = "\n".join(
        m.content if isinstance(m.content, str) else str(m.content)
        for m in prompt.messages)
    assert "## Retrieved Evidence" in body
    assert "Answer is 42." in body and "fakekb" in body        # 检索：内容 + 来源
    assert "prior session learned 42." in body                 # 召回：内容
    assert "(0.900)" in body and "(0.800)" in body             # score 标识


@pytest.mark.asyncio
async def test_pipeline_pruned_evidence_leaves_no_ghost():
    doc = KnowledgeDoc(id="d1", content="HUGE" * 5_000, score=0.9, source="fakekb")
    # 窗口收紧 → budget 必须裁证据块（唯一大块）→ 最终请求不含证据内容。
    prompt = await _pipeline([doc], [], session_limit=300).assemble(
        _request(session_limit=300))
    body = "\n".join(
        m.content if isinstance(m.content, str) else str(m.content)
        for m in prompt.messages)
    assert "Retrieved Evidence" not in body and "HUGE" not in body


# ── cache 前缀稳定：检索变化不打穿锚回合 ─────────────────────────────────────


def _anchor_blocks(ref_content: str):
    def hist(bid, role, mtype, content, ts, task_id="tsk_1", score=None):
        md = {"timestamp": ts, "role": role, "type": mtype, "task_id": task_id,
              "seq_no": 0}
        if score is not None:
            md["score"] = score
        return ContextBlock(
            id=bid, source="agent_recall", kind="history", target="messages",
            content=content, priority=6, token_estimate=10, metadata=md)

    # 两条 user 回合（interactive 形态）：首条 = 任务锚（## Current Task 落点），
    # 末条 = 当前消息（证据随其尾部动态区变化）——锚 ≠ 末条，前缀稳定性可观察。
    return [
        hist("u1", "user", "user_prompt", "how to X?", "2026-01-01T00:00:00"),
        hist("u2", "user", "user_prompt", "follow-up question", "2026-01-02T00:00:00"),
        ContextBlock(
            id="task_spec", source="task_spec", kind="task_spec", target="messages",
            content="", priority=0, token_estimate=1,
            metadata={"title": "T", "description": "", "user_prompt": "how to X?"}),
        ContextBlock(
            id=generate_id("blk"), source="knowledge:fakekb", kind="reference",
            target="messages", content=ref_content, priority=slot_priority("reference"),
            token_estimate=10,
            metadata={"score": 0.9, "doc_id": "d1", "source": "fakekb"}),
    ]


def _anchor_messages(blocks):
    composer = DefaultComposer()
    task = SimpleNamespace(
        id="tsk_1", title="T", description="", user_prompt="how to X?",
        user_prompt_in_memory=True, outputs=None)
    request = SimpleNamespace(purpose="act", task=task, extra={})
    return composer._build_actor_messages(blocks, request)


def test_cache_prefix_stable_across_retrieval_change():
    m1 = _anchor_messages(_anchor_blocks("evidence round one"))
    m2 = _anchor_messages(_anchor_blocks("evidence round TWO different"))
    # 锚回合（首条 user：## Current Task / Opening Message 框）两轮字节一致。
    assert m1[0].content == m2[0].content
    assert "## Current Task" in m1[0].content
    # 证据仅随尾部动态区（末条 user）变化。
    assert "evidence round one" in m1[-1].content
    assert "round TWO" in m2[-1].content
    assert "evidence" not in m1[0].content


# ── budget：提级边界 + 对照 + 地板 + 留痕 ────────────────────────────────────


def _ev(bid, score, tokens=50, *, source="knowledge:kb",
          ts="2026-01-05T00:00:00", kind="reference"):
    return ContextBlock(
        id=bid, source=source, kind=kind, target="messages", content="e",
        priority=slot_priority(kind), token_estimate=tokens,
        metadata={"score": score, "timestamp": ts})


def _hist(bid, priority, tokens, ts, *, mtype="task_compact_summary", role="user"):
    return ContextBlock(
        id=bid, source="agent_recall", kind="history", target="messages",
        content="h", priority=priority, token_estimate=tokens,
        metadata={"timestamp": ts, "role": role, "type": mtype, "task_id": "past"})


def _req():
    return SimpleNamespace(
        task=SimpleNamespace(id="cur"),
        session=SimpleNamespace(context_limit=180_000, reserved_output_tokens=8192),
    )


def test_promotion_hard_cap_ten_high_scores_only_k_promoted():
    blocks = [_ev(f"e{i}", 0.9, ts=f"2026-01-0{i + 1}T00:00:00") for i in range(10)]
    strat = PriorityBudgetStrategy(evidence_top_k=3)
    promoted = strat._promoted_evidence_ids(blocks)
    assert len(promoted) == 3                       # AND 语义硬上界：十条均 0.9 只提 3


def test_promotion_k0_and_floor():
    blocks = [_ev("e1", 0.99), _ev("e2", 0.1)]
    assert PriorityBudgetStrategy(evidence_top_k=0)._promoted_evidence_ids(blocks) == set()
    # floor 加严：排名达标但分数不达标的条目不提级。
    strat = PriorityBudgetStrategy(evidence_top_k=3, evidence_score_floor=0.5)
    assert "e1" in strat._promoted_evidence_ids(blocks)
    assert "e2" not in strat._promoted_evidence_ids(blocks)


def test_promotion_per_source_independent():
    blocks = [_ev("k1", 0.9, source="knowledge:a"), _ev("k2", 0.8, source="knowledge:a"),
              _ev("k3", 0.7, source="knowledge:a"),
              _ev("s1", 0.6, source="long_memory"), _ev("s2", 0.5, source="long_memory")]
    strat = PriorityBudgetStrategy(evidence_top_k=2)
    promoted = strat._promoted_evidence_ids(blocks)
    assert promoted == {"k1", "k2", "s1", "s2"}     # 多 source 各自计 K


@pytest.mark.asyncio
async def test_low_score_dropped_before_high_score_within_tier():
    """同档（7 档未提级）内低分先丢——score 正号方向的回归（探针教训）。"""
    blocks = [_ev("low", 0.2, tokens=100), _ev("high", 0.9, tokens=100),
              _hist("cap", 6, 100, "2026-01-01T00:00:00")]
    kept = await PriorityBudgetStrategy(evidence_top_k=0).apply(
        blocks, token_limit=200, request=_req())
    ids = {b.id for b in kept}
    assert "low" not in ids and "high" in ids


@pytest.mark.asyncio
async def test_promoted_evidence_outlives_stale_capsules():
    """对照（spec R2 场景）：同预算下顺序反转——未提级证据 → 陈旧胶囊 → 提级证据。"""
    hi = _ev("hi", 0.95, tokens=120, ts="2026-01-01T00:00:00")     # 提级 4
    lo = _ev("lo", 0.30, tokens=120, ts="2026-01-02T00:00:00")     # 维持 7
    cap_old = _hist("cap_old", 6, 120, "2026-01-03T00:00:00")      # 陈旧胶囊 6
    cap_new = _hist("cap_new", 6, 120, "2026-01-04T00:00:00")
    kept = await PriorityBudgetStrategy(evidence_top_k=1).apply(
        [hi, lo, cap_old, cap_new], token_limit=240, request=_req())
    ids = {b.id for b in kept}
    # 预算 240：先丢 lo（7 档低分）→ 仍超 → 丢 cap_old（6 档最老）→ 达标。
    assert ids == {"hi", "cap_new"}


@pytest.mark.asyncio
async def test_floor_survives_evidence_pressure():
    """地板（pin 的当前 user_prompt）不受证据提级侵蚀。"""
    cur = ContextBlock(
        id="cur", source="agent_recall", kind="history", target="messages",
        content="current message", priority=6, token_estimate=50,
        metadata={"timestamp": "2026-01-09T00:00:00", "role": "user",
                  "type": "user_prompt", "task_id": "cur"})
    blocks = [cur] + [_ev(f"e{i}", 0.9, tokens=60) for i in range(5)]
    kept = await PriorityBudgetStrategy(evidence_top_k=5).apply(
        blocks, token_limit=60, request=_req())
    assert "cur" in {b.id for b in kept}


@pytest.mark.asyncio
async def test_no_score_blocks_last_within_tier(caplog):
    """同档（4 档）内：有 score 的提级证据先丢，无 score 的当前任务历史殿后。"""
    cur_hist = ContextBlock(
        id="curh", source="agent_recall", kind="history", target="messages",
        content="cur", priority=6, token_estimate=100,
        metadata={"timestamp": "2026-01-01T00:00:00", "role": "assistant",
                  "type": "llm_response", "task_id": "cur"})       # 提级 4（当前 task）
    hi = _ev("hi", 0.95, tokens=100, ts="2026-01-01T00:00:00")     # 提级 4（证据）
    with caplog.at_level(logging.WARNING, logger="ctx_weft.core.assembler.budget"):
        kept = await PriorityBudgetStrategy(evidence_top_k=1).apply(
            [cur_hist, hi], token_limit=100, request=_req())
    assert {b.id for b in kept} == {"curh"}          # 证据先丢，无 score 历史殿后
    # 证据丢弃单列 warning（含 score），可被单独检索。
    assert any("dropped evidence" in r.message and "hi" not in r.getMessage()
               or "dropped evidence" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_drop_logging_reconstructible(caplog):
    with caplog.at_level(logging.INFO, logger="ctx_weft.core.assembler.budget"):
        await PriorityBudgetStrategy(evidence_top_k=0).apply(
            [_ev("e1", 0.5), _hist("cap1", 6, 50, "2026-01-01T00:00:00")],
            token_limit=40, request=_req())
    msgs = " | ".join(r.getMessage() for r in caplog.records)
    assert "kind=reference" in msgs and "kind=history" in msgs    # 丢了什么
    assert "eff_priority=" in msgs                                # 当时什么档


@pytest.mark.asyncio
async def test_unsupported_kind_warns_including_blackboard(caplog):
    with caplog.at_level(logging.WARNING, logger="ctx_weft.core.assembler.composer"):
        await DefaultComposer().compose(
            [ContextBlock(
                id="bb1", source="blackboard:topic", kind="blackboard", target="messages",
                content="note", priority=3, token_estimate=5, metadata={})],
            _request())
    assert any("blackboard" in r.getMessage() for r in caplog.records)


def test_slot_priority_static_baseline_unchanged():
    """既有断言语义（test_source_priorities 同口径）：静态基线仍是 7；提级在 budget 层。"""
    assert slot_priority("reference") == 7
    assert slot_priority("summary") == 7
