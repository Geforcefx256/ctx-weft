"""跨层 fold（spec 2026-06-29，删 L1，单一阈值 keep_last）。

通过 finalize_task_memory 造真实结束单元（task 层胶囊 + agent 层 finish 对），再调
fold_root_experience 验证：超 keep_last 的最老单元**整体折成摘要**——task 层胶囊 + finish 对
一起 supersede + 一条 AGENT_COMPACT_SUMMARY；未超的保留为完整胶囊（不存在「body 删、finish
留」的 L1 中间态）。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.compaction_testkit import _count_root_residues, fold_root_experience
from ctx_weft.core.loop.steps.finalize import finalize_task_memory
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    MemoryEvent,
    MemoryEventType,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
AGENT = "ag1"


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _task_scope(task_id: str) -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=AGENT)


def _agent_scope() -> MemoryAddress:
    """fold 跑在 agent scope（task_id 为当前 task，但 agent 层召回只看 agent_id）。"""
    return MemoryAddress(session_id="s1", task_id="cur", agent_id=AGENT)


class _FakeTM:
    def children_of(self, task_id: str) -> set[str]:
        return set()


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


def _finalize_state(task: Task, scope: MemoryAddress, cfg: LoopConfig):
    agent = SimpleNamespace(id=scope.agent_id, loop_config=cfg)
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="run1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent)


def _finalize_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_pctx(), task_manager=_FakeTM(),
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


def _fold_state(cfg: LoopConfig):
    agent = SimpleNamespace(id=AGENT, loop_config=cfg)
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="run1", sequence_counter=0, session=session,
                           scope=_agent_scope(), task=SimpleNamespace(id="cur"), agent=agent)


def _fold_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_pctx())


def _root_task(task_id: str) -> Task:
    return Task(id=task_id, session_id="s1", status="FINISHED", tenant_id="default",
                assigned_agent_id=AGENT, creator_agent_id=AGENT, parent_task_id=None,
                title=f"Root {task_id}", user_prompt="hi", settings=NormalTaskSettings())


async def _make_l0_unit(mem, task_id: str, t0: int) -> None:
    """造一个真实 L0 顶层单元：task 层 raw body（短任务，留全 raw）+ agent 层 finish 对。"""
    scope = _task_scope(task_id)
    # short body：1 轮 assistant（< short_task_turn_cap=2, small → 留全 raw）
    await mem.ingest(_ev(T.USER_PROMPT, scope, f"prompt {task_id}", t0, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, scope, f"reply {task_id}", t0 + 1, role="assistant"), _pctx())
    task = _root_task(task_id)
    state = _finalize_state(task, scope, LoopConfig())
    await finalize_task_memory(mem, state, task, f"out {task_id}", "success", _finalize_ctx(mem),
                               act_recap=f"act recap {task_id}", task_summary=f"summary {task_id}")


async def _alive_body(mem, task_id: str):
    """该 task 的 task 层 body（USER_PROMPT/LLM_RESPONSE）中未 superseded 的。"""
    return await mem.recall_recent(_task_scope(task_id),
                                   [T.USER_PROMPT, T.LLM_RESPONSE], 2000, _pctx())


async def _alive_finish_pair(mem, origin: str):
    """该 origin 在 agent scope 的 finish 对（AGENT_CONVERSATION_TURN）中未 superseded 的。"""
    recs = await mem.recall_recent(_agent_scope(), [T.AGENT_CONVERSATION_TURN], 2000, _pctx())
    return [r for r in recs if r.metadata.get("origin_task_id") == origin]


async def _alive_summaries(mem):
    return await mem.recall_recent(_agent_scope(), [T.AGENT_COMPACT_SUMMARY], 2000, _pctx())


# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l0_units_have_body_and_finish_pair() -> None:
    """sanity：finalize 后每个 L0 单元都有 task 层 body + agent 层 finish 对。"""
    mem = InMemoryMemoryProvider()
    await _make_l0_unit(mem, "R0", 0)
    assert await _alive_body(mem, "R0"), "L0 unit must have task-layer body"
    assert await _alive_finish_pair(mem, "R0"), "L0 unit must have agent-layer finish pair"


@pytest.mark.asyncio
async def test_fold_drops_whole_unit_body_and_finish() -> None:
    """超 keep_last 的最老单元**整体**折成摘要：task 层胶囊 + finish 对一起 supersede（不再有
    「body 删、finish 留」的 L1 中间态）；未超的保留完整胶囊。"""
    mem = InMemoryMemoryProvider()
    # keep_last=2，开 4 个结束 root → 最老 2 个折成摘要
    for i in range(4):
        await _make_l0_unit(mem, f"R{i}", i * 10)
    cfg = LoopConfig(compact_keep_last=2)
    await fold_root_experience(_fold_state(cfg), _fold_ctx(mem),
                               cfg.compact_keep_last, "folded")

    # 最老 2 个（R0,R1）：body + finish 对都没了（整单元折掉）
    for old in ("R0", "R1"):
        assert await _alive_body(mem, old) == [], f"{old}: body must be folded"
        assert await _alive_finish_pair(mem, old) == [], f"{old}: finish pair must be folded"
    # 最新 2 个（R2,R3）：完整胶囊 body + finish 对都在
    for kept in ("R2", "R3"):
        assert await _alive_body(mem, kept), f"{kept}: kept capsule body must survive"
        assert await _alive_finish_pair(mem, kept), f"{kept}: kept finish pair must survive"
    # 折出一条摘要
    summ = await _alive_summaries(mem)
    assert len(summ) == 1, "exactly one AGENT_COMPACT_SUMMARY"
    assert summ[0].role == "user"


@pytest.mark.asyncio
async def test_fold_keeps_only_keep_last_capsules() -> None:
    """keep_last=1，开 4 个 → 最老 3 个整体折成一条摘要，仅最新 1 个保留为完整胶囊。"""
    mem = InMemoryMemoryProvider()
    for i in range(4):
        await _make_l0_unit(mem, f"R{i}", i * 10)
    cfg = LoopConfig(compact_keep_last=1)
    await fold_root_experience(_fold_state(cfg), _fold_ctx(mem),
                               cfg.compact_keep_last, "folded")

    for folded in ("R0", "R1", "R2"):
        assert await _alive_body(mem, folded) == [], f"{folded}: body folded"
        assert await _alive_finish_pair(mem, folded) == [], f"{folded}: finish pair folded"
    assert await _alive_body(mem, "R3"), "R3: kept capsule body survives"
    assert await _alive_finish_pair(mem, "R3"), "R3: kept finish pair survives"
    assert len(await _alive_summaries(mem)) == 1


@pytest.mark.asyncio
async def test_count_root_residues_counts_capsules() -> None:
    """_count_root_residues 数「结束顶层单元（胶囊）」；折成摘要的单元不计。"""
    mem = InMemoryMemoryProvider()
    for i in range(4):
        await _make_l0_unit(mem, f"R{i}", i * 10)
    cfg = LoopConfig(compact_keep_last=2)
    state, ctx = _fold_state(cfg), _fold_ctx(mem)

    assert await _count_root_residues(state, ctx) == 4, "4 capsules before fold"
    await fold_root_experience(state, ctx, cfg.compact_keep_last, "folded")
    # 最老 2 个折成摘要 → 只剩 R2/R3 两个胶囊
    assert await _count_root_residues(state, ctx) == 2, "after fold, 2 capsules remain"


@pytest.mark.asyncio
async def test_summary_anchored_before_kept() -> None:
    """摘要锚到保留胶囊最早 ts − 1µs（不变量 A）。"""
    mem = InMemoryMemoryProvider()
    for i in range(4):
        await _make_l0_unit(mem, f"R{i}", i * 100)
    cfg = LoopConfig(compact_keep_last=1)
    await fold_root_experience(_fold_state(cfg), _fold_ctx(mem),
                               cfg.compact_keep_last, "folded")
    summ = await _alive_summaries(mem)
    assert len(summ) == 1
    # 保留胶囊 R3 的最早 ts；摘要须早于之
    kept_finish = await _alive_finish_pair(mem, "R3")
    kept_min = min(r.timestamp for r in kept_finish)
    assert summ[0].timestamp < kept_min, "summary must anchor before kept capsule"
