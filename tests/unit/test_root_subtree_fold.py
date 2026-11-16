"""fold_root_experience：新 Task-6 格式（AGENT_CONVERSATION_TURN 胶囊）的检测与折叠（§3.11）。

旧格式（TASK_DISPATCH_RESULT(parent=None) 作 root 标记）已在 spec §3.11 废弃；
新实现按 origin_task_id 分组 AGENT_CONVERSATION_TURN，parent_task_id 判据决定顶层单元。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.compaction_testkit import _count_root_residues, fold_root_experience
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _sc(agent="ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=None, agent_id=agent)


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


def _state(scope):
    """spec 2026-06-29 单一阈值 fold：keep_last 是唯一门槛（保 N 个胶囊、更老折成摘要），
    keep_pair 已删。keep_last 由各用例直接传给 fold_root_experience。"""
    agent = SimpleNamespace(id=scope.agent_id, loop_config=LoopConfig())
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="run1", sequence_counter=0, session=session,
                           scope=scope, task=SimpleNamespace(id="cur"), agent=agent)


def _ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_pctx())


async def _seed_root_capsule(mem, scope, task_id: str, t0: int, parent_task_id=None):
    """Task-4 task-resident：一份结束单元 = task 层 body（task_id scope 的 USER_PROMPT）
    + agent 层 finish 对（4x AGENT_CONVERSATION_TURN，parent_task_id=None for root）。

    body → user prompt → assistant summary → assistant finish_task tool_call → tool result
    （L1 删 task 层 body；L2 连 finish 对一并删）。
    """
    tcid = f"tc_{task_id}"
    # task 层 body（按 origin task_id 的 task scope，同 agent_id 供 recall_recent_by_agent 召回）
    body_scope = MemoryAddress(session_id="s1", task_id=task_id, agent_id=scope.agent_id)
    await mem.ingest(_ev(T.USER_PROMPT, body_scope, f"body prompt {task_id}", t0,
                         role="user"), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, f"user prompt {task_id}", t0,
                         role="user", origin_task_id=task_id, parent_task_id=parent_task_id), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, f"assistant summary {task_id}", t0 + 1,
                         role="assistant", origin_task_id=task_id, parent_task_id=parent_task_id), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, "", t0 + 2,
                         role="assistant", origin_task_id=task_id, parent_task_id=parent_task_id,
                         tool_calls=[{"id": tcid, "name": "control:finish_task", "input": {}}]), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, f"Process Report: done {task_id}", t0 + 3,
                         role="tool", origin_task_id=task_id, parent_task_id=parent_task_id,
                         tool_call_id=tcid), _pctx())


async def _seed_root_capsule_with_dispatch(mem, scope, task_id: str, t0: int, parent_task_id=None):
    """新格式胶囊 + cross-agent dispatch pair（parent_task_id 指向 delegating task）。"""
    await _seed_root_capsule(mem, scope, task_id, t0, parent_task_id=parent_task_id)


async def _seed_capsule_offset(mem, scope, task_id: str, turn_t: int, result_t: int):
    """多回合胶囊：前两轮在 turn_t（早），finish 对在 result_t（晚）。
    用于验证 anchor_ts 必须扫描全部元素（不能只看最新回合）。"""
    tcid = f"tc_{task_id}"
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, f"user prompt {task_id}", turn_t,
                         role="user", origin_task_id=task_id, parent_task_id=None), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, f"assistant summary {task_id}", turn_t + 1,
                         role="assistant", origin_task_id=task_id, parent_task_id=None), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, "", result_t - 1,
                         role="assistant", origin_task_id=task_id, parent_task_id=None,
                         tool_calls=[{"id": tcid, "name": "control:finish_task", "input": {}}]), _pctx())
    await mem.ingest(_ev(T.AGENT_CONVERSATION_TURN, scope, f"Process Report: done {task_id}", result_t,
                         role="tool", origin_task_id=task_id, parent_task_id=None,
                         tool_call_id=tcid), _pctx())


async def _alive(mem, scope, type_, role=None):
    recs = await mem.recall_recent(scope, [type_], 2000, _pctx())
    return [r for r in recs if role is None or r.role == role]


# ──────────────────────────────────────────────────────────────────────────────
# 既有测试（已迁移到 Task-6 真实格式）
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_folded_capsule_conversation_turns_superseded():
    """被折胶囊（最旧）的所有 AGENT_CONVERSATION_TURN 被 superseded，不落单。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    # keep_last=1 → 3 个胶囊中折掉最旧 2 个
    for i, t in enumerate([0, 10, 20]):
        await _seed_root_capsule(mem, scope, f"task{i}", t * 10)
    n = await fold_root_experience(_state(scope), _ctx(mem), keep_last=1, summary_text="folded")
    assert n > 0
    turns = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    alive_tasks = {r.metadata.get("origin_task_id") for r in turns}
    assert alive_tasks == {"task2"}  # 只剩最新胶囊的 turns


@pytest.mark.asyncio
async def test_kept_capsule_intact():
    """保留胶囊（最新 keep_last 个）的所有 AGENT_CONVERSATION_TURN 完整存活。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    for i, t in enumerate([0, 10, 20]):
        await _seed_root_capsule(mem, scope, f"task{i}", t * 10)
    await fold_root_experience(_state(scope), _ctx(mem), keep_last=2, summary_text="folded")
    turns = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    alive_tasks = {r.metadata.get("origin_task_id") for r in turns}
    assert alive_tasks == {"task1", "task2"}


@pytest.mark.asyncio
async def test_compact_summary_sorts_before_kept_capsule():
    """新 AGENT_COMPACT_SUMMARY 的 timestamp 早于最旧保留胶囊所有元素（anchor − 1µs）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    for i, t in enumerate([0, 10, 20]):
        await _seed_root_capsule(mem, scope, f"task{i}", t * 10)
    await fold_root_experience(_state(scope), _ctx(mem), keep_last=1, summary_text="folded")
    summ = await _alive(mem, scope, T.AGENT_COMPACT_SUMMARY)
    assert len(summ) == 1
    # 保留胶囊 task2 的最早回合在 t=200s
    kept_min_ts = _BASE + timedelta(seconds=200)
    assert summ[0].timestamp < kept_min_ts


@pytest.mark.asyncio
async def test_nothing_to_fold_returns_zero():
    """root 残留 ≤ keep_last → 不折，conversation turns 全留。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    await _seed_root_capsule(mem, scope, "task0", 0)
    n = await fold_root_experience(_state(scope), _ctx(mem), keep_last=6, summary_text="x")
    assert n == 0
    turns = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    assert {r.metadata.get("origin_task_id") for r in turns} == {"task0"}


@pytest.mark.asyncio
async def test_anchor_ts_covers_all_kept_capsule_elements():
    """anchor_ts 必须扫描保留胶囊的全部回合（最早 user 回合），不能只看最晚回合。

    构造：2 个胶囊，keep_last=1（折掉第 1 个，保留第 2 个）。
    第 2 个（被保留）胶囊的首回合 user 在 turn_t=5，finish 回合在 result_t=20。
    正确 anchor < _BASE + 5s；错误 anchor = _BASE + 20s - 1µs。
    """
    mem = InMemoryMemoryProvider()
    scope = _sc()
    # 第 1 个胶囊（会被折掉）
    await _seed_capsule_offset(mem, scope, "task0", turn_t=0, result_t=3)
    # 第 2 个胶囊（会被保留）：首 user 回合在 t=5，finish 对在 t=20/21
    await _seed_capsule_offset(mem, scope, "task1", turn_t=5, result_t=20)
    await fold_root_experience(_state(scope), _ctx(mem), keep_last=1, summary_text="compacted")
    summ = await _alive(mem, scope, T.AGENT_COMPACT_SUMMARY)
    assert len(summ) == 1
    kept_turn_earliest = _BASE + timedelta(seconds=5)
    assert summ[0].timestamp < kept_turn_earliest, (
        f"AGENT_COMPACT_SUMMARY.timestamp={summ[0].timestamp} should be < {kept_turn_earliest} "
        f"(min of kept capsule's first AGENT_CONVERSATION_TURN)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 新增测试（Task 12：§3.11 新判据）
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_count_root_residues_detection():
    """_count_root_residues 数「结束顶层单元（胶囊）」（spec 2026-06-29）。

    keep_last+2 个胶囊 → count == keep_last+2。fold（keep_last=2）后，最旧 2 个折成摘要
    （task 层胶囊 + finish 对均 supersede），count == keep_last，AGENT_COMPACT_SUMMARY 存在
    （role=user）。
    """
    keep_last = 2
    total = keep_last + 2
    mem = InMemoryMemoryProvider()
    scope = _sc()
    for i in range(total):
        await _seed_root_capsule(mem, scope, f"R{i}", i * 20)
    state = _state(scope)
    ctx = _ctx(mem)

    count_before = await _count_root_residues(state, ctx)
    assert count_before == total, f"expected {total}, got {count_before}"

    await fold_root_experience(state, ctx, keep_last=keep_last, summary_text="compacted")

    count_after = await _count_root_residues(state, ctx)
    assert count_after == keep_last, f"after fold expected {keep_last}, got {count_after}"

    summ = await _alive(mem, scope, T.AGENT_COMPACT_SUMMARY)
    assert len(summ) == 1
    assert summ[0].role == "user"


@pytest.mark.asyncio
async def test_no_l1_capsule_keeps_body_until_folded_to_summary():
    """删 L1 后只剩 3 级（执行中 / 胶囊 / 总结）：胶囊（task 层 body + finish 对）一直完整保留,
    直到超 keep_last 才**整体**折成摘要。不存在「body 删了但 finish 对还在」的 L1 中间态。

    构造 3 个胶囊,keep_last=2（keep_pair 在旧两档模型下是 30 > keep_last → 旧码会把 R0 降 L1：
    删 body、留 finish、不出摘要。新模型应：R0 整体折成摘要,R1/R2 保留完整胶囊含 task 层 body）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    for i, t in enumerate([0, 10, 20]):
        await _seed_root_capsule(mem, scope, f"R{i}", t * 10)

    n = await fold_root_experience(_state(scope), _ctx(mem),
                                   keep_last=2, summary_text="folded")
    assert n > 0

    turns = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    body = await mem.recall_recent_by_agent(scope, [T.USER_PROMPT], 2000, _pctx())
    alive_origins = {r.metadata.get("origin_task_id") for r in turns}
    alive_body = {r.metadata.get("task_id") for r in body}

    # R0 超出 keep_last → 整体折成摘要：finish 对 + task 层 body 都没了（不是只删 body 的 L1）
    assert "R0" not in alive_origins, "R0 finish 对应随单元折掉（无 L1 中间态）"
    assert "R0" not in alive_body, "R0 task 层 body 应随单元一起折掉"
    # R1/R2 保留为完整胶囊：finish 对在 + task 层 body 也在（胶囊不做 L1 黑盒化）
    assert {"R1", "R2"} <= alive_origins, "保留胶囊的 finish 对须存活"
    assert {"R1", "R2"} <= alive_body, "保留胶囊的 task 层 body 不应被删（无 L1）"
    # 折出一条摘要
    summ = await _alive(mem, scope, T.AGENT_COMPACT_SUMMARY)
    assert len(summ) == 1


@pytest.mark.asyncio
async def test_same_agent_nested_child_folded_with_parent():
    """折顶层 R 时，同 agent 内嵌子胶囊 C1（origin=C1, parent=R）也被 supersede。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    # R0：顶层 root，t=0
    await _seed_root_capsule(mem, scope, "R0", 0, parent_task_id=None)
    # C1：R0 的同 agent 子胶囊，t=10
    await _seed_root_capsule(mem, scope, "C1", 10, parent_task_id="R0")
    # R1：另一个顶层 root，t=40（保留）
    await _seed_root_capsule(mem, scope, "R1", 40, parent_task_id=None)

    # keep_last=1 → 折 R0（R0 是最旧顶层），R0 的后代 C1 也应被折
    n = await fold_root_experience(_state(scope), _ctx(mem), keep_last=1, summary_text="folded")
    assert n > 0

    turns = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    alive_origins = {r.metadata.get("origin_task_id") for r in turns}
    # R0 和 C1 均应被 supersede
    assert "R0" not in alive_origins, "R0 should be folded"
    assert "C1" not in alive_origins, "C1 (child of R0) should be folded with R0"
    assert "R1" in alive_origins, "R1 (kept) must survive"


def _seed_dispatch_pair(mem_ingest_coros, scope, origin, tc, t0, result="child result"):
    """造一组 dispatch 对（新表示，§2.3）：delegate assistant 回合 + result tool 回合，
    均为 AGENT_CONVERSATION_TURN、origin=delegating task → 与该单元 finish 对同 origin、同命运。
    返回待 ingest 的协程列表（调用方逐个 await）。"""
    return [
        mem_ingest_coros(_ev(T.AGENT_CONVERSATION_TURN, scope, "", t0, role="assistant",
                             origin_task_id=origin, parent_task_id=None,
                             tool_calls=[{"id": tc, "name": "control:delegate_task", "input": {}}]),
                         _pctx()),
        mem_ingest_coros(_ev(T.AGENT_CONVERSATION_TURN, scope, result, t0 + 1, role="tool",
                             origin_task_id=origin, tool_call_id=tc), _pctx()),
    ]


@pytest.mark.asyncio
async def test_legacy_dispatch_pair_folds_with_unit_via_adapter():
    """§5.5：存量 legacy TASK_DISPATCH/RESULT 经读侧适配归一化后，作为 R0 单元的 conversation turn
    随单元 L2 一并折——验证 fold 已接入 normalize_legacy_dispatch（旧数据不丢、行为与新数据一致）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    await _seed_root_capsule(mem, scope, "R0", 0, parent_task_id=None)
    # legacy 表示：gateway 旧写 TASK_DISPATCH（scope task_id=R0 → 适配取 origin=R0）+ finalize 旧写
    # TASK_DISPATCH_RESULT（parent_task_id=R0 → 适配取 origin=R0）
    tc = "tc_legacy"
    legacy_scope = MemoryAddress(session_id="s1", task_id="R0", agent_id=scope.agent_id)
    await mem.ingest(_ev(T.TASK_DISPATCH, legacy_scope, "", 10, role="assistant",
                         tool_call_id=tc, tool_name="control:delegate_task", arguments={}), _pctx())
    await mem.ingest(_ev(T.TASK_DISPATCH_RESULT, legacy_scope, "legacy child result", 11, role="tool",
                         tool_call_id=tc, parent_task_id="R0", child_task_id="C2"), _pctx())
    await _seed_root_capsule(mem, scope, "R1", 40, parent_task_id=None)

    n = await fold_root_experience(_state(scope), _ctx(mem), keep_last=1, summary_text="folded")
    assert n > 0

    # legacy dispatch 对随 R0 单元 L2 一并 supersede（适配后归 origin=R0）
    assert await _alive(mem, scope, T.TASK_DISPATCH_RESULT) == [], \
        "legacy dispatch result must fold with R0 unit via adapter"
    assert await _alive(mem, scope, T.TASK_DISPATCH) == [], \
        "legacy dispatch must fold with R0 unit via adapter"
    # R0 finish 对没了（整单元 L2）；R1 保留
    alive = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    assert "R0" not in {r.metadata.get("origin_task_id") for r in alive}
    assert "R1" in {r.metadata.get("origin_task_id") for r in alive}


@pytest.mark.asyncio
async def test_cross_agent_dispatch_pair_folded_with_parent():
    """§2.3：折顶层 R0 到 L2 时，其 dispatch 对（delegate + result conversation turn, origin=R0）
    作为 R0 对话里的一组普通 message，与 finish 对同命运一起被 supersede。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    # R0：顶层 root，t=0
    await _seed_root_capsule(mem, scope, "R0", 0, parent_task_id=None)
    # dispatch 对：R0 派发的子任务 C2（delegate + result，origin=R0）
    tc = "tc_cross_C2"
    for coro in _seed_dispatch_pair(mem.ingest, scope, "R0", tc, 10):
        await coro
    # R1：另一个顶层 root，t=40（保留）
    await _seed_root_capsule(mem, scope, "R1", 40, parent_task_id=None)

    n = await fold_root_experience(_state(scope), _ctx(mem), keep_last=1, summary_text="folded")
    assert n > 0

    alive = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    # dispatch result（origin=R0）随 R0 单元 L2 一并 supersede
    assert tc not in {r.metadata.get("tool_call_id") for r in alive if r.role == "tool"}, \
        "dispatch result (origin=R0) must fold with R0 unit at L2"
    # delegate 回合（origin=R0）也被 supersede
    assert [r for r in alive if r.role == "assistant"
            and any(t.get("id") == tc for t in (r.metadata.get("tool_calls") or []))] == [], \
        "delegate turn (origin=R0) must fold with R0 unit at L2"
    # R0 finish 对没了（整单元 L2）；R1 保留
    assert "R0" not in {r.metadata.get("origin_task_id") for r in alive}
    assert "R1" in {r.metadata.get("origin_task_id") for r in alive}


@pytest.mark.asyncio
async def test_kept_dispatch_capsule_retains_body_no_l1():
    """spec 2026-06-29：删 L1 后，未超 keep_last 的胶囊**完整保留**——含 dispatch 对的单元,其
    task 层 body + dispatch 对 + finish 对都不动（不存在「body 删、对话留」的 L1 黑盒态）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    await _seed_root_capsule(mem, scope, "R0", 0, parent_task_id=None)
    tc = "tc_C2"
    for coro in _seed_dispatch_pair(mem.ingest, scope, "R0", tc, 10):
        await coro
    await _seed_root_capsule(mem, scope, "R1", 40, parent_task_id=None)
    await _seed_root_capsule(mem, scope, "R2", 80, parent_task_id=None)

    # keep_last=3 → 3 个胶囊都保留，无折叠
    n = await fold_root_experience(_state(scope), _ctx(mem), keep_last=3, summary_text="folded")
    assert n == 0

    # R0 task 层 body 完整在（不做 L1 删）
    body = await mem.recall_recent_by_agent(scope, [T.USER_PROMPT], 2000, _pctx())
    assert "R0" in {r.metadata.get("task_id") for r in body}, "kept capsule's task body must remain"
    alive = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    # R0 的 dispatch 对 + finish 对都在
    assert tc in {r.metadata.get("tool_call_id") for r in alive if r.role == "tool"}, \
        "kept capsule's dispatch result must remain"
    assert "R0" in {r.metadata.get("origin_task_id") for r in alive}, "kept capsule's finish pair must remain"
    # 无折叠 → 无 AGENT_COMPACT_SUMMARY
    assert await _alive(mem, scope, T.AGENT_COMPACT_SUMMARY) == []


@pytest.mark.asyncio
async def test_cross_agent_child_counts_as_top_level_in_own_scope():
    """在 child agent 的 scope 内，胶囊 origin=C2, parent=R（R 不在该 scope origin 集）→ 算 1 顶层单元。

    OR 判据：parent_task_id ∉ scope origin 集 → 顶层单元。
    """
    mem = InMemoryMemoryProvider()
    # child agent 有自己的 scope（agent_id 不同或 task_id 不同均可）
    child_scope = MemoryAddress(session_id="s1", task_id=None, agent_id="child_agent")

    # 仅植入 C2 的胶囊（parent=R，但 R 不在 child_scope 的任何胶囊 origin 中）
    await _seed_root_capsule(mem, child_scope, "C2", 0, parent_task_id="R")

    child_state = SimpleNamespace(
        run_id="run1", sequence_counter=0,
        session=SimpleNamespace(id="s1", tenant_id="default"),
        scope=child_scope,
        task=SimpleNamespace(id="cur"),
        agent=SimpleNamespace(id="child_agent", loop_config=LoopConfig()),
    )
    ctx = _ctx(mem)

    count = await _count_root_residues(child_state, ctx)
    assert count == 1, (
        f"C2 (parent=R, R not in scope) should count as 1 top-level unit, got {count}"
    )


@pytest.mark.asyncio
async def test_anchor_ordering_before_all_kept_capsule_elements():
    """AGENT_COMPACT_SUMMARY.timestamp < min timestamp of ALL kept capsule elements（§3.11 anchor）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc()
    # 3 capsules: fold R0, keep R1+R2
    await _seed_root_capsule(mem, scope, "R0", 0)
    await _seed_root_capsule(mem, scope, "R1", 100)  # earliest kept: t=100
    await _seed_root_capsule(mem, scope, "R2", 200)

    await fold_root_experience(_state(scope), _ctx(mem), keep_last=2, summary_text="anchor test")

    summ = await _alive(mem, scope, T.AGENT_COMPACT_SUMMARY)
    assert len(summ) == 1

    kept_turns = await _alive(mem, scope, T.AGENT_CONVERSATION_TURN)
    min_kept_ts = min(r.timestamp for r in kept_turns)
    assert summ[0].timestamp < min_kept_ts, (
        f"AGENT_COMPACT_SUMMARY.timestamp={summ[0].timestamp} must be < "
        f"min kept element ts={min_kept_ts}"
    )
