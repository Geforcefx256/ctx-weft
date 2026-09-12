"""conversation-pairing 回归（spec: conversation-integrity）。

覆盖：铸造形态与确定性、摄入锚采纳与伴随字段（两平面同值）、跨轮次/跨 task 重复
raw id 的重建配对、裁剪原子性、存量裸 id 行为不回退 + 歧义留痕、adapter 原样透传。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import ContextBlock
from ctx_weft.core.assembler.budget import PriorityBudgetStrategy
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources._history import record_to_history_block
from ctx_weft.core.loop.llm_gateway import legalize_messages, reorder_tool_results_after_calls
from ctx_weft.core.loop.steps.act import _ingest_assistant_turn
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import (
    INTERNAL_CALL_ID_MAX_LEN,
    INTERNAL_CALL_ID_RE,
    mint_call_id,
    mint_turn_call_ids,
)
from ctx_weft.protocols import (
    LLMMessage,
    LLMUsage,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
    ToolCall,
)
from ctx_weft.protocols.operations import operation_id_for
from ctx_weft.providers.llm.anthropic import _serialize_messages as _anthropic_serialize
from ctx_weft.providers.llm.openai import _serialize_messages as _openai_serialize
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

# ── 铸造原语 ──────────────────────────────────────────────────────────────────


def test_mint_form_charset_and_length():
    for raw in ("call_1", "a", "toolu_01XFDUDYJgAACzvnptvVoYEL", "x" * 200, ""):
        mid = mint_call_id(
            anchor="asst_01HTEST000000000000000000", ordinal=3, raw_id=raw, turn_seq=12
        )
        assert INTERNAL_CALL_ID_RE.match(mid), mid
        assert len(mid) <= INTERNAL_CALL_ID_MAX_LEN


def test_mint_deterministic_and_differentiating():
    a1 = mint_call_id(anchor="A", ordinal=0, raw_id="call_1", turn_seq=1)
    assert a1 == mint_call_id(anchor="A", ordinal=0, raw_id="call_1", turn_seq=1)
    assert a1 != mint_call_id(anchor="B", ordinal=0, raw_id="call_1", turn_seq=1)  # 跨回合
    assert a1 != mint_call_id(anchor="A", ordinal=1, raw_id="call_1", turn_seq=1)  # 跨调用


def test_mint_same_raw_within_turn_yields_distinct_ids():
    minted = mint_turn_call_ids(
        [
            ToolCall(id="call_1", name="t__a", arguments={}),
            ToolCall(id="call_1", name="t__b", arguments={}),
        ],
        anchor="asst_X",
        turn_seq=2,
    )
    ids = [m.call.id for m in minted]
    assert len(set(ids)) == 2
    assert [m.raw_id for m in minted] == ["call_1", "call_1"]
    assert [m.ordinal for m in minted] == [0, 1]


def test_mint_accepts_duck_typed_call_and_does_not_mutate():
    tc = SimpleNamespace(id="call_9", name="t__a", arguments={})
    minted = mint_turn_call_ids([tc], anchor="asst_Y", turn_seq=1)
    assert minted[0].call.id.startswith("tc_")
    assert minted[0].raw_id == "call_9"
    assert tc.id == "call_9"  # 原对象不动


# ── 摄入：锚采纳 + 伴随字段 + 两平面同值 ────────────────────────────────────


def _ingest_harness():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    state = SimpleNamespace(
        scope=scope,
        session=SimpleNamespace(id="s1", tenant_id="default"),
        agent=SimpleNamespace(id="ag1"),
    )
    ctx = SimpleNamespace(
        memory=mem,
        provider_ctx=SimpleNamespace(session_id="s1", tenant_id="default", task_id="t1"),
    )
    return state, ctx, mem


@pytest.mark.asyncio
async def test_ingest_adopts_anchor_and_records_companions():
    from ctx_weft.core.utils.ids import generate_id

    state, ctx, mem = _ingest_harness()
    anchor = generate_id("asst")
    minted = mint_turn_call_ids(
        [ToolCall(id="call_1", name="mcp__t__a", arguments={"q": 1})], anchor=anchor, turn_seq=5
    )

    persisted = await _ingest_assistant_turn(
        state, ctx, "hi", [m.call for m in minted], LLMUsage(), 5, anchor=anchor, minted=minted
    )

    # 锚被 provider 采纳（record_id == 预铸锚）；两平面取同一份铸造值。
    assert persisted.record_id == anchor
    assert persisted.tool_calls[0]["id"] == minted[0].call.id
    # 伴随字段：raw wire id 可追溯；op_id 与 _execute_tool_calls 的派生同口径。
    rec = await mem.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    asst = [r for r in rec if r.role == "assistant"][0]
    tc_md = asst.metadata["tool_calls"][0]
    assert tc_md["id"] == minted[0].call.id
    assert tc_md["raw_id"] == "call_1"
    assert tc_md["op_id"] == operation_id_for("default", "s1", "ag1", anchor, 0)


@pytest.mark.asyncio
async def test_ingest_without_minting_keeps_legacy_shape():
    """旧调用（无 anchor/minted）：行为逐字节同前——id 原样、无伴随字段。"""
    state, ctx, mem = _ingest_harness()
    persisted = await _ingest_assistant_turn(
        state, ctx, "hi", [ToolCall(id="call_1", name="mcp__t__a", arguments={})], LLMUsage(), 1
    )
    assert persisted.tool_calls[0]["id"] == "call_1"
    rec = await mem.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    tc_md = [r for r in rec if r.role == "assistant"][0].metadata["tool_calls"][0]
    assert "raw_id" not in tc_md and "op_id" not in tc_md


# ── 重建与合法化：重复 raw id 不复制不错配（内部标识平面） ───────────────────


def _pairing_records(raw_id: str = "call_1"):
    """两轮 assistant（同 raw id、不同锚）+ 各自 result + 间隔 user，模拟跨轮次复用。"""
    t0 = datetime(2026, 9, 1, tzinfo=UTC)

    def rec(rid, ts, role, content, md):
        return MemoryRecord(
            id=rid,
            type=MemoryEventType.LLM_RESPONSE
            if role == "assistant"
            else (MemoryEventType.TOOL_RESULT if role == "tool" else MemoryEventType.USER_PROMPT),
            content=content,
            timestamp=ts,
            role=role,
            metadata=md,
        )

    out = []
    for turn, (anchor, seq) in enumerate([("asst_A", 1), ("asst_B", 2)], start=1):
        minted = mint_turn_call_ids(
            [ToolCall(id=raw_id, name="mcp__t__x", arguments={})], anchor=anchor, turn_seq=seq
        )
        mid = minted[0].call.id
        out.append(
            rec(
                f"u{turn}",
                t0 + timedelta(minutes=10 * turn - 10),
                "user",
                f"q{turn}",
                {"task_id": "t1"},
            )
        )
        out.append(
            rec(
                f"a{turn}",
                t0 + timedelta(minutes=10 * turn - 5),
                "assistant",
                f"r{turn}",
                {"task_id": "t1", "tool_calls": [{"id": mid, "name": "mcp__t__x", "input": {}}]},
            )
        )
        out.append(
            rec(
                f"res{turn}",
                t0 + timedelta(minutes=10 * turn - 1),
                "tool",
                f"out{turn}",
                {"task_id": "t1", "tool_call_id": mid},
            )
        )
    return out


def _to_messages(records):
    req = SimpleNamespace(token_counter=lambda t: max(1, len(t) // 3))
    blocks = [
        record_to_history_block(r, "agent_recall", i, request=req, current_task_id="t1")
        for i, r in enumerate(records)
    ]
    return DefaultComposer()._history_to_messages_with_sources(blocks)


def test_rebuild_pairs_each_result_to_its_true_owner():
    msgs = [m for m, *_ in _to_messages(_pairing_records())]
    # 每条 result 恰出现一次、紧邻其 owner（重建按 ts 正序，无跨轮串扰）。
    assistant_ids = [[tc["id"] for tc in m.tool_calls] for m in msgs if m.role == "assistant"]
    tool_ids = [m.tool_call_id for m in msgs if m.role == "tool"]
    assert len(tool_ids) == 2 and len(set(tool_ids)) == 2
    for aid in [i for ids in assistant_ids for i in ids]:
        assert aid in tool_ids


def test_legalize_no_duplication_no_orphans_on_internal_ids():
    msgs = [m for m, *_ in _to_messages(_pairing_records())]
    out = legalize_messages(msgs)
    # a(tool_call) → tool(result) 交替，各一次；无孤立、无悬挂。
    tool_calls = [(m, tc["id"]) for m in out if m.role == "assistant" for tc in m.tool_calls]
    results = [m for m in out if m.role == "tool"]
    assert len(results) == 2
    assert {m.tool_call_id for m in results} == {tcid for _, tcid in tool_calls}
    for i, m in enumerate(out):
        if m.role == "assistant" and m.tool_calls:
            assert out[i + 1].role == "tool" and out[i + 1].tool_call_id == m.tool_calls[0]["id"]


# ── 跨 task 召回混合 + 裁剪原子性 ─────────────────────────────────────────────


def _blk(bid, priority, tokens, *, ts, role, mtype, task_id, tool_calls=None, tool_call_id=""):
    md = {"timestamp": ts, "role": role, "type": mtype, "task_id": task_id}
    if tool_calls is not None:
        md["tool_calls"] = tool_calls
    if tool_call_id:
        md["tool_call_id"] = tool_call_id
    return ContextBlock(
        id=bid,
        source="agent_recall",
        kind="history",
        target="messages",
        content="x",
        priority=priority,
        token_estimate=tokens,
        metadata=md,
    )


def _dup_raw_blocks(task_a: str, task_b: str):
    """两个 task 各一组 assistant+result，raw id 同为 call_1（内部 id 因锚不同而互异）。"""
    m1 = mint_turn_call_ids(
        [ToolCall(id="call_1", name="t", arguments={})], anchor="asst_A", turn_seq=1
    )[0].call.id
    m2 = mint_turn_call_ids(
        [ToolCall(id="call_1", name="t", arguments={})], anchor="asst_B", turn_seq=2
    )[0].call.id
    return [
        _blk(
            "a1",
            6,
            100,
            ts="2026-01-01T00:00:10",
            role="assistant",
            mtype="llm_response",
            task_id=task_a,
            tool_calls=[{"id": m1, "name": "t", "input": {}}],
        ),
        _blk(
            "r1",
            6,
            100,
            ts="2026-01-01T00:00:11",
            role="tool",
            mtype="tool_result",
            task_id=task_a,
            tool_call_id=m1,
        ),
        _blk(
            "a2",
            6,
            100,
            ts="2026-01-01T00:00:20",
            role="assistant",
            mtype="llm_response",
            task_id=task_b,
            tool_calls=[{"id": m2, "name": "t", "input": {}}],
        ),
        _blk(
            "r2",
            6,
            100,
            ts="2026-01-01T00:00:21",
            role="tool",
            mtype="tool_result",
            task_id=task_b,
            tool_call_id=m2,
        ),
    ]


def test_coalesce_groups_by_true_owner_across_tasks():
    blocks = _dup_raw_blocks("tA", "tB")
    units = PriorityBudgetStrategy._coalesce_tool_pairs(blocks)
    by_first = {u[0].id: sorted(b.id for b in u) for u in units}
    assert by_first["a1"] == ["a1", "r1"]  # 旧 result 不归到新调用下
    assert by_first["a2"] == ["a2", "r2"]


def _req(task_id="cur"):
    return SimpleNamespace(
        task=SimpleNamespace(id=task_id),
        session=SimpleNamespace(context_limit=180_000, reserved_output_tokens=8192),
    )


@pytest.mark.asyncio
async def test_pruning_drops_call_and_result_as_unit():
    """超限裁剪丢 a1 单元 → r1 同批消失；幸存面无孤立 result。"""
    blocks = _dup_raw_blocks("tA", "tB")
    kept = await PriorityBudgetStrategy().apply(blocks, token_limit=200, request=_req())
    assert {b.id for b in kept} == {"a2", "r2"}  # 最老单元（a1+r1）整体丢弃
    kept_tools = {b.metadata.get("tool_call_id") for b in kept if b.metadata.get("role") == "tool"}
    kept_calls = {
        tc["id"] for b in kept if b.metadata.get("tool_calls") for tc in b.metadata["tool_calls"]
    }
    assert kept_tools == kept_calls  # 无孤立 result / 无悬挂 call


# ── 存量裸 id：行为不回退 + 歧义留痕 ──────────────────────────────────────────


def _legacy_dup_messages():
    """改造前形态：两个 assistant 携带同一裸 wire id call_1。"""
    tc = [{"id": "call_1", "name": "t", "input": {}}]
    return [
        LLMMessage(role="user", content="q"),
        LLMMessage(role="assistant", content="a1", tool_calls=list(tc)),
        LLMMessage(role="tool", content="r1", tool_call_id="call_1"),
        LLMMessage(role="assistant", content="a2", tool_calls=list(tc)),
        LLMMessage(role="tool", content="r2", tool_call_id="call_1"),
    ]


def test_legacy_duplicate_ids_behavior_unchanged_but_logged(caplog):
    with caplog.at_level(logging.ERROR, logger="ctx_weft.core.loop.llm_gateway"):
        out = reorder_tool_results_after_calls(_legacy_dup_messages())
    # 行为维持现状：每个携带者后各发一遍（r1, r2 在 a1 与 a2 之后都出现）。
    seq = [(m.role, m.content if m.role == "tool" else "") for m in out]
    tools = [c for r, c in seq if r == "tool"]
    assert tools == ["r1", "r2", "r1", "r2"]
    # 歧义留痕：ERROR 指明重复 id。
    assert any(
        "call_1" in r.message and "multiple assistant" in r.message for r in caplog.records
    ), caplog.text


def test_internal_ids_no_duplication_no_log(caplog):
    msgs = [m for m, *_ in _to_messages(_pairing_records())]
    with caplog.at_level(logging.ERROR, logger="ctx_weft.core.loop.llm_gateway"):
        out = legalize_messages(msgs)
    tools = [m.content for m in out if m.role == "tool"]
    assert sorted(tools) == ["out1", "out2"]  # 各一次
    assert not any("multiple assistant" in r.message for r in caplog.records)


# ── 段压缩后重建（贯通：多轮工具循环 → fold 早期段 → 重建 → 配对完整） ─────


@pytest.mark.asyncio
async def test_rebuild_after_fold_keeps_pairing():
    from ctx_weft.core.utils.ids import generate_id

    state, ctx, mem = _ingest_harness()

    async def _ingest_user(text: str) -> str:
        return await ctx.memory.ingest(
            MemoryEvent(
                kind=MemoryKind.CONVERSATION_TURN,
                scope=MemoryScope.TASK,
                address=state.scope,
                content=text,
                timestamp=now_utc(),
                role="user",
                metadata={"task_id": "t1"},
            ),
            ctx.provider_ctx,
        )

    turn_ids: list[tuple[list[str], str]] = []
    for turn, tool in enumerate(["mcp__t__a", "mcp__t__b"], start=1):
        anchor = generate_id("asst")
        user_id = await _ingest_user(f"q{turn}")
        minted = mint_turn_call_ids(
            [ToolCall(id="call_1", name=tool, arguments={})], anchor=anchor, turn_seq=turn
        )
        persisted = await _ingest_assistant_turn(
            state,
            ctx,
            f"a{turn}",
            [m.call for m in minted],
            LLMUsage(),
            turn,
            anchor=anchor,
            minted=minted,
        )
        res_id = await ctx.memory.ingest(
            MemoryEvent(
                kind=MemoryKind.CONVERSATION_TURN,
                scope=MemoryScope.TASK,
                address=state.scope,
                content=f"out{turn}",
                timestamp=now_utc(),
                role="tool",
                metadata={
                    "tool_call_id": minted[0].call.id,
                    "tool_name": tool,
                    "invocation_id": f"inv{turn}",
                },
            ),
            ctx.provider_ctx,
        )
        turn_ids.append(([user_id, persisted.record_id, res_id], minted[0].call.id))

    # 段压缩语义：fold 掉第一轮（user + assistant + result，supersede 不可逆）。
    await mem.fold(turn_ids[0][0], [], ctx.provider_ctx)

    view = await mem.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
    req = SimpleNamespace(token_counter=lambda t: max(1, len(t) // 3))
    blocks = [
        record_to_history_block(r, "agent_recall", i, request=req, current_task_id="t1")
        for i, r in enumerate(view)
    ]
    msgs = [m for m, *_ in DefaultComposer()._history_to_messages_with_sources(blocks)]
    out = legalize_messages(msgs)

    # 幸存轮配对完整：assistant(tool_call) 紧邻其唯一 result，无孤立/重复/悬挂。
    assert [m.role for m in out] == ["user", "assistant", "tool"]
    a2, r2 = out[1], out[2]
    assert a2.tool_calls[0]["id"] == turn_ids[1][1] == r2.tool_call_id


# ── adapter 原样透传 ─────────────────────────────────────────────────────────


def _minted_pair():
    minted = mint_turn_call_ids(
        [ToolCall(id="call_1", name="t__x", arguments={"q": 1})],
        anchor="asst_01HZTEST000000000000000000",
        turn_seq=1,
    )
    return minted[0].call


def test_anthropic_serializes_minted_ids_verbatim():
    tc = _minted_pair()
    msgs = [
        LLMMessage(role="user", content="q"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": tc.id, "name": tc.name, "input": tc.arguments}],
        ),
        LLMMessage(role="tool", content="ok", tool_call_id=tc.id),
    ]
    payload = _anthropic_serialize(msgs)
    tool_use = payload[1]["content"][0]
    assert tool_use["type"] == "tool_use" and tool_use["id"] == tc.id
    tool_result = payload[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == tc.id
    # provider 约束：字符集与长度。
    assert INTERNAL_CALL_ID_RE.match(tc.id) and len(tc.id) <= INTERNAL_CALL_ID_MAX_LEN


def test_openai_serializes_minted_ids_verbatim():
    tc = _minted_pair()
    msgs = [
        LLMMessage(role="user", content="q"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": tc.id, "name": tc.name, "input": tc.arguments}],
        ),
        LLMMessage(role="tool", content="ok", tool_call_id=tc.id),
    ]
    payload = _openai_serialize("", msgs)
    wire_tc = payload[1]["tool_calls"][0]
    assert wire_tc["id"] == tc.id
    assert payload[2]["tool_call_id"] == tc.id
