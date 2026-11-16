"""Exercise the actual stage policy and lifecycle events, without mocking phases."""

from types import SimpleNamespace

import pytest

from ctx_weft.protocols import MemoryScope, load_records
from ctx_weft.protocols.events import EventType
from tests import compaction_testkit as cm
from tests.model_testkit import model_snapshot
from tests.unit.test_compact_history_priority import capsule, setup



def _state(ratio_trigger=0.8, ratio_target=0.6, limit=1000):
    agent = SimpleNamespace(
        id="a",
        loop_config=SimpleNamespace(
            compact_keep_last=6,
            collapse_keep_last=3,
            compact_token_ratio=ratio_trigger,
            compact_target_ratio=ratio_target,
        ),
        loop_guard=SimpleNamespace(context_limit=limit),
    )
    return SimpleNamespace(
        resolved_model=model_snapshot(context_limit=limit),
        scope=SimpleNamespace(session_id="s1", agent_id="a", task_id="t1"),
        task=SimpleNamespace(id="t1"),
        agent=agent,
        session=SimpleNamespace(id="s1", tenant_id="tn"),
        extra={},
        run_id="r1",
        sequence_counter=0,
    )


async def _const(value):
    return value


@pytest.mark.asyncio
async def test_stops_after_l1_when_under_target(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    for i in range(3):
        await capsule(state, ctx, f"old{i}", i)

    async def unexpected(*args, **kwargs):
        pytest.fail("L1 already meets the target")

    cm.patch_summary(monkeypatch, unexpected)
    events = await cm.escalating_compact(state, ctx, token_estimate=58_100)
    levels = [e.payload for e in events if e.type == EventType.MEMORY_COMPACTED]
    assert [p["source"] for p in levels] == ["demote_lean"]
    assert levels[0]["demoted_capsules"] == 1
    body = await load_records(
        ctx.memory, cm._agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx
    )
    assert "old0-body" not in {r.id for r in body}
    assert "old1-body" in {r.id for r in body}


@pytest.mark.asyncio
async def test_escalates_l1_l2_l3(monkeypatch):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    for i in range(8):
        await capsule(state, ctx, f"old{i}", i)
    scopes = []

    async def summary(state, ctx, *, scope="task", record_ids=None):
        scopes.append(scope)
        return "digest"

    cm.patch_summary(monkeypatch, summary)
    events = await cm.escalating_compact(state, ctx, token_estimate=90_000)
    assert scopes == ["agent", "agent", "task"]
    assert [e.payload["source"] for e in events if e.type == EventType.MEMORY_COMPACTED] == [
        "demote_lean",
        "root_experience",
        "collapse",
    ]


@pytest.mark.asyncio
async def test_invalid_window_rejected_before_compaction():
    with pytest.raises(ValueError, match="context_limit"):
        _state(limit=0)


@pytest.mark.asyncio
async def test_finished_event_aggregates(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0)
    events = await cm.escalating_compact(state, ctx, token_estimate=58_100)
    levels = [e.payload for e in events if e.type == EventType.MEMORY_COMPACTED]
    finished = events[-1].payload
    assert finished["total_superseded"] == sum(p["superseded_count"] for p in levels)
    assert finished["freed_tokens"] == sum(p["freed_tokens"] for p in levels) > 0
    assert finished["levels"] == ["demote_lean"]
    assert finished["est_after"] < finished["target_tokens"]


@pytest.mark.asyncio
async def test_no_started_no_finished_when_under_target():
    assert (
        await cm.escalating_compact(
            _state(), SimpleNamespace(memory=None, provider_ctx=None), token_estimate=500
        )
        == []
    )


@pytest.mark.asyncio
async def test_finished_emitted_when_nothing_folded(monkeypatch):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    state.agent.loop_config.collapse_keep_last = 100
    events = await cm.escalating_compact(state, ctx, token_estimate=90_000)
    assert events[-1].type == EventType.MEMORY_COMPACT_FINISHED
    assert events[-1].payload["levels"] == []
    assert events[-1].payload["total_superseded"] == 0


@pytest.mark.asyncio
async def test_started_emitted_live_via_bus(monkeypatch):
    state, ctx, bus = await setup(monkeypatch, target=0.01)
    await capsule(state, ctx, "old", 0)

    async def summary(*args, **kwargs):
        assert bus.events[0].type == EventType.MEMORY_COMPACT_STARTED
        return "digest"

    cm.patch_summary(monkeypatch, summary)
    events = await cm.escalating_compact(state, ctx, token_estimate=90_000)
    assert not any(e.type == EventType.MEMORY_COMPACT_STARTED for e in events)
    assert events[-1].type == EventType.MEMORY_COMPACT_FINISHED


@pytest.mark.asyncio
async def test_force_preserves_order_even_below_target(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0)

    async def summary(*args, **kwargs):
        return "digest"

    cm.patch_summary(monkeypatch, summary)
    events = await cm.escalating_compact(state, ctx, token_estimate=1, force=True)
    assert [e.payload["source"] for e in events if e.type == EventType.MEMORY_COMPACTED] == [
        "demote_lean",
        "root_experience",
        "collapse",
    ]
