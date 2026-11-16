"""Completed history is compacted before the live task, including Act entry."""

import asyncio
from contextlib import aclosing
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS

import pytest

from ctx_weft.core.assembler.builder import compile_history
from ctx_weft.core.loop.driver import StepDriver
from tests import compaction_testkit as cm
from ctx_weft.core.loop.steps.act import ActStep
from ctx_weft.core.loop.steps.compact import CompactStep
from ctx_weft.core.loop.steps.finalize import _ensure_dispatch_frame
from ctx_weft.core.loop.steps.prepare import PrepareStep
from ctx_weft.core.loop.task_memory_fold import build_finish_slots, replace_finish_report
from ctx_weft.protocols import (
    LLMChunk, LLMUsage, MemoryAddress, MemoryEvent, MemoryKind, MemoryScope,
    ToolCall, load_records,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.template import LoopConfig
from tests.assembler_testkit import compiled_test_inputs, recalled_blocks, scripted_inputs
from tests.unit.test_compact_step_transitions import ScriptedLLM
from tests.unit.test_llm_event_convergence import _prepare_compact_transition

BASE = datetime(2026, 1, 1, tzinfo=UTC)


async def capsule(state, ctx, tid, number, *, parent=None, summary=True, dispatch=None):
    address = MemoryAddress(session_id=state.scope.session_id,
                            agent_id=state.scope.agent_id, task_id=tid)
    half = MemoryAddress(session_id=address.session_id, agent_id=address.agent_id)
    ts = BASE + timedelta(seconds=number * 10)
    await ctx.memory.ingest(MemoryEvent(
        id=f"{tid}-body", kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=address, role="user", content=f"DETAILS {tid} " + "x" * 3000,
        timestamp=ts), ctx.provider_ctx)
    slots = build_finish_slots(
        scope=half, task_id=tid, parent_task_id=parent, title=tid, outcome="success",
        act_recap=f"RECAP {tid}", task_summary=f"RESULT {tid}", final_reply=f"ANSWER {tid}",
        base=ts + timedelta(seconds=5), tool_call_id=f"close-{tid}")
    for index, event in enumerate(slots):
        event.id = f"{tid}-close-{index}"
        event.metadata["finish_policy"] = {
            "has_summary": summary, "capsule_state": "full", "slot_count": len(slots),
            "preserve_raw": True, "outcome": "success",
        }
        await ctx.memory.ingest(event, ctx.provider_ctx)
    if dispatch:
        # Exact historical metadata: child-parent fact must not become an owner self-edge.
        await ctx.memory.ingest(MemoryEvent(
            id=f"{tid}-start", kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
            address=half, role="user", content="start", timestamp=ts,
            metadata={"origin_task_id": tid, "parent_task_id": tid,
                      "child_task_id": dispatch,
                      "task_memory": {"version": 1, "kind": "start", "group_id": tid}},
        ), ctx.provider_ctx)


async def setup(monkeypatch, *, target=0.58, llm=None):
    state, ctx, bus, _ = await _prepare_compact_transition(monkeypatch, llm=llm)
    state.agent.loop_config = LoopConfig(
        compact_keep_last=6, collapse_keep_last=1, compact_target_ratio=target,
        predispatch_compact_token_ratio=0.5, max_turns_per_act=2)
    state.task.parent_task_id = None
    state.task.title = "current"
    state.task.user_prompt = None
    state.task.user_prompt_in_memory = True
    state.task.switch_requested = False
    state.task.actor_done = False
    state.task.suspend_requested = False
    state.task.unattended = True

    async def estimate(*args):
        return 0, False

    monkeypatch.setattr(PrepareStep, "_estimate_tokens", estimate)
    return state, ctx, bus


@pytest.mark.asyncio
async def test_writer_separates_child_parent_from_origin_parent(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    child = NS(id="child", parent_task_id="old", title="child", description="",
               started_at=BASE, created_at=BASE, origin_tool_call_id="dispatch")
    await _ensure_dispatch_frame(ctx.memory, state.scope, child, ctx.provider_ctx)
    records = await load_records(ctx.memory, cm._agent_half(state.scope),
                                 MemoryScope.AGENT, ctx.provider_ctx)
    assert records[0].metadata["origin_task_id"] == "old"
    assert records[0].metadata["child_task_id"] == "child"
    assert "parent_task_id" not in records[0].metadata
    assert 'parent_task_id="old"' in records[0].content


@pytest.mark.asyncio
async def test_legacy_starts_do_not_hide_roots_or_detach_nested_children(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0, dispatch="child")
    await capsule(state, ctx, "child", 1, parent="old", dispatch="grandchild")
    await capsule(state, ctx, "grandchild", 2, parent="child")
    await capsule(state, ctx, "newer", 3)
    assert await cm._count_root_residues(state, ctx) == 2
    assert await cm._kept_origin_ids(state, ctx, 6) == ["old", "newer"]
    assert await cm.fold_root_experience(state, ctx, 1, "old subtree digest") > 0
    body = await load_records(ctx.memory, cm._agent_half(state.scope),
                              MemoryScope.TASK, ctx.provider_ctx)
    assert {r.id for r in body if r.id.endswith("-body")} == {"newer-body"}


@pytest.mark.asyncio
async def test_closed_cross_agent_result_does_not_block_history_demotion(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0, dispatch="remote")
    await ctx.memory.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
        address=cm._agent_half(state.scope), role="user", content="remote delivered",
        timestamp=BASE + timedelta(seconds=2),
        metadata={"origin_task_id": "old", "child_task_id": "remote", "dispatch_state": "closed",
                  "task_memory": {"version": 1, "kind": "result", "group_id": "old"}},
    ), ctx.provider_ctx)
    assert await cm.demote_kept_capsules(state, ctx, {"old"}) > 0
    assert not any(r.id == "old-body" for r in await load_records(
        ctx.memory, cm._agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx))


@pytest.mark.asyncio
async def test_running_descendant_is_protected_from_whole_history_fold(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0, dispatch="running-child")
    await capsule(state, ctx, "newer", 1)
    assert await cm._kept_origin_ids(state, ctx, 6) == ["newer"]
    await cm.fold_root_experience(state, ctx, 0, "completed history only")
    body = await load_records(ctx.memory, cm._agent_half(state.scope),
                              MemoryScope.TASK, ctx.provider_ctx)
    assert any(r.id == "old-body" for r in body)
    assert not any(r.id == "newer-body" for r in body)


@pytest.mark.asyncio
async def test_unclosed_history_is_not_demoted_as_a_completed_task(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await ctx.memory.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
        address=cm._agent_half(state.scope), role="user", content="unfinished request",
        timestamp=BASE, metadata={"origin_task_id": "unfinished", "parent_task_id": None}),
        ctx.provider_ctx)
    assert await cm._kept_origin_ids(state, ctx, None) == []
    assert await cm.demote_kept_capsules(state, ctx, {"unfinished"}) == 0


@pytest.mark.asyncio
async def test_l2_selection_excludes_rich_history_and_current_task(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "rich", 0)
    await capsule(state, ctx, "lean", 1)
    assert await cm.demote_kept_capsules(state, ctx, {"lean"}) > 0
    selected = []

    async def summarize(plan):
        rows = await ctx.memory.read_records(list(plan.summary_ids), ctx.provider_ctx)
        selected.extend(rows)
        return "lean history digest"

    assert await cm.fold_root_experience(state, ctx, 0, summarize, lean_only=True) > 0
    assert len(selected) == 1 and cm.capsule_origin(selected[0]) == "lean"
    assert selected[0].metadata["finish_policy"]["capsule_state"] == "lean"
    body = await load_records(ctx.memory, cm._agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx)
    assert any(r.id == "rich-body" for r in body)
    assert any(r.address.task_id == state.scope.task_id for r in body)


@pytest.mark.asyncio
@pytest.mark.parametrize("current_is_parent", [False, True])
async def test_completed_branches_of_live_task_tree_are_compacted_first(monkeypatch, current_is_parent):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    parent = state.scope.task_id if current_is_parent else "ancestor"
    await capsule(state, ctx, parent, 0,
                  dispatch="running" if current_is_parent else state.scope.task_id)
    for index in range(8):
        await capsule(state, ctx, f"old{index}", index + 1, parent=parent)
    if not current_is_parent:
        await capsule(state, ctx, state.scope.task_id, 9, parent=parent)
    scopes = []

    async def summarize(current, local_ctx, *, scope="task", record_ids=()):
        scopes.append(scope)
        if scope == "agent":
            rows = await ctx.memory.read_records(list(record_ids), ctx.provider_ctx)
            assert all(r.scope == MemoryScope.AGENT for r in rows)
            assert not any(cm.capsule_origin(r) == parent for r in rows)
            assert all(r.kind == MemoryKind.SUMMARY or
                       r.metadata["finish_policy"]["capsule_state"] == "lean" for r in rows)
        else:
            body = await load_records(ctx.memory, cm._agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx)
            assert not any(r.id.startswith("old") for r in body)
        return "digest"

    cm.patch_summary(monkeypatch, summarize)
    events = await cm.escalating_compact(state, ctx, token_estimate=90_000)
    assert scopes == ["agent", "agent", "task"]
    assert [e.payload["source"] for e in events if e.type == EventType.MEMORY_COMPACTED] == [
        "demote_lean", "root_experience", "collapse"]
    turns = await load_records(ctx.memory, cm._agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx)
    assert any(cm.capsule_origin(r) == parent for r in turns)


@pytest.mark.asyncio
@pytest.mark.parametrize("history_kind", ["lean", "summary"])
@pytest.mark.parametrize("old_count", [1, 6])
async def test_condensed_history_is_summarized_before_current_task(monkeypatch, history_kind, old_count):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    for index in range(old_count):
        tid = f"old{index}"
        if history_kind == "lean":
            await capsule(state, ctx, tid, index)
            assert await cm.demote_kept_capsules(state, ctx, {tid}) > 0
        else:
            await ctx.memory.ingest(MemoryEvent(
                id=tid, kind=MemoryKind.SUMMARY, scope=MemoryScope.AGENT,
                address=cm._agent_half(state.scope), role="user", content=f"OLD DIGEST {tid}",
                timestamp=BASE + timedelta(seconds=index)), ctx.provider_ctx)
    scopes = []

    async def summarize(current, local_ctx, *, scope="task", record_ids=()):
        scopes.append(scope)
        if scope == "agent":
            rows = await ctx.memory.read_records(list(record_ids), ctx.provider_ctx)
            assert len(rows) == old_count
            assert all(r.scope == MemoryScope.AGENT for r in rows)
        else:
            turns = await load_records(ctx.memory, cm._agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx)
            assert [r.content for r in turns] == ["combined history digest"]
        return "combined history digest" if scope == "agent" else "current digest"

    cm.patch_summary(monkeypatch, summarize)
    events = await cm.escalating_compact(state, ctx, token_estimate=90_000)
    assert scopes == ["agent", "task"]
    assert [e.payload["source"] for e in events if e.type == EventType.MEMORY_COMPACTED] == [
        "root_experience", "collapse"]


@pytest.mark.asyncio
async def test_l2_stops_before_current_task_when_history_digest_meets_target(monkeypatch):
    state, ctx, _ = await setup(monkeypatch, target=0.58)
    await capsule(state, ctx, "old", 0)
    assert await cm.demote_kept_capsules(state, ctx, {"old"}) > 0
    scopes = []

    async def summarize(current, local_ctx, *, scope="task"):
        scopes.append(scope)
        return "digest"

    cm.patch_summary(monkeypatch, summarize)
    events = await cm.escalating_compact(state, ctx, token_estimate=58_001)
    assert scopes == ["agent"]
    assert [e.payload["source"] for e in events if e.type == EventType.MEMORY_COMPACTED] == ["root_experience"]
    current = await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
    assert not any(r.metadata.get("collapsed") for r in current)


@pytest.mark.asyncio
async def test_late_lean_replacement_during_l2_blocks_current_compaction(monkeypatch):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    await capsule(state, ctx, "old", 0)
    scopes = []

    async def summarize(current, local_ctx, *, scope="task"):
        scopes.append(scope)
        assert scope == "agent"
        registry = StepDriver.for_context(ctx).executions
        async with registry.task_guard("old", "finish_report"):
            await replace_finish_report(
                ctx.memory, ctx.provider_ctx, cm._agent_half(state.scope), "old", "close-old",
                "LATE RECAP", "LATE RESULT", "success", "old", operation_id="late-l2")
        return "snapshot digest"

    cm.patch_summary(monkeypatch, summarize)
    await asyncio.wait_for(cm.escalating_compact(state, ctx, token_estimate=90_000), timeout=2)
    assert scopes == ["agent"]
    assert not await cm._old_history_remaining(state, ctx)
    assert "_compact_deferred_history" not in state.extra
    turns = await load_records(ctx.memory, cm._agent_half(state.scope), MemoryScope.AGENT, ctx.provider_ctx)
    assert any("LATE RESULT" in str(r.content) for r in turns)
    current = await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
    assert not any(r.metadata.get("collapsed") for r in current)


@pytest.mark.asyncio
@pytest.mark.parametrize("collapse_current", [False, True])
async def test_act_compact_removes_old_history_before_current_window(monkeypatch, collapse_current):
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS

    dispatch = ToolCall(id="dispatch", name=next(iter(DISPATCH_TOOLS)), arguments={})
    responses = [
        [LLMChunk(kind="tool_call", tool_call=dispatch),
         LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=60_000, completion_tokens=1))],
    ]
    if collapse_current:
        responses.extend([
            [LLMChunk(kind="token", text="OLDER HISTORY DIGEST")],
            [LLMChunk(kind="token", text="OLD HISTORY DIGEST")],
            [LLMChunk(kind="token", text="CURRENT TASK DIGEST")],
        ])
    responses.append([LLMChunk(kind="token", text="done"),
                      LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=20, completion_tokens=1))])
    llm = ScriptedLLM(responses)
    state, ctx, bus = await setup(monkeypatch, llm=llm, target=0.01 if collapse_current else 0.58)
    for i in range(10):
        await capsule(state, ctx, f"old{i}", i * 2, dispatch=f"child{i}")
        await capsule(state, ctx, f"child{i}", i * 2 + 1, parent=f"old{i}")
    assert await cm._count_root_residues(state, ctx) == 10
    compact_inputs = []

    async def assemble(request):
        blocks = await recalled_blocks(request, ctx)
        records = [b.value for b in blocks]
        if request.purpose == "compact":
            compact_inputs.append((request.intent.scope, records))
            if request.intent.scope == "task":
                assert not any(r.id.startswith(("old", "child")) and r.scope == MemoryScope.TASK
                               for r in records), "current task must wait for old details to disappear"
            else:
                assert all(r.scope == MemoryScope.AGENT for r in records)
                assert all(r.kind == MemoryKind.SUMMARY or
                           r.metadata["finish_policy"]["capsule_state"] == "lean" for r in records)
        prompt = compile_history(records, token_counter=ctx.llm.tokenizer.count,
                                 task_id=state.task.id)
        return scripted_inputs(prompt, purpose=request.purpose)

    ctx.assembler = NS(assemble=assemble, compile=compiled_test_inputs)

    invoked = []

    async def tools(current, local_ctx, calls):
        assert [tc.name for tc in calls] == [dispatch.name]
        assert calls == current.pending_act_turn.record.tool_calls
        invoked.extend(calls)
        return []

    monkeypatch.setattr("ctx_weft.core.loop.steps.act._execute_tool_calls", tools)
    driver = StepDriver({"prepare": PrepareStep(), "act": ActStep(), "compact": CompactStep()})
    transitions = []
    async with aclosing(driver.run(state, ctx)) as chain:
        async for outcome in chain:
            transitions.append(outcome.next_step)
            if outcome.next_step == "observe":
                break
    assert transitions == ["act", "compact", "prepare", "act", "observe"]
    assert len(invoked) == 1
    sequences = [event.sequence for event in bus.events]
    assert sequences == sorted(set(sequences))
    events = [e for e in bus.events if e.type == EventType.MEMORY_COMPACTED]
    levels = [e.payload["source"] for e in events]
    assert levels == (["demote_lean", "root_experience", "collapse"] if collapse_current
                      else ["demote_lean"])
    assert all(e.payload["trigger"] == "pre_dispatch" for e in events)
    assert [scope for scope, _ in compact_inputs] == (["agent", "agent", "task"] if collapse_current else [])
    if collapse_current:
        assert {cm.capsule_origin(r) for r in compact_inputs[0][1]} == {f"old{i}" for i in range(4)}
    current = await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
    assert any(r.metadata.get("collapsed") for r in current) is collapse_current
    fresh = await recalled_blocks(NS(scope=state.scope, extra={}), ctx)
    fresh_ids = {b.value.id for b in fresh}
    assert "old0-body" not in fresh_ids and "child0-body" not in fresh_ids
    sent = "\n".join(str(m.content) for m in llm.requests[-1].messages)
    assert "DETAILS old0" not in sent
    assert ("OLD HISTORY DIGEST" in sent) is collapse_current
    assert ("history 1" in sent) is not collapse_current


@pytest.mark.asyncio
async def test_missing_summary_is_regenerated_before_current_task(monkeypatch):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    await capsule(state, ctx, "old", 0, summary=False)
    scopes = []

    async def summarize(current, local_ctx, *, scope="task"):
        scopes.append(scope)
        if scope == "task":
            assert not any(r.id == "old-body" for r in await load_records(
                ctx.memory, cm._agent_half(state.scope), MemoryScope.TASK, ctx.provider_ctx))
        return "fresh digest"

    cm.patch_summary(monkeypatch, summarize)
    events = await cm.escalating_compact(state, ctx, token_estimate=90_000)
    # L1 now only reuses existing summaries. Missing reports are summarized in
    # L2 once, before any current-task summary, rather than repaired by an L1 LLM.
    assert scopes == ["agent", "task"]
    assert [e.payload["source"] for e in events if e.type == EventType.MEMORY_COMPACTED] == [
        "root_experience", "collapse"]


@pytest.mark.asyncio
async def test_late_observer_replacement_keeps_one_digest_and_folds_only_selected_ids(monkeypatch):
    state, ctx, _ = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0)
    await capsule(state, ctx, "newer", 1)
    calls = []

    async def summarize(plan):
        selected = plan.summary_ids
        rows = await ctx.memory.read_records(list(selected), ctx.provider_ctx)
        calls.append(rows)
        async def observer_write():
            registry = StepDriver.for_context(ctx).executions
            async with registry.task_guard("old", "finish_report"):
                await replace_finish_report(
                    ctx.memory, ctx.provider_ctx, cm._agent_half(state.scope), "old", "close-old",
                    "LATE RECAP", "LATE RESULT", "success", "old", operation_id="late")

        # The model await must not own the observer's write lock.
        await asyncio.wait_for(asyncio.create_task(observer_write()), timeout=2)
        return "SNAPSHOT DIGEST"

    assert await cm.fold_root_experience(state, ctx, 1, summarize) > 0
    assert len(calls) == 1
    turns = await load_records(ctx.memory, cm._agent_half(state.scope),
                               MemoryScope.AGENT, ctx.provider_ctx)
    body = await load_records(ctx.memory, cm._agent_half(state.scope),
                              MemoryScope.TASK, ctx.provider_ctx)
    assert not {r.id for r in calls[0]} & {r.id for r in [*turns, *body]}
    assert any("LATE RECAP" in str(r.content) for r in turns)
    assert any("LATE RESULT" in str(r.content) for r in turns)
    assert [r.content for r in turns if r.kind == MemoryKind.SUMMARY] == ["SNAPSHOT DIGEST"]
    assert any(r.id == "newer-body" for r in body)


@pytest.mark.asyncio
@pytest.mark.parametrize("old_count", [1, 8])
async def test_new_history_during_digest_is_retained_and_blocks_current_compaction(monkeypatch, old_count):
    state, ctx, _ = await setup(monkeypatch, target=0.01)
    await capsule(state, ctx, "old", 0, summary=False)
    for index in range(1, old_count):
        await capsule(state, ctx, f"extra{index}", index)
    calls = []

    async def summarize(current, local_ctx, *, scope="task"):
        assert scope == "agent"
        calls.append(scope)
        await ctx.memory.ingest(MemoryEvent(
            kind=MemoryKind.SUMMARY, scope=MemoryScope.TASK,
            address=MemoryAddress(session_id=state.scope.session_id,
                                  agent_id=state.scope.agent_id, task_id="old"),
            role="assistant", content="new evidence", timestamp=BASE + timedelta(seconds=len(calls))),
            ctx.provider_ctx)
        return "snapshot digest"

    cm.patch_summary(monkeypatch, summarize)
    await cm.escalating_compact(state, ctx, token_estimate=90_000)
    assert len(calls) == 1
    current = await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
    assert not any(r.metadata.get("collapsed") for r in current)
    old = await load_records(ctx.memory, MemoryAddress(
        session_id=state.scope.session_id, agent_id=state.scope.agent_id, task_id="old"),
        MemoryScope.TASK, ctx.provider_ctx)
    assert [r.content for r in old] == ["new evidence"]
    turns = await load_records(ctx.memory, cm._agent_half(state.scope),
                               MemoryScope.AGENT, ctx.provider_ctx)
    summary = next(r for r in turns if "old" in r.metadata.get("residual_task_ids", []))
    # Missing-summary work now belongs to fold_history (L2), which emits a
    # rolling summary while preserving ownership of late residual evidence.
    assert summary.kind == MemoryKind.SUMMARY
    assert "snapshot digest" in summary.content
    assert await cm._old_history_remaining(state, ctx)
    assert "old" in await cm._kept_origin_ids(state, ctx, old_count + 1)

    async def summarize_later(current, local_ctx, *, scope="task"):
        assert scope == "agent"
        return "later evidence digest"

    # A subsequent snapshot can consume the residual without needing its old closure.
    cm.patch_summary(monkeypatch, summarize_later)
    assert await cm.fold_root_experience(state, ctx, 0, lambda: summarize_later(state, ctx, scope="agent")) > 0
    assert not await cm._old_history_remaining(state, ctx)
