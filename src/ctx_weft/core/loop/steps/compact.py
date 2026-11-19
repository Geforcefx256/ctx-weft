"""Agent-owned compaction levels, budgets, LLM summaries and lifecycle events.

Memory providers own structural selection and atomic transformations. This step
never builds task capsules or computes a list of protected record IDs.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from contextlib import aclosing
from dataclasses import dataclass

from ctx_weft.core.assembler import CompactIntent, ContextRequest
from ctx_weft.core.assembler.builder import compile_history, history_token_costs
from ctx_weft.core.loop.driver import (
    COMPACT_NOOP_KEY,
    LoopContext,
    LoopState,
    Step,
    StepDriver,
    StepOutcome,
    assistant_memory_checkpoint,
    make_event,
)
from ctx_weft.core.loop.llm_gateway import (
    build_llm_request,
    llm_request_id,
    resolve_llm_identity,
    stream_llm_resilient,
)
from ctx_weft.core.loop.llm_stream import ensure_collected_stream
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.core.utils.estimate import estimate_assembled_tokens
from ctx_weft.protocols import LLMCallError, MemoryAddress, MemoryKind, MemoryScope
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.protocols.memory import MemoryOrganizeDeferred

logger = logging.getLogger(__name__)
_SUMMARY_MAX_ATTEMPTS = 3
_TASK_KINDS = [MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY, MemoryKind.TOOL_AUDIT]


@dataclass(frozen=True)
class _LevelResult:
    source: str
    layer: str
    superseded: int
    freed_tokens: int
    capsules: int = 0


@dataclass(frozen=True)
class _CompactResult:
    before: int
    after: int
    target: int
    levels: tuple = ()
    stop: str = "exhausted"

    @property
    def changed(self):
        return bool(self.levels)


async def _read_memory(state, ctx, known=None):
    """Refresh view membership while reusing immutable bodies already in hand."""
    address = MemoryAddress(
        session_id=ctx.provider_ctx.session_id,
        agent_id=ctx.provider_ctx.agent_id or state.scope.agent_id,
    )
    ids = []
    for scope in (MemoryScope.TASK, MemoryScope.AGENT):
        ids.extend(
            await ctx.memory.load_view_ids(address, scope, ctx.provider_ctx, kinds=_TASK_KINDS)
        )
    rows = {} if known is None else dict(known)
    missing = [rid for rid in ids if rid not in rows]
    if missing:
        rows.update((r.id, r) for r in await ctx.memory.read_records(missing, ctx.provider_ctx))
    return {rid: rows[rid] for rid in ids}


async def _prepare_summary_records(ctx, records):
    """Explicit image transformation; recall itself remains read-only."""
    if not records:
        return records
    result = await ctx.memory.organize(
        ctx.provider_ctx,
        operation="demote_images",
        record_ids=[r.id for r in records],
        keep_recent_images=0,
    )
    if result.status == "deferred":
        raise MemoryOrganizeDeferred("summary material changed during image demotion")
    removed = set(result.superseded_ids)
    by_id = {r.id: r for r in records if r.id not in removed}
    by_id.update((r.id, r) for r in result.replacements)
    return sorted(by_id.values(), key=lambda r: (r.timestamp, r.metadata.get("seq_no", 0)))


async def _run_compact(state, ctx, *, token_estimate, target_tokens, force=False):
    if token_estimate < target_tokens and not force:
        return _CompactResult(token_estimate, token_estimate, target_tokens, stop="target")
    async with assistant_memory_checkpoint(state, ctx):
        rows = await _read_memory(state, ctx)
    current = state.scope.task_id
    cost, costs = history_token_costs(
        rows.values(), token_counter=ctx.llm.tokenizer.count, task_id=current
    )
    baseline = compile_history(
        rows.values(), token_counter=ctx.llm.tokenizer.count, task_id=current
    ).token_count
    cost, estimate = baseline, token_estimate
    config = state.agent.loop_config
    levels = []

    def fits():
        return estimate < target_tokens and not force

    def measure(*, full=False):
        nonlocal cost, costs, estimate
        cost, costs = history_token_costs(
            rows.values(), token_counter=ctx.llm.tokenizer.count, task_id=current, known=costs
        )
        if full:
            cost = compile_history(
                rows.values(), token_counter=ctx.llm.tokenizer.count, task_id=current
            ).token_count
        estimate = token_estimate + cost - baseline

    def finished(deferred=False):
        return _CompactResult(
            token_estimate,
            estimate,
            target_tokens,
            tuple(levels),
            "history_changed" if deferred else "target" if fits() else "exhausted",
        )

    async def apply(operation, *, records=None, summary=None, **kwargs):
        nonlocal rows
        before = cost
        result = await ctx.memory.organize(
            ctx.provider_ctx,
            operation=operation,
            record_ids=None if records is None else [r.id for r in records],
            summary=summary,
            **kwargs,
        )
        removed = set(result.superseded_ids)
        rows = {rid: r for rid, r in rows.items() if rid not in removed}
        rows.update((r.id, r) for r in result.replacements)
        if result.status == "deferred":
            # A competing writer may have replaced rows outside this commit's
            # active subset. Report the fresh view's cost before yielding.
            rows = await _read_memory(state, ctx, rows)
        measure()
        return result, max(0, before - cost)

    async def checkpoint():
        nonlocal rows
        previous = set(rows)
        rows = await _read_memory(state, ctx, rows)
        measure(full=True)
        return previous != set(rows)

    if bool(getattr(getattr(ctx, "blob_store", None), "can_externalize", False)):
        result, freed = await apply(
            "demote_images", keep_recent_images=getattr(config, "compact_keep_recent_images", 2)
        )
        if result.changed:
            levels.append(_LevelResult("demote_images", "task", result.demoted_images, freed))
        if result.status == "deferred":
            return finished(True)
    if fits():
        return finished()

    count = freed = capsules = 0
    while True:
        result, saved = await apply("reduce_history")
        count += len(result.superseded_ids)
        freed += saved
        capsules += int(result.changed)
        if result.status == "deferred" or result.status == "exhausted" or fits():
            break
    if count:
        levels.append(_LevelResult("demote_lean", "agent", count, freed, capsules))
    changed = await checkpoint()
    if result.status == "deferred" or changed or fits():
        return finished(result.status == "deferred" or changed)

    count = freed = 0
    keep_values = (0,) if force else tuple(dict.fromkeys((config.compact_keep_last, 0)))
    for keep in keep_values:
        # Each fixed range is summarized once; replacements are not recursively
        # fed back to the same level during this run.
        level_before = cost
        try:
            records = await ctx.memory.recall_for_organize(
                ctx.provider_ctx, operation="fold_history", keep_last=keep
            )
            records = await _prepare_summary_records(ctx, records)
        except MemoryOrganizeDeferred:
            return finished(True)
        if records:
            # Preparing reversible image material may replace source IDs. Refresh
            # membership, but reuse the returned bodies rather than reading again.
            rows.update((r.id, r) for r in records)
            rows = await _read_memory(state, ctx, rows)
            measure()
            summary = await _summarize(state, ctx, scope="agent", records=records)
            result, saved = await apply("fold_history", records=records, summary=summary)
            count += len(result.superseded_ids)
            freed += max(0, level_before - cost)
            if result.status == "deferred":
                if count:
                    levels.append(_LevelResult("root_experience", "agent", count, freed))
                return finished(True)
        if fits():
            break
    if count:
        levels.append(_LevelResult("root_experience", "agent", count, freed))
    changed = await checkpoint()
    if changed or fits():
        return finished(changed)

    level_before = cost
    try:
        records = await ctx.memory.recall_for_organize(
            ctx.provider_ctx,
            operation="fold_current",
            keep_last=getattr(config, "collapse_keep_last", config.compact_keep_last),
        )
        records = await _prepare_summary_records(ctx, records)
    except MemoryOrganizeDeferred:
        return finished(True)
    if records:
        rows.update((r.id, r) for r in records)
        rows = await _read_memory(state, ctx, rows)
        measure()
        summary = await _summarize(state, ctx, scope="task", records=records)
        result, freed = await apply("fold_current", records=records, summary=summary)
        if result.changed:
            levels.append(
                _LevelResult(
                    "collapse", "task", len(result.superseded_ids), max(0, level_before - cost)
                )
            )
        if result.status == "deferred":
            return finished(True)
    measure(full=True)
    return finished()


async def _summarize(
    state: LoopState,
    ctx: LoopContext,
    *,
    scope: str = "task",
    records: list,
) -> str:
    """装配 purpose=compact 上下文并生成非空文本摘要，返回解析后的 digest。"""
    agent = state.agent
    extra = {"compact_scope": scope, "organize_records": records}
    request = ContextRequest(
        purpose="compact",
        scope=state.scope,
        task=state.task,
        agent=agent,
        session=state.session,
        resolved_model=state.resolved_model,
        template=state.extra.get("template"),
        bound_capabilities=state.extra.get("bound_capabilities", []),
        token_counter=ctx.llm.tokenizer.count,
        extra=extra,
        intent=CompactIntent(scope="agent" if scope == "agent" else "task"),
    )
    inputs = await ctx.assembler.assemble(request)
    llm_request = await build_llm_request(ctx, state, inputs)
    model, llm_account = resolve_llm_identity(state)

    bus = getattr(ctx, "event_bus", None)
    # 同 gateway 那 4 个流式事件一个门禁：bus 存在 且 origin==LOOP_COMPACT（见上方 docstring）。
    should_emit_finished = (
        bus is not None and getattr(state, "origin", None) == EventOrigin.LOOP_COMPACT
    )
    for attempt in range(1, _SUMMARY_MAX_ATTEMPTS + 1):
        req_id = llm_request_id(state) if should_emit_finished else None
        stream = ensure_collected_stream(
            stream_llm_resilient(ctx, state, llm_request, inputs=inputs)
        )
        try:
            async with aclosing(stream):
                async for _ in stream:
                    stream.accept_current()
        except asyncio.CancelledError:
            usage_payload = dataclasses.asdict(stream.partial_result().usage)
            if should_emit_finished and any(value is not None for value in usage_payload.values()):
                pause = getattr(ctx, "pause_token", None)
                tok = getattr(ctx, "cancel_token", None)
                paused = pause is not None and pause.is_paused and not (tok and tok.is_cancelled)
                await bus.emit_batch((make_event(
                    state, EventType.LLM_RESPONSE_FINISHED, payload={
                        "request_id": req_id, "content": "", "reasoning": "", "tool_calls": [],
                        "usage": usage_payload, "llm_model": model, "llm_account": llm_account,
                        "finish_reason": "paused" if paused else "cancelled", "turn": 0}),))
            raise
        result = stream.get_final_result()
        if should_emit_finished:
            await bus.emit(
                make_event(
                    state,
                    EventType.LLM_RESPONSE_FINISHED,
                    payload={
                        "request_id": req_id,
                        "content": result.text,
                        "reasoning": result.reasoning,
                        "tool_calls": [dataclasses.asdict(tc) for tc in result.tool_calls],
                        "usage": dataclasses.asdict(result.usage),
                        "llm_model": model,
                        "llm_account": llm_account,
                        "finish_reason": result.finish_reason or "stop",
                        "turn": 0,
                    },
                )
            )
        # Validate before creating any replacement MemoryEvent. Prose remains a
        # supported degraded digest; an empty or tool-only response cannot be one.
        if result.text.strip() and not result.tool_calls:
            return result.text
        if attempt < _SUMMARY_MAX_ATTEMPTS:
            logger.warning(
                "Invalid compaction summary; retrying (%d/%d)", attempt, _SUMMARY_MAX_ATTEMPTS
            )
            if should_emit_finished:
                await bus.emit(
                    make_event(
                        state,
                        EventType.LLM_RETRY_TRIGGERED,
                        payload={
                            "request_id": req_id,
                            "attempt": attempt,
                            "max_attempts": _SUMMARY_MAX_ATTEMPTS,
                            "next_delay_sec": 0,
                            "error_code": 0,
                            "error": "Empty or non-text compaction summary",
                        },
                    )
                )
    raise LLMCallError(
        "Invalid compaction summary after 3 attempts; original records retained", retriable=False
    )


async def _wait_for_observes(state, ctx, registry):
    from ctx_weft.core.loop.steps.act import _await_tool_or_stop, _interrupt_checkpoint

    await _interrupt_checkpoint(state, ctx)
    waiter = asyncio.create_task(registry.wait_agent_observes(state.agent.id))
    try:
        await _await_tool_or_stop(waiter, ctx)
        await _interrupt_checkpoint(state, ctx)
        await waiter
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


async def _estimate_after_observe(state, ctx):
    """Rebuild the complete actor input after observations have changed memory."""
    # Confirm pending assistant writes before the new budget reads memory. Keep
    # the checkpoint lock out of both Observe waiting and prompt assembly.
    async with assistant_memory_checkpoint(state, ctx):
        pass
    staged = getattr(getattr(ctx, "task_manager", None), "staged_memory", None)
    settings = state.task.settings
    request = ContextRequest(
        purpose=settings.purpose if isinstance(settings, NormalTaskSettings) else "act",
        scope=state.scope, task=state.task, agent=state.agent, session=state.session,
        resolved_model=state.resolved_model, template=state.extra.get("template"),
        bound_capabilities=state.extra.get("bound_capabilities", []),
        token_counter=ctx.llm.tokenizer.count,
        extra={"staged_memory": staged(state.task.id) if staged is not None else []},
    )
    inputs = await ctx.assembler.assemble(request)
    prompt = await ctx.assembler.compile(
        inputs, template=request.template, token_counter=request.token_counter,
        reuse=True, trim=False)
    return estimate_assembled_tokens(prompt, request.token_counter)


class CompactStep(Step):
    name = "compact"

    async def execute(self, state, ctx) -> StepOutcome:
        options = getattr(getattr(ctx, "execution", None), "options", {})
        after = options.get("after")
        if after not in (None, "prepare"):
            raise ValueError(f"Unsupported compact continuation: {after!r}")
        config, guard = state.agent.loop_config, state.agent.loop_guard
        estimate = options.get("token_estimate", guard.context_tokens)
        ratio = getattr(config, "compact_target_ratio", 0) or config.compact_token_ratio
        limit = state.resolved_model.effective_input_limit
        target = int(limit * ratio)
        force, trigger = options.get("force", False), options.get("trigger", "compact")
        events = []
        registry = StepDriver.for_context(ctx).executions
        pending = registry.pending_agent_observes(state.agent.id)
        started = bool(pending) or (limit > 0 and (force or estimate >= target))
        if started:
            started_event = make_event(
                state,
                EventType.MEMORY_COMPACT_STARTED,
                payload={
                    "task_id": state.task.id,
                    "agent_id": state.agent.id,
                    "trigger": trigger,
                    "token_estimate": estimate,
                    "target_tokens": target,
                    **({"phase": "waiting_observe", "pending_observe_count": len(pending)}
                       if pending else {}),
                },
            )
            bus = getattr(ctx, "event_bus", None)
            if bus is not None:
                await bus.emit(started_event)
            else:
                events.append(started_event)
        if pending:
            await _wait_for_observes(state, ctx, registry)
            estimate = await _estimate_after_observe(state, ctx)
        guard.context_tokens = guard.context_message_count = 0
        state.extra[COMPACT_NOOP_KEY] = True
        result = None
        if limit > 0 and (force or estimate >= target):
            result = await _run_compact(
                state, ctx, token_estimate=estimate, target_tokens=target, force=force
            )
        elif started:
            result = _CompactResult(estimate, estimate, target, stop="target")
        if result is not None:
            state.extra[COMPACT_NOOP_KEY] = not result.changed
            for level in result.levels:
                payload = dict(
                    superseded_count=level.superseded,
                    layer=level.layer,
                    source=level.source,
                    trigger=trigger,
                    freed_tokens=level.freed_tokens,
                )
                if level.source == "demote_images":
                    payload["demoted_images"] = level.superseded
                if level.source == "demote_lean":
                    payload["demoted_capsules"] = level.capsules
                events.append(make_event(state, EventType.MEMORY_COMPACTED, payload=payload))
            events.append(
                make_event(
                    state,
                    EventType.MEMORY_COMPACT_FINISHED,
                    payload={
                        "task_id": state.task.id,
                        "agent_id": state.agent.id,
                        "trigger": trigger,
                        "total_superseded": sum(x.superseded for x in result.levels),
                        "freed_tokens": sum(x.freed_tokens for x in result.levels),
                        "levels": [x.source for x in result.levels],
                        "est_before": result.before,
                        "est_after": result.after,
                        "target_tokens": result.target,
                    },
                )
            )
        return StepOutcome(
            next_step=after,
            next_step_options={"after_compact": True} if after == "prepare" else {},
            state_patch={"prompt_inputs": None},
            events=events,
        )
