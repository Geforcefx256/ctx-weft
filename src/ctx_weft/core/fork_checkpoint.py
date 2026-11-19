"""Observe-owned, immutable AGENT-memory checkpoints for explicitly anchored replies.

Preparation precedes actor handoff; only the committed ObserveApplied publishes
the checkpoint. No query materializes or repairs source execution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime

from ctx_weft.core.utils.event import new_event
from ctx_weft.core.utils.ids import record_id
from ctx_weft.protocols import MemoryAddress, MemoryScope
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.protocols.runtime import MemorySeedConflictError


def validate_reply_anchor(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"response_event_id", "response_memory_id"}
        or any(not isinstance(v, str) or not v for v in value.values())
    ):
        raise ValueError("reply_anchor must identify one persisted response and assistant memory")
    return dict(value)


def fork_options(state, ctx):
    options = getattr(getattr(ctx, "execution", None), "options", {})
    if options.get("reply_anchor"):
        return options
    return state.extra.get("observe_fork_options", {})


def _group(record):
    return record.metadata.get("task_memory", {}).get("group_id")


async def prepare_checkpoint(state, ctx):
    options = fork_options(state, ctx)
    if not options.get("reply_anchor") or state.task.status == "CANCELED":
        return None
    if state.extra.get("observe_decision", {}).get("verdict"):
        return None
    completion = state.extra.get("observe_completion") or options.get("completion")
    if not completion:
        return None
    saved = options.get("fork_checkpoint") or state.extra.get("fork_checkpoint")
    if saved is not None:
        return deepcopy(saved)
    pending = state.extra.get("fork_checkpoint_prepared_event")
    if pending is None:
        anchor = validate_reply_anchor(options["reply_anchor"])
        memory = ctx.memory
        ids = options.get("fork_base_record_ids")
        if ids is None:
            raise RuntimeError("Anchored Observe has no frozen AGENT inputs")
        base = await memory.read_records(ids, ctx.provider_ctx)
        if set(ids) != {r.id for r in base} or any(
            r.scope != MemoryScope.AGENT or r.address.agent_id != state.agent.id for r in base
        ):
            raise MemorySeedConflictError(
                "Fork checkpoint AGENT inputs are missing or have wrong ownership"
            )
        address = MemoryAddress(session_id=state.session.id, agent_id=state.agent.id)
        current_ids = await memory.load_view_ids(address, MemoryScope.AGENT, ctx.provider_ctx)
        current = await memory.read_records(current_ids, ctx.provider_ctx)
        observe_id = state.extra["observe_id"]
        # Only this observation's writes may augment its frozen input. Later user
        # turns can already be running, and must never enter this reply's fork.
        own = [
            r
            for r in current
            if (
                r.metadata.get("finish_policy", {}).get("observe_id") == observe_id
                or r.metadata.get("observe_id") == observe_id
            )
        ]
        groups = {_group(r) for r in own if _group(r)}
        additions = [r for r in current if r in own or _group(r) in groups]
        selected = {r.id: r for r in base if _group(r) not in groups or not _group(r)}
        selected.update({r.id: r for r in additions})
        ordered = sorted(
            selected.values(), key=lambda r: (r.timestamp, r.metadata.get("seq_no", 0), r.id)
        )
        checkpoint = {
            "version": 1,
            "observe_id": observe_id,
            "reply_anchor": anchor,
            "agent_id": state.agent.id,
            "task_id": state.task.id,
            "agent_memory_ids": [r.id for r in ordered],
            "segment_end_at": options["segment_end_at"],
        }
        pending = new_event(
            EventType.REPLY_FORK_CHECKPOINT_PREPARED,
            session_id=state.session.id,
            tenant_id=ctx.provider_ctx.tenant_id,
            task_id=state.task.id,
            agent_id=state.agent.id,
            origin=EventOrigin.LOOP_OBSERVE,
            timestamp=datetime.fromisoformat(options["segment_end_at"]),
            causation_id=observe_id,
            payload={
                "task_id": state.task.id,
                "observe_id": observe_id,
                "fork_checkpoint": checkpoint,
            },
        )
        pending.id = record_id(observe_id, "fork_checkpoint_prepared")
        state.extra["fork_checkpoint_prepared_event"] = pending
    await ctx.event_bus.emit_batch((pending,))
    draft = deepcopy(pending.payload["fork_checkpoint"])
    state.extra["fork_checkpoint"] = draft
    return deepcopy(draft)


def stabilize_applied(event, checkpoint):
    """Identical receipt identity for foreground, background and cold recovery."""
    return replace(
        event,
        id=record_id(checkpoint["observe_id"], "reply_fork_applied"),
        timestamp=datetime.fromisoformat(checkpoint["segment_end_at"]),
        origin=EventOrigin.LOOP_OBSERVE,
        run_id=None,
        sequence=0,
        causation_id=checkpoint["observe_id"],
    )


def applied_event(state, ctx, checkpoint, *, reason=""):
    from ctx_weft.core.loop.driver import make_event

    payload = {
        "task_id": state.task.id,
        "observe_id": state.extra["observe_id"],
        **({"reason": reason} if reason else {}),
    }
    if checkpoint is None:
        return make_event(state, EventType.OBSERVE_APPLIED, payload)
    options = fork_options(state, ctx)
    payload["fork_checkpoint"] = deepcopy(checkpoint)
    event = new_event(
        EventType.OBSERVE_APPLIED,
        session_id=state.session.id,
        tenant_id=ctx.provider_ctx.tenant_id,
        task_id=state.task.id,
        agent_id=state.agent.id,
        origin=EventOrigin.LOOP_OBSERVE,
        timestamp=datetime.fromisoformat(options["segment_end_at"]),
        payload=payload,
    )
    return stabilize_applied(event, checkpoint)
