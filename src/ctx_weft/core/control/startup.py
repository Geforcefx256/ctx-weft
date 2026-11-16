"""Persist crash interruption facts without materializing runtime owners."""

from __future__ import annotations

import logging
from time import perf_counter

from ctx_weft.core.control.session_queries import session_state_summary
from ctx_weft.core.control.startup_view import (
    apply_startup_events,
    rebuild_startup_view,
)
from ctx_weft.core.models.discriminators import CancelReason, TaskErrorCode
from ctx_weft.core.orchestrator.lifecycle.agent_state import AgentInput, next_agent_transition
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.protocols.hitl import HITL_OUTCOME_CANCELLED, NoResumeDelivery
from ctx_weft.protocols.runtime import (
    SessionInitializationResult,
)

logger = logging.getLogger(__name__)


async def calibrate_session(store, sid: str, *, reason: str = "crash_recovery", publish=None) -> SessionInitializationResult:
    """Calibrate only the addressed session at its durable cut; never enumerate."""
    started = perf_counter()
    head = await store.committed_head(sid)
    view = await rebuild_startup_view(store, sid, through_position=head)
    session = view.sessions.get(sid)
    if session is None:
        from ctx_weft.core.models.errors import SessionNotFound
        raise SessionNotFound(sid)
    events = []
    # Run lifetimes are independent of task status (including background steps).
    for run_id, run in view.open_runs.items():
        sequence = run["sequence"]
        for kind, payload in (
            (EventType.RUN_INTERRUPTED, {"reason": reason}),
            (EventType.RUN_FINISHED, {"outcome": "interrupted"}),
        ):
            sequence += 1
            events.append(new_event(
                kind, session_id=sid, tenant_id=session.tenant_id,
                origin=EventOrigin.RUNTIME, run_id=run_id,
                sequence=sequence,
                task_id=run["task_id"], agent_id=run["agent_id"], payload=payload))
    affected_agents = set()
    settled_agents = set()
    if session.threshold_tripped:
        for req in view.hitl.pending.values():
            events.append(new_event(
                EventType.HITL_RESOLVED, session_id=sid,
                tenant_id=session.tenant_id, origin=EventOrigin.RUNTIME,
                task_id=req.task_id, agent_id=req.agent_id,
                payload={"hitl_id": req.id, "outcome": HITL_OUTCOME_CANCELLED,
                         "message": ""}))
    waiting = {r.task_id: r for r in view.hitl.pending.values()
               if not session.threshold_tripped and not isinstance(r.delivery, NoResumeDelivery)}
    for task in view.tasks.values():
        aid = task.assigned_agent_id or task.creator_agent_id or None
        terminal_kind = None
        if task.status not in {"FINISHED", "FAILED", "CANCELED"}:
            if task.id in session.threshold_root_task_ids:
                terminal_kind = EventType.TASK_FAILED
                payload = {"error_code": TaskErrorCode.BY_THRESHOLD,
                           "error_message": "Session failure threshold reached"}
            elif task.id in session.threshold_cancel_task_ids:
                terminal_kind = EventType.TASK_CANCELED
                payload = {"reason": CancelReason.FAILURE_THRESHOLD}
        if terminal_kind is not None:
            events.append(new_event(
                terminal_kind, session_id=sid, tenant_id=session.tenant_id,
                origin=EventOrigin.RUNTIME, task_id=task.id, agent_id=aid,
                payload=payload))
            if aid:
                affected_agents.add(aid)
                settled_agents.add(aid)
            continue
        # A crash can land after TaskCreated commits but before the first
        # TaskStarted. The new process has no queue/runner for PENDING
        # work either; expose explicit recovery rather than phantom RUNNING.
        # Preserve REVIEWING and SUSPENDED dependency/switch semantics.
        if task.status not in {"ACTIVE", "PENDING"}:
            continue
        if task.id in waiting:
            events.append(new_event(
                EventType.TASK_AWAITING_HUMAN, session_id=sid,
                tenant_id=session.tenant_id, origin=EventOrigin.RUNTIME,
                task_id=task.id, agent_id=aid,
                payload={"hitl_id": waiting[task.id].id},
            ))
            if aid:
                affected_agents.add(aid)
            continue
        events.append(new_event(
            EventType.TASK_INTERRUPTED, session_id=sid,
            tenant_id=session.tenant_id, origin=EventOrigin.RUNTIME,
            task_id=task.id, agent_id=aid,
            payload={"reason": reason, "error_message": "Process interrupted"},
        ))
        if aid:
            affected_agents.add(aid)
    for aid, agent in view.agents.items():
        if agent.status == "terminated":
            continue
        current_task = view.tasks.get(agent.current_task_id)
        if (current_task is not None and current_task.status in {"FINISHED", "FAILED", "CANCELED"}
                and agent.status in {"running", "interrupted"}):
            settled_agents.add(aid)
            affected_agents.add(aid)
        if session.threshold_tripped and agent.current_task_id in (
                session.threshold_root_task_ids + session.threshold_cancel_task_ids):
            # TaskFailed may commit just before ALM's AgentIdle. A snapshot
            # can already have pruned that terminal task, but retains agents.
            settled_agents.add(aid)
            affected_agents.add(aid)
        req = waiting.get(agent.current_task_id)
        if req is not None and req.agent_id and req.agent_id != aid:
            req = None
        # A new process has no executor for parked dependency/review work or a
        # human wait whose request is gone. Preserve the task state for resume,
        # but mark its agent interrupted even if it was already idle at the crash.
        needs_recovery = (
            current_task is not None
            and current_task.status in {"SUSPENDED", "REVIEWING", "AWAITING_HUMAN", "INTERRUPTED"}
            and req is None
        )
        if agent.status == "running" or aid in affected_agents or needs_recovery:
            transition = next_agent_transition(
                agent.status,
                (AgentInput.SETTLED if aid in settled_agents else
                 AgentInput.AWAITING_HUMAN if req else AgentInput.INTERRUPTED),
                task_id=agent.current_task_id, hitl_id=req.id if req else "",
                reason=reason)
            if transition is None:
                continue
            events.append(new_event(
                transition.event_type, session_id=sid,
                tenant_id=session.tenant_id, origin=EventOrigin.RUNTIME,
                agent_id=aid, task_id=agent.current_task_id,
                payload=transition.payload,
            ))
    stored = []
    committed = head
    if events:
        # No bus fanout: startup must not populate runtime registries or snapshots.
        if publish is None:
            receipt = await store.append_batch(sid, f"startup:{sid}:{head}", events)
            stored = list(receipt.records)
        else:
            await publish(events)
            ids = {event.id for event in events}
            stored = [row for row in await store.read_range(sid, after_position=head)
                      if row.event.id in ids]
        committed = stored[-1].position
        apply_startup_events([row.event for row in stored], view)
    result = SessionInitializationResult(
        session_id=sid, before_position=head, committed_position=committed,
        state=await session_state_summary(store, view, session, committed),
        startup_events=stored,
    )
    logger.debug("Session initialized: session=%s duration_ms=%.1f", sid,
                 (perf_counter() - started) * 1000)
    return result
