"""Autonomous startup reads: the shared reducer, with only control-plane fields.

This view is never handed to execution/owner materialization or snapshot writers.
It deliberately carries no conversation, tool results, observation bodies or HITL
answers. Every event envelope is retained to advance unfinished run sequences.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace
from time import perf_counter

from ctx_weft.core.control.reducers import (
    _AGENT_STATUS_BY_EVENT,
    REPLAY_BATCH,
    REPLAY_EXCLUDE_TYPES,
    TASK_STATUS_BY_EVENT,
    _apply,
    _rebuild_agents,
    deserialize_view,
    fold_hitl_snapshot,
    restore_legacy_canceled_tasks,
    snapshot_facts_from_event,
    snapshot_is_usable,
)
from ctx_weft.core.control.types import RunStateView
from ctx_weft.protocols.events import Event, EventReadProjection, EventType, ProjectedEvent
from ctx_weft.providers.events.store.projection import project_value, projection_tree

logger = logging.getLogger(__name__)


def _fields(*keys):
    return tuple((key,) for key in keys)


_TASK_FIELDS = _fields("id", "status", "session_id", "assigned_agent_id", "creator_agent_id",
                       "parent_task_id", "dag_deps", "finalized", "error", "error_code",
                       "failure_retry_count", "blocked_by_task_id", "created_at", "finished_at",
                       "details_complete")
_TASK_FIELDS += (("settings", "use_subagent"), ("settings_raw", "use_subagent"))
_SESSION_FIELDS = _fields("id", "root_agent_id", "tenant_id", "goal", "failure_counter",
                          "threshold_tripped", "threshold_root_task_ids", "threshold_cancel_task_ids")
_AGENT_FIELDS = _fields("id", "status", "status_reason", "current_task_id", "template_id",
                        "parent_agent_id", "spawn_depth", "llm_account", "llm_model",
                        "llm_config", "created_at")
_HITL_FIELDS = _fields("id", "hitl_id", "form", "session_id", "task_id", "agent_id", "tenant_id",
                       "delivery", "tool_call_id", "stage", "outcome")

_PAYLOAD_FIELDS = {
    EventType.SESSION_CREATED: _fields("root_agent_id", "tenant_id"),
    EventType.RECOGNIZE_INTENT_TOOL_CALL: _fields("session_goal"),
    EventType.FAILURE_THRESHOLD_HIT: _fields("root_task_ids", "cancel_task_ids"),
    EventType.AGENT_INSTANTIATED: _fields("template_id", "llm_account", "llm_model", "llm_config",
                                         "parent_agent_id", "spawn_depth", "created_at"),
    EventType.AGENT_LLM_CHANGED: _fields("llm_account", "llm_model", "llm_config"),
    EventType.TASK_CREATED: tuple(("task", *path) for path in _TASK_FIELDS),
    EventType.TASK_REVIEWING: (
        *tuple(("successor", "task", *path) for path in _TASK_FIELDS),
        *_fields("replaced_task_ids", "rewired_dependencies"),
    ),
    **{kind: _fields("reason", "assigned_agent_id", "error_code", "error_message", "retry_count",
                    "blocked_by_task_id", "finished_at") for kind in TASK_STATUS_BY_EVENT
       if kind != EventType.TASK_REVIEWING},
    **{kind: _fields("reason") for kind in _AGENT_STATUS_BY_EVENT},
    EventType.HITL_OPENED: _HITL_FIELDS,
    EventType.HITL_REQUIRED: _HITL_FIELDS,
    EventType.HITL_RESOLVED: _fields("hitl_id", "outcome"),
    **{kind: _fields("hitl_id") for kind in (
        EventType.HITL_REPLY_RETRACTED, EventType.HITL_CLOSED, EventType.HITL_APPROVED,
        EventType.HITL_MODIFIED, EventType.HITL_ANSWERED, EventType.HITL_REJECTED,
        EventType.HITL_CANCELLED)},
}

_BLOB_PATHS = _fields("run_id", "session_id", "task_id", "agent_id",
                      "task_status", "events_total", "tasks_total", "open_runs")
_BLOB_PATHS += tuple(("sessions", "*", *path) for path in _SESSION_FIELDS)
_BLOB_PATHS += tuple(("tasks", "*", *path) for path in _TASK_FIELDS)
_BLOB_PATHS += tuple(("agents", "*", *path) for path in _AGENT_FIELDS)
_BLOB_PATHS += tuple(("hitl", group, "*", *path)
                     for group in ("pending", "resolved") for path in _HITL_FIELDS)
_PAYLOAD_FIELDS[EventType.STATE_SNAPSHOT] = (
    *_fields("cut_position", "projection_version", "chain_depth", "reason"),
    *tuple(("state_blob", *path) for path in _BLOB_PATHS),
)
STARTUP_PROJECTION = EventReadProjection(_PAYLOAD_FIELDS)
STARTUP_SNAPSHOT_VERSION = 3
STARTUP_SNAPSHOT_PROJECTION = EventReadProjection({EventType.STATE_SNAPSHOT: _fields(
    "cut_position", "projection_version", "chain_depth", "reason",
    "startup_state_blob", "startup_projection_version",
)})


def startup_snapshot_blob(blob: dict, current_task_statuses: dict[str, str] | None = None) -> dict:
    """Derive a compact sibling from the exact same snapshot cut and reducer."""
    compact = project_value(blob, projection_tree(_BLOB_PATHS))
    # Normal execution snapshots prune idle agents' terminal tasks. Startup only
    # needs their statuses; retain those tiny facts to avoid rereading task bodies.
    for task_id, status in (current_task_statuses or {}).items():
        compact.setdefault("tasks", {}).setdefault(task_id, {
            "id": task_id, "session_id": blob.get("session_id", ""), "status": status,
        })
    return compact


@dataclass
class StartupRecoveryView(RunStateView):
    """A body-free accumulator; intentionally not an execution recovery view."""

    # Ephemeral locators, never execution state or another HITL ledger. Entries
    # refer to an opening event or the snapshot that supplied the live request.
    hitl_source_positions: dict[str, int] = field(default_factory=dict)


def apply_startup_events(events: list[Event | ProjectedEvent], view: StartupRecoveryView):
    for event in events:
        if not view.session_id:
            view.session_id = event.session_id
        if not view.task_id and event.task_id:
            view.task_id = event.task_id
        if not view.agent_id and event.agent_id:
            view.agent_id = event.agent_id
        view.events_total += 1
        # One state machine for both full recovery and startup calibration.
        _apply(view, event)
    _rebuild_agents(view)
    for task in view.tasks.values():
        task.details_complete = False
    return view


async def rebuild_startup_view(store, session_id: str, *, through_position: int):
    started = perf_counter()
    stored = await store.read_last_of_type(session_id, EventType.STATE_SNAPSHOT,
                                          projection=STARTUP_SNAPSHOT_PROJECTION)
    snapshot = None
    view = None

    def decode(candidate, source_position):
        blob = candidate.state_blob
        sessions = blob.get("sessions")
        if (not isinstance(sessions, dict) or not isinstance(sessions.get(session_id), dict)
                or sessions[session_id].get("id") != session_id):
            raise ValueError("Startup snapshot has no matching session")
        result = StartupRecoveryView(**vars(deserialize_view(blob)))
        result.hitl_source_positions = dict.fromkeys(result.hitl.opened, source_position)
        return result
    if stored is not None:
        payload = stored.event.payload
        # Validate metadata before reading an old full blob: incompatible
        # snapshots need no body at all. New snapshots carry a compact sibling.
        metadata = snapshot_facts_from_event(replace(
            stored.event, payload={**payload, "state_blob": {}},
        ))
        if (snapshot_is_usable(metadata, through_position)
                and stored.position <= through_position):
            compact = payload.get("startup_state_blob")
            if payload.get("startup_projection_version") == STARTUP_SNAPSHOT_VERSION and isinstance(compact, dict):
                candidate = replace(metadata, state_blob=compact)
                try:
                    view = decode(candidate, stored.position)
                    snapshot = candidate
                except (KeyError, TypeError, ValueError, AttributeError):
                    logger.warning("Invalid compact startup snapshot for %s; trying legacy projection", session_id)
            if view is None:
                legacy = await store.read_last_of_type(session_id, EventType.STATE_SNAPSHOT,
                                                      projection=STARTUP_PROJECTION)
                # A concurrent newer snapshot is usable only at this fixed cut.
                if legacy is not None and legacy.position <= through_position:
                    candidate = snapshot_facts_from_event(legacy.event)
                    if snapshot_is_usable(candidate, through_position):
                        try:
                            view = decode(candidate, legacy.position)
                            snapshot = candidate
                        except (KeyError, TypeError, ValueError, AttributeError):
                            logger.warning("Invalid startup snapshot for %s; replaying control facts", session_id)
    cursor = 0
    mode = "replay"
    if view is not None:
        cursor = snapshot.last_commit_position
        mode = "snapshot"
    else:
        view = StartupRecoveryView(run_id=session_id, session_id="", task_id="", agent_id="")
    replayed = 0
    # Batch delta as well as full replay: a very stale snapshot must not create
    # an unbounded list of envelopes, even though their payloads are now small.
    while cursor < through_position:
        upper = min(cursor + REPLAY_BATCH, through_position)
        rows = await store.read_range(session_id, after_position=cursor, through_position=upper,
                                      exclude_types=REPLAY_EXCLUDE_TYPES, projection=STARTUP_PROJECTION)
        apply_startup_events([row.event for row in rows], view)
        for row in rows:
            if row.event.type in (EventType.HITL_OPENED, EventType.HITL_REQUIRED):
                rid = row.event.payload.get("hitl_id")
                if rid in view.hitl.opened:
                    view.hitl_source_positions[rid] = row.position
        view.hitl_source_positions = {
            rid: position for rid, position in view.hitl_source_positions.items()
            if rid in view.hitl.opened
        }
        replayed += len(rows)
        cursor = upper
    if mode == "snapshot" and snapshot.projection_version == 10:
        await restore_legacy_canceled_tasks(store, session_id, view, through_position,
                                            projection=STARTUP_PROJECTION)
    duration = perf_counter() - started
    for task in view.tasks.values():
        task.details_complete = False
    logger.log(logging.INFO if duration >= 1 else logging.DEBUG,
               "Startup rebuild: session=%s mode=%s head=%d replayed=%d duration_ms=%.1f",
               session_id, mode, through_position, replayed, duration * 1000)
    return view


async def read_startup_task(store, session_id: str, task_id: str, *, through_position: int):
    view = StartupRecoveryView(run_id=session_id, session_id=session_id, task_id=task_id, agent_id="")
    cursor = 0
    while cursor < through_position:
        upper = min(cursor + REPLAY_BATCH, through_position)
        rows = await store.read_range(
            session_id, after_position=cursor, through_position=upper,
            include_types=tuple({EventType.TASK_CREATED, EventType.TASK_FINALIZED,
                                 EventType.RECOGNIZE_INTENT_TOOL_CALL, *TASK_STATUS_BY_EVENT}),
            related_task_id=task_id, projection=STARTUP_PROJECTION,
        )
        apply_startup_events([row.event for row in rows], view)
        cursor = upper
    return view.tasks.get(task_id)


async def read_startup_pending_hitl(store, view, session_id, head):
    by_position = {}
    for rid in view.hitl.pending:
        position = view.hitl_source_positions[rid]
        if not 0 < position <= head:
            raise ValueError(f"Invalid HITL source position for {rid}")
        by_position.setdefault(position, []).append(rid)
    requests = {}
    fields = ("id", "hitl_id", "form", "session_id", "task_id", "agent_id", "tenant_id",
              "created_at", "subject_id", "prompt", "detail", "fields", "proposal",
              "delivery", "tool_call_id", "stage", "capability_id", "question",
              "context", "questions", "arguments")
    for position, ids in by_position.items():
        # Read exactly one durable envelope. Snapshot selection is restricted to
        # the requested live entries; answers and execution payloads stay unread.
        projection = EventReadProjection({
            EventType.HITL_OPENED: tuple((key,) for key in fields),
            EventType.HITL_REQUIRED: tuple((key,) for key in fields),
            EventType.STATE_SNAPSHOT: tuple(
                ("state_blob", "hitl", group, rid, key)
                for group in ("pending", "resolved") for rid in ids for key in fields
            ),
        })
        rows = await store.read_range(session_id, after_position=position - 1,
                                      through_position=position, projection=projection)
        if len(rows) != 1:
            raise ValueError(f"Missing HITL source at position {position}")
        event = rows[0].event
        if event.type == EventType.STATE_SNAPSHOT:
            hitl = deserialize_view(event.payload.get("state_blob") or {}).hitl
        else:
            hitl = fold_hitl_snapshot([event])
        for rid in ids:
            requests[rid] = hitl.opened[rid]
    return tuple(deepcopy(requests[rid].to_view()) for rid in view.hitl.pending)
