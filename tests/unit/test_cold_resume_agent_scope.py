"""Cold-restart resume must ingest the user's reply into the resumed task's real agent
scope — the "first prompt after restart is lost / blank reply" bug.

On a cold app restart the pending wait_for_user HITL is rebuilt from the event store.
The HITL_REQUIRED event never persisted agent_id, so the rebuilt request has
agent_id="". _inject_user_reply then ingested the reply into a memory scope with
agent_id="".

That reply IS findable by recall_recent (task-layer recall keys on task_id), but the
actor's history comes from AgentRecallSource → recall_recent_by_agent, which filters by
scope.agent_id. With agent_id="" the reply is invisible to assembly, so the actor sees
history ending in the prior assistant turn, appends the "continue task" resume cue, and
produces a blank reply. Only the FIRST post-restart reply is affected; later parks create
live HITLs that still carry agent_id.

Two layers:
1. _inject_user_reply falls back to the resumed task's agent scope when req.agent_id is
   missing (robust even if the projection ever loses it).
2. The HITL fold round-trips agent_id off the event (fix at source).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ctx_weft.core.control.reducers import fold_hitl_snapshot
from ctx_weft.core.hitl.registry import PendingHitl
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_NORMAL,
    HitlDecision,
    UserTurnDelivery,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


_TS = datetime(2026, 7, 13, tzinfo=UTC)

def _user_turn_req(
    *, hitl_id="hit1", session_id="s1", task_id="t1", agent_id="ag1",
    outcome="accepted", message="ship it", preface=PREFACE_NORMAL,
):
    """一条**已终局**的 `UserTurn` 请求——`_inject_user_reply` / `_write_hitl_reply_turn`
    收的就是这个形态（core 内部的活记录，不是 host 视图）。"""
    req = PendingHitl(
        id=hitl_id, form=HITL_FORM_WAIT, session_id=session_id, task_id=task_id,
        agent_id=agent_id,
        delivery=UserTurnDelivery(task_id=task_id, preface=preface),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    req.decision = HitlDecision(outcome=outcome, message=message)
    req.resolved_at = datetime(2026, 1, 1, tzinfo=UTC)
    return req


# ── Layer 1: _inject_user_reply scope fallback ─────────────────────────────────


@pytest.mark.asyncio
async def test_inject_user_reply_reply_visible_to_agent_recall(monkeypatch):
    """When the rebuilt request lost agent_id, the reply must still be recallable via
    recall_recent_by_agent (the actor's history path), which filters by scope.agent_id."""
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
    from ctx_weft.core import CtxWeftRuntime


    runtime = make_runtime(agent_provider=InlineAgentTemplateProvider())
    mem = InMemoryMemoryProvider()
    runtime.providers.register_memory(mem)

    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    task = Task(id="t1", session_id="s1", status="SUSPENDED",
                assigned_agent_id="ag_root", creator_agent_id="ag_root")
    async def _mark_human_resolved(tid, *, hitl_id):
        if tid == "t1" and task.status not in ("FINISHED", "FAILED", "CANCELED"):
            task.status = "PENDING"

    task_manager = SimpleNamespace(get_task=lambda tid: task if tid == "t1" else None,
                                   children_of=lambda tid: set(),
                                   mark_human_resolved=_mark_human_resolved)

    # Rebuilt-from-projection HITL: agent_id was NOT persisted → empty.
    req = _user_turn_req(hitl_id="h1", agent_id="", message="用户回复")

    await runtime._inject_user_reply(req, session, task_manager)

    # The actor recalls task body by agent (recall_recent_by_agent filters on scope.agent_id).
    agent_scope = MemoryAddress(session_id="s1", task_id=None, agent_id="ag_root")
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    recs = await mem.recall_recent_by_agent(agent_scope, [MemoryEventType.USER_PROMPT], 10, ctx)
    assert [r.content for r in recs] == ["用户回复"], (
        "reply invisible to recall_recent_by_agent — it was ingested into the empty agent "
        "scope, so the actor never sees it (→ continue-cue → blank reply)"
    )


# ── Layer 2: HITL_REQUIRED projection preserves agent_id ───────────────────────


@pytest.mark.asyncio
async def test_hitl_required_projection_preserves_agent_id():
    """fold_hitl_snapshot must round-trip agent_id from the HITL_REQUIRED event so cold
    resume refills a request that still knows its agent scope."""
    ev = Event(
        id="evt_1", run_id="r1", sequence=1, session_id="s1", task_id="t1",
        agent_id="ag_root", type=EventType.HITL_REQUIRED, timestamp=_TS,
        payload={"hitl_id": "h1", "form": "wait", "agent_id": "ag_root",
                 "context": "plain_text"},
    )
    pending = fold_hitl_snapshot([ev]).pending
    assert "h1" in pending
    assert pending["h1"].agent_id == "ag_root", (
        f"agent_id lost through HITL projection: {pending['h1'].agent_id!r}"
    )
