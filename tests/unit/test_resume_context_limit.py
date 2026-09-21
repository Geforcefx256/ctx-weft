"""resume_session must propagate context_limit from the persisted SessionView.

Fix 1 regression guard: before the fix, resume_session built Session(...)
without passing context_limit, so it silently fell back to the model-level
default (180_000) regardless of what was stored in the event at session
creation time.  After the fix it must round-trip any non-default value.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.core.orchestrator.lifecycle.session_registry import SessionRegistry
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.events import EventPersister
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 6, 13, tzinfo=timezone.utc)

NON_DEFAULT_CONTEXT_LIMIT = 64_000


def _ev(seq: int, sid: str, type_: EventType, **payload) -> Event:
    return Event(
        id=f"evt_{sid}_{seq:04d}",
        run_id="run_1",
        sequence=seq,
        session_id=sid,
        type=type_,
        timestamp=_TS,
        payload=payload,
    )


async def test_resume_session_preserves_context_limit() -> None:
    """Session built by resume_session carries context_limit from persisted event."""
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    EventPersister(store, bus)

    sid = "ses_resume_ctx"

    # Seed the store with a SESSION_CREATED event that carries a custom context_limit.
    await append_one(store, _ev(1, sid, EventType.SESSION_CREATED,
                           template_id="tpl_echo",
                           user_prompt="hello",
                           root_agent_id="agt_root",
                           token_budget=200_000,
                           context_limit=NON_DEFAULT_CONTEXT_LIMIT))

    # Build a minimal agent_lifecycle_manager mock — resume_session does NOT call
    # instantiate, so any object with the right shape works.
    lm = MagicMock()

    sm = SessionRegistry(agent_lifecycle_manager=lm, event_bus=bus)

    # resume_session emits events on the bus; we don't need to assert them here.
    session, root_task, task_manager = await sm.resume_session(
        session_id=sid,
        event_store=store,
        user_prompt="continue",
        tenant_id="default",
    )

    assert session.context_limit == NON_DEFAULT_CONTEXT_LIMIT, (
        f"Expected context_limit={NON_DEFAULT_CONTEXT_LIMIT}, got {session.context_limit}"
    )
