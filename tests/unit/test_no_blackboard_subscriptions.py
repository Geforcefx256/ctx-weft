"""Phase 3, Task 2: the loop driver no longer creates predecessor/subtask blackboard subscriptions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.driver import StepDriver
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import ProviderContext


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


@pytest.mark.asyncio
async def test_no_blackboard_subscriptions_created() -> None:
    """After Phase 3, _ensure_blackboard_subscriptions is a no-op and must never call subscribe_topic."""
    calls: list[dict] = []

    class FakeMem:
        async def subscribe_topic(self, **kw):
            calls.append(kw)

    class FakeTM:
        def children_of(self, tid: str) -> set[str]:
            return {"c1", "c2"}

    task = Task(id="T", session_id="s1", status="ACTIVE")
    state = SimpleNamespace(task=task)
    ctx = SimpleNamespace(task_manager=FakeTM(), memory=FakeMem(), provider_ctx=_ctx())

    driver = StepDriver(steps={})
    await driver._ensure_blackboard_subscriptions(state, ctx)

    assert calls == [], (
        "no blackboard subscriptions should be created in Phase 3; "
        f"got calls: {calls}"
    )
