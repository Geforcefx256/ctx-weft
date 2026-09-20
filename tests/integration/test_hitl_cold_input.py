"""Integration: the tool-stage COLD reconcile path, end to end.

Proves that a human answer that was already recorded for a ``tool_call_id`` — the
state you are in after "user answered, then the process restarted" — is written back
as that call's TOOL_RESULT when ``ReconcileStep`` re-invokes the dangling tool call
through a REAL ``CapabilityGateway`` + real ``ControlCapabilityProvider``, **without
asking the human again**.

This is the tool-stage twin of
``test_gateway_authz_hitl.py::test_a_cached_decision_short_circuits_without_asking_again``.
It is not redundant with it: the authorization stage short-circuits at the top of
``invoke`` (before ``authorizer.authorize()``), while the tool stage can only
short-circuit inside ``_resolve_human`` — a provider has to run far enough to yield
``needs_human`` before anyone knows a human is involved. Without that second
short-circuit, ``open()`` idempotently returns the already-resolved request,
``HitlWaiter.wait()`` sees ``resolved`` and reports eviction, and the gateway parks:
the answer the human already gave never reaches the model and the task re-hangs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL, HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.core.hitl.snapshot import HitlSnapshot
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.hitl_waiter import HitlWaiter
from ctx_weft.core.loop.steps.reconcile import ReconcileStep
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import (
    PROVIDER_NAME,
    ControlCapabilityProvider,
)
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.protocols.hitl import HitlDecision
from ctx_weft.protocols.memory import MemoryEvent, MemoryEventType
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


class _PassthroughNormalizer:
    async def __call__(self, content, session_id, tenant_id):
        return content, content


@pytest.mark.asyncio
async def test_cold_input_reconcile_writes_tool_result() -> None:
    """Cold reconcile: pre-answered HITL → ReconcileStep short-circuits → TOOL_RESULT in memory."""

    session_id = "s1"
    task_id = "tsk_1"
    agent_id = "agt_1"
    tool_call_id = "tcA"

    mem = InMemoryMemoryProvider()
    bus = InProcessEventBus()

    # ── CapabilityCache: seed the control tool capability ────────────────────────
    # The gateway looks up capability by tool name → ToolCapability → finds provider by cap.id prefix
    cap = ToolCapability(
        id=f"{PROVIDER_NAME}:ask_user",
        name="ask_user",
        kind="tool",
        description="Pause execution and request input or approval from a human.",
    )
    cache = CapabilityCache()
    cache.put(agent_id, [cap])

    # ── HITL subsystem refilled from a restart: the decision is already on record ──
    # This is exactly what `rebuild_hitl` produces after a crash — a decision keyed by
    # (session, tool_call, stage) with no wait slot. No live pending, no live coroutine.
    registry = HitlRegistry()
    registry.load_snapshot(HitlSnapshot(decisions_for={
        (session_id, tool_call_id, HITL_STAGE_TOOL): (
            HitlDecision(outcome="accepted", message="use postgres"), None,
        ),
    }))
    service = HitlService(registry=registry, event_bus=bus,
                          reply_intake=ReplyIntake(_PassthroughNormalizer()))
    assert registry.decision_for(session_id, tool_call_id, HITL_STAGE_TOOL) is not None

    # ── ControlCapabilityProvider: register the session ─────────────────────────
    provider = ControlCapabilityProvider()
    session = Session(id=session_id, tenant_id="default", user_prompt="do it", status="RUNNING")
    task = Task(id=task_id, session_id=session_id, status="ACTIVE", title="T1")
    tm = SimpleNamespace(get_task=lambda tid: task)
    provider.register_session(session_id, tm, session)

    # ── CapabilityGateway: use the SAME mem instance as LoopContext ──────────────
    gateway = CapabilityGateway(
        capability_cache=cache,
        capability_providers=[provider],
        memory=mem,          # <-- SAME instance as ctx.memory below
        event_bus=bus,
    )

    # ── LoopState + LoopContext ──────────────────────────────────────────────────
    # template 必须在 extra：reconcile 现会先 resolve_and_bind（spec/07 §6 修复）——control provider
    # 据此把 ask_user 等控制工具绑入 cache,gateway.invoke 才找得到。
    from tests.integration.test_minimal_loop import make_echo_template

    scope = MemoryAddress(session_id=session_id, task_id=task_id, agent_id=agent_id)
    agent = SimpleNamespace(id=agent_id, template_id="tmpl_a", session_id=session_id)
    state = LoopState(run_id="run_1", session=session, task=task, agent=agent, scope=scope,
                      extra={"template": make_echo_template()}, resolved_model=SimpleNamespace(model="mock", account=""))

    pctx = ProviderContext(
        session_id=session_id,
        tenant_id="default",
        task_id=task_id,
        agent_id=agent_id,
    )
    ctx = LoopContext(
        assembler=None,
        llm=None,
        memory=mem,          # <-- SAME instance as gateway._memory above
        event_bus=bus,
        provider_ctx=pctx,
        capability_gateway=gateway,
        capability_cache=cache,
        capability_providers=[provider],
        hitl=service,
        # timeout_sec=0：真要走到「开一个新请求然后等人」这条路，它会立刻驱逐 → HitlPark，
        # 用例响亮地红，而不是挂住整场测试。短路生效时这个 waiter 根本不会被用到。
        waiter=HitlWaiter(registry, timeout_sec=0),
    )

    # ── Pre-ingest the assistant LLM_RESPONSE with the dangling tool_call ───────
    # ReconcileStep scans for LLM_RESPONSE events, finds tool_calls without matching TOOL_RESULT.
    await mem.ingest(
        MemoryEvent(
            type=MemoryEventType.LLM_RESPONSE,
            address=scope,
            content="",
            timestamp=now_utc(),
            role="assistant",
            metadata={
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "name": "control__ask_user",
                        "input": {"questions": [{"question": "Which DB?"}]},
                    }
                ]
            },
        ),
        pctx,
    )

    # ── Run ReconcileStep ────────────────────────────────────────────────────────
    step = ReconcileStep()
    outcome = await step.execute(state, ctx)

    # 填完 dangling → prepare 重装 prompt（再 act 调 LLM）
    assert outcome.next_step == "prepare"

    # ── Assert: TOOL_RESULT written with the human's answer ──────────────────────
    results = await mem.recall_recent(
        scope=scope,
        types=[MemoryEventType.TOOL_RESULT],
        limit=10,
        ctx=pctx,
    )
    assert any(
        r.metadata.get("tool_call_id") == tool_call_id and "use postgres" in r.content
        for r in results
    ), f"Expected TOOL_RESULT for {tool_call_id!r} with 'use postgres'; got: {results}"

    # ── Assert: no NEW pending HITL was created (cold short-circuit worked) ──────
    assert registry.list_pending() == [], f"Expected no pending HITL; got: {registry.list_pending()}"


@pytest.mark.asyncio
async def test_the_cold_decision_of_another_session_does_not_apply() -> None:
    """同名 tool_call_id、不同 session → 短路**不得**命中。

    LLM 的 tool_call id 常是 `call_1` 这类短值；旧 `HitlManager.find_for_tool_call`
    不过滤 session，A 会话里人给的答案会被 B 会话的同名调用直接取用。三维键
    (session_id, tool_call_id, stage) 从构造上关掉它——这条用例守住那扇门。
    """
    registry = HitlRegistry()
    registry.load_snapshot(HitlSnapshot(decisions_for={
        ("session_A", "call_1", HITL_STAGE_TOOL): (
            HitlDecision(outcome="accepted", message="use postgres"), None,
        ),
    }))
    assert registry.decision_for("session_A", "call_1", HITL_STAGE_TOOL) is not None
    assert registry.decision_for("session_B", "call_1", HITL_STAGE_TOOL) is None
    # 阶段也是键的一维：授权步不得吃掉工具步的答复（反之亦然）。
    assert registry.decision_for("session_A", "call_1", "authz") is None


@pytest.mark.asyncio
async def test_snapshot_placeholder_keeps_its_epoch_created_at() -> None:
    """装填出来的占位项不参与 pending 列表，`created_at` 取最小值即可（回归守卫）。"""
    registry = HitlRegistry()
    registry.load_snapshot(HitlSnapshot(decisions_for={
        ("s", "c", HITL_STAGE_TOOL): (HitlDecision(outcome="accepted"), None),
    }))
    assert registry.list_pending() == []
    found = registry.find_for_tool_call("s", "c", HITL_STAGE_TOOL)
    assert found is not None and found.created_at == datetime(1970, 1, 1, tzinfo=UTC)
