"""HitlPark 信号本身，以及 `_run_loop` 对它的处理（挂起而非失败）。

原先本文件还覆盖 `defer`、`HitlManager.wait/wait_for_decision` 的双出口、以及
approval 的冷路径短路。三者都随重设计消失或换了归属：

- `AuthorizationDecision.defer` 已删——它只能说「挂起」、说不出问什么，于是 authorizer
  必须自己先去登记请求（耦合的源头）。取代它的是 `needs_human: HitlAsk`，见
  `test_gateway_authz_hitl.py` 与 `test_authorizer_human_stateless.py`。
- `wait()` 抛 `HitlPark` / `wait_for_decision()` 返 `None` 这组双出口已合并成
  `HitlWaiter.wait()` 单一出口（驱逐返 `None`，翻译成 park 的权力只在 gateway）：
  `test_hitl_waiter.py` + `test_gateway_*_needs_human` 的驱逐用例。
- 冷决定短路：`test_gateway_authz_hitl.py::test_a_cached_decision_short_circuits_without_asking_again`
  与 `test_hitl_registry_load.py`。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from types import SimpleNamespace


def test_hitl_park_is_base_exception_not_exception() -> None:
    from ctx_weft.core.loop.signals import HitlPark
    assert issubclass(HitlPark, BaseException)
    assert not issubclass(HitlPark, Exception)


def test_hitl_park_carries_ids() -> None:
    from ctx_weft.core.loop.signals import HitlPark
    p = HitlPark(hitl_id="hit_1", tool_call_id="tc1")
    assert p.hitl_id == "hit_1" and p.tool_call_id == "tc1"


async def test_run_loop_catches_park_returns_awaiting_human() -> None:
    """_run_loop 必须接住 HitlPark、报出 awaiting_human 结局、且不抛（不落 FAILED）。

    Task 4 起 task.status / TaskAwaitingHuman 都不在 run 里落地：run 只交回
    `RunOutcome(kind=awaiting_human, hitl_id=...)`，TaskManager 据它写状态发事件
    （TM 侧的对照断言见 tests/unit/test_task_manager_owns_status.py）。
    """
    from collections.abc import AsyncIterator

    from ctx_weft.core import CtxWeftRuntime, ProviderRegistry
    from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind
    from ctx_weft.protocols.events import EventType
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.loop.signals import HitlPark
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task
    from ctx_weft.protocols import (
        LoopConfig,
        MemoryConfig,
        MemoryAddress,
        ProviderContext,
    )
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    # ── real objects ──────────────────────────────────────────────────────────
    session = Session(
        id="s_park_1",
        user_prompt="park test",

        tenant_id="default",
        root_agent_id="agt_p1",
    )
    task = Task(
        id="tsk_park_1",
        session_id="s_park_1",
        status="ACTIVE",
        tenant_id="default",
        assigned_agent_id="agt_p1",
    )
    agent = Agent(
        id="agt_p1",
        session_id="s_park_1",
        template_id="tpl_test",
        tenant_id="default",
        loop_guard=LoopGuard(),
        memory_config=MemoryConfig(),
        loop_config=LoopConfig(),
    )
    scope = MemoryAddress(session_id="s_park_1", task_id="tsk_park_1", agent_id="agt_p1")
    state = LoopState(run_id="run_p1", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    # ── stub driver that raises HitlPark immediately ──────────────────────────
    class _ParkingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext
        ) -> AsyncIterator[StepOutcome]:
            raise HitlPark(hitl_id="req_park", tool_call_id="tc_park")
            yield  # make it an async generator

    # ── runtime wired with real bus + memory (no LLM needed) ─────────────────
    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(InMemoryMemoryProvider())

    rt = make_runtime(
        agent_provider=InlineAgentTemplateProvider(),
        providers=registry,
        event_store=None,  # uses InMemoryEventStore default
    )
    rt._event_bus = bus  # replace with observable bus

    # minimal LoopContext (only event_bus needed by _run_loop's make_event)
    loop_ctx = LoopContext(
        assembler=None,  # type: ignore[arg-type]
        llm=None,  # type: ignore[arg-type]
        memory=InMemoryMemoryProvider(),
        event_bus=bus,
        provider_ctx=ProviderContext(
            session_id="s_park_1", tenant_id="default",
            task_id="tsk_park_1", agent_id="agt_p1",
        ),
        capability_cache=CapabilityCache(),
    )

    # ── collect events via handler (registered BEFORE run so no events are missed) ──
    collected: list = []

    async def _handler(ev) -> None:
        collected.append(ev)

    bus.subscribe(None, _handler)

    # ── call _run_loop; must NOT raise ────────────────────────────────────────
    final_state = await rt._run_loop(
        state=state,
        loop_ctx=loop_ctx,
        driver=_ParkingDriver(),  # type: ignore[arg-type]
        run_id="run_p1",
        initial_step="act",
        task=task,
        agent=agent,
    )

    # ── assertions ───────────────────────────────────────────────────────────
    # run 侧不写 task 状态了；park 的全部信息落在 RunOutcome 上，交给 TaskManager。
    assert task.status == "ACTIVE", f"run 不该改 task.status, got {task.status!r}"
    outcome = final_state.run_outcome
    assert outcome is not None and outcome.kind is RunOutcomeKind.AWAITING_HUMAN
    assert outcome.hitl_id == "req_park"

    run_finished = next((e for e in collected if e.type == EventType.RUN_FINISHED), None)
    assert run_finished is not None, "RUN_FINISHED not emitted"
    assert run_finished.payload["outcome"] == "awaiting_human"
    assert run_finished.payload.get("error") is None

    task_failed_events = [e for e in collected if e.type == "TaskFailed"]
    assert task_failed_events == [], f"unexpected TaskFailed events: {task_failed_events}"

    # park 是 **task 级事实**：这个 task 卡住了、卡它的是哪个 HITL 请求——但它由
    # TaskManager 宣布（Task 4：task 状态事件只从 TM 出），run 里一条都不该有。
    assert [e for e in collected if e.type == EventType.TASK_AWAITING_HUMAN] == []
    assert [e for e in collected if e.type == EventType.TASK_SUSPENDED] == []


def test_authorizer_filter_is_gone():
    """filter 零调用点，且对 HumanConfirmation 会真的发一个 HITL 请求并等人——是陷阱。"""
    from ctx_weft.protocols.capability import Authorizer
    assert not hasattr(Authorizer, "filter")
