"""loop 在 run 结束时产出 RunOutcome（Task 2）。

本 task 只加产出、不删旧路径——旧的写状态与发事件仍在，
所以既有行为一字不变，全量应当仍是那 2 条基线失败。

搭台方式参照 `tests/unit/test_finalize_fail_reason.py`（SimpleNamespace state + Step().execute()
直调）、`tests/unit/test_hitl_park.py`（`_run_loop` + stub driver）、
`tests/unit/test_run_loop_outage.py`（`run_single_task` 全链路 + `_OutageLLM`）——三种既有自
包含搭台手法的合集，本文件不改动那两个文件本身。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import FinalizeStep
from ctx_weft.core.loop.steps.observe import Verdict
from ctx_weft.core.loop.steps.suspend import SuspendStep
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryConfig,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
)
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all


T = MemoryEventType


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


class _FakeTM:
    def children_of(self, task_id: str) -> set[str]:
        return set()


def _loop_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_ctx(), task_manager=_FakeTM(),
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


# ── finalize / observer verdict ────────────────────────────────────────────────


async def _finalize_state(mem, *, verdict_outcome: str, summary: str, retry_count: int = 0):
    scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="ag2")
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=scope, content="do it",
                                 timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                                 role="user"), _ctx())
    task = Task(id="c1", session_id="s1", status="ACTIVE", tenant_id="default",
                assigned_agent_id="ag2", creator_agent_id="ag1", parent_task_id="p1",
                title="Doing Task", user_prompt="do it", settings=NormalTaskSettings())
    task.retry_count = retry_count
    verdict = Verdict(task_outcome=verdict_outcome, act_recap=summary, task_summary="")
    agent = SimpleNamespace(id="ag2", loop_config=LoopConfig())
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="r1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent, verdict=verdict,
                           run_outcome=None, extra={})


async def run_until_finalize(*, verdict: str, summary: str = "", retry_count: int = 0):
    """自包含搭台：直调 `FinalizeStep().execute()`（同 test_finalize_fail_reason.py 手法），
    再把它 state_patch 里的 run_outcome 应用回 state（driver.apply_patch 在真链路里做的事）。
    """
    mem = InMemoryMemoryProvider()
    state = await _finalize_state(mem, verdict_outcome=verdict, summary=summary,
                                  retry_count=retry_count)
    outcome = await FinalizeStep().execute(state, _loop_ctx(mem))
    state.run_outcome = outcome.state_patch.get("run_outcome")
    return state


@pytest.mark.asyncio
async def test_observer_success_produces_a_completed_outcome() -> None:
    state = await run_until_finalize(verdict="success", summary="做完了")
    assert state.run_outcome is not None
    assert state.run_outcome.kind is RunOutcomeKind.COMPLETED
    assert state.run_outcome.verdict == "success"
    assert state.run_outcome.summary == "做完了"


@pytest.mark.asyncio
async def test_observer_retry_produces_the_verdict_not_the_disposition() -> None:
    """loop 只报「observer 说重试」，**不判**预算够不够——那是 TM 的活。"""
    state = await run_until_finalize(verdict="continue", summary="进行中", retry_count=99)
    assert state.run_outcome.kind is RunOutcomeKind.COMPLETED
    # 即使预算早耗尽（max_retries=3），这里仍是 retry
    assert state.run_outcome.verdict == "continue"


# ── suspend ─────────────────────────────────────────────────────────────────


async def run_until_suspend(*, titles: list[str]):
    """真 `LoopState`（非 SimpleNamespace）：SuspendStep 末尾会 fire-and-forget 一个
    `dataclasses.replace(state)` 快照（background_observe），SimpleNamespace 不是
    dataclass 会在那一步炸掉。"""
    from ctx_weft.core.loop.driver import LoopState
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session

    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="c1", agent_id="ag2")
    task = Task(id="c1", session_id="s1", status="SUSPENDED", tenant_id="default",
                assigned_agent_id="ag2", creator_agent_id="ag1", parent_task_id="p1",
                title="Suspending Task", user_prompt=None,
                settings=NormalTaskSettings(spawn_titles=list(titles)))
    task.user_prompt_in_memory = True  # 跳过 ingest 分支（不是本测试关心的）
    session = Session(id="s1", user_prompt="do it",
                      tenant_id="default", root_agent_id="ag2")
    agent = Agent(id="ag2", session_id="s1", template_id="tpl_test", tenant_id="default", loop_guard=LoopGuard(),
                 memory_config=MemoryConfig(), loop_config=LoopConfig())
    state = LoopState(run_id="r1", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))
    # launch_recap 是 fire-and-forget：给它一个真 event_bus，否则它的
    # finally 块（无条件发 OBSERVE_APPLIED）会在测试结束后抛 "exception was never
    # retrieved"。段里没有任何 CONVERSATION_TURN → n_raw==0 幂等护栏立即 return，
    # 不会真的调 LLM。
    from ctx_weft.core.loop.driver import StepDriver
    from ctx_weft.providers.events import InProcessEventBus
    ctx = _loop_ctx(mem)
    ctx.event_bus = InProcessEventBus()
    outcome = await SuspendStep().execute(state, ctx)
    state.run_outcome = outcome.state_patch.get("run_outcome")
    # 等后台 recap task 真正跑完（它的幂等护栏会立即 return），不留悬空 task 到测试之外。
    await StepDriver.for_context(ctx).executions.wait(task.id)
    return state


@pytest.mark.asyncio
async def test_suspend_on_children_carries_spawn_titles() -> None:
    state = await run_until_suspend(titles=["查资料"])
    assert state.run_outcome.kind is RunOutcomeKind.SUSPENDED_ON_CHILDREN
    assert list(state.run_outcome.spawn_titles) == ["查资料"]


# ── HITL park（runtime._run_loop） ───────────────────────────────────────────


async def run_until_park():
    """同 test_hitl_park.py::test_run_loop_catches_park_returns_awaiting_human 的搭台。"""
    from collections.abc import AsyncIterator

    from ctx_weft.core import ProviderRegistry
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.loop.signals import HitlPark
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.protocols import LoopConfig as RTLoopConfig
    from ctx_weft.protocols import MemoryConfig
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider as MemProv
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    session = Session(id="s_park_2", user_prompt="park test",
                      tenant_id="default", root_agent_id="agt_p2")
    task = Task(id="tsk_park_2", session_id="s_park_2", status="ACTIVE",
               tenant_id="default", assigned_agent_id="agt_p2")
    agent = Agent(id="agt_p2", session_id="s_park_2", template_id="tpl_test",
                 tenant_id="default",
                 loop_guard=LoopGuard(), memory_config=MemoryConfig(),
                 loop_config=RTLoopConfig())
    scope = MemoryAddress(session_id="s_park_2", task_id="tsk_park_2", agent_id="agt_p2")
    state = LoopState(run_id="run_p2", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    class _ParkingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext,
        ) -> AsyncIterator[StepOutcome]:
            raise HitlPark(hitl_id="req_park_2", tool_call_id="tc_park_2")
            yield

    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(MemProv())
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider(), providers=registry,
                      event_store=None)
    rt._event_bus = bus

    loop_ctx = LoopContext(
        assembler=None, llm=None, memory=MemProv(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s_park_2", tenant_id="default",
                                     task_id="tsk_park_2", agent_id="agt_p2"),
        capability_cache=CapabilityCache(),
    )

    return await rt._run_loop(state=state, loop_ctx=loop_ctx, driver=_ParkingDriver(),
                              run_id="run_p2", initial_step="act", task=task, agent=agent)


@pytest.mark.asyncio
async def test_hitl_park_produces_awaiting_human_with_hitl_id() -> None:
    state = await run_until_park()
    assert state.run_outcome.kind is RunOutcomeKind.AWAITING_HUMAN
    assert state.run_outcome.hitl_id == "req_park_2"


# ── LLM outage（runtime._run_loop 经 run_single_task 全链路） ─────────────────


async def run_until_outage():
    """同 test_run_loop_outage.py::test_outage_marks_session_interrupted_not_failed 的搭台。"""
    from ctx_weft.core.models.config import RuntimeConfig
    from ctx_weft.protocols import LLMOutageError
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    class _OutageLLM(MockLLMAdapter):
        def complete(self, request, stream=True):
            self.last_request = request
            async def _gen():
                raise LLMOutageError("simulated outage exhausted")
                yield  # pragma: no cover
            return _gen()

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=_OutageLLM(responses=[]), agent_provider=resolver,
                           config=RuntimeConfig(llm_self_heal_max_attempts=1))
    runtime.providers.register_memory(InMemoryMemoryProvider())

    await initialize_all(runtime)
    _handle, state = await runtime.run_single_task(
        template_id="agent:tpl_echo", user_prompt="hi")
    return state


@pytest.mark.asyncio
async def test_llm_outage_produces_interrupted_not_retriable() -> None:
    state = await run_until_outage()
    assert state.run_outcome.kind is RunOutcomeKind.INTERRUPTED
    assert state.run_outcome.reason == "llm_outage"
    assert state.run_outcome.retriable is False     # outage 从不原地重试


# ── cancellation（runtime._run_loop 的 except asyncio.CancelledError） ────────


async def run_until_cancel():
    """同 run_until_park 的搭台，只是驱动改抛 asyncio.CancelledError（BaseException，
    今天 `except asyncio.CancelledError` 分支不重抛，正常 return state）。"""
    from collections.abc import AsyncIterator

    from ctx_weft.core import ProviderRegistry
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.protocols import LoopConfig as RTLoopConfig
    from ctx_weft.protocols import MemoryConfig
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider as MemProv
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    session = Session(id="s_cancel_2", user_prompt="cancel test",
                      tenant_id="default", root_agent_id="agt_c2")
    task = Task(id="tsk_cancel_2", session_id="s_cancel_2", status="ACTIVE",
               tenant_id="default", assigned_agent_id="agt_c2")
    agent = Agent(id="agt_c2", session_id="s_cancel_2", template_id="tpl_test",
                 tenant_id="default",
                 loop_guard=LoopGuard(), memory_config=MemoryConfig(),
                 loop_config=RTLoopConfig())
    scope = MemoryAddress(session_id="s_cancel_2", task_id="tsk_cancel_2", agent_id="agt_c2")
    state = LoopState(run_id="run_c2", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    class _CancellingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext,
        ) -> AsyncIterator[StepOutcome]:
            raise asyncio.CancelledError
            yield

    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(MemProv())
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider(), providers=registry,
                      event_store=None)
    rt._event_bus = bus

    loop_ctx = LoopContext(
        assembler=None, llm=None, memory=MemProv(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s_cancel_2", tenant_id="default",
                                     task_id="tsk_cancel_2", agent_id="agt_c2"),
        capability_cache=CapabilityCache(),
    )

    result_state = await rt._run_loop(state=state, loop_ctx=loop_ctx, driver=_CancellingDriver(),
                                      run_id="run_c2", initial_step="act", task=task, agent=agent)
    return result_state, task


@pytest.mark.asyncio
async def test_cancellation_produces_canceled_outcome_with_no_reason() -> None:
    """取消支不编造 reason（R5：payload 该是 `{}` 而不是 `{"reason": ""}`）。"""
    state, task = await run_until_cancel()
    assert state.run_outcome.kind is RunOutcomeKind.CANCELED
    assert state.run_outcome.reason == ""
    # Task 4：run 不写 task 状态了——CANCELED 由 TaskManager 据本 outcome 落
    # （对照断言见 tests/unit/test_task_manager_owns_status.py）。
    assert task.status == "ACTIVE"


# ── crash（runtime._run_loop 的泛 except Exception） ──────────────────────────


async def run_until_crash(exc: BaseException):
    """同 run_until_park 的搭台，驱动改抛 `exc`（retriable 的任意异常）。
    `_run_loop` 的泛 `except Exception` 分支 `raise run_error`——调用方拿不到
    返回值，这里返回的是抛出的那个异常本身供 `pytest.raises` 断言。"""
    from collections.abc import AsyncIterator

    from ctx_weft.core import ProviderRegistry
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.protocols import LoopConfig as RTLoopConfig
    from ctx_weft.protocols import MemoryConfig
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider as MemProv
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    session = Session(id="s_crash_2", user_prompt="crash test",
                      tenant_id="default", root_agent_id="agt_x2")
    task = Task(id="tsk_crash_2", session_id="s_crash_2", status="ACTIVE",
               tenant_id="default", assigned_agent_id="agt_x2")
    agent = Agent(id="agt_x2", session_id="s_crash_2", template_id="tpl_test",
                 tenant_id="default",
                 loop_guard=LoopGuard(), memory_config=MemoryConfig(),
                 loop_config=RTLoopConfig())
    scope = MemoryAddress(session_id="s_crash_2", task_id="tsk_crash_2", agent_id="agt_x2")
    state = LoopState(run_id="run_x2", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    class _CrashingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext,
        ) -> AsyncIterator[StepOutcome]:
            raise exc
            yield

    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(MemProv())
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider(), providers=registry,
                      event_store=None)
    rt._event_bus = bus

    loop_ctx = LoopContext(
        assembler=None, llm=None, memory=MemProv(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s_crash_2", tenant_id="default",
                                     task_id="tsk_crash_2", agent_id="agt_x2"),
        capability_cache=CapabilityCache(),
    )

    return await rt._run_loop(state=state, loop_ctx=loop_ctx, driver=_CrashingDriver(),
                              run_id="run_x2", initial_step="act", task=task, agent=agent)


@pytest.mark.asyncio
async def test_run_crash_reraises_and_outcome_is_unreachable_to_caller() -> None:
    """崩溃支 `raise run_error`——`state`（连同它上面挂的 `run_outcome`）根本不
    返回给调用方。这里断言的正是这件事本身：调用方只拿到重抛的异常，拿不到
    任何返回值。**不为了好测而改生产代码的控制流**（R6）。"""
    with pytest.raises(ValueError, match="boom"):
        await run_until_crash(ValueError("boom"))


# ── 崩溃支 outcome 构造的契约（退一步：不改控制流断不到那个 RunOutcome 本体，
#    改测 `disposition_for` 吃到「按 runtime.py 崩溃支同样方式构造」的 RunOutcome
#    后，是否按 retriable 分流）───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_crash_outcome_contract_non_retriable_exception_never_requeues() -> None:
    """`ContextOverflowError.retriable = False`——按崩溃支 `getattr(exc, "retriable",
    True)` 的同一构造方式喂给 disposition_for，即使预算充足也不该原地重试。"""
    from ctx_weft.core.models.errors import ContextOverflowError, crash_error_code
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome, disposition_for

    exc = ContextOverflowError("溢出了")
    outcome = RunOutcome(
        kind=RunOutcomeKind.INTERRUPTED, reason="run_crash",
        error_code=crash_error_code(exc), retriable=getattr(exc, "retriable", True),
    )
    d = disposition_for(outcome, retry_count=0, max_retries=3)
    assert d.status == "INTERRUPTED"
    assert d.event_type == "TaskInterrupted"


@pytest.mark.asyncio
async def test_crash_outcome_contract_default_exception_retries_with_budget() -> None:
    """没有 `retriable` 属性的普通异常按契约缺省为 `True`——预算充足时该原地重试。"""
    from ctx_weft.core.models.errors import crash_error_code
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome, disposition_for

    exc = ValueError("随便什么崩溃")
    outcome = RunOutcome(
        kind=RunOutcomeKind.INTERRUPTED, reason="run_crash",
        error_code=crash_error_code(exc), retriable=getattr(exc, "retriable", True),
    )
    d = disposition_for(outcome, retry_count=0, max_retries=3)
    assert d.status == "PENDING"
    assert d.event_type == "TaskRequeued"
