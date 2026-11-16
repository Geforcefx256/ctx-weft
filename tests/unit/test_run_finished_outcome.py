"""RunFinished 说的是 run 自己的结局，不是 task 的状态（Task 3）。

搭台手法：不用 brief 骨架里假想的 `outage_case` / `park_case` / `normal_case`
fixture（仓里不存在），改用与 Task 2 的 `test_loop_reports_outcome.py` 同源的
既有手法——`test_run_loop_outage.py`（`run_single_task` + spy `emit`）与
`test_hitl_park.py`（`_run_loop` + stub driver 直调 + `bus.subscribe`）。

修复轮 1（裁定 R7）：崩溃支 / 取消支的 `RunFinished.outcome` 也在此文件补真断言。
崩溃支此前只做过静态走读，理由是"调用方拿不到 `state` 返回值"——但
`RunFinished` 是在 `finally` 块里发的（`runtime.py:2587`），`finally` 先于
`raise run_error`（`runtime.py:2600`）执行，事件在异常抛出前已经进了 bus。
用 `bus.subscribe` 收集事件、配合 `pytest.raises` 断言重抛，两者不冲突。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.protocols.events import EventType
from tests._initialization import initialize_all



async def _events_from_outage() -> list:
    """同 test_run_loop_outage.py::test_outage_marks_session_interrupted_not_failed 的搭台。"""
    from ctx_weft.core.models.config import RuntimeConfig
    from ctx_weft.protocols import LLMOutageError
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
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
    runtime = make_runtime(
        llm=_OutageLLM(responses=[]),
        agent_provider=resolver,
        config=RuntimeConfig(llm_self_heal_max_attempts=1),
    )
    runtime.providers.register_memory(InMemoryMemoryProvider())

    seen: list = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    await initialize_all(runtime)
    await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="hi")
    return seen


async def _events_from_park() -> list:
    """同 test_hitl_park.py::test_run_loop_catches_park_returns_awaiting_human 的搭台。"""
    from collections.abc import AsyncIterator

    from ctx_weft.core import ProviderRegistry
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.loop.signals import HitlPark
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task
    from ctx_weft.protocols import LoopConfig, MemoryAddress, MemoryConfig, ProviderContext
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    session = Session(id="s_park_3", user_prompt="park test",
                      tenant_id="default", root_agent_id="agt_p3")
    task = Task(id="tsk_park_3", session_id="s_park_3", status="ACTIVE",
               tenant_id="default", assigned_agent_id="agt_p3")
    agent = Agent(id="agt_p3", session_id="s_park_3", template_id="tpl_test",
                 tenant_id="default",
                 loop_guard=LoopGuard(), memory_config=MemoryConfig(),
                 loop_config=LoopConfig())
    scope = MemoryAddress(session_id="s_park_3", task_id="tsk_park_3", agent_id="agt_p3")
    state = LoopState(run_id="run_p3", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    class _ParkingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext,
        ) -> AsyncIterator[StepOutcome]:
            raise HitlPark(hitl_id="req_park_3", tool_call_id="tc_park_3")
            yield

    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(InMemoryMemoryProvider())

    rt = make_runtime(agent_provider=InlineAgentTemplateProvider(), providers=registry,
                      event_store=None)
    rt._event_bus = bus

    loop_ctx = LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s_park_3", tenant_id="default",
                                     task_id="tsk_park_3", agent_id="agt_p3"),
        capability_cache=CapabilityCache(),
    )

    collected: list = []

    async def _handler(ev) -> None:
        collected.append(ev)

    bus.subscribe(None, _handler)

    await rt._run_loop(state=state, loop_ctx=loop_ctx, driver=_ParkingDriver(),
                       run_id="run_p3", initial_step="act", task=task, agent=agent)
    return collected


async def _events_from_normal() -> list:
    """同 test_minimal_loop.py::test_minimal_echo_loop 的搭台，加 spy 收事件。"""
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="Hello! You said: say hello")])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    seen: list = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    await initialize_all(runtime)
    await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="say hello")
    return seen


async def _events_from_cancel() -> list:
    """同 test_loop_reports_outcome.py::run_until_cancel 的搭台，加 bus.subscribe 收事件。
    取消支不重抛（run_error 保持 None），`_run_loop` 正常 return，不需要 pytest.raises。"""
    from collections.abc import AsyncIterator

    from ctx_weft.core import ProviderRegistry
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task
    from ctx_weft.protocols import LoopConfig, MemoryAddress, MemoryConfig, ProviderContext
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    session = Session(id="s_cancel_3", user_prompt="cancel test",
                      tenant_id="default", root_agent_id="agt_c3")
    task = Task(id="tsk_cancel_3", session_id="s_cancel_3", status="ACTIVE",
               tenant_id="default", assigned_agent_id="agt_c3")
    agent = Agent(id="agt_c3", session_id="s_cancel_3", template_id="tpl_test",
                 tenant_id="default",
                 loop_guard=LoopGuard(), memory_config=MemoryConfig(),
                 loop_config=LoopConfig())
    scope = MemoryAddress(session_id="s_cancel_3", task_id="tsk_cancel_3", agent_id="agt_c3")
    state = LoopState(run_id="run_c3", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    class _CancellingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext,
        ) -> AsyncIterator[StepOutcome]:
            raise asyncio.CancelledError
            yield

    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(InMemoryMemoryProvider())
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider(), providers=registry,
                      event_store=None)
    rt._event_bus = bus

    loop_ctx = LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s_cancel_3", tenant_id="default",
                                     task_id="tsk_cancel_3", agent_id="agt_c3"),
        capability_cache=CapabilityCache(),
    )

    collected: list = []

    async def _handler(ev) -> None:
        collected.append(ev)

    bus.subscribe(None, _handler)

    await rt._run_loop(state=state, loop_ctx=loop_ctx, driver=_CancellingDriver(),
                       run_id="run_c3", initial_step="act", task=task, agent=agent)
    return collected


async def _events_from_crash(exc: BaseException) -> tuple[list, BaseException]:
    """同 test_loop_reports_outcome.py::run_until_crash 的搭台，加 bus.subscribe 收事件，
    并用 pytest.raises 捕获重抛的 `exc`——`finally` 块（含 RUN_FINISHED emit）先于
    `raise run_error` 执行（runtime.py:2587 vs :2600），两者不冲突（裁定 R7）。"""
    from collections.abc import AsyncIterator

    from ctx_weft.core import ProviderRegistry
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.loop.driver import LoopContext, LoopState, StepOutcome
    from ctx_weft.core.models.agent import Agent, LoopGuard
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task
    from ctx_weft.protocols import LoopConfig, MemoryAddress, MemoryConfig, ProviderContext
    from ctx_weft.providers.events import InProcessEventBus
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    session = Session(id="s_crash_3", user_prompt="crash test",
                      tenant_id="default", root_agent_id="agt_x3")
    task = Task(id="tsk_crash_3", session_id="s_crash_3", status="ACTIVE",
               tenant_id="default", assigned_agent_id="agt_x3")
    agent = Agent(id="agt_x3", session_id="s_crash_3", template_id="tpl_test",
                 tenant_id="default",
                 loop_guard=LoopGuard(), memory_config=MemoryConfig(),
                 loop_config=LoopConfig())
    scope = MemoryAddress(session_id="s_crash_3", task_id="tsk_crash_3", agent_id="agt_x3")
    state = LoopState(run_id="run_x3", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))

    class _CrashingDriver:
        async def run(
            self, initial_state: LoopState, ctx: LoopContext,
        ) -> AsyncIterator[StepOutcome]:
            raise exc
            yield

    bus = InProcessEventBus()
    registry = ProviderRegistry()
    registry.register_memory(InMemoryMemoryProvider())
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider(), providers=registry,
                      event_store=None)
    rt._event_bus = bus

    loop_ctx = LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s_crash_3", tenant_id="default",
                                     task_id="tsk_crash_3", agent_id="agt_x3"),
        capability_cache=CapabilityCache(),
    )

    collected: list = []

    async def _handler(ev) -> None:
        collected.append(ev)

    bus.subscribe(None, _handler)

    caught: BaseException | None = None
    try:
        await rt._run_loop(state=state, loop_ctx=loop_ctx, driver=_CrashingDriver(),
                           run_id="run_x3", initial_step="act", task=task, agent=agent)
    except type(exc) as e:  # noqa: BLE001 - 就是要拿住重抛的那一个异常
        caught = e
    return collected, caught


@pytest.mark.asyncio
async def test_cancel_run_finished_outcome_is_canceled() -> None:
    """取消支（`except asyncio.CancelledError`）同样走 `finally`，`outcome` 应为 canceled。"""
    events = await _events_from_cancel()
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert ev.payload["outcome"] == "canceled"


@pytest.mark.asyncio
async def test_crash_run_finished_outcome_is_interrupted() -> None:
    """裁定 R7：崩溃支重抛前，`RunFinished` 已在 `finally` 里发到 bus——用
    `bus.subscribe` 收事件，配合 `pytest.raises` 风格的显式捕获断言重抛，两者
    互不干扰（订阅回调与 raise 走的是两条独立路径，不需要 spy `emit`）。"""
    exc = ValueError("boom")
    events, caught = await _events_from_crash(exc)
    assert caught is exc
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert ev.payload["outcome"] == "interrupted"


@pytest.mark.asyncio
async def test_run_finished_carries_the_run_outcome_not_task_status() -> None:
    events = await _events_from_outage()
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert ev.payload["outcome"] == "interrupted"


@pytest.mark.asyncio
async def test_park_run_reports_awaiting_human() -> None:
    events = await _events_from_park()
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert ev.payload["outcome"] == "awaiting_human"


@pytest.mark.asyncio
async def test_normal_run_reports_completed() -> None:
    events = await _events_from_normal()
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert ev.payload["outcome"] == "completed"


@pytest.mark.asyncio
async def test_will_retry_is_unchanged() -> None:
    """will_retry 是给 host 决定关不关流的，语义不变。"""
    events = await _events_from_normal()
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert "will_retry" in ev.payload


@pytest.mark.asyncio
async def test_final_status_key_still_present_but_deprecated() -> None:
    """`final_status` 保留一个发布周期，值是**发 RUN_FINISHED 那一刻**的 task.status。

    Task 4 起 run 不写 task 状态，这个值因此多半是 ACTIVE——它已经不是 task 的终态
    （终态由随后 TaskManager 的处置写定），正是 host 必须改读 `outcome` 的原因。
    """
    events = await _events_from_park()
    ev = next(e for e in events if e.type == EventType.RUN_FINISHED)
    assert ev.payload["final_status"] == "ACTIVE"
