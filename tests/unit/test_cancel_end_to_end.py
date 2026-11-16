"""取消这条路径的**接缝**：`_run_loop` → `execute` → `_run_task` → 处置，一次跑通。

此前取消被拆成两半各测：run 侧（`test_run_loop_cancel_guard.py`，真 `_run_loop` +
抛 `CancelledError` 的替身 driver）、TM 侧（`test_task_manager_owns_status.py`，替身
runner 直接交回 `RunOutcome(CANCELED)`）。**接缝本身**——`_run_loop` 把 CANCELED 挂进
`state.run_outcome`、`execute` 取 `s.run_outcome` 交回、`_run_task` 喂 `disposition_for`
——只有走读没有测试，而这条路径上曾判错过一次（误以为 `CancelledError` 会穿透
`except Exception` 由更外层处置）。这里用真 driver 的 step 边界检查点驱动一次完整的
`TaskManager._run_task`。

第二条（R17）：**observer 判 fail 之后、finalize 之前命中取消**，终态从 FAILED 变成
CANCELED。旧路径 `report_observation` 判 fail 时就地写 `task.status="FAILED"`，
`_run_loop` 的 `except CancelledError` 里那道 `if task.status not in (...)` 守卫挡住
覆写 → 终态 FAILED → `on_task_finished(FAILED)` → `failure_counter += 1`（可触发熔断）。
新路径 observe 只写 `task.observer_outcome`，取消时 `task.status` 仍是 ACTIVE →
`RunOutcome(CANCELED)` → 终态 CANCELED、不计失败。与 R13 同形：FinalizeStep 没跑，
`TaskFailed` 从未进事件流，投影一直停在 ACTIVE——旧行为是拿一个事件流没见过的内存状态
压掉了一次真实发生的取消。见 docs/upgrade/2026-09-02-task-status-ownership.md。
"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
from tests.model_testkit import model_snapshot

import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import ControlContext
from ctx_weft.core.control.tokens import CancelToken
from ctx_weft.core.loop.driver import LoopContext, LoopState, StepDriver, StepOutcome
from ctx_weft.core.models.agent import Agent, LoopGuard
from ctx_weft.core.models.errors import AgentNotFound
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.orchestrator.task.runner import AgentBinding
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit.observation_helpers import submit_and_apply
from tests.unit.test_runtime_agent_api import _plant


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")



class _SpyBus:
    """转发给真 bus，同时留一份底稿。TM 与 loop 共用一条总线，顺序才是真的。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.events: list = []

    async def emit_batch(self, events, *, on_committed=None):
        if on_committed is not None:
            on_committed()
        for event in events:
            await self.emit(event)

    async def emit(self, event) -> None:
        self.events.append(event)
        await self._inner.emit(event)

    def types(self) -> list:
        return [e.type for e in self.events]

    def of(self, et):
        return next(e for e in self.events if e.type == et)


class _CancelAtBoundaryStep:
    """一个 step：跑完之后置取消——取消因此**落在 step 边界**（driver.run 循环顶部）。

    `report_verdict` 为真时先真调 `report_observation(fail)`，复现 R17 的场景：
    observer 已判死，但 FinalizeStep 还没来得及跑。
    """

    name = "checkpoint"

    def __init__(self, token: CancelToken, *, report_verdict: bool = False) -> None:
        self._tok = token
        self._report = report_verdict

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        if self._report:
            submit_and_apply(
                task_status="fail",
                act_recap="tried and failed",
                task_summary="",
                task_failure_reason="broken",
                ctx=ControlContext(
                    session_id=state.session.id, task_id=state.task.id,
                    agent_id=state.agent.id, task=state.task,
                    task_manager=None, session=state.session,
                ),
            )
        self._tok.cancel()
        return StepOutcome(next_step="finalize")


class _LoopRunner:
    """两阶段 runner 的最小替身：execute 真跑 `runtime._run_loop`，交回 `s.run_outcome`。

    与生产 `_SessionTaskRunner.execute` 同形（那一份还要装配 assembler/memory/令牌注册，
    与本测试要钉的接缝无关）：**结局取自 `_run_loop` 最终返回的那个 state**。
    """

    def __init__(self, runtime: CtxWeftRuntime, session: Session, agent: Agent,
                 bus: _SpyBus, step: _CancelAtBoundaryStep, token: CancelToken) -> None:
        self._runtime = runtime
        self._session = session
        self._agent = agent
        self._bus = bus
        self._step = step
        self._token = token
        self.task: Task | None = None

    async def assemble(self, task_id: str) -> AgentBinding | None:
        return AgentBinding(agent_id=self._agent.id)

    async def execute(self, binding: AgentBinding, task_id: str):
        task = self.task
        assert task is not None
        scope = MemoryAddress(session_id=self._session.id, task_id=task.id,
                              agent_id=self._agent.id)
        state = LoopState(run_id=generate_id("run"), session=self._session, task=task,
                          agent=self._agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))
        provider_ctx = ProviderContext(session_id=self._session.id, tenant_id="default",
                                       task_id=task.id, agent_id=self._agent.id)
        ctx = LoopContext(
            assembler=None, llm=self._runtime._resolve_llm(ctx=_PROVIDER_CONTEXT),
            memory=self._runtime.providers.get_memory(), event_bus=self._bus,
            provider_ctx=provider_ctx, capability_cache=CapabilityCache(),
            cancel_token=self._token,
        )
        driver = StepDriver(steps={"checkpoint": self._step}, initial_step="checkpoint")
        s = await self._runtime._run_loop(
            state, ctx, driver, run_id=state.run_id, initial_step="checkpoint",
            task=task, agent=self._agent,
        )
        return s.run_outcome


def _wire(*, report_verdict: bool) -> tuple[TaskManager, Task, Session, _SpyBus]:
    resolver = InlineAgentTemplateProvider()
    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    bus = _SpyBus(runtime._event_bus)
    runtime._event_bus = bus       # run 域事实（RunStarted/RunCanceled/RunFinished）也走这条

    session = Session(id="s1", user_prompt="hi",  root_agent_id="agt1")
    agent = Agent(id="agt1", session_id="s1", template_id="tpl", loop_guard=LoopGuard())
    task = Task(id="A", session_id="s1", status="PENDING")

    tm = TaskManager(session_id="s1", event_bus=bus)
    tm.set_session(session)
    tm.register_task(task)
    token = CancelToken()
    runner = _LoopRunner(runtime, session, agent, bus,
                         _CancelAtBoundaryStep(token, report_verdict=report_verdict), token)
    runner.task = task
    tm.set_runner(runner)
    return tm, task, session, bus


@pytest.mark.asyncio
async def test_cancel_seam_run_loop_to_task_manager() -> None:
    """R20：真 driver 的 step 边界取消 → 一次完整的 `_run_task` → 终态 CANCELED。"""
    tm, task, _session, bus = _wire(report_verdict=False)

    await tm._run_task("A")

    assert task.status == "CANCELED"
    # payload 恰好 `{}`——取消从不编造 reason（disposition_for 的 CANCELED 支）。
    assert bus.of(EventType.TASK_CANCELED).payload == {}
    assert bus.of(EventType.TASK_CANCELED).run_id is None      # TM 在 run 外面
    # run 域的事实照发，且在 task 域之前（RunFinished 在 _run_loop 的 finally 里）。
    types = bus.types()
    assert EventType.RUN_CANCELED in types
    assert types.index(EventType.RUN_FINISHED) < types.index(EventType.TASK_CANCELED)
    # 队列动作：终态出口 → 移出 running set、不重排、会话收尾（队列空 → 终态落定）。
    # 2026-09-04（Task 12，events-v2 §5）起收尾不再额外发一条会话级 TaskQueueDrained
    # ——落定的终态直接写在 `_session.status` 上，这里改读它。
    assert "A" not in tm.running_task_ids()
    assert tm._queue.pop() is None
    # CANCELED 走的是 on_task_finished 自己的显式分支（"用户主动中断：标记 session 为
    # CANCELED"），不经 `_final_status()`——与 FINISHED/FAILED 走的收尾支同源但判据不同。
    assert task.status == "CANCELED"
    assert EventType.TASK_REQUEUED not in types


@pytest.mark.asyncio
async def test_observer_fail_then_boundary_cancel_lands_on_canceled() -> None:
    """R17：observer 判 fail 之后、finalize 之前被取消 → 终态 CANCELED，失败计数不动。"""
    tm, task, session, bus = _wire(report_verdict=True)

    await tm._run_task("A")

    # observe 侧只写判决不写状态（Task 4）——旧路径这里会写死 FAILED。
    assert task.observer_outcome == "fail"
    assert task.status == "CANCELED"
    types = bus.types()
    assert EventType.TASK_FAILED not in types       # TaskFailed 从未进事件流
    assert EventType.TASK_CANCELED in types
    assert session.failure_counter == 0             # 不计失败 → 不会触发熔断
    # 会话随之落 CANCELED（`on_task_finished` 的 CANCELED 分支）——旧路径这里是
    # `on_task_finished(FAILED)`、failure_counter +1、会话按 _final_status() 判 FAILED。
    assert not hasattr(session, "status")


# ── Task 3：`_run_loop` 分得清取消来源（总账 A3，仅剩这一条） ──────────────────


def _make_loop_fixture() -> tuple[CtxWeftRuntime, Session, Agent, Task, _SpyBus]:
    """搭一份最小的 `_run_loop` 入参：真 runtime，替身 driver 直接跑。"""
    resolver = InlineAgentTemplateProvider()
    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    bus = _SpyBus(runtime._event_bus)
    runtime._event_bus = bus

    session = Session(id="s2", user_prompt="hi",  root_agent_id="agt1")
    agent = Agent(id="agt1", session_id="s2", template_id="tpl", loop_guard=LoopGuard())
    task = Task(id="B", session_id="s2", status="ACTIVE")
    return runtime, session, agent, task, bus


async def _run_loop_with(
    runtime: CtxWeftRuntime, session: Session, agent: Agent, task: Task,
    bus: _SpyBus, driver: StepDriver, token: CancelToken | None,
) -> LoopState:
    scope = MemoryAddress(session_id=session.id, task_id=task.id, agent_id=agent.id)
    state = LoopState(run_id=generate_id("run"), session=session, task=task, agent=agent,
                      scope=scope, resolved_model=model_snapshot(model="mock", account=""))
    provider_ctx = ProviderContext(session_id=session.id, tenant_id="default",
                                   task_id=task.id, agent_id=agent.id)
    ctx = LoopContext(
        assembler=None, llm=runtime._resolve_llm(ctx=_PROVIDER_CONTEXT), memory=runtime.providers.get_memory(),
        event_bus=bus, provider_ctx=provider_ctx, capability_cache=CapabilityCache(),
        cancel_token=token,
    )
    return await runtime._run_loop(
        state, ctx, driver, run_id=state.run_id, initial_step="checkpoint",
        task=task, agent=agent,
    )


class _CancelTokenAtBoundaryStep:
    """跑完之后置取消 token——由 driver 的 step 边界检查点抛出 CancelledError。"""

    name = "checkpoint"

    def __init__(self, token: CancelToken) -> None:
        self._tok = token

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        self._tok.cancel()
        return StepOutcome(next_step="finalize")


class _ExternalCancelDriver:
    """替身 driver：`run()` 直接抛 `CancelledError`，不碰 token——模拟外部 asyncio 取消
    （进程 shutdown / `wait_for` 超时）。与 `StepDriver` 同形的 `run(state, ctx)` 接口。
    """

    async def run(self, state: LoopState, ctx: LoopContext):
        raise asyncio.CancelledError("external shutdown")
        yield  # pragma: no cover - 让本方法成为 async generator，永不执行到这里


async def _run_until_token_cancel() -> tuple[LoopState, _SpyBus]:
    runtime, session, agent, task, bus = _make_loop_fixture()
    token = CancelToken()
    driver = StepDriver(steps={"checkpoint": _CancelTokenAtBoundaryStep(token)},
                        initial_step="checkpoint")
    state = await _run_loop_with(runtime, session, agent, task, bus, driver, token)
    return state, bus


async def _run_with_external_cancelled_error() -> tuple[LoopState, _SpyBus]:
    runtime, session, agent, task, bus = _make_loop_fixture()
    driver = _ExternalCancelDriver()
    state = await _run_loop_with(runtime, session, agent, task, bus, driver, token=None)
    return state, bus


@pytest.mark.asyncio
async def test_token_cancel_is_labelled_token() -> None:
    """CancelToken 触发的取消，`RunCanceled` 的 payload 标 `source: token`。

    裁定（task-3-report.md「裁定后的实现」）：来源标签**不进** `RunOutcome.reason`
    ——那会流进 `TaskCanceled.payload`，撞上刻意定下的 R5（CANCELED 不编造 reason）。
    `RUN_CANCELED` 没有 reducer 消费它，纯增量放来源标签不破契约。
    """
    state, bus = await _run_until_token_cancel()
    assert state.run_outcome.kind is RunOutcomeKind.CANCELED
    assert state.run_outcome.reason == ""            # R5：TaskCanceled 不编造 reason
    assert bus.of(EventType.RUN_CANCELED).payload["source"] == "token"


@pytest.mark.asyncio
async def test_external_cancel_is_labelled_external() -> None:
    """非 token 来源的 CancelledError（如进程 shutdown），`RunCanceled` 标 `source: external`。"""
    state, bus = await _run_with_external_cancelled_error()
    assert state.run_outcome.kind is RunOutcomeKind.CANCELED
    assert state.run_outcome.reason == ""
    assert bus.of(EventType.RUN_CANCELED).payload["source"] == "external"


# ── 2026-09-04 spec §7.2：cancel_session 收成三步 ──────────────────────────


class _RecordingBus:
    """挂在 runtime 真实事件总线上的记录桩——不替换总线，只旁听（与
    `test_cancel_session_hitl.py` 同一手法）。"""

    def __init__(self) -> None:
        self.events: list = []

    async def _record(self, ev) -> None:
        self.events.append(ev)


async def _wire_multi_agent_session(
    rt, session_id: str, task_id: str, root_id: str, child_id: str,
) -> None:
    """搭一个「root + 子 agent 都在跑」的会话：两条 agent record（`_plant` 挂好
    parent/child 关系）+ 一个挂着的 TaskManager，让 `cancel_session` 的守卫
    （`per` 或 `task_manager` 非空）落在真实组件上，`agent_ids_of_session` 也确实能
    读到两个目标。"""
    session = Session(
        id=session_id, user_prompt="hi",
        tenant_id="default", root_agent_id=root_id, created_at=now_utc(),
    )
    tm = TaskManager(session_id=session_id, event_bus=rt._event_bus, max_concurrent=0)
    tm.set_session(session)
    task = Task(id=task_id, session_id=session_id, status="SUSPENDED", tenant_id="default")
    tm.register_task(task)
    rt._task_managers[session_id] = tm
    rt._loaded_sessions.add(session_id)
    rt._session_registry.register_session(session_id, tenant_id="default")
    _plant(rt, root_id, None, session_id=session_id, status="idle")
    _plant(rt, child_id, root_id, session_id=session_id, status="idle")


@pytest.mark.asyncio
async def test_cancel_session_terminates_every_agent() -> None:
    """回归护栏：`cancel_session` 必须把每个 agent 都终态化（2026-09-04 spec §7.2）。

    2026-09-08 生命周期改造前这里只能断言 `AgentTerminated` 事件、然后断言
    `get_agent` 抛 `AgentNotFound`——因为那时 `cancel_session` 对已空闲挂起的会话会走
    `_release_session`，把 agent record 从 registry 摘掉，最终 status 根本读不到。

    现在取消**不逐出**（回收只由显式 `forget_session` 触发），于是可以直接断言真正
    在意的那件事：record 还在，且状态是 `terminated`。事件断言一并保留——两者是不同
    层面的事实（一个是持久事件流，一个是内存现状），都该成立。
    """
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    bus = _RecordingBus()
    rt._event_bus.subscribe(None, bus._record)  # type: ignore[attr-defined]

    session_id, root_id, child_id = "s3", "root", "kid"
    await _wire_multi_agent_session(rt, session_id, "t1", root_id, child_id)

    await initialize_all(rt)
    result = await rt.cancel_session(session_id)

    assert result is True
    terminated_ids = {e.agent_id for e in bus.events if e.type == EventType.AGENT_TERMINATED}
    assert terminated_ids == {root_id, child_id}

    # 取消 ≠ 逐出：record 还在，状态是终态。
    await initialize_all(rt)
    assert (await rt.get_agent(root_id)).data.state.status == "terminated"
    await initialize_all(rt)
    assert (await rt.get_agent(child_id)).data.state.status == "terminated"
    # 默认过滤掉终态 agent（避免列表随时间膨胀），显式要才给。
    await initialize_all(rt)
    assert [item.data for item in await rt.list_agents(session_id=session_id)] == []
    await initialize_all(rt)
    assert {a.agent_id for a in [item.data for item in await rt.list_agents(session_id=session_id, include_terminated=True)]} == {
        root_id, child_id}

    # 逐出之后才真的查无此 agent。
    assert rt.forget_session(session_id) is True, "全部 terminated → 安静了，该能忘掉"
    with pytest.raises(AgentNotFound):
        await initialize_all(rt)
        (await rt.get_agent(root_id)).data


@pytest.mark.asyncio
async def test_cancel_session_keeps_agent_termination_in_cancel_agent() -> None:
    """Early run stop signals do not move agent state ownership into runtime."""
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime.cancel_session)
    assert "tokens.cancel.cancel()" not in src
    assert "await self.cancel_agent(" in src
