"""Observe dispatch follows the shared driver; mechanical verdicts request one report."""

from __future__ import annotations

from tests.model_testkit import model_snapshot

import asyncio
import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest

from ctx_weft.core.loop.driver import LoopContext, LoopState, StepDriver
from ctx_weft.core.loop.signals import HitlPark
from ctx_weft.core.loop.steps.observe import ObserveStep
from ctx_weft.core.loop.steps.suspend import SuspendStep
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    LLMChunk,
    MemoryAddress,
    ProviderContext,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs
from tests.step_helpers import patch_observe_start


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_root_task(**kwargs) -> Task:
    """Root task: parent_task_id=None, same creator/assigned agent."""
    defaults = dict(
        id="t1",
        session_id="s1",
        status="ACTIVE",
        assigned_agent_id="ag1",
        creator_agent_id="ag1",
        parent_task_id=None,
        settings=NormalTaskSettings(),
        actor_done=False,
        observer_outcome=None,
    )
    defaults.update(kwargs)
    return Task(**defaults)


def _make_child_task(**kwargs) -> Task:
    """Child task: parent_task_id set, same agent (same_agent=True, cross_agent=False)."""
    defaults = dict(
        id="t2",
        session_id="s1",
        status="ACTIVE",
        assigned_agent_id="ag1",
        creator_agent_id="ag1",
        parent_task_id="t0",  # has a parent → is child
        settings=NormalTaskSettings(),
        actor_done=False,
        observer_outcome=None,
    )
    defaults.update(kwargs)
    return Task(**defaults)


def _make_observe_state_ctx(task: Task, act_exit_reason: str):
    """Build minimal LoopState + LoopContext for ObserveStep.execute()."""
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id=task.id, agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id=task.id, agent_id="ag1")
    session = Session(id="s1", tenant_id="default", user_prompt="hello")

    agent = SimpleNamespace(
        id="ag1",
        loop_config=SimpleNamespace(
            max_turns_per_observe=1,
            compact_keep_last=2,
            short_task_turn_cap=4,
            short_task_token_threshold=1000,
        ),
        runtime={"llm_model": "mock"},
        loop_guard=SimpleNamespace(context_limit=100_000, context_tokens=0),
    )

    state = LoopState(
        run_id="r1",
        session=session,
        task=task,
        agent=agent,
        scope=scope,
        act_exit_reason=act_exit_reason,
        extra={"template": None},
        resolved_model=model_snapshot(model="mock", account=""),
    )

    class _FakeEventBus:
        def __init__(self):
            self.emitted = []

        async def emit(self, event: Any) -> None:
            self.emitted.append(event)

    class _FakeAssembler:
        async def compile(self, inputs, **kwargs):
            return await compiled_test_inputs(inputs, **kwargs)

        async def assemble(self, request: Any) -> Any:
            return scripted_inputs(SimpleNamespace(system="SYS", messages=[], tools=[]), purpose=request.purpose)

    class _FakeLLM:
        context_limit = 100_000
        output_reserve = 4096
        tokenizer = HeuristicTokenizer()

        async def complete(self, request: Any, stream: bool = True):
            yield LLMChunk(kind="token", text="summary")

    ctx = LoopContext(
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        memory=mem,
        event_bus=_FakeEventBus(),
        provider_ctx=pctx,
    )
    return state, ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("child,boundary,report_boundary", [
    (False, "max_turns", "mechanical"),
    (False, "actor_done", "finish"),
    (True, "actor_done", "finish"),
])
async def test_mechanical_verdict_requests_one_report(monkeypatch, child, boundary, report_boundary):
    task = (_make_child_task if child else _make_root_task)(actor_done=boundary == "actor_done")
    state, ctx = _make_observe_state_ctx(task, boundary)
    state.transcript.append({"role": "assistant", "content": "done"})
    launched = []
    pending = []

    def start(self, name, state, ctx, *, options=None):
        assert name == "observe"
        requested, = ctx.event_bus.emitted
        assert requested.type == EventType.OBSERVE_REQUESTED
        launched.append(dict(options))
        pending.append(asyncio.create_task(asyncio.sleep(0)))
        return pending[-1]

    monkeypatch.setattr(StepDriver, "start", start)
    outcome = await StepDriver.for_context(ctx).execute("observe", state, ctx)
    assert outcome.next_step == "finalize"
    assert launched == []
    stages = [e for e in ctx.event_bus.emitted if e.type.startswith("Observe")]
    assert [e.type for e in stages] == [EventType.OBSERVE_REQUESTED,
        EventType.OBSERVE_STARTED, EventType.OBSERVE_COMPLETED]
    assert stages[0].payload["judge_task"] is True
    assert stages[-1].payload["result"]["verdict"]["task_outcome"] == (
        "continue" if boundary == "max_turns" else "success")
    assert outcome.state_patch["verdict"].reported is False
    assert outcome.state_patch["verdict"].act_recap == ""



def _spy_launch(monkeypatch) -> list:
    """把 driver 的后台 observe 启动换成记录器，返回 (task_id, boundary) 列表。"""
    launched: list = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id, boundary))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)
    return launched


def _with_observe_role(state) -> None:
    """给 state 装上一个有 observe facet 的 template，让 `_should_use_llm` 放行。"""
    state.extra["template"] = SimpleNamespace(identity={"observe": SimpleNamespace(text="ROLE")})


@pytest.mark.asyncio
async def test_foreground_verdict_and_report_use_one_observe(monkeypatch):
    """前台 LLM 在同一次 observe 中生成判决和报告，不另起后台调用。"""
    launched = _spy_launch(monkeypatch)
    from ctx_weft.core.loop.steps.observe import ObservationResult, Verdict

    async def fake_llm_observe(self, state, ctx, events):
        return ObservationResult("真 recap", "真 summary",
            Verdict(task_outcome="success", act_recap="真 recap",
                    task_summary="真 summary", reported=True))

    monkeypatch.setattr(ObserveStep, "_llm_observe", fake_llm_observe, raising=True)

    task = _make_root_task(actor_done=True)
    state, ctx = _make_observe_state_ctx(task, act_exit_reason="actor_done")
    _with_observe_role(state)

    outcome = await StepDriver.for_context(ctx).execute("observe", state, ctx)

    assert launched == [], f"前台已产判决，不该再发任何后台 observe；实得 {launched}"
    assert outcome.state_patch["verdict"].reported is True
    stages = [e for e in ctx.event_bus.emitted if e.type.startswith("Observe")]
    assert [e.type for e in stages] == [EventType.OBSERVE_REQUESTED, EventType.OBSERVE_STARTED, EventType.OBSERVE_COMPLETED]
    assert len({e.payload["observe_id"] for e in stages}) == 1


@pytest.mark.asyncio
async def test_failed_foreground_verdict_stays_in_the_same_observe(monkeypatch):
    """LLM failure falls back within the existing observe, without a second request."""
    launched = _spy_launch(monkeypatch)

    async def boom(self, state, ctx, events):
        raise RuntimeError("observer 掉线")

    monkeypatch.setattr(ObserveStep, "_llm_observe", boom, raising=True)

    task = _make_root_task(actor_done=True)
    state, ctx = _make_observe_state_ctx(task, act_exit_reason="actor_done")
    _with_observe_role(state)

    outcome = await StepDriver.for_context(ctx).execute("observe", state, ctx)

    assert launched == []
    stages = [event.type for event in ctx.event_bus.emitted if event.type.startswith("Observe")]
    assert stages == [EventType.OBSERVE_REQUESTED, EventType.OBSERVE_STARTED, EventType.OBSERVE_COMPLETED]
    # No valid LLM report: retain the raw segment rather than launch another observe.
    assert outcome.state_patch["verdict"].reported is False
    assert outcome.state_patch["verdict"].act_recap == ""


# ── act.py: soft-interrupt park fires for root ───────────────────────────────


@pytest.mark.asyncio
async def test_act_soft_interrupt_fires_for_root(monkeypatch):
    """act soft-interrupt park (_park_for_interrupt, root task) → background observe starts once."""
    from ctx_weft.core.assembler.assembler import AssembledPrompt
    from ctx_weft.core.control.tokens import PauseToken
    from ctx_weft.core.loop.steps.act import ActStep
    from ctx_weft.protocols import LLMMessage
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from tests.hitl_env import make_hitl

    launched = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id,))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)

    from ctx_weft.providers.events import InProcessEventBus
    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    hitl, _hitl_reg = make_hitl(bus)

    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    task = _make_root_task(status="ACTIVE")
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1")

    prompt = AssembledPrompt(
        system="", messages=[LLMMessage(role="user", content="hi")], tools=[], token_count=1,
    )
    state = LoopState(
        run_id="r1", session=session, task=task, agent=agent, scope=scope, prompt_inputs=scripted_inputs(prompt),
        resolved_model=model_snapshot(model="mock", account=""),
    )
    llm = MockLLMAdapter(responses=[MockResponse(text="partial")])
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=llm, memory=mem, event_bus=bus,
        provider_ctx=pctx, hitl=hitl,
    )

    # Fire pause before act starts
    pause = PauseToken()
    pause.pause()
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    assert len(launched) == 1, f"Expected 1 launch for root soft-interrupt, got {len(launched)}"


# ── act.py: soft-interrupt park does NOT fire for child ──────────────────────


@pytest.mark.asyncio
async def test_act_soft_interrupt_child_task_schedules_recap(monkeypatch):
    """act soft-interrupt park with child task → launch_recap NOT called."""
    from ctx_weft.core.assembler.assembler import AssembledPrompt
    from ctx_weft.core.control.tokens import PauseToken
    from ctx_weft.core.loop.steps.act import ActStep
    from ctx_weft.protocols import LLMMessage
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from tests.hitl_env import make_hitl

    launched = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id,))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)

    from ctx_weft.providers.events import InProcessEventBus
    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    hitl, _hitl_reg = make_hitl(bus)

    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    child = _make_child_task(status="ACTIVE")
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t2", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t2", agent_id="ag1")

    prompt = AssembledPrompt(
        system="", messages=[LLMMessage(role="user", content="hi")], tools=[], token_count=1,
    )
    state = LoopState(
        run_id="r1", session=session, task=child, agent=agent, scope=scope, prompt_inputs=scripted_inputs(prompt),
        resolved_model=model_snapshot(model="mock", account=""),
    )
    llm = MockLLMAdapter(responses=[MockResponse(text="partial")])
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=llm, memory=mem, event_bus=bus,
        provider_ctx=pctx, hitl=hitl,
    )

    pause = PauseToken()
    pause.pause()
    ctx.pause_token = pause

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    assert len(launched) == 1, "same-agent child interruption also needs a durable recap"


# ── act.py: plain-text pause fires for root (Task 13) ───────────────────────


@pytest.mark.asyncio
async def test_act_plain_text_pause_fires_for_root(monkeypatch):
    """act plain-text pause (_park_await_user, root task, interactive) → background observe starts once."""
    from ctx_weft.core.loop.steps.act import _finish_plain_text_turn
    from ctx_weft.providers.events import InProcessEventBus
    from tests.hitl_env import make_hitl

    launched = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id,))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)

    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    hitl, _hitl_reg = make_hitl(bus)

    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    task = dataclasses.replace(_make_root_task(status="ACTIVE"))
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1")

    state = LoopState(
        run_id="r1", session=session, task=task, agent=agent, scope=scope,
        resolved_model=model_snapshot(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=None, memory=mem, event_bus=bus,
        provider_ctx=pctx, hitl=hitl,
    )

    with pytest.raises(HitlPark):
        await _finish_plain_text_turn(state, ctx, turn_num=1, transcript=[])

    assert len(launched) == 1, f"Expected 1 launch for root plain-text pause, got {len(launched)}"


# ── act.py: plain-text pause does NOT fire for child (Task 13) ──────────────


@pytest.mark.asyncio
async def test_act_plain_text_pause_child_task_fires(monkeypatch):
    """子任务的纯文本 park **也**触发后台 observe（2026-09-22 去掉 `_is_own_root` 闸）。

    那个闸原先的理由是「非 root 走 LLM observe 向 parent 上报」——但纯文本回合压根到不了
    前台 observe（它在 park 处就退出了），于是子任务的这一段既没人判也没人折。并发不会
    因此失控：单交互线闸门保证一个 session 同时只有一条非 unattended 的线。
    """
    from ctx_weft.core.loop.steps.act import _finish_plain_text_turn
    from ctx_weft.providers.events import InProcessEventBus
    from tests.hitl_env import make_hitl

    launched = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id,))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)

    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    hitl, _hitl_reg = make_hitl(bus)

    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    child = dataclasses.replace(_make_child_task(status="ACTIVE"), unattended=False)
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t2", agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id="t2", agent_id="ag1")

    state = LoopState(
        run_id="r1", session=session, task=child, agent=agent, scope=scope,
        resolved_model=model_snapshot(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=None, memory=mem, event_bus=bus,
        provider_ctx=pctx, hitl=hitl,
    )

    with pytest.raises(HitlPark):
        await _finish_plain_text_turn(state, ctx, turn_num=1, transcript=[])

    assert [t for (t,) in launched] == ["t2"],         f"子任务的纯文本 park 应触发一次后台 observe，实得 {launched}"


# ── suspend.py: dispatch boundary (spec 2026-07-16) fires for all delegate parents ──


def _make_suspend_state_ctx(task: Task):
    """Minimal LoopState + LoopContext for SuspendStep.execute()."""
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id=task.id, agent_id="ag1")
    pctx = ProviderContext(session_id="s1", tenant_id="default", task_id=task.id, agent_id="ag1")
    session = Session(id="s1", tenant_id="default", user_prompt="hello")
    agent = SimpleNamespace(id="ag1")

    state = LoopState(
        run_id="r1", session=session, task=task, agent=agent, scope=scope,
        extra={"template": None},
        resolved_model=model_snapshot(model="mock", account=""),
    )

    class _FakeEventBus:
        def __init__(self):
            self.emitted = []

        async def emit(self, event: Any) -> None:
            self.emitted.append(event)

    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=None, memory=mem, event_bus=_FakeEventBus(),
        provider_ctx=pctx,
    )
    return state, ctx


@pytest.mark.asyncio
async def test_suspend_step_fires_dispatch_boundary_for_root(monkeypatch):
    """root task delegate suspend → launch_recap(boundary='dispatch') once."""
    launched = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id, boundary))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)

    task = _make_root_task(status="SUSPENDED")
    state, ctx = _make_suspend_state_ctx(task)

    await SuspendStep().execute(state, ctx)

    assert launched == [("t1", "dispatch")], f"Expected one dispatch launch, got {launched}"


@pytest.mark.asyncio
async def test_suspend_step_fires_dispatch_boundary_for_child(monkeypatch):
    """non-root delegate parent fires same way (spec: no _is_own_root gating)."""
    launched = []

    def fake_launch(state, ctx, *, boundary=""):
        launched.append((state.task.id, boundary))
        return asyncio.ensure_future(asyncio.sleep(0))

    patch_observe_start(monkeypatch, fake_launch)

    child = _make_child_task(status="SUSPENDED")
    state, ctx = _make_suspend_state_ctx(child)

    await SuspendStep().execute(state, ctx)

    assert launched == [("t2", "dispatch")], f"Expected one dispatch launch, got {launched}"
