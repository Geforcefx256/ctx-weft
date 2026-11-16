"""Goal switches hand execution off while the old goal receives terminal review."""
from __future__ import annotations

from tests.model_testkit import model_snapshot


import asyncio
from types import SimpleNamespace as NS

import pytest

from ctx_weft.core.assembler import ReviewIntent
from ctx_weft.core.assembler.sources.capability import build_llm_tools
from ctx_weft.core.capabilities.control_tools import (
    _CONTROL_TOOLS,
    SWITCH_TASK_NAME,
    ControlContext,
    report_observation,
    switch_task,
)
from ctx_weft.core.control.reducers import _apply
from ctx_weft.core.control.types import RunStateView
from ctx_weft.core.loop.steps.act import ActStep, _execute_tool_calls
from ctx_weft.core.loop.steps.observe import ObservationResult, ObserveStep, Verdict
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols import ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from tests._initialization import initialize_all
from tests.integration.test_interactive_task import _act_state_ctx
from tests.step_helpers import patch_observe_start
from tests.unit.test_observe_react_helper import _make_ctx, _make_state
from tests.unit.test_task_scheduling import _CapturingBus


def setup_switch():
    bus = _CapturingBus()
    tm = TaskManager(session_id="s1", event_bus=bus, max_concurrent=0)
    from tests.unit._stub_runner import StubRunner
    tm.set_runner(StubRunner(tm))
    old = Task(id="old", session_id="s1", status="ACTIVE", assigned_agent_id="a1")
    tm._tasks[old.id] = old
    ctx = ControlContext("s1", old.id, "a1", old, tm, None, "call-switch")
    return tm, old, ctx, bus


async def test_switch_stages_independent_successor_and_commits_recoverable_transition():
    tm, old, ctx, bus = setup_switch()
    result = switch_task("new", "new goal", "new user request", ctx=ctx)
    assert old.switch_requested and old.actor_done and not old.suspend_requested
    successor = tm._staged[old.id][0][0]
    assert successor.parent_task_id is None
    assert successor.assigned_agent_id == "a1"
    assert successor.settings.inherit_from_agent_id == "a1"
    assert successor.id in result.content
    status = await tm.apply_run_outcome(old.id, RunOutcome(RunOutcomeKind.TASK_SWITCHED))
    assert status == "REVIEWING" and not old.switch_review_ready.is_set()
    event = bus.events[-1]
    assert event.type == EventType.TASK_REVIEWING
    assert event.payload["successor"]["task"]["user_prompt"] == "new user request"
    # A crash here precedes flush: the durable transition still reconstructs both tasks.
    view = RunStateView(run_id="r", session_id="s1", task_id="old", agent_id="a1")
    _apply(view, event)
    assert successor.id in view.tasks
    assert view.tasks[successor.id].user_prompt == "new user request"
    await tm._flush_staged(old.id)
    await tm._settle(old.id, status)
    assert old.switch_review_ready.is_set()
    assert tm._queue.pop().task_id == successor.id
    assert not tm._held_ports()


@pytest.mark.parametrize("parent,unattended", [("parent", False), (None, True)])
def test_switch_accepts_subtasks_and_unattended_tasks(parent, unattended):
    tm, old, ctx, _ = setup_switch()
    old.parent_task_id, old.unattended = parent, unattended
    switch_task("new", "goal", "request", ctx=ctx)
    successor = tm._staged[old.id][0][0]
    assert successor.parent_task_id == parent
    assert successor.unattended == unattended
    assert old.actor_done


def test_terminal_schema_is_local_and_requires_terminal_verdict():
    cap, _ = _CONTROL_TOOLS["report_observation"]
    normal = build_llm_tools([cap], "observe")[0]
    terminal = build_llm_tools([cap], "observe", terminal_only=True)[0]
    assert terminal.input_schema["properties"]["task_status"]["enum"] == ["success", "fail"]
    assert "task_status" in terminal.input_schema["required"]
    assert normal.input_schema == cap.input_schema
    assert terminal.input_schema != normal.input_schema
    with pytest.raises(ValueError):
        ReviewIntent(judge_task=False, terminal_only=True)


@pytest.mark.parametrize("verdict", ["success", "fail"])
async def test_switched_review_settles_both_terminal_results_without_failure_threshold(verdict):
    tm, old, _, bus = setup_switch()
    tm.set_session(Session(id="s1", tenant_id="default",  user_prompt="old"))
    old.status = "REVIEWING"
    new = Task(id="new", session_id="s1", status="ACTIVE", assigned_agent_id="a1")
    tm._tasks[new.id] = new
    tm._running_tasks.add(new.id)
    closed = []

    async def finalize():
        closed.append(old.status)

    accepted = await tm.apply_out_of_band_verdict(
        old.id, RunOutcome(RunOutcomeKind.COMPLETED, verdict=verdict, error="old goal unfinished"),
        finalize=finalize,
    )
    assert accepted and old.status == ("FINISHED" if verdict == "success" else "FAILED")
    assert closed == [old.status]
    assert new.status == "ACTIVE" and new.id in tm._running_tasks
    assert tm.session.failure_counter == 0
    assert any(e.payload.get("reason") == "user_topic_switch" for e in bus.events)
    assert not await tm.apply_out_of_band_verdict(old.id, RunOutcome(RunOutcomeKind.COMPLETED, verdict=verdict))


@pytest.mark.parametrize("verdict", ["continue", "waiting"])
async def test_nonterminal_verdict_cannot_resume_a_switched_task(verdict):
    tm, old, _, _ = setup_switch()
    old.status = "REVIEWING"
    assert not await tm.apply_out_of_band_verdict(old.id, RunOutcome(RunOutcomeKind.COMPLETED, verdict=verdict))
    assert old.status == "REVIEWING"
    assert tm.requeue_resumable() == []
    assert ObserveStep()._submission_error(
        report_observation("recap", verdict), terminal_only=True,
    )


async def test_terminal_review_failure_has_terminal_fallback(monkeypatch):
    state, ctx = _make_state(), _make_ctx("")
    state.task = Task(id="t1", session_id="s1", status="REVIEWING")

    async def failed(*args):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(ObserveStep, "_llm_observe", failed)
    result = await ObserveStep()._observe(state, ctx, [], True, "task_switch")
    assert result.verdict.task_outcome == "success"
    assert not result.act_recap and not result.verdict.failure_reason


def test_terminal_success_does_not_require_closing_prose():
    state = _make_state()
    state.task = Task(id="t1", session_id="s1", status="REVIEWING")
    result = ObserveStep()._interpret_submission(
        state, report_observation("did the work", "success"), terminal_only=True,
    )
    assert result.verdict.task_outcome == "success"


async def test_act_switch_exits_without_finish_park_or_output_synthesis(monkeypatch):
    launched = []

    def launch(state, ctx, *, boundary=""):
        launched.append(boundary)
        return asyncio.create_task(asyncio.sleep(0))

    patch_observe_start(monkeypatch, launch)

    class Gateway:
        async def invoke(self, **kwargs):
            kwargs["state"].task.switch_requested = True
            kwargs["state"].task.actor_done = True
            return NS(content="switched", is_error=False)

    llm = MockLLMAdapter(responses=[MockResponse(
        text="Switching to your new request.",
        tool_calls=[ToolCall(id="sw", name=SWITCH_TASK_NAME, arguments={})],
    )])
    state, ctx, old, hitl, _ = _act_state_ctx(False, llm, gateway=Gateway())
    outcome = await ActStep().execute(state, ctx)
    assert outcome.next_step is None
    assert outcome.state_patch["run_outcome"].kind == RunOutcomeKind.TASK_SWITCHED
    assert old.outputs is None and not hitl.list_pending("s1")
    assert launched == ["task_switch"]


async def test_conflicting_batch_executes_no_tools(monkeypatch):
    from ctx_weft.core.loop.steps import act
    state, ctx = _make_state(), _make_ctx("")
    paired = []

    async def record(state, ctx, tc, content, **kwargs):
        paired.append(tc.id)

    monkeypatch.setattr(act, "_ingest_synthetic_tool_result", record)
    calls = [ToolCall(id="switch", name=SWITCH_TASK_NAME, arguments={}),
             ToolCall(id="ask", name="control__ask_user", arguments={})]
    result = await _execute_tool_calls(state, ctx, calls)
    assert paired == ["switch", "ask"]
    assert all(row["is_error"] for row in result)


@pytest.mark.parametrize("old_verdict", ["success", "fail"])
async def test_runtime_new_goal_runs_before_old_review_and_late_review_keeps_agent_running(monkeypatch, old_verdict):
    from ctx_weft.core.runtime import SessionStartParams
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    review_started, release_review = asyncio.Event(), asyncio.Event()
    new_started, release_new = asyncio.Event(), asyncio.Event()
    old_closed = asyncio.Event()

    class Router(MockLLMAdapter):
        def __init__(self):
            super().__init__(responses=[], observer_verdict="success")
            self.acts = 0

        def complete(self, request, stream=True):
            names = {t.name for t in request.tools or []}
            if "control__update_task_metadata" in names:
                response = MockResponse(tool_calls=[ToolCall(id="meta", name="control__update_task_metadata", arguments={"title": "old", "description": "old goal"})])
            elif "control__report_observation" in names:
                return super().complete(request, stream=stream)
            else:
                self.acts += 1
                if self.acts == 1:
                    response = MockResponse(tool_calls=[ToolCall(id="sw", name=SWITCH_TASK_NAME, arguments={"title": "new", "description": "new goal", "task_prompt": "new request"})])
                else:
                    return self.new_goal(request)
            return self._stream(response, request)

        async def new_goal(self, request):
            new_started.set()
            await release_new.wait()
            async for chunk in self._stream(MockResponse(text="New goal delivered.", tool_calls=[]), request):
                yield chunk

    original = ObserveStep._llm_observe

    async def review(self, state, ctx, events):
        if ctx.execution.options.get("boundary") == "task_switch":
            review_started.set()
            await release_review.wait()
            return ObservationResult("Worked on old goal.", "Old goal review.",
                                     Verdict(old_verdict, "Worked on old goal.", "Old goal review.",
                                             reported=True, failure_reason="Old goal unfinished after user switch." if old_verdict == "fail" else ""))
        return await original(self, state, ctx, events)

    monkeypatch.setattr(ObserveStep, "_llm_observe", review)
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    rt = make_runtime(llm=Router(), agent_provider=provider)
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    rt.providers.register_memory(InMemoryMemoryProvider())
    events = []

    async def record(event):
        events.append(event)
        if event.type in (EventType.TASK_FINISHED, EventType.TASK_FAILED) and event.payload.get("reason") == "user_topic_switch":
            old_closed.set()

    rt.event_bus.subscribe(None, record)
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(template_id="agent:tpl_echo", user_prompt="switch to new request"))
    try:
        await asyncio.wait_for(new_started.wait(), 5)
        await asyncio.wait_for(review_started.wait(), 5)
        tm = rt._task_managers[handle.session_id]
        old = tm.get_task(handle.task_id)
        assert old.status == "REVIEWING"
        successor = next(t for t in tm.all_tasks() if t.id != old.id)
        assert successor.status == "ACTIVE"
        release_review.set()
        await asyncio.wait_for(old_closed.wait(), 5)
        await initialize_all(rt)
        agent = (await rt.get_agent(handle.agent_id)).data
        assert agent.state.current_task.task_id == successor.id
        assert agent.state.status == "running"
        assert old.status == ("FINISHED" if old_verdict == "success" else "FAILED")
        release_new.set()
        for _ in range(100):
            if successor.status == "FINISHED":
                break
            await asyncio.sleep(0.01)
        assert successor.status == "FINISHED"
        await asyncio.wait_for(asyncio.gather(*list(tm._background_asyncio_tasks)), 5)
        if tm._session_done_waiter is not None:
            await asyncio.wait_for(tm._session_done_waiter, 5)
        assert not any(e.type == EventType.TASK_AWAITING_HUMAN and e.task_id == old.id for e in events)
    finally:
        release_review.set()
        release_new.set()

@pytest.mark.parametrize("locale", ["en", "zh"])
def test_switch_review_prompt_uses_terminal_contract(locale, monkeypatch):
    from functools import partial

    from ctx_weft.core.assembler import prompts
    from ctx_weft.core.assembler.builder import PromptBuilder
    from tests.assembler_testkit import context_from_blocks
    monkeypatch.setattr(prompts, "render_purpose", partial(prompts.render_purpose, locale=locale))
    request = NS(resolved_model=model_snapshot(100000, 0),
        purpose="observe", intent=ReviewIntent(terminal_only=True),
        extra={"observe_boundary": "task_switch", "locale": locale},
        task=Task(id="t1", session_id="s1", status="REVIEWING", user_prompt_in_memory=True),
        agent=NS(id="a1"), template=None, token_counter=len,
        session=NS(context_limit=100_000, reserved_output_tokens=0),
    )
    prompt = PromptBuilder(len).compile(context_from_blocks(request, ()), tools=[])
    text = prompt.system + "\n" + "\n".join(str(m.content) for m in prompt.messages)
    assert "success" in text and "fail" in text
    assert "task_failure_reason" in text
    assert "next_step_hint" in text


async def test_recovery_keeps_old_task_in_review_and_only_enqueues_successor():
    from dataclasses import replace

    from ctx_weft.core.control.converters import task_from_projection
    from ctx_weft.core.orchestrator.task.manager import task_payload
    tm, old, ctx, bus = setup_switch()
    switch_task("new", "new goal", "new request", ctx=ctx)
    old_payload = task_payload(old, user_prompt_jsonable="old request")
    await tm.apply_run_outcome(old.id, RunOutcome(RunOutcomeKind.TASK_SWITCHED))
    created = replace(bus.events[-1], type=EventType.TASK_CREATED, payload=old_payload)
    view = RunStateView(run_id="r", session_id="s1", task_id=old.id, agent_id="a1")
    _apply(view, created)
    _apply(view, bus.events[-1])
    recovered = TaskManager(session_id="s1")
    recovered.restore([task_from_projection(t) for t in view.tasks.values()], set())
    assert recovered.get_task(old.id).status == "REVIEWING"
    successor = recovered._queue.pop()
    assert successor.task_id != old.id
    assert recovered._queue.pop() is None

@pytest.mark.parametrize("target_id", ["sibling", "unrelated", "missing"])
def test_switch_target_must_be_current_or_ancestor(target_id):
    tm, old, ctx, _ = setup_switch()
    for tid in ("sibling", "unrelated"):
        tm._tasks[tid] = Task(id=tid, session_id="s1", status="PENDING")
    with pytest.raises(ValueError, match="current task or one of its ancestors"):
        switch_task("new", "goal", "request", target_task_id=target_id, reason="new instruction", ctx=ctx)
    assert not tm._staged


async def test_ancestor_switch_rewires_parent_dependencies_and_survives_replay():
    from dataclasses import replace

    from ctx_weft.core.control.converters import task_from_projection
    from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
    from ctx_weft.core.orchestrator.task.manager import task_payload
    from ctx_weft.core.orchestrator.task.queue import QueueEntry

    tm, caller, ctx, bus = setup_switch()
    parent = Task(id="parent", session_id="s1", status="SUSPENDED", assigned_agent_id="root-agent")
    branch = Task(id="branch", session_id="s1", status="SUSPENDED", parent_task_id=parent.id,
                  creator_agent_id="root-agent", assigned_agent_id="branch-agent", origin_tool_call_id="original-dispatch",
                  origin_tool_name="control__delegate_task", unattended=True)
    caller.parent_task_id = branch.id
    sibling = Task(id="sibling", session_id="s1", status="ACTIVE", parent_task_id=branch.id, assigned_agent_id="other-agent")
    follow = Task(id="follow", session_id="s1", status="PENDING", parent_task_id=parent.id, dag_deps=[branch.id])
    for task in (parent, branch, sibling, follow):
        tm._tasks[task.id] = task
    for task in (branch, caller, sibling, follow):
        tm._parent_map[task.id] = task.parent_task_id
        tm._children_of.setdefault(task.parent_task_id, set()).add(task.id)
    tm._queue.push(QueueEntry(task_id=follow.id, session_id="s1", blocked_by={branch.id}))
    tm._running_tasks.update((caller.id, sibling.id))
    cancelled, reviewed = [], []

    async def review(task):
        reviewed.append(task.id)

    tm.set_hooks(TaskManagerHooks(cancel_inflight=lambda tid: cancelled.append(tid) or True, review_switched_task=review))
    with pytest.raises(ValueError, match="reason"):
        switch_task("new", "goal", "request", target_task_id=branch.id, ctx=ctx)
    initial = {tid: task_payload(t, user_prompt_jsonable=t.user_prompt) for tid, t in tm._tasks.items()}
    switch_task("new", "goal", "request", target_task_id=branch.id, reason="replace price branch", ctx=ctx)
    successor = tm._staged[caller.id][0][0]
    assert successor.parent_task_id == parent.id
    assert successor.assigned_agent_id == branch.assigned_agent_id
    assert successor.unattended
    assert successor.origin_tool_call_id == "original-dispatch"
    await tm.apply_run_outcome(caller.id, RunOutcome(RunOutcomeKind.TASK_SWITCHED))
    assert cancelled == [sibling.id]
    assert branch.id in reviewed and sibling.id not in reviewed
    assert all(tm.get_task(tid).status == "REVIEWING" for tid in (caller.id, sibling.id, branch.id))
    assert tm.children_of(parent.id) == {successor.id, follow.id}
    assert follow.dag_deps == [successor.id]
    assert tm._queue.peek_all()[0].blocked_by == {successor.id}
    # A late canceled outcome must not cancel the old branch task.
    assert await tm.apply_run_outcome(sibling.id, RunOutcome(RunOutcomeKind.CANCELED)) == "REVIEWING"
    await tm._settle(sibling.id, "REVIEWING")
    assert sibling.id in reviewed
    view = RunStateView(run_id="r", session_id="s1", task_id=caller.id, agent_id="a1")
    boundary = next(e for e in bus.events if e.payload.get("replaced_task_ids"))
    for tid, payload in initial.items():
        _apply(view, replace(boundary, type=EventType.TASK_CREATED, task_id=tid, payload=payload))
    _apply(view, boundary)
    from ctx_weft.core.control.reducers import deserialize_view, serialize_view
    view = deserialize_view(serialize_view(view))
    recovered = TaskManager(session_id="s1")
    recovered.restore([task_from_projection(t) for t in view.tasks.values()], set(), enqueue=False)
    assert recovered.children_of(parent.id) == {successor.id, follow.id}
    assert recovered.get_task(follow.id).dag_deps == [successor.id]
    assert recovered.get_task(caller.id).switched_to_task_id == successor.id
    await tm._flush_staged(caller.id)
    await tm._try_resume_parent(successor.id)
    assert parent.status == "SUSPENDED"


@pytest.mark.parametrize("use_subagent", [False, True])
@pytest.mark.parametrize("replace_ancestor", [False, True])
async def test_subtask_replacement_completes_parent_before_old_review(monkeypatch, use_subagent, replace_ancestor):
    from ctx_weft.core.runtime import SessionStartParams
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    release_old = asyncio.Event()
    replacement_done = asyncio.Event()
    root_id = ""
    original = ObserveStep._llm_observe

    async def review(self, state, ctx, events):
        if ctx.execution.options.get("boundary") == "task_switch":
            await release_old.wait()
            return ObservationResult("Old subgoal stopped.", "Replaced by the new goal.",
                                     Verdict("fail", "Old subgoal stopped.", "Replaced by the new goal.",
                                             reported=True, failure_reason="User replaced the subgoal."))
        return await original(self, state, ctx, events)

    monkeypatch.setattr(ObserveStep, "_llm_observe", review)

    class Router(MockLLMAdapter):
        def __init__(self):
            super().__init__(responses=[], observer_verdict="success")
            self.acts = 0

        def complete(self, request, stream=True):
            names = {t.name for t in request.tools or []}
            if "control__report_observation" in names:
                return super().complete(request, stream=stream)
            if "control__update_task_metadata" in names:
                response = MockResponse(tool_calls=[ToolCall(id="meta", name="control__update_task_metadata", arguments={"title": "parent", "description": "parent goal"})])
            else:
                self.acts += 1
                if self.acts == 1:
                    response = MockResponse(tool_calls=[ToolCall(id="delegate", name="control__delegate_task", arguments={"title": "old child", "description": "old subgoal", "use_subagent": use_subagent})])
                elif self.acts == 2:
                    response = MockResponse(tool_calls=[ToolCall(id="switch", name=SWITCH_TASK_NAME, arguments={"title": "replacement child", "description": "new subgoal", "task_prompt": "new request", "target_task_id": root_id if replace_ancestor else "", "reason": "Replace the whole root goal" if replace_ancestor else ""})])
                else:
                    response = MockResponse(text="Goal delivered.", tool_calls=[])
            return self._stream(response, request)

    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    rt = make_runtime(llm=Router(), agent_provider=provider)
    rt.providers.register_memory(InMemoryMemoryProvider())

    async def record(event):
        nonlocal root_id
        if event.type == EventType.TASK_CREATED and not root_id:
            root_id = event.task_id
        if event.type == EventType.TASK_FINISHED:
            replacement_done.set()

    rt.event_bus.subscribe(None, record)
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(template_id="agent:tpl_echo", user_prompt="parent goal"))
    try:
        await asyncio.wait_for(replacement_done.wait(), 5)
        tm = rt._task_managers[handle.session_id]
        old = next(t for t in tm.all_tasks() if t.switched_to_task_id and t.parent_task_id == handle.task_id)
        replacement = tm.get_task(old.switched_to_task_id)
        assert replacement.status == "FINISHED"
        assert old.status == "REVIEWING"
        assert tm.children_of(handle.task_id) == (set() if replace_ancestor else {replacement.id})
        for _ in range(100):
            if tm.get_task(handle.task_id).status == ("REVIEWING" if replace_ancestor else "FINISHED"):
                break
            await asyncio.sleep(0.01)
        assert tm.get_task(handle.task_id).status == ("REVIEWING" if replace_ancestor else "FINISHED")
        release_old.set()
        await asyncio.wait_for(asyncio.gather(*list(tm._background_asyncio_tasks)), 5)
        assert old.status == "FAILED"
    finally:
        release_old.set()
