"""ObserveStep owns both modes, regardless of chain/standalone scheduling."""
import dataclasses
from types import SimpleNamespace as NS

import pytest

from ctx_weft.core.capabilities.control_tools import report_observation
from ctx_weft.core.loop.driver import ExecutionMode, StepExecution
from ctx_weft.core.loop.steps import observe as observing
from ctx_weft.core.loop.steps.observe import ObservationResult, ObserveStep, Verdict
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import IdentityFacet, LoopConfig, MemoryEvent, MemoryKind, MemoryScope
from tests.unit.test_observe_react_helper import (
    _make_ctx,
    _make_state,
    _make_tool_call_chunk,
)


def state():
    result = _make_state()
    result.agent.loop_config = LoopConfig(short_segment_token_threshold=0, max_turns_per_observe=3)
    result.task = Task(id="t1", session_id="s1", status="ACTIVE", outputs="Delivered",
                       parent_task_id="parent")
    result.extra["template"] = NS(identity={"observe": IdentityFacet(text="ROLE")})
    result.act_exit_reason = "actor_done"
    result.transcript = [NS()]
    return result


@pytest.mark.parametrize("locale", ["en", "zh"])
@pytest.mark.parametrize("boundary", ["max_turns", "context_limit", "plain_text"])
@pytest.mark.parametrize("judge_task", [False, True])
def test_turn_limit_prompt_limits_verdicts_and_shows_rounds(monkeypatch, locale, boundary, judge_task):
    from functools import partial

    from ctx_weft.core.assembler import ReviewIntent, prompts
    from ctx_weft.core.assembler.builder import PromptBuilder
    from tests.assembler_testkit import context_from_blocks
    from tests.model_testkit import model_snapshot

    monkeypatch.setattr(prompts, "render_purpose", partial(prompts.render_purpose, locale=locale))
    request = NS(resolved_model=model_snapshot(100000, 0), purpose="observe",
        intent=ReviewIntent(judge_task=judge_task),
        extra={"observe_boundary": boundary, "observe_act_turns": 37, "observe_index": 4},
        task=Task(id="t1", session_id="s1", status="ACTIVE", user_prompt_in_memory=True),
        agent=NS(id="a1"), template=None, token_counter=len,
        session=NS(context_limit=100000, reserved_output_tokens=0))
    context = context_from_blocks(request, ())
    context = type(context).from_payload(context.to_payload())
    prompt = PromptBuilder(len).compile(context, tools=[])
    text = prompt.system + "\n" + "\n".join(str(m.content) for m in prompt.messages)
    instruction = ("Allowed `task_status` values for this review are `continue` and `fail`"
                   if locale == "en" else
                   "本次 `task_status` 只能填 `continue` 或 `fail`")
    assert (instruction in text) == (judge_task and boundary == "max_turns")
    assert ("Observe review 4" in text if locale == "en" else "第 4 次 Observe 复核" in text) == judge_task
    assert ("3 previous reviews" in text if locale == "en" else "此前已进行 3 次" in text) == judge_task
    assert ("currently available capabilities" in text if locale == "en" else "当前可用能力范围" in text) == judge_task
    if judge_task:
        assert ("Do not judge failure based on the review count alone" in text if locale == "en" else
                "不要仅凭复核次数判失败" in text)
        assert ("concrete, feasible next step" in text if locale == "en" else "下一步可行的推进路径" in text)
    if judge_task and boundary == "max_turns":
        assert ("37 actor turns" if locale == "en" else "37 轮") in text
        assert "task_failure_reason" in text
        assert ("do not return `waiting` or `success`" if locale == "en" else
                "不得填 `waiting` 或 `success`") in text


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("judge_task", [False, True])
async def test_same_tool_and_purpose_with_optional_judgement(monkeypatch, standalone, judge_task):
    current = state()
    before = dataclasses.asdict(current.task)
    during_review = {**before, "observe_count": int(judge_task and not standalone)}
    calls, requests = [], []

    class Gateway:
        async def invoke(self, **kwargs):
            assert dataclasses.asdict(current.task) == during_review
            calls.append(kwargs)
            submitted = report_observation(
                act_recap="incremental", task_summary="whole task", task_status="success",
                next_step_hint="hint", task_failure_reason="ignored if recap",
            )
            return NS(content=submitted.content, metadata=submitted.metadata, is_error=False)

    ctx = _make_ctx("", gateway=Gateway())
    from ctx_weft.core.utils.clock import now_utc
    await ctx.memory.ingest(MemoryEvent(kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
        address=current.scope, role="assistant", content="Delivered", timestamp=now_utc()), ctx.provider_ctx)
    original = ctx.assembler.assemble

    async def assemble(request):
        requests.append(request)
        return await original(request)

    ctx.assembler.assemble = assemble

    async def stream(*args, **kwargs):
        yield _make_tool_call_chunk("control__report_observation")

    monkeypatch.setattr(observing, "stream_llm_resilient", stream)
    outcome = await _collect(current, ctx, judge_task=judge_task, background=standalone)
    observed = outcome.state_patch["observation"]
    assert requests[0].purpose == "observe"
    assert requests[0].intent.judge_task == judge_task
    assert len(calls) == 1
    assert observed.act_recap == "incremental"
    assert observed.task_summary == "whole task"
    assert (observed.verdict is not None) == judge_task
    assert outcome.next_step == ("finalize" if judge_task and not standalone else None)
    if standalone or not judge_task:
        assert dataclasses.asdict(current.task) == before
    else:
        assert current.task.observer_outcome == "success"
        assert current.task.process_report == "incremental"


async def test_missing_verdict_is_retried_before_accepting_submission(monkeypatch):
    current = state()
    received = []

    class Gateway:
        async def invoke(self, **kwargs):
            submitted = report_observation(
                act_recap="recap", task_status="continue" if len(received) > 1 else None)
            return NS(content=submitted.content, metadata=submitted.metadata, is_error=False)

    async def stream(ctx, state, request, **kwargs):
        received.append(request)
        yield _make_tool_call_chunk("control__report_observation")

    monkeypatch.setattr(observing, "stream_llm_resilient", stream)
    outcome = await _collect(current, _make_ctx("", gateway=Gateway()), background=True)
    assert len(received) == 2
    assert "task_status" in received[1].messages[-1].content
    assert outcome.state_patch["verdict"].task_outcome == "continue"
    assert outcome.state_patch["verdict"].reported
    assert current.task.observer_outcome is None


@pytest.mark.parametrize("standalone", [False, True])
@pytest.mark.parametrize("boundary", ["max_turns", "context_limit"])
async def test_only_turn_limit_forces_continuation(monkeypatch, standalone, boundary):
    current = state()
    current.act_exit_reason = boundary

    async def observe(*args):
        return ObservationResult("recap", "", Verdict("success", "recap", reported=True))

    async def fold(*args):
        pass

    monkeypatch.setattr(ObserveStep, "_llm_observe", observe)
    monkeypatch.setattr(ObserveStep, "_fold_retry_segment", fold)
    outcome = await _collect(current, _make_ctx(""), background=standalone)
    expected = "continue" if boundary == "max_turns" else "success"
    assert outcome.state_patch["verdict"].task_outcome == expected
    assert outcome.state_patch["observation"].act_recap == "recap"
    assert current.task.observer_outcome == (None if standalone else expected)


@pytest.mark.parametrize("verdict", ["success", "waiting", "continue", "fail", None, "error"])
@pytest.mark.parametrize("boundary", ["max_turns", "normal"])
async def test_turn_limit_is_deterministic_even_when_observation_fails(monkeypatch, verdict, boundary):
    current = state()

    async def observe(*args):
        if verdict == "error":
            raise RuntimeError("observer unavailable")
        return ObservationResult("segment recap", "whole task",
            Verdict(verdict, "segment recap", "whole task", reported=True) if verdict else None)

    monkeypatch.setattr(ObserveStep, "_llm_observe", observe)
    result = await ObserveStep()._observe(current, _make_ctx(""), [], True, boundary)
    expected = "continue" if boundary == "max_turns" and verdict != "fail" else (
        "success" if verdict in (None, "error") else verdict)
    assert result.verdict.task_outcome == expected
    if boundary == "max_turns" and verdict != "fail":
        assert result.task_summary == result.verdict.task_summary == ""
    elif verdict == "fail":
        assert result.verdict.task_summary == "whole task"


@pytest.mark.parametrize("boundary", ["plain_text", "finish_park"])
async def test_background_failure_does_not_close_a_parked_task(monkeypatch, boundary):
    current = state()
    current.task.status = "AWAITING_HUMAN"

    async def fail(*args):
        raise RuntimeError("model failed")

    monkeypatch.setattr(ObserveStep, "_llm_observe", fail)
    outcome = await _collect(current, _make_ctx(""), background=True, boundary=boundary)
    observation = outcome.state_patch["observation"]
    assert observation.verdict.task_outcome == "success"
    assert not observation.verdict.reported
    assert observation.act_recap == observation.task_summary == ""
    assert current.task.status == "AWAITING_HUMAN"
    assert current.task.observer_outcome is None


async def test_unadvertised_ask_user_is_never_dispatched(monkeypatch):
    current = state()

    class Gateway:
        async def invoke(self, **kwargs):
            pytest.fail("ask_user must not reach the gateway")

    async def stream(*args, **kwargs):
        yield _make_tool_call_chunk("control__ask_user")

    monkeypatch.setattr(observing, "stream_llm_resilient", stream)
    outcome = await _collect(current, _make_ctx("", gateway=Gateway()), judge_task=False, background=True)
    assert outcome.state_patch["observation"].verdict is None
    assert current.task.observer_outcome is None

async def _collect(current, ctx, *, background=False, judge_task=True, boundary=None):
    """Exercise observation decision logic; settlement is covered by background tests."""
    boundary = boundary or current.act_exit_reason
    ctx.execution = StepExecution(
        ExecutionMode.BACKGROUND if background else ExecutionMode.FOREGROUND,
        {"judge_task": judge_task, "boundary": boundary},
    )
    step = ObserveStep()
    if background:
        result = await step._observe(current, ctx, [], judge_task=judge_task, boundary=boundary)
        return step._outcome(result)
    return await step.execute(current, ctx)
