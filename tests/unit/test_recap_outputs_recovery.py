"""Recover final output from actor response facts at the original intent."""
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

from ctx_weft.core.capabilities.control_tools import FINISH_TASK_NAME
from ctx_weft.core.runtime import CtxWeftRuntime
from ctx_weft.protocols.events import EventOrigin, EventType


def response(text, *, run="actor", task="task", origin=EventOrigin.LOOP_ACT, summary=None):
    calls = [] if summary is None else [dict(
        id="finish", name=FINISH_TASK_NAME, arguments={"deliverables_summary": summary})]
    return NS(event=NS(type=EventType.LLM_RESPONSE_FINISHED, task_id=task, run_id=run,
                       origin=origin, payload={"content": text, "tool_calls": calls}))


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("texts,summary,expected", [
    (["answer"], None, "answer"),
    (["early", " answer ", ""], " checklist ", "answer\n\nchecklist"),
    ([""], "checklist", "checklist"),
    ([""], None, None),
])
async def test_restore_output_uses_original_actor_run_and_shared_composition(nested, texts, summary, expected):
    intent = {"observe_id": "obs"}
    rows = [response("old attempt", run="old"), response("another task", task="other")]
    rows.extend(response(text, summary=summary if i == len(texts) - 1 else None)
                for i, text in enumerate(texts))
    rows.append(response("observer text", origin=EventOrigin.LOOP_BACKGROUND_OBSERVE))
    rows.append(NS(position=42, event=NS(type=EventType.HITL_OPENED if nested else EventType.OBSERVE_REQUESTED,
                           run_id=None if nested else "actor",
                           payload={"observe_request": intent} if nested else intent)))
    rows.append(response("later reply"))
    runtime = NS(event_store=NS(read_range=AsyncMock(return_value=rows)),
                 _historical_task_view=AsyncMock(return_value=NS(tasks={})))
    task = NS(id="task", outputs=None, suspend_requested=False)
    first = await CtxWeftRuntime._restore_observe_outputs(runtime, "session", task, intent)
    assert task.outputs == expected
    second = await CtxWeftRuntime._restore_observe_outputs(runtime, "session", task, intent)
    assert task.outputs == expected and first == second
    if expected:
        assert first["final_body"] == ("answer" if any(texts) else "")


async def test_missing_intent_does_not_use_unrelated_response():
    runtime = NS(event_store=NS(read_range=AsyncMock(return_value=[response("unrelated")])))
    task = NS(id="task", outputs=None)
    with pytest.raises(RuntimeError, match="Missing observe scheduling intent"):
        await CtxWeftRuntime._restore_observe_outputs(runtime, "session", task, {"observe_id": "missing"})
    assert task.outputs is None


@pytest.mark.parametrize("reason", ["paused", "cancelled"])
async def test_usage_only_completion_is_not_restored_as_actor_output(reason):
    intent = {"observe_id": "obs"}
    aborted = response("", summary="must not become an output")
    aborted.event.payload.update(finish_reason=reason, usage={"prompt_tokens": 5})
    rows = [response("actual answer"), aborted, NS(position=42, event=NS(
        type=EventType.OBSERVE_REQUESTED, run_id="actor", payload=intent))]
    runtime = NS(event_store=NS(read_range=AsyncMock(return_value=rows)),
                 _historical_task_view=AsyncMock(return_value=NS(tasks={})))
    task = NS(id="task", outputs=None)
    extra = await CtxWeftRuntime._restore_observe_outputs(runtime, "session", task, intent)
    assert task.outputs == "actual answer"
    assert len(extra["restored_transcript"]) == 1
