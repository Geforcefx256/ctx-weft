"""Real actor/observer scheduling distinguishes waiting from automatic continuation."""

import asyncio
from dataclasses import replace

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import CapabilityRef, ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime
from tests.integration.test_observe_turn_limit_status import _WorkTool
from tests.integration.test_pause_on_first_message_e2e import _NamesTasksLLM



class ScriptedVerdicts(_NamesTasksLLM):
    def __init__(self, verdicts, *, tool_first=False):
        super().__init__(responses=[], context_limit=100_000)
        self.stalling = False
        self.verdicts = iter(verdicts)
        self.actor_count = 0
        self.tool_first = tool_first

    def complete(self, request, stream=True):
        if self._is_sidecar(request):
            return super().complete(request, stream)
        if any(t.name == "control__report_observation" for t in request.tools):
            verdict = next(self.verdicts)
            return self._stream(MockResponse(tool_calls=[ToolCall(
                id="review", name="control__report_observation",
                arguments={"task_status": verdict, "act_recap": f"Reviewed {verdict}",
                           "task_summary": "Whole task" if verdict in ("success", "fail") else "",
                           "task_failure_reason": "Cannot finish" if verdict == "fail" else ""},
            )]), request)
        self.actor_count += 1
        if self.tool_first and self.actor_count == 1:
            return self._stream(MockResponse(tool_calls=[ToolCall(
                id="work", name="work__advance", arguments={})]), request)
        return self._stream(MockResponse(text=f"Actor reply {self.actor_count}"), request)


async def start(verdicts, *, tool_first=False):
    llm = ScriptedVerdicts(verdicts, tool_first=tool_first)
    template = make_echo_template()
    template.loop_config = replace(template.loop_config, short_segment_token_threshold=0)
    if tool_first:
        template.capability_refs = [CapabilityRef(capability_id="work:advance", mode="required")]
        template.loop_config = replace(template.loop_config, max_turns_per_act=1)
    provider = InlineAgentTemplateProvider()
    provider.register(template)
    rt = make_runtime(llm=llm, agent_provider=provider)
    rt.providers.register_memory(InMemoryMemoryProvider())
    if tool_first:
        rt.providers.register_capability(_WorkTool())
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="Complete the work"))
    return rt, llm, handle


async def quiescent(rt, handle):
    tm = rt._task_managers[handle.session_id]
    async with asyncio.timeout(10):
        while (tm.get_task(handle.task_id).status in ("ACTIVE", "PENDING")
               or tm._dispatch_tasks or tm._background_asyncio_tasks):
            await asyncio.sleep(0.01)
    return tm.get_task(handle.task_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("last", ["success", "fail"])
async def test_background_continue_runs_without_user_input_then_closes(last):
    rt, llm, handle = await start(["continue", last])
    task = await quiescent(rt, handle)
    assert llm.actor_count == 2
    assert task.status == ("FINISHED" if last == "success" else "FAILED")
    assert task.retry_count == 0
    assert not rt.hitl.registry.list_pending(handle.session_id)
    view = await rebuild_view(rt.event_store, handle.session_id)
    assert not view.pending_observe
    rows = await rt.event_store.read_range(handle.session_id)
    continuation, = [r for r in rows if r.event.type == EventType.OBSERVE_COMPLETED
                     and r.event.payload.get("action") == "continue"]
    requeue, = [r for r in rows if r.event.type == EventType.TASK_REQUEUED]
    assert continuation.position < requeue.position


@pytest.mark.asyncio
async def test_waiting_stops_until_user_reply_after_plain_text():
    rt, llm, handle = await start(["waiting", "success"])
    task = await quiescent(rt, handle)
    assert task.status == "AWAITING_HUMAN"
    assert task.observer_outcome == "waiting"
    assert task.error is None
    assert llm.actor_count == 1
    pending, = rt.hitl.registry.list_pending(handle.session_id)
    assert pending.task_id == task.id
    rows = await rt.event_store.read_range(handle.session_id)
    assert not any(r.event.type == EventType.TASK_REQUEUED for r in rows)
    await rt.send_message(handle.agent_id, "Here is the information")
    task = await quiescent(rt, handle)
    assert llm.actor_count == 2
    assert task.status == "FINISHED"
    assert not rt.hitl.registry.list_pending(handle.session_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("reported", ["success", "waiting", "continue"])
async def test_max_turns_always_runs_again_without_waiting(reported):
    rt, llm, handle = await start([reported, "success"], tool_first=True)
    task = await quiescent(rt, handle)
    assert task.status == "FINISHED" and llm.actor_count == 2
    assert task.retry_count == 0
    rows = await rt.event_store.read_range(handle.session_id)
    requeued, = [r for r in rows if r.event.type == EventType.TASK_REQUEUED]
    assert requeued.event.payload["outcome"] == "continue"
    completed = [r for r in rows if r.event.type == EventType.OBSERVE_COMPLETED]
    assert completed[0].event.payload["result"]["verdict"]["task_outcome"] == "continue"
    assert any(r.event.type == EventType.MAX_TURNS_REACHED for r in rows)
    assert not any(r.event.type in (EventType.HITL_OPENED, EventType.TASK_AWAITING_HUMAN)
                   and r.position < requeued.position for r in rows)
    assert not rt.hitl.registry.list_pending(handle.session_id)


@pytest.mark.asyncio
async def test_max_turns_can_fail_when_the_task_exceeds_capability():
    rt, llm, handle = await start(["fail"], tool_first=True)
    task = await quiescent(rt, handle)
    assert task.status == "FAILED" and llm.actor_count == 1
    assert task.error == "Cannot finish"
    rows = await rt.event_store.read_range(handle.session_id)
    assert any(r.event.type == EventType.MAX_TURNS_REACHED for r in rows)
    assert not any(r.event.type in (EventType.TASK_REQUEUED, EventType.TASK_AWAITING_HUMAN,
                                   EventType.HITL_OPENED) for r in rows)


@pytest.mark.asyncio
async def test_continue_then_waiting_preserves_only_the_current_wait():
    rt, llm, handle = await start(["continue", "waiting", "success"])
    task = await quiescent(rt, handle)
    assert task.status == "AWAITING_HUMAN" and llm.actor_count == 2
    pending, = rt.hitl.registry.list_pending(handle.session_id)
    rows = await rt.event_store.read_range(handle.session_id)
    opened = [r.event.payload["hitl_id"] for r in rows if r.event.type == EventType.HITL_OPENED]
    assert len(opened) == 2 and pending.id == opened[-1]
    await rt.send_message(handle.agent_id, "Proceed")
    assert (await quiescent(rt, handle)).status == "FINISHED"
    assert llm.actor_count == 3


@pytest.mark.asyncio
async def test_user_reply_during_continuation_application_does_not_duplicate_actor(monkeypatch):
    from ctx_weft.providers.memory import organize as memory_ops

    applying, release = asyncio.Event(), asyncio.Event()
    original = memory_ops.organize

    async def held_fold(*args, **kwargs):
        if kwargs.get("operation") == "fold_segment":
            applying.set()
            await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(memory_ops, "organize", held_fold)
    rt, llm, handle = await start(["continue", "success"])
    await asyncio.wait_for(applying.wait(), 5)
    reply = asyncio.create_task(rt.send_message(handle.agent_id, "Additional information"))
    try:
        await asyncio.sleep(0.03)
        assert llm.actor_count == 1, "history must settle before the next actor starts"
    finally:
        release.set()
    await reply
    assert (await quiescent(rt, handle)).status == "FINISHED"
    assert llm.actor_count == 2
    assert not rt.hitl.registry.list_pending(handle.session_id)


@pytest.mark.asyncio
async def test_cancel_during_continuation_application_prevents_requeue(monkeypatch):
    from ctx_weft.providers.memory import organize as memory_ops

    applying, release = asyncio.Event(), asyncio.Event()
    original = memory_ops.organize

    async def held_fold(*args, **kwargs):
        if kwargs.get("operation") == "fold_segment":
            applying.set()
            await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(memory_ops, "organize", held_fold)
    rt, llm, handle = await start(["continue"])
    await asyncio.wait_for(applying.wait(), 5)
    cancel = asyncio.create_task(rt.cancel_agent(handle.agent_id, reason="stop continuing"))
    await asyncio.sleep(0.03)
    release.set()
    await asyncio.wait_for(cancel, 5)
    assert (await quiescent(rt, handle)).status == "CANCELED"
    assert llm.actor_count == 1
    rows = await rt.event_store.read_range(handle.session_id)
    assert not any(r.event.type == EventType.TASK_REQUEUED for r in rows)
