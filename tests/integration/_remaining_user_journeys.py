
from tests._recovery import recover_fixture_session
"""Real user journeys; only external model, tool, and storage I/O are controlled."""
import asyncio
import base64
import copy
import json
import os
import re
from itertools import pairwise

from ctx_weft import RuntimeConfig, SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import ImagePart, LLMChunk, TextPart, ToolCall
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.authorizer import HumanConfirmationAuthorizer
from ctx_weft.providers.blob.fs import FsBlobStore
from ctx_weft.providers.llm.adapters.mock import MockResponse
from tests.integration._session_cancel_worker import wait_until
from tests.integration._user_control_journey import (
    SID,
    drained,
    memory_state,
    request_text,
    save,
)
from tests.integration.test_dispatch_boundary_recap_e2e import _is_background_recap
from tests.integration.test_hitl_e2e_v2 import _RecordingBashTool
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_multimodal_end_to_end import (
    _image_sources,
    _WireCapturingAnthropicAdapter,
)
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM

PROJECT = "PROJECT=CEDAR"
APPROVED_ARGS = {"command": "list release artifacts"}
WITHDRAWN = b"withdrawn screenshot bytes"
SCREENSHOT = b"accepted screenshot bytes"


def evidence(text):
    return " ".join(sorted(set(re.findall(
        r"(?:PROJECT|PART_ONE|PART_TWO|PART_THREE|ANSWER_A|ANSWER_B|SCREENSHOT_RESULT)=\w+", text))))


def plan_evidence(failed):
    return (["PART_ONE=verified", "PART_TWO=unavailable"] if failed else
            ["PART_ONE=verified", "PART_TWO=approved", "PART_THREE=ready"])


class JourneyLLM(_FinishEveryTaskLLM):
    def __init__(self, scenario, stage, variant):
        super().__init__()
        self.scenario, self.stage, self.variant = scenario, stage, variant
        self.calls = []
        self.actors = 0
        self.failed_review = False
        self.withdrawn_started, self.withdrawn_closed = asyncio.Event(), asyncio.Event()
        self.wire = _WireCapturingAnthropicAdapter()
        self.actor_payloads = []

    def response(self, request, text="", name=None, arguments=None):
        return self._stream(MockResponse(text=text, tool_calls=([ToolCall(
            id=self._id("journey"), name=name, arguments=arguments or {})] if name else [])), request)

    def question(self, request, text):
        return self.response(request, name="control__ask_user",
                             arguments={"questions": [{"question": text}]})

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        text = request_text(request)
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__report_observation" in names:
            kind = "report" if _is_background_recap(request) else "judge"
            self.calls.append((kind, request))
            outcome = "success"
            if self.scenario == "plan" and kind == "judge" and self.failed_review:
                outcome, self.failed_review = "fail", False
            if self.scenario == "image" and "SCREENSHOT_ACCEPTED" not in text:
                outcome = "waiting"
            return self.response(request, name="control__report_observation", arguments={
                "task_status": outcome,
                "act_recap": "Evidence checked. " + evidence(text),
                "task_summary": "Work summary. " + evidence(text),
                "task_failure_reason": "Step two could not validate the unavailable artifact." if outcome == "fail" else "",
            })
        self.calls.append(("actor", request))
        self.actors += 1
        if self.scenario == "image":
            self.actor_payloads.append(self.wire._build_payload(request))
        # Following tasks must read the actual accumulated context, rather than
        # receiving a fixed answer containing evidence the model never saw.
        if "NEXT_USER_TASK" in text or "RESTART_USER_TASK" in text:
            return self.response(request, "Followup completed. " + evidence(text))
        if self.scenario == "plan":
            return self.plan(request)
        if self.scenario == "approval":
            if "TOOL_OUTPUT_MARKER" not in text:
                return self.response(request, name="fs__bash_exec", arguments={"command": "original proposal"})
            return self.response(request, "Approved tool result read. " + evidence(text))
        if self.scenario == "ports":
            # The overall plan legitimately lists OTHER active tasks. Choose
            # the current task explicitly, never by finding a sibling's title.
            goals = re.findall(r"(?m)^Current task: '([^'\n]+)'", text)
            assert goals, "actor prompt is missing its current task"
            side = goals[-1].startswith("SIDE_PORT_REQUEST")
            answer = "ANSWER_B=green" if side else "ANSWER_A=blue"
            if answer not in text:
                return self.question(request, "Choose side environment?" if side else "Choose main environment?")
            return self.response(request, "Environment selected. " + evidence(text))
        if "SCREENSHOT_WITHDRAWN" in text:
            async def stalled():
                try:
                    self.withdrawn_started.set()
                    await asyncio.Event().wait()
                    yield LLMChunk(kind="done")
                finally:
                    self.withdrawn_closed.set()
            return stalled()
        if "SCREENSHOT_ACCEPTED" in text:
            return self.response(request, "Screenshot checked. SCREENSHOT_RESULT=accepted " + evidence(text))
        return self._stream(MockResponse(text="Please attach the screenshot."), request)

    def plan(self, request):
        # A strict plan has a deterministic actor order. Observer and metadata
        # calls use their own branches and never consume this response script.
        initial = self.stage in {"hot", "prepare"}
        if initial and self.actors == 1:
            return self.response(request, name="control__delegate_plan", arguments={"tasks": [
                {"title": f"Step {i}", "task_prompt": f"Perform release step {i}. " + PROJECT,
                 "use_subagent": self.variant.startswith("cross"), "inherit_memory": True}
                for i in (1, 2, 3)]})
        if initial and self.actors == 2:
            return self.response(request, "Step one done. PART_ONE=verified " + PROJECT)
        if initial and self.actors == 3:
            return self.question(request, "May I validate the release artifact?")
        offset = self.actors - (3 if initial else 0)
        failing = self.variant.endswith("fail")
        if offset == 1:
            self.failed_review = failing
            return self.response(request, ("Step two unavailable. PART_TWO=unavailable " if failing
                                          else "Step two done. PART_TWO=approved ") + PROJECT)
        if not failing and offset == 2:
            return self.response(request, "Step three done. PART_THREE=ready " + PROJECT)
        return self.response(request, "Plan handled. " + evidence(request_text(request)))


class DurableTool(_RecordingBashTool):
    """The external effect survives every runtime process and is independently read."""
    def __init__(self, path):
        super().__init__()
        self.path = path / "tool-effects.jsonl"

    async def _run(self, args):
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(args) + "\n")
            output.flush()
            os.fsync(output.fileno())
        async for event in super()._run(args):
            yield event

    @staticmethod
    def _cap():
        cap = _RecordingBashTool._cap()
        cap.side_effects = True
        return cap


def effects(path):
    file = path / "tool-effects.jsonl"
    return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] if file.exists() else []


class BoundaryStore:
    """Real SQL commits with failures/exits only at provider I/O boundaries."""
    def __init__(self, store, scenario, variant, stage):
        self.store, self.scenario, self.variant, self.stage = store, scenario, variant, stage
        self.attempts = []
        self.source = stage in {"hot", "prepare"}

    def __getattr__(self, name):
        return getattr(self.store, name)

    async def read_range(self, *args, **kwargs):
        return await self.store.read_range(*args, **kwargs)

    async def committed_head(self, *args, **kwargs):
        return await self.store.committed_head(*args, **kwargs)

    async def append_batch(self, sid, bid, events, *args, **kwargs):
        accepted = self.source and self.scenario == "approval" and any(
            e.type == "HitlResolved" and e.payload.get("outcome") == "accepted" for e in events)
        if accepted:
            self.attempts.append((bid, copy.deepcopy(events)))
            if self.variant == "before" and len(self.attempts) <= 2:
                raise OSError("approval commit unavailable")
            if self.variant in {"unknown", "decision_crash"}:
                result = await self.store.append_batch(sid, bid, events, *args, **kwargs)
                if self.variant == "decision_crash":
                    os._exit(73)
                if len(self.attempts) <= 2:
                    raise OSError("approval receipt unavailable")
                return result
        result = await self.store.append_batch(sid, bid, events, *args, **kwargs)
        if self.source and self.scenario == "approval" and self.variant == "tool_crash" and any(
                e.type == "CapabilityFinished" and "TOOL_OUTPUT_MARKER" in str(e.payload) for e in events):
            os._exit(73)
        if self.source and self.scenario == "image" and self.variant == "message_crash" and any(
                e.type == "TaskMessageAppended" and "SCREENSHOT_ACCEPTED" in str(e.payload) for e in events):
            os._exit(73)
        return result


def runtime(path, store, memory, scenario, variant, stage):
    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    llm = JourneyLLM(scenario, stage, variant)
    timeout = None if (variant.endswith("_hot") or variant in {
        "hot", "before", "unknown", "decision_crash", "tool_crash"}) else 0
    rt = make_runtime(llm=llm, agent_provider=templates, event_store=store,
                      config=RuntimeConfig(hitl_timeout_sec=timeout, task_max_concurrent=2))
    rt.providers.register_memory(memory)
    tool = DurableTool(path)
    if scenario == "approval":
        rt.providers.register_capability(tool, tool_authorizers={"fs:bash_exec": HumanConfirmationAuthorizer()})
    if scenario == "image":
        rt.providers.register_event_blob_store(FsBlobStore(path / "event-blobs"))
        rt.providers.register_memory_blob_store(FsBlobStore(path / "memory-blobs"))
    return rt, llm, tool


async def task_finished(rt, task_id, status="FINISHED"):
    await wait_until(lambda: rt._task_managers[SID].get_task(task_id).status == status)


async def question(rt, task_id):
    await wait_until(lambda: any(q.task_id == task_id for q in rt.hitl_registry.list_pending(session_id=SID)))
    return next(q for q in rt.hitl_registry.list_pending(session_id=SID) if q.task_id == task_id)


async def reply(rt, q, message, *, arguments=None):
    return await rt.reply_to_hitl(HitlReply(hitl_id=q.id, agent_id=q.agent_id,
        outcome="accepted", message=message, modified_arguments=arguments))


async def prepare(rt, llm, path, store, scenario, variant, stage):
    prompt = {"plan": "Prepare a release in three ordered steps. ",
              "approval": "Inspect the release artifacts after approval. ",
              "ports": "MAIN_PORT_REQUEST: choose the main environment. ",
              "image": "Inspect my deployment screenshot. "}[scenario] + PROJECT
    h = await rt.start_session(SessionStartParams.create(session_id=SID,
        template_id="agent:tpl_echo", user_prompt=prompt))
    data = {"root": h.agent_id, "task": h.task_id, "agents": [h.agent_id], "tasks": [h.task_id]}
    if scenario == "plan":
        await wait_until(lambda: bool(rt.hitl_registry.list_pending(session_id=SID)))
        await drained(rt)
        children = sorted((t for t in rt._task_managers[SID].all_tasks() if t.parent_task_id == h.task_id),
                          key=lambda t: t.title)
        assert len(children) == 3
        assert [t.status for t in children] == ["FINISHED", "AWAITING_HUMAN", "PENDING"]
        assert children[1].dag_deps == [children[0].id] and children[2].dag_deps == [children[1].id]
        assert rt._task_managers[SID].get_task(h.task_id).status == "SUSPENDED"
        data["children"] = [t.id for t in children]
        data["tasks"] += data["children"]
        data["agents"] = sorted(set(data["agents"] + [t.assigned_agent_id for t in children if t.assigned_agent_id]))
        data["question"] = (await question(rt, children[1].id)).id
    elif scenario == "ports":
        a = await question(rt, h.task_id)
        side = await rt.dispatch_task(SID, "SIDE_PORT_REQUEST: choose the side environment.",
            settings={"use_subagent": True, "subagent_template": "agent:tpl_echo", "inherit_memory": False},
            port_key="side")
        b = await question(rt, side.task_id)
        data.update(side=side.agent_id, side_task=side.task_id, question_a=a.id, question_b=b.id)
        data["agents"].append(side.agent_id)
        data["tasks"].append(side.task_id)
        assert len(await rt.list_pending_hitl(session_id=SID)) == 2
        await answer_first_port(rt, llm, data, variant)
    elif scenario == "approval":
        q = await question(rt, h.task_id)
        assert q.form == "approval" and effects(path) == []
        data["question"] = q.id
        if variant in {"cold", "restart"}:
            await drained(rt)
        save(path, data)
        if variant in {"decision_crash", "tool_crash"}:
            await reply(rt, q, "Approved release listing", arguments=APPROVED_ARGS)
            await drained(rt)
            raise AssertionError("approval did not reach the real commit crash")
    else:
        old = await question(rt, h.task_id)
        await drained(rt)
        await rt.send_message(h.agent_id, screenshot("SCREENSHOT_WITHDRAWN", WITHDRAWN), session_id=SID)
        await asyncio.wait_for(llm.withdrawn_started.wait(), 10)
        withdrawn_images = _image_sources(llm.actor_payloads[-1])
        assert withdrawn_images and all(base64.b64decode(s["data"]) == WITHDRAWN for s in withdrawn_images)
        await rt.pause_agent(h.agent_id)
        await asyncio.wait_for(llm.withdrawn_closed.wait(), 5)
        await drained(rt)
        pending, = rt.hitl_registry.list_pending(session_id=SID)
        assert pending.id == old.id
        await assert_withdrawn_absent(rt, store)
        data["question"] = pending.id
        save(path, data)
        if variant == "message_crash":
            await rt.send_message(h.agent_id, screenshot("SCREENSHOT_ACCEPTED", SCREENSHOT), session_id=SID)
            await task_finished(rt, h.task_id)
            await drained(rt)
            raise AssertionError("image input did not reach its actual commit")
    return data


async def answer_first_port(rt, llm, data, variant):
    first = "a" if variant.startswith("main") else "b"
    tid = data["task"] if first == "a" else data["side_task"]
    q = rt.hitl_registry.get(data["question_" + first])
    await reply(rt, q, "ANSWER_A=blue" if first == "a" else "ANSWER_B=green")
    await task_finished(rt, tid)
    # The other agent may still own a live hot question waiter. Join only the
    # completed producer, without requiring that unrelated waiter to exit.
    tm = rt._task_managers[SID]
    await wait_until(lambda: tid not in tm.running_task_ids() and not tm._background_asyncio_tasks)
    other = "b" if first == "a" else "a"
    pending, = rt.hitl_registry.list_pending(session_id=SID)
    assert pending.id == data["question_" + other]
    assert not any("ANSWER_A=blue" in request_text(r) and "ANSWER_B=green" in request_text(r)
                   for k, r in llm.calls if k == "actor")
    data["second"] = other


def screenshot(label, raw):
    return [TextPart(text=label), ImagePart(data=base64.b64encode(raw).decode(), media_type="image/png")]


async def assert_withdrawn_absent(rt, store):
    rows = await store.read_range(SID)
    assert not any("SCREENSHOT_WITHDRAWN" in str(r.event.payload) for r in rows)
    contents = await memory_state(rt.providers.get_memory(), rt._agent_lifecycle_manager.agent_ids_of_session(SID))
    assert "SCREENSHOT_WITHDRAWN" not in str(contents)


async def assert_settled(rt, store):
    await drained(rt)
    tm = rt._task_managers[SID]
    assert not tm.running_task_ids() and not tm.open_round_task_ids
    assert not rt._run_tokens.get(SID)
    view = await rebuild_view(store, SID)
    assert not view.open_runs and not view.pending_observe and not view.hitl.pending
    assert all(agent.status == "idle" for agent in view.agents.values())
    rows = await store.read_range(SID)
    assert_lifecycles(rows)
    starts = [r for r in rows if r.event.type == "RunStarted"]
    ends = [r for r in rows if r.event.type == "RunFinished"]
    assert len({r.event.run_id for r in starts}) == len(starts)
    assert len({r.event.run_id for r in ends}) == len(ends)
    assert {r.event.run_id for r in starts} == {r.event.run_id for r in ends}
    for start in starts:
        end, = [r for r in ends if r.event.run_id == start.event.run_id]
        assert start.position < end.position
        if start.event.origin == "runtime" and start.event.payload.get("initial_step") == "prepare":
            terminal, = [r for r in rows if r.event.task_id == start.event.task_id and
                         r.event.type in {"TaskFinished", "TaskFailed", "TaskCanceled"}]
            assert end.position < terminal.position
            finalizations = [r for r in rows if r.event.type == "TaskFinalized" and r.event.task_id == start.event.task_id]
            # Foreground FinalizeStep may publish before TaskManager commits
            # the status; background close publishes after it. Both finalize
            # once and retain the required RunFinished -> task terminal order.
            assert len(finalizations) == 1
    for opened in [r for r in rows if r.event.type == "HitlOpened"]:
        rid = opened.event.payload["hitl_id"]
        resolved, = [r for r in rows if r.event.type == "HitlResolved" and r.event.payload["hitl_id"] == rid]
        assert opened.position < resolved.position
        closed = [r for r in rows if r.event.type == "HitlClosed" and r.event.payload["hitl_id"] == rid]
        if resolved.event.payload["outcome"] != "cancelled":
            assert len(closed) == 1 and resolved.position < closed[0].position
        else:
            # Superseding a wait with send_message cancels it. Cancellation is
            # already its consumption boundary; no user reply is left to close.
            assert len(closed) <= 1


async def finish_journey(rt, llm, path, provider, store, memory, data, scenario, variant, stage):
    if scenario == "plan":
        q = rt.hitl_registry.get(data["question"])
        await reply(rt, q, "Validate the release artifact now.")
        await task_finished(rt, data["task"])
        children = [rt._task_managers[SID].get_task(tid) for tid in data["children"]]
        failing = variant.endswith("fail")
        assert [t.status for t in children] == (["FINISHED", "FAILED", "CANCELED"] if failing
                                              else ["FINISHED"] * 3)
        if failing:
            assert children[-1].error_code == "BLOCKED_BY_FAILED_DEP"
        parent = rt._task_managers[SID].get_task(data["task"])
        assert all(part in parent.outputs for part in plan_evidence(failing))
        rows = await store.read_range(SID)
        for prev, nxt in pairwise(children):
            if nxt.status == "CANCELED":
                assert not any(r.event.type == "TaskStarted" and r.event.task_id == nxt.id for r in rows)
                continue
            ended, = [r for r in rows if r.event.type == "TaskFinished" and r.event.task_id == prev.id]
            starts = [r for r in rows if r.event.type == "TaskStarted" and r.event.task_id == nxt.id]
            assert starts and ended.position < starts[0].position
        data["agents"] = sorted(set(data["agents"] + [t.assigned_agent_id for t in children if t.assigned_agent_id]))
    elif scenario == "ports":
        second = data["second"]
        q = rt.hitl_registry.get(data["question_" + second])
        await reply(rt, q, "ANSWER_A=blue" if second == "a" else "ANSWER_B=green")
        await task_finished(rt, data["task"])
        await task_finished(rt, data["side_task"])
        for tid, own, other in [(data["task"], "ANSWER_A=blue", "ANSWER_B=green"),
                                (data["side_task"], "ANSWER_B=green", "ANSWER_A=blue")]:
            output = rt._task_managers[SID].get_task(tid).outputs
            assert own in output and other not in output
        assert rt._task_managers[SID].get_task(data["side_task"]).port_key == "side"
    elif scenario == "approval":
        if variant not in {"decision_crash", "tool_crash"}:
            q = rt.hitl_registry.get(data["question"])
            if variant in {"before", "unknown"}:
                for attempt in range(2):
                    try:
                        await reply(rt, q, "Approved release listing", arguments=(APPROVED_ARGS if not attempt else {"command": "changed retry"}))
                    except PersistenceUnavailableError:
                        assert effects(path) == []
                        rt.clear_storage_isolation(SID)
                    else:
                        raise AssertionError("provider failure did not propagate")
            await reply(rt, q, "Approved release listing", arguments=(
                {"command": "changed retry"} if variant in {"before", "unknown"} else APPROVED_ARGS))
            if variant in {"before", "unknown"}:
                assert len(provider.attempts) >= 3
                assert all(a == provider.attempts[0] for a in provider.attempts)
        else:
            await recover_fixture_session(rt, data["root"], session_id=SID)
        assert await rt.recover_storage(SID)
        await task_finished(rt, data["task"])
        assert effects(path) == [APPROVED_ARGS]
        actors = [request_text(r) for k, r in llm.calls if k == "actor"]
        assert any("TOOL_OUTPUT_MARKER" in t for t in actors)
    else:
        if variant == "message_crash":
            await recover_fixture_session(rt, data["root"], session_id=SID)
        else:
            await rt.send_message(data["root"], screenshot("SCREENSHOT_ACCEPTED", SCREENSHOT), session_id=SID)
        await task_finished(rt, data["task"])
        accepted_payloads = [p for p in llm.actor_payloads if "SCREENSHOT_ACCEPTED" in str(p)]
        assert accepted_payloads
        images = [s for p in accepted_payloads for s in _image_sources(p)]
        assert images and all(base64.b64decode(s["data"]) == SCREENSHOT for s in images)
        await assert_withdrawn_absent(rt, store)
    await drained(rt)
    await next_tasks(rt, llm, data, scenario, "NEXT_USER_TASK")
    await assert_settled(rt, store)
    data["statuses"] = {t.id: t.status for t in rt._task_managers[SID].all_tasks()}
    data["head"] = await store.committed_head(SID)
    data["memory"] = await memory_state(memory, data["agents"])
    return data


async def next_tasks(rt, llm, data, scenario, marker):
    for agent in data["agents"] if scenario == "ports" else [data["root"]]:
        before = len(llm.calls)
        turn = await rt.send_message(agent, "Review the next item. " + marker, session_id=SID, unattended=True)
        assert turn.task_id not in data["tasks"]
        await turn.wait_for_finish(timeout=10)
        await task_finished(rt, turn.task_id)
        await drained(rt)
        requests = [request_text(r) for k, r in llm.calls[before:] if k == "actor"]
        assert requests
        if scenario == "ports":
            own, other = ("ANSWER_A=blue", "ANSWER_B=green") if agent == data["root"] else ("ANSWER_B=green", "ANSWER_A=blue")
            assert own in requests[0] and other not in requests[0]
        else:
            assert PROJECT in requests[0]
        if scenario == "plan":
            assert all(part in requests[0] for part in plan_evidence(data.get("failed")))
        if scenario == "image":
            assert "SCREENSHOT_RESULT=accepted" in requests[0]
            assert "SCREENSHOT_WITHDRAWN" not in requests[0]
        data["tasks"].append(turn.task_id)


async def verify(rt, llm, path, store, memory, data, scenario):
    for _ in range(2):
        for agent in data["agents"]:
            await recover_fixture_session(rt, agent, session_id=SID)
        await drained(rt)
        assert await store.committed_head(SID) == data["head"]
        assert await memory_state(memory, data["agents"]) == data["memory"]
        assert not llm.calls, "recovery re-executed finished work"
    assert not await rt.list_pending_hitl(session_id=SID)
    for tid, status in data["statuses"].items():
        assert (await rt.get_task(SID, tid)).data.status == status
    if scenario == "plan":
        output = (await rt.get_task(SID, data["task"])).data.outputs
        assert all(part in output for part in plan_evidence(data.get("failed")))
    await next_tasks(rt, llm, data, scenario, "RESTART_USER_TASK")
    await assert_settled(rt, store)
    if scenario == "approval":
        assert effects(path) == [APPROVED_ARGS]
    if scenario == "image":
        await assert_withdrawn_absent(rt, store)
