
from tests._recovery import recover_fixture_session
"""Providers and process phases for public user-control journeys.

Only provider responses/latency and the external tool are controlled. No events,
task states, memory records or recovery decisions are planted by these tests.
"""
import asyncio
import json
import re

from ctx_weft import RuntimeConfig, SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.content import content_to_text
from ctx_weft.protocols import (
    LLMChunk,
    MemoryAddress,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    ToolCall,
    load_records,
)
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.authorizer import HumanConfirmationAuthorizer
from ctx_weft.providers.llm.adapters.mock import MockResponse
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_dispatch_boundary_recap_e2e import _is_background_recap
from tests.integration.test_hitl_e2e_v2 import _RecordingBashTool
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM

SID = "public-user-controls"
PROJECT = "PROJECT=BRONZE"
REFUSAL = "Please only list files. REFUSAL_REASON=NO_DELETE"
FACTS = ["FACT_0=blue", "FACT_1=staging", "FACT_2=friday", "FACT_3=manual"]
EVIDENCE = "Verified build tests, release artifacts and deployment checks. " * 180


def request_text(request):
    return request.system + "\n" + "\n".join(content_to_text(m.content) for m in request.messages)


def facts(text):
    return " ".join(sorted(set(re.findall(r"(?:PROJECT|FACT_\d+|REFUSAL_REASON)=\w+", text))))


class ControlLLM(_FinishEveryTaskLLM):
    """Real call routing, with controllable slow actor/observer streams."""

    def __init__(self, scenario, phase, variant, model="model-a"):
        super().__init__()
        self.scenario, self.phase, self.variant = scenario, phase, variant
        self._model = model
        self.calls = []
        self.blocked = False
        self.started, self.closed = asyncio.Event(), asyncio.Event()
        self.report_started, self.release_report = asyncio.Event(), asyncio.Event()
        self.side_asked = False
        self.tool_asked = False

    @property
    def model(self):
        return self._model

    def finish(self, request, text=None):
        text = text or "Release checklist ready. " + facts(request_text(request))
        return self._stream(MockResponse(text=text, tool_calls=[]), request)

    def stall(self):
        async def stream():
            try:
                yield LLMChunk(kind="token", text="Checking release evidence")
                self.started.set()
                await asyncio.Event().wait()
            finally:
                self.closed.set()
        return stream()

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        text = request_text(request)
        if "control__update_task_metadata" in names:
            kind = "metadata"
        elif "control__report_observation" in names:
            kind = "report" if _is_background_recap(request) else "judge"
        elif not names:
            kind = "compact"
        else:
            kind = "actor"
        self.calls.append((kind, request))
        if kind == "metadata":
            return self._stream(MockResponse(text=""), request)
        if kind == "compact":
            digest = ("## Goal\nPrepare the release.\n## Constraints\nKeep the recorded decisions.\n"
                      "## Done\n" + facts(text) + "\n## Remaining\nReview the next item.")
            return self._stream(MockResponse(text=digest), request)
        if kind in {"judge", "report"}:
            initial_review = (self.phase == "initial" and self._model == "model-a"
                              and self.scenario in {"resume", "model"}
                              and (self.scenario == "model" or self.variant == "observe"))
            if initial_review and kind == "judge" and not self.blocked:
                self.blocked = True
                return self.stall()
            response = MockResponse(tool_calls=[ToolCall(
                id=self._id("observation"), name="control__report_observation",
                arguments={"task_status": "success", "act_recap": "Verified progress. " + facts(text),
                           "task_summary": "Release checklist. " + facts(text)})])
            if self.scenario == "model" and self.phase == "initial" and kind == "report":
                async def delayed_report():
                    self.report_started.set()
                    await self.release_report.wait()
                    async for chunk in self._stream(response, request):
                        yield chunk
                return delayed_report()
            return self._stream(response, request)
        # A real child agent asks a real question while its parent is paused.
        if self.scenario == "resume" and "SIDE_QUESTION" in text:
            if "ENVIRONMENT=staging" not in text and not self.side_asked:
                self.side_asked = True
                return self._stream(MockResponse(tool_calls=[ToolCall(
                    id=self._id("ask"), name="control__ask_user",
                    arguments={"questions": [{"question": "Which environment should be used?"}]})]), request)
            return self.finish(request)
        if self.scenario == "approval" and "FOLLOWUP_TASK" not in text and "RESTART_TASK" not in text:
            if "REFUSAL_REASON=NO_DELETE" not in text and not self.tool_asked:
                self.tool_asked = True
                return self._stream(MockResponse(tool_calls=[ToolCall(
                    id=self._id("bash"), name="fs__bash_exec", arguments={"command": "remove old builds"})]), request)
            return self.finish(request)
        if (self.scenario == "model" and self.phase == "initial" and self._model == "model-a"
                and not self.tool_asked):
            self.tool_asked = True
            # Two real actor turns make this a foldable segment under the normal
            # single-reply retention rule. No short-segment guard is patched.
            return self._stream(MockResponse(text=EVIDENCE, tool_calls=[ToolCall(
                id=self._id("inspect"), name="fs__bash_exec",
                arguments={"command": "list release artifacts"})]), request)
        if (self.scenario == "resume" and self.phase == "initial"
                and self.variant == "act" and not self.blocked):
            self.blocked = True
            return self.stall()
        if (self.phase == "initial" and self._model == "model-a" and self.scenario in {"resume", "model"}
                and not self.blocked and "FOLLOWUP_TASK" not in text):
            return self.finish(request, "Release evidence. " + PROJECT + "\n" + EVIDENCE)
        return self.finish(request)


class ModelProvider:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def get_client(self, account=None, model=None, *, ctx):
        return self.b if model == "model-b" else self.a


class SlowApprovalStore:
    """Forward all real SQL I/O, letting a slow decision commit yield normally.

    The timeout also permits a correct implementation to persist the decision
    before delivering it: in that case no consumer can close the request yet.
    No events or receipts are changed, and no recovery methods are overridden.
    """

    def __init__(self, store):
        self.store = store
        self.closed_ids = set()
        self.closed = asyncio.Event()
        self.decision_id = None

    def __getattr__(self, name):
        return getattr(self.store, name)

    async def read_range(self, *args, **kwargs):
        return await self.store.read_range(*args, **kwargs)

    async def committed_head(self, *args, **kwargs):
        return await self.store.committed_head(*args, **kwargs)

    async def append_batch(self, session_id, batch_id, events, *args, **kwargs):
        for event in events:
            if event.type == "HitlResolved" and event.payload.get("outcome") == "rejected":
                self.decision_id = event.payload["hitl_id"]
                if self.decision_id not in self.closed_ids:
                    try:
                        async with asyncio.timeout(2):
                            await self.closed.wait()
                    except TimeoutError:
                        pass
        result = await self.store.append_batch(session_id, batch_id, events, *args, **kwargs)
        for event in events:
            if event.type == "HitlClosed":
                self.closed_ids.add(event.payload["hitl_id"])
                if self.decision_id in self.closed_ids:
                    self.closed.set()
        return result


def runtime(store, memory, scenario, phase, variant):
    templates = InlineAgentTemplateProvider()
    template = make_echo_template()
    # A small configured retention floor makes four actual completed tasks
    # sufficient for a user-requested compact; the history is never seeded.
    template.loop_config.compact_keep_last = 1
    templates.register(template)
    a = ControlLLM(scenario, phase, variant)
    b = ControlLLM(scenario, phase, variant, "model-b")
    timeout = None if scenario == "approval" and variant == "hot" else 0
    rt = make_runtime(llm=a, agent_provider=templates, event_store=store,
                      config=RuntimeConfig(hitl_timeout_sec=timeout, task_max_concurrent=2))
    rt.providers.register_llm_provider(ModelProvider(a, b))
    rt.providers.register_memory(memory)
    tool = _RecordingBashTool()
    rt.providers.register_capability(tool, tool_authorizers=(
        {"fs:bash_exec": HumanConfirmationAuthorizer()} if scenario == "approval" else {}))
    return rt, a, b, tool


async def drained(rt, *, reports=True):
    tm = rt._task_managers[SID]
    await wait_until(lambda: not tm._dispatch_tasks and (not reports or not tm._background_asyncio_tasks))


async def finished(rt, task_id):
    await wait_until(lambda: rt._task_managers[SID].get_task(task_id).status == "FINISHED")
    await drained(rt)
    assert (await rt.get_task(SID, task_id)).data.status == "FINISHED"


async def memory_state(memory, agent_ids):
    result = []
    for agent_id in sorted(agent_ids):
        for scope in (MemoryScope.TASK, MemoryScope.AGENT):
            records = await load_records(memory, MemoryAddress(session_id=SID, agent_id=agent_id),
                                              scope, ProviderContext(session_id=SID))
            result.extend([agent_id, str(scope), r.id, r.role, str(r.content), r.metadata] for r in records)
    return result


def save(path, data):
    (path / "receipt.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


async def check_settled(rt, store, *, legacy_hitl=None):
    await drained(rt)
    tm = rt._task_managers[SID]
    assert not tm.running_task_ids() and not tm.open_round_task_ids
    assert not rt._run_tokens.get(SID)
    view = await rebuild_view(store, SID)
    assert not view.open_runs, view.open_runs
    assert not view.pending_observe, view.pending_observe
    rows = await store.read_range(SID)
    assert_lifecycles(rows)
    start_rows = [r for r in rows if r.event.type == "RunStarted"]
    starts = {r.event.run_id: r for r in start_rows}
    assert len(starts) == len(start_rows), "duplicate RunStarted"
    ends = {}
    for row in rows:
        if row.event.type == "RunFinished":
            ends.setdefault(row.event.run_id, []).append(row)
    assert starts.keys() == ends.keys()
    for run, started in starts.items():
        assert len(ends[run]) == 1 and started.position < ends[run][0].position
    for terminal in [r for r in rows if r.event.type in {"TaskFinished", "TaskCanceled", "TaskFailed"}]:
        for run, started in starts.items():
            if (started.event.task_id == terminal.event.task_id and started.event.origin == "runtime"
                    and started.event.payload.get("initial_step") == "prepare"):
                # A user may compact an already completed task later. That
                # separate compact run must not be mistaken for its actor run.
                assert ends[run][0].position < terminal.position
    for opened in [r for r in rows if r.event.type == "HitlOpened"]:
        rid = opened.event.payload["hitl_id"]
        resolved = [r for r in rows if r.event.type == "HitlResolved" and r.event.payload["hitl_id"] == rid]
        closed = [r for r in rows if r.event.type == "HitlClosed" and r.event.payload["hitl_id"] == rid]
        assert len(resolved) == len(closed) == 1
        if rid == legacy_hitl:
            # This exact historical request was recorded by the old producer.
            # Its journal is preserved, while Closed must retire it on replay.
            assert opened.position < closed[0].position < resolved[0].position
            continue
        assert opened.position < resolved[0].position < closed[0].position, (
            f"HITL {rid}: Opened@{opened.position}, Resolved@{resolved[0].position}, "
            f"Closed@{closed[0].position}; recovered pending={list(view.hitl.pending)}")
    assert not view.hitl.pending, "completed tasks left pending HITL: " + str(list(view.hitl.pending))


async def prepare(rt, a, store, memory, scenario, variant):
    h = await rt.start_session(SessionStartParams.create(session_id=SID,
        template_id="agent:tpl_echo", user_prompt="Prepare the release checklist. " + PROJECT,
        unattended=scenario != "approval"))
    data = {"root": h.agent_id, "task": h.task_id, "agents": [h.agent_id], "tasks": [h.task_id]}
    if scenario in {"resume", "model"}:
        await asyncio.wait_for(a.started.wait(), 10)
        assert await rt.pause_agent(h.agent_id) == [h.agent_id]
        await asyncio.wait_for(a.closed.wait(), 5)
        await drained(rt, reports=scenario != "model")
        root_req, = rt.hitl_registry.list_pending(session_id=SID)
        data["pause_hitl"] = root_req.id
        if scenario == "resume":
            child = await rt.dispatch_task(SID, "Check deployment environment. SIDE_QUESTION",
                agent_id=h.agent_id, settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                port_key="environment")
            await wait_until(lambda: any(q.agent_id == child.agent_id and q.form == "question"
                            for q in rt.hitl_registry.list_pending(session_id=SID)))
            await drained(rt)
            question, = [q for q in rt.hitl_registry.list_pending(session_id=SID) if q.agent_id == child.agent_id]
            data.update(child=child.agent_id, child_task=child.task_id, question=question.id)
            data["agents"].append(child.agent_id)
            data["tasks"].append(child.task_id)
        else:
            await asyncio.wait_for(a.report_started.wait(), 10)
            change = rt.set_agent_llm(h.agent_id, llm_model="model-b") if variant == "agent" else rt.set_session_llm(SID, llm_model="model-b")
            assert await change
            assert all(r.model == "model-a" for kind, r in a.calls if kind == "report")
    elif scenario == "approval":
        await wait_until(lambda: bool(rt.hitl_registry.list_pending(session_id=SID)))
        if variant != "hot":
            await drained(rt)
        q, = rt.hitl_registry.list_pending(session_id=SID)
        assert q.form == "approval"
        data["approval"] = q.id
    else:
        await h.wait_for_finish(timeout=10)
        await finished(rt, h.task_id)
        for fact in FACTS:
            turn = await rt.send_message(h.agent_id, "Record this release decision: " + fact,
                                         session_id=SID, unattended=True)
            data["tasks"].append(turn.task_id)
            await turn.wait_for_finish(timeout=10)
            await finished(rt, turn.task_id)
        before = await memory_state(memory, data["agents"])
        compact = await rt.compact_agent(h.agent_id, **({"task_id": data["tasks"][-1]} if variant == "task" else {}))
        assert compact.task_id_is_transient == (variant == "agent")
        assert [r for kind, r in a.calls if kind == "compact"], "manual compact did not call its real LLM path"
        after = await memory_state(memory, data["agents"])
        assert before != after
        summaries = await load_records(memory, MemoryAddress(session_id=SID, agent_id=h.agent_id),
            MemoryScope.AGENT, ProviderContext(session_id=SID), kinds=[MemoryKind.SUMMARY])
        assert any(FACTS[0] in str(r.content) for r in summaries)
        oldest = await load_records(memory, MemoryAddress(session_id=SID, agent_id=h.agent_id,
            task_id=data["tasks"][1]), MemoryScope.TASK, ProviderContext(session_id=SID))
        assert not oldest, "manual compact did not fold the old task's real body"
        data["compacted_memory"] = after
        data["compact_requests"] = [request_text(r) for kind, r in a.calls if kind == "compact"]
    data["prepared_head"] = await store.committed_head(SID)
    return data


async def complete(rt, a, b, tool, store, memory, data, scenario, *, restarted, answered=False):
    root, tid = data["root"], data["task"]
    calls_before = len(a.calls) + len(b.calls)
    if scenario == "resume":
        assert await rt.resume_agent(root, session_id=SID) == [root]
        await finished(rt, tid)
        question, = rt.hitl_registry.list_pending(session_id=SID)
        assert question.id == data["question"] and question.agent_id == data["child"]
        rows = await store.read_range(SID)
        assert not any(r.event.type == "HitlResolved" and r.event.payload.get("hitl_id") == question.id for r in rows)
        await rt.reply_to_hitl(HitlReply(hitl_id=question.id, agent_id=data["child"],
            outcome="accepted", message="ENVIRONMENT=staging"))
        await finished(rt, data["child_task"])
        assert any("ENVIRONMENT=staging" in request_text(r) for k, r in a.calls if k == "actor")
    elif scenario == "approval":
        assert tool.invocations == 0
        if not answered:
            answer = await rt.reply_to_hitl(HitlReply(hitl_id=data["approval"], agent_id=root,
                                                    outcome="rejected", message=REFUSAL))
            assert answer.outcome == "rejected"
        await finished(rt, tid)
        assert tool.invocations == 0
        assert any(REFUSAL in request_text(r) for k, r in a.calls if k == "actor")
    elif scenario == "model":
        assert await rt.resume_agent(root, session_id=SID) == [root]
        await wait_until(lambda: any(k == "actor" for k, _ in b.calls))
        if not restarted:
            assert not a.release_report.is_set()
            assert a.report_started.is_set()
            a.release_report.set()
        await finished(rt, tid)
        assert (await rt.get_task(SID, tid)).data.outputs == "Release checklist ready. " + PROJECT
        old_reports = [request_text(r) for k, r in (b.calls if restarted else a.calls)
                       if k == "report" and "Verified build tests" in request_text(r)]
        assert old_reports and all(PROJECT in text and "TOOL_OUTPUT_MARKER" in text for text in old_reports)
        assert all(r.model == "model-b" for k, r in b.calls if k != "metadata")
        assert not any(k == "actor" for k, r in a.calls[calls_before:])
    else:
        assert await memory_state(memory, data["agents"]) == data["compacted_memory"]
    # First user request after the operation must really execute, using history.
    client = b if scenario == "model" else a
    turn = await rt.send_message(root, "Check one more release item. FOLLOWUP_TASK", session_id=SID, unattended=True)
    assert turn.task_id not in data["tasks"]
    await turn.wait_for_finish(timeout=10)
    await finished(rt, turn.task_id)
    followup = [request_text(r) for k, r in client.calls if k == "actor" and "FOLLOWUP_TASK" in request_text(r)]
    assert followup and PROJECT in followup[-1]
    if scenario == "compact":
        assert all(fact in followup[-1] for fact in FACTS)
        assert "Keep the recorded decisions." in followup[-1]
    data["tasks"].append(turn.task_id)
    await check_settled(rt, store)
    data["memory"] = await memory_state(memory, data["agents"])
    data["head"] = await store.committed_head(SID)
    return data


async def verify_restart(rt, a, b, store, memory, data, scenario):
    root = data["root"]
    assert not await rt.list_pending_hitl(session_id=SID)
    # Actual reconstruction and activation are deliberately repeated, without
    # replacing recovery or filling any missing state in the test.
    await recover_fixture_session(rt, root, session_id=SID)
    await drained(rt)
    await recover_fixture_session(rt, root, session_id=SID)
    await drained(rt)
    assert await store.committed_head(SID) == data["head"]
    assert await memory_state(memory, data["agents"]) == data["memory"]
    assert not a.calls and not b.calls, "completed work was re-executed on restart"
    for task_id in data["tasks"]:
        assert (await rt.get_task(SID, task_id)).data.status == "FINISHED"
    turn = await rt.send_message(root, "Review the last release item. RESTART_TASK", session_id=SID, unattended=True)
    assert turn.task_id not in data["tasks"]
    await turn.wait_for_finish(timeout=10)
    await finished(rt, turn.task_id)
    client = b if scenario == "model" else a
    actors = [r for k, r in client.calls if k == "actor"]
    assert actors and PROJECT in request_text(actors[0])
    if scenario == "compact":
        assert all(fact in request_text(actors[0]) for fact in FACTS)
        assert "Keep the recorded decisions." in request_text(actors[0])
    if scenario == "model":
        assert all(r.model == "model-b" for r in actors)
        assert not a.calls
    await check_settled(rt, store, legacy_hitl=data.get("legacy_hitl"))
