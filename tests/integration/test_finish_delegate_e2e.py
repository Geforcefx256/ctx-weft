"""Delegation suspends the parent, then normal replies close child and resumed parent.

The actor no longer has a finish tool that can detach work from its task tree.
Observer and metadata calls use separate mock routes from actor responses.
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


_CHILD_TITLE = "followup"


class _RouterLLM(MockLLMAdapter):
    """Request-aware mock: pick the response from the tools offered in the request.

    - recognize_intent (has update_task_metadata) → empty (single-shot, no tool).
    - observe (has report_observation)           → report success.
    - background_observe (has report_observation) → report success.
    - background_recap (has report_observation) → emit a segment report.
    - act: first delegates; child and resumed parent then reply in prose.
    """

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._act_calls = 0
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def complete(self, request, stream=True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}

        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)

        if "control__report_observation" in names:
            text = "\n".join(str(m.content) for m in request.messages)
            arguments = {"act_recap": "segment summary", "task_summary": "whole task"}
            if "`task_status`" in text:
                arguments["task_status"] = "success"
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("obs"), name="control__report_observation",
                         arguments=arguments),
            ]), request)

        # act
        self._act_calls += 1
        if self._act_calls == 1:
            # A dispatch keeps the child attached and suspends its parent.
            return self._stream(MockResponse(text="root done", tool_calls=[
                ToolCall(id=self._id("del"), name="control__delegate_task",
                         arguments={"title": _CHILD_TITLE,
                                    "task_prompt": "do the unrelated follow-up"}),
            ]), request)
        # Child and resumed parent both deliver ordinary replies.
        return self._stream(MockResponse(text="child done"), request)


def _make_runtime(llm: _RouterLLM) -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


async def _wait_for_n_finished(runtime, session_id, n, timeout=5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        view = await rebuild_view(runtime.event_store, session_id)
        finished = [t for t in view.tasks.values() if t.status == "FINISHED"]
        if len(finished) >= n:
            return view
        await asyncio.sleep(0.02)
    view = await rebuild_view(runtime.event_store, session_id)
    raise TimeoutError(
        f"expected >={n} FINISHED tasks within {timeout}s; "
        f"got {[(t.title, t.status) for t in view.tasks.values()]}"
    )


@pytest.mark.asyncio
async def test_delegate_same_batch_e2e():
    llm = _RouterLLM()
    runtime = _make_runtime(llm)

    # Spy events to check suspension and the number of actual executions.
    seen = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    await initialize_all(runtime)
    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="root request",
        )
    )
    session_id = handle.session_id
    await handle.wait_for_finish(timeout=5.0)

    # Both parent and child must finish.
    view = await _wait_for_n_finished(runtime, session_id, n=2)
    tasks = list(view.tasks.values())
    assert len(tasks) == 2, f"expected exactly 2 tasks, got {[t.title for t in tasks]}"
    assert all(t.status == "FINISHED" for t in tasks), (
        f"expected both FINISHED, got {[(t.title, t.status) for t in tasks]}"
    )

    # Delegated work remains a blocking child of its parent.
    child = next((t for t in tasks if t.title == _CHILD_TITLE), None)
    assert child is not None, f"delegated task {_CHILD_TITLE!r} not found"
    assert child.parent_task_id == handle.task_id, (
        f"delegated work must stay attached to its parent, got {child.parent_task_id!r}"
    )

    # The non-subagent task must record its real execution agent id at start (run_task),
    # so same/cross-agent classification compares creator==assigned instead of creator==None.
    assert child.assigned_agent_id, (
        "non-subagent task must get its execution agent id recorded at start"
    )
    assert child.assigned_agent_id == child.creator_agent_id, (
        f"non-subagent runs in creator's agent → same-agent; got "
        f"assigned={child.assigned_agent_id!r} creator={child.creator_agent_id!r}"
    )

    # Parent runs twice (before/after the child); the child runs once.
    from collections import Counter
    started = Counter(
        e.task_id for e in seen if getattr(e, "type", None) == EventType.TASK_STARTED
    )
    assert started == Counter({handle.task_id: 2, child.id: 1}), dict(started)

    # Dispatch suspends the parent until its child has returned.
    from tests._event_helpers import all_events
    suspended = [e for e in await all_events(runtime.event_store, session_id)
                 if e.type == EventType.TASK_SUSPENDED]
    assert any(e.task_id == handle.task_id for e in suspended), (
        f"parent must suspend while waiting for its child, got "
        f"{[(e.payload or {}).get('task_id') for e in suspended]}"
    )
