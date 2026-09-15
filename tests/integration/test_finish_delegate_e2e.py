"""End-to-end: actor calls finish_task + delegate_task in the SAME act batch.

Locks the cross-layer behavior of ActStep._reconcile_finish_vs_dispatch +
TaskManager.detach_staged driven through a real session:

1. finish wins  → the current (root) task FINISHES and never SUSPENDS
   (delegate did NOT hijack it into awaiting a child).
2. delegate still dispatches → the delegated work runs as an INDEPENDENT task
   (re-parented to the root's parent = top level), and also reaches FINISHED.

The mock LLM routes by the tools present in each request (recognize_intent /
observe / background_observe / act), so it is robust to call ordering. The first
act turn (root) returns finish+delegate in one batch; later act turns (the
detached follow-up) return finish only.
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio

_CHILD_TITLE = "followup"


class _RouterLLM(MockLLMAdapter):
    """Request-aware mock: pick the response from the tools offered in the request.

    - recognize_intent (has update_task_metadata) → empty (single-shot, no tool).
    - observe (has report_task_outcome)           → report success.
    - background_observe (has collect_process_report) → emit a segment report.
    - act (everything else): 1st call → finish_task + delegate_task batch (the
      scenario under test); subsequent calls → finish_task only (the follow-up).
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

        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("obs"), name="control__report_task_outcome",
                         arguments={"task_status": "success",
                                    "act_recap": "done",
                                    "task_summary": "done"}),
            ]), request)

        if "control__collect_process_report" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("bg"), name="control__collect_process_report",
                         arguments={"act_recap": "segment summary"}),
            ]), request)

        # act
        self._act_calls += 1
        if self._act_calls == 1:
            # Root act: finish + delegate IN THE SAME BATCH (the case under test).
            # 交付物 = 收尾回合正文（finish_task 不带 result 参数——未声明参数会被
            # gateway 严格校验拒绝，见 spec: capability-gateway）。
            return self._stream(MockResponse(text="root done", tool_calls=[
                ToolCall(id=self._id("fin"), name="control__finish_task",
                         arguments={}),
                ToolCall(id=self._id("del"), name="control__delegate_task",
                         arguments={"title": _CHILD_TITLE,
                                    "task_prompt": "do the unrelated follow-up"}),
            ]), request)
        # Detached follow-up act (and any later act): finish only.
        return self._stream(MockResponse(text="child done", tool_calls=[
            ToolCall(id=self._id("fin"), name="control__finish_task",
                     arguments={}),
        ]), request)


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


async def test_finish_plus_delegate_same_batch_e2e():
    llm = _RouterLLM()
    runtime = _make_runtime(llm)

    # Spy events to prove the root never suspended (finish won over delegate).
    seen = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="root request", context_limit=100_000,
        )
    )
    session_id = handle.session_id
    await handle.wait_for_finish(timeout=5.0)

    # Both the root and the detached follow-up must reach FINISHED.
    view = await _wait_for_n_finished(runtime, session_id, n=2)
    tasks = list(view.tasks.values())
    assert len(tasks) == 2, f"expected exactly 2 tasks, got {[t.title for t in tasks]}"
    assert all(t.status == "FINISHED" for t in tasks), (
        f"expected both FINISHED, got {[(t.title, t.status) for t in tasks]}"
    )

    # The delegated follow-up was re-parented to the root's parent (= top level),
    # i.e. dispatched as an INDEPENDENT task, not as a blocking child of the root.
    child = next((t for t in tasks if t.title == _CHILD_TITLE), None)
    assert child is not None, f"delegated task {_CHILD_TITLE!r} not found"
    assert not child.parent_task_id, (
        f"detached follow-up must be top-level (no parent), got parent={child.parent_task_id!r}"
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

    # 每个 task 的 TaskStarted 恰好一条：防止 _run_task(派发前) 与 run_task(_resolve 后) 双发回归。
    # 权威那条由 run_task 发（带 resolved agent id，见上 child.assigned_agent_id 断言）。
    from collections import Counter
    started = Counter(
        e.task_id for e in seen if getattr(e, "type", None) == EventType.TASK_STARTED
    )
    assert started, "expected at least one TaskStarted"
    assert all(c == 1 for c in started.values()), (
        f"TaskStarted must fire exactly once per task, got {dict(started)}"
    )

    # finish WON: the root must never have been suspended by the delegate.
    suspended = [e for e in seen if getattr(e, "type", None) == EventType.TASK_SUSPENDED]
    assert not suspended, (
        f"finish must win — no task should suspend, got "
        f"{[(e.payload or {}).get('task_id') for e in suspended]}"
    )
