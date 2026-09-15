"""End-to-end: transient LLM outage interrupts a session; /resume re-drives to completion.

Two invariants:
1. After outage: task is SUSPENDED (non-terminal), NOT FAILED; the root agent goes
   ``interrupted`` (AgentInterrupted emitted).
2. After resume with healthy LLM: task reaches FINISHED; agent not left interrupted.
3. Idempotent: a second outage+resume cycle behaves identically; final resume completes.

Task 16 note: ``SessionInterrupted`` (and the SessionRegistry state machine that used to
emit it) is retired — the SM was demoted to a plain agent registry in Task 15. These
assertions now pin the agent-level ALM verdict (``AgentInterrupted`` / ``status_of() ==
"interrupted"``) instead: TaskInterrupted (the task-domain fact TaskManager still emits
on this exact path) drives the ALM five-state machine the same way TaskQueueInterrupted
used to drive SM, so the observation point moves down one layer without losing strength.
The event-store projection's ``session_status`` can no longer reach ``INTERRUPTED``
either (nothing emits the retired event to drive that reducer branch any more), so this
file stops asserting on it — an accepted consequence of session-state ownership moving
to the host (docs/events-v2.md §2.1.1), not a regression.

API contract (confirmed by reading runtime.py):
- ``start_session(params)`` → ``TurnHandle``; emits SESSION_CREATED so the event store can
  be queried by ``recover_agent``.  ``run_single_task`` is a phase-1 compat shim that
  does NOT emit SESSION_CREATED, making ``recover_agent`` fail with "not found".
- ``handle.wait_for_finish(timeout)`` streams events until RunFinished (or timeout).
- ``recover_agent(agent_id)`` → ``None``; re-queues non-terminal tasks and launches
  drain as ``asyncio.create_task`` (fire-and-forget).
- Task status and session status are read back via ``rebuild_view`` from the event store.

Self-heal budget note
--------------------
``stream_llm_resilient`` reads budget config from ``ctx.config`` (the LoopContext).
We inject ``RuntimeConfig(llm_self_heal_max_attempts=1, ...)`` into ``make_runtime(config=...)`` directly.
This also proves the production wiring: if ``ctx.config`` were still ``None``, the 8-attempt / 2s-base-delay
defaults would make these tests take minutes instead of completing instantly.
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import LLMOutageError, ToolCall
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime

pytestmark = pytest.mark.asyncio

# The mock LLM's default context_limit (from MockLLMAdapter constructor default)
_MOCK_CONTEXT_LIMIT = 100_000


def _finish_task_response(text: str = "done") -> MockResponse:
    """Return a MockResponse that calls control__finish_task.

    start_session creates the root task with interaction_mode='interactive', which
    requires the LLM to call control__finish_task (not plain-text reply).
    交付物 = 收尾回合正文（finish_task 不带未声明参数，spec: capability-gateway）。
    """
    return MockResponse(text=text, tool_calls=[
        ToolCall(id="tc1", name="control__finish_task", arguments={}),
    ])


# ── Test fixtures ─────────────────────────────────────────────────────────────


class _FlakyLLM(MockLLMAdapter):
    """First ``fail_for`` **act** calls raise LLMOutageError; afterwards behave as the mock.

    Only act-step LLM calls count toward the failure budget.  Recognize-intent calls
    (identified by the ``control__update_task_metadata`` tool in the request) are routed
    to a separate ``ri_responses`` queue so they don't consume failure tokens or compete
    with the act responses.  This prevents race conditions between a leftover background
    recognize_intent asyncio.Task from a previous run and the current run's act step.
    """

    def __init__(self, *, fail_for: int, ri_responses: list | None = None, **kw):
        super().__init__(**kw)
        self._fail_for = fail_for
        self._raised = 0
        # Separate responses for recognize_intent calls (always succeed, empty by default).
        self._ri_responses = list(ri_responses or [])
        self._ri_idx = 0

    @staticmethod
    def _is_recognize_intent(request) -> bool:
        """Return True if this is a recognize_intent call (has update_task_metadata tool)."""
        tools = getattr(request, "tools", None) or []
        return any(
            getattr(t, "name", "") == "control__update_task_metadata"
            for t in tools
        )

    def complete(self, request, stream=True):
        self.last_request = request
        # Recognize-intent calls use the ri_responses queue (never fail, no act tokens).
        if self._is_recognize_intent(request):
            if self._ri_responses and self._ri_idx < len(self._ri_responses):
                resp = self._ri_responses[self._ri_idx]
                self._ri_idx += 1
                return self._stream(resp, request)
            # If ri_responses exhausted, return empty text response.
            return self._stream(MockResponse(text=""), request)
        # Act calls: fail for the first `fail_for` times, then succeed.
        if self._raised < self._fail_for:
            self._raised += 1

            async def _gen():
                raise LLMOutageError("transient outage")
                yield  # pragma: no cover  (make this an async generator)

            return _gen()
        return super().complete(request, stream=stream)


def _make_runtime(flaky_llm: _FlakyLLM) -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    config = RuntimeConfig(
        llm_self_heal_max_attempts=1,
        llm_self_heal_base_delay_sec=0.0,
        llm_self_heal_max_interval_sec=0.0,
        llm_self_heal_max_duration_sec=0.1,
    )
    runtime = make_runtime(
        llm=flaky_llm,
        agent_provider=resolver,
        config=config,
    )
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


def _new_params(prompt: str = "hi") -> SessionStartParams:
    """Build SessionStartParams for a new session using the echo template."""
    return SessionStartParams.create(
        template_id="agent:tpl_echo",
        user_prompt=prompt,
        context_limit=_MOCK_CONTEXT_LIMIT,
    )


async def _wait_for_run_finish(handle, timeout: float = 5.0) -> None:
    """Wait for the handle's run to emit RunFinished (with a short timeout)."""
    await handle.wait_for_finish(timeout=timeout)


async def _wait_for_task_finished(
    runtime: CtxWeftRuntime,
    session_id: str,
    timeout: float = 5.0,
) -> None:
    """Poll rebuild_view until at least one task reaches FINISHED.

    recover_agent() launches drain as an asyncio background task
    (asyncio.create_task inside _register_and_drain), so we must yield control
    until the drain completes.
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        view = await rebuild_view(runtime.event_store, session_id)
        tasks = list(view.tasks.values())
        if any(t.status == "FINISHED" for t in tasks):
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"No task in session {session_id!r} reached FINISHED within {timeout}s")


async def _wait_for_interrupted_event(
    seen: list,
    timeout: float = 5.0,
) -> None:
    """Wait until a new AgentInterrupted event appears in the spy list.

    recover_agent() launches drain as asyncio.create_task, so we need to yield
    to the event loop and poll until the event is captured by the spy. (Task 16:
    SessionInterrupted is retired along with the SessionRegistry state machine —
    AgentInterrupted is the still-live successor signal, see module docstring.)
    """
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        has_interrupted = any(
            getattr(e, "type", None) == EventType.AGENT_INTERRUPTED
            for e in seen
        )
        if has_interrupted:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("Did not receive AgentInterrupted within timeout")


def _agent_not_left_interrupted(seen: list, agent_id: str) -> bool:
    """True unless this agent's most recent ALM event in ``seen`` is AgentInterrupted.

    Deliberately event-log-based rather than a live ``status_of(agent_id)`` check: a
    session that finishes draining releases its agent registry records
    (``CtxWeftRuntime._release_session`` → ``AgentLifecycleManager.release_session``), so by
    the time "after resume" assertions run following a successful completion the
    agent may already be gone from the registry — that's expected, not a bug. Also
    tolerant of a ``seen.clear()`` between phases (test_idempotent_outage_resume_
    completes does this): if this agent has no ALM event at all in the current
    window, there is nothing to have left it interrupted, so that counts as fine too.
    """
    alm_types = (EventType.AGENT_RUNNING, EventType.AGENT_IDLE,
                 EventType.AGENT_INTERRUPTED, EventType.AGENT_WAITING_HUMAN)
    relevant = [e for e in seen
                if getattr(e, "agent_id", None) == agent_id and getattr(e, "type", None) in alm_types]
    if not relevant:
        return True
    return getattr(relevant[-1], "type", None) != EventType.AGENT_INTERRUPTED


# ── Tests ─────────────────────────────────────────────────────────────────────


async def test_outage_then_resume_completes():
    """Single outage → INTERRUPTED; then recover_agent re-drives to FINISHED.

    With _DEFAULT_MAX_ATTEMPTS=1 (patched above), the first complete() call raises
    LLMOutageError immediately and stream_llm_resilient exhausts its budget after 1
    attempt, propagating LLMOutageError to _run_loop which sets task SUSPENDED.
    After recover_agent(), _FlakyLLM(fail_for=1) succeeds (raised >= fail_for), so
    the resume re-drive produces a normal response and task reaches FINISHED.
    """
    # Act responses (only consumed by non-recognize_intent LLM calls):
    # - One _finish_task_response for the resume's act step.
    # Recognize-intent responses (separate queue, never fail):
    # - Extra buffers for any background recognize_intent tasks from leftover runs.
    llm = _FlakyLLM(
        fail_for=1,
        responses=[_finish_task_response("done")],
        ri_responses=[MockResponse(text=""), MockResponse(text=""), MockResponse(text="")],
    )
    runtime = _make_runtime(llm)

    # Spy: collect all emitted events for post-hoc assertions.
    seen = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    # ── Initial run (uses start_session so SESSION_CREATED is persisted) ──────
    handle = await runtime.start_session(_new_params())
    session_id = handle.session_id

    # Wait for the run's RunFinished event (outage causes it to emit RunFinished).
    await _wait_for_run_finish(handle)

    # ── Invariant 1: after outage ──────────────────────────────────────────────
    view_after_outage = await rebuild_view(runtime.event_store, session_id)
    tasks_after_outage = list(view_after_outage.tasks.values())
    assert tasks_after_outage, "Expected tasks in event store after initial run"

    # Task must be non-terminal (NOT FAILED / CANCELED / FINISHED).
    # Note: the LLMOutageError path sets task.status=SUSPENDED in memory but does NOT
    # emit TASK_SUSPENDED to the event store (SuspendStep emits it; the outage handler
    # doesn't go through SuspendStep).  So the event-store projection shows the last
    # recorded TASK_STARTED → ACTIVE.  The invariant is that the task is not terminal.
    _TERMINAL = {"FAILED", "CANCELED", "FINISHED"}
    assert all(t.status not in _TERMINAL for t in tasks_after_outage), (
        f"Task must not be terminal after transient outage, got {[t.status for t in tasks_after_outage]}"
    )

    # An AgentInterrupted must have been emitted, and the root agent's ALM status
    # must reflect it (Task 16: replaces the retired session-level SessionInterrupted).
    interrupted_events = [
        e for e in seen
        if getattr(e, "type", None) == EventType.AGENT_INTERRUPTED
    ]
    assert interrupted_events, "Expected AgentInterrupted after outage"

    sess_after_outage = view_after_outage.sessions.get(session_id)
    assert sess_after_outage is not None
    root_agent_id = sess_after_outage.root_agent_id
    assert root_agent_id
    assert runtime._agent_lifecycle_manager.status_of(root_agent_id) == "interrupted", (
        f"Expected root agent {root_agent_id!r} interrupted after outage, got "
        f"{runtime._agent_lifecycle_manager.status_of(root_agent_id)!r}"
    )

    # No TASK_FAILED must have been emitted.
    assert not [e for e in seen if getattr(e, "type", None) == EventType.TASK_FAILED], (
        "TASK_FAILED must not be emitted on transient outage"
    )

    # ── Resume with healthy LLM ────────────────────────────────────────────────
    # recover_agent() returns None; drain runs as asyncio background task.
    result = await runtime.recover_agent(root_agent_id)
    assert result is None  # confirm the API contract (returns None)

    # Wait for the async drain to complete.
    await _wait_for_task_finished(runtime, session_id)

    # ── Invariant 2: after resume ──────────────────────────────────────────────
    view = await rebuild_view(runtime.event_store, session_id)
    tasks = list(view.tasks.values())
    assert tasks, "Expected at least one task in view after resume"
    assert all(t.status == "FINISHED" for t in tasks), (
        f"Expected all tasks FINISHED after resume, got {[t.status for t in tasks]}"
    )

    sess = view.sessions.get(session_id)
    assert sess is not None
    assert _agent_not_left_interrupted(seen, root_agent_id), (
        f"Root agent {root_agent_id!r} must not be left interrupted after successful resume"
    )


async def test_idempotent_outage_resume_completes():
    """Two outages + two resumes; each outage re-interrupts; final resume completes.

    With _DEFAULT_MAX_ATTEMPTS=1 (patched), each _FlakyLLM call that raises will
    immediately exhaust the budget and produce LLMOutageError → SUSPENDED.
    fail_for=2 means:
      - Initial run: call 1 raises → SUSPENDED.
      - First resume re-drive: call 2 raises → SUSPENDED again.
      - Second resume re-drive: call 3 succeeds → FINISHED.
    No exception must leak out of recover_agent(); the final state must be FINISHED.
    """
    # Act responses (two failures + one success): fail 1 (first run), fail 2 (first resume),
    # then succeed on the second resume.
    # Recognize-intent responses: extra buffers for concurrent/leftover RI tasks.
    llm = _FlakyLLM(
        fail_for=2,
        responses=[_finish_task_response("ok")],
        ri_responses=[
            MockResponse(text=""),
            MockResponse(text=""),
            MockResponse(text=""),
            MockResponse(text=""),
        ],
    )
    runtime = _make_runtime(llm)

    seen = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy

    # ── First run → outage → SUSPENDED ────────────────────────────────────────
    handle = await runtime.start_session(_new_params())
    session_id = handle.session_id

    await _wait_for_run_finish(handle)

    view_after_first = await rebuild_view(runtime.event_store, session_id)
    tasks_after_first = list(view_after_first.tasks.values())
    assert tasks_after_first, "Expected tasks after initial run"
    # Task is non-terminal (not FAILED/CANCELED/FINISHED); outage doesn't emit TASK_SUSPENDED.
    _TERMINAL = {"FAILED", "CANCELED", "FINISHED"}
    assert all(t.status not in _TERMINAL for t in tasks_after_first), (
        f"Task must not be terminal after first outage, got {[t.status for t in tasks_after_first]}"
    )
    sess_after_first = view_after_first.sessions.get(session_id)
    assert sess_after_first is not None
    root_agent_id = sess_after_first.root_agent_id
    assert root_agent_id
    assert runtime._agent_lifecycle_manager.status_of(root_agent_id) == "interrupted", (
        f"Expected root agent {root_agent_id!r} interrupted after first outage, got "
        f"{runtime._agent_lifecycle_manager.status_of(root_agent_id)!r}"
    )
    interrupted_1 = [
        e for e in seen
        if getattr(e, "type", None) == EventType.AGENT_INTERRUPTED
    ]
    assert interrupted_1, "Expected AgentInterrupted after first outage"

    # No TASK_FAILED in the first cycle.
    assert not [e for e in seen if getattr(e, "type", None) == EventType.TASK_FAILED], (
        "TASK_FAILED must not be emitted on first transient outage"
    )

    # ── First resume → second outage → still SUSPENDED ────────────────────────
    seen.clear()
    await runtime.recover_agent(root_agent_id)

    # Wait until the second outage fires: poll the spy list for the new
    # SessionInterrupted event from the background drain.
    await _wait_for_interrupted_event(seen)

    view_mid = await rebuild_view(runtime.event_store, session_id)
    tasks_mid = list(view_mid.tasks.values())
    assert tasks_mid, "Expected tasks in view after first resume"
    # Tasks are non-terminal (second outage, not FAILED).
    _TERMINAL = {"FAILED", "CANCELED", "FINISHED"}
    assert all(t.status not in _TERMINAL for t in tasks_mid), (
        f"Task must not be terminal after second outage, got {[t.status for t in tasks_mid]}"
    )
    sess_mid = view_mid.sessions.get(session_id)
    assert sess_mid is not None
    assert runtime._agent_lifecycle_manager.status_of(root_agent_id) == "interrupted", (
        f"Expected root agent {root_agent_id!r} interrupted after second outage, got "
        f"{runtime._agent_lifecycle_manager.status_of(root_agent_id)!r}"
    )

    # _wait_for_interrupted_event already confirmed AgentInterrupted is in seen.
    assert any(
        getattr(e, "type", None) == EventType.AGENT_INTERRUPTED
        for e in seen
    ), "Expected AgentInterrupted again after second outage"

    # No TASK_FAILED in any outage cycle.
    assert not [e for e in seen if getattr(e, "type", None) == EventType.TASK_FAILED], (
        "TASK_FAILED must not be emitted on any transient outage"
    )

    # ── Second resume → healthy LLM → FINISHED ────────────────────────────────
    seen.clear()
    await runtime.recover_agent(root_agent_id)
    await _wait_for_task_finished(runtime, session_id)

    view_final = await rebuild_view(runtime.event_store, session_id)
    tasks_final = list(view_final.tasks.values())
    assert all(t.status == "FINISHED" for t in tasks_final), (
        f"Expected all tasks FINISHED after final resume, got {[t.status for t in tasks_final]}"
    )

    sess_final = view_final.sessions.get(session_id)
    assert sess_final is not None
    assert _agent_not_left_interrupted(seen, root_agent_id), (
        f"Root agent {root_agent_id!r} must not be left interrupted after final resume"
    )
