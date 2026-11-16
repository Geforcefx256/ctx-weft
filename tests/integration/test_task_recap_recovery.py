"""Crash-recovery of a session stuck in the finish-boundary background recap.

The reported bug: a session whose root task finished (``TASK_FINISHED`` persisted)
but whose finish-boundary background observe had NOT yet completed gets stuck —
``_fire_session_done`` was blocked awaiting that observe when the crash happened, so
``SESSION_FINISHED`` / terminal status never persisted (the session projection stays
RUNNING). On ``/resume`` the old ``_recover_session_locked`` found no resumable tasks
(the only task is terminal) and raised ``RuntimeError("has no resumable tasks")`` →
the resume click hung.

Test A reproduces that stuck state and asserts recover_agent re-runs the pending
recap and finalizes the session to SUCCEEDED. Test B guards the genuinely-empty
projection: a session with no tasks at all must still raise RuntimeError.

Task 16 note: ``SessionFinished`` itself (and the SessionRegistry state machine that
used to emit it) is retired — the SM was demoted to a plain agent registry in Task 15.
The tests below used to pin ``TaskQueueDrained`` instead (the TM aggregate signal that
``SessionFinished`` was translated from). Task 12 (2026-09-04, events-v2 §5) retired
that signal too — ``TaskManager.announce_queue_state`` stopped emitting it, since its
one consumer (the session state machine) was already gone. ``finalize_idle_session``
still computes the same terminal verdict, it just writes it directly onto the
``TaskManager``-held ``Session`` object instead of broadcasting it as an event. Two of
the tests below spy on ``finalize_idle_session``'s ``status`` argument directly (by the
time ``recover_agent()`` returns, the session may already be fully released and the
``TaskManager`` gone from ``runtime._task_managers``, so reading ``tm.session.status``
back after the fact isn't reliable there); the third reads ``tm.session.status`` off a
``TaskManager`` reference captured while it's still guaranteed live. They still don't
assert on the event-store projection's ``session_status`` reaching a terminal value,
because nothing emits ``SessionFinished`` any more to drive that reducer branch — that's
an accepted consequence of session-state ownership moving to the host
(docs/events-v2.md §2.1.1), not a regression this file should guard against.
"""

from __future__ import annotations

from tests._recovery import recover_fixture_session

import asyncio
from datetime import datetime, timezone

import pytest

from ctx_weft import RuntimeConfig
from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.capabilities.control_tools import REPORT_OBSERVATION_NAME
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    ProviderContext,
    ToolCall,
)
from ctx_weft.protocols.capability import qualify
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._event_helpers import append_one
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


_TS = datetime(2026, 7, 12, tzinfo=timezone.utc)


def _make_runtime() -> tuple[CtxWeftRuntime, InMemoryMemoryProvider, list]:
    """Build a runtime with the echo template + in-memory memory, spying on emits.

    The mock LLM returns a fixed recap for the relaunched background observe so the
    re-run completes deterministically (plain-text fallback across react rounds).
    """
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    # Enough plain-text responses to outlast the observe ReAct loop
    # (max_turns_per_observe defaults to 5) without exhausting the mock.
    # `observer_verdict="success"`：actor 回一段正文就收尾，而 2026-09-28 起纯文本不再被机械
    # 判 success（`_mechanical_verdict` 判 continue）——不给这条路由，task 会一路重试到
    # max_retries 然后落 FAILED。
    llm = MockLLMAdapter(responses=[MockResponse(text="recovered recap") for _ in range(8)],
                         observer_verdict="success")
    # 夹具的事件流只发 SESSION_CREATED，没有 AGENT_INSTANTIATED（恢复用的最小夹具，
    # agent 的出身不是这些用例的关注点），所以装填出来的 agent 没有自己的 template_id——
    # 而 registry 不再借会话那份（2026-09-30）。配上 host 声明的母版，装配时由
    # TemplateLookup 兜底，与生产同一条链路。
    runtime = make_runtime(llm=llm, agent_provider=resolver,
                           config=RuntimeConfig(fallback_template_ref="agent:tpl_echo"))
    mem = InMemoryMemoryProvider()
    runtime.providers.register_memory(mem)

    seen: list = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy
    return runtime, mem, seen


def _ev(seq: int, type_, **payload) -> Event:
    # task_id populates both the Event field (used by the status reducers) and stays
    # available; the recap fold specifically reads it from the *payload* (mirroring how
    # background_observe emits OBSERVE_STARTED), so callers that need it there keep it.
    task_id = payload.get("task_id")
    if type_ not in (EventType.OBSERVE_STARTED, EventType.OBSERVE_APPLIED):
        task_id = payload.pop("task_id", None)
    return Event(
        id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id="ses_recap",
        type=type_, timestamp=_TS, task_id=task_id, payload=payload,
    )


@pytest.mark.asyncio
async def test_stuck_finish_session_recovers_and_finalizes(monkeypatch) -> None:
    """root task FINISHED + TaskRecapStarted (no Done) + session projection RUNNING
    (no SESSION_FINISHED) → recover_agent re-runs the recap and finalizes SUCCEEDED."""
    runtime, mem, seen = _make_runtime()
    sid, tid, aid = "ses_recap", "tsk_recap", "agt_root"

    seed = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="agent:tpl_echo", root_agent_id=aid),
        _ev(2, EventType.RUN_STARTED),
        _ev(3, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            # 崩溃恢复的种子：**无人值守**。这些用例驱动的是「进程死掉之后把 task 捡
            # 回来跑完」，没有人在等下一条消息；不标的话纯文本收尾会 park 等一个永远
            # 不来的人（判据自 2026-09-22 起是「有没有人在」）。
            "unattended": True,
            "assigned_agent_id": aid, "creator_agent_id": aid}),
        _ev(4, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
        _ev(5, EventType.TASK_FINISHED, task_id=tid,
            outcome="success", summary="done", outputs={"result": "ok"}),
        # TaskFinalized 的真实发射侧只带 {task_id, outcome}（finalize.py:766）——
        # outputs 已由上面的 TASK_FINISHED 折进投影，这里不再伪造它。
        _ev(6, EventType.TASK_FINALIZED, task_id=tid, outcome="success"),
        # Crash mid finish-boundary observe: STARTED persisted, DONE never was.
        _ev(7, EventType.OBSERVE_STARTED, task_id=tid, boundary="finish", agent_id=aid),
    ]
    for e in seed:
        await append_one(runtime.event_store, e)

    # Pre-crash state: session projection is still RUNNING (no SESSION_FINISHED).
    view_before = await rebuild_view(runtime.event_store, sid)
    assert not hasattr(view_before.sessions[sid], "status")
    assert view_before.tasks[tid].outputs == {"result": "ok"}

    # Seed the placeholder finish pair in memory so _relaunch's close-synth path can
    # locate the finish_task tool_call and supersede the placeholder Process Report.
    scope = MemoryAddress(session_id=sid, task_id=tid, agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id=tid, agent_id=aid)
    fin_tcid = "tc_finish"
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
        content="Process Report: (placeholder)", timestamp=_TS, role="assistant",
        metadata={"origin_task_id": tid, "tool_calls": [
            {"id": fin_tcid, "name": qualify("control:finish_task"), "input": {}}]},
    ), pctx)
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
        content="(placeholder finish result)", timestamp=_TS, role="tool",
        metadata={"origin_task_id": tid, "tool_call_id": fin_tcid},
    ), pctx)

    # finalize_idle_session's terminal verdict used to be observable as a
    # TaskQueueDrained(final_status) event; 2026-09-04 (Task 12, events-v2 §5) retired
    # that broadcast (its one consumer, the session state machine, was already gone).
    # Spy on the call itself instead — by the time recover_agent() returns, the
    # session may already be fully released (`_release_session` pops the TaskManager
    # out of `runtime._task_managers`), so reading `tm.session.status` back after the
    # fact isn't reliable; capturing the argument at the call site is.
    finalized: list[str] = []
    orig_finalize = TaskManager.finalize_idle_session

    async def _spy_finalize(self, *, _orig=orig_finalize):
        finalized.append(self.get_task(tid).status)
        return await _orig(self)

    monkeypatch.setattr(TaskManager, "finalize_idle_session", _spy_finalize)

    # ── Recover ────────────────────────────────────────────────────────────────
    # 装填是调用方的责任（2026-09-21：`recover_agent` 对 registry miss 直接抛
    # `AgentNotLoaded`，按 agent 扫全库的 sweep 已删）。只喂内存，不建 TM、不跑。
    await initialize_all(runtime)
    await runtime.rebuild_session(sid)
    await initialize_all(runtime)
    await recover_fixture_session(runtime, aid)

    # finalize_idle_session awaits the relaunched recap before finalizing the session,
    # so it should already be persisted; drain any stragglers defensively.
    tm = runtime._task_managers.get(sid)
    if tm is not None and tm._background_asyncio_tasks:
        await asyncio.gather(*list(tm._background_asyncio_tasks), return_exceptions=True)

    assert finalized == ["FINISHED"], (
        f"expected finalize_idle_session to finalize SUCCEEDED exactly once, got {finalized!r}"
    )

    # NOTE: the event-store *projection*'s session_status can no longer reach a
    # terminal value here — nothing emits SessionFinished any more (Task 15/16), and
    # reducers.py's session_status write only happens on that event. This is an
    # accepted consequence of the ownership move (per docs/events-v2.md §2.1.1, the
    # host now aggregates "is the session busy" itself rather than core broadcasting
    # a precomputed verdict); it is not something this test should assert on any more.

    # The pending recap was closed out (OBSERVE_APPLIED emitted by the re-run's finally).
    assert any(
        getattr(e, "type", None) == EventType.OBSERVE_APPLIED
        and (e.payload or {}).get("task_id") == tid
        for e in seen
    ), "expected OBSERVE_APPLIED from the relaunched recap"


@pytest.mark.asyncio
async def test_stuck_failed_session_recovers_and_finalizes_failed(monkeypatch) -> None:
    """Variant of Test A seeded with TASK_FAILED (+ FAILURE_THRESHOLD_HIT) instead of
    TASK_FINISHED: finalize_idle_session's FAILED branch (``session.failure_counter > 0``
    in ``_recover_session_locked``) must report TaskQueueDrained(FAILED), not SUCCEEDED.

    ``session.failure_counter`` is only incremented by the FAILURE_THRESHOLD_HIT reducer
    rule (``TASK_FAILED`` alone does not touch it — see
    ``TaskManager.on_task_finished``/``reducers._apply``), so both events are seeded to
    reach the same failure_counter > 0 state a real run would reach when a task's
    retries are exhausted and the session-level failure threshold trips.
    """
    runtime, mem, seen = _make_runtime()
    sid, tid, aid = "ses_recap", "tsk_recap", "agt_root"

    seed = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="agent:tpl_echo", root_agent_id=aid),
        _ev(2, EventType.RUN_STARTED),
        _ev(3, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            # 崩溃恢复的种子：**无人值守**。这些用例驱动的是「进程死掉之后把 task 捡
            # 回来跑完」，没有人在等下一条消息；不标的话纯文本收尾会 park 等一个永远
            # 不来的人（判据自 2026-09-22 起是「有没有人在」）。
            "unattended": True,
            "assigned_agent_id": aid, "creator_agent_id": aid}),
        _ev(4, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
        _ev(5, EventType.TASK_FAILED, task_id=tid,
            error_code="TASK_FAILED_AT_RUN", error_message="boom", retry_count=3),
        _ev(6, EventType.FAILURE_THRESHOLD_HIT, failure_counter=1, threshold=1),
        # TaskFinalized 的真实发射侧只带 {task_id, outcome}（finalize.py:766）——
        # error 已由上面的 TASK_FAILED 折进投影，这里不再伪造它。
        _ev(7, EventType.TASK_FINALIZED, task_id=tid, outcome="fail"),
        # Crash mid finish-boundary observe: STARTED persisted, DONE never was.
        _ev(8, EventType.OBSERVE_STARTED, task_id=tid, boundary="finish", agent_id=aid),
    ]
    for e in seed:
        await append_one(runtime.event_store, e)

    # Pre-crash state: task FAILED, session projection failure_counter > 0, still RUNNING.
    view_before = await rebuild_view(runtime.event_store, sid)
    assert view_before.tasks[tid].status == "FAILED"
    assert view_before.tasks[tid].error == "boom"
    assert view_before.sessions[sid].failure_counter > 0
    assert not hasattr(view_before.sessions[sid], "status")

    # Seed the placeholder finish pair in memory so _relaunch's close-synth path can
    # locate the finish_task tool_call (register_close_synth infers outcome="fail"
    # from task.status == "FAILED" — see runtime._relaunch_observe).
    scope = MemoryAddress(session_id=sid, task_id=tid, agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id=tid, agent_id=aid)
    fin_tcid = "tc_finish_fail"
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
        content="Process Report: (placeholder)", timestamp=_TS, role="assistant",
        metadata={"origin_task_id": tid, "tool_calls": [
            {"id": fin_tcid, "name": qualify("control:finish_task"), "input": {}}]},
    ), pctx)
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.AGENT_CONVERSATION_TURN, address=scope,
        content="(placeholder finish result)", timestamp=_TS, role="tool",
        metadata={"origin_task_id": tid, "tool_call_id": fin_tcid},
    ), pctx)

    # See the identically-worded note in test_stuck_finish_session_recovers_and_
    # finalizes just above for why this spies on finalize_idle_session's argument
    # rather than pinning a TaskQueueDrained event (Task 12 retired it, events-v2 §5).
    finalized: list[str] = []
    orig_finalize = TaskManager.finalize_idle_session

    async def _spy_finalize(self, *, _orig=orig_finalize):
        finalized.append(self.get_task(tid).status)
        return await _orig(self)

    monkeypatch.setattr(TaskManager, "finalize_idle_session", _spy_finalize)

    # ── Recover ────────────────────────────────────────────────────────────────
    await initialize_all(runtime)
    await runtime.rebuild_session(sid)
    await initialize_all(runtime)
    await recover_fixture_session(runtime, aid)

    tm = runtime._task_managers.get(sid)
    if tm is not None and tm._background_asyncio_tasks:
        await asyncio.gather(*list(tm._background_asyncio_tasks), return_exceptions=True)

    # finalize_idle_session finalized FAILED — not SUCCEEDED (Test A's happy path).
    assert finalized == ["FAILED"], (
        f"expected finalize_idle_session to finalize FAILED exactly once, got {finalized!r}"
    )

    # The pending finish-boundary recap was still closed out despite the failure.
    assert any(
        getattr(e, "type", None) == EventType.OBSERVE_APPLIED
        and (e.payload or {}).get("task_id") == tid
        for e in seen
    ), "expected OBSERVE_APPLIED from the relaunched recap"


class _RoutingLLM(MockLLMAdapter):
    """Routes each ``complete()`` call by request shape into one of two independent
    FIFO queues, instead of sharing a single index (the base ``MockLLMAdapter``).

    Recovery of a SUSPENDED task with a pending recap spins up two *concurrent*
    coroutines: the resumed task's own act loop, and the relaunched background-observe
    ReAct loop. A single shared response queue would race between them (whichever
    coroutine's LLM call lands first "steals" the next response, regardless of which
    queue it was meant for) — flaky by construction. Routing on the presence of the
    recap-report tool (only bound to background-observe requests) keeps each
    coroutine's responses deterministic regardless of scheduling order.
    """

    def __init__(self, *, responses: list[MockResponse], recap_responses: list[MockResponse], **kw):
        super().__init__(responses=responses, **kw)
        self._recap_responses = list(recap_responses)
        self._recap_idx = 0

    @staticmethod
    def _is_background_observe(request) -> bool:
        tools = getattr(request, "tools", None) or []
        return any(getattr(t, "name", "") == REPORT_OBSERVATION_NAME for t in tools)

    def complete(self, request, stream: bool = True):
        self.last_request = request
        if self._is_background_observe(request):
            if self._recap_idx >= len(self._recap_responses):
                raise RuntimeError(
                    f"_RoutingLLM: recap responses exhausted after {self._recap_idx} calls"
                )
            resp = self._recap_responses[self._recap_idx]
            self._recap_idx += 1
            return self._stream(resp, request)
        return super().complete(request, stream=stream)


@pytest.mark.asyncio
async def test_suspended_task_with_pending_interrupt_recap_recovers() -> None:
    """Spec §7: a SUSPENDED (resumable) task with a pending interrupt-boundary recap.

    Simulates a crash where the root task was interrupted mid-act (e.g. an LLM outage,
    TaskSuspended persisted) while a background observe segment recap was also
    mid-flight (TaskRecapStarted boundary="interrupt", no Done). recover_agent must
    do BOTH: re-queue/re-dispatch the SUSPENDED task (TaskManager.restore) AND
    re-launch the pending recap (_relaunch_observe) — and the re-fold guard in
    background_observe._run_recap must let it fold exactly once (no
    duplicate TaskRecapDone / no hang). The resumed task then runs to completion
    (its own normal-boundary close triggers a *second*, unrelated recap — expected,
    accepted best-effort concurrency per spec §5.1/§3.6), and the session reaches a
    terminal status.
    """
    runtime, mem, seen = _make_runtime()
    sid, tid, aid = "ses_recap", "tsk_recap", "agt_root"

    seed = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="agent:tpl_echo", root_agent_id=aid),
        _ev(2, EventType.RUN_STARTED),
        _ev(3, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            # 崩溃恢复的种子：**无人值守**。这些用例驱动的是「进程死掉之后把 task 捡
            # 回来跑完」，没有人在等下一条消息；不标的话纯文本收尾会 park 等一个永远
            # 不来的人（判据自 2026-09-22 起是「有没有人在」）。
            "unattended": True,
            "assigned_agent_id": aid, "creator_agent_id": aid}),
        _ev(4, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
        # Crash mid interrupt-boundary observe while the task itself is suspended
        # (e.g. LLM outage during act, spec/07 outage path): STARTED persisted, DONE never was.
        _ev(5, EventType.TASK_SUSPENDED, task_id=tid, summary="outage"),
        _ev(6, EventType.OBSERVE_STARTED, task_id=tid, boundary="interrupt", agent_id=aid),
    ]
    for e in seed:
        await append_one(runtime.event_store, e)

    # Route the routing-aware mock LLM in for this test (the module-level _make_runtime
    # builds a plain MockLLMAdapter; swap it for one that separates the two concurrent
    # LLM consumers deterministically — see _RoutingLLM docstring).
    runtime.llm = _RoutingLLM(
        responses=[MockResponse(text="resumed and done")],
        recap_responses=[
            MockResponse(tool_calls=[
                ToolCall(id=f"tc_recap_{i}", name=REPORT_OBSERVATION_NAME,
                          arguments={"task_status": "success",
                                     "act_recap": "recovered recap"}),
            ])
            for i in range(4)
        ],
    )

    view_before = await rebuild_view(runtime.event_store, sid)
    assert view_before.tasks[tid].status == "SUSPENDED"
    assert not hasattr(view_before.sessions[sid], "status")

    # Seed one un-folded raw LLM_RESPONSE so the interrupt-boundary recap's re-fold
    # guard sees active raw and takes the real fold path (rather than the "already
    # folded, nothing to do" no-op skip — see background_observe._run_recap).
    scope = MemoryAddress(session_id=sid, task_id=tid, agent_id=aid)
    pctx = ProviderContext(session_id=sid, tenant_id="default", task_id=tid, agent_id=aid)
    await mem.ingest(MemoryEvent(
        type=MemoryEventType.LLM_RESPONSE, address=scope, content="(raw turn, unfolded)",
        timestamp=_TS, role="assistant", metadata={"tool_calls": []},
    ), pctx)

    # ── Recover ────────────────────────────────────────────────────────────────
    await initialize_all(runtime)
    await runtime.rebuild_session(sid)
    await initialize_all(runtime)
    await recover_fixture_session(runtime, aid)

    # Drive to quiescence: repeatedly gather whatever background tasks the TM is
    # tracking (the relaunched recap, plus any recap the resumed task's own close
    # spawns) until the resumed task itself reaches a terminal status. Task 16 retired
    # SessionFinished (SessionRegistry's state machine was demoted to an agent registry
    # in Task 15), and with it every live path that could ever move the event-store
    # projection's session_status off RUNNING — polling on the task's own status is
    # the still-real completion signal for this loop.
    tm = runtime._task_managers.get(sid)
    for _ in range(100):
        if tm is not None and tm._background_asyncio_tasks:
            await asyncio.gather(*list(tm._background_asyncio_tasks), return_exceptions=True)
        view = await rebuild_view(runtime.event_store, sid)
        if view.tasks[tid].status in ("FINISHED", "FAILED", "CANCELED"):
            break
        await asyncio.sleep(0.02)

    # The SUSPENDED task was re-dispatched: a fresh TASK_STARTED was emitted for it
    # (recovery didn't just leave it parked) and it ran through to FINISHED.
    started_events = [
        e for e in seen
        if getattr(e, "type", None) == EventType.TASK_STARTED and e.task_id == tid
    ]
    assert started_events, "expected a re-dispatch TASK_STARTED for the resumed SUSPENDED task"
    view = await rebuild_view(runtime.event_store, sid)
    assert view.tasks[tid].status == "FINISHED", (
        f"expected resumed task to reach FINISHED, got {view.tasks[tid].status!r}"
    )

    # The interrupt-boundary recap pending at crash time was re-folded exactly once:
    # a OBSERVE_APPLIED closes it out — no duplicate fold, no hang.
    recap_done = [
        e for e in seen
        if getattr(e, "type", None) == EventType.OBSERVE_APPLIED
        and (e.payload or {}).get("task_id") == tid
    ]
    assert recap_done, "expected OBSERVE_APPLIED from the relaunched interrupt-boundary recap"

    # Recovery composed both threads to completion — the TM-held Session reaches a
    # terminal status (2026-09-04 Task 12 retired TaskQueueDrained, the aggregate
    # signal that used to feed the now-retired SessionFinished translation; see the
    # note above the polling loop — the verdict is written directly onto
    # ``tm.session.status`` now, not broadcast as an event).
    assert tm is not None, "expected a TaskManager to still be registered for the session"
    assert tm.is_done() and all(t.status in {"FINISHED", "FAILED", "CANCELED"} for t in tm.all_tasks()), (
        f"expected the session to reach a terminal status after recovery, got "
        f"{[(t.id, t.status) for t in tm.all_tasks()]!r}"
    )


@pytest.mark.asyncio
async def test_no_tasks_at_all_still_raises() -> None:
    """A session with no tasks at all is a genuinely empty/broken projection and must
    still raise RuntimeError (the empty-session guard)."""
    runtime, _mem, _seen = _make_runtime()
    aid = "agt_root"

    await append_one(runtime.event_store, _ev(
        1, EventType.SESSION_CREATED, user_prompt="do it",
        template_id="agent:tpl_echo", root_agent_id=aid))

    await initialize_all(runtime)
    await runtime.rebuild_session("ses_recap")
    await initialize_all(runtime)
    await runtime.recover_session("ses_recap")
    assert not runtime._task_managers["ses_recap"].all_tasks()


@pytest.mark.asyncio
async def test_all_tasks_terminal_with_a_pruned_snapshot_does_not_raise() -> None:
    """`test_no_tasks_at_all_still_raises` 的孪生反面：**所有 task 都已终态**的会话，
    在快照被裁过（v2，`prune_view_for_snapshot`）之后恢复，不得抛「no resumable tasks」。

    裁剪之后这类会话的 `view.tasks` 是空的——与「从来没有过 task」在集合上不可区分。
    闸门若按 `not all_tasks` 判，就会把一个正常完工的会话当成坏投影抛错，正是本文件
    docstring 里记的那个「resume 点了没反应」的老 bug。判据因此改用 `view.tasks_total`。

    ⚠️ 这条**必须有快照参与**：不写快照时 `rebuild_view` 走全量回放、`view.tasks` 是
    全量的，裁剪不参与，旧判据同样是绿的（上面那条 Test A 就是这样）。
    """
    from ctx_weft.core.control.reducers import prune_view_for_snapshot

    runtime, _mem, _seen = _make_runtime()
    sid, tid, aid = "ses_recap", "tsk_done", "agt_root"

    seed = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="agent:tpl_echo", root_agent_id=aid),
        _ev(2, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            # 崩溃恢复的种子：**无人值守**。这些用例驱动的是「进程死掉之后把 task 捡
            # 回来跑完」，没有人在等下一条消息；不标的话纯文本收尾会 park 等一个永远
            # 不来的人（判据自 2026-09-22 起是「有没有人在」）。
            "unattended": True,
            "assigned_agent_id": aid, "creator_agent_id": aid}),
        _ev(3, EventType.TASK_STARTED, task_id=tid, assigned_agent_id=aid),
        _ev(4, EventType.TASK_FINISHED, task_id=tid, outcome="success",
            summary="done", outputs={"result": "ok"}),
    ]
    for e in seed:
        await append_one(runtime.event_store, e)

    # 按生产口径写一张**裁过的**快照，并让它成为可用基底
    head = await runtime.event_store.committed_head(sid)
    pruned = prune_view_for_snapshot(await rebuild_view(runtime.event_store, sid))
    assert pruned.tasks == {}, "前置条件：唯一的 task 已终态，活闭包为空"
    assert pruned.tasks_total == 1
    await seed_snapshot(runtime.event_store, sid, pruned, cut=head, reason="test")

    # 恢复读到的就是那份空 tasks —— 闸门必须靠 tasks_total 认出「这不是坏投影」
    view = await rebuild_view(runtime.event_store, sid)
    assert view.tasks == {} and view.tasks_total == 1

    await initialize_all(runtime)
    await runtime.rebuild_session(sid)
    await initialize_all(runtime)
    await recover_fixture_session(runtime, aid)   # 不抛即通过（没有活可重排，但会话是好的）


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot", [False, True])
async def test_resume_uses_snapshot_or_bounded_history(tmp_path, snapshot) -> None:
    """Real pending recap: snapshots bound replay; no-snapshot replay has a fixed cut."""
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store
    from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
    from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime
    from tests.integration._session_cancel_worker import wait_until
    from tests.integration.test_lifecycle_crash_boundaries import run_worker

    await run_worker(tmp_path, "finish_park", "task_finalized")
    sid = "crash-session"
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            view = await rebuild_view(store, sid)
            aid = view.sessions[sid].root_agent_id
            assert view.pending_observe
            if snapshot:
                await seed_snapshot(store, sid, prune_view_for_snapshot(view),
                                    cut=await store.committed_head(sid))
            full_reads = []
            orig_range = store.read_range

            async def counting(session_id, **kwargs):
                # A narrow query for a known producer is not a projection replay.
                if (kwargs.get("after_position", 0) == 0
                        and not kwargs.get("include_types")):
                    full_reads.append(kwargs)
                return await orig_range(session_id, **kwargs)

            store.read_range = counting
            runtime, _ = make_lifecycle_runtime(store, memory, "finish_park")
            await initialize_all(runtime)
            await runtime.rebuild_session(sid, strict=True)
            # No snapshot requires the initial projection replay. A valid snapshot
            # must avoid unbounded replay even while a terminal task awaits recap.
            if snapshot:
                assert full_reads == []
            else:
                assert full_reads
            initial_reads = len(full_reads)
            tm = runtime._task_managers[sid]
            for _ in range(2):
                await recover_fixture_session(runtime, aid, session_id=sid)
                await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
                if snapshot:
                    assert len(full_reads) == initial_reads
                else:
                    assert all(read.get("through_position") is not None for read in full_reads)
            # Restore instrumentation before inspecting the final public result.
            store.read_range = orig_range
            assert not (await rebuild_view(store, sid)).pending_observe
            task_id = next(iter(view.tasks))
            task = (await runtime.get_task(sid, task_id)).data
            assert task.status == "FINISHED"
