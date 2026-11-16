
from tests._recovery import recover_fixture_session
"""Whole user journeys: interrupt, retract, edit, finish, restart and work again.

Only provider I/O and model latency are controlled. Runtime entry points, task
dispatch, observation, HITL, persistence and memory folding execute normally.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import LLMChunk, MemoryAddress, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_foreground_observe_interrupt import runtime
from tests.integration.test_send_message_after_session_done import (
    _FinishEveryTaskLLM,
    _is_background_observe,
)

SID = "whole-user-journey"


class JourneyLLM(_FinishEveryTaskLLM):
    def __init__(self, late=False):
        super().__init__()
        self.old_review = asyncio.Event()
        self.edit_started = asyncio.Event()
        self.retracted_started = asyncio.Event()
        self.release_recaps = asyncio.Event()
        if not late:
            self.release_recaps.set()
        self.closed = set()
        self.prompts = []
        self.initial_review = False
        self.actor_requests = []

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        text = str([m.content for m in request.messages])
        if "control__update_task_metadata" in names:
            return super().complete(request, stream)
        if "control__report_observation" in names:
            if _is_background_observe(request):
                self.prompts.append(text)
                async def recap():
                    await self.release_recaps.wait()
                    async for chunk in super(JourneyLLM, self).complete(request, stream):
                        yield chunk
                return recap()
            if not self.initial_review and "FINAL_REQUEST" not in text and "NEXT_TASK" not in text:
                self.initial_review = True
                return self.hold("initial_review", self.old_review, first=True)
            return super().complete(request, stream)
        self.actor_requests.append(text)
        if "WITHDRAWN_INPUT" in text:
            return self.hold("withdrawn", self.retracted_started, first=False)
        if "FINAL_REQUEST" in text or "NEXT_TASK" in text:
            self.release_recaps.set()
            return super().complete(request, stream)
        if "EDIT_ONE" in text:
            return self.hold("edit", self.edit_started, first=True)
        return super().complete(request, stream)

    def hold(self, label, entered, *, first):
        async def stream():
            try:
                entered.set()
                if first:
                    yield LLMChunk(kind="token", text=label + " work")
                await asyncio.Event().wait()
            finally:
                self.closed.add(label)
        return stream()


async def idle(rt):
    tm = rt._task_managers[SID]
    await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)


async def answer(rt, handle, entry, text):
    if entry == "send":
        return await rt.send_message(handle.agent_id, text, session_id=SID, unattended=True)
    req, = rt.hitl_registry.list_pending(session_id=SID)
    return await rt.reply_to_hitl(HitlReply(hitl_id=req.id, agent_id=handle.agent_id,
                                         outcome="accepted", message=text))


async def settled(rt, handle):
    try:
        await wait_until(lambda: rt._task_managers[SID].get_task(handle.task_id).status == "FINISHED")
    except TimeoutError:
        tm = rt._task_managers[SID]
        view = await rebuild_view(rt.event_store, SID)
        pytest.fail(json.dumps({"tasks": [(t.id, t.status) for t in tm.all_tasks()],
            "running_slots": sorted(tm.running_task_ids()), "dispatch": list(tm._dispatch_tasks),
            "rounds": list(tm.open_round_task_ids), "open_runs": list(view.open_runs),
            "pending_observe": list(view.pending_observe),
            "health": rt.storage_health(SID)}, ensure_ascii=False))
    await idle(rt)
    await handle.wait_for_finish(timeout=5)
    tm = rt._task_managers[SID]
    assert not tm.running_task_ids() and not tm.open_round_task_ids
    assert not rt._run_tokens.get(SID)
    view = await rebuild_view(rt.event_store, SID)
    assert not view.open_runs and not view.pending_observe and not view.hitl.pending


async def body(memory, task_id):
    return await load_records(memory, MemoryAddress(session_id=SID, task_id=task_id),
                                  MemoryScope.TASK, ProviderContext(session_id=SID))


async def assert_event_contract(store, sid):
    from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles
    rows = await store.read_range(sid)
    assert_lifecycles(rows)
    starts = {r.event.run_id: r for r in rows if r.event.type == "RunStarted"}
    finishes = {}
    for row in rows:
        if row.event.type == "RunFinished":
            finishes.setdefault(row.event.run_id, []).append(row)
    assert starts.keys() == finishes.keys()
    for run, started in starts.items():
        assert len(finishes[run]) == 1 and started.position < finishes[run][0].position
    for terminal in [r for r in rows if r.event.type in {"TaskFinished", "TaskCanceled", "TaskFailed"}]:
        for run, started in starts.items():
            if started.event.task_id == terminal.event.task_id and started.event.origin == "runtime":
                assert finishes[run][0].position < terminal.position


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["send", "reply"])
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("late", [False, True])
async def test_repeated_edit_retract_finish_restart_and_next_task(tmp_path, entry, short, late):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            old_report_waiting, release_old_report = asyncio.Event(), asyncio.Event()
            hold_old = late and short and entry == "reply"
            append = store.append_batch

            async def slow_old_report(sid, bid, events, *args, **kwargs):
                if hold_old and not old_report_waiting.is_set() and any(
                        e.type == "ObserveCompleted" and e.origin == "loop.background_observe" for e in events):
                    old_report_waiting.set()
                    await release_old_report.wait()
                return await append(sid, bid, events, *args, **kwargs)

            store.append_batch = slow_old_report
            llm = JourneyLLM(late)
            rt = runtime(store, memory, llm, short)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="INITIAL_REQUEST", unattended=True))
            await asyncio.wait_for(llm.old_review.wait(), 10)
            await rt.pause_agent(h.agent_id)
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm._dispatch_tasks)
            if hold_old:
                await asyncio.wait_for(old_report_waiting.wait(), 10)
            first, = rt.hitl_registry.list_pending(session_id=SID)
            if hold_old and entry == "reply":
                reply = asyncio.create_task(answer(rt, h, entry, "WITHDRAWN_INPUT"))
                await asyncio.sleep(0)
                assert not reply.done()
                release_old_report.set()
                await asyncio.wait_for(reply, 10)
            else:
                await answer(rt, h, entry, "WITHDRAWN_INPUT")
            await asyncio.wait_for(llm.retracted_started.wait(), 10)
            await rt.pause_agent(h.agent_id)
            await wait_until(lambda: not tm._dispatch_tasks)
            pending, = rt.hitl_registry.list_pending(session_id=SID)
            assert pending.id == first.id
            await answer(rt, h, entry, "EDIT_ONE")
            await asyncio.wait_for(llm.edit_started.wait(), 10)
            await wait_until(lambda: not tm.open_round_task_ids)
            await rt.pause_agent(h.agent_id)
            await wait_until(lambda: not tm._dispatch_tasks)
            if hold_old:
                registry = rt._step_executions[SID]
                await wait_until(lambda: registry.pending_run_id(h.task_id) is None)
                if entry == "reply":
                    assert release_old_report.is_set()
                else:
                    assert registry.tasks  # send can proceed while the old report is held.
            await answer(rt, h, entry, "FINAL_REQUEST")
            release_old_report.set()
            await settled(rt, h)
            await assert_event_contract(store, SID)
            assert llm.closed == {"initial_review", "withdrawn", "edit"}
            assert all("WITHDRAWN_INPUT" not in p and "FINAL_REQUEST" not in p for p in llm.prompts)
            rows = await store.read_range(SID)
            assert not any("WITHDRAWN_INPUT" in str(r.event.payload) for r in rows)
            records = await body(memory, h.task_id)
            users = [str(r.content) for r in records if r.role == "user"]
            assert any("FINAL_REQUEST" in s for s in users)
            assert any("EDIT_ONE" in s and "INITIAL_REQUEST" in s for s in users)
            assert not any("WITHDRAWN_INPUT" in str(r.content) for r in records)
            before = [(r.id, r.content) for r in records]
            head = await store.committed_head(SID)
            cold = runtime(store, memory, _FinishEveryTaskLLM(), short)
            await initialize_all(cold)
            await cold.rebuild_session(SID, strict=True)
            await recover_fixture_session(cold, h.agent_id, session_id=SID)
            await idle(cold)
            assert await store.committed_head(SID) == head
            assert [(r.id, r.content) for r in await body(memory, h.task_id)] == before
            nxt = await cold.send_message(h.agent_id, "NEXT_TASK", session_id=SID, unattended=True)
            assert nxt.task_id != h.task_id
            await settled(cold, nxt)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["send", "reply"])
async def test_immediate_input_from_confirmed_pause_notice_remains_retractable(tmp_path, entry):
    """Hosts may post either API as soon as they display the confirmed pause."""
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = JourneyLLM()
            rt = runtime(store, memory, llm, False)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="INITIAL_REQUEST", unattended=True))
            await asyncio.wait_for(llm.old_review.wait(), 10)
            old = []
            rejected = []
            attempted = asyncio.Event()
            async def receive(event):
                if old:
                    return
                old.append(event.payload["hitl_id"])
                from ctx_weft.core.models.errors import AgentBusyError
                try:
                    await answer(rt, h, entry, "WITHDRAWN_INPUT")
                except AgentBusyError:
                    # send_message explicitly rejects running agents. The host
                    # retries after AgentInterrupted/TaskAwaitingHuman arrives.
                    rejected.append(True)
                finally:
                    attempted.set()
            sub = rt.event_bus.subscribe("HitlOpened", receive)
            try:
                await rt.pause_agent(h.agent_id)
                await asyncio.wait_for(attempted.wait(), 10)
                if rejected:
                    await wait_until(lambda: not rt._task_managers[SID]._dispatch_tasks)
                    await answer(rt, h, entry, "WITHDRAWN_INPUT")
                await asyncio.wait_for(llm.retracted_started.wait(), 10)
                await rt.pause_agent(h.agent_id)
                await idle(rt)
                pending, = rt.hitl_registry.list_pending(session_id=SID)
                assert pending.id == old[0]
                assert not any("WITHDRAWN_INPUT" in str(r.event.payload) for r in await store.read_range(SID))
                await answer(rt, h, entry, "FINAL_REQUEST")
                await settled(rt, h)
                nxt = await rt.send_message(h.agent_id, "NEXT_TASK", session_id=SID, unattended=True)
                await settled(rt, nxt)
            finally:
                await sub.unsubscribe()
                await rt.cancel_session(SID)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["send", "reply"])
@pytest.mark.parametrize("fault,action", [
    ("before", "resume"), ("unknown", "resume"), ("memory", "resume"),
    ("before", "cancel"), ("unknown", "cancel"), ("memory", "cancel"),
])
async def test_resume_after_input_commit_failure_finishes_and_accepts_next_work(tmp_path, entry, fault, action):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = JourneyLLM()
            rt = runtime(store, memory, llm, False)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="INITIAL_REQUEST", unattended=True))
            await asyncio.wait_for(llm.old_review.wait(), 10)
            await rt.pause_agent(h.agent_id)
            await idle(rt)
            original_append, original_ingest = store.append_batch, memory.ingest
            failures = []
            async def append(sid, bid, events, *args, **kwargs):
                if (fault != "memory"
                        and any("FINAL_REQUEST" in str(e.payload) for e in events)):
                    failures.append(bid)
                    if fault == "unknown":
                        await original_append(sid, bid, events, *args, **kwargs)
                    raise OSError("input commit receipt unavailable")
                return await original_append(sid, bid, events, *args, **kwargs)
            async def ingest(event, ctx):
                if fault == "memory" and "FINAL_REQUEST" in str(event.content):
                    failures.append(event.id)
                    raise OSError("memory temporarily unavailable")
                return await original_ingest(event, ctx)
            store.append_batch, memory.ingest = append, ingest
            await answer(rt, h, entry, "FINAL_REQUEST")
            tm = rt._task_managers[SID]
            await wait_until(lambda: bool(failures) and not tm._dispatch_tasks)
            for _ in range(2):
                rt.clear_storage_isolation(SID)
                try:
                    await recover_fixture_session(rt, h.agent_id, session_id=SID)
                except Exception as exc:
                    from ctx_weft.protocols.events import PersistenceUnavailableError
                    assert isinstance(exc, OSError if fault == "memory" else PersistenceUnavailableError)
                    assert not tm._dispatch_tasks and h.task_id in tm.running_task_ids()
                await wait_until(lambda: not tm._queue.has_pending() or rt.storage_health(SID) is not None)
                await wait_until(lambda: not tm._dispatch_tasks)
            # Storage has recovered. A normal /resume must finish the accepted
            # input without requiring another message or losing the occupied slot.
            store.append_batch, memory.ingest = original_append, original_ingest
            rt.clear_storage_isolation(SID)
            if action == "cancel":
                await rt.cancel_agent(h.agent_id)
                await idle(rt)
                assert tm.get_task(h.task_id).status == "CANCELED"
                assert not tm.running_task_ids() and not tm.open_round_task_ids
                rows = await store.read_range(SID)
                canceled, = [r for r in rows if r.event.type == "TaskCanceled" and r.event.task_id == h.task_id]
                for started in [r for r in rows if r.event.type == "RunStarted" and r.event.origin == "runtime"]:
                    finished, = [r for r in rows if r.event.type == "RunFinished" and r.event.run_id == started.event.run_id]
                    assert started.position < finished.position < canceled.position
                view = await rebuild_view(store, SID)
                assert not view.open_runs and not view.pending_observe and not view.hitl.pending
                return
            await recover_fixture_session(rt, h.agent_id, session_id=SID)
            await settled(rt, h)
            records = await body(memory, h.task_id)
            assert sum("FINAL_REQUEST" in str(r.content) for r in records if r.role == "user") == 1
            nxt = await rt.send_message(h.agent_id, "NEXT_TASK", session_id=SID, unattended=True)
            await settled(rt, nxt)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["send", "reply"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("recovery_fault", ["none", "crash_again", "memory_write"])
async def test_cold_input_recovery_preserves_request_through_to_next_task(tmp_path, entry, snapshot, recovery_fault):
    worker = Path(__file__).with_name("_user_journey_audit_worker.py")
    for mode in ["source", *(["recover"] if recovery_fault == "crash_again" else [])]:
        proc = await asyncio.to_thread(subprocess.run,
            [sys.executable, str(worker), str(tmp_path), entry, "snapshot" if snapshot else "no_snapshot", mode],
            cwd=worker.parents[2], capture_output=True, text=True, timeout=40)
        assert proc.returncode == 73, proc.stdout + proc.stderr
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = JourneyLLM()
            llm.initial_review = True
            rt = runtime(store, memory, llm, False)
            original = memory.ingest
            failed = []
            async def ingest(event, ctx):
                if not failed and "FINAL_REQUEST" in str(event.content):
                    failed.append(event.id)
                    raise OSError("memory writes unavailable during materialization")
                return await original(event, ctx)
            if recovery_fault == "memory_write":
                memory.ingest = ingest
            await initialize_all(rt)
            try:
                await rt.rebuild_session(SID, strict=True)
            except OSError:
                # Refusing incomplete materialization is safe; retry once the
                # memory provider is writable, then finish the same request.
                assert recovery_fault == "memory_write" and failed
                memory.ingest = original
                await rt.rebuild_session(SID, strict=True)
            if recovery_fault != "memory_write":
                memory.ingest = original
            tm = rt._task_managers[SID]
            task, = tm.all_tasks()
            aid = tm.session.root_agent_id
            try:
                await recover_fixture_session(rt, aid, session_id=SID)
            except OSError:
                assert recovery_fault == "memory_write" and failed
                assert not tm._dispatch_tasks and not llm.actor_requests
                assert not rt._memory_settled(SID)
                await tm.drain()  # An unrelated settlement cannot bypass recovery.
                assert not tm._dispatch_tasks and not llm.actor_requests
                memory.ingest = original
                await recover_fixture_session(rt, aid, session_id=SID)
            h = type("Handle", (), {"task_id": task.id, "agent_id": aid})()
            await wait_until(lambda: task.status == "FINISHED")
            await idle(rt)
            assert llm.actor_requests and all("FINAL_REQUEST" in p for p in llm.actor_requests)
            records = await body(memory, task.id)
            assert sum("FINAL_REQUEST" in str(r.content) for r in records if r.role == "user") == 1
            head = await store.committed_head(SID)
            before = [(r.id, r.content) for r in records]
            again = runtime(store, memory, _FinishEveryTaskLLM(), False)
            await initialize_all(again)
            await again.rebuild_session(SID, strict=True)
            await recover_fixture_session(again, h.agent_id, session_id=SID)
            await idle(again)
            assert await store.committed_head(SID) == head
            assert [(r.id, r.content) for r in await body(memory, task.id)] == before
            nxt = await again.send_message(aid, "NEXT_TASK", session_id=SID, unattended=True)
            await settled(again, nxt)
            await assert_event_contract(store, SID)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_fault", ["before", "unknown"])
async def test_uncertain_input_then_repeated_finish_failure_retains_original_run(tmp_path, finish_fault):
    from copy import deepcopy

    from ctx_weft.protocols.events import PersistenceUnavailableError

    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = JourneyLLM()
            rt = runtime(store, memory, llm, False)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="INITIAL_REQUEST", unattended=True))
            await asyncio.wait_for(llm.old_review.wait(), 10)
            await rt.pause_agent(h.agent_id)
            await idle(rt)
            tm = rt._task_managers[SID]
            original = store.append_batch
            input_failed = []
            finish_attempts = []
            finish_available = asyncio.Event()

            async def append(sid, bid, events, *args, **kwargs):
                if not input_failed and any("FINAL_REQUEST" in str(e.payload) for e in events):
                    input_failed.append(bid)
                    await original(sid, bid, events, *args, **kwargs)
                    raise OSError("input receipt lost")
                if (not finish_available.is_set() and any(
                        e.type == "RunFinished" and e.origin == "runtime" for e in events)):
                    finish_attempts.append((bid, deepcopy(events)))
                    if finish_fault == "unknown":
                        await original(sid, bid, events, *args, **kwargs)
                    raise OSError("finish receipt unavailable")
                return await original(sid, bid, events, *args, **kwargs)

            store.append_batch = append
            await answer(rt, h, "send", "FINAL_REQUEST")
            await wait_until(lambda: input_failed and not tm._dispatch_tasks)
            pending = rt._pending_actor_runs[(SID, h.task_id)]
            original_finish = pending.finished.id
            for _ in range(2):
                rt.clear_storage_isolation(SID)
                with pytest.raises(PersistenceUnavailableError):
                    await recover_fixture_session(rt, h.agent_id, session_id=SID)
                assert rt._pending_actor_runs[(SID, h.task_id)].finished.id == original_finish
                assert h.task_id in tm.running_task_ids() and not tm._dispatch_tasks
            assert len(finish_attempts) >= 2
            assert all(attempt == finish_attempts[0] for attempt in finish_attempts)
            finish_available.set()
            store.append_batch = original
            rt.clear_storage_isolation(SID)
            await recover_fixture_session(rt, h.agent_id, session_id=SID)
            await settled(rt, h)
            assert (SID, h.task_id) not in rt._pending_actor_runs
            await assert_event_contract(store, SID)
            nxt = await rt.send_message(h.agent_id, "NEXT_TASK", session_id=SID, unattended=True)
            await settled(rt, nxt)
