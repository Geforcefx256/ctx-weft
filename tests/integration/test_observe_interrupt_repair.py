
from tests._recovery import recover_fixture_session
"""Public-entry regressions for interrupted reviews, immutable input and hot recovery."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import LLMChunk, MemoryAddress, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_foreground_observe_interrupt import SID, InterruptedReviewLLM, runtime

_NORMAL_SEGMENTS = {}


async def task_body(memory, task_id):
    records = await load_records(memory, MemoryAddress(session_id=SID, task_id=task_id),
        MemoryScope.TASK, ProviderContext(session_id=SID))
    return [(str(r.kind), r.role, str(r.content)) for r in records]


async def normal_segment(tmp_path_factory, short, new_input):
    """Capture the same interrupted segment without crashes, including late new input."""
    key = (short, new_input)
    if key in _NORMAL_SEGMENTS:
        return _NORMAL_SEGMENTS[key]
    path = tmp_path_factory.mktemp("normal_observe_segment")
    async with open_sqlite_event_store(path / "e.db") as store:
        async with open_sqlite_memory(path / "m.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, short)
            h = await start(rt, llm)
            await rt.pause_agent(h.agent_id)
            await drained(rt)
            if new_input:
                entered = asyncio.Event()
                original = llm.complete
                def complete(request, stream=True):
                    if not {t.name for t in request.tools}.intersection(
                            {"control__report_observation", "control__update_task_metadata"}):
                        async def hold():
                            entered.set()
                            yield LLMChunk(kind="token", text="new work")
                            await asyncio.Event().wait()
                        return hold()
                    return original(request, stream)
                llm.complete = complete
                await request_turn(rt, h, "send")
                await asyncio.wait_for(entered.wait(), 10)
                await wait_until(lambda: not rt._task_managers[SID].open_round_task_ids)
            result = await task_body(memory, h.task_id)
            await rt.cancel_session(SID)
            await drained(rt)
    _NORMAL_SEGMENTS[key] = result
    return result


async def drained(rt, sid=SID):
    tm = rt._task_managers[sid]
    await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)


async def start(rt, llm):
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(session_id=SID,
        template_id="agent:tpl_echo", user_prompt="OLD REQUEST", unattended=True))
    await asyncio.wait_for(llm.review_started.wait(), 10)
    return handle


async def request_turn(rt, handle, entry, text="NEW REQUEST SECRET"):
    if entry == "send":
        await rt.send_message(handle.agent_id, text, session_id=SID, unattended=True)
    else:
        req, = rt.hitl_registry.list_pending(session_id=SID)
        await rt.reply_to_hitl(HitlReply(hitl_id=req.id, agent_id=handle.agent_id,
                                       outcome="accepted", message=text))


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("entry", ["send", "reply"])
async def test_handoff_hot_retries_complete_stopped_actor(tmp_path, unknown, entry):
    async with open_sqlite_event_store(tmp_path / "e.db") as store:
        async with open_sqlite_memory(tmp_path / "m.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, False)
            handle = await start(rt, llm)
            original = store.append_batch
            attempts, signatures = 0, []

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal attempts
                if any(e.type == "HitlOpened" for e in events):
                    attempts += 1
                    signatures.append((bid, [(e.id, e.payload, e.timestamp) for e in events]))
                    if attempts <= 3:
                        if unknown:
                            await original(sid, bid, events, *args, **kwargs)
                        raise OSError("handoff receipt unavailable")
                return await original(sid, bid, events, *args, **kwargs)

            store.append_batch = append
            await rt.pause_agent(handle.agent_id)
            tm = rt._task_managers[SID]
            await wait_until(lambda: rt.storage_health(SID) is not None and not tm._dispatch_tasks)
            assert not rt.hitl_registry.list_pending(session_id=SID)
            assert handle.task_id in tm.running_task_ids()
            assert not rt.clear_storage_isolation(SID)
            # Resume now runs the bounded storage coordinator before taking the
            # session guard, confirming the same envelopes across all attempts.
            await recover_fixture_session(rt, handle.agent_id, session_id=SID)
            store.append_batch = original
            await drained(rt)
            assert llm.review_closed.is_set()
            assert signatures and all(s == signatures[0] for s in signatures)
            assert not tm.running_task_ids()
            view = await rebuild_view(store, SID)
            assert view.tasks[handle.task_id].status == "AWAITING_HUMAN"
            assert len(view.hitl.pending) == 1 and not view.pending_observe and not view.open_runs
            rows = await store.read_range(SID)
            opened, = [r for r in rows if r.event.type == "HitlOpened"]
            actor_finish, = [r for r in rows if r.event.type == "RunFinished" and r.event.origin == "runtime"]
            waiting, = [r for r in rows if r.event.type == "TaskAwaitingHuman"]
            assert opened.position < actor_finish.position < waiting.position
            head = await store.committed_head(SID)
            await recover_fixture_session(rt, handle.agent_id, session_id=SID)
            assert await store.committed_head(SID) == head
            await request_turn(rt, handle, entry)
            await wait_until(lambda: tm.get_task(handle.task_id).status == "FINISHED")
            await drained(rt)
            records = await load_records(memory, MemoryAddress(session_id=SID, agent_id=handle.agent_id,
                task_id=handle.task_id), MemoryScope.TASK, ProviderContext(session_id=SID))
            assert any(r.role == "user" and "NEW REQUEST SECRET" in str(r.content)
                       and "OLD REQUEST" in str(r.content) for r in records)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["send", "reply"])
async def test_late_recap_assembly_cannot_read_new_turn(tmp_path, monkeypatch, entry):
    from ctx_weft.core.assembler.assembler import ContextAssembler
    original_assemble = ContextAssembler.assemble
    entered, release, captured, new_started = [asyncio.Event() for _ in range(4)]
    seen = []

    async def assemble(self, request):
        if (request.extra or {}).get("observe_boundary") == "interrupt":
            entered.set()
            await release.wait()
        return await original_assemble(self, request)

    monkeypatch.setattr(ContextAssembler, "assemble", assemble)
    async with open_sqlite_event_store(tmp_path / "e.db") as store:
        async with open_sqlite_memory(tmp_path / "m.db") as memory:
            llm = InterruptedReviewLLM()
            original_complete = llm.complete

            def complete(request, stream=True):
                names = {t.name for t in request.tools}
                if "control__report_observation" in names and entered.is_set():
                    seen.append(str([m.content for m in request.messages]))
                    captured.set()
                if (not names.intersection({"control__report_observation", "control__update_task_metadata"})
                        and any("NEW REQUEST SECRET" in str(m.content) for m in request.messages)):
                    async def hold():
                        new_started.set()
                        yield LLMChunk(kind="token", text="new work")
                        await asyncio.Event().wait()
                    return hold()
                return original_complete(request, stream)

            llm.complete = complete
            rt = runtime(store, memory, llm, False)
            handle = await start(rt, llm)
            await rt.pause_agent(handle.agent_id)
            await asyncio.wait_for(entered.wait(), 10)
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm._dispatch_tasks)
            await request_turn(rt, handle, entry)
            await asyncio.wait_for(new_started.wait(), 10)
            await wait_until(lambda: not tm.open_round_task_ids)
            release.set()
            await asyncio.wait_for(captured.wait(), 10)
            assert seen and all("NEW REQUEST SECRET" not in text and "OLD REQUEST" in text for text in seen)
            await rt.cancel_session(SID)
            await drained(rt)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["pause", "cancel"])
async def test_recovered_foreground_has_real_cancel_lifecycle(tmp_path, stop):
    worker = Path(__file__).with_name("_lifecycle_crash_worker.py")
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(worker), str(tmp_path), "foreground", "observe_started"],
        cwd=worker.parents[2], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 73, proc.stderr
    sid = "crash-session"
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, False)
            await initialize_all(rt)
            await rt.rebuild_session(sid, strict=True)
            tm = rt._task_managers[sid]
            aid = tm.session.root_agent_id
            await recover_fixture_session(rt, aid, session_id=sid)
            await asyncio.wait_for(llm.review_started.wait(), 10)
            assert rt._agent_lifecycle_manager.status_of(aid) == "running"
            assert rt._run_tokens.get(sid)
            if stop == "pause":
                assert await rt.pause_agent(aid) == [aid]
            else:
                await rt.cancel_agent(aid)
            await asyncio.wait_for(llm.review_closed.wait(), 10)
            await drained(rt, sid)
            assert not tm.running_task_ids() and not rt._run_tokens.get(sid)
            view = await rebuild_view(store, sid)
            assert not view.open_runs and not view.pending_observe
            task, = tm.all_tasks()
            assert task.status == ("AWAITING_HUMAN" if stop == "pause" else "CANCELED")


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
@pytest.mark.parametrize("stage", ["TaskStarted", "RunStarted", "RunFinished"])
async def test_recovered_foreground_retries_original_run_envelopes(tmp_path, unknown, stage):
    worker = Path(__file__).with_name("_lifecycle_crash_worker.py")
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(worker), str(tmp_path), "foreground", "observe_started"],
        cwd=worker.parents[2], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 73, proc.stderr
    sid = "crash-session"
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, False)
            await initialize_all(rt)
            await rt.rebuild_session(sid, strict=True)
            tm = rt._task_managers[sid]
            aid = tm.session.root_agent_id
            original = store.append_batch
            signatures = []
            async def append(session_id, bid, events, *args, **kwargs):
                if any(e.type == stage and (stage == "TaskStarted" or e.origin == "runtime") for e in events):
                    signatures.append((bid, [(e.id, e.payload, e.timestamp) for e in events]))
                    if len(signatures) <= 3:
                        if unknown:
                            await original(session_id, bid, events, *args, **kwargs)
                        raise OSError("recovered actor receipt unavailable")
                return await original(session_id, bid, events, *args, **kwargs)
            store.append_batch = append
            await recover_fixture_session(rt, aid, session_id=sid)
            if stage == "RunFinished":
                await asyncio.wait_for(llm.review_started.wait(), 10)
                await rt.pause_agent(aid)
            await wait_until(lambda: rt.storage_health(sid) is not None and not tm._dispatch_tasks)
            assert not rt.clear_storage_isolation(sid)
            await recover_fixture_session(rt, aid, session_id=sid)
            store.append_batch = original
            assert len(signatures) >= 4 and all(s == signatures[0] for s in signatures[:4])
            if stage != "RunFinished":
                await asyncio.wait_for(llm.review_started.wait(), 10)
                await rt.pause_agent(aid)
            await drained(rt, sid)
            assert llm.review_closed.is_set() and not tm.running_task_ids()
            rows = await store.read_range(sid)
            for started in [r for r in rows if r.event.type == "RunStarted"]:
                finished, = [r for r in rows if r.event.type == "RunFinished"
                             and r.event.run_id == started.event.run_id]
                assert started.position < finished.position
            assert not (await rebuild_view(store, sid)).open_runs
            head = await store.committed_head(sid)
            await recover_fixture_session(rt, aid, session_id=sid)
            assert await store.committed_head(sid) == head


@pytest.mark.asyncio
async def test_cancel_during_recovered_foreground_registration_is_not_lost(tmp_path):
    worker = Path(__file__).with_name("_lifecycle_crash_worker.py")
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(worker), str(tmp_path), "foreground", "observe_started"],
        cwd=worker.parents[2], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 73, proc.stderr
    sid = "crash-session"
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, False)
            await initialize_all(rt)
            await rt.rebuild_session(sid, strict=True)
            tm = rt._task_managers[sid]
            aid = tm.session.root_agent_id
            entered, release = asyncio.Event(), asyncio.Event()
            original = store.append_batch
            async def append(session_id, bid, events, *args, **kwargs):
                result = await original(session_id, bid, events, *args, **kwargs)
                if any(e.type == "TaskStarted" for e in events):
                    entered.set()
                    await release.wait()
                return result
            store.append_batch = append
            await recover_fixture_session(rt, aid, session_id=sid)
            await asyncio.wait_for(entered.wait(), 10)
            await rt.cancel_agent(aid)
            release.set()
            await drained(rt, sid)
            assert not llm.review_started.is_set() and not rt._run_tokens.get(sid)
            task, = tm.all_tasks()
            assert task.status == "CANCELED" and not tm.running_task_ids()
            view = await rebuild_view(store, sid)
            assert not view.open_runs and not view.pending_observe


@pytest.mark.asyncio
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("entry", ["send", "reply"])
@pytest.mark.parametrize("report_first", [False, True])
async def test_same_agent_child_has_durable_recap_and_new_turn(tmp_path, monkeypatch, short, entry, report_first):
    from ctx_weft.core.loop.steps.observe import ObserveStep
    from tests.integration.test_dispatch_boundary_recap_e2e import _is_background_recap, _RouterLLM
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )
    entered, release, new_actor = asyncio.Event(), asyncio.Event(), asyncio.Event()
    body = ObserveStep._background_body
    async def hold_report(self, state, ctx, boundary, judging):
        if boundary == "interrupt":
            entered.set()
            await release.wait()
        return await body(self, state, ctx, boundary, judging)
    monkeypatch.setattr(ObserveStep, "_background_body", hold_report)

    class ChildReviewLLM(_RouterLLM):
        def __init__(self):
            super().__init__()
            self.started, self.closed = asyncio.Event(), asyncio.Event()
            self.reviewed = False

        def complete(self, request, stream=True):
            if (not {t.name for t in request.tools}.intersection(
                    {"control__report_observation", "control__update_task_metadata"})
                    and any("NEW REQUEST SECRET" in str(m.content) for m in request.messages)):
                new_actor.set()
            if ("control__report_observation" in {t.name for t in request.tools}
                    and not _is_background_recap(request) and not self.reviewed):
                self.reviewed = True
                async def hold():
                    try:
                        self.started.set()
                        yield LLMChunk(kind="token", text="child judgement")
                        await asyncio.Event().wait()
                    finally:
                        self.closed.set()
                return hold()
            return super().complete(request, stream)

    async with open_sqlite_event_store(tmp_path / "e.db") as store:
        async with open_sqlite_memory(tmp_path / "m.db") as memory:
            provider = InlineAgentTemplateProvider()
            template = make_echo_template()
            template.loop_config.short_segment_token_threshold = 100000 if short else 0
            provider.register(template)
            llm = ChildReviewLLM()
            rt = make_runtime(llm=llm, agent_provider=provider, event_store=store)
            rt.providers.register_memory(memory)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="ROOT DELEGATE", unattended=True))
            await asyncio.wait_for(llm.started.wait(), 10)
            tm = rt._task_managers[SID]
            child = next(t for t in tm.all_tasks() if t.parent_task_id)
            assert child.creator_agent_id == child.assigned_agent_id
            await rt.pause_agent(h.agent_id)
            await asyncio.wait_for(llm.closed.wait(), 10)
            await asyncio.wait_for(entered.wait(), 10)
            await wait_until(lambda: not tm._dispatch_tasks)
            assert tm.get_task(h.task_id).status == "SUSPENDED"
            assert child.status == "AWAITING_HUMAN"
            rows = await store.read_range(SID)
            opened, = [r for r in rows if r.event.type == "HitlOpened" and r.event.task_id == child.id]
            intent = opened.event.payload["observe_request"]
            assert intent["boundary"] == "interrupt" and not intent["judge_task"]
            assert "context_record_ids" in intent
            if report_first:
                release.set()
                await drained(rt)
            await request_turn(rt, h, entry)
            if not report_first:
                await asyncio.wait_for(new_actor.wait(), 10)
                release.set()
            await wait_until(lambda: tm.get_task(h.task_id).status == "FINISHED")
            await drained(rt)
            assert child.status == "FINISHED" and not tm.running_task_ids()
            view = await rebuild_view(store, SID)
            assert not view.pending_observe and not view.open_runs
            rows = await store.read_range(SID)
            for event_type in ("ObserveStarted", "ObserveCompleted", "ObserveApplied"):
                assert len([r for r in rows if r.event.type == event_type
                            and r.event.payload.get("observe_id") == intent["observe_id"]]) == 1
            users = [text for _, role, text in await task_body(memory, child.id) if role == "user"]
            assert any("NEW REQUEST SECRET" in text and "do the delegated work" in text for text in users)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["send", "reply"])
async def test_replacement_before_first_chunk_can_be_retracted_twice(tmp_path, entry):
    async with open_sqlite_event_store(tmp_path / "e.db") as store:
        async with open_sqlite_memory(tmp_path / "m.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, False)
            h = await start(rt, llm)
            await rt.pause_agent(h.agent_id)
            await drained(rt)
            original = llm.complete
            started = asyncio.Event()

            def complete(request, stream=True):
                if not {t.name for t in request.tools}.intersection(
                        {"control__report_observation", "control__update_task_metadata"}):
                    async def hold():
                        started.set()
                        await asyncio.Event().wait()
                        yield LLMChunk(kind="token", text="unreachable")
                    return hold()
                return original(request, stream)

            llm.complete = complete
            for _ in range(2):
                started.clear()
                await request_turn(rt, h, entry, "RETRACT THIS")
                await asyncio.wait_for(started.wait(), 10)
                await rt.pause_agent(h.agent_id)
                await drained(rt)
                tm = rt._task_managers[SID]
                assert not tm.running_task_ids()
                assert len(rt.hitl_registry.list_pending(session_id=SID)) == 1
                rows = await store.read_range(SID)
                assert not any("RETRACT THIS" in str(r.event.payload) for r in rows)
            llm.complete = original
            await request_turn(rt, h, entry, "FINAL NEW REQUEST")
            await wait_until(lambda: rt._task_managers[SID].get_task(h.task_id).status == "FINISHED")
            await drained(rt)
            records = await load_records(memory, MemoryAddress(session_id=SID, agent_id=h.agent_id,
                task_id=h.task_id), MemoryScope.TASK, ProviderContext(session_id=SID))
            users = [str(r.content) for r in records if r.role == "user"]
            assert any("FINAL NEW REQUEST" in text and "OLD REQUEST" in text for text in users)
            assert not any("RETRACT THIS" in text for text in users)


@pytest.mark.asyncio
async def test_reply_on_open_notification_does_not_commit_the_next_round(tmp_path):
    """The host can answer a confirmed bubble before the old actor releases its slot."""
    async with open_sqlite_event_store(tmp_path / "e.db") as store:
        async with open_sqlite_memory(tmp_path / "m.db") as memory:
            llm = InterruptedReviewLLM()
            original = llm.complete
            new_started = asyncio.Event()
            def complete(request, stream=True):
                names = {t.name for t in request.tools}
                if (not names.intersection({"control__report_observation", "control__update_task_metadata"})
                        and any("UNCOMMITTED REPLACEMENT" in str(m.content) for m in request.messages)):
                    async def hold():
                        new_started.set()
                        await asyncio.Event().wait()
                        yield LLMChunk(kind="token", text="unreachable")
                    return hold()
                return original(request, stream)
            llm.complete = complete
            rt = runtime(store, memory, llm, False)
            h = await start(rt, llm)
            bubbles = []
            async def answer(event):
                if bubbles:
                    return
                bubbles.append(event.payload["hitl_id"])
                await rt.reply_to_hitl(HitlReply(hitl_id=bubbles[0], agent_id=h.agent_id,
                    outcome="accepted", message="UNCOMMITTED REPLACEMENT"))
            subscription = rt.event_bus.subscribe("HitlOpened", answer)
            await rt.pause_agent(h.agent_id)
            await asyncio.wait_for(new_started.wait(), 10)
            await rt.pause_agent(h.agent_id)
            await drained(rt)
            await subscription.unsubscribe()
            pending, = rt.hitl_registry.list_pending(session_id=SID)
            assert pending.id == bubbles[0]
            assert not (await rebuild_view(store, SID)).open_runs
            assert not any("UNCOMMITTED REPLACEMENT" in str(r.event.payload)
                           for r in await store.read_range(SID))
            await rt.cancel_session(SID)
            await drained(rt)


@pytest.mark.asyncio
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("first,second", [
    ("before", "recover_handoff"), ("handoff", None),
    ("handoff", "recover_completed"), ("handoff", "recover_applied"),
    ("late", None), ("late", "recover_completed"), ("late", "recover_applied"),
])
async def test_handoff_process_crashes_and_legacy_snapshot(tmp_path, tmp_path_factory, first, second, short):
    worker = Path(__file__).with_name("_observe_interrupt_repair_worker.py")
    for cut in [first, *([second] if second else [])]:
        proc = await asyncio.to_thread(subprocess.run,
            [sys.executable, str(worker), str(tmp_path), cut, "short" if short else "long"],
            cwd=worker.parents[2], capture_output=True, text=True, timeout=35)
        assert proc.returncode == 73, proc.stdout + proc.stderr
    async with open_sqlite_event_store(tmp_path / "e.db") as store:
        async with open_sqlite_memory(tmp_path / "m.db") as memory:
            from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM
            llm = _FinishEveryTaskLLM()
            seen = []
            original = llm.complete
            def complete(request, stream=True):
                if "control__report_observation" in {t.name for t in request.tools}:
                    seen.append(str([m.content for m in request.messages]))
                return original(request, stream)
            llm.complete = complete
            rt = runtime(store, memory, llm, short)
            await initialize_all(rt)
            await rt.rebuild_session(SID, strict=True)
            tm = rt._task_managers[SID]
            tm._max_concurrent = 0
            await recover_fixture_session(rt, tm.session.root_agent_id, session_id=SID)
            await drained(rt)
            assert all("NEW REQUEST SECRET" not in text for text in seen)
            view = await rebuild_view(store, SID)
            assert not view.open_runs and not view.pending_observe
            assert not tm.running_task_ids()
            assert len(rt.hitl_registry.list_pending(session_id=SID)) == (0 if first == "late" else 1)
            task, = tm.all_tasks()
            assert await task_body(memory, task.id) == await normal_segment(
                tmp_path_factory, short, first == "late")
            rows = await store.read_range(SID)
            intents = [r.event.payload.get("observe_request", {}) for r in rows if r.event.type == "HitlOpened"]
            assert len(intents) == 1
            oid = intents[0]["observe_id"]
            for event_type in ("ObserveStarted", "ObserveCompleted", "ObserveApplied"):
                assert len([r for r in rows if r.event.type == event_type and r.event.payload.get("observe_id") == oid]) == 1
            from tests.integration.test_queued_session_cancel_recovery import content
            before = await content(memory, view.tasks.values())
            head = await store.committed_head(SID)
            cold = runtime(store, memory, _FinishEveryTaskLLM(), short)
            await initialize_all(cold)
            await cold.rebuild_session(SID, strict=True)
            cold._task_managers[SID]._max_concurrent = 0
            await recover_fixture_session(cold, tm.session.root_agent_id, session_id=SID)
            await drained(cold)
            assert await store.committed_head(SID) == head
            assert await content(memory, view.tasks.values()) == before
