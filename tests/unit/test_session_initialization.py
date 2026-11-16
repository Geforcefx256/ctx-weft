
from tests._recovery import recover_fixture_session
"""Per-session recovery: isolation, shared work, retries and bounded retention."""
import asyncio
import gc
import weakref
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from ctx_weft import SessionNotFound
from ctx_weft.core.control import startup
from ctx_weft.protocols.events import EventType
from tests.unit.test_runtime_initialization_queries import append, container, runtime



async def seeded():
    old = runtime()
    h = await container(old)
    await append(old.event_store, h.session_id,
        (EventType.TASK_CREATED, "work", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "work", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "work", h.root_agent_id, {}))
    return runtime(old.event_store), h


@pytest.mark.asyncio
async def test_single_session_never_enumerates_or_touches_other_sessions(monkeypatch):
    rt, h = await seeded()
    other = await container(runtime(rt.event_store), "other")
    await append(rt.event_store, other.session_id,
        (EventType.TASK_CREATED, "other-work", other.root_agent_id, {}))
    head = await rt.event_store.committed_head("other")
    monkeypatch.setattr(rt.event_store, "list_session_ids", AsyncMock(side_effect=AssertionError("enumeration")))
    result = await rt.initialize(h.session_id)
    assert result.state.root_task.status == "INTERRUPTED"
    assert await rt.event_store.committed_head("other") == head
    assert rt._initialized_sessions == {h.session_id}
    assert not rt._initializing_sessions and not rt._task_managers
    assert not rt._agent_lifecycle_manager.all_agent_ids()


@pytest.mark.asyncio
async def test_concurrent_waiters_share_task_and_receive_detached_reports(monkeypatch):
    rt, h = await seeded()
    entered, release = asyncio.Event(), asyncio.Event()
    original = startup.calibrate_session
    calls = 0
    async def calibrate(*args):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return await original(*args)
    monkeypatch.setattr(startup, "calibrate_session", calibrate)
    waiters = [asyncio.create_task(rt.initialize(h.session_id)) for _ in range(20)]
    await entered.wait()
    assert len(rt._initializing_sessions) == 1
    release.set()
    reports = await asyncio.gather(*waiters)
    assert calls == 1 and all(r == reports[0] for r in reports)
    reports[0].startup_events.clear()
    assert reports[1].startup_events
    assert not rt._initializing_sessions
    monkeypatch.setattr(rt.event_store, "committed_head", AsyncMock(side_effect=AssertionError("repeat I/O")))
    assert await rt.initialize(h.session_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_all", [False, True])
async def test_waiter_cancellation_does_not_cancel_initialization(monkeypatch, cancel_all):
    rt, h = await seeded()
    entered, release = asyncio.Event(), asyncio.Event()
    original = startup.calibrate_session
    async def calibrate(*args):
        entered.set()
        await release.wait()
        return await original(*args)
    monkeypatch.setattr(startup, "calibrate_session", calibrate)
    a, b = [asyncio.create_task(rt.initialize(h.session_id)) for _ in range(2)]
    await entered.wait()
    shared = rt._initializing_sessions[h.session_id]
    a.cancel()
    if cancel_all:
        b.cancel()
    await asyncio.gather(a, *([b] if cancel_all else []), return_exceptions=True)
    assert not shared.cancelled()
    release.set()
    await shared
    if not cancel_all:
        assert (await b).state.has_interrupted_agents
    await asyncio.sleep(0)
    assert h.session_id in rt._initialized_sessions and not rt._initializing_sessions


@pytest.mark.asyncio
async def test_shared_failure_clears_task_and_can_retry(monkeypatch):
    rt, h = await seeded()
    entered, release = asyncio.Event(), asyncio.Event()
    original = startup.calibrate_session
    calls = 0
    async def fail(*args):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise OSError("offline")
    monkeypatch.setattr(startup, "calibrate_session", fail)
    waiters = [asyncio.create_task(rt.initialize(h.session_id)) for _ in range(8)]
    await entered.wait()
    release.set()
    failures = await asyncio.gather(*waiters, return_exceptions=True)
    assert calls == 1 and all(isinstance(e, OSError) for e in failures)
    assert not rt._initializing_sessions and not rt._initialized_sessions
    monkeypatch.setattr(startup, "calibrate_session", original)
    assert (await rt.initialize(h.session_id)).state.has_interrupted_agents


@pytest.mark.asyncio
async def test_different_sessions_initialize_in_parallel(monkeypatch):
    rt, h = await seeded()
    await container(runtime(rt.event_store), "other")
    original = startup.calibrate_session
    entered, release = set(), asyncio.Event()
    async def calibrate(store, sid):
        entered.add(sid)
        if len(entered) == 2:
            release.set()
        await release.wait()
        return await original(store, sid)
    monkeypatch.setattr(startup, "calibrate_session", calibrate)
    async with asyncio.timeout(2):
        await asyncio.gather(rt.initialize(h.session_id), rt.initialize("other"))
    assert entered == {h.session_id, "other"}


@pytest.mark.asyncio
async def test_read_queries_do_not_initialize_or_repair():
    rt, h = await seeded()
    head = await rt.event_store.committed_head(h.session_id)
    assert (await rt.get_session(h.session_id)).data.session_id == h.session_id
    assert (await rt.get_task(h.session_id, "work")).data.status == "ACTIVE"
    assert await rt.event_store.committed_head(h.session_id) == head
    assert not rt._initialized_sessions and not rt._task_managers


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["rebuild", "recover", "dispatch", "send", "resume"])
async def test_execution_entry_initializes_before_loading(monkeypatch, entry):
    from ctx_weft import SessionStartParams
    rt, h = await seeded()
    loaded = []
    original = rt._rebuild_session_locked
    async def rebuild(sid, **kwargs):
        assert sid in rt._initialized_sessions
        loaded.append(sid)
        return await original(sid, **kwargs)
    monkeypatch.setattr(rt, "_rebuild_session_locked", rebuild)
    async with asyncio.timeout(3):
        if entry == "rebuild":
            await rt.rebuild_session(h.session_id, strict=True)
        elif entry == "recover":
            await recover_fixture_session(rt, h.root_agent_id, session_id=h.session_id, keep_alive=True)
        elif entry == "dispatch":
            await rt.dispatch_task(h.session_id, "next", agent_id=h.root_agent_id, unattended=True)
        elif entry == "send":
            await rt.send_message(h.root_agent_id, "next", session_id=h.session_id)
        else:
            from ctx_weft.core.models.errors import UnfinishedTasksError
            with pytest.raises(UnfinishedTasksError):
                await rt.start_session(SessionStartParams.create(
                    session_id=h.session_id, resume=True, template_id="agent:tpl_echo",
                    user_prompt="next"))
    assert loaded and not rt._initializing_sessions


@pytest.mark.asyncio
async def test_missing_session_is_not_marked_and_new_creation_is_initialized():
    rt = runtime()
    with pytest.raises(SessionNotFound):
        await rt.initialize("missing")
    assert not rt._initialized_sessions and not rt._initializing_sessions
    h = await container(rt, "missing")
    assert await rt.initialize(h.session_id) is None
    assert rt.forget_session(h.session_id)
    assert await rt.initialize(h.session_id) is None
    await rt.purge_session(h.session_id)
    assert h.session_id not in rt._initialized_sessions


@pytest.mark.asyncio
async def test_success_does_not_retain_task_or_report(monkeypatch):
    rt, h = await seeded()
    references = []
    original = startup.calibrate_session
    async def calibrate(*args):
        report = await original(*args)
        report = replace(report, startup_events=report.startup_events * 1000)
        references.extend([weakref.ref(asyncio.current_task()), weakref.ref(report)])
        return report
    monkeypatch.setattr(startup, "calibrate_session", calibrate)
    async def use():
        await rt.initialize(h.session_id)
    await use()
    await asyncio.sleep(0)
    gc.collect()
    assert not rt._initializing_sessions and rt._initialized_sessions == {h.session_id}
    assert all(ref() is None for ref in references)


@pytest.mark.asyncio
async def test_cold_hitl_reply_initializes_before_resolving(monkeypatch):
    from ctx_weft.protocols.hitl import HitlReply

    rt, h = await seeded()
    await append(rt.event_store, h.session_id,
        (EventType.HITL_OPENED, "work", h.root_agent_id, {
            "hitl_id": "question", "form": "question", "stage": "tool",
            "delivery": {"kind": "no_resume"}}))
    original = rt.hitl.resolve
    async def resolve(*args, **kwargs):
        assert h.session_id in rt._initialized_sessions
        return await original(*args, **kwargs)
    monkeypatch.setattr(rt.hitl, "resolve", resolve)
    result = await rt.reply_to_hitl(HitlReply(
        hitl_id="question", outcome="accepted", agent_id=h.root_agent_id,
        message="answer"), session_id=h.session_id)
    assert result.outcome == "accepted"
    assert not rt._initializing_sessions


@pytest.mark.asyncio
async def test_missing_session_purge_nested_commands_do_not_deadlock():
    rt = runtime()
    async with asyncio.timeout(1):
        await rt.purge_session("never-created")
        await rt.purge_session("never-created")
    assert not rt._initializing_sessions and not rt._initialized_sessions
    assert not rt._task_managers
