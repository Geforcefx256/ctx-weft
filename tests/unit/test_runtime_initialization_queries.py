"""Host lifecycle: startup facts, read-only queries and one cold owner."""

from __future__ import annotations

from tests._recovery import recover_fixture_session

import asyncio

import pytest

from ctx_weft import RuntimeConfig, SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.errors import SessionAlreadyExistsError
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["get_session", "get_task", "list_tasks", "get_agent"])
async def test_dispatch_callback_can_query_same_session(query):
    rt = runtime()
    h = await container(rt)
    results = []

    async def observe(event):
        if query == "get_session":
            result = await rt.get_session(h.session_id)
        elif query == "get_task":
            result = await rt.get_task(h.session_id, event.task_id)
        elif query == "list_tasks":
            result = await rt.list_tasks(h.session_id)
        else:
            result = await rt.get_agent(h.root_agent_id)
        results.append(result)

    subscription = rt.event_bus.subscribe(EventType.TASK_CREATED, observe)
    try:
        async with asyncio.timeout(2):
            handle = await rt.dispatch_task(
                h.session_id, "work", agent_id=h.root_agent_id, unattended=True)
        assert len(results) == 1
        if query == "get_session":
            assert results[0].data.session_id == h.session_id
        elif query == "get_task":
            assert results[0].data.task_id == handle.task_id
        elif query == "list_tasks":
            assert handle.task_id in {row.data.task_id for row in results[0]}
        else:
            assert results[0].data.agent_id == h.root_agent_id
    finally:
        await subscription.unsubscribe()


@pytest.mark.asyncio
async def test_successor_task_query_uses_cross_task_history_without_loading():
    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "old", h.root_agent_id, {"task": {"id": "old"}}),
        (EventType.TASK_REVIEWING, "old", h.root_agent_id, {
            "successor": {"task": {
                "id": "successor", "title": "new goal",
                "assigned_agent_id": h.root_agent_id}},
            "replaced_task_ids": ["old"],
        }),
    )
    assert rt.forget_session(h.session_id)
    head = await rt.event_store.committed_head(h.session_id)
    rows = {r.data.task_id: r for r in await rt.list_tasks(h.session_id)}
    result = await rt.get_task(h.session_id, "successor")
    assert result.data == rows["successor"].data
    assert result.data.title == "new goal" and result.data.status == "PENDING"
    assert result.source == "events" and not result.loaded
    assert result.committed_position == head
    assert not await rt._task_is_settled(h.session_id, "successor")
    await append(rt.event_store, h.session_id,
        (EventType.TASK_FINISHED, "successor", h.root_agent_id, {"outputs": "done"}),
    )
    assert (await rt.get_task(h.session_id, "successor")).data.outputs == "done"
    assert await rt._task_is_settled(h.session_id, "successor")
    assert not rt._task_managers


def runtime(store=None, *, max_concurrent=0):
    agents = InlineAgentTemplateProvider()
    agents.register(make_echo_template())
    rt = make_runtime(
        agent_provider=agents, event_store=store,
        llm=MockLLMAdapter(responses=[]), config=RuntimeConfig(task_max_concurrent=max_concurrent))
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def container(rt, sid="s1"):
    await initialize_all(rt)
    return await rt.create_session(template_id="agent:tpl_echo",  session_id=sid)


async def append(store, sid, *specs):
    events = [new_event(kind, session_id=sid, tenant_id="default", origin="test",
                        task_id=tid, agent_id=aid, payload=payload)
              for kind, tid, aid, payload in specs]
    await store.append_batch(sid, events[0].id, events)


@pytest.mark.asyncio
async def test_initialize_and_rebuild_accept_pruned_dependency_without_timestamp():
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    from tests._snapshot_helpers import seed_snapshot

    store = InMemoryEventStore()
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "dep", h.root_agent_id, {"task": {"id": "dep"}}),
        (EventType.TASK_FINISHED, "dep", h.root_agent_id, {"outputs": "done"}),
        (EventType.TASK_CREATED, "live", h.root_agent_id, {
            "task": {"id": "live", "dag_deps": ["dep"]}}),
    )
    view = prune_view_for_snapshot(await rebuild_view(store, h.session_id))
    assert view.tasks["dep"].created_at is None
    assert view.tasks["live"].created_at is not None
    await seed_snapshot(store, h.session_id, view, cut=await store.committed_head(h.session_id))
    rt = runtime(store)
    assert (await initialize_all(rt)).checked_sessions == 1
    assert (await rt.get_task(h.session_id, "dep")).data.outputs == "done"
    await rt.rebuild_session(h.session_id, strict=True)
    assert rt._task_managers[h.session_id].get_task("live") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded", [False, True])
async def test_wait_for_finish_requires_durable_recap_done(loaded):
    from ctx_weft.core.runtime import TurnHandle

    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id, {"task": {"id": "t"}}),
        (EventType.TASK_FINISHED, "t", h.root_agent_id, {"outputs": "done"}),
        (EventType.OBSERVE_STARTED, "t", h.root_agent_id,
         {"task_id": "t", "agent_id": h.root_agent_id, "boundary": "finish"}),
    )
    assert rt.forget_session(h.session_id)
    if loaded:
        await rt.rebuild_session(h.session_id, strict=True)
        # Reproduce a failed relaunch after deferred recovery was consumed.
        rt._prepared_recoveries.pop(h.session_id)
    handle = TurnHandle(
        session_id=h.session_id, agent_id=h.root_agent_id, task_id="t",
        template_id="agent:tpl_echo", event_bus=rt.event_bus,
        _is_finished=rt._task_is_settled, _pending_observe=rt._pending_observe_run_id,
    )
    subscribers = set(rt.event_bus._subscribers)
    with pytest.raises(TimeoutError):
        await handle.wait_for_finish(timeout=0.05)
    assert set(rt.event_bus._subscribers) == subscribers
    await rt.event_bus.emit(new_event(
        EventType.OBSERVE_APPLIED, session_id=h.session_id, tenant_id="default",
        origin="test", task_id="t", agent_id=h.root_agent_id,
        payload={"task_id": "t", "agent_id": h.root_agent_id, "boundary": "finish"}))
    await handle.wait_for_finish(timeout=0.1)


@pytest.mark.asyncio
async def test_wait_does_not_bypass_deferred_recap_on_terminal_event(monkeypatch):
    from ctx_weft.core.runtime import TurnHandle

    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id, {"task": {"id": "t"}}),
        (EventType.TASK_FINISHED, "t", h.root_agent_id, {}),
        (EventType.OBSERVE_STARTED, "t", h.root_agent_id,
         {"task_id": "t", "agent_id": h.root_agent_id, "boundary": "finish"}),
    )
    assert rt.forget_session(h.session_id)
    await rt.rebuild_session(h.session_id, strict=True)
    handle = TurnHandle(
        session_id=h.session_id, agent_id=h.root_agent_id, task_id="t",
        template_id="agent:tpl_echo", event_bus=rt.event_bus,
        _is_finished=rt._task_is_settled, _pending_observe=rt._pending_observe_run_id,
    )
    waiter = asyncio.create_task(handle.wait_for_finish(timeout=0.1))
    await asyncio.sleep(0)
    await rt.event_bus.emit(new_event(
        EventType.TASK_FINISHED, session_id=h.session_id, tenant_id="default",
        origin="test", task_id="t", agent_id=h.root_agent_id))
    with pytest.raises(TimeoutError):
        await waiter
    # Activating recovery is an execution concern; waiting must not consume it.
    assert "t" in rt._prepared_recoveries[h.session_id][2]

    async def complete_recap(**kwargs):
        await rt.event_bus.emit(new_event(
            EventType.OBSERVE_APPLIED, session_id=h.session_id, tenant_id="default",
            origin="test", task_id="t", agent_id=h.root_agent_id,
            payload={"task_id": "t", "boundary": kwargs["boundary"]}))
        return True

    monkeypatch.setattr(rt, "_relaunch_observe", complete_recap)
    await recover_fixture_session(rt, h.root_agent_id)
    await handle.wait_for_finish(timeout=0.1)


@pytest.mark.asyncio
async def test_new_sessions_do_not_require_global_initialize():
    rt = runtime()
    h = await rt.create_session(template_id="agent:tpl_echo")
    assert h.session_id in rt._initialized_sessions
    assert await rt.initialize(h.session_id) is None
    assert (await rt.get_session(h.session_id)).loaded


@pytest.mark.asyncio
async def test_slim_dependency_queries_keep_details_after_loading():
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    from tests._snapshot_helpers import seed_snapshot

    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "dep", h.root_agent_id,
         {"task": {"id": "dep", "title": "original title", "description": "details"}}),
        (EventType.TASK_FINISHED, "dep", h.root_agent_id, {"outputs": {"answer": 42}}),
        (EventType.TASK_CREATED, "live", h.root_agent_id,
         {"task": {"id": "live", "dag_deps": ["dep"]}}),
    )
    view = prune_view_for_snapshot(await rebuild_view(rt.event_store, h.session_id))
    assert not view.tasks["dep"].details_complete
    await seed_snapshot(rt.event_store, h.session_id, view,
                        cut=await rt.event_store.committed_head(h.session_id))
    assert rt.forget_session(h.session_id)
    before = (await rt.get_task(h.session_id, "dep")).data
    await rt.rebuild_session(h.session_id, strict=True)
    slim = rt._task_managers[h.session_id].get_task("dep")
    assert not slim.details_complete and slim.outputs is None
    result = await rt.get_task(h.session_id, "dep")
    rows = {r.data.task_id: r for r in await rt.list_tasks(h.session_id)}
    assert result.data == before == rows["dep"].data
    assert result.source == rows["dep"].source == "events"
    assert not slim.details_complete  # Queries do not hydrate runtime owners.


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_kind", ["new", "missing_task", "legacy"])
async def test_pending_observe_snapshot_recovers_and_retries(snapshot_kind, monkeypatch):
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    from ctx_weft.core.runtime import TurnHandle
    from tests._snapshot_helpers import seed_snapshot

    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "t", h.root_agent_id,
         {"task": {"id": "t", "title": "recap task", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_FINISHED, "t", h.root_agent_id, {"outputs": "result"}),
        (EventType.OBSERVE_STARTED, "t", h.root_agent_id,
         {"task_id": "t", "agent_id": h.root_agent_id, "boundary": "finish"}),
    )
    view = prune_view_for_snapshot(await rebuild_view(rt.event_store, h.session_id))
    assert view.tasks["t"].details_complete
    assert view.tasks["t"].outputs == "result"
    if snapshot_kind != "new":
        view.tasks.pop("t")
    await seed_snapshot(rt.event_store, h.session_id, view,
                        cut=await rt.event_store.committed_head(h.session_id),
                        overrides={"projection_version": 4} if snapshot_kind == "legacy" else None)
    assert rt.forget_session(h.session_id)
    await rt.rebuild_session(h.session_id, strict=True)
    assert rt._task_managers[h.session_id].get_task("t").outputs == "result"
    handle = TurnHandle(
        session_id=h.session_id, agent_id=h.root_agent_id, task_id="t",
        template_id="agent:tpl_echo", event_bus=rt.event_bus,
        _is_finished=rt._task_is_settled, _pending_observe=rt._pending_observe_run_id)
    # Exercise the real relaunch failure path, then retry through the public entry.
    original = rt._agent_lifecycle_manager.materialize
    def fail(*args, **kwargs):
        raise OSError("temporary failure")
    monkeypatch.setattr(rt._agent_lifecycle_manager, "materialize", fail)
    await recover_fixture_session(rt, h.root_agent_id)
    assert "t" in rt._prepared_recoveries[h.session_id][2]
    with pytest.raises(TimeoutError):
        await handle.wait_for_finish(timeout=0.05)
    monkeypatch.setattr(rt._agent_lifecycle_manager, "materialize", original)
    started = []
    async def complete(**kwargs):
        started.append(kwargs["task"].id)
        await rt.event_bus.emit(new_event(
            EventType.OBSERVE_APPLIED, session_id=h.session_id, tenant_id="default",
            origin="test", task_id="t", agent_id=h.root_agent_id,
            payload={"task_id": "t", "boundary": "finish"}))
        return True
    monkeypatch.setattr(rt, "_relaunch_observe", complete)
    await recover_fixture_session(rt, h.root_agent_id)
    assert started == ["t"]
    assert h.session_id not in rt._prepared_recoveries
    await handle.wait_for_finish(timeout=0.1)


@pytest.mark.asyncio
async def test_startup_calibrates_without_loading_and_preserves_waits():
    store = InMemoryEventStore()
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "active", h.root_agent_id, {}),
        (EventType.TASK_CREATED, "waiting", None, {}),
        (EventType.TASK_AWAITING_HUMAN, "waiting", None, {"hitl_id": "question"}),
        (EventType.TASK_CREATED, "queued", None, {}),
        (EventType.TASK_CREATED, "review", None, {}),
        (EventType.TASK_REVIEWING, "review", None, {}),
    )
    rt = runtime(store)
    report = await initialize_all(rt)
    assert report.interrupted_task_ids == ["active", "queued"]
    assert report.interrupted_session_ids == [h.session_id]
    assert not rt._task_managers and not rt._session_registry._states
    assert rt._agent_lifecycle_manager.all_agent_ids() == []
    assert not rt.hitl_registry.list_pending()
    view = await rebuild_view(store, h.session_id)
    assert view.tasks["active"].status == "INTERRUPTED"
    assert view.tasks["waiting"].status == "AWAITING_HUMAN"
    assert view.tasks["queued"].status == "INTERRUPTED"
    assert view.tasks["review"].status == "REVIEWING"
    assert view.agents[h.root_agent_id].status == "interrupted"
    rows = await store.read_range(h.session_id)
    assert not any(row.event.type == EventType.SESSION_STATUS_CHANGED for row in rows)
    head = await store.committed_head(h.session_id)
    await initialize_all(runtime(store))
    assert await store.committed_head(h.session_id) == head


@pytest.mark.asyncio
async def test_initialization_failure_can_retry_without_duplicate_wiring(monkeypatch):
    store = InMemoryEventStore()
    h = await container(runtime(store))
    rt = runtime(store)
    subscribers = dict(rt.event_bus._subscribers)
    original = store.committed_head
    async def fail(*args, **kwargs):
        raise OSError("unavailable")
    monkeypatch.setattr(store, "committed_head", fail)
    with pytest.raises(OSError):
        await rt.initialize(h.session_id)
    assert not rt._initialized_sessions and not rt._initializing_sessions
    assert rt.event_bus._subscribers == subscribers
    monkeypatch.setattr(store, "committed_head", original)
    await rt.initialize(h.session_id)
    assert rt._initialized_sessions == {h.session_id}
    assert rt.event_bus._subscribers == subscribers


@pytest.mark.asyncio
async def test_query_cold_hot_and_evicted_without_loading():
    rt = runtime()
    h = await container(rt)
    assert (await rt.get_session(h.session_id)).source == "memory"
    assert rt.forget_session(h.session_id)
    head = await rt.event_store.committed_head(h.session_id)
    result = await rt.get_session(h.session_id)
    assert result.source == "events" and not result.loaded
    assert result.committed_position == head
    agent = await rt.get_agent(h.root_agent_id, session_id=h.session_id)
    assert agent.source == "events" and agent.data.state.status == "idle"
    assert not rt._task_managers
    assert rt._agent_lifecycle_manager.all_agent_ids() == []
    assert await rt.event_store.committed_head(h.session_id) == head
    await rt.rebuild_session(h.session_id, strict=True)
    assert (await rt.get_agent(h.root_agent_id)).source == "memory"


@pytest.mark.asyncio
async def test_concurrent_rebuild_and_dispatch_share_exactly_one_owner(monkeypatch):
    rt = runtime()
    h = await container(rt)
    assert rt.forget_session(h.session_id)
    owners = []
    original = rt._session_registry._new_task_manager
    def create(session):
        owner = original(session)
        owners.append(owner)
        return owner
    monkeypatch.setattr(rt._session_registry, "_new_task_manager", create)
    operations = [rt.rebuild_session(h.session_id, strict=True) for _ in range(6)]
    operations += [rt.dispatch_task(h.session_id, f"work {i}", agent_id=h.root_agent_id,
                                   unattended=True) for i in range(6)]
    await asyncio.gather(*operations)
    assert len(owners) == 1
    assert rt._task_managers[h.session_id] is owners[0]
    assert len(owners[0].all_tasks()) == 6
    await rt.rebuild_session(h.session_id, strict=True)
    assert rt._task_managers[h.session_id] is owners[0]


@pytest.mark.asyncio
async def test_rebuild_failure_rolls_back_all_runtime_records(monkeypatch):
    rt = runtime()
    h = await container(rt)
    assert rt.forget_session(h.session_id)
    original = rt._load_task_manager_view
    async def fail(*args, **kwargs):
        raise OSError("failed preparation")
    monkeypatch.setattr(rt, "_load_task_manager_view", fail)
    with pytest.raises(OSError):
        await rt.rebuild_session(h.session_id, strict=True)
    assert h.session_id not in rt._task_managers
    assert not rt._agent_lifecycle_manager.agent_ids_of_session(h.session_id)
    assert h.session_id not in rt._session_registry._states
    monkeypatch.setattr(rt, "_load_task_manager_view", original)
    await rt.rebuild_session(h.session_id, strict=True)
    assert rt._session_is_loaded(h.session_id)


@pytest.mark.asyncio
async def test_missing_member_rebuild_keeps_live_owner():
    rt = runtime()
    h = await container(rt)
    owner = rt._task_managers[h.session_id]
    assert rt.forget_agent(h.root_agent_id)
    await rt.rebuild_session(h.session_id, strict=True)
    assert rt._task_managers[h.session_id] is owner
    assert (await rt.get_agent(h.root_agent_id)).data.agent_id == h.root_agent_id


@pytest.mark.asyncio
async def test_cold_existing_id_cannot_be_created_twice():
    rt = runtime()
    h = await container(rt)
    assert rt.forget_session(h.session_id)
    with pytest.raises(SessionAlreadyExistsError):
        await rt.create_session(template_id="agent:tpl_echo",  session_id=h.session_id)
    with pytest.raises(SessionAlreadyExistsError):
        await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="duplicate",
            session_id=h.session_id))


@pytest.mark.asyncio
async def test_task_queries_merge_live_state_and_history():
    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "old", h.root_agent_id,
         {"assigned_agent_id": h.root_agent_id, "title": "old work"}),
        (EventType.TASK_FINISHED, "old", h.root_agent_id, {"outputs": {"answer": 42}}),
    )
    handle = await rt.dispatch_task(h.session_id, "new", agent_id=h.root_agent_id, unattended=True)
    old = await rt.get_task(h.session_id, "old")
    assert old.source == "events" and old.data.outputs == {"answer": 42}
    live = await rt.get_task(h.session_id, handle.task_id)
    assert live.source == "memory"
    rows = await rt.list_tasks(h.session_id)
    assert {r.data.task_id for r in rows} == {"old", handle.task_id}
    old.data.outputs["answer"] = 0
    assert (await rt.get_task(h.session_id, "old")).data.outputs == {"answer": 42}


@pytest.mark.asyncio
async def test_store_session_enumeration_is_partition_only():
    store = InMemoryEventStore()
    for sid in ("c", "a", "b"):
        await append(store, sid, (EventType.SESSION_CREATED, None, None, {}))
    assert await store.list_session_ids(limit=2) == ["a", "b"]
    assert await store.list_session_ids(after_session_id="b", limit=2) == ["c"]


@pytest.mark.asyncio
async def test_historical_task_survives_pruned_recovery_snapshot():
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    from tests._snapshot_helpers import seed_snapshot

    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "old", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_FINISHED, "old", h.root_agent_id, {"outputs": "kept in history"}),
    )
    view = prune_view_for_snapshot(await rebuild_view(rt.event_store, h.session_id))
    assert "old" not in view.tasks
    await seed_snapshot(rt.event_store, h.session_id, view,
                        cut=await rt.event_store.committed_head(h.session_id))
    assert rt.forget_session(h.session_id)
    assert (await rt.get_task(h.session_id, "old")).data.outputs == "kept in history"
    assert [row.data.task_id for row in await rt.list_tasks(h.session_id)] == ["old"]
    assert not rt._task_managers


@pytest.mark.asyncio
async def test_sql_restart_calibration_and_queries(tmp_path):
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store

    path = tmp_path / "events.db"
    async with open_sqlite_event_store(path) as store:
        rt = runtime(store)
        h = await container(rt)
        await append(store, h.session_id,
            (EventType.TASK_CREATED, "crashed", h.root_agent_id,
             {"assigned_agent_id": h.root_agent_id}),
            (EventType.TASK_STARTED, "crashed", h.root_agent_id,
             {"assigned_agent_id": h.root_agent_id}),
        )
    async with open_sqlite_event_store(path) as store:
        rt = runtime(store)
        report = await initialize_all(rt)
        assert report.interrupted_task_ids == ["crashed"]
        task = await rt.get_task(h.session_id, "crashed")
        assert task.data.status == "INTERRUPTED" and task.source == "events"
        assert not rt._task_managers
        assert await store.list_session_ids(limit=1) == [h.session_id]
        await asyncio.gather(*(rt.rebuild_session(h.session_id, strict=True) for _ in range(5)))
        assert len(rt._task_managers) == 1


@pytest.mark.asyncio
async def test_wait_after_terminal_and_repeated_wait_are_immediate():
    from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM
    rt = runtime(max_concurrent=4)
    rt._llm = _FinishEveryTaskLLM()
    rt._config = RuntimeConfig()
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="finish",
        unattended=True))
    await handle.wait_for_finish(timeout=5)
    await handle.wait_for_finish(timeout=0.1)
    assert (await rt.get_task(handle.session_id, handle.task_id)).data.status == "FINISHED"


@pytest.mark.asyncio
async def test_wait_does_not_close_subscription_on_an_idle_tick():
    from ctx_weft.core.runtime import TurnHandle
    from ctx_weft.providers.events import InProcessEventBus
    bus = InProcessEventBus()
    handle = TurnHandle(session_id="s", agent_id="a", task_id="t", template_id="tpl", event_bus=bus)
    waiter = asyncio.create_task(handle.wait_for_finish(timeout=2))
    await asyncio.sleep(0.4)
    assert not waiter.done()
    await bus.emit(new_event(EventType.TASK_FINISHED, session_id="s", tenant_id="default",
                             origin="test", agent_id="a", task_id="t"))
    await waiter
    assert not bus._subscribers


@pytest.mark.asyncio
async def test_simultaneous_cold_send_reuses_owner(monkeypatch):
    from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM
    rt = runtime()
    rt._llm = _FinishEveryTaskLLM()
    rt._config = RuntimeConfig()
    h = await container(rt)
    assert rt.forget_session(h.session_id)
    owners = []
    original = rt._session_registry._new_task_manager
    def create(session):
        owner = original(session)
        owners.append(owner)
        return owner
    monkeypatch.setattr(rt._session_registry, "_new_task_manager", create)
    # Disable dispatch only; both commands deliver their input to the same owner.
    rt._config = RuntimeConfig(task_max_concurrent=0)
    handles = await asyncio.gather(*(rt.send_message(h.root_agent_id, str(i), session_id=h.session_id)
                                     for i in range(4)))
    assert len(owners) == 1
    assert len({handle.task_id for handle in handles}) == 1
    assert rt._task_managers[h.session_id] is owners[0]


@pytest.mark.asyncio
async def test_initialize_retry_after_commit_does_not_duplicate_facts(monkeypatch):
    store = InMemoryEventStore()
    h = await container(runtime(store))
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
    )
    rt = runtime(store)
    original = store.append_batch
    async def lose_confirmation(*args, **kwargs):
        await original(*args, **kwargs)
        raise OSError("commit acknowledgement lost")
    monkeypatch.setattr(store, "append_batch", lose_confirmation)
    with pytest.raises(OSError):
        await initialize_all(rt)
    head = await store.committed_head(h.session_id)
    assert not rt._initialized_sessions and not rt._task_managers
    monkeypatch.setattr(store, "append_batch", original)
    await initialize_all(rt)
    assert await store.committed_head(h.session_id) == head
    facts = await store.read_range(h.session_id, include_types=(EventType.TASK_INTERRUPTED,))
    assert len(facts) == 1


@pytest.mark.asyncio
async def test_startup_closes_abandoned_run_envelope():
    store = InMemoryEventStore()
    h = await container(runtime(store))
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
    )
    start = new_event(EventType.RUN_STARTED, session_id=h.session_id,
                      tenant_id="default", origin="test", run_id="abandoned",
                      task_id="active", agent_id=h.root_agent_id)
    await store.append_batch(h.session_id, start.id, [start])
    await initialize_all(runtime(store))
    rows = await store.read_range(h.session_id,
                                  include_types=(EventType.RUN_INTERRUPTED, EventType.RUN_FINISHED))
    assert [r.event.type for r in rows] == [EventType.RUN_INTERRUPTED, EventType.RUN_FINISHED]
    assert all(r.event.run_id == "abandoned" for r in rows)


@pytest.mark.asyncio
async def test_startup_preserves_a_durable_hot_human_wait():
    from ctx_weft.protocols.hitl import HitlAsk, ToolResultDelivery
    store = InMemoryEventStore()
    old = runtime(store)
    h = await container(old)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.TASK_STARTED, "active", h.root_agent_id, {"assigned_agent_id": h.root_agent_id}),
        (EventType.AGENT_RUNNING, "active", h.root_agent_id, {"task_id": "active"}),
    )
    await old.hitl.open(HitlAsk(form="approval", delivery=ToolResultDelivery("call")),
                        session_id=h.session_id, task_id="active", agent_id=h.root_agent_id,
                        stage="authz", unattended=False)
    rt = runtime(store)
    report = await initialize_all(rt)
    assert not report.interrupted_task_ids
    assert (await rt.get_task(h.session_id, "active")).data.status == "AWAITING_HUMAN"
    assert (await rt.get_agent(h.root_agent_id, session_id=h.session_id)).data.state.status == "waiting_human"
    assert not rt._task_managers


@pytest.mark.asyncio
@pytest.mark.parametrize("base_kind", ["full", "slim", "pruned", "absent"])
async def test_get_task_reads_only_target_facts_and_uses_snapshot(base_kind):
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    from tests._snapshot_helpers import seed_snapshot

    class CountingStore(InMemoryEventStore):
        def __init__(self):
            super().__init__()
            self.reads = []

        async def read_range(self, session_id, **kwargs):
            rows = await super().read_range(session_id, **kwargs)
            self.reads.append((kwargs, [r.event.task_id for r in rows]))
            return rows

    store = CountingStore()
    rt = runtime(store)
    h = await container(rt)
    await append(store, h.session_id,
        (EventType.TASK_CREATED, "target", h.root_agent_id,
         {"task": {"id": "target", "title": "title"}}),
        (EventType.TASK_FINISHED, "target", h.root_agent_id, {"outputs": {"value": 42}}),
        (EventType.TASK_CREATED, "live", h.root_agent_id,
         {"task": {"id": "live", "dag_deps": ["target"]}}))
    for i in range(20):
        await append(store, h.session_id,
            (EventType.TASK_CREATED, f"unrelated-{i}", h.root_agent_id,
             {"task": {"id": f"unrelated-{i}"}}),
            (EventType.TASK_FINISHED, f"unrelated-{i}", h.root_agent_id, {"outputs": "large history"}))
    cut = await store.committed_head(h.session_id)
    if base_kind != "absent":
        view = await rebuild_view(store, h.session_id)
        if base_kind != "full":
            view = prune_view_for_snapshot(view)
        if base_kind == "pruned":
            view.tasks.pop("target")
        await seed_snapshot(store, h.session_id, view, cut=cut)
    await append(store, h.session_id,
        (EventType.TASK_FINALIZED, "target", h.root_agent_id, {}))
    assert rt.forget_session(h.session_id)
    store.reads.clear()
    result = await rt.get_task(h.session_id, "target")
    assert result.data.outputs == {"value": 42} and result.data.finished_at is not None
    assert result.source == "events" and not result.loaded
    assert len(store.reads) == 1
    kwargs, ids = store.reads[0]
    assert kwargs["related_task_id"] == "target"
    assert kwargs["after_position"] == (cut if base_kind == "full" else 0)
    assert set(ids) == {"target"}
    assert not rt._task_managers


@pytest.mark.asyncio
async def test_get_task_applies_review_of_target_under_another_task_id():
    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "parent", h.root_agent_id, {"task": {"id": "parent"}}),
        (EventType.TASK_CREATED, "child", h.root_agent_id, {"task": {"id": "child"}}),
        (EventType.TASK_REVIEWING, "child", h.root_agent_id, {
            "successor": {"task": {"id": "new"}}, "replaced_task_ids": ["parent", "child"]}))
    rt.forget_session(h.session_id)
    assert (await rt.get_task(h.session_id, "parent")).data.status == "REVIEWING"
    assert (await rt.get_task(h.session_id, "new")).data.status == "PENDING"
