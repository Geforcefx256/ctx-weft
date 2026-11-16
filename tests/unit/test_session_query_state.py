"""Session and agent queries share complete execution facts, hot and cold."""

import pytest

from ctx_weft import AgentDetail, AgentState, HitlRequestView, TaskDetail
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from tests._snapshot_helpers import seed_snapshot
from tests.unit.test_runtime_initialization_queries import append, container, runtime


async def test_hot_state_exposes_agent_reasons_and_goal_without_folding_notifications():
    rt = runtime()
    h = await container(rt)
    rt._task_managers[h.session_id].session.goal = "current goal"
    rec = rt._agent_lifecycle_manager._agents[h.root_agent_id]
    rec.status, rec.status_reason = "interrupted", "llm_outage:timeout"
    state = (await rt.get_session(h.session_id)).data.state
    agent = (await rt.get_agent(h.root_agent_id)).data
    assert state.goal == "current goal"
    assert state.agents == (agent,)
    assert agent.state == AgentState(h.root_agent_id, "interrupted", "llm_outage:timeout")
    rec.status, rec.status_reason = "idle", "settled"
    state = (await rt.get_session(h.session_id)).data.state
    assert not state.has_interrupted_agents
    assert state.agents[0].state.status == "idle"
    assert state.agents[0].state.status_reason == "settled"


@pytest.mark.parametrize("snapshot", ["none", "current", "old_compact"])
async def test_cold_state_and_initialization_preserve_goal_and_all_agent_sources(snapshot):
    rt = runtime()
    h = await container(rt)
    sid, aid = h.session_id, h.root_agent_id
    await append(rt.event_store, sid,
        (EventType.RECOGNIZE_INTENT_TOOL_CALL, None, aid, {"session_goal": "saved goal"}),
        (EventType.AGENT_INSTANTIATED, None, "child", {"template_id": "agent:tpl_echo"}),
        (EventType.AGENT_INTERRUPTED, None, "child", {"reason": "child error"}),
        (EventType.AGENT_INTERRUPTED, None, aid, {"reason": "root error"}),
        (EventType.AGENT_INSTANTIATED, None, "unknown", {"template_id": "agent:tpl_echo"}),
        (EventType.AGENT_INTERRUPTED, None, "unknown", {}),
    )
    if snapshot != "none":
        view = await rebuild_view(rt.event_store, sid)
        overrides = None
        if snapshot == "old_compact":
            from ctx_weft.core.control.reducers import snapshot_event_payload
            payload = snapshot_event_payload(view, cut=await rt.event_store.committed_head(sid),
                                             chain_depth=0, reason="test")
            compact = payload["startup_state_blob"]
            compact["sessions"][sid].pop("goal")
            overrides = {"startup_state_blob": compact, "startup_projection_version": 1}
        await seed_snapshot(rt.event_store, sid, view,
                            cut=await rt.event_store.committed_head(sid), overrides=overrides)
    assert rt.forget_session(sid)
    head = await rt.event_store.committed_head(sid)
    result = await rt.get_session(sid)
    state = result.data.state
    assert result.source == "events" and not result.loaded
    assert state.goal == "saved goal"
    assert state.has_interrupted_agents
    assert {a.agent_id: a.state.status_reason for a in state.agents} == {
        aid: "root error", "child": "child error", "unknown": "",
    }
    assert [a.agent_id for a in state.agents] == sorted(a.agent_id for a in state.agents)
    for expected in state.agents:
        assert (await rt.get_agent(expected.agent_id, session_id=sid)).data == expected
    assert await rt.event_store.committed_head(sid) == head
    assert not rt._task_managers
    fresh = runtime(rt.event_store)
    initialized = await fresh.initialize(sid)
    assert initialized.state == state
    assert not fresh._task_managers
    await fresh.rebuild_session(sid)
    assert (await fresh.get_session(sid)).data.state == state


async def test_agent_current_task_state_agrees_with_session_and_task_query():
    rt = runtime()
    h = await container(rt)
    handle = await rt.dispatch_task(h.session_id, "queued", agent_id=h.root_agent_id)
    task = (await rt.get_task(h.session_id, handle.task_id)).data
    assert task.status == "ACTIVE"  # admitted task can still be waiting for a run slot
    expected = AgentState(h.root_agent_id, "idle", "", task)
    assert (await rt.get_agent(h.root_agent_id)).data.state == expected
    assert (await rt.get_session(h.session_id)).data.state.agents[0].state == expected
    cold = runtime(rt.event_store)
    assert (await cold.get_agent(h.root_agent_id, session_id=h.session_id)).data.state == expected
    assert (await cold.get_session(h.session_id)).data.state.agents[0].state == expected
    assert not cold._task_managers


@pytest.mark.parametrize("snapshot", [False, True, "old_full"])
async def test_complete_public_objects_survive_failed_task_recovery_and_copy_isolation(snapshot):
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
    from ctx_weft.core.control.reducers import prune_view_for_snapshot
    rt = runtime()
    h = await container(rt)
    handle = await rt.dispatch_task(h.session_id, "failure task", agent_id=h.root_agent_id)
    manager = rt._task_managers[h.session_id]
    manager.get_task(handle.task_id).retry_count = 3
    await manager.apply_run_outcome(handle.task_id, RunOutcome(
        RunOutcomeKind.COMPLETED, verdict="fail", error="recorded root cause"))
    await manager._settle(handle.task_id, "FAILED")
    # Complete the finalization envelope, as the real loop does before settling.
    task = manager.get_task(handle.task_id)
    await manager._emit(EventType.TASK_FINALIZED, task_id=task.id,
                        payload={"finished_at": task.finished_at.isoformat()})
    task.finalized = True
    hot = (await rt.get_session(h.session_id)).data.state
    assert isinstance(hot.root_agent, AgentDetail) and isinstance(hot.root_task, TaskDetail)
    assert hot.root_task.error_code == "TASK_FAILED_BY_OBSERVER"
    assert hot.root_task.error == "recorded root cause" and hot.root_task.failure_retry_count == 3
    assert hot.root_agent.state.current_task == hot.root_task
    if snapshot:
        view = await rebuild_view(rt.event_store, h.session_id)
        overrides = None
        if snapshot == "old_full":
            from ctx_weft.core.control.reducers import serialize_view
            blob = serialize_view(view)
            for task in blob["tasks"].values():
                task.pop("failure_retry_count", None)
            overrides = {"state_blob": blob}
        else:
            view = prune_view_for_snapshot(view)
        await seed_snapshot(rt.event_store, h.session_id, view,
                            cut=await rt.event_store.committed_head(h.session_id), overrides=overrides)
    cold = runtime(rt.event_store)
    before = await rt.event_store.committed_head(h.session_id)
    assert (await cold.get_session(h.session_id)).data.state == hot
    initialized = await cold.initialize(h.session_id)
    assert initialized.state == hot and not cold._task_managers
    assert await rt.event_store.committed_head(h.session_id) == before
    await cold.rebuild_session(h.session_id)
    assert (await cold.get_session(h.session_id)).data.state == hot
    # Model configs in the aggregate are detached using the existing query DTOs.
    hot.root_agent.llm_config.reasoning_effort = "low"
    assert (await rt.get_session(h.session_id)).data.state.root_agent.llm_config.reasoning_effort != "low"


@pytest.mark.parametrize("snapshot", [False, True])
async def test_initialization_and_session_query_include_real_outputs_and_hitl_bodies(snapshot):
    from ctx_weft.protocols.hitl import HitlAsk, ToolResultDelivery
    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "done", h.root_agent_id,
         {"task": {"id": "done", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_FINISHED, "done", h.root_agent_id, {"outputs": {"answer": [42]}}),
        (EventType.AGENT_IDLE, "done", h.root_agent_id, {}),
    )
    ask = HitlAsk(form="approval", delivery=ToolResultDelivery("call"), prompt="actual question",
                  detail="actual detail", fields=[{"name": "choice"}], proposal={"command": "run"})
    await rt.hitl.open(ask, session_id=h.session_id, task_id="done", agent_id=h.root_agent_id,
                       stage="authz", unattended=False)
    hot_request, = (await rt.get_session(h.session_id)).data.state.pending_hitl
    hot_request.fields[0]["name"] = "changed"
    hot_request.proposal["command"] = "changed"
    again, = (await rt.get_session(h.session_id)).data.state.pending_hitl
    assert again.fields == [{"name": "choice"}] and again.proposal == {"command": "run"}
    if snapshot:
        from ctx_weft.core.control.reducers import prune_view_for_snapshot
        await seed_snapshot(rt.event_store, h.session_id,
            prune_view_for_snapshot(await rebuild_view(rt.event_store, h.session_id)),
            cut=await rt.event_store.committed_head(h.session_id))
    cold = runtime(rt.event_store)
    report = await cold.initialize(h.session_id)
    state = (await cold.get_session(h.session_id)).data.state
    assert report.state == state and not cold._task_managers
    assert state.root_task.outputs == {"answer": [42]}
    request, = state.pending_hitl
    assert isinstance(request, HitlRequestView)
    assert (request.prompt, request.detail, request.fields, request.proposal) == (
        "actual question", "actual detail", [{"name": "choice"}], {"command": "run"})
    state.root_task.outputs["answer"].append(99)
    request.fields[0]["name"] = "changed"
    again = (await cold.get_session(h.session_id)).data.state
    assert again.root_task.outputs == {"answer": [42]}
    assert again.pending_hitl[0].fields == [{"name": "choice"}]


def test_experimental_wrappers_and_removed_query_field_aliases_are_absent():
    import ctx_weft.protocols.task as tasks
    import ctx_weft.protocols.hitl as hitl
    import ctx_weft.protocols.runtime as values
    assert not hasattr(tasks, "TaskState") and not hasattr(tasks, "TaskFailure")
    assert not hasattr(hitl, "HitlState") and not hasattr(hitl, "HitlRequestState")
    assert not hasattr(values, "TaskDetail")
    fields = values.SessionStateSummary.__dataclass_fields__
    assert not {"root_agent_id", "root_agent_status", "root_task_id", "root_task_status",
                "pending_hitl_count", "waiting_user_turn_agent_id"} & fields.keys()


async def test_missing_legacy_parent_does_not_fabricate_a_root_task_or_hide_the_current_task():
    rt = runtime()
    h = await container(rt)
    await append(rt.event_store, h.session_id,
        (EventType.TASK_CREATED, "child", h.root_agent_id,
         {"task": {"id": "child", "parent_task_id": "missing", "assigned_agent_id": h.root_agent_id}}),
        (EventType.TASK_FINISHED, "child", h.root_agent_id, {"outputs": "actual output"}),
        (EventType.AGENT_IDLE, "child", h.root_agent_id, {}))
    cold = runtime(rt.event_store)
    state = (await cold.get_session(h.session_id)).data.state
    assert state.root_task is None
    assert state.root_agent.state.current_task.task_id == "child"
    assert state.root_agent.state.current_task.outputs == "actual output"
