"""Session recovery and query facts are public SDK contracts."""
import asyncio
from unittest.mock import AsyncMock
import pytest
from ctx_weft import CtxWeftRuntime, RuntimeClosedError
from ctx_weft.protocols import ModelCallConfig
from ctx_weft.protocols.events import EventType, TRANSIENT_EVENT_TYPES
from tests.unit.test_runtime_initialization_queries import runtime, container

@pytest.mark.asyncio
async def test_cold_and_hot_empty_container_recovery_are_both_noops():
    first = runtime()
    handle = await container(first)
    await first.recover_session(handle.session_id)
    second = runtime(first.event_store)
    try:
        await second.recover_session(handle.session_id)
        owner = second._task_managers[handle.session_id]
        assert owner.all_tasks() == []
        assert not owner.recovery_pending
        same = owner
        await second.recover_session(handle.session_id)
        assert second._task_managers[handle.session_id] is same
    finally:
        await second.close()
        await first.close()

@pytest.mark.asyncio
async def test_storage_confirmation_precedes_the_session_command_lock(monkeypatch):
    core = runtime()
    order = []
    monkeypatch.setattr(core, "storage_health", lambda sid: "isolated")
    async def storage(sid):
        assert core._session_lock_owners.get(sid) is not asyncio.current_task()
        order.append("storage")
        return True
    async def recover(sid):
        assert core._session_lock_owners[sid] is asyncio.current_task()
        order.append("recover")
    monkeypatch.setattr(core, "recover_storage", storage)
    monkeypatch.setattr(core, "initialize", AsyncMock(return_value=None))
    monkeypatch.setattr(core, "_rebuild_session_locked", AsyncMock())
    monkeypatch.setattr(core, "_recover_session_locked", recover)
    await core.recover_session("session")
    assert order == ["storage", "recover"]
    monkeypatch.setattr(core, "storage_health", lambda sid: None)
    await core.close()

@pytest.mark.asyncio
async def test_existing_session_and_agent_queries_include_detached_facts_and_config():
    core = runtime()
    handle = await container(core)
    try:
        cfg = ModelCallConfig(reasoning_effort="high")
        await core.set_agent_llm(handle.root_agent_id, llm_account="account", llm_model="model", llm_config=cfg)
        current = await core.get_session(handle.session_id)
        assert current.data.state.root_agent.agent_id == handle.root_agent_id
        assert not current.data.state.has_running_agents
        detail = (await core.get_agent(handle.root_agent_id, session_id=handle.session_id)).data
        assert (detail.llm_account, detail.llm_model) == ("account", "model")
        detail.llm_config.reasoning_effort = "low"
        again = (await core.get_agent(handle.root_agent_id, session_id=handle.session_id)).data
        assert again.llm_config.reasoning_effort == "high"
        assert core.forget_session(handle.session_id)
        cold = await core.get_session(handle.session_id)
        assert cold.data.state == current.data.state
        saved = (await core.get_agent(handle.root_agent_id, session_id=handle.session_id)).data
        assert saved.llm_config.reasoning_effort == "high"
        assert not cold.loaded
    finally:
        await core.close()

@pytest.mark.asyncio
async def test_recovery_is_rejected_after_close():
    core = runtime()
    handle = await container(core)
    await core.close()
    with pytest.raises(RuntimeClosedError):
        await core.recover_session(handle.session_id)

def test_removed_alias_and_transient_round_contract():
    assert not hasattr(CtxWeftRuntime, "recover_agent")
    assert not hasattr(CtxWeftRuntime, "prepare_agent")
    assert not hasattr(CtxWeftRuntime, "rebuild_hitl")
    assert not hasattr(CtxWeftRuntime, "send_session_message")
    assert {EventType.ROUND_COMMITTED, EventType.ROUND_DISCARDED} <= TRANSIENT_EVENT_TYPES

@pytest.mark.asyncio
async def test_successful_close_clears_confirmed_storage_isolation():
    core = runtime()
    handle = await container(core)
    core._mark_storage_unavailable(handle.session_id, asyncio.CancelledError())
    report = await core.close()
    assert report.durable and report.quiescent
    assert core.storage_health(handle.session_id) is None
    assert (await core.get_session(handle.session_id)).data.state.root_agent.agent_id == handle.root_agent_id
