"""Public entry points persist agent choices and deliver request copies to adapters."""

from copy import deepcopy

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import ModelCallConfig, deserialize_call_config
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


class CapturingAdapter(MockLLMAdapter):
    def __init__(self):
        super().__init__([MockResponse(text="done") for _ in range(8)], observer_verdict="success")
        self.received = []

    def complete(self, request, stream=True):
        self.received.append((request.model, deepcopy(request.config)))
        request.config.reasoning_effort = "adapter-local"
        return super().complete(request, stream)


async def runtime(**kwargs):
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    llm = CapturingAdapter()
    rt = make_runtime(llm=llm, agent_provider=provider, **kwargs)
    rt.providers.register_memory(InMemoryMemoryProvider())
    await initialize_all(rt)
    return rt, llm


@pytest.mark.asyncio
async def test_start_session_delivers_config_without_adding_session_state():
    rt, llm = await runtime()
    config = ModelCallConfig(reasoning_effort="high", temperature=0.3, max_tokens=4096)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hello", llm_config=config,
    ))
    state = await handle.wait_for_finish(timeout=10)
    assert state.task.status == "FINISHED"
    assert llm.received and all(c.reasoning_effort == "high" for _, c in llm.received)
    assert all(c.max_tokens <= 4096 for _, c in llm.received)
    assert config.reasoning_effort == "high"
    assert state.resolved_model.config.reasoning_effort == "high"
    view = await rebuild_view(rt.event_store, handle.session_id)
    assert deserialize_call_config(view.agents[handle.agent_id].llm_config) == config
    assert not hasattr(view.sessions[handle.session_id], "llm_config")
    rows = await rt.event_store.read_range(handle.session_id)
    created = next(r.event for r in rows if r.event.type == EventType.SESSION_CREATED)
    assert "llm_config" not in created.payload


@pytest.mark.asyncio
async def test_container_creation_setting_and_dispatch_config_do_not_leak_between_agents():
    rt, _ = await runtime()
    a = await rt.create_session(template_id="agent:tpl_echo", llm_config=ModelCallConfig(reasoning_effort="high"))
    b = await rt.create_session(template_id="agent:tpl_echo", llm_config=ModelCallConfig(reasoning_effort="low"))
    assert await rt.set_session_llm(a.session_id, llm_config=ModelCallConfig(reasoning_effort="medium")) == 1
    av = await rebuild_view(rt.event_store, a.session_id)
    bv = await rebuild_view(rt.event_store, b.session_id)
    assert deserialize_call_config(av.agents[a.root_agent_id].llm_config).reasoning_effort == "medium"
    assert deserialize_call_config(bv.agents[b.root_agent_id].llm_config).reasoning_effort == "low"
    # A new independent tree has an explicit choice, not a session-global default.
    handle = await rt.dispatch_task(
        a.session_id, "hello", settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
        llm_config=ModelCallConfig(reasoning_effort="high"),
    )
    await handle.wait_for_finish(timeout=10)
    view = await rebuild_view(rt.event_store, a.session_id)
    assert view.tasks[handle.task_id].status == "FINISHED"
    assert deserialize_call_config(view.agents[handle.agent_id].llm_config).reasoning_effort == "high"


@pytest.mark.asyncio
async def test_run_single_task_forwards_config_to_act_and_observe():
    rt, llm = await runtime()
    _, state = await rt.run_single_task(
        template_id="agent:tpl_echo", user_prompt="hello",
        llm_config=ModelCallConfig(reasoning_effort="low"),
    )
    assert state.task.status == "FINISHED"
    assert all(c.reasoning_effort == "low" for _, c in llm.received)


@pytest.mark.asyncio
async def test_llm_provider_context_follows_session_subagent_and_cold_rebuild(tmp_path):
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store
    from tests.runtime_helpers import close_runtime

    async with open_sqlite_event_store(tmp_path / "context.sqlite") as store:
        rt, llm = await runtime(event_store=store)
        seen = []

        class Resolver:
            def get_client(self, account=None, model=None, *, ctx):
                seen.append(ctx)
                return llm

        rt.providers.register_llm_provider(Resolver())
        expected = {}
        root_ids = {}
        try:
            for tenant in ("opaque-a", "opaque-b"):
                session = await rt.create_session(template_id="agent:tpl_echo", tenant_id=tenant)
                expected[session.session_id] = tenant
                root_ids[session.session_id] = session.root_agent_id
                turn = await rt.dispatch_task(
                    session.session_id, "hello", tenant_id=tenant,
                    settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                )
                await turn.wait_for_finish(timeout=10)
                assert any(c.agent_id == turn.agent_id for c in seen)
            sid = next(iter(expected))
            assert rt.forget_session(sid)
            await rt.rebuild_session(sid)
            resumed = await rt.dispatch_task(
                sid, "again", agent_id=root_ids[sid], tenant_id=expected[sid],
                settings={"use_subagent": False},
            )
            await resumed.wait_for_finish(timeout=10)
            assert seen
            assert all(c.tenant_id == expected[c.session_id] for c in seen), [
                (c.session_id, c.tenant_id, expected[c.session_id]) for c in seen
            ]
            assert all(c.agent_id and c.agent_template_id == "agent:tpl_echo" for c in seen)
        finally:
            await close_runtime(rt)


@pytest.mark.asyncio
async def test_sql_persistence_and_cold_rebuild_deliver_the_subclass(tmp_path):
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store
    from ctx_weft.providers.llm import AnthropicCallConfig

    async with open_sqlite_event_store(tmp_path / "events.sqlite") as store:
        rt, llm = await runtime(event_store=store)
        handle = await rt.create_session(
            template_id="agent:tpl_echo",
            llm_config=AnthropicCallConfig(reasoning_effort="high", thinking_budget_tokens=2048),
        )
        selected = AnthropicCallConfig(reasoning_effort="low", thinking_budget_tokens=1024)
        await rt.set_agent_llm(handle.root_agent_id, llm_config=selected)
        assert rt.forget_session(handle.session_id)
        await rt.rebuild_session(handle.session_id)
        turn = await rt.dispatch_task(
            handle.session_id, "hello", agent_id=handle.root_agent_id,
            settings={"use_subagent": False},
        )
        await turn.wait_for_finish(timeout=10)
        assert llm.received
        assert all(type(c) is AnthropicCallConfig and c.thinking_budget_tokens == 1024
                   and c.reasoning_effort == "low" for _, c in llm.received)
        view = await rebuild_view(store, handle.session_id)
        assert deserialize_call_config(view.agents[handle.root_agent_id].llm_config) == selected


@pytest.mark.asyncio
@pytest.mark.parametrize("cold", [False, True], ids=["hot", "cold-reload"])
async def test_hitl_reply_keeps_running_snapshot_and_cold_reply_uses_updated_choice(cold):
    from ctx_weft.protocols import HitlReply
    from tests.integration.test_hitl_e2e_v2 import _finish_call, _poll
    from tests.integration.test_hitl_hot_reply_round_window_e2e import _ask, _runtime, _ScriptedLLM
    from tests.runtime_helpers import close_runtime, wait_for_park

    llm = _ScriptedLLM([_ask("question", "Continue?"), _finish_call(), _finish_call()],
                       observer_verdicts=["success"])
    rt = _runtime(llm, hitl_timeout_sec=0 if cold else None)
    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go",
        llm_config=ModelCallConfig(reasoning_effort="high"),
    ))
    sid = handle.session_id
    try:
        if cold:
            await wait_for_park(handle)
            # Restart with the durable stores, rather than evicting a live HITL.
            from ctx_weft import RuntimeConfig

            old_manager = rt._task_managers[sid]
            await _poll(lambda: not old_manager._dispatch_tasks)
            templates = InlineAgentTemplateProvider()
            templates.register(make_echo_template())
            fresh = make_runtime(
                llm=llm, agent_provider=templates, event_store=rt.event_store,
                config=RuntimeConfig(hitl_timeout_sec=0),
            )
            fresh.providers.register_memory(rt.providers.get_memory())
            rt = fresh
            await initialize_all(rt)
            await rt.rebuild_session(sid, strict=True)
        pending = await _poll(lambda: rt.hitl_registry.list_pending(session_id=sid))
        question = pending[0]
        if not cold:
            await _poll(lambda: question.slot)
        await rt.set_agent_llm(question.agent_id, llm_config=ModelCallConfig(reasoning_effort="low"))
        await rt.reply_to_hitl(HitlReply(
            hitl_id=question.id, agent_id=question.agent_id, outcome="accepted", message="continue",
        ), session_id=sid)
        await _poll(lambda: rt._task_managers[sid].get_task(question.task_id).status == "FINISHED")
        assert llm.act_requests[0].config.reasoning_effort == "high"
        assert llm.act_requests[1].config.reasoning_effort == ("low" if cold else "high")
        view = await rebuild_view(rt.event_store, sid)
        assert deserialize_call_config(view.agents[question.agent_id].llm_config).reasoning_effort == "low"
        following = await rt.send_message(question.agent_id, "next turn", session_id=sid)
        await following.wait_for_finish(timeout=10)
        assert llm.act_requests[2].config.reasoning_effort == "low"
    finally:
        await close_runtime(rt)
