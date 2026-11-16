
from tests._recovery import recover_fixture_session
"""Two real questions, withdraw one answer, finish, restart and send more work."""

import asyncio

import pytest

from ctx_weft import RuntimeConfig, SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_hitl_e2e_v2 import _all_request_text, _finish_call
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    STALL_BEFORE_CHUNK,
    _ask,
    _next_question,
    _ScriptedLLM,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM
from tests.integration.test_user_journey_audit import assert_event_contract

SID = "whole-hitl-journey"


def runtime(store, memory, llm, timeout):
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store,
                      config=RuntimeConfig(hitl_timeout_sec=timeout))
    rt.providers.register_memory(memory)
    return rt


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0], ids=["hot", "cold"])
@pytest.mark.parametrize("long_answer", [False, True], ids=["short", "long"])
@pytest.mark.parametrize("restart_after_retraction", [False, True], ids=["same_process", "restart_before_reanswer"])
async def test_two_questions_with_retraction_then_restart_and_new_task(tmp_path, timeout, long_answer, restart_after_retraction):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            q2 = _ask("q2", "Second decision?")
            # Use real longer conversation content with the default folding
            # configuration; do not force long/short classification in core.
            if long_answer:
                q2.text = "Evidence from the completed investigation. " * 1200
            llm = _ScriptedLLM([_ask("q1", "First decision?"), q2,
                STALL_BEFORE_CHUNK, _finish_call()], observer_verdicts=["success"])
            rt = runtime(store, memory, llm, timeout)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="PROCESS_REQUEST"))
            first = await _next_question(rt, SID)
            if timeout == 0:
                await wait_until(lambda: not rt._task_managers[SID]._dispatch_tasks)
            await rt.reply_to_hitl(HitlReply(hitl_id=first.id, agent_id=h.agent_id,
                outcome="accepted", message="FIRST_APPROVAL"))
            second = await _next_question(rt, SID, exclude={first.id})
            if timeout == 0:
                await wait_until(lambda: not rt._task_managers[SID]._dispatch_tasks)
            await rt.reply_to_hitl(HitlReply(hitl_id=second.id, agent_id=h.agent_id,
                outcome="accepted", message="WITHDRAWN_APPROVAL"))
            await asyncio.wait_for(llm.stalled.wait(), 10)
            await rt.pause_agent(h.agent_id)
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm._dispatch_tasks)
            pending, = rt.hitl_registry.list_pending(session_id=SID)
            assert pending.id == second.id
            if restart_after_retraction:
                llm = _ScriptedLLM([_finish_call()], observer_verdicts=["success"])
                rt = runtime(store, memory, llm, timeout)
                await initialize_all(rt)
                await rt.rebuild_session(SID, strict=True)
                tm = rt._task_managers[SID]
            await rt.reply_to_hitl(HitlReply(hitl_id=second.id, agent_id=h.agent_id,
                outcome="accepted", message="FINAL_APPROVAL"))
            await wait_until(lambda: tm.get_task(h.task_id).status == "FINISHED")
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            last = _all_request_text(llm.act_requests[-1])
            assert "FIRST_APPROVAL" in last and "FINAL_APPROVAL" in last and "WITHDRAWN_APPROVAL" not in last
            rows = await store.read_range(SID)
            leaks = [(r.event.type, r.event.origin, str(r.event.payload)[:600]) for r in rows
                     if "WITHDRAWN_APPROVAL" in str(r.event.payload)]
            assert not leaks, leaks
            assert sum(r.event.type == "HitlResolved" and r.event.payload.get("hitl_id") == first.id for r in rows) == 1
            assert sum(r.event.type == "HitlResolved" and r.event.payload.get("hitl_id") == second.id for r in rows) == 1
            view = await rebuild_view(store, SID)
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert not tm.running_task_ids() and not tm.open_round_task_ids and not rt._run_tokens.get(SID)
            async def content():
                records = await load_records(memory, MemoryAddress(session_id=SID, agent_id=h.agent_id),
                    MemoryScope.AGENT, ProviderContext(session_id=SID))
                return [(r.id, r.content) for r in records]
            before, head = await content(), await store.committed_head(SID)
            cold = runtime(store, memory, _FinishEveryTaskLLM(), timeout)
            await initialize_all(cold)
            await cold.rebuild_session(SID, strict=True)
            await recover_fixture_session(cold, h.agent_id, session_id=SID)
            await wait_until(lambda: not cold._task_managers[SID]._background_asyncio_tasks)
            assert await store.committed_head(SID) == head and await content() == before
            nxt = await cold.send_message(h.agent_id, "NEW_REQUEST", session_id=SID, unattended=True)
            assert nxt.task_id != h.task_id
            await nxt.wait_for_finish(timeout=15)
            await assert_event_contract(store, SID)
