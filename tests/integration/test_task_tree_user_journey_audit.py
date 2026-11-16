"""Cancel a real delegated tree, finish unaffected work, then reuse a child agent."""

import asyncio

import pytest

from ctx_weft import RuntimeConfig, SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._cancel_tree_worker import TreeLLM
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM
from tests.integration.test_user_journey_audit import assert_event_contract

SID = "whole-tree-journey"


class FinishingTreeLLM(TreeLLM):
    def __init__(self):
        super().__init__()
        self.release_others = asyncio.Event()
        self.unrelated_closed = asyncio.Event()
        self.release_reports.set()

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        if names.intersection({"control__report_observation", "control__update_task_metadata"}):
            return super().complete(request, stream)
        text = str([m.content for m in request.messages])
        if "UNRELATED" in text:
            async def other():
                try:
                    self.unrelated_started.set()
                    from ctx_weft.protocols import LLMChunk
                    yield LLMChunk(kind="token", text="unrelated working")
                    await self.release_others.wait()
                    async for chunk in _FinishEveryTaskLLM.complete(self, request, stream):
                        yield chunk
                finally:
                    self.unrelated_closed.set()
            return other()
        return super().complete(request, stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["none", "before", "unknown"])
@pytest.mark.parametrize("short", [False, True])
async def test_cancel_tree_preserves_queued_work_and_child_agent_reuse(tmp_path, fault, short):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            provider = InlineAgentTemplateProvider()
            template = make_echo_template()
            template.loop_config.short_segment_token_threshold = 100000 if short else 0
            provider.register(template)
            llm = FinishingTreeLLM()
            rt = make_runtime(llm=llm, agent_provider=provider, event_store=store,
                              config=RuntimeConfig(task_max_concurrent=2))
            rt.providers.register_memory(memory)
            await initialize_all(rt)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="TREE ROOT",  unattended=True))
            await asyncio.wait_for(llm.grandchild_started.wait(), 10)
            tm = rt._task_managers[SID]
            other = await rt.dispatch_task(SID, "UNRELATED RUN", agent_id=h.agent_id,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                unattended=True, port_key="other")
            await asyncio.wait_for(llm.unrelated_started.wait(), 10)
            await wait_until(lambda: not tm.open_round_task_ids)
            queued = await rt.dispatch_task(SID, "UNRELATED QUEUED", agent_id=h.agent_id,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                unattended=True, port_key="queued")
            assert queued.task_id not in tm.running_task_ids()
            original = store.append_batch
            failed = []
            async def append(sid, bid, events, *args, **kwargs):
                if (fault != "none" and not failed
                        and any(e.type == "TaskCanceled" and e.task_id == h.task_id for e in events)):
                    failed.append(bid)
                    if fault == "unknown":
                        await original(sid, bid, events, *args, **kwargs)
                    raise OSError("tree cancellation receipt unavailable")
                return await original(sid, bid, events, *args, **kwargs)
            store.append_batch = append
            try:
                await rt.cancel_agent(h.agent_id)
            except PersistenceUnavailableError:
                assert failed
                rt.clear_storage_isolation(SID)
                await rt.cancel_agent(h.agent_id)
            store.append_batch = original
            assert await rt.recover_storage(SID)
            tree = [t for t in tm.all_tasks() if str(t.user_prompt).startswith("TREE")]
            await wait_until(lambda: all(t.status == "CANCELED" for t in tree))
            assert len(tree) == 3 and tm.get_task(other.task_id).status == "ACTIVE"
            assert not llm.unrelated_closed.is_set()
            llm.release_others.set()
            await other.wait_for_finish(timeout=15)
            await queued.wait_for_finish(timeout=15)
            assert tm.get_task(queued.task_id).status == "FINISHED"
            assert rt._agent_lifecycle_manager.status_of(other.agent_id) == "idle"
            followup = await rt.send_message(other.agent_id, "UNRELATED FOLLOWUP", session_id=SID, unattended=True)
            assert followup.task_id != other.task_id
            await followup.wait_for_finish(timeout=15)
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            view = await rebuild_view(store, SID)
            assert not view.pending_observe and not view.open_runs and not view.hitl.pending
            assert not tm.running_task_ids()
            assert view.agents[h.agent_id].status == "terminated"
            assert all(a.status != "terminated" for aid, a in view.agents.items() if aid != h.agent_id)
            await assert_event_contract(store, SID)
