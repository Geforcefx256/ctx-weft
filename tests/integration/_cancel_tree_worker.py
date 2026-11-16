
"""Real task-tree cancellation across agents, including failures and crash recovery."""

import asyncio
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import LLMChunk, ToolCall
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM


class TreeLLM(_FinishEveryTaskLLM):
    def __init__(self):
        super().__init__()
        self.actors = 0
        self.tree_actors = 0
        self.grandchild_started = asyncio.Event()
        self.unrelated_started = asyncio.Event()
        self.unrelated_tokens = 0
        self.release_reports = asyncio.Event()

    def complete(self, request, stream=True):
        names = {t.name for t in request.tools}
        if "control__update_task_metadata" in names:
            return super().complete(request, stream)
        if "control__report_observation" in names:
            async def report():
                await self.release_reports.wait()
                async for chunk in self._stream(MockResponse(tool_calls=[ToolCall(id=self._id("obs"),
                    name="control__report_observation", arguments={"task_status": "success", "act_recap": "OBSOLETE REPORT"})]), request):
                    yield chunk
            return report()
        self.actors += 1
        unrelated = any("UNRELATED RUN" in str(getattr(m, "content", "")) for m in request.messages)
        if not unrelated:
            self.tree_actors += 1
        if not unrelated and self.tree_actors <= 2:
            arguments = {"title": "child" if self.tree_actors == 1 else "grandchild",
                         "task_prompt": "TREE CHILD" if self.tree_actors == 1 else "TREE GRANDCHILD"}
            if self.tree_actors == 1:
                arguments.update(use_subagent=True, subagent_template="agent__tpl_echo")
            return self._stream(MockResponse(tool_calls=[ToolCall(id=self._id("delegate"),
                name="control__delegate_task", arguments=arguments)]), request)
        if not unrelated and getattr(self, "finish_grandchild", False):
            self.grandchild_started.set()
            return self._stream(MockResponse(text="GRAND FINAL ANSWER", tool_calls=[]), request)
        ready = self.unrelated_started if unrelated else self.grandchild_started
        async def hold():
            ready.set()
            while True:
                if ready is self.unrelated_started:
                    self.unrelated_tokens += 1
                yield LLMChunk(kind="token", text="working ")
                await asyncio.sleep(.01)
        return hold()


def make_tree_runtime(store, memory, recovering=False, mechanical=False):
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    if mechanical:
        template.identity.pop("observe", None)
    provider.register(template)
    llm = TreeLLM()
    from ctx_weft.core.models.config import RuntimeConfig
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store,
                      config=RuntimeConfig(task_max_concurrent=0) if recovering else None)
    rt.providers.register_memory(memory)
    if recovering:
        llm.release_reports.set()
    return rt, llm


async def worker(path, cut, recovering=False):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            rt, llm = make_tree_runtime(store, memory, recovering=recovering)
            root_task = root_agent = child_task = None
            if recovering:
                view = await rebuild_view(store, "tree-session")
                root_agent = view.sessions["tree-session"].root_agent_id
                root_task = next(t.id for t in view.tasks.values() if str(t.user_prompt) == "TREE ROOT")
            failed = False
            original = store.append_batch
            async def append(sid, bid, events, *args, **kwargs):
                nonlocal failed
                for e in events:
                    if e.type == "AgentTerminated" and root_agent is not None:
                        assert e.agent_id == root_agent, "cancel_agent attempted to terminate a descendant agent"
                    failure_kind = {
                        "agent_commit_failed": "AgentTerminated",
                        "task_cancel_commit_failed": "TaskCanceled",
                        "task_cancel_commit_failed_crash": "TaskCanceled",
                        "child_cancel_commit_failed": "TaskCanceled",
                        "child_cancel_commit_failed_crash": "TaskCanceled",
                        "finalized_commit_failed": "TaskFinalized",
                        "finalized_commit_failed_crash": "TaskFinalized",
                    }.get(cut)
                    target = child_task if cut.startswith("child_cancel") else root_task
                    if not failed and failure_kind == e.type and e.task_id == target:
                        failed = True
                        raise OSError("injected cancellation commit failure")
                result = await original(sid, bid, events, *args, **kwargs)
                for e in events:
                    if cut == "agent_terminated" and e.type == "AgentTerminated" and e.payload.get("reason", "").startswith("cancel_agent:"):
                        os._exit(73)
                    if e.type == "TaskCanceled":
                        if cut == "root_canceled" and e.task_id == root_task:
                            os._exit(73)
                        if cut == "descendant_canceled" and e.task_id != root_task:
                            os._exit(73)
                    if cut == "finalized" and e.type == "TaskFinalized" and e.payload.get("outcome") == "cancelled":
                        os._exit(73)
                return result
            store.append_batch = append
            original_ingest = memory.ingest
            async def ingest(event, ctx):
                result = await original_ingest(event, ctx)
                if cut == "cancel_slot" and event.metadata.get("finish_policy", {}).get("outcome") == "cancelled":
                    os._exit(73)
                return result
            memory.ingest = ingest
            await initialize_all(rt)
            if recovering:
                await recover_fixture_session(rt, root_agent, session_id="tree-session")
                tm = rt._task_managers["tree-session"]
                while tm._background_asyncio_tasks:
                    await asyncio.gather(*list(tm._background_asyncio_tasks))
                raise AssertionError("recovery did not reach the crash injection")
            handle = await rt.start_session(SessionStartParams.create(session_id="tree-session",
                template_id="agent:tpl_echo", user_prompt="TREE ROOT",  unattended=True))
            root_task, root_agent = handle.task_id, handle.agent_id
            await asyncio.wait_for(llm.grandchild_started.wait(), 10)
            tm = rt._task_managers["tree-session"]
            async with asyncio.timeout(10):
                while tm.open_round_task_ids:
                    await asyncio.sleep(.01)
            # This agent is a descendant of the root agent, but its task is outside
            # the root task tree and must survive cancellation.
            other = await rt.dispatch_task("tree-session", "UNRELATED RUN", agent_id=handle.agent_id,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                port_key="other", unattended=True)
            await asyncio.wait_for(llm.unrelated_started.wait(), 10)
            async with asyncio.timeout(10):
                while tm.open_round_task_ids:
                    await asyncio.sleep(.01)
            tm._max_concurrent = 0
            queued = await rt.dispatch_task("tree-session", "UNRELATED QUEUED", agent_id=handle.agent_id,
                settings={"use_subagent": True, "subagent_template": "agent:tpl_echo"},
                port_key="queued", unattended=True)
            queued_status = tm.get_task(queued.task_id).status
            assert queued.task_id not in tm.running_task_ids()
            assert rt._agent_lifecycle_manager.record_of(handle.agent_id).current_task_id == root_task, (rt._agent_lifecycle_manager.record_of(handle.agent_id), [(t.id,t.status,t.user_prompt,t.parent_task_id,t.assigned_agent_id) for t in tm.all_tasks()])
            signals = []
            original_cancel = rt._cancel_run_token
            def signal(sid, tid):
                signals.append(tid)
                return original_cancel(sid, tid)
            rt._cancel_run_token = signal
            original_agent_status = rt._agent_lifecycle_manager.status_of(root_agent)
            tree_tasks = [t for t in tm.all_tasks() if str(t.user_prompt).startswith("TREE")]
            child_task = next(t.id for t in tree_tasks if str(t.user_prompt) == "TREE CHILD")
            try:
                await rt.cancel_agent(handle.agent_id, reason="cancel tree")
            except PersistenceUnavailableError:
                assert failed and rt.storage_health("tree-session") is not None
                rows = await store.read_range("tree-session")
                if cut == "agent_commit_failed":
                    assert rt._agent_lifecycle_manager.status_of(root_agent) == original_agent_status
                    assert not signals
                    assert all(t.status not in {"FINISHED", "FAILED", "CANCELED"}
                               and not (t.error or "").startswith("cancel_agent:") for t in tree_tasks)
                    assert not any(r.event.type == "AgentTerminated" for r in rows)
                else:
                    assert rt._agent_lifecycle_manager.status_of(root_agent) == "terminated"
                    assert all(t.status == "CANCELED" or (t.error or "").startswith("cancel_agent:")
                               for t in tree_tasks)
                    missing_kind = "TaskCanceled" if cut.startswith(("task_cancel", "child_cancel")) else "TaskFinalized"
                    target = child_task if cut.startswith("child_cancel") else root_task
                    assert not any(r.event.type == missing_kind and r.event.task_id == target for r in rows)
                assert rt._agent_lifecycle_manager.status_of(other.agent_id) == "running"
                if cut.endswith("_crash"):
                    os._exit(73)
                rt.clear_storage_isolation("tree-session")
                await rt.cancel_agent(handle.agent_id, reason="cancel tree")
            else:
                assert "commit_failed" not in cut, "persistence failure was not injected"
            affected = [t.id for t in tm.all_tasks() if str(t.user_prompt).startswith("TREE")]
            assert other.task_id not in affected
            assert set(signals).issubset(set(affected))
            llm.release_reports.set()
            async with asyncio.timeout(15):
                while any(tm.get_task(tid).status != "CANCELED" or not tm.get_task(tid).finalized for tid in affected):
                    await asyncio.sleep(.01)
                while tm._background_asyncio_tasks:
                    await asyncio.gather(*list(tm._background_asyncio_tasks))
            before = llm.unrelated_tokens
            await asyncio.sleep(.03)
            assert llm.unrelated_tokens > before, "unrelated producer stopped during cancellation"
            assert tm.get_task(other.task_id).status == "ACTIVE"
            lm = rt._agent_lifecycle_manager
            assert lm.status_of(other.agent_id) == "running"
            assert tm.get_task(queued.task_id).status == queued_status
            assert queued.task_id not in tm.running_task_ids()
            assert lm.status_of(queued.agent_id) == "idle"
            rows = await store.read_range("tree-session")
            terminated = [r.event.agent_id for r in rows if r.event.type == "AgentTerminated"]
            assert terminated == [root_agent]
            for task in tree_tasks:
                if task.assigned_agent_id != root_agent:
                    assert lm.status_of(task.assigned_agent_id) == "idle"
                    lm.assert_can_receive(task.assigned_agent_id)
            os._exit(0 if cut == "none" or cut.endswith("commit_failed") else 74)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2], recovering="recover" in sys.argv[3:]))
