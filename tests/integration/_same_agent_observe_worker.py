
"""Crash the real delegate/observe/finalize path for same-agent children."""

import asyncio
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.core.loop.steps.finalize import FinalizeStep
from ctx_weft.core.loop.steps.observe import ObserveStep
from ctx_weft.protocols import ToolCall
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_dispatch_boundary_recap_e2e import _RouterLLM
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


class ChildLLM(_RouterLLM):
    def __init__(self, recovering=False):
        super().__init__()
        self._act_calls = 2 if recovering else 0
        self.new_actor_calls = 0

    def complete(self, request, stream=True):
        names = {tool.name for tool in request.tools}
        if "control__update_task_metadata" in names:
            return super().complete(request, stream)
        if "control__report_observation" in names:
            return self._stream(MockResponse(tool_calls=[ToolCall(id=self._id("report"),
                name="control__report_observation", arguments={"task_status": "success",
                "act_recap": "FORMAL REPORT", "task_summary": "FORMAL REPORT"})]), request)
        self.new_actor_calls += 1
        if self._act_calls == 0:
            return super().complete(request, stream)
        self._act_calls += 1
        answer = "CHILD FINAL ANSWER" if self._act_calls == 2 else "ROOT FINAL ANSWER"
        return self._stream(MockResponse(text=answer, tool_calls=[]), request)


def make_child_runtime(store, memory, size, recovering=False):
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.identity.pop("observe")
    template.loop_config.short_segment_token_threshold = 100000 if size == "short" else 0
    template.loop_config.short_task_token_threshold = 100000 if size == "short" else 0
    provider.register(template)
    llm = ChildLLM(recovering)
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store)
    rt.providers.register_memory(memory)
    return rt, llm


async def worker(path, order, size, cut, recovering=False):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            rt, llm = make_child_runtime(store, memory, size, recovering)
            from tests.runtime_helpers import install_mechanical_report_fixture
            install_mechanical_report_fixture(lambda state: bool(state.task.parent_task_id))
            children = set()
            report_ready = asyncio.Event()
            finalized = asyncio.Event()
            original_append = store.append_batch
            async def append(sid, bid, events, *args, **kwargs):
                if cut == "before_applied" and any(e.type == "ObserveApplied" and e.task_id in children for e in events):
                    os._exit(73)
                result = await original_append(sid, bid, events, *args, **kwargs)
                for event in events:
                    if event.type == "TaskCreated" and event.payload["task"].get("parent_task_id"):
                        children.add(event.payload["task"]["id"])
                    if event.task_id not in children:
                        continue
                    if event.type == "ObserveCompleted":
                        report_ready.set()
                    if event.type == "TaskFinalized":
                        finalized.set()
                    targets = {"requested": "ObserveRequested", "completed": "ObserveCompleted",
                        "task_finalized": "TaskFinalized", "task_finished": "TaskFinished", "applied": "ObserveApplied"}
                    if targets.get(cut) == event.type:
                        os._exit(73)
                return result
            store.append_batch = append
            original_ingest = memory.ingest
            async def ingest(event, ctx):
                result = await original_ingest(event, ctx)
                if cut == "finish_slot" and event.metadata.get("origin_task_id") in children and event.metadata.get("finish_policy"):
                    os._exit(73)
                return result
            memory.ingest = ingest
            original_fold = memory.fold
            async def fold(ids, events, ctx):
                result = await original_fold(ids, events, ctx)
                if cut == "report_replaced" and any(e.metadata.get("origin_task_id") in children and e.metadata.get("finish_policy") for e in events):
                    os._exit(73)
                if cut == "raw_folded" and ctx.task_id in children and ids and not events:
                    os._exit(73)
                return result
            memory.fold = fold
            commit_organize = memory._commit_organize
            async def organize_commit(ids, events, ctx):
                result = await commit_organize(ids, events, ctx)
                if result is not None and cut == "raw_folded" and ctx.task_id in children and ids and not events:
                    os._exit(73)
                return result
            memory._commit_organize = organize_commit

            if not recovering:
                original_finalize = FinalizeStep.execute
                async def ordered_finalize(self, state, ctx):
                    if state.task.parent_task_id and order == "early":
                        await report_ready.wait()
                    return await original_finalize(self, state, ctx)
                FinalizeStep.execute = ordered_finalize
                original_observe = ObserveStep._llm_observe
                async def ordered_report(self, state, ctx, events):
                    if state.task.parent_task_id and order == "late":
                        await finalized.wait()
                    return await original_observe(self, state, ctx, events)
                ObserveStep._llm_observe = ordered_report
            await initialize_all(rt)
            if recovering:
                from ctx_weft.core.control.reducers import rebuild_view
                view = await rebuild_view(store, "child-session")
                children.update(t.id for t in view.tasks.values() if t.parent_task_id)
                root = view.sessions["child-session"].root_agent_id
                await recover_fixture_session(rt, root, session_id="child-session")
                tm = rt._task_managers["child-session"]
            else:
                handle = await rt.start_session(SessionStartParams.create(session_id="child-session",
                    template_id="agent:tpl_echo", user_prompt="delegate then finish",
                    unattended=True))
                await handle.wait_for_finish(timeout=20)
                tm = rt._task_managers["child-session"]
            async with asyncio.timeout(20):
                while tm.running_task_ids() or tm._background_asyncio_tasks:
                    if tm._background_asyncio_tasks:
                        await asyncio.gather(*list(tm._background_asyncio_tasks))
                    await asyncio.sleep(.01)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], len(sys.argv) > 5))
