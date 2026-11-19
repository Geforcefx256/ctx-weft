
from tests._recovery import recover_fixture_session
"""Interrupt an actual foreground review, then deliver a replacement user turn."""

import asyncio
import dataclasses

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import LLMChunk, load_records
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM

SID = "foreground-observe-interrupt"


class InterruptedReviewLLM(_FinishEveryTaskLLM):
    def __init__(self):
        super().__init__()
        self.review_started = asyncio.Event()
        self.review_closed = asyncio.Event()
        self.reviews = 0
        self.requests = []

    def complete(self, request, stream=True):
        self.requests.append(request)
        if "control__report_observation" in {t.name for t in request.tools}:
            self.reviews += 1
            if self.reviews == 1:
                async def stalled():
                    try:
                        self.review_started.set()
                        yield LLMChunk(kind="token", text="unfinished judgement")
                        await asyncio.Event().wait()
                    finally:
                        self.review_closed.set()
                return stalled()
        return super().complete(request, stream)


def runtime(store, memory, llm, short):
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    template.loop_config = dataclasses.replace(template.loop_config,
        short_segment_token_threshold=100000 if short else 0)
    provider.register(template)
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store)
    rt.providers.register_memory(memory)
    return rt


@pytest.mark.asyncio
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("cold", [False, True])
async def test_observe_pause_closes_stream_and_replaces_prompt(tmp_path, short, cold):
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            llm = InterruptedReviewLLM()
            rt = runtime(store, memory, llm, short)
            await initialize_all(rt)
            handle = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="OLD REQUEST", unattended=True))
            await asyncio.wait_for(llm.review_started.wait(), 10)
            await rt.pause_agent(handle.agent_id)
            await asyncio.wait_for(llm.review_closed.wait(), 5)
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            assert not tm.running_task_ids()
            view = await rebuild_view(store, SID)
            assert view.tasks[handle.task_id].status == "AWAITING_HUMAN"
            assert not view.open_runs and not view.pending_observe
            rows = await store.read_range(SID)
            old_request, = [r for r in rows if r.event.type == "ObserveRequested"
                              and r.event.payload["execution_mode"] == "foreground"]
            old_id = old_request.event.payload["observe_id"]
            completed, = [r for r in rows if r.event.type == "ObserveCompleted"
                             and r.event.payload["observe_id"] == old_id]
            applied, = [r for r in rows if r.event.type == "ObserveApplied"
                           and r.event.payload["observe_id"] == old_id]
            opened, = [r for r in rows if r.event.type == "HitlOpened"]
            assert completed.position < applied.position < opened.position
            assert completed.event.payload["accepted"] is False
            assert opened.event.payload["observe_request"]["judge_task"] is False
            assert opened.event.payload["observe_request"]["boundary"] == "interrupt"
            anchor = old_request.event.payload["reply_anchor"]
            handoff = opened.event.payload["observe_request"]
            assert handoff["reply_anchor"] == anchor
            assert handoff["replaces_observe_id"] == old_id
            assert "fork_checkpoint" not in applied.event.payload
            recap_applied, = [r for r in rows if r.event.type == "ObserveApplied"
                              and r.event.payload["observe_id"] == handoff["observe_id"]]
            assert recap_applied.event.payload["fork_checkpoint"]["reply_anchor"] == anchor
            if cold:
                # Rebuild twice before input: no duplicate judgement/report or slots.
                llm = _FinishEveryTaskLLM()
                rt = runtime(store, memory, llm, short)
                head = await store.committed_head(SID)
                await initialize_all(rt)
                await rt.rebuild_session(SID, strict=True)
                await recover_fixture_session(rt, handle.agent_id, session_id=SID)
                await recover_fixture_session(rt, handle.agent_id, session_id=SID)
                assert await store.committed_head(SID) == head
            await rt.send_message(handle.agent_id, "NEW REQUEST", session_id=SID, unattended=True)
            tm = rt._task_managers[SID]
            await wait_until(lambda: tm.get_task(handle.task_id).status == "FINISHED")
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext
            records = await load_records(memory, MemoryAddress(session_id=SID, agent_id=handle.agent_id,
                task_id=handle.task_id), MemoryScope.TASK, ProviderContext(session_id=SID))
            user = [str(r.content) for r in records if r.role == "user"]
            assert any("NEW REQUEST" in text and "OLD REQUEST" in text for text in user)
            assert not tm.running_task_ids()
            view = await rebuild_view(store, SID)
            assert not view.pending_observe and not view.open_runs


@pytest.mark.asyncio
@pytest.mark.parametrize("second_crash", [None, "ObserveCompleted", "ObserveApplied"])
async def test_interrupt_handoff_survives_process_and_recovery_crashes(tmp_path, second_crash):
    import subprocess
    import sys
    from pathlib import Path

    from tests.integration.test_queued_session_cancel_recovery import content
    worker = Path(__file__).with_name("_foreground_observe_interrupt_worker.py")
    for cut, recovering in [("HitlOpened", False), *([(second_crash, True)] if second_crash else [])]:
        result = await asyncio.to_thread(subprocess.run,
            [sys.executable, str(worker), str(tmp_path), cut, *(["recover"] if recovering else [])],
            cwd=worker.parents[2], capture_output=True, text=True, timeout=30)
        assert result.returncode == 73, result.stdout + result.stderr
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = runtime(store, memory, _FinishEveryTaskLLM(), False)
            await initialize_all(rt)
            await rt.rebuild_session(SID, strict=True)
            tm = rt._task_managers[SID]
            agent_id = tm.session.root_agent_id
            await recover_fixture_session(rt, agent_id, session_id=SID)
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            view = await rebuild_view(store, SID)
            assert len(view.hitl.pending) == 1
            assert not view.pending_observe and not view.open_runs
            assert not tm.running_task_ids()
            head = await store.committed_head(SID)
            before = await content(memory, view.tasks.values())
            await recover_fixture_session(rt, agent_id, session_id=SID)
            assert await store.committed_head(SID) == head
            assert await content(memory, view.tasks.values()) == before
            rows = await store.read_range(SID)
            for requested in [r.event for r in rows if r.event.type == "ObserveRequested"]:
                oid = requested.payload["observe_id"]
                assert len([r for r in rows if r.event.type == "ObserveCompleted"
                            and r.event.payload["observe_id"] == oid]) == 1
                assert len([r for r in rows if r.event.type == "ObserveApplied"
                            and r.event.payload["observe_id"] == oid]) == 1
