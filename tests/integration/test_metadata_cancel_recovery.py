
from tests._recovery import recover_fixture_session
"""Slow metadata cannot hold cancellation hostage or cancel an independent report."""

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration._metadata_cancel_worker import REPLY, SID, content
from tests.integration._session_cancel_worker import runtime, wait_until
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM

WORKER = Path(__file__).with_name("_metadata_cancel_worker.py")


async def run_worker(path, mode, cut):
    proc = await asyncio.to_thread(subprocess.run,
        [sys.executable, str(WORKER), str(path), mode, cut],
        cwd=WORKER.parents[2], capture_output=True, text=True, timeout=40)
    assert proc.returncode == (0 if cut == "none" or cut.endswith("_hot_receipt") else 73), proc.stdout + proc.stderr


def assert_order(events):
    for start in [e for e in events if e.type == "RunStarted"]:
        finish, = [e for e in events if e.type == "RunFinished" and e.run_id == start.run_id]
        assert events.index(start) < events.index(finish)
        if start.origin == "runtime":
            for cancel in [e for e in events if e.type == "TaskCanceled" and e.task_id == start.task_id]:
                assert events.index(finish) < events.index(cancel)
    for cancel in [e for e in events if e.type == "TaskCanceled"]:
        finalized, = [e for e in events if e.type == "TaskFinalized" and e.task_id == cancel.task_id]
        assert events.index(cancel) < events.index(finalized)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["slow", "return", "detached", "report_short", "report_long"])
async def test_cancel_exits_dispatch_without_waiting_for_auxiliary_results(tmp_path, mode):
    await run_worker(tmp_path, mode, "none")
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        assert_order([r.event for r in await store.read_range(SID)])


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", ["metadata_hot_receipt", "actor_hot_receipt", "response_hot_receipt"])
async def test_cancellation_finish_receipt_loss_can_be_retried_hot(tmp_path, cut):
    await run_worker(tmp_path, "slow", cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        assert_order([r.event for r in await store.read_range(SID)])


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,cut", [
    ("slow", "metadata_closed"), ("slow", "metadata_receipt"),
    ("slow", "response_receipt"),
    ("slow", "task_canceled"), ("slow", "agent_terminated"),
    ("return", "metadata_receipt"), ("return", "task_canceled"),
    ("detached", "metadata_closed"), ("slow", "actor_receipt"),
    ("report_short", "task_canceled"), ("report_long", "task_canceled"),
    ("report_short", "agent_terminated"), ("report_long", "observe_completed"),
])
async def test_cancel_recovery_keeps_run_order_and_report_handoff(tmp_path, mode, cut):
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            rt = runtime(store, memory, _FinishEveryTaskLLM(), recovering=True)
            await initialize_all(rt)
            await rt.rebuild_session(SID, strict=True)
            root = (await rebuild_view(store, SID)).sessions[SID].root_agent_id
            await rt.cancel_session(SID)
            await recover_fixture_session(rt, root, session_id=SID)
            tm = rt._task_managers[SID]
            await wait_until(lambda: not tm.running_task_ids() and not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            view = await rebuild_view(store, SID)
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert all(task.title != "LATE TITLE" for task in view.tasks.values())
            assert rt._agent_lifecycle_manager.record_of(root).status == "terminated"
            events = [r.event for r in await store.read_range(SID)]
            assert_order(events)
            assert sum(e.type == "RunStarted" and e.origin == "loop.recognize_intent" for e in events) == 1
            task_id = next(e.task_id for e in events if e.type == "TaskCreated")
            raw = await content(memory, task_id, root)
            assert REPLY not in str(raw)
            if mode.startswith("report_"):
                assert "ORIGINAL ANSWER" in str(raw)
            head = await store.committed_head(SID)
            await rt.cancel_session(SID)
            rt2 = runtime(store, memory, _FinishEveryTaskLLM(), recovering=True)
            await initialize_all(rt2)
            await rt2.rebuild_session(SID, strict=True)
            await recover_fixture_session(rt2, root, session_id=SID)
            await wait_until(lambda: not rt2._task_managers[SID]._background_asyncio_tasks)
            assert await store.committed_head(SID) == head
            assert await content(memory, task_id, root) == raw
