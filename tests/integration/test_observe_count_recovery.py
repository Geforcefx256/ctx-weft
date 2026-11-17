"""Real SQLite/process crashes preserve the task review count and prompt ordinal."""
import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_lifecycle_crash_boundaries import restart, run_worker
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["fourth_requested", "fourth_started", "fourth_completed"])
async def test_fourth_review_uses_original_ordinal_after_real_cold_recovery(tmp_path, snapshot, cut):
    await run_worker(tmp_path, "review_count", cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            before = await rebuild_view(store, "crash-session")
            task_before, = before.tasks.values()
            assert task_before.observe_count == (3 if cut == "fourth_requested" else 4)
            pending, = before.pending_observe.values()
            if cut != "fourth_requested":
                assert pending["observe_index"] == 4
            rt = await restart(store, memory, "review_count", snapshot)
            # Three prior reviews continued. An unfinished fourth must be retried
            # once, while a completed fourth must only finish applying its decision.
            rt._llm.judgements = 3
            prompts = []
            original = rt._llm.complete

            def capture(request, stream=True):
                if any(t.name == "control__report_observation" for t in request.tools):
                    prompts.append(request.system + "\n" + "\n".join(str(m.content) for m in request.messages))
                return original(request, stream)

            rt._llm.complete = capture
            try:
                await rt.recover_session("crash-session")
                tm = rt._task_managers["crash-session"]
                async with asyncio.timeout(15):
                    while (tm._dispatch_tasks or tm._background_asyncio_tasks
                           or any(t.status in {"ACTIVE", "PENDING"} for t in tm._tasks.values())):
                        await asyncio.sleep(0.01)
                task = tm.get_task(task_before.id)
                assert task.status == "FINISHED" and task.observe_count == 4
                assert rt._llm.actor_count == 0
                assert len(prompts) == (0 if cut == "fourth_completed" else 1)
                if prompts:
                    assert "Observe review 4" in prompts[0]
                    assert "3 previous reviews" in prompts[0]
                    assert "currently available capabilities" in prompts[0]
                rows = await store.read_range("crash-session")
                starts = [r.event.payload for r in rows if r.event.type == EventType.OBSERVE_STARTED
                          and r.event.payload.get("judge_task")]
                assert [p["observe_index"] for p in starts] == [1, 2, 3, 4]
                assert starts[-1]["observe_id"] == pending["observe_id"]
                assert_lifecycles(rows)
                after = await rebuild_view(store, "crash-session")
                assert after.tasks[task.id].observe_count == 4
                assert not after.pending_observe
                head = await store.committed_head("crash-session")
                await rt.recover_session("crash-session")
                assert await store.committed_head("crash-session") == head
            finally:
                await rt.close()
