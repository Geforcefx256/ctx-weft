
from tests._recovery import recover_fixture_session
"""Finish real work, crash its background judgement, then finish the user journey."""

import pytest

from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests._snapshot_helpers import seed_snapshot
from tests.integration._lifecycle_crash_worker import make_lifecycle_runtime
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_lifecycle_crash_boundaries import run_worker

SID = "crash-session"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["plain_text", "finish_park"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("cut", ["none", "hitl_opened", "observer_response", "observe_completed"])
async def test_finished_actor_background_judgement_preserves_result_and_allows_next_task(tmp_path, mode, snapshot, cut):
    await run_worker(tmp_path, mode, cut)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            if snapshot:
                await seed_snapshot(store, SID, prune_view_for_snapshot(await rebuild_view(store, SID)),
                    cut=await store.committed_head(SID), overrides={"projection_version": 10})
            rt, llm = make_lifecycle_runtime(store, memory, mode)
            await initialize_all(rt)
            await rt.rebuild_session(SID, strict=True)
            tm = rt._task_managers[SID]
            history = await store.read_range(SID, include_types=["TaskCreated"])
            task_id = history[0].event.task_id
            aid = tm.session.root_agent_id
            await recover_fixture_session(rt, aid, session_id=SID)
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            # /resume twice must not be necessary to rediscover an already
            # completed actor's deliverable, and must never silently change it.
            head = await store.committed_head(SID)
            await recover_fixture_session(rt, aid, session_id=SID)
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)
            assert await store.committed_head(SID) == head
            rows = await store.read_range(SID)
            results = [(r.event.payload.get("result", {}).get("verdict"), r.event.payload.get("action"))
                       for r in rows if r.event.type == "ObserveCompleted"]
            expected = "completed answer" if mode == "plain_text" else "done"
            task = (await rt.get_task(SID, task_id)).data
            assert (task.status, task.outputs) == ("FINISHED", expected), {
                "actual": (task.status, task.outputs), "decisions": results,
                "actor_reruns": llm.actor_count,
            }
            nxt = await rt.send_message(aid, "NEXT_REQUEST", session_id=SID, unattended=True)
            assert nxt.task_id != task_id
            await nxt.wait_for_finish(timeout=15)
