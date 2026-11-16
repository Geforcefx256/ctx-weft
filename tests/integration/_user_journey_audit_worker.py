
"""Exit after a real input commit, or while its recovery writes are in progress."""

import asyncio
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import prune_view_for_snapshot, rebuild_view
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._snapshot_helpers import seed_snapshot
from tests.integration.test_foreground_observe_interrupt import runtime
from tests.integration.test_user_journey_audit import SID, JourneyLLM, answer, idle


async def run(path, entry, snapshot, recovering):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            llm = JourneyLLM()
            rt = runtime(store, memory, llm, False)
            if recovering:
                original = memory.ingest
                async def ingest(event, ctx):
                    result = await original(event, ctx)
                    if "FINAL_REQUEST" in str(event.content):
                        os._exit(73)
                    return result
                memory.ingest = ingest
                await initialize_all(rt)
                await rt.rebuild_session(SID, strict=True)
                aid = rt._task_managers[SID].session.root_agent_id
                await recover_fixture_session(rt, aid, session_id=SID)
            else:
                await initialize_all(rt)
                h = await rt.start_session(SessionStartParams.create(session_id=SID,
                    template_id="agent:tpl_echo", user_prompt="INITIAL_REQUEST", unattended=True))
                await asyncio.wait_for(llm.old_review.wait(), 10)
                await rt.pause_agent(h.agent_id)
                await idle(rt)
                if snapshot:
                    await seed_snapshot(store, SID, prune_view_for_snapshot(await rebuild_view(store, SID)),
                        cut=await store.committed_head(SID), overrides={"projection_version": 10})
                original = store.append_batch
                async def append(sid, bid, events, *args, **kwargs):
                    result = await original(sid, bid, events, *args, **kwargs)
                    if any(e.type in {"TaskMessageAppended", "HitlResolved"}
                           and "FINAL_REQUEST" in str(e.payload) for e in events):
                        os._exit(73)
                    return result
                store.append_batch = append
                await answer(rt, h, entry, "FINAL_REQUEST")
            await asyncio.sleep(10)
            raise AssertionError("did not reach the crash injection")


if __name__ == "__main__":
    asyncio.run(run(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "snapshot", sys.argv[4] == "recover"))
