
"""Crash at the atomic queued cancellation boundary, or during its recovery."""

import asyncio
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._queued_session_cancel_worker import SID, runtime
from tests.integration.test_terminal_submission_repair import setup_queue


async def worker(path, cut):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            if cut == "recover_finalized":
                rt, _ = runtime(store, memory, recovering=True)
                view = await rebuild_view(store, SID)
                owner = next(t.assigned_agent_id for t in view.tasks.values() if t.user_prompt == "QUEUED ONE")
                target = None
            else:
                rt, llm, first, _ = await setup_queue(store, memory)
                llm.release_reports.set()
                target = first.task_id
            original = store.append_batch

            async def append(sid, bid, events, *args, **kwargs):
                match = any((e.type == "TaskCanceled" and e.task_id == target) or
                            (cut == "recover_finalized" and e.type == "TaskFinalized") for e in events)
                if match and cut == "before":
                    os._exit(73)
                result = await original(sid, bid, events, *args, **kwargs)
                if match:
                    os._exit(73)
                return result

            store.append_batch = append
            if cut == "recover_finalized":
                await initialize_all(rt)
                await rt.rebuild_session(SID, strict=True)
                await recover_fixture_session(rt, owner, session_id=SID)
            else:
                await rt.pause_session(SID)
            await asyncio.sleep(10)
            raise AssertionError("crash injection not reached")


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2]))
