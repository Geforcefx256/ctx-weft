"""Run each public user workflow phase in a fresh OS process."""

import asyncio
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._initialization import initialize_all

from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._user_control_journey import (
    SID,
    SlowApprovalStore,
    complete,
    prepare,
    runtime,
    save,
    verify_restart,
)


async def main(path, scenario, stage, variant):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            provider = SlowApprovalStore(store) if scenario == "approval" and variant == "hot" else store
            rt, a, b, tool = runtime(provider, memory, scenario,
                                      "initial" if stage in {"prepare", "hot"} else "continued", variant)
            await initialize_all(rt)
            if stage in {"prepare", "hot"}:
                data = await prepare(rt, a, store, memory, scenario, variant)
                if stage == "prepare":
                    save(path, data)
                    # A real application exit while paused/awaiting approval, or
                    # while its independent old report is still in flight.
                    os._exit(73)
            else:
                data = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
                await rt.rebuild_session(SID, strict=True)
            if stage == "verify":
                await verify_restart(rt, a, b, store, memory, data, scenario)
            else:
                data = await complete(rt, a, b, tool, store, memory, data, scenario, restarted=stage == "continue")
                save(path, data)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), *sys.argv[2:]))
