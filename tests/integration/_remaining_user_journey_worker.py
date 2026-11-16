"""Each stage runs in a separate operating-system process against the real stores."""

import asyncio
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._initialization import initialize_all

from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._remaining_user_journeys import (
    SID,
    BoundaryStore,
    finish_journey,
    prepare,
    request_text,
    runtime,
    save,
    verify,
)


async def main(path, scenario, variant, stage):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            provider = BoundaryStore(store, scenario, variant, stage)
            rt, llm, _tool = runtime(path, provider, memory, scenario, variant, stage)
            await initialize_all(rt)
            try:
                if stage in {"hot", "prepare"}:
                    data = await prepare(rt, llm, path, store, scenario, variant, stage)
                    data["failed"] = scenario == "plan" and variant.endswith("fail")
                    if stage == "prepare":
                        save(path, data)
                        os._exit(73)
                else:
                    data = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
                    await rt.rebuild_session(SID, strict=True)
                if stage == "verify":
                    await verify(rt, llm, path, store, memory, data, scenario)
                else:
                    data = await finish_journey(rt, llm, path, provider, store, memory, data, scenario, variant, stage)
                    save(path, data)
            except Exception:
                tm = rt._task_managers.get(SID)
                rows = await store.read_range(SID)
                diagnostic = {
                    "tasks": [(t.id, t.assigned_agent_id, t.status, t.outputs, t.error)
                              for t in tm.all_tasks()] if tm else [],
                    "pending": [(q.id, q.task_id, q.agent_id) for q in rt.hitl_registry.list_pending(session_id=SID)],
                    "requests": [(kind, request_text(request)) for kind, request in llm.calls],
                    "events": [(r.position, r.event.type, r.event.task_id, r.event.payload) for r in rows],
                }
                (path / "diagnostics.json").write_text(json.dumps(diagnostic, default=str), encoding="utf-8")
                print("Journey diagnostics: " + str(path / "diagnostics.json"), file=sys.stderr)
                raise


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), *sys.argv[2:]))
