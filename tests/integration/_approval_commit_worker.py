
"""Inject storage failures at an actual user's hot approval reply."""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._user_control_journey import (
    REFUSAL,
    SID,
    SlowApprovalStore,
    complete,
    prepare,
    request_text,
    runtime,
    save,
)


class DecisionStore(SlowApprovalStore):
    """All commits are real SQL transactions; only provider failures are injected."""

    def __init__(self, store, mode):
        super().__init__(store)
        self.mode = mode.removesuffix("_recover")
        self.attempts = []
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def append_batch(self, sid, bid, events, *args, **kwargs):
        decision = any(e.type == "HitlResolved" and e.payload.get("outcome") == "rejected" for e in events)
        if self.mode == "recrash" and any(e.type == "HitlClosed" for e in events):
            await self.store.append_batch(sid, bid, events, *args, **kwargs)
            os._exit(73)
        if decision:
            self.attempts.append((bid, copy.deepcopy(events)))
            if self.mode == "before_crash":
                os._exit(73)
            if self.mode in {"after_crash", "before", "unknown"}:
                if self.mode != "before":
                    result = await self.store.append_batch(sid, bid, events, *args, **kwargs)
                    if self.mode == "after_crash":
                        os._exit(73)
                if len(self.attempts) <= 2:
                    raise OSError("decision commit unavailable" if self.mode == "before" else "decision receipt lost")
                if self.mode == "unknown":
                    return result
            if self.mode == "concurrent":
                self.entered.set()
                await self.release.wait()
        return await self.store.append_batch(sid, bid, events, *args, **kwargs)


async def main(path, mode, stage):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            provider = DecisionStore(store, mode if stage in {"hot", "source", "recrash"} else "normal")
            rt, a, b, tool = runtime(provider, memory, "approval",
                                      "initial" if stage in {"hot", "source"} else "continued", "hot")
            await initialize_all(rt)
            if stage in {"hot", "source"}:
                data = await prepare(rt, a, store, memory, "approval", "hot")
                save(path, data)
            else:
                data = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
                await rt.rebuild_session(SID, strict=True)

            def reply(message=REFUSAL):
                return rt.reply_to_hitl(HitlReply(hitl_id=data["approval"], agent_id=data["root"],
                                                outcome="rejected", message=message))

            if stage == "source":
                await reply()
                raise AssertionError("producer did not reach the real commit crash")
            if stage == "hot" and mode in {"before", "unknown", "before_recover", "unknown_recover"}:
                initial_actor_calls = len([r for k, r in a.calls if k == "actor"])
                for message in [REFUSAL, "RETRY_ALTERNATE must not replace the first reply"]:
                    try:
                        await reply(message)
                    except PersistenceUnavailableError:
                        pass
                    else:
                        raise AssertionError("storage failure was swallowed")
                    req = rt.hitl_registry.get(data["approval"])
                    assert not req.resolved and req.slot is not None
                    assert len([r for k, r in a.calls if k == "actor"]) == initial_actor_calls
                    assert tool.invocations == 0
                    rows = await store.read_range(SID)
                    assert not any(r.event.type == "HitlClosed" for r in rows)
                    rt.clear_storage_isolation(SID)
                if mode.endswith("_recover"):
                    await recover_fixture_session(rt, data["root"], session_id=SID)
                else:
                    answer = await reply("RETRY_ALTERNATE still must not replace it")
                    assert answer.outcome == "rejected"
                assert len(provider.attempts) >= 3
                assert all(batch == provider.attempts[0] for batch in provider.attempts)
            elif stage == "hot" and mode == "concurrent":
                first = asyncio.create_task(reply())
                await asyncio.wait_for(provider.entered.wait(), 5)
                second = asyncio.create_task(reply("RETRY_ALTERNATE duplicate"))
                # Public commands contend for the same original resolution.
                await asyncio.sleep(0)
                provider.release.set()
                answers = await asyncio.gather(first, second)
                assert sum(answer is not None for answer in answers) == 1
                assert len(provider.attempts) == 1
            elif stage in {"continue", "recrash"}:
                pending = await rt.list_pending_hitl(session_id=SID)
                if any(q.data.id == data["approval"] for q in pending):
                    await reply()
                else:
                    await recover_fixture_session(rt, data["root"], session_id=SID)
            assert await rt.recover_storage(SID)
            data = await complete(rt, a, b, tool, store, memory, data, "approval", restarted=stage != "hot", answered=True)
            assert not any("RETRY_ALTERNATE" in request_text(r) for k, r in a.calls if k == "actor")
            save(path, data)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), *sys.argv[2:]))
