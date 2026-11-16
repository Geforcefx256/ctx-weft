
"""Exercise session cancellation with a full pool and multiple queued tasks per agent."""

import asyncio
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._recovery import recover_fixture_session
from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._cancel_tree_worker import TreeLLM
from tests.integration._session_cancel_worker import wait_until
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

SID = "queued-session-cancel"


def runtime(store, memory, *, recovering=False):
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    llm = TreeLLM()
    llm.release_reports.set()
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store,
                      config=RuntimeConfig(task_max_concurrent=0 if recovering else 1))
    rt.providers.register_memory(memory)
    return rt, llm


async def drain(rt):
    tm = rt._task_managers[SID]
    await wait_until(lambda: not tm.running_task_ids() and not tm._dispatch_tasks
                     and not tm._background_asyncio_tasks)


async def worker(path, fault, recovering=False):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            rt, llm = runtime(store, memory, recovering=recovering)
            await initialize_all(rt)
            if recovering:
                view = await rebuild_view(store, SID)
                target = next(t.id for t in view.tasks.values() if t.user_prompt == "QUEUED ONE")
                owner = view.tasks[target].assigned_agent_id
            else:
                handle = await rt.start_session(SessionStartParams.create(session_id=SID,
                    template_id="agent:tpl_echo", user_prompt="TREE ROOT",
                    unattended=True))
                await asyncio.wait_for(llm.grandchild_started.wait(), 15)
                tm = rt._task_managers[SID]
                await wait_until(lambda: not tm.open_round_task_ids)
                first = await rt.dispatch_task(SID, "QUEUED ONE", agent_id=handle.agent_id,
                    unattended=True, settings=NormalTaskSettings(use_subagent=True,
                                                               subagent_template="agent:tpl_echo"))
                await rt.dispatch_task(SID, "QUEUED TWO", agent_id=first.agent_id, unattended=True)
                target, owner = first.task_id, first.agent_id
                assert target not in tm.running_task_ids()
                assert tm.get_task(target).status == "ACTIVE"

            original = store.append_batch
            attempts = 0
            signature = None
            envelopes = []

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal attempts, signature
                relevant = any(e.type == "TaskCanceled" and e.task_id == target for e in events)
                if relevant and fault in {"failed", "unknown", "unknown_retries", "failed_crash", "unknown_crash"}:
                    attempts += 1
                    envelopes.append((bid, [(e.id, e.payload) for e in events]))
                    if fault.startswith("unknown"):
                        current = (bid, [(e.id, e.payload) for e in events])
                        if signature is None:
                            signature = current
                        assert current == signature, "unconfirmed cancellation envelope changed"
                        result = await original(sid, bid, events, *args, **kwargs)
                        if attempts <= 3:
                            raise OSError("queued cancellation receipt lost")
                        return result
                    if attempts <= 3:
                        raise OSError("queued cancellation not committed")
                result = await original(sid, bid, events, *args, **kwargs)
                for event in events:
                    if fault == "terminated" and event.type == "AgentTerminated" and event.agent_id == owner:
                        os._exit(73)
                    if event.task_id == target and event.type == {
                        "canceled": "TaskCanceled", "finalized": "TaskFinalized",
                    }.get(fault):
                        os._exit(73)
                return result

            store.append_batch = append
            read_task = rt._read_persisted_task
            unavailable_reads = 0
            async def read(session_id, task_id):
                nonlocal unavailable_reads
                if fault == "unknown_retries" and task_id == target and attempts and unavailable_reads < 2:
                    unavailable_reads += 1
                    raise PersistenceUnavailableError("storage still unavailable while confirming lost receipt")
                return await read_task(session_id, task_id)
            rt._read_persisted_task = read
            if recovering:
                await rt.rebuild_session(SID, strict=True)
                await recover_fixture_session(rt, owner, session_id=SID)
                await drain(rt)
                raise AssertionError("recovery missed crash injection")
            async with asyncio.timeout(15):
                while True:
                    try:
                        accepted = await rt.cancel_session(SID)
                    except PersistenceUnavailableError:
                        assert fault in {"failed", "unknown", "unknown_retries", "failed_crash", "unknown_crash"}
                        if fault.endswith("_crash"):
                            os._exit(73)
                        assert rt._task_managers[SID].get_task(target).status == "ACTIVE"
                        assert any(e.task_id == target for e in rt._task_managers[SID]._queue.peek_all())
                        rt.clear_storage_isolation(SID)
                    else:
                        if accepted:
                            break
                    # A stop refused during TaskFinalized publication is not an
                    # accepted cancellation. Let that writer settle before retrying.
                    await rt._task_managers[SID].wait_for_finalized_writes()
                    await asyncio.sleep(0)
            await drain(rt)
            view = await rebuild_view(store, SID)
            assert len(view.tasks) == 5
            assert all(t.status == "CANCELED" and t.finalized for t in view.tasks.values())
            if fault == "failed":
                assert attempts >= 4
            if fault in {"unknown", "unknown_retries"}:
                assert attempts >= 1
                assert unavailable_reads == (2 if fault == "unknown_retries" else 0)
            if envelopes:
                assert all(envelope == envelopes[0] for envelope in envelopes)
                assert sum(r.event.type == "TaskCanceled" and r.event.task_id == target for r in await store.read_range(SID)) == 1
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending
            assert fault in {"none", "failed", "unknown", "unknown_retries"}, "crash point not reached"
            os._exit(0)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2], "recover" in sys.argv[3:]))
