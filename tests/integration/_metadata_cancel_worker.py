"""Cancel through public APIs while independent model producers are still active."""

import asyncio
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols import MemoryAddress, MemoryScope, ProviderContext, ToolCall, load_records
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration._session_cancel_worker import runtime, wait_until
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    STALL_BEFORE_CHUNK,
    _ask,
    _next_question,
    _ScriptedLLM,
)

SID = "metadata-cancel"
REPLY = "RETRACT THIS REPLY"


async def content(memory, task_id, agent_id):
    result = []
    for scope in (MemoryScope.TASK, MemoryScope.AGENT):
        records = await load_records(memory, MemoryAddress(session_id=SID,
            task_id=task_id if scope == MemoryScope.TASK else None, agent_id=agent_id),
            scope, ProviderContext(session_id=SID, tenant_id="default"))
        result.extend((str(scope), r.role, str(r.content).replace(task_id, "<task>")) for r in records)
    return sorted(result)


async def worker(path, mode, cut):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            report_mode = mode.startswith("report_")
            cold_reply = report_mode or mode == "detached"
            answer = "ORIGINAL ANSWER " + ("long text " * 2500 if mode == "report_long" else "short")
            llm = _ScriptedLLM([MockResponse(text=answer) if cold_reply else _ask("question", "Which file?"),
                                STALL_BEFORE_CHUNK])
            original_complete = llm.complete
            metadata_started, metadata_closed, release_metadata = (asyncio.Event() for _ in range(3))
            metadata_finished, returned, allow_return = (asyncio.Event() for _ in range(3))
            report_started, report_closed, release_report = (asyncio.Event() for _ in range(3))
            actor_closed = asyncio.Event()
            actors = 0

            def complete(request, stream=True):
                nonlocal actors
                names = {t.name for t in request.tools}
                if "control__update_task_metadata" in names and not report_mode:
                    async def metadata():
                        metadata_started.set()
                        source = llm._stream(MockResponse(tool_calls=[ToolCall(id="metadata", name="control__update_task_metadata",
                            arguments={"title": "LATE TITLE", "description": "late metadata"})]), request)
                        try:
                            await release_metadata.wait()
                            async for chunk in source:
                                yield chunk
                        finally:
                            await source.aclose()
                            metadata_closed.set()
                            if cut == "metadata_closed":
                                os._exit(73)
                    return metadata()
                source = original_complete(request, stream)
                if "control__report_observation" in names and report_mode:
                    async def report():
                        report_started.set()
                        try:
                            await release_report.wait()
                            async for chunk in source:
                                yield chunk
                        finally:
                            await source.aclose()
                            report_closed.set()
                    return report()
                if names.intersection({"control__update_task_metadata", "control__report_observation"}):
                    return source
                actors += 1
                if actors != 2:
                    return source
                async def actor():
                    try:
                        async for chunk in source:
                            yield chunk
                    finally:
                        await source.aclose()
                        actor_closed.set()
                return actor()
            llm.complete = complete
            rt = runtime(store, memory, llm)
            await initialize_all(rt)

            async def notice(event):
                if event.type == "RunFinished" and event.origin == "loop.recognize_intent":
                    metadata_finished.set()
                if (mode == "return" and event.type == "LLMResponseFinished"
                        and event.origin == "loop.recognize_intent" and event.payload.get("finish_reason") != "cancelled"):
                    returned.set()
                    await allow_return.wait()
            rt.event_bus.subscribe(None, notice)
            h = await rt.start_session(SessionStartParams.create(session_id=SID, template_id="agent:tpl_echo",
                user_prompt="help with this file"))
            tm = rt._task_managers[SID]
            if cold_reply:
                await wait_until(lambda: any(p.form == "wait" for p in rt.hitl_registry.list_pending(session_id=SID)))
                question = next(p for p in rt.hitl_registry.list_pending(session_id=SID) if p.form == "wait")
                await wait_until(lambda: not rt._run_tokens.get(SID))
                if report_mode:
                    await report_started.wait()
                else:
                    await metadata_started.wait()
            else:
                question = await _next_question(rt, SID)
                await wait_until(lambda: rt.hitl_registry.get(question.id).slot is not None)
                await metadata_started.wait()
            await rt.reply_to_hitl(HitlReply(hitl_id=question.id, outcome="accepted", agent_id=h.agent_id, message=REPLY))
            await asyncio.wait_for(llm.stalled.wait(), 10)
            dispatch = tm._dispatch_tasks[h.task_id]
            cancel_token = rt._run_tokens[SID][h.task_id].cancel
            if mode == "return":
                release_metadata.set()
                await asyncio.wait_for(returned.wait(), 10)
            original_append = store.append_batch
            receipt_lost = False
            receipt_signatures = []
            async def append(sid, bid, events, *args, **kwargs):
                nonlocal receipt_lost
                receipt = await original_append(sid, bid, events, *args, **kwargs)
                kind, origin = {
                    "actor_finished": ("RunFinished", "runtime"),
                    "actor_receipt": ("RunFinished", "runtime"),
                    "actor_hot_receipt": ("RunFinished", "runtime"),
                    "metadata_receipt": ("RunFinished", "loop.recognize_intent"),
                    "metadata_hot_receipt": ("RunFinished", "loop.recognize_intent"),
                    "response_hot_receipt": ("LLMResponseFinished", "loop.recognize_intent"),
                    "response_receipt": ("LLMResponseFinished", "loop.recognize_intent"),
                    "task_canceled": ("TaskCanceled", None),
                    "agent_terminated": ("AgentTerminated", None),
                    "observe_completed": ("ObserveCompleted", None),
                }.get(cut, (None, None))
                if any(e.type == kind and (origin is None or e.origin == origin) for e in events):
                    if cut.endswith("_hot_receipt"):
                        from copy import deepcopy
                        receipt_signatures.append(deepcopy((bid, [(e.id, e.payload) for e in events])))
                        assert receipt_signatures[-1] == receipt_signatures[0]
                        if not receipt_lost:
                            receipt_lost = True
                            raise OSError("committed cancellation finish, receipt lost")
                        return receipt
                    # Receipt loss followed by process death: the caller never sees
                    # confirmation, although the transaction is actually durable.
                    os._exit(73)
                return receipt
            store.append_batch = append
            cancellation = asyncio.create_task(rt.cancel_session(SID))
            await wait_until(lambda: cancel_token.is_cancelled)
            allow_return.set()
            try:
                await asyncio.wait_for(cancellation, 10)
            except PersistenceUnavailableError:
                assert cut.endswith("_hot_receipt")
            if cut.endswith("_hot_receipt"):
                await wait_until(lambda: receipt_lost)
                await wait_until(lambda: not tm._dispatch_is_live(h.task_id))
                rt.clear_storage_isolation(SID)
                await rt.cancel_session(SID)
                assert len(receipt_signatures) >= 2
            await wait_until(lambda: dispatch.done() and not tm._dispatch_tasks and not tm.running_task_ids())
            assert actor_closed.is_set()
            assert not rt._run_tokens.get(SID) and not tm.open_round_task_ids
            assert rt._agent_lifecycle_manager.record_of(h.agent_id).status == "terminated"
            if not report_mode:
                await asyncio.wait_for(metadata_closed.wait(), 10)
                await asyncio.wait_for(metadata_finished.wait(), 10)
                assert not release_metadata.is_set() or mode == "return"
                assert tm.get_task(h.task_id).title != "LATE TITLE"
                rows = await store.read_range(SID)
                starts = [r.event for r in rows if r.event.type == "LLMRequestStarted" and r.event.origin == "loop.recognize_intent"]
                ends = [r.event for r in rows if r.event.type == "LLMResponseFinished" and r.event.origin == "loop.recognize_intent"]
                assert len(starts) == len(ends) == 1
                assert starts[0].payload["request_id"] == ends[0].payload["request_id"]
            if report_mode:
                assert not report_closed.is_set(), "cancellation stopped the independent observation"
                before = await content(memory, h.task_id, h.agent_id)
                release_report.set()
                await wait_until(lambda: not tm._background_asyncio_tasks)
                assert await content(memory, h.task_id, h.agent_id) == before
                assert answer in str(before)
                assert REPLY not in str(before)
            await wait_until(lambda: not tm._background_asyncio_tasks)
            view = await rebuild_view(store, SID)
            assert not view.open_runs and not view.pending_observe and not view.hitl.pending


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]))
