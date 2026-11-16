"""Public session cancellation at real stream, commit and retraction boundaries."""

import asyncio
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests._initialization import initialize_all

from ctx_weft import SessionStartParams
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.protocols import LLMChunk, MemoryAddress, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.events import PersistenceUnavailableError
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._event_helpers import all_events
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    _ask,
    _next_question,
    _ScriptedLLM,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM

SID = "user-cancel"
RETRACTED = "this message should be retracted"


def runtime(store, memory, llm, *, recovering=False):
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=provider, event_store=store,
                      config=RuntimeConfig(task_max_concurrent=0 if recovering else 4,
                                           hitl_timeout_sec=None))
    rt.providers.register_memory(memory)
    return rt


async def wait_until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(.01)


async def worker(path, mode, cut):
    async with open_sqlite_event_store(path / "events.db") as store:
        async with open_sqlite_memory(path / "memory.db") as memory:
            if mode == "recover":
                original_ingest = memory.ingest
                async def ingest(event, ctx):
                    receipt = await original_ingest(event, ctx)
                    if (event.id or "").startswith("uprompt:") and RETRACTED in str(event.content):
                        os._exit(73)
                    return receipt
                memory.ingest = ingest
                rt = runtime(store, memory, _FinishEveryTaskLLM(), recovering=True)
                await initialize_all(rt)
                await rt.rebuild_session(SID, strict=True)
                raise AssertionError("recovery did not repair the missing committed input")
            llm = _ScriptedLLM([_ask("q", "What should I do?")]) if mode == "hot" else _FinishEveryTaskLLM()
            metadata_release, metadata_finished = asyncio.Event(), asyncio.Event()
            if mode == "hot":
                initial_complete = llm.complete
                def hold_metadata(request, stream=True):
                    source = initial_complete(request, stream)
                    if "control__update_task_metadata" not in {t.name for t in request.tools}:
                        return source
                    async def response():
                        try:
                            await metadata_release.wait()
                            async for chunk in source:
                                yield chunk
                        finally:
                            await source.aclose()
                    return response()
                llm.complete = hold_metadata
            rt = runtime(store, memory, llm)
            await initialize_all(rt)
            async def background_finished(event):
                if event.type == "RunFinished" and event.origin == "loop.recognize_intent":
                    metadata_finished.set()
            rt.event_bus.subscribe(None, background_finished, provisional=True)
            async def crash_on_discard(event):
                if cut == "round_discarded" and event.type == "RoundDiscarded":
                    os._exit(73)
            rt.event_bus.subscribe(None, crash_on_discard)
            h = await rt.start_session(SessionStartParams.create(session_id=SID,
                template_id="agent:tpl_echo", user_prompt="FIRST",
                unattended=mode != "hot"))
            tm = rt._task_managers[SID]
            question = None
            if mode == "hot":
                question = await _next_question(rt, SID)
                await wait_until(lambda: rt.hitl_registry.get(question.id).slot is not None)
            else:
                await h.wait_for_finish(timeout=15)
                await wait_until(lambda: not tm.running_task_ids() and not tm._background_asyncio_tasks)
            before = await all_events(store, SID)
            started, closed, gate, commit_entered = (asyncio.Event() for _ in range(4))
            receipt_gate = asyncio.Event()
            original_complete = llm.complete
            original_append = store.append_batch
            target = h.task_id if mode == "hot" else None
            calls = []
            frozen = None
            def complete(request, stream=True):
                if {t.name for t in request.tools}.intersection(
                        {"control__update_task_metadata", "control__report_observation"}):
                    return original_complete(request, stream)
                async def response():
                    started.set()
                    try:
                        if mode in ("arrival", "unknown", "after"):
                            await gate.wait()
                            yield LLMChunk(kind="token", text="partial response")
                        await asyncio.Event().wait()
                        yield LLMChunk(kind="token", text="unreachable")
                    finally:
                        if mode == "hot":
                            metadata_release.set()
                            await asyncio.wait_for(metadata_finished.wait(), 10)
                        closed.set()
                        if cut == "stream_closed":
                            os._exit(73)
                return response()
            llm.complete = complete

            async def append(sid, bid, events, *args, **kwargs):
                nonlocal frozen
                relevant = target is not None and any(e.type == "TaskCreated" and e.task_id == target for e in events)
                if relevant and mode == "unknown":
                    signature = (bid, [(e.id, e.type, e.payload) for e in events])
                    if frozen is None:
                        from copy import deepcopy
                        frozen = deepcopy(signature)
                    assert signature == frozen, "uncertain batch changed during cancellation"
                    calls.append(signature)
                    receipt = await original_append(sid, bid, events, *args, **kwargs)
                    if len(calls) == 1:
                        commit_entered.set()
                        await receipt_gate.wait()
                        if cut == "unknown_committed":
                            os._exit(73)
                        raise OSError("receipt lost after provisional commit")
                    return receipt
                receipt = await original_append(sid, bid, events, *args, **kwargs)
                if cut == "agent_terminated" and any(e.type == "AgentTerminated" for e in events):
                    os._exit(73)
                return receipt
            store.append_batch = append
            if question is not None:
                await rt.reply_to_hitl(HitlReply(hitl_id=question.id, outcome="accepted",
                    agent_id=question.agent_id, message=RETRACTED))
            else:
                turn = await rt.send_message(h.agent_id, RETRACTED, unattended=True)
                target = turn.task_id
            await asyncio.wait_for(started.wait(), 10)
            dispatch = tm._dispatch_tasks[target]
            token = rt._run_tokens[SID][target].cancel
            original_discard = tm.discard_round
            async def discard(tid):
                if tid == target:
                    assert token.is_cancelled, "round removed before stop signal"
                    assert closed.is_set(), "round removed before stream closed"
                return await original_discard(tid)
            tm.discard_round = discard
            if mode in ("after", "unknown"):
                gate.set()
                if mode == "unknown":
                    # append pauses after commit, while cancel delivers its signal.
                    await asyncio.wait_for(commit_entered.wait(), 10)
                else:
                    await wait_until(lambda: not tm.open_round_task_ids)
            cancellation = asyncio.create_task(rt.cancel_session(SID))
            await wait_until(lambda: token.is_cancelled)
            if mode in ("arrival", "unknown"):
                gate.set()
                receipt_gate.set()
            try:
                await asyncio.wait_for(cancellation, 15)
            except PersistenceUnavailableError:
                assert mode == "unknown"
            for _ in range(3):
                rt.clear_storage_isolation(SID)
                await rt.cancel_session(SID)
            await wait_until(lambda: dispatch.done() and not tm.running_task_ids()
                             and not tm._background_asyncio_tasks)
            assert closed.is_set()
            assert not rt._run_tokens.get(SID)
            assert not tm._dispatch_tasks
            assert not tm.open_round_task_ids
            assert rt._agent_lifecycle_manager.record_of(h.agent_id).status == "terminated"
            events = await all_events(store, SID)
            raw = await load_records(memory,
                MemoryAddress(session_id=SID, agent_id=h.agent_id, task_id=target),
                MemoryScope.TASK,
                ProviderContext(session_id=SID, tenant_id="default", agent_id=h.agent_id, task_id=target))
            assert any(RETRACTED in str(r.content) for r in raw) == (mode in ("after", "unknown"))
            if mode in ("before", "arrival"):
                assert not any(e.task_id == target for e in events), "retracted run leaked durable events"
                assert tm.get_task(target) is None
                assert [e.id for e in events[:len(before)]] == [e.id for e in before]
            if question is not None:
                resolved = [e for e in events if e.type == "HitlResolved" and e.payload.get("hitl_id") == question.id]
                assert len(resolved) == 1 and resolved[0].payload["outcome"] == "cancelled", [e.payload for e in events if e.type == "HitlResolved"]
            if mode == "unknown":
                assert len(calls) >= 2


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]))
