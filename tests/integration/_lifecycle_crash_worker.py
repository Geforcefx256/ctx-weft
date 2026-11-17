
"""Real runtime worker; terminate only after a selected SQLite commit succeeds."""

import asyncio
import dataclasses
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests._recovery import recover_fixture_session

from tests._initialization import initialize_all
from ctx_weft import SessionStartParams
from ctx_weft.protocols import LLMChunk, MemoryKind, MemoryScope, ToolCall
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.llm.adapters.mock import MockResponse
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM


class LifecycleLLM(_FinishEveryTaskLLM):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.actor_started = asyncio.Event()
        self.actor_count = 0
        self.tool_names = []
        self.judgements = 0

    def complete(self, request, stream=True):
        names = {tool.name for tool in request.tools}
        self.tool_names.append(sorted(names))
        if "control__update_task_metadata" in names:
            return super().complete(request, stream)
        if "control__report_observation" in names:
            self.judgements += 1
            if self.mode == "mechanical":
                async def unavailable():
                    raise RuntimeError("observer unavailable")
                    yield
                return unavailable()
            verdict = ("continue" if self.mode == "review_count" and self.judgements <= 3 else
                       "continue" if self.mode == "background_continue" and self.actor_count == 1 and self.judgements == 1 else
                       "waiting" if self.mode == "background_waiting" else
                       "continue" if self.mode in { "parked_cancel_retry", "parked_interrupt_retry"} or
                       self.mode in {"retry", "mechanical_retry"} and self.judgements == 1 else
                       "fail" if self.mode == "threshold" else "success")
            return self._stream(MockResponse(tool_calls=[ToolCall(
                id=self._id("obs"), name="control__report_observation",
                arguments={"task_status": verdict, "act_recap": "reviewed",
                           "task_summary": "reviewed", "task_failure_reason": "test failure"})]), request)
        self.actor_count += 1
        if self.mode in {"threshold", "interrupt"} and self.actor_count == 1:
            async def hold_root():
                self.actor_started.set()
                while True:
                    yield LLMChunk(kind="token", text="working ")
                    await asyncio.sleep(0.01)
            return hold_root()
        if self.mode.startswith("exhausted_") or self.mode in {"review_count", "plain_text", "background_waiting", "parked_cancel_retry", "parked_interrupt_retry"} or (self.mode in {"mechanical_retry", "background_continue"} and self.actor_count == 1):
            return self._stream(MockResponse(text="completed answer"), request)
        return super().complete(request, stream)


def make_lifecycle_runtime(store, memory, mode):
    provider = InlineAgentTemplateProvider()
    template = make_echo_template()
    if mode in {"mechanical", "mechanical_retry"} or mode.startswith(("handoff_", "exhausted_")):
        template.identity.pop("observe")
    if mode in {"background_waiting", "background_continue", "review_count"}:
        template.loop_config = dataclasses.replace(template.loop_config, short_segment_token_threshold=0)
    if mode.startswith("handoff_"):
        template.loop_config = dataclasses.replace(
            template.loop_config, short_task_token_threshold=100000 if mode.endswith("short") else 0)
    if mode.startswith("exhausted_") or mode.startswith(("parked_cancel", "parked_interrupt")):
        template.loop_config = dataclasses.replace(template.loop_config,
            short_segment_token_threshold=100000 if mode.startswith("exhausted_") and mode.endswith("short") else 0,
            short_task_token_threshold=100000 if mode.endswith("short") else 0)
    provider.register(template)
    llm = LifecycleLLM(mode)
    from ctx_weft.core.models.config import RuntimeConfig
    rt = make_runtime(agent_provider=provider, event_store=store, llm=llm,
                      config=RuntimeConfig(task_max_retries=1 if "budget" in mode else 0)
                      if mode.startswith("exhausted_") or mode in {"retry", "mechanical_retry"} else None)
    rt.providers.register_memory(memory)
    return rt, llm


async def worker(directory, mode, cut, recovering=False):
    async with open_sqlite_event_store(directory / "events.db") as store:
        async with open_sqlite_memory(directory / "memory.db") as memory:
            rt, llm = make_lifecycle_runtime(store, memory, mode)
            if mode.startswith(("handoff_", "exhausted_")):
                from tests.runtime_helpers import install_mechanical_report_fixture
                install_mechanical_report_fixture()
            original = store.append_batch
            root_task_id = None
            observation_completed = False
            recap_ids = set()
            recap_completed = asyncio.Event()
            foreground_finalized = asyncio.Event()
            handoff = mode.startswith(("handoff_", "exhausted_"))
            allow_finish = asyncio.Event()
            producer_settled = asyncio.Event()
            report_waiting = asyncio.Event()
            observe_calls = []
            review_requests = set()
            fourth_review_id = None
            if mode == "mechanical" or handoff:
                from ctx_weft.core.loop.driver import ExecutionMode
                from ctx_weft.core.loop.steps.observe import ObserveStep
                original_execute = ObserveStep.execute
                async def assert_single_background(self, state, ctx):
                    expected_mode = (ExecutionMode.FOREGROUND if mode == "mechanical"
                                     else ExecutionMode.BACKGROUND)
                    assert ctx.execution.mode == expected_mode
                    assert ctx.execution.options["judge_task"] is (mode == "mechanical")
                    observe_calls.append(state.run_id)
                    assert len(observe_calls) <= (2 if "budget" in mode else 1)
                    return await original_execute(self, state, ctx)
                ObserveStep.execute = assert_single_background
            if mode.startswith("handoff_") and "cancel" in mode:
                from ctx_weft.core.loop.driver import StepDriver
                original_foreground = StepDriver._execute_foreground
                async def hold_before_finalize(self, name, state, ctx, *, options=None):
                    result = await original_foreground(self, name, state, ctx, options=options)
                    if name == "observe":
                        await allow_finish.wait()
                    return result
                if not recovering:
                    StepDriver._execute_foreground = hold_before_finalize
                if cut == "cancel_closed":
                    import ctx_weft.core.loop.steps.finalize as finalize_module
                    original_cancel = finalize_module.synthesize_cancel_closure
                    async def crash_cancel(*args, **kwargs):
                        await original_cancel(*args, **kwargs)
                        os._exit(73)
                    finalize_module.synthesize_cancel_closure = crash_cancel
            if "late_subscribe" in mode and not recovering:
                from ctx_weft.core.loop.steps.finalize import FinalizeStep
                from ctx_weft.core.loop.steps.observe import ObserveStep
                class FinishError(OSError):
                    retriable = False
                async def fail_finalize(self, state, ctx):
                    raise FinishError("injected foreground producer failure")
                FinalizeStep.execute = fail_finalize
                original_background = ObserveStep._execute_background
                async def late_subscribe(self, state, ctx, boundary, judging):
                    await producer_settled.wait()
                    return await original_background(self, state, ctx, boundary, judging)
                ObserveStep._execute_background = late_subscribe
                import ctx_weft.core.loop.steps.observe as observe_module
                from ctx_weft.core.orchestrator.task.manager import TaskManager
                original_wait = observe_module.wait_disposition
                async def announce_wait(state, ctx, ready):
                    report_waiting.set()
                    return await original_wait(state, ctx, ready)
                observe_module.wait_disposition = announce_wait
                original_apply = TaskManager.apply_run_outcome
                async def release_before_disposition(self, task_id, outcome):
                    # The foreground driver's final notification already happened.
                    # Let the report subscribe before the task disposition commits.
                    producer_settled.set()
                    await report_waiting.wait()
                    return await original_apply(self, task_id, outcome)
                TaskManager.apply_run_outcome = release_before_disposition

            if handoff and not recovering and "cancel" not in mode and "late_subscribe" not in mode:
                from ctx_weft.core.loop.steps.finalize import FinalizeStep
                from ctx_weft.core.loop.steps.observe import ObserveStep
                if "early" in mode:
                    original_finalize = FinalizeStep.execute
                    async def ordered_finalize(self, state, ctx):
                        await recap_completed.wait()
                        return await original_finalize(self, state, ctx)
                    FinalizeStep.execute = ordered_finalize
                else:
                    original_observe = ObserveStep._llm_observe
                    async def ordered_observe(self, state, ctx, events):
                        await foreground_finalized.wait()
                        return await original_observe(self, state, ctx, events)
                    ObserveStep._llm_observe = ordered_observe
            if mode.startswith(("parked_cancel", "parked_interrupt")) and not recovering:
                original_complete = llm.complete
                def delayed_complete(request, stream=True):
                    response = original_complete(request, stream)
                    if "control__report_observation" not in {tool.name for tool in request.tools}:
                        return response
                    async def held():
                        report_waiting.set()
                        await allow_finish.wait()
                        async for chunk in response:
                            yield chunk
                    return held()
                llm.complete = delayed_complete
            if mode.startswith(("parked_cancel", "parked_interrupt")) and cut == "cancel_closed":
                import ctx_weft.core.loop.steps.finalize as finalize_module
                original_cancel = finalize_module.synthesize_cancel_closure
                async def crash_parked_cancel(*args, **kwargs):
                    await original_cancel(*args, **kwargs)
                    os._exit(73)
                finalize_module.synthesize_cancel_closure = crash_parked_cancel
            if mode == "accepted_interrupt" and not recovering:
                import ctx_weft.core.loop.steps.observe as observe_module
                original_close = observe_module._out_of_band_finalize
                async def hold_accepted_close(*args, **kwargs):
                    report_waiting.set()
                    await allow_finish.wait()
                    return await original_close(*args, **kwargs)
                observe_module._out_of_band_finalize = hold_accepted_close
            original_ingest = memory.ingest
            original_fold = memory.fold
            hitl_commit_failed = False

            async def crash_after_fold(ids, events, ctx):
                result = await original_fold(ids, events, ctx)
                if cut == "report_replaced" and any(e.metadata.get("finish_policy") for e in events):
                    os._exit(73)
                if cut == "raw_folded" and ids and not events:
                    os._exit(73)
                if cut == "segment_fold" and any(
                        event.kind == MemoryKind.SUMMARY for event in events):
                    os._exit(73)
                return result

            memory.fold = crash_after_fold
            original_organize_commit = memory._commit_organize
            async def crash_after_organize_commit(ids, events, ctx):
                result = await original_organize_commit(ids, events, ctx)
                if result is not None and cut == "raw_folded" and ids and not events:
                    os._exit(73)
                if result is not None and cut == "segment_fold" and any(
                        event.kind == MemoryKind.SUMMARY for event in events):
                    os._exit(73)
                return result
            memory._commit_organize = crash_after_organize_commit

            async def crash_after_memory_write(event, ctx):
                result = await original_ingest(event, ctx)
                if observation_completed and (
                        cut == "blackboard_memory" and event.kind == MemoryKind.PUBLICATION or
                        cut == "finish_memory" and event.kind == MemoryKind.CONVERSATION_TURN
                        and event.scope == MemoryScope.AGENT):
                    os._exit(73)
                return result

            memory.ingest = crash_after_memory_write

            async def crash_after_commit(sid, batch_id, events, *args, **kwargs):
                nonlocal observation_completed, hitl_commit_failed, fourth_review_id
                if cut == "before_summary_applied" and any(
                        e.type == EventType.OBSERVE_APPLIED and e.payload.get("observe_id") in recap_ids
                        for e in events):
                    os._exit(73)
                if cut == "hitl_commit_failed" and not hitl_commit_failed and any(
                        event.type == EventType.HITL_RESOLVED for event in events):
                    hitl_commit_failed = True
                    raise OSError("injected HitlResolved SQLite commit failure")
                receipt = await original(sid, batch_id, events, *args, **kwargs)
                for event in events:
                    if mode == "review_count":
                        intent = (event.payload.get("observe_request", {})
                                  if event.type == EventType.HITL_OPENED else
                                  event.payload if event.type == EventType.OBSERVE_REQUESTED else {})
                        if intent.get("judge_task"):
                            review_requests.add(intent["observe_id"])
                            if len(review_requests) == 4 and cut == "fourth_requested":
                                os._exit(73)
                        if event.type == EventType.OBSERVE_STARTED and event.payload.get("observe_index") == 4:
                            fourth_review_id = event.payload["observe_id"]
                            if cut == "fourth_started":
                                os._exit(73)
                        if (event.type == EventType.OBSERVE_COMPLETED
                                and event.payload.get("observe_id") == fourth_review_id
                                and cut == "fourth_completed"):
                            os._exit(73)
                    if event.type == EventType.OBSERVE_REQUESTED and event.payload.get("verdict"):
                        assert event.payload["judge_task"] is False
                    if cut == "agent_terminated" and event.type == EventType.AGENT_TERMINATED and event.payload.get("reason", "").startswith("cancel_agent:"):
                        os._exit(73)
                    if cut == "task_canceled" and event.type == EventType.TASK_CANCELED:
                        os._exit(73)
                    if event.type == EventType.TASK_INTERRUPTED:
                        producer_settled.set()
                    if event.type == EventType.TASK_FINALIZED:
                        foreground_finalized.set()
                    if event.type == EventType.OBSERVE_COMPLETED and not event.payload.get("result", {}).get("verdict"):
                        recap_completed.set()
                    if recovering and event.type == EventType.OBSERVE_STARTED and not event.payload.get("judge_task", True):
                        recap_ids.add(event.payload["observe_id"])
                    if hitl_commit_failed and event.type == EventType.RUN_FINISHED:
                        os._exit(73)
                    if event.type == EventType.OBSERVE_REQUESTED and (event.payload.get("verdict") or not event.payload.get("judge_task", True)):
                        recap_ids.add(event.payload["observe_id"])
                    if cut.startswith("summary_") and event.payload.get("observe_id") in recap_ids:
                        target = {"summary_requested": EventType.OBSERVE_REQUESTED,
                                  "summary_started": EventType.OBSERVE_STARTED,
                                  "summary_completed": EventType.OBSERVE_COMPLETED,
                                  "summary_applied": EventType.OBSERVE_APPLIED}[cut]
                        if event.type == target:
                            os._exit(73)
                    if event.type == EventType.OBSERVE_COMPLETED or (event.type == EventType.OBSERVE_REQUESTED and event.payload.get("verdict")):
                        observation_completed = True
                    matched = (
                        (cut == "observe_requested" and event.type == EventType.OBSERVE_REQUESTED)
                        or (cut == "observe_started" and event.type == EventType.OBSERVE_STARTED)
                        or (cut == "observe_completed" and event.type == EventType.OBSERVE_COMPLETED)
                        or (cut == "observe_applied" and event.type == EventType.OBSERVE_APPLIED)
                        or (cut == "hitl_opened" and event.type == EventType.HITL_OPENED
                         and event.payload.get("observe_request"))
                        or (cut == "background_run_started" and event.type == EventType.RUN_STARTED
                            and event.payload.get("initial_step") == "background_observe")
                        or (cut == "recap_started" and event.type == EventType.OBSERVE_STARTED)
                        or (cut == "recap_done" and event.type == EventType.OBSERVE_APPLIED)
                        or (cut == "observer_response" and event.type == EventType.LLM_RESPONSE_FINISHED
                            and event.origin == "loop.background_observe")
                        or (cut == "task_requeued" and event.type == EventType.TASK_REQUEUED)
                        or (cut == "hitl_resolved" and event.type == EventType.HITL_RESOLVED)
                        or (cut == "task_finished" and event.type == EventType.TASK_FINISHED)
                        or (cut == "task_finalized" and event.type == EventType.TASK_FINALIZED
                            and ("budget" not in mode or event.payload.get("outcome") == "fail"))
                        or (cut == "trip_decision" and event.type == EventType.FAILURE_THRESHOLD_HIT)
                        or (cut == "root_run_finished" and event.type == EventType.RUN_FINISHED
                            and event.task_id == root_task_id and event.origin == "runtime")
                        or (cut == "root_task_failed" and event.type == EventType.TASK_FAILED
                            and event.task_id == root_task_id))
                    if matched:
                        os._exit(73)
                return receipt

            store.append_batch = crash_after_commit
            await initialize_all(rt)
            if recovering:
                if mode == "background_continue":
                    llm.actor_count = llm.judgements = 1
                session = await rt.get_session("crash-session")
                await recover_fixture_session(rt, session.data.root_agent_id, session_id="crash-session")
                tm = rt._task_managers.get("crash-session")
                while tm and tm._background_asyncio_tasks:
                    await asyncio.gather(*list(tm._background_asyncio_tasks))
                return
            handle = await rt.start_session(SessionStartParams.create(
                session_id="crash-session", template_id="agent:tpl_echo",
                user_prompt="perform work",
                unattended=mode.startswith(("handoff_", "exhausted_")) or mode in {"threshold", "interrupt", "foreground", "mechanical", "mechanical_retry", "retry"}))
            root_task_id = handle.task_id
            if mode == "accepted_interrupt":
                await asyncio.wait_for(report_waiting.wait(), 10)
                await rt.cancel_agent(handle.agent_id, reason="after accepted verdict")
                allow_finish.set()
            if mode.startswith(("parked_cancel", "parked_interrupt")):
                await asyncio.wait_for(report_waiting.wait(), 10)
                tm = rt._task_managers[handle.session_id]
                async with asyncio.timeout(10):
                    while tm.get_task(handle.task_id).status != "AWAITING_HUMAN":
                        await asyncio.sleep(0.01)
                await rt.cancel_agent(handle.agent_id, reason="cancel parked task")
                allow_finish.set()
            if mode.startswith("handoff_") and "cancel" in mode:
                await asyncio.wait_for(recap_completed.wait(), 10)
                await rt.cancel_agent(handle.agent_id, reason="cancel during report handoff")
                allow_finish.set()
            if mode in {"threshold", "interrupt"}:
                await asyncio.wait_for(llm.actor_started.wait(), timeout=10)
                async with asyncio.timeout(10):
                    while rt._task_managers[handle.session_id].is_round_open(handle.task_id):
                        await asyncio.sleep(0.01)
                await asyncio.sleep(0.02)
            if mode == "interrupt":
                await rt.pause_session(handle.session_id)
            if mode == "threshold":
                rt._task_managers[handle.session_id].session.failure_threshold = 1
                await rt.dispatch_task(handle.session_id, "fail this independent task",
                    settings={"use_subagent": True, "subagent_template": "agent:tpl_echo",
                              "inherit_from_agent_id": handle.agent_id}, unattended=True)
            try:
                if "late_subscribe" in mode:
                    await asyncio.wait_for(producer_settled.wait(), 10)
                else:
                    await handle.wait_for_finish(timeout=20)
            except PersistenceUnavailableError:
                if cut != "hitl_commit_failed":
                    raise
            # A normal run must drain its background execution before the process exits.
            tm = rt._task_managers[handle.session_id]
            async with asyncio.timeout(10):
                while tm.running_task_ids():
                    await asyncio.sleep(0.01)
            while tm._background_asyncio_tasks:
                await asyncio.gather(*list(tm._background_asyncio_tasks))
            if mode == "mechanical" or handoff:
                assert len(observe_calls) == (2 if "budget" in mode else 1)


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2], sys.argv[3], len(sys.argv) > 4))
