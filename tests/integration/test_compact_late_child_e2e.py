"""Exercise late child closure through real dispatch, observe and compact steps.

Only the test LLM controls response timing. No step, scheduler, memory write,
budget estimate or organize implementation is patched.
"""

import asyncio
from contextlib import asynccontextmanager

import pytest

from ctx_weft import SessionStartParams
from ctx_weft.protocols import (
    CapabilityEvent,
    CapabilityRef,
    MemoryAddress,
    MemoryScope,
    ProviderContext,
    ToolCall,
    load_records,
)
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests._initialization import initialize_all
from tests.integration.test_compact_flow_e2e import _NoopToolProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

SID = "compact-late-child"
CHILD_ANSWER = "CHILD PARKED ANSWER"
CURRENT_PROMPT = "INDEPENDENT CURRENT WORK"
WARMUP_PROMPT = "ALREADY COMPLETED OLD WORK"


class EvidenceTool(_NoopToolProvider):
    # The real tool result, rather than an injected usage/budget estimate, grows
    # the next actor prompt past the default 90% context recovery threshold.
    def __init__(self):
        self.calls = 0

    async def _handle(self):
        self.calls += 1
        content = {1: "EARLY EVIDENCE " * 200, 2: "TOOL EVIDENCE " * 7600}.get(self.calls, "ok")
        yield CapabilityEvent(kind="result", payload={"content": content, "metadata": {}})


class ScheduledLLM(MockLLMAdapter):
    def __init__(self):
        super().__init__(responses=[], context_limit=20_000, output_reserve=0)
        self.child_review_started = asyncio.Event()
        self.release_child_review = asyncio.Event()
        self.current_summary_started = asyncio.Event()
        self.release_current_summary = asyncio.Event()
        self.current_calls = 0
        self.trace = []

    def complete(self, request, stream=True):
        names = {tool.name for tool in request.tools}
        text = "\n".join(str(message.content) for message in request.messages)
        if "control__update_task_metadata" in names:
            response = MockResponse(
                tool_calls=[
                    ToolCall(
                        id="metadata",
                        name="control__update_task_metadata",
                        arguments={"title": "Work", "description": "Work", "session_goal": "Work"},
                    )
                ]
            )
        elif not names:
            if "agent's completed-task history" in text or "为这个 agent 已完成的任务历史" in text:
                self.trace.append("l2_history_summary")
                return self._stream(MockResponse(text="Completed history digest"), request)
            self.trace.append("l3_summary_started")
            self.current_summary_started.set()

            async def summary():
                await self.release_current_summary.wait()
                self.trace.append("l3_summary_returned")
                async for chunk in self._stream(
                    MockResponse(text="Current progress digest"), request
                ):
                    yield chunk

            return summary()
        elif "control__report_observation" in names:
            response = MockResponse(
                tool_calls=[
                    ToolCall(
                        id="review",
                        name="control__report_observation",
                        arguments={
                            "task_status": "success",
                            "act_recap": "Completed work",
                            "task_summary": "Completed work",
                        },
                    )
                ]
            )
            if (
                CHILD_ANSWER in text
                and CURRENT_PROMPT not in text
                and not self.child_review_started.is_set()
            ):
                self.trace.append("child_review_started")
                self.child_review_started.set()

                async def child_review():
                    await self.release_child_review.wait()
                    self.trace.append("child_review_returned")
                    async for chunk in self._stream(response, request):
                        yield chunk

                return child_review()
        elif CURRENT_PROMPT in text:
            self.current_calls += 1
            self.trace.append(f"work_response_{self.current_calls}")
            response = (
                MockResponse(
                    text="Continuing work",
                    tool_calls=[
                        ToolCall(id=f"work_{self.current_calls}", name="probe__noop", arguments={})
                    ],
                )
                if self.current_calls <= 3
                else MockResponse(text="Current final answer")
            )
        elif WARMUP_PROMPT in text:
            self.trace.append("old_work_actor")
            response = MockResponse(text="Previous completed answer")
        elif CHILD_ANSWER in text:
            self.trace.append("parent_resumed")
            response = MockResponse(text="Parent final answer")
        elif "do the delegated work" in text:
            self.trace.append("child_actor")
            response = MockResponse(text=CHILD_ANSWER)
        else:
            self.trace.append("parent_delegate")
            response = MockResponse(
                tool_calls=[
                    ToolCall(
                        id="delegate",
                        name="control__delegate_task",
                        arguments={"title": "Child", "task_prompt": "do the delegated work"},
                    )
                ]
            )
        return self._stream(response, request)


@asynccontextmanager
async def memory_backend(backend, tmp_path):
    if backend == "memory":
        yield InMemoryMemoryProvider()
    else:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            yield memory


async def wait_until(predicate):
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("finish_timing", ["before_compact", "while_compact_waits"])
async def test_late_child_closure_via_scheduler(tmp_path, backend, finish_timing):
    """Compact joins the real child's Observe before selecting old history."""
    async with memory_backend(backend, tmp_path) as memory:
        template = make_echo_template()
        template.capability_refs = [CapabilityRef(capability_id="probe:noop", mode="required")]
        # Keep every LoopConfig default, including collapse_keep_last=3.
        provider = InlineAgentTemplateProvider()
        provider.register(template)
        llm = ScheduledLLM()
        rt = make_runtime(llm=llm, agent_provider=provider)
        rt.providers.register_memory(memory)
        rt.providers.register_capability(EvidenceTool())
        await initialize_all(rt)
        events = []
        compact_done = asyncio.Event()
        compact_waiting = asyncio.Event()
        child_applied = asyncio.Event()
        current_id = None
        snapshot = {}

        async def collect(event):
            events.append(event)
            if (
                event.type == "MemoryCompactStarted"
                and event.task_id == current_id
                and event.payload.get("phase") == "waiting_observe"
            ):
                llm.trace.append("compact_waiting_observe")
                compact_waiting.set()
            if event.type == "ObserveApplied" and event.task_id == snapshot.get("child_id"):
                snapshot["child_observe_run"] = event.run_id
                llm.trace.append("child_observe_applied")
                child_applied.set()
            if event.type == "RunFinished" and event.run_id == snapshot.get("child_observe_run"):
                llm.trace.append("child_observe_finished")
            if event.type == "MemoryCompactFinished" and event.task_id == current_id:
                pctx = ProviderContext(
                    session_id=SID,
                    tenant_id="default",
                    agent_id=handle.agent_id,
                    task_id=current_id,
                )
                scope = MemoryAddress(session_id=SID, agent_id=handle.agent_id, task_id=current_id)
                snapshot["current_records"] = await load_records(
                    memory, scope, MemoryScope.TASK, pctx
                )
                snapshot["remaining_history"] = await memory.recall_for_organize(
                    pctx, operation="fold_history", keep_last=0
                )
                child_scope = MemoryAddress(
                    session_id=SID, agent_id=handle.agent_id, task_id=snapshot["child_id"]
                )
                snapshot["child_records"] = await load_records(
                    memory, child_scope, MemoryScope.TASK, pctx
                )
                snapshot["compact"] = event.payload
                llm.trace.append("compact_finished")
                compact_done.set()

        subscription = rt.event_bus.subscribe(None, collect)
        try:
            handle = await rt.start_session(
                SessionStartParams.create(
                    session_id=SID, template_id="agent:tpl_echo", user_prompt="delegate then finish"
                )
            )
            await asyncio.wait_for(llm.child_review_started.wait(), 15)
            tm = rt._task_managers[SID]
            (child,) = [task for task in tm._tasks.values() if task.parent_task_id]
            snapshot["child_id"] = child.id
            assert child.assigned_agent_id == handle.agent_id
            await wait_until(lambda: child.status == "AWAITING_HUMAN" and not tm._dispatch_tasks)
            assert tm.get_task(handle.task_id).status == "SUSPENDED"
            pctx = ProviderContext(
                session_id=SID, tenant_id="default", agent_id=handle.agent_id, task_id=child.id
            )
            scope = MemoryAddress(session_id=SID, agent_id=handle.agent_id, task_id=child.id)
            child_raw = await load_records(memory, scope, MemoryScope.TASK, pctx)
            snapshot["child_raw_ids"] = {r.id for r in child_raw if CHILD_ANSWER in str(r.content)}
            assert snapshot["child_raw_ids"]

            if finish_timing == "before_compact":
                llm.release_child_review.set()
                await asyncio.wait_for(child_applied.wait(), 15)
                await wait_until(
                    lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks
                )

            # Supply real completed history, so this compact actually processes
            # L1/L2 after the child transitions from parked to completed.
            warmup = await rt.dispatch_task(
                SID, WARMUP_PROMPT, agent_id=handle.agent_id, unattended=True
            )
            await wait_until(
                lambda: tm.get_task(warmup.task_id).status == "FINISHED" and not tm._dispatch_tasks
            )
            assert await memory.recall_for_organize(pctx, operation="fold_history", keep_last=0)

            current = await rt.dispatch_task(
                SID, CURRENT_PROMPT, agent_id=handle.agent_id, unattended=True
            )
            current_id = current.task_id
            if finish_timing == "while_compact_waits":
                await asyncio.wait_for(compact_waiting.wait(), 15)
                assert child.status == "AWAITING_HUMAN"
                assert tm.get_task(current_id).status == "ACTIVE"
                assert not llm.current_summary_started.is_set()
                assert "l2_history_summary" not in llm.trace
                assert any(
                    e.type == "StepStarted"
                    and e.task_id == current_id
                    and e.payload["step_name"] == "compact"
                    for e in events
                )
                llm.release_child_review.set()
                await asyncio.wait_for(child_applied.wait(), 15)
            await asyncio.wait_for(llm.current_summary_started.wait(), 15)
            assert child.status == "FINISHED"
            assert current.agent_id == child.assigned_agent_id
            current_ctx = ProviderContext(
                session_id=SID, tenant_id="default", agent_id=current.agent_id, task_id=current_id
            )
            history_at_l3 = await memory.recall_for_organize(
                current_ctx, operation="fold_history", keep_last=0
            )
            assert not any(r.address.task_id == child.id for r in history_at_l3)
            llm.release_current_summary.set()
            await asyncio.wait_for(compact_done.wait(), 15)
            await wait_until(lambda: not tm._dispatch_tasks and not tm._background_asyncio_tasks)

            child_after = snapshot["child_records"]
            raw_retained = bool(snapshot["child_raw_ids"] & {r.id for r in child_after})
            old_child_available = any(
                r.address.task_id == child.id for r in snapshot["remaining_history"]
            )
            current_folded = any(r.metadata.get("collapsed") for r in snapshot["current_records"])
            print(
                {
                    "backend": backend,
                    "timing": finish_timing,
                    "trace": llm.trace,
                    "compact": snapshot["compact"],
                    "child_raw_retained": raw_retained,
                    "old_child_available": old_child_available,
                    "current_folded": current_folded,
                    "child_rows": [(r.id, r.role, str(r.content)[:120]) for r in child_after],
                }
            )
            assert current_folded
            assert not raw_retained
            assert not old_child_available
            assert tm.get_task(current_id).status == "FINISHED"
            assert tm.get_task(handle.task_id).status == "FINISHED"
            assert snapshot["compact"]["est_before"] >= 20_000 * 0.8
            assert snapshot["compact"]["target_tokens"] == 8000
            assert snapshot["compact"]["freed_tokens"] > 0
            assert snapshot["compact"]["levels"] == ["demote_lean", "root_experience", "collapse"]
            assert llm.trace.index("l2_history_summary") < llm.trace.index("l3_summary_started")
            assert llm.trace.index("child_observe_applied") < llm.trace.index("l2_history_summary")
            assert llm.trace.index("child_observe_applied") < llm.trace.index("child_observe_finished")
            assert llm.trace.index("child_observe_finished") < llm.trace.index("l2_history_summary")
            if finish_timing == "while_compact_waits":
                assert llm.trace.index("compact_waiting_observe") < llm.trace.index(
                    "child_observe_applied"
                )
            assert any(
                e.type == "MemoryCompactStarted"
                and e.task_id == current_id
                and e.payload["trigger"] == "prepare_budget"
                for e in events
            )
        finally:
            llm.release_child_review.set()
            llm.release_current_summary.set()
            await subscription.unsubscribe()
