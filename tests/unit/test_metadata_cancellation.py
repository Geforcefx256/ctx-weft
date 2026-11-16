"""Metadata stop handles follow actual executions, not the latest-run index."""
import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.core.control.tokens import CancelToken
from ctx_weft.core.loop.driver import ExecutionRegistry, LoopState, StepDriver, StepOutcome
from ctx_weft.core.loop.llm_stream import stream_until_cancel
from ctx_weft.core.loop.steps.recognize_intent import RecognizeIntentStep
from ctx_weft.protocols import LLMChunk



@pytest.mark.asyncio
async def test_metadata_cancel_before_execute_emits_no_request_completion():
    token = CancelToken()
    token.cancel()
    emitted = []
    async def emit(event):
        emitted.append(event)
    with pytest.raises(asyncio.CancelledError):
        await RecognizeIntentStep().execute(SimpleNamespace(),
            SimpleNamespace(cancel_token=token, event_bus=SimpleNamespace(emit=emit)))
    assert emitted == []


@pytest.mark.asyncio
async def test_cancel_wins_over_a_ready_chunk_and_closes_the_source():
    token = CancelToken()
    closed = asyncio.Event()
    async def response():
        try:
            token.cancel()
            yield LLMChunk(kind="token", text="late")
        finally:
            closed.set()
    with pytest.raises(asyncio.CancelledError):
        async for _ in stream_until_cancel(response(), token):
            pytest.fail("a canceled result reached the consumer")
    assert closed.is_set()


@pytest.mark.asyncio
async def test_stop_metadata_signals_all_live_runs_without_canceling_observe():
    registry = ExecutionRegistry()
    tokens = [CancelToken(), CancelToken()]
    tasks = [asyncio.create_task(token.wait()) for token in tokens]
    report_release = asyncio.Event()
    report = asyncio.create_task(report_release.wait())
    for index, (token, task) in enumerate(zip(tokens, tasks)):
        registry.register(f"metadata-{index}", "same-task", "recognize_intent", task,
                          agent_id="agent", stop_metadata=token.cancel)
    registry.register("report", "same-task", "observe", report, agent_id="agent")
    registry.stop_metadata()
    await asyncio.gather(*tasks)
    assert not report.done()
    assert all(token.is_cancelled for token in tokens)
    report_release.set()
    await report
    await asyncio.sleep(0)
    assert not registry.tasks and not registry.metadata_stops


@pytest.mark.asyncio
async def test_metadata_stop_does_not_signal_the_original_observe_token():
    original = CancelToken()
    report_release = asyncio.Event()
    metadata_ready, report_ready = asyncio.Event(), asyncio.Event()
    seen = {}
    class Step:
        def __init__(self, name):
            self.name = name
        async def execute(self, state, ctx):
            seen[self.name] = ctx.cancel_token
            if self.name == "recognize_intent":
                metadata_ready.set()
                await ctx.cancel_token.wait()
            else:
                report_ready.set()
                await report_release.wait()
            return StepOutcome(None)
    async def emit(event):
        pass
    state = LoopState(run_id="actor", session=SimpleNamespace(id="session", tenant_id="default"),
        task=SimpleNamespace(id="task", status="ACTIVE"), agent=SimpleNamespace(id="agent"),
        scope=SimpleNamespace())
    ctx = SimpleNamespace(cancel_token=original, event_bus=SimpleNamespace(emit=emit), task_manager=None)
    driver = StepDriver(steps={name: Step(name) for name in ("recognize_intent", "observe")})
    metadata = driver.start("recognize_intent", state, ctx)
    report = driver.start("observe", state, ctx)
    await metadata_ready.wait()
    await report_ready.wait()
    driver.executions.stop_metadata()
    await metadata
    assert seen["recognize_intent"] is not original
    assert seen["observe"] is None and not original.is_cancelled
    assert not report.done()
    report_release.set()
    await report
