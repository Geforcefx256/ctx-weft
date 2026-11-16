"""Agent-scoped Observe draining and fresh compact budgets."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from ctx_weft.core.control.tokens import CancelToken, PauseToken
from ctx_weft.core.loop.driver import ExecutionRegistry
from ctx_weft.core.utils.estimate import estimate_assembled_tokens
from ctx_weft.protocols import LLMMessage, LLMTool, MemoryScope, load_records
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs
from tests.unit.test_compact_history_priority import capsule, setup



async def wait_started(bus):
    async with asyncio.timeout(2):
        while not any(e.type == "MemoryCompactStarted" for e in bus.events):
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_registry_waits_for_all_tasks_but_only_this_agents_observes():
    registry = ExecutionRegistry()
    releases = [asyncio.Event() for _ in range(4)]
    tasks = [asyncio.create_task(release.wait()) for release in releases]
    for index, (agent, step) in enumerate(
        [("a", "observe"), ("a", "observe"), ("b", "observe"), ("a", "recognize_intent")]
    ):
        registry.register(str(index), str(index), step, tasks[index], agent_id=agent)
    try:
        waiter = asyncio.create_task(registry.wait_agent_observes("a"))
        releases[0].set()
        await tasks[0]
        assert not waiter.done()
        releases[1].set()
        assert await asyncio.wait_for(waiter, 2)
        assert not tasks[2].done() and not tasks[3].done()
        assert not await registry.wait_agent_observes("missing")
    finally:
        for release in releases:
            release.set()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_registry_includes_observes_registered_while_waiting():
    registry = ExecutionRegistry()
    release = asyncio.Event()
    later_release = asyncio.Event()
    registered = asyncio.Event()
    later = []

    async def producer():
        await release.wait()
        task = asyncio.create_task(later_release.wait())
        later.append(task)
        registry.register("later", "other-task", "observe", task, agent_id="a")
        registered.set()

    first = asyncio.create_task(producer())
    registry.register("first", "task", "observe", first, agent_id="a")
    waiter = asyncio.create_task(registry.wait_agent_observes("a"))
    try:
        await asyncio.sleep(0)
        release.set()
        await registered.wait()
        await first
        assert not waiter.done()
        later_release.set()
        assert await asyncio.wait_for(waiter, 2)
    finally:
        release.set()
        later_release.set()
        await asyncio.gather(first, *later, waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "force,incoming,grow", [(False, 90_000, False), (True, 90_000, False), (False, 1, True)]
)
async def test_compact_rebuilds_budget_after_waiting(monkeypatch, force, incoming, grow):
    state, ctx, bus = await setup(monkeypatch)
    await capsule(state, ctx, "old", 0)
    release = asyncio.Event()
    observer = asyncio.create_task(release.wait())
    from ctx_weft.core.loop.driver import StepDriver
    from ctx_weft.core.loop.steps.compact import CompactStep

    driver = StepDriver({"compact": CompactStep()})
    driver.executions.register("observe", "old", "observe", observer, agent_id=state.agent.id)
    assembled = []
    prompt = NS(
        system="word " * 45_000 if grow else "x" * 100,
        messages=[LLMMessage(role="user", content="fresh input")],
        tools=[LLMTool(name="work", description="description", input_schema={"type": "object"})],
    )

    async def assemble(request):
        assert observer.done(), "Budget assembly must follow the entire Observe execution"
        assembled.append(request)
        return scripted_inputs(
            prompt if request.purpose == "act" else None,
            system="Summarize",
            purpose=request.purpose,
        )

    async def compile(inputs, **kwargs):
        if inputs.purpose == "act":
            assert kwargs["trim"] is False
        return await compiled_test_inputs(inputs, **kwargs)

    ctx.assembler = NS(assemble=assemble, compile=compile)
    pending = asyncio.create_task(
        driver.execute(
            "compact",
            state,
            ctx,
            options={"token_estimate": incoming, "force": force, "after": "prepare"},
        )
    )
    try:
        await wait_started(bus)
        started = next(e for e in bus.events if e.type == "MemoryCompactStarted")
        assert started.payload["phase"] == "waiting_observe"
        assert started.payload["pending_observe_count"] == 1
        assert not assembled and not pending.done()
        release.set()
        outcome = await asyncio.wait_for(pending, 3)
        finished = next(e for e in outcome.events if e.type == "MemoryCompactFinished")
        expected = estimate_assembled_tokens(prompt, ctx.llm.tokenizer.count)
        assert finished.payload["est_before"] == expected
        assert expected != incoming
        assert outcome.next_step == "prepare"
        records = await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
        if force or grow:
            assert finished.payload["levels"] == ["demote_lean", "root_experience", "collapse"]
            assert any(r.metadata.get("collapsed") for r in records)
        else:
            assert finished.payload["levels"] == []
            assert not any(r.metadata.get("collapsed") for r in records)
            assert not any(request.purpose == "compact" for request in assembled)
    finally:
        release.set()
        await asyncio.gather(observer, pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel_token", "pause_token", "caller_cancel", "observe_error"])
async def test_stopping_or_failing_the_wait_does_not_start_compression(monkeypatch, stop):
    state, ctx, bus = await setup(monkeypatch)
    from ctx_weft.core.loop.driver import StepDriver
    from ctx_weft.core.loop.steps.compact import CompactStep

    driver = StepDriver({"compact": CompactStep()})
    release = asyncio.Event()

    async def observe():
        await release.wait()
        if stop == "observe_error":
            raise RuntimeError("observe write failed")

    observer = asyncio.create_task(observe())
    driver.executions.register("observe", "old", "observe", observer, agent_id=state.agent.id)
    ctx.cancel_token = CancelToken()
    if stop == "pause_token":
        from ctx_weft.core.loop.commit import ROUND_COMMITTED_KEY
        from ctx_weft.core.loop.signals import HitlPark
        from ctx_weft.core.loop.steps import act

        ctx.pause_token = PauseToken()
        ctx.hitl = NS()
        state.extra[ROUND_COMMITTED_KEY] = True

        async def park(*args, **kwargs):
            raise HitlPark(hitl_id="paused-compact")

        monkeypatch.setattr(act, "_park_for_interrupt", park)
    pending = asyncio.create_task(
        driver.execute("compact", state, ctx, options={"token_estimate": 90_000})
    )
    try:
        await wait_started(bus)
        if stop == "cancel_token":
            ctx.cancel_token.cancel()
        elif stop == "pause_token":
            ctx.pause_token.pause()
        elif stop == "caller_cancel":
            pending.cancel()
        else:
            release.set()
        error = (
            RuntimeError
            if stop == "observe_error"
            else HitlPark
            if stop == "pause_token"
            else asyncio.CancelledError
        )
        with pytest.raises(error):
            await asyncio.wait_for(pending, 2)
        if stop != "observe_error":
            assert not observer.done()
        assert not any(e.type in {"MemoryCompacted", "LLMRequestStarted"} for e in bus.events)
    finally:
        release.set()
        await asyncio.gather(observer, pending, return_exceptions=True)
