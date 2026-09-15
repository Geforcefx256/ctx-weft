"""Interrupt during the tool loop: completed / interrupted / cancelled three-state."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from ctx_weft.core.control.tokens import CancelToken, PauseToken
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.park import HitlPark
from ctx_weft.core.loop.steps.act import CANCELLED_MARK, INTERRUPTED_MARK, ActStep
from ctx_weft.core.capabilities.cache import CapabilityCache
from tests.hitl_env import make_hitl
from ctx_weft.core.assembler.assembler import AssembledPrompt
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    LLMMessage, MemoryEventType, MemoryAddress, ProviderContext, ToolCall,
)
from ctx_weft.protocols.capability import (
    CapabilityEvent, CapabilityProviderInfo, ToolCapability, ToolCapabilityProvider,
)
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio


class _ToolProvider(ToolCapabilityProvider):
    name = "mcp:t"

    def __init__(self, *, on_invoke=None, block: asyncio.Event | None = None,
                 started: asyncio.Event | None = None) -> None:
        self._on_invoke = on_invoke
        self._block = block
        self._started = started
        self.calls: list[str] = []

    def _caps(self):
        return [
            ToolCapability(id="mcp:t:a", name="a", description=""),
            ToolCapability(id="mcp:t:b", name="b", description=""),
        ]

    async def list(self, ctx): return self._caps()
    async def retrieve(self, ctx): return self._caps()
    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name, capability_count=2)

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            self.calls.append(capability_id)
            if self._on_invoke:
                self._on_invoke(capability_id)
            if self._block is not None:
                if self._started is not None:
                    self._started.set()
                await self._block.wait()
            yield CapabilityEvent(kind="result", payload={"content": "ok"})
        return _run()

    async def cancel(self, invocation_id, ctx): return None


def _harness(llm, provider):
    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    hitl, _hitl_reg = make_hitl(bus)
    cache = CapabilityCache()
    cache.put("ag1", provider._caps())
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[provider], memory=mem, event_bus=bus,
    )
    session = Session(id="s1", tenant_id="default", user_prompt="hi", status="RUNNING", token_budget=0)
    task = Task(id="t1", session_id="s1", status="ACTIVE", title="T",
                interaction_mode="interactive", settings=NormalTaskSettings())
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    prompt = AssembledPrompt(system="", messages=[LLMMessage(role="user", content="hi")],
                             tools=[], token_count=1)
    state = LoopState(run_id="r1", session=session, task=task, agent=agent, scope=scope,
                      assembled_prompt=prompt,
                      resolved_model=SimpleNamespace(model="mock", account=""))
    pause = PauseToken()
    ctx = LoopContext(
        assembler=None, llm=llm, memory=mem, event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1"),
        hitl=hitl, capability_gateway=gw, pause_token=pause,
    )
    return state, ctx, task, mem, pause, provider


async def _tool_results(mem, ctx, scope):
    return await mem.recall_recent(scope, [MemoryEventType.TOOL_RESULT], 20, ctx.provider_ctx)


async def test_interrupt_between_tools_cancels_not_started():
    # tc_a completes (and fires interrupt); tc_b never starts → cancelled.
    provider = _ToolProvider()
    llm = MockLLMAdapter(responses=[MockResponse(tool_calls=[
        ToolCall(id="c_a", name="mcp__t__a", arguments={}),
        ToolCall(id="c_b", name="mcp__t__b", arguments={}),
    ])])
    state, ctx, task, mem, pause, _p = _harness(llm, provider)
    provider._on_invoke = lambda cid: pause.pause() if cid == "mcp:t:a" else None

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    # park 不写状态：AWAITING_HUMAN 由 TaskManager 据 RunOutcome 落（Task 4）
    assert task.status == "ACTIVE"
    assert provider.calls == ["mcp:t:a"]            # tc_b never invoked
    res = await _tool_results(mem, ctx, state.scope)
    # tool_call_id 已是摄入点铸造的内部标识（spec: conversation-integrity），按 tool_name
    # 索引——被断言的是「哪个调用得到什么标记」，与 id 值域无关。
    by_name = {r.metadata.get("tool_name"): r for r in res}
    assert by_name["mcp__t__a"].metadata.get("interrupted") is not True   # completed normally
    assert by_name["mcp__t__b"].metadata.get("cancelled") is True
    assert CANCELLED_MARK in by_name["mcp__t__b"].content


async def test_interrupt_during_tool_marks_interrupted_and_cancels_rest():
    block = asyncio.Event()
    started = asyncio.Event()
    provider = _ToolProvider(block=block, started=started)
    llm = MockLLMAdapter(responses=[MockResponse(tool_calls=[
        ToolCall(id="c_a", name="mcp__t__a", arguments={}),
        ToolCall(id="c_b", name="mcp__t__b", arguments={}),
    ])])
    state, ctx, task, mem, pause, _p = _harness(llm, provider)

    run = asyncio.create_task(ActStep().execute(state, ctx))
    await started.wait()        # tc_a is mid-execution (blocking)
    pause.pause()               # interrupt while tc_a in flight

    with pytest.raises(HitlPark):
        await run

    # park 不写状态：AWAITING_HUMAN 由 TaskManager 据 RunOutcome 落（Task 4）
    assert task.status == "ACTIVE"
    res = await _tool_results(mem, ctx, state.scope)
    # 同上：按 tool_name 索引（内部标识值域下的等价断言）。
    by_name = {r.metadata.get("tool_name"): r for r in res}
    assert by_name["mcp__t__a"].metadata.get("interrupted") is True
    assert INTERRUPTED_MARK in by_name["mcp__t__a"].content
    assert by_name["mcp__t__b"].metadata.get("cancelled") is True
