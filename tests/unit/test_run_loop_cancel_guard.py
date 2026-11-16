"""_run_loop A1 守卫（Task 10）：只有这次取消真的会让 task 翻成 CANCELED 时才发
RUN_CANCELED；已被熔断标 FAILED 的 run 被协作取消属内部清场，不发（否则 host/postgres
投影会把 FAILED 盖成 CANCELED）。RUN_FINISHED 无条件发出。

Task 4 起 **TASK_CANCELED 不再由 _run_loop 发**（task 状态事件只从 TaskManager 出），
run 侧也不再写 task.status；守卫的判据改成 except 分支里取的 cancel_takes_effect，
task 侧的同一守卫在 `TaskManager.apply_run_outcome`（终态已坐实 → 不应用 run 的结局）。
"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
from tests.model_testkit import model_snapshot

from types import SimpleNamespace

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.protocols.events import EventType
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.models.agent import Agent, LoopGuard
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import MemoryAddress, ProviderContext
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")



class _CancelingDriver:
    """driver.run() 的替身：立刻抛 asyncio.CancelledError，模拟协作取消命中检查点。"""

    async def run(self, state, ctx):
        raise asyncio.CancelledError("simulated cooperative cancel")
        yield  # pragma: no cover — 让方法保持 async generator 形状


def _runtime() -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    return runtime


def _build_state_and_ctx(runtime: CtxWeftRuntime, task: Task):
    session = Session(id="s1", user_prompt="hi")
    agent = Agent(id="agt1", session_id="s1", template_id="tpl", loop_guard=LoopGuard())
    scope = MemoryAddress(session_id="s1", task_id=task.id, agent_id="agt1")
    state = LoopState(run_id=generate_id("run"), session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))
    provider_ctx = ProviderContext(session_id="s1", tenant_id="default", task_id=task.id, agent_id="agt1")
    ctx = LoopContext(
        assembler=None, llm=runtime._resolve_llm(ctx=_PROVIDER_CONTEXT), memory=runtime.providers.get_memory(),
        event_bus=runtime._event_bus, provider_ctx=provider_ctx,
        capability_cache=CapabilityCache(),
    )
    return state, ctx


async def _run_and_capture(runtime: CtxWeftRuntime, task: Task):
    state, ctx = _build_state_and_ctx(runtime, task)
    seen = []
    orig_emit = runtime._event_bus.emit

    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)

    runtime._event_bus.emit = _spy
    try:
        await runtime._run_loop(
            state, ctx, _CancelingDriver(), run_id=state.run_id,
            initial_step="prepare", task=task, agent=state.agent,
        )
    except asyncio.CancelledError:
        pass
    return seen


@pytest.mark.asyncio
async def test_was_cancelled_and_task_failed_suppresses_cancel_events():
    """熔断已先手把 task 标 FAILED 后再协作取消：不发 RUN_CANCELED/TASK_CANCELED，RUN_FINISHED 照发。"""
    runtime = _runtime()
    task = Task(id="root", session_id="s1", status="FAILED", error_code="TASK_FAILED_BY_THRESHOLD")

    seen = await _run_and_capture(runtime, task)
    types = [e.type for e in seen]

    assert EventType.RUN_CANCELED not in types
    assert EventType.TASK_CANCELED not in types
    assert EventType.RUN_FINISHED in types
    assert task.status == "FAILED"  # run 侧不写状态；FAILED 是熔断先手写的，原样保留


@pytest.mark.asyncio
async def test_was_cancelled_and_task_not_terminal_emits_run_canceled():
    """回归：普通协作取消（task 未处于任何终态）→ RUN_CANCELED 照发。

    TASK_CANCELED 不在这里发了（Task 4）：run 交回 `RunOutcome(kind=canceled)`，
    由 TaskManager 落 CANCELED 并发事件。
    """
    runtime = _runtime()
    task = Task(id="c1", session_id="s1", status="ACTIVE")

    seen = await _run_and_capture(runtime, task)
    types = [e.type for e in seen]

    assert EventType.RUN_CANCELED in types
    assert EventType.TASK_CANCELED not in types      # 只从 TaskManager 出
    assert EventType.RUN_FINISHED in types
    assert task.status == "ACTIVE"                   # run 侧不写状态
