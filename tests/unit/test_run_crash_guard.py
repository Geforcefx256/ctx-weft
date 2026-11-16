"""M1 · 崩溃支的 `RunInterrupted` 受终态守卫约束（评审 minor）。

`except Exception` 分支上方置 `SUSPENDED` 那三行有个
`if task.status not in ("FINISHED", "FAILED", "CANCELED")` 守卫——异常抛出时 task
已是终态确有其事（observer 已判 FAILED、之后 FinalizeStep 又抛异常那条窄路径）。
但 `RunInterrupted` 的 emit 曾经在守卫外面、无条件发，导致这条窄路径上流里会
`TaskFailed` + `RunInterrupted{run_crash}` + `RunFinished{FAILED}` 并存——
`docs/events-v2.md` §2.4 让 host 靠 `RunInterrupted` 类型存在与否判断这次执行是否
非正常终止，这里会误报一次。修法：把这条 emit 挪进同一个 `if`。
"""

from __future__ import annotations

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
from tests.model_testkit import model_snapshot

from types import SimpleNamespace

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



class _CrashingDriver:
    """driver.run() 的替身：立刻抛一个普通异常，模拟运行层崩溃（非 outage、非取消）。"""

    async def run(self, state, ctx):
        raise RuntimeError("simulated run crash")
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
            state, ctx, _CrashingDriver(), run_id=state.run_id,
            initial_step="prepare", task=task, agent=state.agent,
        )
    except RuntimeError:
        pass  # _run_loop re-raises run_error at the end (caller normally routes to
              # TaskManager._handle_task_failure); we only care about the events emitted.
    return seen


@pytest.mark.asyncio
async def test_run_interrupted_not_emitted_when_task_already_terminal():
    """observer 已判 FAILED，随后 FinalizeStep 又抛异常那条窄路径：不再补发 RunInterrupted。"""
    runtime = _runtime()
    task = Task(id="root", session_id="s1", status="FAILED", error_code="TASK_FAILED_BY_OBSERVER")

    seen = await _run_and_capture(runtime, task)
    types = [e.type for e in seen]

    assert EventType.RUN_INTERRUPTED not in types
    assert EventType.RUN_FINISHED in types  # RUN_FINISHED 不受该守卫约束，照发
    assert task.status == "FAILED"  # 未被 except 分支覆写（FAILED 在终态三元组里）


@pytest.mark.asyncio
async def test_run_interrupted_still_emitted_when_task_not_terminal():
    """正向对照：task 还在 ACTIVE 时崩溃——RunInterrupted 照发，行为不受本修法影响。"""
    runtime = _runtime()
    task = Task(id="c1", session_id="s1", status="ACTIVE")

    seen = await _run_and_capture(runtime, task)
    types = [e.type for e in seen]

    assert EventType.RUN_INTERRUPTED in types
    run_interrupted = [e for e in seen if e.type == EventType.RUN_INTERRUPTED]
    assert run_interrupted[0].payload["reason"] == "run_crash"
    # Task 4：崩溃支不再写 task.status（那句 `= "SUSPENDED"` 是过渡态，随即被 TM 的
    # 处置覆盖）。task 落 PENDING/INTERRUPTED 由 TaskManager 据崩溃 RunOutcome 定。
    assert task.status == "ACTIVE"
