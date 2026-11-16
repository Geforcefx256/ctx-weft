"""New foreground runs do not wait for a previous observation to finish."""

from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.loop.driver import ExecutionRegistry
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



@pytest.mark.asyncio
async def test_registry_waits_until_the_pending_fold_completes():
    registry = ExecutionRegistry()
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    registry.register("run", "task", "observe", task, agent_id="agent")
    waiter = asyncio.create_task(registry.wait("task"))
    await asyncio.sleep(0)
    assert not waiter.done()
    release.set()
    await asyncio.wait_for(waiter, timeout=1)
    assert registry.pending_run_id("task") is None


@pytest.mark.asyncio
async def test_registry_wait_is_a_noop_without_pending_work():
    registry = ExecutionRegistry()
    await asyncio.wait_for(registry.wait("task"), timeout=1)
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    registry.register("run", "task", "observe", done, agent_id="agent")
    await asyncio.wait_for(registry.wait("task"), timeout=1)
    assert registry.pending_run_id("task") is None


@pytest.mark.asyncio
async def test_old_completion_does_not_clear_new_registration():
    registry = ExecutionRegistry()
    release = asyncio.Event()
    old = asyncio.create_task(asyncio.sleep(0))
    new = asyncio.create_task(release.wait())
    registry.register("old", "task", "observe", old, agent_id="agent")
    registry.register("new", "task", "observe", new, agent_id="agent")
    await old
    await asyncio.sleep(0)
    assert registry.pending_run_id("task") == "new"
    assert "old" not in registry.tasks
    release.set()
    await new
    await asyncio.sleep(0)
    assert not registry.tasks
    assert not registry.latest


@pytest.mark.asyncio
async def test_new_task_run_does_not_wait_for_previous_tasks_pending_fold():
    """agent 已终态 → `send_message` 建新 task，其 run **不等**上一轮折叠落地。

    这条曾是「agent 轴 recap 强一致」的回归守卫，2026-09-22 随两道屏障一并反转（理由见
    模块 docstring）。它现在守的是那次拆除真的生效了：新一轮不为上一轮的后台 LLM 往返
    买单。代价是这一轮的装配可能读到尚未 supersede 的 raw——prompt 白胀一轮，已接受。
    """
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse

    fold_done = False
    release = asyncio.Event()
    llm_saw_fold_done: list[bool] = []

    class _RecordingLLM(MockLLMAdapter):
        # `complete` 是**同步返回 AsyncIterator** 的（不是 async def），照原形状覆盖：
        # 每次被调时记一笔「此刻上一轮的折叠落地了没」，其余原样委托。
        def complete(self, request, stream: bool = True):      # type: ignore[override]
            llm_saw_fold_done.append(fold_done)
            return super().complete(request, stream)

    llm = _RecordingLLM(observer_verdict="success",
                        responses=[MockResponse(text="ok") for _ in range(40)])
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(agent_provider=resolver, llm=llm)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    from ctx_weft.core.runtime import SessionStartParams
    await initialize_all(runtime)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="第一轮",
    ))
    await handle.wait_for_finish(timeout=30)
    agent_id = handle.agent_id

    # 造出「上一轮的 close 边界折叠还在飞」这个必然窗口：登记一个我们能掐着放行的
    # 在途折叠，key 是 agent（对应真实代码里 launch 时的两轴同步登记）。
    async def _slow_fold() -> None:
        nonlocal fold_done
        await release.wait()
        fold_done = True

    pending = asyncio.create_task(_slow_fold())
    runtime._step_executions[handle.session_id].register(
        "previous-observe", handle.task_id, "observe", pending, agent_id=agent_id)

    llm_saw_fold_done.clear()
    await initialize_all(runtime)
    send = asyncio.create_task(runtime.send_message(agent_id, "第二轮"))

    # 给新 run 充分的调度机会：屏障拆掉之后，它该在 `release` 之前就把 LLM 调起来。
    for _ in range(50):
        await asyncio.sleep(0)
    assert llm_saw_fold_done and not any(llm_saw_fold_done), (
        "新 task 的 run 仍在等上一轮折叠 —— 两道屏障没拆干净，人回复还在为后台往返买单"
    )

    release.set()
    await pending
    handle2 = await asyncio.wait_for(send, timeout=30)
    await handle2.wait_for_finish(timeout=30)

    assert llm_saw_fold_done, "新一轮始终没跑起来"
    # 这一轮从头到尾都跑在折叠之前——正是「不等」的直接证据。反过来说，若将来有人给
    # 这条路重新加上等待，上面那条断言会先炸，不会静默地把延迟加回来。
    assert not any(llm_saw_fold_done), (
        "新一轮有 LLM 调用发生在折叠落地之后 —— 屏障可能被重新引入，"
        f"逐次观测到的 fold_done={llm_saw_fold_done}"
    )
