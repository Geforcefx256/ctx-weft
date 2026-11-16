"""端到端：`RunInterrupted` 从 run 里发，`TaskInterrupted` 发在重试判定之后。

两条异常路径**不对称**：

- outage：`_run_loop` 自己捕获、不重抛 → 没有重试判定，直接挂起等 `/resume`。
- run 崩溃：捕获后重抛 → `TaskManager._handle_task_failure` → 原地重试或挂起。

于是 `RunInterrupted` 无条件发（那次 run 确实死了），`TaskInterrupted` 只在
「挂起等 /resume」那一支发；走重试的那一支发的是 `TaskRequeued`。
"""

from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import LLMCallError
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


TASK_INTERRUPTED = "TaskInterrupted"
_MOCK_CONTEXT_LIMIT = 100_000


class _CrashLLM(MockLLMAdapter):
    """每次 act 调用都抛**可重试、非 outage** 的错：走 `_run_loop` 的泛 except → 重抛。"""

    @staticmethod
    def _is_recognize_intent(request) -> bool:
        tools = getattr(request, "tools", None) or []
        return any(getattr(t, "name", "") == "control__update_task_metadata" for t in tools)

    def complete(self, request, stream=True):
        self.last_request = request
        if self._is_recognize_intent(request):
            return super().complete(request, stream=stream)

        async def _gen():
            raise LLMCallError("degenerate output", retriable=True, outage=False)
            yield  # pragma: no cover  (make this an async generator)

        return _gen()


def _runtime(llm) -> object:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(
        llm=llm,
        agent_provider=resolver,
        config=RuntimeConfig(
            task_max_retries=1,
            llm_self_heal_max_attempts=1,
            llm_self_heal_base_delay_sec=0.0,
            llm_self_heal_max_interval_sec=0.0,
            llm_self_heal_max_duration_sec=0.1,
        ),
    )
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _wait_for(seen: list, type_: str, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if any(getattr(e, "type", None) == type_ for e in seen):
            return
        await asyncio.sleep(0.02)
    raise TimeoutError(f"{type_} not seen within {timeout}s; got {[e.type for e in seen]}")


@pytest.mark.asyncio
async def test_crash_retry_then_suspend_order() -> None:
    runtime = _runtime(_CrashLLM(responses=[]))
    seen: list = []
    async def _spy(ev):
        seen.append(ev)
    runtime._event_bus.subscribe(None, _spy)

    await initialize_all(runtime)
    handle = await runtime.start_session(
        SessionStartParams.create(template_id="agent:tpl_echo", user_prompt="hi"))
    await _wait_for(seen, TASK_INTERRUPTED)

    types = [e.type for e in seen]
    # 崩溃 + 可重试：重排之前只有 run 级事实，没有 task 级的「断了」。
    first_requeue = types.index(EventType.TASK_REQUEUED)
    prefix = types[:first_requeue]
    assert EventType.RUN_INTERRUPTED in prefix, "run 崩溃必须在 run 里发 RunInterrupted"
    assert TASK_INTERRUPTED not in prefix, "还能重试的 task 不得先被打成 INTERRUPTED"
    # 重试耗尽之后才轮到 task 级的挂起事实。
    assert types.index(TASK_INTERRUPTED) > first_requeue
    assert types.count(TASK_INTERRUPTED) == 1
    # task_max_retries=1 → 两次 run，各死一次。
    assert types.count(EventType.RUN_INTERRUPTED) == 2
    # 问题 1：run 域的事实必须带 run_id。
    assert all(e.run_id for e in seen if e.type == EventType.RUN_INTERRUPTED)

    view = await rebuild_view(runtime.event_store, handle.session_id)
    assert [t.status for t in view.tasks.values()] == ["INTERRUPTED"]


@pytest.mark.asyncio
async def test_outage_emits_both_run_and_task_interrupted() -> None:
    """outage 支没有重试判定：run 级与 task 级两条事实都从 `_run_loop` 里发。"""

    class _OutageLLM(MockLLMAdapter):
        def complete(self, request, stream=True):
            from ctx_weft.protocols import LLMOutageError

            async def _gen():
                raise LLMOutageError("simulated outage exhausted")
                yield  # pragma: no cover

            return _gen()

    runtime = _runtime(_OutageLLM(responses=[]))
    seen: list = []
    async def _spy(ev):
        seen.append(ev)
    runtime._event_bus.subscribe(None, _spy)

    await initialize_all(runtime)
    _handle, state = await runtime.run_single_task(
        template_id="agent:tpl_echo", user_prompt="hi")

    assert state.task.status == "INTERRUPTED"
    run_evs = [e for e in seen if e.type == EventType.RUN_INTERRUPTED]
    task_evs = [e for e in seen if e.type == TASK_INTERRUPTED]
    assert run_evs and run_evs[0].payload["reason"] == "llm_outage"
    assert run_evs[0].run_id
    assert task_evs, "outage 挂起也要有 task 域的事实，否则投影停在 ACTIVE"
    # Task 4：TaskInterrupted 改由 TaskManager 发——TM 在 run 外面、拿不到 run_id，
    # 故这条 task 域事实的 run_id 是 None（run 域那条照旧带，见上面 run_evs 的断言）。
    assert task_evs[0].run_id is None
    assert task_evs[0].payload["reason"] == "llm_outage"
    # 2026-09-04（Task 12，events-v2 §5）起 TaskManager 不再额外聚合出一条会话级
    # TaskQueueInterrupted——上面两条 task/run 域事实已经是完整的观测点。
    assert "TaskQueueInterrupted" not in [str(e.type) for e in seen]
