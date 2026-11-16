"""_run_loop routes LLMOutageError to INTERRUPTED, not FAILED."""

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.protocols import LLMOutageError
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



class _OutageLLM(MockLLMAdapter):
    """Raises LLMOutageError on every complete() (outage already self-heal-exhausted)."""

    def complete(self, request, stream=True):
        self.last_request = request
        async def _gen():
            raise LLMOutageError("simulated outage exhausted")
            yield  # pragma: no cover  (make this an async generator)
        return _gen()


@pytest.mark.asyncio
async def test_outage_marks_session_interrupted_not_failed():
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    # max_attempts=1 so self-heal exhausts immediately (no retry delays in tests)
    runtime = make_runtime(
        llm=_OutageLLM(responses=[]),
        agent_provider=resolver,
        config=RuntimeConfig(llm_self_heal_max_attempts=1),
    )
    runtime.providers.register_memory(InMemoryMemoryProvider())

    seen = []
    orig_emit = runtime._event_bus.emit
    async def _spy(ev):
        seen.append(ev)
        return await orig_emit(ev)
    runtime._event_bus.emit = _spy

    await initialize_all(runtime)
    handle, state = await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="hi")

    # task must NOT be terminal-failed
    assert state.task.status == "INTERRUPTED"

    types = [getattr(e, "type", None) for e in seen]
    # 各层各发各的（Task 6）：loop 报 run 级事实，TM 折 task 级事实，ALM 判 agent 状态。
    # SM 的会话级 SessionInterrupted 已随会话状态机一并退役（Task 16），TM 代它聚合的
    # 会话级 TaskQueueInterrupted 也已停发（2026-09-04 Task 12，events-v2 §5）——观测点
    # 下移到仍然真实存在的 ALM 五态机（agent_state.py），验证强度不降：仍然要求
    # 「异常时相关 agent 进入 interrupted」，不是弱化成「没抛异常」。
    run_interrupted = [e for e in seen if e.type == EventType.RUN_INTERRUPTED]
    assert run_interrupted, "expected RunInterrupted (run-level fact)"
    assert run_interrupted[0].payload["reason"] == "llm_outage"
    assert "TaskQueueInterrupted" not in [str(t) for t in types], "TaskQueueInterrupted 已停发/已删"
    agent_interrupted = [e for e in seen if e.type == EventType.AGENT_INTERRUPTED]
    assert agent_interrupted, "expected AgentInterrupted (ALM verdict, replaces retired SM session verdict)"
    aid = agent_interrupted[0].agent_id
    assert aid and runtime._agent_lifecycle_manager.status_of(aid) == "interrupted"
    # 通用 setter 退役：谁都不许再发它。
    assert EventType.SESSION_STATUS_CHANGED not in types
    # no TASK_FAILED emitted
    assert not [e for e in seen if getattr(e, "type", None) == EventType.TASK_FAILED]
