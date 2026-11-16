"""LLM-outage interrupt is tagged with reason='llm_outage' on the **run-level** event,
so the host/frontend can tell it apart from a generic interrupt (e.g. restart).

Task 6 起会话状态不在这里宣布：loop 发 RunInterrupted（带 reason），TM 据处置表把它折成
task 级的 TaskInterrupted（同样带 reason/error_code）。2026-09-04（Task 12，events-v2 §5）
起 TM 不再额外聚合出一条会话级 TaskQueueInterrupted——那条信号的消费者（SessionRegistry
的会话状态机）早已退役——恢复期的可观测性改由 AGENT_* 现状广播承担（Task 11），实时路径
上则是 TaskInterrupted 本身驱动 AgentLifecycleManager 五态机转 interrupted。
"""

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
    def complete(self, request, stream=True):
        async def _gen():
            raise LLMOutageError("simulated outage exhausted")
            yield  # pragma: no cover  (make this an async generator)
        return _gen()


@pytest.mark.asyncio
async def test_outage_interrupt_carries_llm_outage_reason():
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(
        llm=_OutageLLM(responses=[]),
        agent_provider=resolver,
        config=RuntimeConfig(llm_self_heal_max_attempts=1),
    )
    runtime.providers.register_memory(InMemoryMemoryProvider())

    seen = []
    async def _spy(ev):
        seen.append(ev)
    runtime._event_bus.subscribe(None, _spy)

    await initialize_all(runtime)
    await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="hi")

    interrupts = [e for e in seen if getattr(e, "type", None) == EventType.RUN_INTERRUPTED]
    assert interrupts, "expected RunInterrupted"
    assert all((e.payload or {}).get("reason") == "llm_outage" for e in interrupts)
    # 成因也随 task.error 抵达 TaskManager 折出的 task 级事实（host 据此区分 LLM 故障
    # vs 重启中断）。精确断言：reason 必须是**码** "llm_outage"，不是 str(exc) 那种
    # 自由文本。host 按码分流（三份契约：升级须知 / docs/events-v2.md §2.1.2 /
    # spec/golden/07）。2026-09-04（Task 12）起不再有会话级 TaskQueueInterrupted 聚合
    # 信号（已停发，events-v2 §5）——这份 reason 现在直接落在 TaskInterrupted 本身。
    task_sig = [e for e in seen if getattr(e, "type", None) == EventType.TASK_INTERRUPTED]
    assert task_sig, "expected TaskInterrupted"
    assert (task_sig[0].payload or {}).get("reason") == "llm_outage"
    assert EventType.SESSION_STATUS_CHANGED not in [getattr(e, "type", None) for e in seen]
    assert "TaskQueueInterrupted" not in [str(getattr(e, "type", "")) for e in seen], (
        "TaskQueueInterrupted 已停发（Task 12）、类型已删（2026-09-05），不应再出现"
    )
