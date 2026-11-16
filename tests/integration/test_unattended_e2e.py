"""无人值守端到端：会触发 HITL 的 task 在后台**跑到终态**，而不是挂起等一个永不到来的人。

两条链路，均从真实 `CtxWeftRuntime` 起（真实 `TaskManager` / `CapabilityGateway` /
`HitlService` / `HitlRegistry`），只在 LLM 与工具 provider 两处打桩：

1. `start_session(unattended=True)`：root task 无人值守且被强制 `unattended=True`；
   一次被 `HumanConfirmationAuthorizer` 门控的工具调用**当场被拒**（不登记、不 park），
   actor 收到 `[Blocked by human: ...]` 后照常收尾 → FINISHED。
2. `send_message(unattended=True)`：它开出的新 task 同样带标记、同样被强制 auto。

装配复用 `tests/integration/test_hitl_e2e_v2.py` 的桩与 `test_minimal_loop` 的模板。
"""

from __future__ import annotations

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.providers.authorizer import HumanConfirmationAuthorizer
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_hitl_e2e_v2 import (
    _BASH_CALL,
    _all_request_text,
    _finish_call,
    _RecordingBashTool,
    _wait_task_terminal,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



class _RouterLLM(MockLLMAdapter):
    """同 `test_hitl_e2e_v2._ActRouterLLM`，但 act 队列耗尽时回一句空文本而不是 IndexError。

    空文本在 `unattended=True` 下不会让位等用户（那正是本文件要验证的），
    只是把这一轮走完——用来吸收 close 边界后台 observe 之类的额外调用，让断言只钉
    真正关心的那几轮。
    """

    def __init__(self, act_responses: list[MockResponse], **kw) -> None:
        super().__init__(responses=[], **kw)
        self._act_responses = list(act_responses)
        self._act_idx = 0
        self.act_requests: list = []

    def complete(self, request, stream: bool = True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__report_observation" in names:
            # 判定回合——**不消费 act 队列**。2026-09-27 起 root 的 `finish_task` 也 park +
            # 后台判定（S-b），而 verdict 缺失 ≡ retry：不应这一轮，本文件那个「对照组：
            # 普通 session 的 root task 仍是 interactive」的前置（root 跑完、FINISHED）就
            # 永远等不到。判 success = 「它确实做完了」，正是这里要的。
            return self._stream(MockResponse(tool_calls=[ToolCall(
                id="obs", name="control__report_observation",
                arguments={"task_status": "success", "act_recap": "seg", "task_summary": "ok"},
            )]), request)
        self.act_requests.append(request)
        if self._act_idx >= len(self._act_responses):
            return self._stream(MockResponse(text=""), request)
        response = self._act_responses[self._act_idx]
        self._act_idx += 1
        return self._stream(response, request)


def _runtime_with_gated_tool(llm):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=llm, agent_provider=resolver, config=RuntimeConfig())
    runtime.providers.register_memory(InMemoryMemoryProvider())
    tool = _RecordingBashTool()
    runtime.providers.register_capability(
        tool, tool_authorizers={"fs:bash_exec": HumanConfirmationAuthorizer()},
    )
    return runtime, tool


@pytest.mark.asyncio
async def test_unattended_session_finishes_instead_of_parking_on_hitl() -> None:
    """驱动：`start_session(unattended=True)` → act 调一个需人工审批的工具 → 守卫在
    `HitlService.open()` 挡下 → gateway 合成拒绝 → 第二轮 act 收尾。

    会因下列任一项回归而失败：
    - 守卫失效（登记了 pending、task 落 AWAITING_HUMAN，后台作业就此永久挂起）；
    - `UnattendedHitl` 逸出到 agent loop（task 变成 FAILED 而不是 FINISHED）；
    - 拒绝没有回灌给模型（第二轮 prompt 里看不到 `[Blocked by human`）。
    """
    llm = _RouterLLM(act_responses=[_BASH_CALL, _finish_call()])
    runtime, tool = _runtime_with_gated_tool(llm)

    await initialize_all(runtime)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="run ls nightly",
         unattended=True,
    ))

    state = await _wait_task_terminal(handle, timeout=10.0)
    assert state is not None and state.task.status == "FINISHED", (
        f"unattended task must reach a terminal state, got "
        f"{state.task.status if state else None}"
    )
    # 一个都不许留：无人值守的 task 本就不该存在「等人回答」的记录。
    assert runtime.hitl_registry.list_pending(session_id=handle.session_id) == []
    assert tool.invocations == 0, "unapprovable tool must not run"
    assert state.task.unattended is True
    assert state.task.unattended

    assert len(llm.act_requests) >= 2
    assert "[Blocked by human" in _all_request_text(llm.act_requests[1]), (
        "the denial must reach the model's next prompt so it can move on"
    )


@pytest.mark.asyncio
async def test_send_message_unattended_marks_the_task_it_opens() -> None:
    """`send_message(unattended=True)` 开出的新 task 带标记，且被强制 auto——
    没有人会发下一条消息，interactive 的纯文本让位在这里等于永久挂起。"""
    llm = _RouterLLM(act_responses=[_finish_call("tc_a"), _finish_call("tc_b")])
    runtime, _tool = _runtime_with_gated_tool(llm)

    await initialize_all(runtime)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="first",
    ))
    first = await _wait_task_terminal(handle, timeout=10.0)
    assert first is not None and first.task.status == "FINISHED"
    # 对照组：普通 session 的 root task 仍是 interactive、仍不是无人值守。
    assert first.task.unattended is False
    assert not first.task.unattended

    await initialize_all(runtime)
    new_handle = await runtime.send_message(handle.agent_id, "second", unattended=True)
    tm = runtime._task_managers[handle.session_id]
    task = tm.get_task(new_handle.task_id)
    assert task is not None
    assert task.unattended is True
    assert task.unattended
