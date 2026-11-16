"""一个 task 同时最多一个「挡住它」的 HITL——由 act 的 tool call 循环串行保证。

docs/events-v2.md §2.3 用 ⚠️ 明写这条不变式是**结构性巧合**（`_execute_tool_calls`
恰好是 `for` 串行执行、第一个 park 就 unwind 整个 run），必须有测试直接钉住：
`TaskAwaitingHuman.hitl_id` 是单值，`SessionWaiting` 也不带计数，若有人把这个循环
改成 `asyncio.gather`，两个 tool call 会同时 park 出两个 HITL，「挡住这个 task 的
那一个请求」就不再唯一。

钉法：两个都会 park 的 tool call，只有**第一个**真的进 invoke——第二个连 invoke
都不该被调用。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.signals import HitlPark
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import MemoryAddress
from ctx_weft.protocols.llm import ToolCall



class _ParkingGateway:
    """每次 invoke 都 park（模拟两个都需要审批的工具），并记下被调用的顺序。"""

    def __init__(self) -> None:
        self.invoked: list[str] = []

    async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id):
        self.invoked.append(tool_call_id)
        raise HitlPark(hitl_id=f"hit_for_{tool_call_id}", tool_call_id=tool_call_id)


def _state() -> LoopState:
    session = Session(id="s1", user_prompt="x")
    task = Task(id="t1", session_id="s1", status="ACTIVE")
    agent = Agent(id="ag1", session_id="s1", template_id="tpl")
    return LoopState(
        run_id="run_1", session=session, task=task, agent=agent,
        scope=MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1"),
        resolved_model=model_snapshot(model="mock", account=""),
    )


@pytest.mark.asyncio
async def test_only_the_first_parking_tool_call_is_invoked() -> None:
    from ctx_weft.core.loop.steps.act import _execute_tool_calls

    gw = _ParkingGateway()
    ctx = LoopContext(assembler=None, llm=None, memory=None, event_bus=None,
                      provider_ctx=None, capability_gateway=gw)
    state = _state()
    tool_calls = [
        ToolCall(id="tc_1", name="needs_approval", arguments={}),
        ToolCall(id="tc_2", name="needs_approval_too", arguments={}),
    ]

    with pytest.raises(HitlPark) as caught:
        await _execute_tool_calls(state, ctx, tool_calls)

    # 串行 unwind：第一个 park 就整个抛出，第二个 tool call 连 invoke 都没进。
    assert gw.invoked == ["tc_1"], (
        "tool call 循环必须串行：第二个 tool call 不得被执行，否则一个 task 会同时"
        "挂着两个 HITL，TaskAwaitingHuman.hitl_id 的单值语义就破了"
    )
    # 抛出的正是**第一个**的 park——「挡住这个 task 的那一个请求」唯一确定。
    assert caught.value.hitl_id == "hit_for_tc_1"
    assert caught.value.tool_call_id == "tc_1"
    # park 不写状态：AWAITING_HUMAN 由 TaskManager 据 RunOutcome 落（Task 4）
    assert state.task.status == "ACTIVE"
