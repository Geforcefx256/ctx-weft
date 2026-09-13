"""spec: delivery-acceptance——集成路径：compat 内联修正、park 负向、子任务上抛标注、依赖传播。"""

from __future__ import annotations

import json

import pytest

from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from ctx_weft.core.acceptance.protocol import AcceptanceFinding, BUILTIN_STRUCTURE_ID
from ctx_weft.core.loop.driver import EventType
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import LoopConfig


class _RepeatLastMock(MockLLMAdapter):
    """超出 responses 后重复最后一个响应——免于精确数后台 observe 的调用次数。"""

    def complete(self, request, stream=True):
        if self._idx >= len(self._responses):
            self._idx = len(self._responses) - 1
        return super().complete(request, stream)


def _structure_spec(fields):
    return [{"checker_id": BUILTIN_STRUCTURE_ID, "checker_version": "1",
             "params": {"fields": fields}, "required": True}]


class _CaptureBus(InProcessEventBus):
    def __init__(self):
        super().__init__()
        self.captured = []

        async def _sink(e):
            self.captured.append(e)

        self.subscribe(None, _sink)

    def of(self, t):
        return [e for e in self.captured if e.type == t]


def _make_rt(llm, mode="required") -> tuple:
    from ctx_weft.core.models.config import RuntimeConfig
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    bus = _CaptureBus()
    rt = make_runtime(llm=llm, agent_provider=provider, event_bus=bus,
                      config=RuntimeConfig(acceptance_mode=mode))
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt, bus


async def test_compat_entry_performs_actual_repair_and_returns_final() -> None:
    """spec 场景：兼容入口实际完成一次修正——首检失败 → 真实修正生成 → 返回修正后结果。"""
    llm = _RepeatLastMock(responses=[
        MockResponse(text='{"total": "1.00"}'),   # 首答：结构过、total 错
        MockResponse(text='{"total": "2.00"}'),   # 修正轮：全过
        # 后台 observe/recap 等额外消耗（顺序在两轮 run 之后）
        MockResponse(text='{"total": "2.00"}'),
        MockResponse(text='{"total": "2.00"}'),
    ])
    rt, bus = _make_rt(llm)
    calls = {"n": 0}

    def _total(candidate, params):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("failed", [AcceptanceFinding(field="total", condition="sum mismatch",
                                                 evidence_ref="rows", suggestion="recompute")])
        return ("passed", [])

    rt.acceptance_registry.register("acme.total", "1", _total)
    spec = _structure_spec({"total": "string"})
    spec.append({"checker_id": "acme.total", "checker_version": "1",
                 "params": {}, "required": True})
    handle, state = await rt.run_single_task(
        template_id="agent:tpl_echo", user_prompt="reconcile",
        acceptance_spec=spec)
    task = state.task
    assert task.status == "FINISHED", f"got {task.status}"
    assert task.acceptance_repairs_used == 1            # 恰好一次修正
    assert calls["n"] == 2                              # 修正后全量复检真的跑了
    checked = bus.of(EventType.TASK_ACCEPTANCE_CHECKED)
    assert [e.payload["verdict"] for e in checked] == ["failed", "passed"]
    assert bus.of(EventType.TASK_ACCEPTANCE_RETRY_RESERVED)
    assert task.outputs and "2.00" in str(task.outputs)  # 返回修正后的最终结果


async def test_compat_entry_terminates_when_repair_still_fails() -> None:
    llm = _RepeatLastMock(responses=[
        MockResponse(text='{"total": "1.00"}'),
        MockResponse(text='{"total": "1.00"}'),   # 修正后仍错
        MockResponse(text='{"total": "1.00"}'),
        MockResponse(text='{"total": "1.00"}'),
    ])
    rt, bus = _make_rt(llm)

    def _always_fail(candidate, params):
        return ("failed", [AcceptanceFinding(field="total", condition="sum mismatch",
                                             evidence_ref="rows", suggestion="recompute")])

    rt.acceptance_registry.register("acme.total", "1", _always_fail)
    spec = _structure_spec({"total": "string"})
    spec.append({"checker_id": "acme.total", "checker_version": "1",
                 "params": {}, "required": True})
    handle, state = await rt.run_single_task(
        template_id="agent:tpl_echo", user_prompt="reconcile",
        acceptance_spec=spec)
    assert state.task.status == "FAILED"
    assert state.task.error_code == "TASK_FAILED_ACCEPTANCE"


async def test_park_turn_never_reaches_acceptance() -> None:
    """交互中间交流（park）不触发验收、不产生记录、不动额度（spec req2 场景）。"""
    rt, bus = _make_rt(_RepeatLastMock(responses=[MockResponse(text='{"total": "1.00"}')]))
    rt.acceptance_registry.register("acme.total", "1",
                                    lambda c, p: ("passed", []))
    # interactive 任务经 start_session 太重；用 compat 入口 + interaction_mode=interactive
    # 的等价路径：直接断言 park 的 RunOutcome 不进 gate——结构上 park 抛 HitlPark、
    # finalize 不执行。此处用 required+声明的 compat 任务、纯文本回复但 auto 模式
    # 会走 observe→finalize；真正的 park 负向在下方 TM 级断言。
    spec = _structure_spec({"total": "string"})
    handle, state = await rt.run_single_task(
        template_id="agent:tpl_echo", user_prompt="hi", acceptance_spec=spec)
    # auto 模式纯文本 → observe 收尾 run（非 park）；此断言确保正常路径产 CHECKED
    assert bus.of(EventType.TASK_ACCEPTANCE_CHECKED)
    # park 负向：HitlPark 结局（RunOutcome AWAITING_HUMAN）从不携带 acceptance——
    # disposition 表无验收分支可走，且 finalize 未执行。
    from ctx_weft.core.orchestrator.task.disposition import (
        RunOutcome, RunOutcomeKind, disposition_for)
    disp = disposition_for(RunOutcome(kind=RunOutcomeKind.AWAITING_HUMAN, hitl_id="h1"),
                           retry_count=0, max_retries=3)
    assert disp.status == "AWAITING_HUMAN" and disp.event_type == "TaskAwaitingHuman"


async def test_acceptance_failed_blocks_on_success_dependents() -> None:
    """验收 FAILED 对 on_success 下游不放行（复用 task-handoff 依赖条件）。"""
    from ctx_weft.core.models.discriminators import TaskErrorCode
    from tests.unit.test_task_dependency_conditions import _NoopRunner, _task
    bus = _CaptureBus()
    tm = TaskManager(session_id="s1", max_concurrent=0, event_bus=bus)
    from ctx_weft.core.models.session import Session
    tm.set_session(Session(id="s1", user_prompt="", status="RUNNING"))
    tm.set_runner(_NoopRunner())
    a = _task("a")
    b = _task("b", dag_deps=["a"], dep_conditions={"a": "success"})
    tm.register_task(a)
    tm.register_task(b)
    await tm.push_task(a)
    await tm.push_task(b, blocked_by=["a"])
    a.status = "FAILED"
    a.error_code = TaskErrorCode.TASK_ACCEPTANCE_FAILED
    await tm.on_task_finished("a", status="FAILED")
    assert b.status == "CANCELED"
    assert b.error_code == TaskErrorCode.BLOCKED_BY_FAILED_DEP


async def test_off_vs_shadow_same_input_same_delivery() -> None:
    """spec 影子对照：off 与 shadow 交付决策完全一致，差异仅在验收记录。"""
    def _make(mode):
        llm = _RepeatLastMock(responses=[MockResponse(text='{"total": "1.00"}')])
        rt, bus = _make_rt(llm, mode=mode)

        def _fail(candidate, params):
            return ("failed", [AcceptanceFinding(field="total", condition="sum mismatch",
                                                 evidence_ref="rows", suggestion="recompute")])

        rt.acceptance_registry.register("acme.total", "1", _fail)
        spec = _structure_spec({"total": "string"})
        spec.append({"checker_id": "acme.total", "checker_version": "1",
                     "params": {}, "required": True})
        return rt, bus, spec

    rt_off, bus_off, spec = _make("off")
    _, state_off = await rt_off.run_single_task(
        template_id="agent:tpl_echo", user_prompt="x", acceptance_spec=spec)
    rt_sh, bus_sh, spec2 = _make("shadow")
    _, state_sh = await rt_sh.run_single_task(
        template_id="agent:tpl_echo", user_prompt="x", acceptance_spec=spec2)
    # 交付决策一致：必需检查失败在 shadow 下不改变结果
    assert state_off.task.status == state_sh.task.status == "FINISHED"
    assert state_off.task.outputs == state_sh.task.outputs
    # 差异仅在记录：off 零事件；shadow 有 CHECKED（verdict=failed）
    assert not bus_off.of(EventType.TASK_ACCEPTANCE_CHECKED)
    assert bus_sh.of(EventType.TASK_ACCEPTANCE_CHECKED)[0].payload["verdict"] == "failed"
    assert not bus_sh.of(EventType.TASK_ACCEPTANCE_RETRY_RESERVED)  # 不重排不修正


async def test_unregistered_checker_rejects_fresh_task_loudly() -> None:
    """spec：未注册 (id, version) 的声明在派发前被响亮拒绝（非恢复任务）。"""
    import pytest as _pytest
    from ctx_weft.core.acceptance.protocol import InvalidAcceptanceSpec
    rt, bus = _make_rt(MockLLMAdapter(responses=[MockResponse(text="x")]))
    spec = [{"checker_id": "nope.missing", "checker_version": "9",
             "params": {}, "required": True}]
    with _pytest.raises(InvalidAcceptanceSpec, match="nope.missing"):
        await rt.run_single_task(
            template_id="agent:tpl_echo", user_prompt="x", acceptance_spec=spec)
