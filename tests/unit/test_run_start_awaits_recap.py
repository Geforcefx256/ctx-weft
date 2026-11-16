"""_run_loop 入口**不再**等在途段 recap（2026-09-22 拆除两道屏障）。

原先 driver 首步必须排在本 task 在途后台 recap 之后（spec 2026-07-16 §2）。拆除的依据：
正确性那一半已由**段界水位线**接管——`launch_recap` 钉住段界，
`segment_fold` 按它算折叠池，迟到的折叠不再抢走段界、也不再排到新消息之后
（见 tests/unit/test_segment_fold.py 的水位线三条）。剩下的只是性能：首次装配可能读到
尚未被 supersede 的 raw，prompt 白胀一轮——这条代价已确认接受。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.driver import ExecutionRegistry, LoopState
from ctx_weft.core.runtime import CtxWeftRuntime
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryAddress



class _FakeBus:
    async def emit(self, event) -> None:
        pass


class _RecordingDriver:
    """driver.run 是 async generator：首次迭代时记录时刻，不产出任何 outcome。"""

    def __init__(self, order: list):
        self._order = order
        self.executions = ExecutionRegistry()

    async def run(self, state, ctx):
        self._order.append("driver_started")
        if False:  # 使函数成为 async generator，且不产出任何 outcome
            yield


def _make_state_and_task():
    task = Task(
        id="t1", session_id="s1", status="ACTIVE",
        assigned_agent_id="ag1", creator_agent_id="ag1",
        settings=NormalTaskSettings(),
    )
    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    agent = SimpleNamespace(id="ag1")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    state = LoopState(run_id="r1", session=session, task=task, agent=agent, scope=scope, resolved_model=model_snapshot(model="mock", account=""))
    return state, task, agent


def _fake_runtime_self():
    return SimpleNamespace(
        _closing=False,
        _event_bus=_FakeBus(),
        _pending_actor_runs={},
        _capability_cache=SimpleNamespace(evict=lambda agent_id: None),
    )


@pytest.mark.asyncio
async def test_run_start_does_not_wait_for_pending_observe():
    """有在途 recap：driver 首步**照常先跑**，不为它等一次后台 LLM 往返。"""
    order: list = []

    async def slow_recap():
        await asyncio.sleep(0.02)
        order.append("recap_done")

    state, task, agent = _make_state_and_task()
    recap = asyncio.create_task(slow_recap())
    driver = _RecordingDriver(order)
    driver.executions.register("recap-run", task.id, "observe", recap, agent_id=agent.id)

    await CtxWeftRuntime._run_loop(
        _fake_runtime_self(), state, SimpleNamespace(task_manager=None), driver,
        "r1", "prepare", task, agent,
    )

    assert order == ["driver_started"], f"run 不该再等 recap，实得 {order}"
    await recap                              # 收尾，别留 pending task
    assert order == ["driver_started", "recap_done"]



@pytest.mark.asyncio
async def test_run_start_passthrough_without_pending():
    """无 pending：直通，driver 正常执行。"""
    order: list = []
    state, task, agent = _make_state_and_task()
    driver = _RecordingDriver(order)

    await CtxWeftRuntime._run_loop(
        _fake_runtime_self(), state, SimpleNamespace(task_manager=None), driver,
        "r1", "prepare", task, agent,
    )

    assert order == ["driver_started"]


@pytest.mark.asyncio
async def test_run_start_unaffected_by_errored_recap():
    """在途 recap 以异常终结：run 压根不看它，driver 照常执行（段保 raw 降级）。

    拆掉屏障之前，这条测的是入口那圈 try/except 把异常吞住；现在没有那圈 await，
    隔离是结构性的——留着它是为了守住「recap 炸了不牵连 run」这个不变量本身。
    """
    order: list = []

    async def errored_recap():
        raise RuntimeError("guard region boom")

    state, task, agent = _make_state_and_task()
    recap = asyncio.create_task(errored_recap())
    driver = _RecordingDriver(order)
    driver.executions.register("recap-run", task.id, "observe", recap, agent_id=agent.id)

    await CtxWeftRuntime._run_loop(
        _fake_runtime_self(), state, SimpleNamespace(task_manager=None), driver,
        "r1", "prepare", task, agent,
    )

    assert order == ["driver_started"], "recap 异常不得阻断 run"
    with pytest.raises(RuntimeError):
        await recap
