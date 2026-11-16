"""装配失败的两个出口必须报同一个 reason（总账 D6）。

`_handle_task_failure` 的不可重试支 / 重试耗尽支都落到 `_suspend_task_interrupted`，
它此前签名里根本没有 `reason` 形参，发射处硬编码 `InterruptReason.RUN_CRASH`——
调用方传的 `InterruptReason.ASSEMBLY_FAILURE`（装配失败专用）被吞掉。搭台手法照抄
`tests/unit/test_run_crash_suspend.py`（`_tm` 建 TaskManager + `_CapturingBus`），
但这里直接调 `_handle_task_failure` 走**装配失败**入口，不经 `_run_task` 的真崩溃路径。
"""

from __future__ import annotations

from ctx_weft.core.models.discriminators import InterruptReason
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols.events import EventType
from tests.unit._stub_runner import StubRunner


class _CapturingBus:
    async def emit_batch(self, events, *, on_committed=None):
        if on_committed is not None:
            on_committed()
        for event in events:
            await self.emit(event)

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:
        self.events.append(event)


class _NonRetriable(Exception):
    retriable = False


def _tm(bus: _CapturingBus) -> tuple[TaskManager, Task]:
    tm = TaskManager(session_id="s1", event_bus=bus)
    tm._max_concurrent = 0  # drain 空转：不真正派发
    session = Session(id="s1", user_prompt="")
    tm.set_session(session)
    tm.set_runner(StubRunner(tm))
    t = Task(id="tsk_1", session_id="s1", status="ACTIVE")
    tm.register_task(t)
    return tm, t


async def test_assembly_failure_suspend_carries_assembly_failure_reason() -> None:
    """不可重试的装配失败 → TaskInterrupted.reason 必须是 assembly_failure，不是 run_crash。"""
    bus = _CapturingBus()
    tm, _t = _tm(bus)

    await tm._handle_task_failure(
        "tsk_1", error="boom", exc=_NonRetriable(), reason=InterruptReason.ASSEMBLY_FAILURE,
    )

    ev = next(e for e in bus.events if e.type == EventType.TASK_INTERRUPTED)
    assert ev.payload["reason"] == InterruptReason.ASSEMBLY_FAILURE


async def test_run_crash_suspend_still_carries_run_crash() -> None:
    """执行崩溃那条路不受影响（同一方法，传 RUN_CRASH 时原样透传）。"""
    bus = _CapturingBus()
    tm, _t = _tm(bus)

    await tm._handle_task_failure(
        "tsk_1", error="boom", exc=_NonRetriable(), reason=InterruptReason.RUN_CRASH,
    )

    ev = next(e for e in bus.events if e.type == EventType.TASK_INTERRUPTED)
    assert ev.payload["reason"] == InterruptReason.RUN_CRASH
