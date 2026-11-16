"""未提交窗口：一轮对话在 LLM 开口之前不算发生（spec 2026-09-09）。

分两层测：

- **总线层**：`begin/commit/discard_provisional` 的分流本身——provisional 订阅者恒收
  全量（进程内状态机反映当下真实），其余订阅者等提交（事件日志只记算数的那些），
  提交按发生顺序补投，丢弃则一条都不出。
- **TaskManager 层**：窗口由它开合，丢弃前必须先发一条 `TASK_CANCELED` 把 agent 送回
  `idle`——那条事件自己也在窗口里、随缓冲一起被丢掉，所以**内存状态正确而日志干净**。
  顺序反了（先关窗再发）那条事件就会直接落盘，整套设计白做，故有一条专门的顺序断言。
"""

from __future__ import annotations

from datetime import UTC, datetime
from dataclasses import replace

import pytest

from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events.bus.in_process.bus import InProcessEventBus

_TS = datetime(2026, 9, 9, tzinfo=UTC)


def _ev(seq: int, type_: str, *, task_id: str | None = "tsk_1") -> Event:
    return Event(
        id=f"evt_{seq:04d}", run_id="r1", sequence=seq, session_id="s1",
        type=type_, timestamp=_TS, task_id=task_id, agent_id="agt_1", payload={},
    )


def _collector() -> tuple[list[Event], object]:
    seen: list[Event] = []

    async def handler(ev: Event) -> None:
        seen.append(ev)

    return seen, handler


# ── 总线层 ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["runtime", "loop.background_observe", "loop.recognize_intent"])
async def test_new_round_cannot_retract_an_already_started_background_run(origin):
    bus = InProcessEventBus()
    seen, handler = _collector()
    bus.subscribe(None, handler)
    start = replace(_ev(1, EventType.RUN_STARTED), origin=origin)
    await bus.emit(start)
    bus.begin_provisional("tsk_1")
    # The actor's reply belongs to the new window, the existing producer does not.
    reply = replace(_ev(2, EventType.LLM_TOKEN_STREAMED), run_id="actor", origin="loop.act")
    progress = replace(_ev(3, EventType.LLM_RESPONSE_FINISHED), origin=origin)
    finish = replace(_ev(4, EventType.RUN_FINISHED), origin=origin)
    await bus.emit(reply)
    await bus.emit(progress)
    await bus.emit(finish)
    bus.discard_provisional("tsk_1")
    assert seen == [start, progress, finish]
    assert not bus._background_runs


@pytest.mark.asyncio
async def test_durable_recap_started_during_new_round_survives_retraction():
    bus = InProcessEventBus()
    seen, handler = _collector()
    bus.subscribe(None, handler)
    bus.begin_provisional("tsk_1")
    start = replace(_ev(1, EventType.RUN_STARTED), origin="loop.background_observe")
    finish = replace(_ev(2, EventType.RUN_FINISHED), origin="loop.background_observe")
    await bus.emit(start)
    await bus.emit(finish)
    bus.discard_provisional("tsk_1")
    assert seen == [start, finish]
    assert not bus._background_runs


@pytest.mark.asyncio
async def test_window_holds_events_from_ordinary_subscribers_but_not_provisional_ones():
    bus = InProcessEventBus()
    inside, h_inside = _collector()
    outside, h_outside = _collector()
    bus.subscribe(None, h_inside, provisional=True)   # ALM 那一类
    bus.subscribe(None, h_outside)                    # 落盘 / host 那一类

    bus.begin_provisional("tsk_1")
    await bus.emit(_ev(1, EventType.TASK_CREATED))
    await bus.emit(_ev(2, EventType.TASK_STARTED))

    assert [e.type for e in inside] == [EventType.TASK_CREATED, EventType.TASK_STARTED], (
        "进程内状态机必须立刻看到——看不到就意味着 agent 停在 idle，"
        "pause_agent 会拒掉这个窗口里的暂停请求"
    )
    assert outside == [], "窗口里的事件不得落盘、不得到达 host"


@pytest.mark.asyncio
async def test_commit_replays_in_order_and_does_not_double_deliver():
    bus = InProcessEventBus()
    inside, h_inside = _collector()
    outside, h_outside = _collector()
    bus.subscribe(None, h_inside, provisional=True)
    bus.subscribe(None, h_outside)

    bus.begin_provisional("tsk_1")
    await bus.emit(_ev(1, EventType.TASK_CREATED))
    await bus.emit(_ev(2, EventType.TASK_STARTED))
    await bus.commit_provisional("tsk_1")

    assert [e.type for e in outside] == [EventType.TASK_CREATED, EventType.TASK_STARTED], (
        "顺序必须是发生顺序：TASK_CREATED 是下游 reducer 与 host 建 task 键的那一条，"
        "它若不在最前面，后面每一条都落在一个不存在的键上"
    )
    assert len(inside) == 2, "provisional 订阅者在 emit 时已收过，提交不得再投一遍"


@pytest.mark.asyncio
async def test_discard_drops_everything_and_closes_the_window():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)

    bus.begin_provisional("tsk_1")
    await bus.emit(_ev(1, EventType.TASK_CREATED))
    bus.discard_provisional("tsk_1")
    assert outside == []

    # 窗口已关：此后的事件直接出去（这正是「丢弃必须是最后一步」的原因）
    await bus.emit(_ev(2, EventType.TASK_FINISHED))
    assert [e.type for e in outside] == [EventType.TASK_FINISHED]


@pytest.mark.asyncio
async def test_window_only_holds_its_own_task():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)

    bus.begin_provisional("tsk_1")
    await bus.emit(_ev(1, EventType.TASK_CREATED, task_id="tsk_other"))
    await bus.emit(_ev(2, EventType.SESSION_CREATED, task_id=None))

    assert [e.type for e in outside] == [EventType.TASK_CREATED, EventType.SESSION_CREATED], (
        "闸按 task_id 定：别的 task 与会话级事件都不受影响"
    )


@pytest.mark.asyncio
async def test_begin_is_idempotent_and_keeps_the_buffer():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)

    bus.begin_provisional("tsk_1")
    await bus.emit(_ev(1, EventType.TASK_CREATED))
    bus.begin_provisional("tsk_1")            # retry 重排会重进同一条路径
    await bus.emit(_ev(2, EventType.TASK_STARTED))
    await bus.commit_provisional("tsk_1")

    assert [e.type for e in outside] == [EventType.TASK_CREATED, EventType.TASK_STARTED], (
        "重复开窗把已攒的缓冲清掉了 —— TASK_CREATED 会永久丢失"
    )


@pytest.mark.asyncio
async def test_commit_and_discard_on_an_unopened_window_are_noops():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)
    await bus.commit_provisional("nope")
    bus.discard_provisional("nope")
    await bus.emit(_ev(1, EventType.TASK_CREATED))
    assert [e.type for e in outside] == [EventType.TASK_CREATED]


# ── TaskManager 层 ────────────────────────────────────────────────────────────


def _task_manager(bus):
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.orchestrator.task.manager import TaskManager

    tm = TaskManager(session_id="s1", event_bus=bus)
    tm._session = Session(id="s1", tenant_id="default", user_prompt="hi")
    return tm


def _task(task_id: str = "tsk_1"):
    from ctx_weft.core.models.task import Task

    return Task(id=task_id, session_id="s1", status="ACTIVE",
                assigned_agent_id="agt_1", creator_agent_id="agt_1",
                description="hi", user_prompt="hi")


@pytest.mark.asyncio
async def test_push_task_provisional_holds_task_created():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)
    tm = _task_manager(bus)

    await tm.push_task(_task(), provisional=True)
    assert outside == [], "provisional push 的 TASK_CREATED 不得直接落盘"
    assert tm.is_round_open("tsk_1")

    await tm.commit_round("tsk_1")
    # `RoundCommitted` 必须排在补投的那批**前面**：host 据它 flush 攒着的用户消息帧，
    # 那一帧要落在这一轮的 task/run 帧之前——用户先说话，agent 才开跑。
    assert [e.type for e in outside] == [
        EventType.ROUND_COMMITTED, EventType.TASK_CREATED,
    ]
    assert not tm.is_round_open("tsk_1")


@pytest.mark.asyncio
async def test_ordinary_push_task_is_unaffected():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)
    tm = _task_manager(bus)

    await tm.push_task(_task("tsk_child"))     # 委派子任务走的这条：创建即落盘
    assert [e.type for e in outside] == [EventType.TASK_CREATED]
    assert not tm.is_round_open("tsk_child")


@pytest.mark.asyncio
async def test_discard_settles_the_agent_in_memory_but_writes_nothing_to_the_log():
    """丢弃的核心不变式：`TASK_CANCELED` 必须在**关窗之前**发。

    它是把 agent 从 `running` 送回 `idle` 的那条转移（ALM 映射成 `AgentInput.SETTLED`），
    进程内订阅者要收到；而它自己也在窗口里，随缓冲一起被丢掉，所以日志上什么都不留。
    顺序反了就会有一条 `TASK_CANCELED` 直接落盘，指向一个从未被创建过的 task。
    """
    bus = InProcessEventBus()
    inside, h_inside = _collector()
    outside, h_outside = _collector()
    bus.subscribe(None, h_inside, provisional=True)
    bus.subscribe(None, h_outside)
    tm = _task_manager(bus)

    await tm.push_task(_task(), provisional=True)
    await tm.discard_round("tsk_1")

    assert [e.type for e in inside] == [
        EventType.TASK_CREATED, EventType.TASK_CANCELED, EventType.ROUND_DISCARDED,
    ], "进程内状态机要看到 TASK_CANCELED，否则 agent 永远停在 running"
    # 唯一出去的是那条会话级的丢弃信号：它不带 task_id（绕开闸），host 据它把攒着的
    # 用户消息帧丢掉。这一轮自己的事件一条都没出去。
    assert [e.type for e in outside] == [EventType.ROUND_DISCARDED]
    assert outside[0].task_id in (None, ""), "丢弃信号带上 task_id 就会被自己那道闸挡住"
    assert tm.get_task("tsk_1") is None, "被丢弃的 task 不得留在登记里"
    assert not tm.is_round_open("tsk_1")


@pytest.mark.asyncio
async def test_discard_is_idempotent():
    bus = InProcessEventBus()
    outside, h_outside = _collector()
    bus.subscribe(None, h_outside)
    tm = _task_manager(bus)

    await tm.push_task(_task(), provisional=True)
    await tm.discard_round("tsk_1")
    await tm.discard_round("tsk_1")      # 二次调用不得再发一条 TASK_CANCELED
    assert [e.type for e in outside] == [EventType.ROUND_DISCARDED], (
        "第二次 discard 必须是彻底的 no-op —— 多发一条信号会让 host 把下一轮的帧丢掉"
    )
