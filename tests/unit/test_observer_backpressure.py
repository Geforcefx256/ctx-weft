"""观察者背压（spec: event-commit；change reliability-wp3，方案 E-T10）。

观察者 = ``stream()`` 订阅者（SSE 宿主消费者）——它们只入队不 drain，队列满才真实
发生丢弃；handler 订阅者是内联 drain（emit 同步等它），不在此列。慢观察者不阻塞
提交与必要状态推进；丢弃可观测——EventsDropped 元事件（transient）携带 position，
可按 read_range 补读；其余观察者不受影响。
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from ctx_weft.core.events.commit_gate import CommitGate
from ctx_weft.protocols.events import Event, EventFilter, EventType
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus

_T0 = datetime(2026, 9, 11, tzinfo=UTC)


def _ev(n: int) -> Event:
    return Event(id=f"evt_{n:04d}", run_id="r1", sequence=n, session_id="s1",
                 type=EventType.RUN_STARTED, timestamp=_T0, payload={"n": n})


async def test_slow_stream_observer_drops_visible_and_backfillable():
    """慢 stream 观察者：队列满 → 丢弃 + EventsDropped（带 position）；提交与 required 不受阻。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus(queue_size=2)
    bus.attach_commit_gate(CommitGate(store))

    required_seen: list[str] = []

    async def _required(ev: Event) -> None:
        required_seen.append(ev.id)

    bus.subscribe(None, _required, provisional=True, required=True)

    slow_out: list[Event] = []
    unpark = asyncio.Event()
    got_drop = asyncio.Event()

    async def _slow_consume() -> None:
        # 真慢消费者：注册订阅、拿到第一条后挂起——队列随即积压、溢出、丢弃。
        async for ev in bus.stream(EventFilter()):
            slow_out.append(ev)
            if ev.type == EventType.EVENTS_DROPPED:
                got_drop.set()
                return
            await unpark.wait()          # 挂起：不再消费

    slow_task = asyncio.create_task(_slow_consume())
    await asyncio.sleep(0); await asyncio.sleep(0)   # 让订阅先注册
    handler_out: list[str] = []

    async def _handler(ev: Event) -> None:         # 内联 drain 的观察者：恒全量
        handler_out.append(ev.id)

    bus.subscribe(None, _handler)

    for n in range(1, 7):                          # 6 条事件冲垮 queue_size=2
        await bus.emit(_ev(n))

    # 提交与必要状态推进不受阻；内联观察者全量无缺口
    assert required_seen == [f"evt_{n:04d}" for n in range(1, 7)]
    assert handler_out == [f"evt_{n:04d}" for n in range(1, 7)]
    assert len(await store.read_by_session("s1")) == 6

    # 慢观察者：放行后 drain 自己的队列——看到 EventsDropped 通报（丢弃可观测）
    unpark.set()
    await asyncio.wait_for(got_drop.wait(), timeout=5)
    slow_task.cancel()

    drop = next(e for e in slow_out if e.type == EventType.EVENTS_DROPPED)
    assert drop.payload["position"] >= 1
    # 补读可用：按 position 从持久日志取回被丢事件
    backfill = await store.read_range(
        "s1", after_position=drop.payload["position"] - 1,
        through_position=drop.payload["position"])
    assert backfill and backfill[0].event.id.startswith("evt_")


async def test_drop_only_affects_the_full_queue():
    """一个 stream 观察者打满只影响它自己：handler 观察者全量收到（无缺口）。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus(queue_size=1)
    bus.attach_commit_gate(CommitGate(store))

    slow_stream = bus.stream(EventFilter())                 # 打满不消费
    handler_out: list[str] = []

    async def _handler(ev: Event) -> None:
        handler_out.append(ev.id)

    bus.subscribe(None, _handler)

    for n in range(1, 5):
        await bus.emit(_ev(n))

    assert handler_out == [f"evt_{n:04d}" for n in range(1, 5)]   # 无缺口、无阻塞


async def test_no_gate_falls_back_to_counter_only():
    """best_effort（无 gate）：丢弃回落纯计数旧行为——不产 EventsDropped。"""
    bus = InProcessEventBus(queue_size=1)
    out: list[Event] = []
    unpark = asyncio.Event()

    async def _slow() -> None:
        async for ev in bus.stream(EventFilter()):
            out.append(ev)
            await unpark.wait()

    task = asyncio.create_task(_slow())
    await asyncio.sleep(0); await asyncio.sleep(0)
    for n in range(1, 4):
        await bus.emit(_ev(n))
    unpark.set()
    await asyncio.sleep(0.05)          # 给消费者 drain 的时间
    task.cancel()
    # 旧行为：丢弃静默（只有 warning 日志与内部计数）——无元事件；且 queue_size=1、
    # 消费者在 unpark 前至多收一条，三条里必有无从补回的静默丢失（对照组：required
    # 时代同样的丢失会以 EventsDropped 通报并可按 position 补读）。
    assert not [e for e in out if e.type == EventType.EVENTS_DROPPED]
    assert {e.id for e in out} <= {"evt_0001", "evt_0002", "evt_0003"}
    assert len(out) < 3
