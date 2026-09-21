"""EventPersister / attach_persistence 的接线契约（spec 2026-08-29 §6.4）。

订阅策略（丢弃瞬态 delta）归 persister，不归 store——store 变成「让存什么就存什么」，
一致性测试才能直接测 append/read 往返而不被 store 悄悄吃掉测试事件。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import TRANSIENT_EVENT_TYPES, Event
from ctx_weft.core.control.snapshot_writer import attach_snapshotting
from ctx_weft.providers.events import (
    EventPersister,
    InMemoryEventStore,
    InProcessEventBus,
    )
from tests._snapshot_helpers import latest_snapshot


def _ev(type_: str, seq: int = 1, session: str = "s1") -> Event:
    return Event(
        id=f"evt_{seq:04d}",
        run_id="r1",
        sequence=seq,
        session_id=session,
        type=type_,
        timestamp=datetime.now(UTC),
    )


async def test_store_no_longer_accepts_event_bus():
    """自订阅已抽出成 EventPersister——store 不再自己订阅。"""
    with pytest.raises(TypeError):
        InMemoryEventStore(event_bus=InProcessEventBus())


async def test_store_append_no_longer_filters_transient():
    """行为变化（spec §6.4）：过滤是订阅策略，归 persister。"""
    transient = next(iter(TRANSIENT_EVENT_TYPES))
    store = InMemoryEventStore()
    await store.append(_ev(transient))
    assert len(await store.read_by_session("s1")) == 1


async def test_persister_drops_transient():
    transient = next(iter(TRANSIENT_EVENT_TYPES))
    store = InMemoryEventStore()
    p = EventPersister(store)
    await p.on_event(_ev(transient, 1))
    await p.on_event(_ev("SessionCreated", 2))
    stored = await store.read_by_session("s1")
    assert [e.type for e in stored] == ["SessionCreated"]


async def test_persister_subscribes_when_given_a_bus():
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    EventPersister(store, bus)
    await bus.emit(_ev("SessionCreated"))
    assert len(await store.read_by_session("s1")) == 1


async def test_persister_swallows_store_errors():
    """append 失败不得掀掉 loop——bus handler 在 emit 里内联执行。"""
    class _Boom:
        async def append(self, event):
            raise RuntimeError("db down")

    p = EventPersister(_Boom())
    await p.on_event(_ev("SessionCreated"))  # 不抛即通过


async def test_detach_stops_receiving():
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    p = EventPersister(store, bus)
    await p.detach()
    await bus.emit(_ev("SessionCreated"))
    assert await store.read_by_session("s1") == []


async def test_attach_persistence_wires_persister():
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    handle = attach_snapshotting(bus, store)
    await bus.emit(_ev("SessionCreated"))
    assert len(await store.read_by_session("s1")) == 1
    await handle.detach()
    await bus.emit(_ev("SessionFinished", 2))
    assert len(await store.read_by_session("s1")) == 1


# ── SnapshotWriter ──────────────────────────────────────────────────────────


async def test_snapshot_written_on_session_finished():
    from ctx_weft.core.control.snapshot_writer import SnapshotWriter

    bus, store = InProcessEventBus(), InMemoryEventStore()
    attach_snapshotting(bus, store, every_n=1)
    await bus.emit(_ev("SessionCreated", 1))
    await bus.emit(_ev("SessionFinished", 2))
    snap = await latest_snapshot(store, "s1")
    assert snap is not None
    assert snap.snapshot_reason == "session_finished"
    # 切面位置，不是「最后那条事件的 id」——后者从前是 position 旁边的第二个游标，而按 ID
    # 当游标正是 H2 的根因，所以新表示里刻意没有它。
    assert snap.last_commit_position == 2


async def test_snapshot_periodic_on_run_finished():
    bus, store = InProcessEventBus(), InMemoryEventStore()
    attach_snapshotting(bus, store, every_n=2)
    await bus.emit(_ev("SessionCreated", 1))
    await bus.emit(_ev("RunFinished", 2))       # n=2 达阈值
    snap = await latest_snapshot(store, "s1")
    assert snap is not None
    assert snap.snapshot_reason == "periodic"


async def test_snapshot_not_written_before_threshold():
    bus, store = InProcessEventBus(), InMemoryEventStore()
    attach_snapshotting(bus, store, every_n=50)
    await bus.emit(_ev("SessionCreated", 1))
    await bus.emit(_ev("RunFinished", 2))
    assert await latest_snapshot(store, "s1") is None


async def test_snapshot_sees_the_triggering_event():
    """顺序契约：persister 必须先落库，snapshot 才折得到这条事件（spec §6.5）。

    切面位置（`last_commit_position`）验不出订阅顺序对不对——它取的是写那一刻的
    `committed_head`，顺序反了它照样等于 2。真正对顺序敏感的是 `state_blob`：它是
    `rebuild_view` 的产物，若 SnapshotWriter 抢在 EventPersister 之前跑，store 里还只有
    1 条事件，`events_total` 就会是 1 而非 2。所以顺序契约靠 `events_total` 钉住，切面位置
    另验「这张快照切在哪」。
    """
    bus, store = InProcessEventBus(), InMemoryEventStore()
    attach_snapshotting(bus, store, every_n=1)
    await bus.emit(_ev("SessionCreated", 1))
    await bus.emit(_ev("SessionFinished", 2))
    snap = await latest_snapshot(store, "s1")
    assert snap.last_commit_position == 2
    assert snap.state_blob["events_total"] == 2


async def test_snapshot_writer_not_attached_by_default():
    bus, store = InProcessEventBus(), InMemoryEventStore()
    handle = attach_snapshotting(bus, store)          # snapshot_every_n 默认 0
    assert handle.snapshot_writer is None
    await bus.emit(_ev("SessionCreated", 1))
    await bus.emit(_ev("SessionFinished", 2))
    assert await latest_snapshot(store, "s1") is None


async def test_snapshot_writer_swallows_errors():
    from ctx_weft.core.control.snapshot_writer import SnapshotWriter

    class _Boom(InMemoryEventStore):
        async def save_snapshot(self, snapshot):
            raise RuntimeError("db down")

    bus, store = InProcessEventBus(), _Boom()
    attach_snapshotting(bus, store, every_n=1)
    await bus.emit(_ev("SessionCreated", 1))
    await bus.emit(_ev("SessionFinished", 2))   # 不抛即通过
