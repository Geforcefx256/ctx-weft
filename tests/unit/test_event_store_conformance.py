"""EventStore 协议一致性测试套（spec 2026-08-29 §8）。

**面向协议、不面向实现。** 每条用例只经 `protocols/events.py` 声明的方法操作 store，
不碰任何实现内部字段（`_events` / `_active` / `_snapshots`）。

═══ 接入点 ═══════════════════════════════════════════════════════════════════
新 store 接进来只需在 `_STORE_FACTORIES` 里加一行 + 一个工厂：

    _STORE_FACTORIES = {
        "in_memory": _make_in_memory,
        "sql": _make_sql,          # ← Task 10 加这一行
    }

工厂签名 `(tmp_path) -> AsyncIterator[EventStore]`（asynccontextmanager）。
═════════════════════════════════════════════════════════════════════════════

本套存在的直接理由：`list_active_session_ids` 的判据决定崩溃恢复捞哪些会话，两个实现
分叉的表现是「重启后某些会话不弹恢复」或「已结束的会话反复被恢复」——生产里极难归因。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.protocols.events import Event, EventStore, RunSnapshot
from ctx_weft.providers.events import InMemoryEventStore


@asynccontextmanager
async def _make_in_memory(tmp_path) -> AsyncIterator[EventStore]:
    yield InMemoryEventStore()


@asynccontextmanager
async def _make_sql(tmp_path) -> AsyncIterator[EventStore]:
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store

    async with open_sqlite_event_store(tmp_path / "events.db") as s:
        yield s


_STORE_FACTORIES = {
    "in_memory": _make_in_memory,
    "sql": _make_sql,
}


@pytest.fixture(params=sorted(_STORE_FACTORIES))
async def store(request, tmp_path) -> AsyncIterator[EventStore]:
    async with _STORE_FACTORIES[request.param](tmp_path) as s:
        yield s


_T0 = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def _ev(
    seq: int,
    type_: str = "RunStarted",
    *,
    session: str = "s1",
    payload: dict | None = None,
    **kw,
) -> Event:
    """事件 id 用 ULID 字典序等价的零填充串——便于断言里按序比对。"""
    return Event(
        id=f"evt_{seq:04d}",
        run_id=kw.pop("run_id", "r1"),
        sequence=seq,
        session_id=session,
        type=type_,
        timestamp=_T0 + timedelta(seconds=seq),
        payload=payload or {},
        **kw,
    )


# ── append / read 往返 ───────────────────────────────────────────────────────


async def test_append_read_roundtrip_preserves_every_field(store):
    """逐字段往返。schema_version 与 causation_id 尤其容易在 SQL 映射里被漏掉。"""
    ev = Event(
        id="evt_0001",
        run_id="r7",
        sequence=42,
        session_id="s1",
        type="RunStarted",
        timestamp=_T0,
        tenant_id="tenant-x",
        task_id="t1",
        agent_id="a1",
        payload={"k": "v", "n": 1},
        metadata={"m": True},
        causation_id="evt_0000",
        schema_version=3,
    )
    await store.append(ev)
    (got,) = await store.read_by_session("s1")
    for field in (
        "id", "run_id", "sequence", "session_id", "type", "tenant_id",
        "task_id", "agent_id", "payload", "metadata", "causation_id",
        "schema_version",
    ):
        assert getattr(got, field) == getattr(ev, field), field
    assert got.timestamp == _T0          # 时区保真：naive 回读会让这条恒 False


async def test_read_by_session_is_ordered(store):
    """read_by_session 按提交序（= position 序）返回。

    旧契约钉的是 id（ULID）排序——对乱序 append 做防御性归一。2026-09 起（change
    reliability-wp2，spec: event-log）有意改为提交序：append/append_batch 全在锁内按
    提交顺序入列，position 是它的记录；全量回放与快照+增量必须同一排序语义（可靠性
    方案 E5），按 id 排会让延迟提交的旧 ID 在全量回放里错位。
    """
    for seq in (3, 1, 2):
        await store.append(_ev(seq))
    assert [e.sequence for e in await store.read_by_session("s1")] == [3, 1, 2]


async def test_read_by_session_isolates_sessions(store):
    await store.append(_ev(1, session="s1"))
    await store.append(_ev(2, session="s2"))
    assert [e.session_id for e in await store.read_by_session("s1")] == ["s1"]


async def test_read_by_session_unknown_returns_empty(store):
    assert await store.read_by_session("nope") == []


# ── read_session_events_of_types ─────────────────────────────────────────────


async def test_read_of_types_filters(store):
    await store.append(_ev(1, "RunStarted"))
    await store.append(_ev(2, "RunFinished"))
    await store.append(_ev(3, "SessionFinished"))
    got = await store.read_session_events_of_types("s1", ("RunFinished", "SessionFinished"))
    assert [e.type for e in got] == ["RunFinished", "SessionFinished"]


async def test_read_of_types_empty_tuple(store):
    await store.append(_ev(1))
    assert await store.read_session_events_of_types("s1", ()) == []


# ── 快照 ─────────────────────────────────────────────────────────────────────


def _snap(sid: str = "s1", *, last_id: str, seq: int, reason: str = "periodic") -> RunSnapshot:
    return RunSnapshot(
        id=f"snp_{last_id}",
        run_id="r1",
        session_id=sid,
        last_event_id=last_id,
        last_event_sequence=seq,
        state_blob={"session_id": sid, "n": seq},
        snapshot_reason=reason,
        snapshot_at=_T0,
    )


async def test_snapshot_roundtrip(store):
    s = _snap(last_id="evt_0005", seq=5)
    await store.save_snapshot(s)
    got = await store.load_latest_snapshot("s1")
    assert got is not None
    for field in (
        "id", "run_id", "session_id", "last_event_id",
        "last_event_sequence", "state_blob", "snapshot_reason",
    ):
        assert getattr(got, field) == getattr(s, field), field


async def test_load_latest_snapshot_returns_newest(store):
    await store.save_snapshot(_snap(last_id="evt_0001", seq=1))
    await store.save_snapshot(_snap(last_id="evt_0009", seq=9))
    got = await store.load_latest_snapshot("s1")
    assert got.last_event_sequence == 9


async def test_load_latest_snapshot_none_when_absent(store):
    assert await store.load_latest_snapshot("s1") is None


async def test_load_latest_snapshot_by_created_at_not_write_order(store):
    """乱序写入时「最新」按 snapshot_at 取最大，不是「最后一次 save_snapshot」。

    协议口径（protocols/events.py::load_latest_snapshot）：按 snapshot_at 取
    最大、同值按 id 取最大。这里先写一条时间更晚的快照，再写一条时间更早的
    （模拟并发/重试补写导致的乱序），"最新"必须仍然是时间更晚的那条——
    两个实现（in_memory 的 last-write-wins、sql 的 ORDER BY created_at DESC）
    在写入顺序与时间顺序一致时看不出分歧，只有乱序写入才会暴露。
    """
    import dataclasses

    later = dataclasses.replace(
        _snap(last_id="evt_0009", seq=9), snapshot_at=_T0 + timedelta(seconds=10)
    )
    earlier = _snap(last_id="evt_0001", seq=1)  # snapshot_at=_T0，比 later 早

    await store.save_snapshot(later)
    await store.save_snapshot(earlier)  # 写入顺序在后，但时间更早

    got = await store.load_latest_snapshot("s1")
    assert got.last_event_sequence == 9, (
        "载入应仍是时间更晚（snapshot_at 更大）的快照，而不是写入顺序最后的那条"
    )


# ── list_active_session_ids：会话生命周期状态机 ──────────────────────────────


async def test_active_after_session_created(store):
    await store.append(_ev(1, "SessionCreated"))
    assert set(await store.list_active_session_ids()) == {"s1"}


async def test_inactive_after_session_finished(store):
    await store.append(_ev(1, "SessionCreated"))
    await store.append(_ev(2, "SessionFinished"))
    assert set(await store.list_active_session_ids()) == set()


async def test_reactivated_by_session_resumed(store):
    """多轮会话：每轮结束发 SessionFinished，下一条消息发 SessionResumed。
    不重新计入的话崩溃恢复会漏掉所有已对话过的会话。"""
    await store.append(_ev(1, "SessionCreated"))
    await store.append(_ev(2, "SessionFinished"))
    await store.append(_ev(3, "SessionResumed"))
    assert set(await store.list_active_session_ids()) == {"s1"}


async def test_inactive_after_terminal_status_changed(store):
    """参考实现的纯 SQL 判据完全忽略 SessionStatusChanged——这条钉住它。"""
    await store.append(_ev(1, "SessionCreated"))
    await store.append(_ev(2, "SessionStatusChanged", payload={"new_status": "FAILED"}))
    assert set(await store.list_active_session_ids()) == set()


async def test_non_terminal_status_changed_keeps_active(store):
    await store.append(_ev(1, "SessionCreated"))
    await store.append(_ev(2, "SessionStatusChanged", payload={"new_status": "RUNNING"}))
    assert set(await store.list_active_session_ids()) == {"s1"}


async def test_non_terminal_status_does_not_resurrect(store):
    """已 finished 的会话不该被一条非终态状态事件重新拉活。"""
    await store.append(_ev(1, "SessionCreated"))
    await store.append(_ev(2, "SessionFinished"))
    await store.append(_ev(3, "SessionStatusChanged", payload={"new_status": "RUNNING"}))
    assert set(await store.list_active_session_ids()) == set()


async def test_ordinary_event_does_not_resurrect(store):
    """普通事件既不激活也不停用——只有四类生命周期事件改变活跃性。"""
    await store.append(_ev(1, "SessionCreated"))
    await store.append(_ev(2, "SessionFinished"))
    await store.append(_ev(3, "RunStarted"))
    assert set(await store.list_active_session_ids()) == set()


async def test_active_sessions_are_independent(store):
    await store.append(_ev(1, "SessionCreated", session="s1"))
    await store.append(_ev(2, "SessionCreated", session="s2"))
    await store.append(_ev(3, "SessionFinished", session="s1"))
    assert set(await store.list_active_session_ids()) == {"s2"}
