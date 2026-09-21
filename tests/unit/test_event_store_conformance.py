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

本套存在的直接理由：两个 store 实现必须在同一份契约上逐字等价。分叉的表现（某条读法在
in_memory 上对、在 SQL 上差一条，或排序口径不同）只会在生产里以「恢复出来的世界不一样」
浮现，极难归因。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.protocols.events import Event, EventStore
from ctx_weft.providers.events import InMemoryEventStore
from tests._event_helpers import all_events, append_one


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
    await append_one(store, ev)
    (got,) = await all_events(store, "s1")
    for field in (
        "id", "run_id", "sequence", "session_id", "type", "tenant_id",
        "task_id", "agent_id", "payload", "metadata", "causation_id",
        "schema_version",
    ):
        assert getattr(got, field) == getattr(ev, field), field
    assert got.timestamp == _T0          # 时区保真：naive 回读会让这条恒 False


async def test_reading_a_session_is_ordered_by_commit(store):
    """整条会话的读（`read_range`）按提交序（= position 序）返回。

    2026-09-21 起这条测的是 `read_range`：`read_by_session` 整个从协议删了（src 零调用者，
    而它是「整条会话读成一个 list」那个形状）。排序口径没变，所以这条测试的内容没变。

    旧契约钉的是 id（ULID）排序——对乱序 append 做防御性归一。2026-09 起（change
    reliability-wp2，spec: event-log）有意改为提交序：append/append_batch 全在锁内按
    提交顺序入列，position 是它的记录；全量回放与快照+增量必须同一排序语义（可靠性
    方案 E5），按 id 排会让延迟提交的旧 ID 在全量回放里错位。
    """
    for seq in (3, 1, 2):
        await append_one(store, _ev(seq))
    assert [e.sequence for e in await all_events(store, "s1")] == [3, 1, 2]


async def test_reading_a_session_isolates_sessions(store):
    await append_one(store, _ev(1, session="s1"))
    await append_one(store, _ev(2, session="s2"))
    assert [e.session_id for e in await all_events(store, "s1")] == ["s1"]


async def test_reading_an_unknown_session_returns_empty(store):
    assert await all_events(store, "nope") == []


# ── read_range(include_types=) ───────────────────────────────────────────────
#
# 从前是独立的 `read_session_events_of_types`。2026-09-21 并进区间读——同一个查询换过滤
# 条件，而它缺的那一样（位置区间）正是它的问题所在（见协议 `read_range` 的 docstring）。


async def test_read_of_types_filters(store):
    await append_one(store, _ev(1, "RunStarted"))
    await append_one(store, _ev(2, "RunFinished"))
    await append_one(store, _ev(3, "SessionFinished"))
    got = [se.event for se in await store.read_range(
        "s1", include_types=("RunFinished", "SessionFinished"))]
    assert [e.type for e in got] == ["RunFinished", "SessionFinished"]


async def test_read_of_types_empty_tuple(store):
    await append_one(store, _ev(1))
    await append_one(store, _ev(2, "RunFinished"))
    # 原语层面 `include_types=()` = **不启用这个过滤器**（与 `exclude_types=()` 对称——原语
    # 该是正交的），所以它读回全部两条。
    assert len(await store.read_range("s1", include_types=())) == 2
    # 而 core 的入口 `load_events_of_types` 在这里**反着来**：空类型 → 空结果。那不是不一致，
    # 是安全阀放在了该放的那一层——调用方的意图是「按类型收窄」，类型列表却空了，用原语的
    # 正交语义就会静默退化成整条会话读。
    from ctx_weft.core.control.reducers import load_events_of_types
    assert await load_events_of_types(store, "s1", ()) == []


# ── 快照 ──────────────────────────────────────────────────────────────────────
#
# 这里曾有 4 条：`save_snapshot` 往返、「最新那张」、无快照返回 None、以及「最新按
# `created_at` 而不是写入顺序」。那套 API 与 `event_snapshots` 表在 2026-09-20 随「快照变成
# 日志里的一条 `StateSnapshot` 事件」一起删除，所以它们的**主题消失了**，不是覆盖变少：
#
#   · 「取最新那一条」现在是通用原语 `read_last_of_type`，覆盖在
#     `tests/unit/test_read_primitives_conformance.py`（同样参数化跑遍两个 store）；
#   · 那条「最新按 created_at 而不是写入顺序」的用例是为「写入序 ≠ 时间序」这个麻烦而写的
#     ——现在「最新」= position 最大，那个麻烦本身不存在了；
#   · 载荷编解码与恢复接合在 `tests/unit/test_snapshot_as_event.py`。



# 这里从前有 8 条会话活跃性用例（`list_active_session_ids` 的判据：SessionCreated 置活、
# SessionFinished/终态 SessionStatusChanged 置停、SessionResumed 复活、普通事件不动、
# 多会话互不干扰）。它们连同被测方法一起于 2026-09-21 删除。
#
# **主题消失了，不是覆盖变少**：那个判据不该存在于 core。「有哪些会话」是 host 自己的
# 数据（它建的会话、它的会话表和状态列），core 从事件流反推是职责倒置——而且推得更差，
# 两条 discard 依据（`SessionFinished` / `SessionStatusChanged`）在 src 下没有任何 emit
# 调用点，判据恒真，返回的实际是「这个库里出现过的全部会话」。这 8 条用例钉的正是一台
# 永远走不到 discard 分支的状态机。
#
# host 要按会话装填，拿自己的清单逐条调 `Runtime.rebuild_hitl` / `rebuild_session`。
