"""两个通用读原语的 conformance：`read_last_of_type` 与 `read_range(exclude_types=)`。

它们是「把状态快照做成一种事件」的地基（那个类型与它的发射者在下一步加）。**刻意不带任何特定
类型的语义**——调用方自己填要取什么、排除什么。store 因此不必懂「什么是快照」：那是 core 的
词表，不是存储契约的一部分。

参数化跑遍两个内置 store：两边口径不一致会让「本机能跑、线上恢复不对」，而这类分歧只有把
同一套断言喂给两个实现才抓得到。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import Event, EventType, StoredEvent

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID = "s1"


def _ev(n: int, type_: str) -> Event:
    return Event(
        id=f"evt_{n:08d}", run_id="r1", sequence=n, session_id=_SID, type=type_,
        timestamp=_T0, tenant_id="acme", task_id="t1", agent_id="ag1",
        payload={"n": n},
    )


@pytest.fixture(params=["in_memory", "sqlite"])
async def store(request, tmp_path):
    if request.param == "in_memory":
        from ctx_weft.providers.events import InMemoryEventStore

        yield InMemoryEventStore()
        return
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store

    async with open_sqlite_event_store(tmp_path / "events.sqlite") as s:
        yield s


async def _seed(store) -> None:
    """RunFinished ×3 中间夹两条同类事件——模拟「某种要单独取最后一条 / 要排除的类型」。"""
    batch = [
        _ev(1, EventType.RUN_FINISHED),
        _ev(2, EventType.MEMORY_COMPACTED),
        _ev(3, EventType.RUN_FINISHED),
        _ev(4, EventType.MEMORY_COMPACTED),
        _ev(5, EventType.RUN_FINISHED),
    ]
    await store.append_batch(_SID, "b1", batch)


# ── read_last_of_type ─────────────────────────────────────────────────────────


async def test_last_of_type_returns_the_highest_position(store) -> None:
    """「最后一条」= position 最大，与 `read_range` / `committed_head` 同一个序。

    这条将替掉快照专属的那套「最新」口径（`snapshot_at` 最大、相同则 `id` 最大，
    因为写入序 ≠ 时间序）。同一个序意味着不存在第二种「最新」，也就不存在两种口径不一致。
    """
    await _seed(store)

    got = await store.read_last_of_type(_SID, EventType.MEMORY_COMPACTED)

    assert isinstance(got, StoredEvent)
    assert got.event.id == "evt_00000004", "该拿第二张（position 更大的）"
    assert got.position == 4


async def test_last_of_type_reads_exactly_one(store) -> None:
    """只读一条——这是它存在的理由。

    用 `read_session_events_of_types` 会把全部同类事件连载荷一起捞回来。对状态快照那种
    一条一条攒下来的类型，那就是 O(快照张数) 的浪费，而要的只是最后一张。
    """
    await _seed(store)

    all_of_type = await store.read_session_events_of_types(
        _SID, (EventType.MEMORY_COMPACTED,))
    assert len(all_of_type) == 2, "前提：日志里有两条"

    got = await store.read_last_of_type(_SID, EventType.MEMORY_COMPACTED)
    assert got is not None and got.event.id == all_of_type[-1].id


async def test_last_of_type_is_none_when_absent(store) -> None:
    """没有这种事件 → None，不是空列表、不抛。"""
    await _seed(store)

    assert await store.read_last_of_type(_SID, EventType.FAILURE_THRESHOLD_HIT) is None
    assert await store.read_last_of_type("s_other", EventType.MEMORY_COMPACTED) is None


# ── read_range(exclude_types=) ────────────────────────────────────────────────


async def test_exclude_types_skips_them(store) -> None:
    """排除掉的不返回，其余原样、仍按 position 升序。

    全量重放将用它排掉状态快照事件：那些事件的存在是为了**省**重放，把它们读回来反而更贵。
    """
    await _seed(store)

    got = await store.read_range(
        _SID, exclude_types=(EventType.MEMORY_COMPACTED,))

    assert [se.event.type for se in got] == [str(EventType.RUN_FINISHED)] * 3
    assert [se.position for se in got] == [1, 3, 5], "position 原样，不重编号"


async def test_exclude_types_default_is_no_filtering(store) -> None:
    """不传 → 行为与从前逐字一致（全取）。"""
    await _seed(store)

    got = await store.read_range(_SID)

    assert len(got) == 5


async def test_exclude_types_composes_with_the_position_window(store) -> None:
    """与 position 窗口叠加，不互相干扰。"""
    await _seed(store)

    got = await store.read_range(
        _SID, after_position=2, through_position=5,
        exclude_types=(EventType.MEMORY_COMPACTED,))

    assert [se.position for se in got] == [3, 5]


async def test_head_is_unaffected_by_exclusion(store) -> None:
    """`committed_head` 不受排除影响——它是提交位点，不是某次查询的结果。

    如果 head 会随排除变化，快照边界就会随「这次读排除了什么」而漂移，而它必须是一个
    与读法无关的事实。
    """
    await _seed(store)

    assert await store.committed_head(_SID) == 5
