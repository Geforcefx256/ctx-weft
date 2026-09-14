"""EventStore 有序提交扩展的 conformance（spec: event-log；change reliability-wp2）。

参数化内存 / SQLite 跑同一套用例，钉住八条 requirement：位置唯一单调、批次原子、
batch_id 幂等与冲突、append 兼容、跨会话隔离、双连接争用、按位置读取、（迁移单测另
文件）。双连接争用是**数据库级并发**（两个独立 session factory），不以单协程顺序调用
代替——内存参数共享同一实例（进程内锁天然串行，design D5 的妥协只承诺正确性）。
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import (
    CommitReceipt,
    Event,
    EventConflictError,
    EventStore,
    StoredEvent,
    supports_ordered_commit,
)
from ctx_weft.providers.events.store.in_memory.store import InMemoryEventStore
from ctx_weft.providers.events.store.sql.store import SqlEventStore, open_sqlite_event_store

_T0 = datetime(2026, 9, 11, tzinfo=UTC)


def _ev(n: int, *, session: str = "s1", type_: str = "RunStarted", payload=None) -> Event:
    return Event(
        id=f"evt_{n:04d}", run_id="r1", sequence=n, session_id=session, type=type_,
        timestamp=_T0, payload=payload if payload is not None else {"n": n},
    )


def _ids(receipt: CommitReceipt) -> list[str]:
    return [r.event.id for r in receipt.records]


def _positions(receipt: CommitReceipt) -> list[int]:
    return [r.position for r in receipt.records]


@pytest.fixture(params=["in_memory", "sqlite"])
async def store(request, tmp_path):
    if request.param == "in_memory":
        yield InMemoryEventStore()
    else:
        async with open_sqlite_event_store(tmp_path / "events.sqlite") as s:
            yield s


@pytest.fixture(params=["in_memory", "sqlite"])
async def store_pair(request, tmp_path):
    """两个独立实例（SQL = 两个连接池，数据库级并发；内存 = 同实例×2）。"""
    if request.param == "in_memory":
        s = InMemoryEventStore()
        yield s, s
    else:
        from ctx_weft.providers._sqlalchemy import make_session_factory
        from sqlalchemy.ext.asyncio import async_sessionmaker

        url = f"sqlite+aiosqlite:///{tmp_path / 'events.sqlite'}"
        engine, f1 = make_session_factory(url, connect_args={"timeout": 15})
        _, f2 = make_session_factory(url, connect_args={"timeout": 15})
        try:
            from ctx_weft.providers.events.store.sql.models import Base
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            yield SqlEventStore(f1), SqlEventStore(f2)
        finally:
            await engine.dispose()


# ── 位置唯一单调 + committed_head ─────────────────────────────────────────────


async def test_positions_increase_and_head_tracks(store):
    r1 = await store.append_batch("s1", "b1", [_ev(1), _ev(2)])
    r2 = await store.append_batch("s1", "b2", [_ev(3)])
    assert _positions(r1) == [1, 2] and _positions(r2) == [3]
    assert min(_positions(r2)) > max(_positions(r1))
    assert await store.committed_head("s1") == 3
    assert await store.committed_head("no-such") == 0


async def test_event_id_cannot_span_sessions(store):
    await store.append_batch("s1", "b1", [_ev(1)])
    with pytest.raises(EventConflictError):
        await store.append_batch("s2", "b2", [_ev(1, session="s2")])
    assert await store.committed_head("s2") == 0


# ── 批次原子 ─────────────────────────────────────────────────────────────────


async def test_batch_atomicity_kth_failure_leaves_no_trace(store):
    """批内 id 重复（第 2 条撞第 1 条主键）→ 整批不存在；同 batch_id 干净重试成功。

    失败批次不留任何痕迹（无事件行、head 不动、批表无行）——所以用同一 batch_id 重发
    修正后的内容是**全新提交**而非内容冲突（spec 场景「同 batch_id 重试成功后恰好落库
    N 条」的确定性实现口径）。
    """
    with pytest.raises(EventConflictError):
        await store.append_batch("s1", "bad", [_ev(1), _ev(1)])  # 同 id 第二条撞主键
    assert await store.read_by_session("s1") == []
    assert await store.committed_head("s1") == 0
    receipt = await store.append_batch("s1", "bad", [_ev(1), _ev(2)])  # 同 batch_id 重试
    assert _ids(receipt) == ["evt_0001", "evt_0002"]
    assert len(await store.read_by_session("s1")) == 2


async def test_batch_must_share_session(store):
    with pytest.raises(ValueError):
        await store.append_batch("s1", "b1", [_ev(1, session="s2")])
    with pytest.raises(ValueError):
        await store.append_batch("s1", "b-empty", [])


# ── batch_id 幂等与冲突 ───────────────────────────────────────────────────────


async def test_idempotent_retry_returns_original_receipt(store):
    r1 = await store.append_batch("s1", "b1", [_ev(1), _ev(2)])
    r2 = await store.append_batch("s1", "b1", [_ev(1), _ev(2)])
    assert r1 == r2  # frozen dataclass：含 position 的全等
    assert len(await store.read_by_session("s1")) == 2
    assert await store.committed_head("s1") == 2


async def test_same_batch_different_content_conflicts(store):
    await store.append_batch("s1", "b1", [_ev(1)])
    with pytest.raises(EventConflictError):
        await store.append_batch("s1", "b1", [_ev(1), _ev(2)])
    with pytest.raises(EventConflictError):  # 同 id 异 payload 也算内容冲突
        await store.append_batch("s1", "b1", [_ev(1, payload={"n": 999})])
    assert len(await store.read_by_session("s1")) == 1


# ── append 兼容 ──────────────────────────────────────────────────────────────


async def test_append_is_single_event_batch_and_mixes(store):
    """逐条 append 与批次提交共存，按提交顺序获得递增 position。

    旧契约行为差异（钉基线用，wp2-1.2）：改道前 in-memory 对重复 id 双存、SQL 抛
    IntegrityError——改道后统一为幂等（同 id 同内容 no-op 返回原 receipt）。
    """
    await store.append(_ev(1))
    await store.append(_ev(2))
    r = await store.append_batch("s1", "b3", [_ev(3), _ev(4)])
    assert _positions(r) == [3, 4]
    assert [e.sequence for e in await store.read_by_session("s1")] == [1, 2, 3, 4]
    # append 的幂等键 = event.id：重复 append 同一事件是 no-op
    await store.append(_ev(1))
    assert len(await store.read_by_session("s1")) == 4


# ── 跨会话隔离 + 双连接争用 ──────────────────────────────────────────────────


async def test_cross_session_isolation(store_pair):
    a, b = store_pair
    ra = await a.append_batch("s1", "b1", [_ev(1)])
    rb = await b.append_batch("s2", "b2", [_ev(2, session="s2")])
    assert _positions(ra) == [1] and _positions(rb) == [1]
    assert await a.committed_head("s1") == 1
    assert await b.committed_head("s2") == 1


async def test_dual_connection_same_session_contention(store_pair):
    """两个独立连接并发提交同会话批次：head 串行化，position 无重复、单调无交错。"""
    a, b = store_pair
    batches = [
        (a, "ba", [_ev(i) for i in range(1, 4)]),
        (b, "bb", [_ev(i) for i in range(4, 7)]),
        (a, "bc", [_ev(i) for i in range(7, 10)]),
        (b, "bd", [_ev(i) for i in range(10, 13)]),
    ]
    receipts = await asyncio.gather(*(
        s.append_batch("s1", bid, evs) for s, bid, evs in batches))
    all_positions = sorted(p for r in receipts for p in _positions(r))
    assert all_positions == list(range(1, 13)), f"duplicate/gap in positions: {all_positions}"
    heads = [await a.committed_head("s1"), await b.committed_head("s1")]
    assert heads == [12, 12]
    # 批内 position 连续（整批一次性分配，不被并发切割）
    for r in receipts:
        ps = _positions(r)
        assert ps == list(range(ps[0], ps[0] + len(ps)))


# ── 按位置读取 ───────────────────────────────────────────────────────────────


async def test_read_range_bounds(store):
    for n in range(1, 6):
        await store.append(_ev(n))
    got = await store.read_range("s1", after_position=2, through_position=4)
    assert [se.position for se in got] == [3, 4]
    assert [se.event.id for se in got] == ["evt_0003", "evt_0004"]
    all_ = await store.read_range("s1")
    assert [se.position for se in all_] == [1, 2, 3, 4, 5]
    assert isinstance(all_[0], StoredEvent)


# ── 协议形状（wp2-1.1）──────────────────────────────────────────────────────


def test_protocol_types_shape():
    """frozen dataclass + 异常可导入（协议面进 protocols/events.py 的形状锚）。"""
    assert CommitReceipt(batch_id="b", records=()).batch_id == "b"
    try:
        raise EventConflictError("x")
    except EventConflictError:
        pass
    assert supports_ordered_commit(InMemoryEventStore())


def test_single_event_protocol_no_parallel_ordered_protocol():
    """有序提交并进 EventStore，不另立第二个 Protocol（避免两套 store 契约）。"""
    import ctx_weft.protocols.events as mod
    assert not hasattr(mod, "OrderedEventStore")
    for name in ("append_batch", "read_range", "committed_head"):
        assert hasattr(mod.EventStore, name)


def test_supports_ordered_commit_rejects_inherited_stubs():
    """只继承协议桩的 store 判为不支持——hasattr 会误判，这正是换掉它的理由。

    required 模式的构造期检查依赖这条：误判会把错误推迟到第一次 emit 才炸。
    """
    class StubStore(EventStore):
        async def append(self, event): ...
        async def read_by_session(self, session_id): return []

    stub = StubStore()
    assert hasattr(stub, "append_batch")          # 继承自协议，恒为真
    assert not supports_ordered_commit(stub)      # 但不是真实现


def test_supports_ordered_commit_accepts_duck_typed_store():
    """不继承协议、自带三个方法的鸭子类型 store 判为支持。"""
    class DuckStore:
        async def append_batch(self, session_id, batch_id, events): ...
        async def read_range(self, session_id, **kw): return []
        async def committed_head(self, session_id): return 0

    assert supports_ordered_commit(DuckStore())


def test_supports_ordered_commit_requires_all_three():
    """部分实现不算支持——半可用的提交门比不支持更难诊断。"""
    class PartialStore:
        async def append_batch(self, session_id, batch_id, events): ...

    assert not supports_ordered_commit(PartialStore())
