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
from tests._event_helpers import all_events, append_one

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
    assert await all_events(store, "s1") == []
    assert await store.committed_head("s1") == 0
    receipt = await store.append_batch("s1", "bad", [_ev(1), _ev(2)])  # 同 batch_id 重试
    assert _ids(receipt) == ["evt_0001", "evt_0002"]
    assert len(await all_events(store, "s1")) == 2


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
    assert len(await all_events(store, "s1")) == 2
    assert await store.committed_head("s1") == 2


async def test_same_batch_different_content_conflicts(store):
    await store.append_batch("s1", "b1", [_ev(1)])
    with pytest.raises(EventConflictError):
        await store.append_batch("s1", "b1", [_ev(1), _ev(2)])
    with pytest.raises(EventConflictError):  # 同 id 异 payload 也算内容冲突
        await store.append_batch("s1", "b1", [_ev(1, payload={"n": 999})])
    assert len(await all_events(store, "s1")) == 1


# ── append 兼容 ──────────────────────────────────────────────────────────────


async def test_append_is_single_event_batch_and_mixes(store):
    """逐条 append 与批次提交共存，按提交顺序获得递增 position。

    旧契约行为差异（钉基线用，wp2-1.2）：改道前 in-memory 对重复 id 双存、SQL 抛
    IntegrityError——改道后统一为幂等（同 id 同内容 no-op 返回原 receipt）。
    """
    await append_one(store, _ev(1))
    await append_one(store, _ev(2))
    r = await store.append_batch("s1", "b3", [_ev(3), _ev(4)])
    assert _positions(r) == [3, 4]
    assert [e.sequence for e in await all_events(store, "s1")] == [1, 2, 3, 4]
    # append 的幂等键 = event.id：重复 append 同一事件是 no-op
    await append_one(store, _ev(1))
    assert len(await all_events(store, "s1")) == 4


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
        await append_one(store, _ev(n))
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
    # 按事件 ID 取增量的 API 已彻底移除（H2 的作案工具）
    assert not hasattr(mod.EventStore, "read_after")
    for name in ("append_batch", "read_range", "committed_head"):
        assert hasattr(mod.EventStore, name)


def test_ordered_commit_methods_are_mandatory():
    """协议面就是这四个，全是 @abstractmethod——没有可选扩展这一档了。

    2026-09-19~21 一路收：`save_snapshot` / `load_latest_snapshot`（快照变成事件）、
    `replay`（只由必需原语组合 → 归调用方）、`read_by_session`（整条会话读成一个 list）、
    `list_active_session_ids`（host 自己的数据，core 不该回答）、
    `read_session_events_of_types`（并进 `read_range`）、`append`（= `append_batch` 加一条
    属于 core 的 batch_id 策略）。

    判据用 `==` 而不是 `>=`：多出来一个方法同样该让这条红。协议面每多一个必需方法，每个自定义
    store 就多一份必须写对的东西，而本仓的记录是——每一个"顺手加上"的方法后来都成了要清的
    形状，或者生出一道能力探测的门。
    """
    assert EventStore.__abstractmethods__ == {
        "append_batch", "read_range", "committed_head",
        # 它后来也升为必需：恢复第一步无条件要取最新快照，做成可选就又要在 core 里探一次
        # 能力——那个坏设计在这个仓生出过三次「两分支 + 两失败形态」。
        "read_last_of_type"}
    # 可选扩展仍是可选：不在 abstractmethods 里
    for name in ("save_snapshot", "load_latest_snapshot"):
        assert name not in EventStore.__abstractmethods__
    # `list_active_session_ids` 2026-09-21 整个从协议删了——**不是**降为可选，同
    # `read_by_session`（见下）。「有哪些会话」是 host 自己的数据（它建的会话、它的会话表
    # 和状态列），core 从事件流反推是职责倒置，而且推得更差：判据的两条 discard 依据
    # （`SessionFinished` / `SessionStatusChanged`）在 src 下没有 emit 调用点，恒真。
    # 钉「协议面上没有它」而不是「不在 abstractmethods 里」：后者在它被降为可选扩展时
    # 也会绿，而那正是不该发生的形态。
    assert not hasattr(EventStore, "list_active_session_ids"), (
        "list_active_session_ids 回来了：会话清单是 host 的数据，不该由 core 从事件流推")
    # `read_by_session`（整条会话读成一个 list）2026-09-21 整个从协议删了——**不是**降为
    # 可选。所以钉「协议面上没有它」，而不是「它不在 abstractmethods 里」：后者在它被降为
    # 可选扩展时也会绿，而那正是不该发生的形态。
    assert not hasattr(EventStore, "read_by_session"), (
        "read_by_session 回来了：它是本仓花很长时间清掉的那个形状，且留着会让「断言它被调"
        "用 0 次」那类守卫重新变得空洞")


def test_subclass_missing_ordered_commit_cannot_instantiate():
    """第一层强制：显式继承 EventStore 而不实现三方法 → ABC 在实例化时就拒绝。"""
    class StubStore(EventStore):
        async def append(self, event): ...

    with pytest.raises(TypeError, match="append_batch"):
        StubStore()


def test_supports_ordered_commit_accepts_duck_typed_store():
    """鸭子类型 store（不继承协议，绕过 ABC）自带三个方法 → 契约校验放行。"""
    class DuckStore:
        async def append_batch(self, session_id, batch_id, events): ...
        async def read_range(self, session_id, **kw): return []
        async def committed_head(self, session_id): return 0

    assert supports_ordered_commit(DuckStore())


def test_supports_ordered_commit_requires_all_three():
    """第二层强制：鸭子类型绕过 ABC，部分实现由构造期校验拦下。

    半可用的提交门比完全没有更难诊断——所以三缺一即判不合格。
    """
    class PartialStore:
        async def append_batch(self, session_id, batch_id, events): ...

    assert not supports_ordered_commit(PartialStore())


def test_the_protocol_has_no_replay():
    """分批重放不在协议面上——它是 core 的函数（`reducers.replay_session`）。

    这条从前叫 `test_replay_default_is_inherited_not_reimplemented`，钉的是「`replay` 是协议上
    一个能用的默认实现，两个内置 store 都不该覆盖它」。2026-09-21 整个搬进 core，理由写在
    `replay_session` 的 docstring 里，一句话：**一个只由必需读原语组合而成的算法属于调用方**。
    放在协议上宣称了「实现方可以换掉它」，而那份自由一次没被行使，代价却是实打实的——鸭子类型
    的 store 继承不到默认实现，于是得补一道 `supports_replay` 门来兜。

    钉「协议面上没有它」而不是「它不在 abstractmethods 里」：后者在它被加回成默认实现时也会
    绿，而那正是不该发生的形态（同 `read_by_session` 那条）。
    """
    assert not hasattr(EventStore, "replay"), (
        "replay 回到协议上了：它只用 read_range + committed_head，给协议加它等于宣称实现方可以"
        "换掉它，而那份自由换来的是一道构造期能力门")
    assert not hasattr(EventStore, "REPLAY_BATCH"), (
        "批大小是 core 的权衡（内存硬约束 vs 耗时软约束），不是 store 的")
    for cls in (InMemoryEventStore, SqlEventStore):
        assert not hasattr(cls, "replay"), f"{cls.__name__} 自己实现了 replay"


# 这里从前有 `test_supports_replay_asks_only_whether_it_exists`：钉「`supports_replay` 只问属性
# 存不存在，判据与 `supports_ordered_commit` 相反」。那个函数 2026-09-21 随协议上的 `replay`
# 一起删了——`replay` 变成 core 的 `reducers.replay_session`（只用 `read_range` +
# `committed_head` 两个必需原语）之后，任何满足协议的 store 都能被重放，不需要这道门，也不
# 需要解释「为什么这一个判据是反的」。


# 这里从前有 `test_runtime_rejects_duck_typed_store_without_replay`：钉「三个必需方法齐全但不
# 继承协议的鸭子 store，因为拿不到 `replay` 的默认实现而被构造期拒绝」。那道门 2026-09-21 随
# `replay` 搬进 core 一起删了——`reducers.replay_session` 只用 `read_range` +
# `committed_head`，这种鸭子 store 现在**能**被重放，拒绝它是错的。下面那条
# `test_runtime_rejects_store_without_ordered_commit` 仍在（它兜的是抽象桩真的缺方法）。




def test_runtime_rejects_store_without_ordered_commit():
    """构造期响亮拒绝：按 ID 排序的 store 不是「功能少一点」，是恢复语义错误。

    鸭子类型 store 绕过 ABC，所以这道校验是第二层网；且**与 commit policy 无关**
    ——best_effort 也一样拒绝，因为错的是恢复语义不是提交确认。
    """
    from ctx_weft.core.models.config import RuntimeConfig
    from ctx_weft.core.runtime import CtxWeftRuntime

    class LegacyStore:
        """只有旧接口的 store——WP2 之前的形态。"""
        async def append(self, event): ...
        # 没有 append_batch / read_range / committed_head

    for policy in ("required", "best_effort"):
        with pytest.raises(ValueError, match="未实现有序提交"):
            CtxWeftRuntime(
                event_store=LegacyStore(),
                config=RuntimeConfig(event_commit_policy=policy),
            )
