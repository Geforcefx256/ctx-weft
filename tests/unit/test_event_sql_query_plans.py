"""SqlEventStore 的 SQL 形态：排序口径、索引命中、以及已删掉的快照表（2026-09-20 审查）。

这一批钉的是**查询计划**，不是返回值。返回值有一致性套盯着，而「同样的结果用什么代价拿到」
没人盯——本轮审查就是在这儿翻出两处：`list_active_session_ids` 按 `id` 排（既错又慢）、
`read_last_of_type` 缺一列索引（temp b-tree 的行数随会话累积的快照张数线性增长）。

计划断言是拿**实际发出的 SQL** 去 EXPLAIN 的（`before_cursor_execute` 抓语句），不是照抄一份
SQL 字面量——照抄的那种断言在查询改写后会继续绿，而它要防的恰恰是查询改写。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest
from sqlalchemy import event as sa_event

from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers._sqlalchemy import make_session_factory
from ctx_weft.providers.events.store.sql.models import Base
from ctx_weft.providers.events.store.sql.store import SqlEventStore

_T0 = datetime(2026, 9, 20, tzinfo=UTC)


def _ev(id_: str, *, session: str, type_: str, seq: int = 1, **payload) -> Event:
    return Event(id=id_, run_id="r", sequence=seq, session_id=session,
                 type=type_, timestamp=_T0, payload=payload)


class _Capture:
    """把 store 实际发给 DBAPI 的语句抓下来，供 EXPLAIN。"""

    def __init__(self) -> None:
        self.rows: list[tuple[str, object]] = []

    def attach(self, engine) -> None:
        @sa_event.listens_for(engine.sync_engine, "before_cursor_execute")
        def _on(conn, cursor, statement, parameters, context, executemany):
            self.rows.append((statement, parameters))

    @property
    def last(self) -> tuple[str, object]:
        return self.rows[-1]


@pytest.fixture
async def sqlite_store(tmp_path):
    """真实 SQLite 文件 + 抓语句的 store（表/索引全走 create_all，与生产同一份定义）。"""
    db = tmp_path / "plans.sqlite"
    engine, factory = make_session_factory(f"sqlite+aiosqlite:///{db}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    cap = _Capture()
    cap.attach(engine)
    try:
        yield SqlEventStore(factory), cap, db
    finally:
        await engine.dispose()


def _plan(db, statement: str, params) -> list[str]:
    con = sqlite3.connect(db)
    try:
        return [r[3] for r in con.execute("EXPLAIN QUERY PLAN " + statement, params)]
    finally:
        con.close()


# ── 1. 排序口径：提交序，不是铸造序 ─────────────────────────────────────────


async def test_active_sessions_replay_in_commit_order_not_mint_order(sqlite_store):
    """活跃判据必须按 position 重放。本例里两种序给出**相反**的答案。

    构造：`SessionResumed` 的 id 大、`SessionFinished` 的 id 小，但提交序相反。
    真实来源是 ULID 铸造 ≠ 提交——一条事件在内存里造好后可能晚几十毫秒才落库，期间别的
    事件先提交了。

    - 按 `id` 排 → Finished(evt_a) 先、Resumed(evt_z) 后 → 判成**仍活跃**。
    - 按 position 排 → Resumed 先、Finished 后 → 判成**已结束**。

    后者才对：真正后发生的是 Finished。判错的表现是「已结束的会话每次重启都被捞起来恢复」。
    """
    store, _cap, _db = sqlite_store
    await store.append(_ev("evt_m", session="s1", type_=EventType.SESSION_CREATED))
    # 先提交 Resumed（id 大），后提交 Finished（id 小）
    await store.append(_ev("evt_z_resumed", session="s1", type_="SessionResumed", seq=2))
    await store.append(_ev("evt_a_finished", session="s1", type_="SessionFinished", seq=3))

    assert "s1" not in await store.list_active_session_ids(), (
        "按 id 序重放会把 s1 判成仍活跃——生命周期事件必须按 position 重放")


async def test_lifecycle_query_orders_by_position(sqlite_store):
    """发出的 SQL 里，生命周期查询的 ORDER BY 不含裸 events.id 作首键。"""
    store, cap, _db = sqlite_store
    await store.append(_ev("evt_1", session="s1", type_=EventType.SESSION_CREATED))
    cap.rows.clear()
    await store.list_active_session_ids()

    lifecycle = [s for s, _p in cap.rows if "ORDER BY" in s and "events.type IN" in s]
    assert lifecycle, f"没抓到生命周期查询：{[s for s, _ in cap.rows]}"
    stmt = lifecycle[-1]
    order = stmt[stmt.index("ORDER BY"):]
    assert "events.position" in order, f"ORDER BY 里没有 position：{order}"
    assert not order.strip().startswith("ORDER BY events.id"), (
        f"又按铸造序排了：{order}")


# ── 2. 索引命中 ────────────────────────────────────────────────────────────


# 这里**刻意没有**「生命周期查询不许全表扫」那条计划断言。写过，反向验证不红：单测规模下
# SQLite 对两种 ORDER BY 给出同一个计划，差异只在库跑过 `ANALYZE` 之后才出现（实测 3 万
# 事件真 schema：ANALYZE 后 id 序 24.3ms 全表扫 vs position 序 1.2ms 索引搜；没 ANALYZE 时
# 两者同计划、1.3 vs 1.2ms）。造一个让优化器翻边的规模属于把测试绑在优化器成本模型上——
# 那种断言会因 SQLite 升级而红，且红的时候说明不了任何事。代价数据记在 store.py 的
# docstring 里，这里只钉无条件成立的那半：顺序口径（上面两条）。


async def test_read_last_of_type_needs_no_temp_btree(sqlite_store):
    """取最新快照必须靠索引直接落到最后一行，不许排序。

    没有 `(session_id, type, position)` 时的计划是「搜出该会话该类型的**所有**行 → temp
    b-tree 排序 → 取 1 条」。排序行数 = 该会话累积的快照张数，**随会话变长线性增长**——
    和本轮一路在清的那族是同一个东西，只是长在排序上而不在返回值上，所以扫读代码时看不见。
    """
    store, cap, db = sqlite_store
    for i in range(1, 31):
        await store.append(_ev(f"evt_{i:04d}", session="s1", seq=i, type_=(
            EventType.STATE_SNAPSHOT if i % 3 == 0 else "TaskMessageAppended")))
    cap.rows.clear()
    got = await store.read_last_of_type("s1", EventType.STATE_SNAPSHOT)
    assert got is not None and got.event.id == "evt_0030"

    plan = _plan(db, *cap.last)
    joined = " | ".join(plan)
    assert not any("TEMP B-TREE" in p.upper() for p in plan), (
        f"缺 (session_id, type, position) 索引，退化成排序：{joined}")
    assert "ix_events_session_type_position" in joined, joined


async def test_read_range_is_a_single_index_hit(sqlite_store):
    """恢复增量（真正的热路径）：等值 + 范围 + 有序全由 uq_events_session_position 吃掉。"""
    store, cap, db = sqlite_store
    for i in range(1, 21):
        await store.append(_ev(f"evt_{i:04d}", session="s1", seq=i,
                               type_="TaskMessageAppended"))
    cap.rows.clear()
    await store.read_range("s1", after_position=5,
                           exclude_types=(EventType.STATE_SNAPSHOT,))

    plan = _plan(db, *cap.last)
    joined = " | ".join(plan)
    assert not any("TEMP B-TREE" in p.upper() for p in plan), joined
    assert "uq_events_session_position" in joined, joined


# ── 3. head 读回不经 ORM identity map ───────────────────────────────────────


async def test_head_readback_does_not_go_through_the_orm(sqlite_store):
    """批次 position 分配读回 head 用裸 SELECT，不用 ORM select。

    `append_batch` 的第 1、2 步走 `text()`（它们不填 identity map），第 3 步从前用 ORM
    `select(SessionHeadModel)` 读回——于是「恰好」拿到 UPDATE 后的新值。那是巧合不是保证：
    只要以后有人在同一事务里先 `db.get(SessionHeadModel, sid)`（加行日志就够了），identity
    map 就会把**陈旧**的 next_position 交回来，start 算错，整批事件的 position 撞唯一索引
    或悄悄覆盖已分配区间。

    ⚠️ 这条是**语句形态**断言，不是行为断言，因为那个未来场景没法从外面注入：要污染
    identity map 就得在同一个 session 上先跑一次查询，而那会开启事务，撞上 store 自己的
    `db.begin()`（「a transaction is already begun」）——测试会因为错误的原因红。试过，放弃了。
    所以这里退一步，钉住「第 3 步不是 ORM 查询」这个可检查的形状。
    """
    store, cap, _db = sqlite_store
    cap.rows.clear()
    receipt = await store.append_batch("s1", "b1", [
        _ev("e1", session="s1", seq=1, type_="TaskMessageAppended"),
        _ev("e2", session="s1", seq=2, type_="TaskMessageAppended"),
    ])
    assert [r.position for r in receipt.records] == [1, 2]

    reads = [st for st, _p in cap.rows
             if "next_position" in st and st.lstrip().upper().startswith("SELECT")]
    assert reads, f"没抓到 head 读回：{[st for st, _ in cap.rows]}"
    assert any("SELECT next_position FROM event_session_head" in st for st in reads), (
        f"head 读回又走 ORM 了（会经 identity map）：{reads}")
    assert not any("event_session_head.next_position" in st for st in reads), (
        f"ORM 形态的列限定名出现了：{reads}")


# ── 4. 死掉的快照表 ────────────────────────────────────────────────────────


def test_the_event_snapshots_table_is_gone() -> None:
    """`event_snapshots` / `SnapshotModel` / `keep_snapshots` 全删（2026-09-20）。

    快照是 `events` 里的一条 `StateSnapshot` 事件之后，这张表零读零写。删它的理由不是省
    磁盘，是 `create_all` 会在每个宿主库里建一张没人用的表——而一张存在的空表就是邀请：
    下一个人看到它，第一反应是「快照应该写这儿」。

    宿主（NetliveCoworkPy）还有存量行要传云端，那张表由宿主自己的模型声明并冻结，与本包无关。
    """
    import inspect

    from ctx_weft.providers.events.store.sql import models as m
    from ctx_weft.providers.events.store.sql import store as st

    assert "event_snapshots" not in Base.metadata.tables, (
        f"create_all 还会建它：{sorted(Base.metadata.tables)}")
    assert not hasattr(m, "SnapshotModel")
    assert "SnapshotModel" not in getattr(
        __import__("ctx_weft.providers.events.store.sql", fromlist=["x"]), "__all__", [])

    for fn in (st.SqlEventStore.__init__, st.open_sqlite_event_store):
        params = inspect.signature(fn).parameters
        assert "keep_snapshots" not in params, (
            f"{fn.__qualname__} 还收 keep_snapshots——「留最新 N 张」是恢复策略，归 core")


def test_the_composite_index_is_declared_not_only_patched_in() -> None:
    """新索引要在 `__table_args__` 里（新建库靠 create_all），不能只在 open 路径 ALTER。

    只在 `open_sqlite_event_store` 里 `CREATE INDEX IF NOT EXISTS` 的话，直接
    `SqlEventStore(factory)`（宿主接 Postgres 的正路）就拿不到它，而恢复慢在那边才要紧。
    """
    names = {i.name for i in Base.metadata.tables["events"].indexes}
    assert "ix_events_session_type_position" in names, names
