"""SqlEventStore：SQLAlchemy async 的 EventStore 实现（默认 SQLite，postgres 同源）。

协议面六个方法齐备。与 `providers/events/store/in_memory` 是同一套契约的两个实现，
`tests/unit/test_event_store_conformance.py` 对两者跑同一套用例。

设计要点：

- **`append` 不过滤瞬态事件**（spec 2026-08-29 §6.4）：那是订阅策略，归 `EventPersister`。
  与 `InMemoryEventStore` 同口径，否则一致性套没法用同一份用例跑两边。
- **一切排序按 position（提交序），不按 `id`**。`id` 是 ULID 铸造序；两者在并发提交下会
  分叉，而按 `id` 当序正是 H2 那一类 bug 的根因。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ctx_weft.protocols.events import (
    CommitReceipt,
    Event,
    EventConflictError,
    EventStore,
    StoredEvent,
)
from ctx_weft.providers._sqlalchemy import make_session_factory
from ctx_weft.providers.events.store.sql.models import (
    Base,
    EventBatchModel,
    EventModel,
    SessionHeadModel,
)

logger = logging.getLogger(__name__)

__all__ = ["SqlEventStore", "open_sqlite_event_store"]


class SqlEventStore(EventStore):
    """SQLAlchemy-backed event store（含有序提交扩展，spec: event-log）。"""

    def __init__(
        self, session_factory: "async_sessionmaker[AsyncSession]",
    ) -> None:
        # 从前这里还有 `keep_snapshots=3`（「每 session 只留最新 N 张快照」）。那是**恢复
        # 策略**，删错一张的后果是恢复退化甚至找不到可用基底——store 不该做这个决定。快照
        # 变成事件之后，保留归入事件保留策略，这个参数连带整张 event_snapshots 表一起删了。
        self._factory = session_factory

    # ── 写（有序提交扩展：原子批次）──────────────────────────────────────

    async def append_batch(
        self, session_id: str, batch_id: str, events: list[Event],
    ) -> CommitReceipt:
        """整批原子提交。

        分配走会话 head 行的**同事务原子 UPDATE**（禁无锁 MAX+1）：head upsert 是事务内
        第一条写语句——SQLite 立即取写锁（BEGIN IMMEDIATE 等价），PG 端由 UPDATE 取行锁，
        两个连接争用同会话时天然串行化。batch 表主键承担幂等与并发下的最终判定：竞态
        重放撞 PK → 回滚后走比对路径（原 receipt / EventConflictError）。
        """
        if not events:
            raise ValueError("append_batch: empty batch")
        for e in events:
            if e.session_id != session_id:
                raise ValueError(
                    f"append_batch: event {e.id} session {e.session_id!r} != batch session "
                    f"{session_id!r}（批内事件必须同属一个 session）")

        # 快路径：已提交过的 batch 直接比对（确认丢失后的原样重试，无写放大）
        async with self._factory() as db:
            row = await db.get(EventBatchModel, batch_id)
            if row is not None:
                return await self._receipt_for_committed(db, row, events)

        try:
            async with self._factory() as db, db.begin():
                # 1) head upsert（写语句，立即取写锁/行锁）
                await db.execute(text(
                    "INSERT INTO event_session_head (session_id, next_position) "
                    "VALUES (:sid, 0) ON CONFLICT (session_id) DO NOTHING"),
                    {"sid": session_id})
                # 2) 原子推进（SQLite/PG 同语法；这是串行化点）
                await db.execute(text(
                    "UPDATE event_session_head SET next_position = next_position + :n "
                    "WHERE session_id = :sid"),
                    {"n": len(events), "sid": session_id})
                # 3) 读回分配区间。**刻意用 text() 而不是 ORM select**：前两步是
                #    `db.execute(text(...))`，它们不填 identity map，所以 ORM 查询「恰好」
                #    是首次加载、因而拿到 UPDATE 后的新值。那是巧合不是保证——只要以后有人
                #    在同一事务里先 `db.get(SessionHeadModel, sid)`（比如加行日志），
                #    identity map 就会把陈旧的 next_position 交回来，后果是整批事件 position
                #    算错：撞唯一索引，或悄悄覆盖已分配区间。直接取标量，没有这一层。
                allocated = (await db.execute(text(
                    "SELECT next_position FROM event_session_head WHERE session_id = :sid"),
                    {"sid": session_id})).scalar_one()
                start = allocated - len(events)
                # 4) 整批事件（position 连续递增；(session_id,position) 唯一与 event.id
                #    主键承担兜底不变式——重复提交的 id 在这里撞 IntegrityError）
                for i, e in enumerate(events):
                    db.add(self._to_row(e, position=start + i + 1))
                # 5) 幂等账
                db.add(EventBatchModel(
                    batch_id=batch_id, session_id=session_id,
                    first_position=start + 1, event_count=len(events)))
                receipt = CommitReceipt(
                    batch_id=batch_id,
                    records=tuple(StoredEvent(event=e, position=start + i + 1)
                                  for i, e in enumerate(events)))
            return receipt
        except IntegrityError:
            # 撞 batch PK（并发同 batch_id 已提交）或撞已提交 event.id——回滚后按已提交
            # 内容判定：原样 → 原 receipt；否则 EventConflictError。
            async with self._factory() as db:
                row = await db.get(EventBatchModel, batch_id)
                if row is not None:
                    return await self._receipt_for_committed(db, row, events)
                raise EventConflictError(
                    f"append_batch {batch_id!r}: event id already committed in another "
                    f"batch (integrity violation rolled back)") from None

    async def _receipt_for_committed(
        self, db: AsyncSession, row: EventBatchModel, events: list[Event],
    ) -> CommitReceipt:
        """已提交批次的幂等/冲突判定：内容逐字段一致（忽略 position）→ 原 receipt。"""
        stored_rows = (await db.execute(
            select(EventModel).where(EventModel.id.in_([e.id for e in events]))
            .order_by(EventModel.position))).scalars().all()
        stored_by_id = {r.id: r for r in stored_rows}
        if len(stored_rows) == len(events):
            same = all(
                self._row_matches(r, e)
                for e, r in ((e, stored_by_id.get(e.id)) for e in events)
            )
            if same:
                records = tuple(
                    StoredEvent(event=e, position=stored_by_id[e.id].position)
                    for e in events)
                return CommitReceipt(batch_id=row.batch_id, records=records)
        raise EventConflictError(
            f"batch {row.batch_id!r} already committed with different content")

    @staticmethod
    def _row_matches(row: EventModel, event: Event) -> bool:
        return (
            row.id == event.id
            and row.run_id == event.run_id
            and row.session_id == event.session_id
            and row.task_id == event.task_id
            and row.agent_id == event.agent_id
            and row.tenant_id == event.tenant_id
            and row.type == event.type
            and row.sequence == event.sequence
            and json.loads(row.payload_json) == event.payload
            and json.loads(row.metadata_json) == event.metadata
            and row.causation_id == event.causation_id
            and row.origin == event.origin
            and row.schema_version == event.schema_version
        )

    @staticmethod
    def _to_row(event: Event, *, position: int) -> EventModel:
        return EventModel(
            id=event.id,
            run_id=event.run_id,
            session_id=event.session_id,
            task_id=event.task_id,
            agent_id=event.agent_id,
            tenant_id=event.tenant_id,
            type=event.type,
            sequence=event.sequence,
            payload_json=json.dumps(event.payload),
            metadata_json=json.dumps(event.metadata),
            causation_id=event.causation_id,
            origin=event.origin,
            schema_version=event.schema_version,
            position=position,
            timestamp=event.timestamp,
        )

    # ── 读 ────────────────────────────────────────────────────────────────────

    async def read_range(
        self,
        session_id: str,
        *,
        after_position: int = 0,
        through_position: int | None = None,
        include_types: tuple[str, ...] = (),
        exclude_types: tuple[str, ...] = (),
        task_id: str = "",
    ) -> list[StoredEvent]:
        """按 position 升序读 (after, through]；只含已提交（position 非空）的事件。

        四个过滤器的语义见协议。`task_id` 那条的 OR-NULL 安全阀是**故意**的，改动前先看那边
        的说明与 `test_read_primitives_conformance.py`。
        """
        conds = [
            EventModel.session_id == session_id,
            EventModel.position.isnot(None),
            EventModel.position > after_position,
        ]
        if through_position is not None:
            conds.append(EventModel.position <= through_position)
        if include_types:
            conds.append(EventModel.type.in_(tuple(str(t) for t in include_types)))
        if exclude_types:
            conds.append(EventModel.type.not_in(tuple(str(t) for t in exclude_types)))
        if task_id:
            conds.append(or_(
                EventModel.task_id == task_id,
                EventModel.task_id.is_(None),
                EventModel.task_id == "",
            ))
        async with self._factory() as db:
            result = await db.execute(
                select(EventModel).where(*conds).order_by(EventModel.position))
            return [StoredEvent(event=_row_to_event(r), position=r.position)
                    for r in result.scalars().all()]

    async def read_last_of_type(
        self, session_id: str, type_: str,
    ) -> "StoredEvent | None":
        """取该会话最后一条指定类型的已提交事件（position 最大）。只读一条。"""
        async with self._factory() as db:
            result = await db.execute(
                select(EventModel)
                .where(
                    EventModel.session_id == session_id,
                    EventModel.type == str(type_),
                    EventModel.position.isnot(None),
                )
                .order_by(EventModel.position.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return StoredEvent(event=_row_to_event(row), position=row.position)

    async def committed_head(self, session_id: str) -> int:
        # next_position 存的是「最后已分配的 position」（0 = 无提交），即 head 本身
        async with self._factory() as db:
            head = await db.get(SessionHeadModel, session_id)
            return head.next_position if head else 0

    # ── 快照 ──────────────────────────────────────────────────────────────────

    # 快照不在这里：它是日志里的一条 `EventType.STATE_SNAPSHOT` 事件。从前这里有
    # `save_snapshot` / `load_latest_snapshot` / `_prune_snapshots` 与 `event_snapshots` 表
    # ——连「保留最新 N 张」那条**恢复策略**都写在 store 里（它自己决定删哪些），而那是 core
    # 的事。快照变成事件之后，保留归入事件保留策略，store 不再做这个决定。


def _row_to_event(row: EventModel) -> Event:
    return Event(
        id=row.id,
        run_id=row.run_id,
        sequence=row.sequence,
        session_id=row.session_id,
        type=row.type,
        timestamp=row.timestamp,
        tenant_id=row.tenant_id,
        task_id=row.task_id,
        agent_id=row.agent_id,
        payload=json.loads(row.payload_json),
        metadata=json.loads(row.metadata_json),
        causation_id=row.causation_id,
        # 存量行该列为 NULL → 回落 ""，与 docs/events-v2.md §0「存量事件读出空串」一致。
        origin=row.origin if row.origin is not None else "",
        # 存量行（参考宿主写的）该列为 NULL → 回落 1，零迁移。
        schema_version=row.schema_version if row.schema_version is not None else 1,
    )


@asynccontextmanager
async def open_sqlite_event_store(
    db_path: str | Path,
) -> AsyncIterator[SqlEventStore]:
    """开一个 SQLite backed 的 event store（建表 → yield → dispose）。

    测试与单机部署用。宿主接 postgres 时自带 engine / migration，直接构造
    ``SqlEventStore(session_factory)`` 即可，不必走这里。

    存量库兼容（spec: event-log）：``create_all`` 只建缺失的表，不会给既有 ``events``
    表加列/索引——这里显式补：``ALTER TABLE ADD COLUMN position``（幂等探测）+ 两条
    ``CREATE [UNIQUE] INDEX IF NOT EXISTS``（``uq_events_session_position`` 与
    ``ix_events_session_type_position``；新建库里 create_all 已建，IF NOT EXISTS 空转）。
    NULL 不参与唯一碰撞，迁移前新旧行共存无碍；回填归
    ``scripts/migrate_event_positions.py``。

    **不再碰 ``event_snapshots``**：从前这里有三段给它加列的幂等 ALTER，而那张表 2026-09-20
    随「快照变成事件」删了——给一张没人读没人写的表跑迁移，是最容易活过删除动作的那种死代码。
    连接带 ``timeout=15``（sqlite3 busy timeout）：双连接争用同会话 head 时等待而非
    立即报 database is locked。
    """
    engine, factory = make_session_factory(
        f"sqlite+aiosqlite:///{db_path}", connect_args={"timeout": 15})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # create_all 只建缺失的表；既有 events 表的 position 列要显式补（幂等探测）
            cols = await conn.execute(text("PRAGMA table_info(events)"))
            if "position" not in {r[1] for r in cols}:
                await conn.execute(text("ALTER TABLE events ADD COLUMN position INTEGER"))
            await conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_session_position "
                "ON events (session_id, position)"))
            # read_last_of_type（恢复第一步：取最新快照）靠这条免掉 temp b-tree；存量库
            # create_all 不会补，所以显式建。
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_events_session_type_position "
                "ON events (session_id, type, position)"))
            # ── 未回填的存量库：**拒绝开**（2026-09-20）─────────────────────────
            # 加了 position 列不等于回填了它。所有按 position 的读都带
            # `position IS NOT NULL`（那是「只读已提交」的判据），所以一条 position 为
            # NULL 的存量行对恢复**完全不可见**——`committed_head` 返回 0、`read_range`
            # 返回空、`replay` 也是按 position 区间分批的，于是 `rebuild_view` 恢复出一个
            # **空视图**：没有会话、没有 task、没有 agent，而且不报任何错。之后新事件从
            # position 1 开始编号（与 NULL 行不撞唯一索引），整段历史就此永久不可见。
            #
            # 那是这套设计里最坏的失效方向：静默、彻底、且发生在恢复路径上。所以这里换成
            # 启动就炸——回填是一个需要过目的操作（脚本默认 dry-run 并出报告），不在这里
            # 悄悄替调用方做掉。宿主接 Postgres 的正路不经过本函数，它们在
            # `open_database` 里先跑迁移再恢复（m020 顺带回填），不受影响。
            #
            # 代价：开库时一次 `LIMIT 1` 探测。命中即停；没有 NULL 行时是一次全表扫，而
            # 本函数只给测试与单机部署用，且只在开库时跑一次。
            pending = (await conn.execute(text(
                "SELECT count(*) FROM (SELECT 1 FROM events "
                "WHERE \"position\" IS NULL LIMIT 1)"))).scalar_one()
            if pending:
                raise RuntimeError(
                    f"{db_path}: events 表里还有 position 为 NULL 的存量行，未回填。"
                    "所有按 position 的读都会跳过它们，恢复会静默地得到一个空视图。"
                    "先跑 `python scripts/migrate_event_positions.py --db <path> --execute`"
                    "（默认 dry-run，出报告后再加 --execute）。")
        yield SqlEventStore(factory)
    finally:
        await engine.dispose()
