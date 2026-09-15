"""SQLAlchemy 表模型：``events`` / ``event_snapshots`` / ``event_session_head`` / ``event_batches``。

**`Base` 与 `memory/sql` 的刻意不共用。** 共用会让只想建 events 表的宿主被迫连 memory
表一起建，而两个包的可选依赖边界本就是分开的。单文件 SQLite 部署照样可以共享同一个
engine——各自跑一次 ``Base.metadata.create_all`` 即可。

表名与列名与参考宿主
（``IpMasterCoworkPy/src/ipmastercowork/persistence/postgres/models.py``）保持一致，
但**补了两列**（那边漏了、导致往返有损）：

- ``events.schema_version`` —— `Event.schema_version` 是给 reducer 分支用的。不存的话
  第一次 bump 版本时会静默把新事件读成旧版本。
- ``event_snapshots.run_id`` —— 参考实现的 ``load_latest_snapshot`` 硬编码 ``run_id=""``。

有序提交扩展（spec: event-log，change reliability-wp2）：

- ``events.position`` —— 存储层分配的提交位置，同会话唯一递增；**存量行 NULL**（迁移
  工具 ``scripts/migrate_event_positions.py`` 回填），NULL 不参与唯一约束碰撞。
- ``event_session_head`` —— 每会话一行 ``next_position``，批次提交在同一事务内对它做
  原子 UPDATE 完成串行化分配（禁无锁 MAX+1）。
- ``event_batches`` —— batch_id 的幂等账：同 id 重试凭它找回原 receipt / 检出内容冲突。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ctx_weft.providers._sqlalchemy import UtcDateTime


class Base(DeclarativeBase):
    """本包自带的 declarative base（宿主可把这两张表映射进自己的 Base）。"""


class EventModel(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    type: Mapped[str] = mapped_column(String(128), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    causation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # V2 新增：哪个组件发出的（Event.origin，docs/events-v2.md §4）。存量行没有这一列
    # → 读侧回落 ""，与 `Event.origin` 默认值同口径，零迁移。
    origin: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # 存量行（参考宿主写的）没有这一列 → 读侧给默认值 1，零迁移。
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    # spec: event-log —— 存储层提交位置。存量行为 NULL（迁移工具回填）；新写入恒有值。
    # 唯一性由显式索引 uq_events_session_position 保证（create_all 建表时带上；存量库由
    # 迁移路径 CREATE UNIQUE INDEX IF NOT EXISTS 补齐——SQLite 的 ALTER TABLE 加不了
    # 表级约束，统一走索引形态）。
    position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(UtcDateTime, server_default=func.now())

    __table_args__ = (
        # list_active_session_ids 按 (session_id, type) 收窄，再按 id 升序重放。
        Index("ix_events_session_type", "session_id", "type"),
        # (session_id, position) 唯一——head 串行化分配之外的兜底不变式（SQL 默认
        # NULL 不参与唯一碰撞，存量行共存无碍）。
        Index("uq_events_session_position", "session_id", "position", unique=True),
    )


class SessionHeadModel(Base):
    """每会话一行 next_position：批次提交的事务内原子推进点（spec: event-log）。"""

    __tablename__ = "event_session_head"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    next_position: Mapped[int] = mapped_column(Integer, default=0)


class EventBatchModel(Base):
    """batch_id 幂等账：确认丢失后原样重试凭它找回原 receipt（spec: event-log）。"""

    __tablename__ = "event_batches"

    batch_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    first_position: Mapped[int] = mapped_column(Integer)
    event_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, server_default=func.now())


class SnapshotModel(Base):
    __tablename__ = "event_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str] = mapped_column(String(64), default="")
    last_event_id: Mapped[str] = mapped_column(String(64))
    last_event_sequence: Mapped[int] = mapped_column(Integer)
    state_blob_json: Mapped[str] = mapped_column(Text, default="{}")
    snapshot_reason: Mapped[str] = mapped_column(String(64), default="")
    # spec: snapshot-recovery——存量行为 NULL/1 → 恢复忽略该快照走全量回放。
    last_commit_position: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chain_depth: Mapped[int | None] = mapped_column(Integer, nullable=True)
    projection_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, server_default=func.now())
