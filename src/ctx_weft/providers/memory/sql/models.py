"""SQLAlchemy 表模型：``memory_events`` / ``memory_subscriptions`` / ``memory_blob_refs``。

**平滑切换约束（裁定 D4）**：表名与既有列**逐字保持**与宿主
（`IpMasterCoworkPy/src/ipmastercowork/persistence/postgres/models.py` 的
`MemoryEventModel` / `MemorySubscriptionModel`）一致，本模块**只新增 nullable 列**，
故同一份 schema 既能建新库，也能直接读宿主存量行（存量行的新列全为 NULL，读侧给默认值）。

新增的两列（都 nullable）：

- ``memory_events.tenant`` / ``memory_subscriptions.tenant`` —— 多租户隔离分区键
  （协议【多租户隔离契约】）。存量行 NULL → 读侧归一为 ``"default"``，
  **不迁移数据即落默认分区**。
- ``memory_events.content_format`` —— 内容判别列（多模态无损存取契约）：
  ``"text"`` = ``content`` 是纯文本；``"parts"`` = ``content`` 是
  ``content_to_jsonable`` 产物的 JSON；``NULL`` = **存量行**（宿主 provider 写的，
  没有判别列），读侧走兼容启发式，见 `provider._row_content`。

索引取向：既有两个索引（`ix_memory_task` / `ix_memory_agent`）**名字与列定义一字未改**
——改了会与宿主已建的索引对不上。租户维度另起新名的索引，增量 DDL 即可。

⚠️ **唯一一处对宿主 schema 的破坏性要求**：`memory_subscriptions` 的幂等唯一键必须
含 tenant（契约第 4 条），而宿主已有一个**同名**唯一索引
`ix_subscriptions_session_task_topic` 建在 3 列上。本模块改用新名
`ix_subscriptions_tenant_session_task_topic`（4 列）。宿主切换时须
**先 DROP 旧唯一索引**，否则两个租户的同 (session, task, topic) 订阅会撞 IntegrityError。
这是「让两个租户能各有一条订阅」的必要条件，无法靠只加列绕开。

**一张全新的表**（Task C3，宿主无存量，直接建即可）：``memory_blob_refs``
（事件 → blob 的引用边）。它不受「只加 nullable 列」约束的限制
（那条约束针对的是宿主已有行的两张表）。

**字节不在本库**（spec 2026-08-29 §5）：``memory_blob_refs`` 只是引用边——它需要与
ingest 同事务，故留在这里；字节归 blob store（``providers/blob/fs`` 或宿主自己的对象
存储）。裁定 D4「blob 与 ingest/fold 同事务」论证的正是这张引用边表，不是字节表。
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ctx_weft.providers._sqlalchemy import UtcDateTime


class Base(DeclarativeBase):
    """本 provider 自带的 declarative base（宿主可把这两张表映射进自己的 Base）。"""


class MemoryEventModel(Base):
    """一条 memory 事件。列顺序/类型与宿主 `MemoryEventModel` 对齐（新列在末尾）。"""

    __tablename__ = "memory_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    layer: Mapped[str] = mapped_column(String(16), default="task", index=True)
    type: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    topic: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    seq_no: Mapped[int] = mapped_column(Integer)
    topic_seq_no: Mapped[int] = mapped_column(Integer, default=0)
    is_superseded: Mapped[bool] = mapped_column(Boolean, default=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    timestamp: Mapped[datetime] = mapped_column(UtcDateTime, server_default=func.now())

    # ── 新增（一律 nullable，存量行读得动）──
    tenant: Mapped[str | None] = mapped_column(String(64), nullable=True)
    content_format: Mapped[str | None] = mapped_column(String(16), nullable=True)

    __table_args__ = (
        # 与宿主逐字相同（名字与列都不改）
        Index("ix_memory_task", "session_id", "layer", "task_id", "type"),
        Index("ix_memory_agent", "session_id", "layer", "agent_id", "type"),
        # 租户维度：新名新索引，宿主增量建即可
        Index("ix_memory_tenant_task", "tenant", "session_id", "layer", "task_id"),
        Index("ix_memory_tenant_agent", "tenant", "session_id", "layer", "agent_id"),
        Index("ix_memory_tenant_topic", "tenant", "topic", "topic_seq_no"),
        # ── load_view / recall_topic 的工作索引（2026-09-19）───────────────────
        #
        # 起因：`load_view` 在长会话上实测走**全表扫**（上面那两条 `ix_memory_*` 帮不上
        # ——`type IN (...)` 是多值，且排序列不在索引里），而它一个回合要被调十几次。
        # 同一个 (session, layer, task) 分区里死行（is_superseded）会累积到活行的百倍，
        # 于是每次都把死行读出来再丢掉。
        #
        # 三处刻意的选择：
        #
        # 1. **`is_superseded` 做等值列，而不是部分索引的谓词。** 部分索引（`WHERE
        #    is_superseded = 0`）在 SQLite 上要求索引谓词与查询谓词**逐字一致**才命中
        #    ——实测 `= 0` 的索引配 `IS 0` 的查询直接不命中、退回全表扫，且不报错。而本
        #    仓两个 provider 的写法本就不同（core 用 `.is_(False)`、宿主用 `== False`），
        #    那样任何一处改写法都会静默失效。做成等值列则两种写法都命中，代价是索引里
        #    含死行（体积大些，查询效率一样——等值定位后只扫活行段）。
        # 2. **`type` 不进索引。** 它是 `IN (...)` 多值，放进去会打断有序性、迫使
        #    `ORDER BY timestamp, seq_no` 走临时 B-tree；挪出去让排序列紧跟等值前缀，
        #    排序就完全由索引提供（实测：临时 B-tree 消失）。type 退为回表过滤，作用在
        #    已经很小的活行集上。
        # 3. **tenant 不进索引。** core 侧的 tenant 过滤是 `COALESCE(tenant,'default')=?`
        #    （表达式，裸列索引本来就匹配不上——上面那三条 `ix_memory_tenant_*` 对它同样
        #    没用），而 `session_id` 已经把范围收到一个会话，tenant 再滤几乎不减行。
        #
        # `agent` 那条同时服务 AGENT scope 与 TASK scope 的跨 task 聚合（半址
        # `task_id=None` → 按 agent_id 过滤，layer 仍是 'task'，故 layer 留在索引里）。
        Index("ix_memory_task_view", "session_id", "layer", "task_id",
              "is_superseded", "timestamp", "seq_no"),
        Index("ix_memory_agent_view", "session_id", "layer", "agent_id",
              "is_superseded", "timestamp", "seq_no"),
        # `recall_topic`（topic + topic_seq_no > cursor，按 topic_seq_no 升序）与写路径的
        # `max(topic_seq_no) WHERE topic=?`。后者实测从回表变成 COVERING INDEX。
        Index("ix_memory_topic_seq", "topic", "topic_seq_no"),
    )


class MemoryBlobRefModel(Base):
    """事件 → blob 的引用边（spec §5.2 设想的 side index，但长在 memory 内部）。

    正因为它在 memory 内部，写入才能与 ``ingest`` **同一个事务**——这是原方案
    （blob 在 filesystem provider、引用索引在别处）做不到的：那边无论如何都会存在
    「事件已落库、引用还没记上」的窗口。

    字节本身不在这张表里（也不在本库任何表里，spec §5）——本表只记「谁引用了
    哪个 sha」。活性判定靠 JOIN，见 `provider.SqlMemoryProvider.live_blob_refs`。

    没有外键约束：宿主的 `memory_events` 是存量表，加 FK 需要 DDL 且会在
    「先写 blob 后写 event」的顺序上反过来添乱。
    """

    __tablename__ = "memory_blob_refs"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sha: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)


class MemorySubscriptionModel(Base):
    """task 对 topic 的订阅。唯一键含 tenant——见模块 docstring 的 ⚠️。"""

    __tablename__ = "memory_subscriptions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str] = mapped_column(String(64), default="")
    topic: Mapped[str] = mapped_column(String(256))
    cursor: Mapped[int] = mapped_column(Integer, default=0)
    intent: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(UtcDateTime, server_default=func.now())

    # ── 新增（nullable）──
    tenant: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        Index(
            "ix_subscriptions_tenant_session_task_topic",
            "tenant", "session_id", "task_id", "topic",
            unique=True,
        ),
    )
