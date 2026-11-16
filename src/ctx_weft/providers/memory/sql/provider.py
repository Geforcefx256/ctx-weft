"""SqlMemoryProvider：SQLAlchemy async 的 MemoryProvider 实现（默认 SQLite，postgres 同源）。

协议面 9 方法齐备：写 ingest / fold、读 load_view_ids / read_records / recall_topic / recall_semantic、
订阅 subscribe_topic / list_subscriptions、能力 describe。

与 `providers/memory/in_memory/provider.py` 是同一套契约的两个实现，
`tests/unit/test_memory_conformance.py` 对两者跑同一套用例。

设计要点：

- **多租户隔离**（协议【多租户隔离契约】）：``tenant`` 列 + 归一
  ``normalize_tenant``（``None`` / 空串 → ``"default"``），**写读两侧同一个函数**。
  比较一律走 ``COALESCE(tenant, 'default')``——存量行的 NULL 因此与显式 ``"default"``
  落同一分区，零数据迁移。隐式跨行扫描（PUBLICATION 覆盖）同样只在本租户分区内进行；
  订阅表唯一键含 tenant。
  按契约第 5 条，``ingest`` 的按 id 幂等与 ``fold`` 的按 id 遗忘**仍是全局 id 命名空间**
  （调用方显式给 id 的操作，不是隐式扫描）——与 in_memory 的划线一致。
- **多模态无损存取**：``content_format`` 判别列（见 models.py）；``list[ContentPart]``
  经 ``content_to_jsonable`` 落 JSON、``content_from_jsonable`` 还原，**持久化层不拍扁**。
- **词汇双读**：新行 ``type`` 列存 kind 字符串，存量行存 legacy 类型字符串；
  load_view_ids 用 ``kind_expansion`` 展开 ``type IN (...)`` 并选择配对可见的 ID；
  read_records 读取原内容并经 ``normalize_view`` 统一表示。
  零数据迁移（v2 设计 §6）。
- **fold 原子**：遗忘 + 补偿在单个事务内（SQLite / postgres 天然满足）。

- **blob 引用边**（spec 2026-08-29 §5）：本 provider **不存字节**——字节归 blob store。
  这里只在 ``memory_blob_refs`` 里维护「事件 → blob」的引用边，**与 ingest 同事务**，
  并经 ``live_blob_refs()`` 把活引用吐给宿主，供 ``FsBlobStore.collect()`` 做 mark。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import and_, case, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ctx_weft.core.utils.clock import as_utc
from ctx_weft.core.utils.content import collect_blob_refs
from ctx_weft.protocols.content import content_from_jsonable, content_to_jsonable, extract_blob_refs
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import (
    BLOB_REF_PREFIX,
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryProvider,
    MemoryProviderInfo,
    MemoryRecord,
    MemoryScope,
    ProviderContext,
    Subscription,
)
from ctx_weft.protocols._legacy_dispatch import visible_dispatch_ids
from ctx_weft.protocols.memory_compat import (
    MemoryKind,
    kind_expansion,
    kind_of,
    layer_of,
    normalize_view,
    validate_half_address,
)
from ctx_weft.providers._sqlalchemy import make_session_factory
from ctx_weft.providers.memory.sql.models import (
    Base,
    MemoryBlobRefModel,
    MemoryEventModel,
    MemorySubscriptionModel,
)

logger = logging.getLogger(__name__)

_DEFAULT_TENANT = "default"
_DEFAULT_KINDS = (MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY)
_READ_BATCH_SIZE = 500

# content_format 判别列的取值（models.py 有完整说明）
_FMT_TEXT = "text"
_FMT_PARTS = "parts"


def normalize_tenant(tenant_id: str | None) -> str:
    """租户归一（协议【多租户隔离契约】第 3 条）：``None`` / 空串 → ``"default"``。

    与 `in_memory.normalize_tenant` 是**同一条规则**——两个 provider 必须一致，
    否则同一套 conformance 断言不可能同时成立。刻意各留一份而不跨包 import：
    provider 之间不该互相依赖（第三方 provider 也只能照契约实现，import 不到）。
    """
    return tenant_id or _DEFAULT_TENANT


def _tenant_expr(column: Any) -> Any:
    """SQL 侧的同一条归一：``COALESCE(tenant, 'default')``。

    存量行（宿主写的，没有 tenant 列值）NULL 因此落默认分区，不迁移即可读。
    """
    return func.coalesce(column, _DEFAULT_TENANT)


# ── 内容编解码（多模态无损存取契约）──────────────────────────────────────────


def _encode_content(content: str | list[Any]) -> tuple[str, str]:
    """``content`` → ``(存储字符串, content_format)``。

    ``str`` 原样存 + ``"text"``；``list[ContentPart]`` 走 ``content_to_jsonable``
    再 ``json.dumps`` + ``"parts"``。**禁止拍扁成纯文本**（契约第 4 条）。

    新行一律写出**非空**的判别列：只有这样 ``NULL`` 才能唯一地表示「存量行」，
    读侧的兼容启发式（见 `_row_content`）才不会误伤新写的纯文本
    （一条正文恰好是 ``"[1, 2]"`` 的用户消息就是反例）。
    """
    jsonable = content_to_jsonable(content)
    if isinstance(jsonable, str):
        return jsonable, _FMT_TEXT
    return json.dumps(jsonable), _FMT_PARTS


def _row_content(raw: str, content_format: str | None) -> str | list[Any]:
    """存储字符串 → ``content``（``_encode_content`` 的逆）。

    ``content_format`` 三态：
    - ``"parts"`` → JSON 解回 ``list[ContentPart]``；
    - ``"text"``  → 纯文本，原样返回；
    - ``NULL``    → **存量行**（宿主 provider 写的，那时还没有判别列）。宿主当时的读法
      就是「试 json.loads，是 list 就当结构化内容」，这里保留同一启发式，
      否则平滑切换后宿主的历史结构化内容会变成一坨 JSON 字符串。
      启发式**只对 NULL 行生效**，新行永不走这条路。
    """
    if content_format == _FMT_PARTS:
        return content_from_jsonable(json.loads(raw))  # type: ignore[return-value]
    if content_format is not None:
        return raw
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, list):
        return content_from_jsonable(parsed)  # type: ignore[return-value]
    return raw


def _parse_row_vocab(row: MemoryEventModel) -> tuple[MemoryEventType | None, MemoryKind | None]:
    """``type`` 列 → ``(legacy type | None, kind | None)``。新行存 kind、存量行存 legacy 类型。"""
    try:
        return MemoryEventType(row.type), None
    except ValueError:
        pass
    try:
        return None, MemoryKind(row.type)
    except ValueError:
        return None, None  # 未知词汇（前向兼容）：存储保留、视图不见


def _row_to_record(
    row: MemoryEventModel, blob_refs: list[str] | None = None
) -> MemoryRecord:
    type_, kind = _parse_row_vocab(row)
    try:
        scope = MemoryScope(row.layer)
    except ValueError:
        scope = None
    try:
        meta = json.loads(row.metadata_json or "{}")
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return MemoryRecord(
        id=row.id,
        type=type_,
        content=_row_content(row.content, row.content_format),
        timestamp=row.timestamp,
        role=row.role,  # type: ignore[arg-type]
        topic=row.topic,
        kind=kind,  # legacy 行 None → normalize_view 按 LEGACY_TRIPLE 重打
        scope=scope,
        address=MemoryAddress(
            session_id=row.session_id, task_id=row.task_id, agent_id=row.agent_id),
        metadata={
            **meta,
            "seq_no": row.seq_no,
            "topic_seq_no": row.topic_seq_no,
            "task_id": row.task_id or "",
        },
        blob_refs=list(blob_refs or []),
    )


async def _declared_refs(
    db: Any, rows: Sequence[MemoryEventModel]
) -> dict[str, list[str]]:
    """按 event_id 取「**没有**出现在 content 结构化字段里」的那部分引用。

    引用表是 mark 结果的物化，它不区分来源；而 `MemoryRecord.blob_refs` 的语义是
    「声明的、content 里看不见的那些」——两者相减才对得上写侧。不相减的话，
    `collect_blob_refs` 会把结构化 ref 数两遍（去重后无害，但语义漂移，且
    `_rebuild` 的累积会把它们写进补偿记录的 blob_refs，越滚越多）。

    为什么不能按 `content_format == "text"` 短路、省掉纯文本批次的这次查询：
    `MemoryEvent.blob_refs` 在协议层是独立字段，理论上可与纯文本 content（
    `content_format == "text"`）共存——当前唯一生产者 `fold._rebuild` 总是产出
    `list[ContentPart]`，故这条路径实际不触发，但按格式过滤会让这种边界情况静默
    丢失已声明的 ref（等于本缺陷的另一种复发形态）。加一列数据库标记区分「这批
    确实没有声明式 ref」又会违反「无存量迁移」约束（不写迁移脚本、不加数据库列）。
    两轮 review 都评估过这个问题，结论一致：多一次查询换来的正确性没有更便宜的
    替代，故按 event_id 无差别查询，不做基于 content_format 的短路。
    """
    ids = [r.id for r in rows]
    if not ids:
        return {}
    result = await db.execute(
        select(MemoryBlobRefModel.event_id, MemoryBlobRefModel.sha)
        .where(MemoryBlobRefModel.event_id.in_(ids))
    )
    all_refs: dict[str, list[str]] = {}
    for event_id, sha in result.all():
        all_refs.setdefault(event_id, []).append(f"{BLOB_REF_PREFIX}{sha}")
    out: dict[str, list[str]] = {}
    for row in rows:
        refs = all_refs.get(row.id)
        if not refs:
            continue
        structural = set(extract_blob_refs(_row_content(row.content, row.content_format)))
        declared = [r for r in refs if r not in structural]
        if declared:
            out[row.id] = declared
    return out


def _partition_where(address: MemoryAddress, scope: MemoryScope, tenant: str) -> Any:
    """归属分区 WHERE（含租户）：seq 计数与视图查询共用同一口径。

    与 in_memory 的 ``_scope_key`` 一一对应——那边把 tenant 编进计数器 key，
    这里把它编进 WHERE。
    """
    conds = [
        _tenant_expr(MemoryEventModel.tenant) == tenant,
        MemoryEventModel.session_id == address.session_id,
        MemoryEventModel.layer == scope.value,
    ]
    if scope is MemoryScope.TASK:
        conds.append(MemoryEventModel.task_id == address.task_id)
    elif scope is MemoryScope.AGENT:
        conds.append(MemoryEventModel.agent_id == address.agent_id)
    return and_(*conds)


class SqlMemoryProvider(MemoryProvider):
    """SQLAlchemy async 的 MemoryProvider 实现。SQLite 是默认后端，postgres 同一份代码。

    **不是 blob store**（spec 2026-08-29 §5）：字节归 blob store（如
    ``providers/blob/fs.FsBlobStore``），本类只维护 ``memory_blob_refs`` 引用边，
    并经 ``live_blob_refs()`` 供宿主做 mark-sweep 的 mark 输入。
    """

    name = "sql_memory"

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._factory = session_factory

    # ── 写（2）────────────────────────────────────────────────────────────────

    async def ingest(self, event: MemoryEvent, ctx: ProviderContext) -> str:
        async with self._factory() as db, db.begin():
            if event.id is not None and await self._id_exists(db, event.id):
                # record-id 契约：已存在（含 superseded）= no-op，不比对内容、不推进计数器
                return event.id
            return await self._ingest_in_tx(db, event, ctx)

    async def recall_for_organize(self, ctx, *, operation, keep_last=0):
        from ctx_weft.providers.memory.organize import recall_for_organize
        return await recall_for_organize(self, ctx, operation=operation, keep_last=keep_last)

    async def organize(self, ctx, *, operation, record_ids=None, summary=None,
                       keep_recent_images=2):
        from ctx_weft.providers.memory.organize import organize
        return await organize(self, ctx, operation=operation, record_ids=record_ids,
                              summary=summary, keep_recent_images=keep_recent_images)

    async def _commit_organize(self, expected_ids, replacements, ctx):
        """Conditional UPDATE and replacements share one database transaction."""
        async with self._factory() as db, db.begin():
            # A conditional UPDATE, rather than an unlocked SELECT followed by
            # fold(), also detects a competing writer on the selected sources.
            result = await db.execute(update(MemoryEventModel).where(
                MemoryEventModel.id.in_(expected_ids),
                MemoryEventModel.is_superseded.is_(False),
                _tenant_expr(MemoryEventModel.tenant) == normalize_tenant(ctx.tenant_id),
                MemoryEventModel.session_id == ctx.session_id,
            ).values(is_superseded=True))
            if result.rowcount != len(set(expected_ids)):
                await db.rollback()
                return None
            ids = []
            for event in replacements:
                if event.id is not None and await self._id_exists(db, event.id):
                    ids.append(event.id)
                else:
                    ids.append(await self._ingest_in_tx(db, event, ctx))
            await db.flush()
            rows = list((await db.execute(select(MemoryEventModel).where(
                MemoryEventModel.id.in_(ids)))).scalars().all())
            declared = await _declared_refs(db, rows)
            by_id = {r.id: _row_to_record(r, declared.get(r.id)) for r in rows}
            return [by_id[rid] for rid in ids]

    async def fold(
        self,
        supersede_ids: list[str],
        replacements: list[MemoryEvent],
        ctx: ProviderContext,
    ) -> list[str]:
        """原子「遗忘 + 补偿」：**单个事务**内标 superseded + 写 replacements。

        已 superseded / 不存在的 id 跳过（幂等）；replacement 带 id 时按 record-id 契约幂等。
        """
        new_ids: list[str] = []
        async with self._factory() as db, db.begin():
            if supersede_ids:
                await db.execute(
                    update(MemoryEventModel)
                    .where(
                        MemoryEventModel.id.in_(supersede_ids),
                        MemoryEventModel.is_superseded.is_(False),
                    )
                    .values(is_superseded=True)
                )
                await db.flush()
            for ev in replacements:
                if ev.id is not None and await self._id_exists(db, ev.id):
                    new_ids.append(ev.id)
                    continue
                new_ids.append(await self._ingest_in_tx(db, ev, ctx))
        return new_ids

    @staticmethod
    async def _id_exists(db: AsyncSession, event_id: str) -> bool:
        """按 id 存在性（**不看 tenant**——契约第 5 条：id 命名空间仍是全局的）。"""
        result = await db.execute(
            select(MemoryEventModel.id).where(MemoryEventModel.id == event_id))
        return result.scalar_one_or_none() is not None

    async def _ingest_in_tx(
        self, db: AsyncSession, event: MemoryEvent, ctx: ProviderContext
    ) -> str:
        """ingest 内核（须在事务内调用）；fold 复用它以保证原子性。"""
        event_id = event.id or generate_id("mev")
        # v2 归一：kind 解析失败 = 死类型（存储保留、视图不见）；scope 恒可解析
        try:
            kind: MemoryKind | None = kind_of(event.type, event.kind)
        except ValueError:
            kind = None
        scope = layer_of(event.type, event.scope)
        tenant = normalize_tenant(ctx.tenant_id)
        addr = event.address
        assert addr is not None  # MemoryEvent.__post_init__ 已强制

        result = await db.execute(
            select(func.coalesce(func.max(MemoryEventModel.seq_no), 0))
            .where(_partition_where(addr, scope, tenant))
        )
        seq_no = (result.scalar_one() or 0) + 1

        topic_seq = 0
        if event.topic:
            if kind is MemoryKind.PUBLICATION:
                # 黑板覆盖语义：同 topic 旧发布标 superseded，只留最新一条。
                # **只在本租户分区内扫**——否则 B 一发布就把 A 的黑板行标掉，
                # 而 B 连那行都读不到（能销毁读不到的数据，比泄漏更差）。
                await db.execute(
                    update(MemoryEventModel)
                    .where(
                        _tenant_expr(MemoryEventModel.tenant) == tenant,
                        MemoryEventModel.topic == event.topic,
                        MemoryEventModel.type.in_(
                            kind_expansion(MemoryKind.PUBLICATION, MemoryScope.SESSION)),
                        MemoryEventModel.is_superseded.is_(False),
                    )
                    .values(is_superseded=True)
                )
            # topic_seq 刻意**不按租户分区**：与 in_memory 的 `_topic_seq` 同口径
            # （台账 L20 已记为已知弱侧信道）。过滤在读侧做，正确性不受影响。
            result2 = await db.execute(
                select(func.coalesce(func.max(MemoryEventModel.topic_seq_no), 0))
                .where(MemoryEventModel.topic == event.topic)
            )
            topic_seq = (result2.scalar_one() or 0) + 1

        content_str, content_format = _encode_content(event.content)  # type: ignore[arg-type]
        db.add(MemoryEventModel(
            id=event_id,
            session_id=addr.session_id,
            task_id=addr.task_id,
            agent_id=addr.agent_id,
            layer=scope.value,
            # 新行存 kind 字符串；legacy 构造（type 给定）原样存 legacy 类型字符串
            type=str(event.type) if event.type is not None else str(event.kind),
            role=event.role,
            topic=event.topic,
            content=content_str,
            content_format=content_format,
            seq_no=seq_no,
            topic_seq_no=topic_seq,
            is_superseded=False,
            metadata_json=json.dumps(event.metadata),
            timestamp=event.timestamp,
            tenant=tenant,
        ))
        # blob 引用边：**与事件行同一个事务**。判据走 collect_blob_refs——它是 GC 的
        # mark 单一真源（结构化 ref part ∪ event.blob_refs）。不在这里另写 isinstance，
        # 也不解析占位文案：判据一旦分叉，「哪些 blob 还活着」就会和「出网时哪些 part
        # 会被 rehydrate」对不上，而那正好是「回收删掉了还在用的图」的成因。
        for ref in collect_blob_refs(event):
            db.add(MemoryBlobRefModel(
                event_id=event_id, sha=ref[len(BLOB_REF_PREFIX):]))
        await db.flush()
        return event_id

    # ── 读（3）────────────────────────────────────────────────────────────────

    async def read_records(self, record_ids, ctx):
        if not record_ids:
            return []
        ids = list(dict.fromkeys(record_ids))
        positions = {record_id: index for index, record_id in enumerate(ids)}
        rows = []
        declared = {}
        async with self._factory() as db:
            # Keep bind counts bounded on SQLite. Normalize only after collecting
            # every chunk: a legacy dispatch and its result may span chunks.
            for offset in range(0, len(ids), _READ_BATCH_SIZE):
                result = await db.execute(select(MemoryEventModel).where(
                    MemoryEventModel.id.in_(ids[offset:offset + _READ_BATCH_SIZE]),
                    _tenant_expr(MemoryEventModel.tenant) == normalize_tenant(ctx.tenant_id),
                    MemoryEventModel.session_id == ctx.session_id,
                ))
                chunk = list(result.scalars().all())
                rows.extend(chunk)
                declared.update(await _declared_refs(db, chunk))
        rows.sort(key=lambda row: (as_utc(row.timestamp), row.seq_no, positions[row.id]))
        records = normalize_view([_row_to_record(r, declared.get(r.id)) for r in rows],
                                 include_unpaired_dispatch=True)
        from ctx_weft.providers.memory.organize import _cache_records
        _cache_records(self, ctx, records)
        return records

    @staticmethod
    def _view_conditions(address, scope, ctx, kinds):
        validate_half_address(address, scope)
        wanted = list(kinds) if kinds is not None else list(_DEFAULT_KINDS)
        type_strs: set[str] = set()
        for k in wanted:
            type_strs |= kind_expansion(k, scope)

        tenant = normalize_tenant(ctx.tenant_id)
        conds = [
            _tenant_expr(MemoryEventModel.tenant) == tenant,
            MemoryEventModel.session_id == address.session_id,
            MemoryEventModel.layer == scope.value,
            MemoryEventModel.is_superseded.is_(False),
            MemoryEventModel.type.in_(type_strs),
        ]
        if scope is MemoryScope.TASK:
            if address.task_id is not None:
                conds.append(MemoryEventModel.task_id == address.task_id)
                if address.agent_id is not None:  # 全址时防御性校验 agent 归属
                    conds.append(MemoryEventModel.agent_id == address.agent_id)
            else:  # 跨 task 聚合
                conds.append(MemoryEventModel.agent_id == address.agent_id)
        elif scope is MemoryScope.AGENT:
            conds.append(MemoryEventModel.agent_id == address.agent_id)

        return conds

    async def load_view_ids(
        self, address: MemoryAddress, scope: MemoryScope, ctx: ProviderContext,
        kinds: list[MemoryKind] | None = None, *, end_at: datetime | None = None,
    ) -> list[str]:
        conds = self._view_conditions(address, scope, ctx, kinds)
        legacy = (MemoryEventType.TASK_DISPATCH.value, MemoryEventType.TASK_DISPATCH_RESULT.value)
        async with self._factory() as db:
            result = await db.execute(select(
                MemoryEventModel.id,
                MemoryEventModel.type,
                MemoryEventModel.timestamp,
                case((MemoryEventModel.type.in_(legacy), MemoryEventModel.metadata_json),
                     else_=None).label("pair_metadata"),
            ).where(*conds).order_by(
                MemoryEventModel.timestamp.asc(), MemoryEventModel.seq_no.asc()))
            rows = result.all()
        entries = []
        for row in rows:
            metadata = {}
            if row.pair_metadata:
                try:
                    metadata = json.loads(row.pair_metadata)
                except (TypeError, json.JSONDecodeError):
                    pass
            entries.append((row.id, row.type, metadata.get("tool_call_id")))
        visible = visible_dispatch_ids(entries)
        end = as_utc(end_at) if end_at is not None else None
        return [row.id for row in rows if row.id in visible
                and (end is None or as_utc(row.timestamp) <= end)]

    async def recall_topic(
        self,
        topic: str,
        since: int,
        ctx: ProviderContext,
    ) -> tuple[list[MemoryRecord], int]:
        tenant = normalize_tenant(ctx.tenant_id)
        async with self._factory() as db:
            result = await db.execute(
                select(MemoryEventModel)
                .where(
                    _tenant_expr(MemoryEventModel.tenant) == tenant,  # 租户隔离
                    MemoryEventModel.topic == topic,
                    MemoryEventModel.topic_seq_no > since,
                    MemoryEventModel.is_superseded.is_(False),
                )
                .order_by(MemoryEventModel.topic_seq_no.asc())
            )
            rows = list(result.scalars().all())
            declared = await _declared_refs(db, rows)
            records = [_row_to_record(r, declared.get(r.id)) for r in rows]
        new_cursor = rows[-1].topic_seq_no if rows else since
        return records, new_cursor

    async def recall_semantic(
        self,
        query: str,
        scope: MemoryAddress,
        top_k: int,
        ctx: ProviderContext,
    ) -> list[MemoryRecord]:
        """未实现向量检索（``describe().supports_semantic=False``）→ 返空。

        真要接向量库时，**预过滤必须带 tenant**（契约类 docstring），否则相似度会
        直接把别的租户的内容捞出来。
        """
        return []

    # ── 订阅（2）──────────────────────────────────────────────────────────────

    async def subscribe_topic(
        self,
        session_id: str,
        topic: str,
        intent: Literal[
            "subtask", "predecessor", "long_term_background", "long_term_project_log"],
        ctx: ProviderContext,
        task_id: str = "",
    ) -> str:
        """幂等键 ``(tenant, session_id, task_id, topic)``——tenant 必须在键里，
        否则同 session_id 的另一个租户会撞进幂等分支、拿到别人的订阅与游标。"""
        tenant = normalize_tenant(ctx.tenant_id)
        async with self._factory() as db, db.begin():
            existing = await db.execute(
                select(MemorySubscriptionModel).where(
                    _tenant_expr(MemorySubscriptionModel.tenant) == tenant,
                    MemorySubscriptionModel.session_id == session_id,
                    MemorySubscriptionModel.task_id == task_id,
                    MemorySubscriptionModel.topic == topic,
                )
            )
            row = existing.scalars().first()
            if row is not None:
                return row.id  # 幂等：保留原订阅（含游标），不重置
            sub_id = generate_id("sub")
            db.add(MemorySubscriptionModel(
                id=sub_id,
                session_id=session_id,
                task_id=task_id,
                topic=topic,
                cursor=0,
                intent=intent,
                tenant=tenant,
            ))
        return sub_id

    async def list_subscriptions(
        self,
        session_id: str,
        ctx: ProviderContext,
        task_id: str | None = None,
    ) -> list[Subscription]:
        tenant = normalize_tenant(ctx.tenant_id)
        stmt = select(MemorySubscriptionModel).where(
            _tenant_expr(MemorySubscriptionModel.tenant) == tenant,  # 租户隔离
            MemorySubscriptionModel.session_id == session_id,
        )
        if task_id is not None:
            # 该 task 自己的订阅 + session 级订阅（task_id == ""）
            stmt = stmt.where(MemorySubscriptionModel.task_id.in_([task_id, ""]))
        async with self._factory() as db:
            result = await db.execute(stmt.order_by(MemorySubscriptionModel.created_at.asc()))
            rows = list(result.scalars().all())
        return [
            Subscription(
                session_id=r.session_id,
                topic=r.topic,
                cursor=r.cursor,
                intent=r.intent,  # type: ignore[arg-type]
                task_id=r.task_id,
            )
            for r in rows
        ]

    # ── blob 引用边（spec 2026-08-29 §5）─────────────────────────────────────────

    async def live_blob_refs(self) -> set[str]:
        """当前仍被活记录引用的全部 blob ref（``blob:<sha>``）。

        这是 mark-sweep 的 **mark 输出**：宿主把它喂给 ``FsBlobStore.collect(live_refs)``
        （或自己的对象存储清扫器）。本 provider 只维护引用边、不持有字节，故回收动作
        不在这里发生——`fold` 之后崩溃只会让字节被孤立（泄漏），**而不会产生悬空 ref**。

        **不按 tenant 过滤。** 内容寻址让同一份字节跨租户共享一行引用目标，只看单租户
        会把别的租户仍在引用的 sha 判成孤儿并删掉。判据与引用边的写侧
        （``collect_blob_refs``，归一层的 mark 单一真源）严格对应：``content`` 里的
        结构化 ref part ∪ ``event.blob_refs`` 的显式声明。

        ⚠️ **返回值是「此刻的活引用」，不含宽限期语义。** 「已 put、尚未 ingest」那个
        窗口里的字节在这里查不到（它还没有引用边），所以清扫器**必须**自己带宽限期——
        `FsBlobStore.collect` 的 mtime 判据就是干这个的。只按本方法的返回值删会把刚上传
        还没入库的图删掉。
        """
        stmt = (
            select(MemoryBlobRefModel.sha)
            .join(MemoryEventModel, MemoryEventModel.id == MemoryBlobRefModel.event_id)
            .where(MemoryEventModel.is_superseded.is_(False))
            .distinct()
        )
        async with self._factory() as db:
            result = await db.execute(stmt)
            return {f"{BLOB_REF_PREFIX}{sha}" for sha in result.scalars().all()}

    # ── 能力 ──────────────────────────────────────────────────────────────────

    async def describe(self, ctx: ProviderContext) -> MemoryProviderInfo:
        return MemoryProviderInfo(
            name=self.name,
            supports_semantic=False,
            supports_topic=True,
            archives_superseded=True,
        )


# ── 建库/开库便利函数 ─────────────────────────────────────────────────────────


@asynccontextmanager
async def open_sqlite_memory(
    db_path: str | Path,
) -> AsyncIterator[SqlMemoryProvider]:
    """开一个 SQLite backed 的 provider（建表 → yield → dispose）。

    测试与单机部署用。宿主接 postgres 时自带 engine / migration，直接构造
    ``SqlMemoryProvider(session_factory)`` 即可，不必走这里。
    """
    engine, factory = make_session_factory(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        yield SqlMemoryProvider(factory)
    finally:
        await engine.dispose()
