"""EventStore 的单进程内存实现。

协议在 `ctx_weft.protocols.events`；本模块只是它的一个实现（spec 2026-08-27 三层划界）。
线程不安全，仅供开发 / 测试 / 单进程 demo；host 上生产要换 Postgres 等持久实现。

有序提交扩展（spec: event-log）：一把 asyncio.Lock 保护「head 分配 + 幂等
查询 + 批量写入」的短临界区——进程内单事件循环下天然串行；跨会话不互相阻塞的并行度
由 SQL 实现承担（内存实现只承诺正确性，见 reliability-wp2 design D5）。
"""

from __future__ import annotations

import asyncio

from ctx_weft.protocols.events import (
    CommitReceipt,
    Event,
    EventConflictError,
    EventStore,
    RunSnapshot,
    StoredEvent,
)
from ctx_weft.providers.events._lifecycle import apply_lifecycle


# ── InMemoryEventStore ────────────────────────────────────────────────────────


class InMemoryEventStore(EventStore):
    """单进程内存版。线程不安全，仅供开发/测试/单进程 demo 使用。

    订阅由 `EventPersister` 负责，见 `providers/events/persister.py`。
    """

    def __init__(self) -> None:
        self._stored: dict[str, list[StoredEvent]] = {}   # session → 按提交序
        self._active: set[str] = set()
        self._snapshots: dict[str, RunSnapshot] = {}
        self._lock = asyncio.Lock()
        self._next_position: dict[str, int] = {}          # session → 下一个 position
        self._batches: dict[str, CommitReceipt] = {}      # batch_id → receipt
        self._event_batch: dict[str, str] = {}            # event.id → batch_id（身份守卫）

    # ── 写（有序提交扩展：原子批次）──────────────────────────────────────

    async def append_batch(
        self, session_id: str, batch_id: str, events: list[Event],
    ) -> CommitReceipt:
        """整批原子提交（进程内单锁即原子）；同 batch_id 幂等、异内容冲突。

        **不过滤瞬态事件**（spec 2026-08-29 §6.4）：那是订阅策略，归 EventPersister。
        """
        if not events:
            raise ValueError("append_batch: empty batch")
        for e in events:
            if e.session_id != session_id:
                raise ValueError(
                    f"append_batch: event {e.id} session {e.session_id!r} != batch session "
                    f"{session_id!r}（批内事件必须同属一个 session）")
        async with self._lock:
            return self._append_batch_locked(session_id, batch_id, events)

    def _append_batch_locked(
        self, session_id: str, batch_id: str, events: list[Event],
    ) -> CommitReceipt:
        # 幂等：同 batch_id 同内容（忽略 position）→ 原 receipt
        existing = self._batches.get(batch_id)
        if existing is not None:
            if self._same_content(existing, events):
                return existing
            raise EventConflictError(
                f"batch {batch_id!r} already committed with different content")
        # 事件身份守卫：同一 event.id 不落第二个批次（含跨 session），也不在批内重复
        # （SQL 由 events.id 主键天然拦截，这里对齐同一行为）
        seen_in_batch: set[str] = set()
        for e in events:
            if e.id in seen_in_batch:
                raise EventConflictError(
                    f"duplicate event id {e.id!r} within batch {batch_id!r}")
            seen_in_batch.add(e.id)
            prior = self._event_batch.get(e.id)
            if prior is not None:
                raise EventConflictError(
                    f"event {e.id!r} already committed in batch {prior!r}")
        base = self._next_position.get(session_id, 0)
        is_new = session_id not in self._stored
        stored = self._stored.setdefault(session_id, [])
        if is_new:
            self._active.add(session_id)   # 种子：仅首次出现时置 active（终态后不复活）
        records = []
        for i, e in enumerate(events):
            stored.append(StoredEvent(event=e, position=base + i + 1))
            records.append(stored[-1])
            self._event_batch[e.id] = batch_id
            apply_lifecycle(self._active, e)
        self._next_position[session_id] = base + len(events)
        receipt = CommitReceipt(batch_id=batch_id, records=tuple(records))
        self._batches[batch_id] = receipt
        return receipt

    @staticmethod
    def _same_content(receipt: CommitReceipt, events: list[Event]) -> bool:
        if len(receipt.records) != len(events):
            return False
        return all(r.event == e for r, e in zip(receipt.records, events))

    async def append(self, event: Event) -> None:
        """单事件 = 单事件批次（batch_id 确定性取 event.id；spec: event-log 兼容要求）。"""
        await self.append_batch(event.session_id, event.id, [event])

    # ── 读 ────────────────────────────────────────────────────────────────────

    async def read_by_session(self, session_id: str) -> list[Event]:
        # 提交序（= position 序）。旧版按 id 排序只在「乱序 append」时分叉——新版
        # append/append_batch 全在锁内按提交顺序入列，position 序即列表序。
        return [se.event for se in self._stored.get(session_id, [])]

    async def read_range(
        self,
        session_id: str,
        *,
        after_position: int = 0,
        through_position: int | None = None,
    ) -> list[StoredEvent]:
        out = []
        for se in self._stored.get(session_id, []):
            if se.position <= after_position:
                continue
            if through_position is not None and se.position > through_position:
                continue
            out.append(se)
        return out

    async def committed_head(self, session_id: str) -> int:
        # _next_position 存的是「最后已分配的 position」（0 = 无提交），即 head 本身
        return self._next_position.get(session_id, 0)

    async def list_active_session_ids(self) -> list[str]:
        return list(self._active)

    async def read_after(self, session_id: str, after_event_id: str) -> list[Event]:
        # legacy 口径（spec: event-log 保留不动）：按 id（ULID）排序过滤，不依赖提交序。
        #
        # ⚠️ after_event_id 不存在于本 session 时，返回 id 大于它的**全部**事件，
        # 不是空列表——协议口径是过滤式的（"id > after_event_id"），不是"从标记处
        # 扫描"。这条路径是活的：core/control/reducers.py 的 rebuild_view 拿快照的
        # last_event_id 调本方法；快照引用了一个不在 store 里的 id 时（事件流被截断、
        # 快照来自别处、或存储不一致），静默返回空会让 view 退化成只剩快照、且这个
        # 数据损失不可观测。多回放一段最坏只是重复折叠（reduce_events 幂等），比
        # 静默丢 delta 安全。
        events = sorted(self._stored.get(session_id, []), key=lambda se: se.event.id)
        return [se.event for se in events if se.event.id > after_event_id]

    async def read_session_events_of_types(
        self, session_id: str, types: tuple[str, ...],
    ) -> list[Event]:
        type_set = set(types)
        return [se.event for se in self._stored.get(session_id, [])
                if se.event.type in type_set]

    async def save_snapshot(self, snapshot: RunSnapshot) -> None:
        # 仅保留每个 session 的最新快照——恢复只需最新一条（snapshot + delta replay）。
        # 「最新」按协议口径（protocols/events.py::load_latest_snapshot）取
        # (snapshot_at, id) 的最大值，**不是**「最后一次调用 save_snapshot」——
        # 写入顺序不保证与时间顺序一致，与 SqlEventStore 的 `ORDER BY created_at
        # DESC, id DESC` 对齐，避免乱序写入时两个实现返回不同快照。
        async with self._lock:
            existing = self._snapshots.get(snapshot.session_id)
            if existing is None or (snapshot.snapshot_at, snapshot.id) >= (
                existing.snapshot_at, existing.id,
            ):
                self._snapshots[snapshot.session_id] = snapshot

    async def load_latest_snapshot(self, session_id: str) -> RunSnapshot | None:
        return self._snapshots.get(session_id)
