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
    StoredEvent,
)


# ── InMemoryEventStore ────────────────────────────────────────────────────────


class InMemoryEventStore(EventStore):
    """单进程内存版。线程不安全，仅供开发/测试/单进程 demo 使用。

    订阅由 `EventPersister` 负责，见 `providers/events/persister.py`。
    """

    def __init__(self) -> None:
        self._stored: dict[str, list[StoredEvent]] = {}   # session → 按提交序
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
        stored = self._stored.setdefault(session_id, [])
        records = []
        for i, e in enumerate(events):
            stored.append(StoredEvent(event=e, position=base + i + 1))
            records.append(stored[-1])
            self._event_batch[e.id] = batch_id
        self._next_position[session_id] = base + len(events)
        receipt = CommitReceipt(batch_id=batch_id, records=tuple(records))
        self._batches[batch_id] = receipt
        return receipt

    @staticmethod
    def _same_content(receipt: CommitReceipt, events: list[Event]) -> bool:
        if len(receipt.records) != len(events):
            return False
        return all(r.event == e for r, e in zip(receipt.records, events))

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
        """四个过滤器的语义见协议；`task_id` 的 OR-NULL 安全阀与 SQL 侧逐字同口径。"""
        want = {str(t) for t in include_types}
        skip = {str(t) for t in exclude_types}
        out = []
        for se in self._stored.get(session_id, []):
            if se.position <= after_position:
                continue
            if through_position is not None and se.position > through_position:
                continue
            if want and se.event.type not in want:
                continue
            if skip and se.event.type in skip:
                continue
            # 只排除**明确属于别的 task** 的：无归属（None / 空串）的照样取回。
            if task_id and se.event.task_id and se.event.task_id != task_id:
                continue
            out.append(se)
        return out

    async def read_last_of_type(
        self, session_id: str, type_: str,
    ) -> "StoredEvent | None":
        """取该会话最后一条指定类型的事件（position 最大）。只读一条。"""
        want = str(type_)
        for se in reversed(self._stored.get(session_id, [])):
            if se.event.type == want:
                return se
        return None

    async def committed_head(self, session_id: str) -> int:
        # _next_position 存的是「最后已分配的 position」（0 = 无提交），即 head 本身
        return self._next_position.get(session_id, 0)

    # 快照不在这里：它是日志里的一条 `EventType.STATE_SNAPSHOT` 事件，跟别的事件一样经
    # `append_batch` 进来、经 `read_last_of_type` 出去。从前这里有 `save_snapshot` /
    # `load_latest_snapshot` 和一份单独的 `_snapshots` 字典，连「最新」的口径都要自己实现一遍
    # （按 `(snapshot_at, id)` 取最大，因为写入序 ≠ 时间序）——现在那套全没了。
