"""SnapshotWriter——会话存活期间定期写状态快照。

崩溃恢复针对的是**没有终态**的 session。若快照只在 `SessionFinished` 写，恢复时永远
没有快照可用，`rebuild_view` 只能 O(全部事件) 全量回放。定期写之后，恢复退化成
「最新快照 + 增量 delta」，回放量被限制在约一个阈值窗口内。

⚠️ **本类的 `on_event` 在 `EventBus.emit()` 里内联执行，不是后台任务。** 所以写入路径
必须用 `rebuild_view`（快照 + 增量，O(delta)）而不是全量 reduce——任何 O(全部事件) 的
读取都会直接阻塞 loop 主路径。

⚠️ **必须在 EventPersister 之后订阅**，否则 `rebuild_view` 看不到当前这条事件。
用 `attach_persistence()` 接线，顺序由它保证。

一致切面（spec: snapshot-recovery，change reliability-wp4）：快照边界 = 写那一刻的
``committed_head``（C），内容 = ``read_range(0..C)`` 的全量折——触发事件只是「现在写
一张」的信号，不是边界。单一 apply 语义让「全量回放 vs 快照+增量」两路恢复天然等价
（E5）。有序提交是 EventStore 的必需部分，故这里没有 legacy 回落分支。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ctx_weft.protocols.events import TRANSIENT_EVENT_TYPES

if TYPE_CHECKING:
    from ctx_weft.protocols.events import Event, EventBus, EventStore

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_SNAPSHOT_EVERY_N_EVENTS", "SnapshotWriter"]

#: RunFinished 边界上，距上次快照累计多少事件后写一张。
DEFAULT_SNAPSHOT_EVERY_N_EVENTS = 50


class SnapshotWriter:
    """EventBus 订阅者：会话存活期间定期 + 结束时写 `RunSnapshot`。

    只在 `RunFinished`（一次 loop run 收尾、状态稳定的恢复边界）与 `SessionFinished`
    两个点落快照——中途落快照会把一个跑到一半的 run 的状态固化下来，恢复时反而更难处理。
    """

    def __init__(
        self,
        event_store: "EventStore",
        event_bus: "EventBus | None" = None,
        *,
        every_n_events: int = DEFAULT_SNAPSHOT_EVERY_N_EVENTS,
    ) -> None:
        self._store = event_store
        self._every_n = max(1, every_n_events)
        self._since_snapshot: dict[str, int] = {}
        self._subscription = event_bus.subscribe(None, self.on_event) if event_bus else None

    async def detach(self) -> None:
        if self._subscription is not None:
            await self._subscription.unsubscribe()
            self._subscription = None

    async def on_event(self, event: "Event") -> None:
        session_id = event.session_id
        if not session_id:
            return
        # 瞬态 delta 不计入阈值，使「每 N 事件」按有意义的持久化事件计数。
        if event.type in TRANSIENT_EVENT_TYPES:
            return
        try:
            if event.type == "SessionFinished":
                await self._write(session_id, event, reason="session_finished")
                self._since_snapshot.pop(session_id, None)
                return
            n = self._since_snapshot.get(session_id, 0) + 1
            if event.type == "RunFinished" and n >= self._every_n:
                await self._write(session_id, event, reason="periodic")
                self._since_snapshot[session_id] = 0
            else:
                self._since_snapshot[session_id] = n
        except Exception:
            logger.exception("SnapshotWriter: failed for session %s", session_id)

    #: 当前投影版本（spec: snapshot-recovery）——apply 语义变更时 bump，旧快照据此
    #: 在恢复路径被忽略并全量重建。
    PROJECTION_VERSION = 1

    async def _write(self, session_id: str, event: "Event", reason: str) -> None:
        from ctx_weft.core.control.reducers import reduce_events, serialize_view
        from ctx_weft.core.utils.clock import now_utc
        from ctx_weft.core.utils.ids import generate_id
        from ctx_weft.protocols.events import RunSnapshot

        # ── 一致切面（spec: snapshot-recovery，reliability-wp4）───────────────
        # C = committed_head；内容 = read_range(0..C) 的全量折。触发事件只是信号
        # （WP3 后 writer 收到的都是已确认事件，C ≥ 触发事件 position，不会漏折它）。
        # 全量折是刻意的：与恢复路径共用同一个 apply 语义，两路等价（E5）不需要额外
        # 证明；增量维护 writer 内存 view 的方案被否决见 design D1。
        head = await self._store.committed_head(session_id)
        stored = await self._store.read_range(
            session_id, after_position=0, through_position=head)
        view = reduce_events([se.event for se in stored], run_id=session_id)
        if not view.session_id:
            return  # 该 session 尚无任何已提交事件，跳过
        snapshot = RunSnapshot(
            id=generate_id("snp"),
            run_id=event.run_id or "",
            session_id=session_id,
            last_event_id=event.id,
            last_event_sequence=event.sequence,
            state_blob=serialize_view(view),
            snapshot_reason=reason,
            snapshot_at=now_utc(),
            last_commit_position=head,
            projection_version=self.PROJECTION_VERSION,
        )
        await self._store.save_snapshot(snapshot)
        logger.info(
            "SnapshotWriter: snapshot %s for session %s (reason=%s, cut=%d events)",
            snapshot.id, session_id, reason, len(stored),
        )
