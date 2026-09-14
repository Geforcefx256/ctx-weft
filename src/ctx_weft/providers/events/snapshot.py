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
``committed_head``（C）——触发事件只是「现在写一张」的信号，不是边界。H2 的修复完全
在这个游标的定义里，与折叠策略无关。

折叠策略（spec: snapshot-recovery）：**常态增量**，内容 = 上一张可用快照 + 
``read_range((base, C])``，满足上面那条 O(delta) 约束。全量 ``read_range(0..C)`` 降为
**重锚点**——无可用基底（首张 / 存量快照 / 版本不匹配 / 位置超前）、链深到顶、或本
进程首次为该 session 写快照时各触发一次，用来纠正历史快照的偏差。两种模式下 blob 都
恒等于 fold(0..C)（``reduce_events`` 就是 ``apply_events`` 在空 view 上的调用，左折叠
可结合），「全量回放 vs 快照+增量」两路恢复因此同样天然等价（E5）。

有序提交是 EventStore 的必需部分，故这里没有 legacy 回落分支。
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
        # 本进程内已写过快照的 session（spec: snapshot-recovery）：空集意味着「服务刚
        # 起来」，第一张快照走全量重锚，纠正上一个进程可能留下的偏差。
        self._anchored: set[str] = set()
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

    #: 快照链的最大深度：连续增量写到这个层数就强制一次全量重锚（spec: snapshot-recovery）。
    #: 存在的理由是 `serialize_view` / `deserialize_view` 往返——链式写让 blob 反复过这
    #: 道往返，任何有损字段会逐次退化。上限把累积层数压成常数，长期不重启的会话也拿得到
    #: 周期性的「对齐日志」。折叠总量因此是 O(n)（增量）+ 每 MAX 张一次 O(n)，而不是每张
    #: 都 O(n)。
    MAX_CHAIN_DEPTH = 32

    async def _usable_base(self, session_id: str, head: int):
        """取可作增量基底的最新快照；取不到返回 None（调用方全量重锚）。

        判据与恢复路径共用 `snapshot_is_usable`，额外叠一层写入侧专属的链深上限。
        另加一条：**本进程内还没为这个 session 写过快照时不认基底**——服务每次起来的
        第一张快照都是全量重锚，让「上一进程留下的快照有误」在启动后被纠正一次。
        """
        from ctx_weft.core.control.reducers import snapshot_is_usable

        if session_id not in self._anchored:
            return None
        try:
            snapshot = await self._store.load_latest_snapshot(session_id)
        except NotImplementedError:
            return None
        if not snapshot_is_usable(snapshot, head, max_chain_depth=self.MAX_CHAIN_DEPTH):
            return None
        return snapshot

    async def _write(self, session_id: str, event: "Event", reason: str) -> None:
        from ctx_weft.core.control.reducers import (
            apply_events,
            deserialize_view,
            reduce_events,
            serialize_view,
        )
        from ctx_weft.core.utils.clock import now_utc
        from ctx_weft.core.utils.ids import generate_id
        from ctx_weft.protocols.events import RunSnapshot

        # ── 一致切面（spec: snapshot-recovery，reliability-wp4）───────────────
        # 边界 C = committed_head，**与折叠策略无关**：触发事件只是「现在写一张」的
        # 信号，不是边界（WP3 后 writer 收到的都是已确认事件，C ≥ 触发事件 position）。
        # H2 的修复完全在这个游标的定义里。
        head = await self._store.committed_head(session_id)
        base = await self._usable_base(session_id, head)

        if base is not None:
            # ── 常态：增量，O(delta)────────────────────────────────────────
            # blob 恒等于 fold(0..cursor)，证明是一行归纳：`reduce_events(evts)` 就是
            # `apply_events(evts, 空 view)`（reducers.py 里两段循环体逐字相同），而
            # apply 是 for 循环左折叠、可结合，故
            #   reduce(0..C) = apply((p, C], reduce(0..p)) = apply((p, C], blob_p)
            # 两路等价（E5）因此与全量折时**同样**成立，不需要额外证明。
            delta = await self._store.read_range(
                session_id, after_position=base.last_commit_position,
                through_position=head)
            view = apply_events(
                [se.event for se in delta], deserialize_view(base.state_blob))
            folded, depth, anchored = len(delta), base.chain_depth + 1, False
        else:
            # ── 重锚：全量，O(n)──────────────────────────────────────────
            # 无可用基底（首张 / 存量快照 / 版本不匹配 / 位置超前 / 链深到顶 /
            # 本进程首次），从日志重新推导一张——这是纠正历史快照偏差的地方。
            stored = await self._store.read_range(
                session_id, after_position=0, through_position=head)
            view = reduce_events([se.event for se in stored], run_id=session_id)
            folded, depth, anchored = len(stored), 0, True

        if not view.session_id:
            return  # 该 session 尚无任何已提交事件，跳过（不记 anchored，下次仍重锚）
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
            chain_depth=depth,
        )
        await self._store.save_snapshot(snapshot)
        self._anchored.add(session_id)
        logger.info(
            "SnapshotWriter: snapshot %s for session %s (reason=%s, cut=%d, "
            "folded=%d events, mode=%s, chain_depth=%d)",
            snapshot.id, session_id, reason, head, folded,
            "anchor" if anchored else "incremental", depth,
        )
