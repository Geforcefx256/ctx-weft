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
进程首次为该 session 写快照时各触发一次，用来纠正历史快照的偏差。

两种模式下 blob 都恒等于 ``prune(fold(0..C))``（``reduce_events`` 就是 ``apply_events``
在空 view 上的调用，左折叠可结合；``prune`` 见 `reducers.prune_view_for_snapshot`）。

⚠️ **v2 起等价性是「裁剪后等价」，不再是逐字节等价**（projection_version 2，2026-09-19）：
blob 里的 ``tasks`` 只有活闭包，而全量回放得到的是全部 task。增量链不漂移靠的是
``prune`` 与 ``apply`` 在闭包上**可交换**：

    prune(apply(δ, prune(v))) == prune(apply(δ, v))

它成立是因为 ``prune`` 的三类判据只读 ``status`` / ``parent_task_id`` / ``dag_deps``
（这几个字段两路相同），而被裁掉的 task 之后再不会被任何事件**有效**改动——唯一的例外
是 ``TASK_FINALIZED`` 只写的 ``finished_at``，那条分叉已在 ``prune_view_for_snapshot``
的 docstring 里论证为无害（被裁的 task 进不了 `restore` 的 `all_tasks`，无人读它）。
给终态 task 加新消费者之前，先回去看那一条。

有序提交是 EventStore 的必需部分，故这里没有 legacy 回落分支。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from collections.abc import Callable

from ctx_weft.core.control.reducers import _PROJECTION_VERSION
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
        memory_settled: "Callable[[str], bool] | None" = None,
    ) -> None:
        self._store = event_store
        self._every_n = max(1, every_n_events)
        #: 「这一刻该 session 的 memory 效果是否都已落地」。由 core 注入——只有它知道
        #: （那是它自己的 ingest 路径）。见 `_is_safe_to_write` 的说明。None = 一律放行
        #: （无 TaskManager 的极简接线 / 单测）。
        self._memory_settled = memory_settled
        self._since_snapshot: dict[str, int] = {}
        # 本进程内已写过快照的 session（spec: snapshot-recovery）：空集意味着「服务刚
        # 起来」，第一张快照走全量重锚，纠正上一个进程可能留下的偏差。
        self._anchored: set[str] = set()
        self._subscription = event_bus.subscribe(None, self.on_event) if event_bus else None

    async def detach(self) -> None:
        if self._subscription is not None:
            await self._subscription.unsubscribe()
            self._subscription = None

    def _is_safe_to_write(self, session_id: str) -> bool:
        """现在写一张快照，会不会写出一张**领先于 memory** 的？

        不变式：**有快照 ⟹ 到它的 `committed_head` 为止，memory 效果都已落地。** 恢复路径
        「以快照为准」全靠它——一张领先的快照就是静默的数据丢失（它说某件事做完了，所以
        `pending_recap` / HITL `resolved` 里没有它，而 memory 里其实没有）。

        为什么需要这道门：未提交窗口下的顺序是

            TaskManager.commit_round:
              1. ROUND_COMMITTED
              2. bus.commit_provisional(task_id)   ← 缓冲里的事件在这里补投给 rest 订阅者
              3. commit_round 钩子：HitlResolved → **暂存的 memory 写入落盘**
              4. _rounds.pop

        本类是 rest 订阅者，所以第 2 步就会被叫醒，而那一刻 `committed_head` 已经涵盖整批
        缓冲事件、第 3 步还没跑。实测确认，见
        `tests/unit/test_snapshot_not_ahead_of_memory.py`。

        这也是「从事件里找一个『已落 memory』的信号」那条路走不通的原因：缓冲里的**每一条**
        事件都排在它自己的 memory 效果之前，`ActTurnCompleted` / `CapabilityFinished` 都是。
        判断只能来自 core 的 ingest 路径，所以这里收一个注入的谓词，不自己猜。

        跳过的代价是**安全的那一侧**：这一刻不写，下一个 `RunFinished` 再写；快照偏旧只意味着
        恢复多重放一段、多做一次幂等写。而快照偏新是不可挽回的。
        """
        if self._memory_settled is None:
            return True
        try:
            return bool(self._memory_settled(session_id))
        except Exception:
            # 谓词自己炸了 → 当作「不安全」。宁可少写一张快照，不可写一张领先的。
            logger.exception(
                "SnapshotWriter: memory_settled 判定失败，跳过本次写入 (%s)", session_id)
            return False

    async def on_event(self, event: "Event") -> None:
        session_id = event.session_id
        if not session_id:
            return
        if not self._is_safe_to_write(session_id):
            # 这一刻该 session 还有没落盘的 memory 写入（见 `_is_safe_to_write`）。**计数照旧
            # 累加**，所以下一个安全的边界会立刻补上这一张，不会因为跳过而拖长重放窗口。
            self._since_snapshot[session_id] = self._since_snapshot.get(session_id, 0) + 1
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
    #:
    #: **不写字面量，直接引读侧那一个常量。** 从前这里是独立的 `1`，与
    #: `reducers._PROJECTION_VERSION` 两处各写一遍：只改一个的后果是所有快照永远判不
    #: 可用（写侧盖 1、读侧要 2），恢复全量回放、writer 每次全量重锚，O(delta) 退回
    #: O(n)，且**不报任何错**。绑成同一个值之后这种漂移不可能再发生。
    PROJECTION_VERSION = _PROJECTION_VERSION

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
            prune_view_for_snapshot,
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
            # blob 恒等于 prune(fold(0..cursor))，证明是一行归纳：`reduce_events(evts)`
            # 就是 `apply_events(evts, 空 view)`（reducers.py 里两段循环体逐字相同），而
            # apply 是 for 循环左折叠、可结合，故
            #   fold(0..C) = apply((p, C], fold(0..p))
            # 基底是裁过的，所以这里成立的是**裁剪后**的那条等式
            #   prune(fold(0..C)) = prune(apply((p, C], prune(fold(0..p))))
            # ——它靠 prune 与 apply 在闭包上可交换（见模块 docstring 的 ⚠️ 段）。
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
        # 裁到「活闭包」再落盘（spec: snapshot-recovery v2）。放在这里而**不是**塞进
        # `serialize_view`：那个函数的语义是「全字段序列化」，一堆往返测试依赖它不丢东西；
        # 裁剪是快照写入侧的策略，显式一步、可单独测。契约因此是
        # `blob == serialize_view(prune(fold(0..C)))`，见 `prune_view_for_snapshot`。
        view = prune_view_for_snapshot(view)
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
