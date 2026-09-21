"""把事件灌进 EventStore 的 EventBus 订阅者，以及接线的便利函数。

抽出成独立组件（而不是像从前那样塞进 `InMemoryEventStore.__init__`）的理由：
换掉内存 store 的宿主否则必须自己重写一遍订阅逻辑——参考宿主
`IpMasterCoworkPy` 里的 `EventPersister` 正是这么来的。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ctx_weft.protocols.events import TRANSIENT_EVENT_TYPES

if TYPE_CHECKING:
    from ctx_weft.protocols.events import Event, EventBus, EventStore

logger = logging.getLogger(__name__)

__all__ = ["EventPersister", "PersistenceHandle", "attach_persistence"]


class EventPersister:
    """订阅 EventBus，把非瞬态事件 append 进任意 `EventStore`。

    **瞬态过滤在这里，不在 store 里**（spec 2026-08-29 §6.4）：每 token 一个的流式
    delta 只为实时流而发，落库会让事件表无界膨胀、且被 `read_by_session` /
    `reduce_events` 全量回放（真相由 `LLMResponseFinished` 承载）。这是**订阅策略**，
    不是存储策略——`EventStore.append` 因此是「让存什么就存什么」，一致性测试才能
    直接测往返而不被 store 悄悄吃掉测试事件。

    ⚠️ `on_event` 吞掉 store 异常**不是**防止掀掉 loop 的唯一防线——
    `InProcessEventBus.emit` 已经捕获 handler 异常并 `logger.exception`。这里再捕一次
    是为了把 event id / type 写进日志（bus 那层只知道 subscriber id）。改动前先看
    那一处，别以为删掉这里就没人兜底了。
    """

    def __init__(self, event_store: "EventStore", event_bus: "EventBus | None" = None) -> None:
        self._store = event_store
        self._subscription = event_bus.subscribe(None, self.on_event) if event_bus else None

    async def on_event(self, event: "Event") -> None:
        if event.type in TRANSIENT_EVENT_TYPES:
            return
        try:
            await self._store.append(event)
        except Exception:
            logger.exception(
                "EventPersister: failed to append event %s (%s)", event.id, event.type
            )

    async def detach(self) -> None:
        """停止订阅。宿主把 runtime 默认的内存 store 换成持久实现时必须调——
        否则旧实例作为孤儿订阅者继续在内存里堆积事件。"""
        if self._subscription is not None:
            await self._subscription.unsubscribe()
            self._subscription = None


class PersistenceHandle:
    """`attach_persistence` 的返回值：一起 detach 掉它接上的全部订阅者。"""

    def __init__(self, persister: EventPersister, snapshot_writer: Any = None) -> None:
        self.persister = persister
        self.snapshot_writer = snapshot_writer

    async def detach(self) -> None:
        """persister 可为 None（required 模式：提交经 CommitGate，无 persister 订阅）。"""
        if self.snapshot_writer is not None:
            await self.snapshot_writer.detach()
        if self.persister is not None:
            await self.persister.detach()


def attach_persistence(
    event_bus: "EventBus",
    event_store: "EventStore",
) -> PersistenceHandle:
    """接上 EventPersister，返回可一起 detach 的 handle。

    **这里不再认识快照**（2026-09-20）。从前本函数收 `snapshot_every_n` 并顺手把
    `SnapshotWriter` 也接上——那让 providers 反向 import `core.control`，而 writer 决定的
    五件事（写在哪个边界、隔多久、安不安全、切面在哪、增量还是重锚）全是领域知识。
    writer 因此搬去了 `core/control/snapshot_writer.py`，连同那条「persister 必须先订阅」
    的顺序契约一起，由那边的 `attach_snapshotting` 承担——**契约仍然只有一处实现**。
    """
    return PersistenceHandle(EventPersister(event_store, event_bus))
