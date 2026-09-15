"""提交门（spec: event-commit；change reliability-wp3，可靠性方案 §4.5/§4.7）。

required 模式的提交确认点：事件先经本门确认存储提交（WP2 的 ``append_batch``），
成功返回带 position 的 StoredEvent，失败先标记会话健康再抛
``PersistenceUnavailableError``——**通知成功从此等价于提交成功**。

位置：编排层（core/），由 Runtime 构造期经 ``EventBus.attach_commit_gate`` 接进
emit 路径；Provider/总线实现不反向依赖本模块。best_effort 模式不接本门（退回
``attach_persistence`` 的旧观察者路径并告警）——本门只有 required 语义，无双实现。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from ctx_weft.protocols.events import (
    TRANSIENT_EVENT_TYPES,
    PersistenceUnavailableError,
    StoredEvent,
)
from ctx_weft.protocols.events import Event

logger = logging.getLogger(__name__)

__all__ = ["CommitGate"]

# 会话不可用回调：(session_id, 原始异常) → 由 Runtime 侧标记健康表（先标记后抛，
# 保证任何并发路径读到的健康状态与异常一致）。
OnUnavailable = Callable[[str, BaseException], None]


class CommitGate:
    """required 提交确认点。线程模型：每个 emit/commit 调用一次，无内部锁——
    串行化由底层 store 的 append_batch 承担（WP2），**不持锁等待回调**（bus 的
    fanout 在 commit 返回之后才发生，派生事件的重入 emit 走自己的提交，无死锁）。
    """

    def __init__(self, event_store: Any, *, on_unavailable: OnUnavailable | None = None):
        self._store = event_store
        self._on_unavailable = on_unavailable

    async def commit(
        self, events: list[Event], *, batch_id: str = "",
    ) -> list[StoredEvent]:
        """确认一批事件的存储提交。

        - 瞬态事件跳过（返回 []）——它们只为实时流而发，不是持久事实；
        - 单事件调用（窗口外 emit）：batch_id 缺省取 event.id（与 ``append`` 兼容键一致）；
        - 批次调用（commit_provisional）：bus 传入 round batch_id，重试不换；
        - 失败：先 ``on_unavailable(session, cause)`` 标记，再抛
          ``PersistenceUnavailableError``——不吞、不重试、不通知。
        """
        non_transient = [e for e in events if e.type not in TRANSIENT_EVENT_TYPES]
        if not non_transient:
            return []
        session_id = non_transient[0].session_id
        for e in non_transient:
            if e.session_id != session_id:
                raise ValueError(
                    f"CommitGate: batch spans sessions ({session_id!r} vs "
                    f"{e.session_id!r})——一个批次必须同属一个 session")
        bid = batch_id or non_transient[0].id
        try:
            receipt = await self._store.append_batch(session_id, bid, non_transient)
        except Exception as exc:
            logger.error(
                "CommitGate: store append failed for session %s batch %s (%d events): %r",
                session_id, bid, len(non_transient), exc)
            if self._on_unavailable is not None:
                try:
                    self._on_unavailable(session_id, exc)
                except Exception:                     # pragma: no cover — 健康标记绝不能反掀提交
                    logger.exception("CommitGate: on_unavailable callback failed")
            raise PersistenceUnavailableError(
                f"event commit not confirmed for session {session_id!r} "
                f"(batch {bid!r}, {len(non_transient)} events); session enters "
                f"storage_unavailable isolation") from exc
        return list(receipt.records)
