"""测试侧读写状态快照的两个小工具。

快照现在是日志里的一条 `StateSnapshot` 事件（不再有 `save_snapshot` /
`load_latest_snapshot` / `event_snapshots` 表）。这里把「取最新那一张」和「播一张」各收成一个
函数，免得每个测试各写一遍两行代码——各写一遍就会各自漂移，而它们是断言的基准。
"""

from __future__ import annotations

from typing import Any

from ctx_weft.core.control.reducers import (
    SnapshotFacts,
    snapshot_event_payload,
    snapshot_facts_from_event,
)
from ctx_weft.protocols.events import EventOrigin, EventType


async def latest_snapshot(store: Any, session_id: str) -> "SnapshotFacts | None":
    """该会话最新那张快照（= position 最大的 `StateSnapshot` 事件）；没有则 None。"""
    stored = await store.read_last_of_type(session_id, EventType.STATE_SNAPSHOT)
    return snapshot_facts_from_event(stored.event if stored else None)


async def seed_snapshot(
    store: Any,
    session_id: str,
    view: Any,
    *,
    cut: int,
    chain_depth: int = 0,
    reason: str = "test",
    tenant_id: str = "default",
    overrides: "dict[str, Any] | None" = None,
) -> None:
    """往日志里播一张快照。

    `overrides` 直接改 payload 的键，专供「坏形态必须被判不可用」那类用例——旧代码用
    `dataclasses.replace(snap, ...)` 造坏快照，快照变成事件之后对应的动作就是改 payload。
    键名见 `reducers` 里的 `_SNAP_*` 常量（`cut_position` / `projection_version` /
    `chain_depth` / `state_blob` / `reason`）。
    """
    from ctx_weft.core.utils.event import new_event

    payload = snapshot_event_payload(
        view, cut=cut, chain_depth=chain_depth, reason=reason)
    if overrides:
        payload.update(overrides)
    ev = new_event(
        EventType.STATE_SNAPSHOT,
        session_id=session_id,
        tenant_id=tenant_id,
        origin=EventOrigin.RUNTIME,
        payload=payload,
    )
    # batch_id 带 payload 指纹：同一 cut 播多张坏形态时不会撞成「已提交、幂等 no-op」
    await store.append_batch(
        session_id, f"seed-snapshot:{session_id}:{cut}:{ev.id}", [ev])
