"""状态快照就是日志里的一条 `StateSnapshot` 事件。

换来两件事：

  · **「哪张最新」= position 最大**，与 `read_range` / `committed_head` 共用同一个序。从前要靠
    协议规定一套快照专属口径（`snapshot_at` 最大、相同则 `id` 最大，因为写入序 ≠ 时间序）。
  · **那三个「必须原样往返」的字段进了 payload**（切面位置 / `projection_version` /
    `chain_depth`），而 payload 对 store 是**整体**不透明的——它没法只丢其中一个。从前它们是
    列，store 可能忘了存：m020 补的就是那几列，而丢了它们 `snapshot_is_usable` 恒判不可用，
    恢复永远全量回放且**不报错**。

blob 直接进 payload，不走 `EventBlobStore`：裁剪后实测只有 2.2 KB
（`prune_view_for_snapshot` 把 946 KB 压到 2.2 KB，436×），走 ref 反而要改 `rebuild_view` 的签名
并引入「有 ref、blob 没写成」这个新的崩溃窗口。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import (
    REPLAY_EXCLUDE_TYPES,
    rebuild_view,
    reduce_events,
    snapshot_event_payload,
    snapshot_facts_from_event,
    snapshot_is_usable,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._snapshot_helpers import latest_snapshot, seed_snapshot

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID = "s1"


def _ev(n: int, type_: str, **payload) -> Event:
    # `task_id` 同时放到**事件头**：`_apply` 的 task 分支读的是 `ev.task_id`，只放 payload 里
    # 那条事件会被静默忽略（我第一版就是这么写的，于是 TASK_FINISHED 没生效、状态还是 ACTIVE）。
    return Event(
        id=f"evt_{n:08d}", run_id="r1", sequence=n, session_id=_SID, type=type_,
        timestamp=_T0, tenant_id="acme", task_id=payload.get("task_id"),
        payload=payload,
    )


def _session_view():
    return reduce_events([
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="ag1"),
    ], run_id=_SID)


# ── payload 编解码 ────────────────────────────────────────────────────────────


def test_payload_round_trips_the_three_load_bearing_fields() -> None:
    """切面位置 / `projection_version` / `chain_depth` 必须原样回来。

    这三个就是 m020 补的那几列。它们现在在 payload 里，而 payload 是整体不透明的——所以
    「store 忘了存其中一个」这个故障形态结构上不成立。这条钉住编解码自己不丢。
    """
    from ctx_weft.core.control.reducers import _PROJECTION_VERSION

    payload = snapshot_event_payload(
        _session_view(), cut=42, chain_depth=3, reason="periodic")
    facts = snapshot_facts_from_event(_ev(9, EventType.STATE_SNAPSHOT, **payload))

    assert facts is not None
    assert facts.last_commit_position == 42
    assert facts.projection_version == _PROJECTION_VERSION
    assert facts.chain_depth == 3
    assert facts.snapshot_reason == "periodic"
    assert facts.state_blob["session_id"] == _SID


def test_cut_is_explicit_not_derived_from_the_events_own_position() -> None:
    """切面位置**显式**写在 payload 里，不是「这条事件的 position 减一」。

    并发追加时快照事件拿到的是 `head+k`，k 不定——按 position-1 推会把切面算错，而算错的方向
    可能是**偏大**（把没折进去的事件当成已折），那就是静默的状态丢失。
    """
    payload = snapshot_event_payload(
        _session_view(), cut=7, chain_depth=0, reason="t")

    assert payload["cut_position"] == 7
    # 事件自己的 position 是 99，与切面无关
    facts = snapshot_facts_from_event(_ev(99, EventType.STATE_SNAPSHOT, **payload))
    assert facts.last_commit_position == 7


def test_the_codec_does_not_prune() -> None:
    """编解码器**不裁剪**——裁剪是写入策略，由调用方喂进来之前自己做。

    混进策略之后，任何想「原样存一张 view」的调用方（测试、工具）都会被悄悄改掉内容。
    """
    view = reduce_events([
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="ag1"),
        _ev(2, EventType.TASK_CREATED, task={
            "id": "t_done", "status": "ACTIVE", "title": "T",
            "assigned_agent_id": "ag1", "creator_agent_id": "ag1"}),
        _ev(3, EventType.TASK_FINISHED, task_id="t_done",
            outcome="success", summary="ok"),
    ], run_id=_SID)
    assert view.tasks["t_done"].status == "FINISHED", "前提：这个 task 已终态"

    from ctx_weft.core.control.reducers import prune_view_for_snapshot

    assert prune_view_for_snapshot(view).tasks == {}, (
        "前提：裁剪会把它丢掉——不然这条测不出编码器有没有偷偷裁"
    )

    payload = snapshot_event_payload(view, cut=3, chain_depth=0, reason="t")

    assert "t_done" in payload["state_blob"]["tasks"], (
        "编码器裁剪了——那是写入策略，不该住在这里"
    )


@pytest.mark.parametrize("bad", [
    {"state_blob": "not-a-dict"},
    {"state_blob": None},
])
def test_malformed_payload_folds_to_none_not_an_exception(bad) -> None:
    """载荷畸形 → None，不抛。

    载荷可能是别人写的（存量、外部导入、手工修过的库）。而「这张不可用」的后果只是全量重放
    ——正确，只是慢；抛出去会把一次慢恢复变成一次恢复失败。
    """
    payload = snapshot_event_payload(_session_view(), cut=1, chain_depth=0, reason="t")
    payload.update(bad)

    assert snapshot_facts_from_event(_ev(9, EventType.STATE_SNAPSHOT, **payload)) is None


def test_non_snapshot_events_fold_to_none() -> None:
    """不是这个类型 → None。折叠不会把别的事件误当快照。"""
    assert snapshot_facts_from_event(_ev(1, EventType.RUN_FINISHED)) is None
    assert snapshot_facts_from_event(None) is None


def test_a_non_int_cut_is_treated_as_missing() -> None:
    """切面位置不是整数 → 当作缺失 → `snapshot_is_usable` 判不可用。

    猜一个位置比全量重放危险得多：猜大了就把没折进去的事件当成已折。
    """
    payload = snapshot_event_payload(_session_view(), cut=5, chain_depth=0, reason="t")
    payload["cut_position"] = "5"

    facts = snapshot_facts_from_event(_ev(9, EventType.STATE_SNAPSHOT, **payload))

    assert facts.last_commit_position is None
    assert snapshot_is_usable(facts, head=10) is False


# ── 与恢复路径的接合 ──────────────────────────────────────────────────────────


async def test_recovery_uses_the_snapshot_event() -> None:
    """恢复走「最后一条快照事件 + 增量」，且与全量重放结果一致。"""
    store = InMemoryEventStore()
    batch = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="ag1"),
        _ev(2, EventType.TASK_CREATED, task={
            "id": "t_a", "status": "ACTIVE", "title": "A",
            "assigned_agent_id": "ag1", "creator_agent_id": "ag1"}),
    ]
    await store.append_batch(_SID, "b1", batch)
    view_at_2 = reduce_events(batch, run_id=_SID)
    await seed_snapshot(store, _SID, view_at_2, cut=2, reason="periodic")

    # 快照之后再来一条（它的 position 在快照事件之后）
    await store.append_batch(_SID, "b2", [
        _ev(4, EventType.TASK_CREATED, task={
            "id": "t_b", "status": "PENDING", "title": "B",
            "assigned_agent_id": "ag1", "creator_agent_id": "ag1"}),
    ])

    rebuilt = await rebuild_view(store, _SID)

    assert sorted(rebuilt.tasks) == ["t_a", "t_b"]
    assert (await latest_snapshot(store, _SID)).last_commit_position == 2


async def test_both_recovery_paths_exclude_snapshot_events() -> None:
    """快照事件本身**不参与折叠**——两条路都排除它。

    不排除的代价是双份的：多读每条 2.2 KB 的 blob（那些事件存在正是为了**省**重放），而且
    `events_total` 这类计数会把快照算进去、两条路给出不同的数。
    """
    store = InMemoryEventStore()
    await store.append_batch(_SID, "b1", [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="ag1"),
    ])
    await seed_snapshot(store, _SID, _session_view(), cut=1, reason="a")
    await seed_snapshot(store, _SID, _session_view(), cut=1, reason="b")

    # 全量路径（把快照全判不可用：版本给个不匹配的）
    await seed_snapshot(store, _SID, _session_view(), cut=1,
                        overrides={"projection_version": 99})
    full = await rebuild_view(store, _SID)

    assert full.events_total == 1, (
        f"折进去了快照事件（events_total={full.events_total}，日志里有 4 条快照）"
    )


def test_the_exclusion_set_is_named_in_core() -> None:
    """要排除什么由 **core** 命名——协议不认识这个类型。

    写死在协议默认实现里就把 core 的词表焊进了存储契约：store 从此要懂「什么是快照」，而这次
    改动的全部意图正是让它**不必**懂。
    """
    import inspect

    from ctx_weft.protocols.events import EventStore

    assert REPLAY_EXCLUDE_TYPES == (EventType.STATE_SNAPSHOT,)
    for name in ("replay", "read_range", "read_last_of_type"):
        src = inspect.getsource(getattr(EventStore, name))
        assert "STATE_SNAPSHOT" not in src, f"{name} 里焊进了快照类型"


def test_snapshot_type_is_persisted_and_not_legacy() -> None:
    """快照事件必须**落库**且不是 L 档。

    落进 `TRANSIENT_EVENT_TYPES` 会让它压根不进存储（那个集合的语义就是「不进任何持久化/
    投影/快照路径」）——快照事件要是瞬态的，等于快照机制整个失效，而且不报错。
    """
    from ctx_weft.protocols.events import (
        EVENT_TYPES,
        L_TIER_EVENT_TYPES,
        TRANSIENT_EVENT_TYPES,
    )

    assert EventType.STATE_SNAPSHOT in EVENT_TYPES
    assert EventType.STATE_SNAPSHOT not in TRANSIENT_EVENT_TYPES
    assert EventType.STATE_SNAPSHOT not in L_TIER_EVENT_TYPES


def test_the_writer_appends_directly_instead_of_going_through_the_bus() -> None:
    """写入侧**直接 `append_batch`**，不经 EventBus。

    三个理由，第三个是决定性的：它不是任何人该反应的领域事实；走 bus 会投给 SSE 转译 / 投影
    更新器 / 分析订阅者（都没理由看见一张快照）；而**写入者自己就是 bus 订阅者**，走 bus 会
    递归调回它自己。
    """
    import inspect

    from ctx_weft.providers.events.snapshot import SnapshotWriter

    src = inspect.getsource(SnapshotWriter._write)
    assert "append_batch" in src
    assert "event_bus.emit" not in src and "_bus.emit" not in src


# ── 协议不提及快照 ────────────────────────────────────────────────────────────


def test_the_protocol_has_no_snapshot_api() -> None:
    """`EventStore` 上不得再有任何快照专属方法或类型。

    留着它们不只是「多几行没人用的代码」：一个可选的 `load_latest_snapshot` 会诱导下一个人
    在 core 里探一次「这个 store 支不支持快照」，而那种能力探测在这个仓里已经生出过两次两分支
    两失败形态（`replay` 与 `read_by_session_after`）。方法不存在，那条路就走不通。
    """
    import ctx_weft.protocols.events as mod
    from ctx_weft.protocols.events import EventStore

    for gone in ("save_snapshot", "load_latest_snapshot"):
        assert not hasattr(EventStore, gone), f"{gone} 还在协议上"
    assert not hasattr(mod, "RunSnapshot")

    import ctx_weft.protocols as pkg

    assert not hasattr(pkg, "RunSnapshot"), "还从 protocols 包里导出着"


def test_the_builtin_stores_have_no_snapshot_api_either() -> None:
    """两个内置 store 也清干净——包括那条「保留最新 N 张」的**恢复策略**。

    `_prune_snapshots` 从前住在 store 里，由它自己决定删哪些快照。那是 core 的恢复策略
    （恢复只取最新一张，所以旧的可删），剪错一张的后果是恢复退化甚至找不到可用基底。快照变成
    事件之后，保留归入事件保留策略，store 不再做这个决定。
    """
    from ctx_weft.providers.events import InMemoryEventStore
    from ctx_weft.providers.events.store.sql import SqlEventStore

    for cls in (InMemoryEventStore, SqlEventStore):
        for gone in ("save_snapshot", "load_latest_snapshot", "_prune_snapshots"):
            assert not hasattr(cls, gone), f"{cls.__name__}.{gone} 还在"
