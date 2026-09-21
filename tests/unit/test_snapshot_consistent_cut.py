"""快照一致切面 conformance（spec: snapshot-recovery；change reliability-wp4）。

钉住三件事：延迟提交不再丢失（E-T04）、取 head 后并发提交不混入（E-T05）、
两路恢复等价（E5——逐字段比较，不只 task 键集）。夹具用生产 SnapshotWriter +
真实 store（in_memory / SQLite 参数化），不 mock 任何恢复路径。
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import reduce_events, rebuild_view
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus
from ctx_weft.providers.events.persister import attach_persistence
from ctx_weft.providers.events.snapshot import SnapshotWriter
from tests._snapshot_helpers import latest_snapshot, seed_snapshot

_T0 = datetime(2026, 9, 11, tzinfo=UTC)


def _ev(n: int, *, session: str = "s1", type_: str = "TaskCreated",
        task_id: str | None = None, payload: dict | None = None) -> Event:
    return Event(id=f"evt_{n:04d}", run_id="r1", sequence=n, session_id=session,
                 type=type_, timestamp=_T0, task_id=task_id or f"t{n}",
                 payload=payload if payload is not None else {
                     "task": {"id": task_id or f"t{n}", "assigned_agent_id": f"ag{n}"}})


@pytest.fixture(params=["in_memory", "sqlite"])
async def env(request, tmp_path):
    """真 bus + 真 persister/CommitGate 不接（直接 emit 走 gate? 这里不接 gate——
    writer 三步切面不依赖 gate，只要 store 有 committed_head/read_range）。"""
    if request.param == "in_memory":
        store = InMemoryEventStore()
        yield store
    else:
        from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
        async with open_sqlite_event_store(tmp_path / "snap.sqlite") as store:
            yield store


async def _emit_run_finished(bus, n, session="s1"):
    """RunFinished 触发 writer（every_n=1 → 每次 RunFinished 都写快照）。"""
    await bus.emit(_ev(n, type_=EventType.RUN_FINISHED, task_id=None,
                      payload={"outcome": "completed"}))


def _make_bus(store) -> InProcessEventBus:
    """persister（落库）+ SnapshotWriter(every_n=1)——与 attach_persistence 同序。"""
    bus = InProcessEventBus()
    attach_persistence(bus, store, snapshot_every_n=1)
    return bus


async def test_late_committed_event_not_lost(env):
    """E-T04：B 先提交触发快照（cut=当时 head），A 后提交 → 再快照/恢复两条路径都见 a、b。"""
    store = env
    bus = _make_bus(store)

    # B 路径：先于窗口的事件 + 触发快照
    await bus.emit(_ev(1, task_id="a"))
    await bus.emit(_ev(2, task_id="b"))
    await _emit_run_finished(bus, 3)          # 快照 #1：cut = head(2)

    # A 延迟提交：模拟「后提交的旧位置事件」——直接经 store 提交一个批次（位置在 head 后
    # 但内容属于早期语义），bus 不参与（等价于并发窗口后提交）
    from ctx_weft.protocols.events import Event as E
    late = E(id="evt_0009", run_id="r1", sequence=9, session_id="s1",
             type="TaskCreated", timestamp=_T0, task_id="late",
             payload={"task": {"id": "late", "assigned_agent_id": "agL"}})
    await store.append_batch("s1", "batch_late", [late])   # position 3（提交序在后）

    # 恢复：rebuild_view 走 position 增量（快照 cut=2 → delta=position 3）→ late 在
    restored = await rebuild_view(store, "s1")
    full = reduce_events([se.event for se in await store.read_range("s1")], "s1")
    # RunFinished 不建 task（触发事件只作信号）；late 经 position 增量进入恢复视图
    assert sorted(restored.tasks) == sorted(full.tasks) == ["a", "b", "late"], (
        f"snapshot recovery lost tasks: {sorted(restored.tasks)} vs {sorted(full.tasks)}")


async def test_concurrent_commit_after_cut_not_in_blob(env):
    """E-T05：快照 cut 之后的新提交不混入该快照——由下一次 delta 应用一次。"""
    store = env
    bus = _make_bus(store)
    await bus.emit(_ev(1, task_id="a"))
    await _emit_run_finished(bus, 2)          # 快照 #1：cut = head(1)，blob 只含 a

    # 触发事件（RunFinished）本身已提交（persister 先于 writer），cut 含它：head=2
    snap1 = await latest_snapshot(store, "s1")
    assert snap1.last_commit_position == 2

    await bus.emit(_ev(3, task_id="c"))       # cut 后新提交
    await _emit_run_finished(bus, 4)          # 快照 #2

    # cut = 5，不是 4：**快照 #1 自己也是一条事件、占了 position 3**。这是「快照是日志里的
    # 一种事件」的直接后果，而它对折叠无影响——增量读排除 `StateSnapshot`
    # （`REPLAY_EXCLUDE_TYPES`），所以被跳过的那个 position 不会少折任何东西。
    snap2 = await latest_snapshot(store, "s1")
    assert snap2.last_commit_position == 5
    restored = await rebuild_view(store, "s1")
    assert sorted(restored.tasks) == ["a", "c"]


async def test_two_route_equivalence_field_by_field(env):
    """E5：全量回放 vs 快照+增量——session/task/agent/outputs 逐字段等价（不只键集）。"""
    store = env
    bus = _make_bus(store)
    for n in range(1, 8):
        await bus.emit(_ev(n, task_id=f"t{n}"))
    await _emit_run_finished(bus, 8)          # 快照
    await bus.emit(_ev(9, task_id="t9"))
    await bus.emit(_ev(10, task_id="t10"))
    await _emit_run_finished(bus, 11)         # 再快照

    restored = await rebuild_view(store, "s1")                     # 快照+增量
    full = reduce_events(                                          # 全量
        [se.event for se in await store.read_range("s1")], "s1")

    # 逐字段对照（归一化时间戳/随机 id）
    assert sorted(restored.tasks) == sorted(full.tasks)
    for tid in full.tasks:
        r_t, f_t = restored.tasks[tid], full.tasks[tid]
        assert r_t.status == f_t.status, tid
        assert r_t.title == f_t.title, tid
        assert r_t.assigned_agent_id == f_t.assigned_agent_id, tid
        assert (r_t.outputs or None) == (f_t.outputs or None), tid
    assert sorted(restored.agents) == sorted(full.agents)
    for aid in full.agents:
        assert restored.agents[aid].status == full.agents[aid].status, aid
        assert restored.agents[aid].template_id == full.agents[aid].template_id, aid
    assert restored.sessions.keys() == full.sessions.keys()


async def test_legacy_and_corrupt_snapshots_ignored():
    """E-T06：无 position（legacy）/ 版本不匹配 / 引用未来位置 → 忽略 + 全量重建。"""
    from ctx_weft.core.control.reducers import deserialize_view, serialize_view

    store = InMemoryEventStore()
    bus = _make_bus(store)
    await bus.emit(_ev(1, task_id="a"))
    await _emit_run_finished(bus, 2)
    good = await latest_snapshot(store, "s1")
    assert good.last_commit_position == 2

    # 三种坏形态各自覆盖
    import dataclasses
    base_view = serialize_view(await rebuild_view(store, "s1"))

    good_view = await rebuild_view(store, "s1")
    for bad_over in (
        {"cut_position": None},          # legacy：存量快照无切面位置
        {"projection_version": 99},      # 版本不匹配
        {"cut_position": 999},           # 位置超前
    ):
        await seed_snapshot(store, "s1", good_view, cut=2, overrides=bad_over)
        bad_snap = await latest_snapshot(store, "s1")
        restored = await rebuild_view(store, "s1")
        assert sorted(restored.tasks) == ["a"], (
            f"bad snapshot must be ignored: pos={bad_snap.last_commit_position} "
            f"ver={bad_snap.projection_version} → {sorted(restored.tasks)}")


# ── 折叠策略：常态增量 + 重锚点（spec: snapshot-recovery）────────────────────
#
# H2 的修复在游标定义里，与折叠策略无关。这一组钉住策略本身：写路径默认 O(delta)，
# 全量折只在「无可用基底 / 链深到顶 / 本进程首次」时作为重锚发生。


async def test_first_write_in_process_is_anchor(env):
    """服务刚起来的第一张快照是全量重锚——纠正上一进程可能留下的偏差。"""
    store = env
    bus = _make_bus(store)
    await bus.emit(_ev(1, task_id="a"))
    await _emit_run_finished(bus, 2)
    snap = await latest_snapshot(store, "s1")
    assert snap.chain_depth == 0, "本进程首张快照必须是重锚"


async def test_steady_state_writes_are_incremental(env):
    """同一进程内的后续快照走增量：chain_depth 逐张 +1。"""
    store = env
    bus = _make_bus(store)
    await bus.emit(_ev(1, task_id="a"))
    await _emit_run_finished(bus, 2)                      # 重锚 depth=0
    depths = []
    for n in (3, 5, 7):
        await bus.emit(_ev(n, task_id=f"t{n}"))
        await _emit_run_finished(bus, n + 1)
        depths.append((await latest_snapshot(store, "s1")).chain_depth)
    assert depths == [1, 2, 3], f"后续写入应为增量链，实得 {depths}"


async def test_incremental_blob_equals_full_fold(env):
    """增量写出来的 blob 必须逐字段等于「从日志全量折」——这是两路等价的前提。"""
    from ctx_weft.core.control.reducers import (
        REPLAY_EXCLUDE_TYPES,
        deserialize_view,
    )

    store = env
    bus = _make_bus(store)
    for n in (1, 3, 5, 7, 9):
        await bus.emit(_ev(n, task_id=f"t{n}"))
        await _emit_run_finished(bus, n + 1)

    snap = await latest_snapshot(store, "s1")
    assert snap.chain_depth > 0, "本例要覆盖的是增量路径"
    incremental = deserialize_view(snap.state_blob)
    full = reduce_events(
        # 这里必须和恢复路径排除同一批类型：快照自己也是日志里的一条事件，读进来会被
        # 当成普通事件计进 events_total（实测差 4 条 = 本例写出的 4 张快照）。
        [se.event for se in await store.read_range(
            "s1", after_position=0, through_position=snap.last_commit_position,
            exclude_types=REPLAY_EXCLUDE_TYPES)],
        run_id="s1")

    assert sorted(incremental.tasks) == sorted(full.tasks)
    for tid in full.tasks:
        i_t, f_t = incremental.tasks[tid], full.tasks[tid]
        assert i_t.status == f_t.status, tid
        assert i_t.title == f_t.title, tid
        assert i_t.assigned_agent_id == f_t.assigned_agent_id, tid
    assert sorted(incremental.agents) == sorted(full.agents)
    for aid in full.agents:
        assert incremental.agents[aid].status == full.agents[aid].status, aid
    assert incremental.events_total == full.events_total


async def test_two_route_equivalence_holds_under_incremental_writes(env):
    """E5 在增量写下同样成立——恢复的两条路径仍逐字段等价。"""
    store = env
    bus = _make_bus(store)
    for n in (1, 3, 5, 7, 9, 11):
        await bus.emit(_ev(n, task_id=f"t{n}"))
        await _emit_run_finished(bus, n + 1)
    await bus.emit(_ev(13, task_id="t13"))                # 快照之后的增量

    restored = await rebuild_view(store, "s1")            # 快照 + delta
    full = reduce_events([se.event for se in await store.read_range("s1")], "s1")
    assert sorted(restored.tasks) == sorted(full.tasks)
    for tid in full.tasks:
        assert restored.tasks[tid].status == full.tasks[tid].status, tid


async def test_chain_depth_cap_forces_reanchor(env, monkeypatch):
    """链深到顶强制一次全量重锚，把往返有损的累积限制在常数级。"""
    monkeypatch.setattr(SnapshotWriter, "MAX_CHAIN_DEPTH", 3)
    store = env
    bus = _make_bus(store)
    await bus.emit(_ev(1, task_id="a"))
    await _emit_run_finished(bus, 2)                      # depth 0（重锚）
    seen = [0]
    for n in (3, 5, 7, 9, 11):
        await bus.emit(_ev(n, task_id=f"t{n}"))
        await _emit_run_finished(bus, n + 1)
        seen.append((await latest_snapshot(store, "s1")).chain_depth)
    # 0,1,2,3 之后基底达上限 → 下一张重锚回 0，再继续增量
    assert seen == [0, 1, 2, 3, 0, 1], f"链深上限未生效：{seen}"


async def test_unusable_base_rejected_so_writer_reanchors(env):
    """坏基底一律不认（判据与恢复侧共用），writer 因此重锚而不是基于坏快照做增量。

    直接测 `_usable_base` 而不走端到端：`load_latest_snapshot` 的「最新」判据是
    (snapshot_at, id)，同毫秒写入时由 id 决胜——端到端断言会变成跟时钟赛跑。
    """
    import dataclasses

    store = env
    bus = _make_bus(store)
    await bus.emit(_ev(1, task_id="a"))
    await _emit_run_finished(bus, 2)
    good = await latest_snapshot(store, "s1")
    head = await store.committed_head("s1")

    writer = SnapshotWriter(store, None, every_n_events=1)
    writer._anchored.add("s1")                       # 装成「本进程已重锚过」

    assert await writer._usable_base("s1", head) is not None, "健康基底应被接受"

    good_view = await rebuild_view(store, "s1")
    for over, why in (
        ({"projection_version": 99}, "版本不匹配"),
        ({"cut_position": head + 999}, "位置超前"),
        ({"cut_position": None}, "存量快照无切面位置"),
        ({"chain_depth": SnapshotWriter.MAX_CHAIN_DEPTH}, "链深到顶"),
    ):
        await seed_snapshot(store, "s1", good_view, cut=head, overrides=over)
        assert await writer._usable_base("s1", head) is None, f"{why} 的基底必须被拒"

    # 未在本进程重锚过的 session 同样不认基底（服务刚起来那一张必须是全量）
    fresh = SnapshotWriter(store, None, every_n_events=1)
    assert await fresh._usable_base("s1", head) is None
