"""全量重锚的两条约束：内存有界，且不再每次重启都发生（2026-09-20）。

`SnapshotWriter.on_event` 跑在 `EventBus.emit()` 的**内联**路径上（见那个类开头的 ⚠️），所以
它的写入路径必须 O(delta)。那条约束当初只管住了增量分支：重锚分支是一次
`read_range(0..head)` + `reduce_events`，整条流同时在内存里。实测 20 万事件的会话峰值
604.9MB，分批折同一段是 9.6MB（62 倍）——600MB 压在 loop 主路径上，并发几个会话就是会崩的
量级。

而重锚发生的频率从前还被一条策略放大：`_anchored` 强制「服务每次起来的第一张快照走全量
重锚」，于是**每次重启、每个会话**一次 O(n)。去掉之后重锚只在真正需要时发生（无快照 /
载荷畸形 / 版本不匹配 / 位置超前 / 链深到顶）。

两条改动的失效方向不同，所以分两组钉：分批写错会让 blob 不等价（`test_snapshot_consistent_cut`
已经在管两路等价，这里只管「真的分批了」）；而基底策略改错会让重锚回来，那只是慢，不会错
——慢且不报错正是它躲了这么久的原因，所以要一条专门的测试盯着。
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

from ctx_weft.core.control.reducers import (
    REPLAY_BATCH,
    REPLAY_EXCLUDE_TYPES,
    apply_events,
    rebuild_view,
    reduce_events,
    replay_session,
)
from ctx_weft.core.control.snapshot_writer import SnapshotWriter
from ctx_weft.core.control.types import RunStateView
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._snapshot_helpers import latest_snapshot, seed_snapshot
from tests._event_helpers import append_one

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_TYPES = ("LLMResponseFinished", "CapabilityInvoked", "TaskMessageAppended",
          "ActTurnCompleted")


def _ev(seq: int, type_: str | None = None) -> Event:
    return Event(
        id=f"e{seq:06d}", run_id="r", sequence=seq, session_id="s1",
        type=(type_ or (EventType.SESSION_CREATED if seq == 1
                        else _TYPES[seq % len(_TYPES)])),
        timestamp=_T0, task_id=f"tsk_{seq // 10}",
        payload=({"template_id": "t", "root_agent_id": "a"} if seq == 1
                 else {"content": "x"}))


async def _seeded(n: int) -> InMemoryEventStore:
    store = InMemoryEventStore()
    for i in range(1, n + 1):
        await append_one(store, _ev(i))
    return store


# ── 1. 重锚分批 ──────────────────────────────────────────────────────────────


async def test_the_anchor_reads_in_batches_not_all_at_once() -> None:
    """重锚的读是分批的：批数随事件数增长，单批不超过 `REPLAY_BATCH`。

    钉「每批的量」而不是「峰值内存」：内存峰值要靠 tracemalloc，而那东西会把耗时放大近一个
    数量级、在 CI 上也不稳。单批上界是那 62 倍差距的直接成因，钉它就够。
    """
    n = REPLAY_BATCH * 2 + 137          # 跨三批，且最后一批不满
    store = await _seeded(n)
    writer = SnapshotWriter(store, None, every_n_events=1)

    sizes: list[int] = []
    real = store.read_range

    async def spy(session_id, **k):
        got = await real(session_id, **k)
        sizes.append(len(got))
        return got

    store.read_range = spy                          # type: ignore[method-assign]
    await writer._write("s1", _ev(n), reason="anchor")

    assert len(sizes) >= 3, f"没有分批，一次读完了：{sizes}"
    assert max(sizes) <= REPLAY_BATCH, (
        f"有一批超过了 REPLAY_BATCH={REPLAY_BATCH}：{sizes}")
    assert sum(sizes) == n, f"分批把事件读漏或读重了：sum={sum(sizes)} != {n}"


async def test_the_batched_anchor_folds_to_the_same_view() -> None:
    """分批重锚写出的 blob 与一次折完全一致——左折叠可结合。"""
    from ctx_weft.core.control.reducers import deserialize_view

    n = REPLAY_BATCH + 50
    store = await _seeded(n)
    writer = SnapshotWriter(store, None, every_n_events=1)
    await writer._write("s1", _ev(n), reason="anchor")

    snap = await latest_snapshot(store, "s1")
    assert snap is not None and snap.chain_depth == 0, "本例要覆盖重锚路径"
    batched = deserialize_view(snap.state_blob)

    one_shot = reduce_events(
        [se.event for se in await store.read_range(
            "s1", after_position=0, through_position=snap.last_commit_position,
            exclude_types=REPLAY_EXCLUDE_TYPES)],
        run_id="s1")
    assert batched.events_total == one_shot.events_total == n
    assert sorted(batched.tasks) == sorted(one_shot.tasks)
    assert sorted(batched.agents) == sorted(one_shot.agents)


async def test_the_anchor_stops_at_the_declared_cut() -> None:
    """重锚只折到它声明的那个切面，一条不多。

    这是 `replay_session` 必须收 `through_position` 的原因：不传的话它自己去取
    `committed_head`，而切面 C 是 `_write` 开头取的那个 head。两者之间若又提交了新事件，就会
    把 (C, head'] 也折进去——blob 领先于它声明的切面，恢复时那段被重复 apply，而 apply 不是
    幂等的（`events_total` 会双计）。
    """
    n = REPLAY_BATCH + 10
    store = await _seeded(n)
    writer = SnapshotWriter(store, None, every_n_events=1)

    # 注入点必须落在**真正的危险窗口**里：`_write` 取完自己的 head 之后、`replay_session` 若
    # 自作主张再取一次 head 之前。所以钩 `committed_head`——第一次调用是 `_write` 自己的，返回
    # 之后立刻塞 25 条新事件；此时若 `replay_session` 没收到 `through_position`，它取到的就是
    # n+25，多折进去的那 25 条会让 blob 领先于声明的切面。
    #
    # ⚠️ 第一版钩的是 `read_range`（在折叠途中注入）。那个位置钉不住：不传
    # `through_position` 时 `replay_session` 取 head 发生在第一次 `read_range` **之前**，于是
    # 注入永远在窗口之外，去掉参数测试照样绿。反向验证时发现的。
    real_head = store.committed_head
    injected = False

    async def spy(session_id):
        nonlocal injected
        got = await real_head(session_id)
        if not injected:
            injected = True
            for i in range(n + 1, n + 26):
                await append_one(store, _ev(i))
        return got

    store.committed_head = spy                      # type: ignore[method-assign]
    await writer._write("s1", _ev(n), reason="anchor")
    store.committed_head = real_head                # type: ignore[method-assign]

    snap = await latest_snapshot(store, "s1")
    from ctx_weft.core.control.reducers import deserialize_view
    assert snap.last_commit_position == n, "切面被期间的新提交带偏了"
    assert deserialize_view(snap.state_blob).events_total == n, (
        f"折进了切面之后的事件（{deserialize_view(snap.state_blob).events_total} != {n}）"
        "——blob 领先于切面，恢复会重复 apply")

    # 恢复照常：快照 + 增量 == 全量
    full = RunStateView(run_id="s1", session_id="", task_id="", agent_id="")
    async for batch in replay_session(store, "s1",
                                      exclude_types=REPLAY_EXCLUDE_TYPES):
        apply_events(batch, full)
    assert (await rebuild_view(store, "s1")).events_total == full.events_total == n + 25


def test_the_anchor_does_not_use_reduce_events() -> None:
    """源码守卫：重锚不许再用「一次读完 + reduce_events」那条路。

    `reduce_events` 收的是 `list[Event]`——用它就意味着整条流同时在内存里，而这个方法跑在
    `EventBus.emit()` 的内联路径上。

    用 AST 数**真实调用**，不数字符串：`_write` 的注释里就写着
    「`reduce_events(evts)` 就是 `apply_events(evts, 空 view)`」那条等价性证明（那是该留的），
    数字符串会把它数进去。这个坑在同一批改动里踩过两次，所以这里直接用 AST。
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(SnapshotWriter._write)))
    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reduce_events" not in called, (
        "重锚又变成一次折了：reduce_events 收整个 list，而这里跑在 emit() 内联路径上")
    assert "apply_events" in called, "分批折叠用的就是它"
    # 分批本身现在由 `reducers.replay_session` 提供（2026-09-21 从协议搬进 core，并加上
    # `through_position`——重锚需要显式截断到自己的切面，协议版给不了那个参数，所以这个循环
    # 一度是就地抄的）。钉「调了它」而不是「有个 while 循环」：形状换了，约束没换。
    assert "replay_session" in called, "重锚没走 replay_session——又把分批循环抄了一遍？"


# ── 2. 不再每次重启都重锚 ────────────────────────────────────────────────────


async def test_a_fresh_writer_accepts_the_previous_processes_snapshot() -> None:
    """刚构造的 writer 直接认上一个进程留下的健康快照——不再强制全量重锚。

    从前有个 `_anchored` 集合强制「服务每次起来第一张走全量」。理由是「纠正上一进程可能留下
    的偏差」，但恢复侧本来就信任那张快照（`rebuild_view` 拿它当基底）：读侧信、写侧不信是个
    说不通的不对称，而代价是每次重启每个会话一次 O(n)。限制往返累积偏差本来就是
    `MAX_CHAIN_DEPTH` 的职责。
    """
    store = await _seeded(40)
    view = await rebuild_view(store, "s1")
    head = await store.committed_head("s1")
    await seed_snapshot(store, "s1", view, cut=head)

    fresh = SnapshotWriter(store, None, every_n_events=1)   # 模拟「进程刚起来」
    assert await fresh._usable_base("s1", head) is not None


async def test_a_fresh_writer_still_writes_an_incremental_snapshot() -> None:
    """端到端：进程重启后的第一张快照是**增量**（chain_depth 递增），不是重锚。"""
    store = await _seeded(40)
    first = SnapshotWriter(store, None, every_n_events=1)
    await first._write("s1", _ev(40), reason="anchor")
    assert (await latest_snapshot(store, "s1")).chain_depth == 0

    for i in range(41, 51):
        await append_one(store, _ev(i))
    reborn = SnapshotWriter(store, None, every_n_events=1)  # 新进程
    await reborn._write("s1", _ev(50), reason="periodic")

    snap = await latest_snapshot(store, "s1")
    assert snap.chain_depth == 1, (
        "进程重启后的第一张又走了全量重锚——每次重启每个会话一次 O(n) 回来了")


async def test_the_chain_depth_cap_still_forces_a_reanchor() -> None:
    """去掉的只是「每进程一次」，链深上限这道闸必须还在——它才是管往返偏差的那个。"""
    store = await _seeded(40)
    view = await rebuild_view(store, "s1")
    head = await store.committed_head("s1")
    await seed_snapshot(store, "s1", view, cut=head,
                        chain_depth=SnapshotWriter.MAX_CHAIN_DEPTH)

    writer = SnapshotWriter(store, None, every_n_events=1)
    assert await writer._usable_base("s1", head) is None


def test_the_per_process_reanchor_state_is_gone() -> None:
    """`_anchored` 整个删掉，不是留着不读——留着下一个人会以为它还有用。"""
    store = InMemoryEventStore()
    writer = SnapshotWriter(store, None, every_n_events=1)
    assert not hasattr(writer, "_anchored")
    assert "_anchored" not in inspect.getsource(SnapshotWriter._usable_base)
