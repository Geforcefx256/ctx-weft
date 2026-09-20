"""超长会话不得出现「读整条事件流」的查询。

一次性把整条流变成 `list[Event]` 的代价是实测过的：一条 3 万事件 / 45MB `events` 表的
会话约 **130MB 常驻、读一次 3.5 秒**（SQLite 本地文件；Postgres 走网更慢）。所以

1. 只关心几种事件类型的折叠，必须按类型收窄——段 recap 折叠曾经是全量读，而它每次
   `/resume` 都走一遍（`_recover_session_locked`），与快照有没有无关；
2. 快照不可用时那趟**正当的**全量重放也要分批，不把整条流驻留内存。

第 2 条的分批**由 store 产出**（`EventStore.replay`，基类默认实现即真分批——`read_range`
与 `committed_head` 都是必需方法，按 position 区间切就行）。core 只管
`async for batch in store.replay(sid)`，不问「你支不支持分页」、不替谁选降级路：那种能力
探测曾经写在 core 里，一个坏设计生出两个分支和两种失败形态。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import (
    TASK_RECAP_EVENT_TYPES,
    rebuild_view,
    reduce_events,
)
from ctx_weft.protocols.events import Event, EventStore, EventType
from ctx_weft.providers.events import InMemoryEventStore

#: 分批大小现在是 store 的事（`EventStore.REPLAY_BATCH`），core 不持有它。
_REPLAY_BATCH = EventStore.REPLAY_BATCH

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID = "s_long"


def _ev(n: int, type_: str, **payload) -> Event:
    return Event(
        id=f"evt_{n:08d}", run_id="r1", sequence=n, session_id=_SID, type=type_,
        timestamp=_T0, tenant_id="acme", payload=payload,
    )


class _CountingStore(InMemoryEventStore):
    """记下每条读路径被调了几次——断言的是**读法**，不只是结果。"""

    def __init__(self) -> None:
        super().__init__()
        self.full_reads = 0
        self.typed_reads: list[tuple[str, ...]] = []
        self.ranges: list[tuple[int, int | None]] = []

    async def read_by_session(self, session_id: str):
        self.full_reads += 1
        return await super().read_by_session(session_id)

    async def read_session_events_of_types(self, session_id: str, types):
        self.typed_reads.append(tuple(str(t) for t in types))
        return await super().read_session_events_of_types(session_id, types)

    async def read_range(self, session_id: str, *, after_position=0, through_position=None):
        self.ranges.append((after_position, through_position))
        return await super().read_range(
            session_id, after_position=after_position, through_position=through_position)


async def _seed(store: InMemoryEventStore, n_noise: int) -> None:
    """一条「长会话」：大量与段 recap 无关的噪音事件 + 两对 recap 事件。"""
    await store.append(_ev(0, EventType.SESSION_CREATED, user_prompt="go",
                           template_id="agent:tpl", root_agent_id="agt_root"))
    for i in range(1, n_noise + 1):
        await store.append(_ev(i, EventType.RUN_FINISHED, outcome="completed"))
    # 一对完成的（started + done）+ 一对被崩溃打断的（只有 started）
    await store.append(_ev(n_noise + 1, EventType.TASK_RECAP_STARTED,
                           task_id="t_done", boundary="finish", agent_id="agt_root"))
    await store.append(_ev(n_noise + 2, EventType.TASK_RECAP_DONE, task_id="t_done"))
    await store.append(_ev(n_noise + 3, EventType.TASK_RECAP_STARTED,
                           task_id="t_stuck", boundary="finish", agent_id="agt_root"))


# ── 1. 段 recap 折叠：按类型收窄，不碰全量 ──────────────────────────────────


async def test_recap_fold_reads_only_the_two_recap_types() -> None:
    """收窄入口本身：给它那个类型集，它只发一次类型查询、不碰全量。

    ⚠️ 这条**只测入口**，护不住「调用点用错工具」——那条在
    `tests/integration/test_task_recap_recovery.py::test_resume_does_not_read_the_whole_event_stream`，
    它走真实的 `/resume` → `restore_session` 路径。两条都要有：这条钉工具本身的行为，
    那条钉真实调用点确实用了它。
    """
    from ctx_weft.core.control.reducers import load_events_of_types

    store = _CountingStore()
    await _seed(store, n_noise=500)

    got = await load_events_of_types(store, _SID, TASK_RECAP_EVENT_TYPES)

    assert store.full_reads == 0, "不该为段 recap 折叠读整条事件流"
    assert store.typed_reads == [tuple(str(t) for t in TASK_RECAP_EVENT_TYPES)]
    assert len(got) == 3, f"503 条噪音里只该取回那两类共 3 条，实际 {len(got)}"


async def test_recap_type_set_matches_what_the_fold_actually_reads() -> None:
    """类型集与折叠实现必须同步——漏一个类型就会静默少折出一个待重跑的段。"""
    import inspect

    from ctx_weft.core.control.reducers import fold_pending_task_recap

    src = inspect.getsource(fold_pending_task_recap)
    for t in TASK_RECAP_EVENT_TYPES:
        assert t.name in src, f"{t.name} 在类型集里但折叠实现没读它"
    declared = {t.name for t in TASK_RECAP_EVENT_TYPES}
    for line in src.splitlines():
        if "EventType.TASK_" in line:
            name = line.split("EventType.")[1].split(":")[0].split()[0].strip(" :")
            assert name in declared, f"折叠读了 {name}，但它不在 TASK_RECAP_EVENT_TYPES 里"


# ── 2. 快照不可用时的全量重放：按 position 区间分批 ─────────────────────────


async def test_full_replay_is_batched_by_position_range() -> None:
    """无可用快照时由 store 分多批产出，**一次全量读都不发**。

    区间的首尾相接由 `EventStore.replay` 的默认实现负责；这里连带验证它，因为 core 正是
    靠「每批首尾相接、末批上界 == head」才等价于整批折叠。
    """
    store = _CountingStore()
    await _seed(store, n_noise=_REPLAY_BATCH * 2 + 17)   # 跨 3 批，末批不满

    view = await rebuild_view(store, _SID)

    assert store.full_reads == 0, "分批路径不该触发 read_by_session"
    assert len(store.ranges) >= 3, f"应分多批，实际 {len(store.ranges)} 批"
    # 区间首尾相接、不重不漏，且都锚在同一个 head 上
    head = await store.committed_head(_SID)
    assert store.ranges[0][0] == 0
    assert store.ranges[-1][1] == head
    for (_, prev_upper), (next_lower, _) in zip(store.ranges, store.ranges[1:]):
        assert prev_upper == next_lower, f"区间断裂：{store.ranges}"
    assert view.session_id == _SID


async def test_batched_replay_equals_one_shot_replay() -> None:
    """分批与整批**逐字段等价**——重放是左折叠、可结合，这条是那个论证的可执行版本。

    不等价的话，恢复出来的状态会依赖「这次走了哪条路」，那种 bug 极难归因。
    """
    store = _CountingStore()
    await _seed(store, n_noise=_REPLAY_BATCH + 5)

    batched = await rebuild_view(store, _SID)
    head = await store.committed_head(_SID)
    stored = await store.read_range(_SID, after_position=0, through_position=head)
    one_shot = reduce_events([se.event for se in stored], run_id=_SID)

    assert batched.session_id == one_shot.session_id
    assert batched.events_total == one_shot.events_total
    assert batched.tasks_total == one_shot.tasks_total
    assert sorted(batched.tasks) == sorted(one_shot.tasks)
    assert sorted(batched.agents) == sorted(one_shot.agents)
    assert batched.session_status == one_shot.session_status
    assert batched.task_status == one_shot.task_status


async def test_replay_of_an_empty_session_does_not_query_at_all() -> None:
    """空会话（head == 0）：一个区间查询都不该发。"""
    store = _CountingStore()

    view = await rebuild_view(store, "s_empty")

    assert store.ranges == [], f"head 为 0 时不该发区间查询，实际 {store.ranges}"
    assert store.full_reads == 0
    assert view.tasks == {}
