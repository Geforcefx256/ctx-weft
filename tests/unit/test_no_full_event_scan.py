"""超长会话不得出现「读整条事件流」的查询。

一次性把整条流变成 `list[Event]` 的代价是实测过的：一条 3 万事件 / 45MB `events` 表的
会话约 **130MB 常驻、读一次 3.5 秒**（SQLite 本地文件；Postgres 走网更慢）。所以

1. 只关心几种事件类型的折叠，必须按类型收窄（HITL 折叠、dangling tool_call 折叠都在这
   一档）——段 recap 折叠曾经是全量读，而它每次 `/resume` 都走一遍
   （`_recover_session_locked`），与快照有没有无关；
2. 快照不可用时那趟**正当的**全量重放也要分批，不把整条流驻留内存。

段 recap 后来**连收窄查询都不发了**：收窄只把斜率降了 15 倍，它仍随会话长度线性增长
（recap 事件只增不减），所以那个账进了投影（`RunStateView.pending_recap`）。这条路的护栏
在 `tests/unit/test_fold_pending_task_recap.py`（折叠 + 快照往返）与
`tests/integration/test_task_recap_recovery.py::test_resume_does_not_read_the_whole_event_stream`
（真实 `/resume` 上一次 recap 查询都不发）。

第 2 条的分批**由 store 产出**（`EventStore.replay`，基类默认实现即真分批——`read_range`
与 `committed_head` 都是必需方法，按 position 区间切就行）。core 只管
`async for batch in store.replay(sid)`，不问「你支不支持分页」、不替谁选降级路：那种能力
探测曾经写在 core 里，一个坏设计生出两个分支和两种失败形态。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import (
    HITL_FOLD_EVENT_TYPES,
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


async def test_typed_read_does_not_touch_the_full_stream() -> None:
    """收窄入口本身：给它一个类型集，它只发一次类型查询、不碰全量。

    ⚠️ 这条**只测入口**，护不住「调用点用错工具」——那种要在真实恢复路径上钉，见
    `tests/integration/test_task_recap_recovery.py::test_resume_does_not_read_the_whole_event_stream`。
    两档都要有：这条钉工具本身的行为，那条钉调用点确实用了它。
    """
    from ctx_weft.core.control.reducers import load_events_of_types

    store = _CountingStore()
    await _seed(store, n_noise=500)

    got = await load_events_of_types(store, _SID, HITL_FOLD_EVENT_TYPES)

    assert store.full_reads == 0, "收窄入口不该读整条事件流"
    assert store.typed_reads == [tuple(str(t) for t in HITL_FOLD_EVENT_TYPES)]
    assert got == [], "503 条噪音里没有 HITL 事件，收窄就该一条都不取回"


async def test_recap_fold_lives_in_the_projection_not_in_a_query() -> None:
    """段 recap 的账只有一处折叠实现，且在 `_apply` 里（进投影）。

    从前它是独立函数 `fold_pending_task_recap` + 恢复路径上一次收窄查询。并存两份折叠正是
    feat 把 HITL 移出投影要治的那类漂移，所以这里钉「旧的那份真的没了」——留着它，下一个人
    就会照旧用法再写一次那条线性增长的查询。
    """
    import ctx_weft.core.control.reducers as mod

    assert not hasattr(mod, "fold_pending_task_recap")
    assert not hasattr(mod, "TASK_RECAP_EVENT_TYPES")

    store = _CountingStore()
    await _seed(store, n_noise=3)
    view = await rebuild_view(store, _SID)

    assert store.typed_reads == [], "重放折出这个账，不发类型查询"
    # _seed 的最后一条是 t_stuck 的 started（无 done），t_done 那对已销账
    assert set(view.pending_recap) == {"t_stuck"}


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
