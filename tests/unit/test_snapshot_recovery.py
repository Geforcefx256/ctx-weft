"""快照恢复：InMemoryEventStore 快照存取 + rebuild_view 的「快照 + 增量」路径。

覆盖：
1. InMemoryEventStore.save_snapshot / load_latest_snapshot 往返，且只保留每 session 最新一张。
2. rebuild_view 在有快照时走「deserialize(snapshot) + read_range(delta)」，
   结果与全量 reduce_events 完全一致——即快照不改变恢复语义，只省回放量。
3. 无快照时 rebuild_view 退回全量回放（向后兼容）。
"""

from __future__ import annotations

from datetime import datetime, timezone

from ctx_weft.core.control.reducers import rebuild_view, reduce_events, serialize_view
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.events import RunSnapshot
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.events import EventPersister
from tests._snapshot_helpers import latest_snapshot, seed_snapshot


def _ts() -> datetime:
    return datetime(2026, 6, 5, tzinfo=timezone.utc)


def _ev(seq: int, type_: str, **payload) -> Event:
    """构造一个 session=s1 的事件；id 单调，便于断言里按序比对。"""
    task_id = payload.pop("task_id", None)
    return Event(
        id=f"evt_{seq:04d}",
        run_id="run_1",
        sequence=seq,
        session_id="s1",
        type=type_,
        timestamp=_ts(),
        task_id=task_id,
        payload=payload,
    )


def _session_events() -> list[Event]:
    """一段典型 session 生命周期事件流：建会话 → 跑两个 task → 第二个仍在进行。"""
    return [
        _ev(1, EventType.SESSION_CREATED, user_prompt="do it",
            template_id="tmpl_a", root_agent_id="agt_root"),
        _ev(2, EventType.RUN_STARTED),
        _ev(3, EventType.TASK_CREATED, task={
            "id": "tsk_1", "status": "PENDING", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root",
        }),
        _ev(4, EventType.TASK_STARTED, task_id="tsk_1", assigned_agent_id="agt_root"),
        _ev(5, EventType.TASK_FINISHED, task_id="tsk_1"),
        _ev(6, EventType.RUN_FINISHED, final_status="FINISHED"),
        # —— 之后的增量（delta）——
        _ev(7, EventType.TASK_CREATED, task={
            "id": "tsk_2", "status": "PENDING", "title": "T2",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root",
        }),
        _ev(8, EventType.TASK_STARTED, task_id="tsk_2", assigned_agent_id="agt_root"),
    ]


async def test_latest_snapshot_is_the_one_with_the_highest_position() -> None:
    """「最新」= position 最大，会话之间互不干扰。

    从前这条测的是存储 API「留哪一张」（`snapshot_at` 最大、相同则 `id` 最大——写入序 ≠ 时间序
    所以那套 tie-break 当初是必须的）。快照变成日志里的事件之后，那套口径整个消失：和
    `read_range` / `committed_head` 共用同一个序。
    """
    store = InMemoryEventStore()
    assert await latest_snapshot(store, "s1") is None

    v1 = reduce_events([_ev(1, EventType.SESSION_CREATED, user_prompt="a",
                            template_id="agent:tpl", root_agent_id="agt_root")],
                       run_id="s1")
    await seed_snapshot(store, "s1", v1, cut=3, reason="first")
    assert (await latest_snapshot(store, "s1")).snapshot_reason == "first"

    await seed_snapshot(store, "s1", v1, cut=6, reason="second")
    got = await latest_snapshot(store, "s1")
    assert got.snapshot_reason == "second" and got.last_commit_position == 6
    # 其它 session 不受影响
    assert await latest_snapshot(store, "s2") is None


async def test_rebuild_view_snapshot_plus_delta_matches_full_replay() -> None:
    events = _session_events()
    store = InMemoryEventStore()
    for ev in events:
        await store.append(ev)

    full = reduce_events(events, run_id="s1")

    # 在第 6 条（RunFinished）处建快照，模拟 SnapshotWriter 的定期写入。
    head = events[:6]
    view_at_6 = reduce_events(head, run_id="s1")
    await seed_snapshot(store, "s1", view_at_6, cut=6, reason="periodic")

    rebuilt = await rebuild_view(store, "s1")

    # 快照 + 增量 == 全量回放：会话/任务投影逐项一致。
    assert rebuilt.session_status == full.session_status
    assert set(rebuilt.tasks) == set(full.tasks) == {"tsk_1", "tsk_2"}
    assert {tid: t.status for tid, t in rebuilt.tasks.items()} == \
           {tid: t.status for tid, t in full.tasks.items()}
    assert rebuilt.tasks["tsk_1"].status == "FINISHED"
    assert rebuilt.tasks["tsk_2"].status == "ACTIVE"
    assert set(rebuilt.agents) == set(full.agents)


async def test_inmemory_store_drops_transient_token_events() -> None:
    """每 token 一个的流式 delta 不入存储——只为实时流而发，真相在 LLMResponseFinished。

    过滤已搬到 EventPersister（spec 2026-08-29 §6.4）：store.append 本身「让存什么就
    存什么」，这里改经 persister 走一遍，验证的仍是端到端「瞬态事件不落库」的效果。
    """
    store = InMemoryEventStore()
    persister = EventPersister(store)
    await persister.on_event(_ev(1, EventType.SESSION_CREATED, template_id="t", root_agent_id="a"))
    await persister.on_event(_ev(2, EventType.LLM_TOKEN_STREAMED, delta="he"))
    await persister.on_event(_ev(3, EventType.LLM_REASONING_STREAMED, delta="..."))
    await persister.on_event(_ev(4, EventType.LLM_RESPONSE_FINISHED, content="hello"))

    types = [e.type for e in await store.read_by_session("s1")]
    assert types == [EventType.SESSION_CREATED, EventType.LLM_RESPONSE_FINISHED]
    # 非瞬态事件仍正常入库并参与 active 追踪
    assert "s1" in await store.list_active_session_ids()


async def test_detach_stops_receiving_events() -> None:
    """detach 后不再从总线收事件（host 切到 Postgres 后避免孤儿堆积）。

    该能力已搬去 EventPersister（spec 2026-08-29 §6.4）——store 自身不再自订阅。
    """
    bus = InProcessEventBus()
    store = InMemoryEventStore()
    persister = EventPersister(store, bus)

    await bus.emit(_ev(1, EventType.SESSION_CREATED, template_id="t", root_agent_id="a"))
    assert len(await store.read_by_session("s1")) == 1

    await persister.detach()
    await bus.emit(_ev(2, EventType.RUN_FINISHED, final_status="FINISHED"))

    # detach 之后的事件不应再落入本 store
    assert len(await store.read_by_session("s1")) == 1
    # 幂等：重复 detach 不报错
    await persister.detach()


async def test_rebuild_view_without_snapshot_falls_back_to_full_replay() -> None:
    events = _session_events()
    store = InMemoryEventStore()
    for ev in events:
        await store.append(ev)

    rebuilt = await rebuild_view(store, "s1")
    full = reduce_events(events, run_id="s1")
    assert {tid: t.status for tid, t in rebuilt.tasks.items()} == \
           {tid: t.status for tid, t in full.tasks.items()}
