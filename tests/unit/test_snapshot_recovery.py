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


async def test_inmemory_snapshot_roundtrip_keeps_latest() -> None:
    store = InMemoryEventStore()
    assert await store.load_latest_snapshot("s1") is None

    snap1 = RunSnapshot(
        id="snp_1", run_id="run_1", session_id="s1",
        last_event_id="evt_0003", last_event_sequence=3, state_blob={"v": 1},
    )
    await store.save_snapshot(snap1)
    assert (await store.load_latest_snapshot("s1")).id == "snp_1"

    snap2 = RunSnapshot(
        id="snp_2", run_id="run_1", session_id="s1",
        last_event_id="evt_0006", last_event_sequence=6, state_blob={"v": 2},
    )
    await store.save_snapshot(snap2)
    got = await store.load_latest_snapshot("s1")
    assert got.id == "snp_2" and got.state_blob == {"v": 2}
    # 其它 session 不受影响
    assert await store.load_latest_snapshot("s2") is None


async def test_rebuild_view_snapshot_plus_delta_matches_full_replay() -> None:
    events = _session_events()
    store = InMemoryEventStore()
    for ev in events:
        await store.append(ev)

    full = reduce_events(events, run_id="s1")

    # 在第 6 条（RunFinished）处建快照，模拟 SnapshotWriter 的定期写入。
    head = events[:6]
    view_at_6 = reduce_events(head, run_id="s1")
    await store.save_snapshot(RunSnapshot(
        id="snp_mid", run_id="run_1", session_id="s1",
        last_event_id=head[-1].id, last_event_sequence=head[-1].sequence,
        state_blob=serialize_view(view_at_6),
    ))

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
