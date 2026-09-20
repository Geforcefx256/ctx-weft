"""裁过的快照在恢复路径上仍然可用（spec: snapshot-recovery v2，2026-09-19）。

`prune_view_for_snapshot` 的单元覆盖在 `tests/unit/test_snapshot_prune.py`；这里钉的是
它**经真 SnapshotWriter 落盘、再被 rebuild_view 当基底读回**之后的两件事：

1. blob 里确实不再有那些无消费者的历史 task（正向确认裁剪真的发生在生产路径上）。
2. 「所有 task 都已终态」的会话恢复时**不抛错**。

第 2 条是这次改动唯一的行为回归面：裁剪之后这类会话的 `view.tasks` 是空的，而恢复路径
原来的空投影闸门判的是 `not all_tasks`——那会把一个正常完工的会话当成「SESSION_CREATED
之后就崩了的坏投影」，抛 RuntimeError("has no resumable tasks")，也就是
`test_task_recap_recovery.py` docstring 里记的那个「resume 点了没反应」的老 bug。判据因此
改成 `view.tasks_total`（这个会话创建过几个 task），集合空不空与它无关。

⚠️ 这条**必须有快照参与才测得到**：直接 append 事件再 `rebuild_view` 走的是全量回放，
`view.tasks` 是全量的，裁剪根本不参与——`test_task_recap_recovery.py` 的 Test A 就是这样，
所以它在旧判据下同样是绿的，护不住这条。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ctx_weft.core.control.reducers import rebuild_view, snapshot_is_usable
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore, InProcessEventBus
from ctx_weft.providers.events.persister import attach_persistence

_T0 = datetime(2026, 9, 19, tzinfo=timezone.utc)
_SID = "s_pruned"


def _ev(seq: int, type_: str, *, task_id: str | None = None, **payload) -> Event:
    return Event(
        id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=_SID, type=type_,
        timestamp=_T0, tenant_id="default", task_id=task_id, agent_id="agt_root",
        payload=payload, metadata={},
    )


def _created(seq: int, tid: str, *, parent: str = "") -> Event:
    return _ev(seq, EventType.TASK_CREATED, task_id=tid, task={
        "id": tid, "status": "PENDING", "title": f"T-{tid}", "parent_task_id": parent,
        "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root",
        "user_prompt": "P" * 300,
    })


async def _finished_session() -> tuple[InMemoryEventStore, InProcessEventBus]:
    """一个「全部 task 已终态」的会话，并让 SnapshotWriter 真写一张快照。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    attach_persistence(bus, store, snapshot_every_n=1)   # 每条 RunFinished 都写

    await bus.emit(_ev(1, EventType.SESSION_CREATED, user_prompt="go",
                       template_id="agent:tpl_echo", root_agent_id="agt_root"))
    await bus.emit(_created(2, "root"))
    await bus.emit(_created(3, "helper", parent="root"))
    await bus.emit(_ev(4, EventType.TASK_STARTED, task_id="root",
                       assigned_agent_id="agt_root"))
    await bus.emit(_ev(5, EventType.TASK_FINISHED, task_id="helper", outputs="done"))
    await bus.emit(_ev(6, EventType.TASK_FINISHED, task_id="root", outputs="all done"))
    # 触发快照：此刻 root / helper 都是终态 → 活闭包为空
    await bus.emit(_ev(7, EventType.RUN_FINISHED, outcome="completed"))
    return store, bus


@pytest.mark.asyncio
async def test_writer_really_prunes_terminal_tasks_out_of_the_blob() -> None:
    store, _bus = await _finished_session()

    snap = await store.load_latest_snapshot(_SID)
    assert snap is not None, "SnapshotWriter 应当在 RunFinished 上写了一张"
    assert snapshot_is_usable(snap, await store.committed_head(_SID)), (
        "这张快照必须是可用基底，否则下面的断言测不到增量路径")

    # 两个 task 都终态、都没有活父、也不被任何活 task 依赖 → blob 里一个都不留
    assert snap.state_blob["tasks"] == {}
    # 但「创建过 2 个」这件事仍在 blob 里
    assert snap.state_blob["tasks_total"] == 2


@pytest.mark.asyncio
async def test_rebuild_from_pruned_snapshot_keeps_the_created_count() -> None:
    """恢复路径读回裁过的 blob：tasks 空，但 tasks_total 如实——闸门据后者判断。"""
    store, _bus = await _finished_session()

    view = await rebuild_view(store, _SID)
    assert view.tasks == {}
    assert view.tasks_total == 2
    # session 投影本身没被裁，恢复仍拿得到模板等必需信息
    assert view.sessions[_SID].template_id == "agent:tpl_echo"
    assert view.sessions[_SID].root_agent_id == "agt_root"


@pytest.mark.asyncio
async def test_live_task_survives_pruning_through_the_writer() -> None:
    """对照组：会话里还有活 task 时，它必须**全字段**留在 blob 里（driver 要读 prompt），
    而同会话里那些纯历史 task 仍被裁掉。"""
    store = InMemoryEventStore()
    bus = InProcessEventBus()
    attach_persistence(bus, store, snapshot_every_n=1)

    await bus.emit(_ev(1, EventType.SESSION_CREATED, user_prompt="go",
                       template_id="agent:tpl_echo", root_agent_id="agt_root"))
    await bus.emit(_created(2, "old"))
    await bus.emit(_ev(3, EventType.TASK_FINISHED, task_id="old", outputs="x"))
    await bus.emit(_created(4, "alive"))
    await bus.emit(_ev(5, EventType.RUN_FINISHED, outcome="completed"))

    snap = await store.load_latest_snapshot(_SID)
    assert snap is not None
    tasks = snap.state_blob["tasks"]
    assert sorted(tasks) == ["alive"], "只有活 task 该留下"
    assert tasks["alive"]["user_prompt"] == "P" * 300, "活 task 不降级字段"
    assert snap.state_blob["tasks_total"] == 2
