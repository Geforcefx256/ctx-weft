"""回归：`AgentView.current_task_id` 必须在 `TASK_CREATED` 就更新，不等 `TASK_STARTED`。

内存那边有一个**无事件的写口**：`_start_task_for_agent` push 完新 task 会立刻
`ALM.set_current_task(agent_id, task.id)`（"task 还没派发、但路由已经必须认它"）。
若投影只从 `AGENT_*` 折这个字段，push 之后、`TASK_STARTED` 之前这段窗口里投影里还是
上一个**已终态**的 task；而 `ALM.load()` 是无条件整条覆盖 record 的，这段窗口内任何
一次热重装（`/resume`、冷 HITL 应答、`send_message` 的自愈重建都会调
`_load_agents_of`）都会把 `send_message` 的路由判据倒回去 → 下一条消息又新建一个
task，而刚才那个还在队列里：同一个 agent 挂两个 task。
"""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.protocols.events import Event, EventType
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio
_TS = datetime(2026, 6, 13, tzinfo=UTC)


def _ev(seq, sid, type_, *, agent_id=None, task_id=None, **payload):
    # id 前缀刻意用 `evt_0000…`：`InMemoryEventStore.read_by_session` 按**事件 id**
    # （ULID 字典序）排序，不按 sequence。手造 id 若排在真实 ULID（`evt_01M2…`）之后，
    # 本用例后面由 runtime 真发出来的事件会被折在种子事件**之前**，回放顺序颠倒。
    return Event(id=f"evt_0000{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                 type=type_, timestamp=_TS, task_id=task_id, agent_id=agent_id, payload=payload)


def _runtime():
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    rt = make_runtime(agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def test_task_created_updates_the_assigned_agents_current_task():
    rt = _runtime()
    sid, aid = "A", "agt_1"
    await append_one(rt.event_store, _ev(1, sid, EventType.SESSION_CREATED,
                                    template_id="agent:tpl_echo", root_agent_id=aid))
    await append_one(rt.event_store, _ev(2, sid, EventType.AGENT_INSTANTIATED, agent_id=aid,
                                    template_id="agent:tpl_echo"))
    # 上一轮：跑过 tsk_old，回 idle（task 已终态）
    await append_one(rt.event_store, _ev(3, sid, EventType.AGENT_RUNNING, agent_id=aid, task_id="tsk_old"))
    await append_one(rt.event_store, _ev(4, sid, EventType.AGENT_IDLE, agent_id=aid, task_id="tsk_old"))
    # 新一轮 push 了，但还没派发（没有 TASK_STARTED / AgentRunning）
    await append_one(rt.event_store, _ev(5, sid, EventType.TASK_CREATED, task_id="tsk_new",
                                    task={"id": "tsk_new", "status": "ACTIVE",
                                          "assigned_agent_id": aid, "creator_agent_id": aid}))

    view = await rebuild_view(rt.event_store, sid)
    assert view.agents[aid].current_task_id == "tsk_new"


async def test_load_no_longer_rewinds_current_task_id(monkeypatch):
    """端到端形态：`send_message` push 完新 task，此刻发生一次热重装，路由判据不得倒回。"""
    from ctx_weft.core.orchestrator.task.manager import TaskManager

    async def _noop_drain(self):   # 停在 push 之后、TASK_STARTED 之前
        return None
    monkeypatch.setattr(TaskManager, "drain", _noop_drain)

    rt = _runtime()
    sid, aid = "A", "agt_1"
    await append_one(rt.event_store, _ev(1, sid, EventType.SESSION_CREATED,
                                    template_id="agent:tpl_echo", root_agent_id=aid))
    await append_one(rt.event_store, _ev(2, sid, EventType.AGENT_INSTANTIATED, agent_id=aid,
                                    template_id="agent:tpl_echo"))
    await append_one(rt.event_store, _ev(3, sid, EventType.AGENT_RUNNING, agent_id=aid, task_id="tsk_old"))
    await append_one(rt.event_store, _ev(4, sid, EventType.AGENT_IDLE, agent_id=aid, task_id="tsk_old"))
    await rebuild_all_active(rt)

    reg = rt._agent_lifecycle_manager
    assert reg.record_of(aid).current_task_id == "tsk_old"

    first = await rt.send_message(aid, "new question", session_id=sid)
    assert reg.record_of(aid).current_task_id == first.task_id

    # 热重装（/resume、冷 HITL 应答、并发 send_message 的自愈都会走到这里）
    await rt._load_agents_of(sid, tenant_id="default")  # 返回 (条数, tenant)
    assert reg.record_of(aid).current_task_id == first.task_id, "路由判据被热重装倒回了"

    # 判据没倒回 → 下一条消息并进同一个 task，不再凭空多开一个
    second = await rt.send_message(aid, "another question", session_id=sid)
    assert second.task_id == first.task_id
