"""冷启动装填 ALM + 恢复期 AGENT_* 现状广播（2026-09-04 spec §6.3 / §6.4）。

「恢复不是一种状态」的纪律保持：发的是从事件折出来的**现状**，不是新状态，
不引入 RECOVERING 之类的值域。

搭台手法照抄 `tests/unit/test_recover_routing.py`——直接把事件写进 event store
再调 `recover()`，不去跑一个真会话再模拟进程重启（`recover()` 的契约就是「只读
事件日志」，唯一需要搭的台就是事件日志本身，见该文件模块 docstring）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ctx_weft.protocols.events import Event, EventType
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit._legacy_recover import rebuild_all_active
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 6, 13, tzinfo=timezone.utc)


def _ev(seq: int, sid: str, type_: EventType, *, agent_id: str | None = None, **payload) -> Event:
    return Event(id=f"evt_{sid}_{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                 type=type_, timestamp=_TS, task_id="t1", agent_id=agent_id, payload=payload)


async def _seed_crashed_session(store, sid: str, root_agent_id: str) -> None:
    """写一个可 `recover()` 的会话事件流：SessionCreated + 一个 root agent，带一条未决 HITL。

    reducer 折 `AgentView` 靠的是 `AGENT_INSTANTIATED`（装填 template_id）+ 五态机
    状态事件（装填 status）——`_rebuild_agents` 只推算树形字段，两者都要种，否则
    `view.agents` 折不出这个 agent，见 `core/control/reducers.py`。这里额外种一条
    `HITL_REQUIRED`，让 root agent 折出来的 `waiting_human` 状态不是巧合：有一条真
    未决 HITL 挂着。
    """
    await append_one(store, _ev(1, sid, EventType.SESSION_CREATED,
                            template_id="tpl_x", root_agent_id=root_agent_id))
    await append_one(store, _ev(2, sid, EventType.AGENT_INSTANTIATED, agent_id=root_agent_id,
                            template_id="tpl_x"))
    await append_one(store, _ev(3, sid, EventType.HITL_REQUIRED, hitl_id=f"h_{sid}", form="question"))
    await append_one(store, _ev(4, sid, EventType.AGENT_WAITING_HUMAN, agent_id=root_agent_id))


async def test_recover_populates_agent_registry() -> None:
    """核心回归：重启后 list_agents 立刻看得见，不必等某条冷应答。"""
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    await _seed_crashed_session(rt.event_store, "S1", "agt_root")

    assert rt.list_agents(session_id="S1") == []      # 装填前
    await rebuild_all_active(rt)
    ids = {a.agent_id for a in rt.list_agents(session_id="S1")}
    assert "agt_root" in ids


async def test_recover_returns_agent_count_not_session_count() -> None:
    """报告单位换成 agent（spec §6.2）。

    两个 session：S1 挂 1 个 agent，S2 挂 2 个 agent（root + 一个额外子 agent）——
    agent 总数 3 与 session 数 2 刻意不相等，防止巧合数值让断言退化成没意义的检查。
    """
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    store = rt.event_store
    await _seed_crashed_session(store, "S1", "agt_root_1")
    await _seed_crashed_session(store, "S2", "agt_root_2")
    await append_one(store, _ev(5, "S2", EventType.AGENT_INSTANTIATED, agent_id="agt_child_2",
                            template_id="tpl_x"))
    await append_one(store, _ev(6, "S2", EventType.AGENT_IDLE, agent_id="agt_child_2"))

    n = await rebuild_all_active(rt)

    assert n == 3
    assert n == len(rt.list_agents())


async def test_get_agent_works_right_after_recover() -> None:
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    await _seed_crashed_session(rt.event_store, "S1", "agt_root")
    await rebuild_all_active(rt)
    d = rt.get_agent("agt_root")
    assert d.session_id == "S1"


async def test_load_broadcasts_current_status() -> None:
    """装填完按折出来的现状发 AGENT_*，host 投影因此不会停在崩溃前的状态。"""
    seen: list[Event] = []

    async def recorder(ev: Event) -> None:
        seen.append(ev)

    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    rt.event_bus.subscribe(None, recorder)
    await _seed_crashed_session(rt.event_store, "S1", "agt_root")

    await rebuild_all_active(rt)

    agent_events = [e.type for e in seen if str(e.type).startswith("Agent")]
    assert any(t in agent_events for t in (
        EventType.AGENT_IDLE, EventType.AGENT_WAITING_HUMAN, EventType.AGENT_INTERRUPTED,
    )), f"恢复期没有 AGENT_* 现状广播: {agent_events}"


async def test_broadcast_carries_the_folded_status_not_a_default() -> None:
    """有未决 HITL 的 agent 恢复后必须是 waiting_human，不是被字段默认值重置成 idle。"""
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    await _seed_crashed_session(rt.event_store, "S1", "agt_root")
    await rebuild_all_active(rt)
    assert rt.get_agent("agt_root").status == "waiting_human"


async def test_crashed_running_is_settled_to_interrupted_and_terminated_stays_silent() -> None:
    """Ruling F 收口 + 2026-09-09 恢复期裁定。

    `running` 折出来的现状**绝不照发**——进程刚起来什么都没派发，发 `AgentRunning` 会让
    host 以为有活在跑。这条测试要挡住的具体改动仍然是：有人往
    `_RECOVERY_BROADCAST_BY_STATUS` 里加一条 `"running": EventType.AGENT_RUNNING`
    （见 fix report 里记录的反向验证）。

    但"不照发"不等于"什么都不说"。折出来的 `running` 在事件流里的真实含义是**崩溃时
    正在跑**，而 core 从前对此保持沉默、把缺口留给宿主自己补（宿主只能去折一遍事件流，
    等于替 core 的五态机下判决）。现在由 `_settle_crashed_agents` 走 `apply_input` 判成
    `interrupted`——那是一次真转移，`interrupted` 本就是为"被打断、可恢复、等 /resume
    重新派发"而存在的。

    `terminated` 仍然零事件：粘滞终态，host 投影本就已经是终态，重发没有信息。
    """
    seen: list[Event] = []

    async def recorder(ev: Event) -> None:
        seen.append(ev)

    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    rt.event_bus.subscribe(None, recorder)
    store = rt.event_store
    await _seed_crashed_session(store, "S1", "agt_root")
    # 额外两个 agent：一个折成 running，一个折成 terminated——同一条 AGENT_INSTANTIATED
    # + 五态机状态事件的种法，只是换了状态事件类型（见 _seed_crashed_session 的折叠依据）。
    await append_one(store, _ev(5, "S1", EventType.AGENT_INSTANTIATED, agent_id="agt_running",
                            template_id="tpl_x"))
    await append_one(store, _ev(6, "S1", EventType.AGENT_RUNNING, agent_id="agt_running"))
    await append_one(store, _ev(7, "S1", EventType.AGENT_INSTANTIATED, agent_id="agt_terminated",
                            template_id="tpl_x"))
    await append_one(store, _ev(8, "S1", EventType.AGENT_TERMINATED, agent_id="agt_terminated"))

    await rebuild_all_active(rt)

    running_events = [e.type for e in seen if getattr(e, "agent_id", None) == "agt_running"]
    assert EventType.AGENT_RUNNING not in running_events, (
        f"绝不能把折出来的 running 照发出去: {running_events}")
    assert running_events == [EventType.AGENT_INTERRUPTED], (
        f"崩溃时在跑的 agent 该被判成 interrupted，实得: {running_events}")
    assert rt.get_agent("agt_running").status == "interrupted"

    terminated_events = [e.type for e in seen if getattr(e, "agent_id", None) == "agt_terminated"]
    assert terminated_events == [], f"unexpected broadcast for agt_terminated: {terminated_events}"

    # 幂等性（再装填一次不重复发 AgentInterrupted）在
    # `tests/integration/test_session_lifecycle_forget_rebuild.py` 里验。**不能在这里验**：
    # 本文件的 `_ev` 造的 id 是 `evt_<sid>_<seq>`，而 `InMemoryEventStore.read_by_session`
    # 按**事件 id** 排序——真 ULID（`evt_01M2…`）会排在 `evt_S1_…` 之前，于是刚发出的
    # `AgentInterrupted` 折叠时反而落在种子事件**前面**，折出来仍是 running。那是 fixture
    # 的假象，不是生产行为（生产里 ULID 单调，后发的必然排在后面）。


# ── 2026-09-04 spec §6.4 / §9：TASK_QUEUE_* 停发 ──────────────────────────


async def test_no_task_queue_events_are_emitted() -> None:
    """三个类型进 L 档：枚举与 reducer 分支保留，但不再有发射点。

    `_seed_crashed_session` 种出一个带未决 HITL 的 root agent（非空装填，避免
    断言在零 agent 上退化成没有意义的检查），`recover()` 走完整条恢复链
    （rebuild_hitl → register_session → ALM.load 广播 AGENT_*）之后，事件流里
    不应再出现任何 TaskQueue* 类型——它们已随 `announce_queue_state` /
    `_announce_queue_state_as_tm_proxy` 一并停发。
    """
    seen: list[Event] = []

    async def recorder(ev: Event) -> None:
        seen.append(ev)

    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    rt.event_bus.subscribe(None, recorder)
    await _seed_crashed_session(rt.event_store, "S1", "agt_root")

    await rebuild_all_active(rt)

    queue_events = [e.type for e in seen if str(e.type).startswith("TaskQueue")]
    assert queue_events == [], f"仍在发 TASK_QUEUE_*: {queue_events}"


def test_task_queue_types_are_deleted_outright() -> None:
    """2026-09-05：三个类型连枚举一并删除，不进 L 档。

    退役闸门（events-v2.md §5 第 2 级）对它们天然成立——`master` 的 `EventType` 里
    从来没有这三个名字，它们只在 2026-09-02→09-04 之间的分支内部存在过，任何从
    master 迁移来的事件流都不可能含有它们。
    """
    from ctx_weft.protocols.events import EVENT_TYPES, L_TIER_EVENT_TYPES
    gone = {"TaskQueueBlocked", "TaskQueueInterrupted", "TaskQueueDrained"}
    assert not (gone & set(EVENT_TYPES))
    assert not (gone & set(L_TIER_EVENT_TYPES))


def test_l_tier_has_fourteen_entries() -> None:
    """20 - 6（3 个 TaskQueue* + 3 个分支内部的 SESSION_* 运行态一并删除）。
    数字写死是为了让「悄悄多停一个」这件事必须显式改测试。"""
    from ctx_weft.protocols.events import L_TIER_EVENT_TYPES
    assert len(L_TIER_EVENT_TYPES) == 14


def test_proxy_announcer_is_gone() -> None:
    from ctx_weft.core.runtime import CtxWeftRuntime
    assert not hasattr(CtxWeftRuntime, "_announce_queue_state_as_tm_proxy")


def test_task_manager_has_no_announce_queue_state() -> None:
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    assert not hasattr(TaskManager, "announce_queue_state")
