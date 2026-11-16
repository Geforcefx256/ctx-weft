"""用户开口时，`send_message` 的两条投递分支都收口该 agent 名下的旧 park 气泡。

**这个文件测的是投递侧，不是判决侧**（2026-09-27 订正）。判决侧的规则已经换了：判 success
就收气泡（`background_observe._close_park_bubble`，不分 root 与子任务）——task 判完成了，那个
让位入口就用掉了，人后面还想说话由 `send_message` 开新的一轮。文件名沿用旧称（它曾经钉的是
`faedd25` 那条「判决不收」），内容如下。

投递侧的收口仍然必需，`retry` / `fail` 那两种判决就是它的用武之地：那时 task 维持 PAUSED、
气泡留着当入口，用户回话时得有人把它终局掉。注入分支（`_inject_user_turn`）一直在做；新建
分支（`_start_task_for_agent`）是 `faedd25` 补的。

本文件把现场手工摆成「气泡挂在一个已终态 task 上」（直接写 `status = "FINISHED"`，不经判决
路径），验的就是这种残留照样能被下一条消息收掉——判决侧现在会自己收，但崩在中途、老数据、
或将来又多一条终结路径时，这道防线还得在。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlAsk, UserTurnDelivery
from tests._event_helpers import append_one
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active


_TS = datetime(2026, 9, 24, tzinfo=UTC)
_SID, _AID = "A", "agt_1"


def _ev(seq: int, type_: EventType, *, agent_id: str | None = None, **payload) -> Event:
    return Event(id=f"evt_{seq:04d}", run_id="r1", sequence=seq, session_id=_SID,
                 type=type_, timestamp=_TS, task_id=None, agent_id=agent_id, payload=payload)


async def _noop_drain(self) -> None:
    return None


async def _live_session(monkeypatch):
    """一个活着的会话 + 一个跑完（FINISHED）的 task + 挂在它上面的 park 气泡。

    这正是「后台 observe 判 success、带外终结了 task」之后的现场：task 终态，气泡未决。
    """
    monkeypatch.setattr(TaskManager, "drain", _noop_drain)   # 不真的派发到 LLM

    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())

    await append_one(rt.event_store, _ev(1, EventType.SESSION_CREATED,
                                         template_id="agent:tpl_echo", root_agent_id=_AID))
    await append_one(rt.event_store, _ev(2, EventType.AGENT_INSTANTIATED, agent_id=_AID,
                                         template_id="agent:tpl_echo"))
    await append_one(rt.event_store, _ev(3, EventType.AGENT_IDLE, agent_id=_AID))
    await rebuild_all_active(rt)

    await initialize_all(rt)
    first = await rt.send_message(_AID, "讲个笑话")
    tm = rt._task_managers[_SID]

    # 纯文本回合让位：`_cold_park` 开的正是这个形状的气泡。
    req = await rt.hitl.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=first.task_id, preface="")),
        session_id=_SID, task_id=first.task_id, agent_id=_AID, tenant_id="default",
        stage="tool", unattended=False,
    )
    # 后台 observe 判 success → 带外判决把 task 终结掉。
    tm.get_task(first.task_id).status = "FINISHED"
    return rt, tm, first.task_id, req.id


def _pending(rt) -> list[str]:
    return [v.id for v in rt.hitl_registry.list_pending(session_id=_SID) if not v.resolved]


# ── 判决终结 task 之后，气泡还在 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_bubble_outlives_the_finished_task(monkeypatch) -> None:
    """**要害**：task 已是终态，那个「等你说话」的入口仍然开着。

    宿主据它把会话折成 PAUSED；若它在这里就没了，会话会跳成已完成。
    """
    rt, _tm, _tid, hid = await _live_session(monkeypatch)
    assert _pending(rt) == [hid]


# ── 用户开口 → 新建分支收口它 ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_next_message_starts_a_new_task(monkeypatch) -> None:
    """气泡留着**不影响路由**：`send_message` 只看 `current_task_id` 是否终态。"""
    rt, _tm, finished_id, _hid = await _live_session(monkeypatch)
    await initialize_all(rt)
    second = await rt.send_message(_AID, "再讲一个")
    assert second.task_id != finished_id, "应当新开一轮，而不是注进已终结的 task"


@pytest.mark.asyncio
async def test_the_next_message_closes_the_stale_bubble(monkeypatch) -> None:
    """用户开口 = 这个入口被用掉了 → 收口。

    不收的后果不在这一轮（新 task 一跑起来宿主就折成 RUNNING，显示不会错），而在之后：
    孤儿气泡把会话**永久钉在 PAUSED**。宿主折状态时 `pending_hitl` 排在 `terminal` 之前
    （`SessionStatusFold.status`），于是 root task 的终态再也浮不出来——SSE 的终态收口不
    触发，投影里 `sessions.status` 一直 PAUSED，重启还被 `list_active_session_ids` 当活
    会话捞回来。

    （2026-09-27 查证：它**不会**被 `resume_agent` 的 `_pause_bubble_of` 误当成暂停气泡
    放行冷续跑——那个函数按 `delivery.preface in (AFTER_INTERRUPT, AFTER_INTERRUPT_EDIT)`
    过滤，而 park 用的是 `PREFACE_NORMAL`；`resume_agent` 还要求 agent 处于
    `waiting_human`，而带外判决已经把它 `SETTLED` 成 idle。两道门都挡住了。曾经写在这里
    的那个担心是错的。）
    """
    rt, _tm, _tid, _hid = await _live_session(monkeypatch)
    await initialize_all(rt)
    await rt.send_message(_AID, "再讲一个")
    assert _pending(rt) == []


@pytest.mark.asyncio
async def test_a_fresh_park_still_works_after_that(monkeypatch) -> None:
    """收口是针对**旧**气泡的：新一轮自己再 park 时照常开得出来。"""
    rt, _tm, _tid, _hid = await _live_session(monkeypatch)
    await initialize_all(rt)
    second = await rt.send_message(_AID, "再讲一个")
    again = await rt.hitl.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=second.task_id, preface="")),
        session_id=_SID, task_id=second.task_id, agent_id=_AID, tenant_id="default",
        stage="tool", unattended=False,
    )
    assert _pending(rt) == [again.id]
