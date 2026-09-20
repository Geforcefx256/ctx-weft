"""`HitlClosed`：把「人答了」与「系统用掉了」分开。

`HitlResolved` 只说人答了。决定落盘之后、答复注入进对话之前，进程可能死掉——任务重排本身
不带注入这一步，不补，人说的那句话就静默消失。补注入的清单是 `HitlSnapshot.resolved`，而它
从前留的是该会话**全部**已终局请求：交互式会话里每条用户消息都是一次 UserTurn HITL，那个
清单随对话轮数线性增长。这条事件让它自己销账。

**一条规则贯穿所有形态：谁消费了这条决定，谁在持久效果落地之后盖章。** 本文件钉 UserTurn
那个发射点的时机，以及 `HitlService.close` 的顺序纪律；gateway 那两个出口在
`tests/unit/test_hitl_closed_by_gateway.py`。
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_NORMAL,
    HitlDecision,
    UserTurnDelivery,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)


def _req(hitl_id="hit_u", message="我的答复"):
    from ctx_weft.core.hitl.registry import PendingHitl

    req = PendingHitl(
        id=hitl_id, form=HITL_FORM_WAIT, session_id="s1", task_id="t1", agent_id="ag_root",
        delivery=UserTurnDelivery(task_id="t1", preface=PREFACE_NORMAL), created_at=_T0,
        tenant_id="acme",
    )
    req.decision = HitlDecision(outcome="accepted", message=message)
    req.resolved_at = _T0
    return req


async def _harness(monkeypatch):
    """runtime + 一条按顺序记录「ingest / emit」的探针。"""
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
    import ctx_weft.core.loop.steps.background_observe as bo

    monkeypatch.setattr(bo, "_task_pending", {})
    rt = make_runtime(agent_provider=InlineAgentTemplateProvider())
    rt.providers.register_memory(InMemoryMemoryProvider())

    trace: list[str] = []
    orig_ingest = rt._ingest_user_turn

    async def _traced_ingest(*a, **kw):
        trace.append("ingest")
        return await orig_ingest(*a, **kw)

    monkeypatch.setattr(rt, "_ingest_user_turn", _traced_ingest)

    emitted: list = []
    orig_emit = rt._event_bus.emit

    async def _traced_emit(ev):
        if ev.type == EventType.HITL_CLOSED:
            trace.append("emit")
            emitted.append(ev)
        return await orig_emit(ev)

    monkeypatch.setattr(rt._event_bus, "emit", _traced_emit)

    session = Session(id="s1", tenant_id="acme", user_prompt="hi", status="RUNNING")
    task = Task(id="t1", session_id="s1", status="SUSPENDED",
                assigned_agent_id="ag_root", creator_agent_id="ag_root")
    return rt, session, task, trace, emitted


async def test_emitted_after_the_ingest_not_before(monkeypatch):
    """顺序就是正确性：先 ingest 再盖章。

    反过来的话，崩在两者之间就留下「日志说已注入、对话里却没有」——而这条事件的全部作用
    正是让折叠据此**不再**补注入，于是那句话永久消失。`CapabilityFinished` 现在就是发在
    ingest 之前的，那条缝真实存在。
    """
    rt, session, task, trace, emitted = await _harness(monkeypatch)

    await rt._write_hitl_reply_turn(_req(), session, task)

    assert trace == ["ingest", "emit"], f"顺序错了：{trace}"
    assert len(emitted) == 1
    ev = emitted[0]
    assert ev.payload == {"hitl_id": "hit_u"}, "只带 hitl_id——正文已经在对话里"
    # tenant 取自**请求**，与 HitlOpened / HitlResolved / HitlReplyRetracted 同口径
    # （`HitlService._emit` 统一用 `req.tenant_id`）——不是从 session 另取一份。
    assert (ev.session_id, ev.task_id, ev.tenant_id) == ("s1", "t1", "acme")


async def test_not_emitted_when_the_ingest_fails(monkeypatch):
    """ingest 失败 → 不许发。发了就等于宣布「已注入」，那条答复再也不会被补。"""
    rt, session, task, trace, emitted = await _harness(monkeypatch)

    async def _boom(*a, **kw):
        trace.append("ingest")
        raise RuntimeError("memory down")

    monkeypatch.setattr(rt, "_ingest_user_turn", _boom)

    with pytest.raises(RuntimeError, match="memory down"):
        await rt._write_hitl_reply_turn(_req(), session, task)

    assert emitted == [], "ingest 没成，绝不能宣布已注入"


async def test_emit_failure_does_not_undo_a_successful_injection(monkeypatch):
    """发事件失败**不抛**：注入已经成功、回滚不了，抛出去会把一次成功变成恢复失败。

    代价是退回本事件引入之前的行为（那条请求继续留在 `resolved` 里、下次恢复幂等补一遍），
    不是数据损坏。
    """
    rt, session, task, trace, _ = await _harness(monkeypatch)

    async def _emit_boom(ev):
        if ev.type == EventType.HITL_CLOSED:
            raise RuntimeError("bus down")

    monkeypatch.setattr(rt._event_bus, "emit", _emit_boom)

    await rt._write_hitl_reply_turn(_req(), session, task)      # 不抛

    from ctx_weft.protocols import MemoryAddress, MemoryEventType, ProviderContext
    recs = await rt.providers.get_memory().recall_recent_by_agent(
        MemoryAddress(session_id="s1", task_id=None, agent_id="ag_root"),
        [MemoryEventType.USER_PROMPT], 10,
        ProviderContext(session_id="s1", tenant_id="acme"))
    assert [r.content for r in recs] == ["我的答复"], "注入本身必须已经生效"


async def test_registry_no_longer_lists_it_for_re_injection(monkeypatch):
    """端到端的那条闭环：发过 injected 之后，折叠 → registry → 补注入清单里没有它。

    这就是「有界」在行为上的样子——清单只剩「答了但还没落进对话」的那几条。
    """
    from ctx_weft.core.control.reducers import fold_hitl_snapshot
    from ctx_weft.core.hitl.registry import HitlRegistry
    from ctx_weft.protocols.events import Event

    def _ev(type_, payload, seq):
        return Event(id=f"evt_{seq}", run_id=None, sequence=seq, session_id="s1",
                     type=type_, timestamp=_T0, tenant_id="acme", task_id="t1",
                     agent_id="ag_root", payload=payload)

    base = [
        _ev(EventType.HITL_OPENED, {
            "hitl_id": "hit_u", "form": "wait",
            "delivery": {"kind": "user_turn", "task_id": "t1", "preface": "normal"},
            "subject_id": "", "prompt": "", "detail": "", "fields": [], "proposal": None,
            "tool_call_id": "", "agent_id": "ag_root", "resume_state": None,
            "reply_as_result": False, "stage": "tool",
        }, 0),
        _ev(EventType.HITL_RESOLVED, {
            "hitl_id": "hit_u", "outcome": "accepted", "claimed": False,
            "message": "我的答复"}, 1),
    ]

    reg = HitlRegistry()
    reg.load_snapshot(fold_hitl_snapshot(base))
    assert [r.id for r in reg.resolved_for_session("s1")] == ["hit_u"], "前提：本来在清单里"

    reg2 = HitlRegistry()
    reg2.load_snapshot(fold_hitl_snapshot(
        [*base, _ev(EventType.HITL_CLOSED, {"hitl_id": "hit_u"}, 2)]))
    assert reg2.resolved_for_session("s1") == [], "注入过了就不该再出现在补注入清单里"
