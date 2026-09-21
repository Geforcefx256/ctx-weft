"""端到端：恢复期补写的盲区（M8）与取消会话时的收口（L5）。

- **M8**：「这个 task 还挂着别的未决问题」曾被当成跳过补写的理由——那是重排要不要放行的
  判据，与「这条答复进没进过对话」无关。代价是同一个 task 上前一条已终局的答复被跳过，
  而人答掉那个未决问题时走的是活 TaskManager 那条路，于是永久缺失。补写的时间戳也要用
  答复真正终局的时刻，否则它会排到本次应答之后，对话顺序颠倒。
- **L5**：取消会话时，开着的未提交窗口要先收掉——不收的话，待终局的答复既不在待答列表里
  （收口遍历不到）、也没人退回，重启后又成了「有 HitlOpened 无终局事件」的未决提问；
  总线那半缓冲也没人清。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest import mock

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.llm.mock import MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _ActRouterLLM, _poll
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    STALL_BEFORE_CHUNK, _ask, _next_question, _ScriptedLLM,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)
from tests._event_helpers import all_events, append_one

pytestmark = pytest.mark.asyncio

SID, TID, AID = "ses_b", "tsk_b", "agt_root"
TS = datetime(2026, 9, 17, tzinfo=timezone.utc)


def _runtime(llm, *, hitl_timeout_sec=None, memory=None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec))
    rt.providers.register_memory(memory or InMemoryMemoryProvider())
    return rt


def _ev(seq, type_, ts=TS, **payload):
    task_id = payload.pop("task_id", None)
    return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id=SID,
                 type=type_, timestamp=ts, task_id=task_id, payload=payload)


# ── M8：还挂着别的未决问题，也要把已终局的答复补进对话 ─────────────────────────────


async def test_answered_reply_is_injected_even_while_another_question_is_pending() -> None:
    llm = _ActRouterLLM(act_responses=[MockResponse(text="done")], context_limit=100_000)
    mem = InMemoryMemoryProvider()
    rt = _runtime(llm, memory=mem)
    earlier = datetime(2026, 9, 17, 0, 0, 10, tzinfo=timezone.utc)

    for e in [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go", template_id="agent:tpl_echo",
            root_agent_id=AID),
        _ev(2, EventType.TASK_CREATED, task={
            "id": TID, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": AID, "creator_agent_id": AID, "user_prompt": "go"}),
        _ev(3, EventType.TASK_STARTED, task_id=TID, assigned_agent_id=AID),
        # ① 一条 wait 气泡被答过、但进程在把它写进对话之前就崩了
        _ev(4, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_1", form="wait",
            delivery={"kind": "user_turn", "task_id": TID, "preface": "normal"},
            stage="tool", agent_id=AID, prompt=""),
        _ev(5, EventType.HITL_RESOLVED, ts=earlier, task_id=TID, hitl_id="hit_1",
            outcome="accepted", message="ANSWERED_EARLIER"),
        # ② 同一个 task 上还挂着另一个**未决**问题 —— 曾经就是这一条让上面那条被跳过
        _ev(6, EventType.HITL_OPENED, task_id=TID, hitl_id="hit_2", form="question",
            delivery={"kind": "tool_result", "tool_call_id": "call_9"},
            tool_call_id="call_9", stage="tool", agent_id=AID, prompt="which db?"),
        _ev(7, EventType.TASK_AWAITING_HUMAN, task_id=TID, hitl_id="hit_2"),
    ]:
        await append_one(rt.event_store, e)

    # 装填是调用方的责任（2026-09-21：`recover_agent` 对 registry miss 直接抛
    # `AgentNotLoaded`，按 agent 扫全库的 sweep 已删）。只喂内存，不建 TM、不跑。
    await rt.rebuild_session(SID)
    with mock.patch(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        return_value=None,
    ):
        await rt.recover_agent(AID)
        await asyncio.sleep(0.3)

    scope = MemoryAddress(session_id=SID, task_id=TID, agent_id=AID)
    pctx = ProviderContext(session_id=SID, tenant_id="default", task_id=TID, agent_id=AID)
    view = await mem.load_view(scope, MemoryScope.TASK, pctx,
                               kinds=[MemoryKind.CONVERSATION_TURN])
    replies = [r for r in view if r.metadata.get("source") == "hitl_reply"]
    assert [content_to_text(r.content) for r in replies] == ["ANSWERED_EARLIER"], (
        "还挂着别的未决问题，不该拦住这条已终局答复进对话")
    assert replies[0].timestamp == earlier, (
        "补写要用那条答复真正终局的时刻，否则它会排到后来的消息之后")
    # task 状态一动不动：补写是**只写**的
    assert rt._task_managers[SID].get_task(TID).status == "AWAITING_HUMAN"


# ── L5：取消会话时先收窗，再收口气泡 ────────────────────────────────────────────


async def test_cancel_session_closes_open_rounds_and_finalizes_claimed_replies() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)

    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    tm = rt._task_managers[sid]
    assert tm.is_round_open(q1.task_id) and rt.hitl_registry.get(q1.id).claim_pending, (
        "前提：这条应答收下了、那一轮还开着（LLM 没开口）")

    assert await rt.cancel_session(sid) is True
    await asyncio.sleep(0.4)

    rec = rt.hitl_registry.get(q1.id)
    assert rec.resolved and rec.decision.outcome == "cancelled", (
        "待终局的答复必须被退回再收口，否则它既不在待答列表里、也永远不终局")
    types = [e.type for e in await all_events(rt.event_store, sid)]
    assert EventType.HITL_CANCELLED in types or EventType.HITL_RESOLVED in types, (
        f"收口事实必须落盘，否则重启后这条提问又成了未决：{types}")
    assert not tm.open_round_task_ids, "取消之后不得留下开着的窗"
    bus = rt.event_bus
    assert not getattr(bus, "_provisional", {}), "总线那半缓冲也要清干净"


async def test_forget_session_refuses_while_a_round_is_open() -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    await rt.reply_to_hitl(HitlReply(
        hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)

    assert rt.session_is_quiescent(sid) is False, (
        "有待终局的答复 / 开着的窗 → 这条会话还不算安静")
    assert rt.forget_session(sid) is False, "不安静就不该被逐出（会把那一轮连暂存一起丢掉）"
    assert sid in rt._task_managers


async def test_quiescence_accounts_for_open_rounds_and_claimed_replies() -> None:
    """安静判据必须认这两样——它们都表示「有一轮正开着」，而且都不出现在待答列表里。

    上一个用例里 task 还在跑，`is_done()` 本来就为假，钉不住这两条新判据；这里把会话跑到
    静止之后再逐个制造它们。
    """
    llm = _ScriptedLLM([MockResponse(text="all done", tool_calls=[])])
    rt = _runtime(llm)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    # interactive 的 root task 说完一段纯文本就 park 出 wait 气泡；取消整条会话让它彻底静止
    # （光收口气泡不够——agent 还停在 waiting_human）。
    wait = await _poll(lambda: [p for p in rt.hitl_registry.list_pending(session_id=sid)
                                if p.form == "wait"] or None)
    task_id, agent_id = wait[0].task_id, wait[0].agent_id
    assert await rt.cancel_session(sid) is True
    await _poll(lambda: rt.session_is_quiescent(sid) or None, timeout=8.0)

    tm = rt._task_managers[sid]

    # ① 开着的未提交窗口
    tm.begin_round(task_id, owns_task=False)
    assert rt.session_is_quiescent(sid) is False
    assert rt.forget_session(sid) is False
    tm.drop_round_buffer(task_id)
    assert rt.session_is_quiescent(sid) is True

    # ② 已收下、尚未终局的答复（它刻意不出现在待答列表里）
    from ctx_weft.protocols.hitl import HitlAsk, HitlDecision, UserTurnDelivery
    req = rt.hitl_registry.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=task_id), prompt=""),
        hitl_id="hit_claimed", session_id=sid, task_id=task_id, agent_id=agent_id,
        stage="tool", created_at=TS)
    rt.hitl_registry.claim(req.id, HitlDecision(outcome="accepted", message="later"))
    assert not rt.hitl_registry.list_pending(session_id=sid), "前提：它不在待答列表里"
    assert rt.session_is_quiescent(sid) is False
    assert rt.forget_session(sid) is False
