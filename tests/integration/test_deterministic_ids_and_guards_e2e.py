"""端到端：剩余遗留问题的修复（审查文档 L1–L4、L7、L8）。

同一味药贯穿其中：**确定性 id**。重入、重放、撤销后重跑都可能让同一条记录被写第二遍，
而 memory 的 id 契约（已存在的 id——含已 superseded——= no-op）本来就能兜住，前提是 id 稳定。

- L4 审计记录：id 由 tool_call 派生 → 冷重入不再一次次叠加；
- L3 继承复制：id 由「子 scope + 源记录」派生 → 撤销一轮后重跑不再整段抄第二遍；
- L1 assistant 回合：也走暂存分流 → 窗口开着时不直写 memory；
- L2 压缩忙判据：有开着的窗 / 有待终局答复时不许开压缩；
- L7 暂存块排序：序号与落盘后可比，时间戳撞上也不会前后不一致；
- L8 跨 task 的气泡：只有同一个 task 的才跟着这一轮收口。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from ctx_weft.core.loop.capability_gateway import tool_audit_record_id, tool_result_record_id
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.models.errors import SessionBusyError
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import (
    MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext,
)
from ctx_weft.protocols.hitl import HitlAsk, HitlDecision, UserTurnDelivery
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _poll
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    STALL_BEFORE_CHUNK, _ask, _next_question, _ScriptedLLM,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

pytestmark = pytest.mark.asyncio
TS = datetime(2026, 9, 18, tzinfo=timezone.utc)


def _runtime(llm, *, hitl_timeout_sec=None, memory=None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=hitl_timeout_sec))
    rt.providers.register_memory(memory or InMemoryMemoryProvider())
    return rt


def test_audit_and_result_ids_are_derived_from_the_same_call() -> None:
    """L4：审计记录也有确定性 id，且与结果那条同源不撞车；裸 wire id 一律 None。"""
    from ctx_weft.core.utils.ids import mint_call_id
    tcid = mint_call_id(anchor="asst_z", ordinal=0, raw_id="tc1", turn_seq=0)
    audit, result = tool_audit_record_id(tcid), tool_result_record_id(tcid)
    assert audit and result and audit != result
    assert tool_audit_record_id(tcid) == audit, "同一次调用恒得同一个 id"
    assert tool_audit_record_id("call_1") is None, "裸 wire id 不存在确定性派生"


async def test_staged_records_sort_after_everything_already_in_memory() -> None:
    """L7：暂存记录的排序键要和落盘后可比——取视图里的最大序号往后排。"""
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)
    await rt.reply_to_hitl(
        __import__("ctx_weft.protocols.hitl", fromlist=["HitlReply"]).HitlReply(
            hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="A1"))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)

    tm = rt._task_managers[sid]
    staged = tm.staged_memory(q1.task_id)
    assert staged, "前提：这一轮有暂存记录"

    mem = rt.providers.get_memory()
    scope = MemoryAddress(session_id=sid, task_id=q1.task_id, agent_id=q1.agent_id)
    pctx = ProviderContext(session_id=sid, tenant_id="default",
                           task_id=q1.task_id, agent_id=q1.agent_id)
    # 只看**会被渲染**的那些：审计记录不进 prompt、不参与 composer 的归并排序。
    view = await mem.load_view(scope, MemoryScope.TASK, pctx,
                               kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY])
    # 装配一次，触发叠加路径给暂存记录补序号
    from ctx_weft.core.assembler.assembler import ContextRequest
    from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource
    from ctx_weft.core.utils.estimate import estimate_tokens
    from types import SimpleNamespace

    deps = SimpleNamespace(memory=mem, provider_ctx=pctx)
    request = SimpleNamespace(
        scope=scope, token_counter=estimate_tokens, task=SimpleNamespace(id=q1.task_id),
        extra={"staged_memory": staged})
    blocks = [b async for b in AgentRecallSource().fetch(request, deps)]
    # 按记录 id 精确挑出暂存那些块：暂存里还有一条 TOOL_AUDIT，它不进 prompt、不产块。
    staged_ids = {e.id for e in staged}
    staged_seq = [b.metadata["seq_no"] for b in blocks
                  if b.metadata.get("memory_event_id") in staged_ids]
    assert staged_seq, "暂存的对话记录必须被叠加进历史"
    memory_seq = [r.metadata.get("seq_no", 0) for r in view if r.metadata]
    assert min(staged_seq) > max(memory_seq or [0]), (
        f"暂存记录必须排在已落盘记录之后：staged={staged_seq} memory={memory_seq}")
    assert staged_seq == sorted(staged_seq)


async def test_compact_refuses_while_a_round_is_open() -> None:
    """L2：有开着的窗（这一轮的内容还在暂存区里）时，压缩必须拒绝——它只折 memory。

    会话先跑到静止（否则 run 令牌本身就会挡住压缩，测不出新判据），再逐个制造这两种状态。
    """
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid, aid = handle.session_id, handle.agent_id
    q1 = await _next_question(rt, sid)
    assert await rt.cancel_session(sid) is True
    await _poll(lambda: rt.session_is_quiescent(sid) or None, timeout=8.0)
    tm = rt._task_managers[sid]

    # ① 开着的窗
    tm.begin_round(q1.task_id, owns_task=False)
    with pytest.raises(SessionBusyError):
        await rt.compact_agent(aid)
    tm.drop_round_buffer(q1.task_id)

    # ② 已收下、尚未终局的答复
    req = rt.hitl_registry.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=q1.task_id), prompt=""),
        hitl_id="hit_claim", session_id=sid, task_id=q1.task_id, agent_id=aid,
        stage="tool", created_at=TS)
    rt.hitl_registry.claim(req.id, HitlDecision(outcome="accepted", message="later"))
    with pytest.raises(SessionBusyError):
        await rt.compact_agent(aid)


async def test_only_the_same_tasks_bubble_follows_the_round() -> None:
    """L8：注入消息时，别的 task 上的气泡必须立即收口，不能跟着这一轮走。

    跟着走就没人管了：提交与撤销两个钩子都按 task 查待终局请求，别的 task 的会永远卡在
    待终局，而且从待答列表里消失。
    """
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), STALL_BEFORE_CHUNK])
    rt = _runtime(llm, hitl_timeout_sec=0)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id
    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.2)

    # 同一个 agent、**另一个** task 上的气泡
    other = rt.hitl_registry.open(
        HitlAsk(form="wait", delivery=UserTurnDelivery(task_id="tsk_other"), prompt=""),
        hitl_id="hit_other", session_id=sid, task_id="tsk_other", agent_id=q1.agent_id,
        stage="tool", created_at=TS)

    await rt._inject_user_turn(q1.task_id, "new instruction", session_id=sid)

    assert rt.hitl_registry.get(other.id).resolved, (
        "别的 task 的气泡必须当场收口——没有哪一轮会替它终局")
    assert not rt.hitl_registry.get(other.id).claim_pending
