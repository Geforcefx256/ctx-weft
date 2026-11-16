"""Task 1: task-resident 模型翻转 — close 只写 finish 对、留 raw body。

复用 test_capsule_golden 的 helper（InMemoryMemoryProvider, _ev, _task_scope,
_agent_scope, _state, _loop_ctx, _make_task, _pctx, T, finalize_task_memory, LoopConfig）。

核心不变量（本 task 后）：
  每个结束 task = task 层 raw body（不被 supersede）+ agent 层 finish 对（2 条）。
"""
from __future__ import annotations

from ctx_weft.core.loop.task_memory_fold import task_record_kind

import pytest

from ctx_weft.protocols.template import LoopConfig

from .test_capsule_golden import (
    T,
    _agent_scope,
    _ev,
    _loop_ctx,
    _make_task,
    _pctx,
    _state,
    _task_scope,
    finalize_task_memory,
    InMemoryMemoryProvider,
)



@pytest.mark.asyncio
async def test_close_writes_finish_pair_keeps_body() -> None:
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "做X", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "在做", 2, role="assistant"), _pctx())
    await mem.ingest(_ev(T.TOOL_RESULT, tsc, "ok", 3, role="tool"), _pctx())
    task = _make_task(outputs="完成")
    await finalize_task_memory(
        mem, _state(task, tsc, LoopConfig()), task,
        "完成\n\nProcess Report: 成功", "success", _loop_ctx(mem),
        act_recap="成功", task_summary="",
    )
    # agent 层：恰 2 条 finish 对
    caps = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    assert len(caps) == 2
    assert task_record_kind(caps[0]) == "finish"  # newest-first: Process Report tool 在前
    asst = [c for c in caps if task_record_kind(c) == "recap"][0]
    assert not asst.metadata.get("tool_calls")
    # task 层 body 仍在（未 supersede）
    body = await mem.recall_recent(
        tsc, [T.USER_PROMPT, T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx(),
    )
    assert len(body) == 3


# ─── Task 2: 短任务 close → 全 raw 留（不 supersede 末段） ─────────────────────


@pytest.mark.asyncio
async def test_short_close_keeps_active_raw() -> None:
    """短叶子 task（1 轮 LLM、token 极小 → _is_short_leaf=True）close 后，task 层
    active LLM_RESPONSE/TOOL_RESULT 仍可 recall（全 raw 保留，Task 1 行为不变）。"""
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "做X", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "在做", 2, role="assistant"), _pctx())
    await mem.ingest(_ev(T.TOOL_RESULT, tsc, "ok", 3, role="tool"), _pctx())
    task = _make_task(outputs="完成")
    await finalize_task_memory(
        mem, _state(task, tsc, LoopConfig()), task,
        "完成\n\nProcess Report: 成功", "success", _loop_ctx(mem),
        act_recap="成功", task_summary="",
    )
    # 短任务：active LLM_RESPONSE/TOOL_RESULT 仍被 recall（未 supersede）
    raw = await mem.recall_recent(tsc, [T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx())
    assert len(raw) == 2
    up = await mem.recall_recent(tsc, [T.USER_PROMPT], 500, _pctx())
    assert len(up) == 1


# ─── Task 2: 长任务 close → supersede 末 raw 段，留锚点 + finish 对 ─────────────


@pytest.mark.asyncio
async def test_long_close_supersedes_final_raw_keeps_anchors() -> None:
    """长 task（LLM_RESPONSE 轮次 > short_task_turn_cap=2 → _is_short_leaf=False）close 后：
    - task 层 active LLM_RESPONSE/TOOL_RESULT/TOOL_INVOCATION 被 supersede（不再 recall）
    - USER_PROMPT 与既有 TASK_COMPACT_SUMMARY 锚点保留
    - agent 层 finish 对（Task 1）仍在
    - 不另产新 TASK_COMPACT_SUMMARY（末段由 finish 对 Process Report 承载）
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()
    cfg = LoopConfig()  # short_task_turn_cap=2

    await mem.ingest(_ev(T.USER_PROMPT, tsc, "做大X", 1, role="user"), _pctx())
    # 既有中间段摘要（前一边界后台 observe 折出）—— 锚点，须保留
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, tsc, "〔段①〕已处理前半", 2, role="assistant"), _pctx())
    # 末 raw 段：> turn_cap 轮 LLM + 工具调用/结果
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "处理1", 3, role="assistant"), _pctx())
    await mem.ingest(_ev(T.TOOL_INVOCATION, tsc, "call", 4, role="assistant"), _pctx())
    await mem.ingest(_ev(T.TOOL_RESULT, tsc, "res1", 5, role="tool"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "处理2", 6, role="assistant"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "处理3", 7, role="assistant"), _pctx())  # 3 轮 > 2

    task = _make_task(outputs="大功告成")
    await finalize_task_memory(
        mem, _state(task, tsc, cfg), task,
        "大功告成\n\nProcess Report: 成功收尾", "success", _loop_ctx(mem),
        act_recap="成功收尾", task_summary="",
    )

    # 末 raw 段被 supersede（active LLM/TOOL/INVOCATION 不再 recall）
    raw = await mem.recall_recent(
        tsc, [T.LLM_RESPONSE, T.TOOL_RESULT, T.TOOL_INVOCATION], 500, _pctx(),
    )
    assert raw == [], f"long task: final raw segment must be superseded; got {raw!r}"

    # USER_PROMPT + TASK_COMPACT_SUMMARY 锚点保留
    anchors = await mem.recall_recent(
        tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY], 500, _pctx(),
    )
    assert {a.type for a in anchors} == {T.USER_PROMPT, T.TASK_COMPACT_SUMMARY}
    assert len(anchors) == 2, f"anchors must survive; got {anchors!r}"
    # 不另产新段摘要：仅原有 1 条 TASK_COMPACT_SUMMARY
    summaries = [a for a in anchors if a.type == T.TASK_COMPACT_SUMMARY]
    assert len(summaries) == 1 and "段①" in summaries[0].content

    # agent 层 finish 槽位仍在（Task 1）：有产出 → recap / 答复+finish_task / 报告 三槽
    caps = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    assert len(caps) == 3
    reply_slot = [c for c in caps if c.metadata.get("final_reply")][0]
    assert task_record_kind(reply_slot) == "final_reply" and not reply_slot.metadata.get("tool_calls")
    assert "大功告成" in reply_slot.content
