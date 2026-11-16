"""close 路径 A1：机会性用结果槽 / 占位+异步替换 finish 对 Process Report。

三个测试：
1. test_a1_slot_hit_uses_background_report — bg 先完成，_close_report 预置；
   _synthesize_dispatch_pair 合成的 finish tool 记录 == "Process Report: 好报告"。
2. test_a1_placeholder_then_async_replace — 槽空 → 先用占位；随后 bg 回调替换；
   旧记录被 supersede，新记录 content=="Process Report: 好报告"，tool_call_id 配对完整。
3. test_a1_no_await_blocking — _synthesize_dispatch_pair 不调 await_pending_observe。
"""
from __future__ import annotations

from tests.task_memory_testkit import render_task_record
from ctx_weft.core.loop.task_memory_fold import task_record_kind, task_record_group

from datetime import UTC, datetime, timedelta

import pytest

from ctx_weft.core.loop.task_memory_fold import (
    find_finish_target,
    replace_finish_report,
)
from ctx_weft.core.loop.steps.finalize import _synthesize_dispatch_pair
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
SESSION = "s1"


# ─── 辅助 ─────────────────────────────────────────────────────────────────────

def _pctx() -> ProviderContext:
    return ProviderContext(session_id=SESSION, tenant_id="default")


def _task_scope(task_id: str = "t1", agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=task_id, agent_id=agent_id)


def _agent_scope(agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=None, agent_id=agent_id)


def _ev(type_: MemoryEventType, scope: MemoryAddress, content: str, t: int,
        role: str | None = None, **meta) -> MemoryEvent:
    return MemoryEvent(
        type=type_, address=scope, content=content,
        timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta,
    )


def _make_task(task_id: str = "t1", agent_id: str = "ag1",
               outputs: str | None = "最终答复") -> Task:
    return Task(
        id=task_id, session_id=SESSION, status="FINISHED", tenant_id="default",
        assigned_agent_id=agent_id, creator_agent_id=agent_id, parent_task_id=None,
        title="测试任务", description="", user_prompt="初始请求",
        settings=NormalTaskSettings(), outputs=outputs,
    )


async def _get_finish_tool(mem: InMemoryMemoryProvider, scope: MemoryAddress) -> MemoryEvent | None:
    """取 agent scope 内最新的 tool role AGENT_CONVERSATION_TURN（finish 记录）。"""
    recs = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    tool_recs = [r for r in recs if task_record_kind(r) == "finish"]
    return tool_recs[0] if tool_recs else None  # recall_recent returns newest-first


async def _get_finish_asst(mem: InMemoryMemoryProvider, scope: MemoryAddress) -> MemoryEvent | None:
    """取 agent scope 内最新的 assistant role AGENT_CONVERSATION_TURN（finish 记录）。"""
    recs = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    asst_recs = [r for r in recs if task_record_kind(r) == "recap"]
    return asst_recs[0] if asst_recs else None


@pytest.mark.asyncio
async def test_a1_slot_hit_uses_background_report() -> None:
    """bg 先完成：_close_report[task.id] 预置 ("好报告_act", "好报告_sum")；
    _synthesize_dispatch_pair 合成后立即替换：finish tool == "好报告_sum"，assistant == "好报告_act"。
    反转契约（spec 2026-07-01）：finish 对 assistant 槽 = act_recap（过程复述，≠ 答复；答复由内联
    body / blackboard 承载，避免重复）。无后续替换（slot 命中即直接用）。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    await mem.ingest(_ev(T.USER_PROMPT, tsc, "初始请求", 1, role="user"), _pctx())

    task = _make_task()

    assert await find_finish_target(mem, _pctx(), asc, task.id) is None
    await _synthesize_dispatch_pair(mem, asc, task, "占位 act_recap", "占位 task_summary", "success", _pctx())
    scope, call, policy = await find_finish_target(mem, _pctx(), asc, task.id)
    assert policy["preserve_raw"]
    await replace_finish_report(mem, _pctx(), scope, task.id, call,
                                "好报告_act", "好报告_sum", policy["outcome"], task.title)

    finish_tool = await _get_finish_tool(mem, asc)
    assert finish_tool is not None, "finish tool record must exist"
    assert finish_tool.content == render_task_record("finish", task_id="t1", title="测试任务", outcome="success", body="好报告_sum"), (
        f"slot hit: finish tool content must be '好报告_sum' (+ task marker); got {finish_tool.content!r}"
    )
    finish_asst = await _get_finish_asst(mem, asc)
    assert finish_asst is not None, "finish assistant record must exist"
    assert "好报告_act" in finish_asst.content, (
        f"slot hit: finish assistant content must be act_recap '好报告_act'; got {finish_asst.content!r}"
    )

@pytest.mark.asyncio
async def test_a1_placeholder_then_async_replace() -> None:
    """槽空 → finalize 先合成用占位 + 登记 _close_synth；
    随后 bg 回调（直接调 _replace_finish_report）替换 finish 对（assistant + tool 两条）：
    - 旧两条被 supersede（不再出现）
    - 新 tool 记录 content == "好报告_sum"，新 assistant content == "好报告_act"
    - tool_call_id 配对相同（不产生悬空 pair）
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    await mem.ingest(_ev(T.USER_PROMPT, tsc, "初始请求", 1, role="user"), _pctx())

    task = _make_task()

    # 槽空（_close_report 为空）→ 占位路径
    await _synthesize_dispatch_pair(mem, asc, task, "占位 act_recap", "占位 task_summary", "success", _pctx())

    # 占位记录已写入（assistant + tool 两条）
    finish_tool_placeholder = await _get_finish_tool(mem, asc)
    assert finish_tool_placeholder is not None, "placeholder finish tool must exist"
    assert finish_tool_placeholder.content, "placeholder tool content must be non-empty"
    placeholder_tool_call_id = task_record_group(finish_tool_placeholder)
    assert placeholder_tool_call_id, "placeholder finish tool must have tool_call_id"

    finish_asst_placeholder = await _get_finish_asst(mem, asc)
    assert finish_asst_placeholder is not None, "placeholder finish assistant must exist"
    # 反转契约：finish 对 assistant 槽 = act_recap（过程复述，≠ 答复）
    assert "占位 act_recap" in finish_asst_placeholder.content, (
        f"placeholder assistant content must be '占位 act_recap'; got {finish_asst_placeholder.content!r}"
    )

    p_scope, p_tool_call_id, policy = await find_finish_target(mem, _pctx(), asc, task.id)
    assert p_tool_call_id == placeholder_tool_call_id
    assert policy["preserve_raw"]
    p_outcome = policy["outcome"]
    await replace_finish_report(
        mem, _pctx(), p_scope, "t1",
        p_tool_call_id, "好报告_act", "好报告_sum", p_outcome, task.title,
    )

    # 旧两条占位记录应被 supersede（不再出现）
    all_recs = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    active_tool_recs = [r for r in all_recs if task_record_kind(r) == "finish"]
    active_asst_recs = [r for r in all_recs if task_record_kind(r) == "recap"]
    assert len(active_tool_recs) == 1, (
        f"after replace, exactly 1 active finish tool record expected; got {len(active_tool_recs)}: "
        f"{[r.content for r in active_tool_recs]}"
    )
    assert len(active_asst_recs) == 1, (
        f"after replace, exactly 1 active finish assistant record expected; got {len(active_asst_recs)}"
    )
    new_finish_tool = active_tool_recs[0]
    new_finish_asst = active_asst_recs[0]
    # 归属前缀须**存活**替换：_replace_finish_report 整条重写 tool 槽，漏传 title 就会把
    # finalize 合成占位时打上的 `[task: …]` 标记抹掉。
    assert new_finish_tool.content == render_task_record("finish", task_id="t1", title="测试任务", outcome="success", body="好报告_sum"), (
        f"new finish tool content must be '好报告_sum' (+ task marker preserved across the "
        f"bg replace); got {new_finish_tool.content!r}"
    )
    # 反转契约：finish 对 assistant 槽 = act_recap（bg 刷新后的 "好报告_act"，≠ 答复）
    assert "好报告_act" in new_finish_asst.content, (
        f"new finish assistant content must be act_recap '好报告_act'; got {new_finish_asst.content!r}"
    )
    assert task_record_group(new_finish_tool) == placeholder_tool_call_id, (
        "replaced record must keep same tool_call_id for proper pair matching"
    )




# ─── TEST 3: no blocking await ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a1_no_await_blocking(monkeypatch) -> None:
    """_synthesize_dispatch_pair 不调 await_pending_observe（非阻塞）。"""
    called = []

    async def _fake_await(self, task_id: str, step="observe") -> None:
        called.append(task_id)

    # Monkeypatch the function in the background_observe module
    from ctx_weft.core.loop.driver import ExecutionRegistry
    monkeypatch.setattr(ExecutionRegistry, "wait", _fake_await)

    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()

    await mem.ingest(_ev(T.USER_PROMPT, tsc, "初始请求", 1, role="user"), _pctx())

    task = _make_task()

    await _synthesize_dispatch_pair(mem, asc, task, "act_recap", "task_summary", "success", _pctx())

    assert called == [], (
        f"_synthesize_dispatch_pair must NOT call await_pending_observe (A1 non-blocking); "
        f"called with: {called}"
    )


# ─── TEST 4/5: 方案2 最终段 raw 折叠 ───────────────────────────────────────────

async def _ingest_user_plus_raw(mem: InMemoryMemoryProvider, tsc: MemoryAddress) -> None:
    """task 层：user 锚点 + 最终段 raw（LLM_RESPONSE + TOOL_RESULT，close 边界未折）。"""
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "初始请求", 1, role="user"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, tsc, "我在读目录…", 2, role="assistant"), _pctx())
    await mem.ingest(_ev(T.TOOL_RESULT, tsc, "目录内容…", 3, role="tool"), _pctx())


@pytest.mark.asyncio
async def test_slot_hit_replaces_report_no_raw_mirror() -> None:
    """task-resident：close 不镜像 body（无 final_segment_raw 镜像）；slot hit 时 finish tool
    用真实 report；user 锚点 + 最终段 raw 留 task 层（不进 agent 层）。

    注：原「方案2 折最终段 raw 镜像」随 mirror 删除而失效——agent 层从不写 final_segment_raw，
    body raw 留 task 层（长任务压缩由 Task 2 处理）。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()
    await _ingest_user_plus_raw(mem, tsc)

    task = _make_task()

    await _synthesize_dispatch_pair(mem, asc, task, "占位_act", "占位_sum", "success", _pctx())
    scope, call, policy = await find_finish_target(mem, _pctx(), asc, task.id)
    await replace_finish_report(mem, _pctx(), scope, task.id, call,
                                "真实段_act", "真实段总结", policy["outcome"], task.title)

    recs = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    # agent 层从不写 final_segment_raw 镜像（task-resident）
    assert [r for r in recs if r.metadata.get("final_segment_raw")] == []
    # agent 层只有 finish 对（无 body 镜像）
    assert len(recs) == 2 and {r.role for r in recs} == {"user"}
    finish_tool = await _get_finish_tool(mem, asc)
    assert finish_tool.content == render_task_record("finish", task_id="t1", title="测试任务", outcome="success", body="真实段总结"), (
        f"slot hit: tool content must be '真实段总结' (+ task marker); got {finish_tool.content!r}"
    )
    finish_asst = await _get_finish_asst(mem, asc)
    # 反转契约：finish 对 assistant 槽 = act_recap（"真实段_act"，≠ 答复）
    assert "真实段_act" in finish_asst.content, (
        f"slot hit: assistant content must be act_recap '真实段_act'; got {finish_asst.content!r}"
    )
    # user 锚点 + 最终段 raw 留 task 层
    body = await mem.recall_recent(tsc, [T.USER_PROMPT, T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx())
    assert any(r.role == "user" and "初始请求" in r.content for r in body), "user 锚点留 task 层"
    assert len(body) == 3, "最终段 raw body 留 task 层（task-resident）"


@pytest.mark.asyncio
async def test_degraded_keeps_task_layer_raw_body() -> None:
    """task-resident：close 不镜像 body——降级（槽空）时 agent 层仍只 finish 对，
    最终段 raw body 留 task 层（不被 supersede）。"""
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()
    await _ingest_user_plus_raw(mem, tsc)

    task = _make_task()
    # 槽空 → 占位 + register

    await _synthesize_dispatch_pair(mem, asc, task, "占位_act", "占位_sum", "success", _pctx())

    recs = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 500, _pctx())
    # 无 final_segment_raw 镜像（agent 层不再镜像 body）
    assert [r for r in recs if r.metadata.get("final_segment_raw")] == []
    assert len(recs) == 2 and {r.role for r in recs} == {"user"}
    # 最终段 raw body 留 task 层
    body = await mem.recall_recent(tsc, [T.LLM_RESPONSE, T.TOOL_RESULT], 500, _pctx())
    assert len(body) == 2, "降级时最终段 raw body 留 task 层（task-resident）"


# ─── TDD: 两段化核心行为 ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_synthesize_dispatch_pair_two_segments() -> None:
    """_synthesize_dispatch_pair 写两段化 finish 对（反转契约 spec 2026-07-01）：
    assistant content = act_recap（过程复述，≠ 答复），tool content = task_summary；
    finish_task 退化为无参标记（input = {}，不再塞 result）。
    """
    from ctx_weft.core.loop.steps.finalize import _synthesize_dispatch_pair
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("t1")
    asc = _agent_scope()
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "初始请求", 1, role="user"), _pctx())

    task = _make_task(outputs="最终产出文本")
    await _synthesize_dispatch_pair(mem, asc, task, "本段我做了 A、B", "整段：A→B→验证，已就绪", "success", _pctx())

    turns = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 50, _pctx())
    asst = [r for r in turns if task_record_kind(r) == "recap"]
    tool = [r for r in turns if task_record_kind(r) == "finish"]
    # assistant 槽 = act_recap（过程复述）；答复由内联 body / blackboard 承载，不在此重复
    assert asst and "本段我做了 A、B" in asst[0].content
    assert tool and "整段：A→B→验证，已就绪" in tool[0].content
    assert task_record_group(asst[0]) == task_record_group(tool[0])
    assert asst[0].timestamp < tool[0].timestamp
    assert all(not r.metadata.get("tool_calls") for r in turns)



@pytest.mark.filterwarnings("ignore::pytest.PytestWarning")
def test_empty_close_has_no_fabricated_report():
    text = render_task_record("finish", task_id="t1", outcome="success", body="")
    assert text == '<task_finish task_id="t1" outcome="success">\n\n</task_finish>'



@pytest.mark.asyncio
async def test_partial_finish_is_not_ready_and_replay_is_idempotent(monkeypatch):
    mem = InMemoryMemoryProvider()
    task = _make_task()
    scope = _agent_scope()
    original = mem.ingest
    count = 0
    async def fail_second(event, ctx):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("interrupted between finish slots")
        return await original(event, ctx)
    monkeypatch.setattr(mem, "ingest", fail_second)
    with pytest.raises(OSError):
        await _synthesize_dispatch_pair(mem, scope, task, "", "", "success", _pctx(), operation_id="obs-partial")
    assert await find_finish_target(mem, _pctx(), scope, task.id) is None
    monkeypatch.setattr(mem, "ingest", original)
    await _synthesize_dispatch_pair(mem, scope, task, "", "", "success", _pctx(), operation_id="obs-partial")
    target = await find_finish_target(mem, _pctx(), scope, task.id)
    assert target is not None and target[2]["preserve_raw"]
    rows = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 100, _pctx())
    assert len(rows) == 2
