"""close 末段 raw 延迟折叠（spec 2026-07-20 deferred-close-raw-fold）：

close 时若尚无 LLM 总结（规则 observe 占位，has_llm_summary=False），末段 raw 不删、
保持 active；折叠推迟到真摘要落地（bg 替换成功 / slot 命中）之后。bg 失败 → raw 永久保留。
不变量：末段 raw 与「真实 Process Report」二者始终至少存其一。
"""

from __future__ import annotations

from ctx_weft.core.loop.task_memory_fold import task_record_kind

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import ctx_weft.core.loop.steps.observe as _obs_mod
from ctx_weft.core.capabilities.control_tools import (
    REPORT_OBSERVATION_NAME,
)
from ctx_weft.core.capabilities.control_tools import ControlMetaKey as K
from ctx_weft.core.loop import task_memory_fold as fp
from ctx_weft.core.loop.driver import StepDriver
from ctx_weft.core.loop.steps.finalize import finalize_task_memory
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)



def _pctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _sc(task_id: str | None, agent_id: str = "ag1") -> MemoryAddress:
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent_id)


class _FakeTM:
    def children_of(self, task_id: str) -> set[str]:
        return set()


def _state(task: Task, scope: MemoryAddress):
    agent = SimpleNamespace(id=scope.agent_id, loop_config=LoopConfig())
    session = SimpleNamespace(id="s1", tenant_id="default")
    return SimpleNamespace(run_id="run1", sequence_counter=0, session=session,
                           scope=scope, task=task, agent=agent)


def _loop_ctx(mem):
    return SimpleNamespace(memory=mem, provider_ctx=_pctx(), task_manager=_FakeTM(),
                           llm=SimpleNamespace(tokenizer=HeuristicTokenizer()))


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(type=type_, address=scope, content=content,
                       timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta)


async def _seed_long_conv(mem, scope) -> None:
    """user 锚点 + 5 轮大体量 assistant（> turn_cap=2 且 > token 阈值 → 非 short）。"""
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hello", 0, role="user"), _pctx())
    for i in range(5):
        await mem.ingest(
            _ev(T.LLM_RESPONSE, scope, "x " * 4000, i + 1, role="assistant"), _pctx())


def _root_task(task_id="t1", agent="ag1") -> Task:
    return Task(id=task_id, session_id="s1", status="FINISHED", tenant_id="default",
                assigned_agent_id=agent, creator_agent_id=agent, parent_task_id=None,
                title="Root", user_prompt="hello", settings=NormalTaskSettings())


async def _active_raw(mem, scope) -> list:
    return await mem.recall_recent(scope, [T.LLM_RESPONSE], 100, _pctx())


# ─── TEST 1: 无 LLM 总结的 close 不删 raw、登记延迟折叠 ───────────────────────

@pytest.mark.asyncio
async def test_close_without_llm_summary_keeps_raw_and_registers_fold() -> None:
    """规则 observe close（has_llm_summary=False）：末段 raw 保持 active；
    _close_synth 登记扩为 4 元组，第 4 元 = raw 所在 task scope（供 bg 替换后补删）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _root_task()

    await finalize_task_memory(mem, _state(task, scope), task, "out", "success",
                               _loop_ctx(mem), act_recap="占位 recap", task_summary="",
                               has_llm_summary=False)

    assert await _active_raw(mem, scope), \
        "无 LLM 总结时 close 不得删末段 raw（占位 finish 对不承载执行内容）"
    target = await fp.find_finish_target(mem, _pctx(), scope, "t1")
    assert target is not None and not target[2]["preserve_raw"]


# ─── TEST 2: 有 LLM 总结的 close 照旧同步删 raw（回归）──────────────────────

@pytest.mark.asyncio
async def test_close_with_llm_summary_supersedes_raw_at_close() -> None:
    """前台 LLM observe close（has_llm_summary=True）：行为不变，close 即删末段 raw。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _root_task()

    await finalize_task_memory(mem, _state(task, scope), task, "out", "success",
                               _loop_ctx(mem), act_recap="真 recap", task_summary="真总结",
                               has_llm_summary=True)

    assert await _active_raw(mem, scope) == [], \
        "有 LLM 总结时 close 应同步 supersede 末段 raw（既有行为）"


# ─── TEST 3: slot 命中（bg 先完成）→ 替换后立即补删 raw ───────────────────────

@pytest.mark.asyncio
async def test_slot_hit_folds_raw_immediately() -> None:
    """bg 先完成（_close_report 预置）+ has_llm_summary=False：
    finish 对直接用真报告，且末段 raw 随之补删（真摘要已落地）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _root_task()

    await finalize_task_memory(mem, _state(task, scope), task, "out", "success",
                               _loop_ctx(mem), act_recap="占位 recap", task_summary="",
                               has_llm_summary=False)

    import asyncio

    from ctx_weft.core.loop.steps.observe import ObservationResult, apply_close_report
    state = _state(task, scope)
    state.extra = {"observe_id": "obs-long-report"}
    await apply_close_report(state, _loop_ctx(mem), ObservationResult("bg_act", "bg_sum"), asyncio.Event())
    turns = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 100, _pctx())
    tool = [r for r in turns if task_record_kind(r) == "finish"]
    assert tool and "bg_sum" in tool[0].content, \
        f"slot 命中 finish tool 应为真报告；实得 {[r.content for r in tool]!r}"
    assert await _active_raw(mem, scope) == [], \
        "真摘要已落地（slot 命中替换后）应立即补删末段 raw"


# ─── TEST 4: 同 agent 子任务延迟折叠登记的 raw scope = 子 task scope ──────────

@pytest.mark.asyncio
async def test_same_agent_child_defers_with_child_task_scope() -> None:
    """同 agent 子任务 close（嵌套 finish 对写 parent scope）：raw_fold_scope 必须是
    子任务自己的 task scope（raw 所在层），不是 finish 对所在的 parent scope。"""
    mem = InMemoryMemoryProvider()
    child_scope = _sc("c1")
    await _seed_long_conv(mem, child_scope)
    child = Task(id="c1", session_id="s1", status="FINISHED", tenant_id="default",
                 assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id="p1",
                 origin_tool_call_id="tc1", origin_tool_name="control__delegate_task",
                 title="Child", user_prompt="do", settings=NormalTaskSettings())

    await finalize_task_memory(mem, _state(child, child_scope), child, "out", "success",
                               _loop_ctx(mem), act_recap="占位 recap", task_summary="",
                               has_llm_summary=False)

    assert await _active_raw(mem, child_scope), "无 LLM 总结时子任务末段 raw 也须保留"
    target = await fp.find_finish_target(mem, _pctx(), child_scope, "c1")
    assert target is not None and not target[2]["preserve_raw"]
    assert target[0].task_id == "p1"


# ─── bg 侧：替换成功 → 补删；无可用报告 → raw 保留 ───────────────────────────

def _tool_call_chunk():
    return SimpleNamespace(
        kind="tool_call",
        tool_call=SimpleNamespace(id="tc_obs", name=REPORT_OBSERVATION_NAME,
                                  arguments={"task_status": "success", "act_recap": "真报告act"}),
        text="", usage=None,
    )


def _token_chunk(text: str):
    return SimpleNamespace(kind="token", text=text, tool_call=None, usage=None)


def _usage_chunk():
    from ctx_weft.protocols import LLMUsage
    return SimpleNamespace(kind="usage", usage=LLMUsage(prompt_tokens=1, completion_tokens=1),
                           tool_call=None, text="")


class _FakeGateway:
    async def invoke(self, *, tool_name, arguments, state, ctx, tool_call_id):
        # 返回形态对齐生产 InvocationResult（含 is_error）——run_observe_react
        # 读该字段判定 terminal 失败，缺字段会 AttributeError。
        from types import SimpleNamespace
        # 对齐 `report_observation`：content 是话术，干净的 recap 在 metadata 里。
        return SimpleNamespace(
            content="Assessment recorded: outcome=success. 真报告act",
            is_error=False,
            metadata={K.OBSERVER_OUTCOME: "success",
                      K.OBSERVER_ACT_RECAP: "真报告act",
                      K.OBSERVER_TASK_SUMMARY: "真报告summary"})


async def _preset_placeholder_pair(mem, scope, task_id: str, pctx) -> None:
    ts = datetime.now(UTC)
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=scope, content="占位 recap",
        timestamp=ts, role="assistant",
        metadata={"origin_task_id": task_id, "finish_policy": {"preserve_raw": False, "outcome": "success", "slot_count": 2},
                  "tool_calls": [{"id": "tc9", "name": "control__finish_task", "input": {}}]},
    ), pctx)
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=scope, content="占位 summary",
        timestamp=ts, role="tool",
        metadata={"origin_task_id": task_id, "finish_policy": {"preserve_raw": False, "outcome": "success", "slot_count": 2}, "tool_call_id": "tc9"},
    ), pctx)


@pytest.mark.asyncio
async def test_bg_replace_success_folds_raw(monkeypatch, fake_state_ctx) -> None:
    """close 边界 bg 回调：_replace_finish_report 成功后按登记的 raw_fold_scope 补删末段 raw。"""
    state, ctx = fake_state_ctx  # task 层预置 [UP, LLM, TOOL]
    ctx.capability_gateway = _FakeGateway()
    state.agent.loop_config = SimpleNamespace(compact_keep_last=2, max_turns_per_observe=2)
    state.session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)
    await _preset_placeholder_pair(ctx.memory, state.scope, state.task.id, ctx.provider_ctx)

    async def _stream(c, s, req, **kwargs):
        yield _token_chunk("…")
        yield _tool_call_chunk()
        yield _usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _stream)
    await StepDriver.for_context(ctx).start("observe", state, ctx, options={"boundary": "finish"})

    turns = await ctx.memory.recall_recent(
        state.scope, [T.AGENT_CONVERSATION_TURN], 100, ctx.provider_ctx)
    asst = [r for r in turns if task_record_kind(r) == "recap"]
    assert asst and "真报告act" in asst[0].content, "finish 对应已被 bg 真报告替换"
    raw = await ctx.memory.recall_recent(
        state.scope, [T.LLM_RESPONSE, T.TOOL_RESULT], 100, ctx.provider_ctx)
    assert raw == [], "真摘要落地后应按 raw_fold_scope 补删末段 raw"
    up = await ctx.memory.recall_recent(state.scope, [T.USER_PROMPT], 100, ctx.provider_ctx)
    assert up, "USER_PROMPT 锚点保留"


@pytest.mark.asyncio
async def test_bg_no_usable_report_keeps_raw_and_placeholder(monkeypatch, fake_state_ctx) -> None:
    """close 边界 bg 无可用报告：raw 保留（延迟折叠不触发）、占位 finish 对不动、登记弹掉。"""
    state, ctx = fake_state_ctx
    ctx.capability_gateway = _FakeGateway()
    state.agent.loop_config = SimpleNamespace(compact_keep_last=2, max_turns_per_observe=2)
    state.session = SimpleNamespace(id="s1", tenant_id="default", token_used=0)
    await _preset_placeholder_pair(ctx.memory, state.scope, state.task.id, ctx.provider_ctx)

    async def _empty(c, s, req, **kwargs):
        yield _usage_chunk()

    monkeypatch.setattr(_obs_mod, "stream_llm_resilient", _empty)
    await StepDriver.for_context(ctx).start("observe", state, ctx, options={"boundary": "finish"})

    raw = await ctx.memory.recall_recent(
        state.scope, [T.LLM_RESPONSE, T.TOOL_RESULT], 100, ctx.provider_ctx)
    assert len(raw) == 2, "bg 失败时末段 raw 必须保留（降级 = 保 raw）"
    turns = await ctx.memory.recall_recent(
        state.scope, [T.AGENT_CONVERSATION_TURN], 100, ctx.provider_ctx)
    assert sorted(r.content for r in turns) == ["占位 recap", "占位 summary"], \
        "占位 finish 对保持原样"
