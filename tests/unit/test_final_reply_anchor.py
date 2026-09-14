"""最终回复锚点：close 合成的 finish 对里那条「答复 + finish_task」消息。

反转契约（spec 2026-07-01）下最终答复正文写在收尾回合的普通消息里，而 close 把整个末段
raw 删掉、也不另产段摘要 —— 答复在胶囊里无处安放。补法是把 finish 对从两条扩成三条：

    assistant  act_recap                        ← 过程复述，不挂 tool_calls
    assistant  提示 + task.outputs + 收束尾注     ← 锚点，finish_task{} 挂在这条
    tool       process report                   ← 与锚点配对

于是重建出的历史示范了 finish_task 的真实用法（答复正文与收尾调用同一条消息）。末段 raw
未折的场景（short leaf / outputs 为空）退回两条形态，避免答复出现两遍。
"""

from __future__ import annotations

from ctx_weft.core.utils.task_ref import task_ref_parts

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import ctx_weft.core.loop.steps.background_observe as bo
from ctx_weft.core.loop.steps.finalize import (
    FINAL_REPLY_NOTE, FINAL_REPLY_NOTE_UNTITLED, finalize_task_memory,
)
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    MemoryAddress, MemoryEvent, MemoryEventType, MemoryKind, MemoryScope, ProviderContext,
)
from ctx_weft.protocols.capability import qualify
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_REPLY = "已把 config 解析拆到 config/parse.py，入口保持 load_config() 不变。"
FINISH_TASK = qualify("control:finish_task")


@pytest.fixture(autouse=True)
def _clear_bg_state():
    bo._close_report.clear()
    bo._close_synth.clear()
    yield
    bo._close_report.clear()
    bo._close_synth.clear()


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
    """user 锚点 + 5 轮大体量 assistant（> turn_cap 且 > token 阈值 → 非 short）。"""
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hello", 0, role="user", task_id="t1"), _pctx())
    for i in range(5):
        await mem.ingest(
            _ev(T.LLM_RESPONSE, scope, "x " * 4000, i + 1, role="assistant"), _pctx())


async def _seed_short_conv(mem, scope) -> None:
    """user 锚点 + 1 轮短 assistant（short leaf → 不折 raw）。"""
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hi", 0, role="user", task_id="t1"), _pctx())
    await mem.ingest(_ev(T.LLM_RESPONSE, scope, "done", 1, role="assistant"), _pctx())


def _task(task_id="t1", title="抽取配置解析", outputs=_REPLY, parent=None) -> Task:
    return Task(id=task_id, session_id="s1", status="FINISHED", tenant_id="default",
                assigned_agent_id="ag1", creator_agent_id="ag1", parent_task_id=parent,
                title=title, user_prompt="hello", outputs=outputs,
                settings=NormalTaskSettings())


async def _close(mem, task, scope, *, has_llm_summary=True):
    await finalize_task_memory(mem, _state(task, scope), task, "out", "success",
                               _loop_ctx(mem), act_recap="recap", task_summary="总结",
                               has_llm_summary=has_llm_summary)


async def _agent_turns(mem, agent_id="ag1") -> list:
    return await mem.load_view(
        MemoryAddress(session_id="s1", task_id=None, agent_id=agent_id),
        MemoryScope.AGENT, _pctx(), kinds=[MemoryKind.CONVERSATION_TURN])


def _has_finish_call(rec) -> bool:
    return any(tc.get("name") == FINISH_TASK for tc in (rec.metadata.get("tool_calls") or []))


# ─── 三槽形态 ────────────────────────────────────────────────────────────────

async def test_close_writes_recap_then_reply_then_report() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 3, f"长任务 close 应写三槽；实得 {[(r.role, r.content[:20]) for r in turns]}"
    recap, anchor, report = turns
    assert (recap.role, anchor.role, report.role) == ("assistant", "assistant", "tool")
    assert recap.content.startswith("recap") and not _has_finish_call(recap), \
        "recap 槽只放过程复述（+ 收束尾注）、不挂 finish_task"
    assert _has_finish_call(anchor), "finish_task{} 必须挂在答复那条消息上"
    assert _REPLY in anchor.content and anchor.metadata.get("final_reply") is True
    assert report.metadata.get("tool_call_id") == \
        (anchor.metadata.get("tool_calls") or [{}])[0].get("id"), "报告须与锚点配对"
    assert recap.timestamp < anchor.timestamp <= report.timestamp


async def test_anchor_wraps_reply_with_notes() -> None:
    from ctx_weft.core.loop.steps.finalize import FINAL_REPLY_CLOSING_NOTE
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    anchor = (await _agent_turns(mem))[1]
    assert anchor.content == (
        f"{FINAL_REPLY_NOTE.format(ref=task_ref_parts('t1', '抽取配置解析'))}\n\n{_REPLY}\n\n{FINAL_REPLY_CLOSING_NOTE}"
    ), f"锚点 = 提示词 + outputs 正文 + 收束尾注；实得 {anchor.content!r}"


async def test_root_task_without_title_falls_back_to_bare_id() -> None:
    """无标题 → 称呼退化为裸 id（仍是可用句柄）；UNTITLED 变体只在连 id 都没有时才用。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(title=""), scope)

    assert (await _agent_turns(mem))[1].content.startswith(
        FINAL_REPLY_NOTE.format(ref="t1"))


def test_untitled_note_used_only_when_nothing_identifies_the_task() -> None:
    from ctx_weft.core.loop.steps.finalize import _final_reply_block
    assert _final_reply_block(task_ref_parts("", ""), "x").startswith(FINAL_REPLY_NOTE_UNTITLED)


async def test_task_layer_keeps_no_anchor_record() -> None:
    """答复归 agent 层的 finish 对；task 层仍是纯遗忘（只剩 USER_PROMPT + 段摘要）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    view = await mem.load_view(scope, MemoryScope.TASK, _pctx(),
                               kinds=[MemoryKind.CONVERSATION_TURN])
    assert [r.role for r in view] == ["user"], \
        f"task 层末段 raw 应被纯遗忘；实得 {[(r.role, str(r.content)[:20]) for r in view]}"


# ─── 退回两槽：短任务 / 无产出 ────────────────────────────────────────────────

async def test_short_leaf_keeps_two_slot_shape() -> None:
    """short leaf 不折末段 raw（原文即胶囊），再塞锚点等于答复出现两遍。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_short_conv(mem, scope)

    await _close(mem, _task(), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 2 and _has_finish_call(turns[0]), \
        f"短任务应退回两槽；实得 {[(r.role, r.content[:20]) for r in turns]}"
    assert await mem.recall_recent(scope, [T.LLM_RESPONSE], 100, _pctx()), "短任务 raw 照留"


async def test_empty_outputs_keeps_two_slot_shape() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(outputs=None), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 2 and _has_finish_call(turns[0])


# ─── 延迟折叠：占位两槽 → bg 真报告落地后升三槽 ──────────────────────────────

async def test_deferred_close_upgrades_to_three_slots_on_bg_report() -> None:
    """占位 close 时末段 raw 还在（答复还在 raw 里）→ 先两槽；bg 真报告落地、raw 补删的
    同时升成三槽，锚点补位。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    bo._close_report["t1"] = ("bg 真 recap", "bg 真 summary")

    await _close(mem, _task(), scope, has_llm_summary=False)

    turns = await _agent_turns(mem)
    assert len(turns) == 3, f"bg 报告已到 → 升三槽；实得 {[(r.role, r.content[:24]) for r in turns]}"
    recap, anchor, report = turns
    assert recap.content.startswith("bg 真 recap"), "recap 槽换成 bg 真报告"
    assert "bg 真 summary" in report.content
    assert _REPLY in anchor.content and _has_finish_call(anchor)


async def test_bg_replacement_does_not_clobber_the_anchor() -> None:
    """bg 事后重写 finish 对时锚点不能被当成 recap 槽冲掉——它才是挂着 finish_task 的那条，
    按 tool_call_id 找 assistant 会先命中它。"""
    from ctx_weft.core.loop.steps.background_observe import _replace_finish_report, pop_close_synth
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _task()

    await _close(mem, task, scope, has_llm_summary=True)
    synth = pop_close_synth("t1")
    assert synth is not None, "前提：close 登记了 bg 替换槽"
    tool_call_id, synth_scope, outcome, _raw = synth
    await _replace_finish_report(mem, _pctx(), synth_scope, "t1", tool_call_id,
                                 "bg 真 recap", "bg 真 summary", outcome, task.title,
                                 final_reply=_REPLY)

    turns = await _agent_turns(mem)
    assert len(turns) == 3, f"替换后仍是三槽；实得 {[(r.role, r.content[:24]) for r in turns]}"
    recap, anchor, report = turns
    assert recap.content.startswith("bg 真 recap")
    assert _REPLY in anchor.content and _has_finish_call(anchor), "锚点不得被 recap 冲掉"
    assert "bg 真 summary" in report.content


# ─── 渲染 ────────────────────────────────────────────────────────────────────

async def test_rendered_messages_put_finish_call_on_the_reply() -> None:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextRequest
    from ctx_weft.core.assembler.composer import DefaultComposer
    from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    await _close(mem, _task(), scope)

    deps = AssemblerDeps(memory=mem, knowledge_providers=[], provider_ctx=_pctx())
    req = ContextRequest(purpose="act", scope=_sc("t2"), task=None, agent=None, session=None,
                         template=None, bound_capabilities=[])
    blocks = [b async for b in AgentRecallSource().fetch(req, deps)]
    msgs = DefaultComposer()._history_to_messages(blocks)

    tail = msgs[-3:]
    assert [m.role for m in tail] == ["assistant", "assistant", "tool"], \
        f"尾部应是 recap → 答复 → 报告；实得 {[(m.role, str(m.content)[:24]) for m in msgs]}"
    assert not tail[0].tool_calls, "recap 那条不带 tool_calls"
    assert [tc.get("name") for tc in tail[1].tool_calls] == [FINISH_TASK]
    assert _REPLY in tail[1].content


async def test_dispatch_ack_points_at_the_final_reply() -> None:
    """同 agent 派发 ack 是内联执行的导读：末尾是 finish_task 与它带的最终回复。"""
    from ctx_weft.core.loop.steps.finalize import _dispatch_ack
    ack = _dispatch_ack("抽取配置解析", "success")
    assert "final reply" in ack.lower(), f"导读须提到最终回复；实得 {ack!r}"


async def test_closed_capsule_summary_has_no_progress_heading() -> None:
    """`## Progress So Far` 是「当前 task 上一段的复述」标题。闭合后的胶囊由**别的** task
    装配，段摘要不得再冠这个标题（否则新任务会把旧任务的进度读成自己的）。"""
    from ctx_weft.core.assembler.sources._history import record_to_history_block
    from ctx_weft.core.assembler.sources._history import PROGRESS_SO_FAR_HEADING
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hello", 0, role="user", task_id="t1"), _pctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, scope, "第1段做了啥", 5,
                         role="assistant", task_id="t1"), _pctx())

    view = await mem.load_view(scope, MemoryScope.TASK, _pctx(),
                               kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY])
    summary = next(r for r in view if r.role == "assistant")
    request = SimpleNamespace(scope=scope, token_counter=len)

    own = record_to_history_block(summary, source="agent_recall", idx=0,
                                  request=request, current_task_id="t1")
    other = record_to_history_block(summary, source="agent_recall", idx=0,
                                    request=request, current_task_id="t2")
    assert own.content.startswith(PROGRESS_SO_FAR_HEADING), "当前 task 自己的段摘要仍要冠标题"
    assert PROGRESS_SO_FAR_HEADING not in other.content, \
        f"跨 task（闭合胶囊）不得冠 Progress So Far；实得 {other.content!r}"


# ─── recap 槽的收束尾注 ──────────────────────────────────────────────────────

async def test_recap_slot_carries_closing_note() -> None:
    """recap 槽是 agent 层普通 assistant 回合，拿不到段摘要那条尾注
    （annotate_assistant_summary 只贴 TASK_COMPACT_SUMMARY）。它形似「我上一轮就是这么答的」，
    同样需要一句系统注解说明它是过程复述、不是答复。"""
    from ctx_weft.core.loop.steps.finalize import PROCESS_RECAP_NOTE
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    recap = (await _agent_turns(mem))[0]
    assert recap.content.startswith("recap"), "复述正文仍在最前"
    assert recap.content.endswith(PROCESS_RECAP_NOTE), \
        f"recap 槽须以系统注解收束；实得 {recap.content!r}"


async def test_two_slot_recap_also_carries_the_note() -> None:
    """两槽形态下 recap 槽同样会被模仿（它还挂着 finish_task），一视同仁。"""
    from ctx_weft.core.loop.steps.finalize import PROCESS_RECAP_NOTE
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(outputs=None), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 2
    assert turns[0].content.endswith(PROCESS_RECAP_NOTE)


async def test_bg_rewritten_recap_keeps_the_note() -> None:
    """bg 事后用真报告重写 recap 槽时，注解不能丢（两处形态共用 build_finish_slots）。"""
    from ctx_weft.core.loop.steps.background_observe import _replace_finish_report, pop_close_synth
    from ctx_weft.core.loop.steps.finalize import PROCESS_RECAP_NOTE
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _task()

    await _close(mem, task, scope, has_llm_summary=True)
    tool_call_id, synth_scope, outcome, _raw = pop_close_synth("t1")
    await _replace_finish_report(mem, _pctx(), synth_scope, "t1", tool_call_id,
                                 "bg 真 recap", "bg 真 summary", outcome, task.title)

    recap = (await _agent_turns(mem))[0]
    assert recap.content.startswith("bg 真 recap")
    assert recap.content.endswith(PROCESS_RECAP_NOTE)
