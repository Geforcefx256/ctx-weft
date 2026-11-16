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

from tests.model_testkit import model_snapshot


from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop import task_memory_fold as fp
from ctx_weft.core.loop.steps.finalize import (
    finalize_task_memory,
)
from tests.task_memory_testkit import render_task_record
from ctx_weft.core.loop.task_memory_fold import task_record_kind, task_record_group
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    load_records,
)
from ctx_weft.protocols.capability import qualify
from ctx_weft.protocols.template import LoopConfig
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import recalled_blocks


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_REPLY = "已把 config 解析拆到 config/parse.py，入口保持 load_config() 不变。"
FINISH_TASK = qualify("control:finish_task")



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
    return await load_records(mem,
        MemoryAddress(session_id="s1", task_id=None, agent_id=agent_id),
        MemoryScope.AGENT, _pctx(), kinds=[MemoryKind.CONVERSATION_TURN])


def _has_finish_call(rec) -> bool:
    return task_record_kind(rec) in {"recap", "final_reply"}


# ─── 三槽形态 ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_writes_recap_then_reply_then_report() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 3, f"长任务 close 应写三槽；实得 {[(r.role, r.content[:20]) for r in turns]}"
    recap, anchor, report = turns
    assert (recap.role, anchor.role, report.role) == ("user", "assistant", "user")
    assert "recap" in recap.content and task_record_kind(recap) == "recap", \
        "recap 槽只放过程复述（+ 收束尾注）、不挂 finish_task"
    assert _has_finish_call(anchor), "finish_task{} 必须挂在答复那条消息上"
    assert _REPLY in anchor.content and anchor.metadata.get("final_reply") is True
    assert task_record_group(report) == \
        task_record_group(anchor), "报告须与锚点配对"
    assert recap.timestamp < anchor.timestamp <= report.timestamp


@pytest.mark.asyncio
async def test_anchor_preserves_plain_reply() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    anchor = (await _agent_turns(mem))[1]
    assert anchor.content == _REPLY
    assert task_record_kind(anchor) == "final_reply"



@pytest.mark.asyncio
async def test_root_task_without_title_keeps_identity_in_metadata() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(title=""), scope)

    anchor = (await _agent_turns(mem))[1]
    assert anchor.content == _REPLY
    assert anchor.metadata["origin_task_id"] == "t1"


@pytest.mark.asyncio
async def test_untitled_record_keeps_its_kind() -> None:
    assert render_task_record("final_reply", body="x") == "x"



@pytest.mark.asyncio
async def test_task_layer_keeps_no_anchor_record() -> None:
    """答复归 agent 层的 finish 对；task 层仍是纯遗忘（只剩 USER_PROMPT + 段摘要）。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    view = await load_records(mem, scope, MemoryScope.TASK, _pctx(),
                               kinds=[MemoryKind.CONVERSATION_TURN])
    assert [r.role for r in view] == ["user"], \
        f"task 层末段 raw 应被纯遗忘；实得 {[(r.role, str(r.content)[:20]) for r in view]}"


# ─── 退回两槽：短任务 / 无产出 ────────────────────────────────────────────────

@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_empty_outputs_keeps_two_slot_shape() -> None:
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(outputs=None), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 2 and _has_finish_call(turns[0])


# ─── 延迟折叠：占位两槽 → bg 真报告落地后升三槽 ──────────────────────────────

@pytest.mark.asyncio
async def test_deferred_close_upgrades_to_three_slots_on_bg_report() -> None:
    """占位 close 时末段 raw 还在（答复还在 raw 里）→ 先两槽；bg 真报告落地、raw 补删的
    同时升成三槽，锚点补位。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope, has_llm_summary=False)
    target_scope, call, policy = await fp.find_finish_target(mem, _pctx(), scope, "t1")
    await fp.replace_finish_report(mem, _pctx(), target_scope, "t1", call,
                                  "bg 真 recap", "bg 真 summary", policy["outcome"], _task().title,
                                  final_reply=_REPLY)
    from ctx_weft.core.loop.steps.finalize import _supersede_final_raw_segment
    await _supersede_final_raw_segment(mem, scope, _pctx())

    turns = await _agent_turns(mem)
    assert len(turns) == 3, f"bg 报告已到 → 升三槽；实得 {[(r.role, r.content[:24]) for r in turns]}"
    recap, anchor, report = turns
    assert "bg 真 recap" in recap.content, "recap 槽换成 bg 真报告"
    assert "bg 真 summary" in report.content
    assert _REPLY in anchor.content and _has_finish_call(anchor)


@pytest.mark.asyncio
async def test_bg_replacement_does_not_clobber_the_anchor() -> None:
    """bg 事后重写 finish 对时锚点不能被当成 recap 槽冲掉——它才是挂着 finish_task 的那条，
    按 tool_call_id 找 assistant 会先命中它。"""
    from ctx_weft.core.loop.task_memory_fold import replace_finish_report
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _task()

    await _close(mem, task, scope, has_llm_summary=True)
    synth_scope, tool_call_id, policy = await fp.find_finish_target(mem, _pctx(), scope, "t1")
    outcome = policy["outcome"]
    await replace_finish_report(mem, _pctx(), synth_scope, "t1", tool_call_id,
                                 "bg 真 recap", "bg 真 summary", outcome, task.title,
                                 final_reply=_REPLY)

    turns = await _agent_turns(mem)
    assert len(turns) == 3, f"替换后仍是三槽；实得 {[(r.role, r.content[:24]) for r in turns]}"
    recap, anchor, report = turns
    assert "bg 真 recap" in recap.content
    assert _REPLY in anchor.content and _has_finish_call(anchor), "锚点不得被 recap 冲掉"
    assert "bg 真 summary" in report.content


# ─── 渲染 ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rendered_messages_put_finish_call_on_the_reply() -> None:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextRequest
    from tests.assembler_testkit import history_messages
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    await _close(mem, _task(), scope)

    deps = AssemblerDeps(memory=mem, knowledge_providers=[], provider_ctx=_pctx())
    req = ContextRequest(resolved_model=model_snapshot(100_000, 0), purpose="act", scope=_sc("t2"), task=None, agent=None, session=None,
                         template=None, bound_capabilities=[])
    blocks = await recalled_blocks(req, deps)
    msgs = history_messages(blocks, task_id="t2")

    tail = msgs[-3:]
    assert [m.role for m in tail] == ["user", "assistant", "user"], \
        f"尾部应是 recap → 答复 → 报告；实得 {[(m.role, str(m.content)[:24]) for m in msgs]}"
    assert not tail[0].tool_calls, "recap 那条不带 tool_calls"
    assert all(not message.tool_calls for message in tail)
    assert _REPLY in tail[1].content
    assert "<task_finish" in str(tail[2].content)


@pytest.mark.asyncio
async def test_dispatch_ack_points_at_the_final_reply() -> None:
    """同 agent 派发 ack 是内联执行的导读：末尾是 finish_task 与它带的最终回复。"""
    from ctx_weft.core.loop.steps.finalize import _dispatch_ack
    ack = _dispatch_ack("抽取配置解析", "success")
    assert "final reply" in ack.lower(), f"导读须提到最终回复；实得 {ack!r}"


@pytest.mark.asyncio
async def test_segment_summary_keeps_its_task_identity_as_a_user_recap() -> None:
    """Current and historical task summaries use user recaps with their source task ID."""
    from ctx_weft.core.assembler.sources._history import record_to_history_block
    from tests.assembler_testkit import compile_prompt, prompt_request
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await mem.ingest(_ev(T.USER_PROMPT, scope, "hello", 0, role="user", task_id="t1"), _pctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, scope, "第1段做了啥", 5,
                         role="assistant", task_id="t1"), _pctx())

    view = await load_records(mem, scope, MemoryScope.TASK, _pctx(),
                               kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY])
    summary = next(r for r in view if r.role == "assistant")
    block = record_to_history_block(summary, source="agent_recall", idx=0)
    assert block.value is summary, "source 必须保持原始记忆，不在这里加标题"

    own = compile_prompt([block], prompt_request(task_id="t1"))
    other = compile_prompt([block], prompt_request(task_id="t2"))
    own_text = "\n".join(str(message.content) for message in own.messages)
    other_text = "\n".join(str(message.content) for message in other.messages)
    for prompt, text in ((own, own_text), (other, other_text)):
        assert prompt.messages[0].role == "user"
        assert '<task_recap task_id="t1">\n第1段做了啥\n</task_recap>' in text
        assert "## Progress So Far" not in text


# ─── recap 槽的收束尾注 ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recap_slot_carries_closing_note() -> None:
    """recap 槽是 agent 层普通 assistant 回合，拿不到段摘要那条尾注
    （annotate_assistant_summary 只贴 TASK_COMPACT_SUMMARY）。它形似「我上一轮就是这么答的」，
    同样需要一句系统注解说明它是过程复述、不是答复。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(), scope)

    recap = (await _agent_turns(mem))[0]
    assert "recap" in recap.content, "复述正文仍在最前"
    assert recap.content.endswith("</task_recap>")


@pytest.mark.asyncio
async def test_two_slot_recap_also_carries_the_note() -> None:
    """两槽形态下 recap 槽同样会被模仿（它还挂着 finish_task），一视同仁。"""
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)

    await _close(mem, _task(outputs=None), scope)

    turns = await _agent_turns(mem)
    assert len(turns) == 2
    assert turns[0].content.endswith("</task_recap>")


@pytest.mark.asyncio
async def test_bg_rewritten_recap_keeps_the_note() -> None:
    """bg 事后用真报告重写 recap 槽时，注解不能丢（两处形态共用 build_finish_slots）。"""
    from ctx_weft.core.loop.task_memory_fold import (
        replace_finish_report,
    )
    mem = InMemoryMemoryProvider()
    scope = _sc("t1")
    await _seed_long_conv(mem, scope)
    task = _task()

    await _close(mem, task, scope, has_llm_summary=True)
    synth_scope, tool_call_id, policy = await fp.find_finish_target(mem, _pctx(), scope, "t1")
    outcome = policy["outcome"]
    await replace_finish_report(mem, _pctx(), synth_scope, "t1", tool_call_id,
                                 "bg 真 recap", "bg 真 summary", outcome, task.title)

    recap = (await _agent_turns(mem))[0]
    assert "bg 真 recap" in recap.content
    assert recap.content.endswith("</task_recap>")
