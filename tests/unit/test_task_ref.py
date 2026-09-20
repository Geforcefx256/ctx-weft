"""spec: task-handoff——任务称呼的规范形式：id 与标题恒同时出现。

模型用**标题**建立心智模型（act 的任务树 / 已完成清单 / 派发对 / finish 对归属），
却用 **id** 指名（observer 在 `next_step_hint` 里点名子任务）。这里钉死两件事：形式只有一种，
且每张模型可见的脸上两者都在——否则模型得在中间做一次没有凭据的映射。
"""

from __future__ import annotations

from types import SimpleNamespace

from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.assembler import ContextBlock
from ctx_weft.core.loop.steps.act_guidance import build_act_guidance, build_resume_cue
from ctx_weft.core.utils.task_ref import task_ref, task_ref_parts


# ── 规范形式 ─────────────────────────────────────────────────────────────────


def test_canonical_form_is_quoted_title_then_parenthesised_id():
    assert task_ref_parts("tsk_1", "Build the parser") == "'Build the parser' (tsk_1)"
    assert task_ref(SimpleNamespace(id="tsk_1", title="Build the parser")) \
        == "'Build the parser' (tsk_1)"


def test_untitled_degrades_to_bare_id_not_empty_quotes():
    assert task_ref_parts("tsk_1", "") == "tsk_1"
    assert task_ref_parts("tsk_1", "   ") == "tsk_1"


def test_nothing_identifying_yields_empty_so_callers_keep_their_fallback():
    """两者皆空 → 空串。不编 `(unidentified task)` 占位：那既无信息量，又会让
    调用点原有的兜底分支（如 finish 对的 UNTITLED 变体）变成死代码。"""
    assert task_ref_parts("", "") == ""


def test_missing_attrs_do_not_raise():
    """鸭子类型：Task / TaskView / 任何投影都能传进来。"""
    assert task_ref(SimpleNamespace()) == ""


# ── 模型可见面：两者必须同时出现 ─────────────────────────────────────────────


def _cur(task_id="t1", title="T"):
    return SimpleNamespace(id=task_id, title=title, description="", interaction_mode="auto",
                           user_prompt="", next_step_hint=None)


def _t(task_id, title, status="PENDING", parent=None):
    return SimpleNamespace(id=task_id, title=title, description="", status=status,
                           parent_task_id=parent, created_at=None, user_prompt="",
                           outputs=None, task_summary=None)


def _tm(tasks):
    return SimpleNamespace(all_tasks=lambda: tasks)


def test_act_guidance_plan_tree_carries_both():
    """任务树是模型建立「计划长什么样」的地方——它随后要按 id 操作其中的子任务。"""
    cur = _t("t1", "Write tests", "ACTIVE")
    other = _t("t2", "Ship it", "PENDING")
    g = build_act_guidance(_cur("t1", "Write tests"), _tm([cur, other]))
    for task_id, title in (("t1", "Write tests"), ("t2", "Ship it")):
        assert task_ref_parts(task_id, title) in g


def test_act_guidance_completed_children_carry_both():
    """这些正是 observe 阶段要按 id 指名的任务。"""
    parent = _t("t1", "Parent", "ACTIVE")
    child = _t("t2", "Research", "FINISHED", parent="t1")
    g = build_act_guidance(_cur("t1", "Parent"), _tm([parent, child]))
    assert "ALREADY COMPLETED" in g
    assert task_ref_parts("t2", "Research") in g


def test_act_guidance_anchor_and_resume_cue_carry_both():
    g = build_act_guidance(_cur("t1", "Fix importer"), _tm([]))
    assert f"Current task: {task_ref_parts('t1', 'Fix importer')}" in g.split("\n")
    cue = build_resume_cue(_cur("t1", "Fix importer"), _tm([]))
    assert task_ref_parts("t1", "Fix importer") in cue


def _observer_cue(subtasks):
    req = SimpleNamespace(
        purpose="observe", scope=SimpleNamespace(session_id="s1", task_id="t1", agent_id="a1"),
        task=SimpleNamespace(id="t1", title="parent", description="", user_prompt="",
                             user_prompt_in_memory=False, process_report=None, outputs=None),
        agent=SimpleNamespace(id="a1"), session=SimpleNamespace(id="s1"),
        template=None, bound_capabilities=[], extra={"subtasks": subtasks},
        token_counter=len,
    )
    blocks = [ContextBlock(id="b1", source="t", kind="history", target="messages",
                           content="do the work", priority=3, token_estimate=3,
                           metadata={"role": "user", "type": "user_prompt",
                                     "timestamp": "2026-06-30T00:00:00+00:00"})]
    return DefaultComposer()._build_observer_messages(blocks, req)[-1].content


def test_observer_review_face_carries_both_for_every_entry():
    """回归：曾因一次缩进错误只渲染出最后一条。"""
    cue = _observer_cue([
        {"task_id": "tsk_a", "title": "Greet Amy", "outcome": "finished"},
        {"task_id": "tsk_b", "title": "Greet Lily", "outcome": "failed"},
    ])
    assert task_ref_parts("tsk_a", "Greet Amy") in cue
    assert task_ref_parts("tsk_b", "Greet Lily") in cue


def test_observer_subtask_instruction_says_task_id_not_task_title():
    """指令行是这条链上最吃重的一句：它决定模型用哪个键指名子任务。"""
    cue = _observer_cue([{"task_id": "tsk_a", "title": "Greet Amy", "outcome": "finished"}])
    assert "task_id" in cue
    assert "task_title" not in cue


def test_current_task_frame_carries_both():
    """任务自己的 `## Current Task` 框——此前它连自己的 id 都不知道。"""
    blk = ContextBlock(
        id="ts1", source="task_spec", kind="task_spec", target="messages",
        content="x", priority=9, token_estimate=1,
        metadata={"task_id": "t1", "title": "Build the parser",
                  "description": "D", "user_prompt": "P"})
    req = SimpleNamespace(
        purpose="act", scope=SimpleNamespace(session_id="s1", task_id="t1", agent_id="a1"),
        task=SimpleNamespace(id="t1", title="Build the parser", description="D",
                             user_prompt="P", user_prompt_in_memory=False, process_report=None),
        agent=SimpleNamespace(id="a1"), session=SimpleNamespace(id="s1"),
        template=None, bound_capabilities=[], extra={}, token_counter=len)
    text = "\n".join(m.content for m in DefaultComposer()._build_actor_messages([blk], req)
                     if isinstance(m.content, str))
    assert f"## Current Task\n{task_ref_parts('t1', 'Build the parser')}" in text
