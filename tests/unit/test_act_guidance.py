"""act guidance：build_act_guidance 内容 + GuidanceSource/composer 装配落位。

内容契约（loop/steps/act_guidance.py）：session 任务树 + 已完成子任务清单（动态段）
+ finish/切换/ask_user 指针级提醒（静态段，完整协议在工具 description/SOUL/observer 护栏）。
装配契约：PrepareStep → extra["act_guidance"] → GuidanceSource（kind="guidance"）→
composer 恒拼末条 user 最尾部（仅 act purpose；其后至多再补一行 Capabilities 指针，
清单全文随当前 task 回合落在 cache 前缀内）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from ctx_weft.core.assembler.assembler import ContextBlock
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.guidance import GuidanceSource
from ctx_weft.core.loop.steps.act_guidance import build_act_guidance
from ctx_weft.core.utils.estimate import estimate_tokens


def _task(id, title="", status="PENDING", parent=None, description="", created_at=None,
          user_prompt="", outputs=None, task_summary=None):
    return SimpleNamespace(
        id=id, title=title, description=description, status=status,
        parent_task_id=parent, created_at=created_at, user_prompt=user_prompt,
        outputs=outputs, task_summary=task_summary,
    )


def _cur(title="", description="", mode="auto", id="t1", next_step_hint=None):
    """当前 task（build_act_guidance 第一参）。"""
    return SimpleNamespace(
        id=id, title=title, description=description,
        interaction_mode=mode, user_prompt="", next_step_hint=next_step_hint,
    )


def _tm(tasks=None):
    """task_manager 替身；None = 无 task_manager。"""
    if tasks is None:
        return None
    return SimpleNamespace(all_tasks=lambda: tasks)


def test_no_task_manager_omits_tree_but_keeps_finish():
    g = build_act_guidance(_cur(title="T"), _tm())
    assert "## The overall plan" not in g
    assert "control__finish_task" in g
    assert "final reply to the user" in g


# ── 上一轮复核的一次性转向（next_step_hint）────────────────────────────────────

def test_next_step_hint_rendered_when_present():
    """observer 的一次性转向经 guidance 投递（不入 memory），故随任务终结自然消失。"""
    g = build_act_guidance(_cur(title="T", next_step_hint="先拆 TokenStore"), _tm())
    assert "先拆 TokenStore" in g
    assert "## Note from the review of your previous attempt" in g


def test_no_hint_omits_the_section_entirely():
    g = build_act_guidance(_cur(title="T"), _tm())
    assert "## Note from the review of your previous attempt" not in g


def test_hint_section_absent_for_task_without_the_field():
    """旧数据 / 替身 task 无该字段 → getattr 兜底，不炸、不出段。"""
    legacy = SimpleNamespace(id="t1", title="T", description="",
                             interaction_mode="auto", user_prompt="")
    g = build_act_guidance(legacy, _tm())
    assert "## Note from the review of your previous attempt" not in g


def test_lone_root_still_shows_itself_in_tree():
    # root 独自 act（唯一非终态 task）：仍出树、列自己一行，但表头不提「别做其他任务」。
    cur = _task("t1", "Current", "ACTIVE")
    g = build_act_guidance(_cur(title="Current", id="t1"), _tm(tasks=[cur]))
    assert "## The overall plan (▶ = your current task):" in g
    assert "- [ACTIVE] ▶ 'Current' (t1)" in g
    assert "do NOT do them yourself" not in g
    assert "Do not start the other tasks yourself." not in g
    assert "control__finish_task" in g


def test_no_open_tasks_omits_tree():
    # 全部终态（含当前）时无非终态节点 → 不出树，但完成方式仍在。
    cur = _task("t1", "Current", "FINISHED")
    g = build_act_guidance(_cur(title="Current", id="t1"), _tm(tasks=[cur]))
    assert "## The overall plan" not in g
    assert "control__finish_task" in g


def test_anchor_line_always_present_description_not_duplicated():
    # 每个 act 回合都有当前任务锚定行（长对话里 ## Current Task 框远在历史深处）；
    # description 不重复（由 Current Task 框承载）。
    g = build_act_guidance(_cur(title="T", description="D"), _tm())
    assert "Current task: 'T' (t1)" in g.split("\n")
    assert "Description: D" not in g


def test_anchor_line_uses_bare_id_when_untitled():
    """无标题时锚定行退化为裸 id——id 本身就是可用的句柄，比「见上文」有信息量。"""
    g = build_act_guidance(_cur(title=""), _tm())
    assert "Current task: t1" in g.split("\n")


def test_ask_user_reminder_present_in_all_modes():
    for mode in ("auto", "interactive"):
        g = build_act_guidance(_cur(title="T", mode=mode), _tm())
        assert "control__ask_user" in g


def test_switch_rule_mentions_both_tools_in_one_response():
    g = build_act_guidance(_cur(title="T"), _tm())
    assert "control__delegate_task" in g
    assert "ONE response" in g


def test_static_reminders_are_pointer_level_not_full_protocol():
    # A 阶段契约：静态段不再复读完整协议（deliverables_summary 详解等留在工具 description）。
    g = build_act_guidance(_cur(title="T"), _tm())
    assert "deliverables_summary" not in g


def test_tree_lists_nonterminal_with_status_and_current_marker():
    cur = _task("t1", "Write report", "ACTIVE")
    sib = _task("t2", "Publish", "PENDING")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[cur, sib]))
    assert "## The overall plan" in g
    assert "- [ACTIVE] ▶ 'Write report' (t1)" in g
    assert "- [PENDING] 'Publish' (t2)" in g
    assert "do NOT do them yourself" in g
    assert "Do not start the other tasks yourself." in g


def test_terminal_tasks_excluded_from_tree():
    cur = _task("t1", "CURTASK", "ACTIVE")
    tasks = [
        cur,
        _task("t2", "DONETASK", "FINISHED"),
        _task("t3", "FAILTASK", "FAILED"),
        _task("t4", "CANCTASK", "CANCELED"),
        _task("t5", "PENDTASK", "PENDING"),
    ]
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=tasks))
    assert "PENDTASK" in g
    assert "CURTASK" in g
    assert "DONETASK" not in g
    assert "FAILTASK" not in g
    assert "CANCTASK" not in g


def test_child_task_indented_under_parent():
    parent = _task("t1", "Parent", "SUSPENDED")
    child = _task("t2", "Child", "PENDING", parent="t1")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, child]))
    assert "- [SUSPENDED] ▶ 'Parent' (t1)" in g
    assert "  - [PENDING] 'Child' (t2)" in g


def test_orphan_nonterminal_promoted_to_root():
    # parent 不在非终态集合里（终态/缺失）→ 提升到 root 层，不缩进、不丢失。
    cur = _task("t1", "Current", "ACTIVE")
    orphan = _task("t2", "Orphan", "PENDING", parent="tX")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[cur, orphan]))
    assert "- [PENDING] 'Orphan' (t2)" in g
    assert "  - [PENDING] 'Orphan' (t2)" not in g


def test_untitled_task_uses_start_prompt_as_label():
    # 无 title 的 task（如 root）用开启它的 prompt 首行作标签。
    cur = _task("t1", "Current", "ACTIVE")
    root = _task("tsk_01", "", "ACTIVE", user_prompt="Summarize the quarterly report\nand email it")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[cur, root]))
    assert "- [ACTIVE] 'Summarize the quarterly report'" in g
    assert "(untitled" not in g


def test_untitled_task_long_prompt_truncated():
    cur = _task("t1", "Current", "ACTIVE")
    long_prompt = "x" * 200
    root = _task("tsk_01", "", "ACTIVE", user_prompt=long_prompt)
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[cur, root]))
    assert ("x" * 80 + "…") in g
    assert ("x" * 81) not in g


def test_untitled_task_no_prompt_falls_back_to_id_prefix():
    cur = _task("t1", "Current", "ACTIVE")
    blank = _task("abcdef123456", "", "PENDING")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[cur, blank]))
    # 无标题且无 prompt → 裸 id（此前只印前 6 位，取不回来）
    assert "- [PENDING] abcdef123456" in g


def test_finished_children_listed_with_no_redo_emphasis():
    # 挂起恢复场景：完成的子任务不在树里，但要在「已完成」清单里点名 + 强调勿重做。
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Research", "FINISHED", parent="t1")
    c2 = _task("t3", "Draft", "FINISHED", parent="t1")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1, c2]))
    assert "ALREADY COMPLETED" in g
    assert "Do NOT redo their work" in g
    assert "- [FINISHED] 'Research' (t2)" in g
    assert "- [FINISHED] 'Draft' (t3)" in g
    # 树里仍只有非终态节点
    assert "▶ 'Parent' (t1)" in g
    assert "  - [FINISHED]" not in g.split("ALREADY COMPLETED")[0]


def test_finished_child_renders_outputs_result_snippet():
    # 完成子任务的 outputs（成果物）作为结果摘要拼在 title 下的缩进 → 行。
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Research", "FINISHED", parent="t1", outputs="Found 3 key sources on X.")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert "- [FINISHED] 'Research'" in g
    assert "    → Found 3 key sources on X." in g


def test_finished_child_falls_back_to_task_summary_when_no_outputs():
    # outputs 空 → 回退 task_summary。
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Draft", "FINISHED", parent="t1",
               outputs=None, task_summary="Drafted intro and outline.")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert "    → Drafted intro and outline." in g


def test_finished_child_outputs_preferred_over_task_summary():
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Draft", "FINISHED", parent="t1",
               outputs="THE DELIVERABLE", task_summary="the process digest")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert "    → THE DELIVERABLE" in g
    assert "the process digest" not in g


def test_finished_child_no_result_renders_title_only():
    # 两者皆空 → 仅 title 行，无 → 行（保持向后兼容）。
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Research", "FINISHED", parent="t1")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert "- [FINISHED] 'Research'" in g
    assert "→" not in g.split("ALREADY COMPLETED")[1]


def test_finished_child_result_truncated():
    parent = _task("t1", "Parent", "ACTIVE")
    long_out = "y" * 300
    c1 = _task("t2", "Research", "FINISHED", parent="t1", outputs=long_out)
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert ("y" * 150 + "…") in g
    assert ("y" * 151) not in g


def test_finished_child_result_multiline_collapsed_to_single_line():
    # 多行结果折叠成单行，避免撑开每回合尾部的 guidance。
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Research", "FINISHED", parent="t1",
               outputs="line one\n\nline two\n  line three")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert "    → line one line two line three" in g


def test_finished_child_outputs_as_content_part_list():
    # outputs 是 ContentPart 列表 → content_to_text 归一。
    parent = _task("t1", "Parent", "ACTIVE")
    c1 = _task("t2", "Research", "FINISHED", parent="t1",
               outputs=[SimpleNamespace(text="part A "), SimpleNamespace(text="part B")])
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, c1]))
    assert "    → part A part B" in g


def test_mixed_naive_aware_created_at_does_not_crash_sort():
    # 崩溃恢复回归：restore 重建的 task 带 naive created_at（事件重放 / DB 丢 tz），与
    # 活任务的 aware created_at 混入同一注册表；排序须归一 tz，不得抛 naive/aware 比较错。
    from datetime import datetime, timezone
    naive = datetime(2026, 7, 13, 10, 0, 0)                    # 恢复重建：无 tz
    aware = datetime(2026, 7, 13, 11, 0, 0, tzinfo=timezone.utc)  # 活任务：aware
    parent = _task("t1", "Parent", "ACTIVE")
    c_recovered = _task("t2", "Recovered", "FINISHED", parent="t1", created_at=naive)
    c_live = _task("t3", "Live", "FINISHED", parent="t1", created_at=aware)
    # 两序都跑一遍，确保比较两侧混排都归一（naive<aware 与 aware<naive 各触发一次）。
    for tasks in ([parent, c_recovered, c_live], [parent, c_live, c_recovered]):
        g = build_act_guidance(_cur(id="t1"), _tm(tasks=tasks))
        assert "- [FINISHED] 'Recovered' (t2)" in g
        assert "- [FINISHED] 'Live' (t3)" in g


def test_finished_children_of_other_tasks_not_listed():
    # 别的 task 的完成子任务、以及无 parent 的完成 task，都不进当前 task 的清单。
    cur = _task("t1", "Current", "ACTIVE")
    other_child = _task("t2", "OtherChild", "FINISHED", parent="tX")
    orphan_done = _task("t3", "OrphanDone", "FINISHED")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[cur, other_child, orphan_done]))
    assert "ALREADY COMPLETED" not in g
    assert "OtherChild" not in g
    assert "OrphanDone" not in g


def test_failed_canceled_children_not_listed_as_completed():
    # FAILED/CANCELED 子任务可能需要重派，不得标成「已完成勿重做」。
    parent = _task("t1", "Parent", "ACTIVE")
    failed = _task("t2", "FailedChild", "FAILED", parent="t1")
    canceled = _task("t3", "CanceledChild", "CANCELED", parent="t1")
    g = build_act_guidance(_cur(id="t1"), _tm(tasks=[parent, failed, canceled]))
    assert "ALREADY COMPLETED" not in g
    assert "FailedChild" not in g
    assert "CanceledChild" not in g


def test_interactive_mode_keeps_pause_note_with_tree():
    cur = _task("t1", "Cur", "ACTIVE")
    sib = _task("t2", "Other", "PENDING")
    g = build_act_guidance(_cur(mode="interactive", id="t1"), _tm(tasks=[cur, sib]))
    assert "## The overall plan" in g
    assert "keeps the task open" in g


def test_finish_reminder_leads_with_completion_gate():
    # 判据在前：仅目标完全达成才 finish，措辞不得以收尾为默认动作。
    for mode in ("interactive", "autonomous"):
        g = build_act_guidance(_cur(mode=mode), _tm())
        assert "ONLY once the task goal is fully achieved" in g
        assert "keep working instead of finishing" in g


# ── 装配管线：GuidanceSource + composer 落位 ──────────────────────────────────


def _collect(source, request):
    async def _run():
        return [b async for b in source.fetch(request, deps=None)]
    return asyncio.run(_run())


def test_guidance_source_yields_block_from_extra():
    req = SimpleNamespace(extra={"act_guidance": "GUIDE TEXT"}, token_counter=estimate_tokens)
    blocks = _collect(GuidanceSource(), req)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.kind == "guidance" and b.target == "messages"
    assert b.content == "GUIDE TEXT"
    assert b.priority == 1


def test_guidance_source_silent_without_extra():
    assert _collect(GuidanceSource(), SimpleNamespace(extra={})) == []
    assert _collect(GuidanceSource(), SimpleNamespace(extra={"act_guidance": ""})) == []


def _guidance_block(text: str) -> ContextBlock:
    return ContextBlock(id="g", source="guidance", kind="guidance", target="messages",
                        content=text, priority=1, token_estimate=1, metadata={})


def _cap_block() -> ContextBlock:
    return ContextBlock(id="cap", source="capability", kind="capabilities", target="system",
                        content="search the web", priority=1, token_estimate=1,
                        metadata={"capability_name": "web_search", "capability_kind": "tool"})


def _act_request() -> SimpleNamespace:
    task = SimpleNamespace(id="t1", user_prompt_in_memory=False, process_report=None,
                           title="T", description="D", user_prompt="hello world")
    return SimpleNamespace(task=task, purpose="act", extra={})


def test_composer_places_capabilities_before_guidance_at_tail():
    # fresh 单消息回合：清单全文随当前 task 回合（cache 前缀内），guidance 殿后
    # 紧贴生成点；清单已在末条消息里，不补指针。
    msgs = DefaultComposer()._build_actor_messages(
        [_cap_block(), _guidance_block("GUIDE TEXT")], _act_request())
    last = msgs[-1].content
    assert "GUIDE TEXT" in last and "## Capabilities" in last
    assert last.index("## Capabilities") < last.index("GUIDE TEXT")
    assert last.rstrip().endswith("GUIDE TEXT")
    assert "Capabilities section of the current task message above" not in last


def test_resume_cue_anchors_task_and_remaining_work():
    from ctx_weft.core.loop.steps.act_guidance import build_resume_cue
    cue = build_resume_cue(_cur(title="Fix importer", id="t1"), _tm())
    assert "the task: 'Fix importer' (t1)" in cue
    assert "do not redo" in cue and "remaining work" in cue
    # 无已完成子任务 → 不指向 guidance 清单
    assert "situational notes" not in cue


def test_resume_cue_points_to_completed_list_only_when_children_finished():
    from ctx_weft.core.loop.steps.act_guidance import build_resume_cue
    parent = _task("t1", "Parent", "ACTIVE")
    child = _task("t2", "Research", "FINISHED", parent="t1")
    cue = build_resume_cue(_cur(title="Parent", id="t1"), _tm(tasks=[parent, child]))
    assert "situational notes" in cue and "re-delegating" in cue
    # 无 title → 退化为裸 id（仍是可用句柄，不再是「见上文」）
    cue2 = build_resume_cue(_cur(title="", id="t1"), _tm())
    assert "the task: t1" in cue2


def test_composer_uses_extra_resume_cue_as_turn_opener():
    # 历史以 assistant 收尾 → 垫续跑回合，内容取 extra["act_resume_cue"]；
    # guidance + Capabilities 指针随后拼进同一回合（清单全文在当前 task 回合）。
    hist_user = ContextBlock(id="h1", source="x", kind="history", target="messages",
                             content="hi", priority=3, token_estimate=1,
                             metadata={"role": "user", "timestamp": "1", "type": "user_prompt"})
    hist_asst = ContextBlock(id="h2", source="x", kind="history", target="messages",
                             content="working...", priority=3, token_estimate=1,
                             metadata={"role": "assistant", "timestamp": "2"})
    task = SimpleNamespace(id="t1", user_prompt_in_memory=True, process_report=None,
                           title="T", description="D", user_prompt="hi")
    req = SimpleNamespace(task=task, purpose="act", extra={"act_resume_cue": "RESUME CUE"})
    msgs = DefaultComposer()._build_actor_messages(
        [hist_user, hist_asst, _cap_block(), _guidance_block("GUIDE TEXT")], req)
    first = msgs[0].content
    last = msgs[-1].content
    assert msgs[-1].role == "user"
    assert "## Capabilities" in first  # 清单全文随当前 task 回合
    assert last.startswith("RESUME CUE")
    assert "## Capabilities" not in last
    assert last.index("RESUME CUE") < last.index("GUIDE TEXT") \
        < last.index("Capabilities section of the current task message above")


def test_composer_ignores_guidance_for_non_act_purpose():
    req = _act_request()
    req.purpose = "observe"
    msgs = DefaultComposer()._build_actor_messages(
        [_cap_block(), _guidance_block("GUIDE TEXT")], req)
    assert all("GUIDE TEXT" not in (m.content if isinstance(m.content, str) else "")
               for m in msgs)
