"""Resources/directive relocated from actor system prompt → first user message."""

from __future__ import annotations

from types import SimpleNamespace

from ctx_weft.core.assembler.assembler import ContextBlock
from ctx_weft.core.assembler.composer import DefaultComposer


def _cap_block(name: str, kind: str, desc: str) -> ContextBlock:
    return ContextBlock(
        id=f"cap-{name}",
        source="capability",
        kind="capabilities",
        target="system",
        content=desc,
        priority=1,
        token_estimate=1,
        metadata={"capability_name": name, "capability_kind": kind},
    )


def _tool_block_kind(
    qname: str, desc: str, provider_name: str, provider_description: str, kind: str,
) -> ContextBlock:
    return ContextBlock(
        id=f"cap-{qname}",
        source="capability",
        kind="capabilities",
        target="system",
        content=desc,
        priority=1,
        token_estimate=1,
        metadata={
            "capability_name": qname,
            "capability_kind": kind,
            "provider_name": provider_name,
            "provider_description": provider_description,
        },
    )


def _tool_block(qname: str, desc: str, provider_name: str, provider_description: str) -> ContextBlock:
    return _tool_block_kind(qname, desc, provider_name, provider_description, "tool")


def _tool_block_schema(qname: str, schema: dict) -> ContextBlock:
    blk = _tool_block_kind(qname, "描述不该进正文", "p", "", "tool")
    blk.metadata["input_schema"] = schema
    return blk


def _identity_block(text: str) -> ContextBlock:
    return ContextBlock(id="id", source="identity", kind="identity", target="system",
                        content=text, priority=0, token_estimate=1, metadata={})


def _background_block(text: str) -> ContextBlock:
    return ContextBlock(id="bg", source="memory", kind="background", target="system",
                        content=text, priority=1, token_estimate=1, metadata={})


def _directive_block(text: str) -> ContextBlock:
    return ContextBlock(id="dir", source="identity:skill", kind="directive", target="system",
                        content=text, priority=1, token_estimate=1,
                        metadata={"kind": "skill_instructions"})


def _fresh_task_request() -> SimpleNamespace:
    task = SimpleNamespace(user_prompt_in_memory=False, process_report=None,
                           title="T", description="D", user_prompt="hello world")
    return SimpleNamespace(task=task, purpose="act")


def test_actor_system_has_only_soul_and_background() -> None:
    blocks = [
        _identity_block("SOUL TEXT"),
        _background_block("BG TEXT"),
        _cap_block("web_search", "tool", "search the web"),
        _cap_block("researcher", "agent", "a research subagent"),
        _directive_block("Instructions for skill 'x':\nDo the thing"),
    ]
    system = DefaultComposer()._build_actor_system(blocks)
    assert "SOUL TEXT" in system
    assert "## Project Background" in system and "BG TEXT" in system
    assert "### Available Tools" not in system
    assert "### Available Sub-Agents" not in system
    assert "## Instructions for the current task" not in system
    assert "Do the thing" not in system


def test_act_directive_and_capabilities_on_current_task_turn() -> None:
    blocks = [
        _identity_block("SOUL TEXT"),
        _background_block("BG TEXT"),
        _cap_block("web_search", "tool", "search the web"),
        _cap_block("researcher", "agent", "a research subagent"),
        _directive_block("Instructions for skill 'x':\nDo the thing"),
        _history_block("user", "## Current Message\nthe original ask", "1"),
        _history_block("assistant", "did some work", "2"),
    ]
    task = SimpleNamespace(user_prompt_in_memory=True, process_report="halfway done",
                           title="T", description="D", user_prompt="the original ask")
    msgs = DefaultComposer()._build_actor_messages(blocks, SimpleNamespace(task=task, purpose="act"))
    user_msgs = [m for m in msgs if m.role == "user"]
    first, last = user_msgs[0].content, user_msgs[-1].content
    # directive AND capabilities both ride the current task's user message
    # (stable in the reconstructed history → inside the prompt-cache prefix)
    assert "## Instructions for the current task" in first
    assert "Do the thing" in first
    assert "the original ask" in first
    assert "## Capabilities" in first
    assert "### Available Tools" in first
    assert "### Available Sub-Agents" in first
    # order: task content → directive → capabilities
    assert first.index("the original ask") < first.index("## Instructions for the current task") \
        < first.index("## Capabilities")
    # the trailing user message carries only a one-line pointer, not the full listing
    assert "### Available Tools" not in last
    assert "Capabilities section of the current task message above" in last


def test_act_directive_follows_current_task() -> None:
    blocks = [
        _identity_block("SOUL TEXT"),
        _background_block("BG TEXT"),
        _directive_block("Do the thing"),
    ]
    msgs = DefaultComposer()._build_actor_messages(blocks, _fresh_task_request())
    content = [m for m in msgs if m.role == "user"][0].content
    assert "## Current Task" in content
    assert "## Instructions for the current task" in content
    # directive sits after the task context, not before it
    assert content.index("## Current Task") < content.index("## Instructions for the current task")


def test_act_capabilities_intro_heading_on_last() -> None:
    blocks = [
        _identity_block("SOUL TEXT"),
        _background_block("BG TEXT"),
        _cap_block("web_search", "tool", "search the web"),
    ]
    msgs = DefaultComposer()._build_actor_messages(blocks, _fresh_task_request())
    last = [m for m in msgs if m.role == "user"][-1].content
    assert "## Capabilities" in last
    assert "complete the current task" in last
    # intro heading precedes the specific capability lists
    assert last.index("## Capabilities") < last.index("### Available Tools")


def _skill_directive_block(text: str, skill_name: str) -> ContextBlock:
    return ContextBlock(id="dir", source="identity:skill", kind="directive", target="system",
                        content=text, priority=1, token_estimate=1,
                        metadata={"kind": "skill_instructions", "skill_name": skill_name})


def test_directive_merges_skill_name_into_heading_and_shifts_levels() -> None:
    md = "# Overview\nsurvey the network\n\n## Steps\n1. collect\n\n### Detail\nx"
    blocks = [_skill_directive_block(md, "sdn-migration-survey")]
    section = DefaultComposer()._build_directive_section(blocks)
    # skill name folded into the H2 heading; no separate "Instructions for skill" line
    assert section.startswith("## Instructions for the current task (skill: sdn-migration-survey)")
    assert "Instructions for skill 'sdn-migration-survey'" not in section
    # skill md headings shifted so its top level (H1) nests under the H2 container (→ H3)
    lines = section.split("\n")
    assert "### Overview" in lines
    assert "#### Steps" in lines
    assert "##### Detail" in lines
    assert "# Overview" not in lines  # original H1 line is gone (shifted)


def test_directive_without_headings_unchanged_body() -> None:
    blocks = [_skill_directive_block("just do the thing", "my-skill")]
    section = DefaultComposer()._build_directive_section(blocks)
    assert section == "## Instructions for the current task (skill: my-skill)\n\njust do the thing"


def test_act_skills_on_last_when_no_directive() -> None:
    blocks = [
        _identity_block("SOUL TEXT"),
        _background_block("BG TEXT"),
        _cap_block("code_runner", "skill", "run code"),
    ]
    msgs = DefaultComposer()._build_actor_messages(blocks, _fresh_task_request())
    last = [m for m in msgs if m.role == "user"][-1].content
    assert "### Available Skills" in last
    assert "code_runner" in last


def test_actor_no_resources_is_noop() -> None:
    blocks = [_identity_block("SOUL TEXT"), _background_block("BG TEXT")]
    msgs = DefaultComposer()._build_actor_messages(blocks, _fresh_task_request())
    first = [m for m in msgs if m.role == "user"][0].content
    assert "---\n\n## Current Task" not in first  # no preamble separator injected
    assert "## Current Task" in first


def test_tools_grouped_by_provider_with_description_and_prefix() -> None:
    blocks = [
        _tool_block("mcp__github__create_issue", "Create an issue", "mcp:github",
                    "Manage GitHub issues & PRs."),
        _tool_block("mcp__github__add_comment", "Comment on an issue", "mcp:github",
                    "Manage GitHub issues & PRs."),
        _tool_block("fs__bash_exec", "Run a shell command", "fs", ""),
    ]
    section = DefaultComposer()._build_resources_section(blocks)
    # provider 分块标题
    assert "#### mcp:github tools" in section
    assert "#### fs tools" in section
    # 有 description 的 provider：写出描述
    assert "Manage GitHub issues & PRs." in section
    # 每个 provider 段都带具体前缀提示（无论是否有 description）
    assert "prefix the tool name with `mcp__github__`" in section
    assert "prefix the tool name with `fs__`" in section
    # 工具仍按 qualified 名列出，但只留名字 + 入参签名（描述随 tools 参数下发）
    assert "- mcp__github__create_issue()" in section
    assert "- fs__bash_exec()" in section
    assert "Create an issue" not in section
    # provider 描述出现在该 provider 标题之后、其工具之前
    assert section.index("#### mcp:github tools") < section.index("Manage GitHub issues") \
        < section.index("- mcp__github__create_issue")


def test_tool_signature_marks_optional_params() -> None:
    """入参签名：必填直出，可选带 ?，类型用短名。"""
    blk = _tool_block_schema("p__create_issue", {
        "type": "object",
        "properties": {"repo": {"type": "string"}, "labels": {"type": "array"},
                       "draft": {"type": "boolean"}},
        "required": ["repo"],
    })
    section = DefaultComposer()._build_resources_section([blk])
    assert "- p__create_issue(repo: str, labels?: list, draft?: bool)" in section
    assert "描述不该进正文" not in section


def test_tool_signature_empty_when_no_properties() -> None:
    """无入参 / schema 缺失 → 空括号，不留噪声。"""
    for schema in ({"type": "object"}, {}, None):
        blk = _tool_block_schema("p__ping", schema)
        assert "- p__ping()" in DefaultComposer()._build_resources_section([blk])


def test_tool_signature_truncates_long_param_lists() -> None:
    """入参过多时截断成省略号，避免一行撑爆索引。"""
    props = {f"p{i}": {"type": "string"} for i in range(12)}
    blk = _tool_block_schema("p__wide", {"properties": props, "required": list(props)})
    section = DefaultComposer()._build_resources_section([blk])
    assert "p7: str, …)" in section
    assert "p8" not in section


def test_tool_signature_union_and_unknown_types() -> None:
    """联合类型取首个；未知/缺失 type 记 any。"""
    blk = _tool_block_schema("p__odd", {
        "properties": {"a": {"type": ["string", "null"]}, "b": {}, "c": {"type": "integer"}},
        "required": ["a", "b", "c"],
    })
    section = DefaultComposer()._build_resources_section([blk])
    assert "- p__odd(a: str, b: any, c: int)" in section


def test_tools_section_carries_index_preamble() -> None:
    """段首说明去哪找完整描述，免得模型以为索引就是全部。"""
    section = DefaultComposer()._build_resources_section(
        [_tool_block_schema("p__x", {})])
    assert "optional params are marked `?`" in section
    assert "tool definitions in this request" in section


def test_skills_and_agents_keep_descriptions() -> None:
    """skill / sub-agent 没有 tools 参数那条通路，描述仍须留在正文。"""
    blocks = [
        _tool_block_kind("local_skill__pdf", "Work with PDFs", "local_skill", "", "skill"),
        _tool_block_kind("template_agent__planner", "Plans work", "template_agent", "", "agent"),
    ]
    section = DefaultComposer()._build_resources_section(blocks)
    assert "- local_skill__pdf: Work with PDFs" in section
    assert "- template_agent__planner: Plans work" in section


def test_skills_grouped_by_provider_with_description_and_prefix() -> None:
    blocks = [
        _tool_block_kind("local_skill__pdf", "Work with PDFs", "local_skill",
                         "Filesystem-backed skills.", "skill"),
        _tool_block_kind("local_skill__csv", "Work with CSVs", "local_skill",
                         "Filesystem-backed skills.", "skill"),
    ]
    section = DefaultComposer()._build_resources_section(blocks)
    assert "### Available Skills (assign to tasks where appropriate)" in section
    assert "#### local_skill skills" in section
    assert "Filesystem-backed skills." in section
    assert "When assigning these skills, prefix the skill name with `local_skill__`" in section
    assert "- local_skill__pdf: Work with PDFs" in section


def test_bound_skill_dropped_from_list_others_kept() -> None:
    """当前 task 绑了某个 skill：清单里去掉它（正文已在 ## Instructions for the current task），
    其余 skill 保留——它们是派发给子任务的候选。"""
    blocks = [
        _tool_block_kind("local_skill__pdf", "Work with PDFs", "local_skill", "", "skill"),
        _tool_block_kind("local_skill__csv", "Work with CSVs", "local_skill", "", "skill"),
    ]
    section = DefaultComposer()._build_resources_section(
        blocks, current_skill_name="local_skill__pdf")
    assert "local_skill__pdf" not in section
    assert "- local_skill__csv: Work with CSVs" in section
    assert "### Available Skills (assign to tasks where appropriate)" in section


def test_bound_skill_is_only_skill_drops_whole_section() -> None:
    """绑定的 skill 是唯一一个 → 整段 Available Skills 消失（不留空标题）。"""
    blocks = [_tool_block_kind("local_skill__pdf", "Work with PDFs", "local_skill", "", "skill")]
    section = DefaultComposer()._build_resources_section(
        blocks, current_skill_name="local_skill__pdf")
    assert "Available Skills" not in section


def test_directive_presence_alone_no_longer_hides_skills() -> None:
    """判据是 skill 名，不是「有没有 directive 块」：没传 skill 名时清单原样渲染。"""
    blocks = [
        _tool_block_kind("local_skill__pdf", "Work with PDFs", "local_skill", "", "skill"),
        _directive_block("skill 正文"),
    ]
    section = DefaultComposer()._build_resources_section(blocks)
    assert "- local_skill__pdf: Work with PDFs" in section


def test_same_bare_name_across_providers_only_bound_one_dropped() -> None:
    """同名不同 provider：只剔掉绑定的那个 qualified 名，另一个 provider 的同名 skill 保留。"""
    blocks = [
        _tool_block_kind("local_skill__pdf", "Local PDFs", "local_skill", "", "skill"),
        _tool_block_kind("mcp__docs__pdf", "Remote PDFs", "mcp:docs", "", "skill"),
    ]
    section = DefaultComposer()._build_resources_section(
        blocks, current_skill_name="local_skill__pdf")
    assert "- mcp__docs__pdf: Remote PDFs" in section
    assert "local_skill__pdf" not in section


def test_bare_skill_name_matches_nothing() -> None:
    """裸名不作判据：它在 get_by_qualified_name 下查不到 skill 定义、正文没进 prompt，
    此时再把条目从清单里藏掉是双输（跨 provider 还会误伤同名 skill）。"""
    blocks = [_tool_block_kind("local_skill__pdf", "Work with PDFs", "local_skill", "", "skill")]
    section = DefaultComposer()._build_resources_section(blocks, current_skill_name="pdf")
    assert "- local_skill__pdf: Work with PDFs" in section


def test_skill_name_matched_via_capability_id_when_name_absent() -> None:
    """block 没带 capability_name 时按 qualify(capability_id) 折算，provider 段不丢。"""
    blk = _tool_block_kind("", "Work with PDFs", "local_skill", "", "skill")
    blk.metadata.pop("capability_name")
    blk.metadata["capability_id"] = "local_skill:pdf"
    section = DefaultComposer()._build_resources_section(
        [blk], current_skill_name="local_skill__pdf")
    assert "Available Skills" not in section


def test_subagents_grouped_by_provider_keep_delegate_preamble() -> None:
    blocks = [
        _tool_block_kind("template_agent__planner", "Plans work", "template_agent",
                         "Built-in agent templates.", "agent"),
    ]
    section = DefaultComposer()._build_resources_section(blocks)
    assert "### Available Sub-Agents" in section
    assert "Delegate via:" in section  # preamble preserved
    assert "#### template_agent sub-agents" in section
    assert "Built-in agent templates." in section
    assert "When delegating these sub-agents, prefix the sub-agent name with `template_agent__`" in section
    assert "- template_agent__planner: Plans work" in section


def test_tools_without_provider_name_render_flat() -> None:
    blocks = [_cap_block("web_search", "tool", "search the web")]
    section = DefaultComposer()._build_resources_section(blocks)
    assert "### Available Tools" in section
    assert "- web_search()" in section
    assert "search the web" not in section  # 描述不入正文
    assert "####" not in section  # 无 provider 分块标题


def test_observer_system_excludes_resources_and_directive() -> None:
    template = SimpleNamespace(identity={"act": SimpleNamespace(text="ACT SOUL")})
    request = SimpleNamespace(template=template)
    blocks = [
        _background_block("BG TEXT"),
        _cap_block("report_task_outcome", "tool", "report the outcome"),
        _directive_block("Instructions for skill 'x':\nDo the thing"),
    ]
    system = DefaultComposer()._build_act_system(blocks, request)
    assert "ACT SOUL" in system
    assert "## Project Background" in system and "BG TEXT" in system
    assert "### Available Tools" not in system
    assert "## Instructions for the current task" not in system
    assert "Do the thing" not in system


def test_observer_messages_inject_resources_and_keep_role() -> None:
    blocks = [
        _identity_block("OBSERVER ROLE"),  # purpose=observe → identity block is the ROLE
        _cap_block("report_task_outcome", "tool", "report the outcome"),
    ]
    task = SimpleNamespace(title="T", description="d", user_prompt="up",
                           user_prompt_in_memory=False, process_report=None)
    request = SimpleNamespace(task=task)
    msgs = DefaultComposer()._build_observer_messages(blocks, request)
    user_msgs = [m for m in msgs if m.role == "user"]
    assert "### Available Tools" in user_msgs[0].content
    # 条目只有名字 + 入参签名，描述随 tools 参数下发、不重复进正文
    assert "- report_task_outcome()" in user_msgs[0].content
    assert "report the outcome" not in user_msgs[0].content
    joined = "\n".join(m.content for m in msgs if isinstance(m.content, str))
    assert "OBSERVER ROLE" in joined


def _history_block(role: str, content: str, ts: str) -> ContextBlock:
    # mem_type="user_prompt" for user blocks so _frame_current_message and
    # current_task_user_idx detection work correctly (they now use mtype, not source).
    mem_type = "user_prompt" if role == "user" else "llm_response"
    return ContextBlock(id=f"h-{ts}", source="task_conversation", kind="history",
                        target="messages", content=content, priority=3, token_estimate=1,
                        metadata={"role": role, "timestamp": ts, "type": mem_type})


def _experience_block(role: str, content: str, ts: str) -> ContextBlock:
    return ContextBlock(id=f"e-{ts}", source="agent_experience", kind="history",
                        target="messages", content=content, priority=3, token_estimate=1,
                        metadata={"role": role, "timestamp": ts})


def _task_history_block(role: str, content: str, ts: str, task_id: str) -> ContextBlock:
    mem_type = "user_prompt" if role == "user" else "llm_response"
    return ContextBlock(id=f"h-{ts}", source="task_conversation", kind="history",
                        target="messages", content=content, priority=3, token_estimate=1,
                        metadata={"role": role, "timestamp": ts, "type": mem_type,
                                  "task_id": task_id})


def test_interactive_task_pins_task_frame_first_current_message_follows_latest() -> None:
    """Interactive task: the user's follow-up messages accumulate as USER_PROMPTs with the SAME
    task_id. The ## Current Task frame + directive + capabilities pin to the FIRST (task-opening)
    user message — drifting them re-decorates a different turn every round, reverting the previous
    turn's bytes and breaking the prompt-cache prefix. The ## Current Message frame (+ same-language
    reply hint) instead follows the LATEST user message of the task — "current message" semantically
    IS the newest one; pinning it first would mislabel a stale message as current."""
    blocks = [
        _identity_block("SOUL TEXT"),
        _cap_block("web_search", "tool", "search the web"),
        _directive_block("Do the thing"),
        _task_history_block("user", "你好", "1", "tsk_1"),
        _task_history_block("assistant", "你好！有什么可以帮你？", "2", "tsk_1"),
        _task_history_block("user", "你是谁", "3", "tsk_1"),
    ]
    task = SimpleNamespace(id="tsk_1", user_prompt_in_memory=True, process_report=None,
                           title="打招呼", description="回应问候", user_prompt="你好")
    msgs = DefaultComposer()._build_actor_messages(blocks, SimpleNamespace(task=task, purpose="act"))
    user_msgs = [m for m in msgs if m.role == "user"]
    first = user_msgs[0].content
    latest = user_msgs[-1].content
    # task frame + directive + capabilities pin to the task-opening message;
    # the raw opening text gets its own ## Opening Message heading
    assert "## Current Task" in first and "你好" in first
    assert "## Opening Message" in first
    assert first.index("## Current Task") < first.index("## Opening Message") \
        < first.index("## Instructions for the current task")
    assert "## Current Message" not in first
    assert "## Capabilities" in first
    # the latest (follow-up) message carries the current-message frame + language hint
    # + tail pointer only — none of the task-level bulk
    assert "## Current Message" in latest and "你是谁" in latest
    assert "Reply in the same language" in latest
    assert "## Current Task" not in latest
    assert "## Instructions for the current task" not in latest
    assert "## Capabilities" not in latest
    assert "Capabilities section of the current task message above" in latest


def test_act_directive_targets_current_task_not_prior_experience() -> None:
    """The current task's skill directive must ride the current task's first user message
    (task_conversation), not an earlier cross-task agent_experience turn that happens to be
    the first user message of the whole list."""
    blocks = [
        _identity_block("SOUL TEXT"),
        _directive_block("Do the thing"),
        # prior FINISHED task recalled via agent_recall as a real user_prompt (task-resident
        # capsule) — mtype="user_prompt" yet must NOT receive the current task's directive.
        _history_block("user", "prior task ask", "1"),
        _history_block("assistant", "prior task work", "2"),
        # the current task's own conversation (newer)
        _history_block("user", "## Current Message\nthe current ask", "3"),
    ]
    task = SimpleNamespace(user_prompt_in_memory=True, process_report=None,
                           title="T", description="D", user_prompt="the current ask")
    msgs = DefaultComposer()._build_actor_messages(blocks, SimpleNamespace(task=task, purpose="act"))
    user_msgs = [m for m in msgs if m.role == "user"]
    prior = next(m for m in user_msgs if "prior task ask" in m.content)
    current = next(m for m in user_msgs if "the current ask" in m.content)
    # directive rides the current task's first user message, not the prior experience turn
    assert "## Instructions for the current task" not in prior.content
    assert "## Instructions for the current task" in current.content
    assert "Do the thing" in current.content


def test_resumed_task_directive_and_capabilities_on_history_pointer_on_fallback() -> None:
    """Resumed act task: directive AND capabilities attach to the first (history-derived,
    current-task) user message; the trailing fallback continue message carries only the
    one-line capabilities pointer.

    Progress So Far no longer has a separate process_report-driven render path (retired 2026-07-01
    Task 3 — retry feedback is now carried by the TASK_COMPACT_SUMMARY segment summary instead), so
    a resumed task with only history blocks + a trailing non-user turn falls back to the generic
    resume user message (task title + review-completed/do-remaining wording) as the trailing
    dynamic-context slot (act purpose only)."""
    blocks = [
        _identity_block("SOUL TEXT"),
        _background_block("BG TEXT"),
        _cap_block("web_search", "tool", "search the web"),
        _directive_block("Instructions for skill 'x':\nDo the thing"),
        _history_block("user", "## Current Message\nthe original ask", "1"),
        _history_block("assistant", "did some work", "2"),
    ]
    task = SimpleNamespace(user_prompt_in_memory=True, process_report=None, process_report_at=None,
                           title="T", description="D", user_prompt="the original ask")
    msgs = DefaultComposer()._build_actor_messages(blocks, SimpleNamespace(task=task, purpose="act"))
    user_msgs = [m for m in msgs if m.role == "user"]
    first, last = user_msgs[0].content, user_msgs[-1].content
    # directive then capabilities on the first (history-derived) user message, AFTER the task content
    assert "## Instructions for the current task" in first
    assert "the original ask" in first
    assert "## Capabilities" in first and "### Available Tools" in first
    assert first.index("the original ask") < first.index("## Instructions for the current task") \
        < first.index("## Capabilities")
    # the trailing dynamic-context message: without extra["act_resume_cue"] the composer's
    # structural fallback line anchors the task and asks to continue with the remaining work
    # (canonical cue lives in act_guidance.py); the full listing stays off it — pointer only
    assert "You are still working on the task: 'T'" in last
    assert "continue with only the remaining work" in last
    assert "### Available Tools" not in last
    assert "Capabilities section of the current task message above" in last
    assert first is not last


def test_facet_purpose_gets_no_resume_filler_cue_rides_new_trailing_user() -> None:
    """Non-act purposes must NOT get the act resume filler. With a non-user-ending history,
    the facet cue (+capabilities pointer) rides a NEW trailing user message appended at the
    END — never glued onto an earlier mid-conversation user turn. The full capabilities
    listing rides the current-task user turn (cache prefix), same as act."""
    blocks = [
        _identity_block("OBSERVER ROLE"),
        _cap_block("report_task_outcome", "tool", "report the outcome"),
        _history_block("user", "## Current Message\nthe original ask", "1"),
        _history_block("assistant", "did some work", "2"),
        _history_block("tool", "big tool result", "3"),
    ]
    task = SimpleNamespace(title="T", description="d", user_prompt="the original ask",
                           user_prompt_in_memory=True, process_report=None, outputs=None)
    request = SimpleNamespace(task=task, purpose="observe")
    msgs = DefaultComposer()._build_observer_messages(blocks, request)
    # no act resume filler anywhere
    assert all("You are still working on" not in (m.content or "") for m in msgs)
    # the cue message is the LAST message of the list (nothing after it)
    assert msgs[-1].role == "user"
    assert "OBSERVER ROLE" in msgs[-1].content
    # trailing message carries the pointer, not the full listing
    assert "Capabilities section of the current task message above" in msgs[-1].content
    assert "### Available Tools" not in msgs[-1].content
    # the full listing rides the current-task user turn; the ROLE facet stays off it
    first_user = next(m for m in msgs if m.role == "user")
    assert "OBSERVER ROLE" not in first_user.content
    assert "## Capabilities" in first_user.content
    assert "### Available Tools" in first_user.content


def test_observer_capabilities_on_current_task_turn_not_prepended() -> None:
    """Observe (with conversation history) must place the Capabilities block on the current-task
    user turn — appended AFTER the task content, same as act — never prepended before it (which
    would bury the task under the full skills/tools/sub-agents listing). The trailing user
    message keeps only the one-line pointer alongside the ROLE + judgment cue."""
    blocks = [
        _identity_block("OBSERVER ROLE"),  # purpose=observe → identity block is the ROLE
        _cap_block("report_task_outcome", "tool", "report the outcome"),
        _cap_block("docx", "skill", "make a docx"),
        _cap_block("planner", "agent", "a planning subagent"),
        _history_block("user", "## Current Message\nthe original ask", "1"),
        _history_block("assistant", "did some work", "2"),
    ]
    task = SimpleNamespace(title="T", description="d", user_prompt="the original ask",
                           user_prompt_in_memory=True, process_report=None, outputs=None)
    request = SimpleNamespace(task=task)  # no purpose attr → non-act (observe) path
    msgs = DefaultComposer()._build_observer_messages(blocks, request)
    user_msgs = [m for m in msgs if m.role == "user"]
    assert len(user_msgs) >= 2
    first, last = user_msgs[0].content, user_msgs[-1].content
    # capabilities ride the current-task user turn, AFTER the task content (not front-loaded)
    assert "the original ask" in first
    assert "## Capabilities" in first
    assert "### Available Tools" in first
    assert "### Available Skills" in first
    assert "### Available Sub-Agents" in first
    assert first.index("the original ask") < first.index("## Capabilities")
    # the trailing user message: pointer only, alongside the observer ROLE + judgment cue
    assert "### Available Tools" not in last
    assert "Capabilities section of the current task message above" in last
    assert "OBSERVER ROLE" in last
