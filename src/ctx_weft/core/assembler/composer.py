"""Composer：把 ContextBlock 渲染成最终 AssembledPrompt。

═══ 槽位总览：所有 purpose 共用同一副「底盘」，差异全在尾部注入与工具面 ═══

system —— 恒为 act 面孔（--- 分隔）。purpose 专属人格不进 system，由尾部 user
message 就近覆盖（_build_actor_system / _build_act_system 同构）：

    ┌──────────────────────────────────────────────┐
    │ identity(act)          ← SOUL.md 正文         │
    │ ---                                           │
    │ ## Project Background  ← background block     │
    └──────────────────────────────────────────────┘

messages —— 组装步骤（_build_actor_messages）：

    ① history 多轮重建：agent_recall 归并所有未折叠 task body + agent 层
       finish/dispatch 对 + 折叠摘要，按 (timestamp, seq_no) 正序（见
       sources/_history.py）。当前 task 段摘要已冠 ## Progress So Far，
       user 身份摘要已套「压缩摘要」消歧前缀。
    ② 当前 task 的 user_prompt 回合就地装饰，分两处（interactive 多轮下不同回合）：
       - **首条**（按 task_id 定位，匹配不到回退末条）：## Current Task 框 +
         ## Opening Message 标注原文；directive（skill 指令）仅 act 拼在此回合尾部；
         ## Capabilities 全文（所有 purpose）拼在 directive 之后——随此稳定回合
         落在 cache 前缀内。
       - **最新一条**（同 task_id）：## Current Message 框 + 同语言回复提示——
         「当前消息」就是最新一条，钉首条会把旧消息冒充成当前消息。
         首条即最新（单条）时两框合并（无 Opening Message）。
    ③ 仅 act：末条以 assistant/tool 收尾时垫续跑衔接 user 回合
       （extra["act_resume_cue"]；缺失时结构兜底一句，保证以 user 收尾）。
    ④ 末条 user 尾部依次拼（_append_to_last_user；末条非 user 则新建，
       绝不回溯粘中部）：guidance（仅 act）→ Capabilities 指针（一行，
       清单不在末条时才补）→ facet 尾注（仅 facet purpose）。

═══ act 末条 user 的三种形态（开场三选一；「--- 以下」的收尾恒同）═══

  A. fresh / 当前消息回合 —— 末条就是当前 task 的 user_prompt 回合：
       ## Current Task {title}\n{description}
       ## Current Message {user_prompt}（+ 同语言回复提示）
       ## Instructions for the current task (skill: x)   ← 绑 skill 时
       ## Capabilities                                   ← 清单全文随此稳定回合
       ---（guidance 起，见下；清单已在末条，不再补指针）
  B. 续跑回合 —— 历史以 assistant/tool 收尾（挂起父任务恢复、段中崩溃
     recover、retry 摘要为空）：
       {resume cue：仍在做任务 X、盘点已完成、只做剩余
        + 确有 FINISHED 子任务时一句「清单见下方态势注记」}
       ---（guidance 起，见下）
  C. 追问回合 —— interactive 任务里用户新消息本身是末条 user：
       ## Current Message {用户消息原文}（+ 同语言回复提示）
       ---（guidance 起；此形态下 Current Task 框远在历史深处，
           就近任务锚由 guidance 锚定行承担）

  三种形态共用的收尾（guidance + capabilities 指针）：
       ---
       Current task: {title}                    ← 任务锚定行（恒有）
       ## The overall plan（▶ 定位当前 task）   ← ≥1 非终态 task 才出
       ## 已派发且 ALREADY COMPLETED 的子任务   ← 有 FINISHED 子任务才出
       finish / 无关新请求双发 / ask_user 三条指针级提醒
       Capabilities 指针（一行）                ← 清单全文在当前 task 回合
                                                  （cache 前缀内），B/C 形态补此行

  guidance 与 resume cue 同源 loop/steps/act_guidance.py（PrepareStep 构建，
  经 extra["act_guidance"] / extra["act_resume_cue"] 传入；guidance 走
  GuidanceSource 成块、参与预算与 token 记账）。

═══ facet purpose（observe/compact/recognize_intent/background_observe）═══

  无续跑 cue、无 guidance。清单全文同样随当前 task 回合；末条 user
  （历史末条是 user 就并入，否则新建）：
       {原内容（如有）}
       Capabilities 指针（一行；清单不在末条时才补）
       ## Your Current Role + facet 正文（ROLE/COMPACT/METADATA.md）
       ---
       cue（各 purpose 专属，见下表）

═══ 各 facet 的 cue 与工具面（act 的 tools = act-purpose 全量）═══

  observe            ROLE facet → [## Final output ← task.outputs（finish 为
                     SILENT 工具，对话里不可见，须显式回填）] → 裁决 cue
                     (_OBSERVE_JUDGMENT_CUE) → [## Your sub-tasks 可 review 清单
                     ← extra["subtask_reviews"]]。tools = report_task_outcome。
  background_observe ROLE facet → [## Actor 的最终产出（仅 close 边界）] →
                     边界 cue（interrupt / plain_text / finish / normal）。
                     tools = collect_process_report。
  compact            COMPACT facet → 压缩 cue，按 extra["compact_scope"] 二选一：
                     task 域=概括整个 task 至今；agent 域=只压派发历史。
                     tools = []（纯文本输出）。
  recognize_intent   METADATA facet → 元数据 cue（update_task_metadata 恰一次）。
                     tools = update_task_metadata。

上表的 tools 一栏是**结果**而非本模块的产物：composer 只渲染散文段，工具数组由
AssembledPrompt.tools 现读 CapabilityCache（按同一个 purpose 过滤，见
sources/capability.py::build_llm_tools）。compact 的 tools = [] 也由那条闭包保证
（assembler 对 compact 不装 tools_fn）。

历史沿革：格式源自 miniAgents prompt_builder.py（refactor 非 redesign）；
Phase 3 (2026-06-30) 移除 ## Task Background blackboard 段，predecessor 结果
改经 memory recall 浮现。裁剪保护阶梯见 priority.py；预算裁剪见 budget.py。
"""

from __future__ import annotations

import dataclasses
import logging
import re
from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ctx_weft.core.capabilities.control_tools import (
    DELEGATE_TASK_NAME,
    REPORT_TASK_OUTCOME_NAME,
)
from ctx_weft.core.utils.content import (
    content_to_text,
    content_with_prefix,
    content_with_suffix,
    downgrade_images_to_text,
    image_tokens,
)
from ctx_weft.core.utils.headings import SUBTASKS_REVIEW_HEADING
from ctx_weft.protocols import LLMMessage
from ctx_weft.protocols.capability import qualify

logger = logging.getLogger(__name__)

# 渲染层有路径的块 kind（spec: context-evidence）：集合外的 kind 在 compose 显式
# warning（今日即 blackboard——Phase 3 裁定不渲染，但要被看见，不留静默盲区）。
_RENDERABLE_KINDS = frozenset({
    "identity", "background", "task_spec", "history",
    "capabilities", "directive", "guidance", "reference", "summary",
})

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import (
        AssembledPrompt,
        ContextBlock,
        ContextRequest,
    )


_OBSERVER_ROLE_FALLBACK = "You are an objective observer evaluating task execution results."

# 尾部 observe user message 的判定提示（拼在 ROLE 之后）。
_OBSERVE_JUDGMENT_CUE = (
    "Now act as the observer for the current task. Based on the execution above, judge the "
    f"task's completion status and call `{REPORT_TASK_OUTCOME_NAME}` exactly once with: a `task_status` "
    "of `success` / `retry` / `fail`; an `act_recap` honestly recapping ONLY this act segment — the actor's "
    "execution AFTER the most recent `## Progress So Far` section OR the latest user message, whichever "
    "comes later (`## Progress So Far` is the previous observation's recap; if neither exists this is the "
    "first observation, so start after `## Current Task`). Don't re-narrate anything before that point. "
    "And — when status is success/fail — a concise `task_summary`: the important steps and lessons of the "
    "whole task (a process report, not verbose, and NOT the final output), incorporating the results of any "
    "sub-tasks you dispatched. Optionally review your own sub-tasks via `task_reviews`. Call no other tools."
)

# task compact cue：整体式——坍缩会替掉原始 prompt + 之前所有 `## Progress So Far`，故须概括
# 整段 task-so-far（不是逐段 recap；逐段 act_recap 契约在 ROLE.md，属 observer）。
# 结构化 digest（spec: compact-fidelity）：五个固定小节，宽松解析见
# compact.parse_structured_digest（缺必需节 → 整文降级存档、标 degraded，不中断压缩）。
# Evidence 小节引用可回取的执行身份——read_tool_output 是 tool-result-recovery 落地的
# 回读工具；收敛版工具结果里自带的 invocation_id 在本 cue 面前的历史中可见。
_COMPACTION_INSTRUCTION = (
    "Now act as a memory compactor. Summarize the ENTIRE task execution so far — from the "
    "user's original request through everything done since — into one concise progress digest "
    "a future turn can continue from. Write it as EXACTLY these five sections, each opened by "
    "its heading on its own line:\n"
    "## Goal — the task's objective, one or two sentences.\n"
    "## Constraints — user-imposed constraints and hard limits that must keep being honored "
    "(write \"none\" only if truly none).\n"
    "## Done — what has been completed: key results, decisions made, important tool outputs.\n"
    "## Remaining — unfinished threads, open problems, and failure causes still to be "
    "addressed (write \"none\" only if nothing remains).\n"
    "## Evidence — references to crucial tool outputs a future turn can read back, one per "
    "line, like `- results__read_tool_output(invocation_id='<id>', tail=200): what it "
    "contains`; omit this section if no large output matters.\n"
    "This replaces the earlier turns, so fold in whatever matters (facts discovered, current "
    "state). Every section except Evidence must be present; keep each section as short as the "
    "material allows. Output only the digest text, no preamble."
)

# agent compact cue：概括本 agent 已完成的任务单元——每个任务被要求做什么、结果/关键产出/
# 教训（含其派发的子任务）。L1 折的是超龄完成单元整体（finish 对 + task 层胶囊），不只是派发
# 记录，故不得只压派发；旧 digest 随折叠被 supersede，须显式要求延续其内容。当前 task 的执行
# 细节由 task 域 cue（L3 坍缩）负责，此处忽略。与 task 域同构的五小节契约（spec:
# compact-fidelity）。
_AGENT_COMPACTION_INSTRUCTION = (
    "Now act as a memory compactor for this agent's work history. Summarize the completed tasks "
    "so far — for each: what it was asked to do and its outcome / key results / lessons, "
    "including any sub-tasks it delegated — into one digest the agent can rely on later. Write "
    "it as EXACTLY these five sections, each opened by its heading on its own line:\n"
    "## Goal — what this body of work is about, overall.\n"
    "## Constraints — constraints and hard limits that carried across tasks and must keep "
    "being honored (write \"none\" only if truly none).\n"
    "## Done — each completed task with its outcome and key lessons.\n"
    "## Remaining — threads that earlier digests left open and are still unfinished (write "
    "\"none\" only if nothing remains).\n"
    "## Evidence — references to crucial tool outputs a future turn can read back, one per "
    "line, like `- results__read_tool_output(invocation_id='<id>', tail=200): what it "
    "contains`; omit this section if no large output matters.\n"
    "This digest replaces the older task records, so fold in whatever matters, and carry "
    "forward everything an earlier compaction digest in the conversation already preserved "
    "(especially its Constraints and Remaining). Ignore the current still-running task's own "
    "execution detail. Every section except Evidence must be present. Output only the digest "
    "text, no preamble."
)

# Capabilities 指针：清单全文随「当前 task」user 回合（cache 前缀内稳定），末条 user 只留
# 这一行就近提醒，不再每回合随动态尾部重付整段清单的 token。措辞避开 "## Capabilities"
# 字面量，免与清单标题的存在性断言/检索混淆。
_CAPABILITIES_POINTER = (
    "(Your available capabilities — tools / skills / sub-agents — are listed in the "
    "Capabilities section of the current task message above.)"
)

# Available Tools 段的引子：正文只留「名字 + 入参形状」的索引，描述与完整 JSON Schema
# 随请求的 tools 参数下发，两处都写就是同一份内容付两遍 token。
_TOOLS_INDEX_PREAMBLE = (
    "Listed as `name(param: type)` — optional params are marked `?`. "
    "Full descriptions and parameter schemas come with the tool definitions in this request."
)

_RECOGNIZE_INTENT_INSTRUCTION = (
    "Now set this task's metadata: call `control__update_task_metadata` exactly once. Both `title` and "
    "`description` are REQUIRED and must be non-empty — always provide a best-effort value even if the "
    "instruction is short or vague; never pass empty strings. Add `session_goal` only if the direction "
    "is now clear (it is the only optional field). Then stop and call no other tools."
)

_BACKGROUND_BOUNDARY_DESC = {
    "interrupt": "this segment was interrupted by the user part-way through",
    "plain_text": "you replied in prose and yielded the floor, pausing for the user's input",
    "finish": "the task was closed out with finish_task",
    "normal": "the task ended normally with its final output",
    "dispatch": "you delegated a sub-task, and this task is suspended until it completes",
    # 机械判决路径（observe 无可用 LLM observer / 已取消）：结局由规则定，摘要全靠这次
    # 后台观察补 —— 段可能刚结束、也可能马上要重跑，故措辞不预设收尾。**非 close 边界**
    # （不进 _CLOSE_BOUNDARIES）：不写 _close_report，走段折。
    "mechanical": ("this segment just ended and was judged by rule (no LLM observer ran), "
                   "so this recap is the only account of it"),
}


# close 段（finish/normal）：actor 纯文本或 finish_task 收尾，收尾回合正文落 task.outputs。
# 与 loop.steps.background_observe._CLOSE_BOUNDARIES 保持一致（此处避免跨层 import）。
_CLOSE_BOUNDARIES = {"finish", "normal"}


def _background_observe_cue(boundary: str) -> str:
    desc = _BACKGROUND_BOUNDARY_DESC.get(boundary, _BACKGROUND_BOUNDARY_DESC["normal"])
    is_close = boundary in _CLOSE_BOUNDARIES
    summary_ask = (
        " Also give `task_summary`: a concise process report of the WHOLE task's execution "
        "(call out the important steps and lessons, skip the trivia; it is not the final output), "
        "incorporating the results of any completed sub-tasks."
        if is_close else ""
    )
    return (
        f"Status of the current task: {desc}. Based on the execution above, call "
        "`collect_process_report` exactly once: give `act_recap` (recap ONLY this segment's act — "
        "the work the actor newly did after the last `## Progress So Far` or the **last user "
        "message** in the conversation, whichever is later; if there is neither, this is the first "
        "observation, so start from after `## Current Task`; do not restate anything before that "
        "point)"
        + summary_ask +
        " Just summarize — do not judge success/retry/fail, and do not call any other tool."
    )


def _finish_result_section(request) -> str:
    """close 段（finish/normal）把 actor 的最终产出（task.outputs）注入 prompt。

    finish_task 是 SILENT 工具：其 result 进 task.outputs，**不写任务层对话**；delegate_task
    是 DISPATCH 工具、也排除出对话重建。若某段仅由 finish(+delegate) 组成，从记忆重建的对话里
    看不到任何 actor 动作，观察者会**虚构**一段完成叙述。把 task.outputs 显式喂进来，让它据实总结。
    返回空串表示无产出可注入（保持原行为）。
    """
    task = getattr(request, "task", None)
    outputs = getattr(task, "outputs", None) if task is not None else None
    if not outputs:
        return ""
    text = outputs if isinstance(outputs, str) else content_to_text(outputs)
    text = (text or "").strip()
    if not text:
        return ""
    return (
        "## Actor's Final Output (this segment was closed out with finish_task)\n\n"
        f"{text}\n\n"
        "(The above is the final result the actor submitted — the only authoritative evidence of "
        "what this segment produced. Summarize the segment faithfully from it; do not invent tool "
        "calls, steps, or artifacts that did not actually happen.)"
    )


_HEADING_RE = re.compile(r"^(#{1,6})\s")

# 携带真实图片的 compose purpose（用户裁定 D2）。只有 act 需要模型真看图：
# compact 恰在上下文超预算时触发且产出按 spec §8 恒为纯文本；recognize_intent 是填元数据；
# observe / background_observe 判任务成败靠 actor 产出与工具结果。其余 purpose 的图在
# compose 出口降级成文本占位（见 compose 末尾）。
_IMAGE_BEARING_PURPOSES = frozenset({"act"})


_JSON_TYPE_ABBR = {
    "string": "str", "integer": "int", "number": "num", "boolean": "bool",
    "array": "list", "object": "obj", "null": "null",
}

# 单个工具最多列几个参数：MCP 工具偶有几十个入参，全列会把索引撑爆，超出部分折成省略号。
_MAX_SIGNATURE_PARAMS = 8


def _abbr_type(spec: object) -> str:
    """JSON Schema 的 type 字段 → 短名；联合类型取首个，缺失为 any。"""
    if not isinstance(spec, dict):
        return "any"
    t = spec.get("type")
    if isinstance(t, list):
        t = t[0] if t else None
    if not isinstance(t, str):
        return "any"
    return _JSON_TYPE_ABBR.get(t, t)


def _compact_signature(input_schema: object) -> str:
    """input_schema → ``a: str, b?: int`` 形式的紧凑入参签名（可选参数带 ?）。

    工具的完整描述与 JSON Schema 随请求的 tools 参数一并下发，prompt 里这份清单只作索引：
    留名字与参数形状，够模型判断「有没有这个能力、要准备什么」，细节去 tool 定义里看。
    """
    if not isinstance(input_schema, dict):
        return ""
    props = input_schema.get("properties")
    if not isinstance(props, dict) or not props:
        return ""
    required = input_schema.get("required")
    required = set(required) if isinstance(required, (list, set, tuple)) else set()
    parts: list[str] = []
    for pname, spec in list(props.items())[:_MAX_SIGNATURE_PARAMS]:
        mark = "" if pname in required else "?"
        parts.append(f"{pname}{mark}: {_abbr_type(spec)}")
    if len(props) > _MAX_SIGNATURE_PARAMS:
        parts.append("…")
    return ", ".join(parts)


def _is_named_skill(block: ContextBlock, skill_name: str) -> bool:
    """capability block 是否就是名为 skill_name 的那个 skill。

    只认 qualified 名（``provider__name``），与 PrepareStep 加载 skill 正文时用的
    capability_cache.get_by_qualified_name 同一口径——两个决定（正文进不进 prompt、
    条目从不从清单里剔除）由同一个键决定，不会一边命中一边落空。

    不接受裸名兜底：裸名跨 provider 撞车（``local_skill:pdf`` 与 ``mcp:skills:pdf``
    都叫 pdf），一剔就是一片；且裸名在 get_by_qualified_name 下本就查不到 skill 定义，
    正文根本没进 prompt，此时再把条目藏掉是双输。
    """
    md = block.metadata
    name = md.get("capability_name") or qualify(str(md.get("capability_id", "")))
    return name == skill_name


def _is_blank_content(content: object) -> bool:
    """内容是否真的为空。

    与 llm_gateway._is_empty_content 同义但更严格地只判「有无内容」：
    str 看是否空；列表看是否为空、或是否只含空 TextPart。
    非文本 part（图片）一律视为有内容——纯图片消息不是空消息。

    注意：本函数**不** strip——纯空白文本（如 "   "）在此判为非空，与
    llm_gateway._is_empty_content（会 strip，纯空白判空）不同义。今日安全仅因为
    legalize_messages 链路最终会再经 _is_empty_content 把纯空白消息滤掉一道；
    这对不同判据不是巧合但也没人写下来过，改动前请先看
    tests/unit/test_composer_vs_gateway_blank_content.py。
    """
    if content is None:
        return True
    if isinstance(content, str):
        return not content
    for p in content:
        if not hasattr(p, "text"):
            return False        # 图片 = 有内容
        if p.text:
            return False
    return True


def _shift_markdown_headings(md: str, base_level: int) -> str:
    """把 md 内的标题层级整体下移，使最浅一级标题成为 base_level 的子级（base_level+1）。

    用于把 skill 正文嵌进 "## Instructions for the current task"（H2）之下：md 里的
    H1/H2 会被下移，避免与容器标题抢层级；已比 base_level+1 更深则不动（不上提）。
    跳过 ``` / ~~~ 围栏代码块内的 # 行。
    """
    lines = md.split("\n")

    def _iter_heading_levels():
        in_fence = False
        for ln in lines:
            stripped = ln.lstrip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _HEADING_RE.match(ln)
            if m:
                yield len(m.group(1))

    levels = list(_iter_heading_levels())
    if not levels:
        return md
    shift = (base_level + 1) - min(levels)
    if shift <= 0:
        return md

    out: list[str] = []
    in_fence = False
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            out.append(ln)
            continue
        if not in_fence:
            m = _HEADING_RE.match(ln)
            if m:
                new_level = min(6, len(m.group(1)) + shift)
                ln = "#" * new_level + ln[len(m.group(1)):]
        out.append(ln)
    return "\n".join(out)


@runtime_checkable
class Composer(Protocol):
    """ContextBlock → AssembledPrompt 渲染器。"""

    @abstractmethod
    async def compose(
        self,
        blocks: list[ContextBlock],
        request: ContextRequest,
    ) -> AssembledPrompt:
        ...


class DefaultComposer(Composer):
    """默认 Composer——按 miniAgents 风格组装。

    purpose=act：Actor 风格
    purpose=observe：Observer 风格（复用 act 会话 + 末尾 observe 指令）
    purpose=compact：复用 act system + 会话 + 末尾压缩指令，tools 为空
    purpose=recognize_intent：复用 act system + 会话 + 末尾元数据指令（update_task_metadata 工具）
    """

    async def compose(
        self,
        blocks: list[ContextBlock],
        request: ContextRequest,
    ) -> AssembledPrompt:
        from ctx_weft.core.assembler.assembler import AssembledPrompt

        # 未识别 kind 显式留痕（spec: context-evidence）：无渲染路径的块不再静默消失。
        unsupported = sorted({b.kind for b in blocks} - _RENDERABLE_KINDS)
        if unsupported:
            sources = sorted({b.source for b in blocks if b.kind in unsupported})
            logger.warning(
                "DefaultComposer: no rendering path for block kind(s) %s "
                "(sources: %s) — these blocks are dropped from the prompt",
                unsupported, sources)

        # tools 恒为 []：**composer 不再产工具数组**。它此前从 capabilities blocks 的
        # metadata["llm_tool"] 收一份拷贝，与 CapabilityCache 各存一套、装配后再无同步。
        # 现由 ContextAssembler 给 AssembledPrompt 装 tools_fn（读 cache 现算），composer
        # 只管散文段渲染——同一批能力在 prompt 文本与工具数组里因此不会各说各话。
        if request.purpose == "act":
            system = self._build_actor_system(blocks)
            messages = self._build_actor_messages(blocks, request)
        elif request.purpose == "observe":
            system = self._build_act_system(blocks, request)
            messages = self._build_observer_messages(blocks, request)
        elif request.purpose == "recognize_intent":
            system = self._build_act_system(blocks, request)
            messages = self._build_facet_trailing_messages(blocks, request, _RECOGNIZE_INTENT_INSTRUCTION)
        elif request.purpose == "background_observe":
            system = self._build_act_system(blocks, request)
            messages = self._build_background_observe_messages(blocks, request)
        else:  # compact
            system = self._build_act_system(blocks, request)
            cue = (_AGENT_COMPACTION_INSTRUCTION
                   if (getattr(request, "extra", None) or {}).get("compact_scope") == "agent"
                   else _COMPACTION_INSTRUCTION)
            messages = self._build_facet_trailing_messages(blocks, request, cue)

        # per-purpose 图片策略（用户裁定 D2）：只有 act 需要模型真看图，其余四个 purpose
        # 一律把图降级成确定性文本占位。**必须在 token_count 之前**（架构裁定 T1）——
        # 否则报出的 token 数含图、实际发出的 prompt 已无图，budget 与 compact 会基于
        # 错误的数字判断。五条 purpose 分支都汇到这里，故只需这一处。
        if request.purpose not in _IMAGE_BEARING_PURPOSES:
            messages = [
                m if (c := downgrade_images_to_text(m.content)) is m.content
                else dataclasses.replace(m, content=c)
                for m in messages
            ]

        token_count = request.token_counter(system) + sum(
            request.token_counter(content_to_text(m.content)) + image_tokens(m.content)
            for m in messages
        )
        return AssembledPrompt(
            system=system,
            messages=messages,
            tools=[],
            token_count=token_count,
        )

    # ── Actor ─────────────────────────────────────────────────────────────────

    def _build_actor_system(self, blocks: list[ContextBlock]) -> str:
        """soul + Project Background，--- 分隔。

        Resources（skills/tools/agents）与 skill 指令不再进 system，
        改由 _build_actor_messages 注入本 task 首条 user message（不入 memory）。
        """
        identity = self._first_kind(blocks, "identity")
        background = self._first_kind(blocks, "background")

        parts: list[str] = []
        if identity:
            parts.append(content_to_text(identity.content))
        if background:
            parts.append(f"## Project Background\n\n{content_to_text(background.content)}")

        return "\n\n---\n\n".join(parts)

    def _build_actor_messages(
        self,
        blocks: list[ContextBlock],
        request: ContextRequest,
    ) -> list[LLMMessage]:
        """Actor messages：历史轮次在前，任务上下文 + 当前消息作为最后一条 user message。

        结构：
          [0..N-1] 历史轮次：user / assistant / tool 各自独立的 LLMMessage
          [N]      user: Current Task + Progress So Far + Current Message
        """
        # Phase 3 (2026-06-30): ## Task Background (blackboard predecessor blocks) removed.
        # Predecessors now surface via memory recall (Phase 2 inherit/recall); no bb_blocks needed.
        history_blocks = [b for b in blocks if b.kind == "history"]

        # Progress So Far 改由 task 层 TASK_COMPACT_SUMMARY 段摘要承载（_history 冠标题渲染），
        # 不再据 task.process_report 单独渲染 → 无重复、来源单一（spec 2026-07-01 §3.7）。
        history_pairs = self._history_to_messages_with_sources(history_blocks)
        messages: list[LLMMessage] = [m for m, _src, _mtype, _tid in history_pairs]
        task = request.task
        # 当前 task 的首条 user 回合（## Current Task 框 / directive / capabilities 的落点）：
        # 定位到 task_id == 当前 task 的那条 USER_PROMPT。task-resident 胶囊下，先前/并行 task 的
        # raw body（含其 USER_PROMPT）也经 agent_recall 召回进 history，故同时存在多条 user_prompt；
        # 不能简单取末条（parent resume 后同 agent 子 body 的 user_prompt 更新，会误顶 parent 头）。
        # task_id 匹配不到（fresh task / 旧数据无 task_id）时回退末条 user_prompt。
        # ## Current Message 框另按 _latest_task_user_index 跟随该 task 最新一条（见
        # _frame_current_message）。
        current_task_user_idx = self._current_task_user_index(history_pairs, getattr(task, "id", ""))
        spec_title, spec_desc, spec_prompt = self._task_spec_fields(blocks, task)
        parts: list[str] = []

        if not task.user_prompt_in_memory:
            # daemon 或尚未持久化的路径：实时构建完整的用户消息（spec 取自 task_spec block）。
            # 注意：spec_prompt 来自 _task_spec_fields，读的是 task_spec block 的拍扁文本
            # （见 spec §3② 允许的拍扁点），图片 part 在此路径上不会被携带。标准 loop 走不到
            # 这个分支——driver.run 先调 _persist_user_prompt 把 user_prompt 落盘成 memory
            # 记录，随后 user_prompt_in_memory 恒为真，落到下面的 else 分支（_frame_current_message，
            # 保 parts）。这里安全仅因为「不可达」，不是因为这条路径本身处理了图片——如果将来
            # daemon 或手工装配路径开始命中这个分支，图片会在此静默丢失，需要照 else 分支的
            # 做法改成保 parts，而不是在这里加一行注释就当没看见。
            if spec_title and spec_desc:
                parts.append(f"## Current Task\n{spec_title}\n{spec_desc}")
            elif spec_title:
                parts.append(f"## Current Task\n{spec_title}")
            if spec_prompt:
                parts.append(
                    f"## Current Message\n{spec_prompt}\n\n"
                    "（Reply in the same language as the Current Message above.）"
                )
        else:
            # in-memory：渲染期就地装饰最近一条 task_conversation user 回合
            self._frame_current_message(messages, history_pairs, task, blocks)

        if parts:
            # 这条实时构建的当前任务上下文也是「当前 task」回合；history 里没有 task_conversation
            # user 时（fresh task），directive 落到它身上。
            if current_task_user_idx is None:
                current_task_user_idx = len(messages)
            messages.append(LLMMessage(role="user", content="\n\n".join(parts)))
        # 续跑衔接（仅 act）：历史以 assistant/tool 收尾时垫一条续跑 user 回合——锚定当前任务、
        # 先盘点已完成再只做剩余（典型命中：挂起父任务恢复、段中崩溃 recover、retry 摘要为空）。
        # 文本与 guidance 同源（loop/steps/act_guidance.py 的 build_resume_cue，经
        # extra["act_resume_cue"] 传入）——两者共同构成 act 的态势感知层：cue 开场、
        # guidance（任务树/已完成清单/收尾提醒）收口，勿重做的具体清单只在 guidance 展开。
        # extra 缺失（手构请求/单测）时用结构兜底一句，保证 act prompt 恒以 user 收尾。
        # facet purpose（observe/compact/recognize_intent/background_observe）不垫续跑句——
        # 它们的 trailing cue 自带角色行为定义；末条非 user 时由 _append_to_last_user 新建
        # user 回合承载 capabilities / cue（不回溯粘中部 user，避免指令沉进对话中部失效）。
        if (
            getattr(request, "purpose", None) == "act"
            and messages
            and messages[-1].role != "user"
        ):
            cue = (getattr(request, "extra", None) or {}).get("act_resume_cue", "")
            if not cue:
                task_ref = f"the task: {spec_title}" if spec_title else "the task above"
                cue = (
                    f"You are still working on {task_ref}, resuming from the state recorded "
                    "above. Review what has already been done and continue with only the "
                    "remaining work."
                )
            messages.append(LLMMessage(role="user", content=cue))
        # 连续同角色 / 孤立 tool result 的合法化不在装配层做——统一交由
        # loop.llm_gateway.stream_llm 在发送前处理，使装配层不反向依赖 loop。
        merged = messages
        # Resources 注入（仅发送，不入 memory）：
        #   - directive（当前 task 指令）：act 追加到首条 user message 尾部（紧跟 ## Current Task
        #     等任务上下文之后）；其他 purpose 仍前置到首条。
        #   - capabilities（skills/tools/agents）：所有 purpose 一律拼到「当前 task」user 回合
        #     尾部（directive 之后）——该回合在重建历史里位置/内容稳定，清单落在 prompt cache
        #     前缀内，不再每回合随动态末条重付整段 token；末条只留一行 _CAPABILITIES_POINTER
        #     保住生成点附近的 recency 提示（tools 的可调用性另有 API tools 参数兜底）。
        directive_text = self._build_directive_section(blocks)
        capabilities_text = self._build_capabilities_section(
            blocks, current_skill_name=getattr(request, "extra", {}).get("skill_name", ""))
        if getattr(request, "purpose", None) == "act":
            # directive 落到「当前 task」的 user message（紧跟任务上下文之后），即 history 里末条
            # user_prompt——而不是整个 message 列表的第一条 user（可能是更早的、经 agent_recall
            # 召回的已结束 task 回合）。兜底退回末条 user。
            target_idx = current_task_user_idx
            if target_idx is None:
                target_idx = self._last_user_index(merged)
            merged = self._append_to_user_at(merged, target_idx, directive_text)
        else:
            if directive_text:
                merged = self._prepend_to_first_user(merged, directive_text)
        # capabilities 全文 → 当前 task user 回合（缺失兜底末条 user；连兜底都没有时
        # 新建末条 user 承载——此时清单已在末条，后续不再补指针）。
        cap_idx: int | None = None
        if capabilities_text:
            cap_idx = current_task_user_idx
            if cap_idx is None:
                cap_idx = self._last_user_index(merged)
            if cap_idx is not None:
                merged = self._append_to_user_at(merged, cap_idx, capabilities_text)
            else:
                merged = self._append_to_last_user(merged, capabilities_text)
                cap_idx = len(merged) - 1
        # 末条 user 收尾（act）：证据（检索/召回）→ guidance（任务锚定/plan 全景/已完成
        # 清单/静态指针）——动态内容居尾不打穿 prompt cache 前缀。证据逐轮随查询变化，
        # 落尾部动态区（guidance 之前，guidance 恒收口）。仅 act 渲染证据（facet purpose
        # 的 trailing cue 自带角色定义，证据非其输入）；空节零渲染。
        if getattr(request, "purpose", None) == "act":
            evidence_text = self._build_evidence_section(blocks)
            if evidence_text:
                merged = self._append_to_last_user(merged, evidence_text)
            guidance = self._first_kind(blocks, "guidance")
            if guidance is not None:
                merged = self._append_to_last_user(merged, content_to_text(guidance.content))
        # Capabilities 指针：清单不是最末条消息时补一行（fresh 单消息回合清单本就在末条，不补）。
        if cap_idx is not None and cap_idx != len(merged) - 1:
            merged = self._append_to_last_user(merged, _CAPABILITIES_POINTER)
        return merged

    @staticmethod
    def _current_task_user_index(history_pairs, task_id) -> int | None:
        """定位「当前 task」**首条** user_prompt 的下标：task_id 精确匹配取首条，回退最后一条 user_prompt。

        history_pairs 是 _history_to_messages_with_sources 的 (msg, src, mem_type, task_id) 四元组。
        - 匹配取首条：interactive 任务同一 task 会累积多条 user_prompt（用户每条新消息一条），
          ## Current Task 框 / directive / capabilities 须钉在**开启该 task 的首条消息**上。
          取末条会让这些重量级注入跟着每条新消息漂移——上一轮被装饰的回合在下一轮重建时恢复
          原文，cache 前缀每轮被打穿。（## Current Message 框不在此列：它语义上就是最新一条，
          由 _latest_task_user_index 定位、随新消息走，代价只是尾部一小段 cache。）
        - parent resume 后召回里存在其它 task 的 user_prompt（更新的同 agent 子 body 等）——task_id
          过滤保证不误顶 parent 头。
        - task_id 匹配不到（fresh task / 旧数据无 task_id）时回退末条 user_prompt。
        """
        match = last = None
        for i, (m, _src, mtype, tid) in enumerate(history_pairs):
            if m.role == "user" and mtype == "user_prompt":
                last = i
                if match is None and task_id and tid == task_id:
                    match = i
        return match if match is not None else last

    def _frame_current_message(self, messages, history_pairs, task, blocks=None) -> None:
        """In-memory 路径：渲染期就地装饰「当前 task」的 USER_PROMPT 回合（不落库）。

        history_pairs 是 _history_to_messages_with_sources 返回的 (msg, src, mem_type, task_id) 四元组。
        两处装饰（interactive 多轮下分属不同回合）：
        - ## Current Task 框 + ## Opening Message 标注原文 → task_id == task.id 的**首条**
          （开启该 task 的消息；与 directive/capabilities 同回合，钉住不随新消息漂移，
          cache 前缀稳定。Opening Message 标题在无任务框时也加，把原文和随后追加的
          directive/capabilities 分隔开）；
        - ## Current Message 框 + 同语言提示 → 该 task 的**最新一条** user_prompt——「当前消息」
          语义上就是最新一条，钉首条会把旧消息冒充成当前消息。新消息到来时上一条的框在重建里
          恢复原文、cache 自该回合起失效，但该回合已近尾部，重付的后缀很小。
        首条即最新（单条）时两框合并在同一回合（A 形态）。task_id 匹配不到时两者同回退末条
        user_prompt（合并框）。spec（title/description）取自 task_spec block 的 metadata
        （无块时回退直读 task）。
        """
        task_id = getattr(task, "id", "")
        anchor = self._current_task_user_index(history_pairs, task_id)
        if anchor is None:
            return
        latest = self._latest_task_user_index(history_pairs, task_id)
        if latest is None:
            latest = anchor
        spec_title, spec_desc, _ = self._task_spec_fields(blocks, task)
        prefix = ""
        if spec_title and spec_desc:
            prefix = f"## Current Task\n{spec_title}\n{spec_desc}\n\n"
        elif spec_title:
            prefix = f"## Current Task\n{spec_title}\n\n"
        if anchor != latest:
            # 首条原文冠 ## Opening Message（开启此 task 的消息）：与最新一条的
            # ## Current Message 区分，也把原文和随后追加的 directive/capabilities 分隔开。
            messages[anchor] = LLMMessage(
                role="user",
                content=content_with_prefix(
                    messages[anchor].content, f"{prefix}## Opening Message\n"
                ),
            )
        framed = content_with_prefix(
            messages[latest].content,
            f"{prefix if anchor == latest else ''}## Current Message\n",
        )
        framed = content_with_suffix(
            framed, "\n\n（Reply in the same language as the Current Message above.）"
        )
        messages[latest] = LLMMessage(role="user", content=framed)

    @staticmethod
    def _latest_task_user_index(history_pairs, task_id) -> int | None:
        """「当前 task」**最新一条** user_prompt 的下标（## Current Message 框的落点）。

        task_id 精确匹配取末条（interactive 追问里最新那条才是「当前消息」；同 agent 更晚的
        其它 task user_prompt——如子 body——被过滤掉），匹配不到回退整个列表的末条 user_prompt
        （与 _current_task_user_index 的回退一致，此时两者同指一条、合并框）。
        """
        match = last = None
        for i, (m, _src, mtype, tid) in enumerate(history_pairs):
            if m.role == "user" and mtype == "user_prompt":
                last = i
                if task_id and tid == task_id:
                    match = i
        return match if match is not None else last

    def _build_facet_trailing_messages(
        self,
        blocks: list[ContextBlock],
        request: ContextRequest,
        cue: str,
        extra_sections: list[str] | None = None,
        pre_cue_sections: list[str] | None = None,
        facet_fallback: str = "",
        facet_heading: str = "## Your Current Role",
    ) -> list[LLMMessage]:
        """act 会话 + 末尾一条 user message：purpose facet + cue（+ 可选附加段）。

        observe / compact / recognize_intent 共用。facet 取 purpose 对应的 identity block
        （IdentitySource 已按 purpose 产出），冠以 facet_heading 标题后并入最后一条 user 回合
        （避免连续 user）。
        """
        messages = self._build_actor_messages(blocks, request)
        facet = self._first_kind(blocks, "identity")
        facet_text = content_to_text(facet.content) if facet else ""
        if not facet_text:
            facet_text = facet_fallback

        sections: list[str] = []
        if facet_text:
            if facet_heading:
                facet_text = f"{facet_heading}\n\n{facet_text}"
            sections.extend([facet_text, "---"])
        if pre_cue_sections:
            sections.extend(pre_cue_sections)
        sections.append(cue)
        if extra_sections:
            sections.extend(extra_sections)

        # facet + cue 落到对话末尾：末条是 user 就地并入（避免连续 user），否则新建一条 user
        # 回合承载（续跑兜底仅 purpose=act 注入，facet 的历史可以 assistant/tool 收尾）。
        return self._append_to_last_user(messages, "\n\n".join(sections))

    def _build_resources_section(
        self, blocks: list[ContextBlock], *, current_skill_name: str = "",
    ) -> str:
        """从 capabilities blocks 按 kind 分组渲染（miniAgents 风格）。

        三类（skills / tools / sub-agents）统一按 provider 分块（分级标题 + provider
        描述 + 带具体前缀的引用提示），见 _render_grouped_section。
        """
        cap_blocks = [b for b in blocks if b.kind == "capabilities"]
        skills = [b for b in cap_blocks if b.metadata.get("capability_kind") == "skill"]
        tools = [b for b in cap_blocks if b.metadata.get("capability_kind") == "tool"]
        agents = [b for b in cap_blocks if b.metadata.get("capability_kind") == "agent"]

        # 当前 task 已绑定的 skill：正文整段进了 ## Instructions for the current task，
        # 列表里再列一遍是重复，故从清单中剔除；其余 skill 保留——它们是派发给子任务的候选，
        # 绑了一个 skill 不该让模型看不见别的。判据是 skill 名（task.settings.skill_name，
        # 与 capability_name = qualify(cap.id) 同一口径），不是「有没有 directive 块」。
        if current_skill_name:
            skills = [b for b in skills if not _is_named_skill(b, current_skill_name)]

        sections: list[str] = []
        if skills:
            sections.append(self._render_grouped_section(
                "### Available Skills (assign to tasks where appropriate)",
                skills, action="assigning", noun="skill", noun_plural="skills",
            ))
        if tools:
            sections.append(self._render_grouped_section(
                "### Available Tools (use them via tool calls)",
                tools, action="calling", noun="tool", noun_plural="tools",
                preamble=_TOOLS_INDEX_PREAMBLE,
                signature=True,
            ))
        if agents:
            sections.append(self._render_grouped_section(
                "### Available Sub-Agents",
                agents, action="delegating", noun="sub-agent", noun_plural="sub-agents",
                preamble=f"Delegate via: {DELEGATE_TASK_NAME}(use_subagent=True, subagent_template='<name>')",
            ))
        return "\n\n".join(sections)

    def _render_grouped_section(
        self,
        header: str,
        blocks: list[ContextBlock],
        *,
        action: str,
        noun: str,
        noun_plural: str,
        preamble: str | None = None,
        signature: bool = False,
    ) -> str:
        """按 provider 分块渲染一个能力段（tools / skills / sub-agents 共用）。

        每个有 provider_name 的能力归入对应 provider 子段（#### 标题）；若该 provider
        有 description 则写出，并附「引用时须带前缀 `<prefix>`」提示（prefix 由 provider 名
        qualify 得到，与下方条目名一致）。无 provider_name 的条目（旧路径/测试构造）平铺在
        标题下，保持向后兼容。

        signature=True（tools）：条目渲染成 ``- name(a: str, b?: int)``，不带描述——
        描述与完整 schema 随请求的 tools 参数下发，正文里再写一遍是纯重复。
        signature=False（skills / sub-agents）：仍是 ``- name: 描述``，这两类没有
        tools 参数那条通路，描述只能由正文承载。
        """
        def _line(b: ContextBlock) -> str:
            name = b.metadata.get("capability_name", "?")
            if signature:
                return f"- {name}({_compact_signature(b.metadata.get('input_schema'))})"
            return f"- {name}: {content_to_text(b.content)}"

        ungrouped: list[ContextBlock] = []
        grouped: dict[str, list[ContextBlock]] = {}
        order: list[str] = []
        for b in blocks:
            pname = b.metadata.get("provider_name") or ""
            if not pname:
                ungrouped.append(b)
                continue
            if pname not in grouped:
                grouped[pname] = []
                order.append(pname)
            grouped[pname].append(b)

        lines = [header]
        if preamble:
            lines.append(preamble)
        for b in ungrouped:
            lines.append(_line(b))

        for pname in order:
            group = grouped[pname]
            lines.append("")
            lines.append(f"#### {pname} {noun_plural}")
            pdesc = (group[0].metadata.get("provider_description") or "").strip()
            prefix = qualify(pname) + "__"
            if pdesc:
                lines.append("")
                lines.append(pdesc)
            lines.append(
                f"When {action} these {noun_plural}, prefix the {noun} name with `{prefix}`, "
                "exactly as shown below."
            )
            lines.append("")
            for b in group:
                lines.append(_line(b))
        return "\n".join(lines)

    def _build_capabilities_section(
        self, blocks: list[ContextBlock], *, current_skill_name: str = "",
    ) -> str:
        """Capabilities（skills/tools/agents）段，带 ## Capabilities 引导标题。空时返回 ""。"""
        resources_section = self._build_resources_section(
            blocks, current_skill_name=current_skill_name)
        if not resources_section:
            return ""
        return (
            "## Capabilities\n\n"
            "You can use the capabilities listed below to complete the current task.\n\n"
            + resources_section
        )

    def _build_evidence_section(self, blocks: list[ContextBlock]) -> str:
        """证据段（spec: context-evidence）：检索参考 + 语义召回，含 score 与来源标识。

        两个 source 的块各自成小节；命中内容是回答当前问题的证据，渲染于尾部动态区
        （调用点在 guidance 之前）。空节零渲染（无命中零开销）。
        """
        refs = [b for b in blocks if b.kind == "reference"]
        sums = [b for b in blocks if b.kind == "summary"]
        if not refs and not sums:
            return ""

        def _items(blks: list[ContextBlock]) -> list[str]:
            lines = []
            for b in blks:
                score = b.metadata.get("score")
                score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
                lines.append(f"- ({score_s}) {b.source}: {content_to_text(b.content)}")
            return lines

        sections: list[str] = []
        if refs:
            sections.append("### References\n" + "\n".join(_items(refs)))
        if sums:
            sections.append("### Recalled Memories\n" + "\n".join(_items(sums)))
        return "## Retrieved Evidence\n" + "\n\n".join(sections)

    def _build_directive_section(self, blocks: list[ContextBlock]) -> str:
        """当前 task 的 skill 指令段。空时返回 ""。

        skill 名并入标题（## Instructions for the current task (skill: <name>)）；
        skill 正文里的标题层级整体下移到该 H2 之下（_shift_markdown_headings）。
        """
        directive = self._first_kind(blocks, "directive")
        if not directive:
            return ""
        skill_name = directive.metadata.get("skill_name", "")
        heading = "## Instructions for the current task"
        if skill_name:
            heading = f"{heading} (skill: {skill_name})"
        body = _shift_markdown_headings(content_to_text(directive.content), base_level=2)
        return f"{heading}\n\n{body}"

    def _prepend_to_first_user(
        self, messages: list[LLMMessage], text: str
    ) -> list[LLMMessage]:
        """把 text 拼到首条 user message 内容前（以 --- 分隔）。空 text 为 no-op。"""
        if not text:
            return messages
        out = list(messages)
        for i, m in enumerate(out):
            if m.role == "user":
                # 保 parts：拍扁会丢图。content_with_prefix 对 str 走朴素拼接、逐字节等价。
                out[i] = dataclasses.replace(
                    m, content=content_with_prefix(m.content, f"{text}\n\n---\n\n"))
                return out
        return out

    def _append_to_user_at(
        self, messages: list[LLMMessage], idx: int | None, text: str
    ) -> list[LLMMessage]:
        """把 text 拼到 messages[idx]（应为 user 回合）内容尾部。idx 越界/None 或空 text 为 no-op。"""
        if not text or idx is None or not (0 <= idx < len(messages)):
            return messages
        out = list(messages)
        m = out[idx]
        out[idx] = dataclasses.replace(m, content=content_with_suffix(m.content, f"\n\n{text}"))
        return out

    def _last_user_index(self, messages: list[LLMMessage]) -> int | None:
        """末条 user message 的下标；无 user 时返回 None。"""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                return i
        return None

    def _append_to_last_user(
        self, messages: list[LLMMessage], text: str
    ) -> list[LLMMessage]:
        """把 text 拼到**末条** message（须为 user）尾部；末条非 user / 无消息时新建一条 user 回合。

        刻意不回溯到更早的 user：facet purpose 的历史可以 assistant/tool 收尾（续跑兜底仅
        act 注入），此时把 text 粘到中部的 user 会让 capabilities / facet cue 沉进对话中部、
        被其后的 assistant/tool 回合淹没。空 text 为 no-op。
        """
        if not text:
            return messages
        out = list(messages)
        if out and out[-1].role == "user":
            last = out[-1]
            out[-1] = dataclasses.replace(
                last, content=content_with_suffix(last.content, f"\n\n{text}"))
            return out
        out.append(LLMMessage(role="user", content=text))
        return out

    def _history_to_messages(self, history_blocks: list[ContextBlock]) -> list[LLMMessage]:
        """将 history blocks 转换为真实多轮 LLMMessage 列表。

        跨层（task_conversation + agent_experience）按 timestamp 正序归并，seq_no 作 tiebreak。
        """
        return [m for m, _src, _mtype, _tid in self._history_to_messages_with_sources(history_blocks)]

    def _history_to_messages_with_sources(
        self, history_blocks: list[ContextBlock]
    ) -> list[tuple[LLMMessage, str, str, str]]:
        """同 _history_to_messages，但每条 message 附带来源 block.source、memory event type、task_id。

        返回 (LLMMessage, source, mem_type, task_id) 四元组。
        - source: 来源标识（"agent_recall" / "task_conversation" / "agent_experience" 等）
        - mem_type: MemoryEventType str（如 "user_prompt" / "agent_conversation_turn"）；
          无 type 的合成 block（如 capabilities / progress）传空字符串。
        - task_id: 记录所属 task（USER_PROMPT 记录带；其余可空）。用于把 ## Current Message 框架
          精确定位到「当前 task（request.task.id）」自己的 user_prompt 回合——而非召回历史里最后
          一条 user_prompt（parent resume 后，同 agent 子 body 的 user_prompt 更新，会误顶 parent 头）。
        用于区分「当前 task 自有对话」（USER_PROMPT）与跨 task 的 agent_experience 回合——
        directive 注入和 ## Current Message 框架需定位到当前 task 的 user_prompt 回合。
        """
        sorted_blocks = sorted(
            history_blocks,
            key=lambda b: (b.metadata.get("timestamp", ""), b.metadata.get("seq_no", 0)),
        )
        out: list[tuple[LLMMessage, str, str, str]] = []
        for b in sorted_blocks:
            role = b.metadata.get("role", "user")
            content = b.content
            tool_calls = b.metadata.get("tool_calls") or [] if role == "assistant" else []
            # 判空必须 parts-aware：纯图片消息的 content_to_text 是空串，
            # 旧写法会把它当空消息静默丢弃（spec §6.4）。
            if _is_blank_content(content) and not tool_calls:
                continue  # 真空且无 tool_call 才跳过（保留仅含 tool_call 的 assistant 回合）
            if role == "assistant":
                msg = LLMMessage(
                    role="assistant",
                    content=content,
                    tool_calls=[
                        {"id": tc.get("id", ""), "name": tc.get("name", ""), "input": tc.get("input", {})}
                        for tc in tool_calls
                    ],
                )
            elif role == "tool":
                msg = LLMMessage(
                    role="tool",
                    content=content,
                    tool_call_id=b.metadata.get("tool_call_id", ""),
                )
            else:
                msg = LLMMessage(role="user", content=content)
            mem_type = str(b.metadata.get("type", ""))
            out.append((msg, b.source, mem_type, str(b.metadata.get("task_id", ""))))
        return out

    # ── Observer ──────────────────────────────────────────────────────────────

    def _build_act_system(
        self, blocks: list[ContextBlock], request: ContextRequest
    ) -> str:
        """system = act identity(SOUL) + Project Background。observe/compact/metadata 共用。

        与 act 同构：resources 与 skill 指令不进 system；purpose 专属 facet（observe ROLE /
        compact / metadata persona）由 trailing user message 承载，不进 system。
        """
        parts: list[str] = []
        act_facet = request.template.identity.get("act") if request.template else None
        if act_facet and getattr(act_facet, "text", ""):
            parts.append(act_facet.text)
        background = self._first_kind(blocks, "background")
        if background:
            parts.append(f"## Project Background\n\n{content_to_text(background.content)}")
        return "\n\n---\n\n".join(parts)

    def _build_observer_messages(
        self,
        blocks: list[ContextBlock],
        request: ContextRequest,
    ) -> list[LLMMessage]:
        """act 风格完整会话 + 尾部一条 observe user message（仅发送，不入 memory）。

        复用 _build_facet_trailing_messages：observe facet（ROLE）+ 判定提示 + 可复核清单。
        subtask 可 confirm/reopen，predecessor 只读。
        """
        # Phase 3 (2026-06-30): blackboard subtask/predecessor block rendering removed.
        # Predecessor results surface via memory recall (Phase 2); subtask review handles come
        # from task_manager via request.extra["subtask_reviews"] (Task 1 below).
        # The blackboard mechanism (subscribe_topic/recall_topic/BlackboardSource) is kept intact.

        extra_sections: list[str] = []

        # Phase 3: reviewable sub-tasks come from task_manager via request.extra (not blackboard).
        # The observer reads each child's RESULT from the conversation (Phase 2); this clause only
        # surfaces the actionable handles (task_id/title/outcome) so it can confirm/reopen via task_reviews.
        reviews = (getattr(request, "extra", {}) or {}).get("subtask_reviews") or []
        if reviews:
            lines = [f"{SUBTASKS_REVIEW_HEADING} (confirm / reopen via `task_reviews`, referencing the exact task_title):"]
            for r in reviews:
                lines.append(f"- {r['task_id']} — {r.get('title', '')} [{r.get('outcome', '')}]")
            extra_sections.append("\n".join(lines))

        # finish_task 的产出走 SILENT，不入 task 层、不在重建的对话里——但 observer 须看到 actor
        # 最终提交了什么。显式补一段并标注来源（act 阶段调用 finish_task 的结果），置于判定提示之前。
        pre_cue_sections: list[str] = []
        outputs = getattr(request.task, "outputs", None)
        outputs_text = outputs if isinstance(outputs, str) else (content_to_text(outputs) if outputs else "")
        if outputs_text:
            pre_cue_sections.append(
                "## Final output\n"
                "The actor ended the act phase by calling the `finish_task` tool; the result it "
                "submitted (shown to the user) was:\n\n"
                + outputs_text
            )

        return self._build_facet_trailing_messages(
            blocks,
            request,
            _OBSERVE_JUDGMENT_CUE,
            extra_sections=extra_sections,
            pre_cue_sections=pre_cue_sections,
            facet_fallback=_OBSERVER_ROLE_FALLBACK,
        )

    def _build_background_observe_messages(self, blocks, request):
        """act 风格会话 + 尾部 background-observe cue（ROLE facet + boundary 状态 + 只给 process_report）。

        close 段（finish/normal）额外把 actor 的最终产出（task.outputs）注入 cue 之前——否则仅由
        finish(+delegate) 组成的段在对话重建里无 actor 动作可见，观察者会虚构完成叙述。
        """
        boundary = (getattr(request, "extra", {}) or {}).get("observe_boundary", "normal")
        pre_cue: list[str] | None = None
        if boundary in _CLOSE_BOUNDARIES:
            section = _finish_result_section(request)
            if section:
                pre_cue = [section]
        return self._build_facet_trailing_messages(
            blocks, request, _background_observe_cue(boundary),
            pre_cue_sections=pre_cue,
            facet_fallback=_OBSERVER_ROLE_FALLBACK,
        )

    @staticmethod
    def _render_bb(blks: list[ContextBlock]) -> str:
        """渲染 blackboard 结果块为复核清单行。"""
        lines: list[str] = []
        for b in blks:
            title = b.metadata.get("title", "")
            outcome = b.metadata.get("outcome", "")
            body = content_to_text(b.content)
            head = f"{title} [{outcome}]" if title else (f"[{outcome}]" if outcome else "")
            lines.append(f"- {head}: {body}" if head else f"- {body}")
        return "\n".join(lines)

    # ── 共用工具 ──────────────────────────────────────────────────────────────

    def _first_kind(
        self,
        blocks: list[ContextBlock],
        kind: str,
    ) -> ContextBlock | None:
        for b in blocks:
            if b.kind == kind:
                return b
        return None

    def _task_spec_fields(self, blocks, task) -> tuple[str, str, str]:
        """当前 task 的 spec 字段 (title, description, user_prompt)。

        优先取 TaskSpecSource 产的 task_spec block 的 metadata；无块时回退直读 task
        （兼容手构 blocks / 未注册 TaskSpecSource 的调用）。block 是元数据载体，
        composer 用它去就地装饰当前消息，而非把它当独立消息渲染。
        """
        blk = self._first_kind(blocks, "task_spec") if blocks else None
        if blk is not None:
            md = blk.metadata
            return (
                md.get("title", "") or "",
                md.get("description", "") or "",
                md.get("user_prompt", "") or "",
            )
        title = task.title or ""
        description = task.description or ""
        up = task.user_prompt
        user_prompt = up if isinstance(up, str) else (content_to_text(up) if up else "")
        return title, description, user_prompt

