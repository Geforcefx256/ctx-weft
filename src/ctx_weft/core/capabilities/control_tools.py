"""ControlCapabilityProvider：内置控制能力。

工具定义用 @control_tool 装饰器 + Annotated 类型注解声明，
input_schema 由 extract_schema() 从函数签名自动提取（自动跳过 ctx 参数）。

工具函数体通过注入的 ControlContext 直接操作 Task / TaskManager / Session，
不再通过 metadata 把控制信号传递给 ActStep。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any

from ctx_weft.core.capabilities.schema import extract_schema
from ctx_weft.core.capabilities.task_inputs import InvalidTaskInputs, normalize_task_inputs
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.headings import SUBTASKS_REVIEW_HEADING
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    Purpose,
    SessionScopedCapabilityProvider,
    ToolCapability,
    ToolCapabilityProvider,
    qualify,
)
from ctx_weft.protocols.context import ProviderContext

if TYPE_CHECKING:
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task

logger = logging.getLogger(__name__)

PROVIDER_NAME = "control"

# runtime-injected context parameter — excluded from LLM schema and from arguments filtering
_SKIP: frozenset[str] = frozenset({"ctx"})

# Qualified (LLM-facing) names for the built-in control tools. Use these anywhere
# a control tool is named to the LLM (prompts, docstrings shown as descriptions).
FINISH_TASK_NAME = qualify(f"{PROVIDER_NAME}:finish_task")
DELEGATE_TASK_NAME = qualify(f"{PROVIDER_NAME}:delegate_task")
DELEGATE_PLAN_NAME = qualify(f"{PROVIDER_NAME}:delegate_plan")
ASK_USER_NAME = qualify(f"{PROVIDER_NAME}:ask_user")
REPORT_TASK_OUTCOME_NAME = qualify(f"{PROVIDER_NAME}:report_task_outcome")
BACKGROUND_PROCESS_REPORT_NAME = qualify(f"{PROVIDER_NAME}:collect_process_report")
UPDATE_TASK_METADATA_NAME = qualify(f"{PROVIDER_NAME}:update_task_metadata")

# delegate_plan 的 actor-visible ack 及 gateway 配对 tool result 内容。
_PLAN_DISPATCH_ACK = ("Plan created. Its sub-tasks will now be started one by one "
                      "via start_task.")

# `ask_user` 撞上无人值守（`UnattendedHitl`）时回灌给 actor 的工具结果。
#
# 文本住在这里而不是 gateway：这是**给 ask_user 的调用者看的**上下文化答复，与该工具
# 自身的语义（「我需要一个人的决定」）配套；gateway 只负责在唯一的登记入口被守卫挡下时
# 把它取出来。绝不能让 `UnattendedHitl` 逸出到 agent loop——那会把「没人可问」变成一次
# run 失败，而 actor 该收到的是一个说得清楚的结果。
ASK_USER_UNATTENDED_RESULT = (
    "[No human available: this task runs unattended in the background, so nobody can "
    "answer your question. Decide for yourself based on the information you already "
    "have, or call control__finish_task and state clearly what blocked you.]"
)


def _mode(interactive: bool) -> str:
    """Map the LLM-facing `interactive` bool to Task.interaction_mode."""
    return "interactive" if interactive else "auto"


def _child_mode(interactive: bool, parent: "Task | None") -> str:
    """Resolve a delegated child's interaction_mode under its parent.

    interactive 只能沿用户面向链路向下传递：父任务非 interactive（auto=自治）时，子任务即使
    请求 interactive 也**静默降级为 auto**——自治分支不该凭空长出对话子任务（要用户输入用 ask_user）。
    否则一个没人盯的自治分支会冷 park 等用户，甚至永不回交父任务而卡死该分支。
    """
    mode = _mode(interactive)
    if mode == "interactive" and parent is not None and parent.interaction_mode != "interactive":
        logger.warning(
            "delegate: downgrading interactive sub-task to auto under non-interactive parent task %s",
            parent.id,
        )
        return "auto"
    return mode


# ── ControlMetaKey ────────────────────────────────────────────────────────────


class ControlMetaKey:
    """Step 间传递的 metadata key。仅保留仍需通过 metadata 传递的信号。"""
    CONTROL_ACTION = "control_action"
    HITL_REQUESTED = "hitl_requested"
    REOPEN_TASK_IDS = "reopen_task_ids"  # observer review：需重新入队的子任务 id（→ reopen_chain）


# ── ControlContext ─────────────────────────────────────────────────────────────


@dataclass
class ControlContext:
    """运行时注入到 control tool 函数体的执行上下文（不暴露给 LLM）。"""
    session_id: str
    task_id: str
    agent_id: str
    task: "Task | None"
    task_manager: "TaskManager | None"
    session: "Session | None"
    tool_call_id: str = ""  # 发起本次调用的 LLM tool_call id（spec/06 §5，委派回填用）


# ── ControlResult ─────────────────────────────────────────────────────────────


@dataclass
class ControlResult:
    """control tool 函数的返回值。

    content  → 给 LLM 看的确认文本（拼入 tool result message）
    metadata → 传递给 Step 的附加信号（可选）
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── @control_tool 装饰器 ──────────────────────────────────────────────────────

# name → (ToolCapability, handler_fn)
_CONTROL_TOOLS: dict[str, tuple[ToolCapability, Callable[..., ControlResult]]] = {}


def control_tool(*, purposes: list[Purpose], input_schema: dict[str, Any] | None = None):
    """装饰器：声明控制工具的 purposes + 自动提取 input_schema，注册到 _CONTROL_TOOLS。

    input_schema 给定时用它覆盖自动提取——extract_schema 只能生成扁平 schema
    （list→array 无 items、dict→object 无 properties），需要嵌套对象/数组的工具
    （如 ask_user 的 questions）须手写 schema 覆盖。
    """
    def decorator(fn: Callable[..., ControlResult]) -> Callable[..., ControlResult]:
        first_line = (fn.__doc__ or "").strip().split("\n")[0].strip()
        cap = ToolCapability(
            id=f"{PROVIDER_NAME}:{fn.__name__}",
            name=fn.__name__,
            kind="tool",
            purposes=list(purposes),
            description=first_line,
            input_schema=input_schema or extract_schema(fn),  # extract_schema 默认排除 "ctx"
            side_effects=False,
        )
        _CONTROL_TOOLS[fn.__name__] = (cap, fn)
        return fn
    return decorator


# ── 工具定义 ──────────────────────────────────────────────────────────────────


@control_tool(purposes=["act"])
def delegate_task(
    title: Annotated[str, "Short imperative title for the sub-task (≤20 chars)"],
    description: Annotated[
        str,
        "WHAT the sub-task must achieve — its goal/content only. No HOW, no tool names, and do NOT "
        "restate dispatch/orchestration choices such as 'use subagent' / 'inherit memory' / which "
        "skill (those go in the use_subagent / inherit_memory / skill_name params). This text becomes "
        "the sub-task's own `## Current Task`, so anything off-goal will mislead it when it runs.",
    ] = "",
    task_prompt: Annotated[str, "Detailed prompt extracted from the user request for this task"] = "",
    skill_name: Annotated[str, "Skill to assign to the task, or empty if none"] = "",
    use_subagent: Annotated[bool, "True if the task should run in a dedicated sub-agent"] = False,
    subagent_template: Annotated[str, "Template id for the sub-agent; empty = system default"] = "",
    inherit_memory: Annotated[bool, "True if the sub-agent should inherit session memory"] = True,
    interactive: Annotated[
        bool,
        "True if this sub-task is human-interactive: a plain-text turn pauses and waits "
        "for the user. Default False = autonomous (the actor must call finish_task "
        "to finish).",
    ] = False,
    inputs: Annotated[dict, "Optional input data for the sub-task"] = None,
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Delegate ONE new sub-task to run separately; the current task suspends until its sub-tasks finish. To finish your OWN task instead, use control__finish_task."""
    from ctx_weft.core.models.task import Task as TaskModel

    if ctx is None or ctx.task_manager is None or ctx.task is None:
        return ControlResult(content=f"Sub-task '{title}' scheduled.")

    # spec: task-handoff——inputs 规整（纯 JSON + 分级收敛），不合格响亮拒绝、不派发。
    try:
        normalized_inputs = normalize_task_inputs(inputs)
    except InvalidTaskInputs as e:
        return ControlResult(content=f"Cannot delegate sub-task {title!r}: {e}")

    child = TaskModel(
        id=generate_id("tsk"),
        session_id=ctx.session_id,
        status="PENDING",
        tenant_id=ctx.task.tenant_id,
        parent_task_id=ctx.task_id,
        creator_agent_id=ctx.agent_id,
        title=title,
        description=description,
        user_prompt=task_prompt or description,
        inputs=normalized_inputs,
        origin_tool_call_id=ctx.tool_call_id or None,
        origin_tool_name=DELEGATE_TASK_NAME,  # 保真：actor 确实调了 delegate_task → finalize 铸框用真名
        interaction_mode=_child_mode(bool(interactive), ctx.task),
        # 无人值守**继承**，不给 LLM 旋钮（schema 里没有这个参数）：「有没有人在」是
        # 作业被怎么起起来的事实，不是 actor 可以自行宣布的。子任务的 interaction_mode
        # 不用在这里单独处理——父任务无人值守即 auto，`_child_mode` 的「父不 interactive
        # 则子不 interactive」规则已经把子任务一并按住了。
        unattended=ctx.task.unattended,
        settings=NormalTaskSettings(
            skill_name=skill_name,
            use_subagent=bool(use_subagent),
            subagent_template=subagent_template,
            inherit_memory=bool(inherit_memory),
        ),
        created_at=now_utc(),
    )
    ctx.task_manager.stage_task(child, parent_task_id=ctx.task_id)

    if isinstance(ctx.task.settings, NormalTaskSettings):
        ctx.task.settings.spawn_titles.append(title)

    ctx.task.suspend_requested = True     # 路由意图；task 落 SUSPENDED 归 TaskManager
    ctx.task.actor_done = True
    # spec: task-handoff——ack 回传子任务 id：模型后续的审核/引用一律用这个稳定句柄。
    return ControlResult(
        content=f"Sub-task '{title}' scheduled (task_id: {child.id})."
    )


@control_tool(purposes=["act"])
def delegate_plan(
    tasks: Annotated[
        list,
        (
            "Ordered list of task specs. Each item: "
            "title (str, the sub-task's goal only — no dispatch flags), "
            "description (str, WHAT the sub-task must achieve — its goal/content only; no HOW, and do NOT "
            "restate use_subagent/inherit_memory/skill in the text: it becomes the sub-task's own "
            "`## Current Task` and off-goal words mislead it), "
            "task_prompt (str, detailed prompt — goal/content only, same rule as description), "
            "skill_name (str, skill to assign or empty), "
            "use_subagent (bool), subagent_template (str), "
            "inherit_memory (bool, default true), "
            "interactive (bool, default false — true makes the task human-interactive: "
            "a plain-text turn pauses and waits for the user instead of requiring control__finish_task), "
            "inputs (object, optional JSON data handed to the sub-task verbatim — shown to it "
            "under a dedicated '## Inputs' section; keep it small), "
            "run_if (str, when this task may start relative to its predecessor: 'success' "
            "(default — only after the predecessor FINISHED) or 'any' (cleanup-style: start "
            "once the predecessor reached any terminal state, including failure))."
        ),
    ],
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Delegate SEVERAL ordered sub-tasks in one call (each runs after the previous); the current task suspends until they all finish. For a single sub-task use control__delegate_task; to finish your OWN task, use control__finish_task."""
    from ctx_weft.core.models.task import Task as TaskModel

    if not isinstance(tasks, list):
        tasks = []

    n = len(tasks)
    if ctx is None or ctx.task_manager is None or ctx.task is None:
        return ControlResult(content=_PLAN_DISPATCH_ACK)

    # spec: task-handoff——先整单校验再铸造：inputs 规整与 run_if 合法性任何一项不过，
    # 整个调用拒绝、零子任务创建（不留下半截 plan）。
    prepared: list[tuple[dict, dict | None, str]] = []  # (spec, inputs, run_if)
    for idx, spec in enumerate(tasks):
        if not isinstance(spec, dict):
            continue
        title = spec.get("title", "subtask")
        run_if = spec.get("run_if", "success")
        if run_if not in ("success", "any"):
            return ControlResult(content=(
                f"Cannot delegate plan: task {idx + 1} ({title!r}) declares invalid "
                f"run_if {run_if!r}; use 'success' or 'any'."
            ))
        try:
            normalized_inputs = normalize_task_inputs(spec.get("inputs"))
        except InvalidTaskInputs as e:
            return ControlResult(content=(
                f"Cannot delegate plan: task {idx + 1} ({title!r}): {e}"
            ))
        prepared.append((spec, normalized_inputs, run_if))

    titles: list[str] = []
    child_ids: list[str] = []
    prev_ids: list[str] = []
    for spec, normalized_inputs, run_if in prepared:
        title = spec.get("title", "subtask")
        child = TaskModel(
            id=generate_id("tsk"),
            session_id=ctx.session_id,
            status="PENDING",
            tenant_id=ctx.task.tenant_id,
            parent_task_id=ctx.task_id,
            creator_agent_id=ctx.agent_id,
            title=title,
            description=spec.get("description", ""),
            user_prompt=spec.get("task_prompt") or spec.get("description", ""),
            inputs=normalized_inputs,
            origin_tool_call_id=generate_id("tcall"),
            tracking_task_ids=list(prev_ids),
            # spec: task-handoff——本任务对前序的依赖条件（run_if）。物化进
            # dep_conditions，push_task 落盘、restore/reopen 重建。
            dep_conditions={prev_ids[-1]: run_if} if prev_ids else None,
            interaction_mode=_child_mode(bool(spec.get("interactive", False)), ctx.task),
            # 同 delegate_task：继承而非声明；interaction_mode 由 `_child_mode` 接住。
            unattended=ctx.task.unattended,
            settings=NormalTaskSettings(
                skill_name=spec.get("skill_name", ""),
                use_subagent=bool(spec.get("use_subagent", False)),
                subagent_template=spec.get("subagent_template", ""),
                inherit_memory=bool(spec.get("inherit_memory", True)),
            ),
            created_at=now_utc(),
        )
        ctx.task_manager.stage_task(
            child,
            parent_task_id=ctx.task_id,
            blocked_by=[prev_ids[-1]] if prev_ids else None,
        )
        prev_ids.append(child.id)
        child_ids.append(child.id)
        titles.append(title)

    if isinstance(ctx.task.settings, NormalTaskSettings):
        ctx.task.settings.spawn_titles = titles
    ctx.task.suspend_requested = True     # 路由意图；task 落 SUSPENDED 归 TaskManager
    ctx.task.actor_done = True
    # spec: task-handoff——ack 回传与 spec 顺序对应的 id 列表（审核/引用的稳定句柄）。
    return ControlResult(content=(
        "Plan created. Its sub-tasks will now be started one by one via start_task "
        f"(task_ids in order: {', '.join(child_ids)})."
    ))


@control_tool(purposes=["act"])
def finish_task(
    deliverables_summary: Annotated[
        str,
        "OPTIONAL. A brief recap of the concrete deliverables of this task (e.g. the key files "
        "changed / artifacts produced), for the observer and whoever delegated this task. This "
        "is NOT your reply to the user — write your final reply as your normal message text in "
        "this same turn; that message is what the user sees AND the deliverable handed off. "
        "Leave this empty when there is nothing concrete to itemize.",
    ] = "",
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Finish the CURRENT task and hand off to review — call ONLY when the task goal is fully achieved; if work remains, keep working instead. Write your final reply to the user as your normal message text in this same turn — that message IS the reply shown to the user and the deliverable handed off; this tool just ends the task. The optional `deliverables_summary` is a brief recap of concrete artifacts for the reviewer, NOT your answer. Use when YOUR work is done — NOT to create new work (use control__delegate_task / control__delegate_plan for that)."""
    if ctx is not None and ctx.task is not None:
        # task.outputs 由 ActStep 收尾时合成（收尾回合正文 + deliverables_summary，spec 2026-07-01）；
        # 此处不写 outputs。actor_done 让 act 循环退出；不置 SUSPENDED → next_step=observe。
        ctx.task.actor_done = True
    return ControlResult(content="Task finished.")


def _collect_reviews(
    task_reviews: list,
    ctx: "ControlContext",
) -> tuple[dict[str, str], str]:
    """匹配**自己派生的子任务**的 review，返回 ({待 reopen 的子任务 id: reasoning}, 给 LLM 的摘要)。

    每条 review: {task_id, review_status('confirmed'|'reopen'|'skip'), reasoning}。

    权限范围：只有当前 task 直接派生的子任务（children_of）才能被 review / reopen。
    同 plan 前序仅作只读上下文（经 memory recall 以对话形态出现，无专门段），不可在此操作；
    任何不在子任务集合内的 task_id 都会被拒绝并在摘要里反馈给 LLM。

    条目级显式校验（spec: task-handoff）：gateway 的控制工具严格校验只查顶层参数，
    嵌套条目它不管——所以在这里逐条把关：缺 task_id / task_id 非字符串 / 携带未知
    字段（含旧 task_title）→ 该条目拒绝且回执说明，其余合法条目照常生效。同名、
    改名、顺序变化都不影响配对（键是稳定 id，标题仅展示）。

    本函数**不直接改状态**——仅收集需 reopen 的 FINISHED 子任务及其 reasoning，交由
    ControlCapabilityProvider 通过 TaskManager.reopen_chain(id, reason) 正规重排
    （head + 其 plan 后续一并入队 + 重写 user_prompt + 发 TASK_REQUEUED）。reasoning 会成为重做指令。
    'confirmed' / 'skip' 仅记录摘要。按 task_id 精确匹配。
    """
    if not isinstance(task_reviews, list) or ctx.task_manager is None or ctx.task is None:
        return {}, ""

    allowed_keys = {"task_id", "review_status", "reasoning"}
    child_ids = ctx.task_manager.children_of(ctx.task.id)
    children = {t.id: t for t in ctx.task_manager.all_tasks() if t.id in child_ids}

    reopen: dict[str, str] = {}
    applied: list[str] = []
    denied: list[str] = []
    for review in task_reviews:
        if not isinstance(review, dict):
            denied.append("non-object entry")
            continue
        # 条目级校验：缺 task_id / 非字符串 / 未知字段（含旧 task_title）→ 整条拒绝
        task_id = review.get("task_id")
        unknown = set(review) - allowed_keys
        if not isinstance(task_id, str) or not task_id:
            denied.append(f"entry without valid 'task_id' ({sorted(review)}); "
                          "reference sub-tasks by their task_id")
            continue
        if unknown:
            denied.append(f"{task_id!r} (unknown field(s) {sorted(unknown)}; "
                          "use exactly task_id / review_status / reasoning — 'task_title' "
                          "is no longer accepted)")
            continue
        review_status = review.get("review_status", "")
        reasoning = review.get("reasoning", "")
        target = children.get(task_id)
        if target is None:
            denied.append(f"{task_id!r}")  # 非自己的子任务（含前序 / 其它）→ 越权
            continue
        title = target.title or task_id
        if not reasoning:
            continue
        if review_status == "reopen":
            if target.status == "FINISHED":
                reopen[target.id] = reasoning
                applied.append(f"reopened {title!r} ({task_id}) (+ its plan successors)")
            else:
                # 已是 PENDING/进行中，本来就会跑，无需 reopen
                applied.append(f"already active {title!r} ({task_id})")
        elif review_status == "confirmed":
            applied.append(f"confirmed {title!r} ({task_id})")
        elif review_status == "skip":
            applied.append(f"skipped {title!r} ({task_id})")

    parts: list[str] = []
    if applied:
        parts.append(f"Reviews applied: {'; '.join(applied)}.")
    if denied:
        parts.append(
            f"Ignored (invalid or out of scope — reference YOUR OWN sub-tasks by task_id "
            f"exactly as listed): {'; '.join(denied)}."
        )
    summary = ("\n" + " ".join(parts)) if parts else ""
    return reopen, summary


@control_tool(purposes=["observe"])
def report_task_outcome(
    task_status: Annotated[
        str,
        "Outcome of the current task — one of 'success' | 'retry' | 'fail'. "
        "Don't over-think — once the situation is clear, call this tool promptly. "
        "'success' if completed successfully (give a thorough act_recap of the outcome and key steps); "
        "'retry' if this attempt fell short but is worth another try (act_recap describes what is missing, "
        "task_failure_reason the concrete blocker of this attempt, next_step_hint the concrete next step); "
        "'fail' if it cannot be completed and should NOT be retried (act_recap/task_failure_reason explain why).",
    ],
    act_recap: Annotated[
        str,
        "An honest recap of what this segment's act did: what you changed or produced, which tools "
        "you called, and whether anything failed. First person, faithful to what actually ran. "
        "Scope = the work the actor newly did after the last `## Progress So Far` or the last user "
        "message in the conversation, whichever is later (on a first observation, start from the "
        "beginning of the task); do not restate anything before that point. Written to memory, and "
        "reused as the next round's `## Progress So Far` on retry.",
    ],
    task_summary: Annotated[
        str,
        "Required when task_status is 'success' or 'fail': a CONCISE process report of the WHOLE task — "
        "the important steps taken and lessons/experience, incorporating any sub-task results. "
        "Keep it high-signal, NOT a verbose blow-by-blow. This is NOT the final output: the final "
        "deliverable shown to the user is the actor's closing message text (the same turn that calls "
        "finish_task), not any tool argument. Leave empty for 'retry'.",
    ] = "",
    task_failure_reason: Annotated[
        str,
        "Required when task_status is 'fail' or 'retry'. "
        "For 'fail': explain specifically what went wrong: which step failed, what error or unexpected result "
        "was encountered, and what the root cause is. "
        "For 'retry': state what concretely blocked or fell short in this attempt — if the retry limit is hit, "
        "this is shown to the user as the failure reason. Leave empty only for 'success'.",
    ] = "",
    task_reviews: Annotated[
        list,
        "Optional reviews of YOUR OWN sub-tasks only — exactly those listed under the "
        f"'{SUBTASKS_REVIEW_HEADING}' section in the context. You may NOT review anything "
        "else (upstream/predecessor tasks appear as read-only conversation context); "
        "such entries are rejected. "
        f"Each entry is an object with exactly these fields: task_id (str, the id shown as "
        f"'{SUBTASKS_REVIEW_HEADING}' entries — NOT the title), "
        "review_status ('confirmed'|'reopen'|'skip'), reasoning (str, required). "
        "Entries missing task_id or carrying other fields (e.g. the old 'task_title') are "
        "rejected individually. "
        "'reopen' re-runs that FINISHED sub-task from scratch: its previous output is "
        "automatically shown to the re-run and your 'reasoning' becomes the revision "
        "instruction — so write 'reasoning' as concrete, actionable feedback (what is "
        "wrong and what must be fixed), not just a verdict. Reopening a sub-task that is "
        "part of a plan AUTOMATICALLY reopens its later plan steps as well (they will be "
        "redone against the corrected result), so you only need to reopen the earliest "
        "step that is wrong. "
        "'confirmed'/'skip' are recorded only, no re-run. Only FINISHED sub-tasks can be "
        "reopened. Omit sub-tasks you have no information about; leave empty otherwise.",
    ] = None,
    next_step_hint: Annotated[
        str,
        "Optional. If there are obvious risks, blockers, or important concerns the next actor turn should be "
        "aware of, describe them here. Leave empty if nothing notable.",
    ] = "",
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Record the review verdict for the current task (success / retry / fail), and optionally review your own sub-tasks in the same call."""
    task = ctx.task if ctx else None
    # observe 裁决三态。机械退出（max_turns/context_limit）由系统在 ObserveStep 归为 retry。
    if task_status not in ("success", "retry", "fail"):
        task_status = "retry"
    # 一次性转向（只对下一次 attempt 有效）与永久记录（act_recap）分开累积：act_recap 会经
    # process_report → 段摘要 / finish 对进永久记忆，把 hint 拌进去会让它在任务完成后仍留在
    # 历史里（过期的 Next Step Hint）。hint 走 task.next_step_hint → guidance，不入 memory。
    hint = f"Next Step Hint: {next_step_hint}" if next_step_hint else ""

    metadata: dict[str, Any] = {}
    if task is not None:
        # 护栏：没有最终产出就不允许判成功，改判 retry（提示下一轮调 finish_task 收尾）。
        if task_status == "success" and not task.outputs:
            task_status = "retry"
            _hint = ("The previous round ended without a final output. Review the recap above "
                     "and judge whether this task still needs more work. If it does, continue with the "
                     "necessary tool calls. Once everything required is done, write your final reply to "
                     "the user as your normal message text and then call the `control__finish_task` tool "
                     "to complete the task — your message text is the reply and the deliverable.")
            hint = f"{hint}\n\n{_hint}" if hint else _hint

        task.process_report = act_recap
        task.next_step_hint = hint or None
        task.task_summary = task_summary
        task.process_report_at = now_utc()
        # **只写判决，不写状态**（Task 4）：三态 verdict 经 FinalizeStep 的 RunOutcome
        # 交给 TaskManager，由处置表决定 task 落 FINISHED / FAILED / PENDING。
        task.observer_outcome = task_status
        task.actor_done = True
        if task_status == "success":
            task.error = None  # 清掉上一轮 retry 暂存的受阻原因，FINISHED 任务不携带 error
        elif task_status == "fail":
            task.error = task_failure_reason
        else:  # retry
            # 本轮受阻原因暂存 task.error：retry 耗尽降级 fail 时它就是真死因
            # （TaskFailed 的 TASK_FAILED_RETRY_EXHAUSTED 携带）；下一轮判决必然覆盖或清空。
            task.error = task_failure_reason or None

    review_msg = ""
    if task_reviews and ctx:
        reopen_map, review_msg = _collect_reviews(task_reviews, ctx)
        if reopen_map:
            metadata[ControlMetaKey.REOPEN_TASK_IDS] = reopen_map

    failure_part = f" Failure reason: {task_failure_reason}" if task_status == "fail" and task_failure_reason else ""
    return ControlResult(
        content=f"Assessment recorded: outcome={task_status}.{failure_part} {act_recap}{review_msg}",
        metadata=metadata,
    )


@control_tool(purposes=["background_observe"])
def collect_process_report(
    act_recap: Annotated[
        str,
        "Honest recap of what the LAST act phase actually did: what was changed/produced, which tools "
        "were called and whether any failed. First-person, faithful to the transcript, this segment only.",
    ],
    task_summary: Annotated[
        str,
        "For a close (finish/normal) segment: a CONCISE process report of the WHOLE task — important steps "
        "and lessons, incorporating any sub-task results. High-signal, not verbose. NOT the final output "
        "(that is the actor's closing message text, delivered in the same turn as finish_task). "
        "Leave empty for non-close segments.",
    ] = "",
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Summarize the current segment. Zero state write: never touches task.status / process_report / etc.
    Returns act_recap as content + task_summary in metadata for the close-out finish 对."""
    return ControlResult(content=act_recap, metadata={"task_summary": task_summary})


@control_tool(purposes=["recognize_intent"])
def update_task_metadata(
    title: Annotated[str, "Task title (REQUIRED, must be non-empty): ≤20 chars, start with a verb, "
                          "summarize the core goal. Never pass an empty string — make a best effort even "
                          "if the instruction is short or vague."],
    description: Annotated[str, "Task description (REQUIRED, must be non-empty): ≤80 chars, state the "
                               "outcome to achieve. Never pass an empty string."],
    session_goal: Annotated[
        str,
        "Overall session goal: ≤60 chars. The only optional field. Set it on the first fill or when "
        "the direction changes; leave empty ONLY to keep the existing goal.",
    ] = "",
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Set the current task's title and description (and optionally the session goal). Call exactly once, then stop."""
    if ctx is not None and ctx.task is not None:
        # recognize_intent 现直跑在 root task 上：直接写当前 task 的 title/description。
        # 不置 actor_done（会让 root 主 act 循环误判完成；单发步骤本就不读它）。
        if title:
            ctx.task.title = title
        if description:
            ctx.task.description = description
    if ctx is not None and ctx.session is not None and session_goal:
        ctx.session.goal = session_goal
    return ControlResult(content=f"Metadata updated: title={title!r}")


# ask_user 的嵌套结构超出 extract_schema 的扁平表达能力,手写覆盖（见 control_tool）。
_ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "description": (
                "One or more questions to ask the human. Each is answered independently; "
                "batch related questions in one call instead of asking one at a time."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question or decision you need from the human.",
                    },
                    "options": {
                        "type": "array",
                        "description": (
                            "Predefined choices the human can pick. Omit or leave empty for a "
                            "free-text-only question; a free-text 'Other' is always offered regardless."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "Short choice text shown on the button."},
                                "description": {"type": "string", "description": "What this choice means or its trade-off."},
                                "recommended": {"type": "boolean", "description": "Set on the option you recommend."},
                            },
                            "required": ["label"],
                        },
                    },
                    "multi_select": {
                        "type": "boolean",
                        "description": "Allow the human to select multiple options.",
                        "default": False,
                    },
                },
                "required": ["question"],
            },
        },
    },
    "required": ["questions"],
}


@control_tool(purposes=["act", "observe"], input_schema=_ASK_USER_SCHEMA)
def ask_user(
    questions: Annotated[list, "List of questions to ask the human (see schema)"],
    *,
    ctx: ControlContext = None,
) -> ControlResult:
    """Ask the user one or more questions and wait for their reply — call this whenever you need information, a decision, a clarification, or a choice that only the user can provide; prefer asking over guessing or assuming. Execution pauses until they answer, then resumes from their reply."""
    # 暂停态由 pending HITL 集合推导（spec §7.1），不再由这里直接写 session.status。
    # needs_human 的产出与 park 均在 ControlCapabilityProvider._handle() 完成。
    n = len(questions) if questions else 0
    return ControlResult(
        content=f"Human input requested ({n} question(s))",
        metadata={
            ControlMetaKey.HITL_REQUESTED: True,
            "questions": questions or [],
        },
    )


# ── ControlCapabilityProvider ─────────────────────────────────────────────────


class ControlCapabilityProvider(ToolCapabilityProvider, SessionScopedCapabilityProvider):
    """Exposes task-orchestration control tools to the LLM.

    工具通过 @control_tool 装饰器注册到 _CONTROL_TOOLS；
    _handle() 构建 ControlContext 并注入，工具函数体直接完成实际变动。
    """

    name = PROVIDER_NAME

    def __init__(self) -> None:
        self._sessions: dict[str, tuple["TaskManager", "Session"]] = {}

    def register_session(
        self,
        session_id: str,
        task_manager: "TaskManager",
        session: "Session",
    ) -> None:
        """注册 session 的 TaskManager + Session，供工具函数体使用。"""
        self._sessions[session_id] = (task_manager, session)

    def deregister_session(self, session_id: str) -> None:
        """会话被**逐出内存**时注销（`forget_session` / `purge_session`），防止内存泄漏。

        不是「会话跑完」时（2026-09-08 生命周期改造前是那样）。重新登记由
        `_register_and_drain` 承担：逐出会连 TaskManager 一起摘掉，下一条执行入口重建
        TM 时那里会把本 provider 一并接回来。
        """
        self._sessions.pop(session_id, None)

    async def list(self, ctx: ProviderContext) -> list[ToolCapability]:
        return [cap for cap, _ in _CONTROL_TOOLS.values()]

    def invoke(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        return self._handle(capability_id, arguments, ctx)

    async def _handle(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        name = capability_id.split(":")[-1]
        entry = _CONTROL_TOOLS.get(name)
        if entry is None:
            yield CapabilityEvent(
                kind="error",
                payload={
                    "code": "UNKNOWN_CONTROL",
                    "message": f"Unknown control capability: {capability_id}",
                },
            )
            return

        _, fn = entry
        tm = None
        try:
            sig = inspect.signature(fn)
            # 过滤掉 LLM arguments 中 schema 未声明的 key
            valid_keys = set(sig.parameters.keys()) - _SKIP
            filtered = {k: v for k, v in arguments.items() if k in valid_keys}

            # 构建并注入 ControlContext
            if "ctx" in sig.parameters:
                tm, session = self._sessions.get(ctx.session_id, (None, None))
                task = tm.get_task(ctx.task_id or "") if tm and ctx.task_id else None
                filtered["ctx"] = ControlContext(
                    session_id=ctx.session_id,
                    task_id=ctx.task_id or "",
                    agent_id=ctx.agent_id or "",
                    task=task,
                    task_manager=tm,
                    session=session,
                    tool_call_id=ctx.extra.get("tool_call_id", ""),
                )

            result: ControlResult = fn(**filtered)
        except Exception as e:
            logger.exception("Control tool %s raised: %s", name, e)
            yield CapabilityEvent(
                kind="error",
                payload={"code": "CONTROL_EXEC_ERROR", "message": str(e)},
            )
            return

        K = ControlMetaKey

        # report_task_outcome 的 review：把命中的子任务正规重排（reopen_chain：
        # head + 其 plan 后续一并入队 + 重写 user_prompt + 发 TASK_REQUEUED）。不在此
        # drain——下一次 drain（当前 run 返回后）会自然调度，避免 run 内重入。
        reopen_map = result.metadata.get(K.REOPEN_TASK_IDS)
        if reopen_map and tm is not None:
            for tid, reason in reopen_map.items():
                await tm.reopen_chain(tid, reason)

        # ask_user：声明「我需要一个人的决定」并立即停——不在此 park、不自己等人。
        # 「答复即结果」：gateway 收到 needs_human 后开等待、拿到答复直接回灌为本次工具结果，
        # 因此本 provider **不需要**实现 HumanResumable（spec §2.3）。
        if result.metadata.get(K.HITL_REQUESTED):
            from ctx_weft.protocols.hitl import HITL_FORM_QUESTION, HitlAsk, ToolResultDelivery
            tool_call_id = (ctx.extra or {}).get("tool_call_id", "")
            yield CapabilityEvent(kind="needs_human", payload={"ask": HitlAsk(
                form=HITL_FORM_QUESTION,
                delivery=ToolResultDelivery(tool_call_id=tool_call_id),
                prompt=result.content,
                fields=list(result.metadata.get("questions") or []),
                subject_id=capability_id,
                reply_as_result=True,
            )})
            return                                     # needs_human 是流的终点

        yield CapabilityEvent(
            kind="result",
            payload={"content": result.content, "metadata": result.metadata},
        )

    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        pass

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(
            name=self.name,
            capability_count=len(_CONTROL_TOOLS),
            supports_streaming=False,
            supports_cancel=False,
            description=self.description,
        )
