"""Act 态势感知文本（两件套）：turn 尾部 guidance + 续跑衔接 cue。

本模块是 act 态势感知文本的唯一 authorship，产两段互补的文本，
均由 PrepareStep 构建、经 request.extra 传入装配、仅发送不入 memory：

- **build_act_guidance** → extra["act_guidance"] → GuidanceSource 产
  kind="guidance" block → composer 恒拼末条 user 最尾部（recency 最强；
  动态内容居尾不打穿 prompt cache 前缀）。
- **build_resume_cue** → extra["act_resume_cue"] → composer 在「历史以
  assistant/tool 收尾」时垫一条续跑衔接 user 回合作为该 turn 的开场
  （何时垫由 composer 按对话形状判定；文本在此与 guidance 同源维护）。
  guidance 承载"已完成子任务"的具体清单后，cue 只留通用续跑框架 +
  （确有已完成子任务时）一句指向清单的提示，不再重复展开。

guidance 内容分两类（composer 拼在末条 user 尾部；## Capabilities 全文随
当前 task 回合落在 cache 前缀内，其后至多再补一行清单指针）：
- **动态段**（只能运行时生成）：当前任务锚定行（每个 act 回合都有，长对话里
  ## Current Task 框远在历史深处时的就近锚）、session 非终态任务树（▶ 定位
  当前 task）、当前 task 已完成子任务清单（防重做/重派——它们已从任务树消失，
  每条附结果摘要 outputs→task_summary，完整结果仍摊在对话上文里，摘要防上文被
  压缩/淹没时丢失）、上一轮复核的一次性转向（task.next_step_hint = observer 的
  next_step_hint + success-without-outputs 护栏文案）。
  转向段走 guidance 而非并进 act_recap，是刻意的生命周期分离：act_recap 会进永久
  记忆（段摘要 / finish 对），一次性指令混进去会在任务完成后仍留在历史里；guidance
  只发不入 memory，随任务终结自然消失，且 priority 1 比段摘要（动态档 4）更难被预算裁掉。
- **静态段**（指针级，刻意压缩）：finish 收尾、无关新请求 finish+delegate 双发、
  ask_user 三条只留一句提醒。完整协议已有三处承载——SOUL、工具 description、
  以及机械兜底（observer 的 success-without-outputs→retry 护栏、interactive
  纯文本自动 park），此处不再复读全文。
"""

from __future__ import annotations

from datetime import datetime, timezone

from ctx_weft.core.capabilities.control_tools import (
    ASK_USER_NAME,
    DELEGATE_TASK_NAME,
    FINISH_TASK_NAME,
)
from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.core.utils.task_ref import task_ref_parts
from ctx_weft.core.utils.clock import as_utc

_TERMINAL_STATUSES = TERMINAL_TASK_STATUSES  # 词表见 core.domain.status
_TASK_LABEL_MAX = 80
_SUBTASK_RESULT_MAX = 150


def _task_label(t) -> str:
    """任务树/已完成清单里一行的标签：**标题与 id 恒同时出现**（spec: task-handoff）。

    这两处是模型建立「整个计划长什么样」的地方，而它随后要用 `task_id` 去
    `report_task_outcome` 操作其中的子任务——只印标题就是让它自己去猜映射。
    无 title 时退回开启该 task 的 prompt 首行（截断）当描述性名字，id 照印。
    """
    title = (t.title or "").strip()
    if not title:
        prompt = content_to_text(getattr(t, "user_prompt", None) or "").strip()
        if prompt:
            first = prompt.splitlines()[0].strip()
            title = first[:_TASK_LABEL_MAX] + "…" if len(first) > _TASK_LABEL_MAX else first
    return task_ref_parts(getattr(t, "id", "") or "", title)


def _nonterminal_tasks(task_manager) -> list:
    """session 内非终态 task（`_TERMINAL_STATUSES` 的补集），无 task_manager 时空表。"""
    if task_manager is None:
        return []
    return [t for t in task_manager.all_tasks() if t.status not in _TERMINAL_STATUSES]


def _has_other_open_tasks(task, task_manager) -> bool:
    """当前 task 之外是否还有其它非终态 task（决定是否提示「别自己做其它任务」）。"""
    return any(t.id != task.id for t in _nonterminal_tasks(task_manager))


def _finished_subtasks(task, task_manager) -> list:
    """当前 task 的 FINISHED 直接子任务，created_at 升序。

    只列 FINISHED：FAILED/CANCELED 的子任务可能需要重派/另行处理，不适合标成
    「已完成勿重做」。挂起恢复时任务树只剩非终态节点，完成的子任务从树里消失——
    这份清单把它们显式钉出来，防止 parent 把子任务的活自己再做一遍或重复派发。
    """
    if task_manager is None:
        return []
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    done = [t for t in task_manager.all_tasks()
            if t.parent_task_id == task.id and t.status == "FINISHED"]
    # created_at 归一为 aware(UTC)：崩溃恢复重建的 task 可能带 naive 时间，与活任务的
    # aware 时间混排会抛 naive/aware 比较错（边界已在 converters 补齐，此处防御兜底）。
    done.sort(key=lambda t: as_utc(t.created_at) if t.created_at else epoch)
    return done


def _subtask_result_snippet(t) -> str:
    """完成子任务的结果摘要：outputs（成果物）优先，回退 task_summary；折叠单行、截断到 _SUBTASK_RESULT_MAX。

    parent 「build on their results」需要的是子任务的实际产出——outputs 是 finish_task 提交的
    deliverable（str 或 ContentPart 列表，content_to_text 归一）；outputs 空时回退 observer 的
    task_summary（整段综合总结）。多行折叠成单行，避免多行结果撑开每回合尾部的 guidance。
    两者皆空返回 ""（回退到只列 title，保持向后兼容）。
    """
    outputs = getattr(t, "outputs", None)
    text = outputs if isinstance(outputs, str) else (content_to_text(outputs) if outputs else "")
    if not (text or "").strip():
        text = getattr(t, "task_summary", None) or ""
    text = " ".join((text or "").split())  # 折叠所有空白（含换行）成单空格
    if not text:
        return ""
    return text[:_SUBTASK_RESULT_MAX] + "…" if len(text) > _SUBTASK_RESULT_MAX else text


def _session_task_tree(task, task_manager) -> str:
    """把 session 内**非终态** task 渲染成缩进任务树，当前 task 以 ``▶`` 标注。

    - 仅列非终态（`_TERMINAL_STATUSES` 的补集）——终态不刷屏、只留「还没做完的活」。
    - 按 parent_task_id 建树；父节点被过滤掉（终态/缺失）的非终态 task 提升到 root 层，避免孤儿丢失。
      roots 及同层子节点按 created_at 升序。
    - 只要有 ≥1 个非终态 task 就出树（含 root 独自 act 时只列它自己一行）——root 派生子任务后即转
      SUSPENDED 停止 act，故「单节点」唯一对应 root 独自 act 的情形，让它也能看到自己的定位。
    """
    tasks = _nonterminal_tasks(task_manager)
    if not tasks:
        return ""
    ids = {t.id for t in tasks}
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    children: dict[str | None, list] = {}
    for t in tasks:
        parent = t.parent_task_id if t.parent_task_id in ids else None
        children.setdefault(parent, []).append(t)
    for lst in children.values():
        lst.sort(key=lambda t: as_utc(t.created_at) if t.created_at else epoch)

    lines: list[str] = []

    def _walk(node_id: str | None, depth: int) -> None:
        for t in children.get(node_id, []):
            indent = "  " * depth
            marker = "▶ " if t.id == task.id else ""
            lines.append(f"{indent}- [{t.status}] {marker}{_task_label(t)}")
            _walk(t.id, depth + 1)

    _walk(None, 0)
    return "\n".join(lines)


def build_resume_cue(task, task_manager) -> str:
    """续跑衔接 cue：历史以 assistant/tool 收尾时，composer 垫的 user 回合开场白。

    通用框架（盘点已完成、只做剩余）+ 确有已完成子任务时一句指向 guidance
    清单的提示（具体条目由 build_act_guidance 的 ALREADY COMPLETED 段承载，
    此处不重复展开）。
    """
    ref = _task_label(task) if getattr(task, "id", None) else ""
    subject = f"the task: {ref}" if ref else "the task above"
    cue = (
        f"You are still working on {subject}, resuming from the state recorded above. "
        "Review the conversation for what has already been done — do not redo it — and "
        "continue with only the remaining work."
    )
    if _finished_subtasks(task, task_manager):
        cue += (
            " Parts of it have ALREADY been delegated to sub-tasks that are now complete — "
            "they are listed with their results in the situational notes below. Never execute "
            "that work a second time: instead of redoing or re-delegating it, build on those "
            "results and judge what, if anything, still remains."
        )
    return cue


def build_act_guidance(task, task_manager) -> str:
    """构造 act 态势 guidance 文本（恒非空；调用方按 purpose/settings 门控）。

    段序 = 当前任务锚定 → plan 全景（任务树）→ 已完成子任务清单 → 静态指针。
    锚定行每个 act 回合都有：fresh 回合虽与同消息前部的 ``## Current Task`` 框
    略有重复，但长对话/续跑/interactive 追问回合里该框远在历史深处，此处是
    生成点附近唯一的任务锚。description 不重复（由 Current Task 框承载）。
    session 仅剩当前 task 一个非终态节点时，任务树整段不出现。
    """
    parts: list[str] = ["---"]

    ref = _task_label(task) if getattr(task, "id", None) else ""
    parts.append(
        f"Current task: {ref}" if ref
        else "Current task: (as framed in the conversation above)"
    )
    parts.append("")

    tree = _session_task_tree(task, task_manager)
    has_other_tasks = _has_other_open_tasks(task, task_manager)
    if tree:
        header = (
            "## The overall plan (▶ = your current task; the rest are handled separately "
            "— do NOT do them yourself):"
            if has_other_tasks
            else "## The overall plan (▶ = your current task):"
        )
        parts.append(header)
        parts.append(tree)
        parts.append("")

    done = _finished_subtasks(task, task_manager)
    if done:
        parts.append(
            "## Your current task has ALREADY been delegated to the sub-tasks below, and they "
            "are ALREADY COMPLETED — the digest under each is what it produced (full results "
            "are in the conversation above). That work is done and must never be executed a "
            "second time: Do NOT redo their work yourself and do NOT delegate them again. Your "
            "next step is to judge whether any work still remains for the task goal beyond "
            "these results — if it does, do only that part, building on what they produced:"
        )
        for t in done:
            parts.append(f"- [FINISHED] {_task_label(t)}")
            snippet = _subtask_result_snippet(t)
            if snippet:
                parts.append(f"    → {snippet}")
        parts.append("")

    hint = (getattr(task, "next_step_hint", None) or "").strip()
    if hint:
        parts.append("## Note from the review of your previous attempt:")
        parts.append(hint)
        parts.append("")

    # 静态提醒（指针级）：完整语义在工具 description / SOUL / observer 护栏，这里只钉最易违反的三条。
    # finish 提醒判据在前——此段处于每回合尾部 recency 最强位，措辞若以收尾为默认会放大过早 finish。
    finish_core = (
        f"Call `{FINISH_TASK_NAME}` ONLY once the task goal is fully achieved — if work "
        "remains, keep working instead of finishing. To finish, write the final reply to the "
        "user as your normal message text and call it in that same turn; the message text is "
        "the reply and the deliverable, not the tool arguments (details in the tool description)."
    )
    if task.interaction_mode == "interactive":
        parts.append(
            finish_core
            + " In this interactive task a plain-text reply keeps the task open and hands "
            "the floor to the user — the right move mid-conversation; finish only when the "
            "whole request is served."
        )
    elif has_other_tasks:
        parts.append(finish_core + " Do not start the other tasks yourself.")
    else:
        parts.append(finish_core)
    parts.append(
        f"If the user's latest message is an unrelated NEW request, emit `{FINISH_TASK_NAME}` "
        f"AND `{DELEGATE_TASK_NAME}` together in ONE response — finishing alone would lose "
        "the new request."
    )
    parts.append(
        "Need information or a decision only the user can provide? "
        f"Call `{ASK_USER_NAME}` instead of guessing."
    )
    return "\n".join(parts)
