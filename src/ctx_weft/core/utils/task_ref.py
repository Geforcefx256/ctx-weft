"""任务在模型可见文本里的统一称呼：**标题与 id 恒同时出现**。

为什么要统一：模型用**标题**建立心智模型（act 每回合的 `## The overall plan` 任务树、
已完成子任务清单、派发对回执、finish 对的归属标记，历来都只印标题），却被要求用 **id**
**指名**一个子任务（observer 在 `next_step_hint` 里点名哪个子任务的产出不合格，下一轮
actor 据此决定重派还是自己做）。两个称呼分居两张脸，模型就得在中间做一次**没有凭据的
映射**——而它手上唯一同时给出两者的地方是 observe 的 `## Your sub-tasks`，那已经是要
指名的时刻，太晚了。

标题还不是稳定标识：同名子任务在叙述面完全无法区分（`[task: {title}]` 这个前缀的
用途恰恰是「消歧嵌套的同 agent 收尾」，而它在标题相同时失效），改名则让旧引用落空。
id 稳定但不可读。两个一起印，各自补上对方的短板，代价是每处多约 10 个 token。

规范形式 —— **`'标题' (tsk_…)`**，全仓一种形状，让模型只需学一次：

    'Build the parser' (tsk_01M2GC4QXWJFYJX1EC5TBYZVGF)

无标题时退化为裸 id（印一个空引号只是噪声）。鸭子类型取字段，不 import `Task`，
因此本模块可被 core 任意层引用而不成环（与 `headings.py` 同一考量）。
"""

from __future__ import annotations

__all__ = ["task_ref", "task_label", "task_label_parts", "task_ref_parts"]

#: `task_label` 的标题回退长度上限（取 prompt 首行时截断）。
TASK_LABEL_MAX = 80


def task_ref(task) -> str:
    """任务的规范称呼：``'标题' (id)``；无标题则裸 id。

    ``task`` 只需有 ``id`` 与 ``title`` 两个属性（Task / TaskView / 任何投影皆可）。
    """
    return task_ref_parts(getattr(task, "id", "") or "", getattr(task, "title", "") or "")


def task_ref_parts(task_id: str, title: str) -> str:
    """同 `task_ref`，但直接收 id 与标题——供手上只有两个字符串的调用点使用。

    两者皆空 → 返回 **空串**，让调用点自己决定兜底措辞（例如 finish 对的「无标题」
    变体）。不在这里编一个 `(unidentified task)` 之类的占位——那既没有信息量，又会
    把调用点原有的兜底分支变成死代码。
    """
    title = (title or "").strip()
    task_id = (task_id or "").strip()
    if not task_id:
        return f"{title!r}" if title else ""
    return f"{title!r} ({task_id})" if title else task_id


def task_label(task) -> str:
    """任务的规范称呼，**标题空时退回开启它的 prompt 首行**（截断）。

    与 `task_ref` 的分野只有一条：这个会回退。用在「生成点附近的任务锚」上——act 每回合
    guidance 的锚定行、observe 的 cue 锚定行、任务树/已完成清单的每一行。那些位置必须
    印出一个人能读的名字，而 root task 在 `recognize_intent` 填完标题之前 `title` 恒空
    （`start_session` 建它时就是空的），只印裸 id 等于什么都没说。

    从 `act_guidance._task_label` 移来（2026-09-28）：observe 的 cue 也要这道锚，而
    composer 在 assembler 层、不该反向 import loop 层。放这里两边共用一份，不写第二套回退。
    """
    return task_label_parts(getattr(task, "id", "") or "", getattr(task, "title", "") or "",
                            getattr(task, "user_prompt", None))


def task_label_parts(task_id: str, title: str, user_prompt=None) -> str:
    """Render the same label from persisted prompt fields without a Task object."""
    title = (title or "").strip()
    if not title:
        from ctx_weft.protocols.content import content_to_text

        prompt = content_to_text(user_prompt or "").strip()
        if prompt:
            first = prompt.splitlines()[0].strip()
            title = first[:TASK_LABEL_MAX] + "…" if len(first) > TASK_LABEL_MAX else first
    return task_ref_parts(task_id, title)
