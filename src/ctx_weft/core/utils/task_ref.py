"""任务在模型可见文本里的统一称呼：**标题与 id 恒同时出现**。

为什么要统一：模型用**标题**建立心智模型（act 每回合的 `## The overall plan` 任务树、
已完成子任务清单、派发对回执、finish 对的归属标记，历来都只印标题），却被要求用 **id**
执行操作（`report_task_outcome` 的 `task_reviews` 只认 `task_id`，非字符串或缺失一律
拒绝）。两个称呼分居两张脸，模型就得在中间做一次**没有凭据的映射**——而它手上唯一
同时给出两者的地方是 observe 的 `## Your sub-tasks`，那已经是操作时刻，太晚了。

标题还不是稳定标识：同名子任务在叙述面完全无法区分（`[task: {title}]` 这个前缀的
用途恰恰是「消歧嵌套的同 agent 收尾」，而它在标题相同时失效），改名则让旧引用落空。
id 稳定但不可读。两个一起印，各自补上对方的短板，代价是每处多约 10 个 token。

规范形式 —— **`'标题' (tsk_…)`**，全仓一种形状，让模型只需学一次：

    'Build the parser' (tsk_01M2GC4QXWJFYJX1EC5TBYZVGF)

无标题时退化为裸 id（印一个空引号只是噪声）。鸭子类型取字段，不 import `Task`，
因此本模块可被 core 任意层引用而不成环（与 `headings.py` 同一考量）。
"""

from __future__ import annotations

__all__ = ["task_ref", "task_ref_parts"]


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
