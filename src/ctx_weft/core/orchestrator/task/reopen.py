"""reopen 的 prompt 改写：memory 侧内容 + event 侧 jsonable，两条路径逐分支同构。

从 `task_manager.py` 搬出——`reopen_task` 的九十来行里只有约二十行是调度（拿锁、
翻状态、push、emit），其余全是内容拼装，与队列无关。切出来之后这段逻辑可以直接
单测，不必先造一个 TaskManager。

两条路径的同构是本模块的**全部难点**：memory 侧（`build_reopen_prompt` 的
`user_prompt`）逐 section 调 `content_with_suffix`，event 侧
（`append_text_sections`）必须产出逐字节等价的结果——否则崩溃恢复重放出来的 prompt
会和内存里跑的那份分叉。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ctx_weft.core.utils.content import content_with_suffix

if TYPE_CHECKING:
    from ctx_weft.core.models.task import Task
    from ctx_weft.protocols import ContentPart

__all__ = [
    "ReopenPrompt",
    "append_text_sections",
    "build_reopen_prompt",
    "outputs_to_text",
]


@dataclass(frozen=True)
class ReopenPrompt:
    """一次 reopen 的四份内容。调用方（`TaskManager.reopen_task`）照原样写回 Task。

    `original_*` 两个字段是**首次 reopen 时拍下的快照**，之后每次 reopen 都以它为
    base——重复 reopen 因此不会叠加修订说明。调用方须把它们写回 task，否则下一次
    reopen 会把已改写过的 prompt 当成 base。
    """

    user_prompt: "str | list[ContentPart]"
    user_prompt_event_jsonable: "str | list[dict] | None"
    original_user_prompt: "str | list[ContentPart]"
    original_user_prompt_event_jsonable: "str | list[dict] | None"


def build_reopen_prompt(
    task: "Task", reason: str = "", upstream: "tuple[str, str] | None" = None,
) -> ReopenPrompt:
    """按 base + 上一轮产出 + 修订说明拼出重跑用的 prompt（memory 侧 + event 侧）。

    改写始终以 `original_user_prompt` 为 base（首次 reopen 时快照下来），所以重复
    reopen 不会把「上轮产出 / 修订提示」反复累加进 prompt。

    `upstream=(head_title, head_reason)` 标记级联重开的后继：它拿到的是「上游任务已
    修订，按更新后的结果重做」的指令（上游的新结果经 blackboard 订阅送达），而不是
    直接的修订说明。
    """
    # base = 首次执行的原始 prompt（首次 reopen 时快照下来）
    if task.original_user_prompt is None:
        # 原样保留（含多模态）：这是 reopen 的 base，拍扁会让重开后图片永久消失。
        original: "str | list[ContentPart]" = task.user_prompt or ""
        original_jsonable = task.user_prompt_event_jsonable
    else:
        original = task.original_user_prompt
        original_jsonable = task.original_user_prompt_event_jsonable

    prev_output = outputs_to_text(task.outputs) or (task.process_report or "")
    sections: list[str] = []
    if prev_output:
        sections.append(f"## Previous attempt (rejected)\n{prev_output}")
    if upstream is not None:
        head_ref, head_reason = upstream
        sections.append(
            f"## Upstream task revised\n"
            f"Predecessor {head_ref} was reopened (reason: {head_reason}). "
            f"Its updated result appears in the conversation above. "
            f"Redo this task based on the updated result."
        )
    elif reason:
        sections.append(f"## Revision required\n{reason}")

    # base 可能是多模态（list[ContentPart]），不能进 "\n\n".join()。
    # 有 base 时从 base 起逐段 content_with_suffix；无 base 时退回纯文本 join。
    # 两条路径对 str base 的产物与改造前**逐字节相同**。
    if original:
        # 无 section 时 new_prompt 必须与 base 是不同对象：list base 若直接复用同一
        # 引用，task.user_prompt 与 task.original_user_prompt 会别名同一份 parts，
        # 日后任一方被就地修改都会污染另一方（str 不可变故无此风险）。
        new_prompt: "str | list[ContentPart]" = (
            list(original) if isinstance(original, list) else original
        )
        for sec in sections:
            new_prompt = content_with_suffix(new_prompt, f"\n\n{sec}")
    else:
        new_prompt = "\n\n".join(sections) if sections else original

    # 兜底：event jsonable 没被填上（历史上 `_restore_task_prompts` 跳过纯文本、
    # `run_single_task` 丢弃它，都出过这个洞——终审 C1），而 base 又是非空 str。
    # 纯文本的事件形态就是它自己，直接补上；决不能让「字段没填」被
    # `append_text_sections` 读成「base 为空」，那会把用户的原始指令从 TASK_REQUEUED
    # 里抹掉、并在下一次重放时永久生效。
    # 只兜 str：list base 的事件形态含 event ref，core 无从凭空重建（重建就意味着拿
    # memory ref 冒充 event ref，正是两个命名空间不得相通的红线）。
    if original_jsonable is None and isinstance(original, str) and original:
        original_jsonable = original

    return ReopenPrompt(
        user_prompt=new_prompt,
        user_prompt_event_jsonable=append_text_sections(original_jsonable, sections),
        original_user_prompt=original,
        original_user_prompt_event_jsonable=original_jsonable,
    )


def append_text_sections(
    jsonable: "str | list[dict] | None", sections: "list[str]",
) -> "str | list[dict] | None":
    """把 reopen 的文本 section 追加到事件侧 jsonable 尾部，与 `build_reopen_prompt`
    对 `new_prompt`（memory 侧）的构造逐分支同构：

    - base 为空（``None`` / ``""`` / ``[]``）→ 与 memory 侧 ``else`` 分支
      （``"\\n\\n".join(sections)``）一致：产出**不带前导空行**的 str，类型也收敛
      为 str（哪怕 base 原本是空 list）——`if original:` 对三者一视同仁地判假，
      event 侧必须跟着一视同仁。
    - base 非空 → 与 memory 侧逐 section 调 `content_with_suffix` 的**等效**结果
      一致：suffix 逐 section 以 ``"\\n\\n"`` 为前缀拼接（迭代调用 `content_with_suffix`
      与一次性拼接完整 suffix 对同一批纯文本 section 等价——合并只发生在字符串层面，
      不受分几次调用影响）；str base 直接接在尾部；list base 若尾部已是 text part
      则原地合并进那个 part（同 `content_with_suffix` 对连续 text part 的合并语义，
      否则事件侧会比 memory 侧多出一个独立 text part、两边形状分歧），否则新增一个
      text part。
    """
    if not sections:
        return jsonable
    if not jsonable:  # None / "" / [] —— 与 memory 侧 `if original:` 判据一致
        return "\n\n".join(sections)
    suffix = "".join(f"\n\n{sec}" for sec in sections)
    if isinstance(jsonable, str):
        return jsonable + suffix
    if isinstance(jsonable[-1], dict) and jsonable[-1].get("type") == "text":
        tail = jsonable[-1]
        merged = {**tail, "text": tail.get("text", "") + suffix}
        return [*jsonable[:-1], merged]
    return [*jsonable, {"type": "text", "text": suffix}]


def outputs_to_text(outputs: Any) -> str:
    """把 task.outputs（str 或 ContentPart 列表）渲染成纯文本，供 reopen prompt 复用。"""
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, list):
        return "\n".join(
            p.get("text", "")
            for p in outputs
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""
