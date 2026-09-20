"""跨层的 prompt 标题契约字符串。

`SUBTASKS_HEADING` 只有一个，但它必须住在这里而不是任何一个包里：它的两个消费者是
`assembler/composer.py`（渲染那一段）与 `capabilities/control_tools.py`
（`report_task_outcome` 的 `next_step_hint` 描述里指路「按这一段列出的标题+id 指名」），
而 `assembler -> capabilities` 已经存在（composer 引控制工具的限定名），放进 assembler
会让 capabilities 反向引它、成环。

它的兄弟 `PROGRESS_SO_FAR_HEADING` 不在这里——那个只有 `assembler/sources/_history.py`
一个消费者（同时也是唯一的渲染者），已内联到那边。
"""

from __future__ import annotations

__all__ = ["SUBTASKS_HEADING"]

# observe prompt 里「自己派生的子任务清单」段的标题前缀。跨层字符串契约，勿散写字面量。
#
# 2026-09-19 之前它叫 SUBTASKS_REVIEW_HEADING，配套的是 `report_task_outcome` 的
# `task_reviews` 参数（可 confirm/reopen 子任务）。那套连同 reopen 一并删除——这一段
# 现在纯是**信息**：observer 据它指名哪个子任务的产出不合格，写进 next_step_hint，
# 由下一轮 actor 自己决定重派还是自己做。
SUBTASKS_HEADING = "## Your sub-tasks"
