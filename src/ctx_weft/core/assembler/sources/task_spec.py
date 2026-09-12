"""TaskSpecSource：当前 task 的 spec（title/description/user_prompt）。

把 task spec 作为结构化 metadata 放进一个 priority=0 的 task_spec block（不可裁）。
Composer 不把它当独立消息渲染，而是读 metadata 去装饰「当前 task」那条 user 回合
（in-memory 就地装饰 / fresh 实时构建），见 DefaultComposer._task_spec_fields。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ctx_weft.core.assembler.priority import slot_priority
from ctx_weft.core.utils.content import content_to_text
from ctx_weft.core.utils.ids import generate_id

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextBlock, ContextRequest


class TaskSpecSource:
    """产出 task_spec block（priority=0 不可裁；字段在 metadata，content 仅作快照）。"""

    name = "task_spec"

    async def fetch(
        self,
        request: "ContextRequest",
        deps: "AssemblerDeps",
    ) -> AsyncIterator["ContextBlock"]:
        from ctx_weft.core.assembler.assembler import ContextBlock

        task = request.task
        title = task.title or ""
        description = task.description or ""
        user_prompt = (
            task.user_prompt
            if isinstance(task.user_prompt, str)
            else content_to_text(task.user_prompt) if task.user_prompt
            else ""
        )
        # spec: task-handoff——显式输入走 metadata（composer 两条渲染路径都读它）；
        # content 仍只是快照，遵守本模块「block 是元数据载体」的契约。
        inputs = getattr(task, "inputs", None)
        # content 仅作可读快照（debug / token 估算）；composer 实际读 metadata 字段。
        parts = [p for p in (title, description, user_prompt) if p]
        if inputs:
            import json as _json
            parts.append("## Inputs\n" + _json.dumps(inputs, ensure_ascii=False, indent=2))
        content = "\n".join(parts) if parts else "[No task spec]"

        yield ContextBlock(
            id=generate_id("blk"),
            source="task_spec",
            kind="task_spec",
            target="messages",
            content=content,
            priority=slot_priority("task_spec"),  # 不可裁
            token_estimate=request.token_counter(content),
            metadata={
                "task_id": task.id,
                "title": title,
                "description": description,
                "user_prompt": user_prompt,
                "inputs": inputs,
            },
        )
