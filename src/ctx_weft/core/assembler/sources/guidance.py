"""Collect raw Act guidance state; Jinja owns all visible wording."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES
from ctx_weft.core.utils.clock import as_utc
from ctx_weft.protocols.content import content_to_text
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.core.utils.task_ref import task_label

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextBlock, ContextRequest


_SUBTASK_RESULT_MAX = 150


def _nonterminal_tasks(task_manager) -> list:
    if task_manager is None:
        return []
    return [
        task for task in task_manager.all_tasks()
        if task.status not in TERMINAL_TASK_STATUSES
    ]


def _finished_subtasks(task, task_manager) -> list:
    if task_manager is None:
        return []
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    done = [
        child for child in task_manager.all_tasks()
        if child.parent_task_id == task.id and child.status == "FINISHED"
    ]
    done.sort(key=lambda child: as_utc(child.created_at) if child.created_at else epoch)
    return done


def _subtask_result_snippet(task) -> str:
    outputs = getattr(task, "outputs", None)
    text = outputs if isinstance(outputs, str) else (
        content_to_text(outputs) if outputs else "")
    if not (text or "").strip():
        text = getattr(task, "task_summary", None) or ""
    text = " ".join((text or "").split())
    if not text:
        return ""
    return text[:_SUBTASK_RESULT_MAX] + "…" if len(text) > _SUBTASK_RESULT_MAX else text


def _session_task_rows(task, task_manager) -> list[dict]:
    tasks = _nonterminal_tasks(task_manager)
    if not tasks:
        return []
    ids = {item.id for item in tasks}
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    children: dict[str | None, list] = {}
    for item in tasks:
        parent = item.parent_task_id if item.parent_task_id in ids else None
        children.setdefault(parent, []).append(item)
    for siblings in children.values():
        siblings.sort(key=lambda item: as_utc(item.created_at) if item.created_at else epoch)

    rows: list[dict] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for item in children.get(parent_id, []):
            rows.append({
                "depth": depth,
                "status": item.status,
                "current": item.id == task.id,
                "ref": task_label(item),
            })
            walk(item.id, depth + 1)

    walk(None, 0)
    return rows


def _collect_state(task, task_manager) -> dict:
    ref = task_label(task) if getattr(task, "id", None) else ""
    nonterminal = _nonterminal_tasks(task_manager)
    has_other_tasks = any(item.id != task.id for item in nonterminal)
    completed_tasks = [
        {
            "ref": task_label(child),
            "result": _subtask_result_snippet(child),
        }
        for child in _finished_subtasks(task, task_manager)
    ]
    hint = (getattr(task, "next_step_hint", None) or "").strip()
    return {
        "guidance": {
            "task_ref": ref,
            "plan_tasks": _session_task_rows(task, task_manager),
            "has_other_tasks": has_other_tasks,
            "completed_tasks": completed_tasks,
            "review_note": hint,
            "attended": not task.unattended,
            "unattended": task.unattended,
            "unattended_others": task.unattended and has_other_tasks,
        },
        "resume": {
            "ref": ref,
            "completed_tasks": bool(completed_tasks),
        },
    }


class GuidanceSource:
    """Read task state directly and emit one detached guidance snapshot."""

    name = "guidance"

    async def fetch(
        self,
        request: "ContextRequest",
        deps: "AssemblerDeps",
    ) -> AsyncIterator["ContextBlock"]:
        from ctx_weft.core.assembler.assembler import ContextBlock

        if request.purpose != "act":
            return
        value = _collect_state(request.task, getattr(deps, "task_manager", None))
        yield ContextBlock(
            id=generate_id("blk"),
            source="guidance",
            kind="guidance",
            value=value,
        )
