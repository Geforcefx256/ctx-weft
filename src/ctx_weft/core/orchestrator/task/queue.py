"""TaskQueue：LIFO 调度 + blocked DAG 支持。

设计文档 §4（Phase 4 §4.1）；spec: task-handoff 扩展双完成集。

依赖条件两集（spec: task-handoff）：
- ``blocked_by``   —— on_any 依赖：前序到达**任一终态**即释放（存量语义）；
- ``blocked_success`` —— on_success 依赖：前序 **FINISHED** 才释放。
对应两个完成集：``_completed``（任一终态）与 ``_succeeded``（仅 FINISHED）。
``mark_failed`` / 取消走「终态但非成功」——只进 ``_completed``，on_success 后继不释放。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class QueueEntry:
    task_id: str
    session_id: str
    priority: int = 5
    blocked_by: set[str] = field(default_factory=set)
    blocked_success: set[str] = field(default_factory=set)


def split_deps(
    dag_deps: "list[str] | None",
    dep_conditions: "dict[str, str] | None",
) -> tuple[set[str], set[str]]:
    """按条件拆依赖（spec: task-handoff）：返回 (on_any 集, on_success 集)。

    条件缺省（None / 未声明）按 ``any`` 解释——存量事件回放的历史保真语义；
    新派发由 ``push_task`` 写入时物化，永远显式。
    """
    any_deps: set[str] = set()
    success_deps: set[str] = set()
    conds = dep_conditions or {}
    for dep in dag_deps or []:
        if conds.get(dep) == "success":
            success_deps.add(dep)
        else:
            any_deps.add(dep)
    return any_deps, success_deps


class TaskQueue:
    """In-process LIFO queue with DAG dependency support.

    Rules:
    - Tasks are scheduled LIFO (stack) when not blocked.
    - A task is blocked if any of its on_any deps are non-terminal, or any of its
      on_success deps are not FINISHED.
    - When a task finishes successfully, its dependents (both kinds) are unblocked;
      when it fails or is canceled, only on_any dependents are unblocked.
    - pop() returns the most recently pushed unblocked task.
    """

    def __init__(self) -> None:
        self._entries: list[QueueEntry] = []
        self._completed: set[str] = set()   # 任一终态（含 FAILED/CANCELED）
        self._succeeded: set[str] = set()   # 仅 FINISHED
        self._running: set[str] = set()

    def push(self, entry: QueueEntry) -> None:
        # Remove already-satisfied dependencies at push time
        entry.blocked_by -= self._completed
        entry.blocked_success -= self._succeeded
        self._entries.append(entry)
        logger.debug(
            "TaskQueue.push: %s (blocked_by=%s, blocked_success=%s)",
            entry.task_id, entry.blocked_by, entry.blocked_success,
        )

    def pop(self, skip: Callable[[QueueEntry], bool] | None = None) -> QueueEntry | None:
        """Return the topmost non-blocked, non-running task (LIFO).

        ``skip``: optional predicate. Entries for which it returns True are left in
        the queue (not popped) — used for same-agent no-concurrency: skip a task whose
        target agent is currently busy. Such entries get picked up on a later pop() once
        the agent frees (a running task completing triggers another drain).
        """
        for i in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[i]
            # Refresh: remove deps that have since completed
            entry.blocked_by -= self._completed
            entry.blocked_success -= self._succeeded
            if entry.blocked_by or entry.blocked_success or entry.task_id in self._running:
                continue
            if skip is not None and skip(entry):
                continue
            self._entries.pop(i)
            self._running.add(entry.task_id)
            return entry
        return None

    def mark_running(self, task_id: str) -> None:
        self._running.add(task_id)

    def unmark_running(self, task_id: str) -> None:
        """Remove from running set without marking as complete/failed. Used for retries."""
        self._running.discard(task_id)

    def seed_completed(self, task_ids: "Iterable[str]") -> None:
        """恢复期批量装填「已终态」集合。

        `TaskManager.restore` 此前直接写 `self._completed`——那是穿透私有。与
        `mark_complete` 的区别：这里不刷新已排队条目的 `blocked_by`，因为 restore 的
        调用顺序是**先装填、后 push**，而 `push` 首行就会摘掉已完成依赖，无需重复扫描。
        """
        self._completed.update(task_ids)

    def seed_succeeded(self, task_ids: "Iterable[str]") -> None:
        """恢复期批量装填「已成功」集合（仅 FINISHED 的任务 id，spec: task-handoff）。"""
        self._succeeded.update(task_ids)

    def unmark_completed(self, task_id: str) -> None:
        """Remove from completed sets so a reopened task can be scheduled again."""
        self._completed.discard(task_id)
        self._succeeded.discard(task_id)

    def mark_complete(self, task_id: str) -> None:
        self._running.discard(task_id)
        self._completed.add(task_id)
        self._succeeded.add(task_id)
        # Refresh all blocked entries
        for entry in self._entries:
            entry.blocked_by.discard(task_id)
            entry.blocked_success.discard(task_id)
        logger.debug("TaskQueue.mark_complete: %s, %d pending", task_id, len(self._entries))

    def mark_failed(self, task_id: str) -> None:
        """终态但非成功（FAILED，以及依赖阻塞/用户取消的 CANCELED）。

        只进 ``_completed``：on_any 后继照常解锁，on_success 后继保持阻塞
        （它们的善后由 TaskManager 的永久阻塞扫描处置，spec: task-handoff）。
        """
        self._running.discard(task_id)
        self._completed.add(task_id)  # any-terminal: unblocks on_any dependents only

    def cancel(self, task_id: str) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.task_id == task_id:
                self._entries.pop(i)
                return True
        return False

    def drain_pending(self) -> list[str]:
        """Remove all queued (pending) entries; return their task ids.

        Does not touch ``_running`` / ``_completed`` — only clears what hasn't started.
        """
        ids = [e.task_id for e in self._entries]
        self._entries.clear()
        return ids

    def pending_count(self) -> int:
        return len(self._entries)

    def has_pending(self) -> bool:
        return bool(self._entries)

    def all_blocked(self) -> bool:
        """True if every pending task is blocked (deadlock signal)."""
        if not self._entries:
            return False
        return all(bool(e.blocked_by or e.blocked_success) for e in self._entries)

    def peek_all(self) -> list[QueueEntry]:
        return list(self._entries)
