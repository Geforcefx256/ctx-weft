"""TaskQueue：LIFO 调度 + blocked DAG 支持。

设计文档 §4（Phase 4 §4.1）；spec: task-handoff 改写解锁判据。

**依赖只有一种语义：前序 FINISHED 才放行**（spec: task-handoff）。依赖边只在
`delegate_plan` 产生——模型在那里声明的就是「这个要在那个之后」，而「之后」预设的是
前序**做成了**；可并行的活儿走多次 `delegate_task`，彼此根本不产生边，也就无从配置。
所以条件不是一个枚举值，是这条边的定义本身。

改造前 `mark_failed` 把失败任务也塞进解锁集（源码注释 "treat failed as done for
unblocking" 标明是故意的），于是前序失败后继照跑——拿着不存在的产出往下做。现在
失败与取消都**不解锁**：后继永不可满足，由 `TaskManager.dispose_blocked_dependents`
判定并落 CANCELED 终态，不滞留队列。

因此本类只有**一个**完成集 `_succeeded`（仅 FINISHED）。它刻意不叫 `_completed`：
一个名为「已完成」却把 FAILED 排除在外的集合是个陷阱，读代码的人会按名字理解。
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
    #: 尚未**成功**的前序任务 id。前序 FAILED/CANCELED 时不会被摘除——该条目就此
    #: 永不可满足，等待 `dispose_blocked_dependents` 判定。
    blocked_by: set[str] = field(default_factory=set)


class TaskQueue:
    """In-process LIFO queue with DAG dependency support.

    Rules:
    - Tasks are scheduled LIFO (stack) when not blocked.
    - A task is blocked until every one of its `blocked_by` deps has FINISHED.
      A dep that failed or was canceled never unblocks it.
    - pop() returns the most recently pushed unblocked task.
    """

    def __init__(self) -> None:
        self._entries: list[QueueEntry] = []
        self._succeeded: set[str] = set()   # 仅 FINISHED——唯一的解锁依据
        self._running: set[str] = set()

    def push(self, entry: QueueEntry) -> None:
        # Remove already-satisfied dependencies at push time
        entry.blocked_by -= self._succeeded
        self._entries.append(entry)
        logger.debug("TaskQueue.push: %s (blocked_by=%s)", entry.task_id, entry.blocked_by)

    def pop(self, skip: Callable[[QueueEntry], bool] | None = None) -> QueueEntry | None:
        """Return the topmost non-blocked, non-running task (LIFO).

        ``skip``: optional predicate. Entries for which it returns True are left in
        the queue (not popped) — used for same-agent no-concurrency: skip a task whose
        target agent is currently busy. Such entries get picked up on a later pop() once
        the agent frees (a running task completing triggers another drain).
        """
        for i in range(len(self._entries) - 1, -1, -1):
            entry = self._entries[i]
            # Refresh: remove deps that have since succeeded
            entry.blocked_by -= self._succeeded
            if entry.blocked_by or entry.task_id in self._running:
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

    def seed_succeeded(self, task_ids: "Iterable[str]") -> None:
        """恢复期批量装填「已成功」集合（**只**传 FINISHED 的 id）。

        `TaskManager.restore` 此前直接写私有集合——那是穿透。与 `mark_complete` 的
        区别：这里不刷新已排队条目的 `blocked_by`，因为 restore 的调用顺序是
        **先装填、后 push**，而 `push` 首行就会摘掉已满足依赖，无需重复扫描。
        """
        self._succeeded.update(task_ids)

    def unmark_succeeded(self, task_id: str) -> None:
        """Remove from the succeeded set so a reopened task can be scheduled again."""
        self._succeeded.discard(task_id)

    def mark_complete(self, task_id: str) -> None:
        """FINISHED——唯一会解锁后继的收尾。"""
        self._running.discard(task_id)
        self._succeeded.add(task_id)
        # Refresh all blocked entries
        for entry in self._entries:
            entry.blocked_by.discard(task_id)
        logger.debug("TaskQueue.mark_complete: %s, %d pending", task_id, len(self._entries))

    def mark_failed(self, task_id: str) -> None:
        """终态但非成功（FAILED，以及依赖阻塞 / 用户取消的 CANCELED）。

        **不解锁任何后继**——它们依赖的是这个任务的产出，而产出不存在。后继的善后
        （落 CANCELED + BLOCKED_BY_FAILED_DEP）由 TaskManager 的永久阻塞扫描负责，
        所以这里只把它移出 running，不动 `_succeeded`。
        """
        self._running.discard(task_id)

    def cancel(self, task_id: str) -> bool:
        for i, entry in enumerate(self._entries):
            if entry.task_id == task_id:
                self._entries.pop(i)
                return True
        return False

    def drain_pending(self) -> list[str]:
        """Remove all queued (pending) entries; return their task ids.

        Does not touch ``_running`` / ``_succeeded`` — only clears what hasn't started.
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
        return all(bool(e.blocked_by) for e in self._entries)

    def peek_all(self) -> list[QueueEntry]:
        return list(self._entries)
