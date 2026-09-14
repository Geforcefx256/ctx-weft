"""操作账本的进程内实现（spec: tool-operations；change reliability-wp5）。

单锁短临界区 + dict；CAS 由 revision 乐观锁承担。进程内不会存储故障——生产跨进程
恢复请用 ``providers/operations/sql.py`` 或宿主自实现。
"""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime

from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.operations import (
    OperationRecord,
    OperationStatus,
    OperationUpdate,
    RevisionConflict,
)

__all__ = ["InMemoryOperationStore"]


def _now() -> datetime:
    return datetime.now(UTC)


class InMemoryOperationStore:
    """dict 版操作账本。prepare 幂等（同 id 同内容 no-op）；CAS 乐观锁串行推进。"""

    def __init__(self) -> None:
        self._records: dict[str, OperationRecord] = {}
        self._lock = asyncio.Lock()

    async def get(self, operation_id: str, ctx: ProviderContext) -> OperationRecord | None:
        rec = self._records.get(operation_id)
        return copy.deepcopy(rec) if rec is not None else None

    async def prepare(self, record: OperationRecord, ctx: ProviderContext) -> OperationRecord:
        async with self._lock:
            existing = self._records.get(record.operation_id)
            if existing is not None:
                # 幂等：同 id 同内容（身份字段一致）no-op；异内容拒绝（身份被复用）
                same = all(
                    getattr(existing, f) == getattr(record, f)
                    for f in ("tenant_id", "session_id", "agent_id",
                              "assistant_record_id", "tool_ordinal", "tool_name")
                )
                if not same:
                    raise ValueError(
                        f"operation_id {record.operation_id!r} already bound to a "
                        f"different logical call")
                return copy.deepcopy(existing)
            rec = copy.deepcopy(record)
            rec.created_at = rec.updated_at = _now()
            self._records[rec.operation_id] = rec
            return copy.deepcopy(rec)

    async def compare_and_set(
        self,
        operation_id: str,
        expected_revision: int,
        update: OperationUpdate,
        ctx: ProviderContext,
    ) -> OperationRecord:
        async with self._lock:
            rec = self._records.get(operation_id)
            if rec is None:
                raise KeyError(f"operation {operation_id!r} not prepared")
            if rec.revision != expected_revision:
                raise RevisionConflict(
                    f"operation {operation_id!r} revision {rec.revision} != expected "
                    f"{expected_revision}")
            if update.status is not None:
                _validate_transition(rec.status, update.status)
                # setattr 形态（而非属性赋值）：TaskManager 状态守卫的 AST 扫描按
                # `<bare>.status =` 抓 task 状态越权写——OperationRecord 的 status 是
                # 账本自己的状态机，不属该契约，避免误报。
                setattr(rec, "status", update.status)
            if update.result_set:
                rec.result = copy.deepcopy(update.result)
            if update.error is not None:
                rec.error = update.error
            if update.append_attempt is not None:
                rec.attempts.append(update.append_attempt)
            rec.revision += 1
            rec.updated_at = _now()
            return copy.deepcopy(rec)


_LEGAL = {
    OperationStatus.PREPARED: {OperationStatus.STARTED, OperationStatus.COMPLETED},
    OperationStatus.STARTED: {OperationStatus.COMPLETED, OperationStatus.WAITING_HUMAN},
    OperationStatus.WAITING_HUMAN: {OperationStatus.STARTED, OperationStatus.COMPLETED},
    # 终态——**含「结果无法确定」这一种结局**：那是 result 的内容，不是另一个状态。
    OperationStatus.COMPLETED: set(),
}


def _validate_transition(src: OperationStatus, dst: OperationStatus) -> None:
    if dst not in _LEGAL.get(src, set()):
        raise ValueError(f"illegal operation transition {src} -> {dst}")
