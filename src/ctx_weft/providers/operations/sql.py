"""操作账本的 SQL 实现（spec: tool-operations；change reliability-wp5）。

SQLite 默认（与 events/memory 的 open 上下文同习惯）；postgres 同源（宿主自带
migration 直接构造 ``SqlOperationStore(session_factory)``）。表独立于 events/memory
两个包的 Base（同一可选依赖边界纪律）。
"""

from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Integer, String, Text, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.operations import (
    OperationRecord,
    OperationStatus,
    OperationUpdate,
    RevisionConflict,
)
from ctx_weft.providers._sqlalchemy import make_session_factory

__all__ = ["SqlOperationStore", "open_sqlite_operation_store"]


class _Base(DeclarativeBase):
    """本包自带的 declarative base（与 events/memory 的刻意不共用）。"""


class OperationModel(_Base):
    __tablename__ = "operations"

    operation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_id: Mapped[str] = mapped_column(String(64))
    assistant_record_id: Mapped[str] = mapped_column(String(64))
    tool_ordinal: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32))
    revision: Mapped[int] = mapped_column(Integer)
    args_hash: Mapped[str] = mapped_column(String(128), default="")
    recovery_policy: Mapped[str] = mapped_column(String(32), default="manual")
    attempts_json: Mapped[str] = mapped_column(Text, default="[]")
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    memory_result_id: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


def _to_model(rec: OperationRecord) -> OperationModel:
    return OperationModel(
        operation_id=rec.operation_id, tenant_id=rec.tenant_id,
        session_id=rec.session_id, agent_id=rec.agent_id,
        assistant_record_id=rec.assistant_record_id, tool_ordinal=rec.tool_ordinal,
        tool_name=rec.tool_name, status=rec.status.value, revision=rec.revision,
        args_hash=rec.args_hash, recovery_policy=rec.recovery_policy,
        attempts_json=json.dumps(rec.attempts),
        result_json=json.dumps(rec.result, ensure_ascii=False, default=str)
        if rec.result is not None else None,
        error=rec.error, memory_result_id=rec.memory_result_id,
    )


def _to_record(row: OperationModel) -> OperationRecord:
    return OperationRecord(
        operation_id=row.operation_id, tenant_id=row.tenant_id,
        session_id=row.session_id, agent_id=row.agent_id,
        assistant_record_id=row.assistant_record_id, tool_ordinal=row.tool_ordinal,
        tool_name=row.tool_name, status=OperationStatus(row.status),
        revision=row.revision, args_hash=row.args_hash,
        recovery_policy=row.recovery_policy,
        attempts=json.loads(row.attempts_json or "[]"),
        result=json.loads(row.result_json) if row.result_json else None,
        error=row.error, memory_result_id=row.memory_result_id,
        created_at=row.created_at, updated_at=row.updated_at,
    )


_LEGAL = {
    "prepared": {"started", "completed"},
    "started": {"completed", "waiting_human", "unknown"},
    "waiting_human": {"started", "completed", "unknown"},
    "unknown": {"completed", "started"},
    "completed": set(),
}


class SqlOperationStore:
    """SQLAlchemy-backed 操作账本（CAS 由 WHERE revision=? 行数判定，天然乐观锁）。"""

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]") -> None:
        self._factory = session_factory

    async def get(self, operation_id: str, ctx: ProviderContext) -> OperationRecord | None:
        async with self._factory() as db:
            row = await db.get(OperationModel, operation_id)
            return _to_record(row) if row is not None else None

    async def prepare(self, record: OperationRecord, ctx: ProviderContext) -> OperationRecord:
        async with self._factory() as db, db.begin():
            existing = await db.get(OperationModel, record.operation_id)
            if existing is not None:
                same = all(
                    getattr(existing, f) == getattr(record, f)
                    for f in ("tenant_id", "session_id", "agent_id",
                              "assistant_record_id", "tool_ordinal", "tool_name")
                )
                if not same:
                    raise ValueError(
                        f"operation_id {record.operation_id!r} already bound to a "
                        f"different logical call")
                # 幂等命中：本事务未做任何修改，退出 begin() 时提交空事务即可。
                # 这里**不调 rollback**——原先写的 `db.rollback()` 漏了 await，返回的
                # 协程从未被等待（RuntimeWarning），回滚实际上从未发生；而补上 await
                # 又会和 begin() 退出时的 commit 撞车。没有要撤销的写，直接返回。
                return _to_record(existing)
            db.add(_to_model(record))
        return copy.deepcopy(record)

    async def compare_and_set(
        self,
        operation_id: str,
        expected_revision: int,
        update: OperationUpdate,
        ctx: ProviderContext,
    ) -> OperationRecord:
        from sqlalchemy import update as sa_update
        async with self._factory() as db, db.begin():
            row = await db.get(OperationModel, operation_id)
            if row is None:
                raise KeyError(f"operation {operation_id!r} not prepared")
            if row.revision != expected_revision:
                raise RevisionConflict(
                    f"operation {operation_id!r} revision {row.revision} != expected "
                    f"{expected_revision}")
            if update.status is not None:
                if update.status.value not in _LEGAL.get(row.status, set()):
                    raise ValueError(
                        f"illegal operation transition {row.status} -> {update.status}")
            patch: dict = {"revision": row.revision + 1,
                           "updated_at": datetime.now(UTC)}
            if update.status is not None:
                patch["status"] = update.status.value
            if update.result_set:
                patch["result_json"] = (
                    json.dumps(update.result, ensure_ascii=False, default=str)
                    if update.result is not None else None)
            if update.error is not None:
                patch["error"] = update.error
            if update.append_attempt is not None:
                attempts = json.loads(row.attempts_json or "[]")
                attempts.append(update.append_attempt)
                patch["attempts_json"] = json.dumps(attempts)
            await db.execute(
                sa_update(OperationModel).where(
                    OperationModel.operation_id == operation_id,
                    OperationModel.revision == expected_revision,
                ).values(**patch)
            )
        got = await self.get(operation_id, ctx)
        assert got is not None
        return got


@asynccontextmanager
async def open_sqlite_operation_store(
    db_path: str | Path,
) -> AsyncIterator[SqlOperationStore]:
    """开一个 SQLite backed 的操作账本（建表 → yield → dispose）。测试与单机部署用。"""
    engine, factory = make_session_factory(
        f"sqlite+aiosqlite:///{db_path}", connect_args={"timeout": 15})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)
        yield SqlOperationStore(factory)
    finally:
        await engine.dispose()
