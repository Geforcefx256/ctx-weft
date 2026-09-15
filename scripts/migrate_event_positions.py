"""事件日志 position 回填迁移（spec: event-log；change reliability-wp2）。

旧数据没有存储层提交位置——本工具按 ``(session_id, event.id)`` 排序为每会话的事件
分配连续 position（1..n），并初始化会话 head。

⚠️ 如实声明（可靠性方案 §4.9）：分配的是**确定性顺序**，不是历史实际提交顺序——
旧契约没有记录提交顺序，本工具不能、也不声称还原它。已按旧 id 序正常恢复过的宿主
重放结果不变（id 序与本工具的排序键一致）。

用法：
  python scripts/migrate_event_positions.py --db PATH            # dry-run（默认，只报告）
  python scripts/migrate_event_positions.py --db PATH --execute  # 实际写入

幂等：已有 position 的行跳过；同会话继续从现有最大 position 之后分配。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select, text

from ctx_weft.providers._sqlalchemy import make_session_factory
from ctx_weft.providers.events.store.sql.models import (
    Base,
    EventModel,
    SessionHeadModel,
)


async def _ensure_schema(engine) -> None:
    """建缺失表 + 给既有 events 表补 position 列与唯一索引（与 open 路径同一套兼容逻辑）。"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        cols = await conn.execute(text("PRAGMA table_info(events)"))
        if "position" not in {r[1] for r in cols}:
            await conn.execute(text("ALTER TABLE events ADD COLUMN position INTEGER"))
        await conn.execute(text(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_events_session_position "
            "ON events (session_id, position)"))


async def _report(factory) -> dict:
    async with factory() as db:
        total = await db.scalar(select(func.count()).select_from(EventModel))
        sessions = await db.scalar(
            select(func.count()).select_from(select(EventModel.session_id).distinct()))
        positioned = await db.scalar(
            select(func.count()).select_from(EventModel)
            .where(EventModel.position.isnot(None)))
        ids = (await db.execute(select(EventModel.session_id, EventModel.id))).all()
        dup_counter = Counter((sid, eid) for sid, eid in ids)
        duplicates = sum(1 for v in dup_counter.values() if v > 1)
    return {
        "sessions": sessions,
        "events_total": total,
        "events_with_position": positioned,
        "events_pending": total - positioned,
        "incomplete_duplicate_event_ids": duplicates,
        "note": "deterministic (session_id, event.id) order — NOT the historical commit order",
    }


async def _execute(factory) -> dict:
    assigned = 0
    sessions_touched = 0
    async with factory() as db:
        session_ids = (await db.execute(
            select(EventModel.session_id).distinct().order_by(EventModel.session_id))
        ).scalars().all()
        for sid in session_ids:
            rows = (await db.execute(
                select(EventModel)
                .where(EventModel.session_id == sid, EventModel.position.is_(None))
                .order_by(EventModel.id))).scalars().all()
            if not rows:
                continue
            current_max = await db.scalar(
                select(func.max(EventModel.position))
                .where(EventModel.session_id == sid))
            pos = current_max or 0
            for row in rows:
                pos += 1
                row.position = pos
                assigned += 1
            head = await db.get(SessionHeadModel, sid)
            if head is None:
                db.add(SessionHeadModel(session_id=sid, next_position=pos))
            else:
                head.next_position = max(head.next_position, pos)
            sessions_touched += 1
        await db.commit()
    return {"sessions_touched": sessions_touched, "positions_assigned": assigned}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="SQLite 数据库文件路径")
    parser.add_argument("--execute", action="store_true",
                        help="实际写入（缺省 dry-run：只输出报告不落任何数据）")
    args = parser.parse_args()
    if not args.db.exists():
        print(json.dumps({"error": f"database not found: {args.db}"}))
        return 2

    engine, factory = make_session_factory(
        f"sqlite+aiosqlite:///{args.db}", connect_args={"timeout": 15})
    try:
        await _ensure_schema(engine)
        report = await _report(factory)
        result = {"dry_run": not args.execute, **report}
        if args.execute:
            result["migration"] = await _execute(factory)
            result.update(await _report(factory))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        await engine.dispose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
