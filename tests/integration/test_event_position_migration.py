"""迁移端到端（spec: snapshot-recovery；wp4-4.1，方案 E-T13）。

旧 schema 库（无 position 列、无快照新列）→ `migrate_event_positions.py --execute`
→ 正式 open 路径起动 → 事件可读、快照按 position 写/恢复、两路等价。
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from ctx_weft.core.control.reducers import reduce_events, rebuild_view
from ctx_weft.protocols.events import Event
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.events.persister import attach_persistence
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_event_positions.py"
_T0 = datetime(2026, 9, 11, tzinfo=UTC)


def _legacy_db(db: Path) -> None:
    """旧 schema：events 无 position、event_snapshots 无新列（WP2/WP4 之前）。"""
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE events (
        id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64), session_id VARCHAR(64),
        task_id VARCHAR(64), agent_id VARCHAR(64), tenant_id VARCHAR(64) DEFAULT 'default',
        type VARCHAR(128), sequence INTEGER, payload_json TEXT DEFAULT '{}',
        metadata_json TEXT DEFAULT '{}', causation_id VARCHAR(64), origin VARCHAR(64),
        schema_version INTEGER DEFAULT 1, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.execute("""CREATE TABLE event_snapshots (
        id VARCHAR(64) PRIMARY KEY, session_id VARCHAR(64), run_id VARCHAR(64) DEFAULT '',
        last_event_id VARCHAR(64), last_event_sequence INTEGER,
        state_blob_json TEXT DEFAULT '{}', snapshot_reason VARCHAR(64) DEFAULT '',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    for i in range(1, 4):
        conn.execute(
            "INSERT INTO events (id, run_id, session_id, type, sequence, task_id, payload_json)"
            " VALUES (?,?,?,?,?,?,?)",
            (f"evt_old_{i:04d}", "r", "s1", "TaskCreated", i, f"old{i}",
             json.dumps({"task": {"id": f"old{i}", "assigned_agent_id": "ag"}})))
    # 一张 legacy 快照（无 position）——恢复必须忽略它走全量
    conn.execute(
        "INSERT INTO event_snapshots (id, session_id, last_event_id, last_event_sequence, state_blob_json)"
        " VALUES ('snp_legacy','s1','evt_old_0001',1,'{\"bogus\": true}')")
    conn.commit()
    conn.close()


def _run_migrate(db: Path, *extra: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--db", str(db), *extra],
        capture_output=True, text=True, cwd=str(_SCRIPT.parents[1]))
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


async def test_migrated_db_recovers_via_position_and_ignores_legacy_snapshot(tmp_path):
    db = tmp_path / "legacy.sqlite"
    _legacy_db(db)

    # dry-run 默认零写入 → execute 回填
    report = _run_migrate(db)
    assert report["dry_run"] is True and report["events_pending"] == 3
    done = _run_migrate(db, "--execute")
    assert done["migration"]["positions_assigned"] == 3

    # 正式 open 路径起动（幂等 ALTER 补快照新列）+ 重建
    async with open_sqlite_event_store(db) as store:
        assert await store.committed_head("s1") == 3
        restored = await rebuild_view(store, "s1")
        # legacy 快照（无 position/含 bogus blob）被忽略——全量回放得到真实三任务
        assert sorted(restored.tasks) == ["old1", "old2", "old3"]

        # 新事件继续提交 + writer 按 position 写快照
        bus = InProcessEventBus()
        attach_persistence(bus, store, snapshot_every_n=1)
        from ctx_weft.protocols.events import EventType
        await bus.emit(Event(id="evt_new_1", run_id="r", sequence=10, session_id="s1",
                             type=EventType.RUN_FINISHED, timestamp=_T0,
                             payload={"outcome": "completed"}))
        snap = await store.load_latest_snapshot("s1")
        assert snap is not None
        assert snap.last_commit_position == 4            # position 游标（非事件 ID）
        assert snap.projection_version == 1

        # 两路等价：全量 vs 快照+增量
        after = await rebuild_view(store, "s1")
        full = reduce_events([se.event for se in await store.read_range("s1")], "s1")
        assert sorted(after.tasks) == sorted(full.tasks) == ["old1", "old2", "old3"]
