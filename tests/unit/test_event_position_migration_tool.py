"""迁移工具 conformance（spec: event-log；wp2-5.1）：dry-run 默认零写入、execute 连续回填。"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from ctx_weft.protocols.events import Event
from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store
from datetime import UTC, datetime

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "migrate_event_positions.py"


def _legacy_db(db: Path) -> None:
    """造一个旧 schema 库（没有 position 列）直插事件——迁移前后的真实起点。"""
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE events (
        id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64), session_id VARCHAR(64),
        task_id VARCHAR(64), agent_id VARCHAR(64), tenant_id VARCHAR(64) DEFAULT 'default',
        type VARCHAR(128), sequence INTEGER, payload_json TEXT DEFAULT '{}',
        metadata_json TEXT DEFAULT '{}', causation_id VARCHAR(64), origin VARCHAR(64),
        schema_version INTEGER DEFAULT 1, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    for sid, n in (("s1", 3), ("s2", 2)):
        for i in range(1, n + 1):
            conn.execute(
                "INSERT INTO events (id, run_id, session_id, type, sequence) VALUES (?,?,?,?,?)",
                (f"evt_{sid}_{i:04d}", "r", sid, "RunStarted", i))
    conn.commit()
    conn.close()


def _run(db: Path, *extra: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--db", str(db), *extra],
        capture_output=True, text=True, cwd=str(_SCRIPT.parents[1]))
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_dry_run_default_writes_nothing(tmp_path):
    db = tmp_path / "legacy.sqlite"
    _legacy_db(db)
    report = _run(db)
    assert report["dry_run"] is True
    assert report["events_pending"] == 5 and report["sessions"] == 2
    assert "NOT the historical commit order" in report["note"]  # 如实声明的口径钉住
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
    assert "position" in cols          # schema 兼容补列允许（幂等探测）
    assert conn.execute("SELECT COUNT(*) FROM events WHERE position IS NOT NULL").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM event_session_head").fetchone()[0] == 0
    conn.close()


async def test_execute_assigns_contiguous_positions_and_heads(tmp_path):
    db = tmp_path / "legacy.sqlite"
    _legacy_db(db)
    report = _run(db, "--execute")
    assert report["migration"] == {"sessions_touched": 2, "positions_assigned": 5}
    assert report["events_pending"] == 0

    async with open_sqlite_event_store(db) as store:   # 迁移后经正式 open 路径核验
        assert await store.committed_head("s1") == 3
        assert await store.committed_head("s2") == 2
        s1 = await store.read_range("s1")
        assert [se.position for se in s1] == [1, 2, 3]
        assert [se.event.id for se in s1] == sorted(se.event.id for se in s1)
        # 幂等：新提交接在迁移后的 head 之后
        extra = Event(id="evt_new", run_id="r", sequence=9, session_id="s2",
                      type="RunStarted", timestamp=datetime.now(UTC), payload={})
        receipt = await store.append_batch("s2", "b_new", [extra])
        assert receipt.records[0].position == 3
