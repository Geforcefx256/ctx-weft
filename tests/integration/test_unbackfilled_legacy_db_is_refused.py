"""未回填 position 的存量库：开库就炸，不许静默恢复成空视图。

这是复核「从快照分表那版升上来」时翻出来的一条静默数据丢失路径（2026-09-20）。

链条：`open_sqlite_event_store` 给存量 `events` 表加 `position` 列（`create_all` 不加列），
但**不回填**——回填是 `scripts/migrate_event_positions.py`，一个默认 dry-run、要过目的独立
操作。而所有按 position 的读都带 `position IS NOT NULL`（那是「只读已提交」的判据）：

- `committed_head` → 0（没有 head 行）
- `read_range(after_position=0)` → 空
- `replay()` → 也是按 position 区间分批，同样空

于是 `rebuild_view` 恢复出一个**空视图**：没有会话、没有 task、没有 agent，**不报任何错**。
更糟的是之后新事件从 position 1 开始编号（NULL 行不参与唯一索引碰撞），整段历史就此永久
不可见——没有任何后续操作会把它捞回来。

失效方向：静默、彻底、发生在恢复路径上。所以换成开库即失败。

**宿主接 Postgres 的正路不经过这里**：`bootstrap/db.py` 的 `open_database` 先
`run_pending`（m020 顺带回填）再恢复，顺序有注释钉着。这条守卫针对的是单机 SQLite 部署与
直接拿旧库做实验的人。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from ctx_weft.providers.events.store.sql.store import open_sqlite_event_store

_LEGACY_EVENTS = """
CREATE TABLE events (
    id VARCHAR(64) PRIMARY KEY, run_id VARCHAR(64), session_id VARCHAR(64),
    task_id VARCHAR(64), agent_id VARCHAR(64), tenant_id VARCHAR(64) DEFAULT 'default',
    type VARCHAR(128), sequence INTEGER, payload_json TEXT DEFAULT '{}',
    metadata_json TEXT DEFAULT '{}', causation_id VARCHAR(64), origin VARCHAR(64),
    schema_version INTEGER DEFAULT 1, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
"""


def _legacy_db(path: Path, *, n: int = 4, with_snapshot_table: bool = True) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(_LEGACY_EVENTS)
        if with_snapshot_table:
            # 快照分表那版留下的表。恢复不再读它，留着正好一起验「有它也不碍事」。
            conn.execute("""CREATE TABLE event_snapshots (
                id VARCHAR(64) PRIMARY KEY, session_id VARCHAR(64),
                run_id VARCHAR(64) DEFAULT '', last_event_id VARCHAR(64),
                last_event_sequence INTEGER, state_blob_json TEXT DEFAULT '{}',
                snapshot_reason VARCHAR(64) DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            conn.execute(
                "INSERT INTO event_snapshots (id, session_id, last_event_id,"
                " last_event_sequence, state_blob_json)"
                " VALUES ('snp','s1','evt_0001',1,'{\"bogus\": true}')")
        conn.execute(
            "INSERT INTO events (id, run_id, session_id, type, sequence, payload_json)"
            " VALUES (?,?,?,?,?,?)",
            ("evt_0001", "r", "s1", "SessionCreated", 0,
             json.dumps({"template_id": "agent:default", "root_agent_id": "agt_root"})))
        for i in range(1, n):
            conn.execute(
                "INSERT INTO events (id, run_id, session_id, task_id, type, sequence,"
                " payload_json) VALUES (?,?,?,?,?,?,?)",
                (f"evt_{i + 1:04d}", "r", "s1", f"tsk_{i}", "TaskCreated", i,
                 json.dumps({"task": {
                     "id": f"tsk_{i}", "status": "PENDING", "title": "T",
                     "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root"}})))
        conn.commit()
    finally:
        conn.close()


async def test_opening_an_unbackfilled_legacy_db_raises(tmp_path):
    """开库即失败，且错误信息里带着该跑的那条命令。"""
    db = tmp_path / "legacy.sqlite"
    _legacy_db(db)

    with pytest.raises(RuntimeError) as ei:
        async with open_sqlite_event_store(db):
            pass                                    # pragma: no cover

    msg = str(ei.value)
    assert "position" in msg
    assert "migrate_event_positions" in msg, f"没告诉人怎么修：{msg}"


async def test_the_silent_failure_it_replaces_really_was_silent(tmp_path):
    """钉住它替掉的是什么：绕过守卫直连时，恢复确实静默返回空视图。

    这条不是在测生产路径，是把「不加守卫会怎样」记在代码里——否则下一个人看到那道探测只会
    觉得是多余的启动开销，顺手删掉。直接构造 `SqlEventStore`（不经 open 路径的守卫），恢复
    出来的视图里一条 task 都没有，而库里明明有 4 条事件。
    """
    from ctx_weft.core.control.reducers import rebuild_view
    from ctx_weft.providers._sqlalchemy import make_session_factory
    from ctx_weft.providers.events.store.sql.models import Base
    from ctx_weft.providers.events.store.sql.store import SqlEventStore

    db = tmp_path / "silent.sqlite"
    _legacy_db(db, n=4)
    engine, factory = make_session_factory(f"sqlite+aiosqlite:///{db}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)     # 只补缺表，不加列
            from sqlalchemy import text
            await conn.execute(text("ALTER TABLE events ADD COLUMN position INTEGER"))
        store = SqlEventStore(factory)

        assert await store.committed_head("s1") == 0
        assert await store.read_range("s1", after_position=0) == []
        view = await rebuild_view(store, "s1")
        assert view.tasks == {}, "居然读到了——那守卫也许可以放宽"
        assert view.events_total == 0
        # 而按 session 的那条读法看得见它们：差别只在「按 position 还是按 id 序」
        assert len(await store.read_by_session("s1")) == 4
    finally:
        await engine.dispose()


async def test_a_backfilled_db_opens_fine(tmp_path):
    """跑过回填脚本的库正常开——守卫只挡未回填的。"""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[2] / "scripts" / "migrate_event_positions.py"
    db = tmp_path / "ok.sqlite"
    _legacy_db(db, n=4)

    proc = subprocess.run(
        [sys.executable, str(script), "--db", str(db), "--execute"],
        capture_output=True, text=True, cwd=str(script.parents[1]))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["migration"]["positions_assigned"] == 4

    async with open_sqlite_event_store(db) as store:
        assert await store.committed_head("s1") == 4
        assert len(await store.read_range("s1", after_position=0)) == 4


async def test_a_fresh_db_opens_fine(tmp_path):
    """新库里一条 NULL position 都不会有——守卫不该给正常路径添麻烦。"""
    async with open_sqlite_event_store(tmp_path / "fresh.sqlite") as store:
        assert await store.committed_head("nope") == 0
