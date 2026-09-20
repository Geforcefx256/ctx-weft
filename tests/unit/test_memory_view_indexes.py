"""`load_view` / `recall_topic` 的工作索引真的被规划器用上（P0，2026-09-19）。

这三条索引（`ix_memory_task_view` / `ix_memory_agent_view` / `ix_memory_topic_seq`）的
「对不对」只能从**查询计划**上看——声明了不等于命中。既有的
`ix_memory_task` / `ix_memory_agent` 就是个例子：它们存在，但 `load_view` 实测走全表扫，
因为 `type IN (...)` 是多值、且 `ORDER BY timestamp, seq_no` 的排序列不在索引里。

所以这里不断言「索引存在」，而是拿 `EXPLAIN QUERY PLAN` 断言两件事：**走索引**，
且**不需要临时排序**。谓词写法也一并钉住——见
`test_index_is_hit_regardless_of_how_the_boolean_is_written` 的 docstring。
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from ctx_weft.providers.memory.sql import open_sqlite_memory

pytestmark = pytest.mark.asyncio

_VIEW_INDEXES = ("ix_memory_task_view", "ix_memory_agent_view", "ix_memory_topic_seq")

_SEED = (
    "INSERT INTO memory_events (id, session_id, task_id, agent_id, layer, type, role,"
    " topic, content, seq_no, topic_seq_no, is_superseded, metadata_json, timestamp)"
    " VALUES (:id,'s1','t1','a1','task','conversation_turn','user','tp1','x',"
    " :sq,:sq,:sup,'{}',:ts)"
)


async def _seeded(mem, *, dead: int = 300, live: int = 20):
    """一个 (session, layer, task) 分区：dead 条死行 + live 条活行，并 ANALYZE。"""
    async with mem._factory() as db:                       # noqa: SLF001 —— 计划断言需要裸连接
        await db.execute(text(_SEED), [
            {"id": f"mev_{i}", "sq": i, "sup": 1 if i < dead else 0,
             "ts": f"2026-01-0{1 if i < dead else 2} 00:{i % 60:02d}:00"}
            for i in range(dead + live)])
        await db.execute(text("ANALYZE"))
        await db.commit()


async def _plan(mem, sql: str, params: dict) -> str:
    async with mem._factory() as db:                       # noqa: SLF001
        rows = (await db.execute(text("EXPLAIN QUERY PLAN " + sql), params)).all()
    return " | ".join(str(r[-1]) for r in rows)


async def test_load_view_task_scope_uses_the_index_and_needs_no_sort(tmp_path) -> None:
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        await _seeded(mem)
        plan = await _plan(mem, (
            "SELECT * FROM memory_events WHERE coalesce(tenant,'default')=:tn"
            " AND session_id=:s AND layer=:l AND is_superseded IS 0"
            " AND type IN ('conversation_turn','summary') AND task_id=:t"
            " ORDER BY timestamp ASC, seq_no ASC"),
            {"tn": "default", "s": "s1", "l": "task", "t": "t1"})

    assert "ix_memory_task_view" in plan, f"应走 task 视图索引，实际：{plan}"
    assert "TEMP B-TREE" not in plan, f"排序该由索引提供，实际：{plan}"


async def test_load_view_cross_task_aggregate_uses_the_agent_index(tmp_path) -> None:
    """跨 task 聚合（半址 `task_id=None` → 按 agent_id 过滤，layer 仍是 'task'）。

    这是 `AgentRecallSource` 的主查询——已结束 task 的对话就是靠它进上下文的。
    """
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        await _seeded(mem)
        plan = await _plan(mem, (
            "SELECT * FROM memory_events WHERE coalesce(tenant,'default')=:tn"
            " AND session_id=:s AND layer=:l AND is_superseded IS 0"
            " AND type IN ('conversation_turn','summary') AND agent_id=:a"
            " ORDER BY timestamp ASC, seq_no ASC"),
            {"tn": "default", "s": "s1", "l": "task", "a": "a1"})

    assert "ix_memory_agent_view" in plan, f"应走 agent 视图索引，实际：{plan}"
    assert "TEMP B-TREE" not in plan, f"排序该由索引提供，实际：{plan}"


async def test_index_is_hit_regardless_of_how_the_boolean_is_written(tmp_path) -> None:
    """`is_superseded` 做**等值列**而不是部分索引谓词——两种布尔写法都必须命中。

    这是刻意的设计：部分索引（`WHERE is_superseded = 0`）在 SQLite 上要求索引谓词与
    查询谓词**逐字一致**才命中，实测 `= 0` 的索引配 `IS 0` 的查询直接退回全表扫、
    且不报错。而本仓两个 provider 的写法本就不同（core 用 `.is_(False)` → `IS 0`，
    宿主用 `== False` → `= 0`），做成部分索引的话任何一处改写法都会静默失效。
    """
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        await _seeded(mem)
        base = ("SELECT * FROM memory_events WHERE session_id=:s AND layer=:l AND {}"
                " AND type IN ('conversation_turn','summary') AND task_id=:t"
                " ORDER BY timestamp ASC, seq_no ASC")
        params = {"s": "s1", "l": "task", "t": "t1"}
        for predicate in ("is_superseded IS 0", "is_superseded = 0"):
            plan = await _plan(mem, base.format(predicate), params)
            assert "ix_memory_task_view" in plan, f"{predicate!r} 未命中索引：{plan}"
            assert "TEMP B-TREE" not in plan, f"{predicate!r} 仍需临时排序：{plan}"


async def test_recall_topic_and_the_write_path_max_use_the_topic_index(tmp_path) -> None:
    """`recall_topic` 的游标查询按 topic_seq_no 升序，写路径要 `max(topic_seq_no)`。

    后者**不带** is_superseded（seq 是分区内单调序号，死行也占号，不能复用），所以它
    要的是全量的 `(topic, topic_seq_no)`——实测从回表变成 COVERING INDEX。
    """
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        await _seeded(mem)
        recall = await _plan(mem, (
            "SELECT * FROM memory_events WHERE topic=:tp AND topic_seq_no>:c"
            " AND is_superseded IS 0 ORDER BY topic_seq_no"), {"tp": "tp1", "c": 0})
        max_seq = await _plan(
            mem, "SELECT max(topic_seq_no) FROM memory_events WHERE topic=:tp",
            {"tp": "tp1"})

    assert "ix_memory_topic_seq" in recall, f"recall_topic 应走 topic 索引：{recall}"
    assert "TEMP B-TREE" not in recall, f"topic_seq_no 的序该由索引提供：{recall}"
    assert "COVERING INDEX ix_memory_topic_seq" in max_seq, (
        f"写路径的 max(topic_seq_no) 该走覆盖索引、不回表：{max_seq}")


async def test_write_path_max_seq_no_is_covered(tmp_path) -> None:
    """写路径的 `max(seq_no)`（分区内下一个序号）同样不该回表。"""
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        await _seeded(mem)
        plan = await _plan(mem, (
            "SELECT max(seq_no) FROM memory_events"
            " WHERE session_id=:s AND layer=:l AND task_id=:t"),
            {"s": "s1", "l": "task", "t": "t1"})

    assert "COVERING INDEX ix_memory_task_view" in plan, (
        f"max(seq_no) 该走覆盖索引、不回表：{plan}")


async def test_all_three_indexes_exist_on_a_fresh_db(tmp_path) -> None:
    """新库由 models.py 的 `__table_args__` 建出——名字是跨仓契约（宿主 m021 照它建）。"""
    async with open_sqlite_memory(tmp_path / "mem.db") as mem:
        async with mem._factory() as db:                   # noqa: SLF001
            names = {r[0] for r in (await db.execute(text(
                "SELECT name FROM sqlite_master WHERE type='index'"
                " AND tbl_name='memory_events'"))).all()}

    assert set(_VIEW_INDEXES) <= names, f"缺索引：{set(_VIEW_INDEXES) - names}"
