"""`CtxWeftRuntime` 与事件持久化接线的集成测试（final review C1/C2）。

全仓此前没有任何测试把 `CtxWeftRuntime` 与 `attach_persistence` / `SnapshotWriter` /
`SqlEventStore` 放在一起构造——这正是两条 Critical 穿过 13 轮 review 都没被抓到的
根因。本文件补上这道缝：

- `CtxWeftRuntime(event_store=<SqlEventStore>)` 构造后，事件真的进了那个 SQL
  store（不是内部默认的 `InMemoryEventStore`）。
- 没有双写：同一条事件在 store 里只有一份，且不产生 `IntegrityError`。
- `snapshot_every_n > 0` 时 `runtime.persistence.snapshot_writer` 非 None 且真的写出快照；
  `snapshot_every_n=0`（默认）时为 None。
- `await runtime.persistence.detach()` 之后事件不再落库。

构造 `CtxWeftRuntime` 的最小接线照抄
`tests/unit/test_protocols_events_relocation.py::test_runtime_still_gets_a_working_default_store`
（同样来自 `tests/unit/test_media_get_image.py::_StubAgents`）：`CtxWeftRuntime()`
无参构造会因「未注册 AgentCapabilityProvider」抛 ValueError，故需要一个最小 stub。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from ctx_weft.core.registry import ProviderRegistry
from ctx_weft.core.runtime import CtxWeftRuntime
from ctx_weft.protocols.capability import (
    AgentCapability,
    AgentCapabilityProvider,
    CapabilityProviderInfo,
)
from ctx_weft.protocols.events import Event
from ctx_weft.providers.events.store.sql import SqlEventStore, open_sqlite_event_store


class _StubAgents(AgentCapabilityProvider):
    """满足 `CtxWeftRuntime` 构造期「至少一个 AgentCapabilityProvider」硬校验。"""

    name = "stub_agents"

    async def list(self, ctx):
        return [AgentCapability(id="stub_agents:a", name="a", kind="agent")]

    async def get_template(self, template_id, version, ctx):
        return None

    async def describe(self, ctx):
        return CapabilityProviderInfo(name=self.name)


def _make_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_capability(_StubAgents())
    return registry


def _ev(seq: int, type_: str = "RunStarted", *, session: str = "s1", run_id: str = "r1") -> Event:
    from datetime import UTC, datetime, timedelta

    t0 = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)
    return Event(
        id=f"evt_{seq:04d}",
        run_id=run_id,
        sequence=seq,
        session_id=session,
        type=type_,
        timestamp=t0 + timedelta(seconds=seq),
        payload={},
    )


@asynccontextmanager
async def _sql_store(tmp_path) -> AsyncIterator[SqlEventStore]:
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        yield store


# ── C1: runtime 的读路径必须真的读到传入的 SqlEventStore ──────────────────────


async def test_runtime_persists_events_into_the_injected_sql_store(tmp_path) -> None:
    async with _sql_store(tmp_path) as store:
        runtime = CtxWeftRuntime(providers=_make_registry(), event_store=store)

        assert runtime.event_store is store  # 不是内部默认的 InMemoryEventStore

        await runtime.event_bus.emit(_ev(1, "SessionCreated"))
        await runtime.event_bus.emit(_ev(2, "RunStarted"))

        got = await store.read_by_session("s1")
        assert [e.type for e in got] == ["SessionCreated", "RunStarted"]


# ── C2: 单一入口，无双写 ───────────────────────────────────────────────────────


async def test_no_double_write_via_attach_persistence_single_entry_point(
    tmp_path, caplog
) -> None:
    """同一条事件在 store 里只有一份，且没有 IntegrityError。

    反事实自查（见 final-fix-report.md）：把下面 `CtxWeftRuntime` 内部的接线换回
    「直接 EventPersister(...) 再额外 attach_persistence 一次」，这条断言会变红
    （len(got) == 2 而不是 1，且日志出现 IntegrityError）。
    """
    import logging

    async with _sql_store(tmp_path) as store:
        runtime = CtxWeftRuntime(providers=_make_registry(), event_store=store)

        with caplog.at_level(logging.ERROR):
            await runtime.event_bus.emit(_ev(1, "SessionCreated"))

        got = await store.read_by_session("s1")
        assert len(got) == 1, f"同一条事件不应该被存两份，实际：{got}"
        assert "IntegrityError" not in caplog.text


# ── snapshot_every_n 的两种取值 ────────────────────────────────────────────────


from ctx_weft.core.models.config import RuntimeConfig


async def test_snapshot_every_n_zero_means_no_snapshot_writer(tmp_path) -> None:
    async with _sql_store(tmp_path) as store:
        runtime = CtxWeftRuntime(providers=_make_registry(), event_store=store)
        assert runtime.persistence.snapshot_writer is None


async def test_snapshot_every_n_positive_attaches_and_writes_snapshot(tmp_path) -> None:
    async with _sql_store(tmp_path) as store:
        runtime = CtxWeftRuntime(
            providers=_make_registry(), event_store=store, snapshot_every_n=1
        )
        assert runtime.persistence.snapshot_writer is not None

        await runtime.event_bus.emit(_ev(1, "SessionCreated"))
        await runtime.event_bus.emit(_ev(2, "RunStarted"))
        await runtime.event_bus.emit(_ev(3, "RunFinished"))  # 触发快照边界

        snap = await store.load_latest_snapshot("s1")
        assert snap is not None
        assert snap.last_event_id == "evt_0003"


# ── detach 之后事件不再落库 ────────────────────────────────────────────────────


async def test_detach_stops_persistence(tmp_path) -> None:
    """best_effort：persister 订阅可 detach（spec: event-commit 的兼容路径）。

    required（默认）没有 persister 订阅可 detach——提交在 CommitGate 里；
    那一路的等价物见下方 test_required_mode_persists_via_gate_not_persister。
    """
    async with _sql_store(tmp_path) as store:
        runtime = CtxWeftRuntime(
            providers=_make_registry(), event_store=store,
            config=RuntimeConfig(event_commit_policy="best_effort"))

        await runtime.event_bus.emit(_ev(1, "SessionCreated"))
        assert len(await store.read_by_session("s1")) == 1

        await runtime.persistence.detach()

        await runtime.event_bus.emit(_ev(2, "RunStarted"))
        # detach 之后新事件不应该再落库——store 里仍然只有 detach 之前那一条。
        assert len(await store.read_by_session("s1")) == 1


async def test_required_mode_persists_via_gate_not_persister(tmp_path) -> None:
    """required（默认，spec: event-commit）：无 persister 订阅，提交走 CommitGate——
    detach 句柄是空壳，事件照常落库（先确认提交、再通知）。"""
    async with _sql_store(tmp_path) as store:
        runtime = CtxWeftRuntime(providers=_make_registry(), event_store=store)
        assert runtime.persistence.persister is None

        await runtime.event_bus.emit(_ev(1, "SessionCreated"))
        assert len(await store.read_by_session("s1")) == 1   # gate 已确认提交

        await runtime.persistence.detach()                    # 空壳 detach：无订阅可停
        await runtime.event_bus.emit(_ev(2, "RunStarted"))
        assert len(await store.read_by_session("s1")) == 2    # 提交不因 detach 停
