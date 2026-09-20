"""capability 折叠按 task 收窄——那是它唯一**安全**的界。

它要回答的是「这次调用跑过没有」，而 `CapabilityInvoked` / `CapabilityFinished` 每次工具调用
留一对、只增不减。实测一条 4000 次调用的会话：取回 8000 条、779ms、24.4MB，而它真正需要的只是
正在 reconcile 的那个 task 里那几个 dangling tool_call。

**为什么不能按「最近 N 条」截尾。** 漏掉一条 `CapabilityInvoked`，gateway 就以为那次调用没跑过
→ 静默重跑一个有副作用的工具。失败方向朝错的那边，所以不猜。按 task 收窄不是猜：dangling 调用
必定属于本 task，而 capability 事件带着 `task_id`（`make_event` 从 `LoopState` 取）。

**但收窄本身也必须朝安全方向失败**：`task_id` 为 NULL / 空串的事件照样取回。那一档是「无从
归属」（存量数据、非 run 域事件），不是「属于别人」——把它们排除掉就把上面那个重跑风险请回来了。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID = "s1"


def _cap(n: int, tool_call_id: str, *, task_id: str | None, finished: bool = False) -> Event:
    return Event(
        id=f"evt_{n:08d}", run_id="r1", sequence=n, session_id=_SID,
        type=EventType.CAPABILITY_FINISHED if finished else EventType.CAPABILITY_INVOKED,
        timestamp=_T0, tenant_id="acme", task_id=task_id, agent_id="ag1",
        payload={"tool_call_id": tool_call_id, "invocation_id": f"iv_{n}",
                 "capability_name": "bash", "capability_id": "fs:bash",
                 "arguments": {}, "outcome": "success", "result": "ok",
                 "result_length": 2},
    )


_TYPES = (EventType.CAPABILITY_INVOKED, EventType.CAPABILITY_FINISHED)


async def _store() -> InMemoryEventStore:
    store = InMemoryEventStore()
    await store.append(_cap(1, "c_mine", task_id="t_mine"))
    await store.append(_cap(2, "c_mine", task_id="t_mine", finished=True))
    await store.append(_cap(3, "c_other", task_id="t_other"))
    await store.append(_cap(4, "c_other", task_id="t_other", finished=True))
    await store.append(_cap(5, "c_orphan", task_id=None))      # 无从归属（存量）
    return store


async def test_narrowing_drops_only_other_tasks() -> None:
    """收窄掉的只有**明确属于别的 task** 的那些。"""
    from ctx_weft.core.control.reducers import load_events_of_types

    got = await load_events_of_types(await _store(), _SID, _TYPES, task_id="t_mine")

    ids = {e.payload["tool_call_id"] for e in got}
    assert "c_other" not in ids, "别的 task 的该被收窄掉——那正是这个界的用处"
    assert ids == {"c_mine", "c_orphan"}


async def test_unattributed_events_are_always_kept() -> None:
    """`task_id` 为空的**必须**取回。

    这条是整个收窄的安全阀。存量日志里的 capability 事件可能没有 task_id（列是 nullable），
    把它们当成「属于别人」排除掉，gateway 就会以为那次调用没跑过 → 静默重跑一个有副作用的
    工具。多取回几条只是多折几下，方向不对称。
    """
    from ctx_weft.core.control.reducers import fold_operations, load_events_of_types

    got = await load_events_of_types(await _store(), _SID, _TYPES, task_id="t_mine")
    facts = fold_operations(got)

    assert "c_orphan" in facts, "没归属的调用必须仍然被折出来"
    assert facts["c_orphan"].invoked is True


async def test_no_task_id_means_no_narrowing() -> None:
    """不传 `task_id` → 行为与从前逐字一致（全取）。"""
    from ctx_weft.core.control.reducers import load_events_of_types

    got = await load_events_of_types(await _store(), _SID, _TYPES)

    assert len(got) == 5


async def test_sql_store_agrees_with_the_in_memory_one(tmp_path) -> None:
    """两个内置 store 的收窄口径必须一致——不一致会让「本机能跑、线上重跑工具」。"""
    from ctx_weft.core.control.reducers import load_events_of_types
    from ctx_weft.providers.events.store.sql import open_sqlite_event_store

    async with open_sqlite_event_store(tmp_path / "events.sqlite") as sql:
        for n, (tcid, tid) in enumerate(
                [("c_mine", "t_mine"), ("c_other", "t_other"), ("c_orphan", None)], start=1):
            await sql.append(_cap(n, tcid, task_id=tid))

        got = await load_events_of_types(sql, _SID, _TYPES, task_id="t_mine")

    assert {e.payload["tool_call_id"] for e in got} == {"c_mine", "c_orphan"}


# ── 调用点真的传了它 ──────────────────────────────────────────────────────────


def test_both_capability_fold_call_sites_narrow_by_task() -> None:
    """两个调用点都要传 `task_id`——漏一个，那一条读就还是随会话长度增长。

    源码检查而非行为检查：要跑到这两处得搭起真 provider + 真 memory，而要钉的只是「传没传」。
    """
    import inspect

    from ctx_weft.core.loop.capability_gateway import CapabilityGateway
    from ctx_weft.core.loop.steps.reconcile import ReconcileStep

    for src in (inspect.getsource(ReconcileStep.execute),
                inspect.getsource(CapabilityGateway.invoke)):
        assert "CAP_FOLD_EVENT_TYPES" in src
        # 用 rindex：第一处是 import 行，真正的调用在后面
        i = src.rindex("CAP_FOLD_EVENT_TYPES")
        assert "task_id=" in src[i:i + 200], "这处 capability 折叠没按 task 收窄"
