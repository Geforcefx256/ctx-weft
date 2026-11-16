
from tests._recovery import recover_fixture_session
"""Compare persisted memory with normal execution across both close-report orderings."""
import pytest

from ctx_weft.core.control.reducers import rebuild_view, reduce_events
from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events.store.sql import open_sqlite_event_store
from ctx_weft.providers.memory.sql.provider import open_sqlite_memory
from tests.integration.test_lifecycle_crash_boundaries import run_worker
from tests.integration.test_observe_application_recovery import recover_and_drain, assert_cold_recovery_is_noop
from tests.integration.test_observe_lifecycle_crashes import assert_lifecycles

_BASELINES = {}


async def memory_result(store, memory):
    view = await rebuild_view(store, "crash-session")
    task, = reduce_events([r.event for r in await store.read_range("crash-session")], "").tasks.values()
    root = view.sessions["crash-session"].root_agent_id
    ctx = ProviderContext(session_id="crash-session", tenant_id="default")
    address = MemoryAddress(session_id="crash-session", agent_id=root, task_id=task.id)
    raw = await load_records(memory, address, MemoryScope.TASK, ctx,
                                 kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.TOOL_AUDIT])
    finish = await load_records(memory, MemoryAddress(session_id="crash-session", agent_id=root), MemoryScope.AGENT, ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    finish = [r for r in finish if r.metadata.get("origin_task_id") == task.id]
    from ctx_weft.core.loop.task_memory_fold import task_record_group, task_record_kind
    assert sum(task_record_kind(r) == "recap" for r in finish) == 1
    assert sum(task_record_kind(r) == "finish" for r in finish) == 1
    assert len({task_record_group(r) for r in finish}) == 1
    assert all(r.metadata.get("finish_policy") for r in finish)
    assert any("reviewed" in str(r.content) for r in finish if task_record_kind(r) == "finish")
    assert not any("no process recap" in str(r.content) for r in finish)
    sources = {r.metadata["finish_policy"]["final_reply_record_id"] for r in finish
               if r.metadata["finish_policy"].get("final_reply_record_id")}
    source_records = await memory.read_records(list(sources), ctx)
    assert {r.id for r in source_records} == sources
    for record in source_records:
        assert record.role == "assistant" and record.address.task_id == task.id
        assert str(record.content) == task.outputs
    def policy(record):
        return {k: ("<final_reply>" if k == "final_reply_record_id" else v)
                for k, v in record.metadata.get("finish_policy", {}).items() if k != "observe_id"}
    def canonical(records):
        return [(r.kind, r.role, str(r.content).replace(task.id, "<task>"),
                 r.metadata.get("final_reply", False), policy(r)) for r in records]
    return {"raw": canonical(raw), "finish": canonical(finish), "outputs": task.outputs}


async def normal_result(tmp_path_factory, mode):
    if mode not in _BASELINES:
        path = tmp_path_factory.mktemp(mode + "_normal")
        await run_worker(path, mode, "none")
        async with open_sqlite_event_store(path / "events.db") as store:
            async with open_sqlite_memory(path / "memory.db") as memory:
                result = await memory_result(store, memory)
                rows = await store.read_range("crash-session")
                assert_lifecycles(rows)
                recap = next(r for r in rows if r.event.type == EventType.OBSERVE_COMPLETED
                             and r.event.payload["action"] == "recap")
                finalized = next(r for r in rows if r.event.type == EventType.TASK_FINALIZED)
                assert (recap.position < finalized.position) == ("early" in mode)
                if mode.endswith("short"):
                    assert any(r[1] == "assistant" for r in result["raw"])
                    assert not any(r[3] for r in result["finish"])
                    assert all(r[4]["preserve_raw"] for r in result["finish"])
                else:
                    assert all(r[1] == "user" for r in result["raw"])
                    assert sum(r[3] for r in result["finish"]) == 1
                    assert not any(r[4]["preserve_raw"] for r in result["finish"])
                _BASELINES[mode] = result
    return _BASELINES[mode]


CASES = [(order, size, snapshot, cut)
         for order in ("early", "late") for size in ("short", "long") for snapshot in (False, True)
         for cut in ("summary_completed", "finish_memory", "report_replaced", "before_summary_applied",
                     "summary_applied", "task_finalized", "recovery_report_replaced")]
CASES += [(order, "long", snapshot, "raw_folded")
          for order in ("early", "late") for snapshot in (False, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("order,size,snapshot,cut", CASES)
async def test_report_and_raw_match_normal_execution(tmp_path, tmp_path_factory, order, size, snapshot, cut):
    mode = f"handoff_{order}_{size}"
    expected = await normal_result(tmp_path_factory, mode)
    initial_cut = "summary_completed" if cut == "recovery_report_replaced" else cut
    await run_worker(tmp_path, mode, initial_cut)
    if cut == "recovery_report_replaced":
        await run_worker(tmp_path, mode, "report_replaced", recovering=True)
    async with open_sqlite_event_store(tmp_path / "events.db") as store:
        async with open_sqlite_memory(tmp_path / "memory.db") as memory:
            before = await rebuild_view(store, "crash-session")
            had_report = any(info.get("completion") for info in before.pending_observe.values())
            rt, root = await recover_and_drain(store, memory, mode, snapshot)
            assert rt._llm.actor_count == 0
            if had_report:
                assert rt._llm.judgements == 0
            actual = await memory_result(store, memory)
            assert actual == expected, (mode, cut, actual, expected)
            rows = await store.read_range("crash-session")
            assert_lifecycles(rows)
            requests = [r.event.payload for r in rows if r.event.type == EventType.OBSERVE_REQUESTED]
            assert len(requests) == 1 and requests[0]["verdict"]["task_outcome"] == "success"
            assert not any("parent_observe_id" in r.event.payload for r in rows)
            for kind in (EventType.TASK_FINALIZED, EventType.TASK_FINISHED):
                assert sum(r.event.type == kind for r in rows) == 1
            recap_id = next(r.event.payload["observe_id"] for r in rows
                            if r.event.type == EventType.OBSERVE_COMPLETED
                            and r.event.payload["action"] == "recap")
            applied = next(r.position for r in rows if r.event.type == EventType.OBSERVE_APPLIED
                           and r.event.payload["observe_id"] == recap_id)
            finalized = next(r.position for r in rows if r.event.type == EventType.TASK_FINALIZED)
            assert finalized < applied
            view = await rebuild_view(store, "crash-session")
            assert not view.pending_observe and not view.open_runs
            head = await store.committed_head("crash-session")
            await recover_fixture_session(rt, root, session_id="crash-session")
            assert await store.committed_head("crash-session") == head
            await assert_cold_recovery_is_noop(store, memory, mode, root)
            assert await memory_result(store, memory) == actual
