"""冷准备后的 send_message：复用同一个 TaskManager，交付输入后再激活。"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import Event, EventType
from tests._event_helpers import append_one
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active


_TS = datetime(2026, 6, 13, tzinfo=UTC)


def _ev(seq: int, sid: str, type_: EventType, *, agent_id: str | None = None, **payload) -> Event:
    return Event(id=f"evt_{sid}_{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                 type=type_, timestamp=_TS, task_id=None, agent_id=agent_id, payload=payload)


def _runtime_with_registered_template():
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _noop_drain(self) -> None:
    return None


@pytest.mark.asyncio
async def test_send_message_reuses_a_cold_prepared_task_manager(
    monkeypatch,
) -> None:
    """冷准备建立 owner，send_message 必须复用它并在输入交付后激活。"""
    monkeypatch.setattr(TaskManager, "drain", _noop_drain)  # 不真的派发到 LLM

    rt = _runtime_with_registered_template()
    sid, aid = "A", "agt_1"
    await append_one(rt.event_store, _ev(1, sid, EventType.SESSION_CREATED,
                                     template_id="agent:tpl_echo", root_agent_id=aid))
    await append_one(rt.event_store, _ev(2, sid, EventType.AGENT_INSTANTIATED, agent_id=aid,
                                     template_id="agent:tpl_echo"))
    await append_one(rt.event_store, _ev(3, sid, EventType.AGENT_IDLE, agent_id=aid))

    n = await rebuild_all_active(rt)
    assert n == 1
    await initialize_all(rt)
    assert {a.agent_id for a in [item.data for item in await rt.list_agents()]} == {aid}
    prepared_tm = rt._task_managers[sid]
    assert not prepared_tm._queue.has_pending()
    assert rt._prepared_recoveries[sid][0] is prepared_tm

    await initialize_all(rt)
    handle = await rt.send_message(aid, "hi")

    assert handle.agent_id == aid
    assert handle.session_id == sid
    assert handle.task_id
    assert rt._task_managers[sid] is prepared_tm
    assert sid not in rt._prepared_recoveries
    tm = rt._task_managers[sid]
    assert tm.get_task(handle.task_id) is not None
    assert tm.get_task(handle.task_id).status not in ("FINISHED", "FAILED", "CANCELED")


@pytest.mark.asyncio
async def test_send_message_fast_path_does_not_rebuild_when_session_already_live(
    monkeypatch,
) -> None:
    """守卫「已经活着的会话不付这次重建」（终审 CRITICAL 1 的性能约束）：TM 已存活时
    `recover_agent` 不应被调用——即便再一次走的是 `_start_task_for_agent`（current
    task 手动置终态，模拟"上一条消息已经跑完，agent 又变回可新建 task 的状态"）。"""
    monkeypatch.setattr(TaskManager, "drain", _noop_drain)

    rt = _runtime_with_registered_template()
    sid, aid = "A", "agt_1"
    await append_one(rt.event_store, _ev(1, sid, EventType.SESSION_CREATED,
                                     template_id="agent:tpl_echo", root_agent_id=aid))
    await append_one(rt.event_store, _ev(2, sid, EventType.AGENT_INSTANTIATED, agent_id=aid,
                                     template_id="agent:tpl_echo"))
    await append_one(rt.event_store, _ev(3, sid, EventType.AGENT_IDLE, agent_id=aid))
    await rebuild_all_active(rt)

    # 先交付一条消息，激活已准备的 TM。
    await initialize_all(rt)
    first = await rt.send_message(aid, "hi")
    tm = rt._task_managers[sid]
    # 模拟这条消息对应的 task 已经跑完终态——下一条消息会再次落到
    # `_start_task_for_agent`（新建分支），而不是注入分支。
    tm.get_task(first.task_id).status = "FINISHED"

    called: list[str] = []
    orig_recover_agent = rt.recover_session

    async def _spy_recover_agent(agent_id, **kw):
        called.append(agent_id)
        return await orig_recover_agent(agent_id, **kw)

    monkeypatch.setattr(rt, "recover_session", _spy_recover_agent, raising=False)

    await initialize_all(rt)
    handle = await rt.send_message(aid, "second message")

    assert called == [], "session 已活，不该再走 recover_agent 重建"
    assert rt._task_managers[sid] is tm, "同一个 TM，不是被重新建了一份"
    assert handle.task_id and handle.task_id != first.task_id
