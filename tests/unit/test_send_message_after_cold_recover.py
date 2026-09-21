"""终审 CRITICAL 1 回归：`send_message` 到一个只被 `recover()` 装填过 ALM、从没真正
跑起来过（没有 TaskManager）的 agent，必须自愈建出 TM 并把消息塞进去，不是裸
`KeyError`。

复现手法照抄 `tests/unit/test_recover_loads_agents.py`：直接把最小事件集写进
event store 再调 `recover()`，模拟「进程重启，这个 agent 在崩溃前就已经 idle，
之后再也没被 `/resume` 或冷 HITL 应答碰过」——`recover()` 只装填
`AgentLifecycleManager`（`_load_agents_of`），从不建 `_task_managers`（`recover()`
自己的 docstring："这里 nothing drains/runs"）。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.protocols.events import Event, EventType
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio

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


async def test_send_message_self_heals_a_cold_recovered_agents_missing_task_manager(
    monkeypatch,
) -> None:
    """seed SESSION_CREATED + AGENT_INSTANTIATED + AGENT_IDLE for "agt_1" in session "A";
    recover() 装填 ALM 但不建 TM; send_message("agt_1", "hi") 此前会撞见裸
    ``KeyError: 'A'``——现在必须成功返回一个可用的 TurnHandle，且自愈建出的 TM
    真的挂进了 `_task_managers`。"""
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
    assert {a.agent_id for a in rt.list_agents()} == {aid}
    assert sid not in rt._task_managers, "recover() 不该建 TM——这是本 bug 的前提条件"

    handle = await rt.send_message(aid, "hi")

    assert handle.agent_id == aid
    assert handle.session_id == sid
    assert handle.task_id
    assert sid in rt._task_managers, "send_message 必须自愈建出并保留一个活的 TM"
    tm = rt._task_managers[sid]
    assert tm.get_task(handle.task_id) is not None
    assert tm.get_task(handle.task_id).status not in ("FINISHED", "FAILED", "CANCELED")


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

    # 先自愈一次，把 TM 建起来、留活。
    first = await rt.send_message(aid, "hi")
    tm = rt._task_managers[sid]
    # 模拟这条消息对应的 task 已经跑完终态——下一条消息会再次落到
    # `_start_task_for_agent`（新建分支），而不是注入分支。
    tm.get_task(first.task_id).status = "FINISHED"

    called: list[str] = []
    orig_recover_agent = rt.recover_agent

    async def _spy_recover_agent(agent_id, **kw):
        called.append(agent_id)
        return await orig_recover_agent(agent_id, **kw)

    monkeypatch.setattr(rt, "recover_agent", _spy_recover_agent, raising=False)

    handle = await rt.send_message(aid, "second message")

    assert called == [], "session 已活，不该再走 recover_agent 重建"
    assert rt._task_managers[sid] is tm, "同一个 TM，不是被重新建了一份"
    assert handle.task_id and handle.task_id != first.task_id
