"""HITL 冷应答解 tenant 的读取成本（P0，2026-09-19）。

`_tenant_for_session` 原本是 `read_by_session(session_id)`——把整条事件流读出来，只为拿
第一条上的一个字段。它在 HITL **冷应答**路径上，长会话里每答一次就付一次 O(全部事件)。

现在按类型收窄到 `SessionCreated`（会话的 tenant 就记在它身上，每会话恰一条，走
`(session_id, type)` 索引）。全量读退为兜底，只在 `SessionCreated` 缺失或不带 tenant
的异常数据上才走——那条兜底保住了改造前「任何一条事件的 tenant 都算」的语义。

这里钉两件事：正常路径**不再**全量读，以及兜底路径仍然管用。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import Event, EventType
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 9, 19, tzinfo=UTC)
_SID = "ses_tenant"
_TENANT = "acme"


def _runtime():
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    return make_runtime(agent_provider=resolver)


def _ev(seq: int, type_: str, *, tenant: str = _TENANT, **payload) -> Event:
    return Event(
        id=f"evt_{seq:04d}", run_id="r1", sequence=seq, session_id=_SID, type=type_,
        timestamp=_TS, tenant_id=tenant, payload=payload,
    )


def _count_full_reads(runtime) -> list[int]:
    """给 `read_by_session` 装一个计数器（返回单元素列表当计数盒）。"""
    box = [0]
    store = runtime.event_store
    original = store.read_by_session

    async def counted(session_id: str):
        box[0] += 1
        return await original(session_id)

    store.read_by_session = counted     # type: ignore[method-assign]
    return box


async def test_tenant_comes_from_session_created_without_a_full_read() -> None:
    runtime = _runtime()
    # 一条正常会话：SessionCreated + 一串后续事件
    await runtime.event_store.append(_ev(
        1, EventType.SESSION_CREATED, user_prompt="go",
        template_id="agent:tpl_echo", root_agent_id="agt_root"))
    for seq in range(2, 12):
        await runtime.event_store.append(_ev(seq, EventType.RUN_STARTED))

    reads = _count_full_reads(runtime)
    assert await runtime._tenant_for_session(_SID) == _TENANT
    assert reads[0] == 0, "正常路径不该再把整条事件流读出来"


async def test_falls_back_to_a_full_read_when_session_created_is_missing() -> None:
    """日志残缺（没有 SessionCreated）时仍要解得出来——这是改造前的语义，不能丢。"""
    runtime = _runtime()
    # 故意不写 SessionCreated，只有后续事件带着 tenant
    for seq in range(1, 5):
        await runtime.event_store.append(_ev(seq, EventType.RUN_STARTED))

    reads = _count_full_reads(runtime)
    assert await runtime._tenant_for_session(_SID) == _TENANT
    assert reads[0] == 1, "兜底才允许全量读，且只读一次"


async def test_unknown_session_falls_back_to_default_without_raising() -> None:
    """解不出一律 `default`，绝不抛——这条在 HITL 应答路径上，抛错会卡住人类应答。"""
    runtime = _runtime()
    assert await runtime._tenant_for_session("ses_never_seen") == "default"


async def test_live_task_manager_short_circuits_before_any_read() -> None:
    """热路径（活 owner TM 的 session 上就有 tenant）连类型收窄那一次查询都不该发。"""
    from types import SimpleNamespace

    runtime = _runtime()
    await runtime.event_store.append(_ev(
        1, EventType.SESSION_CREATED, user_prompt="go",
        template_id="agent:tpl_echo", root_agent_id="agt_root"))

    hits = [0]
    original = runtime._read_session_events_of_types

    async def counted(session_id: str, types):
        hits[0] += 1
        return await original(session_id, types)

    runtime._read_session_events_of_types = counted   # type: ignore[method-assign]
    reads = _count_full_reads(runtime)

    runtime._task_managers[_SID] = SimpleNamespace(   # type: ignore[assignment]
        session=SimpleNamespace(tenant_id="hot-tenant"))
    assert await runtime._tenant_for_session(_SID) == "hot-tenant"
    assert (hits[0], reads[0]) == (0, 0), "热路径是纯内存查表，不该碰事件库"
