"""终审 CRITICAL 2 回归：一条折叠自 legacy `HITL_REQUIRED` 事件的 `PendingHitl` 永远
没有 `agent_id`（该事件从未持久化它，见 `_reply_turn_agent_id` 与
`tests/unit/test_cold_resume_agent_scope.py`）。`_resume_after_hitl` 此前按
`req.agent_id` 路由到 `recover_agent`：`recover_agent("")` 必然 `record_of("")` miss
→ 而键为 `""` 的记录无论怎么装填都不会出现 → 最终抛错——而这一步发生在
`HitlService._commit` 已经把这条 HITL 判成终局**之后**，没有第二次机会，会话永久
卡住。

修法：`agent_id` 为空时回退到 `req.session_id`（`PendingHitl` 两个字段本来就都有），
与换轴前 `recover_session(session_id)` 同一条路。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import HitlReply
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests.unit._legacy_recover import rebuild_all_active
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio

_TS = datetime(2026, 6, 13, tzinfo=UTC)


def _runtime_for_legacy_resume():
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _seed_legacy_wait_for_user_session(rt, sid: str, aid: str, tid: str, hid: str) -> None:
    """种一条「actor 纯文本暂停等用户回复」的旧版事件流：`HITL_REQUIRED` 的
    `form="wait"` 折成 `UserTurnDelivery`（`_legacy_delivery`），且**不带 agent_id**
    ——既不在事件信封上，也不在 payload 里，模拟升级前从未持久化过这个字段的存量
    数据。task 停在 `SUSPENDED`，与 `test_hitl_ask_human_cold.py` 的搭台口径一致。
    """

    def ev(seq: int, type_: EventType, *, agent_id: str | None = None, **payload) -> Event:
        return Event(id=f"evt_{sid}_{seq:04d}", run_id="r1", sequence=seq, session_id=sid,
                     type=type_, timestamp=_TS, task_id=tid, agent_id=agent_id, payload=payload)

    await append_one(rt.event_store, ev(1, EventType.SESSION_CREATED, task_id=None,
                                    template_id="agent:tpl_echo", root_agent_id=aid,
                                    user_prompt="do it"))
    await append_one(rt.event_store, ev(2, EventType.TASK_CREATED, task={
        "id": tid, "status": "PENDING", "title": "T1",
        "assigned_agent_id": aid, "creator_agent_id": aid}))
    await append_one(rt.event_store, ev(3, EventType.TASK_STARTED, assigned_agent_id=aid))
    # legacy：无 agent_id，既不在信封上也不在 payload 里。
    await append_one(rt.event_store, ev(4, EventType.HITL_REQUIRED, hitl_id=hid, form="wait",
                                    context="plain_text"))
    await append_one(rt.event_store, ev(5, EventType.TASK_SUSPENDED))


async def test_legacy_cold_hitl_reply_resumes_via_session_id_fallback() -> None:
    rt = _runtime_for_legacy_resume()
    sid, aid, tid, hid = "ses_legacy", "agt_root", "tsk_1", "hit_1"
    await _seed_legacy_wait_for_user_session(rt, sid, aid, tid, hid)

    await rebuild_all_active(rt)  # ALM.load() 装填了 aid，但键是 aid，不是 ""

    pending = rt.hitl_registry.get(hid)
    assert pending is not None and pending.agent_id == "", (
        "搭台前提：legacy 折叠出的 agent_id 必须是空串，否则这条测试没有测到本 bug"
    )

    reply = HitlReply(hitl_id=hid, outcome="accepted", agent_id="", message="ship it")

    # 这是本次修复要守住的行为：不抛 AgentNotFound，且真的把会话续上。
    view = await rt.reply_to_hitl(reply)

    assert view is not None
    assert rt.hitl_registry.list_pending(session_id=sid) == [], (
        "HITL 应已终局，不再挂在 pending 里"
    )
    assert sid in rt._task_managers, (
        "回退到 session_id 之后必须真的把这个 session 的 TM 续跑起来"
    )


async def test_legacy_cold_hitl_reply_without_fix_would_raise_agent_not_found() -> None:
    """锁死 bug 本身的可复现性：直接调用未走回退的 `recover_agent("")`（模拟修复前
    `_resume_after_hitl` 的行为）必须撞见 `AgentNotFound`——证明 `_recover_after_
    cold_hitl` 的回退分支不是无的放矢。"""
    from ctx_weft.core.models.errors import AgentNotFound

    rt = _runtime_for_legacy_resume()
    sid, aid, tid, hid = "ses_legacy2", "agt_root2", "tsk_2", "hit_2"
    await _seed_legacy_wait_for_user_session(rt, sid, aid, tid, hid)
    await rebuild_all_active(rt)

    with pytest.raises(AgentNotFound):
        await rt.recover_agent("", resumed_task_id=tid, hitl_id=hid)
