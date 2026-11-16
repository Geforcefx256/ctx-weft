"""存量日志里的 `SessionPausedHitl` 折进当前词表的 `WAITING`（Task 7 起，两种 form 都是）。

`SessionStatus` 值域自 Task 9 起不再含 `PAUSED` / `PAUSED_HITL`——值域本身由
`tests/unit/test_session_status_domain.py` 钉住。"""

from __future__ import annotations

import pytest

from ctx_weft.protocols.events import EventType



# `SessionPausedHitl` 不再由 core 发出——「等的是面板还是一句话」改由 pending 集合的
# delivery 承载，host 直接从 `list_pending_hitl(...)` 的 `delivery` 字段读（2026-09-04
# Task 14 起 core 不再替 host 派生这个串；delivery 相关断言见 `test_hitl_recovery_v2.py`
# 的 delivery 用例）。旧事件仍在存量日志里，故下面的 reducer 投影分支保留并继续被测。


def _paused_event(session_id: str, form: str):
    from ctx_weft.protocols.events import Event
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.core.utils.ids import generate_id
    return Event(
        id=generate_id("evt"), run_id=None, sequence=0, session_id=session_id,
        type=EventType.SESSION_PAUSED_HITL, timestamp=now_utc(), task_id=None,
        payload={"capability_id": "whatever", "form": form},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["wait", "question", "approval"])
async def test_session_paused_hitl_folds_to_waiting_regardless_of_form(form) -> None:
    """L 档折叠目标是当前词表：PAUSED/PAUSED_HITL 两档合并进单一 WAITING
    （Task 9 已把两者从 SessionStatus 值域里删掉），故不再按 form 分流。"""
    from ctx_weft.core.control.reducers import reduce_events
    view = reduce_events([_paused_event("s1", form)], run_id="r1")
    assert not hasattr(view, "session_status")


# ── Task 7（会话状态所有权重构）之后：HITL_OPENED / HITL_RESOLVED 不再驱动投影 ──
#
# 上面这段旧注释描述的是 Task 6 引入、Task 7 收回的行为：`HITL_OPENED` 分支已整个删除，
# `HITL_RESOLVED`（新模型）也已从会话回 RUNNING 的 elif 元组里移出——「等人」这件事的
# 会话状态改由 SessionRegistry 承载（docs/events-v2.md §2.1.1）。正面用例见
# `tests/unit/test_domain_facts_do_not_write_session_status.py`
# （`test_hitl_opened_no_longer_pauses_the_session` /
# `test_hitl_resolved_no_longer_returns_the_session_to_running`）。
# 下面只留一条：确认 HITL_RESOLVED 对已终结的会话仍是无操作（无论是不是因为它已被削掉写入）。


def _hitl_event(session_id: str, etype, payload: dict):
    from ctx_weft.protocols.events import Event
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.core.utils.ids import generate_id
    return Event(
        id=generate_id("evt"), run_id=None, sequence=0, session_id=session_id,
        type=etype, timestamp=now_utc(), task_id="t1", payload=payload,
    )


def _opened(session_id: str, delivery: dict):
    return _hitl_event(session_id, EventType.HITL_OPENED,
                       {"hitl_id": "h1", "form": "whatever", "delivery": delivery})


@pytest.mark.asyncio
async def test_hitl_resolved_does_not_overwrite_a_terminal_status() -> None:
    """仅当仍处暂停态才掰回 RUNNING——迟到的 HITL_RESOLVED 不得复活一个已收尾的会话。"""
    from ctx_weft.core.control.reducers import reduce_events
    from ctx_weft.protocols.events import Event
    from ctx_weft.core.utils.clock import now_utc
    from ctx_weft.core.utils.ids import generate_id
    finished = Event(
        id=generate_id("evt"), run_id=None, sequence=0, session_id="s1",
        type=EventType.SESSION_FINISHED, timestamp=now_utc(), task_id=None,
        payload={"final_status": "SUCCEEDED"},
    )
    view = reduce_events(
        [_opened("s1", {"kind": "no_resume"}), finished,
         _hitl_event("s1", EventType.HITL_RESOLVED,
                     {"hitl_id": "h1", "outcome": "accepted"})],
        run_id="r1")
    assert not hasattr(view, "session_status")
