"""段 recap 的待重跑账：折叠语义 + 它进快照往返后还在。

这个账从前是独立函数 `fold_pending_task_recap` + 恢复路径上一次按类型收窄的查询。查询随
会话长度线性增长（实测 1000 个 task 的会话：取回 1999 条折出 1 个，272ms / 5.1MB，每次
`/resume` 付一遍），所以它搬进了投影（`RunStateView.pending_recap`）——**折叠实现只此一
处**（`_apply`），不与旧函数并存：feat 把 HITL 移出投影的理由正是「两份口径不同的折叠会
漂移」。
"""

from ctx_weft.core.control.reducers import (
    deserialize_view,
    prune_view_for_snapshot,
    reduce_events,
    serialize_view,
)
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.events import Event, EventType


def _ev(type_, payload):
    return Event(
        id=generate_id("evt"), run_id=None, sequence=0, session_id="ses1",
        type=type_, timestamp=now_utc(), tenant_id="default",
        task_id=payload.get("task_id"), payload=payload,
    )


def _pending(events):
    return reduce_events(events, run_id="ses1").pending_recap


def test_started_without_done_is_pending():
    assert _pending([
        _ev(EventType.TASK_RECAP_STARTED,
            {"task_id": "t1", "boundary": "finish", "agent_id": "a1"}),
    ]) == {"t1": {"boundary": "finish", "agent_id": "a1"}}


def test_started_then_done_is_empty():
    assert _pending([
        _ev(EventType.TASK_RECAP_STARTED,
            {"task_id": "t1", "boundary": "finish", "agent_id": "a1"}),
        _ev(EventType.TASK_RECAP_DONE, {"task_id": "t1"}),
    ]) == {}


def test_last_write_wins_per_task():
    # 同 task 二次 started（如 recover 又崩一次）：以最后一次 boundary 为准
    assert _pending([
        _ev(EventType.TASK_RECAP_STARTED,
            {"task_id": "t1", "boundary": "interrupt", "agent_id": "a1"}),
        _ev(EventType.TASK_RECAP_DONE, {"task_id": "t1"}),
        _ev(EventType.TASK_RECAP_STARTED,
            {"task_id": "t1", "boundary": "finish", "agent_id": "a1"}),
    ]) == {"t1": {"boundary": "finish", "agent_id": "a1"}}


def test_key_comes_from_the_payload_not_the_event_header():
    """键取 payload 的 `task_id`，不是 `ev.task_id`——段边界上事件头可能是父任务。

    取错了会把待重跑的账记到父任务名下：恢复时按父任务重跑，那个子段的 memory 写永远
    补不上，而且不报错。这里刻意让两者不同，只有读对了才过。
    """
    ev = Event(
        id=generate_id("evt"), run_id=None, sequence=0, session_id="ses1",
        type=EventType.TASK_RECAP_STARTED, timestamp=now_utc(), tenant_id="default",
        task_id="t_parent",                                   # 事件头：父任务
        payload={"task_id": "t_child", "boundary": "finish", "agent_id": "a1"},
    )

    assert _pending([ev]) == {"t_child": {"boundary": "finish", "agent_id": "a1"}}


# ── 进快照往返 ────────────────────────────────────────────────────────────────


def test_pending_recap_survives_a_snapshot_round_trip():
    """它进投影的全部意义就在这条：写进 blob、读回来还在。

    掉了的话，恢复会认为「没有待重跑的 recap」，那段 memory 写就永远补不上——而且不报错。
    """
    view = reduce_events([
        _ev(EventType.TASK_RECAP_STARTED,
            {"task_id": "t_stuck", "boundary": "finish", "agent_id": "a1"}),
    ], run_id="ses1")

    back = deserialize_view(serialize_view(view))

    assert back.pending_recap == {"t_stuck": {"boundary": "finish", "agent_id": "a1"}}


def test_snapshot_prune_does_not_drop_pending_recap():
    """⚠️ 裁快照**不得**按 task 存活性过滤它。

    recap 跑在 finish 边界上，待重跑的那条往往属于一个已 FINISHED、已被
    `prune_view_for_snapshot` 裁掉的 task。跟着裁就等于把这个机制关掉。
    """
    view = reduce_events([
        _ev(EventType.TASK_RECAP_STARTED,
            {"task_id": "t_done_but_stuck", "boundary": "finish", "agent_id": "a1"}),
    ], run_id="ses1")
    assert view.tasks == {}, "前提：这个 task 根本不在投影里"

    pruned = prune_view_for_snapshot(view)

    assert pruned.pending_recap == {
        "t_done_but_stuck": {"boundary": "finish", "agent_id": "a1"}}


def test_old_blob_without_the_key_deserializes_to_empty():
    """v2 及更早的 blob 没有这个键 → 空 dict，不是 KeyError。

    那类 blob 已因 projection_version 不匹配被判不可用、走全量回放（全量回放会正确折出
    它），所以这条只钉形态完整性——但它得是**形态**完整，不能炸。
    """
    assert deserialize_view({"session_id": "ses1"}).pending_recap == {}
