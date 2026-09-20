"""快照裁剪：blob 的 `tasks` 只存活闭包（spec: snapshot-recovery v2，2026-09-19）。

`tasks` 是 blob 里唯一随会话历史线性增长的部分，而 `SnapshotWriter.on_event` 在
`EventBus.emit()` 里内联执行 —— 所以这里钉的是「blob 大小只跟当下有多少活有关」。

覆盖：
1. 三类闭包各自被保留（活 / 活的直接子任务 / 被活 task 的 dag_deps 引用），纯历史被裁。
2. 第 3 类瘦身成 id+session_id+status，且状态**如实**保留（FAILED/CANCELED 不能丢）。
3. 幂等：裁两次 == 裁一次（增量链反复以裁过的 blob 为基底）。
4. 有界性：历史里多堆 N 个已终态 task，裁完的条数不变。
5. `agents` 不裁，且历史 task 被裁之后 `_rebuild_agents` 不会把老 agent 的
   spawn_depth 重置回 0（这条是裁剪能不能做的前提）。
6. `tasks_total` 不受裁剪影响 —— 恢复路径的空投影闸门靠它区分「全部完工」与「坏投影」。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ctx_weft.core.control.reducers import (
    apply_events,
    deserialize_view,
    prune_view_for_snapshot,
    reduce_events,
    serialize_view,
)
from ctx_weft.protocols.events import Event, EventType

_T0 = datetime(2026, 9, 19, tzinfo=timezone.utc)


def _ev(seq: int, type_: str, task_id: str | None = None, **payload) -> Event:
    return Event(
        id=f"evt_{seq:04d}", run_id="r1", sequence=seq, session_id="s1", type=type_,
        timestamp=_T0 + timedelta(seconds=seq), tenant_id="default",
        task_id=task_id, agent_id="agt_root", payload=payload, metadata={},
    )


def _created(seq: int, tid: str, status: str = "PENDING", *, parent: str = "",
             deps: list[str] | None = None, agent: str = "agt_root",
             creator: str = "agt_root", use_subagent: bool = False) -> Event:
    return _ev(seq, EventType.TASK_CREATED, task_id=tid, task={
        "id": tid, "status": status, "title": f"T-{tid}", "parent_task_id": parent,
        "assigned_agent_id": agent, "creator_agent_id": creator,
        "dag_deps": deps or [], "settings": {"use_subagent": use_subagent},
        # 全文字段：裁剪要省的正是这些
        "user_prompt": "P" * 400, "description": "D" * 200,
    })


def _base_events() -> list[Event]:
    """一个含四类 task 的会话：活 / 活的子任务 / 被依赖的前驱 / 纯历史。"""
    return [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="agt_root"),
        _created(2, "live", "ACTIVE"),
        _created(3, "child", parent="live"),
        _created(4, "dep"),
        _created(5, "waiter", deps=["dep"]),
        _created(6, "ancient"),
        _ev(7, EventType.TASK_FINISHED, task_id="child", outputs="child deliverable"),
        _ev(8, EventType.TASK_FINISHED, task_id="dep"),
        _ev(9, EventType.TASK_FINISHED, task_id="ancient"),
    ]


def test_prune_keeps_the_three_closure_classes_and_drops_pure_history() -> None:
    view = reduce_events(_base_events(), run_id="s1")
    assert sorted(view.tasks) == ["ancient", "child", "dep", "live", "waiter"]

    pruned = prune_view_for_snapshot(view)

    # live / waiter 非终态；child 是 live 的直接子任务；dep 被 waiter 的 dag_deps 引用
    assert sorted(pruned.tasks) == ["child", "dep", "live", "waiter"]
    # ancient 终态、非任何活 task 的子任务、也不被依赖 → 无消费者，丢
    assert "ancient" not in pruned.tasks


def test_live_and_child_keep_full_fields() -> None:
    """第 1/2 类不降级：act guidance 的已完成清单要 title + outputs。"""
    pruned = prune_view_for_snapshot(reduce_events(_base_events(), run_id="s1"))

    live = pruned.tasks["live"]
    assert live.user_prompt == "P" * 400        # driver 重排时立刻要读
    assert live.title == "T-live"

    child = pruned.tasks["child"]
    assert child.title == "T-child"             # _task_label
    assert child.outputs == "child deliverable"  # _subtask_result_snippet 的首选来源
    assert child.status == "FINISHED"            # restore 的 SUSPENDED 闸门


def test_dependency_class_is_slimmed_but_keeps_status() -> None:
    """第 3 类只留 id/session_id/status —— 消费者只看状态。"""
    pruned = prune_view_for_snapshot(reduce_events(_base_events(), run_id="s1"))

    dep = pruned.tasks["dep"]
    assert dep.id == "dep" and dep.session_id == "s1"
    assert dep.status == "FINISHED"              # TaskQueue.seed_succeeded 的判据
    # 全文字段被省掉 —— 这一类唯一的用途是回答「前驱成了没有」
    assert dep.user_prompt == ""
    assert dep.title == ""
    assert dep.description == ""


def test_dependency_class_keeps_failed_and_canceled_verbatim() -> None:
    """**不能只留 FINISHED 的**：漏掉 FAILED/CANCELED 的前驱会让
    `TaskManager._find_blocked_forever` 认不出永久阻塞，后继就永远卡在队列里。"""
    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="agt_root"),
        _created(2, "bad"),
        _created(3, "gone"),
        _created(4, "waiter", deps=["bad", "gone"]),
        _ev(5, EventType.TASK_FAILED, task_id="bad", error_message="boom"),
        _ev(6, EventType.TASK_CANCELED, task_id="gone", reason="user"),
    ]
    pruned = prune_view_for_snapshot(reduce_events(events, run_id="s1"))

    assert pruned.tasks["bad"].status == "FAILED"
    assert pruned.tasks["gone"].status == "CANCELED"


def test_prune_is_idempotent() -> None:
    """增量链会反复以裁过的 blob 为基底，所以裁剪必须幂等。"""
    once = prune_view_for_snapshot(reduce_events(_base_events(), run_id="s1"))
    twice = prune_view_for_snapshot(once)

    assert sorted(twice.tasks) == sorted(once.tasks)
    assert twice.tasks["dep"].status == once.tasks["dep"].status
    assert twice.tasks["child"].outputs == once.tasks["child"].outputs


def test_blob_size_does_not_grow_with_finished_history() -> None:
    """有界性：再堆 40 个已终态的历史 task，裁完的条数一个不多。"""
    events = _base_events()
    seq = 100
    for i in range(40):
        events.append(_created(seq, f"hist_{i}"))
        events.append(_ev(seq + 1, EventType.TASK_FINISHED, task_id=f"hist_{i}"))
        seq += 2

    view = reduce_events(events, run_id="s1")
    assert len(view.tasks) == 45                      # 全量投影确实涨了
    pruned = prune_view_for_snapshot(view)
    assert sorted(pruned.tasks) == ["child", "dep", "live", "waiter"]

    # 且 blob 里不再留着那些历史 task 的 prompt 全文
    blob = serialize_view(pruned)
    assert "hist_0" not in blob["tasks"]
    assert len(blob["tasks"]) == 4


def test_pruning_history_does_not_reset_agent_spawn_depth() -> None:
    """裁剪能成立的前提：`agents` 不裁，且 `_rebuild_agents` 不会把老 agent 的树形字段
    重置回 0。

    `_rebuild_agents` 是从 `view.tasks` 推 spawn_depth 的——历史 task 被裁之后它推不出
    老 agent 的深度。安全的理由是它只覆盖**能在 tasks 里找到**的 agent：循环碰不到的
    agent 原样保留 blob 里那份。这条一旦破掉，子 agent 的 spawn_depth 会在某次恢复后
    悄悄归零，spawn 深度限制随之失效。
    """
    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="agt_root"),
        # 子 agent 的深度只有这个 task 知道（use_subagent + creator=root）
        _created(2, "spawned", agent="agt_sub", creator="agt_root", use_subagent=True),
        _ev(3, EventType.TASK_FINISHED, task_id="spawned"),
        _created(4, "live", "ACTIVE"),
    ]
    view = reduce_events(events, run_id="s1")
    assert view.agents["agt_sub"].spawn_depth == 1
    assert view.agents["agt_sub"].parent_agent_id == "agt_root"

    # 裁剪 → 落盘 → 读回 → 再 apply 一条 delta（末尾会调 _rebuild_agents）
    pruned = prune_view_for_snapshot(view)
    assert "spawned" not in pruned.tasks, "前置条件：那个 task 确实被裁掉了"
    restored = deserialize_view(serialize_view(pruned))
    after = apply_events([_ev(5, EventType.RUN_FINISHED, outcome="completed")], restored)

    assert after.agents["agt_sub"].spawn_depth == 1, "老 agent 的深度不得被重置回 0"
    assert after.agents["agt_sub"].parent_agent_id == "agt_root"


def test_tasks_total_survives_pruning_and_roundtrip() -> None:
    """恢复路径的空投影闸门靠 `tasks_total` 区分「全部完工」与「坏投影」——
    裁剪与往返都不能动它。"""
    view = reduce_events(_base_events(), run_id="s1")
    assert view.tasks_total == 5

    pruned = prune_view_for_snapshot(view)
    assert pruned.tasks_total == 5
    assert deserialize_view(serialize_view(pruned)).tasks_total == 5


def test_all_tasks_finished_prunes_to_empty_but_still_counts() -> None:
    """一个全部完工的会话裁完是空 tasks —— 与「从来没有过 task」在集合上不可区分，
    只有 `tasks_total` 分得开。恢复闸门因此不能用 `not all_tasks` 判。"""
    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="agt_root"),
        _created(2, "only"),
        _ev(3, EventType.TASK_FINISHED, task_id="only"),
    ]
    pruned = prune_view_for_snapshot(reduce_events(events, run_id="s1"))

    assert pruned.tasks == {}
    assert pruned.tasks_total == 1          # ← 闸门据此知道「这不是坏投影」

    empty = reduce_events(events[:1], run_id="s1")
    assert empty.tasks == {} and empty.tasks_total == 0   # 真正的空投影


async def test_pruning_does_not_affect_memory_recall_of_finished_tasks() -> None:
    """裁掉的 task **照样**能把它的对话召回进上下文——两套存储互不相干。

    「已结束的 task 出现在 LLM 上下文里」走的是 memory 的跨 task 聚合：
    `AgentRecallSource` 用半址 `MemoryAddress(session_id, agent_id)` + `MemoryScope.TASK`
    （`task_id=None`）把该 agent 名下**所有** task 的记录一次取回。那是 `memory_events`，
    与 `snapshots.state_blob` 是两套存储，裁剪一个字节都不碰它。

    快照里的 `tasks` 投影喂的是 `TaskManager`，消费者是**调度**（重排 / DAG 闸门 / 父子
    唤醒）和 guidance 的清单，不是对话内容。这条测试把这个边界钉住：投影裁到只剩活
    task，召回仍拿得到已完工 task 的全部对话。
    """
    from ctx_weft.protocols import (
        MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext,
    )
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    events = [
        _ev(1, EventType.SESSION_CREATED, user_prompt="go",
            template_id="agent:tpl", root_agent_id="agt_root"),
        _created(2, "old"),
        _ev(3, EventType.TASK_FINISHED, task_id="old", outputs="done"),
        _created(4, "now", "ACTIVE"),
    ]
    pruned = prune_view_for_snapshot(reduce_events(events, run_id="s1"))
    assert sorted(pruned.tasks) == ["now"], "前置条件：old 确实被裁出了投影"

    mem = InMemoryMemoryProvider()
    ctx = ProviderContext(session_id="s1", tenant_id="default")
    for tid in ("old", "now"):
        await mem.ingest(MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=MemoryAddress(session_id="s1", task_id=tid, agent_id="agt_root"),
            content=f"[{tid}] 结论", role="assistant", timestamp=_T0), ctx)

    # AgentRecallSource 的那个半址查询
    recalled = [r.content for r in await mem.load_view(
        MemoryAddress(session_id="s1", agent_id="agt_root"), MemoryScope.TASK, ctx)]

    assert "[old] 结论" in recalled, "被裁出投影的 task，其对话仍必须可召回"
    assert "[now] 结论" in recalled
