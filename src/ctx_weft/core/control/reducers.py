"""Event reducers：把 event 序列重建成 state。

Phase 6 §6.6。核心是 reduce(events) → RunStateView。
RunStateView.sessions / .tasks 包含完整的 Session/Task 投影。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from ctx_weft.core.utils.content import (
    content_from_jsonable,
    content_to_jsonable,
)
from ctx_weft.core.control.types import AgentView, RunStateView, SessionView, TaskView
from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HITL_STAGE_TOOL, PendingHitl
from ctx_weft.core.hitl.snapshot import HitlSnapshot
from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES, WAITING, TaskStatus
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.protocols.hitl import (
    HITL_FORM_QUESTION,
    HITL_FORM_WAIT,
    HITL_OUTCOME_ACCEPTED,
    HITL_OUTCOME_CANCELLED,
    HITL_OUTCOME_REJECTED,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    Delivery,
    HitlDecision,
    NoResumeDelivery,
    ToolResultDelivery,
    UserTurnDelivery,
)

logger = logging.getLogger(__name__)

# 事件类型 → 任务状态的投影映射。
#
# 留在 core 而不进 `protocols/events.py` 的理由（spec 2026-08-27 三层划界）：这是 core
# 的**投影逻辑**，不是 host 要按之编程的契约；且值类型 `TaskStatus` 来自
# `core/state/models.py`，放进 protocols 会让协议层反向依赖 core。
#
# 落在本文件而不是单独模块：`reduce_events` 是它唯一的消费者，就在下方几十行处用到。
TASK_STATUS_BY_EVENT: dict[EventType, TaskStatus] = {
    EventType.TASK_STARTED: "ACTIVE",
    EventType.TASK_SUSPENDED: "SUSPENDED",
    EventType.TASK_AWAITING_HUMAN: "AWAITING_HUMAN",
    # 被打断由 task 域的 TASK_INTERRUPTED 写；run 域的 RUN_INTERRUPTED 只说
    # 「这次执行死了」，不写 task 状态——那次 run 死了不等于 task 停在 INTERRUPTED
    # （还能重试的走 TASK_REQUEUED → PENDING）。见 docs/events-v2.md §2.3 / §2.4。
    EventType.TASK_INTERRUPTED: "INTERRUPTED",
    EventType.TASK_FINISHED: "FINISHED",
    EventType.TASK_FAILED: "FAILED",
    EventType.TASK_CANCELED: "CANCELED",
    # ACTIVE 的正主是 TASK_STARTED（TM 派发时发、回填 assigned_agent_id）。TaskResumed
    # 只说「不再被子任务挡住了」——解挂到真正派发之间那段窗口 task 在队列里，不是在跑
    # （D5：此前这里映射 ACTIVE 是在镜像 `_try_resume_parent` 入队前抢跑置位的缺陷）。
    EventType.TASK_RESUMED: "PENDING",
    EventType.TASK_REQUEUED: "PENDING",
    # 见下方专属分支（与 TASK_REQUEUED 一样需要清 outputs，这里仅供直接查表的消费方用）。
    EventType.TASK_HUMAN_RESOLVED: "PENDING",
}

# L 档翻译表：存量日志里 `SessionStatusChanged.new_status` 可能带旧词表的值。
# 旧模型按 form 分 PAUSED / PAUSED_HITL 两档，新模型合并成一个 WAITING
# （docs/events-v2.md §2.1.3）——与 SESSION_PAUSED_HITL 分支同口径。
# 未列出的值仍在当前值域内，原样通过。
_LEGACY_SESSION_STATUS: dict[str, str] = {
    "PAUSED": WAITING,
    "PAUSED_HITL": WAITING,
}

# HITL 各终态事件。`HITL_FOLD_EVENT_TYPES`（见下方 v2 折叠段）与 `RECOVERY_EVENT_TYPES`
# 共用它，语义与 `fold_hitl_snapshot` 一致（单一真相）。
_HITL_RESOLVE_TYPES = (
    EventType.HITL_APPROVED, EventType.HITL_MODIFIED, EventType.HITL_ANSWERED,
    EventType.HITL_REJECTED, EventType.HITL_CANCELLED,
)

# Task 14：5 个 AGENT_* 状态事件 → AgentView.status。与 `AgentLifecycleManager` 的五态机
# （agent_state.py）同一词表——reducer 只折叠，不重新判定转移是否合法（那是 ALM
# 在事件产生时的职责，此处只读它已经发生的结果）。
_AGENT_STATUS_BY_EVENT: dict[str, str] = {
    EventType.AGENT_RUNNING: "running",
    EventType.AGENT_IDLE: "idle",
    EventType.AGENT_WAITING_HUMAN: "waiting_human",
    EventType.AGENT_INTERRUPTED: "interrupted",
    EventType.AGENT_TERMINATED: "terminated",
}


def apply_events(events: list[Event], view: RunStateView) -> RunStateView:
    """在已有 RunStateView 上增量应用事件列表（快照恢复后的 delta replay）。"""
    for ev in events:
        if not view.session_id:
            view.session_id = ev.session_id
        if not view.task_id and ev.task_id:
            view.task_id = ev.task_id
        if not view.agent_id and ev.agent_id:
            view.agent_id = ev.agent_id
        view.events_total += 1
        _apply(view, ev)
    _rebuild_agents(view)
    return view


def prune_view_for_snapshot(view: RunStateView) -> RunStateView:
    """裁出快照该存的那部分投影：`tasks` 只留「活闭包」，其余原样返回。

    **为什么要裁。** `tasks` 是 blob 里唯一随会话历史**线性增长**的部分——每个 task 一条，
    且带 `user_prompt` / `outputs` 全文。而 `SnapshotWriter.on_event` 是在 `EventBus.emit()`
    里**内联**执行的（见那个类的 ⚠️ 注释），blob 多大，代价就直接压在 loop 主路径上，每
    ~50 条持久事件付一次；增量写还要把整份 blob 过一遍 deserialize→serialize 往返。裁完
    之后 blob 的大小只跟「当前有多少活」有关，与会话跑了多久无关。

    **闭包三类，判据全部来自真实消费者：**

    1. **非终态 task** → 全字段。`TaskManager.restore` 要重排它们，driver 立刻要读
       `user_prompt` 等。
    2. **(1) 的直接子任务**（含终态）→ 全字段。两个消费者：`restore` 的 SUSPENDED 闸门
       （`all(cid in terminal_ids for cid in children)`，缺一个终态子任务父任务就永不
       重排），以及 act guidance 的「已完成子任务」清单（`_task_label` 读 `title`、
       `_subtask_result_snippet` 读 `outputs`）。这个集合的大小是「活 task 数 × 分支度」，
       本身有界，所以**刻意不降级字段**——降级只省常数，却要为 title/outputs 的每条
       fallback 各做一遍正确性论证，不值当。
    3. **被 (1) 的 `dag_deps` 引用、又不在 1/2 里的 task** → 只留 `id` / `session_id` /
       `status`。这一类的消费者只看状态：`TaskQueue.seed_succeeded`（`== "FINISHED"` 才
       释放后继）与 `TaskManager._find_blocked_forever`（`in ("FAILED", "CANCELED")` 才
       判永久阻塞）。**状态要如实留，不能只留 FINISHED 的**——漏掉 FAILED/CANCELED 的
       前驱会让 `_find_blocked_forever` 认不出永久阻塞，那个后继就永远卡在队列里。

    其余一律丢弃：它们终态，且既不是任何活 task 的子任务、也不是它的依赖，没有消费者。

    **唯一的已知分叉，无害。** 丢掉的 task 之后仍可能收到一条 `TASK_FINALIZED`（它在
    task 终态**之后**才发，只写 `finished_at`）。`_apply` 那一支是 `view.tasks.get()`，
    取不到就跳过，于是「全量回放」与「快照+增量」在这个字段上不再逐字节等价。无害的
    理由是被裁掉的 task 进不了 `restore` 的 `all_tasks`，没有任何消费者读它的
    `finished_at`。⚠️ **要给终态 task 加新消费者时，先回来看这一条。**

    **`agents` 不裁**，也不需要为它补偿：它们是纯标量、每 agent 一条，而 `_rebuild_agents`
    只覆盖它能在 `tasks` 里找到的 agent（循环碰不到的 agent 原样保留）。所以历史 agent 的
    `spawn_depth` / `parent_agent_id` 保持 blob 里存的那份，不会因为它的 task 被裁掉而
    被重置回 0。

    幂等：裁两次与裁一次等价（第 1/2 类的判据只看 status 与 parent_task_id，第 3 类的
    瘦身结果仍带 status）。增量链因此可以反复以裁过的 blob 为基底。
    """
    live_ids = {tid for tid, t in view.tasks.items()
                if t.status not in TERMINAL_TASK_STATUSES}
    child_ids = {tid for tid, t in view.tasks.items()
                 if t.parent_task_id and t.parent_task_id in live_ids}
    keep_full = live_ids | child_ids

    dep_ids: set[str] = set()
    for tid in live_ids:
        dep_ids.update(view.tasks[tid].dag_deps or ())
    dep_ids -= keep_full

    tasks: dict[str, TaskView] = {tid: view.tasks[tid] for tid in keep_full}
    for tid in dep_ids:
        dep = view.tasks.get(tid)
        if dep is None:
            continue          # 引用了投影里没有的 id（数据异常）——跳过，不造占位
        tasks[tid] = TaskView(id=dep.id, session_id=dep.session_id, status=dep.status)

    return replace(view, tasks=tasks)


# ── HITL 活账的 blob 往返 ──────────────────────────────────────────────────────
#
# 复用既有往返器,不另造一套编码：`delivery_to_payload` / `_delivery_from_payload` 就是
# `HitlOpened` 载荷用的那对,`content_to_jsonable` / `content_from_jsonable` 就是决定 message
# 用的那对。两处各自只有一份实现,blob 与事件因此不会在编码上分叉。
#
# **不存的字段,每一个都有理由：**
#   slot                  活的 asyncio `WaitSlot`——进程内对象,序列化没有意义
#   pending_decision      待终局。它对应的 `HitlResolved` 压根没落库,冷重建必须把这条请求
#   pending_event_payload  看成未决。这是结构保证的（折叠只在 `HitlResolved` 时写
#                         `decision`）,不靠这里记得排除,但仍然显式不写,免得将来有人
#                         「顺手补全字段」把它加回来
#   claimed               同上,它描述的是热投递有没有接住,那是进程内的事
#   opened                由 `pending` ∪ `resolved` 完全重建（见 `_hitl_from_blob`）——
#                         存它是把同一份信息写两遍,而两份会分叉
#   decision_owner        同上,由 `resolved` 重建


def _pending_to_blob(r: "PendingHitl") -> dict[str, Any]:
    from ctx_weft.core.hitl.service import delivery_to_payload

    d = r.decision
    return {
        "id": r.id, "form": r.form, "session_id": r.session_id, "task_id": r.task_id,
        "agent_id": r.agent_id, "delivery": delivery_to_payload(r.delivery),
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "tenant_id": r.tenant_id, "subject_id": r.subject_id,
        "prompt": r.prompt, "detail": r.detail,
        "fields": list(r.fields), "proposal": r.proposal,
        "tool_call_id": r.tool_call_id, "stage": r.stage,
        "invocation_key": r.invocation_key, "resume_state": r.resume_state,
        "reply_as_result": r.reply_as_result,
        "reply_attempt": r.reply_attempt,
        "resolved_at": r.resolved_at.isoformat() if r.resolved_at else None,
        "legacy_origin": r.legacy_origin,
        "closed": r.closed,
        "decision": None if d is None else {
            "outcome": d.outcome,
            "message": content_to_jsonable(d.message),
            "modified_arguments": d.modified_arguments,
        },
    }


#: 存量 blob 缺 `created_at` 时的回落。**不用 `now_utc()`**：那会让一条几个月前开出的请求
#: 每次读快照都「刚刚创建」，而 `created_at` 是 `find_for_tool_call`「取最近一条」的排序键,
#: 也是 host 展示待答列表的排序键——用当下时刻会让它插到队首。固定值至少是稳定的。
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _pending_from_blob(b: dict[str, Any]) -> "PendingHitl":
    def _dt(x):
        return datetime.fromisoformat(x) if x else None

    d = b.get("decision")
    req = PendingHitl(
        id=b["id"], form=b.get("form", ""), session_id=b.get("session_id", ""),
        task_id=b.get("task_id", ""), agent_id=b.get("agent_id", ""),
        delivery=_delivery_from_payload(b.get("delivery") or {}, hitl_id=b["id"]),
        created_at=_dt(b.get("created_at")) or _EPOCH,
        tenant_id=b.get("tenant_id", "default"), subject_id=b.get("subject_id", ""),
        prompt=b.get("prompt", ""), detail=b.get("detail", ""),
        fields=list(b.get("fields") or []), proposal=b.get("proposal"),
        tool_call_id=b.get("tool_call_id", ""), stage=b.get("stage", ""),
        invocation_key=b.get("invocation_key", ""), resume_state=b.get("resume_state"),
        reply_as_result=bool(b.get("reply_as_result", False)),
    )
    req.reply_attempt = int(b.get("reply_attempt", 0) or 0)
    req.resolved_at = _dt(b.get("resolved_at"))
    req.legacy_origin = bool(b.get("legacy_origin", False))
    req.closed = bool(b.get("closed", False))
    if d is not None:
        req.decision = HitlDecision(
            outcome=d.get("outcome", ""),
            message=content_from_jsonable(d.get("message") or ""),
            modified_arguments=d.get("modified_arguments"),
        )
    return req


def _hitl_to_blob(snap: HitlSnapshot) -> dict[str, Any]:
    """HITL 活账 → blob。只写 `pending` / `resolved`,其余可重建（见上方说明）。"""
    return {
        "pending": {rid: _pending_to_blob(r) for rid, r in snap.pending.items()},
        "resolved": {rid: _pending_to_blob(r) for rid, r in snap.resolved.items()},
    }


def _hitl_from_blob(b: dict[str, Any]) -> HitlSnapshot:
    """blob → HITL 活账,并把三份派生账重建出来。

    `opened` / `decision_owner` / `decisions_for` 都从 `pending` ∪ `resolved` 推出来,不从
    blob 读——存它们等于把同一份信息写两遍,而两份会分叉。推导必须和 `apply_hitl_event` 的
    记账口径一致,所以：
      · `opened` = 两者的并（折叠里正是「开过且还没了结」留在 opened 的那批）；
      · `decisions_for` / `decision_owner` 只收 `resolved` 里带 `tool_call_id` 的,键三维,
        与折叠那两处写入逐字对应。
    """
    snap = HitlSnapshot()
    for rid, raw in (b.get("pending") or {}).items():
        snap.pending[rid] = _pending_from_blob(raw)
    for rid, raw in (b.get("resolved") or {}).items():
        snap.resolved[rid] = _pending_from_blob(raw)
    snap.opened.update(snap.pending)
    snap.opened.update(snap.resolved)
    for rid, r in snap.resolved.items():
        if r.tool_call_id and r.decision is not None:
            key = (r.session_id, r.tool_call_id, r.stage)
            snap.decisions_for[key] = (r.decision, r.resume_state)
            snap.decision_owner[key] = rid
    return snap


def serialize_view(view: RunStateView) -> dict[str, Any]:
    """把 RunStateView 序列化为可 JSON 存储的 dict（用于快照写入）。"""
    def _dt(d: datetime | None) -> str | None:
        return d.isoformat() if d is not None else None

    return {
        "run_id": view.run_id,
        "session_id": view.session_id,
        "task_id": view.task_id,
        "agent_id": view.agent_id,
        "current_step": view.current_step,
        "task_status": view.task_status,
        "session_status": view.session_status,
        "assembled_prompt_tokens": view.assembled_prompt_tokens,
        "transcript_turns": view.transcript_turns,
        "events_total": view.events_total,
        "tasks_total": view.tasks_total,
        # 不裁：它由自己的 DONE 销账，不随会话长度增长（见字段 docstring）。
        "hitl": _hitl_to_blob(view.hitl),
        "pending_recap": {
            tid: {"boundary": info.get("boundary", ""), "agent_id": info.get("agent_id", "")}
            for tid, info in view.pending_recap.items()
        },
        "sessions": {
            sid: {
                "id": s.id,
                # 保 ref 形态（裁定 2026-08-27）：session prompt 与 task 侧同为
                # 状态源，拍扁会让重放后「曾有一张图」无痕。同步的 content_to_jsonable
                # 即可——view 从事件还原，本就是 ref 形态，无 base64 可外部化，不需要
                # content_to_event_jsonable 的 put（那个是 async，且只用在发射点）。
                # ⚠️ 这条推理链继承自「事件里恒无字节」这个不变量本身，本处不是独立的
                # 第二道防线（final review M1）：**存量**（双写机制上线之前发出的）
                # 含 base64 的 SESSION_CREATED / TASK_CREATED 事件被重放到这里时，
                # content_to_jsonable 会原样保留 base64，字节就此写进
                # RunSnapshot.state_blob——即写回事件库外的另一处持久层。窄（只影响
                # 存量事件的重放快照），但真实，此处不修，仅记录在案。
                "user_prompt": content_to_jsonable(s.user_prompt),
                "template_id": s.template_id,
                "status": s.status,
                "goal": s.goal,
                "root_agent_id": s.root_agent_id,
                "llm_model": s.llm_model,
                "llm_account": s.llm_account,
                "tenant_id": s.tenant_id,
                "token_budget": s.token_budget,
                "context_limit": s.context_limit,
                "reserved_output_tokens": s.reserved_output_tokens,
                "failure_counter": s.failure_counter,
                "threshold_tripped": s.threshold_tripped,
                "created_at": _dt(s.created_at),
            }
            for sid, s in view.sessions.items()
        },
        "tasks": {
            tid: {
                "id": t.id,
                "session_id": t.session_id,
                "status": t.status,
                "title": t.title,
                "description": t.description,
                "assigned_agent_id": t.assigned_agent_id,
                "creator_agent_id": t.creator_agent_id,
                "parent_task_id": t.parent_task_id,
                "user_prompt": content_to_jsonable(t.user_prompt),
                "interaction_mode": t.interaction_mode,
                "unattended": t.unattended,
                "origin_tool_call_id": t.origin_tool_call_id,
                "origin_tool_name": t.origin_tool_name,
                "settings_raw": t.settings_raw,
                "dag_deps": t.dag_deps,
                # spec: task-handoff——两个新字段进快照（error_code /
                # blocked_by_task_id）；旧快照无键 → deserialize 落缺省。
                "priority": t.priority,
                "max_retries": t.max_retries,
                "tenant_id": t.tenant_id,
                "outputs": t.outputs,
                "error": t.error,
                "error_code": getattr(t, "error_code", None),
                "blocked_by_task_id": getattr(t, "blocked_by_task_id", None),
                "created_at": _dt(t.created_at),
                "finished_at": _dt(t.finished_at),
            }
            for tid, t in view.tasks.items()
        },
        "agents": {
            aid: {
                "id": a.id,
                "spawn_depth": a.spawn_depth,
                "parent_agent_id": a.parent_agent_id,
                "template_id": a.template_id,
                "llm_account": a.llm_account,
                "llm_model": a.llm_model,
                "status": a.status,
                "current_task_id": a.current_task_id,
            }
            for aid, a in view.agents.items()
        },
    }


def deserialize_view(data: dict[str, Any]) -> RunStateView:
    """从快照 dict 还原 RunStateView（用于快照读取后恢复）。"""
    def _dt(s: str | None) -> datetime | None:
        return datetime.fromisoformat(s) if s else None

    sessions: dict[str, SessionView] = {}
    for sid, s in data.get("sessions", {}).items():
        sessions[sid] = SessionView(
            id=s["id"],
            user_prompt=content_from_jsonable(s.get("user_prompt", "")),
            template_id=s.get("template_id", ""),
            status=s.get("status", "UNKNOWN"),
            goal=s.get("goal", ""),
            root_agent_id=s.get("root_agent_id", ""),
            llm_model=s.get("llm_model", ""),
            llm_account=s.get("llm_account", ""),
            tenant_id=s.get("tenant_id", "default"),
            token_budget=s.get("token_budget", 200_000),
            context_limit=s.get("context_limit", 180_000),
            reserved_output_tokens=s.get("reserved_output_tokens", 8192),
            failure_counter=s.get("failure_counter", 0),
            threshold_tripped=s.get("threshold_tripped", False),
            created_at=_dt(s.get("created_at")),
        )

    tasks: dict[str, TaskView] = {}
    for tid, t in data.get("tasks", {}).items():
        tasks[tid] = TaskView(
            id=t["id"],
            session_id=t.get("session_id", ""),
            status=t.get("status", "UNKNOWN"),
            title=t.get("title", ""),
            description=t.get("description", ""),
            assigned_agent_id=t.get("assigned_agent_id", ""),
            creator_agent_id=t.get("creator_agent_id", ""),
            parent_task_id=t.get("parent_task_id", ""),
            user_prompt=content_from_jsonable(t.get("user_prompt", "")),
            interaction_mode=t.get("interaction_mode", "auto"),
            # 存量快照无此键 → False（无人值守是新增语义，旧数据一律「有人在」）。
            unattended=t.get("unattended", False),
            origin_tool_call_id=t.get("origin_tool_call_id", ""),
            origin_tool_name=t.get("origin_tool_name", ""),
            settings_raw=t.get("settings_raw", {}),
            dag_deps=t.get("dag_deps", []),
            priority=t.get("priority", 5),
            max_retries=t.get("max_retries", 3),
            tenant_id=t.get("tenant_id", "default"),
            outputs=t.get("outputs"),
            error=t.get("error"),
            error_code=t.get("error_code"),
            blocked_by_task_id=t.get("blocked_by_task_id"),
            created_at=_dt(t.get("created_at")),
            finished_at=_dt(t.get("finished_at")),
        )

    agents: dict[str, AgentView] = {}
    for aid, a in data.get("agents", {}).items():
        agents[aid] = AgentView(
            id=a["id"],
            spawn_depth=a.get("spawn_depth", 0),
            parent_agent_id=a.get("parent_agent_id"),
            # 旧快照无该键 → 留空，调用方回落 session 模板（零数据迁移）。
            template_id=a.get("template_id", ""),
            # 同上：旧快照无模型选择 → 空 ModelChoice，回落账号默认。
            llm_account=a.get("llm_account", ""),
            llm_model=a.get("llm_model", ""),
            # 旧快照无该键 → 回落默认值 "idle"/None，与其余字段同口径（零数据迁移）。
            status=a.get("status", "idle"),
            current_task_id=a.get("current_task_id"),
        )

    return RunStateView(
        run_id=data.get("run_id", ""),
        session_id=data.get("session_id", ""),
        task_id=data.get("task_id", ""),
        agent_id=data.get("agent_id", ""),
        current_step=data.get("current_step"),
        task_status=data.get("task_status", "UNKNOWN"),
        session_status=data.get("session_status", "UNKNOWN"),
        assembled_prompt_tokens=data.get("assembled_prompt_tokens", 0),
        transcript_turns=data.get("transcript_turns", 0),
        events_total=data.get("events_total", 0),
        # 存量（v1）blob 无此键 → 0。那类 blob 的 `tasks` 是全量的，恢复闸门走
        # `not all_tasks` 那一半仍然对；且 v1 blob 已因 projection_version 不匹配
        # 被判不可用、走全量回放，这里的回落只是形态完整性。
        tasks_total=data.get("tasks_total", 0),
        # v2 及更早的 blob 无此键 → 空。那些 blob 已因 projection_version 不匹配被判不可用、
        # 走全量回放（全量回放会正确折出它），所以这里的回落只是形态完整性——**这正是
        # bump 版本号换来的东西**：不必为「老快照里没有这个字段」另留一条兜底查询。
        hitl=_hitl_from_blob(data.get("hitl") or {}),
        pending_recap={
            tid: {"boundary": v.get("boundary", ""), "agent_id": v.get("agent_id", "")}
            for tid, v in (data.get("pending_recap") or {}).items()
        },
        sessions=sessions,
        tasks=tasks,
        agents=agents,
    )


#: 快照 blob 的投影版本（spec: snapshot-recovery）：不匹配的快照被忽略、走全量回放。
#:
#: **读写两侧共用这一个常量**——`snapshot_is_usable` 判定用它，`SnapshotWriter` 盖章
#: 也从这里取。从前是两处独立字面量（本文件与 `providers/events/snapshot.py` 各一个），
#: 只 bump 一个的后果是所有快照**永远**判不可用：恢复全量回放、writer 每次全量重锚，
#: O(delta) 退回 O(n)，而且不报任何错——只能从 SnapshotWriter 日志里的 `mode=anchor`
#: 看出来。合成一个常量，把这个类别消灭掉。
#:
#: v2（2026-09-19）：blob 的 `tasks` 改为只存「活闭包」（见 `prune_view_for_snapshot`）。
#: v1 的 blob 是全量 tasks、信息上是 v2 的超集，拿它当增量基底本来也是对的；bump 是为了
#: **回滚安全**——v1 的代码读到 v2 blob 会因版本不匹配走全量回放，而不是把一份裁过的
#: tasks 当全量用（那会让 `restore` 静默少掉 task）。代价是存量快照集体失效一次、
#: 各会话首次写快照时全量重锚一次。
#: v3（2026-09-20）：blob 新增 `pending_recap`（段 recap 的待重跑账，从前是恢复路径上
#: 一次独立的按类型收窄查询）。必须 bump：v2 的 blob 没有这个键，拿它当增量基底会让
#: 「崩溃打断的 recap」静默变成「没有待重跑的」，那段 memory 写就永远补不上了。bump 之后
#: v2 blob 判不可用 → 全量回放 → 正确折出该字段，无需任何迁移分支。
#: v4（2026-09-20）：blob 新增 `hitl`（HITL 活账，从前是 `rebuild_hitl` 每次冷应答一次的全量
#: 折叠）。必须 bump：v3 的 blob 没有这个键，拿它当增量基底会让**未决 HITL 集合凭空为空**
#: ——`parked_task_ids` 随之为空，于是「人还没答，任务却自己跑起来了」。这是这套版本号防的
#: 最严重的一种退化，不是性能问题。
_PROJECTION_VERSION = 4


def snapshot_is_usable(
    snapshot: Any, head: int, *, max_chain_depth: int | None = None,
) -> bool:
    """该快照能否作为增量基底（spec: snapshot-recovery）。

    **恢复侧与写入侧共用这一套判据**——两边各写一遍是漂移的现成来源：判据一旦分叉，
    writer 会基于一张 reader 根本不认的快照做增量，两路等价就此失守。

    四条硬判据（任一不满足 → 不可用，调用方全量重建）：

    1. 有快照；
    2. 带 ``last_commit_position``——缺它就是改造前的存量快照，位置不可猜；
    3. ``projection_version`` 与当前实现一致——apply 语义变过则旧 blob 不可复用；
    4. 位置不超前于 ``head``——引用未来位置说明数据异常。

    ``max_chain_depth``：仅**写入侧**传。快照链每增量一次深一层，达到上限即强制一次
    全量重锚，把「serialize/deserialize 往返有损」这类逐次累积的偏差限制在常数级内。
    恢复侧不传——读一张已存在的快照时，链深不影响这一次读取的正确性，那是写入策略。
    """
    if snapshot is None:
        return False
    if getattr(snapshot, "last_commit_position", None) is None:
        return False
    if getattr(snapshot, "projection_version", 1) != _PROJECTION_VERSION:
        return False
    if snapshot.last_commit_position > head:
        return False
    if (max_chain_depth is not None
            and getattr(snapshot, "chain_depth", 0) >= max_chain_depth):
        return False
    return True


async def rebuild_view(event_store: Any, session_id: str) -> RunStateView:
    """按快照+增量或全量回放重建 RunStateView（spec: snapshot-recovery，wp4 改造）。

    恢复**只有一条口径**：position 一致切面。有序提交是 `EventStore` 的必需部分，
    所以这里不再按 store 能力分路——没有「无 position 时怎么办」这个分支。

    路径选择（按序）：

    1. **快照 + 增量**（快照有效）：快照携带 ``last_commit_position``、
       ``projection_version`` 匹配、且位置不超前于 ``committed_head``
       → ``read_range((cursor, head])`` 增量 apply。
    2. **全量回放**：无快照 / 快照缺 position（改造前的存量快照）/ 版本不匹配 /
       引用未来位置（数据异常）→ ``read_range(0..head)`` 全量折。
       忽略坏快照是性能降级不是数据丢失（日志是真相）。

    「按事件 ID 取增量」这种读法已从 `EventStore` 彻底移除：ID 铸造序 ≠ 提交序，
    按 ID 当游标正是 H2 的根因。
    """
    try:
        snapshot = await event_store.load_latest_snapshot(session_id)
    except NotImplementedError:
        snapshot = None

    head = await event_store.committed_head(session_id)
    if snapshot_is_usable(snapshot, head):
        view = deserialize_view(snapshot.state_blob)
        delta = await event_store.read_range(
            session_id,
            after_position=snapshot.last_commit_position,
            through_position=head,
        )
        return apply_events([se.event for se in delta], view)
    # 全量：忽略快照（存量/损坏/超前），按提交序重放；重造快照由 writer 负责。
    #
    # **分批由 store 产出**（`EventStore.replay`），这里只管折叠。从前这段自己算 position
    # 区间、还先探一次「这个 store 支不支持分页」再选路——那是把 store 的知识和能力判断
    # 漏进了 core，一个坏设计生出两个分支与两种失败形态。现在 core 不问、不选、不降级。
    #
    # 分批与整批逐字段等价：重放是左折叠（`reduce_events` 就是 `apply_events` 在空 view
    # 上的调用，本文件里两段循环体逐字相同），而 apply 是 for 循环、可结合。
    view = RunStateView(run_id=session_id, session_id="", task_id="", agent_id="")
    async for batch in event_store.replay(session_id):
        apply_events(batch, view)
    return view


def reduce_events(events: list[Event], run_id: str) -> RunStateView:
    """Replay event list into a RunStateView."""
    view = RunStateView(
        run_id=run_id,
        session_id="",
        task_id="",
        agent_id="",
    )

    for ev in events:
        if not view.session_id:
            view.session_id = ev.session_id
        if not view.task_id and ev.task_id:
            view.task_id = ev.task_id
        if not view.agent_id and ev.agent_id:
            view.agent_id = ev.agent_id

        view.events_total += 1
        _apply(view, ev)

    _rebuild_agents(view)
    return view


def _agent_slot(view: RunStateView, aid: str) -> AgentView:
    """取（或建）某 agent 的投影槽位。

    槽位可能已被 `_apply` 的 AGENT_INSTANTIATED 分支建出来（只带 template_id）——
    树形字段与 template_id 来自两个不同的来源（推算 vs 事件），谁先到都不能踩掉对方。
    """
    av = view.agents.get(aid)
    if av is None:
        av = AgentView(id=aid)
        view.agents[aid] = av
    return av


def _rebuild_agents(view: RunStateView) -> None:
    """从 session/task 层级推算所有 AgentView 的树形字段（spawn_depth / parent）。

    只写树形字段，不碰 `template_id`——后者只有 AGENT_INSTANTIATED 事件知道。
    「首个来源胜出」的原有语义由 `_inferred` 保持（此前靠 `aid in view.agents` 判定，
    在 _apply 会预建槽位之后那个判据已失效）。
    """
    _inferred: set[str] = set()

    # Root agents from sessions
    for sess in view.sessions.values():
        if sess.root_agent_id:
            av = _agent_slot(view, sess.root_agent_id)
            av.spawn_depth = 0
            av.parent_agent_id = None
            _inferred.add(sess.root_agent_id)

    # Subagent tasks sorted by creation time (parent tasks precede children in event stream)
    ordered = sorted(view.tasks.values(), key=lambda t: t.created_at or datetime.min)
    for task in ordered:
        aid = task.assigned_agent_id
        if not aid or aid in _inferred:
            continue
        use_subagent = task.settings_raw.get("use_subagent", False)
        creator = task.creator_agent_id
        if use_subagent and creator and creator in view.agents:
            depth = view.agents[creator].spawn_depth + 1
        else:
            depth = 0
        av = _agent_slot(view, aid)
        av.spawn_depth = depth
        av.parent_agent_id = creator or None
        _inferred.add(aid)


def _apply(view: RunStateView, ev: Event) -> None:
    t = ev.type
    p: dict[str, Any] = ev.payload or {}

    # ── Run / Step ────────────────────────────────────────────────────────────
    if t == EventType.STEP_STARTED:
        view.current_step = p.get("step_name")
    elif t == EventType.STEP_COMPLETED:
        view.current_step = p.get("next_step")
    elif t == EventType.RUN_STARTED:
        # run 起止不写 task 状态：ACTIVE 的正主是 TASK_STARTED（TASK_STATUS_BY_EVENT，
        # TM 派发时发）。recap / recognize_intent 等观测型 run 也会发 RunStarted，
        # 若这里再写 ACTIVE，纯文本冷 park 场景下会把已经 AWAITING_HUMAN 的 task 错误
        # 拉回 ACTIVE——标量与逐 task 视图自相矛盾。会话状态同样不在此写：run 是任务
        # 级的，会话状态归 SessionRegistry（docs/events-v2.md §2.1.1）。
        pass
    elif t == EventType.RUN_FINISHED:
        pass   # run 的记账，不承载状态——它在 §3.2 已是 O 档

    # ── Session projection ────────────────────────────────────────────────────
    elif t == EventType.SESSION_CREATED:
        sess = SessionView(
            id=ev.session_id,
            # 存量事件是裸 str → content_from_jsonable 原样返回，零数据迁移。
            user_prompt=content_from_jsonable(p.get("user_prompt", "")),
            template_id=p.get("template_id", ""),
            root_agent_id=p.get("root_agent_id", ""),
            llm_model=p.get("llm_model", ""),
            llm_account=p.get("llm_account", ""),
            tenant_id=p.get("tenant_id", ev.tenant_id),
            token_budget=p.get("token_budget", 200_000),
            context_limit=p.get("context_limit", 180_000),
            reserved_output_tokens=p.get("reserved_output_tokens", 8192),
            status="RUNNING",
            created_at=ev.timestamp,
        )
        view.sessions[ev.session_id] = sess
        view.session_status = "RUNNING"

    elif t == EventType.FAILURE_THRESHOLD_HIT:
        # 只置闩，不动计数：这条事件本身不是一次新失败，它是「计数已经到阈值」的宣告。
        # trip 序列第 2 步发它，第 6 步才给 root 判死——闩位因此恒在那条 TaskFailed 之前。
        sess = view.sessions.get(ev.session_id)
        if sess is not None:
            sess.threshold_tripped = True

    elif t == EventType.SESSION_RESUMED:
        sess = view.sessions.get(ev.session_id)
        if sess is not None:
            # 续跑开新一轮 → 清闩。与内存侧同形：`resume_session` 造的是全新 Session
            # 与全新 TaskManager（`_threshold_tripped` 回到 False），新一轮的失败照常计。
            sess.threshold_tripped = False
            _up = p.get("user_prompt")
            if _up is not None:
                sess.user_prompt = content_from_jsonable(_up)
            sess.status = "RUNNING"
        view.session_status = "RUNNING"

    elif t == EventType.SESSION_STATUS_CHANGED:
        # L 档：只读存量日志（新代码不再发这条通用 setter）。与兄弟分支
        # SESSION_PAUSED_HITL 同口径——旧值域里的 PAUSED / PAUSED_HITL 在新词表
        # 里不存在，必须**翻译进当前词表**再写，否则重放存量日志会往
        # `session_status` 写一个类型里没有的值（models.py 的承诺）。
        # 重构前 `recover()` 发的 `_emit_session_status(... or "PAUSED_HITL")`
        # 走的正是这条，故存量日志里旧值的主要产地就在这儿。
        # 其余值（RUNNING / INTERRUPTED / SUCCEEDED / FAILED / CANCELED）仍在
        # 值域内，原样通过。
        new_status = _LEGACY_SESSION_STATUS.get(
            p.get("new_status", ""), p.get("new_status", ""))
        if new_status:
            view.session_status = new_status
            sess = view.sessions.get(ev.session_id)
            if sess is not None:
                sess.status = new_status

    elif t == EventType.SESSION_PAUSED_HITL:
        # L 档：只读存量日志。旧模型按 form 分 PAUSED / PAUSED_HITL 两档，
        # 新模型合并成一个 WAITING——「等的是审批面板还是一句话」是 delivery 的性质，
        # 由 HitlOpened 承载，不进会话状态（docs/events-v2.md §2.1.3）。
        # 故这里**刻意**把两档都折进 WAITING，不再读 form。
        _set_session_status(view, ev.session_id, WAITING)

    elif t == EventType.SESSION_FINISHED:
        final_status = p.get("final_status", "SUCCEEDED")
        view.session_status = final_status
        sess = view.sessions.get(ev.session_id)
        if sess is not None:
            sess.status = final_status

    elif t == EventType.RECOGNIZE_INTENT_TOOL_CALL:
        goal = p.get("session_goal", "")
        if goal:
            sess = view.sessions.get(ev.session_id)
            if sess is not None:
                sess.goal = goal
        # root task 创建时 title 为空、由 recognize_intent 并发补填——回放须同样补进
        # TaskView，否则导入/重启重建的投影里 root task 永远无名。空值不覆盖已有值。
        if ev.task_id:
            task = view.tasks.get(ev.task_id)
            if task is not None:
                if p.get("title"):
                    task.title = p["title"]
                if p.get("description"):
                    task.description = p["description"]

    # ── Task projection ───────────────────────────────────────────────────────
    # ── Agent projection ──────────────────────────────────────────────────────
    elif t == EventType.AGENT_INSTANTIATED and ev.agent_id:
        # 事件流里唯一记录「该 agent 用的哪个模板」的地方：树形推算得不出模板。
        # 授权按模板做策略（AllowListAuthorizer 读 ctx.agent_template_id），冷 resume
        # 若拿不到就只能回落 session 模板，子 agent 会顶着 root 的模板身份。
        tmpl = p.get("template_id", "")
        if tmpl:
            _agent_slot(view, ev.agent_id).template_id = tmpl
        # 同一事件里的初始模型选择——空值不覆盖，与 template_id 同口径。
        account = p.get("llm_account", "")
        if account:
            _agent_slot(view, ev.agent_id).llm_account = account
        model = p.get("llm_model", "")
        if model:
            _agent_slot(view, ev.agent_id).llm_model = model

    elif t == EventType.AGENT_LLM_CHANGED and ev.agent_id:
        # 纯赋值：agent 的模型选择变了。不改任何 task / session 状态——
        # 「换模型」和「让 task 跑起来」是两件事（见 spec §06 的三条命令）。
        slot = _agent_slot(view, ev.agent_id)
        slot.llm_account = p.get("llm_account", "")
        slot.llm_model = p.get("llm_model", "")

    elif t in _AGENT_STATUS_BY_EVENT and ev.agent_id:
        # terminated 粘滞：一旦进入终态就不再被迟到事件改回——与 SessionRegistry
        # 已有的「已终态就不再转移」同构。
        agent = view.agents.get(ev.agent_id)
        if agent is not None and agent.status != "terminated":
            agent.status = _AGENT_STATUS_BY_EVENT[t]
            if ev.task_id:
                agent.current_task_id = ev.task_id

    elif t == EventType.TASK_CREATED:
        task_data: dict = p.get("task", {})
        task_id = task_data.get("id") or ev.task_id
        if task_id:
            task = TaskView(
                id=task_id,
                session_id=ev.session_id,
                status=task_data.get("status", "PENDING"),
                title=task_data.get("title", ""),
                description=task_data.get("description", ""),
                assigned_agent_id=task_data.get("assigned_agent_id", ""),
                creator_agent_id=task_data.get("creator_agent_id", ""),
                parent_task_id=task_data.get("parent_task_id", ""),
                user_prompt=content_from_jsonable(task_data.get("user_prompt", "")),
                interaction_mode=task_data.get("interaction_mode", "auto"),
                # 存量事件流无此键 → False，见 deserialize_view 同一口径。
                unattended=task_data.get("unattended", False),
                origin_tool_call_id=task_data.get("origin_tool_call_id", ""),
                origin_tool_name=task_data.get("origin_tool_name", ""),
                settings_raw=task_data.get("settings", {}),
                dag_deps=task_data.get("dag_deps", []),
                priority=task_data.get("priority", 5),
                max_retries=task_data.get("max_retries", 3),
                tenant_id=ev.tenant_id or "default",
                created_at=ev.timestamp,
            )
            if task_id not in view.tasks:
                view.tasks_total += 1      # 只数「新建」，重放同一条事件不重复计
            view.tasks[task_id] = task
            if not view.task_id:
                view.task_id = task_id
            # 被指派的 agent 的 `current_task_id` 在这里就更新，不等 `TASK_STARTED`
            # 折出的 `AgentRunning`。
            #
            # 理由是内存那边有一个**无事件的写口**：`_start_task_for_agent` push 完新
            # task 会立刻 `ALM.set_current_task(agent_id, task.id)`（"task 还没派发、但
            # 路由已经必须认它"，见该方法自己的 docstring），而这一步不发任何事件。若
            # 投影只从 `AGENT_*` 折这个字段，push 之后、`TASK_STARTED` 之前这段窗口里
            # 投影里还是**上一个已终态的 task**——而 `ALM.load()` 是无条件整条覆盖
            # record 的，这段窗口内任何一次热重装（`/resume`、冷 HITL 应答、
            # `send_message` 的自愈重建都会调 `_load_agents_of`）都会把路由判据倒回去。
            # 倒回之后下一条消息看到的 `current_task_id` 已终态 → 又新建一个 task，
            # 而刚才那个还在队列里：同一个 agent 挂两个 task（`assert_can_receive` 拦
            # 不住——那个窗口里 agent 还是 `idle`，`TASK_STARTED` 没发）。
            #
            # 补上这一折之后，内存写口与事件轴同源：`load()` 覆盖回来的就是同一个值，
            # "恢复是喂进来、不是查回去"那条纪律不必为此开特例。
            #
            # `terminated` 粘滞：与下面 `_AGENT_STATUS_BY_EVENT` 分支同一口径，已终态的
            # agent 不被迟到的 task 事件改回。
            assigned = task_data.get("assigned_agent_id", "")
            if assigned:
                slot = view.agents.get(assigned)
                if slot is not None and slot.status != "terminated":
                    slot.current_task_id = task_id

    elif t == EventType.TASK_REQUEUED and ev.task_id:
        # 重排（observer active / retry）：状态回 PENDING、清旧产出。
        #
        # `user_prompt` 这一读是**存量兼容**：今天没有发射点往 TASK_REQUEUED 里放它
        # （已删的 reopen 是唯一一个），但存量日志里有 reopen 写下的改写版 prompt——
        # 不读它，那些会话重放后 prompt 会退回 TaskCreated 的原始值，即重启前后不一致。
        task = view.tasks.get(ev.task_id)
        if task is not None:
            task.status = "PENDING"
            task.outputs = None
            up = p.get("user_prompt")
            if up is not None:
                task.user_prompt = content_from_jsonable(up)
        view.task_status = "PENDING"

    elif t == EventType.TASK_HUMAN_RESOLVED and ev.task_id:
        # 「解除阻塞」不是「开始执行」：人已答复/放行，回 PENDING、清旧产出——与
        # TASK_REQUEUED 效果相同（判据是类型不是 payload，D4/见 docs/events-v2.md §2.3），
        # 但这是 TaskAwaitingHuman{hitl_id} 的配对解除事件，不是重排重试。
        task = view.tasks.get(ev.task_id)
        if task is not None:
            task.status = "PENDING"
            task.outputs = None
        view.task_status = "PENDING"

    elif t in TASK_STATUS_BY_EVENT and ev.task_id:
        task = view.tasks.get(ev.task_id)
        if task is not None:
            task.status = TASK_STATUS_BY_EVENT[t]
            if t == EventType.TASK_STARTED:
                assigned = p.get("assigned_agent_id", "")
                if assigned:
                    task.assigned_agent_id = assigned
        view.task_status = TASK_STATUS_BY_EVENT[t]

        # ── failure_counter 折叠 ──
        # TASK_FAILED：普通失败 +1；TASK_FINISHED：成功清零（连败语义）。
        # 熔断闩位（`FailureThresholdHit` 置位）期间一律不计：trip 序列第 6 步给 root
        # 判死也发一条 TaskFailed，那是**聚合结果**而非第 N+1 次新败，计入会让恢复后的
        # 计数比内存真值多一，进而可能误触发下一次熔断。
        #
        # 判据是**事件**，不是 payload 里的 `error_code` 字符串。改造前这里比的是
        # `error_code != TaskErrorCode.BY_THRESHOLD`，那依赖一条没有类型保障的隐含
        # 约定——`error_code` 字段同时接受 `TaskErrorCode` 与 `InterruptReason` 的值
        # （见 runtime 的 outage 支 `error_code=InterruptReason.LLM_OUTAGE`），一旦
        # 将来有谁给 TaskFailed 也塞一个非 TaskErrorCode 的码，那个 `!=` 会静默放行。
        # 闩位与 `TaskManager._threshold_tripped` 同源同形，不看任何自由文本。
        if t == EventType.TASK_FAILED:
            sess = view.sessions.get(ev.session_id)
            if sess is not None and not sess.threshold_tripped:
                sess.failure_counter += 1
        elif t == EventType.TASK_FINISHED:
            sess = view.sessions.get(ev.session_id)
            if sess is not None:
                sess.failure_counter = 0

        # 成果物与死因：数据在 task 终态事件的 payload 里（总账 A1）。
        # TaskFinalized 从不发这两个键，故不能挂在它上面读。
        if task is not None:
            if t == EventType.TASK_FINISHED:
                if "outputs" in p:
                    task.outputs = p.get("outputs")
            elif t in (EventType.TASK_FAILED, EventType.TASK_INTERRUPTED):
                msg = p.get("error_message")
                if msg:
                    task.error = msg
            elif t == EventType.TASK_CANCELED:
                # spec: task-handoff——依赖阻塞取消的结局码与阻塞源随 TASK_CANCELED
                # payload 持久化；不读则回放后只剩 CANCELED、原因不可解释。
                # 用户侧取消（reason 文本）也顺手折进 error，供恢复后观察面解释。
                code = p.get("error_code")
                if code:
                    task.error_code = code
                    task.blocked_by_task_id = p.get("blocked_by_task_id") or None
                msg = p.get("reason") or p.get("error_message")
                if msg and not task.error:
                    task.error = msg

    elif t == EventType.TASK_FINALIZED and ev.task_id:
        task = view.tasks.get(ev.task_id)
        if task is not None:
            # 本事件的 payload 里**有** outputs{output,summary}/error（2026-09-05 起，
            # 供 host 落 tasks 表与 CLI 打印），但这里刻意不折：投影的 outputs/error 只认
            # TaskFinished/TaskFailed/TaskInterrupted 一个来源（总账 A1）——两处都写会让
            # 先到的 TaskFinalized 覆盖或清空后到的成果，正是 A1 当初要治的那个漂移。
            task.finished_at = ev.timestamp

    # ── 段 recap（被崩溃打断的 background observe）──────────────────────────────
    # started 记账、done 销账，剩下的就是要重跑的。**这里是唯一的折叠实现**——从前它是
    # 独立函数 `fold_pending_task_recap` + 恢复路径上一次按类型收窄的查询，那条查询随会话
    # 长度线性增长（见 `RunStateView.pending_recap`）。搬进来而不是并存：feat 把 HITL 移出
    # 投影的理由正是「两份口径不同的折叠会漂移」，同一条原则在这里就是「只留一份」。
    #
    # 键取 payload 的 `task_id` 而不是 `ev.task_id`：这两个事件把它写在 payload 里，
    # 事件头上的 task_id 在段边界上可能是父任务。
    elif t in (EventType.TASK_RECAP_STARTED, EventType.TASK_RECAP_DONE):
        rtid = p.get("task_id", "")
        if rtid:
            if t == EventType.TASK_RECAP_STARTED:
                # 同 task_id last-write-wins：重跑失败会再发一条 started。
                view.pending_recap[rtid] = {
                    "boundary": p.get("boundary", ""),
                    "agent_id": p.get("agent_id", ""),
                }
            else:
                view.pending_recap.pop(rtid, None)

    # ── LLM / Context ─────────────────────────────────────────────────────────
    elif t == EventType.PREPARE_COMPLETED:
        view.assembled_prompt_tokens = p.get("assembled_token_count", 0)
    elif t == EventType.ACT_TURN_COMPLETED:
        view.transcript_turns = p.get("turn", view.transcript_turns)

    # ── HITL ──────────────────────────────────────────────────────────────────
    # **同一份折叠**：`apply_hitl_event` 就是 `fold_hitl_snapshot` 在已有累加器上的一次调用，
    # 不是投影另写的一版。当初把 HITL 移出投影是因为「两份口径不同的折叠会漂移」（旧实现
    # 「重建了 pending 却没重建已解决」就是那么来的）；现在折叠只有一份，那条理由不再适用，
    # 而它回到投影换来的是 `rebuild_hitl` 不必再全量读（实测 1600 条 / 218ms / 读放大 1600×）。
    #
    # `HitlRegistry` 仍是**查询**的唯一真相源，投影只是它的装填来源（spec §3.1：恢复是
    # 喂进来，不是查回去）。
    if t in HITL_FOLD_EVENT_TYPES:
        apply_hitl_event(ev, view.hitl)

    # ── HITL 会话状态 ──────────────────────────────────────────────────────────
    elif t in (
        # L 档：这五个不再发射，保留只为读存量日志。HITL_RESOLVED（新模型）**不在其中**
        # ——新流量里会话状态由 AGENT_* 折叠推出，不再由会话级事件承载。
        EventType.HITL_APPROVED, EventType.HITL_MODIFIED, EventType.HITL_ANSWERED,
        EventType.HITL_REJECTED, EventType.HITL_CANCELLED,
    ):
        if view.session_status == WAITING:
            _set_session_status(view, ev.session_id, "RUNNING")


def _set_session_status(view: RunStateView, session_id: str, status: str) -> None:
    """把状态同时写进 run 级标量与 SessionView。两处必须同写——只写一处是
    「投影和视图对不上」那类 bug 的来源。"""
    view.session_status = status
    sess = view.sessions.get(session_id)
    if sess is not None:
        sess.status = status


# ══════════════════════════════════════════════════════════════════════════════
# HITL v2 双读折叠（2026-09-01 重设计 · 段 1）
# 设计：docs/superpowers/specs/2026-09-01-hitl-redesign-design.md §12.3
# ══════════════════════════════════════════════════════════════════════════════

#: 折叠所需的事件类型全集（新 2 类 + 旧 6 类）。供事件库按类型过滤读取，
#: 无需全量回放。`SessionPausedHitl` 不在其中——新模型由 pending 集合推导。
HITL_FOLD_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.HITL_OPENED,
    EventType.HITL_RESOLVED,
    EventType.HITL_REPLY_RETRACTED,
    EventType.HITL_CLOSED,
    EventType.HITL_REQUIRED,
    *_HITL_RESOLVE_TYPES,
)

#: 旧 `context` 字符串 → 新 `UserTurnDelivery.preface` 枚举。
_LEGACY_PREFACE: dict[str, str] = {
    "plain_text": PREFACE_NORMAL,
    "interrupt": PREFACE_AFTER_INTERRUPT,
    "interrupt:edit": PREFACE_AFTER_INTERRUPT_EDIT,
}


def _delivery_from_payload(payload: dict, hitl_id: str = "") -> Delivery:
    """新事件的 delivery 载荷 → Delivery。封闭值域，穷举即完备。"""
    kind = payload.get("kind", "")
    if kind == "tool_result":
        return ToolResultDelivery(tool_call_id=payload.get("tool_call_id", ""))
    if kind == "user_turn":
        return UserTurnDelivery(task_id=payload.get("task_id", ""),
                                preface=payload.get("preface", PREFACE_NORMAL))
    # 未知 kind（更新版本写下的、本版本不认识的取值）：只能降级为「不可续跑」。
    # 有真人正等着这条请求被 claim——spec §12.3.2「NoResume + 告警」，不静默丢。
    logger.warning(
        "fold_hitl_snapshot: unknown delivery kind %r for hitl_id=%s, degrading to NoResume",
        kind, hitl_id)
    return NoResumeDelivery()


def _legacy_delivery(form: str, tool_call_id: str, context: str, task_id: str,
                     hitl_id: str = "") -> Delivery:
    """旧请求 → Delivery 的反推。

    **复刻旧的「判据」，而不是旧的「意图」**：已删除的 `_resume_after_cold_hitl` 分流
    时看的是 `req.form == "wait"`，不是 sentinel capability_id。用 sentinel 反推会让
    「form 是 wait 但 capability_id 不是 sentinel」的在途请求从「注入」变成「补
    tool_call」——迁移本身改变了行为。迁移的正确性判据是与升级前逐条同构（spec §12.3.2）。

    这里的字面量 `"wait"` 是**存量事件的数据值**，不是活路由判据：活路由只认
    `Delivery`（spec §5），旧模型的这一维已在此处一次性翻译掉。
    """
    if form == "wait":
        return UserTurnDelivery(task_id=task_id,
                                preface=_LEGACY_PREFACE.get(context, PREFACE_NORMAL))
    if tool_call_id:
        return ToolResultDelivery(tool_call_id=tool_call_id)
    # 既非 wait、又无 tool_call 可补：续跑无从谈起。显式「只可取消」，不静默丢——有真人正
    # 等着这条请求的答复（spec §12.3.2「NoResume + 告警」）。
    logger.warning(
        "fold_hitl_snapshot: legacy hitl_id=%s form=%r has no tool_call_id, "
        "degrading to NoResume", hitl_id, form)
    return NoResumeDelivery()


def _legacy_decision(event_type: str, payload: dict) -> HitlDecision | None:
    """旧 resolve 事件 → 决定；**不可用**则返回 None（按未决重问，绝不臆造）。

    可用性规则（旧事件）：Answered 须带 message、
    Modified 须带 modified_arguments、Cancelled 不是决定（spec §12.3.3）。
    """
    # 事件载荷存的是 jsonable 形态（str | list[dict]）——须经 content_from_jsonable 转回
    # ContentPart，否则多模态答复（图片）在 split_for_tool_result 里因无 .text 被拆成空文本
    # 静默丢损（Finding 1）。
    message = content_from_jsonable(payload.get("message") or "")
    if event_type == EventType.HITL_APPROVED:
        return HitlDecision(outcome=HITL_OUTCOME_ACCEPTED, message=message)
    if event_type == EventType.HITL_MODIFIED:
        args = payload.get("modified_arguments")
        if args is None:
            return None
        return HitlDecision(outcome=HITL_OUTCOME_ACCEPTED, message=message,
                            modified_arguments=args)
    if event_type == EventType.HITL_ANSWERED:
        # 判据是真值而非 is None：空串同样还原不出答案——
        # 空串/空表同样视为「还原不出答案」，按未决重问，不臆造（Finding 2）。
        if not payload.get("message"):
            return None
        return HitlDecision(outcome=HITL_OUTCOME_ACCEPTED, message=message)
    if event_type == EventType.HITL_REJECTED:
        return HitlDecision(outcome=HITL_OUTCOME_REJECTED, message=message)
    return None                                   # HITL_CANCELLED：不是决定


def _as_resolved(req: PendingHitl, decision: HitlDecision, resolved_at: datetime,
                 *, legacy: bool = False) -> PendingHitl:
    """把折出的请求标成已终局，供 `HitlSnapshot.resolved` 收录。

    就地改**同一个对象**：它此刻已被 `snap.pending.pop` 摘掉，不再是 pending 的一员，
    没有第二个持有者。复制一份反而会让「同一 hitl_id 两个对象」这种更难查的状态出现。

    `slot` 恒为 None（折叠出来的东西没有等待槽——重启后一切皆冷）。

    `legacy`：这条终局出自**旧模型**的终态事件。恢复期的 `UserTurn` 补写据此跳过它——
    旧路径写下的记忆记录没有 `hitlreply:` 幂等键，补写会重复（Task 9 复审）。
    """
    req.decision = decision
    req.resolved_at = resolved_at
    req.slot = None
    req.legacy_origin = legacy
    return req


def apply_hitl_event(ev: Event, snap: HitlSnapshot) -> HitlSnapshot:
    """把**一条** HITL 事件折进已有的 `snap`（增量 apply 的那一步）。

    它和 `fold_hitl_snapshot` 是**同一份实现**——后者就是在空 `snap` 上把事件逐条喂进来。
    并存两份口径不同的 HITL 折叠正是旧实现里「重建了 pending 却没重建已解决」那类漂移的
    来源，也是本分支当初把 HITL 移出投影的理由，所以这里绝不另写一遍。

    能这样复用，前提是 `snap` 是**唯一**的累加器：连折叠期的工作账（`opened` /
    `decision_owner`）都在它里面。放在外面，一次性折叠与增量折叠各带一份，两份迟早分叉。
    """
    return fold_hitl_snapshot([ev], snap)


def fold_hitl_snapshot(
    events: list[Event], snap: "HitlSnapshot | None" = None,
) -> HitlSnapshot:
    """双读折叠：新旧两套 HITL 事件 → `HitlSnapshot`。

    `snap` 非 None 时**在它上面继续折**（增量 apply，见 `apply_hitl_event`）；折叠是左折叠，
    所以「一次喂 n 条」与「分 n 次各喂 1 条」逐字段等价。

    同 tool_call 有多条请求（重问副本）时，**最后一条可用决定胜出**。

    **event 侧 ref 尚未 hydrate**：`decisions_for[*]` 里的 `HitlDecision.message` 就是
    事件载荷里存的、经 `content_from_jsonable` 转回的 `ContentPart`——但那仍是**事件**
    blob store 命名空间下的引用。装填进 registry、被工具结果/记忆消费之前，调用方须比照
    `runtime.py:1943-1961` 的 `_cold_hitl_decision`：先 `hydrate_event_content`，再
    `normalize_content` 写入**记忆** blob store，并对失败做 `downgrade_images_to_text`
    兜底（该兜底绝不可再抛）。本函数本身保持同步、不做这一步（spec §12.3.3）。
    """
    if snap is None:
        snap = HitlSnapshot()
    opened = snap.opened
    _decision_owner = snap.decision_owner
    for ev in events:
        p = ev.payload or {}
        rid = p.get("hitl_id", "")
        if not rid:
            continue

        if ev.type == EventType.HITL_OPENED:
            req = PendingHitl(
                id=rid, form=p.get("form", ""), session_id=ev.session_id,
                task_id=ev.task_id or "", agent_id=p.get("agent_id", "") or (ev.agent_id or ""),
                delivery=_delivery_from_payload(p.get("delivery") or {}, hitl_id=rid),
                created_at=ev.timestamp, tenant_id=ev.tenant_id, subject_id=p.get("subject_id", ""),
                prompt=p.get("prompt", ""), detail=p.get("detail", ""),
                fields=list(p.get("fields") or []), proposal=p.get("proposal"),
                tool_call_id=p.get("tool_call_id", ""), stage=p.get("stage", ""),
                # 决定缓存的第四维（复审 I3）。旧事件没有这一维 → "" → 通配，
                # 迁移期行为逐条同构。
                invocation_key=p.get("invocation_key", ""),
                resume_state=p.get("resume_state"),
                reply_as_result=bool(p.get("reply_as_result", False)),
            )
            opened[rid] = req
            snap.pending[rid] = req

        elif ev.type == EventType.HITL_REQUIRED:
            form = p.get("form", "approval")
            tool_call_id = p.get("tool_call_id", "")
            # 旧模型没有 stage 字段：approval 出自授权步（HumanGatedAuthorizer 的
            # needs_human），question/wait 等其余 form 出自工具步（ask_user 等 provider 自问）。
            stage = HITL_STAGE_AUTHZ if form == "approval" else HITL_STAGE_TOOL
            req = PendingHitl(
                id=rid, form=form, session_id=ev.session_id, task_id=ev.task_id or "",
                agent_id=p.get("agent_id", "") or (ev.agent_id or ""),
                delivery=_legacy_delivery(form, tool_call_id, p.get("context", ""),
                                          ev.task_id or "", hitl_id=rid),
                created_at=ev.timestamp, tenant_id=ev.tenant_id, subject_id=p.get("capability_id", ""),
                stage=stage,
                prompt=p.get("question", ""),
                # wait 表单的旧 context 存的是模式标记（"plain_text"/"interrupt"/
                # "interrupt:edit"），不是人类可读文案——那份语义已经由上面的 delivery.preface
                # 承接。塞进 detail 会把私有的续跑模式当成展示文案泄给人看（spec §4）。
                detail=("" if form == HITL_FORM_WAIT else p.get("context", "")),
                fields=list(p.get("questions") or []), proposal=p.get("arguments") or None,
                tool_call_id=tool_call_id,
                # 唯一生产 form="question" 的路径是 ask_user
                # （control_capability.py:714），其契约就是「人类答案直接当工具结果」——
                # reply_as_result=True。旧 payload 没有这个字段，默认值会让在途 ask_user
                # 请求跨升级边界后被判成需要 HumanResumable provider（ask_user 从不实现），
                # gateway 按 spec §12.3.5 直接拒绝（Finding: reply_as_result 丢失）。
                reply_as_result=(form == HITL_FORM_QUESTION),
            )
            opened[rid] = req
            snap.pending[rid] = req

        elif ev.type == EventType.HITL_RESOLVED:
            req = opened.get(rid)
            outcome = p.get("outcome", "")
            if req is None or not outcome:
                # 畸形事件（outcome 为空）：不 pop——留在 pending 比消失安全，大不了被
                # 重新问一遍；pop 在校验之前会让这条请求既不在 pending、也没留下决定，
                # 凭空消失（Minor C）。
                continue
            snap.pending.pop(rid, None)
            decision = HitlDecision(
                outcome=outcome, message=content_from_jsonable(p.get("message") or ""),
                modified_arguments=p.get("modified_arguments"),
            )
            if outcome == HITL_OUTCOME_CANCELLED:
                # **取消即了结**，不必等一枚 `HitlClosed`：cancelled 压根不是决定
                # （`fold_cold_hitl_decision` 同一口径），没有后续消费可言——取消本身就是
                # 终点。所以这条既不进 `resolved`/`decisions_for`（下面那个分支管），也不该
                # 继续占着 `opened`：会话关闭 / 熔断 / 逐出都会批量取消，不摘就是一条按
                # 「被取消过多少次」增长的账。
                #
                # 这不是「第二套机制」：它不跨引另一份折叠、不看别的条件，就是本事件自己
                # 载荷里的 `outcome`。而且它天然覆盖 `defer=True` 那条路——延迟收口最终
                # 也走到这条 `HitlResolved`，另发一枚章反而容易漏掉那条分支。
                #
                # 代价同上：取消之后再撤回变成 no-op，方向同样是安全的那边。
                opened.pop(rid, None)
            else:
                # `resolved` **不看 tool_call_id**：`UserTurn` 的 park 本就没有 tool_call，
                # 按它过滤会把整整一类已终局请求丢掉（复审 Finding 2）。
                snap.resolved[rid] = _as_resolved(req, decision, ev.timestamp)
                if req.tool_call_id:
                    key = (req.session_id, req.tool_call_id, req.stage)
                    snap.decisions_for[key] = (decision, req.resume_state)
                    _decision_owner[key] = rid

        elif ev.type == EventType.HITL_CLOSED:
            # 已了结 → 从**两份**账上销掉：补注入清单（`resolved`）与决定缓存
            # （`decisions_for`）。这枚章的语义是「这条决定再也不会被问」，所以两份一起销
            # ——前身设计里它只销前者、后者另靠 capability 事件推，那正是「两套机制」。
            #
            # **只销自己那一条**：`decisions_for` 的键是 `(session, tool_call, stage)` 三维，
            # 同一个键会被「重问副本」的后一条决定覆盖（最后一条可用决定胜出）。不记主人就
            # 直接 pop，一条**旧**请求的了结会把**新**请求的决定连带删掉 → 同一个工具重新
            # 求批。`_decision_owner` 是这一步的本地账。
            #
            snap.resolved.pop(rid, None)
            req = opened.pop(rid, None)
            if req is not None and req.tool_call_id:
                key = (req.session_id, req.tool_call_id, req.stage)
                if _decision_owner.get(key) == rid:
                    snap.decisions_for.pop(key, None)
                    _decision_owner.pop(key, None)
            # **连 `opened` 一起摘**。这是「有界」的最后一环：`opened` 保留每一条开过的
            # 请求、不看结局，所以不摘它，把这个折叠搬进投影的那一步就还是无界的（今天它
            # 只是本地变量，代价藏在 O(n) 的瞬时分配里）。
            #
            # 摘得掉，是因为了结之后没有任何事件还会合法引用它：`HitlResolved` 早过去了，
            # 而撤回（`HitlReplyRetracted`）与了结在活路径上互斥——撤回发生在这一轮提交
            # **之前**，了结发生在持久效果落地**之后**。
            #
            # 于是「了结之后再撤回」变成 no-op（气泡不回 pending）。这比从前留着它更安全：
            # 一条被撤回复活的已了结请求没有任何人在等，而它会经 `parked_task_ids` 永久挡住
            # 那个 task 的重排。失败方向朝「少一条没人等的未决」，不朝「多一条永挡」。

        elif ev.type == EventType.HITL_REPLY_RETRACTED:
            # 一次已收下的答复被收回（spec 2026-09-09）：气泡回到未决，并记一笔
            # 「撤过几次」——后者是 memory 幂等键的第二维（`reply_memory_id`），
            # 撤销之后重启时只有这个数能让重答不撞上上一次的 superseded 记录。
            #
            # 正常流程下这条之前**没有** `HitlResolved`（两阶段：撤销的那次从不发终局
            # 事实），所以这里 pop 通常是 no-op。仍然 pop 是为了对付「先 Resolved 再
            # Retracted」这种外部/存量流：撤回就是撤回，不管它之前算没算终局。
            req = opened.get(rid)
            if req is None:
                continue
            snap.resolved.pop(rid, None)
            req.decision = None
            req.resolved_at = None
            req.reply_attempt += 1
            snap.pending[rid] = req

        elif ev.type in _HITL_RESOLVE_TYPES:
            snap.pending.pop(rid, None)
            req = opened.get(rid)
            decision = _legacy_decision(ev.type, p)
            if ev.type == EventType.HITL_CANCELLED:
                opened.pop(rid, None)         # 取消即了结，与新模型同口径（见上）
                continue
            if req is None or decision is None:
                continue                      # 不可用决定：按未决重问（那会开一条新请求，
                                              # 所以这条留在 opened 里也不会再长）
            snap.resolved[rid] = _as_resolved(req, decision, ev.timestamp, legacy=True)
            if req.tool_call_id:              # 决定缓存只对得上 tool_call 的请求有意义
                key = (req.session_id, req.tool_call_id, req.stage)
                snap.decisions_for[key] = (decision, req.resume_state)
                _decision_owner[key] = rid

    return snap


# ══════════════════════════════════════════════════════════════════════════════
# capability 调用折叠（spec: tool-operations）
# ══════════════════════════════════════════════════════════════════════════════
#
# 一次工具调用在事件流里留下的痕迹，就是它的执行记录——不需要第二个存储。
#
# 曾经有一张 `operations` 表（`OperationStore` / `OperationRecord` / CAS revision）
# 记同样的事。它的核心承诺是「在调 provider 之前把『我要动手了』持久确认下来」，
# 而 `CapabilityInvoked` 经提交门 emit 时**本来就**是先拿到存储确认才返回的——
# 那两次 CAS 是重复劳动。其余几档也各有归宿：
#
#   prepared      → 没有 INVOKED 就是没跑过（授权拒绝时事件流一条都没有）
#   started       → 有 INVOKED、无 FINISHED
#   completed     → 有 FINISHED，payload 带的正是进对话的那份收敛版
#   waiting_human → HITL 有自己的事件集与 `fold_hitl_snapshot`
#   revision CAS  → 唯一消费者（宿主并发处置 API）已删，无并发写者
#
# 与 `reduce_events` 的分工：那个折的是常驻状态（每步都读、进快照）；这个只在恢复
# 与重入时问一次，所以和 `fold_hitl_snapshot` 同类——折出来就用，不存。比 HITL 折叠
# 还省一层：未决 HITL 的年龄没有上界所以不能截尾，而 dangling tool_call 必定在最后一个
# assistant 回合之后，读最近一段即可。
#
# 段 recap 曾经也在这一档，后来进了投影（`RunStateView.pending_recap`）：它和 HITL 一样
# 没有年龄上界，但它的账**自己会销**（每条由它的 DONE 删掉），所以进得起快照；HITL 的
# 未决集则由 `HitlRegistry` 持有，投影里不另存一份。

async def load_events_of_types(
    store, session_id: str, types: "tuple[EventType, ...]",
) -> list[Event]:
    """按类型取该会话的事件——**本文件这些折叠的统一数据入口**。

    `read_session_events_of_types` 是 `EventStore` 的可选扩展；未实现时退化为全量读 +
    内存过滤。这段降级本来在 runtime 里是个只有一个调用方的私有方法，capability 折叠
    落地后成了第二个需要它的人——与其各写各的，不如和折叠住在一起：需要它的理由完全
    一样（「恢复决策按事件折叠，不必全量回放」）。
    """
    try:
        return await store.read_session_events_of_types(session_id, types)
    except NotImplementedError:
        return [e for e in await store.read_by_session(session_id) if e.type in types]


#: 折叠所需的事件类型。供事件库按类型过滤读取，无需全量回放。
CAP_FOLD_EVENT_TYPES: tuple[EventType, ...] = (
    EventType.CAPABILITY_INVOKED,
    EventType.CAPABILITY_FINISHED,
)


@dataclass(frozen=True)
class OperationFacts:
    """一次逻辑调用在事件流里留下的痕迹。**折出来的，不存。**"""

    #: 有 `CapabilityInvoked` = provider 被调用过，副作用可能已发生。
    #: 反过来更有用：**没有就是确定没跑过**——gateway 的 `_record_invocation` 在
    #: provider 之前、在授权与参数校验之后，所以拒绝与校验失败都不会留下这条。
    invoked: bool = False
    finished: bool = False
    #: 各次执行尝试的 invocation_id，按事件顺序。
    attempts: tuple[str, ...] = ()
    #: `CapabilityFinished` 里那份结果——**就是进对话的收敛版**，重放直接用，
    #: 不必也不该再生成一遍。
    result: str | None = None
    outcome: str | None = None
    #: 结果原始长度。`> len(result)` 说明事件 payload 的 8000 上限把它截断了——
    #: 只可能发生在 `spillable=False`（不收敛）的工具上，而那类全是只读可重新派生的。
    result_length: int | None = None
    last_attempt_at: "datetime | None" = None

    @property
    def truncated(self) -> bool:
        """事件里这份结果是不是被截断的（重放前要据此决定重跑还是复用）。"""
        return (self.result_length or 0) > len(self.result or "")


def fold_operations(events: list[Event]) -> dict[str, OperationFacts]:
    """按 tool_call_id 折 `CapabilityInvoked` / `CapabilityFinished`。

    键是摄入点铸造的内部 tool_call 标识（`tc_...`）——裸 wire id 的调用同样会进来，
    调用方自行按 `is_internal_call_id` 取舍（跨回合复用的裸 id 会互相覆盖，那是存量
    数据的既定歧义，见 capability `conversation-integrity`）。
    """
    out: dict[str, OperationFacts] = {}
    for ev in events:
        payload = ev.payload or {}
        tool_call_id = payload.get("tool_call_id") or ""
        if not tool_call_id:
            continue
        facts = out.get(tool_call_id, OperationFacts())
        if ev.type == EventType.CAPABILITY_INVOKED:
            facts = replace(
                facts,
                invoked=True,
                attempts=(*facts.attempts, payload.get("invocation_id", "")),
                last_attempt_at=ev.timestamp,
            )
        elif ev.type == EventType.CAPABILITY_FINISHED:
            facts = replace(
                facts,
                finished=True,
                result=payload.get("result"),
                outcome=payload.get("outcome"),
                result_length=payload.get("result_length"),
            )
        out[tool_call_id] = facts
    return out
