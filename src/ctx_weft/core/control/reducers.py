"""Event reducers：把 event 序列重建成 state。

Phase 6 §6.6。核心是 reduce(events) → RunStateView。
RunStateView.sessions / .tasks 包含完整的 Session/Task 投影。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ctx_weft.core.utils.content import (
    content_from_jsonable,
    content_to_jsonable,
)
from ctx_weft.core.control.types import AgentView, RunStateView, SessionView, TaskView
from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HITL_STAGE_TOOL, PendingHitl
from ctx_weft.core.hitl.snapshot import HitlSnapshot
from ctx_weft.core.models.status import WAITING, TaskStatus
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


def fold_pending_task_recap(events: list[Event]) -> dict[str, dict]:
    """折叠 TaskRecap 事件 → 仍未完成的 {task_id: {"boundary", "agent_id"}}（started 减去 done）。

    某 task 有 TASK_RECAP_STARTED 而无其后的 TASK_RECAP_DONE，说明该段 background observe 的
    memory 写未持久完成（崩溃在中途）——恢复据此重跑。同 task_id last-write-wins。
    """
    pending: dict[str, dict] = {}
    for ev in events:
        p = ev.payload or {}
        tid = p.get("task_id", "")
        if not tid:
            continue
        if ev.type == EventType.TASK_RECAP_STARTED:
            pending[tid] = {"boundary": p.get("boundary", ""), "agent_id": p.get("agent_id", "")}
        elif ev.type == EventType.TASK_RECAP_DONE:
            pending.pop(tid, None)
    return pending


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
                "original_user_prompt": content_to_jsonable(t.original_user_prompt),
                "interaction_mode": t.interaction_mode,
                "unattended": t.unattended,
                "origin_tool_call_id": t.origin_tool_call_id,
                "origin_tool_name": t.origin_tool_name,
                "settings_raw": t.settings_raw,
                "dag_deps": t.dag_deps,
                "priority": t.priority,
                "max_retries": t.max_retries,
                "timeout_ms": t.timeout_ms,
                "budget_consumed": getattr(t, "budget_consumed", None),
                "tenant_id": t.tenant_id,
                "outputs": t.outputs,
                "error": t.error,
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
            original_user_prompt=content_from_jsonable(t.get("original_user_prompt", "")),
            interaction_mode=t.get("interaction_mode", "auto"),
            # 存量快照无此键 → False（无人值守是新增语义，旧数据一律「有人在」）。
            unattended=t.get("unattended", False),
            origin_tool_call_id=t.get("origin_tool_call_id", ""),
            origin_tool_name=t.get("origin_tool_name", ""),
            settings_raw=t.get("settings_raw", {}),
            dag_deps=t.get("dag_deps", []),
            priority=t.get("priority", 5),
            max_retries=t.get("max_retries", 3),
            timeout_ms=t.get("timeout_ms", 60_000),
            budget_consumed=t.get("budget_consumed"),
            tenant_id=t.get("tenant_id", "default"),
            outputs=t.get("outputs"),
            error=t.get("error"),
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
        sessions=sessions,
        tasks=tasks,
        agents=agents,
    )


#: 恢复路径认可的投影版本（spec: snapshot-recovery）：不匹配的快照被忽略走全量。
#: 与 SnapshotWriter.PROJECTION_VERSION 同步 bump。
_PROJECTION_VERSION = 1


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
    if (
        snapshot is not None
        and snapshot.last_commit_position is not None
        and snapshot.projection_version == _PROJECTION_VERSION
        and snapshot.last_commit_position <= head
    ):
        view = deserialize_view(snapshot.state_blob)
        delta = await event_store.read_range(
            session_id,
            after_position=snapshot.last_commit_position,
            through_position=head,
        )
        return apply_events([se.event for se in delta], view)
    # 全量：忽略快照（存量/损坏/超前），按 position 序重放；重造快照由 writer 负责
    stored = await event_store.read_range(
        session_id, after_position=0, through_position=head)
    return reduce_events([se.event for se in stored], run_id=session_id)


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
        view.task_status = "ACTIVE"
        # 会话状态不在此写：run 是任务级的，会话状态归 SessionRegistry
        # （docs/events-v2.md §2.1.1）。
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
                timeout_ms=task_data.get("timeout_ms", 60_000),
                budget_consumed=task_data.get("budget_consumed"),
                tenant_id=ev.tenant_id or "default",
                created_at=ev.timestamp,
            )
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
        # 重排（observer active 或 review reopen）：状态回 PENDING、清旧产出；
        # reopen 还会携带改写后的 user_prompt / 原始 prompt 快照，replay 时一并恢复。
        task = view.tasks.get(ev.task_id)
        if task is not None:
            task.status = "PENDING"
            task.outputs = None
            up = p.get("user_prompt")
            if up is not None:
                task.user_prompt = content_from_jsonable(up)
            oup = p.get("original_user_prompt")
            if oup is not None:
                task.original_user_prompt = content_from_jsonable(oup)
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

    elif t == EventType.TASK_FINALIZED and ev.task_id:
        task = view.tasks.get(ev.task_id)
        if task is not None:
            # 本事件的 payload 里**有** outputs{output,summary}/error（2026-09-05 起，
            # 供 host 落 tasks 表与 CLI 打印），但这里刻意不折：投影的 outputs/error 只认
            # TaskFinished/TaskFailed/TaskInterrupted 一个来源（总账 A1）——两处都写会让
            # 先到的 TaskFinalized 覆盖或清空后到的成果，正是 A1 当初要治的那个漂移。
            task.finished_at = ev.timestamp

    # ── LLM / Context ─────────────────────────────────────────────────────────
    elif t == EventType.PREPARE_COMPLETED:
        view.assembled_prompt_tokens = p.get("assembled_token_count", 0)
    elif t == EventType.ACT_TURN_COMPLETED:
        view.transcript_turns = p.get("turn", view.transcript_turns)

    # ── HITL projection ───────────────────────────────────────────────────────
    # **只投影会话状态**：pending HITL 的真相源是 `HitlRegistry`（由 `fold_hitl_snapshot`
    # 装填），不再在 RunStateView 里另存一份——两份口径不同的 HITL 折叠正是旧实现里
    # 「重建了 pending 却没重建已解决」那类漂移的来源。
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


def fold_hitl_snapshot(events: list[Event]) -> HitlSnapshot:
    """双读折叠：新旧两套 HITL 事件 → `HitlSnapshot`。

    同 tool_call 有多条请求（重问副本）时，**最后一条可用决定胜出**。

    **event 侧 ref 尚未 hydrate**：`decisions_for[*]` 里的 `HitlDecision.message` 就是
    事件载荷里存的、经 `content_from_jsonable` 转回的 `ContentPart`——但那仍是**事件**
    blob store 命名空间下的引用。装填进 registry、被工具结果/记忆消费之前，调用方须比照
    `runtime.py:1943-1961` 的 `_cold_hitl_decision`：先 `hydrate_event_content`，再
    `normalize_content` 写入**记忆** blob store，并对失败做 `downgrade_images_to_text`
    兜底（该兜底绝不可再抛）。本函数本身保持同步、不做这一步（spec §12.3.3）。
    """
    snap = HitlSnapshot()
    opened: dict[str, PendingHitl] = {}
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
            if outcome != HITL_OUTCOME_CANCELLED:
                # `resolved` **不看 tool_call_id**：`UserTurn` 的 park 本就没有 tool_call，
                # 按它过滤会把整整一类已终局请求丢掉（复审 Finding 2）。
                snap.resolved[rid] = _as_resolved(req, decision, ev.timestamp)
                if req.tool_call_id:
                    key = (req.session_id, req.tool_call_id, req.stage)
                    snap.decisions_for[key] = (decision, req.resume_state)

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
            if req is None or decision is None:
                continue                      # HITL_CANCELLED / 不可用决定：按未决重问
            snap.resolved[rid] = _as_resolved(req, decision, ev.timestamp, legacy=True)
            if req.tool_call_id:              # 决定缓存只对得上 tool_call 的请求有意义
                key = (req.session_id, req.tool_call_id, req.stage)
                snap.decisions_for[key] = (decision, req.resume_state)

    return snap
