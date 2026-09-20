"""Control plane types.

Phase 6 §6.7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart


@dataclass
class SessionView:
    """Session 的轻量投影，从事件流 reduce 而来。"""

    id: str
    user_prompt: "str | list[ContentPart]" = ""
    template_id: str = ""
    status: str = "RUNNING"
    goal: str = ""
    root_agent_id: str = ""
    llm_model: str = ""
    llm_account: str = ""
    tenant_id: str = "default"
    token_budget: int = 200_000
    context_limit: int = 180_000
    reserved_output_tokens: int = 8192
    failure_counter: int = 0
    #: 本轮是否已熔断（`FailureThresholdHit` 置位，`SessionResumed` 清零）。
    #: `failure_counter` 的折叠据此判断一条 `TaskFailed` 是不是「熔断自己发的那条」
    #: ——trip 序列第 6 步给 root 判死也发 TaskFailed，那是聚合结果、不是第 N+1 次
    #: 新败，重复计入会让恢复后的计数比内存真值多一。
    threshold_tripped: bool = False
    created_at: datetime | None = None


@dataclass
class TaskView:
    """Task 的轻量投影，从事件流 reduce 而来。"""

    id: str
    session_id: str
    status: str = "PENDING"
    title: str = ""
    description: str = ""
    assigned_agent_id: str = ""
    creator_agent_id: str = ""
    parent_task_id: str = ""
    user_prompt: "str | list[ContentPart]" = ""
    interaction_mode: str = "auto"  # interactive=纯文本暂停等用户 / auto=自治（跨重启保留，否则 resume 后丢失暂停语义）
    # 无人值守（跨重启保留）：丢了它，resume 之后一个后台自治任务就变回「有人看顾」，
    # 随后第一次 HITL 会把它 park 到死。见 `Task.unattended`。
    unattended: bool = False
    # 派发来源（跨重启保留）：子任务是被父的哪一次 delegate 调用派出来的。丢了则 finalize
    # 认不出自己的派发框，子任务 close 时既不闭合父的 ack、也不合成 finish 对（胶囊丢失）。
    # memory 侧另有 child_task_id 做一等事实，本字段是 in-run 快捷路径 + 存量数据回退。
    origin_tool_call_id: str = ""
    origin_tool_name: str = ""
    settings_raw: dict[str, Any] = field(default_factory=dict)
    dag_deps: list[str] = field(default_factory=list)
    priority: int = 5
    max_retries: int = 3
    tenant_id: str = "default"
    outputs: Any = None
    error: str | None = None
    # spec: task-handoff——终态事件的结局码（当前唯一写入者：依赖阻塞取消
    # BLOCKED_BY_FAILED_DEP）与阻塞源任务 id。存量事件无此键 → None。
    error_code: str | None = None
    blocked_by_task_id: str | None = None
    created_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass
class AgentView:
    """Agent 的轻量投影，从 task 层级推算而来。"""

    id: str
    spawn_depth: int = 0
    parent_agent_id: str | None = None
    # 该 agent 实例化时用的模板 id，来自 AgentInstantiated 事件（唯一记录它的地方——
    # 树形推算得不出模板）。存量事件流里子 agent 没发过该事件 → 留空，调用方回落
    # session 模板，与改动前行为一致。
    template_id: str = ""
    # 该 agent 当前的模型选择，来自 AgentInstantiated（初值）/ AgentLlmChanged（切换）。
    # D1 修复：跨重启存活——`load()` 据此重建 registry 里的 ModelChoice。
    llm_account: str = ""
    llm_model: str = ""
    # Task 14：五态机当前值，来自 5 个 AGENT_* 事件的折叠（terminated 粘滞，见
    # reducers._AGENT_STATUS_BY_EVENT）。跨重启存活——`load()` 据此重建
    # `_AgentRecord.status`，否则冷恢复后每个 agent 都会被重置成 idle。
    status: str = "idle"
    # 该 agent 正在处理的 task（AGENT_* 事件的 task_id 非空时同步）。
    current_task_id: str | None = None


def _new_hitl_snapshot():
    """延迟 import：`core.hitl` 在 `core.control` **之上**，模块级 import 会成环。"""
    from ctx_weft.core.hitl.snapshot import HitlSnapshot

    return HitlSnapshot()


@dataclass
class RunStateView:
    """Point-in-time view of a run's state for inspect/replay."""

    run_id: str
    session_id: str
    task_id: str
    agent_id: str

    current_step: str | None = None
    task_status: str = "UNKNOWN"
    session_status: str = "UNKNOWN"

    assembled_prompt_tokens: int = 0
    transcript_turns: int = 0
    events_total: int = 0
    #: 这个会话**创建过**多少个 task（只增不减，与 `tasks` 里当下留了几条无关）。
    #:
    #: 存在的理由：快照的 `tasks` 只存活闭包（`prune_view_for_snapshot`），于是「所有
    #: task 都已终态」与「从来没有过 task」都表现为 `tasks` 为空——而恢复路径的空投影
    #: 闸门要区分这两者（前者是正常的已完工会话，后者才是 SESSION_CREATED 之后就崩的
    #: 坏投影）。集合区分不了，就用计数。
    tasks_total: int = 0

    target_event_id: str | None = None
    events_replayed: int = 0

    # Full projections rebuilt from events
    #: HITL 的**活账**（`HitlSnapshot`）：未决请求、已终局未消费的决定、以及折叠期的两份
    #: 工作账。由 `apply_hitl_event` 逐条折出，与一次性 `fold_hitl_snapshot` 同一份实现。
    #:
    #: **为什么它回到了投影里。** 这个判断没有年龄上界（三个月前开出、至今未决的请求今天仍
    #: 必须被看见），所以 `rebuild_hitl` 从前每次冷应答都要把该会话**全部** HITL 事件读回来
    #: 折一遍——实测 800 次人工确认的会话取回 1600 条、读放大 1600×、218ms。进投影之后随
    #: 快照 + 增量走，变成 O(delta)。
    #:
    #: **为什么现在进得起。** 四份账都有退役了（`HitlClosed` 销 `resolved` /
    #: `decisions_for` / `opened`，`outcome=cancelled` 销 `opened`），所以大小 ≈ 活跃数，
    #: 不随会话长度增长。当初把 HITL 移出投影是因为**两份折叠会漂移**，而现在折叠只有一份
    #: （`apply_hitl_event`），那条理由不再适用——registry 仍是查询的唯一真相源，投影只是
    #: 它的装填来源。
    #:
    #: ⚠️ 两样东西**不进** blob（见 `serialize_view`）：`slot`（活的 asyncio 对象）与
    #: `pending_decision`（待终局——它对应的 `HitlResolved` 压根没落库，冷重建必须把那条
    #: 请求看成未决）。后者是结构保证的：折叠只在 `HitlResolved` 时写 `decision`。
    hitl: "HitlSnapshot" = field(default_factory=lambda: _new_hitl_snapshot())

    #: 被崩溃打断的段 recap：{task_id: {"boundary", "agent_id"}}。
    #:
    #: 某 task 有 `TASK_RECAP_STARTED` 而无其后的 `TASK_RECAP_DONE`，说明那段 background
    #: observe 的 memory 写没落完，恢复据此重跑。
    #:
    #: **为什么它该进投影**：这个判断没有时间下界——几个月前那条无 done 的 started 今天
    #: 仍然要重跑。所以从前每次 `/resume` 都要把该会话**全部** recap 事件读回来折一遍，
    #: 即便收窄到那两种类型，代价仍随会话长度线性增长（实测 1000 个 task 的会话：取回
    #: 1999 条折出 1 个，272ms / 5.1MB）。进了投影就随快照 + 增量走，变成 O(delta)。
    #:
    #: **为什么它进得起投影**（不像 `tasks` 那样要裁）：每个条目由它自己的
    #: `TASK_RECAP_DONE` 删掉，所以大小 = 「崩溃打断且尚未修复的 recap 数」≈ 0~2，
    #: 与会话长度无关。
    #:
    #: ⚠️ **不得按 task 存活性裁**：recap 跑在 finish 边界上，待重跑的 recap 往往属于一个
    #: 已 FINISHED 的 task——拿 `tasks` 的活闭包过滤它会把它们全丢掉。
    pending_recap: dict[str, dict] = field(default_factory=dict)

    sessions: dict[str, SessionView] = field(default_factory=dict)
    tasks: dict[str, TaskView] = field(default_factory=dict)
    agents: dict[str, AgentView] = field(default_factory=dict)
