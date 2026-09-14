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
    # reopen 重写前的原始 prompt 快照（防多轮累加，跨重启保留）
    original_user_prompt: "str | list[ContentPart]" = ""
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
    # spec: execution-limits（wp8）——budget 已消费量投影
    tenant_id: str = "default"
    outputs: Any = None
    error: str | None = None
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

    target_event_id: str | None = None
    events_replayed: int = 0

    # Full projections rebuilt from events
    sessions: dict[str, SessionView] = field(default_factory=dict)
    tasks: dict[str, TaskView] = field(default_factory=dict)
    agents: dict[str, AgentView] = field(default_factory=dict)
