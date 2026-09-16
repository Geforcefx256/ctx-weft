"""Task 与它的 settings。

`TaskSettings` 三种形态里只有 `NormalTaskSettings` 还在运行期产生，另两种保留
只为反序列化存量事件流（见各自 docstring）。它们与 `Task` 同住一个模块，因为
`Task.settings` 直接引用这个联合类型，两者一起改、一起读。

`TaskInteractionMode` 也在这里：它是「纯文本 turn 怎么处理」的行为模式，不是生命
周期状态，故不进 `status.py`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from ctx_weft.core.models.status import TaskStatus

if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart


# ── TaskSettings ──────────────────────────────────────────────────────────────


@dataclass
class NormalTaskSettings:
    """Settings for a regular reasoning task.

    Created by delegate_task / delegate_plan control tools, or supplied
    as initial_task_settings to start_session() / create_session().
    """

    skill_name: str = ""
    use_subagent: bool = False
    subagent_template: str = ""
    inherit_memory: bool = True
    # 继承源的**显式**指定（spec/09 §6）：`inherit_memory` 回答「要不要继承」，本字段
    # 回答「继承谁的」。留空 = 沿用推导（父任务的 agent → 上一条 root task 的 agent），
    # 存量行为逐字不变；非空则直接锚定该 agent，绕过整条推导。
    # 之所以住在 settings 而不是 Task 上：它是**装配决策**（这个 agent 起跑时手里有什么），
    # 与 use_subagent/subagent_template 同族，且只在装配那一刻被读一次——不像
    # interaction_mode/unattended 那样要被 reducer、HitlService、restore 反复直读，
    # 不需要自己的投影槽位，跟着 settings_raw 走即可。
    inherit_from_agent_id: str = ""
    purpose: str = "act"
    # Transient accumulator: written by submit_* control tools, cleared by SuspendStep.
    spawn_titles: list[str] = field(default_factory=list)


@dataclass
class CompactTaskSettings:
    """Settings for a legacy memory-compact sub-task (initial_step='compact').

    No longer created at runtime (compact now runs as an inline step). Retained for
    deserializing old persisted event streams.
    compact_scope_task_id is NOT stored here — Task.parent_task_id carries it.
    """

    compact_scope_agent_id: str = ""
    keep_last: int = 6
    subagent_template: str = "memory-compactor"
    layer: str = "agent"  # spec/06 §7：'agent'=折叠派发日志 / 'task'=折叠 task 对话


@dataclass
class MetadataFillerTaskSettings:
    """Settings for a legacy metadata-filler daemon task.

    No longer created at runtime (recognize_intent now runs as a background
    coroutine). Retained for deserializing old persisted event streams.
    """

    target_task_id: str = ""
    is_daemon: bool = True


TaskSettings = NormalTaskSettings | CompactTaskSettings | MetadataFillerTaskSettings


def deserialize_settings(d: dict | None) -> TaskSettings:
    """Reconstruct a typed TaskSettings from a raw dict (as stored in events)."""
    if not d:
        return NormalTaskSettings()
    d = dict(d)
    t = d.pop("_type", "NormalTaskSettings")
    if t == "CompactTaskSettings":
        known = {k: v for k, v in d.items() if k in CompactTaskSettings.__dataclass_fields__}
        return CompactTaskSettings(**known)
    if t == "MetadataFillerTaskSettings":
        known = {k: v for k, v in d.items() if k in MetadataFillerTaskSettings.__dataclass_fields__}
        return MetadataFillerTaskSettings(**known)
    known = {k: v for k, v in d.items() if k in NormalTaskSettings.__dataclass_fields__}
    return NormalTaskSettings(**known)


# How the actor's plain-text (no tool call) turn is handled:
#   "interactive" → pause and wait for a user message (HITL input cold park)
#   "auto"        → autonomous; the actor must call finish_task to finish
TaskInteractionMode = Literal["interactive", "auto"]


# ── Task ──────────────────────────────────────────────────────────────────────


@dataclass
class Task:
    """会话内的最小工作单元。"""

    id: str
    session_id: str
    status: TaskStatus
    tenant_id: str = "default"

    assigned_agent_id: str | None = None
    creator_agent_id: str | None = None
    parent_task_id: str | None = None

    dag_deps: list[str] = field(default_factory=list)
    tracking_task_ids: list[str] = field(default_factory=list)

    title: str = ""
    description: str = ""
    user_prompt: "str | list[ContentPart] | None" = None
    # 首次执行时的原始 user_prompt 快照；reopen 重写 user_prompt 时以此为 base，
    # 避免多轮 reopen 把"上轮产出/修订提示"反复累加进 prompt。None = 尚未快照。
    original_user_prompt: "str | list[ContentPart] | None" = None
    # 事件侧的 prompt 形态（event ref，`content_to_event_jsonable` 的产物）。
    # **纯瞬态**：不进 TaskProjection / 快照——恢复时由 Runtime._restore_task_prompts
    # 从事件 payload 直接重填，那本来就是这份数据的原样形态，没必要再持久化第二遍。
    # reopen_task 据此发 TASK_REQUEUED，零 blob IO：reopen 只追加文本，不可能引入新图。
    user_prompt_event_jsonable: "str | list[dict] | None" = None
    original_user_prompt_event_jsonable: "str | list[dict] | None" = None
    user_prompt_in_memory: bool = False
    # 这条 user_prompt 落进 memory 时拿到的 record id（`_persist_user_prompt` 记下）。
    # 用途只有一个：用户在 LLM 开口之前按了暂停，act 的丢弃路径据它 `memory.fold([id], [])`
    # 把这一轮的用户消息纯遗忘掉（spec 2026-09-09）。**纯瞬态**，不进投影/快照——
    # 跨重启之后这一轮要么早已提交、要么根本不在日志里，两种情形都用不到它。
    user_prompt_memory_id: str | None = None
    settings: TaskSettings = field(default_factory=NormalTaskSettings)
    # 纯文本(无 tool call)turn 的处理方式：interactive=暂停等用户 / auto=自治需调 finish_task。
    # root task 由 session_registry 设为 interactive；委派子任务默认 auto（delegate_task/plan 可显式置 interactive）。
    interaction_mode: TaskInteractionMode = "auto"
    # 「这个任务没有人看顾」——后台自治作业的标记，**不是** interaction_mode 的别名。
    # 三个字段各答一个不同的问题：`interaction_mode` 答「纯文本回合要不要停下来等人」，
    # `settings.token_budget` 之类答「允许花多少」，本字段答的是**有没有人在**。
    # 唯一用途：在 `HitlService.open()`（HITL 的唯一登记入口）一处堵死——无人值守的
    # task 发起任何 HITL 都会 park 到死，因为没有人会来应答。
    # 不变式 `unattended ⟹ interaction_mode == "auto"` 由**设置点**保证（root task 见
    # `SessionRegistry._make_root_task_manager`、外部消息见 `Runtime._start_task_for_agent`，
    # 委派子任务见 `control_tools._child_mode` 的「父不 interactive 则子不 interactive」）：
    # 无人值守却 interactive，意味着一次纯文本回合就永久挂起。
    unattended: bool = False
    outputs: Any | None = None
    process_report: str | None = None
    # 何时设置 process_report（= 上一轮 observe 产出反馈的时刻，落在该 attempt 之后、下一 attempt 之前）。
    # process_report/process_report_at 现仅服务终态 finish 对 / 子任务 bubble；retry 进度已改由 task 层 TASK_COMPACT_SUMMARY 段摘要承载（spec 2026-07-01 §3.7）。
    process_report_at: datetime | None = None
    # observer 给「下一次 act attempt」的一次性转向：next_step_hint + success-without-outputs
    # 护栏文案（见 control_capability.report_task_outcome）。
    # **刻意不并进 process_report/act_recap**：后者是永久记录（→ TASK_COMPACT_SUMMARY 段摘要、
    # finish 对 assistant 槽），一次性指令混进去会在任务完成后仍留在历史里，跨 task 召回时表现为
    # 「一句已经过期的 Next Step Hint 夹在已完成任务的对话中间」。本字段改由 act_guidance 渲染进
    # 每轮 guidance（只发不入 memory、priority 1 受保护），随任务终结自然消失。
    next_step_hint: str | None = None

    retry_count: int = 0
    max_retries: int = 3

    compensation: dict[str, Any] | None = None
    priority: int = 5
    error: str | None = None
    error_code: str | None = None
    actor_done: bool = False
    # 本轮 act 里 actor 调了 delegate_task / delegate_plan：**这一次 run 该停在
    # 「等子任务」**。纯瞬态、纯 loop 内路由用（ActStep 据它路由到 SuspendStep 并
    # 跳过最终产出合成），每次 _run_task 派发时归零。
    # 它**不是** task 状态：task 落不落 SUSPENDED 由 TaskManager 据 RunOutcome 定
    # （Task 4）。此前这个意图借 `task.status = "SUSPENDED"` 表达，是判决越界写状态。
    suspend_requested: bool = False
    # observe 裁决（三态）：success|retry|fail。retry 置 status=PENDING 重排（机械退出也归 retry）。
    observer_outcome: str | None = None
    # observer 产出的整段综合总结（执行历程+结果）→ finish 对 tool 槽（spec 2026-06-30）。
    task_summary: str | None = None
    # 发起本任务的 parent delegate_task/delegate_plan 的 tool_call_id（spec/06 §5）。
    # finalize 据此把 output+report 作为 TASK_DISPATCH_RESULT 回填 parent agent 层、按它配对。
    origin_tool_call_id: str | None = None
    # 发起本任务的派发工具的**限定名**，finalize 铸派发框时用作 tool_calls[].name（保真）。
    # delegate_task → 真名 control__delegate_task（actor 确实调过）；delegate_plan 子 → None
    # （actor 只调过一次 delegate_plan、无 per-child 调用）→ finalize 回退 start_task 叙事名。
    # 与 origin_tool_call_id 同走 TASK_CREATED payload → TaskView → 恢复链（2026-08-19 补，
    # 此前是纯瞬态、重放后为 None）；存量事件仍无此字段 → 重建后为 None → 回退 start_task。
    origin_tool_name: str | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
    # task manager 真正启动本 task（本 attempt）的时刻。派发框（start_task/dispatch 对）以此为锚，
    # 排在子 body 之前、反映真实启动顺序（避免 created_at 的兄弟碰撞/乱序）。每次 run 重置。
    started_at: datetime | None = None
    finished_at: datetime | None = None
