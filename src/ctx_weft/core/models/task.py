"""Task 与它的 settings。

`TaskSettings` 三种形态里只有 `NormalTaskSettings` 还在运行期产生，另两种保留
只为反序列化存量事件流（见各自 docstring）。它们与 `Task` 同住一个模块，因为
`Task.settings` 直接引用这个联合类型，两者一起改、一起读。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from ctx_weft.core.models.status import TaskStatus

if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart


# ── 交互口（port）────────────────────────────────────────────────────────────
#
# port 是 session 的**对外交互端口**：session 边界之内的事靠 task DAG（依赖、父子
# 挂起），跨出边界的往返靠 port。对端是人、是另一个 agent、还是外部系统，SDK 不
# 关心——它只保证**同一个口上至多一个 task 占着往返**（`TaskManager._held_ports`）。
#
# 互斥的理由不是「人只有一个」，而是一次往返的应答必须能路由回发起方：两个 task
# 同时在一个口上问话，回来的答复无从归属。所以需要串行化的往返才叫 port；不需要
# 串行的对端（无状态、带 correlation id、随便打）根本不必是 port，那就是普通并发。

#: 默认交互口。没有显式声明的 task 都接在这里——这个库里最常见的情形是有人在对话。
PORT_MAIN = "main"
#: 不接任何口：自治作业，既不往返也不推送。
PORT_NONE = ""


def default_port_for(unattended: bool) -> str:
    """没有显式 port 声明时的回落。

    `unattended` 与 `port_key` 是**两个问题**（见 `Task.port_key`），但在没有第二
    个字段的年代，「无人值守」同时兼任了「不接口」。存量事件流与尚未打通 port 参数
    的构造点都靠这条公式回落，两边共用同一个真相，避免 reducer 与运行期各推一套。
    """
    return PORT_NONE if unattended else PORT_MAIN


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
    # `unattended` 那样要被 reducer、HitlService、restore 反复直读，
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


# ── Task ──────────────────────────────────────────────────────────────────────


@dataclass
class Task:
    """会话内的最小工作单元。"""

    id: str
    session_id: str
    status: TaskStatus
    # Recovery may retain a dependency solely for scheduling status checks.
    details_complete: bool = field(default=True, kw_only=True)
    tenant_id: str = "default"

    assigned_agent_id: str | None = None
    creator_agent_id: str | None = None
    parent_task_id: str | None = None

    dag_deps: list[str] = field(default_factory=list)

    title: str = ""
    description: str = ""
    user_prompt: "str | list[ContentPart] | None" = None
    # 事件侧的 prompt 形态（event ref，`content_to_event_jsonable` 的产物）。
    # **纯瞬态**：不进 TaskProjection / 快照——恢复时由 Runtime._restore_task_prompts
    # 从事件 payload 直接重填，那本来就是这份数据的原样形态，没必要再持久化第二遍。
    user_prompt_event_jsonable: "str | list[dict] | None" = None
    #: 这条 user_prompt 已经进了对话历史（memory，或这一轮未提交窗口的暂存区——装配时两者
    #: 都看得见，见 `_persist_user_prompt`）。
    user_prompt_in_memory: bool = False
    settings: TaskSettings = field(default_factory=NormalTaskSettings)
    # 「这个任务没有人看顾」——后台自治作业的标记。
    #
    # 2026-09-22 起它是**纯文本回合归宿的唯一判据**：有人在就 park 让位（park 与后台
    # observe 并行，observer 判 success 则终结本 task、判 retry/fail 则维持 park 等人），
    # 没人在就当作任务产出、走前台 observe。此前那个 `interaction_mode` 字段答的是
    # 「纯文本回合要不要停下来等人」——一个 task 级静态配置回答一个**逐回合**的语义，
    # 必然有一半时候是错的，已随判定权交给 observer 而删除。
    #
    # 它还有两个更早的用途，都源自同一个事实「有没有人在」：`HitlService.open()`
    # （HITL 的唯一登记入口）据它堵死无人值守的 HITL——没人应答就是 park 到死；
    # `TaskManager._held_ports` 据它判断这个 task 占不占交互口。
    #
    # 与 `port_key` 的分工（见下）：本字段答「这一轮**要不要等**对端」，那个答
    # 「对端**是谁**」。两者正交，四种组合里三种合法——尤其 `port_key` 非空 +
    # `unattended=True` 是「接口但自治」：跑完把产出推到那个口，中途不问、不让位。
    unattended: bool = False
    #: 这个 task 接在 session 的哪个交互口上（见模块顶部的 port 段）。空 = 不接任何口。
    #:
    #: 不变式：``unattended=False ⟹ port_key 非空``——要等对端，就得先有个口。
    #: 校验落在 `TaskManager.push_task`（唯一的新建路径），不落 `__post_init__`：
    #: 崩溃重建走 `restore` 不经那条路，把校验放构造器只会让恢复在坏数据上炸。
    #:
    #: **占口判据**（`TaskManager._held_ports`）：``port_key 非空 且 not unattended
    #: 且（在跑 or AWAITING_HUMAN）``。`AWAITING_HUMAN` 必须算——park 一发生槽位就
    #: 还回来了，只看在跑会让同一个口上出现两条同时等应答的往返。
    #:
    #: 沿 task 树继承（`delegate_task` / `delegate_plan` 与 `unattended` 一起传），
    #: 不给 LLM 旋钮：子任务的产出流向同一个对端，这是作业被怎么起起来的事实。
    port_key: str = PORT_MAIN
    outputs: Any | None = None
    process_report: str | None = None
    # 何时设置 process_report（= 上一轮 observe 产出反馈的时刻，落在该 attempt 之后、下一 attempt 之前）。
    # process_report/process_report_at 现仅服务终态 finish 对 / 子任务 bubble；retry 进度已改由 task 层 TASK_COMPACT_SUMMARY 段摘要承载（spec 2026-07-01 §3.7）。
    process_report_at: datetime | None = None
    # observer 给「下一次 act attempt」的一次性转向：next_step_hint + success-without-outputs
    # 护栏文案（见 control_capability.report_observation）。
    # **刻意不并进 process_report/act_recap**：后者是永久记录（→ TASK_COMPACT_SUMMARY 段摘要、
    # finish 对 assistant 槽），一次性指令混进去会在任务完成后仍留在历史里，跨 task 召回时表现为
    # 「一句已经过期的 Next Step Hint 夹在已完成任务的对话中间」。本字段改由 act_guidance 渲染进
    # 每轮 guidance（只发不入 memory、priority 1 受保护），随任务终结自然消失。
    next_step_hint: str | None = None

    #: 运行异常的重试次数；observer 判 continue 重排不递增、不受 max_retries 限制。
    retry_count: int = 0
    max_retries: int = 3
    #: 自治作业从 `INTERRUPTED` 被自动退避重排过几次（见
    #: `TaskManager._schedule_autonomous_requeue`）。与 `retry_count` **不同源**：那个数
    #: 的是 run 内的原地重试，这个数的是「停下来、退避、再起一次」的轮数，而后者的触发
    #: 条件（LLM outage、崩溃）恰恰是前者不允许原地重试的那些。
    #:
    #: **刻意不持久化**（不进 TASK_CREATED payload / `TaskView` / reducers）：`restore`
    #: 今天就无条件、无预算地重排一切非终态 task，所以「进程重启后预算归零」与既有行为
    #: 一致，不是新的放松；持久化要多改四处，换的是一个既有路径本来就没有的严格性。
    interrupt_requeue_count: int = 0

    compensation: dict[str, Any] | None = None
    priority: int = 5
    error: str | None = None
    error_code: str | None = None
    failure_retry_count: int | None = None
    blocked_by_task_id: str | None = None
    actor_done: bool = False
    # 本轮 act 里 actor 调了 delegate_task / delegate_plan：**这一次 run 该停在
    # 「等子任务」**。纯瞬态、纯 loop 内路由用（ActStep 据它路由到 SuspendStep 并
    # 跳过最终产出合成），每次 _run_task 派发时归零。
    # 它**不是** task 状态：task 落不落 SUSPENDED 由 TaskManager 据 RunOutcome 定
    # （Task 4）。此前这个意图借 `task.status = "SUSPENDED"` 表达，是判决越界写状态。
    replaces_task_id: str = ""
    switched_to_task_id: str = ""
    switch_reason: str = ""
    switch_review_started: bool = False
    switch_requested: bool = False  # transient routing intent; status belongs to TaskManager
    switch_review_ready: Any = field(default=None, repr=False, compare=False)
    suspend_requested: bool = False
    # observe verdict: success/fail close, continue requeues, waiting awaits user input.
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
    finalized: bool = False
