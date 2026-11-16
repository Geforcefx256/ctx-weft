"""HITL 契约：host-facing 的请求形态、等待形态与终局值域。

`form` 是**开放扩展点**（``str`` 而非闭 ``Literal``）：host 可定义自己的等待形态，
core 只负责原样透传、不做白名单校验。``outcome`` 与之对称，同样开放——core 只认
``accepted``/``rejected``/``cancelled`` 三个内建值，其余原样透传、不校验。真正封闭的
是 `Delivery`：host 可以定义新的等待形态，但不能定义新的回灌方式，因此续跑路由的每个
取值 core 都认识、都有确定行为（spec 2026-09-01 §5）。

交换类型见下方「新契约」段：provider 产 `HitlAsk`，core 回 `HitlDecision`，host 读
`HitlRequestView`、经 `HitlReply` 应答。core 内部的活记录 `PendingHitl` 不出 core。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctx_weft.protocols.content import ContentPart


# 三种内建等待形态（spec 2026-07-05，替代旧 kind + capability_id sentinel 拼判）：
#   approval — 审批门控：放行/拒绝一次工具调用（HumanConfirmationAuthorizer 触发）
#   question — ask_user 结构化提问，答复回灌 LLM
#   wait     — act 纯文本暂停 / 软打断（wait_for_user 冷 park）
# host 可自定义其它值；core 不校验。
HITL_FORM_APPROVAL = "approval"
HITL_FORM_QUESTION = "question"
HITL_FORM_WAIT = "wait"

HitlForm = str

#: 终局结果。**开放值域**，与 `HitlForm` 对称——host 定义了自己的 form，就该能定义自己的
#: 结局。core 只认下面三个内建值，其余原样透传、不校验。
#: 空串 `""` 是「未决」哨兵，host 自定义 outcome 不得使用它。
HitlOutcome = str

HITL_OUTCOME_ACCEPTED = "accepted"
HITL_OUTCOME_REJECTED = "rejected"
HITL_OUTCOME_CANCELLED = "cancelled"


# ══════════════════════════════════════════════════════════════════════════════
# 新契约（2026-09-01 重设计）。legacy `HitlRequest` 已于段 2 删除。
# 设计：docs/superpowers/specs/2026-09-01-hitl-redesign-design.md §4 / §5
# ══════════════════════════════════════════════════════════════════════════════

#: `UserTurnDelivery.preface`：注入用户回合时的续接修饰。取代 legacy 的
#: `context` 字符串 sniffing（"plain_text" / "interrupt" / "interrupt:edit"）。
PREFACE_NORMAL = "normal"
PREFACE_AFTER_INTERRUPT = "interrupt"
PREFACE_AFTER_INTERRUPT_EDIT = "interrupt_edit"


@dataclass(frozen=True)
class ToolResultDelivery:
    """决定作为该 tool_call 的结果送达 → 热路径就地重入 / 冷路径 reconcile 精确重入。"""

    tool_call_id: str


@dataclass(frozen=True)
class UserTurnDelivery:
    """决定作为一条 user 消息注入任务对话 → 置 PENDING 重排。"""

    task_id: str
    preface: str = PREFACE_NORMAL


@dataclass(frozen=True)
class NoResumeDelivery:
    """纯通知 / 取消，不续跑。"""


#: **封闭值域**——与开放的 `form` 正交（spec §5）。host 可以定义新的等待形态，
#: 但不能定义新的回灌方式；因此续跑路由的每个取值 core 都认识、都有确定行为。
Delivery = ToolResultDelivery | UserTurnDelivery | NoResumeDelivery


@dataclass
class HitlAsk:
    """「我需要一个人的决定」——provider 产出的纯意图。provider 唯一需要构造的类型。

    展示槽位（prompt / detail / fields / proposal）是**通用**的：因为 form 是开放值域，
    不可能做穷举的 tagged union，host 自定义 form 复用同一组槽位。私有语义不得混进来。
    """

    form: str
    delivery: Delivery
    prompt: str = ""                                  # 给人看的主问题
    detail: str = ""                                  # 展示用补充说明
    fields: list[dict[str, Any]] = field(default_factory=list)   # 结构化提问
    proposal: dict[str, Any] | None = None            # 被门控的参数（approval 用）
    subject_id: str = ""                              # 被门控的能力 id（展示与审计）
    #: 不透明续跑载荷：core 原样保存、重入时原样回传，**永不解读**。必须可序列化。
    resume_state: dict[str, Any] | None = None
    #: True = 人的答复直接作工具结果，重入不发生（`ask_user` 走这条）。
    reply_as_result: bool = False


@dataclass
class HitlDecision:
    """「人给了什么」——core 喂给发起方的结果。无 id、无时间、无 session。"""

    outcome: HitlOutcome
    message: "str | list[ContentPart]" = ""
    modified_arguments: dict[str, Any] | None = None


@dataclass
class HitlReply:
    """host → core 的一次应答命令。

    不再携带模型选择——换模型走 `CtxWeftRuntime.set_agent_llm`/`set_session_llm`
    两条独立命令（批次 B）。原来一次 `reply_to_hitl(reply, resume_hint=...)`
    同时做「换模型」+「应答」两件事；现在是两条调用。

    `agent_id` 是**防呆**，不是路由必需——路由仍全靠 `hitl_id`（全局唯一）。
    要求调用方显式声明「我以为在回复哪个 agent」，与 `PendingHitl.agent_id`
    不符则 `reply_to_hitl` 拒绝，而不是静默按 `hitl_id` 走掉（spec 4.3）。
    """

    hitl_id: str
    outcome: HitlOutcome
    agent_id: str  # 必填：调用方声明「我以为在回复哪个 agent」，与记录不符则拒绝（spec 4.3）
    message: "str | list[ContentPart]" = ""
    modified_arguments: dict[str, Any] | None = None


@dataclass
class HitlRequestView:
    """core → host 的只读视图：渲染 UI 与 pending 列表用。

    刻意**不含** `tool_call_id`——那是 core 的幂等键，host 不需要（spec §4）。
    """

    id: str
    form: HitlForm
    session_id: str
    task_id: str
    created_at: datetime
    agent_id: str = ""
    subject_id: str = ""
    prompt: str = ""
    detail: str = ""
    fields: list[dict[str, Any]] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    outcome: HitlOutcome = ""
    #: 这条未决项要怎么把人的答复送回去——`UserTurnDelivery` = 等一句话，
    #: `ToolResultDelivery` / `NoResumeDelivery` = 有面板要拍板。host 据此
    #: 自行渲染，core 不再提供派生的会话级状态串（2026-09-04 spec §6.5）。
    delivery: Delivery = field(default_factory=NoResumeDelivery)
    resolved_at: datetime | None = None

    @property
    def resolved(self) -> bool:
        """推导而非存储——存两份就有一条要维护的不变量，而漏维护是静默的。"""
        return bool(self.outcome)
