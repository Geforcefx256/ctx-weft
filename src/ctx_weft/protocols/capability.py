"""Capability 协议层。

三种 capability 子类：
  ToolCapability   — LLM 可直接调用，走 ToolCapabilityProvider.invoke()
  SkillCapability  — 轻量描述符，不含 dir/source；provider 负责所有加载/执行
  AgentCapability  — sub-agent 模板描述符，由 orchestrator spawn

三种 provider 子类：
  ToolCapabilityProvider  — 实现 invoke() / cancel()
  SkillCapabilityProvider — 实现 Level2 load_definition() + Level3 list_files/load_resource/exec_script
  AgentCapabilityProvider — list() 发现 + get_template() 加载，发现与加载同源
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.operations import RecoveryPolicy

if TYPE_CHECKING:
    from ctx_weft.protocols.context import ContentPart
    from ctx_weft.protocols.hitl import HitlAsk, HitlDecision
    from ctx_weft.protocols.template import AgentTemplate


# ── Purpose ───────────────────────────────────────────────────────────────────

Purpose = Literal["act", "observe", "compact", "recognize_intent"]


# ── Qualified tool name ────────────────────────────────────────────────────────


def qualify(capability_id: str) -> str:
    """capability_id → provider-qualified, LLM-safe tool name.

    LLM function names allow only ``^[A-Za-z0-9_-]{1,64}$`` — colons are invalid,
    so the ``provider:tool`` id is encoded with ``__`` as the separator
    (``mcp:github:create_issue`` → ``mcp__github__create_issue``). This is the
    single canonical tool name the LLM sees, stores in memory, and is classified
    by. Routing still uses the raw ``cap.id``.
    """
    return capability_id.replace(":", "__")


# ── Capability 基类 + 三个子类 ─────────────────────────────────────────────────

@dataclass
class Capability:
    id: str
    name: str
    kind: Literal["tool", "skill", "agent"]
    description: str = ""
    purposes: list[Purpose] = field(default_factory=lambda: ["act"])

    def __post_init__(self) -> None:
        # 归一 description：声明为 str，但 provider 可能透传 None（如无描述的 MCP 工具 /
        # sub-agent 模板）。None 会经 CapabilitySource 落成 ContextBlock.content=None，
        # 装配期 content_to_text 迭代 None 崩溃。在唯一构造入口堵住，覆盖所有子类/provider。
        if self.description is None:
            self.description = ""


@dataclass
class ToolCapability(Capability):
    """LLM 可直接调用，走 ToolCapabilityProvider.invoke()。"""
    kind: str = "tool"
    input_schema: dict[str, Any] = field(default_factory=dict)
    side_effects: bool = False
    spillable: bool = True  # 输出超长时是否允许 gateway 落盘；可重新派生的只读工具置 False
    # spec: tool-operations（wp6）——恢复策略（崩溃后该工具的 started 操作能不能自动
    # 重跑）。**只有两类**，判据是「core 要不要做决定」：
    #   idempotent  重跑安全 → core 同 op_id 直接重跑
    #   reviewed    默认——core 绝不自行重跑，交裁决链：Provider 实现
    #               OperationAdjudicator 则由它答「该不该重跑」，未实现则 core 代为
    #               作结「无从查证」。两条都不停机——不确定是工具结果不是控制流。
    # 刻意不从 side_effects 推断：MCP/旧 Provider 的副作用声明可能不完整（方案 §5.4）。
    # **这里是恢复策略的唯一读取点**（reconcile 读活声明）；账本行上的同名列不参与判定。
    recovery_policy: RecoveryPolicy = RecoveryPolicy.REVIEWED


@dataclass
class SkillCapability(Capability):
    """轻量描述符：不含 dir / source / remote_source_name。
    provider 通过 cap.id prefix 隐式关联（"local_skill:x" → LocalSkillCapabilityProvider）。
    """
    kind: str = "skill"
    triggers: list[str] = field(default_factory=list)
    version: str = ""


@dataclass
class AgentCapability(Capability):
    """Sub-agent 模板描述符，由 orchestrator 负责 spawn。"""
    kind: str = "agent"
    template_name: str = ""
    version: str = ""  # 信息性（listing 展示）；加载一律 version=None 取最新


# ── Level 2（按需加载，不进 cache）────────────────────────────────────────────

@dataclass
class SkillDefinition:
    skill_id: str      # 对应 SkillCapability.id
    skill_name: str
    instructions: str  # SKILL.md frontmatter 之后的全文


# ── 公共事件 / 元信息 ──────────────────────────────────────────────────────────

@dataclass
class CapabilityEvent:
    kind: Literal["progress", "stdout", "stderr", "result", "error", "needs_human", "pin"]
    # pin：provider 声明「把 payload["capabilities"]（list[Capability]）加进当前 task 的可用面」。
    # **与 needs_human 不同，它不终止流**——gateway 就地把这批能力 pin 进 CapabilityCache，
    # 然后继续消费，工具随后照常 yield 自己的 result。用于「一个工具在运行中才发现下一步该用
    # 哪些工具」（按需展开工具面）：pin 进去的能力当轮即对 LLM 可见，因为 AssembledPrompt.tools
    # 是活的（每次读都问 cache），不是装配期的快照。
    # 生命周期挂在 task 上：同 task 的 retry 保留，task 落终态或 context_limit 退出时清。
    # needs_human：provider 声明「我需要一个人的决定」，payload["ask"] 是 HitlAsk。
    # **必须是流的最后一个事件**——gateway 见之即停止消费本流，其后 yield 的一律不可见
    # （spec §2）。让出时生成器被关闭，局部状态随之消失，故让出前的工作要放进
    # ask.resume_state。
    #
    # ── result 的 payload 形状（工具返图看这里）────────────────────────────────
    # {"content": str | list[ContentPart], "metadata": dict}。``content`` 与三个执行
    # 入口、``HitlReply.message``、``AuthorizationDecision.message`` **是同一个联合
    # 类型**：要返图就把 ``ImagePart`` 直接放进去，不必分两处交，也不必自己拆文本。
    #
    #     yield CapabilityEvent(kind="result", payload={"content": [
    #         TextPart(text="这是刚才那个页面的截图"),
    #         ImagePart(data=b64, media_type="image/png"),   # source_type 默认 base64
    #     ]})
    #
    # 交 **inline base64 是允许**的：字节的校验（media_type 白名单 / 单图 5 MiB）与
    # 外部化（写进宿主的 MemoryBlobStore、换成 ``blob:<sha>`` ref）由 gateway 统一
    # 做（`core.utils.content.legalize_tool_result_parts`），provider 不必认识 blob
    # store。不合格的那张会被换成确定性文本占位、其余 part 照走——**gateway 恒不抛**，
    # 一张图不合格不会掀掉整次工具调用。
    #
    # 文本侧的既有加工（spill 落盘截断、``[Human note: …]`` 前缀、事件 payload 脱敏）
    # 只作用于 content 里的文本 part：gateway 收到就用 `split_for_tool_result` 拆开、
    # 加工完再拼回去。provider 不需要知道这些加工存在。
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityProviderInfo:
    name: str
    capability_count: int = 0
    supports_streaming: bool = True
    supports_cancel: bool = True
    description: str = ""


# ── Provider 基类 + 三个子类 ───────────────────────────────────────────────────

class CapabilityProvider(ABC):
    """所有 provider 的基类：仅声明 list()、retrieve() 和 describe()。"""

    name: str
    # 可选：描述本 provider 的 capability 集合能做什么、应当如何使用。
    # 内置 provider 可在定义时由用户自定义；MCP provider 用 server 的
    # initialize.instructions 填充。装配时按 provider 分块写入 LLM prompt。
    description: str = ""

    @abstractmethod
    async def list(self, ctx: ProviderContext) -> list[Capability]: ...

    async def retrieve(self, ctx: ProviderContext) -> list[Capability]:
        """根据当前上下文返回可能需要的 capability 子集。

        默认回落到 list()；provider 可按需覆盖以实现语义检索或按
        ctx.task_settings / ctx.extra 过滤。
        """
        return await self.list(ctx)

    @abstractmethod
    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo: ...


class ToolCapabilityProvider(CapabilityProvider, ABC):
    """暴露 LLM 可直接调用的工具，经由 CapabilityGateway dispatch。"""

    @abstractmethod
    def invoke(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]: ...
    # Gateway 注入 ``ctx.invocation_id``（本次执行的唯一 id）：需要支持取消的 provider 应据此登记
    # 在途句柄（任务/进程/请求），以便后续 cancel(invocation_id) 对应。``ctx.extra["tool_call_id"]``
    # 是发起本次调用的模型 tool_call id（可重放，用于 §6 配对），与 invocation_id 区别见两者文档。

    @abstractmethod
    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        """取消一次在途执行。``invocation_id`` 即 invoke 时经 ``ctx.invocation_id`` 注入的同一个 id。

        多数本地 provider 无需实现（取消经 ``CancelledError`` 传播到 invoke 协程的 finally，如
        bash 的 terminate_tree 杀进程树）；需要显式取消通知的 provider（如 MCP 远端）按 invocation_id
        查到登记的句柄并取消。best-effort：被 gateway 在取消路径上调用，不应抛出。
        """
        ...


class SessionScopedCapabilityProvider(CapabilityProvider, ABC):
    """持有 per-session 内存状态的 provider：core 在 session 结束时统一清理。

    与具体状态语义无关——control provider 持有 TaskManager、文件系统 provider 持有
    workspace 路径等，各自如何登记是实现细节；协议只约定「session 结束时按 id 释放」
    这一通用清理钩子。core（runtime）遍历所有此类 provider 调 deregister_session()，
    无需知道它们各自持有什么。
    """

    @abstractmethod
    def deregister_session(self, session_id: str) -> None: ...


class SkillCapabilityProvider(CapabilityProvider, ABC):
    """管理技能目录/远端源。list() 返回 SkillCapability（Level 1）。
    Level 2 / Level 3 均由子类实现；SkillExecutorCapabilityProvider 路由到此。
    """

    @abstractmethod
    async def load_definition(
        self, skill_name: str, ctx: ProviderContext,
    ) -> SkillDefinition | None:
        """Level 2：加载 SKILL.md 主体（instructions）。"""
        ...

    @abstractmethod
    async def list_files(
        self, skill_name: str, pattern: str, limit: int, ctx: ProviderContext,
    ) -> str:
        """Level 3：列举技能目录下匹配 pattern 的文件。"""
        ...

    @abstractmethod
    async def load_resource(
        self, skill_name: str, resource_path: str, ctx: ProviderContext,
    ) -> str:
        """Level 3：读取技能目录内的参考文件。"""
        ...

    @abstractmethod
    async def exec_script(
        self, skill_name: str, script_path: str, args: str, ctx: ProviderContext,
    ) -> str:
        """Level 3：执行技能目录内的脚本，返回 stdout。"""
        ...


class AgentCapabilityProvider(CapabilityProvider, ABC):
    """列出可用 sub-agent 模板，并负责加载自己列出的模板。

    发现与加载同源（spec 2026-07-22）：list() 返回的每个 AgentCapability.template_name，
    本 provider 的 get_template() 必须能加载。TemplateLookup 按 cap.id 前缀路由到本
    provider 后，传入的是**局部模板名**（前缀已剥掉）。
    """

    @abstractmethod
    async def get_template(
        self, template_id: str, version: str | None, ctx: ProviderContext,
    ) -> "AgentTemplate | None":
        """加载模板定义。不认识该 id → 返回 None（由 TemplateLookup 转成
        TemplateNotFoundError）；仅真实故障（IO/网络/解析错误）才抛异常。
        version=None 取最新。"""
        ...

    async def retrieve(self, ctx: ProviderContext) -> list[Capability]:
        """默认不自动召回：sub-agent 只经模板声明的 `subagents` required refs 绑定
        （allowlist），永不把整个模板目录泄漏给 agent。确需语义召回的 provider 可覆盖。"""
        return []


# ── 授权契约 ───────────────────────────────────────────────────────────────────


@dataclass
class AuthorizationDecision:
    """一次授权的结构化结果。"""

    allowed: bool
    # 反馈 / 拒绝指导，回灌给 LLM（allow / deny 都可带）。
    # **可以是 `list[ContentPart]`**：人类经 HITL 递进来的备注可能带图
    # （`HumanConfirmationAuthorizer` 直接透传 `HitlDecision.message`）。gateway 的两处
    # 拼接走 `content_with_prefix` / `content_with_suffix`，对 str 逐字节原样。
    message: "str | list[ContentPart]" = ""
    modified_arguments: dict[str, Any] | None = None  # allow 时的有效参数（None = 用原参）
    #: 「挂起并问这个问题」。取代只能说「挂起」的 `defer`（已删除）——后者说不出问什么，
    #: 所以旧实现必须让 authorizer 自己先去登记请求（那正是耦合的源头）。
    #: 非 None 时 `allowed` 必须为 False；gateway 先判 allowed，安全不变式不依赖本字段。
    needs_human: "HitlAsk | None" = None


class Authorizer(ABC):
    """对一次 capability 调用作授权决定。

    核心方法 ``authorize`` 对**一次工具调用**作放行/拦截决定，并可携带回灌给 LLM 的
    ``message``（反馈/拒绝指导）与 allow 时的 ``modified_arguments``（改写参数）。
    曾有一个基于 ``authorize`` 的批量 ``filter`` 默认实现，**已删除**：它零调用点，且对
    ``HumanConfirmationAuthorizer`` 会**真的发出一个 HITL 请求并等人**——把「列一下有哪些
    工具可见」变成「向人类逐个求批」。真需要装配期可见性过滤时应另行设计，届时必须显式
    排除会挂起的 authorizer。

    只收 ``ProviderContext``（session/task/agent/模板 标识齐备），不收 core 的 Agent/Task
    状态对象——契约层不依赖 core 状态，host 自实现时也只需面对 protocols。
    内置实现见 ``ctx_weft.providers.authorizer``。
    """

    @abstractmethod
    async def authorize(
        self,
        capability: Capability,
        ctx: ProviderContext,
        arguments: dict[str, Any] | None = None,
        *,
        tool_call_id: str = "",
    ) -> AuthorizationDecision: ...


@runtime_checkable
class SkillIndexConsumer(Protocol):
    """持有 skill 索引、需要在 skill 目录变动时被通知重建的 provider。

    存在的理由是**打断一条 import 环**：`ProviderRegistry` 在
    `register_capability` / `deregister_capability` 里要通知
    `SkillExecutorCapabilityProvider` 重建索引，但后者住在 `core.orchestrator`，
    而 orchestrator 反过来要 `ProviderRegistry` 的类型——registry 靠函数内惰性
    import 绕了这个环。改成对本 Protocol 做 `isinstance`，registry 就完全不必知道
    那个具体类，环真正消失而不是被挪个地方继续绕。

    `runtime_checkable` 的 isinstance 只查方法名存在性，对这里够用：能被通知的
    条件就是「有 mark_dirty」。
    """

    def mark_dirty(self) -> None:
        """下次 list() 前重建索引。"""
        ...


@runtime_checkable
class HumanGatedAuthorizer(Protocol):
    """**可选**能力接口：只有会问人的 authorizer 实现它。

    加法式而非分叉式（spec §2.1）：基础 `Authorizer` 的签名一个字不变，不问人的实现
    看不到任何 HITL 概念。分叉基类会连带要求分叉返回类型联合，否则「不需要 HITL 的
    基类」在类型上仍允许返回 NeedsHuman——非法组合只是换了个地方藏。
    """

    async def on_decision(
        self,
        capability: Capability,
        ctx: ProviderContext,
        arguments: dict[str, Any] | None,
        tool_call_id: str,
        decision: "HitlDecision",
    ) -> AuthorizationDecision: ...


@runtime_checkable
class HumanResumable(Protocol):
    """**可选**能力接口：只有会问人、且答复不直接作结果的工具 provider 实现它。

    同样是流式（spec §2）。重入是**重新调用**而非恢复挂起的生成器，故 `resume_state`
    承载让出前的全部状态。`reply_as_result=True` 的 ask（如 `ask_user`）不需要它。

    **`resume_state` 只省掉热重入**（spec §2.2 / §9.3）。热路径上进程还活着，gateway
    拿到人的决定直接调 `resume()`，`invoke` 不跑第二遍。**冷路径不然**：热窗口被驱逐
    或进程崩了之后，续跑由 `ReconcileStep` 重新 `invoke`，
    **`invoke` 从头再跑一遍**——gateway 在 provider yield `needs_human` 之前无从知道
    这次调用要问人，只能等它跑到那一刻才查到缓存的决定。而 `resume()` 收到的
    `resume_state` 是**这一次刚 yield 的那一份**，不是崩溃前落盘的那一份。

    因此契约是：**`needs_human` 之前做的工作必须幂等，或者便宜到重做无所谓。**
    不可重复的副作用只能放进 `resume()`。
    """

    def resume(
        self,
        ask_id: str,
        decision: "HitlDecision",
        resume_state: dict[str, Any] | None,
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]: ...
