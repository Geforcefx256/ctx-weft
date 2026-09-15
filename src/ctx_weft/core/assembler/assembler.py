"""Context Assembler：装配流水线核心。

ContextRequest 是装配入口；ContextBlock 是中间表示；AssembledPrompt 是最终输出。
ContextSource 是数据获取接口。

═══ 流水线 ═══

  sources（并发 fetch，每个产 0..N 个 block）
    IdentitySource        identity + directive
    CapabilitySource      capabilities（tools/skills/agents 三路）
    TaskSpecSource        task_spec（metadata 载体）
    AgentRecallSource     history（task body + agent 层回合）
    BlackboardSource      blackboard + background
    SemanticRecallSource  summary
    KnowledgeRetrieval…   reference
    GuidanceSource        guidance（act 态势，extra["act_guidance"]）
          │
          ▼  ContextBlock[]（kind / target / priority / token_estimate）
    BudgetStrategy.apply  超限时按保护阶梯裁剪（阶梯见 priority.py，
          │               动态覆盖与配对丢弃见 budget.py）
          ▼  kept blocks
    Composer.compose      按 request.purpose 渲染（槽位布局见 composer.py）
          │
          ▼
    AssembledPrompt（system / messages / tools / token_count）
                    tools 不是这条流水线的产物：它由 assemble() 装上的闭包现读
                    CapabilityCache（唯一真相源），故运行期 pin 进来的能力当轮可见。

详见设计文档 §5.2。
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ctx_weft.protocols import (
    AgentTemplate,
    LLMMessage,
    LLMTool,
    LoopConfig,
    MemoryProvider,
    MemoryAddress,
    ProviderContext,
    Purpose,
)
from ctx_weft.core.utils.estimate import effective_limit, estimate_tokens
from ctx_weft.protocols.knowledge import KnowledgeProvider

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ctx_weft.core.models.agent import Agent
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task


# ── ContextRequest ────────────────────────────────────────────────────────────


@dataclass
class ContextRequest:
    """装配请求。

    purpose 标识装配的 prompt 给哪个 LLM 调用阶段消费：
      - "act"     → PrepareStep 装配给下一个 ActStep 的 LLM 用
      - "observe" → ObserveStep 装配给自己的 LLM 用
      - "compact" → CompactStep 装配给摘要 LLM 用
    """

    purpose: Purpose
    scope: MemoryAddress
    task: "Task"
    agent: "Agent"
    session: "Session"
    template: AgentTemplate | None  # 实例化时绑定的 template（pin 了 version）；None 时 IdentitySource 跳过
    bound_capabilities: list[Any]  # list[Capability]，但避免循环导入
    # token 计数回调：生产由 step 构造时传 ctx.llm.tokenizer.count（已校准、随当次模型）；
    # 默认回退未校准启发式（测试/无 llm 场景）。sources/composer 统一经它计数。
    token_counter: Callable[[str], int] = estimate_tokens
    extra: dict[str, Any] = field(default_factory=dict)


# ── ContextBlock ──────────────────────────────────────────────────────────────


BlockKind = Literal[
    # ── 渲染进 system ──
    "identity",  # SOUL / facet 正文（IdentitySource）
    "background",  # ## Project Background（BlackboardSource long_term_background topic）
    # ── 注入段：composer 拼进 user 回合（不进 system，也不是独立消息）──
    "capabilities",  # ## Capabilities → 当前 task 的 user 回合尾部，末条留一行指针（CapabilitySource）
    "directive",  # ## Instructions for the current task → 当前 task 的 user 回合（skill_instructions）
    "guidance",  # act 运行时态势 guidance → 末条 user 最尾部（GuidanceSource）
    # ── 渲染进 messages ──
    "history",  # 多轮对话无损重建（AgentRecallSource）
    "blackboard",  # 相关任务 topic 通信 / 长期 project_log（BlackboardSource）
    "summary",  # 语义召回结果（SemanticRecallSource）
    "reference",  # 知识检索结果（KnowledgeRetrievalSource）
    "task_spec",  # 当前 task spec 的 metadata 载体（TaskSpecSource）——composer 读其
    #              metadata 去装饰当前 user 回合（## Current Task/Message 框），不独立渲染
]

BlockTarget = Literal["system", "messages"]


@dataclass
class ContextBlock:
    """装配阶段的中间块。每个 Source 产出 0..N 个 block。"""

    id: str
    source: str  # 来源标识（"capability:identity" / "memory:blackboard:bg" 等）
    kind: BlockKind
    target: BlockTarget
    content: str | list[Any]  # 文本或 ContentPart 列表
    priority: int  # 0 最高，budget 不够时高 priority 优先保留
    token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ── AssembledPrompt ───────────────────────────────────────────────────────────


class AssembledPrompt:
    """Composer 输出：装配好的最终 prompt。

    **`tools` 是 property，不是字段**——这是本类不再是 dataclass 的唯一理由。ActStep 的
    每一轮读的都是**同一个** prompt 对象（PrepareStep 一次装配、act 循环内不重装配），
    若 tools 是装配期的一份拷贝，运行期新增的能力就永远进不了工具面，且它与
    CapabilityCache 两份之间没有任何同步机制。装上 `tools_fn` 后每次读都问 cache 现算
    （闭包见 ContextAssembler.assemble），cache 成为唯一真相源。

    构造签名与参数名保持与旧 dataclass 逐字相同（含按位置传），既有构造点零改动；
    不装 tools_fn 时 `tools` 就是构造时传进来的那份，行为逐字节不变。
    """

    def __init__(
        self,
        system: str,
        messages: list[LLMMessage],
        tools: list[LLMTool],
        token_count: int,
        metadata: dict[str, Any] | None = None,
        tools_fn: Callable[[], list[LLMTool]] | None = None,
    ) -> None:
        self.system = system
        self.messages = messages
        self._tools = tools
        self.token_count = token_count
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        self.tools_fn = tools_fn

    @property
    def tools(self) -> list[LLMTool]:
        if self.tools_fn is None:
            return self._tools
        try:
            return self.tools_fn()
        except Exception:
            # 活来源出问题不该让整轮 LLM 调用崩：回落装配期那份快照（可能偏旧，但可用）。
            logger.warning("AssembledPrompt: tools_fn failed, falling back to assembled snapshot",
                           exc_info=True)
            return self._tools

    @tools.setter
    def tools(self, value: list[LLMTool]) -> None:
        # 少数调用方（测试替身、临时裁剪）直接赋值：写进快照槽，语义与旧 dataclass 一致。
        self._tools = value

    def __eq__(self, other: object) -> bool:
        # 逐字段相等，与改造前的 dataclass 语义一致（比较的是**当前** tools 视图）。
        # 有测试拿它作「两次装配逐字节相同」的判据，不能退化成身份比较。
        if not isinstance(other, AssembledPrompt):
            return NotImplemented
        return (
            self.system == other.system
            and self.messages == other.messages
            and self.tools == other.tools
            and self.token_count == other.token_count
            and self.metadata == other.metadata
        )

    __hash__ = None  # type: ignore[assignment]  # 同 dataclass(eq=True)：可变、不可哈希

    def __repr__(self) -> str:  # pragma: no cover — 调试用
        return (f"AssembledPrompt(system={self.system!r}, messages={self.messages!r}, "
                f"tools={self.tools!r}, token_count={self.token_count!r}, "
                f"metadata={self.metadata!r})")


# ── Source protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class ContextSource(Protocol):
    """装配源——把 provider 数据转换为 ContextBlock。"""

    name: str

    @abstractmethod
    def fetch(
        self,
        request: ContextRequest,
        deps: "AssemblerDeps",
    ) -> AsyncIterator[ContextBlock]: ...


# ── AssemblerDeps ─────────────────────────────────────────────────────────────


@dataclass
class AssemblerDeps:
    """注入给 Source / Composer / Budget 的依赖集合。"""

    memory: MemoryProvider
    knowledge_providers: list[KnowledgeProvider]
    provider_ctx: ProviderContext
    skill_provider_index: dict[str, Any] = field(default_factory=dict)  # provider_name → SkillCapabilityProvider
    # provider_name → CapabilityProvider（持有 live 对象，fetch 时读取 .description；
    # MCP 在 connect 后才有 description，故须 live 读取而非装配时快照）
    capability_provider_index: dict[str, Any] = field(default_factory=dict)
    # CapabilityCache（工具面唯一真相源）+ 本次装配的 agent。assemble() 据此给
    # AssembledPrompt 装上活工具面闭包；为 None 时退回「composer 产什么就是什么」的旧行为，
    # 故直接构造 ContextAssembler 的测试无需改动。类型写 Any 避免 assembler → core.capabilities
    # 的硬依赖（assembler 层不认识 cache 的具体类型）。
    capability_cache: Any = None
    agent_id: str = ""


# ── ContextAssembler ───────────────────────────────────────────────────────────


@dataclass
class ContextAssembler:
    """装配流水线主控。"""

    sources: list[ContextSource]
    budget: "BudgetStrategy"
    composer: "Composer"
    deps: AssemblerDeps

    async def assemble(self, request: ContextRequest) -> AssembledPrompt:
        """1) 并发触发所有 sources；2) budget（先扣工具面预留）；3) compose。"""
        all_blocks: list[ContextBlock] = []
        async with asyncio.TaskGroup() as tg:
            futs = [tg.create_task(self._collect(s, request)) for s in self.sources]
        for f in futs:
            all_blocks.extend(f.result())

        # budget：内容预算 = 有效窗口 − 工具面预留（spec: tool-schema-budget）。工具
        # schema 是每个请求的固定占用（API tools 参数随请求下发且不受裁剪）——不预留
        # 则窗口实际超限。预留按当次装配的能力快照现算；compact 工具面恒空 → 零预留；
        # cache 缺失（测试直构 assembler）→ 零预留、行为同旧。指纹供 act 循环增量估算
        # 感知工具面变化。
        token_limit = effective_limit(
            request.session.context_limit, request.session.reserved_output_tokens
        )
        tools_reserved, tools_sig = self._tools_reservation(request)
        kept = await self.budget.apply(
            all_blocks, max(0, token_limit - tools_reserved), request,
            overflow_limit=token_limit)

        # compose
        prompt = await self.composer.compose(kept, request)
        self._install_live_tools(prompt, request)
        prompt.metadata["tools_reserved_tokens"] = tools_reserved
        prompt.metadata["tools_signature"] = tools_sig
        return prompt

    def _tools_reservation(self, request: ContextRequest) -> tuple[int, str]:
        """装配期工具面预留与指纹（紧凑口径：compact / 无 cache → (0, "")）。"""
        cache = self.deps.capability_cache
        if cache is None or request.purpose == "compact":
            return 0, ""
        from ctx_weft.core.assembler.sources.capability import build_llm_tools
        from ctx_weft.core.utils.estimate import estimate_tools_tokens, tools_signature

        agent_id = getattr(request.agent, "id", "") or self.deps.agent_id
        task_id = getattr(request.task, "id", "") or getattr(
            self.deps.provider_ctx, "task_id", "")
        snapshot = build_llm_tools(cache.available(agent_id, task_id), request.purpose)
        return estimate_tools_tokens(snapshot, request.token_counter), tools_signature(snapshot)

    def _install_live_tools(self, prompt: AssembledPrompt, request: ContextRequest) -> None:
        """给 prompt 装上活工具面：每次读 `.tools` 都问 CapabilityCache 现算。

        装配期的一份拷贝与 cache 之间没有同步机制，而 ActStep 整轮循环读的是同一个 prompt
        对象——运行期 pin 进来的能力若不走这条闭包就永远不可见。渲染仍复用
        `build_llm_tools`（同一个 purpose 过滤），故活工具面与散文段清单不可能算出两套。

        compact 不装：它的工具面按契约恒为空（composer 的 else 分支），装了会凭空长出工具。
        cache 缺失（直接构造 assembler 的测试路径）也不装，保持旧行为。
        """
        cache = self.deps.capability_cache
        if cache is None or request.purpose == "compact":
            return
        from ctx_weft.core.assembler.sources.capability import build_llm_tools

        # agent/task 优先取 request 自己的（它就是这次装配的对象）；deps 侧作兜底——
        # provider_ctx 与 request 恒同源，但 deps 是按 run 构造的、request 是按次装配的。
        agent_id = getattr(request.agent, "id", "") or self.deps.agent_id
        task_id = getattr(request.task, "id", "") or getattr(
            self.deps.provider_ctx, "task_id", "")
        purpose = request.purpose
        prompt.tools_fn = lambda: build_llm_tools(cache.available(agent_id, task_id), purpose)

    async def _collect(self, source: ContextSource, request: ContextRequest) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        async for blk in source.fetch(request, self.deps):
            blocks.append(blk)
        return blocks




# Forward references to avoid circular imports at module load
from ctx_weft.core.assembler.budget import BudgetStrategy  # noqa: E402
from ctx_weft.core.assembler.composer import Composer  # noqa: E402
