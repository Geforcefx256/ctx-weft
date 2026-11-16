"""Assemble event-ready prompt inputs, then compile them through purpose templates."""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable
from copy import copy
from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

from ctx_weft.core.utils.estimate import effective_limit, estimate_tokens
from ctx_weft.protocols import (
    AgentTemplate,
    LLMMessage,
    LLMTool,
    MemoryAddress,
    MemoryProvider,
    ProviderContext,
    Purpose,
)
from ctx_weft.protocols.knowledge import KnowledgeProvider

if TYPE_CHECKING:
    from ctx_weft.core.assembler.context import PromptContext
    from ctx_weft.core.capabilities.cache import CapabilityCache
    from ctx_weft.core.orchestrator.model import ResolvedModel
    from ctx_weft.core.models.agent import Agent
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import Task
    from ctx_weft.core.orchestrator.task.manager import TaskManager


# ── Purpose parameters ───────────────────────────────────────────────────────

CompactScope: TypeAlias = Literal["task", "agent"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActIntent:
    """Execute the current task."""


@dataclass(frozen=True)
class ReviewIntent:
    """Observe the latest segment; judgement is optional."""

    judge_task: bool = True
    terminal_only: bool = False

    def __post_init__(self) -> None:
        if self.terminal_only and not self.judge_task:
            raise ValueError("terminal_only requires judge_task")


@dataclass(frozen=True)
class CompactIntent:
    """Compact task or agent memory."""

    scope: CompactScope = "task"

    def __post_init__(self) -> None:
        if self.scope not in ("task", "agent"):
            raise ValueError(f"unsupported compact scope: {self.scope!r}")


@dataclass(frozen=True)
class MetadataIntent:
    """Recognize and fill task metadata."""


PromptIntent: TypeAlias = ActIntent | ReviewIntent | CompactIntent | MetadataIntent


# ── ContextRequest ────────────────────────────────────────────────────────────


@dataclass
class ContextRequest:
    """装配请求。

    purpose selects both the prompt and the capability surface. ReviewIntent
    controls optional task judgement; CompactIntent controls compression scope.
    Foreground/background execution does not change the purpose.
    """

    purpose: Purpose
    scope: MemoryAddress
    task: Task
    agent: Agent
    session: Session
    template: AgentTemplate | None
    bound_capabilities: list[Any]
    resolved_model: ResolvedModel
    token_counter: Callable[[str], int] = estimate_tokens
    extra: dict[str, Any] = field(default_factory=dict)
    # Optional compilation options; their purpose must match the request.
    intent: PromptIntent | None = None

    def __post_init__(self) -> None:
        expected = {
            "act": ActIntent,
            "observe": ReviewIntent,
            "compact": CompactIntent,
            "recognize_intent": MetadataIntent,
        }
        intent_type = expected.get(self.purpose)
        if intent_type is None:
            raise ValueError(f"unsupported prompt purpose: {self.purpose!r}")
        if self.intent is not None and not isinstance(self.intent, intent_type):
            raise ValueError(
                f"prompt intent {type(self.intent).__name__} does not match purpose "
                f"{self.purpose!r}"
            )


# ── ContextBlock ──────────────────────────────────────────────────────────────


BlockKind = Literal[
    "identity",
    "background",
    "capabilities",
    "directive",
    "guidance",
    "history",
    "blackboard",
    "summary",
    "reference",
    "task_spec",
]


@dataclass
class ContextBlock:
    """Source 的统一输出信封。

    Source 只声明来源、种类和原始值。落位、文字装饰、required 与裁剪 priority
    全部属于 purpose Jinja；Source 不生成 prompt 投影。
    """

    id: str
    source: str  # 来源标识（"capability:identity" / "memory:blackboard:bg" 等）
    kind: BlockKind
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_id: str | None = None


# ── AssembledPrompt ───────────────────────────────────────────────────────────


class AssembledPrompt:
    """Compiled text/tools plus the same inputs used to produce them.

    Runtime pins are resolved by the request builder. tools_fn remains available
    to custom callers that explicitly supply a resolver.
    """

    def __init__(
        self,
        system: str,
        messages: list[LLMMessage],
        tools: list[LLMTool],
        token_count: int,
        metadata: dict[str, Any] | None = None,
        tools_fn: Callable[[], list[LLMTool]] | None = None,
        token_budget: int = 0,
        capability_errors: list[dict] | None = None,
    ) -> None:
        self.system = system
        self.messages = messages
        self._tools = tools
        self.token_count = token_count
        # Input budget including tools; 0 means compilation was unbounded.
        self.token_budget = token_budget
        self.metadata = metadata if metadata is not None else {}
        self.tools_fn = tools_fn
        self.inputs: PromptContext | None = None
        self.tool_refs: list[dict] = []
        self.capability_errors = capability_errors if capability_errors is not None else []

    @property
    def tools(self) -> list[LLMTool]:
        if self.tools_fn is None:
            return self._tools
        try:
            return self.tools_fn()
        except Exception:
            logger.warning(
                "live tool resolution failed; using the assembled snapshot",
                exc_info=True,
            )
            return self._tools

    @tools.setter
    def tools(self, value: list[LLMTool]) -> None:
        self._tools = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AssembledPrompt):
            return NotImplemented
        return (
            self.system == other.system
            and self.messages == other.messages
            and self.tools == other.tools
            and self.token_count == other.token_count
            and self.metadata == other.metadata
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"AssembledPrompt(system={self.system!r}, messages={self.messages!r}, "
            f"tools={self.tools!r}, token_count={self.token_count!r}, "
            f"metadata={self.metadata!r})"
        )


# ── Source protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class ContextSource(Protocol):
    """Fetch provider data and wrap each raw object in a ContextBlock."""

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
    """Dependencies available while Sources collect raw context."""

    memory: MemoryProvider
    knowledge_providers: list[KnowledgeProvider]
    provider_ctx: ProviderContext
    capability_cache: "CapabilityCache | None" = None
    task_manager: "TaskManager | None" = None
    # provider_name → CapabilityProvider. Compilation resolves saved IDs through
    # list() and reads the current provider description/skill definition.
    capability_provider_index: dict[str, Any] = field(default_factory=dict)
    template_lookup: Any = None


# ── ContextAssembler ───────────────────────────────────────────────────────────


@dataclass
class ContextAssembler:
    """装配流水线主控。"""

    sources: list[ContextSource]
    deps: AssemblerDeps

    async def assemble(self, request: ContextRequest) -> PromptContext:
        """Select sources and freeze their inputs directly in event form."""
        from ctx_weft.core.assembler.context import PromptContext

        context = PromptContext.from_request(request)
        context._record_context = (getattr(self.deps.provider_ctx, "tenant_id", "default"),
                                   self.deps.provider_ctx.session_id)
        staged_ids = {event.id for event in request.extra.get("staged_memory", [])}
        async with asyncio.TaskGroup() as tg:
            futs = [tg.create_task(self._collect(s, request)) for s in self.sources]
        for f in futs:
            for block in f.result():
                context.add(block, staged_ids=staged_ids)
        return context

    async def compile(self, inputs: PromptContext | dict, *, template=None,
                      token_counter=estimate_tokens, reuse: bool = False,
                      trim: bool = True, tool_refs: list[dict] | None = None) -> AssembledPrompt:
        """Compile inputs; opt into reuse only for an unchanged runtime base.

        Use trim=False to estimate complete inputs before compaction. Otherwise
        enforce the context window minus reserved output, including tools.
        A cached complete result can also serve a bounded request if it fits.
        Reassemble after changing inputs. Historical/request inputs containing
        continuations or a frozen tool override are always compiled independently.
        """
        from ctx_weft.core.assembler.builder import PromptBuilder
        from ctx_weft.core.assembler.context import PromptContext, record_from_json
        from ctx_weft.core.assembler.sources.capability import (
            capability_blocks,
            resolve_capabilities,
        )
        from ctx_weft.protocols.content import normalize_content_parts

        if isinstance(inputs, dict):
            inputs = PromptContext.from_payload(inputs)
        selected_tools = inputs.tools if tool_refs is None else tool_refs
        budget = effective_limit(inputs.context_limit, inputs.reserved_output_tokens) if trim else 0
        if inputs.scope.get("session_id") not in (None, self.deps.provider_ctx.session_id):
            raise ValueError("prompt inputs belong to a different session")
        record_context = (getattr(self.deps.provider_ctx, "tenant_id", "default"),
                          self.deps.provider_ctx.session_id)
        if reuse and (inputs.appended_messages or inputs.tools is not None):
            raise ValueError("only base prompt inputs can reuse compilation")
        resolved, errors = await resolve_capabilities(inputs.capabilities + (selected_tools or []), self.deps)
        capabilities = capability_blocks(inputs.capabilities, resolved, self.deps)
        cached = inputs._compiled_base
        if reuse and cached is not None:
            owner, memory, cached_template, counter, prompt = cached
            # Budget 0 denotes an unbounded compilation. A bounded request
            # must have positive room; relaxing a previous bound recompiles
            # to restore any content it may have discarded.
            same_budget = prompt.token_budget == budget and (not trim or budget > 0)
            fits_budget = (
                trim and budget > 0 and prompt.token_count <= budget
                and (prompt.token_budget == 0 or budget < prompt.token_budget)
            )
            if (owner == record_context and memory is self.deps.memory
                    and cached_template is template and counter == token_counter
                    and prompt._capability_facts == capabilities
                    and (same_budget or fits_budget)):
                return await self._complete_request(prompt, inputs, selected_tools,
                                                    resolved=resolved, errors=errors)
        requested_template = template
        if template is None and inputs.template_id:
            if self.deps.template_lookup is None:
                raise ValueError(f"template lookup is required for {inputs.template_id!r}")
            template = await self.deps.template_lookup.get_template(
                inputs.template_id, None, self.deps.provider_ctx)
        # Cached reads retain the provider's tenant/session boundary.
        records = dict(inputs._records) if inputs._record_context == record_context else {}
        for item in inputs.memory_blocks:
            if "record" in item:
                records[item["memory_id"]] = record_from_json(item["record"])
        missing = [record_id for record_id in inputs.memory_ids() if record_id not in records]
        if missing:
            records.update((record.id, record) for record in await self.deps.memory.read_records(
                missing, self.deps.provider_ctx))
        missing = set(inputs.memory_ids()) - records.keys()
        if missing:
            raise ValueError(f"prompt memory records missing: {sorted(missing)}")
        inputs._records = records
        inputs._record_context = record_context
        # Avoid repeating task input/output content already available by ID.
        for name in ("user_prompt", "outputs"):
            value = normalize_content_parts(inputs.task.get(name))
            if value:
                match = next((r for r in records.values() if r.id and r.content == value), None)
                if match is not None:
                    inputs.task.pop(name)
                    inputs.task[f"{name}_memory_id"] = match.id

        def count(text):
            digest = sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
            if digest not in inputs.token_counts:
                inputs.token_counts[digest] = token_counter(text)
            return inputs.token_counts[digest]

        base_keys = {(ref["provider"], ref["id"]) for ref in inputs.capabilities}
        base_errors = [error for error in errors if (error["provider"], error["id"]) in base_keys]
        skill_definition = await self._load_skill_definition(inputs, resolved, base_errors)
        prompt = PromptBuilder(count).compile(
            inputs, tools=self._compile_tools(inputs, inputs.capabilities, resolved),
            records=records, template=template, trim=trim,
            capabilities=capabilities,
            skill_definition=skill_definition)
        prompt._capability_facts = capabilities
        prompt.tool_refs = self._tool_refs(inputs, inputs.capabilities, resolved)
        prompt.capability_errors = base_errors
        prompt.inputs = inputs
        if reuse:
            inputs._compiled_base = (record_context, self.deps.memory, requested_template,
                                     token_counter, prompt)
        return await self._complete_request(prompt, inputs, selected_tools, resolved=resolved, errors=errors)

    @staticmethod
    def _tool_refs(context, refs, resolved):
        if context.purpose == "compact":
            return []
        return [dict(ref) for ref in refs
                if (cap := resolved.get((ref["provider"], ref["id"]))) is not None
                and cap.kind == "tool" and context.purpose in cap.purposes]

    @classmethod
    def _compile_tools(cls, context: PromptContext, refs, resolved) -> list[LLMTool]:
        from ctx_weft.core.assembler.sources.capability import build_llm_tools

        return build_llm_tools([resolved[(ref["provider"], ref["id"])]
                                for ref in cls._tool_refs(context, refs, resolved)], context.purpose,
                               terminal_only=context.terminal_only)

    async def _load_skill_definition(self, inputs, resolved, errors):
        from ctx_weft.core.assembler.context import json_value
        from ctx_weft.protocols.capability import qualify

        if inputs.purpose != "act" or not inputs.skill_name:
            return {}
        for ref in inputs.capabilities:
            cap = resolved.get((ref["provider"], ref["id"]))
            if cap is None or cap.kind != "skill" or qualify(cap.id) != inputs.skill_name:
                continue
            provider = self.deps.capability_provider_index[ref["provider"]]
            try:
                definition = await provider.load_definition(cap.name, self.deps.provider_ctx)
                if definition is not None:
                    value = json_value(definition)
                    execution_instructions = getattr(provider, "execution_instructions", "").strip()
                    if execution_instructions:
                        value["instructions"] = (
                            f"{definition.instructions.rstrip()}\n\n{execution_instructions}"
                        ).lstrip("\n")
                    return value
                error = "Skill definition was not returned by provider.load_definition()"
            except Exception as exc:
                error = f"provider.load_definition() failed: {exc}"
            logger.warning("capability unavailable: provider=%s capability_id=%s: %s",
                           ref["provider"], ref["id"], error)
            errors.append({**ref, "error": error})
            break
        return {}

    async def _complete_request(self, base, inputs, tool_refs, *, resolved=None, errors=None):
        from ctx_weft.core.assembler.sources.capability import resolve_capabilities

        if tool_refs is None and not inputs.appended_messages:
            return base
        prompt = copy(base)
        prompt.capability_errors = list(base.capability_errors)
        if tool_refs is not None:
            if resolved is None:
                resolved, errors = await resolve_capabilities(tool_refs, self.deps)
            prompt.tools = self._compile_tools(inputs, tool_refs, resolved)
            # Keep the attempted selection, including unavailable references.
            prompt.tool_refs = [dict(ref) for ref in tool_refs]
            for error in errors or []:
                if error not in prompt.capability_errors:
                    prompt.capability_errors.append(error)
        prompt.messages = list(base.messages)
        for item in inputs.appended_messages:
            content = inputs._records[item["memory_id"]].content if item.get("memory_id") else item["content"]
            prompt.messages.append(LLMMessage(
                role=item["role"], content=content, tool_calls=item["tool_calls"],
                tool_call_id=item["tool_call_id"]))
        return prompt

    async def _collect(self, source: ContextSource, request: ContextRequest) -> list[ContextBlock]:
        blocks: list[ContextBlock] = []
        async for blk in source.fetch(request, self.deps):
            blocks.append(blk)
        return blocks
