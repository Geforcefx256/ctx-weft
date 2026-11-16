"""SkillExecutorCapabilityProvider：技能 Level3 执行工具。

与 ControlCapabilityProvider 同级，runtime 初始化时自动注册。
@skill_executor_tool 装饰器参照 @control_tool，从 Annotated 注解提取 schema；
runtime-injected 参数（ctx / skill_provider / skill_name）通过 _SKIP 排除在外。

每次操作使用本次 ProviderContext 查找已分配的 skill。
上下文相关的列表不在执行器中缓存，资源索引和可见性由 Skill Provider 管理。
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any

from ctx_weft.core.capabilities._text_window import window_text
from ctx_weft.core.capabilities.schema import extract_schema
from ctx_weft.protocols.capability import (
    Capability,
    CapabilityEvent,
    CapabilityProviderInfo,
    SkillCapability,
    SkillCapabilityProvider,
    ToolCapability,
    ToolCapabilityProvider,
    qualify,
)
from ctx_weft.protocols.context import ProviderContext

if TYPE_CHECKING:
    from ctx_weft.core.registry import ProviderRegistry

logger = logging.getLogger(__name__)

PROVIDER_NAME = "skill_executor"
_SKIP: frozenset[str] = frozenset({"ctx", "skill_provider", "skill_name"})

# Qualified (LLM-facing) names for the skill_executor tools — use anywhere these tools
# are named to the LLM in prose (e.g. the PrepareStep runtime note).
# The `skill_executor` prefix already conveys "skill", so the bare tool names omit it.
LIST_FILES_NAME = qualify(f"{PROVIDER_NAME}:list_files")
READ_FILE_NAME = qualify(f"{PROVIDER_NAME}:read_file")
EXEC_SCRIPT_NAME = qualify(f"{PROVIDER_NAME}:exec_script")

# ── @skill_executor_tool 装饰器 ───────────────────────────────────────────────

# name → (ToolCapability, async handler fn, valid_keys)
_SKILL_EXECUTOR_TOOLS: dict[str, tuple[ToolCapability, Callable, frozenset[str]]] = {}


def skill_executor_tool(fn: Callable | None = None, *, spillable: bool = True) -> Callable:
    """装饰器：从 Annotated 注解提取 schema，注册到 _SKILL_EXECUTOR_TOOLS。
    runtime-injected 参数（ctx / skill_provider / skill_name）从 schema 排除。
    spillable=False 的只读工具（read_file / list_files）输出原样返回，不截断/落盘。
    支持裸用 @skill_executor_tool 与带参 @skill_executor_tool(spillable=False)。
    """
    if fn is None:
        return lambda f: skill_executor_tool(f, spillable=spillable)
    first_line = (fn.__doc__ or "").strip().split("\n")[0].strip()
    cap = ToolCapability(
        id=f"{PROVIDER_NAME}:{fn.__name__}",
        name=fn.__name__,
        kind="tool",
        purposes=["act"],
        description=first_line,
        input_schema=extract_schema(fn, exclude=_SKIP),
        side_effects=False,
        spillable=spillable,
    )
    valid_keys = frozenset(inspect.signature(fn).parameters.keys()) - _SKIP
    _SKILL_EXECUTOR_TOOLS[fn.__name__] = (cap, fn, valid_keys)
    return fn


# ── SkillResult ───────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    content: str
    is_error: bool = False


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@skill_executor_tool(spillable=False)
async def list_files(
    pattern: Annotated[str, "Skill resource glob pattern"] = "**/*",
    limit: Annotated[int, "Maximum result count"] = 200,
    *,
    skill_provider: SkillCapabilityProvider,
    skill_name: str,
    ctx: ProviderContext,
) -> SkillResult:
    """List resources bundled with the skill assigned to this task."""
    content = await skill_provider.list_files(skill_name, pattern, limit, ctx)
    return SkillResult(content=content)


@skill_executor_tool(spillable=False)
async def read_file(
    path: Annotated[str, "Relative path to a file within the skill directory"],
    offset: Annotated[int | None, "Starting line number"] = None,
    limit: Annotated[int | None, "Maximum line count"] = None,
    char_offset: Annotated[int | None, "Starting character offset"] = None,
    char_limit: Annotated[int | None, "Maximum character count"] = None,
    *,
    skill_provider: SkillCapabilityProvider,
    skill_name: str,
    ctx: ProviderContext,
) -> SkillResult:
    """Read a text resource bundled with the assigned skill, one page per call.

    Returns a numbered line window, not the whole file — a long reference doc would
    otherwise blow up the context. When more remains, the tail of the content carries the
    continuation hint (the next offset / char_offset); call again with it to page on.
    """
    try:
        # 协议上 load_resource 整份返回；开窗在这里做，模型只看到一页。
        content = await skill_provider.load_resource(skill_name, path, ctx)
        return SkillResult(content=window_text(
            content, offset=offset, limit=limit,
            char_offset=char_offset, char_limit=char_limit,
        ))
    except (ValueError, FileNotFoundError) as exc:
        return SkillResult(content=str(exc), is_error=True)


@skill_executor_tool
async def exec_script(
    command: Annotated[str, "Command text"],
    *,
    skill_provider: SkillCapabilityProvider,
    skill_name: str,
    ctx: ProviderContext,
) -> SkillResult:
    """Run a command for the skill assigned to this task."""
    try:
        content = await skill_provider.exec_script(skill_name, command, ctx)
        return SkillResult(content=content)
    except Exception as exc:
        return SkillResult(content=str(exc), is_error=True)


# ── SkillExecutorCapabilityProvider ──────────────────────────────────────────

class AmbiguousSkillError(ValueError):
    """A qualified skill reference must identify exactly one capability."""


class SkillExecutorCapabilityProvider(ToolCapabilityProvider):
    """默认内置，runtime 初始化时自动注册，与 ControlCapabilityProvider 同级。

    只解析路由和转交参数，不解释 ProviderContext 中的业务身份。
    """

    name = PROVIDER_NAME

    def __init__(self, provider_registry: "ProviderRegistry") -> None:
        self._registry = provider_registry

    async def _resolve_skill(
        self, skill_ref: str, ctx: ProviderContext,
    ) -> tuple[SkillCapabilityProvider, str] | None:
        matches: list[tuple[SkillCapabilityProvider, str]] = []
        for p in self._registry.get_capability_providers():
            if not isinstance(p, SkillCapabilityProvider):
                continue
            try:
                for cap in await p.list(ctx):
                    if isinstance(cap, SkillCapability):
                        if qualify(cap.id) == skill_ref:
                            matches.append((p, cap.name))
            except Exception:
                logger.exception("SkillExecutorCapabilityProvider: skill discovery failed for %s", p.name)
                raise
        if len(matches) > 1:
            raise AmbiguousSkillError(f"skill '{skill_ref}' matches multiple capabilities")
        return matches[0] if matches else None

    # ── ToolCapabilityProvider ────────────────────────────────────────────────

    async def list(self, ctx: ProviderContext) -> list[Capability]:
        return [cap for cap, *_ in _SKILL_EXECUTOR_TOOLS.values()]

    def invoke(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        return self._dispatch(capability_id, arguments, ctx)

    async def _dispatch(
        self,
        capability_id: str,
        arguments: dict[str, Any],
        ctx: ProviderContext,
    ) -> AsyncIterator[CapabilityEvent]:
        tool_name = capability_id.split(":")[-1]
        entry = _SKILL_EXECUTOR_TOOLS.get(tool_name)
        if entry is None:
            yield CapabilityEvent(kind="error",
                                  payload={"code": "UNKNOWN_TOOL", "message": tool_name})
            return

        skill_name = ctx.skill_name
        if not skill_name:
            yield CapabilityEvent(kind="error",
                                  payload={"code": "NO_SKILL_ASSIGNED",
                                           "message": "task.settings has no skill_name"})
            return

        try:
            entry_p = await self._resolve_skill(skill_name, ctx)
        except AmbiguousSkillError as exc:
            yield CapabilityEvent(kind="error", payload={"code": "AMBIGUOUS_SKILL", "message": str(exc)})
            return
        except Exception as exc:
            yield CapabilityEvent(kind="error", payload={"code": "SKILL_DISCOVERY_ERROR", "message": str(exc)})
            return
        if entry_p is None:
            yield CapabilityEvent(kind="error",
                                  payload={"code": "SKILL_NOT_FOUND",
                                           "message": f"skill '{skill_name}' not available"})
            return
        provider, bare_skill_name = entry_p

        _, fn, valid_keys = entry
        filtered = {k: v for k, v in arguments.items() if k in valid_keys}

        try:
            result: SkillResult = await fn(
                **filtered,
                skill_provider=provider,
                skill_name=bare_skill_name,
                ctx=ctx,
            )
        except Exception as exc:
            logger.exception("skill_executor: %s failed", tool_name)
            yield CapabilityEvent(kind="error",
                                  payload={"code": "SKILL_EXEC_ERROR", "message": str(exc)})
            return

        if result.is_error:
            yield CapabilityEvent(kind="error",
                                  payload={"code": "SKILL_EXEC_ERROR", "message": result.content})
        else:
            yield CapabilityEvent(kind="result", payload={"content": result.content})

    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        pass

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(
            name=self.name,
            capability_count=len(_SKILL_EXECUTOR_TOOLS),
            supports_streaming=False,
            supports_cancel=False,
            description=self.description,
        )
