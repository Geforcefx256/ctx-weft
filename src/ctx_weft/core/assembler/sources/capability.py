"""CapabilitySource：按 cap.kind 分三路产出 capabilities blocks。

  kind="tool"   → raw ToolCapability
  kind="skill"  → raw SkillCapability
  kind="agent"  → raw AgentCapability

Source 从 cache 的当前可用面返回原始能力；无 cache 的独立使用场景回退到
request.bound_capabilities。purpose 模板负责用途过滤、分组、排版和落位。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.capability import (
    AgentCapability,
    Capability,
    SkillCapability,
    ToolCapability,
    qualify,
)
from ctx_weft.protocols.llm import LLMTool

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import AssemblerDeps, ContextBlock, ContextRequest


logger = logging.getLogger(__name__)


def capability_ref(capability: Capability, provider: str | None = None) -> dict[str, str]:
    return {"provider": provider or capability.id.rsplit(":", 1)[0], "id": capability.id}


async def resolve_capabilities(refs: list[dict], deps: AssemblerDeps):
    """Resolve fixed references through list(), never repeat relevance retrieval."""
    provider_names = list(dict.fromkeys(ref["provider"] for ref in refs))

    async def listed(name):
        provider = deps.capability_provider_index.get(name)
        if provider is None:
            return {}, "Provider is not registered"
        try:
            return {cap.id: cap for cap in await provider.list(deps.provider_ctx)}, None
        except Exception as exc:
            return {}, f"provider.list() failed: {exc}"

    results = await asyncio.gather(*(listed(n) for n in provider_names))
    listings = dict(zip(provider_names, results, strict=True))
    resolved, errors = {}, []
    seen = set()
    for ref in refs:
        name, cid = ref["provider"], ref["id"]
        key = (name, cid)
        if key in seen:
            continue
        seen.add(key)
        capabilities, error = listings[name]
        cap = capabilities.get(cid)
        if cap is not None:
            resolved[key] = cap
            continue
        error = error or "Capability was not returned by provider.list()"
        logger.warning("capability unavailable: provider=%s capability_id=%s: %s", name, cid, error)
        errors.append({"provider": name, "id": cid, "error": error})
    return resolved, errors


async def provider_descriptions(names, deps):
    """Read descriptions in the same explicit context as capability discovery."""
    index = getattr(deps, "capability_provider_index", None) or {}
    descriptions = {}
    for name in dict.fromkeys(names):
        provider = index.get(name)
        if provider is not None:
            info = await provider.describe(deps.provider_ctx)
            descriptions[name] = info.description or ""
    return descriptions


async def capability_blocks(refs, resolved, deps):
    """Transient rendering facts, excluded from PromptContext's event payload."""
    from ctx_weft.core.assembler.context import json_value

    descriptions = await provider_descriptions(
        [ref["provider"] for ref in refs if (ref["provider"], ref["id"]) in resolved], deps)
    blocks = []
    for ref in refs:
        cap = resolved.get((ref["provider"], ref["id"]))
        if cap is not None:
            blocks.append({"definition": json_value(cap), "metadata": {
                "provider_name": ref["provider"],
                "provider_description": descriptions.get(ref["provider"], ""),
            }})
    return blocks


def build_llm_tools(caps: list[Capability], purpose: str, *, terminal_only: bool = False) -> list[LLMTool]:
    """Build the frozen LLM tool snapshot for one purpose."""
    tools = []
    for cap in caps:
        definition = cap if isinstance(cap, dict) else vars(cap)
        if definition.get("kind") == "tool" and purpose in definition.get("purposes", []):
            tools.append(LLMTool(name=qualify(definition["id"]),
                                 description=definition["description"],
                                 input_schema=definition["input_schema"]))

    if terminal_only:
        from copy import deepcopy

        from ctx_weft.core.capabilities.control_tools import REPORT_OBSERVATION_NAME

        for tool in tools:
            if tool.name != REPORT_OBSERVATION_NAME:
                continue
            tool.input_schema = deepcopy(tool.input_schema)
            props = tool.input_schema.get("properties", {})
            if "task_status" in props:
                props["task_status"] = {"type": "string", "enum": ["success", "fail"],
                                        "description": "Final verdict on the old goal; continue and waiting are not allowed."}
                props.get("next_step_hint", {}).update(description="Leave empty: the old task will not resume.")
                props["task_failure_reason"] = {"type": "string", "description": "Required for fail: explain the unfinished original goal and the user topic switch. Empty for success."}
                required = tool.input_schema.setdefault("required", [])
                if "task_status" not in required:
                    required.append("task_status")
    return tools


class CapabilitySource:
    name = "capability"

    async def fetch(
        self,
        request: "ContextRequest",
        deps: "AssemblerDeps",
    ) -> AsyncIterator["ContextBlock"]:
        from ctx_weft.core.assembler.assembler import ContextBlock

        cache = getattr(deps, "capability_cache", None)
        if cache is None:
            capabilities = request.bound_capabilities
        else:
            agent_id = getattr(request.agent, "id", "") or request.scope.agent_id
            task_id = getattr(request.task, "id", "") or request.scope.task_id
            capabilities = cache.available(agent_id, task_id)

        tools: list[ToolCapability] = []
        skills: list[SkillCapability] = []
        agents: list[AgentCapability] = []
        for cap in capabilities:
            if isinstance(cap, ToolCapability):
                tools.append(cap)
            elif isinstance(cap, SkillCapability):
                skills.append(cap)
            elif isinstance(cap, AgentCapability):
                agents.append(cap)

        # ── kind="tool" → LLMTool ────────────────────────────────────────────
        # provider_name → description（live：MCP connect 后才有；deps 持有 provider 对象）
        descriptions = await provider_descriptions(
            [cap.id.rsplit(":", 1)[0] for cap in capabilities], deps)

        def _provider_meta(cap_id: str) -> tuple[str, str]:
            # provider 名 = cap.id 去掉末段（mcp:github:create_issue → mcp:github，
            # local_skill:pdf → local_skill），与 provider.name 对齐；描述 live 读取。
            pname = cap_id.rsplit(":", 1)[0]
            return pname, descriptions.get(pname, "")

        for cap in tools:
            qname = qualify(cap.id)
            provider_name, provider_description = _provider_meta(cap.id)
            yield ContextBlock(
                id=generate_id("blk"),
                source="capability",
                kind="capabilities",
                metadata={
                    "capability_id": cap.id,
                    "capability_name": qname,  # qualified: rendered into "Available Tools" prose
                    "capability_kind": "tool",
                    "input_schema": cap.input_schema,
                    "side_effects": cap.side_effects,
                    "provider_name": provider_name,
                    "provider_description": provider_description,
                },
                value=cap,
            )

        # ── kind="skill" → 每个 skill 独立 block，由 composer 聚合成列表 ──────
        for cap in skills:
            provider_name, provider_description = _provider_meta(cap.id)
            yield ContextBlock(
                id=generate_id("blk"),
                source="capability",
                kind="capabilities",
                metadata={
                    "capability_id": cap.id,
                    "capability_name": qualify(cap.id),
                    "capability_kind": "skill",
                    "provider_name": provider_name,
                    "provider_description": provider_description,
                },
                value=cap,
            )

        for cap in agents:
            provider_name, provider_description = _provider_meta(cap.id)
            yield ContextBlock(
                id=generate_id("blk"),
                source="capability",
                kind="capabilities",
                metadata={
                    "capability_id": cap.id,
                    "capability_name": qualify(cap.id),
                    "capability_kind": "agent",
                    "provider_name": provider_name,
                    "provider_description": provider_description,
                },
                value=cap,
            )
