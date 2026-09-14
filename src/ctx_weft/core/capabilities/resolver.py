"""CapabilityResolver：每次 PrepareStep 执行时调用，根据 template + task 解析 capability。

注：ControlCapabilityProvider 和 SkillExecutorCapabilityProvider 由 PrepareStep 直接召回，
不会出现在传入的 providers 列表中。

三阶段流程：
  1. required  — 强制加载 CapabilityRef.mode="required" 的 capability（走 list()）
  2. retrieve  — 调用各 provider.retrieve(ctx) 召回上下文相关 capability
  3. forbidden — 从合并结果中删除 CapabilityRef.mode="forbidden" 的 capability
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ctx_weft.protocols.capability import Capability, CapabilityProvider

if TYPE_CHECKING:
    from ctx_weft.core.models.task import Task
    from ctx_weft.protocols.context import ProviderContext
    from ctx_weft.protocols.template import AgentTemplate

logger = logging.getLogger(__name__)


class CapabilityResolver:

    async def resolve(
        self,
        template: AgentTemplate,
        task: Task,
        providers: list[CapabilityProvider],
        ctx: ProviderContext,
    ) -> list[Capability]:
        """三阶段解析：required → retrieve → 去 forbidden。"""
        # spec: tool-operations（wp6）——recovery_policy 取值校验（响亮，不静默兜底）：
        # 旧实现是裸 str + `else: manual`，拼错 'retry-safe' 会无声降级成「每次崩溃都
        # 等人」。这里在首个绑定点一次性拦下。
        # 裁决能力**不在此校验**：OperationAdjudicator 靠 isinstance 发现，没有「声明了
        # 却没实现」这个失败模式可堵。
        from ctx_weft.protocols.capability import ToolCapability, ToolCapabilityProvider
        from ctx_weft.protocols.operations import normalize_recovery_policy
        for p in providers:
            if not isinstance(p, ToolCapabilityProvider):
                continue
            try:
                caps = await p.list(ctx)
            except Exception:
                continue
            for cap in caps:
                if not isinstance(cap, ToolCapability):
                    continue
                try:
                    normalize_recovery_policy(getattr(cap, "recovery_policy", None))
                except ValueError as exc:
                    raise ValueError(
                        f"capability {cap.id!r} from provider {p.name!r}: {exc}") from exc
        refs_by_id = {ref.capability_id: ref for ref in template.capability_refs}
        forbidden_ids = {
            ref.capability_id
            for ref in template.capability_refs
            if ref.mode == "forbidden"
        }

        # ── 1. 加载 required capability ──────────────────────────────────────
        required_caps: list[Capability] = []
        for ref in template.capability_refs:
            if ref.mode != "required":
                continue
            cap = await self._find(ref.capability_id, providers, ctx)
            if cap is None:
                logger.warning(
                    "CapabilityResolver: required '%s' not found (template: %s)",
                    ref.capability_id, template.id,
                )
                continue
            required_caps.append(cap)

        # ── 2. 用 ctx 召回相关 capability ────────────────────────────────────
        retrieved: list[Capability] = []
        for provider in providers:
            try:
                retrieved.extend(await provider.retrieve(ctx))
            except Exception:
                logger.exception(
                    "CapabilityResolver: provider '%s'.retrieve() failed", provider.name
                )

        # 合并：required 优先，retrieved 去重追加（跳过非 optional 的声明 ref）
        seen: set[str] = {c.id for c in required_caps}
        bound = list(required_caps)
        for cap in retrieved:
            if cap.id in seen:
                continue
            ref = refs_by_id.get(cap.id)
            if ref is not None and ref.mode != "optional":
                continue
            seen.add(cap.id)
            bound.append(cap)

        # ── 3. 删除 forbidden capability ─────────────────────────────────────
        bound = [c for c in bound if c.id not in forbidden_ids]

        return bound

    async def _find(
        self,
        capability_id: str,
        providers: list[CapabilityProvider],
        ctx: ProviderContext,
    ) -> Capability | None:
        for provider in providers:
            try:
                for cap in await provider.list(ctx):
                    if cap.id == capability_id:
                        return cap
            except Exception:
                logger.exception(
                    "CapabilityResolver: provider '%s'.list() failed", provider.name
                )
        return None
