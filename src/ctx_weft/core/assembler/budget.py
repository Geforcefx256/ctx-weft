"""BudgetStrategy：context token 预算裁剪。

详见设计文档 §5.4 / §6.8.7。
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ctx_weft.core.models.errors import ContextOverflowError
from ctx_weft.core.utils.content import image_part_count

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ctx_weft.core.assembler.assembler import ContextBlock, ContextRequest


@runtime_checkable
class BudgetStrategy(Protocol):
    """token 预算裁剪策略。"""

    @abstractmethod
    async def apply(
        self,
        blocks: list["ContextBlock"],
        token_limit: int,
        request: "ContextRequest",
        overflow_limit: int | None = None,
    ) -> list["ContextBlock"]:
        """根据 token_limit 裁剪 blocks；不够时抛 ContextOverflowError。

        ``token_limit`` 是**内容预算**——`ContextAssembler` 传入时已扣掉工具面预留
        （spec: tool-schema-budget）。``overflow_limit`` 是**真窗口**，只用于溢出报错里的
        ``effective_limit`` 字段：报错数字要对得上宿主配的 context_limit，否则排查时对不上账。
        None → 退化为 token_limit（直构路径同旧行为）。

        ⚠️ 本参数是 `ContextAssembler.assemble` 的**关键字实参**，自实现本协议的宿主
        必须接受它，否则 assemble 时 TypeError。
        """
        ...


class PriorityBudgetStrategy(BudgetStrategy):
    """按 eff_priority 保留（0 永不丢，丢序大→小）；同档按最老先丢、再按体积。

    eff_priority = slot_priority 静态阶梯（见 priority.py 的槽位表）+ 动态覆盖：

        静态基线 5/6（history）──┬─ task_id == 当前 且 type=user_prompt ──→ 0（pin，不可裁）
                                 ├─ task_id / origin_task_id == 当前 ────→ 4（提级）
                                 └─ 其余（已完成 task）───────────────────→ 维持 5/6
    丢序：(-eff_prio, 最老 ts, -token)。
    丢弃流程：tool_call↔tool_result 先聚成同生共死单元（防孤立 tool result）
    → 逐单元丢 → 丢到限内为止；只剩 priority-0 地板仍超限时抛富信息 ContextOverflowError。
    详见 spec §4.2 / §4.2.1。"""

    async def apply(
        self,
        blocks: list["ContextBlock"],
        token_limit: int,
        request: "ContextRequest",
        overflow_limit: int | None = None,
    ) -> list["ContextBlock"]:
        """按 token_limit（**已扣工具面预留的内容预算**）裁剪。

        overflow_limit（spec: tool-schema-budget）：溢出报错里 ``effective_limit``
        字段要反映**真窗口**——报错信息用真值、裁剪用扣减值，两者分开传（None = 退化
        为 token_limit，测试直构路径同旧行为）。
        """
        total = sum(b.token_estimate for b in blocks)
        if total <= token_limit:
            return blocks

        eff_prio_fn = lambda b: self._effective_priority(b, request)  # noqa: E731
        eff_prio = {b.id: eff_prio_fn(b) for b in blocks}
        units = self._coalesce_tool_pairs(blocks)

        def _unit_prio(u: list["ContextBlock"]) -> int:
            return max(eff_prio[b.id] for b in u)  # 配对成员同 task 同层 → 一致，max 无碍

        def _unit_sort_key(u: list["ContextBlock"]):
            # 统一键：(-priority, 最老 ts, -总 token)。
            p = _unit_prio(u)
            ts = min((b.metadata.get("timestamp", "") for b in u), default="")
            tok = sum(b.token_estimate for b in u)
            return (-p, ts, -tok)

        droppable = sorted(units, key=_unit_sort_key)
        kept_ids = {b.id for b in blocks}
        for unit in droppable:
            if total <= token_limit:
                break
            if _unit_prio(unit) == 0:
                continue  # priority-0 地板永不丢
            for b in unit:
                if b.id in kept_ids:
                    kept_ids.discard(b.id)
                    total -= b.token_estimate
                    # 丢弃留痕：kind/source/token/当时档位。
                    logger.info(
                        "budget: dropped block kind=%s source=%s tokens=%d eff_priority=%d",
                        b.kind, b.source, b.token_estimate, eff_prio[b.id])

        if total > token_limit:
            floor = [b for b in blocks if eff_prio[b.id] == 0]
            required = sum(b.token_estimate for b in floor)
            # 地板（pin 住的当前消息）里的图片数——它们不可裁，是溢出的直接成因时
            # 用户该做的是删图而非删字，故单独报出（见
            # docs/superpowers/specs/2026-08-20-multimodal-design.md §6.5）。
            n_images = sum(image_part_count(b.content) for b in floor)
            sess = getattr(request, "session", None)
            raise ContextOverflowError(
                required=required,
                effective_limit=overflow_limit if overflow_limit is not None else token_limit,
                context_limit=getattr(sess, "context_limit", 0),
                reserved_output_tokens=getattr(sess, "reserved_output_tokens", 0),
                image_count=n_images,
            )

        return [b for b in blocks if b.id in kept_ids]

    @staticmethod
    def _effective_priority(b: "ContextBlock", request: "ContextRequest") -> int:
        """slot_priority 静态基线 + 两个动态覆盖（依赖 request.task.id）：
        ① pin：当前 task 的 user_prompt → 0（不可裁，当前消息锚）；
        ② 当前 task 内容（task_id 或 origin_task_id == 当前）→ 4（比已完成 5/6 更保）。"""
        task = getattr(request, "task", None)
        cur = getattr(task, "id", None) if task is not None else None
        md = b.metadata or {}
        if cur is not None:
            if md.get("task_id") == cur and str(md.get("type", "")) == "user_prompt":
                return 0
            if md.get("task_id") == cur or md.get("origin_task_id") == cur:
                return 4
        return b.priority

    @staticmethod
    def _coalesce_tool_pairs(blocks: list["ContextBlock"]) -> list[list["ContextBlock"]]:
        """把 assistant(tool_calls) 与其 tool(tool_call_id) 聚成同生共死单元；
        其余 block 各自单元素单元。仅按 id 配对，不改顺序。"""
        by_id = {b.id: b for b in blocks}
        # tool_call_id -> 拥有它的 assistant block id
        owner: dict[str, str] = {}
        for b in blocks:
            if b.metadata.get("role") == "assistant":
                for tc in (b.metadata.get("tool_calls") or []):
                    tcid = tc.get("id")
                    if tcid:
                        owner[tcid] = b.id
        # 归组：assistant id -> [assistant, *其 tool results]
        groups: dict[str, list[str]] = {}
        grouped: set[str] = set()
        for b in blocks:
            if b.metadata.get("role") == "assistant" and b.metadata.get("tool_calls"):
                groups.setdefault(b.id, [b.id])
                grouped.add(b.id)
        for b in blocks:
            if b.metadata.get("role") == "tool":
                tcid = b.metadata.get("tool_call_id", "")
                oid = owner.get(tcid)
                if oid is not None and oid in groups:
                    groups[oid].append(b.id)
                    grouped.add(b.id)
        units: list[list["ContextBlock"]] = []
        for b in blocks:
            if b.id in grouped and b.id not in groups:
                continue  # tool result 已并入其 owner 单元
            if b.id in groups:
                units.append([by_id[i] for i in groups[b.id]])
            else:
                units.append([b])
        return units
