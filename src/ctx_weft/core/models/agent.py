"""Agent 与它的运行期计数 LoopGuard。

`LoopGuard` 与 `Agent` 同住：它是 `Agent.loop_guard` 的类型，且是 agent 生命周期内
唯一可变的那块（`context_tokens` 由 act 改写）。

agent 的**状态**不在这里——它住在 `orchestrator/lifecycle/agent_manager.py` 的
`_AgentRecord`，状态词表在同包的 `status.py`（`AgentStatus`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ctx_weft.protocols import LoopConfig, MemoryConfig


# ── LoopGuard ─────────────────────────────────────────────────────────────────


@dataclass
class LoopGuard:
    """Agent 运行时计数与测量值（mutable，每轮可能更新）。"""

    turns_used: int = 0
    context_tokens: int = 0
    context_message_count: int = 0
    context_limit: int = 180_000
    reserved_output_tokens: int = 8192
    # spec: tool-schema-budget——瞬态（不进持久快照）：上次发送请求的工具面指纹与估算。
    # 增量估算在指纹变化时计入 schema 差值（新面估算 − 上次面估算；真实基线已含旧面）。
    last_tools_signature: str = ""
    last_tools_est: int = 0


# ── Agent ─────────────────────────────────────────────────────────────────────


@dataclass
class Agent:
    """Agent 实例。"""

    id: str
    session_id: str
    template_id: str
    tenant_id: str = "default"

    parent_agent_id: str | None = None
    spawn_depth: int = 0

    loop_guard: LoopGuard = field(default_factory=LoopGuard)

    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    loop_config: LoopConfig = field(default_factory=LoopConfig)

    runtime: dict[str, Any] = field(default_factory=dict)

    created_at: datetime | None = None
    updated_at: datetime | None = None
