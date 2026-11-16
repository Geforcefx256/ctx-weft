"""Provider call context and knowledge citations.

Text and image content is defined in protocols.content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# ── Citation（KnowledgeDoc 引用信息）────────────────────────────────────────────


@dataclass
class Citation:
    """知识检索结果的引用信息。"""

    url: str | None = None
    title: str | None = None
    anchor: str | None = None  # 段落锚点
    metadata: dict[str, Any] = field(default_factory=dict)


# ── ProviderContext ────────────────────────────────────────────────────────────


@dataclass
class ProviderContext:
    """Provider 调用上下文。core 在每次 provider 操作时构造并注入。

    携带：
    - 当前 session/task/agent 标识 + agent 模板 id（限定操作范围 / 授权维度）
    - tenant_id（多租户预留）
    - trace_id（全链路追踪）
    - invocation_id（capability invoke 时分配，用于 cancel）
    - 时间戳
    """

    session_id: str
    tenant_id: str = "default"
    task_id: str | None = None
    agent_id: str | None = None
    agent_template_id: str = ""  # 该 agent 的模板 id；授权按模板维度做策略（AllowListAuthorizer）
    trace_id: str | None = None
    invocation_id: str | None = None
    request_id: str | None = None
    timestamp: datetime | None = None  # None 时 provider 自填 now
    skill_name: str = ""  # 供 SkillExecutorCapabilityProvider 读取
    extra: dict[str, Any] = field(default_factory=dict)
