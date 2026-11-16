"""agent 发现接口的 host-facing 视图类型（spec 5）。

层级关系不在接口层嵌套——各条自带 parent_agent_id，调用方按需还原成树。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ctx_weft.protocols.llm import ModelCallConfig
from ctx_weft.protocols.task import TaskDetail


@dataclass(frozen=True)
class AgentState:
    """Execution facts shared by session and individual agent queries."""

    agent_id: str
    status: str
    status_reason: str = ""
    current_task: TaskDetail | None = None


@dataclass(frozen=True)
class AgentSummary:
    agent_id: str
    parent_agent_id: str | None
    status: str
    current_task_id: str | None
    spawn_depth: int
    created_at: datetime | None = None


@dataclass(frozen=True)
class AgentDetail:
    agent_id: str
    parent_agent_id: str | None
    state: AgentState
    spawn_depth: int
    session_id: str
    template_id: str
    created_at: datetime | None = None
    llm_account: str = ""
    llm_model: str = ""
    llm_config: ModelCallConfig | None = None


@dataclass(frozen=True)
class CompactReceipt:
    """`compact_agent` 的回执。

    `task_id_is_transient=True` 表示 `task_id` 是一个**只用于圈定折叠范围的内存
    载体 id**，事件库里没有对应记录——调用方不要拿它去查。改动前这条约定只活在
    `compact_session` 的 docstring 里，调用方只能靠「我传没传 task_id」自己反推。
    """

    session_id: str
    agent_id: str
    task_id: str
    task_id_is_transient: bool
