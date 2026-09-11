"""ctx-weft: Agent Runtime SDK."""

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.models.errors import (
    AgentBusyError,
    AgentNotFound,
    AgentNotRunningError,
    AgentTerminatedError,
    SessionAlreadyExistsError,
    SessionBusyError,
)
from ctx_weft.core.models.task import (
    CompactTaskSettings,
    MetadataFillerTaskSettings,
    NormalTaskSettings,
    TaskSettings,
)
from ctx_weft.core.registry import ProviderRegistry
from ctx_weft.core.runtime import CtxWeftRuntime, SessionStartParams, TurnHandle
from ctx_weft.protocols.agent import AgentDetail, AgentSummary, CompactReceipt
from ctx_weft.protocols.events import EventStore
from ctx_weft.protocols.hitl import HitlReply, HitlRequestView
from ctx_weft.providers.events import InMemoryEventStore

__all__ = [
    # 运行时与入参
    "CtxWeftRuntime",
    "ProviderRegistry",
    "SessionStartParams",
    "TurnHandle",
    # 运行时配置：`CtxWeftRuntime(config=...)` 要求宿主传它，故它必须在受支持的
    # 导出面上——否则「只从顶层与 protocols 取名字」这条纪律对它无解，宿主只能深挖
    # `core.models.config`（迁移期实测踩到）。
    "RuntimeConfig",
    # agent 发现与回执
    "AgentDetail",
    "AgentSummary",
    "CompactReceipt",
    # HITL
    "HitlReply",
    "HitlRequestView",
    # 错误
    "AgentNotFound",
    "AgentNotRunningError",
    "AgentBusyError",
    "AgentTerminatedError",
    "SessionAlreadyExistsError",
    "SessionBusyError",
    # 事件
    "EventStore",
    "InMemoryEventStore",
    # task 配置
    "TaskSettings",
    "NormalTaskSettings",
    "CompactTaskSettings",
    "MetadataFillerTaskSettings",
]
