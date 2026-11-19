"""Host-facing runtime initialization and query values."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from ctx_weft.protocols.agent import AgentDetail
from ctx_weft.protocols.events import StoredEvent
from ctx_weft.protocols.hitl import HitlRequestView
from ctx_weft.protocols import task

T = TypeVar("T")


@dataclass(frozen=True)
class SessionForkResult:
    """A committed, memory-initialized container; no task has been dispatched."""

    session_id: str
    root_agent_id: str
    seed_id: str
    source_session_id: str
    response_event_id: str
    prompt_event_id: str
    record_count: int


class MemorySeedConflictError(ValueError):
    """An existing target, record or receipt disagrees with its frozen seed."""


class ForkCheckpointUnavailableError(ValueError):
    """The reply has no committed ObserveApplied fork checkpoint."""


@dataclass(frozen=True)
class ShutdownReport:
    """Execution quiescence and confirmed recoverability are separate guarantees."""

    quiescent: bool
    durable: bool
    session_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class RuntimeClosedError(RuntimeError):
    """A command was submitted after instance shutdown began."""


class RuntimeShutdownError(RuntimeError):
    def __init__(self, report: ShutdownReport):
        self.report = report
        super().__init__("Runtime shutdown incomplete: " + "; ".join(report.errors))


@dataclass(frozen=True)
class PromptDiagnostic:
    """Redacted recompilation of saved inputs, not historical wire bytes."""

    event_id: str
    system_prompt: str
    messages: list[dict[str, Any]]
    tool_names: list[str]
    capability_errors: list
    trim: bool
    source: Literal["recompiled"] = "recompiled"


@dataclass(frozen=True)
class QueryResult(Generic[T]):
    data: T
    source: Literal["memory", "events"]
    loaded: bool
    observed_at: datetime
    committed_position: int | None = None


@dataclass(frozen=True)
class InitializationReport:
    """Host-side aggregate of explicitly selected session initialization results."""

    checked_sessions: int
    interrupted_session_ids: list[str]
    interrupted_task_ids: list[str]
    sessions: list[SessionInitializationResult] = field(default_factory=list)


@dataclass(frozen=True)
class SessionStateSummary:
    """One independent query snapshot composed of the existing public query values."""

    root_agent: AgentDetail | None = None
    root_task: task.TaskDetail | None = None
    agents: tuple[AgentDetail, ...] = ()
    pending_hitl: tuple[HitlRequestView, ...] = ()
    has_running_agents: bool = False
    has_interrupted_agents: bool = False
    failure_counter: int = 0
    threshold_tripped: bool = False
    goal: str = ""


@dataclass(frozen=True)
class SessionInitializationResult:
    """An initialization-time cut, including sessions requiring no new repairs.

    Returned to concurrent first callers only; repeat initialization returns None.
    This is not a live query cache. A fresh runtime reconstructs these results
    even when an earlier process committed repairs but lost its host projection.
    """

    session_id: str
    before_position: int
    committed_position: int
    state: SessionStateSummary
    startup_events: list[StoredEvent] = field(default_factory=list)


@dataclass(frozen=True)
class SessionDetail:
    session_id: str
    root_agent_id: str
    created_at: datetime | None
    state: SessionStateSummary
