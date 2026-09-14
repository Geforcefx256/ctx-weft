"""Conversion helpers: Projection → core state models.

Used by resume_session to rebuild Session/Task objects from event replay.
"""

from __future__ import annotations

from ctx_weft.core.control.types import SessionView, TaskView
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task, deserialize_settings
from ctx_weft.core.utils.clock import as_utc


def _as_utc_opt(dt):
    """归一可选 datetime 为 aware(UTC)；None 透传。事件重放 / DB 反序列化可能丢 tz，
    此处在重建 core 模型的边界统一补齐，使内存态 created_at 恒 aware——与 now_utc()
    新建的活任务同 tz，避免二者混入同一注册表后排序抛 naive/aware 比较错。"""
    return as_utc(dt) if dt is not None else None


def session_from_projection(proj: SessionView) -> Session:
    """Rebuild a Session dataclass from its event-sourced projection."""
    return Session(
        id=proj.id,
        user_prompt=proj.user_prompt,
        status=proj.status,  # type: ignore[arg-type]
        goal=proj.goal,
        tenant_id=proj.tenant_id,
        root_agent_id=proj.root_agent_id or None,
        token_budget=proj.token_budget,
        context_limit=proj.context_limit,
        reserved_output_tokens=getattr(proj, "reserved_output_tokens", 8192),
        failure_counter=proj.failure_counter,
        llm_provider=proj.llm_account or None,
        llm_model=proj.llm_model or None,
        config={"template_id": proj.template_id},
        created_at=_as_utc_opt(proj.created_at),
    )


def task_from_projection(proj: TaskView) -> Task:
    """Rebuild a Task dataclass from its event-sourced projection."""
    # Tasks that were already executing (ACTIVE or SUSPENDED) had their user
    # prompt ingested into memory before the crash; mark it so the driver
    # doesn't ingest it a second time on recovery.
    prompt_in_memory = bool(proj.user_prompt) and proj.status in ("ACTIVE", "SUSPENDED")
    return Task(
        id=proj.id,
        session_id=proj.session_id,
        status=proj.status,  # type: ignore[arg-type]
        tenant_id=proj.tenant_id,
        title=proj.title,
        description=proj.description,
        user_prompt=proj.user_prompt or None,
        original_user_prompt=proj.original_user_prompt or None,
        assigned_agent_id=proj.assigned_agent_id or None,
        creator_agent_id=proj.creator_agent_id or None,
        parent_task_id=proj.parent_task_id or None,
        dag_deps=list(proj.dag_deps),
        priority=proj.priority,
        max_retries=proj.max_retries,
        settings=deserialize_settings(proj.settings_raw),
        interaction_mode=proj.interaction_mode,  # type: ignore[arg-type]
        unattended=proj.unattended,
        origin_tool_call_id=proj.origin_tool_call_id or None,
        origin_tool_name=proj.origin_tool_name or None,
        outputs=proj.outputs,
        error=proj.error,
        created_at=_as_utc_opt(proj.created_at),
        finished_at=_as_utc_opt(proj.finished_at),
        user_prompt_in_memory=prompt_in_memory,
    )
