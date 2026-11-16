"""Conversion helpers: Projection → core state models.

Used by resume_session to rebuild Session/Task objects from event replay.
"""

from __future__ import annotations

from copy import deepcopy

from ctx_weft.core.control.types import SessionView, TaskView
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task, deserialize_settings
from ctx_weft.core.utils.clock import as_utc
from ctx_weft.protocols.agent import AgentDetail, AgentState
from ctx_weft.protocols.llm import deserialize_call_config
from ctx_weft.protocols.task import TaskDetail


def task_detail(task) -> TaskDetail:
    return TaskDetail(
        task.id, task.session_id, task.assigned_agent_id or None,
        task.creator_agent_id or None, task.parent_task_id or None,
        task.status, task.title, task.description, deepcopy(task.outputs),
        task.error, task.error_code, task.created_at, task.finished_at,
        task.failure_retry_count, task.blocked_by_task_id, task.finalized,
    )


def agent_detail(agent_id, agent, session_id, current_task) -> AgentDetail:
    choice = getattr(agent, "llm", None)
    return AgentDetail(
        agent_id=agent_id, parent_agent_id=agent.parent_agent_id,
        state=AgentState(agent_id, agent.status, agent.status_reason, current_task),
        spawn_depth=agent.spawn_depth, session_id=session_id, template_id=agent.template_id,
        created_at=getattr(agent, "created_at", None),
        llm_account=choice.account if choice is not None else agent.llm_account,
        llm_model=choice.model if choice is not None else agent.llm_model,
        llm_config=deepcopy(choice.config) if choice is not None else deserialize_call_config(agent.llm_config),
    )


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
        goal=proj.goal,
        tenant_id=proj.tenant_id,
        root_agent_id=proj.root_agent_id or None,
        failure_counter=proj.failure_counter,
        llm_provider=proj.llm_account or None,
        llm_model=proj.llm_model or None,
        # `config` 不带 template_id（2026-09-30）：core 内部零读者，留着它只是给
        # 「会话有一份自己的模板」这个已经删掉的概念再开一个后门。2026-10-07 起
        # `SessionView` 与 `SessionCreated` 载荷里那份也一并删了（见 types.py），
        # 「这条会话属于哪个智能体」归宿主自己的 sessions.cowork_id 列管。
        created_at=_as_utc_opt(proj.created_at),
    )


def task_from_projection(proj: TaskView) -> Task:
    """Rebuild a Task dataclass from its event-sourced projection."""
    # Tasks that were already executing (ACTIVE or SUSPENDED) normally had their user
    # prompt ingested into memory before the crash; mark it so the driver doesn't
    # ingest it a second time on recovery. This is only an inference — the recovery
    # path verifies it against memory (`Runtime._verify_task_prompts_in_memory`),
    # because a crash between a round's commit and its staged-memory flush leaves a
    # started task whose prompt never reached memory.
    prompt_in_memory = bool(proj.user_prompt) and proj.status in ("ACTIVE", "SUSPENDED")
    return Task(
        id=proj.id,
        session_id=proj.session_id,
        details_complete=proj.details_complete,
        status=proj.status,  # type: ignore[arg-type]
        tenant_id=proj.tenant_id,
        title=proj.title,
        description=proj.description,
        user_prompt=proj.user_prompt or None,
        assigned_agent_id=proj.assigned_agent_id or None,
        creator_agent_id=proj.creator_agent_id or None,
        parent_task_id=proj.parent_task_id or None,
        dag_deps=list(proj.dag_deps),
        priority=proj.priority,
        max_retries=proj.max_retries,
        settings=deserialize_settings(proj.settings_raw),
        unattended=proj.unattended,
        port_key=proj.port_key,
        origin_tool_call_id=proj.origin_tool_call_id or None,
        origin_tool_name=proj.origin_tool_name or None,
        replaces_task_id=proj.replaces_task_id,
        switched_to_task_id=proj.switched_to_task_id,
        switch_reason=proj.switch_reason,
        outputs=proj.outputs,
        error=proj.error,
        error_code=proj.error_code,
        failure_retry_count=proj.failure_retry_count,
        blocked_by_task_id=proj.blocked_by_task_id,
        created_at=_as_utc_opt(proj.created_at),
        finished_at=_as_utc_opt(proj.finished_at),
        finalized=proj.finalized,
        user_prompt_in_memory=prompt_in_memory,
    )
