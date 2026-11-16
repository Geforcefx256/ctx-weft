"""Compose session snapshots from the same public objects as individual queries."""

from copy import deepcopy

from ctx_weft.core.control.converters import agent_detail, task_detail
from ctx_weft.core.control.reducers import (
    read_task_view,
    rebuild_view,
)
from ctx_weft.core.control.startup_view import StartupRecoveryView, read_startup_pending_hitl
from ctx_weft.protocols.runtime import SessionStateSummary


async def public_task_detail(store, session_id, task_id, task, head):
    if not task_id:
        return None
    if task is None or not task.details_complete:
        task = await read_task_view(store, session_id, task_id, through_position=head)
    # Legacy logs can retain an agent's task reference without its TaskCreated.
    # A missing body remains absent; explicit get_task still reports TaskNotFound.
    return task_detail(task) if task is not None else None


async def session_state_summary(
    store, view, session, head: int, *, threshold_tripped: bool | None = None,
) -> SessionStateSummary:
    agent_records = view.agents
    root = agent_records.get(session.root_agent_id)
    if root is None and session.root_agent_id:
        durable = await rebuild_view(store, session.id, through_position=head)
        agent_records = {**durable.agents, **agent_records}
        root = agent_records.get(session.root_agent_id)
    if isinstance(view, StartupRecoveryView):
        pending = await read_startup_pending_hitl(store, view, session.id, head)
    else:
        pending = tuple(deepcopy(request.to_view()) for request in view.hitl.pending.values())
    running = any(agent.status == "running" for agent in agent_records.values())
    interrupted = any(agent.status == "interrupted" for agent in agent_records.values())
    tasks = {}

    async def task_for(task_id):
        if not task_id:
            return None
        if task_id not in tasks:
            tasks[task_id] = await public_task_detail(
                store, session.id, task_id, view.tasks.get(task_id), head,
            )
        return tasks[task_id]

    agents = tuple([agent_detail(aid, agent, session.id, await task_for(agent.current_task_id))
                    for aid, agent in sorted(agent_records.items())])
    root_task = await task_for(root.current_task_id) if root is not None else None
    if root_task and not running and not pending and not interrupted and root.status != "terminated":
        seen = set()
        while root_task is not None and root_task.parent_task_id and root_task.task_id not in seen:
            seen.add(root_task.task_id)
            root_task = await task_for(root_task.parent_task_id)
    return SessionStateSummary(
        root_agent=next((agent for agent in agents if agent.agent_id == session.root_agent_id), None),
        root_task=root_task, agents=agents, pending_hitl=pending,
        has_running_agents=running, has_interrupted_agents=interrupted,
        failure_counter=session.failure_counter,
        threshold_tripped=session.threshold_tripped if threshold_tripped is None else threshold_tripped,
        goal=session.goal,
    )
