"""SuspendStep：当 actor 调用 delegate_task / delegate_plan 时写挂起摘要。

Phase 4 §4.5。
"""

from __future__ import annotations

import logging
from typing import Any

from ctx_weft.core.loop.driver import (
    LoopContext, LoopState, Step, StepOutcome, _persist_user_prompt,
)
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import MemoryEvent, MemoryKind, MemoryScope

logger = logging.getLogger(__name__)


class SuspendStep(Step):
    """写挂起摘要，并把「这次 run 停在等子任务」报成 RunOutcome（状态归 TaskManager）。"""

    name = "suspend"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        task = state.task
        events: list[Any] = []

        # 1) 提问还没进对话就补上（走落库的唯一入口：确定性 id + 窗口暂存分流）
        await _persist_user_prompt(state, ctx)

        # 2) Build suspension summary from spawn titles written by control tool
        from ctx_weft.core.models.task import NormalTaskSettings
        titles: list[str] = []
        if isinstance(task.settings, NormalTaskSettings):
            titles = list(task.settings.spawn_titles)
            task.settings.spawn_titles = []
        if titles:
            summary = f"Delegated to sub-task(s): {', '.join(repr(t) for t in titles)}. Awaiting completion."
        else:
            summary = "Agent suspended, awaiting sub-task completion."

        # v2 P1（2026-07-27）：不再写 OBSERVER_SUMMARY（不进装配的死写点）；
        # summary 仅进 TASK_SUSPENDED 事件 payload。

        # TASK_SUSPENDED 不在这里发（Task 4）：本 step 只报「这次 run 停在等子任务」，
        # summary / spawn_titles 随 RunOutcome 交给 TaskManager，由它落状态并发事件。
        # dispatch 段边界（spec 2026-07-16）：父坐实 SUSPENDED 后 fire-and-forget 后台
        # recap，折派发前 raw——挂起空窗跑 LLM。所有委派父生效（不加 _is_own_root 门控）；
        # resume 竞态由 _run_loop 入口 await_pending_background_observe 封死。
        from ctx_weft.core.loop.steps.background_observe import launch_background_observe
        launch_background_observe(state, ctx, boundary="dispatch")

        run_outcome = RunOutcome(
            kind=RunOutcomeKind.SUSPENDED_ON_CHILDREN,
            summary=summary,
            spawn_titles=tuple(titles),
        )

        return StepOutcome(
            next_step=None,  # loop stops; TaskManager will re-queue when children done
            state_patch={"act_exit_reason": "suspended", "run_outcome": run_outcome},
            events=events,
        )
