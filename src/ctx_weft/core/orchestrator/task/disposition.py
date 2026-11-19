"""run 结局 + 重试预算 → task 处置。**唯一**一份「task 下一步是什么」的判据。

为什么单独成模块、且是纯函数：这份判断今天散在四个地方——
`FinalizeStep` 判重试耗尽、`ObserveStep._apply_assessment` 把 verdict 写成状态、
`TaskManager._handle_task_failure` 判重试预算、`_run_loop` 的 except 链写挂起态。
把它们收进一张表，`TaskManager` 才可能成为 task 状态的唯一改写者
（docs/events-v2.md §2.1 的分层原则推到 task 层）。

纯函数是刻意的：它不该知道事件总线、不该知道队列、不该 await 任何东西。
loop 报「发生了什么」，这张表回答「那么 task 变成什么」，TaskManager 负责执行。

本模块只引一个叶子枚举模块 `discriminators`（无 bus / 无队列 / 不 await，见其
模块 docstring），不引任何其它 `ctx_weft` 运行期东西——纯函数的约束仍然成立。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ctx_weft.core.models.discriminators import TaskErrorCode
from ctx_weft.core.utils.verdict import (
    VERDICT_CONTINUE,
    VERDICT_WAITING,
    VERDICT_SUCCESS,
    normalize_verdict,
)

__all__ = ["Disposition", "RunOutcome", "RunOutcomeKind", "disposition_for"]


class RunOutcomeKind(StrEnum):
    """一次 run 是怎么结束的。**run 自己的词表**，不是 task 状态。"""

    TASK_SWITCHED = "task_switched"
    COMPLETED = "completed"                        # step 链跑到头，带 observer 判决
    AWAITING_HUMAN = "awaiting_human"              # HITL 冷 park
    SUSPENDED_ON_CHILDREN = "suspended_on_children"  # 等子任务
    INTERRUPTED = "interrupted"                    # LLM outage / run 崩溃
    CANCELED = "canceled"                          # 被取消


@dataclass(frozen=True)
class RunOutcome:
    """loop 交给 TaskManager 的全部信息：**发生了什么**，不含「task 该变成什么」。"""

    kind: RunOutcomeKind
    verdict: str = ""            # COMPLETED 时：success / waiting / continue / fail
    summary: str = ""
    outputs: object = None
    error: str = ""              # 死因 / 受阻原因（自由文本，只作溯源）
    error_code: str = ""
    reason: str = ""             # INTERRUPTED：InterruptReason 成员；CANCELED：CancelReason 成员
    # INTERRUPTED 专用：这次打断允不允许原地重试。默认 False（不重试）与真实判据
    # `getattr(exc, "retriable", True)` 的缺省方向相反，是有意的：漏传导致"不重试"
    # 只是让 task 停在 INTERRUPTED 等 /resume，可恢复；漏传导致"重试"会烧预算。
    # crash 入口必须显式传 `getattr(exc, "retriable", True)`。
    retriable: bool = False
    hitl_id: str = ""            # AWAITING_HUMAN
    observe_id: str = ""
    observe_disposition: dict | None = None
    observe_report_pending: bool = False
    fork_checkpoint: dict | None = None
    spawn_titles: tuple[str, ...] = ()   # SUSPENDED_ON_CHILDREN


@dataclass(frozen=True)
class Disposition:
    """处置结果：task 落到哪个状态 + TaskManager 该发哪条事件。"""

    status: str
    event_type: str
    payload: dict


def disposition_for(
    outcome: RunOutcome, *, retry_count: int, max_retries: int,
) -> Disposition:
    """结局 → 处置；重试预算只用于运行异常。

    observer 判 continue 表示任务尚未完成：保持 retry_count、重新入队，不按预算
    降级成失败。只有明确的 fail 判决产生观察者失败。
    可重试的运行异常在预算内消耗一次 retry，耗尽后停在 INTERRUPTED 等恢复。
    outage 由调用方显式传 retriable=False，从不原地重试。
    """
    if outcome.kind is RunOutcomeKind.TASK_SWITCHED:
        return Disposition("REVIEWING", "TaskReviewing", {"reason": "user_topic_switch"})

    if outcome.kind is RunOutcomeKind.AWAITING_HUMAN:
        return Disposition("AWAITING_HUMAN", "TaskAwaitingHuman",
                           {"hitl_id": outcome.hitl_id})

    if outcome.kind is RunOutcomeKind.SUSPENDED_ON_CHILDREN:
        return Disposition("SUSPENDED", "TaskSuspended", {
            "summary": outcome.summary,
            "spawn_titles": list(outcome.spawn_titles),
        })

    if outcome.kind is RunOutcomeKind.CANCELED:
        # 今天 `runtime.py` 的 `TASK_CANCELED` payload 是字面 `{}`（取消从不编造
        # `reason`）。`reason` 为空时不放这个键，避免平白多出一个 `{"reason": ""}`
        # 破坏行为等价；非空时照放，给未来真有取消原因的调用方留口。
        payload: dict = {}
        if outcome.reason:
            payload["reason"] = outcome.reason
        return Disposition("CANCELED", "TaskCanceled", payload)

    if outcome.kind is RunOutcomeKind.INTERRUPTED:
        # 调用方契约：outage 必须显式传 retriable=False，**不得**转发 exc.retriable
        # ——`LLMOutageError.retriable` 是 True，转发会让 outage 在预算充足时被错误地
        # 原地重试。本字段是"这次打断允不允许重试"的判据，不是异常标志位的镜像。
        if outcome.retriable and retry_count < max_retries:
            return Disposition("PENDING", "TaskRequeued", {
                "reason": outcome.reason, "retry_count": retry_count + 1,
            })
        return Disposition("INTERRUPTED", "TaskInterrupted", {
            "reason": outcome.reason,
            "error_code": outcome.error_code,
            "error_message": outcome.error,
            "retry_count": retry_count,
        })

    # COMPLETED：observer 的判决，不消耗运行异常的重试预算。
    #
    # 三态**显式匹配**（2026-09-28）：改名前这里是「success → / retry → / 其余 → FAILED」，
    # 那个兜底同时服务 `fail` 和**任何认不出的值**——一个错别字就让 task 无声判死。现在先经
    # `normalize_verdict` 归一（认不出归 continue，不是 fail），`fail` 单独一支。
    verdict = normalize_verdict(outcome.verdict)
    if verdict == VERDICT_SUCCESS:
        return Disposition("FINISHED", "TaskFinished", {
            "outcome": VERDICT_SUCCESS, "summary": outcome.summary, "outputs": outcome.outputs,
        })
    if verdict == VERDICT_WAITING:
        return Disposition("AWAITING_HUMAN", "TaskAwaitingHuman", {
            "hitl_id": outcome.hitl_id, "outcome": VERDICT_WAITING,
        })
    if verdict == VERDICT_CONTINUE:
        return Disposition("PENDING", "TaskRequeued", {
            "outcome": VERDICT_CONTINUE, "summary": outcome.summary,
            "retry_count": retry_count,
        })
    return Disposition("FAILED", "TaskFailed", {
        "error_code": TaskErrorCode.BY_OBSERVER,
        "error_message": outcome.error,
        "retry_count": retry_count,
    })
