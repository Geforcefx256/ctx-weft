"""判别值的集中定义。

事件 payload 里的 `reason` / `error_code` 是 host 用来分流的对外契约。
它们此前是散落各处的裸字面量（`"llm_outage"` 一个值在同一个函数里写过 4 遍），
改一处会静默分叉——与 `crash_error_code` / `crash_run_outcome` 当初被抽出来
是同一条理由（M3）。

**这些 StrEnum 的值就是上线上的字符串，改值等于改对外契约。**

**这个模块必须保持纯 stdlib**，不 import 任何 `ctx_weft` 运行期模块（无 bus、无队列、
不 await）—— 它被 `task_disposition.py` 引用，而后者的模块 docstring 自称
只引这一个叶子枚举模块、不引任何 `ctx_weft` 运行期东西。若本模块开始 import
运行期依赖，那条纯净性防线就名存实亡了。
"""

from __future__ import annotations

from enum import StrEnum


class InterruptReason(StrEnum):
    """`TaskInterrupted` / `RunInterrupted` / `TaskRequeued` 的 `reason`。"""

    LLM_OUTAGE = "llm_outage"
    RUN_CRASH = "run_crash"
    ASSEMBLY_FAILURE = "assembly_failure"


class CancelReason(StrEnum):
    """`TaskCanceled` 的 `reason`。"""

    USER_CANCEL = "user_cancel"
    FAILURE_THRESHOLD = "failure_threshold"
    PAUSE_ABANDON = "pause_abandon"


class TaskErrorCode(StrEnum):
    """task 结局码。异常派生的码走 `CtxWeftError.code`，不在此列。"""

    BY_OBSERVER = "TASK_FAILED_BY_OBSERVER"
    RETRY_EXHAUSTED = "TASK_FAILED_RETRY_EXHAUSTED"
    BY_THRESHOLD = "TASK_FAILED_BY_THRESHOLD"
    # spec: tool-operations（wp6）——工具副作用结果未知：恢复保守停住等宿主
    # resolve_operation 处置（supply_result / retry_confirmed / cancel_task）。
    TOOL_OUTCOME_UNKNOWN = "TASK_INTERRUPTED_TOOL_OUTCOME_UNKNOWN"
    # spec: execution-limits（wp7）——opt-in 执行限制的四态超限码（不混淆 USER_CANCEL）
    TASK_DEADLINE_EXCEEDED = "TASK_INTERRUPTED_TASK_DEADLINE_EXCEEDED"
    STEP_DEADLINE_EXCEEDED = "TASK_INTERRUPTED_STEP_DEADLINE_EXCEEDED"
    PROVIDER_DEADLINE_EXCEEDED = "TASK_INTERRUPTED_PROVIDER_DEADLINE_EXCEEDED"
    ACTOR_TURN_LIMIT = "TASK_INTERRUPTED_ACTOR_TURN_LIMIT"
    # spec: task-handoff——依赖阻塞取消：on_success 依赖落 FAILED/CANCELED、后继判明
    # 永不可满足而落 CANCELED 的结局码。区别于用户取消：不改写会话终态、不计失败阈值。
    BLOCKED_BY_FAILED_DEP = "BLOCKED_BY_FAILED_DEP"
    # spec: delivery-acceptance——必需检查未通过且修正终止（缺口重复/额度耗尽/预算不足/
    # 检查器故障）的失败结局码。执行可能成功，但交付未通过业务验收。
    TASK_ACCEPTANCE_FAILED = "TASK_FAILED_ACCEPTANCE"
