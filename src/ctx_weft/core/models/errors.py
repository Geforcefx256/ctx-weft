"""ctx-weft 错误类型。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ctx_weft.core.models.discriminators import InterruptReason, TaskErrorCode

if TYPE_CHECKING:
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome


class CtxWeftError(Exception):
    """所有 ctx-weft 异常的基类。"""

    code: str = "CTX_WEFT_ERROR"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


# ── 状态相关 ───────────────────────────────────────────────────────────────────


class AgentNotFound(CtxWeftError):
    code = "AGENT_NOT_FOUND"


class AgentBusyError(CtxWeftError):
    """agent 正在执行（running），拒收新消息（spec §4.1：忙碌直接拒绝，不排队）。

    调用方自行重试，或先 pause/cancel 该 agent。对位既有的 `SessionBusyError`。"""

    code = "AGENT_BUSY"


class AgentTerminatedError(CtxWeftError):
    """agent 已被显式 cancel（终态 'terminated'），不再接受任何输入。"""

    code = "AGENT_TERMINATED"


class AgentNotRunningError(CtxWeftError):
    """`pause_agent` 的目标不是 `running`（spec 7：只对 running 生效，其余状态直接
    报错，不静默降级）。对位 `AgentBusyError`（那个是「忙碌拒收」，这个是「不忙没得停」）。
    """

    code = "AGENT_NOT_RUNNING"


class TaskNotFound(CtxWeftError):
    code = "TASK_NOT_FOUND"


class SessionNotFound(CtxWeftError):
    code = "SESSION_NOT_FOUND"


# ── Capability 相关 ────────────────────────────────────────────────────────────


class UnknownCapability(CtxWeftError):
    code = "UNKNOWN_CAPABILITY"


class CapabilityNotAuthorized(CtxWeftError):
    code = "CAPABILITY_NOT_AUTHORIZED"


class DuplicateCapabilityName(CtxWeftError):
    code = "DUPLICATE_CAPABILITY_NAME"


class TemplateNotFoundError(CtxWeftError):
    """模板查找失败：裸 id 无前缀 / 前缀路由不到 provider / provider 不认识局部名。

    模板引用必须是规范形式 'provider:name'（如 'agent:planner'）——边界强制前缀
    （spec 2026-07-22），core 不做扫描回落。"""

    code = "TEMPLATE_NOT_FOUND"

    def __init__(self, template_ref: str, *, providers: list[str] | None = None) -> None:
        self.template_ref = template_ref
        self.providers = list(providers or [])
        hint = (
            f"registered agent providers: {', '.join(self.providers)}"
            if self.providers else "no AgentCapabilityProvider registered"
        )
        super().__init__(
            f"Template {template_ref!r} not found ({hint}); template refs must be "
            f"canonical 'provider:name', e.g. 'agent:planner'"
        )


# ── Guard 相关 ─────────────────────────────────────────────────────────────────


class BudgetExceededError(CtxWeftError):
    code = "TOKEN_BUDGET_EXCEEDED"


class ContextOverflowError(CtxWeftError):
    code = "CONTEXT_OVERFLOW"
    retriable = False  # 非瞬时：同批 block 重装配必再溢出，不可 resume（spec §4.5）

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        required: int = 0,
        effective_limit: int = 0,
        context_limit: int = 0,
        reserved_output_tokens: int = 0,
        image_count: int = 0,
    ) -> None:
        self.required = required
        self.effective_limit = effective_limit
        self.context_limit = context_limit
        self.reserved_output_tokens = reserved_output_tokens
        self.image_count = image_count
        if not message and context_limit:
            message = (
                f"上下文超出模型可用窗口：保护槽位（角色设定 + 当前任务/消息）约 {required} tokens，"
                f"已超过为输出预留后的可用窗口 effective_limit={effective_limit}"
                f"（= 模型窗口 {context_limit} − 输出预留 {reserved_output_tokens}）。"
            )
            if image_count:
                # 函数内 import，且现在是**必须**的：`core.utils.content` 模块级就引本模块
                # （ContextOverflowError 等），顶层引它会成真环。常量随 utils 拆分搬进
                # content 之后，这条理由从「刻意保持轻量」升级成「不这样写就成环」。
                from ctx_weft.core.utils.content import _IMAGE_PART_TOKENS
                message += (
                    f"其中不可裁的当前消息含 {image_count} 张图片，"
                    f"约占 {image_count * _IMAGE_PART_TOKENS} tokens。"
                    "请先减少图片数量，或改用更大上下文窗口的模型。"
                )
            else:
                message += "请改用更大上下文窗口的模型，或缩短当前消息 / 任务描述。"
        super().__init__(message, code=code)


class MaxTurnsExceeded(CtxWeftError):
    code = "MAX_TURNS_EXCEEDED"


def crash_error_code(exc: BaseException | None) -> str:
    """运行层崩溃的 error_code 推导，供 `RunInterrupted` 与 `TaskInterrupted` 共用。

    同一次崩溃发出的两条事件（run 域 `RunInterrupted` / task 域 `TaskInterrupted`）
    必须同码——此前各自内联一份同构表达式，日后改一处会静默分叉。`exc is None`
    的兜底只在 `TaskManager._suspend_task_interrupted` 的重排/恢复等非崩溃调用路径上
    触发（那里 `exc` 可以是 None）；runtime._run_loop 的 crash 分支里 `exc` 恒非 None。
    """
    if exc is None:
        return "RUN_CRASH"
    return getattr(exc, "code", None) or type(exc).__name__


def crash_run_outcome(exc: BaseException) -> "RunOutcome":
    """运行层崩溃 → `RunOutcome`。**崩溃这份结局只有这一个构造点。**

    三处调用方（`TaskManager._run_task` 的 except、`runtime.run_single_task` 的
    except、`_run_loop` 崩溃支给 `RunFinished.outcome` 用的那份）此前各内联一份同构
    字面量——与 `crash_error_code` 当年被抽出来的理由一模一样（M3：别让同一条判据在
    多处各写一份，改一处会静默分叉）。

    `retriable=getattr(exc, "retriable", True)` 是**崩溃专用**的取法，与 outage 支
    硬编码的 `retriable=False` **不同源、不许合并**：`LLMOutageError.retriable` 恒为
    True（`protocols/llm.py`），转发它会让 outage 在预算充足时被错误地原地重试
    （见 `task_disposition.py` 顶部契约）。故本工厂只服务崩溃，outage 不走这里。

    放在 errors.py 而不是 task_disposition.py：与 `crash_error_code` 同处，且
    `task_disposition` 是纯 stdlib、不引任何 ctx_weft 模块，此方向无环。
    """
    from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
    return RunOutcome(
        kind=RunOutcomeKind.INTERRUPTED,
        reason=InterruptReason.RUN_CRASH,
        error=str(exc),
        error_code=crash_error_code(exc),
        retriable=getattr(exc, "retriable", True),
    )


# ── 入口内容校验 ───────────────────────────────────────────────────────────────


class InvalidContentError(CtxWeftError):
    """入口内容格式非法：未知 media_type / base64 畸形 / 单图超限。

    入口即拒，不落库——比让畸形内容流到 provider 侧再 400 更早、更可诊断。
    """

    code = "INVALID_CONTENT"


class BlobStoreRequiredError(CtxWeftError):
    """携图内容要求宿主注册 EventBlobStore（spec 2026-08-27 双 blob store §7）。

    事件库的口径是**恒不含字节、恒可回读**，没有例外分支——没有 event blob store 就
    无处放字节，只能在入口拒绝。纯文本会话不受影响。
    """

    code = "BLOB_STORE_REQUIRED"


# ── Loop 控制流 ────────────────────────────────────────────────────────────────


class TaskFailedByObserver(CtxWeftError):
    code = TaskErrorCode.BY_OBSERVER


class RunCanceledError(CtxWeftError):
    code = "RUN_CANCELED"


# ── Session 状态 ───────────────────────────────────────────────────────────────


class SessionBusyError(CtxWeftError):
    """Raised when an operation needs an idle session but the session is currently
    running (draining / compacting). The caller should retry once it is idle."""

    code = "SESSION_BUSY"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id!r} is busy (currently running); try again when idle")


class SessionAlreadyExistsError(CtxWeftError):
    """要**新建**的 session 用了一个本进程里已经有主的 session_id。

    一个 session 同一时刻只能有一个 TaskManager（单例，见
    `CtxWeftRuntime._bind_task_manager`）。拿一个已存在的 id 再建一次 session，
    要么再造第二个 TaskManager 与第一个并存，要么往事件库里写第二条
    SESSION_CREATED——两者都不可接受，所以在任何持久化之前直接拒绝。
    续聊已有 session 走 `send_message` / `start_session(resume=True)`。"""

    code = "SESSION_ALREADY_EXISTS"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} already exists in this runtime; "
            f"continue it instead of creating it again"
        )


class UnfinishedTasksError(CtxWeftError):
    """开新一轮（resume_session）被拒：事件里仍有未终结任务。

    弃轮（不恢复、直接开新轮）会把滞留的非终态任务永久遗弃在事件库,之后任何
    recover_agent 全量重建又会把它们复活重跑（僵尸重跑）。调用方应引导用户走
    恢复路径（/resume → recover_agent）续跑或收尾这些任务。"""

    code = "SESSION_HAS_UNFINISHED_TASKS"

    def __init__(self, session_id: str, task_ids: list[str]) -> None:
        self.session_id = session_id
        self.task_ids = list(task_ids)
        shown = ", ".join(self.task_ids[:5]) + ("…" if len(self.task_ids) > 5 else "")
        super().__init__(
            f"Session {session_id!r} still has {len(self.task_ids)} unfinished task(s) "
            f"({shown}); resume the session instead of starting a new turn"
        )
