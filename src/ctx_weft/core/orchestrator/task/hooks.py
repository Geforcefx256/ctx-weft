"""TaskManager 的一次性接线载荷。

**它到底装什么**：`TaskManager` 向上够到 runtime 的那些手——调度器需要、但结构上
不该自己拥有的东西。逐条看，每一个都是一次越界：

    is_current           「我还不还是这个 session 的 owner」——只有 runtime 的
                          _task_managers 映射知道
    cancel_pending_hitl   HITL 归 core.hitl，而 hitl 在 orchestrator **之下**
    cancel_inflight       run 的 CancelToken 存在 runtime._run_tokens；TM 在 run
                          外面，连 run_id 都拿不到
    cancel_finalizer      写 memory——TM 不碰 memory
    threshold_finalizer   同上
    on_session_done       回收 run 令牌 / pause 闩 / scoped provider，全是 runtime 内存
    on_session_idle       同上
    on_task_terminal      清 CapabilityCache 里该 task 的 pin（cache 归 core.capabilities）、
                          给该 task 已终局未消费的 HITL 决定盖章（hitl 在 orchestrator 之下），
                          TM 不认识它，也不该知道「工具面」这回事

改造前它们是 8 个独立 setter（外加一个 `set_session_registry`，实测**从未被读过**，
已随本次一并删除），构造完的 TaskManager 因此是个半成品，接线顺序成了隐含契约。

**收成一个 frozen dataclass、整体替换、不逐字段合并**：这样「装到一半」的中间态在
类型层面就表示不出来。逐字段 setter 恰恰相反，允许任意子集，而任意子集里绝大多数
是错的。

**这不是消除坏味道，是把它集中起来。** 8 个 Optional 回调本身就意味着 TM 有 8 处
越界；真正消除得让 TM 不再需要它们（例如把熔断整条序列移交一个 SessionSupervisor），
那是比本次大得多的一次重构。本模块的价值在于：以后有人想再加一个，得先在这个
dataclass 上加字段，那一刻就会被问一句「这东西为什么不能是 TM 自己的」。

仍是独立方法的三个（它们在不同时刻被多次调用，不属于一次性接线）：
`set_runner`（start / recover 各一次）、`set_session`（三处）、
`set_pause_abandon`（pause 窗口的运行期开关）。
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ctx_weft.core.models.task import Task

__all__ = ["TaskManagerHooks"]


@dataclass(frozen=True)
class TaskManagerHooks:
    """全部字段 None-tolerant：缺注入时对应副作用整体跳过，绝不崩溃。"""

    #: 归属权谓词：本 TM 是否仍是该 session 的当前 owner。None = 不受管（永远视为
    #: current）。被同 session 上更新的 TM 顶替后返回 False → 迟到的收尾变 no-op
    #: （不发 SessionFinished、不触发 on_session_done）。
    is_current: "Callable[[], bool] | None" = None

    #: 「取消该 session 所有未决 pending HITL」。trip 序列第 3 步 best-effort 调用；
    #: HitlCancelled 需全部先于会话终态发出。
    cancel_pending_hitl: "Callable[[], Coroutine[Any, Any, None]] | None" = None

    #: 「对指定在途 task 发协作取消信号」。只发信号不代表任务立即终结——非 root 的
    #: TASK_CANCELED 由 run 结束后的 `apply_run_outcome` 发；root 由 trip 序列先标 FAILED。
    cancel_inflight: "Callable[[str], bool] | None" = None

    #: 「撤销这一轮里**不属于 TM** 的那部分」（两阶段提交的丢弃路径，spec 2026-09-09）。
    #: 由 `discard_round` / `abandon_round` 在关窗**之前**调用：`hitl.release` 把这一轮
    #: 收下的答复退回 pending，并清暂停闩锁。memory 不在其列——这一轮的写入只在快照的
    #: 暂存区里（`TaskManager.stage_memory`），关窗时随快照扔掉。
    #: 与本 dataclass 里其余几条同一理由：hitl 在 orchestrator **之下**，TM 向上够不到；
    #: 而「什么时候撤」只有 TM 知道。best-effort，抛异常只记日志——一次撤销失败不该把
    #: 「用户按了暂停」变成一次 run 崩溃。
    revert_round: "Callable[[str], Coroutine[Any, Any, None]] | None" = None

    #: 「这一轮算数了，把**不属于 TM** 的那部分也落定」——`revert_round` 的对偶。
    #: 参数：(task_id, 暂存的 memory 写入)。由 `commit_round` 在补投缓冲**之后**调用，
    #: 顺序固定：① 把该 task 上待终局的 HITL 答复终局（发 `HitlResolved`）；② 把暂存的
    #: memory 写入按序落盘——**终局事实先于答复进 memory**。
    #: 收在这里而不是散在各提交点，是因为提交点不止一个——act 的首 chunk、gateway 调工具前、
    #: act 离开循环前、prepare 压缩前、`_run_task` 的收尾兜底、消息并进在跑 task 的就地
    #: 提交。best-effort，抛异常只记日志。
    commit_round: "Callable[[str, list[Any]], Coroutine[Any, Any, None]] | None" = None

    #: 「这个 task 停下来等的那个问题，此刻是不是已经有答复了」——run 收尾时的自检判据。
    #: 参数 (task_id, hitl_id)，同步只读。
    #:
    #: 存在的理由是一道结构性的时序缝：park 信号要一路上抛到 `_run_task` 才被 TM 看见，
    #: 而运行槽位要到 `_settle` 才释放（提前释放会让同一个 task 被派发两次）。落在这段
    #: 收尾里的冷应答，其 `resume_task` 会被「它还在跑」挡掉，而那次唤醒没有任何地方
    #: 记得——答复还在，会话却停摆到下一次 `/resume`。
    #:
    #: 修法是把唤醒从「一次性推送」改成「可从状态推导」：收尾末尾自己问一句，答案由
    #: HITL 的内存态给（runtime 侧实现，TM 够不到 hitl）。判据只认**这次 park 所等的
    #: 那个问题**，所以不会退化成无条件重排：下一次 park 等的是另一个 hitl_id。
    human_answer_ready: "Callable[[str, str], bool] | None" = None

    #: 熔断收尾：(root_we_failed_and_started|None, ack_tasks, failures) -> None。
    #: trip 序列第 7 步内联 await（不是后台甩），保证 memory 落盘先于 SESSION_FINISHED
    #: （SSE 关闭）；异常只记日志不阻断终结。
    threshold_finalizer: (
        "Callable[[Task | None, list[Task], list[tuple[str, str]]], "
        "Coroutine[Any, Any, None]] | None"
    ) = None

    #: 统一取消胶囊闭合：(tasks, reason) -> None。三个调用点共用：`cancel_all`、
    #: 熔断清场对已启动的挂起/排队任务、`on_task_finished` 的 CANCELED funnel。
    cancel_finalizer: "Callable[[list[Task], str], Coroutine[Any, Any, None]] | None" = None

    #: session 真正结束时调用。幂等由调用方承担（runtime 侧 `_release_round` 本就幂等）。
#: ⚠ 2026-09-08 起它**不再回收 TaskManager / agent record**——那是显式
#: `forget_session` 的职责。这里只清这一轮的 per-run 控制信号残余。
    on_session_done: "Callable[[], Coroutine[Any, Any, None]] | None" = None

    #: task 落**终态**（FINISHED/FAILED/CANCELED）时调用，参数是 task_id。用于回收挂在
    #: task 上、run 生命周期管不着的运行期状态（当前唯一使用者：清 CapabilityCache 里该
    #: task 的 pin）。**只在终态那条路上触发**：`_settle` 的 PENDING（retry）分支先 return，
    #: 故重试天然不触发——重试保住 pin 正是它该有的语义。同步回调、异常只记日志不阻断收尾。
    #: task 落终态时调用（async）。当前两件事：清该 task 运行期 pin 进来的能力，以及给它
    #: 名下**已终局却再也不会被消费**的 HITL 决定盖 `HitlClosed`——终态 task 不会再跑，
    #: `_inject_resolved_user_turns` 对它本就直接跳过，gateway 也不会再为它求批。
    #: best-effort：回调失败不该拦住任务收尾。
    on_task_terminal: "Callable[[str], Coroutine[Any, Any, None]] | None" = None

    #: session 进入**空闲挂起**（park/suspend 且无其它在跑任务、非终结）时调用。
    #: 区别于 `on_session_done`：那是终结回调；这是「暂停待续接」的信号，供 runtime
    #: 回收按 run 计的控制信号。可多次触发；回调须幂等。
    on_session_idle: "Callable[[], Coroutine[Any, Any, None]] | None" = None
