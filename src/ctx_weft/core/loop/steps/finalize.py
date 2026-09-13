"""FinalizeStep：task 收尾——写 memory + blackboard publish + 更新 task 状态。

miniAgents 对齐版：
- memory 内容 = task.outputs + "\\n\\nProcess Report: " + verdict.act_recap（合并写入）
- 新增 BLACKBOARD_PUBLISH：让父 agent 通过 recall_topic(task.id) 读到子任务结果
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from ctx_weft.core.models.discriminators import TaskErrorCode
from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome, make_event
from ctx_weft.protocols.events import EventType
from ctx_weft.core.media import placeholder_refs
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.utils.content import content_to_text, image_tokens
from ctx_weft.core.utils.clock import as_utc, now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryKind, MemoryScope, MemoryAddress
from ctx_weft.protocols.capability import qualify

logger = logging.getLogger(__name__)

# v2 P3：旧类型清单常量（_OWN_CONV_TYPES/_FINAL_RAW_TYPES）随读侧 kind+role 谓词化删除。

def _dispatch_running_ack(title: str) -> str:
    """派发对 tool 槽的 **running 态**：子任务真正 start 时写，close 时被 `_dispatch_ack` 终态替换。

    与终态同理不含任何子任务产出——此刻也还没有。只说明「在跑」+ 导读下方的内联执行。
    """
    return f"Sub-task '{title}' is running now — its execution follows below."


# 同 agent 派发：派发对 tool 结果（不含任何子任务结果，spec 2026-06-30 §2.5）。
def _dispatch_ack(title: str, outcome: str) -> str:
    """同 agent 派发对的 tool 槽：终态 + 内联导读，**绝不含子任务产出**。

    「不回填真实结果」是结构性的，不是保守：llm_gateway.reorder_tool_results_after_calls 会把
    每条 tool result 强行挪到其 assistant tool_call 之后（OpenAI 兼容端点要求二者相邻，否则
    400），所以本条**必定**渲染在子 body 之前——无论它的 timestamp 是什么。塞进真实产出就成了
    「先结论、后过程」的倒叙，且与内联 body + 嵌套 finish 对重复。

    但本条是在 close 时刻才写的，**已知 outcome**，故可以带两样位置正确的前向信息：
      ① 终态——让 delegate_task 这个调用真正解析出结果，而不是永远停在「started」；
      ② 导读——告诉读者子任务的完整执行内联在下方、以自己的 finish_task 收尾。导读同时解释了
         紧随其后的 user 回合（子任务的 task_prompt，非用户发言），否则易被读成用户插话。
    刻意保持指针级长度：每个派发都会复读一份。
    """
    verdict = "FAILED" if outcome == "fail" else "completed"
    return (
        f"Sub-task '{title}' ran here — outcome: {verdict}. Its execution is inlined "
        f"below, ending with its own {qualify('control:finish_task')} and the final reply "
        "it delivered."
    )

# 派发框的叙事工具名（仅出现在重建历史的 tool_calls 里，非可调用能力）。
START_TASK_NAME = qualify("control:start_task")


# 归一可能 naive 的 datetime 为 aware(UTC)——事件重放 / DB 反序列化可能丢 tz，
# 比较前统一，避免 naive 与 aware 直接比较报 TypeError。统一实现见 utils.as_utc。
_as_utc = as_utc


def _is_frame_of(record, task) -> bool:
    """这条 agent 层回合是不是**本子任务**的派发框 / ack。

    首选 `child_task_id`：铸框时写死的子任务 id（v2「新机制 = 新 metadata 约定」）。它让
    「这次派发对应哪个子任务」成为 memory 里的一等事实，close 时不再依赖任何进程内状态。
    回退按 `tool_calls[].id / tool_call_id == task.origin_tool_call_id` 匹配，仅为存量数据
    ——那是纯瞬态字段（不入 TASK_CREATED payload / TaskView），重启重建后为 None，届时只有
    child_task_id 认得出框（子任务 close 后父只剩「派发框 + 停在 running 的 ack」即由此来）。
    """
    md = record.metadata or {}
    if md.get("child_task_id") == task.id:
        return True
    tcid = task.origin_tool_call_id
    if not tcid:
        return False
    if md.get("tool_call_id") == tcid:
        return True
    return any(tc.get("id") == tcid for tc in (md.get("tool_calls") or []))


def _frame_tool_call_id(frame, task) -> str | None:
    """框自己携带的配对 id：优先取与 task.origin_tool_call_id 相符的那条，否则取首条。

    取首条是安全的：框由 core 合成，`_ensure_dispatch_frame` 每个子任务铸一条、恒单调用；
    存量 legacy 记录经 `_legacy_dispatch` 归一后同样单调用。actor 的真实回合（可能并行多调用）
    落在 task 层，不进本视图。
    """
    tcs = frame.metadata.get("tool_calls") or []
    tcid = task.origin_tool_call_id
    if tcid and any(tc.get("id") == tcid for tc in tcs):
        return tcid
    return tcs[0].get("id") if tcs else None


async def _find_dispatch_frame(memory, parent_scope, task, provider_ctx):
    """在 parent scope 只查已存在的派发框（属于本子任务的 assistant 回合），不铸新框。
    找到 → 返回 `(框的时间戳（归一 aware UTC）, 配对 tool_call_id)`；没有 → None。

    供 `_ensure_dispatch_frame`（find+create）与 `synthesize_cancel_closure`（find-only，
    born-cancel 未铸框时整体跳过、不补铸）共用。
    """
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope
    existing = await memory.load_view(
        MemoryAddress(session_id=parent_scope.session_id, agent_id=parent_scope.agent_id),
        MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    frame = next(
        (r for r in existing if r.role == "assistant" and _is_frame_of(r, task)),
        None,
    )
    if frame is None:
        return None
    # 归一：框可能来自事件重放 / DB 反序列化而丢 tz（naive），直接返回会让调用方拿它与
    # aware 时间比较时炸 TypeError。
    return _as_utc(frame.timestamp), _frame_tool_call_id(frame, task)


async def _ensure_dispatch_frame(memory, parent_scope, task, provider_ctx):
    """在 parent scope 铸一条属于本子任务的 assistant 派发框，返回
    `(时间戳, 配对 tool_call_id)`——时间戳 == 配对 tool result 应锚定的时刻 = task
    真正开始执行的时刻（started_at）。

    **框与 result 同锚 started_at**：二者共用同一时间戳 → 按 (timestamp, seq_no) 排序时严格相邻
    （框先写 seq 小、result 后写 seq 大），且落在「任务开始执行」这条时间线上（而非派发时刻）。
    started_at 晚于 actor 派发那一轮、早于子 body（body 由 driver 在启动后才 ingest USER_PROMPT），
    故「派发框 → Task started/result → 子 body → finish 对」顺序天然成立、胶囊连续。回退 created_at
    （历史/无 started_at）再回退 now，统一归一为 aware(UTC)。

    delegate_task 与 delegate_plan 子统一走此铸框路径（gateway 不再为 delegate_task eager 写框——
    eager 框只能带派发时刻，无法落在 started_at 时间线上，见 capability_gateway._record_invocation）。
    框的 tool name 取 task.origin_tool_name（保真）：delegate_task 子 = 真名 control__delegate_task
    （actor 确实调过）；delegate_plan 子 = None → 回退 START_TASK_NAME 叙事名（无 per-child 真实调用）。
    幂等：若本子任务的框已存在（终态 finalize 单入本不会重入，此为防御），直接返回、不重复铸。
    origin_task_id 留父（delegate_task 的父 = 派发 task；delegate_plan 子 = 留 plan task）；
    **child_task_id 记子**——派发框与 ack 里唯一指向子任务的字段，close 时靠它认框。

    配对 id 的取值序：框已存在 → 用框自己的；否则 task.origin_tool_call_id；再否则现生成一个。
    生成是安全的：actor 真实的 delegate_task 调用**不入 memory**（gateway 对 DISPATCH 工具不写
    对话，见 capability_gateway._record_invocation），这里的框和 ack 整对都是 core 合成的叙事，
    id 只用于本对内部配对，从不与 actor 侧的 id 对账（delegate_plan 子任务本就是现生成的，
    见 control_capability）。有了它，重启丢掉 origin_tool_call_id 的子任务也能正常闭合。

    find + create：find 部分委托 `_find_dispatch_frame`（与 cancel closure 的 find-only 用法共用）。
    """
    found = await _find_dispatch_frame(memory, parent_scope, task, provider_ctx)
    if found is not None:
        # 幂等：框已铸（常态——ensure_dispatch_frame_at_start 在子 start 时就铸好了）。
        # **返回框自己的 ts，而不是按当下 started_at 重算的 ts**：TaskManager 每次派发都刷新
        # task.started_at，故 retry 过的子任务在 close 时算出的是**末次**派发时刻，而框停在首次。
        # 用重算值会把终态 ack 写到框之外（漂进子 body 中间），断掉「框与 result 同锚、严格相邻」
        # 的不变量——渲染虽有 reorder_tool_results_after_calls 兜底，但那是把正确性转嫁给 legalize。
        return found
    # 框与 result 的公共锚点：task 真正启动执行的时刻。归一为 aware(UTC)：started_at/created_at
    # 可能来自事件重放而为 naive（历史无 started_at 时回退 created_at 再回退 now）。
    ts = _as_utc(task.started_at or task.created_at or now_utc())
    tool_call_id = task.origin_tool_call_id or generate_id("tcall")
    await memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=parent_scope,
            content="", timestamp=ts, role="assistant",
            metadata={"origin_task_id": task.parent_task_id,
                      "parent_task_id": task.parent_task_id,
                      "child_task_id": task.id,
                      "tool_calls": [{"id": tool_call_id,
                                      "name": task.origin_tool_name or START_TASK_NAME,
                                      "input": {"title": task.title,
                                                "description": task.description or ""}}]},
        ),
        provider_ctx,
    )
    return ts, tool_call_id


def _parent_scope_of(state, task) -> MemoryAddress:
    """派发方（parent task + creator agent）的 scope——派发框/result 的落点。"""
    return MemoryAddress(
        session_id=state.scope.session_id,
        task_id=task.parent_task_id,
        agent_id=task.creator_agent_id,
    )


async def _put_dispatch_result(memory, parent_scope, task, content: str, ts, provider_ctx,
                               *, replace: bool, tool_call_id: str) -> None:
    """写派发对的 tool 槽（tool_call_id 由 `_ensure_dispatch_frame` / `_find_dispatch_frame`
    给出，与框配对）。

    replace=False（start）：已有配对 result 时 no-op —— retry/resume 会重跑 driver.run，
      不能每轮都 churn 一遍、更不能改写原锚点。
    replace=True（close）：先 supersede 旧的 running ack 再写终态。**必须替换而非新增**：
      同一 tool_call_id 若有两条 active result，reorder_tool_results_after_calls 会把两条
      都排到框之后，于是「在跑」和「已完成」并列出现。

    旧 ack 的识别用 `_is_frame_of`（child_task_id 优先、存量回退 tool_call_id）而非只比
    tool_call_id：重启后拿到的 id 可能是新生成的，只比 id 会漏掉 start 时那条 running ack，
    于是两条 active result 并存。新写的 ack 同样带 child_task_id，start/close/取消三个时刻
    共用本函数，字段一路带下去不会在替换时掉。
    """
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope
    recs = await memory.load_view(
        MemoryAddress(session_id=parent_scope.session_id, agent_id=parent_scope.agent_id),
        MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    stale = [r.id for r in recs
             if r.role == "tool"
             and (_is_frame_of(r, task) or r.metadata.get("tool_call_id") == tool_call_id)]
    if stale and not replace:
        return
    # v2 P3d：旧 ack 遗忘 + 终态写入一次原子 fold（stale 空 = 纯写入）
    await memory.fold(stale, [
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=parent_scope,
            content=content, timestamp=ts, role="tool",
            metadata={"origin_task_id": task.parent_task_id,
                      "child_task_id": task.id,
                      "tool_call_id": tool_call_id},
        ),
    ], provider_ctx)


async def ensure_dispatch_frame_at_start(state, ctx) -> None:
    """子任务**真正开始执行**时铸派发框 + running ack（由 driver.run 调用，紧邻 _persist_user_prompt）。

    **为什么在 start、而不是派发时刻**：gateway 的 `_record_invocation` 早于工具执行，那时子任务
    的生死未定——pause 弃子（`TaskManager.abandon_pending`）、`_pause_abandon` 的 staged 静默丢弃、
    `delegate_task` 自身抛异常，都会让子任务胎死腹中，而框已落库、成为永远 pending 的孤儿。其中
    staged 丢弃连 task id 都不返回、也不发 TASK_CANCELED，根本无从清理。改在 start 铸后：
      - 没跑起来的子任务从不写框 → 零清理，不需要任何取消路径配合（弃子后父会 resume，
        `abandon_pending` 不置 session CANCELED，所以「说在跑却什么都没有」的谎会被读到）；
      - `started_at` 此刻已存在 → 框直接生在正确锚点，无需 close 时重定位。

    **为什么不能等到 close**：执行期间唯一的读者是子任务自己（同 agent 时父挂起、不装配）。
    有了框 + running ack，子任务才看得见自己的来历；否则它只看到一条凭空出现的 user 回合
    （自己的 task_prompt），容易读成用户插话。跨 agent 子任务的 agent scope 不同、召回不到本框，
    与其 lean 隔离形态一致。

    幂等：框由 `_ensure_dispatch_frame` 查重，result 由 `_put_dispatch_result(replace=False)` 查重。

    准入判据只看 `parent_task_id`（持久化字段）：origin_tool_call_id 是瞬态的，重启重建后为
    None，若拿它当条件，重启后续跑的子任务连框都不铸，父永远看不到这次派发。
    """
    task = state.task
    if not task.parent_task_id:
        return  # root task：无派发方，没有框可铸
    parent_scope = _parent_scope_of(state, task)
    ts, tool_call_id = await _ensure_dispatch_frame(
        ctx.memory, parent_scope, task, ctx.provider_ctx)
    await _put_dispatch_result(
        ctx.memory, parent_scope, task, _dispatch_running_ack(task.title), ts, ctx.provider_ctx,
        replace=False, tool_call_id=tool_call_id,
    )


def _finish_report_prefix(title: str, outcome: str) -> str:
    """finish 对 tool 槽的前缀：`[task: <title>] ` 标明归属（+ fail 标记）。

    归属标记的必要性：finish 对的 assistant 槽是**无参**收尾标记（`finish_task{}`，反转契约），
    自身不带任何归属信息。派发侧不存在这个问题——框带 `input:{title, description}` 自描述；
    收尾侧此前没有对称物，于是同 agent 嵌套（孙→子相继 close）时，父重建出的对话里会出现两组
    完全同形的 `[assistant finish_task{}][tool …]`，只能靠正文猜是哪个 task 收的尾。

    标记落 tool 槽而非 `input`：`input` 受 finish_task 的 schema 约束，而
    ControlCapabilityProvider._handle 会过滤 schema 未声明的 key（见 control_capability.py），
    塞进去只会被静默丢弃；且历史里出现未声明参数会诱导模型照此形状调用。tool 槽是自由文本，
    与既有 `[outcome=fail]` 前缀同一惯例。归属在前、outcome 标记在后；title 为空则不产空标记。

    `outcome == "cancelled"`（统一取消胶囊闭合，见 synthesize_cancel_closure）→ `[outcome=cancelled]`，
    与既有 `[outcome=fail]` 同一惯例，标明这条 finish 对是取消收尾而非正常/失败终态。
    """
    parts: list[str] = []
    title = (title or "").strip()
    if title:
        parts.append(f"[task: {title}]")
    if outcome == "fail":
        parts.append("[outcome=fail]")
    elif outcome == "cancelled":
        parts.append("[outcome=cancelled]")
    return "".join(f"{p} " for p in parts)


# 最终回复锚点（close 折末段 raw 时补位）：反转契约下答复正文是收尾回合的普通消息，
# 随末段 raw 一起被删；锚点把它留在 task 层胶囊里。提示词用 assistant 第一人称，与
# 「以上为系统压缩摘要」的尾注互不交叉（各自只描述自己那条消息的正文）。
FINAL_REPLY_NOTE = (
    "[Final reply for task '{title}' — the answer I delivered on finishing it, "
    "verbatim; not a summary.]"
)
FINAL_REPLY_NOTE_UNTITLED = (
    "[Final reply — the answer I delivered on finishing this task, verbatim; not a summary.]"
)
# 收束尾注：锚点是**真回复**，比段摘要更容易被读成「我刚刚就是这么答的」——紧随其后的往往是
# 新的用户消息或另一个 task，没有收束就会照抄/续写它。作用与段摘要的 ASSISTANT_SUMMARY_NOTE
# 对称，措辞同为方括号系统注解。
FINAL_REPLY_CLOSING_NOTE = (
    "[End of that final reply. It was delivered to the user when the task closed — do not repeat "
    "it, do not imitate its form, and do not treat it as the answer to whatever is being asked "
    "now.]"
)


def _finish_tool_text(task_summary: str, act_recap: str, outcome: str) -> str:
    """finish 对 tool 槽内容 = task_summary（process report）。R2 兜底：空则退 act_recap，
    再空给占位。**不掺 outputs**——最终输出由 task 层的最终回复锚点承载
    （见 _supersede_final_raw_segment），tool 槽只讲过程、不重复它。
    绝不返回空串（避免空 tool 回合 / 400）。"""
    for cand in (task_summary, act_recap):
        if cand and cand.strip():
            return cand
    return ("(no final output)" if outcome == "fail"
            else "(nothing further to report for this segment)")


# recap 槽的收束尾注：它是 agent 层普通 assistant 回合，拿不到段摘要那条尾注
# （_history.annotate_assistant_summary 只贴 TASK_COMPACT_SUMMARY），而形态上同样像
# 「我上一轮就是这么答的」，同样会被模仿。与 FINAL_REPLY_CLOSING_NOTE 一并夹住胶囊：
# 前者说「这段是过程」，后者说「那段答复已经交付过了」。
PROCESS_RECAP_NOTE = (
    "[The above is a system-written recap of how this task was carried out, kept for context — "
    "it is not your reply to the user. Do not imitate its form when you answer.]"
)

_RECAP_PLACEHOLDER = "(no process recap for this segment)"


def _recap_block(act_recap: str) -> str:
    """recap 槽正文 = 过程复述 + 收束尾注（正文为空时用占位，注解照贴）。"""
    body = (act_recap or "").strip() or _RECAP_PLACEHOLDER
    return f"{body}\n\n{PROCESS_RECAP_NOTE}"


def _final_reply_block(title: str, reply: str) -> str:
    """锚点正文 = 前置提示 + 答复原文 + 收束尾注。"""
    note = (FINAL_REPLY_NOTE.format(title=title.strip()) if (title or "").strip()
            else FINAL_REPLY_NOTE_UNTITLED)
    return f"{note}\n\n{reply.strip()}\n\n{FINAL_REPLY_CLOSING_NOTE}"


def build_finish_slots(*, scope, task_id: str, parent_task_id, title: str, outcome: str,
                       act_recap: str, task_summary: str, final_reply: str,
                       base, tool_call_id: str) -> list[MemoryEvent]:
    """finish 对的槽位事件。close 合成与 bg 事后重写共用，保证两处形态永远一致。

    `final_reply` 非空 → **三槽**：

        assistant  act_recap                      过程复述，不挂 tool_calls
        assistant  提示 + 答复 + 收束尾注           锚点，finish_task{} 挂这条（metadata.final_reply）
        tool       process report                 与锚点配对

    finish_task 挂在答复那条、而不是 recap 那条：重建出的历史因此示范了 finish_task 的真实
    用法——答复正文与收尾调用同一轮（见 control_capability.finish_task 的说明）。挂在 recap
    上等于反过来教模型「收尾时正文写过程复述」。

    `final_reply` 空 → **两槽**（旧形态，finish_task 挂 recap）。用于末段 raw 没被折的场景：
    short leaf 原文即胶囊、observer 护栏兜底导致 outputs 为空、以及占位 close（raw 还在，
    等 bg 真报告落地时再由 _replace_finish_report 升三槽）。

    时间戳按槽位递增 1µs：顺序由写入点定死，不依赖后续记录的时刻。
    """
    md = {"origin_task_id": task_id, "parent_task_id": parent_task_id}
    call = [{"id": tool_call_id, "name": qualify("control:finish_task"), "input": {}}]
    report = f"{_finish_report_prefix(title, outcome)}"              f"{_finish_tool_text(task_summary, act_recap, outcome)}"
    reply = (final_reply or "").strip()

    def turn(content, ts, role, extra):
        # act_recap / task_summary / final_reply 都可能是折叠产物，逐字带着 L0.5 占位向前走
        # （见 `placeholder_refs` docstring）。这三槽既经 `_replace_finish_report` 的 fold
        # 写入（旧记录被 supersede），也经 `_synthesize_dispatch_pair` 的裸 ingest 写入
        # （无 supersede，但 mark 判据一视同仁、同样只看结构化字段）——两条路都要声明，
        # 故放在两个调用方共用的这一层，而不是分别在各自的调用点补。
        return MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=scope,
            content=content, timestamp=ts, role=role, metadata={**md, **extra},
            blob_refs=placeholder_refs(content),
        )

    if reply:
        return [
            turn(_recap_block(act_recap), base, "assistant", {}),
            turn(_final_reply_block(title, reply), base + timedelta(microseconds=1),
                 "assistant", {"tool_calls": call, "final_reply": True}),
            turn(report, base + timedelta(microseconds=2), "tool", {"tool_call_id": tool_call_id}),
        ]
    return [
        turn(_recap_block(act_recap), base, "assistant", {"tool_calls": call}),
        turn(report, base, "tool", {"tool_call_id": tool_call_id}),
    ]


def _descendant_task_ids(root_id: str, task_manager) -> set[str]:
    """BFS over children_of → 该 task 名下所有后代 task_id（不含自身）。"""
    if task_manager is None:
        return set()
    out: set[str] = set()
    stack = [root_id]
    while stack:
        cur = stack.pop()
        for cid in task_manager.children_of(cur):
            if cid not in out:
                out.add(cid)
                stack.append(cid)
    return out


async def _is_short_leaf(memory, scope, task, loop_config, ctx, has_descendants: bool) -> bool:
    """叶子(无后代) 且 对话 token ≤ threshold 且 LLM_RESPONSE 轮次 ≤ turn_cap → short。"""
    if has_descendants:
        return False  # 非叶（委派过子任务）永不 short
    from ctx_weft.protocols import MemoryKind, MemoryScope
    # v2 P3a：全 task 层视图（对话+摘要+audit = 旧 _OWN_CONV_TYPES 五类型）一次取回，
    # assistant 轮次与 token 估算共用。
    records = await memory.load_view(
        scope, MemoryScope.TASK, ctx.provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY, MemoryKind.TOOL_AUDIT])
    n_assistant = sum(
        1 for r in records
        if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "assistant"
    )
    if n_assistant > loop_config.short_task_turn_cap:
        return False
    text = " ".join(
        content_to_text(r.content) if not isinstance(r.content, str) else r.content
        for r in records
    )
    # 同 background_observe.is_short_segment：保留 join 后数一次的文本口径，图片另计。
    images = sum(image_tokens(r.content) for r in records)
    return (ctx.llm.tokenizer.count(text) + images) <= loop_config.short_task_token_threshold


async def finalize_task_memory(memory, state, task, mem_content: str, outcome: str, ctx,
                               *, act_recap: str, task_summary: str,
                               has_llm_summary: bool = True) -> list:
    """finish 时调用：算 short / descendants 后委派 _close_one。返回事件列表。

    task-resident（spec 2026-06-28）：short 不再 gate 合成/supersede——每个结束 task 都写
    finish 对、body 留 task 层。`short` 仅算出后透传给 _close_one（Task 2 用于 body raw-vs-压末段）。

    has_llm_summary（spec 2026-07-20 延迟折叠）：close 时刻 finish 对内容是否已是 LLM 真摘要
    （= verdict.reported）。False = 规则 observe 占位 → 末段 raw 不在 close 时删，推迟到
    bg 替换真摘要落地后补删（默认 True 保守 = 旧行为，close 即折）。
    """
    descendants = _descendant_task_ids(task.id, ctx.task_manager)
    short = await _is_short_leaf(
        memory, state.scope, task, state.agent.loop_config, ctx, bool(descendants),
    )
    return await _close_one(
        memory, state, task, mem_content, outcome, ctx,
        short=short, act_recap=act_recap, task_summary=task_summary,
        has_llm_summary=has_llm_summary,
    )


async def _supersede_final_raw_segment(memory, scope, provider_ctx) -> None:
    """长任务 close：supersede task 层**末段** raw（active LLM_RESPONSE/TOOL_INVOCATION/
    TOOL_RESULT），保留 USER_PROMPT + TASK_COMPACT_SUMMARY 锚点（spec 2026-06-28 §3.2）。

    段作用域（2026-07-21）：末段 = 最后一条 active USER_PROMPT 之后。此前假设「active raw
    即末段」（中间段在各自边界已折），但短段免折（background_observe.is_short_segment）会让
    前段 raw 以 active 状态残留——它们无胶囊代表，删了即信息丢失（其 UP 失去回答位），故保留
    （「短 → 原文成胶囊」）。无 UP（防御）→ 全删（旧行为）。

    调用时机（spec 2026-07-20 修订的不变量：末段 raw 与「真实 Process Report」至少存其一）：
    finish 对已承载 LLM 真摘要 → close 时同步删；占位 finish 对 → 推迟到 bg 替换真摘要后
    补删（background_observe close 回调）。**不另产新 TASK_COMPACT_SUMMARY**（避免与 finish
    对重复）。幂等：raw 已删则 no-op。

"""
    from ctx_weft.protocols import MemoryKind, MemoryScope
    # v2 P3a：升序视图（对话 + audit，SUMMARY 锚点天然不在），段界 = 末条 role=user 回合。
    view = await memory.load_view(
        scope, MemoryScope.TASK, provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.TOOL_AUDIT])
    ids: list[str] = []
    for r in view:
        if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "user":
            ids = []  # 新段界：只删末段
            continue
        ids.append(r.id)
    if ids:
        await memory.fold(ids, [], provider_ctx)  # 纯遗忘（v2 P3d）；答复由 finish 对的锚点槽承载


async def _close_one(memory, state, task, mem_content: str, outcome: str, ctx,
                     *, short: bool, act_recap: str, task_summary: str,
                     has_llm_summary: bool = True) -> list:
    """close 主体（task-resident，spec 2026-06-28）：bubble / 写 finish 对（不镜像 body）。

    每个结束 task 无条件写 finish 对、body 留 task 层（不 GC 子树）。长任务额外 supersede 末 raw
    段（短任务留全 raw）——`short` 决定 body raw-vs-压末段（spec §3.2）。返回事件列表。

    延迟折叠（spec 2026-07-20）：`has_llm_summary=False`（规则 observe 占位）时末段 raw 不在
    close 时删，`raw_fold_scope` 随 finish 对合成登记下去，bg 替换真摘要后补删；bg 失败则
    raw 永久保留（降级 = 保 raw，信息不丢）。
    """
    # Terminal finalize is single-entry by construction (retry is non-terminal; restore reschedules
    # only non-terminal tasks; re-dispatch uses a new task id), so the residue/bubble writes here
    # need no idempotency guard. [2026-06-23]
    events: list[Any] = []
    same_agent = task.creator_agent_id == task.assigned_agent_id
    cross_agent = bool(task.parent_task_id) and not same_agent
    is_own_root = (task.parent_task_id is None) or cross_agent
    # raw 所在层恒为本 task 的 task scope（≠ 嵌套 finish 对的 parent scope）
    raw_fold_scope = state.scope if (not short and not has_llm_summary) else None
    # 末段 raw 会被折 → 答复须由 finish 对的锚点槽承载；short leaf 不折（原文即胶囊）→ 不塞。
    final_reply = "" if short else _output_text(task.outputs)

    # 1) bubble 到 parent scope（dispatch marker 所在 scope）
    # 判据只看 parent_task_id + mem_content：origin_tool_call_id 是瞬态字段，重启重建后为 None，
    # 曾经把它写进条件 → 跨重启收尾的子任务整段 bubble 被跳过（父的 ack 停在 running、finish 对
    # 不合成 = 胶囊丢失），而末段 raw 照删。配对 id 现由框自己给出，不再需要它当门槛。
    if task.parent_task_id and mem_content:
        parent_scope = _parent_scope_of(state, task)
        # 框通常已由 ensure_dispatch_frame_at_start（子 start 时）铸好，_ensure_dispatch_frame 在此
        # 幂等命中；仅对没走过 start 钩子的路径（旧数据 / start 与 close 之间崩溃）才真正补铸。
        # tool 槽用 replace=True：start 时写的是 running 态，此处须**替换**为终态而非新增一条。
        if cross_agent:
            # 跨 agent（spec 2026-06-28 §2.3）：dispatch result 写成 agent 层普通 conversation turn
            # （tool 回合），与 start_task / delegate 框靠 tool_call_id 配对、时间戳对齐保证相邻。
            report_prefix = "[outcome=fail] " if outcome == "fail" else ""
            frame_ts, frame_tcid = await _ensure_dispatch_frame(
                memory, parent_scope, task, ctx.provider_ctx)
            await _put_dispatch_result(
                memory, parent_scope, task, f"{report_prefix}{mem_content}", frame_ts,
                ctx.provider_ctx, replace=True, tool_call_id=frame_tcid,
            )
            events.append(make_event(
                state, EventType.MEMORY_INGESTED,
                payload={"memory_event_type": MemoryEventType.AGENT_CONVERSATION_TURN.value,
                         "source": "dispatch_result", "content_length": len(mem_content)},
            ))
        elif same_agent:
            # 同 agent（spec 2026-06-30 §2.5）：配对 tool result 换终态文案，时间戳对齐框 → 严格
            # 相邻、排在子 body 之前。子真实产出由内联胶囊 body + 嵌套 finish 对承载。
            frame_ts, frame_tcid = await _ensure_dispatch_frame(
                memory, parent_scope, task, ctx.provider_ctx)
            await _put_dispatch_result(
                memory, parent_scope, task, _dispatch_ack(task.title, outcome), frame_ts,
                ctx.provider_ctx, replace=True, tool_call_id=frame_tcid,
            )
            # 嵌套合成子自己的 finish 对（写进共享 agent scope，@close 时刻）
            await _synthesize_dispatch_pair(
                memory, parent_scope, task, act_recap, task_summary, outcome, ctx.provider_ctx,
                raw_fold_scope=raw_fold_scope, final_reply=final_reply)

    # 2) 自身 finish 对：own root（session 根或跨 agent 根）close 时无条件在 own scope 合成
    if is_own_root and mem_content:
        await _synthesize_dispatch_pair(memory, state.scope, task, act_recap, task_summary, outcome,
                                        ctx.provider_ctx, raw_fold_scope=raw_fold_scope,
                                        final_reply=final_reply)
        events.append(make_event(
            state, EventType.MEMORY_INGESTED,
            payload={"memory_event_type": MemoryEventType.AGENT_CONVERSATION_TURN.value,
                     "source": "root_finish_pair", "content_length": len(mem_content)},
        ))

    # task-resident（spec 2026-06-28 §3.2）：body 留 task 层、不 GC 子树。
    # 长任务 supersede 末 raw 段（保留 USER_PROMPT/TASK_COMPACT_SUMMARY 锚点）；短任务留全 raw。
    # 仅当 finish 对已承载 LLM 真摘要时同步删；占位（has_llm_summary=False）由 raw_fold_scope
    # 走延迟折叠——slot 命中在 _synthesize_dispatch_pair 内已补删，登记路径等 bg 替换后补删。
    if not short and has_llm_summary:
        await _supersede_final_raw_segment(memory, state.scope, ctx.provider_ctx)
    return events


async def synthesize_cancel_closure(memory, session_id: str, task, provider_ctx, reason_text: str) -> None:
    """统一取消胶囊闭合（Task 14）：所有 CANCELED 终态的公共收尾——父 ack 终态化 +
    `[outcome=cancelled]` finish 对。镜像 `_close_one` 的分支，但没有 LoopState，用
    `session_id` + task 字段直接推导 scope。

    调用点（TaskManager → runtime 侧 `_finalize_cancel_memory`）：
      - `cancel_all` 清队（reason=`CancelReason.USER_CANCEL`）；
      - 熔断清场对已启动的挂起/排队任务（reason=`CancelReason.FAILURE_THRESHOLD`）；
      - `on_task_finished(CANCELED)` funnel：在途协作取消终态坐实后（reason=task.error or 通用文案）。

    分支（同 `_close_one`）：
      ① task 有 parent_task_id → 父 scope 的派发对 tool 槽终态化替换为
         取消文案。**find-only**：找不到框（born-cancel，子任务从未真正 start 过、从未铸框）→
         整体跳过、不补铸（`_find_dispatch_frame`，不用 `_ensure_dispatch_frame`）。
      ② 同 agent（`creator_agent_id == assigned_agent_id`）→ 嵌套合成子自己的 finish 对，写进
         ①的父 scope（同 agent 共享 scope）。
      ③ 跨 agent 子任务 / root（`parent_task_id is None`）→ 在自己的 agent scope 合成 finish 对：
         `MemoryAddress(session_id, task.id, task.assigned_agent_id or task.creator_agent_id)`。
    finish 对 `register_bg=False`：这不是正常 close 流程产生的，没有对应的后台 observe 会来替换它，
    登记只会累积永不消费的状态（同熔断收尾路径的既有惯例）。
    """
    same_agent = task.creator_agent_id == task.assigned_agent_id
    cross_agent = bool(task.parent_task_id) and not same_agent
    is_own_root = (task.parent_task_id is None) or cross_agent

    act_recap = "Task was cancelled before completion."
    task_summary = (
        f"Cancelled ({reason_text}) — no final output was produced; "
        f"the partial execution above is all that ran."
    )

    if task.parent_task_id:
        parent_scope = MemoryAddress(
            session_id=session_id, task_id=task.parent_task_id, agent_id=task.creator_agent_id,
        )
        found = await _find_dispatch_frame(memory, parent_scope, task, provider_ctx)
        if found is None:
            # born-cancel：子任务从未真正 start（未走 ensure_dispatch_frame_at_start）、无框可闭——
            # 整体跳过，不补铸空框（补铸出的框只会带取消时刻，与「派发框=started_at」的不变量矛盾）。
            return
        frame_ts, frame_tcid = found
        ack_text = (
            f"Sub-task '{task.title}' was cancelled before completion ({reason_text}); "
            f"its partial execution below is incomplete."
        )
        await _put_dispatch_result(
            memory, parent_scope, task, ack_text, frame_ts, provider_ctx, replace=True,
            tool_call_id=frame_tcid,
        )
        if same_agent:
            await _synthesize_dispatch_pair(
                memory, parent_scope, task, act_recap, task_summary, "cancelled", provider_ctx,
                register_bg=False,
            )
            return

    if is_own_root:
        own_scope = MemoryAddress(
            session_id=session_id, task_id=task.id,
            agent_id=task.assigned_agent_id or task.creator_agent_id,
        )
        await _synthesize_dispatch_pair(
            memory, own_scope, task, act_recap, task_summary, "cancelled", provider_ctx,
            register_bg=False,
        )


async def _synthesize_dispatch_pair(memory, scope, task, act_recap: str, task_summary: str,
                                    outcome: str, provider_ctx, *, register_bg: bool = True,
                                    raw_fold_scope=None, final_reply: str = "") -> None:
    """close 合成 agent 层 finish 对（spec 2026-06-30 两段化）：
    assistant{content=act_recap + finish_task 调用} / tool{content=task_summary 综合总结}。
    own-root：占位先写，bg close observe 产新两段后经 _replace_finish_report 替换（A1）。

    register_bg=False（熔断收尾等 runtime 侧一次性合成路径）：跳过 pop_close_report /
    register_close_synth / _replace_finish_report 整段 bg-observe 联动——这条 finish 对不是
    正常 close 流程产生的、没有对应的后台 observe 会来替换它，登记只会累积永不消费的状态。

    raw_fold_scope（spec 2026-07-20 延迟折叠）：非 None = close 时占位、末段 raw 尚未删，
    此 scope 即 raw 所在 task 层。slot 命中（bg 真摘要已到）→ 替换后立即补删；否则随
    register_close_synth 登记，bg 替换成功后补删。register_bg=False 路径忽略（取消/熔断
    收尾从不折 raw）。
    """
    from ctx_weft.core.loop.steps.background_observe import (
        pop_close_report, register_close_synth, _replace_finish_report,
    )
    base = now_utc()
    tool_call_id = generate_id("tcall")
    # 占位 close（raw_fold_scope 非 None = 末段 raw 还没折）先写两槽：答复此刻仍在 raw 里，
    # 提前塞锚点会让它出现两遍；等 bg 真报告落地、raw 真被折时由 _replace_finish_report 升三槽。
    embed_reply = "" if raw_fold_scope is not None else final_reply
    # 反转契约（spec 2026-07-01）：finish_task 退化为无参收尾标记（答复不再塞进 input.result）。
    # 槽位形态见 build_finish_slots：有答复 → recap / 答复+finish_task / 报告 三槽，
    # 无答复 → recap+finish_task / 报告 两槽。
    for event in build_finish_slots(
        scope=scope, task_id=task.id, parent_task_id=task.parent_task_id,
        title=task.title or "", outcome=outcome, act_recap=act_recap,
        task_summary=task_summary, final_reply=embed_reply, base=base,
        tool_call_id=tool_call_id,
    ):
        await memory.ingest(event, provider_ctx)

    if register_bg:
        bg = pop_close_report(task.id)
        if bg is not None:
            bg_recap, bg_summary = bg
            await _replace_finish_report(memory, provider_ctx, scope, task.id, tool_call_id,
                                         bg_recap, bg_summary, outcome, task.title or "",
                                         final_reply=(final_reply if raw_fold_scope else ""))
            if raw_fold_scope is not None:
                # 真摘要已落地（slot 命中替换完成）→ 立即补删末段 raw（延迟折叠的即时分支）
                await _supersede_final_raw_segment(memory, raw_fold_scope, provider_ctx)
        else:
            register_close_synth(task.id, tool_call_id, scope, outcome, raw_fold_scope)


async def _delivery_acceptance_gate(state, ctx, task, mode: str) -> "dict | None":
    """spec: delivery-acceptance——候选-提交分离边界的执行体。

    顺序为定案：①首次启用初始化（缺 effective_input_turn_id 时经 memory 重算并落
    TASK_INPUT_ADVANCED）→ ②计算不可用检查器集合（恢复后版本缺失）→ ③跑一次验收
    （必需先结构后领域 + 提示性只记录）→ ④候选引用（小直存/超限外存）→ ⑤发
    TASK_ACCEPTANCE_CHECKED → ⑥仅 required 模式产生决策（retry / failed / None）。
    shadow 模式执行①-⑤但永远返回 None（行为矩阵：记录不改变交付）。
    """
    from ctx_weft.core.acceptance.support import (
        candidate_of, compose_gap_hint, findings_equal, recompute_effective_turn,
        run_acceptance_pass,
    )
    from ctx_weft.core.acceptance.protocol import InvalidAcceptanceSpec

    # ① 首次启用初始化（与恢复对账同一原语：从 memory 重算并幂等落盘）
    if task.effective_input_turn_id is None:
        recalced = await recompute_effective_turn(ctx.memory, ctx.provider_ctx, state.scope)
        if recalced is not None:
            task.effective_input_turn_id = recalced
            await ctx.event_bus.emit(make_event(
                state, EventType.TASK_INPUT_ADVANCED, payload={"turn_record_id": recalced}))

    # ② 恢复中版本缺失 → 该检查判 unverified（off 不进此函数；shadow 记录、required 阻止成功）
    unavailable: set = set()
    for check in (task.acceptance_spec or []):
        try:
            ctx.acceptance_registry.resolve(check["checker_id"], check["checker_version"])
        except InvalidAcceptanceSpec:
            unavailable.add((check["checker_id"], check["checker_version"]))
            task.acceptance_unavailable = True

    # ③ 一次验收
    record = await run_acceptance_pass(
        registry=ctx.acceptance_registry, executor=ctx.acceptance_executor,
        spec=task.acceptance_spec or [], candidate=candidate_of(task, state.extra),
        task=task, mode=mode, unavailable_checkers=unavailable)
    previous = task.acceptance
    record["repairs_used_after"] = task.acceptance_repairs_used

    # ④ 候选引用：小直存，超限外存（memory 记录 id，恢复可解引用取回）
    import json as _json
    body = candidate_of(task, state.extra)["final_body"]
    candidate_ref: dict = {"inline": None, "record_id": None}
    if isinstance(body, str) and len(body) <= 4000:
        candidate_ref["inline"] = body
    else:
        record_id = await ctx.memory.ingest(
            MemoryEvent(kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                        address=state.scope, content=body, timestamp=now_utc(),
                        role="assistant",
                        metadata={"task_id": task.id, "acceptance_candidate": True}),
            ctx.provider_ctx)
        candidate_ref["record_id"] = record_id

    # ⑤ 持久化记录（先于任何修正生成落盘——跨崩溃额度保证的①②③序）
    from ctx_weft.core.acceptance.support import acceptance_event_payload
    payload = acceptance_event_payload(record)
    payload["candidate_ref"] = candidate_ref
    await ctx.event_bus.emit(make_event(state, EventType.TASK_ACCEPTANCE_CHECKED, payload=payload))
    task.acceptance = record

    if mode != "required":
        return None  # shadow：到此为止，交付决策不变
    if record["overall"] == "passed":
        return None
    if record["overall"] == "unverified":
        return {"phase": "failed", "verdict": "unverified", "findings": record["findings"],
                "error_message": "required acceptance check(s) unverified (checker fault/timeout/unavailable)"}
    # failed：修正可行性（额度 / 缺口非重复 / 预算未被限额熔断）
    limit_codes = {TaskErrorCode.TASK_DEADLINE_EXCEEDED, TaskErrorCode.STEP_DEADLINE_EXCEEDED,
                   TaskErrorCode.PROVIDER_DEADLINE_EXCEEDED, TaskErrorCode.ACTOR_TURN_LIMIT}
    budget_ok = task.error_code not in limit_codes
    repeated = bool(previous and previous.get("overall") == "failed"
                    and findings_equal(previous.get("findings", []), record["findings"]))
    if task.acceptance_repairs_used == 0 and budget_ok and not repeated:
        return {"phase": "retry", "verdict": "failed", "findings": record["findings"],
                "repairs_used_after": task.acceptance_repairs_used + 1,
                "attempt": record["attempt"], "error_message": compose_gap_hint(record["findings"])}
    return {"phase": "failed", "verdict": "failed", "findings": record["findings"],
            "error_message": "required acceptance checks failed and repair is unavailable "
                             "(budget exhausted / repeated gap / repairs used)"}


class FinalizeStep(Step):
    name = "finalize"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        task = state.task
        verdict = state.verdict
        outcome = verdict.task_outcome if verdict else "fail"
        summary = verdict.act_recap if verdict else ""            # → task.process_report（success/fail finish 对；retry 不再写，进度由段摘要承载）
        task_summary = verdict.task_summary if verdict else ""    # → 汇报给 parent 的 process report
        events: list[Any] = []

        # 交给 TaskManager 的结局：用 observer 的**原值** outcome，不经下面 retry_exhausted
        # 的重试耗尽降级——那是 TM 的活（task_disposition.disposition_for），loop 只报
        # 「observer 说了什么」。task 状态与状态事件都不在本 step 落地了（Task 4）。
        run_outcome = RunOutcome(
            kind=RunOutcomeKind.COMPLETED,
            verdict=outcome,
            summary=summary,
            outputs=task.outputs,
            error=task.error or "",
        )

        # retry 超过上限 → 降级 fail（不再重试）。专属 error_code 区分「程序按重试上限
        # 熔断」与 observer 主动判死；死因 = 最后一轮 retry 判决暂存的受阻原因（task.error，
        # report_task_outcome 判 retry 时写入），机械退出轮没有判决则为空。
        # **本地降级只为收尾副作用服务**（下面的 close/bubble/blackboard 与 TASK_FINALIZED
        # 的 outcome），不再决定 task 状态：状态与 TaskFailed 事件由 TM 据同一条判据算出
        # （disposition_for 的 exhausted 分支）。两处判据逐字同形，改一处必须改另一处。
        retry_exhausted = outcome == "retry" and task.retry_count >= task.max_retries
        if retry_exhausted:
            outcome = "fail"
            task.observer_outcome = "fail"
            task.error_code = TaskErrorCode.RETRY_EXHAUSTED

        # spec: delivery-acceptance——成功候选必经验收边界（required 门禁；shadow 只记录；
        # off 零行为）。候选 = 分开的 final_body/final_summary（design D2），检查、指纹与
        # 提交同一份。验收失败时 success 不再是终态：retry → 一次修正（额度预占在 TM）；
        # failed → 落失败终态（TASK_ACCEPTANCE_FAILED）。
        if outcome == "success" and getattr(task, "acceptance_spec", None):
            mode = getattr(ctx, "acceptance_mode", "off") or "off"
            if mode != "off" and ctx.acceptance_registry is not None and ctx.acceptance_executor is not None:
                gate = await _delivery_acceptance_gate(state, ctx, task, mode)
                if gate is not None:
                    if gate["phase"] == "retry":
                        outcome = "retry"  # 复用既有 retry 收尾形态（无 finish 副作用、清 outputs）
                    else:
                        outcome = "fail"
                        task.observer_outcome = "fail"
                        task.error_code = TaskErrorCode.TASK_ACCEPTANCE_FAILED
                        task.error = gate.get("error_message", "")
                    import dataclasses as _dc
                    run_outcome = _dc.replace(
                        run_outcome, verdict=outcome, outputs=None,
                        error=task.error or "", error_code=task.error_code or "",
                        acceptance=gate)

        terminal = outcome in ("success", "fail")
        # 汇报给 parent（blackboard + cross_agent bubble）= 最终输出 + task_summary（process report 作用）；
        # task_summary 空时回退 act_recap。
        mem_content = _build_memory_content(task.outputs, task_summary or summary)

        # 1) 统一 close：bubble / 自身残留 / 软删自身对话 / GC 子树（spec 2026-06-23）。
        # has_llm_summary=verdict.reported：规则 observe 占位 close 不即折末段 raw，
        # 等 bg 真摘要落地后补删（spec 2026-07-20 延迟折叠）。
        if terminal and mem_content:
            events.extend(await finalize_task_memory(
                ctx.memory, state, task, mem_content, outcome, ctx,
                act_recap=summary, task_summary=task_summary,
                has_llm_summary=bool(verdict and verdict.reported),
            ))

        # 2) 按 outcome 收尾（只写「判决的产物」，不写状态、不发状态事件——Task 4：
        #    TaskFinished / TaskFailed / TaskRequeued 三条统一由 TaskManager 据 RunOutcome 发）
        if outcome in ("success", "fail"):
            task.finished_at = now_utc()
            task.process_report = summary
        elif outcome == "retry":
            # retry 反馈由 observe 前台段折写的 TASK_COMPACT_SUMMARY 承载（spec 2026-07-01 §3.1）；
            # 不再写 process_report/process_report_at（旧 Progress So Far 字段路径已废）。
            # 机械退出（max_turns/context_limit）也归到这里：重排再跑，受 max_retries 兜底。
            # retry_count 的 +1 也搬走了：处置表算出新值，TaskManager 写回 task
            # （在这里先加会让 TM 二次自增，重试预算一轮烧两格）。
            task.outputs = None

        # 3) success 时发布 BLACKBOARD，供任何 agent 按 task_id 精确召回
        if outcome == "success" and mem_content:
            await ctx.memory.ingest(
                MemoryEvent(
                    kind=MemoryKind.PUBLICATION, scope=MemoryScope.SESSION,
                    address=state.scope,
                    content=mem_content,
                    timestamp=now_utc(),
                    role="assistant",
                    topic=task.id,
                    metadata={"task_id": task.id, "title": task.title, "outcome": outcome,
                              "parent_task_id": task.parent_task_id},
                ),
                ctx.provider_ctx,
            )
            events.append(make_event(
                state, EventType.BLACKBOARD_PUBLISHED,
                payload={"topic": task.id, "content_length": len(mem_content), "parent_task_id": task.parent_task_id},
            ))

        # ── TASK_FINALIZED：结局 + 交付物出核 ────────────────────────────────
        # 交付物分两段送：`output` 是答复本身，`summary` 是 agent 给 reviewer 的自评
        # 清单（finish_task 的 deliverables_summary）。**必须分开**——host 打印「最终
        # 答复」只该拿 output；此前只送 task_id/outcome，host 的 tasks 表 outputs_json /
        # error 两列因此恒空。
        #
        # 判据以 `task.outputs` 为准绳，不直接读 state.extra，两条边界都在这一句里：
        # ① retry 驳回：observer 判本段不合格会把 task.outputs 置空（上面 retry 分支、
        #    task/manager.py 的 requeue 各有一次），但 state.extra 里还留着上一轮的废稿；
        #    直接读 extra 会让 host 把一份已被驳回的稿子写进 tasks 表。所以 deliverable
        #    为空 → 两段一起归空。
        # ② extra 空但 outputs 有货：多段任务的末段可能没正文，recap 重启路径更是根本
        #    没跑过 ActStep（state.extra 全空）——这时 task.outputs 才是真答复，回落它。
        # `getattr(..., None) or {}` 只为容忍用 SimpleNamespace 造 state 的单测替身
        # （test_finalize_fail_reason / test_subtask_nesting），生产路径必命中真字段。
        #
        # output 送**纯文本**（_output_text 提取），不送 content parts：这个字段的消费者
        # 是 tasks.outputs_json 与 CLI 打印，两个都是文本场景；多模态交付物有 blob store
        # 那条正路，不该让事件载荷背图片。
        extra = getattr(state, "extra", None) or {}
        deliverable = _output_text(task.outputs)
        events.append(make_event(
            state, EventType.TASK_FINALIZED,
            payload={
                "task_id": task.id,
                "outcome": outcome,
                "outputs": {
                    "output": (extra.get("final_body", "") or deliverable) if deliverable else "",
                    "summary": extra.get("final_summary", "") if deliverable else "",
                },
                "error": task.error or "",
            },
        ))

        return StepOutcome(next_step=None, state_patch={"run_outcome": run_outcome}, events=events)


def _output_text(outputs: Any) -> str:
    """Extract the plain answer text from task.outputs (str or [{type:text,text:...}])."""
    if isinstance(outputs, list):
        return next(
            (p.get("text", "") for p in outputs if isinstance(p, dict) and p.get("type") == "text"),
            "",
        )
    if isinstance(outputs, str):
        return outputs
    return ""


def _build_memory_content(outputs: Any, summary: str) -> str:
    """合并 task outputs 和 observer summary，对齐 miniAgents _write_execution_memory。

    格式："{output_text}\\n\\nProcess Report: {summary}"
    只有 summary 时："{summary}"
    """
    parts = [p for p in [_output_text(outputs), summary] if p]
    return "\n\nProcess Report: ".join(parts)
