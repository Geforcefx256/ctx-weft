"""FinalizeStep：持久化收尾记忆和黑板，向 TaskManager 报告结局。

miniAgents 对齐版：
- memory 内容 = task.outputs + "\\n\\nProcess Report: " + verdict.act_recap（合并写入）
- 新增 BLACKBOARD_PUBLISH：让父 agent 通过 recall_topic(task.id) 读到子任务结果
"""

from __future__ import annotations

from ctx_weft.core.loop.task_memory_fold import (
    task_record_group, task_record_kind, task_record_metadata,
)

import logging
from datetime import datetime, timedelta
from typing import Any

from ctx_weft.core.assembler.prompts import runtime_prompt
from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome, make_event
from ctx_weft.core.loop.task_memory_fold import (
    build_finish_slots,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from ctx_weft.core.utils.content import image_tokens
from ctx_weft.protocols.content import content_to_text
from ctx_weft.core.utils.clock import as_utc, now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.core.utils.task_ref import task_ref
from ctx_weft.core.utils.verdict import (
    VERDICT_CONTINUE,
    VERDICT_FAIL,
    VERDICT_SUCCESS,
    VERDICT_WAITING,
)
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryKind, MemoryScope, MemoryAddress, load_records

logger = logging.getLogger(__name__)

# v2 P3：旧类型清单常量（_OWN_CONV_TYPES/_FINAL_RAW_TYPES）随读侧 kind+role 谓词化删除。

def _final_reply_source(records, outputs: str) -> str | None:
    """Locate the actual retained actor answer, including the old body+summary form."""
    if not outputs.strip():
        return None
    candidates = []
    for record in records:
        md = record.metadata
        if (
            record.scope != MemoryScope.TASK
            or record.role != "assistant"
            or md.get("tool_calls")
            or md.get("tool_call_count")
            or md.get("interrupted")
            or md.get("cancelled")
            or not isinstance(record.content, str)
        ):
            continue
        body = record.content.strip()
        if body and (outputs.strip() == body or outputs.strip().startswith(body + "\n\n")):
            candidates.append(record)
    return max(candidates, key=lambda r: (
        r.timestamp.isoformat() if r.timestamp else "", r.metadata.get("seq_no", 0)
    )).id if candidates else None


def _dispatch_running_ack(ref: str) -> str:
    """Running acknowledgement for the child task; its body follows inline."""
    return runtime_prompt("dispatch_running", ref=ref)


# 同 agent 派发：派发对 tool 结果（不含任何子任务结果，spec 2026-06-30 §2.5）。
def _dispatch_ack(ref: str, outcome: str) -> str:
    """Terminal acknowledgement pointing to the inline child execution and closure."""
    return runtime_prompt("dispatch_closed", ref=ref, failed=outcome == "fail")



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
    if frame.metadata.get("task_memory"):
        return task_record_group(frame)
    tcs = frame.metadata.get("tool_calls") or []
    tcid = task.origin_tool_call_id
    if tcid and any(tc.get("id") == tcid for tc in tcs):
        return tcid
    return tcs[0].get("id") if tcs else None


async def _find_dispatch_frame(memory, parent_scope, task, provider_ctx):
    """Find an existing child start record, accepting legacy synthetic assistant frames."""
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope
    existing = await load_records(memory,
        MemoryAddress(session_id=parent_scope.session_id, agent_id=parent_scope.agent_id),
        MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    frame = next(
        (r for r in existing if (task_record_kind(r) == "start" or r.role == "assistant") and _is_frame_of(r, task)),
        None,
    )
    if frame is None:
        return None
    # 归一：框可能来自事件重放 / DB 反序列化而丢 tz（naive），直接返回会让调用方拿它与
    # aware 时间比较时炸 TypeError。
    return _as_utc(frame.timestamp), _frame_tool_call_id(frame, task)


async def _ensure_dispatch_frame(memory, parent_scope, task, provider_ctx):
    """Persist one child start label at its first started_at (or created_at).

    Retain the existing timestamp and group on retry/recovery. child_task_id
    identifies the dispatch after transient origin_tool_call_id has been lost."""
    found = await _find_dispatch_frame(memory, parent_scope, task, provider_ctx)
    if found is not None:
        # Retry may refresh started_at. Preserve the original frame and running
        # acknowledgement anchor; terminal results use their completion time.
        return found
    # 框与 running ack 的公共锚点：task 真正启动执行的时刻。归一为 aware(UTC)：started_at/created_at
    # 可能来自事件重放而为 naive（历史无 started_at 时回退 created_at 再回退 now）。
    ts = _as_utc(task.started_at or task.created_at or now_utc())
    tool_call_id = task.origin_tool_call_id or generate_id("tcall")
    await memory.ingest(MemoryEvent(
        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=parent_scope,
        content=runtime_prompt("task_memory_record", record=dict(kind="start", task_id=task.id,
            parent_task_id=task.parent_task_id or "", title=task.title, description=task.description or "", body="", outcome="")).strip(),
        timestamp=ts, role="user",
        # origin owns this dispatch; the child's parent is already represented
        # by origin + child_task_id and by the rendered start label. It is not
        # the owning task's parent relationship.
        metadata={"origin_task_id": task.parent_task_id,
                  "child_task_id": task.id, "task_memory": task_record_metadata("start", tool_call_id)},
    ), provider_ctx)
    return ts, tool_call_id


def _parent_scope_of(state, task) -> MemoryAddress:
    """派发方（parent task + creator agent）的 scope——派发框/result 的落点。"""
    return MemoryAddress(
        session_id=state.scope.session_id,
        task_id=task.parent_task_id,
        agent_id=task.creator_agent_id,
    )


async def _put_dispatch_result(memory, parent_scope, task, content: str, ts, provider_ctx,
                               *, replace: bool, tool_call_id: str, operation_id: str = "",
                               completed_at: str = "") -> None:
    """Atomically replace the running child acknowledgement with a labeled result.

    Start is idempotent; close/cancel supersedes the previous result. A terminal
    result follows the child's closure, rather than sharing the start timestamp.
    The committed completion time keeps this position stable during recovery."""
    from ctx_weft.protocols import MemoryAddress, MemoryKind, MemoryScope
    recs = await load_records(memory,
        MemoryAddress(session_id=parent_scope.session_id, agent_id=parent_scope.agent_id),
        MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    stale = [r.id for r in recs
             if (task_record_kind(r) == "result" or r.role == "tool")
             and (_is_frame_of(r, task) or task_record_group(r) == tool_call_id)]
    from ctx_weft.core.utils.ids import record_id
    result_id = record_id(operation_id, "dispatch_result", task.id, parent_scope)
    if result_id and result_id in stale:
        return
    if stale and not replace:
        return
    # Closure has at most three slots: recap, final reply, finish. The dispatch
    # result follows all of them, including when a late report adds a reply slot.
    result_ts = ts
    if replace:
        completion = (datetime.fromisoformat(completed_at) if completed_at
                      else getattr(task, "finished_at", None) or now_utc())
        result_ts = _as_utc(completion) + timedelta(microseconds=3)
    # v2 P3d：旧 ack 遗忘 + 终态写入一次原子 fold（stale 空 = 纯写入）
    await memory.fold(stale, [
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=parent_scope,
            content=runtime_prompt("task_memory_record", record=dict(kind="result", task_id=task.id,
                parent_task_id="", title=task.title, description="", body=content, outcome="")).strip(),
            timestamp=result_ts, role="user", id=result_id,
            metadata={"origin_task_id": task.parent_task_id,
                      "child_task_id": task.id,
                      "dispatch_state": "closed" if replace else "running",
                      **({"observe_id": operation_id} if operation_id else {}),
                      "task_memory": task_record_metadata("result", tool_call_id)},
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
        ctx.memory, parent_scope, task, _dispatch_running_ack(task_ref(task)), ts, ctx.provider_ctx,
        replace=False, tool_call_id=tool_call_id,
    )


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
    records = await load_records(memory,
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
    # 同 fold.is_short_segment：保留 join 后数一次的文本口径，图片另计。
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
    completion = getattr(state, "extra", {}).get("observe_completion", {})
    short = (getattr(state, "extra", {}).get("observe_decision") or completion).get("short_leaf")
    if short is None:
        short = await _is_short_leaf(
            memory, state.scope, task, state.agent.loop_config, ctx, bool(descendants))
    return await _close_one(
        memory, state, task, mem_content, outcome, ctx,
        short=short, act_recap=act_recap, task_summary=task_summary,
        has_llm_summary=has_llm_summary,
    )


async def _supersede_final_raw_segment(memory, scope, provider_ctx) -> None:
    """长任务 close：supersede task 层**末段** raw（active LLM_RESPONSE/TOOL_INVOCATION/
    TOOL_RESULT），保留 USER_PROMPT + TASK_COMPACT_SUMMARY 锚点（spec 2026-06-28 §3.2）。

    段作用域（2026-07-21）：末段 = 最后一条 active USER_PROMPT 之后。此前假设「active raw
    即末段」（中间段在各自边界已折），但短段免折（fold.is_short_segment）会让
    前段 raw 以 active 状态残留——它们无胶囊代表，删了即信息丢失（其 UP 失去回答位），故保留
    （「短 → 原文成胶囊」）。无 UP（防御）→ 全删（旧行为）。

    调用时机（spec 2026-07-20 修订的不变量：末段 raw 与「真实 Process Report」至少存其一）：
    finish 对已承载 LLM 真摘要 → close 时同步删；占位 finish 对 → 推迟到 bg 替换真摘要后
    补删（background_observe close 回调）。**不另产新 TASK_COMPACT_SUMMARY**（避免与 finish
    对重复）。幂等：raw 已删则 no-op。

"""
    from ctx_weft.protocols import MemoryKind, MemoryScope
    # v2 P3a：升序视图（对话 + audit，SUMMARY 锚点天然不在），段界 = 末条 role=user 回合。
    view = await load_records(memory,
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
    close 时删；保留策略随 finish 对持久化，bg 替换真摘要后按原决定补删；bg 失败则
    raw 永久保留（降级 = 保 raw，信息不丢）。
    """
    # A committed observation can be replayed after a crash during close. Its operation
    # identity gives finish slots, dispatch results and publications stable record IDs.
    events: list[Any] = []
    same_agent = task.creator_agent_id == task.assigned_agent_id
    short = short or bool(getattr(state, "extra", {}).get("observe_keep_raw"))
    cross_agent = bool(task.parent_task_id) and not same_agent
    is_own_root = (task.parent_task_id is None) or cross_agent
    # raw 所在层恒为本 task 的 task scope（≠ 嵌套 finish 对的 parent scope）
    raw_fold_scope = state.scope if (not short and not has_llm_summary) else None
    # 末段 raw 会被折 → 答复须由 finish 对的锚点槽承载；short leaf 不折（原文即胶囊）→ 不塞。
    final_reply = "" if short else _output_text(task.outputs)
    completed_at = (getattr(state, "extra", {}).get("observe_completion", {}).get("completed_at")
                    or _as_utc(getattr(task, "finished_at", None) or now_utc()).isoformat())

    # 1) bubble 到 parent scope（dispatch marker 所在 scope）
    # 判据只看 parent_task_id + mem_content：origin_tool_call_id 是瞬态字段，重启重建后为 None，
    # 曾经把它写进条件 → 跨重启收尾的子任务整段 bubble 被跳过（父的 ack 停在 running、finish 对
    # 不合成 = 胶囊丢失），而末段 raw 照删。配对 id 现由框自己给出，不再需要它当门槛。
    if task.parent_task_id and not getattr(task, "switched_to_task_id", ""):
        parent_scope = _parent_scope_of(state, task)
        # 框通常已由 ensure_dispatch_frame_at_start（子 start 时）铸好，_ensure_dispatch_frame 在此
        # 幂等命中；仅对没走过 start 钩子的路径（旧数据 / start 与 close 之间崩溃）才真正补铸。
        # tool 槽用 replace=True：start 时写的是 running 态，此处须**替换**为终态而非新增一条。
        if cross_agent:
            # Cross-agent results carry the delivered output at completion time;
            # the child's private body remains in its own agent scope.
            report_prefix = "[outcome=fail] " if outcome == "fail" else ""
            frame_ts, frame_tcid = await _ensure_dispatch_frame(
                memory, parent_scope, task, ctx.provider_ctx)
            await _put_dispatch_result(
                memory, parent_scope, task, f"{report_prefix}{mem_content}", frame_ts,
                ctx.provider_ctx, replace=True, tool_call_id=frame_tcid,
                operation_id=getattr(state, "extra", {}).get("observe_id", ""),
                completed_at=completed_at,
            )
            events.append(make_event(
                state, EventType.MEMORY_INGESTED,
                payload={"memory_event_type": MemoryEventType.AGENT_CONVERSATION_TURN.value,
                         "source": "dispatch_result", "content_length": len(mem_content)},
            ))
        elif same_agent:
            # The terminal result closes the inline child capsule. It no longer
            # needs tool-call adjacency with the start label.
            frame_ts, frame_tcid = await _ensure_dispatch_frame(
                memory, parent_scope, task, ctx.provider_ctx)
            await _put_dispatch_result(
                memory, parent_scope, task, _dispatch_ack(task_ref(task), outcome), frame_ts,
                ctx.provider_ctx, replace=True, tool_call_id=frame_tcid,
                operation_id=getattr(state, "extra", {}).get("observe_id", ""),
                completed_at=completed_at,
            )
            # 嵌套合成子自己的 finish 对（写进共享 agent scope，@close 时刻）
            await _synthesize_dispatch_pair(
                memory, parent_scope, task, act_recap, task_summary, outcome, ctx.provider_ctx,
                raw_fold_scope=raw_fold_scope, final_reply=final_reply, preserve_raw=short,
                summary_available=has_llm_summary,
                operation_id=getattr(state, "extra", {}).get("observe_id", ""),
                completed_at=completed_at)

    # 2) 自身 finish 对：own root（session 根或跨 agent 根）close 时无条件在 own scope 合成
    if is_own_root or getattr(task, "switched_to_task_id", ""):
        await _synthesize_dispatch_pair(memory, state.scope, task, act_recap, task_summary, outcome,
                                        ctx.provider_ctx, raw_fold_scope=raw_fold_scope,
                                        final_reply=final_reply, preserve_raw=short,
                                        summary_available=has_llm_summary,
                                        operation_id=getattr(state, "extra", {}).get("observe_id", ""),
                                        completed_at=completed_at)
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
    取消是粘滞终态：按 task 固定记录 ID，按持久化取消时间固定槽位时间；
    重放半写入的 finish 对或父 ack 时不追加新记录。
    """
    same_agent = task.creator_agent_id == task.assigned_agent_id
    cross_agent = bool(task.parent_task_id) and not same_agent
    is_own_root = (task.parent_task_id is None) or cross_agent

    act_recap = runtime_prompt('cancelled_act_recap')
    task_summary = runtime_prompt('cancelled_task_summary', reason=reason_text)
    completed_at = _as_utc(task.finished_at or now_utc()).isoformat()

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
        ack_text = runtime_prompt("dispatch_cancelled", ref=task_ref(task), reason=reason_text)
        await _put_dispatch_result(
            memory, parent_scope, task, ack_text, frame_ts, provider_ctx, replace=True,
            tool_call_id=frame_tcid, operation_id=f"cancel:{task.id}",
            completed_at=completed_at,
        )
        if same_agent:
            await _synthesize_dispatch_pair(
                memory, parent_scope, task, act_recap, task_summary, "cancelled", provider_ctx,
                operation_id=f"cancel:{task.id}",
                completed_at=completed_at,
            )
            return

    if is_own_root:
        own_scope = MemoryAddress(
            session_id=session_id, task_id=task.id,
            agent_id=task.assigned_agent_id or task.creator_agent_id,
        )
        await _synthesize_dispatch_pair(
            memory, own_scope, task, act_recap, task_summary, "cancelled", provider_ctx,
            operation_id=f"cancel:{task.id}",
            completed_at=completed_at,
        )


async def _synthesize_dispatch_pair(memory, scope, task, act_recap: str, task_summary: str,
                                    outcome: str, provider_ctx, *,
                                    raw_fold_scope=None, final_reply: str = "", preserve_raw: bool | None = None,
                                    summary_available: bool | None = None,
                                    operation_id: str = "", completed_at: str = "") -> None:
    """Persist finish slots with the original raw-retention policy.

    Background reports replace only a complete pair. No report is handed off
    through process-local state.
    """
    from ctx_weft.core.utils.ids import record_id
    from datetime import datetime
    base = datetime.fromisoformat(completed_at) if completed_at else now_utc()
    tool_call_id = record_id(operation_id, "finish_call", scope, task.id) or generate_id("tcall")
    # 占位 close（raw_fold_scope 非 None = 末段 raw 还没折）先写两槽：答复此刻仍在 raw 里，
    # 提前塞锚点会让它出现两遍；等 bg 真报告落地、raw 真被折时由 replace_finish_report 升三槽。
    embed_reply = "" if raw_fold_scope is not None else final_reply
    # Closure records carry process recap, optional final reply and terminal status.
    slots = build_finish_slots(
        scope=scope, task_id=task.id, parent_task_id=task.parent_task_id,
        title=task.title or "", outcome=outcome, act_recap=act_recap,
        task_summary=task_summary, final_reply=embed_reply, base=base,
        tool_call_id=tool_call_id,
        switched=getattr(task, "switch_requested", False),
    )
    policy = {"preserve_raw": (raw_fold_scope is None and not final_reply) if preserve_raw is None else preserve_raw,
              "outcome": outcome, "slot_count": len(slots), "observe_id": operation_id,
              "capsule_state": "full",
              "has_summary": bool((task_summary or act_recap).strip()) and summary_available is not False}
    raw = await load_records(memory, MemoryAddress(session_id=scope.session_id, task_id=task.id,
        agent_id=getattr(task, "assigned_agent_id", None) or scope.agent_id),
        MemoryScope.TASK, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    source = _final_reply_source(raw, _output_text(getattr(task, "outputs", None)))
    if source:
        policy["final_reply_record_id"] = source
    for index, event in enumerate(slots):
        event.metadata["finish_policy"] = policy
        event.id = record_id(operation_id, "finish_v1", scope, task.id, index)
    # A pre-upgrade producer may already have written part of its old closure.
    # Replace those rows atomically instead of retaining an orphan legacy pair.
    existing = await load_records(memory,
        MemoryAddress(session_id=scope.session_id, agent_id=scope.agent_id),
        MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN])
    legacy = [r.id for r in existing
              if r.metadata.get("origin_task_id") == task.id
              and not r.metadata.get("task_memory")
              and (task_record_group(r) == tool_call_id or
                   operation_id and r.metadata.get("finish_policy", {}).get("observe_id") == operation_id)]
    if legacy:
        await memory.fold(legacy, slots, provider_ctx)
    else:
        for event in slots:
            await memory.ingest(event, provider_ctx)




async def apply_task_close(
    state, task, ctx, *, outcome: str, act_recap: str, task_summary: str,
    has_llm_summary: bool,
) -> list:
    """终态收尾的 memory 侧：统一 close + blackboard。返回要发的事件。

    **抽出来是为了让带外判决也能做这件事**（2026-09-27）。park 之后 run 就地结束
    （`_cold_park` 抛 `HitlPark`），driver 到不了 observe/finalize——于是 S5/S6 之后每个
    「park → 后台判 success」的 task 都跳过了整个 `FinalizeStep`，而本函数里这两件事
    （加上 `task_finalized_event`）此前只在那里发生过。子任务漏得最重：parent 醒来只看到
    「派发框 + 停在 running 的 ack」，拿不到任何产出。调用方见
    `steps.observe._out_of_band_finalize`。

    终态总是写入收尾记录；只有真实内容非空的 success 才发布黑板。
    mem_content 在这里现算（`task.outputs` + `task_summary or act_recap`），两个调用方
    因此不会各算一份、算歪一份。
    """
    from ctx_weft.core.utils.ids import record_id
    events: list[Any] = []
    if "BlackboardPublished:" in getattr(state, "extra", {}).get("observe_effects", []):
        return events
    mem_content = _build_memory_content(task.outputs, task_summary or act_recap)

    # 统一 close：bubble / 自身残留 / 软删自身对话 / GC 子树（spec 2026-06-23）。
    # Only a genuine recap permits discarding the raw segment. Mechanical success
    # records the terminal status while retaining execution evidence.
    events.extend(await finalize_task_memory(
        ctx.memory, state, task, mem_content, outcome, ctx,
        act_recap=act_recap, task_summary=task_summary,
        has_llm_summary=has_llm_summary,
    ))

    # success 时发布 BLACKBOARD，供任何 agent 按 task_id 精确召回
    if outcome == "success" and mem_content:
        await ctx.memory.ingest(
            MemoryEvent(
                kind=MemoryKind.PUBLICATION, scope=MemoryScope.SESSION,
                id=record_id(
                    getattr(state, "extra", {}).get("observe_id", ""), "blackboard", task.id),
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
            payload={"topic": task.id, "content_length": len(mem_content),
                     "parent_task_id": task.parent_task_id},
        ))
    return events


def task_finalized_event(state, task, *, outcome: str):
    """Publish the actual final answer and optional legacy deliverables summary.

    task.outputs is authoritative, so rejected drafts are never published."""
    extra = getattr(state, "extra", None) or {}
    deliverable = _output_text(task.outputs)
    return make_event(
        state, EventType.TASK_FINALIZED,
        payload={
            "task_id": task.id,
            "finished_at": task.finished_at.isoformat() if task.finished_at else None,
            **({"reason": "user_topic_switch"} if getattr(task, "switch_requested", False) else {}),
            "outcome": outcome,
            "outputs": {
                "output": (extra.get("final_body", "") or deliverable) if deliverable else "",
                "summary": extra.get("final_summary", "") if deliverable else "",
            },
            "error": task.error or "",
        },
    )


class FinalizeStep(Step):
    name = "finalize"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        # Stop signals can arrive while the driver publishes observe's outcome
        # events, after the fold has returned but before this step starts.
        from ctx_weft.core.loop.steps.observe import foreground_stop_checkpoint
        await foreground_stop_checkpoint(state, ctx)
        task = state.task
        verdict = state.verdict
        outcome = verdict.task_outcome if verdict else VERDICT_CONTINUE
        summary = verdict.act_recap if verdict else ""            # → task.process_report（success/fail finish 对；retry 不再写，进度由段摘要承载）
        task_summary = verdict.task_summary if verdict else ""    # → 汇报给 parent 的 process report
        events: list[Any] = []

        # 交给 TaskManager 的结局保留 observer 的原值；continue 不按 retry_count 降级。
        # task 状态与状态事件由 TaskManager 据此落地。
        extra = getattr(state, "extra", {})
        run_outcome = RunOutcome(
            kind=RunOutcomeKind.COMPLETED,
            verdict=outcome,
            summary=summary,
            outputs=task.outputs,
            error=task.error or "",
            hitl_id=extra.get("waiting_hitl_id", ""),
            observe_id=extra.get("observe_id", ""),
            observe_disposition=(extra.get("observe_decision") or extra.get("observe_completion", {})).get("disposition"),
            observe_report_pending=bool(extra.get("observe_decision", {}).get("verdict")),
        )

        terminal = outcome in (VERDICT_SUCCESS, VERDICT_FAIL)
        # mem_content（= 最终输出 + task_summary，空则回退 act_recap）现在由
        # `apply_task_close` 自己算——两个调用方各算一份必然有一份算歪。

        # 1) 统一 close + blackboard（`apply_task_close`，与带外 finalize 共用同一份）。
        if terminal and "TaskFinalized:" not in getattr(state, "extra", {}).get("observe_effects", []):
            events.extend(await apply_task_close(
                state, task, ctx, outcome=outcome, act_recap=summary,
                task_summary=task_summary,
                has_llm_summary=bool(verdict and verdict.reported),
            ))

        # 2) 按 outcome 收尾（只写「判决的产物」，不写状态、不发状态事件——Task 4：
        #    TaskFinished / TaskFailed / TaskRequeued 三条统一由 TaskManager 据 RunOutcome 发）
        if outcome in (VERDICT_SUCCESS, VERDICT_FAIL):
            task.finished_at = now_utc()
            task.process_report = summary
        elif outcome in (VERDICT_CONTINUE, VERDICT_WAITING):
            # 未完的反馈由 observe 前台段折写的 TASK_COMPACT_SUMMARY 承载（spec 2026-07-01 §3.1）；
            # 不再写 process_report/process_report_at（旧 Progress So Far 字段路径已废）。
            # 机械退出（max_turns/context_limit）后的 continue 也重排再跑，
            # 不消耗运行异常的 retry_count，也不因 max_retries 耗尽而收尾。
            task.outputs = None

        # 3) TASK_FINALIZED：**每条路径都发**，retry 也发（它把 host 投影里上一次的
        #    outputs_json 清掉）。blackboard 已随 close 搬进 `apply_task_close`。
        events.append(task_finalized_event(state, task, outcome=outcome))

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
