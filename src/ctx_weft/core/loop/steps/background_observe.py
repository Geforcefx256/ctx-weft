"""root 后台异步 observe：在交互/finish 段边界产段摘要并折 raw（spec §3.2）。

仿 recognize_intent 的 fire-and-forget：快照 state、create_task、track_background。
同一 task 至多一个在跑（_task_locks 串行），失败吞掉（降级 = 该段保 raw，spec §3.6）。

boundary 分流（Task 6）：
  - finish / normal → 结果落 _close_report 槽，不写 memory（finalize Task 8 取用）
  - 其他（interrupt、plain_text、dispatch 等）→ apply_compact 写 TASK_COMPACT_SUMMARY；
    但**当前段**（末条 UP 之后）active raw token ≤ short_segment_token_threshold 时
    **免折**（短段保 raw，不跑后台 LLM——「短 → 原文成胶囊」决策在段边界的延伸）。
    免折的段 raw **永久保 raw**：折叠带 since_last=USER_PROMPT 只折当前段，前段残留
    从不跨段合折（否则合并摘要会锚到前一条 UP 之前，UP 失去回答位，2026-07-21）

崩溃恢复竞态（spec §5.1/§3.6；2026-07-16 起同进程内闭合）：`recover_agent` 对一个
SUSPENDED-且-有待完成段 recap 的 task，会（a）经 TaskManager.restore 重排该 task 的新一轮
run，（b）经 `_relaunch_task_recap` 重跑被打断的段 recap。relaunch 先于 register_and_drain
发生，且 `_run_loop` 入口 await_pending_background_observe——新 run 开跑前必等 recap 完成，
两者不再并发写同一段。跨进程/其他极端时序仍是 best-effort：最坏该段保 raw，不影响正确性。
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING

from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.core.loop.driver import make_event
from ctx_weft.core.loop.steps.observe import run_observe_react
from ctx_weft.core.orchestrator.task.disposition import RunOutcomeKind
from ctx_weft.core.utils.content import content_to_text, image_tokens
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import MemoryEventType, MemoryScope

if TYPE_CHECKING:
    from ctx_weft.core.loop.driver import LoopContext, LoopState

logger = logging.getLogger(__name__)

_task_locks: dict[str, asyncio.Lock] = {}
_task_pending: dict[str, asyncio.Task] = {}
# task_id -> 该 task 当前在途后台 observe 那次 launch 的 run_id（与 _task_pending 同步写入，
# launch_background_observe 里两行相邻赋值；TaskRecapDone 事件信封上的 run_id 就是它——
# 见 pending_background_observe_run_id docstring）。
_task_pending_run_id: dict[str, str] = {}
# agent_id -> 该 agent **最近一次** launch 的在途后台 observe。与 _task_pending 同步写入。
#
# 为什么按 task 键还不够：`send_message` 打到一个已终态 agent 上时走
# `_start_task_for_agent` 建的是**新 task**，新 task 的 `_run_loop` 入口等的是
# `_task_pending[新 task_id]`（空），上一个 task 的折叠挂在旧 task_id 下、无人等。
# 而 close 边界的 launch 恒发生在 `TaskFinished` 之前（同协程、无 await 间隔），
# 所以「上一轮刚结束、用户立刻发下一条」这个窗口里折叠必然在飞——新 task 的首次装配
# 会经 `recall_recent_by_agent`（过滤 `not is_superseded`）读到上一轮**未被 supersede
# 的 raw**，而不是折出来的胶囊：prompt 白白胀一轮的量。
#
# 一个 agent 同时只可能有一次在途后台 observe（它自己那条 run 是串行的），故这里
# 一个槽足够；子 agent 各自一个 agent_id、互不干扰。
_agent_pending: dict[str, asyncio.Task] = {}
_orphan_tasks: set[asyncio.Task] = set()

# close 路径结果槽：task_id → (act_recap, task_summary)（finalize Task 8 通过 pop_close_report 取用）
_close_report: dict[str, tuple[str, str]] = {}

# close 路径合成槽：task_id → (tool_call_id, scope, outcome, raw_fold_scope)
# finalize 先到时登记，bg 回调后替换 finish tool 记录；raw_fold_scope 非 None 时
# 替换成功后按它补删末段 raw（spec 2026-07-20 延迟折叠——占位 close 不即折）。
_close_synth: dict[str, tuple] = {}

_CLOSE_BOUNDARIES = {"finish", "normal"}

# 段边界折叠会 supersede 的 raw 类型（= apply_compact 的非保护类型；与 finalize._FINAL_RAW_TYPES
# 同构，本地定义避免与 finalize 交叉 import——finalize 已反向 import 本模块的 pop_close_report）
_SEGMENT_RAW_TYPES = [
    MemoryEventType.LLM_RESPONSE,
    MemoryEventType.TOOL_INVOCATION,
    MemoryEventType.TOOL_RESULT,
]


async def is_short_segment(state: "LoopState", ctx: "LoopContext") -> bool:
    """短段免折门：**当前段**（最后一条 active USER_PROMPT 之后）满足以下任一即为短段：

    - 段内 LLM 回复（role=assistant 回合）≤ 1 条——**不看 token**：一条回复折成摘要
      是净亏（recap 常比原文还长，且原文对下一轮信息更全），折它只是白花一次 LLM；
    - 段内 raw token ≤ short_segment_token_threshold。

    「短 → 原文成胶囊」决策（finalize._is_short_leaf）在段级的判定，
    `_run_background_observe`（interactive/interrupt 边界）与
    `observe._fold_retry_segment`（retry 段折）共用。配置缺失（手构 state /
    单测）→ False = 门关闭，照常折叠。

    段作用域（2026-07-21）：只数末条 UP 之后的 raw，与折叠的 since_last=USER_PROMPT
    对齐——免折残留的前段 raw 不计入，否则「前段累积 + 当前段极短」会被误判为可折，
    而折叠又只折当前段，产出比原文还长的摘要。
    """
    threshold = getattr(state.agent.loop_config, "short_segment_token_threshold", 0)
    if threshold <= 0:
        return False
    from ctx_weft.protocols import MemoryKind, MemoryScope

    # v2 P3a：TASK 视图（对话 + audit，无 SUMMARY——旧类型清单不含段摘要）升序；
    # 段界 = 末条 role=user 回合，其后即当前段 raw。
    view = await ctx.memory.load_view(
        state.scope, MemoryScope.TASK, ctx.provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.TOOL_AUDIT],
    )
    seg_records: list = []
    for r in view:
        if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "user":
            seg_records = []  # 新段界：清空重计
            continue
        seg_records.append(r)
    # 单回复段免折（2026-08-19）：token 再多也不折——见 docstring。
    n_llm = sum(1 for r in seg_records
                if r.kind is MemoryKind.CONVERSATION_TURN and r.role == "assistant")
    if n_llm <= 1:
        return True
    seg_text = " ".join(
        r.content if isinstance(r.content, str) else content_to_text(r.content)
        for r in seg_records
    )
    # 保留「join 后数一次」的文本口径（逐条估算会引入 4×N 的 framing 漂移），
    # 图片另行求和补上——不补则图片密集段被误判短段免折、该段 raw 永久保留。
    seg_images = sum(image_tokens(r.content) for r in seg_records)
    return ctx.llm.tokenizer.count(seg_text) + seg_images <= threshold


def pop_close_report(task_id: str) -> tuple[str, str] | None:
    """取走 close 路径产出的 (act_recap, task_summary)；不存在则返回 None。"""
    return _close_report.pop(task_id, None)


def register_close_synth(task_id: str, tool_call_id: str, scope, outcome: str,
                         raw_fold_scope=None) -> None:
    """finalize 先到时登记：finish 对已合成，待 bg 回调替换 Process Report。

    raw_fold_scope 非 None = close 时是占位 finish 对、末段 raw 未删（task scope），
    bg 替换成功后按它补删；bg 失败/无报告 → 不删（降级 = 保 raw）。"""
    _close_synth[task_id] = (tool_call_id, scope, outcome, raw_fold_scope)


def pop_close_synth(task_id: str) -> tuple | None:
    """bg 回调取走合成登记；不存在则返回 None。"""
    return _close_synth.pop(task_id, None)


async def _replace_finish_report(memory, provider_ctx, scope, task_id: str,
                                 tool_call_id: str, act_recap: str, task_summary: str,
                                 outcome: str, title: str, *, final_reply: str = "") -> None:
    """supersede 本 task 的 finish 对占位，按新 act_recap / task_summary 整体重写。
    按 (tool_call_id + origin_task_id) 定位，不再靠 'Process Report:' 文本（spec 2026-06-30 §2.4）。

    title：归属 task 的标题，用于重建 tool 槽的 `[task: …]` 前缀（与 finalize 合成占位时同源，
    见 finalize._finish_report_prefix）。本函数整条重写 tool 槽，不传就会把占位里的归属标记抹掉。

    final_reply：延迟折叠路径下末段 raw 此刻才被折，答复要随之补进锚点槽——占位是两槽，
    重写后升成三槽（recap / 答复+finish_task / 报告，见 finalize.build_finish_slots）。空则
    保持原形态；若占位已是三槽（close 即折路径），沿用已有锚点正文，**不得**被 act_recap 冲掉
    ——按 tool_call_id 找 assistant 命中的正是锚点那条。"""
    from datetime import timedelta

    from ctx_weft.core.loop.steps.finalize import build_finish_slots
    from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryKind, MemoryScope

    turns = await memory.load_view(
        MemoryAddress(session_id=scope.session_id, agent_id=scope.agent_id),
        MemoryScope.AGENT, provider_ctx, kinds=[MemoryKind.CONVERSATION_TURN],
    )
    asst = [r for r in turns
            if r.role == "assistant" and r.metadata.get("origin_task_id") == task_id
            and any(tc.get("id") == tool_call_id for tc in (r.metadata.get("tool_calls") or []))]
    tool = [r for r in turns
            if r.role == "tool" and r.metadata.get("origin_task_id") == task_id
            and r.metadata.get("tool_call_id") == tool_call_id]
    # 三槽占位里 recap 是**不挂 tool_calls** 的那条 assistant（挂着的是锚点）。
    recap_slot = [r for r in turns
                  if r.role == "assistant" and r.metadata.get("origin_task_id") == task_id
                  and not r.metadata.get("tool_calls")]
    if not asst and not tool:
        logger.warning("A1 _replace_finish_report: no finish 对 for task=%s tcid=%s; skip (best-effort)",
                       task_id, tool_call_id)
        return

    first = (recap_slot or asst or tool)[0]
    ts = first.timestamp
    parent_task_id = first.metadata.get("parent_task_id")
    # 已是三槽 → 沿用已有锚点正文（它已含前后两条注解，直接透传会被再包一层）。
    existing_anchor = next((r for r in asst if r.metadata.get("final_reply")), None)

    # 槽位形态与 close 合成同源（finalize.build_finish_slots）：答复在则三槽、不在则两槽。
    # v2 P3d：占位遗忘 + 新槽写入一次原子 fold（关旧「占位已删而真报告未写」窗口）。
    events = build_finish_slots(
        scope=scope, task_id=task_id, parent_task_id=parent_task_id, title=title,
        outcome=outcome, act_recap=act_recap, task_summary=task_summary,
        final_reply=final_reply, base=ts, tool_call_id=tool_call_id,
    )
    if existing_anchor is not None and not (final_reply or "").strip():
        # 占位已是三槽而本次没带答复：锚点正文原样保留，只换 recap 与报告两槽。
        events = [e for e in events if not e.metadata.get("final_reply")]
        events.insert(1, MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT, address=scope,
            content=existing_anchor.content, timestamp=ts + timedelta(microseconds=1),
            role="assistant",
            metadata={"origin_task_id": task_id, "parent_task_id": parent_task_id,
                      "tool_calls": existing_anchor.metadata.get("tool_calls"),
                      "final_reply": True},
        ))
    await memory.fold([r.id for r in (*recap_slot, *asst, *tool)], events, provider_ctx)


def _clear_pending(t: asyncio.Task, tid: str, aid: str = "") -> None:
    """Compare-and-clear：两个轴各自比对身份后再删，互不影响。

    比对身份（`is t`）而不是无条件 del：同一个 key 上可能已经被更晚的一次 launch 覆盖，
    那时这次 done 回调不该把别人的登记删掉。
    """
    if _task_pending.get(tid) is t:
        del _task_pending[tid]
        _task_pending_run_id.pop(tid, None)
    if aid and _agent_pending.get(aid) is t:
        del _agent_pending[aid]


def _lock_for(task_id: str) -> asyncio.Lock:
    lock = _task_locks.get(task_id)
    if lock is None:
        lock = asyncio.Lock()
        _task_locks[task_id] = lock
    return lock


async def _run_background_observe(state: "LoopState", ctx: "LoopContext", boundary: str) -> None:
    from ctx_weft.core.assembler import ContextRequest
    from ctx_weft.core.capabilities.control_tools import BACKGROUND_PROCESS_REPORT_NAME

    # Task 5 复审修复：origin 必须在本函数**任何**发射点之前钉住，不能拖到调
    # run_observe_react 前才改——RUN_STARTED/TASK_RECAP_STARTED（下面紧接着）以及
    # 两条早退路径（re-fold 幂等护栏、短段免折，均在拿到 origin 之前就 return）此前
    # 都带着调用方快照进来的 origin（正常路径是 LOOP_OBSERVE，act.py interrupt 边界
    # 路径是 LOOP_ACT）发出，host 按 origin 前缀过滤后台事件时会把它们误判成前台
    # 事件渲染进对话流（docs/events-v2.md §4）。这份 state 是 launch_background_observe
    # 给这段后台工作单独快照出的（独立 run_id/sequence_counter），改写 origin 不会
    # 污染主 run 的 state。
    state.origin = EventOrigin.LOOP_BACKGROUND_OBSERVE

    # 总账 C5（控制方裁定 R1）：`launch_background_observe` 给这段快照发了自己的
    # run_id（解 A4）——它在 host 眼里就是一段独立的 run，得有起有止，不能只补
    # recognize_intent / compact_agent 两处而漏掉它自己。payload 结构照抄
    # `_run_loop` 的实际发射点（`runtime.py`，见 task-5-report）；`initial_step`
    # 用它实际做的事 "background_observe"（这条路径没有 StepDriver，取不到真实
    # step 名）。TaskRecapStarted/Done 是任务级的 recap 记账，与这里的 run 级起止
    # 是两层，不互相顶替。
    await ctx.event_bus.emit(make_event(state, EventType.RUN_STARTED, payload={
        "run_id": state.run_id,
        "initial_step": "background_observe",
    }))
    await ctx.event_bus.emit(make_event(
        state, EventType.TASK_RECAP_STARTED,
        payload={"task_id": state.task.id, "boundary": boundary, "agent_id": state.agent.id},
    ))
    run_error: Exception | None = None
    was_cancelled = False
    try:
        async with _lock_for(state.task.id):
            # 重跑幂等护栏（恢复重跑时才生效）：非 close 边界若该段已无 active raw，说明上次
            # 崩溃前已折叠（raw 被 supersede），再折会产冗余胶囊 → 跳过（finally 仍发 DONE）。
            # 正常运行时该段刚产生 raw、计数 > 0，护栏为 no-op。
            if boundary not in _CLOSE_BOUNDARIES:
                from ctx_weft.protocols import MemoryKind, MemoryScope
                view = await ctx.memory.load_view(
                    state.scope, MemoryScope.TASK, ctx.provider_ctx,
                    kinds=[MemoryKind.CONVERSATION_TURN],
                )
                # 旧口径 = LLM_RESPONSE 计数 = assistant 回合
                n_raw = sum(1 for r in view if r.role == "assistant")
                if n_raw == 0:
                    logger.info(
                        "task recap re-fold guard: segment already folded (task=%s); skip",
                        state.task.id,
                    )
                    return
                # 短段免折（is_short_segment）：当前段（末条 UP 之后）active raw 低于阈值时
                # 跳过折叠——花一次后台 LLM 调用换一段常比原文还长的摘要不划算。跳过 = 该段
                # **永久**保 raw（与观察失败的降级同语义）：后续折叠带 since_last=USER_PROMPT
                # 只折各自的当前段，免折残留不会被跨段合折。
                if await is_short_segment(state, ctx):
                    logger.info(
                        "short segment kept raw (task=%s boundary=%s); skip fold",
                        state.task.id, boundary,
                    )
                    return
            try:
                agent = state.agent
                # 不能像 observe._llm_observe 那样用 has_agent() 短路：background observe 是
                # fire-and-forget（launch_background_observe → asyncio.create_task），常在
                # 本 run 的 _run_loop finally evict(agent.id) 之后才真正跑到这——per-agent 快照
                # 已被逐出，has_agent 为 False。控制工具（collect_process_report 等）是 session
                # 全局区（register_global，不随 evict 逐出），故这里应始终尝试 .get()（内部自动
                # 合并全局区），不能因 per-agent 快照缺失就整体清零、连全局控制工具也丢了。
                bound_caps = (
                    ctx.capability_cache.get(agent.id)
                    if ctx.capability_cache is not None
                    else []
                )
                request = ContextRequest(
                    purpose="background_observe",
                    scope=state.scope,
                    task=state.task,
                    agent=agent,
                    session=state.session,
                    template=state.extra.get("template"),
                    bound_capabilities=bound_caps,
                    token_counter=ctx.llm.tokenizer.count,
                    extra={"observe_boundary": boundary},
                )
                prompt = await ctx.assembler.assemble(request)
                # Task 5：不再有 event_types 间接层——LLM_* 的「后台 vs 前台」区分改靠
                # state.origin（EventOrigin.LOOP_BACKGROUND_OBSERVE，已在函数入口钉住），
                # 由 host 据此决定后台 observe 的 LLM 交互不进前端对话流。
                result, last_text = await run_observe_react(
                    state, ctx,
                    system=prompt.system,
                    messages=list(prompt.messages),
                    tools=prompt.tools,
                    max_rounds=agent.loop_config.max_turns_per_observe,
                    terminal_tool_name=BACKGROUND_PROCESS_REPORT_NAME,
                )
                # 报告取值：terminal 工具产出 → 纯文本复述兜底（observer 把复述写成正文而没调工具）。
                # content_to_text：InvocationResult.content 可能是 list[ContentPart]
                # （provider 的 result content 里带了图，任何 provider 都可能）——对 list
                # 直接 `.strip()` 会 AttributeError 掀掉后台 observe。str 输入原样返回。
                act_recap = (content_to_text(result.content if result else "") or last_text or "").strip()
                task_summary = (result.metadata or {}).get("task_summary", "") if result else ""
                if not act_recap:
                    # 无任何可用报告：与异常路径同语义——段保 raw，不写占位摘要、不动 finish 对。
                    if boundary in _CLOSE_BOUNDARIES:
                        pop_close_synth(state.task.id)  # 弹掉登记防泄漏；finalize 占位 finish 对保持原样
                    logger.warning(
                        "background observe produced no usable report (task=%s boundary=%s); "
                        "segment kept raw", state.task.id, boundary,
                    )
                    return
                if boundary in _CLOSE_BOUNDARIES:
                    synth = pop_close_synth(state.task.id)  # sync check-and-clear（无 await）
                    if synth is not None:
                        tool_call_id, scope, outcome, raw_fold_scope = synth
                        from ctx_weft.core.loop.steps.finalize import _output_text
                        await _replace_finish_report(
                            ctx.memory, ctx.provider_ctx, scope, state.task.id,
                            tool_call_id, act_recap, task_summary, outcome,
                            state.task.title or "",
                            # raw 此刻才折 → 答复随之补进锚点槽（占位两槽升三槽）
                            # getattr 兜底：本路径异常被整段吞掉并「保 raw」，缺字段的
                            # task 替身若在此炸 AttributeError，真报告就永远替换不上去。
                            final_reply=(_output_text(getattr(state.task, "outputs", None))
                                         if raw_fold_scope is not None else ""),
                        )
                        if raw_fold_scope is not None:
                            # 真摘要已替换进 finish 对 → 补删末段 raw（延迟折叠收口，
                            # spec 2026-07-20：占位 close 不即折，真报告落地才折）。
                            from ctx_weft.core.loop.steps.finalize import (
                                _supersede_final_raw_segment,
                            )
                            await _supersede_final_raw_segment(
                                ctx.memory, raw_fold_scope, ctx.provider_ctx)
                    else:
                        # root 的 finish/normal 是终结点（单次 close）：槽写一次弹一次，不存在
                        # 跨 rerun 乱序覆盖（retry 仅在机械退出时产生，不经此路径）。
                        _close_report[state.task.id] = (act_recap, task_summary)  # 不写 memory（不变量 3）
                else:
                    # v2 P3c：策展上移——段作用域折叠（护 user 回合与既有段摘要、锚点/
                    # 段尾语义，与 observe._fold_retry_segment 同门）由框架侧 segment_fold
                    # 执行原子 fold。旧 apply_compact 的 TypeError 协议错配特判随之消亡
                    # （segment_fold 是框架内函数，签名错配不再是运行时 provider 风险）。
                    from ctx_weft.core.loop.steps.segment_fold import segment_fold
                    await segment_fold(
                        ctx.memory, state.scope, MemoryScope.TASK, act_recap,
                        ctx.provider_ctx,
                    )
            except Exception:
                # close 边界防泄漏：finalize 可能已 register_close_synth，本次失败后永远无人
                # 消费（task_id 唯一 + close 单入口），弹掉——与「无可用报告」分支对称。
                # 已知残余窗口（接受，不另引状态同步）：bg 比 finalize 先死时登记发生在 pop
                # 之后，仍漏一条；对称地，bg 先写 _close_report 而 finalize 异常中止也漏。
                # 两者触发概率与单次代价（几百字节/次）都低一个量级。
                if boundary in _CLOSE_BOUNDARIES:
                    pop_close_synth(state.task.id)
                logger.exception("background observe failed (ignored); segment kept raw")
    except asyncio.CancelledError:
        # F2：CancelledError 不是 Exception 子类（3.8+ 起继承 BaseException），上面
        # `except Exception` 接不住它——不接住就意味着 run_error 仍是 None，下面
        # RunFinished 会把一次真取消谎报成 completed。同一口径抄 runtime.py::_run_loop
        # 的 `except asyncio.CancelledError`（R1 定下）：记下来、不重新抛出——这段 run
        # 本就是 fire-and-forget，没有调用方在等它的异常，吞掉与 _run_loop 一致。
        was_cancelled = True
    except Exception as exc:
        # 上面那个 except 只吞真正跑出 fold/observe 的失败（业务已降级 = 段保
        # raw，run 仍算跑完）；这里接的是护栏段（幂等检查 / is_short_segment）本身
        # 炸的异常——那条既有行为是原样往外传，不吞。只是顺手记一笔，好让下面的
        # RunFinished 如实报 outcome=interrupted，不撒谎报 completed。
        run_error = exc
        raise
    finally:
        await ctx.event_bus.emit(make_event(
            state, EventType.TASK_RECAP_DONE, payload={"task_id": state.task.id},
        ))
        await ctx.event_bus.emit(make_event(state, EventType.RUN_FINISHED, payload={
            "outcome": (
                RunOutcomeKind.CANCELED.value if was_cancelled
                else RunOutcomeKind.COMPLETED.value if run_error is None
                else RunOutcomeKind.INTERRUPTED.value
            ),
            "final_status": state.task.status,
            "will_retry": False,
            "total_events": state.sequence_counter,
            "total_turns": len(state.transcript),
            "error": str(run_error) if run_error else None,
            "error_type": type(run_error).__name__ if run_error else None,
        }))


def launch_background_observe(
    state: "LoopState", ctx: "LoopContext", *, boundary: str
) -> asyncio.Task:
    # 总账 A4：旧写法只 `dataclasses.replace(state)`——run_id 与主 run 相同，但
    # sequence_counter 是一份独立的 int 副本（普通字段，不可变），此后两边各自
    # +=1，(run_id, sequence) 就会撞号。后台 recap 本来就是一段独立的工作，有自己
    # 的起止事件（TaskRecapStarted/Done，现在还加了 RunStarted/Finished），不该
    # 蹭主 run 的号——给它自己的 run_id、序号从 0 重开。
    snapshot = dataclasses.replace(
        state,
        run_id=generate_id("run"),
        sequence_counter=0,
    )
    task = asyncio.create_task(_run_background_observe(snapshot, ctx, boundary))
    _task_pending[state.task.id] = task
    _task_pending_run_id[state.task.id] = snapshot.run_id
    _agent_pending[state.agent.id] = task
    tm = getattr(ctx, "task_manager", None)
    if tm is not None and hasattr(tm, "track_background"):
        tm.track_background(task)
    else:
        _orphan_tasks.add(task)
        task.add_done_callback(_orphan_tasks.discard)
    task.add_done_callback(
        lambda t, tid=state.task.id, aid=state.agent.id: _clear_pending(t, tid, aid))
    return task


async def await_pending_background_observe(task_id: str) -> None:
    """等该 task 在途后台 observe 完成（强一致）。调用点：`_run_loop` 入口（覆盖常规
    `prepare` 与 `reconcile` dangling tool_call 重放两条 resume 路径，`runtime.py`）、
    用户冷应答注入（`_inject_user_reply`，`runtime.py`）。"""
    pending = _task_pending.get(task_id)
    if pending is not None and not pending.done():
        await asyncio.shield(pending)


async def await_pending_background_observe_for_agent(agent_id: str) -> None:
    """等该 **agent** 在途后台 observe 完成。

    与 `await_pending_background_observe(task_id)` 是同一份强一致保证的两个轴，一起用：
    task 轴覆盖「同一个 task 的下一轮 run」（retry / resume / reconcile 重放），agent 轴
    覆盖「同一个 agent 的**下一个 task**」——后者正是 `send_message` 在 agent 已终态时
    走 `_start_task_for_agent` 建新 task 的那条路，task 轴够不着（见 `_agent_pending`）。

    两轴常常指向同一个对象（同 task 的下一轮），重复 await 无害：第二次 `pending.done()`
    为真，直接返回。
    """
    pending = _agent_pending.get(agent_id)
    if pending is not None and not pending.done():
        await asyncio.shield(pending)


def pending_background_observe_run_id(task_id: str) -> str | None:
    """非阻塞地探一眼：该 task 此刻是否有一个还没跑完的后台 observe（同步字典读，无 await）；
    有就返回**那次 launch** 的 `run_id`，没有返回 `None`。

    `TurnHandle.wait_for_finish`（`runtime.py`）用它决定终态事件到达后要不要在**同一条
    事件流订阅**上继续等对应的 `TaskRecapDone`——而不是另起一个等这个 asyncio.Task 本身
    （`await_pending_background_observe`/`asyncio.shield`）的等待者。两者看似等价，调度
    顺序却不同：`TaskManager._fire_session_done` 也在等这同一个后台任务（`asyncio.gather`），
    且它的等待**总是先注册**（`_run_task` 那条协程从发 TaskFinished 到那次 gather 之间不
    过几行同步代码，中途不会把控制权交还事件循环）；若 `wait_for_finish` 也去等**同一个
    Task 对象**，两个等待者的回调都挂在它的完成清单上，`_fire_session_done` 那个先注册、
    先被唤醒——它会在 `wait_for_finish` 之前跑完 `on_session_done`，抢先把这一轮判成
    收尾（2026-09-08 之前那一步还连着 `_release_session`，会把 session 连同 agent
    record 一并拆掉，后果更重；现在只剩清控制信号，但顺序问题本身没变）。
    改成等**事件总线上的 TaskRecapDone**
    则不出现这个问题：后台 observe 在 `finally` 里先 `emit(TaskRecapDone)`（这一步只是把
    事件塞进订阅者各自的队列，不等任何人处理）、之后才真正从协程函数 return、其
    `asyncio.Task` 才转入 done 态——事件总线的那次唤醒排在 Task-done 的唤醒之前，
    `wait_for_finish` 借着「早就在等这条流」的事件订阅，比 `_fire_session_done` 的
    `gather` 更早被唤醒返回，不会撞见会话已经被拆完的中间态。

    **为什么要返回 run_id、不能只返回 bool（2026-09-04 二轮修复）**：`_task_pending`
    只挂**最新一次** launch——同一个 task 的一生里可能有多次 fire-and-forget 后台
    observe（`interrupt`/`mechanical`/`dispatch`/`finish` 等不同 boundary，跨
    suspend/resume、重试多轮发生），彼此不重叠是**通常**情况，不是**保证**情况。
    若这次 close 边界 launch 之前，上一次的后台任务碰巧还没来得及被下一轮 `_run_loop`
    入口的 `await_pending_background_observe` 收口（例如两次 launch 之间调度得足够
    密集），`wait_for_finish` 在终态事件之后看到的第一个 `TaskRecapDone` 可能是**那次
    更早的 launch** 发的，不是我们真正要等的这次——只按事件类型匹配会在这种重叠窗口里
    复现同一类「提前返回」问题，只是窗口更窄。`launch_background_observe` 给每次
    launch 都铸一个独立 `run_id`（`dataclasses.replace(state, run_id=generate_id(
    "run"), ...)`），`make_event` 把它写进事件信封的 `run_id` 字段——`TaskRecapDone`
    也不例外。返回这个 run_id，让调用方把匹配钉死到「这一次 launch 发的 TaskRecapDone」，
    而不是「随便哪次 launch 发的、类型对得上的事件」。"""
    pending = _task_pending.get(task_id)
    if pending is None or pending.done():
        return None
    return _task_pending_run_id.get(task_id)
