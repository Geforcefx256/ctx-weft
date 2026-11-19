"""ActStep：多 turn LLM 子循环 + capability 调用。

工具调用通过 CapabilityGateway 统一执行（授权 + 审计 + 内存），
ActStep 只处理 LLM 流 + turn 循环逻辑。
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ctx_weft.core.assembler.context import PromptContext
from ctx_weft.core.assembler.prompts import runtime_prompt
from ctx_weft.core.capabilities.control_tools import SWITCH_TASK_NAME
from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
from ctx_weft.core.loop.commit import commit_round, discard_uncommitted_round
from ctx_weft.core.loop.driver import (
    ACT_TURNS_USED_KEY,
    COMPACT_NOOP_KEY,
    CONTEXT_RECOVERY_COUNT_KEY,
    LoopContext,
    LoopState,
    Step,
    StepDriver,
    StepOutcome,
    make_event,
)
from ctx_weft.core.loop.llm_gateway import (
    PROMPT_EST_BASE_KEY,
    PROMPT_EST_SEG_KEY,
    build_llm_request,
    llm_request_id,
    resolve_llm_identity,
    stream_llm_resilient,
)
from ctx_weft.core.loop.llm_stream import ensure_collected_stream
from ctx_weft.core.loop.signals import HitlPark
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import MintedCall, generate_id, mint_turn_call_ids
from ctx_weft.protocols import (
    LLMMessage,
    LLMUsage,
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryScope,
    ToolCall,
    load_records,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import (
    HITL_FORM_WAIT,
    PREFACE_AFTER_INTERRUPT,
    PREFACE_AFTER_INTERRUPT_EDIT,
    PREFACE_NORMAL,
    HitlAsk,
    UserTurnDelivery,
)

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """Single LLM turn record (for Observer + memory)."""

    turn: int
    messages_sent: list[LLMMessage]
    assistant_text: str
    tool_calls: list[ToolCall] = dataclasses.field(default_factory=list)
    tool_results: list[Any] = dataclasses.field(default_factory=list)
    usage: LLMUsage = dataclasses.field(default_factory=LLMUsage)


@dataclass
class PendingActTurn:
    """A persisted, accounted LLM turn whose tools await foreground compaction."""

    record: TurnRecord
    assistant_record_id: str
    context_limit_hit: bool


class ActStep(Step):
    """ActStep: multi-turn LLM loop + capability invocation via CapabilityGateway."""

    name = "act"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        inputs = state.prompt_inputs
        if inputs is None:
            raise RuntimeError("ActStep requires prompt_inputs — PrepareStep must run first")

        agent = state.agent
        # 轮数预算**跨「上下文恢复」累计**：越过停机线后本步会经 PrepareStep 压缩再重入
        # （见 `_can_recover_context`），若每次重入都从 1 重计，等于每恢复一次白送一整份
        # `max_turns_per_act`——而这个配置的语义是「一次执行最多几轮」，不是「每段最多几轮」。
        # `turn_num` 同样用累计值，使 ACT_TURN_STARTED 的轮号在一个 run 内单调递增。
        max_turns = agent.loop_config.max_turns_per_act
        turns_used = state.extra.get(ACT_TURNS_USED_KEY, 0)
        # transcript 同样累加而非覆盖：state_patch 是整字段覆盖，重入时本地变量若从空起头，
        # 压缩前那些轮次就从 state.transcript 消失了（消费方：observe 的
        # `_mechanical_verdict` 空 transcript → fail、background_observe 的 total_turns）。
        transcript: list[TurnRecord] = list(state.transcript)
        exit_reason = "normal"
        reply_anchor = None

        # 运行时态势 guidance（任务树/已完成子任务/收尾提醒）已随装配管线注入
        # （PrepareStep → GuidanceSource → composer 末条 user 尾部），此处不再修饰。
        appended_messages: list[LLMMessage] = []
        # 动态 max_tokens 的增量基线：上一轮实际发送的 message 条数（= 本轮 usage 对应的真实
        # prompt 基线）。None=本轮无循环内真实基线（首轮）→ request_prompt_estimate 走整份估算。
        baseline_msg_count: int | None = None

        pending = getattr(state, "pending_act_turn", None)
        while pending is not None or turns_used < max_turns:
            if pending is not None:
                await _interrupt_checkpoint(state, ctx)
                turn_record = pending.record
                turn_num = turn_record.turn
                context_limit_hit = pending.context_limit_hit
                state.extra["assistant_record_id"] = pending.assistant_record_id
            else:
                turn_num = turns_used + 1
                turns_used = turn_num
                state.extra[ACT_TURNS_USED_KEY] = turn_num
                await _interrupt_checkpoint(state, ctx)
                await ctx.event_bus.emit(make_event(
                    state, EventType.ACT_TURN_STARTED, payload={"turn": turn_num}))

                # 1) 单轮 LLM：流式累积文本 / reasoning / tool_calls / usage（软打断在内部 park）
                turn = await _run_llm_turn(state, ctx, inputs, appended_messages, turn_num, baseline_msg_count)

                # 2) token 记账 + context_limit 判定
                context_limit_hit = await _account_tokens(state, ctx, turn.usage)
                # 仅当本轮真实刷新了 context_tokens（usage>0）才前移基线；否则保持旧值/None，
                # 让下一轮退回整份估算（更保守），避免基线与陈旧 context_tokens 错配。
                if turn.usage.prompt_tokens > 0:
                    baseline_msg_count = len(turn.messages_sent)

                # 3) assistant 回合落 memory + 接回 message 历史 + 记 transcript
                persisted_turn = turn.persisted_turn
                if persisted_turn is None:
                    persisted_turn = await _ingest_assistant_turn(
                        state, ctx, turn.text, turn.tool_calls, turn.usage, turn_num,
                        anchor=turn.anchor, minted=turn.minted, timestamp=turn.memory_timestamp)
                asst_tool_dicts = persisted_turn.tool_calls
                state.extra["assistant_record_id"] = persisted_turn.record_id
                appended_messages.append(LLMMessage(
                    role="assistant", content=turn.text, tool_calls=asst_tool_dicts,
                    reasoning_content=turn.reasoning or None, provider_state=turn.provider_state,
                    memory_id=persisted_turn.record_id))
                turn_record = TurnRecord(
                    turn=turn_num, messages_sent=[*turn.messages_sent, appended_messages[-1]],
                    assistant_text=turn.text, tool_calls=turn.tool_calls, usage=turn.usage)

                # 4) 纯文本（无 tool call）：interactive 让位用户 / 否则即任务产出 → observe
                #
                # **这一支刻意排在 context_limit 之前**（改造前相反）：纯文本就是本 run 的收尾，
                # 模型已经把答复交出来了，此刻压缩没有任何收益——而让 context_limit 抢先的代价是
                # 把一次正常收尾误报成机械退出，经 observe 强制改判 retry、白吃一次重试预算。
                if not turn.tool_calls:
                    transcript.append(turn_record)
                    reply_anchor = ({"response_event_id": getattr(turn, "response_event_id", ""),
                                     "response_memory_id": persisted_turn.record_id}
                                    if turn.text.strip() and getattr(turn, "response_event_id", "") else None)
                    await _finish_plain_text_turn(state, ctx, turn_num, transcript,
                                                  reply_anchor=reply_anchor)
                    break

                compact_options = _predispatch_compact_options(state, turn.tool_calls, turn.usage)
                if compact_options is not None:
                    await commit_round(state, ctx)
                    return StepOutcome(
                        next_step="compact", next_step_options=compact_options,
                        state_patch={
                            "transcript": transcript, "prompt_inputs": None,
                            "pending_act_turn": PendingActTurn(
                                turn_record, persisted_turn.record_id, context_limit_hit),
                        },
                    )

            # 6) 执行 tool calls（同批次延后退出信号，保证 tool_call ↔ result 一一对应）
            #
            # **越过停机线也照执行完**（改造前在 4) 就 break 了，工具一个都不跑）。三条理由：
            # ① 工具调用已经发生过，结果是净收益，丢掉等于白烧一轮 LLM；
            # ② 不执行就没有 TOOL_RESULT，memory 里留下一条**悬挂的** assistant tool_call。
            #    改造前它被 observe 的 `_fold_retry_segment`（keep_last=0）连整段 raw 一起
            #    supersede 掉，坑被掩埋；一旦改成压缩后原地续跑，`collapse_keep_last` 会把它
            #    留在保留区，重装配出的 messages 就带一个无 result 的 tool_call，撞上
            #    `llm_gateway.drop_dangling_tool_calls`——不 400（有兜底），但那条 assistant
            #    会被剥成空消息再被 `remove_empty_messages` 整条删掉，模型看不到自己调过这些
            #    工具，且每次恢复都打一条 ERROR（那个兜底的语义是「上游对账漏了」）。
            # ③ 执行完可能进一步超出窗口，但这是安全的：装配预算层把 assistant(tool_calls)
            #    与其 tool 结果聚成同生共死单元再整单元丢弃（`budget._coalesce_tool_pairs`），
            #    永不产孤儿；且紧接着就是 compact，折的正是这批新料。
            tool_results = await _execute_tool_calls(state, ctx, turn_record.tool_calls)
            await _record_hidden_call_completion(state, ctx, turn_record.tool_calls)
            for tr in tool_results:
                appended_messages.append(LLMMessage(
                    role="tool", content=tr["result"], tool_call_id=tr["tool_call_id"],
                    memory_id=tr.get("memory_id")))
            state.pending_act_turn = None
            pending = None
            turn_record.tool_results = tool_results
            transcript.append(turn_record)
            await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
                "turn": turn_num,
                "reason": "context_limit" if context_limit_hit else "tool_calls_processed"}))


            # 7) actor 收尾 —— 同样**优先于** context_limit：任务已经做完了，没有可续的跑。
            if state.task.switch_requested:
                exit_reason = "task_switch"
                break
            if state.task.suspend_requested:
                exit_reason = "suspended"
                break
            if state.task.actor_done:
                exit_reason = "actor_done"
                break

            # 8) 越过上下文停机线：工具结果已全部落定、配对完整 → 结束本段。
            #    去向由 `_can_recover_context` 决定（回 prepare 压缩续跑 / 退回 observe），
            #    见下方路由。**不清 pin**：`_pinned` 按 task_id 分区（新 task 看不到旧 pin）、
            #    有 max_pins 的 LRU 上界、且 task 落终态时由 `on_task_finished` 清——在这里
            #    清等于让 agent 续跑后从零重新发现工具，正是 `CapabilityCache.evict` 的注释
            #    明确拒绝在 run 边界做的那件事，而续跑连 run 边界都不跨。
            if context_limit_hit:
                exit_reason = "context_limit"
                logger.warning(
                    "ActStep context_limit_hit: prompt_tokens=%d >= %.2f * eff(limit=%d) "
                    "for agent %s", turn_record.usage.prompt_tokens,
                    getattr(agent.loop_config, "context_limit_stop_ratio", 0.8),
                    state.resolved_model.context_limit, agent.id)
                break

        else:
            exit_reason = "max_turns"
            logger.warning("ActStep hit max_turns=%d for agent %s", max_turns, agent.id)
            await ctx.event_bus.emit(make_event(state, EventType.MAX_TURNS_REACHED, payload={
                "max_turns": max_turns}))

        # 离开 act 的循环 = 这一轮已经推进到别的步骤（suspend / prepare 压缩续跑 / observe），
        # 那些步骤直接读 memory、不认暂存区，所以先提交：暂存的答复与消息此刻落盘。
        # 热应答醒来后一个 chunk 都没等到就走到这里（例如同批的 finish_task），也算数。
        await commit_round(state, ctx)

        if exit_reason == "task_switch":
            from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind

            state.transcript = transcript
            state.task.switch_review_started = True
            await StepDriver.for_context(ctx).start_durable(
                "observe", state, ctx, options={"boundary": "task_switch"})
            return StepOutcome(None, state_patch={
                "transcript": transcript, "act_exit_reason": exit_reason,
                "pending_act_turn": None,
                "run_outcome": RunOutcome(kind=RunOutcomeKind.TASK_SWITCHED),
            })

        # Delegation parks first; recover context through Prepare, otherwise review.
        if state.task.suspend_requested:
            next_step = "suspend"
        elif exit_reason == "context_limit" and _can_recover_context(state):
            state.extra[CONTEXT_RECOVERY_COUNT_KEY] = (
                state.extra.get(CONTEXT_RECOVERY_COUNT_KEY, 0) + 1)
            next_step = "prepare"
        else:
            next_step = "observe"

        # ── task.outputs：收尾时合成最终交付物（spec 2026-07-01 反转契约）──
        # Persist an actual final reply; limit and delegation boundaries are not deliverables.
        if exit_reason in ("normal", "actor_done"):
            _synthesize_final_outputs(state, transcript)

        return StepOutcome(
            next_step=next_step,
            next_step_options=({"boundary": exit_reason, "judge_task": True,
                                **({"reply_anchor": reply_anchor} if reply_anchor else {})}
                               if next_step == "observe" else {}),
            state_patch={
                "transcript": transcript,
                "act_exit_reason": exit_reason,
                "pending_act_turn": None,
            },
        )


def _compose_final_outputs(transcript: list[TurnRecord]) -> tuple[str, str]:
    """Extract the last actual assistant answer and any legacy completion summary."""
    last = transcript[-1]
    body = (last.assistant_text or "").strip()
    if not body:
        body = next(
            (t.assistant_text.strip() for t in reversed(transcript)
             if (t.assistant_text or "").strip()),
            "",
        )
    # Historical response events can still carry the former completion summary.
    # New actors cannot call that tool, but recovery must retain old deliverables.
    summary = ""
    for call in last.tool_calls:
        if call.name == "control__finish_task":
            value = (call.arguments or {}).get("deliverables_summary", "")
            summary = value.strip() if isinstance(value, str) else ""
            break
    return body, summary



async def _stream_until_stop(agen: Any, ctx: LoopContext) -> "AsyncIterator[Any]":
    """按 chunk 迭代 LLM 流；**等待下一个 chunk 的过程中**也能被软打断/硬取消掐断。

    改造前这里是裸 `async for`，暂停检查写在循环体里——只有 chunk 到达才执行得到。
    于是请求发出到首个 chunk 之间（TTFT；思考模型、provider 排队、冷路由都能到几十秒）
    协程挂在 `__anext__` 上，`pause_token` 被置位也没有任何人读它：用户按下暂停毫无反应。
    退避重试期间更长——gateway 的自愈最多 300s，adapter 自己的 HTTP 重试还能各睡 60s，
    那两段同样一个 chunk 都不产出。

    这里改成与 `_await_tool_or_stop` **同一形状**（那条路早就是对的：在跑的工具能被当场
    掐掉）：把 `__anext__` 和两个停止信号赛跑。

    停止信号胜出 → 取消挂起的 `__anext__`。那个 `CancelledError` 会打进生成器内部当前
    挂着的 await —— 包括 adapter 退避里的 `asyncio.sleep` 与 httpx 的
    `async with client.stream(...)`，后者退出上下文即**真正掐断这次请求**，不再空烧
    token。`CancelledError` 是 `BaseException`，gateway 的 `except LLMCallError` 与
    adapter 的各处 `except Exception` 都拦不住它（已核实两处均无裸 `except`）。

    本函数**不判断是 pause 还是 cancel**，只负责停下来并收干净——分流由调用方在循环
    之后做，判据仍是那两个 token 本身，不引入第二处真相。
    """
    waiters: list[asyncio.Future] = []
    if ctx.cancel_token is not None:
        waiters.append(asyncio.ensure_future(ctx.cancel_token.wait()))
    if ctx.pause_token is not None:
        waiters.append(asyncio.ensure_future(ctx.pause_token.wait_paused()))
    nxt = None
    try:
        while True:
            nxt = asyncio.ensure_future(agen.__anext__())
            if waiters:
                await asyncio.wait({nxt, *waiters}, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.wait({nxt})
            if not nxt.done():
                # 停止信号先到：掐断挂起的取流，等它真的收完尾再走。
                nxt.cancel()
                await asyncio.wait({nxt})
                return
            try:
                chunk = nxt.result()
            except StopAsyncIteration:
                return
            yield chunk
    finally:
        if nxt is not None:
            if not nxt.done():
                nxt.cancel()
            await asyncio.gather(nxt, return_exceptions=True)
        for w in waiters:
            w.cancel()
        if waiters:
            await asyncio.wait(set(waiters))
        # 生成器可能仍挂在某个 yield 上（调用方 break 出去时）：关掉它，让 httpx 的
        # 流上下文退出。已经被上面的 cancel 拆完的生成器对此是 no-op。
        await agen.aclose()


@dataclass
class _LLMTurnOutput:
    """单轮 LLM 流式产出。"""

    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    usage: LLMUsage
    # 摄入锚（预铸的 assistant 记录 id，ingest 按 MemoryEvent.id 采纳）与铸造伴随。
    # tool_calls 内的 id 已是内部标识；minted 保留原始 wire id / ordinal 供落库伴随字段。
    anchor: str = ""
    response_event_id: str = ""
    minted: list[MintedCall] = dataclasses.field(default_factory=list)
    provider_state: dict[str, Any] = dataclasses.field(default_factory=dict)
    messages_sent: list[LLMMessage] = dataclasses.field(default_factory=list)
    memory_timestamp: datetime | None = None
    persisted_turn: PersistedAssistantTurn | None = None


async def _run_llm_turn(
    state: LoopState, ctx: LoopContext, inputs: PromptContext, appended_messages: list[LLMMessage],
    turn_num: int, baseline_msg_count: int | None = None,
) -> _LLMTurnOutput:
    """发请求事件 → 流式累积 token/reasoning/tool_calls/usage → 处理软打断 → 发 RESPONSE_FINISHED。

    软打断（pause）时提交半截 assistant 文本并 park（raises HitlPark），不返回；硬取消 →
    CancelledError。正常结束返回 _LLMTurnOutput。
    """
    agent = state.agent
    model, llm_account = resolve_llm_identity(state)
    # req_id: 与 llm_gateway.stream_llm_resilient 内部用同一个确定性公式独立算出——两边都在
    # 「本次 LLM 调用的任何事件被发射之前」求值（此处在调用 stream_llm_resilient 之前；gateway
    # 侧在其 while 重试循环、也就是第一次 emit 之前），故 state.sequence_counter 两处读到同一个
    # 值，无需新增参数或跨函数传值即可对齐（详见 llm_gateway.stream_llm_resilient 文档字符串）。
    req_id = llm_request_id(state)

    llm_request = await build_llm_request(
        ctx, state, inputs, appended_messages=appended_messages,
        baseline_msg_count=baseline_msg_count, turn=turn_num)
    # turn：LLM_REQUEST_STARTED / LLM_PROMPT_SENT 现由 gateway 发射，但 turn_num 是本函数的局部
    # 循环变量、state 上没有对应字段——经 metadata 这个既有的瞬态透传通道带给 gateway（同
    # PROMPT_EST_BASE_KEY/PROMPT_EST_SEG_KEY 的做法），不新增函数参数。

    interrupted = False
    stream = ensure_collected_stream(stream_llm_resilient(
        ctx, state, llm_request, inputs=inputs, appended_messages=appended_messages))

    # A stop and a chunk can arrive together. A body exception/break does not
    # close an async generator automatically; await closure before run unwind.
    async with aclosing(_stream_until_stop(stream, ctx)) as chunks:
        async for chunk in chunks:
            tok = ctx.cancel_token
            if _interrupt_pending(ctx):
                interrupted = True
                break
            if tok is not None and tok.is_cancelled:
                tok.raise_if_cancelled()
            # Any first chunk commits the round, including a tool call with no text.
            await commit_round(state, ctx)
            if chunk.kind == "token":
                ctx.run_phase.produced = True
            stream.accept_current()

    result = stream.partial_result()
    text, reasoning, usage = result.text, result.reasoning, result.usage
    tool_calls = list(result.tool_calls)

    # 流在**首个 chunk 到达之前**被停止信号掐断时，上面的循环体一次都没执行到——
    # 判定必须补在循环之后，否则一次 TTFT 窗口里的暂停会被当成「流正常结束」。
    if not interrupted and _interrupt_pending(ctx):
        interrupted = True
    tok = ctx.cancel_token
    if tok is not None and tok.is_cancelled:
        tok.raise_if_cancelled()

    if interrupted:
        # ⓪ 一个 chunk 都没到就被中止 → 整轮丢弃，不 park、不留任何痕迹。
        await discard_uncommitted_round(state, ctx)   # 命中则 raises RoundDiscarded
        # ② 已吐 token：把半截 assistant 文本入 memory 并标注「被用户打断」；
        # ① 已开口但本回合没吐正文（纯工具调用后被打断）：不留记录。随后 park 待用户续接。
        has_partial = bool(text.strip() or reasoning.strip())
        await _commit_interrupted_partial(state, ctx, text, reasoning, turn_num)
        await _park_for_interrupt(state, ctx, edit=not has_partial)

    # 摄入前铸造内部调用标识（spec: conversation-integrity）：LLM 复用 wire id（call_1
    # 类短值）是常态，跨轮次/跨任务重建后按裸 id 配对会错配。本函数是唯一供值点——
    # 事件 payload、memory 记录、appended_messages、gateway/HITL 全部消费同一次铸造的
    # 值。锚 = 预铸的 assistant 记录 id（ingest 经 MemoryEvent.id 采纳，见
    # _ingest_assistant_turn），跨平面一致且与 tool_call_id 派生同源。observe 的
    # ObserveStep._run_react 对自己的回合同口径（live 平面，不入 memory）。
    anchor = generate_id("asst")
    minted: list[MintedCall] = []
    if tool_calls:
        minted = mint_turn_call_ids(tool_calls, anchor=anchor, turn_seq=turn_num)
        tool_calls = [m.call for m in minted]

    # token 自校准回喂：真实 usage 与发送前估算段作比（基线不参与），喂给该模型 tokenizer。
    # 估算段取 PROMPT_EST_SEG_KEY（tokenizer.count 直接产出）而非
    # prompt_token_estimate − base——整份路径上返回值可能被 max(full, ctx_tokens) floor 成
    # 上一轮真实值，拿 floor 后的值回喂会产出 ratio≈1 的假样本，把伺服系统性拖向 1。
    base = llm_request.metadata.get(PROMPT_EST_BASE_KEY)
    est_seg = llm_request.metadata.get(PROMPT_EST_SEG_KEY)
    if usage.prompt_tokens > 0 and est_seg is not None and base is not None:
        ctx.llm.tokenizer.observe(est_seg, usage.prompt_tokens - base)
        # spec: tool-schema-budget——偏差留痕（SHALL）：工具面分项（指纹/估算）与实测同现，
        # 供校准与回归分析（R4「偏差可归因」）。
        guard = getattr(agent, "loop_guard", None)
        logger.info(
            "token estimate feedback: est_seg=%s base=%s actual_prompt=%d "
            "tools_signature=%s tools_est=%d",
            est_seg, base, usage.prompt_tokens,
            getattr(guard, "last_tools_signature", ""),
            getattr(guard, "last_tools_est", 0))

    finished = make_event(
        state, EventType.LLM_RESPONSE_FINISHED,
        payload={
            "request_id": req_id, "content": text, "reasoning": reasoning,
            "memory_id": anchor,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ],
            "usage": dataclasses.asdict(usage),
            # 本次调用实际使用的模型/账号（host 云端上报按此计账，不再受切换竞态影响）
            "llm_model": model, "llm_account": llm_account,
            "finish_reason": "tool_use" if tool_calls else "stop", "turn": turn_num})
    persist = getattr(ctx, "persist_assistant_turn", None)
    persisted_turn = None
    if persist is None:
        await ctx.event_bus.emit(finished)
    else:
        event, tool_dicts = _assistant_memory_event(
            state.scope, text, tool_calls, usage, turn_num,
            anchor=anchor, minted=minted, timestamp=finished.timestamp)
        await persist(state, ctx, finished, event)
        persisted_turn = PersistedAssistantTurn(record_id=anchor, tool_calls=tool_dicts)

    return _LLMTurnOutput(
        text=text, reasoning=reasoning, tool_calls=tool_calls, usage=usage,
        response_event_id=finished.id,
        anchor=anchor, minted=minted, provider_state=result.provider_state,
        messages_sent=llm_request.messages, memory_timestamp=finished.timestamp,
        persisted_turn=persisted_turn)


def _can_recover_context(state: LoopState) -> bool:
    """越过上下文停机线之后，该不该回 PrepareStep 压缩续跑（而非退回 observe 吃一次 retry）。

    两个否决条件都必须有，各管一种失败模式：

    - **配额**（`max_context_recoveries`，0 = 关闭恢复）：compact 压得动一点、但离 target
      总是差一口气时，prepare↔act 会来回 ping-pong，每圈烧一次真实 act LLM 调用 + 一到两次
      摘要调用。配额是这条的硬止损。
    - **compact 已压不动**（`COMPACT_NOOP_KEY`）：`escalating_compact` 的每一级都有可折性
      guard，全 noop 时它返回的事件里一条 `MEMORY_COMPACTED` 都没有——说明这个 scope 再压
      也挤不出东西（例如 task 层只剩 ≤ collapse_keep_last 条、agent 层没有已完成胶囊、
      单轮塞进来一条超大 tool result）。此时再恢复一次纯属白烧，立即放弃，不等配额耗尽。

    ⚠️ 这个否决只在 `used > 0`（即已经恢复过至少一次）时生效。**本 run 开头那次 prepare 的
    compact 也会写这个标志**，而那一次几乎必然是 noop——run 刚起，task 层往往只有一条
    USER_PROMPT，各级 guard 全不满足。若不区分，那面「压不动」的旗会在 act 还没发出第一个
    请求时就立起来，把整个 run 的恢复资格一笔勾销（实测：`prepare→act→observe`，恢复从不
    发生）。所以第一次恢复只受配额约束——想知道压得动压不动，总得先真压一次。
    """
    limit = getattr(state.agent.loop_config, "max_context_recoveries", 0)
    if limit <= 0:
        return False
    used = state.extra.get(CONTEXT_RECOVERY_COUNT_KEY, 0)
    if used >= limit:
        return False
    return not (used > 0 and state.extra.get(COMPACT_NOOP_KEY))


async def _account_tokens(state: LoopState, ctx: LoopContext, usage: LLMUsage) -> bool:
    """更新 LoopGuard 基线 + session.token_used；返回是否越过上下文停机线。

    阈值取 `loop_config.context_limit_stop_ratio`（默认 0.9），**曾硬编码 0.8**——那个值
    与 `compact_token_ratio` 的默认值重合，于是 prepare 刚把估算压到「刚低于 0.8」，本步
    第一轮就又越线，一次恢复配额当场蒸发。两条线现在分工：compact_token_ratio 管 run 开头
    的常规压缩（估算口径），本比率管 mid-act 兜底（真实 usage 口径），故设得更高。

    对齐 miniAgents actor.py context_limit_hit 逻辑。
    """
    agent = state.agent
    if usage.prompt_tokens > 0:
        agent.loop_guard.context_tokens = usage.prompt_tokens
        try:
            from ctx_weft.protocols import MemoryKind, MemoryScope

            # v2 P3a：len(视图谓词) 取代 count_recent（= 旧 USER_PROMPT+LLM_RESPONSE 口径，
            # 与 prepare._estimate_tokens 对齐）。
            view = await load_records(ctx.memory, state.scope, MemoryScope.TASK, ctx.provider_ctx)
            agent.loop_guard.context_message_count = sum(
                1 for r in view
                if r.kind is MemoryKind.CONVERSATION_TURN and r.role in ("user", "assistant")
            )
        except Exception as exc:
            # best-effort 计数：失败不影响主流程，但记 debug 便于排查（不静默吞）。
            logger.debug("load_records count for loop_guard failed: %s", exc)

    # Accumulate actual usage for reporting; sessions have no token spending limit.
    state.session.token_used += usage.prompt_tokens + usage.completion_tokens

    eff = state.resolved_model.effective_input_limit
    ratio = getattr(agent.loop_config, "context_limit_stop_ratio", 0.8)
    return (
        eff > 0
        and usage.prompt_tokens > 0
        and usage.prompt_tokens >= int(eff * ratio)
    )


@dataclasses.dataclass
class PersistedAssistantTurn:
    """`_ingest_assistant_turn` 的结构化返回（spec: tool-operations，wp5）。

    ``record_id`` 是 LLM_RESPONSE 回合的 memory 记录 id——tool_call_id 的派生输入之一
    （确定性派生 → 同一逻辑调用跨重启同 id）。``tool_calls`` 直通旧形态（message
    重建消费点零语义变化）。
    """

    record_id: str
    tool_calls: list[dict]


def _assistant_memory_event(
    address: MemoryAddress, text: str, tool_calls: list[ToolCall],
    usage: LLMUsage, turn_num: int,
    anchor: str = "", minted: list[MintedCall] | None = None,
    timestamp: datetime | None = None,
) -> tuple[MemoryEvent, list[dict]]:
    """Build the same assistant record for live writes and finished-event recovery.

    派发(submit_*)/silent 工具的 tool_call 排除出 LLM_RESPONSE.metadata（派发落 agent 层
    delegate conversation turn、silent 结果不入对话），避免无配对 TOOL_RESULT 的悬挂调用破坏
    无损重建（spec 2026-06-28 §2.3）。

    anchor（spec: conversation-integrity）：预铸的记录 id，经 MemoryEvent.id 交给 provider
    采纳——它同时是内部调用标识与 tool_call_id 的派生锚，三者由此同源。minted 携带每个
    调用的 raw wire id 与 ordinal，随 metadata 落库（raw_tool_call_id 供追溯）；
    两者缺省（旧测试直调 / 无工具回合）时退化为无伴随字段的旧行为。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS, SILENT_TOOLS
    by_ordinal = {m.ordinal: m for m in (minted or [])}
    asst_tool_dicts: list[dict] = []
    for i, tc in enumerate(tool_calls):
        d: dict = {"id": tc.id, "name": tc.name, "input": tc.arguments}
        m = by_ordinal.get(i)
        if m is not None:
            d["raw_tool_call_id"] = m.raw_id
        asst_tool_dicts.append(d)
    _excluded = DISPATCH_TOOLS | SILENT_TOOLS
    non_dispatch_tool_dicts = [d for d in asst_tool_dicts if d["name"] not in _excluded]
    record_id = anchor or generate_id("asst")
    return (
        MemoryEvent(
            id=record_id,
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=address,
            content=text,
            timestamp=timestamp or now_utc(),
            role="assistant",
            metadata={
                "turn": turn_num,
                "tool_call_count": len(tool_calls),
                # Memory can protect unfinished calls, including dispatch/silent
                # calls omitted from the model-visible tool protocol.
                "execution_call_ids": [tc.id for tc in tool_calls],
                "tool_calls": non_dispatch_tool_dicts,
                "usage": dataclasses.asdict(usage),
            },
        ),
        asst_tool_dicts,
    )


async def _ingest_assistant_turn(
    state: LoopState, ctx: LoopContext, text: str, tool_calls: list[ToolCall],
    usage: LLMUsage, turn_num: int,
    anchor: str = "", minted: list[MintedCall] | None = None,
    timestamp: datetime | None = None,
) -> PersistedAssistantTurn:
    """Persist or stage a live turn; recovery ingests the built event directly."""
    from ctx_weft.core.loop.driver import ingest_or_stage

    event, tool_dicts = _assistant_memory_event(
        state.scope, text, tool_calls, usage, turn_num, anchor, minted, timestamp)
    # Empty streams may still belong to a provisional round. Preserve staging.
    await ingest_or_stage(
        ctx.memory, event, ctx.provider_ctx,
        task_manager=getattr(ctx, "task_manager", None),
        # 鸭子类型的 state 替身（单测直调本函数）可能没有 task。
        task_id=getattr(getattr(state, "task", None), "id", ""),
    )
    return PersistedAssistantTurn(record_id=event.id or "", tool_calls=tool_dicts)


def _predispatch_compact_options(
    state: LoopState, tool_calls: list[ToolCall], usage: LLMUsage,
) -> dict[str, Any] | None:
    """Decide the next step without running compaction or invoking tools."""
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS
    if not any(tc.name in DISPATCH_TOOLS for tc in tool_calls):
        return None
    ratio = getattr(state.agent.loop_config, "predispatch_compact_token_ratio", 0.0)
    tokens = usage.prompt_tokens or state.agent.loop_guard.context_tokens
    limit = state.resolved_model.effective_input_limit
    if ratio <= 0 or limit <= 0 or tokens <= 0 or tokens / limit < ratio:
        return None
    return {"trigger": "pre_dispatch", "token_estimate": tokens,
            "force": False, "after": "prepare"}



async def _record_hidden_call_completion(state, ctx, tool_calls):
    """Durable, non-rendered completion for calls absent from task tool messages."""
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS, SILENT_TOOLS
    from ctx_weft.core.loop.driver import ingest_or_stage
    from ctx_weft.core.utils.ids import record_id

    for call in tool_calls:
        if call.name not in DISPATCH_TOOLS | SILENT_TOOLS:
            continue
        await ingest_or_stage(
            ctx.memory,
            MemoryEvent(
                id=record_id("execution_completed", call.id), kind=MemoryKind.TOOL_AUDIT,
                scope=MemoryScope.TASK, address=state.scope, content="",
                role="assistant", timestamp=now_utc(),
                metadata={"tool_call_id": call.id, "execution_completed": True},
            ),
            ctx.provider_ctx, task_manager=getattr(ctx, "task_manager", None),
            task_id=state.task.id,
        )


async def _execute_tool_calls(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall],
) -> list[dict]:
    """逐个经 CapabilityGateway 执行 tool calls，返回 tool_results。

    工具间 / 执行中命中软打断 → 给本 tc 及之后补「已取消/被打断」result + park（raises HitlPark）；
    硬取消 → CancelledError；被 park 的工具 → task 转 SUSPENDED 并上抛（spec/07 §7）。
    """
    tool_results: list[dict] = []
    if any(tc.name == SWITCH_TASK_NAME for tc in tool_calls) and len(tool_calls) != 1:
        message = "switch_task must be called alone. Resubmit the switch as a single tool call."
        for tc in tool_calls:
            await _ingest_synthetic_tool_result(state, ctx, tc, message)
            tool_results.append({"tool_call_id": tc.id, "name": tc.name,
                                 "result": message, "is_error": True})
        return tool_results
    ctx.run_phase.in_tool_loop = True

    for i, tc in enumerate(tool_calls):
        # 账本键就是 tc.id（摄入点铸造的内部标识），gateway 自己从 tool_call_id 取——
        # 调用侧不再铸第二条身份、也不再经共享 provider_ctx 转移所有权。
        # 工具间命中：软打断 → 本 tc 及之后全部「未开始」→ 补「已取消」+ park；硬取消 → CancelledError。
        if _interrupt_pending(ctx):
            # 这一轮若还没提交（热应答醒来、到下一个提交点之前），暂停就是撤回那条答复：
            # 走 `discard_uncommitted_round`（热撤销 → park 在原问题上，抛出）。剩下的
            # 工具调用**不补**结果——人重答之后 reconcile 会把它们当作从未开始的调用执行。
            await discard_uncommitted_round(state, ctx)
            for rest in tool_calls[i:]:
                await _ingest_synthetic_tool_result(state, ctx, rest, CANCELLED_MARK, cancelled=True)
            ctx.run_phase.in_tool_loop = False
            await _park_for_interrupt(state, ctx)
        if ctx.cancel_token is not None and ctx.cancel_token.is_cancelled:
            ctx.cancel_token.raise_if_cancelled()

        # 可取消地执行在途工具：执行中被打断 → 取消该工具、补「被打断」result。
        invoke_task = asyncio.ensure_future(_invoke_tool(tc, state, ctx))
        completed = await _await_tool_or_stop(invoke_task, ctx)
        if not completed:
            if _interrupt_pending(ctx):
                # 同上。被打断的这个工具若已过 gateway step 5，这一轮已提交，下面这句 no-op；
                # 没过 step 5 就意味着 provider 从未被调用，当作没开始即可。
                await discard_uncommitted_round(state, ctx)
                await _ingest_synthetic_tool_result(state, ctx, tc, INTERRUPTED_MARK, interrupted=True)
                for rest in tool_calls[i + 1:]:
                    await _ingest_synthetic_tool_result(state, ctx, rest, CANCELLED_MARK, cancelled=True)
                ctx.run_phase.in_tool_loop = False
                await _park_for_interrupt(state, ctx)
            if ctx.cancel_token is not None:
                ctx.cancel_token.raise_if_cancelled()  # 硬取消
        try:
            result = invoke_task.result()
        except HitlPark:
            # 被 park 的工具未执行，直接 unwind（spec/07 §7）；task 落 AWAITING_HUMAN
            # 由 TaskManager 据 RunOutcome 定（Task 4：这里原先那句 `task.status =
            # "SUSPENDED"` 是写给自己看的死值，_run_loop 的 HitlPark 支随即覆盖）。
            raise

        tool_results.append({
            "tool_call_id": tc.id,
            "name": tc.name,
            "result": result.content,
            "is_error": result.is_error,
            "memory_id": getattr(result, "memory_id", None),
        })

    ctx.run_phase.in_tool_loop = False
    return tool_results


def _synthesize_final_outputs(state: LoopState, transcript: list) -> None:
    """Persist the actual closing assistant body before foreground observation or park."""
    # getattr 防御：手构的替身 task 未必带这个字段（本函数现在也从 park 路径被调，
    # 而那条路上的单测惯用 SimpleNamespace）。
    if getattr(state.task, "suspend_requested", False) or not transcript:
        return
    body, summary = _compose_final_outputs(transcript)
    # 拼接留在这里：`task.outputs` 的既有契约（**拼好的单串**）一个字不变——它有 6 处
    # 读取方（_build_memory_content / background_observe / finalize 的 final_reply 等），
    # 改形态会把这些一并掀翻。
    outputs = f"{body}\n\n{summary}" if (body and summary) else (body or summary)
    if outputs:
        state.task.outputs = outputs
    # 两段另存一份：TASK_FINALIZED 需要它们**分开**出核——output 是交付物本身，summary
    # 是 agent 给 reviewer 的自评清单。混在一起正是 host 侧打印「最终答复」时把自评清单
    # 当答案一起打出来的成因。
    state.extra["final_body"] = body
    state.extra["final_summary"] = summary


async def _finish_plain_text_turn(
    state: LoopState, ctx: LoopContext, turn_num: int, transcript: list,
    *, reply_anchor=None,
) -> None:
    """纯文本回合（无 tool call）收尾。

    **有人在场**就让位 → `_park_await_user`（HITL input 冷 park、raises HitlPark；park
    与后台 observe 并行，observer 判 success 则经带外入口终结本 task、判 continue 则自动续跑，判 waiting 则
    维持 park 等人，判 fail 则失败收尾）。unattended / 非普通任务 / 无 hitl：纯文本即任务产出，发 stop 事件
    路由前台 observe。

    判据 2026-09-22 从 `interaction_mode == "interactive"` 换成 `not unattended`：
    「这段话是想问人还是交付完了」是**逐回合**的语义，不该由 task 上的一个静态字段
    统一回答——那样 interactive 的 task 里 agent 交付完了也得 park，auto 的 task 里
    agent 想停下来说句话也停不了。现在交给 observer 逐次判定，这里只问「有没有人」。

    下面这条 `stop` 是「没让位」的收尾，与让位严格互斥：真让位了就抛 HitlPark，压根
    走不到这里；`_park_await_user` 正常返回就意味着这一轮不让位（无人值守），它保证
    零副作用返回，所以 `stop` 是这个回合唯一的一条 ACT_TURN_COMPLETED。
    """
    # **先合成 outputs，再决定让不让位**：下面那条 park 会抛 `HitlPark` 就地中断整个
    # ActStep，循环之后那次合成到不了。见 `_synthesize_final_outputs` 的 docstring。
    _synthesize_final_outputs(state, transcript)

    if (
        isinstance(state.task.settings, NormalTaskSettings)
        and not state.task.unattended
        and ctx.hitl is not None
    ):
        await _park_await_user(state, ctx, turn_num, reply_anchor=reply_anchor)
    await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
        "turn": turn_num, "reason": "stop"}))


async def _invoke_tool(tc: ToolCall, state: LoopState, ctx: LoopContext) -> Any:
    """通过 CapabilityGateway 执行工具调用。gateway 不存在时降级为旧路径。"""
    if ctx.capability_gateway is not None:
        return await ctx.capability_gateway.invoke(
            tool_name=tc.name,
            arguments=tc.arguments,
            state=state,
            ctx=ctx,
            tool_call_id=tc.id,
        )

    # 降级：无 gateway 时的最简路径（保持向后兼容）
    from ctx_weft.core.loop.capability_gateway import InvocationResult
    logger.warning("No CapabilityGateway in LoopContext; tool '%s' cannot be dispatched", tc.name)
    return InvocationResult(
        invocation_id="noop",
        tool_name=tc.name,
        content=runtime_prompt('gateway_missing', tool=tc.name),
        is_error=True,
    )


INTERRUPTED_MARK = "[interrupted by the user]"
CANCELLED_MARK = "[cancelled]"


async def _ingest_synthetic_tool_result(
    state: LoopState, ctx: LoopContext, tc: ToolCall, content: str, *,
    interrupted: bool = False, cancelled: bool = False,
) -> None:
    """为被打断/未执行的工具补一条 TOOL_RESULT，使 tool_call↔result 一一对应（无 dangling）。

    走 `ingest_tool_result`：记录 id 必须是确定性的那一个，否则这条补位在 dangling 判定
    眼里不存在——「补它就是为了不悬挂」这个目的会落空，恢复时该调用被重新执行一遍。
    """
    from ctx_weft.core.loop.capability_gateway import ingest_tool_result
    await ingest_tool_result(
        ctx.memory, ctx.provider_ctx, state.scope,
        tool_call_id=tc.id, content=content, is_error=True, tool_name=tc.name,
        extra_metadata={"interrupted": interrupted, "cancelled": cancelled},
        task_manager=getattr(ctx, "task_manager", None))


async def _await_tool_or_stop(invoke_task: "asyncio.Future", ctx: LoopContext) -> bool:
    """运行在途工具；任一停止信号（硬取消 cancel_token / 软打断 pause_token）在执行中触发
    → 取消该工具并等其清理跑完，返回 False；正常完成返回 True。
    """
    waiters: list[asyncio.Future] = []
    if ctx.cancel_token is not None:
        waiters.append(asyncio.ensure_future(ctx.cancel_token.wait()))
    if ctx.pause_token is not None:
        waiters.append(asyncio.ensure_future(ctx.pause_token.wait_paused()))
    try:
        if not waiters:
            await asyncio.wait({invoke_task})
            return True
        await asyncio.wait({invoke_task, *waiters}, return_when=asyncio.FIRST_COMPLETED)
    except asyncio.CancelledError:
        # Instance shutdown cancels the actor rather than the user cancel token.
        # A tool may own an open transaction; do not leave it writing after exit.
        invoke_task.cancel()
        await asyncio.gather(invoke_task, return_exceptions=True)
        raise
    finally:
        for w in waiters:
            w.cancel()
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
    if invoke_task.done():
        return True
    invoke_task.cancel()
    await asyncio.wait({invoke_task})
    return False


def _interrupt_pending(ctx: LoopContext) -> bool:
    """软打断挂起中：pause_token 被 pause 且有 hitl 可 park。"""
    tok = ctx.pause_token
    return (
        tok is not None
        and tok.is_paused
        and ctx.hitl is not None
    )


def _is_own_root(task) -> bool:
    """root task 判定（与 observe._is_own_root / finalize._close_one 保持一致）。

    True  → session root（parent_task_id is None）或跨 agent own-root。
    False → 同 agent 子任务（parent 非空且 creator==assigned）。
    """
    same_agent = task.creator_agent_id == task.assigned_agent_id
    cross_agent = bool(task.parent_task_id) and not same_agent
    return task.parent_task_id is None or cross_agent


async def _commit_interrupted_partial(
    state: LoopState, ctx: LoopContext, text: str, reasoning: str, turn_num: int,
) -> None:
    """把被打断的半截 assistant 文本入 memory（标注「被用户打断」）。无内容则不留记录（①）。"""
    if not (text.strip() or (reasoning or "").strip()):
        return
    content = f"{text}\n\n{INTERRUPTED_MARK}" if text.strip() else INTERRUPTED_MARK
    await ctx.memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            content=content,
            timestamp=now_utc(),
            role="assistant",
            metadata={
                "turn": turn_num,
                "interrupted": True,
                "reasoning": reasoning or None,
            },
        ),
        ctx.provider_ctx,
    )


def _interrupt_edit_prefix(prev_request: str) -> str:
    """产出①打断续接的前缀说明；prev 为空则返回空串。

    抽出成纯文本→纯文本的小函数，供多模态调用点用 content_with_prefix 组装
    （new_input 可能是 list[ContentPart]，f-string 直接拼会把图片拍扁）。
    """
    prev = (prev_request or "").strip()
    if not prev:
        return ""
    return runtime_prompt('interrupt_edit_preface', previous=prev) + "\n\n"


def interrupt_edit_note(prev_request: str, new_input: str) -> str:
    """① 打断（未吐 token）续接时的说明：上一条请求被取消、改为新请求。空 prev 时原样返回。"""
    prefix = _interrupt_edit_prefix(prev_request)
    return f"{prefix}{new_input}" if prefix else new_input


async def _cold_park(
    state: LoopState, ctx: LoopContext, preface: str, *, unattended: bool,
    observe_boundary: str | None = None, await_user_turn: int | None = None,
    preceding_events: tuple = (),
    replace_user_turn_id: str = "",
    reply_anchor=None, replaces_observe_id="",
) -> None:
    """起 wait_for_user 冷 park：登记 HITL 请求 + 抛 HitlPark（task 落 AWAITING_HUMAN
    由 TaskManager 据 RunOutcome 定）。

    **纯机械动作，不含任何策略判断**：preface 选哪条、这次让不让位，都由上面两个具名
    包装（`_park_for_interrupt` / `_park_await_user`）决定后传进来。此前这些判断和动作
    挤在同一个函数里，靠一个 `source` 字符串同时表达「选哪条 preface」与「要不要豁免
    无人值守守卫」两件毫不相干的事。

    **不写会话状态**：这不是本函数的职责。2026-09-02 那次重构曾把会话状态的唯一写者
    定为 `SessionRegistry`，由一条队列级聚合信号翻译成会话级状态；那整条链路已随会话
    状态机退役——`SessionRegistry` 自 2026-09-03 起降格为纯 agent 登记表，两端的事件
    类型也已于 2026-09-05 一并删除。会话级别的展示状态目前不由 core 预先算好广播，
    由 host 自行按 agent 状态聚合推导（docs/events-v2.md §2.1.1）。

    续跑方式由 **delivery 显式声明**，不再靠 `form == "wait"` + sentinel capability_id
    这组跨三个模块的魔法字符串（spec §5）。
    """
    observe_request = None
    if observe_boundary is not None:
        from ctx_weft.core.loop.steps.observe import capture_context, make_observe_request
        # A stream can end without yielding any chunk. Its finished/memory pair
        # is still provisional; commit it before pinning the observer's inputs.
        await commit_round(state, ctx)
        observe_request = make_observe_request(
            state, observe_boundary,
            observe_boundary in {"plain_text", "finish_park", "task_switch"},
            reply_anchor=reply_anchor, replaces_observe_id=replaces_observe_id)
        await capture_context(state, ctx, observe_request)
    req = await ctx.hitl.open(
        HitlAsk(
            form=HITL_FORM_WAIT,
            delivery=UserTurnDelivery(task_id=state.task.id, preface=preface),
        ),
        session_id=state.session.id,
        task_id=state.task.id,
        agent_id=state.agent.id,
        stage=HITL_STAGE_TOOL,
        unattended=unattended,
        tenant_id=state.session.tenant_id,
        observe_request=observe_request,
        **({"preceding_events": preceding_events} if preceding_events else {}),
        **({"replace_user_turn_id": replace_user_turn_id} if replace_user_turn_id else {}),
    )
    if await_user_turn is not None:
        await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
            "turn": await_user_turn, "reason": "await_user"}))
    if observe_request is not None:
        observe_request["waiting_hitl_id"] = req.id
        await StepDriver.for_context(ctx).start_durable(
            "observe", state, ctx, options=observe_request)
    # 不建等待槽 —— 本调用方随即 park 释放协程而非 await，应答必然走冷续跑。
    raise HitlPark(hitl_id=req.id)


async def _park_for_interrupt(
    state: LoopState, ctx: LoopContext, *, edit: bool = False, preceding_events: tuple = (),
    replace_user_turn_id: str = "",
    reply_anchor=None, replaces_observe_id="",
) -> None:
    """人按了暂停键 → park 等他续接。**对无人值守守卫豁免**（`unattended=False`）。

    守卫要挡的是「没有人可问」，不是「没有人在场」：按下暂停键的就是一个人，续接的
    也会是那个人——运维暂停一个后台无人值守作业是完全合法的操作，那时 park 正是对的
    行为。这条路径不属于守卫要挡的场景，所以豁免写死在这个具名函数里，而不是让四个
    调用点各传一个 `allow_unattended=True`（那种写法太容易被后来人顺手改掉）。

    ``edit`` 只决定 preface：已吐过 token / 已进工具循环 → after_interrupt；未吐任何
    token 且未进工具 → after_interrupt_edit（续接时需补一句「上一条请求被取消」）。
    """
    preface = PREFACE_AFTER_INTERRUPT_EDIT if edit else PREFACE_AFTER_INTERRUPT
    await _cold_park(state, ctx, preface, unattended=False,
                     observe_boundary="interrupt",
                     reply_anchor=reply_anchor, replaces_observe_id=replaces_observe_id,
                     preceding_events=preceding_events, replace_user_turn_id=replace_user_turn_id)


async def _park_await_user(
    state: LoopState, ctx: LoopContext, turn_num: int, *, boundary: str = "plain_text",
    reply_anchor=None,
) -> None:
    """Yield a prose reply for background judgement when a user is present.

    Unattended tasks stop normally for foreground observation. Historical
    finish_park boundaries remain readable by recovery."""
    if state.task.unattended:
        logger.warning(
            "act: task %s is unattended — not yielding to the user after a plain-text turn; "
            "treating the text as this turn's output and stopping normally",
            state.task.id,
        )
        return
    # 纯文本暂停 = 软待命(允许但不强制回复) → PAUSED,区别于 ask_user 的 PAUSED_HITL。
    #
    # 2026-09-22 去掉了这里的 `_is_own_root` 闸：子任务的纯文本回合同样 park、同样要
    # 判定。当时那个闸的理由是「非 root 走 LLM observe 向 parent 上报」——但纯文本回合
    # 压根到不了前台 observe（它在这里就 park 了），于是子任务的这一段既没人判也没人折。
    # 并发不会因此失控：单交互线闸门保证一个 session 同时只有一条非 unattended 的线
    # （见 `TaskManager._interactive_line_held`）。
    await _cold_park(state, ctx, PREFACE_NORMAL, unattended=state.task.unattended,
                     observe_boundary=boundary, await_user_turn=turn_num,
                     reply_anchor=reply_anchor)


async def _interrupt_checkpoint(state: LoopState, ctx: LoopContext, *, reply_anchor=None) -> None:
    """协作式停止点：软打断（pause）→ park；硬取消（cancel）→ CancelledError。

    提交点之前命中（act 第一轮的循环顶部，LLM 还没被调用过）→ 整轮丢弃而非 park，
    见 `discard_uncommitted_round`。
    """
    if _interrupt_pending(ctx):
        await discard_uncommitted_round(state, ctx)   # 命中则 raises RoundDiscarded
        edit = not ctx.run_phase.produced and not ctx.run_phase.in_tool_loop
        await _park_for_interrupt(state, ctx, edit=edit, reply_anchor=reply_anchor)  # raises HitlPark
    tok = ctx.cancel_token
    if tok is not None and tok.is_cancelled:
        tok.raise_if_cancelled()
