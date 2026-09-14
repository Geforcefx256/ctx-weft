"""ActStep：多 turn LLM 子循环 + capability 调用。

工具调用通过 CapabilityGateway 统一执行（授权 + 审计 + 内存），
ActStep 只处理 LLM 流 + turn 循环逻辑。
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ctx_weft.protocols import LLMMessage, LLMRequest, LLMUsage, ToolCall
from ctx_weft.core.loop.driver import (
    RECOGNIZE_INTENT_PENDING_KEY, ROUND_COMMITTED_KEY,
    LoopContext, LoopState, Step, StepOutcome, make_event,
)
from ctx_weft.core.loop.llm_gateway import (
    PROMPT_EST_BASE_KEY, PROMPT_EST_SEG_KEY, request_prompt_estimate, resolve_llm_identity,
    stream_llm_resilient,
)
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
from ctx_weft.core.loop.park import HitlPark, RoundDiscarded
from ctx_weft.core.capabilities.control_tools import FINISH_TASK_NAME
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.core.utils.estimate import effective_limit
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.protocols import MemoryEvent, MemoryKind, MemoryScope
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


class ActStep(Step):
    """ActStep: multi-turn LLM loop + capability invocation via CapabilityGateway."""

    name = "act"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        prompt = state.assembled_prompt
        if prompt is None:
            raise RuntimeError("ActStep requires assembled_prompt — PrepareStep must run first")

        agent = state.agent
        max_turns = agent.loop_config.max_turns_per_act
        transcript: list[TurnRecord] = []
        exit_reason = "normal"

        # 运行时态势 guidance（任务树/已完成子任务/收尾提醒）已随装配管线注入
        # （PrepareStep → GuidanceSource → composer 末条 user 尾部），此处不再修饰。
        current_messages = list(prompt.messages)
        # 动态 max_tokens 的增量基线：上一轮实际发送的 message 条数（= 本轮 usage 对应的真实
        # prompt 基线）。None=本轮无循环内真实基线（首轮）→ request_prompt_estimate 走整份估算。
        baseline_msg_count: int | None = None

        for turn_num in range(1, max_turns + 1):
            await _interrupt_checkpoint(state, ctx)
            await ctx.event_bus.emit(make_event(
                state, EventType.ACT_TURN_STARTED, payload={"turn": turn_num}))

            # 1) 单轮 LLM：流式累积文本 / reasoning / tool_calls / usage（软打断在内部 park）
            sent_msg_count = len(current_messages)  # 本轮发送条数（append 前）→ 下轮增量基线
            turn = await _run_llm_turn(state, ctx, prompt, current_messages, turn_num, baseline_msg_count)

            # 2) token 记账 + context_limit 判定
            context_limit_hit = await _account_tokens(state, ctx, turn.usage)
            # 仅当本轮真实刷新了 context_tokens（usage>0）才前移基线；否则保持旧值/None，
            # 让下一轮退回整份估算（更保守），避免基线与陈旧 context_tokens 错配。
            if turn.usage.prompt_tokens > 0:
                baseline_msg_count = sent_msg_count

            # 3) assistant 回合落 memory + 接回 message 历史 + 记 transcript
            persisted_turn = await _ingest_assistant_turn(
                state, ctx, turn.text, turn.tool_calls, turn.usage, turn_num)
            asst_tool_dicts = persisted_turn.tool_calls
            state.extra["assistant_record_id"] = persisted_turn.record_id
            current_messages.append(LLMMessage(
                role="assistant", content=turn.text, tool_calls=asst_tool_dicts,
                reasoning_content=turn.reasoning or None))
            turn_record = TurnRecord(
                turn=turn_num, messages_sent=list(current_messages),
                assistant_text=turn.text, tool_calls=turn.tool_calls, usage=turn.usage)

            # 4) context_limit 命中：记录本轮后停止
            if context_limit_hit:
                transcript.append(turn_record)
                await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
                    "turn": turn_num, "reason": "context_limit"}))
                exit_reason = "context_limit"
                # 上下文爆了还留着运行期 pin 进来的额外工具描述，是朝根因加码——下一轮
                # 重新装配时那些工具会原样再占一遍额度。**max_turns 那条退出不清**：它走
                # 另一个事件、另一条路径，且语义相反——agent 接近轮数上限才找到对的工具，
                # 清掉等于让它从零重来、再次耗尽轮数，是个会自我复现的活锁。
                if ctx.capability_cache is not None:
                    ctx.capability_cache.clear_pins(state.task.id)
                logger.warning(
                    "ActStep context_limit_hit: prompt_tokens=%d >= %d * 0.8 for agent %s",
                    turn.usage.prompt_tokens, agent.loop_guard.context_limit, agent.id)
                break

            # 5) 纯文本（无 tool call）：interactive 让位用户 / 否则即任务产出 → observe
            if not turn.tool_calls:
                transcript.append(turn_record)
                await _finish_plain_text_turn(state, ctx, turn_num)
                break

            # 6) 派发前压缩（含派发调用且越阈值时；在 dispatch 执行 / 子 spawn-inherit 之前）
            await _maybe_predispatch_compact(state, ctx, turn.tool_calls, turn.usage)

            # 7) 执行 tool calls（同批次延后退出信号，保证 tool_call ↔ result 一一对应）
            tool_results = await _execute_tool_calls(state, ctx, turn.tool_calls)
            for tr in tool_results:
                current_messages.append(LLMMessage(
                    role="tool", content=tr["result"], tool_call_id=tr["tool_call_id"]))
            turn_record.tool_results = tool_results
            transcript.append(turn_record)
            await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
                "turn": turn_num, "reason": "tool_calls_processed"}))

            # finish_task 与 delegate 同批：finish 胜出（派发改投为独立后继）。
            _reconcile_finish_vs_dispatch(state, ctx, turn.tool_calls)

            if state.task.actor_done:
                exit_reason = "actor_done"
                break

        else:
            exit_reason = "max_turns"
            logger.warning("ActStep hit max_turns=%d for agent %s", max_turns, agent.id)
            await ctx.event_bus.emit(make_event(state, EventType.MAX_TURNS_REACHED, payload={
                "max_turns": max_turns}))

        # suspend_requested 表示 actor 本轮派了活、本 run 要停在等子任务 → SuspendStep。
        # （判据从 `task.status == "SUSPENDED"` 换成意图位：状态归 TaskManager，Task 4）
        next_step = "suspend" if state.task.suspend_requested else "observe"

        # ── task.outputs：收尾时合成最终交付物（spec 2026-07-01 反转契约）──
        # 收尾路径 = 纯文本收尾(normal) 或 finish_task 收尾(actor_done 且未挂起)；答复即消息正文，
        # finish_task 的 deliverables_summary 为可选产出小结。max_turns / context_limit /
        # delegate-suspend 不在此列（不产最终输出，维持现状）。
        if (
            exit_reason in ("normal", "actor_done")
            and not state.task.suspend_requested
            and transcript
        ):
            body, summary = _compose_final_outputs(transcript)
            # 拼接留在调用方：`task.outputs` 的既有契约（**拼好的单串**）一个字不变——
            # 它有 6 处读取方（_build_memory_content / background_observe / finalize 的
            # final_reply 等），改形态会把这些一并掀翻。
            outputs = f"{body}\n\n{summary}" if (body and summary) else (body or summary)
            if outputs:
                state.task.outputs = outputs
            # 两段另存一份：TASK_FINALIZED 需要它们**分开**出核——output 是交付物本身，
            # summary 是 agent 给 reviewer 的自评清单。混在一起正是 host 侧打印「最终
            # 答复」时把自评清单当答案一起打出来的成因。
            state.extra["final_body"] = body
            state.extra["final_summary"] = summary

        return StepOutcome(
            next_step=next_step,
            state_patch={
                "transcript": transcript,
                "act_exit_reason": exit_reason,
            },
        )


def _compose_final_outputs(transcript: list[TurnRecord]) -> tuple[str, str]:
    """拆出收尾交付物的两段 `(body, summary)`（spec 2026-07-01）。

    body = 收尾回合(transcript[-1])正文；空则回溯本段最近一段非空 assistant_text（兼容模型把
    答复写在上一回合、收尾回合只调 finish_task 的情况）。summary = 收尾回合 finish_task 调用的
    deliverables_summary（可空）。两段都空 → `("", "")`，交给 observer 护栏。

    **只拆不拼**：拼接是调用方的事。两段语义不同（body 是答复本身，summary 是给 reviewer 的
    交付物清单），下游有需要分开的消费者（TASK_FINALIZED 事件），在这里拼死就再也分不开了。
    """
    last = transcript[-1]
    body = (last.assistant_text or "").strip()
    if not body:
        body = next(
            (t.assistant_text.strip() for t in reversed(transcript)
             if (t.assistant_text or "").strip()),
            "",
        )
    summary = ""
    for tc in last.tool_calls:
        if tc.name == FINISH_TASK_NAME:
            val = (tc.arguments or {}).get("deliverables_summary", "")
            summary = val.strip() if isinstance(val, str) else ""
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
        for w in waiters:
            w.cancel()
        if waiters:
            await asyncio.wait(set(waiters))
        # 生成器可能仍挂在某个 yield 上（调用方 break 出去时）：关掉它，让 httpx 的
        # 流上下文退出。已经被上面的 cancel 拆完的生成器对此是 no-op。
        await agen.aclose()


async def _commit_round(state: LoopState, ctx: LoopContext) -> None:
    """提交本轮：LLM 开口了，这一轮算数（spec 2026-09-09）。**幂等**，每个 chunk 都调。

    在此之前，这一轮的两样东西还没落地：

    1. `TASK_CREATED` / `TASK_STARTED` / `RUN_STARTED` / `LLM_PROMPT_SENT` 攒在总线的
       未提交窗口里（只到达了进程内状态机，见 `EventBus` 类 docstring）；
    2. 把这一轮唤醒的那条 HITL 答复还停在**待终局**（`pending_decision`），
       `HitlResolved` 一直没发——日志里那个气泡仍是 pending。

    3. `recognize_intent` 还没起飞（判定在 `PrepareStep`，起飞在这里）——旁路只该为
       **真的发生过**的那一轮花一次 LLM 调用。它跑在 root task 上，而 root task 从第二句
       起每一轮都开窗，放在 prepare 起飞就会在窗口里跑、被撤销时整串事件白丢。

    **用户消息的落库不在此列**：它照旧在 run 启动时就写进 memory（`_persist_user_prompt`）。
    推迟它的代价是 PrepareStep 的预算折叠（L0.5 图片降级 / L1 / L3）在每一轮的首次装配
    都看不见这条记录，带图的第一条消息一张也降不了——`test_media_fold_replay_e2e` 实测钉住。
    它的撤销走 `_discard_round_if_uncommitted` 里的 `memory.fold`，那是纯遗忘原语，不是补偿写。

    顺序不可换：`TASK_CREATED` 必须最先出闸，它是下游 reducer 与 host 建 task 键的那一条；
    `recognize_intent` 必须在它之后起飞，它会发自己的一串事件，且要写 `task.title/description`
    ——那两个字段正是窗口里 `TASK_CREATED` 的 payload 在出闸那一刻现算的来源。
    """
    if state.extra.get(ROUND_COMMITTED_KEY):
        return
    state.extra[ROUND_COMMITTED_KEY] = True

    tm = ctx.task_manager
    if tm is not None:
        await tm.commit_round(state.task.id)

    # 两阶段终局的第二阶段：这一轮真的开跑了，那条把它唤醒的答复（HITL 冷续跑 /
    # `send_message` 对旧气泡的收口）现在才算数，`HitlResolved` 在此刻才发。
    # **必须排在 `commit_round` 之后**：那一句先把窗口里攒的 `TASK_*` / `RUN_STARTED`
    # 放出去，`HitlResolved` 才不会落在一个下游还没建键的 task 上。
    if state.extra.pop(RECOGNIZE_INTENT_PENDING_KEY, False):
        from ctx_weft.core.loop.steps.recognize_intent import launch_recognize_intent
        launch_recognize_intent(state, ctx)

    if ctx.hitl is not None:
        # 按 task 查，而不是把一串 hitl_id 顺着 LoopState 穿三层管道下来——与丢弃侧的
        # `_revert_round` 同一口径（它也按 task 全量 release），两边判据只有一份。
        for req in ctx.hitl.registry.claim_pending_for_task(
                state.session.id, state.task.id):
            await ctx.hitl.commit(req.id)



async def _discard_round_if_uncommitted(state: LoopState, ctx: LoopContext) -> None:
    """用户在 LLM 开口之前按了停 → 整轮丢弃（spec 2026-09-09）。已提交则原样返回。

    只对**由一条用户消息新开出来的 task**（`send_message` 的新建分支，`TaskManager`
    给它开了未提交窗口）成立。注入既有 task 的那两条路径（agent 本就在
    `AWAITING_HUMAN` / 挂起等子任务）不在此列——那个 task 早就提交过、是一段正在进行的
    对话，丢不得；它们照常 park 出续跑气泡，只是那条刚注入的用户消息同样还没落 memory，
    所以照样零残留。

    本函数做两件事，然后抛信号：

    1. **把这一轮的用户消息从 memory 里纯遗忘掉**——`fold([id], [])`，标 superseded，
       `load_view` 自然滤掉。这是 provider 早就有的原语（compact / finalize /
       background_observe 都在用），不是补偿写。id 来自 `task.user_prompt_memory_id`，
       在落库那一刻记下的；**不能**改用「读视图取最后一条 user」那种事后推断——用户连发
       两条、或上一条是 HITL 应答时会撤错人。
    2. 抛 `RoundDiscarded`，一路 unwind 到 `_run_task`。

    **一条事件都不发**：task 状态事件的发射点只有 TaskManager 一处（Task 4 的不变式，
    `test_task_manager_owns_status` 有静态守卫盯着）。把 agent 送回 `idle` 的那条
    `TASK_CANCELED`、以及关窗丢弃，都在 `TaskManager.discard_provisional` 里按正确顺序完成。
    """
    if state.extra.get(ROUND_COMMITTED_KEY):
        return
    tm = ctx.task_manager
    if tm is None or not tm.is_round_open(state.task.id):
        return

    record_id = getattr(state.task, "user_prompt_memory_id", None)
    if record_id and ctx.memory is not None:
        try:
            await ctx.memory.fold([record_id], [], ctx.provider_ctx)
        except Exception:
            # 遗忘失败不该把「用户按了暂停」变成一次 run 崩溃：那会把一个干净的丢弃
            # 变成一条 TASK_FAILED + 满屏栈。记一行，照常丢弃——最坏结果是 memory 里
            # 多留一条没人应答的 user 回合，比会话炸掉轻得多。
            logger.exception(
                "discard_round: failed to fold user prompt %s of task %s",
                record_id, state.task.id)
        else:
            state.task.user_prompt_memory_id = None
            # 记录没了，落库标志也要跟着回落：这个 task 若被重排（本路径下不会，但
            # 语义上必须自洽），`_persist_user_prompt` 应当重新写一条，而不是以为写过了。
            state.task.user_prompt_in_memory = False

    raise RoundDiscarded(state.task.id)


@dataclass
class _LLMTurnOutput:
    """单轮 LLM 流式产出。"""

    text: str
    reasoning: str
    tool_calls: list[ToolCall]
    usage: LLMUsage


async def _run_llm_turn(
    state: LoopState, ctx: LoopContext, prompt: Any, current_messages: list[LLMMessage],
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
    req_id = f"req_{agent.id}_{state.sequence_counter}"

    llm_request = LLMRequest(
        model=model, system=prompt.system, messages=list(current_messages), tools=prompt.tools)
    llm_request.prompt_token_estimate = request_prompt_estimate(
        ctx.llm.tokenizer, llm_request, getattr(agent, "loop_guard", None), baseline_msg_count)
    # turn：LLM_REQUEST_STARTED / LLM_PROMPT_SENT 现由 gateway 发射，但 turn_num 是本函数的局部
    # 循环变量、state 上没有对应字段——经 metadata 这个既有的瞬态透传通道带给 gateway（同
    # PROMPT_EST_BASE_KEY/PROMPT_EST_SEG_KEY 的做法），不新增函数参数。
    llm_request.metadata["turn"] = turn_num

    text = ""
    reasoning = ""
    tool_calls: list[ToolCall] = []
    usage = LLMUsage()
    interrupted = False

    async for chunk in _stream_until_stop(
        stream_llm_resilient(ctx, state, llm_request), ctx,
    ):
        tok = ctx.cancel_token
        if _interrupt_pending(ctx):
            interrupted = True          # ② 软打断：停收 token，下面提交半截
            break
        if tok is not None and tok.is_cancelled:
            tok.raise_if_cancelled()    # 硬取消 → CancelledError
        # ── 提交点：本轮第一个 chunk ──────────────────────────────────────────
        # 「一轮对话直到 LLM 真的开口才算发生」（spec 2026-09-09）。判据是**任意
        # chunk**，不是第一个 token：模型第一句就调工具、一个字都不吐的回合很常见，
        # 那种回合 `chunk.kind` 恒不是 "token"，等 token 就是永远等不到、整轮永不提交。
        # 这也正是 gateway 内部 `yielded_anything` 的那条界（首 chunk 之前失败可重试、
        # 之后断流直接抛 outage）——同一条界，这里把它从 gateway 的局部规则升成整轮的。
        await _commit_round(state, ctx)
        if chunk.kind == "token":
            ctx.run_phase.produced = True
            text += chunk.text
        elif chunk.kind == "reasoning":
            reasoning += chunk.text
        elif chunk.kind == "tool_call" and chunk.tool_call is not None:
            tool_calls.append(chunk.tool_call)
        elif chunk.kind == "tool_call_partial":
            # 工具调用参数流式期间的心跳：不携带数据，仅为让上面的暂停/取消检查点
            # 有机会触发（adapter 累积工具调用参数时不产出 token）。此处刻意不累积。
            pass
        elif chunk.kind == "usage" and chunk.usage is not None:
            usage = chunk.usage

    # 流在**首个 chunk 到达之前**被停止信号掐断时，上面的循环体一次都没执行到——
    # 判定必须补在循环之后，否则一次 TTFT 窗口里的暂停会被当成「流正常结束」。
    if not interrupted and _interrupt_pending(ctx):
        interrupted = True
    tok = ctx.cancel_token
    if tok is not None and tok.is_cancelled:
        tok.raise_if_cancelled()

    if interrupted:
        # ⓪ 一个 chunk 都没到就被中止 → 整轮丢弃，不 park、不留任何痕迹。
        await _discard_round_if_uncommitted(state, ctx)   # 命中则 raises RoundDiscarded
        # ② 已吐 token：把半截 assistant 文本入 memory 并标注「被用户打断」；
        # ① 已开口但本回合没吐正文（纯工具调用后被打断）：不留记录。随后 park 待用户续接。
        has_partial = bool(text.strip() or reasoning.strip())
        await _commit_interrupted_partial(state, ctx, text, reasoning, turn_num)
        if _is_own_root(state.task):
            from ctx_weft.core.loop.steps.background_observe import launch_background_observe
            launch_background_observe(state, ctx, boundary="interrupt")
        await _park_for_interrupt(state, ctx, edit=not has_partial)

    # token 自校准回喂：真实 usage 与发送前估算段作比（基线不参与），喂给该模型 tokenizer。
    # 估算段取 PROMPT_EST_SEG_KEY（tokenizer.count 直接产出）而非
    # prompt_token_estimate − base——整份路径上返回值可能被 max(full, ctx_tokens) floor 成
    # 上一轮真实值，拿 floor 后的值回喂会产出 ratio≈1 的假样本，把伺服系统性拖向 1。
    base = llm_request.metadata.get(PROMPT_EST_BASE_KEY)
    est_seg = llm_request.metadata.get(PROMPT_EST_SEG_KEY)
    if usage.prompt_tokens > 0 and est_seg is not None and base is not None:
        ctx.llm.tokenizer.observe(est_seg, usage.prompt_tokens - base)

    await ctx.event_bus.emit(make_event(
        state, EventType.LLM_RESPONSE_FINISHED,
        payload={
            "request_id": req_id, "content": text, "reasoning": reasoning,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ],
            "usage": dataclasses.asdict(usage),
            # 本次调用实际使用的模型/账号（host 云端上报按此计账，不再受切换竞态影响）
            "llm_model": model, "llm_account": llm_account,
            "finish_reason": "tool_use" if tool_calls else "stop", "turn": turn_num}))

    return _LLMTurnOutput(text=text, reasoning=reasoning, tool_calls=tool_calls, usage=usage)


async def _account_tokens(state: LoopState, ctx: LoopContext, usage: LLMUsage) -> bool:
    """更新 LoopGuard 基线 + session.token_used；返回是否命中 context_limit（80% 阈值）。

    对齐 miniAgents actor.py context_limit_hit 逻辑。
    """
    agent = state.agent
    if usage.prompt_tokens > 0:
        agent.loop_guard.context_tokens = usage.prompt_tokens
        try:
            from ctx_weft.protocols import MemoryKind, MemoryScope

            # v2 P3a：len(视图谓词) 取代 count_recent（= 旧 USER_PROMPT+LLM_RESPONSE 口径，
            # 与 prepare._estimate_tokens 对齐）。
            view = await ctx.memory.load_view(state.scope, MemoryScope.TASK, ctx.provider_ctx)
            agent.loop_guard.context_message_count = sum(
                1 for r in view
                if r.kind is MemoryKind.CONVERSATION_TURN and r.role in ("user", "assistant")
            )
        except Exception as exc:
            # best-effort 计数：失败不影响主流程，但记 debug 便于排查（不静默吞）。
            logger.debug("load_view count for loop_guard failed: %s", exc)

    # 累加 session.token_used（供 token_budget 检查使用）
    state.session.token_used += usage.prompt_tokens + usage.completion_tokens

    context_limit = agent.loop_guard.context_limit
    reserve = getattr(agent.loop_guard, "reserved_output_tokens", 0)
    eff = effective_limit(context_limit, reserve)
    return (
        eff > 0
        and usage.prompt_tokens > 0
        and usage.prompt_tokens >= int(eff * 0.8)
    )


@dataclasses.dataclass
class PersistedAssistantTurn:
    """`_ingest_assistant_turn` 的结构化返回（spec: tool-operations，wp5）。

    ``record_id`` 是 LLM_RESPONSE 回合的 memory 记录 id——operation_id 的派生输入之一
    （确定性派生 → 同一逻辑调用跨重启同 id）。``tool_calls`` 直通旧形态（message
    重建消费点零语义变化）。
    """

    record_id: str
    tool_calls: list[dict]


async def _ingest_assistant_turn(
    state: LoopState, ctx: LoopContext, text: str, tool_calls: list[ToolCall],
    usage: LLMUsage, turn_num: int,
) -> PersistedAssistantTurn:
    """把本轮 assistant 回合入 task 层 memory；返回结构化（record_id + tool_calls）。

    派发(submit_*)/silent 工具的 tool_call 排除出 LLM_RESPONSE.metadata（派发落 agent 层
    delegate conversation turn、silent 结果不入对话），避免无配对 TOOL_RESULT 的悬挂调用破坏
    无损重建（spec 2026-06-28 §2.3）。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS, SILENT_TOOLS
    asst_tool_dicts = [{"id": tc.id, "name": tc.name, "input": tc.arguments} for tc in tool_calls]
    _excluded = DISPATCH_TOOLS | SILENT_TOOLS
    non_dispatch_tool_dicts = [d for d in asst_tool_dicts if d["name"] not in _excluded]
    record_id = await ctx.memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            content=text,
            timestamp=now_utc(),
            role="assistant",
            metadata={
                "turn": turn_num,
                "tool_call_count": len(tool_calls),
                "tool_calls": non_dispatch_tool_dicts,
                "usage": dataclasses.asdict(usage),
            },
        ),
        ctx.provider_ctx,
    )
    return PersistedAssistantTurn(record_id=record_id or "", tool_calls=asst_tool_dicts)


async def _maybe_predispatch_compact(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall], usage: LLMUsage,
) -> None:
    """本轮含派发调用且越过 predispatch 阈值 → 派发执行前先走一遍 compact（同 CompactStep）。

    在 gateway 写 delegate conversation turn / 子 spawn-inherit 之前完成，使子继承到压缩后的记忆；每轮至多一次。
    随后 dispatch 执行（父转 SUSPENDED）→ 本 Act 末尾路由到 SuspendStep。usage.prompt_tokens 是本轮真实计数。

    task-4 复审修复：driver 只在**步骤边界**切换 state.origin，此刻仍是 Act 步骤执行期间、
    origin 还是 LOOP_ACT——但接下来经 maybe_compact_before_dispatch → escalating_compact →
    summarize_for_compact 发起的这次 LLM 调用实际是 compact 摘要，不是 act 的一次 LLM turn。
    不临时切换的话，llm_gateway.stream_llm_resilient 的 origin 门禁会把它错当成 act 调用发出
    LLM_REQUEST_STARTED/LLM_PROMPT_SENT（turn 还取不到值恒为 0），escalating_compact 内部
    make_event(state, ...) 发的 MEMORY_COMPACT_* 也会被错标成 LOOP_ACT。这里临时切到
    LOOP_COMPACT、调用结束（含异常路径）后用 finally 还原，使这次调用的所有事件来源标注
    正确，也走上 summarize_for_compact 里新补的收尾事件。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS
    if not any(tc.name in DISPATCH_TOOLS for tc in tool_calls):
        return
    from ctx_weft.core.loop.steps.compact import maybe_compact_before_dispatch
    prev_origin = state.origin
    state.origin = EventOrigin.LOOP_COMPACT
    try:
        events = await maybe_compact_before_dispatch(state, ctx, prompt_tokens=usage.prompt_tokens)
    finally:
        state.origin = prev_origin
    for ev in events:
        await ctx.event_bus.emit(ev)


async def _execute_tool_calls(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall],
) -> list[dict]:
    """逐个经 CapabilityGateway 执行 tool calls，返回 tool_results。

    工具间 / 执行中命中软打断 → 给本 tc 及之后补「已取消/被打断」result + park（raises HitlPark）；
    硬取消 → CancelledError；被 park 的工具 → task 转 SUSPENDED 并上抛（spec/07 §7）。
    """
    tool_results: list[dict] = []
    ctx.run_phase.in_tool_loop = True

    from ctx_weft.protocols.operations import operation_id_for
    record_id = state.extra.get("assistant_record_id", "")
    for i, tc in enumerate(tool_calls):
        # spec: tool-operations（wp5）——调用侧铸稳定逻辑身份（gateway 保持无状态）：
        # (tenant, session, agent, assistant_record_id, ordinal) 确定性派生 → 跨重启同 id。
        # record_id 缺失（理论不可达 / 测试替身无 provider_ctx）→ None/跳过 →
        # gateway 账本旁路（D6 兼容缝），不阻断执行。
        if ctx.provider_ctx is not None:
            ctx.provider_ctx.operation_id = (
                operation_id_for(
                    getattr(state.session, "tenant_id", "default"), state.session.id,
                    state.agent.id, record_id, i)
                if record_id else None)
        # 工具间命中：软打断 → 本 tc 及之后全部「未开始」→ 补「已取消」+ park；硬取消 → CancelledError。
        if _interrupt_pending(ctx):
            for rest in tool_calls[i:]:
                await _ingest_synthetic_tool_result(state, ctx, rest, CANCELLED_MARK, cancelled=True)
            ctx.run_phase.in_tool_loop = False
            if _is_own_root(state.task):
                from ctx_weft.core.loop.steps.background_observe import launch_background_observe
                launch_background_observe(state, ctx, boundary="interrupt")
            await _park_for_interrupt(state, ctx)
        if ctx.cancel_token is not None and ctx.cancel_token.is_cancelled:
            ctx.cancel_token.raise_if_cancelled()

        # 可取消地执行在途工具：执行中被打断 → 取消该工具、补「被打断」result。
        invoke_task = asyncio.ensure_future(_invoke_tool(tc, state, ctx))
        completed = await _await_tool_or_stop(invoke_task, ctx)
        if not completed:
            if _interrupt_pending(ctx):
                await _ingest_synthetic_tool_result(state, ctx, tc, INTERRUPTED_MARK, interrupted=True)
                for rest in tool_calls[i + 1:]:
                    await _ingest_synthetic_tool_result(state, ctx, rest, CANCELLED_MARK, cancelled=True)
                ctx.run_phase.in_tool_loop = False
                if _is_own_root(state.task):
                    from ctx_weft.core.loop.steps.background_observe import launch_background_observe
                    launch_background_observe(state, ctx, boundary="interrupt")
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
        })

    ctx.run_phase.in_tool_loop = False
    return tool_results


def _reconcile_finish_vs_dispatch(
    state: LoopState, ctx: LoopContext, tool_calls: list[ToolCall],
) -> None:
    """finish_task 与 delegate_task / delegate_plan 同批出现时仲裁：finish 胜出。

    两类工具语义互斥——一个要当前 task 收尾(→observe)，一个要它挂起等子任务(→suspend)。
    用户意图是「我做完了，顺手派生独立后续」：故 finish 胜出，被派发任务从「当前 task 的
    阻塞子任务」改投为「当前 task 的 parent 名下的独立后继」(当前是 root 则为顶层)，自行调度。

    - detach_staged：把本轮 staged 子任务改挂到 parent，切断与收尾 task 的阻塞链。
    - 清 suspend_requested：撤销 delegate 置的挂起意图，使路由走 observe(task.outputs
      已由 finish_task 写好)。
    - 清 spawn_titles：避免 SuspendStep 误报(虽已不路由到 suspend，仍清掉防脏状态)。

    顺序无关：只看本批最终是否两类工具都出现。
    """
    from ctx_weft.core.loop.capability_gateway import DISPATCH_TOOLS
    if ctx.task_manager is None:
        return
    names = {tc.name for tc in tool_calls}
    if FINISH_TASK_NAME not in names or not any(n in DISPATCH_TOOLS for n in names):
        return
    ctx.task_manager.detach_staged(state.task.id, state.task.parent_task_id)
    state.task.suspend_requested = False
    if isinstance(state.task.settings, NormalTaskSettings):
        state.task.settings.spawn_titles = []
    logger.info(
        "act: finish_task + dispatch in same batch on task %s — finishing it; "
        "detaching delegated work to parent %s",
        state.task.id, state.task.parent_task_id,
    )


async def _finish_plain_text_turn(state: LoopState, ctx: LoopContext, turn_num: int) -> None:
    """纯文本回合（无 tool call）收尾。

    interactive 普通任务：请求让位给用户 → `_park_await_user`（有人值守则 HITL input
    冷 park、raises HitlPark，用户回复经 runtime 冷 resume 作 USER_PROMPT 注入后重入
    act）。auto / 非普通任务 / 无 hitl：纯文本即任务产出，发 stop 事件路由 observe。

    下面这条 `stop` 是「没让位」的收尾，与让位严格互斥：真让位了就抛 HitlPark，压根
    走不到这里；`_park_await_user` 正常返回就意味着这一轮不让位（无人值守），它保证
    零副作用返回，所以 `stop` 是这个回合唯一的一条 ACT_TURN_COMPLETED。
    """
    if (
        isinstance(state.task.settings, NormalTaskSettings)
        and state.task.interaction_mode == "interactive"
        and ctx.hitl is not None
    ):
        await _park_await_user(state, ctx, turn_num)
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
    from ctx_weft.core.loop.capability_gateway import CapabilityGateway, InvocationResult
    logger.warning("No CapabilityGateway in LoopContext; tool '%s' cannot be dispatched", tc.name)
    return InvocationResult(
        invocation_id="noop",
        tool_name=tc.name,
        content=f"[Error: CapabilityGateway not configured, tool '{tc.name}' skipped]",
        is_error=True,
    )


INTERRUPTED_MARK = "[interrupted by the user]"
CANCELLED_MARK = "[cancelled]"


async def _ingest_synthetic_tool_result(
    state: LoopState, ctx: LoopContext, tc: ToolCall, content: str, *,
    interrupted: bool = False, cancelled: bool = False,
) -> None:
    """为被打断/未执行的工具补一条 TOOL_RESULT，使 tool_call↔result 一一对应（无 dangling）。"""
    await ctx.memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=state.scope,
            content=content,
            timestamp=now_utc(),
            role="tool",
            metadata={
                "tool_name": tc.name,
                "tool_call_id": tc.id,
                "is_error": True,
                "interrupted": interrupted,
                "cancelled": cancelled,
            },
        ),
        ctx.provider_ctx,
    )


async def _await_tool_or_stop(invoke_task: "asyncio.Future", ctx: LoopContext) -> bool:
    """运行在途工具；任一停止信号（硬取消 cancel_token / 软打断 pause_token）在执行中触发
    → 取消该工具并等其清理跑完，返回 False；正常完成返回 True。
    """
    waiters: list[asyncio.Future] = []
    if ctx.cancel_token is not None:
        waiters.append(asyncio.ensure_future(ctx.cancel_token.wait()))
    if ctx.pause_token is not None:
        waiters.append(asyncio.ensure_future(ctx.pause_token.wait_paused()))
    if not waiters:
        await asyncio.wait({invoke_task})
        return True
    try:
        await asyncio.wait({invoke_task, *waiters}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for w in waiters:
            w.cancel()
        await asyncio.wait(set(waiters))
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
    return f'(I cancelled my previous request: "{prev}" — replacing it with the request below.)\n\n'


def interrupt_edit_note(prev_request: str, new_input: str) -> str:
    """① 打断（未吐 token）续接时的说明：上一条请求被取消、改为新请求。空 prev 时原样返回。"""
    prefix = _interrupt_edit_prefix(prev_request)
    return f"{prefix}{new_input}" if prefix else new_input


async def _cold_park(
    state: LoopState, ctx: LoopContext, preface: str, *, unattended: bool,
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
    )
    # 不建等待槽 —— 本调用方随即 park 释放协程而非 await，应答必然走冷续跑。
    raise HitlPark(hitl_id=req.id)


async def _park_for_interrupt(
    state: LoopState, ctx: LoopContext, *, edit: bool = False,
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
    await _cold_park(state, ctx, preface, unattended=False)


async def _park_await_user(state: LoopState, ctx: LoopContext, turn_num: int) -> None:
    """agent 说完一段纯文本、想让位给用户 → 发 await_user + 折叠 + 冷 park。

    与 `_park_for_interrupt` 相反，这条**没有人保证会回来**：让位是 agent 自己提的，
    无人值守的 task 里根本没人会发下一条消息，park 即永久挂起。所以守卫在这里生效。

    **判断前置于一切副作用**，这是本函数存在的全部意义：`hitl.open()` 里的守卫抛
    `UnattendedHitl` 时，`await_user` 事件早已发出、background observe 早已起飞，
    两个结局都坏——异常逸出打挂整个 run，或就地 catch 继续走、同一回合发出两条
    `ACT_TURN_COMPLETED`（`await_user` 一条、`stop` 一条），前端看到的是「agent 说
    它在等用户，紧接着又说它停了」。

    这里**刻意不包 `try/except UnattendedHitl` 兜底**：判断读的和守卫查的是同一个
    `Task.unattended`，只有一个真相源，不存在漂移。守卫仍留在 `open()` 里，所以将来
    若有人加了新路径又忘了前置判断，行为是当场抛异常而不是静默挂死——那正是我们要的。

    让位与不让位严格互斥：让位则抛 `HitlPark`，调用方那条 `stop` 根本到不了；不让位
    则零副作用返回，`stop` 是这个回合唯一的收尾事件。
    """
    if state.task.unattended:
        logger.warning(
            "act: task %s is unattended — not yielding to the user after a plain-text turn; "
            "treating the text as this turn's output and stopping normally",
            state.task.id,
        )
        return
    await ctx.event_bus.emit(make_event(state, EventType.ACT_TURN_COMPLETED, payload={
        "turn": turn_num, "reason": "await_user"}))
    # 纯文本暂停 = 软待命(允许但不强制回复) → PAUSED,区别于 ask_user 的 PAUSED_HITL。
    if _is_own_root(state.task):
        from ctx_weft.core.loop.steps.background_observe import launch_background_observe
        launch_background_observe(state, ctx, boundary="plain_text")
    await _cold_park(state, ctx, PREFACE_NORMAL, unattended=state.task.unattended)


async def _interrupt_checkpoint(state: LoopState, ctx: LoopContext) -> None:
    """协作式停止点：软打断（pause）→ park；硬取消（cancel）→ CancelledError。

    提交点之前命中（act 第一轮的循环顶部，LLM 还没被调用过）→ 整轮丢弃而非 park，
    见 `_discard_round_if_uncommitted`。
    """
    if _interrupt_pending(ctx):
        await _discard_round_if_uncommitted(state, ctx)   # 命中则 raises RoundDiscarded
        edit = not ctx.run_phase.produced and not ctx.run_phase.in_tool_loop
        if _is_own_root(state.task):
            from ctx_weft.core.loop.steps.background_observe import launch_background_observe
            launch_background_observe(state, ctx, boundary="interrupt")
        await _park_for_interrupt(state, ctx, edit=edit)  # raises HitlPark
    tok = ctx.cancel_token
    if tok is not None and tok.is_cancelled:
        tok.raise_if_cancelled()
