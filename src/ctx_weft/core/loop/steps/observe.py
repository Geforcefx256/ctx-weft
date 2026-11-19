"""Observe execution: task judgement (optional) and execution-history recap.

The driver selects foreground/background execution. This step owns observation,
judgement and settlement in both modes.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from contextlib import AsyncExitStack, aclosing, nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Any

from ctx_weft.core.assembler import ContextRequest, ReviewIntent
from ctx_weft.core.assembler.prompts import runtime_prompt
from ctx_weft.core.capabilities.control_tools import REPORT_OBSERVATION_NAME
from ctx_weft.core.capabilities.control_tools import ControlMetaKey as K
from ctx_weft.core.loop.driver import (
    ACT_TURNS_USED_KEY,
    ExecutionMode,
    LoopContext,
    LoopState,
    Step,
    StepDriver,
    StepExecution,
    StepOutcome,
    make_event,
)
from ctx_weft.core.loop.llm_gateway import (
    build_llm_request,
    llm_request_id,
    resolve_llm_identity,
    stream_llm_resilient,
)
from ctx_weft.core.loop.llm_stream import ensure_collected_stream
from ctx_weft.core.loop.task_memory_fold import (
    find_finish_target,
    replace_finish_report,
)
from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind, disposition_for
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.content import image_tokens
from ctx_weft.protocols.content import content_to_text
from ctx_weft.core.utils.ids import generate_id, mint_turn_call_ids
from ctx_weft.core.utils.verdict import (
    VERDICT_CONTINUE,
    VERDICT_FAIL,
    VERDICT_SUCCESS,
    VERDICT_WAITING,
    normalize_verdict,
)
from ctx_weft.protocols import LLMMessage, MemoryScope, load_records
from ctx_weft.protocols.events import EventType

if TYPE_CHECKING:
    from ctx_weft.core.assembler.context import PromptContext
    from ctx_weft.core.capabilities.control_tools import ControlResult

logger = logging.getLogger(__name__)


CLOSE_BOUNDARIES = {"finish", "normal"}
JUDGING_BOUNDARIES = {"plain_text", "finish_park", "task_switch"}


def judges(boundary: str) -> bool:
    return boundary in JUDGING_BOUNDARIES


def _execution(ctx) -> StepExecution:
    return getattr(ctx, "execution", StepExecution())


@dataclass
class Verdict:
    """Final business judgement, not a task-status transition."""
    task_outcome: str
    act_recap: str
    task_summary: str = ""
    reported: bool = False
    next_step_hint: str = ""
    failure_reason: str = ""


@dataclass
class ObservationResult:
    """A recap may exist without a verdict, and a mechanical verdict without a recap."""
    act_recap: str = ""
    task_summary: str = ""
    verdict: Verdict | None = None

    def metadata(self) -> dict:
        result = {K.OBSERVER_ACT_RECAP: self.act_recap,
                  K.OBSERVER_TASK_SUMMARY: self.task_summary}
        if self.verdict is not None:
            result.update({
                K.OBSERVER_OUTCOME: self.verdict.task_outcome,
                K.OBSERVER_NEXT_STEP_HINT: self.verdict.next_step_hint,
                K.OBSERVER_FAILURE_REASON: self.verdict.failure_reason,
            })
        return result


class ObserveStep(Step):
    name = "observe"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        execution = _execution(ctx)
        boundary = self.boundary_or_exit(state, execution.options.get("boundary"))
        background = execution.mode == ExecutionMode.BACKGROUND
        judge_task = execution.options.get(
            "judge_task", (boundary == "task_switch" or judges(boundary)) if background else True)
        await begin(state, ctx, boundary, judge_task)
        completion = state.extra.get("observe_completion")
        decision = completion.get("disposition") if completion else None
        if state.task.status == "CANCELED":
            if not completion:
                await complete(state, ctx, ObservationResult(), accepted=False, action="keep_raw")
            await applied(state, ctx, reason="task_already_terminal")
            return self._outcome(restore_result(completion) if completion else ObservationResult())
        if (decision and state.task.status in {"FINISHED", "FAILED", "CANCELED"}
                and state.task.status != decision["status"]):
            await applied(state, ctx, reason="task_already_terminal")
            return self._outcome(restore_result(completion))
        if not judge_task or background:
            return await self._execute_background(state, ctx, boundary, judge_task)

        events = []
        if completion:
            result = restore_result(completion)
        else:
            result = await await_foreground(
                self._observe(state, ctx, events, True, boundary), state, ctx)
        await foreground_stop_checkpoint(state, ctx)
        if result.verdict and result.verdict.task_outcome == VERDICT_WAITING:
            await ensure_waiting(state, ctx)
        if not completion:
            await complete(state, ctx, result, disposition=planned_disposition(state, result)
                           if result.verdict else None, action="foreground")
        # Persist the decision before any fold, but do not start applying it when
        # a stop arrived during the commit. Never cancel an uncertain event commit.
        await foreground_stop_checkpoint(state, ctx)
        apply_foreground_assessment(state, result, boundary)
        if result.verdict and result.verdict.task_outcome in (VERDICT_CONTINUE, VERDICT_WAITING):
            await await_foreground(
                self._fold_retry_segment(state, ctx, result.verdict, events), state, ctx)
        return self._outcome(result, "finalize", events)

    @staticmethod
    def _outcome(result, next_step=None, events=None):
        patch = {"observation": result}
        if result.verdict is not None:
            patch["verdict"] = result.verdict
        return StepOutcome(next_step, state_patch=patch, events=events or [])

    async def _observe(self, state, ctx, events, judge_task, boundary):
        result = ObservationResult()
        tok = getattr(ctx, "cancel_token", None)
        background = _execution(ctx).mode == ExecutionMode.BACKGROUND
        cancelled = tok is not None and tok.is_cancelled
        if ((background or not cancelled)
                and (boundary == "task_switch" or not judge_task or self._should_use_llm(state))):
            try:
                result = await self._llm_observe(state, ctx, events)
            except Exception as exc:
                logger.warning("ObserveStep LLM failed; retaining raw history: %s", exc)
        if judge_task:
            verdict = result.verdict
            if boundary == "max_turns" and (verdict is None or verdict.task_outcome != VERDICT_FAIL):
                # At the turn limit, only an explicit inability verdict ends
                # the task. Other results (including report failure) continue.
                verdict = (dataclasses.replace(verdict, task_outcome=VERDICT_CONTINUE,
                                               task_summary="") if verdict is not None else
                           Verdict(VERDICT_CONTINUE, result.act_recap))
                result.task_summary = ""
            elif verdict is None:
                verdict = dataclasses.replace(
                    self._mechanical_verdict(state, boundary),
                    act_recap=result.act_recap, task_summary=result.task_summary)
            result.verdict = verdict
            if boundary == "task_switch" and not result.act_recap:
                result.act_recap = verdict.act_recap or verdict.failure_reason
                result.verdict = dataclasses.replace(verdict, act_recap=result.act_recap)
        return result

    def boundary_or_exit(self, state, boundary=None) -> str:
        return boundary or state.act_exit_reason or "normal"

    async def _llm_observe(self, state, ctx, events) -> ObservationResult:
        execution = _execution(ctx)
        background = execution.mode == ExecutionMode.BACKGROUND
        boundary = self.boundary_or_exit(state, execution.options.get("boundary"))
        judge_task = execution.options.get(
            "judge_task", (boundary == "task_switch" or judges(boundary)) if background else True)
        cache = ctx.capability_cache
        # Session-global control tools survive per-agent cache eviction.
        bound = cache.get(state.agent.id) if cache is not None else []
        extra = {"observe_boundary": boundary,
                 "observe_index": execution.options.get("observe_index"),
                 "observe_act_turns": execution.options.get("act_turns")}
        record_ids = execution.options.get("context_record_ids")
        if record_ids is not None:
            extra["observe_record_ids"] = list(record_ids)
        else:
            extra["observe_end_at"] = _execution(ctx).started_at
        context_task = state.extra.get("observe_context_task", state.task)
        if judge_task:
            extra["subtasks"] = self._subtask_handles(state, ctx)
        request = ContextRequest(
            purpose="observe", scope=state.scope, task=context_task, agent=state.agent,
            session=state.session,
            resolved_model=state.resolved_model, template=state.extra.get("template"),
            bound_capabilities=bound, token_counter=ctx.llm.tokenizer.count,
            extra=extra, intent=ReviewIntent(judge_task=judge_task, terminal_only=boundary == "task_switch"),
        )
        inputs = await ctx.assembler.assemble(request)
        submitted, last_text = await self._run_react(
            state, ctx, inputs=inputs, max_rounds=state.agent.loop_config.max_turns_per_observe,
            terminal_tool_name=REPORT_OBSERVATION_NAME,
            validate_submission=lambda result: self._submission_error(
                result, judge_task=judge_task, terminal_only=boundary == "task_switch",
                waiting_allowed=not getattr(state.task, "unattended", False) and getattr(ctx, "hitl", None) is not None),
        )
        if record_ids is not None and all(record_id in inputs._records for record_id in record_ids):
            # The gateway's compiler loaded these immutable records for prompting.
            records = [inputs._records[record_id] for record_id in record_ids]
            records.sort(key=lambda r: (r.timestamp, r.metadata.get("seq_no", 0)))
            state.extra["observe_record_cache"] = ((state.extra.get("observe_id"), tuple(record_ids)), records)
        return self._interpret_submission(state, submitted, last_text,
                                          judge_task=judge_task, background=background, terminal_only=boundary == "task_switch")

    async def _run_react(
        self,
        state: "Any",
        ctx: "Any",
        *,
        inputs: "PromptContext",
        max_rounds: int,
        terminal_tool_name: str,
        validate_submission: Callable[[Any], str | None] | None = None,
    ) -> "tuple[ControlResult | None, str]":
        """Run observation rounds until a valid terminal submission or the round limit.

        Returns the accepted tool result (or None) and the last nonempty text.
        Only advertised tools may execute; submission validation comes from this step.
        This method does not apply task judgement or fold memory.

        The driver supplies the foreground/background event origin. The LLM gateway
        emits streaming events; this method emits LLM_RESPONSE_FINISHED.
        """
        agent = state.agent
        model, llm_account = resolve_llm_identity(state)
        appended_messages: list[LLMMessage] = []
        last_text = ""
        # 动态 max_tokens 的增量基线：上一轮实际发送条数（本轮 usage 对应的真实 prompt 基线）。
        baseline_msg_count: int | None = None
        idle_seconds = getattr(getattr(ctx, "config", None), "observe_llm_idle_timeout_sec",
                               RuntimeConfig.observe_llm_idle_timeout_sec)
        loop = asyncio.get_running_loop()

        for round_num in range(max_rounds):
            # req_id：与 llm_gateway.stream_llm_resilient 内部同一确定性公式独立算出（同
            # act.py._run_llm_turn 的手法）——两边都在本次 LLM 调用任何事件发射前求值，故
            # state.sequence_counter 两处读到同一个值，天然一致，不需要新增参数或跨函数传值。
            req_id = llm_request_id(state)

            llm_request = await build_llm_request(
                ctx, state, inputs, appended_messages=appended_messages,
                baseline_msg_count=baseline_msg_count, turn=round_num)
            sent_msg_count = len(llm_request.messages)
            tools = llm_request.tools
            # turn：gateway 从 metadata["turn"] 取（spec §9.4 统一口径：LLM_* 事件一律用 turn，
            # 不用 round）；round_num 仍是本函数内部的轮次计数局部变量，只是不再直接进 payload。

            stream = ensure_collected_stream(stream_llm_resilient(
                ctx, state, llm_request, inputs=inputs, appended_messages=appended_messages))
            try:
                # The gateway's first-chunk deadline is disarmed once output
                # starts. HTTP heartbeats can then keep its socket alive forever.
                # Bound time without model progress, including hidden retries,
                # while allowing healthy reasoning/tool streams to run longer.
                async with asyncio.timeout(idle_seconds) as idle_timeout:
                    async with aclosing(stream):
                        async for chunk in stream:
                            if ((chunk.kind in {"token", "reasoning"} and chunk.text)
                                    or chunk.kind in {"tool_call", "tool_call_partial"}):
                                idle_timeout.reschedule(loop.time() + idle_seconds)
                            stream.accept_current()
            except TimeoutError as exc:
                if idle_timeout.expired():
                    raise TimeoutError(
                        f"Observe LLM made no progress for {idle_seconds:g} seconds") from exc
                raise
            result = stream.get_final_result()
            accumulated_text, reasoning_text = result.text, result.reasoning
            tool_calls, usage = list(result.tool_calls), result.usage

            if accumulated_text:
                last_text = accumulated_text

            # 摄入前铸造内部调用标识（spec: conversation-integrity，与 act._run_llm_turn 同口径）：
            # observer 回合不入 task 层 memory（工具是 SILENT/控制面），锚仅取唯一性、不持久；
            # live 消息面（appended_messages 的 tool_call↔result 配对）与事件 payload 用同一份值。
            if tool_calls:
                tool_calls = [
                    m.call for m in mint_turn_call_ids(
                        tool_calls, anchor=generate_id("asst"), turn_seq=round_num)
                ]

            # 更新 loop_guard（对齐 miniAgents _run_observer：取 actor/observer 的最大值）
            if usage.prompt_tokens > 0:
                agent.loop_guard.context_tokens = max(
                    agent.loop_guard.context_tokens, usage.prompt_tokens
                )
                baseline_msg_count = sent_msg_count  # 真实刷新才前移基线（否则下轮退回整份估算）
            # 同步累加 session.token_used
            state.session.token_used += usage.prompt_tokens + usage.completion_tokens

            await ctx.event_bus.emit(make_event(
                state, EventType.LLM_RESPONSE_FINISHED,
                payload={
                    "request_id": req_id,
                    "content": accumulated_text,
                    "reasoning": reasoning_text,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                    ],
                    "usage": dataclasses.asdict(usage),
                    # 本次调用实际使用的模型/账号（host 云端上报按此计账，不受切换竞态影响）
                    "llm_model": model,
                    "llm_account": llm_account,
                    "finish_reason": "tool_use" if tool_calls else "stop",
                    "turn": round_num,
                },
            ))

            if not tool_calls:
                # 纯文本轮不直接放弃：observer 常把复述写成正文而忘了调 terminal 工具——
                # 还有剩余轮次时催促其改用工具提交后重试；耗尽轮次才返回 (None, last_text)。
                if round_num + 1 >= max_rounds:
                    break
                appended_messages.append(LLMMessage(
                    role="assistant", content=accumulated_text or runtime_prompt('no_reply'),
                    provider_state=result.provider_state,
                ))
                appended_messages.append(LLMMessage(
                    role="user",
                    content=runtime_prompt("react_nudge", terminal_tool=terminal_tool_name),
                ))
                continue

            appended_messages.append(LLMMessage(
                role="assistant",
                content=accumulated_text,
                tool_calls=[{"id": tc.id, "name": tc.name, "input": tc.arguments} for tc in tool_calls],
                provider_state=result.provider_state,
            ))

            terminal_result = None
            for tc in tool_calls:
                if tc.name not in {tool.name for tool in tools}:
                    content = runtime_prompt("observation_tool_unavailable", tool=tc.name)
                elif ctx.capability_gateway is not None:
                    result = await ctx.capability_gateway.invoke(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        state=state,
                        ctx=ctx,
                        tool_call_id=tc.id,
                    )
                    content = result.content
                    # terminal 工具的失败（is_error，如参数非法）不终止循环：错误内容照常 append
                    # 进 messages（下方）供模型下一轮改参重试；只有成功结果才是终止结果。否则一次
                    # 坏调用的错误文案会被当成 Process Report 终结整个观察（spec: capability-gateway）。
                    if tc.name == terminal_tool_name and not result.is_error:
                        error = validate_submission(result) if validate_submission else None
                        if error:
                            content = error
                        else:
                            terminal_result = result
                else:
                    logger.warning(
                        "run_observe_react: no CapabilityGateway for tool '%s'", tc.name
                    )
                    content = runtime_prompt('gateway_missing', tool=tc.name)
                appended_messages.append(LLMMessage(
                    role="tool",
                    content=content,
                    tool_call_id=tc.id,
                ))

            if terminal_result is not None:
                return terminal_result, last_text

        return None, last_text

    def _interpret_submission(self, state, submitted, last_text="", *, judge_task=True, background=False, terminal_only=False) -> ObservationResult:
        meta = (submitted.metadata or {}) if submitted is not None else {}
        recap = meta.get(K.OBSERVER_ACT_RECAP, "") or ""
        # Preserve the existing background plain-text recap fallback; it is never a verdict.
        if background and not recap:
            recap = content_to_text(last_text or "").strip()
        elif submitted is not None and not recap:
            recap = content_to_text(last_text or "")[:500]
        result = ObservationResult(recap, meta.get(K.OBSERVER_TASK_SUMMARY, "") or "")
        if judge_task and submitted is not None and meta.get(K.OBSERVER_OUTCOME):
            outcome = normalize_verdict(meta[K.OBSERVER_OUTCOME])
            hint = meta.get(K.OBSERVER_NEXT_STEP_HINT, "") or ""
            hint = f"Next Step Hint: {hint}" if hint else ""
            # A successful model verdict still requires an actual final deliverable.
            if outcome == VERDICT_SUCCESS and not observation_task(state).outputs and not terminal_only:
                outcome = VERDICT_CONTINUE
                missing_output = runtime_prompt("observation_missing_output")
                hint = f"{hint}\n\n{missing_output}" if hint else missing_output
            failure_reason = meta.get(K.OBSERVER_FAILURE_REASON, "") or ""
            if outcome in (VERDICT_SUCCESS, VERDICT_WAITING):
                failure_reason = ""
            result.verdict = Verdict(outcome, result.act_recap, result.task_summary,
                                     reported=True, next_step_hint=hint,
                                     failure_reason=failure_reason)
        return result

    def _submission_error(self, result, *, judge_task=True, terminal_only=False, waiting_allowed=True) -> str | None:
        if (judge_task and not waiting_allowed
                and (result.metadata or {}).get(K.OBSERVER_OUTCOME) == VERDICT_WAITING):
            return "waiting requires an available user and HITL service. Continue executable work, or report fail if the goal cannot be achieved."
        if terminal_only:
            meta = result.metadata or {}
            if meta.get(K.OBSERVER_OUTCOME) not in (VERDICT_SUCCESS, VERDICT_FAIL):
                return "This review requires a terminal task_status: success or fail. continue and waiting are not allowed."
            if meta.get(K.OBSERVER_OUTCOME) == VERDICT_FAIL and not str(meta.get(K.OBSERVER_FAILURE_REASON) or "").strip():
                return "A failed terminal review requires task_failure_reason describing the unfinished goal."
            if meta.get(K.OBSERVER_NEXT_STEP_HINT):
                return "Leave next_step_hint empty; this task cannot resume."
        if judge_task and not str((result.metadata or {}).get(K.OBSERVER_OUTCOME) or "").strip():
            return runtime_prompt("observation_missing_verdict")
        return None

    def _mechanical_verdict(self, state, boundary=None) -> Verdict:
        # Only called after an observation produced no usable judgement.
        return Verdict(VERDICT_SUCCESS, "")

    @staticmethod
    def _apply_assessment(task, verdict: Verdict) -> None:
        """Apply an accepted foreground assessment, never task.status."""
        task.process_report = verdict.act_recap
        task.process_report_at = now_utc()
        task.task_summary = verdict.task_summary
        task.next_step_hint = verdict.next_step_hint or None
        task.observer_outcome = verdict.task_outcome
        task.actor_done = True
        task.error = None if verdict.task_outcome == VERDICT_SUCCESS else (
            verdict.failure_reason or None)

    def _should_use_llm(self, state) -> bool:
        return True

    @staticmethod
    def _subtask_handles(state, ctx) -> list[dict]:
        tm = getattr(ctx, "task_manager", None)
        if tm is None or not hasattr(tm, "children_of"):
            return []
        out = []
        for cid in tm.children_of(state.task.id):
            child = tm.get_task(cid)
            if child is None:
                continue
            entry = {"task_id": child.id, "title": child.title or "",
                     "outcome": (child.status or "").lower()}
            if child.status == "CANCELED" and child.error_code:
                entry["note"] = child.error or child.error_code
            out.append(entry)
        return out

    async def _is_short_segment(
        self,
        state: "LoopState", ctx: "LoopContext", segment_end_at: "datetime | None" = None,
    ) -> bool:
        """短段免折门：**当前段**（最后一条 active USER_PROMPT 之后）满足以下任一即为短段：

        - 段内 LLM 回复（role=assistant 回合）≤ 1 条——**不看 token**：一条回复折成摘要
          是净亏（recap 常比原文还长，且原文对下一轮信息更全），折它只是白花一次 LLM；
        - 段内 raw token ≤ short_segment_token_threshold。

        「短 → 原文成胶囊」决策（finalize._is_short_leaf）在段级的判定，
        `ObserveStep._execute_background`（plain_text/interrupt 边界）与
        `ObserveStep._fold_retry_segment`（retry 段折）共用。配置缺失（手构 state /
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
        view = await context_records(state, ctx)
        if view is not None:
            view = [r for r in view if r.scope == MemoryScope.TASK and r.address.task_id == state.task.id
                    and r.kind in (MemoryKind.CONVERSATION_TURN, MemoryKind.TOOL_AUDIT)]
        else:
            view = await load_records(ctx.memory,
                state.scope, MemoryScope.TASK, ctx.provider_ctx,
                kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.TOOL_AUDIT],
            )
        # 段结束时间（2026-09-22，同 `segment_fold`）：只认这段后台 observe 开跑时就已存在
        # 的记录。不设限的话，人中途说的话会成为新段界，把「当前段」判成空段而误判为短段。
        if segment_end_at is not None:
            view = [r for r in view if r.timestamp <= segment_end_at]
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

    async def _fold_retry_segment(
        self, state: LoopState, ctx: LoopContext, verdict: Verdict, events: list[Any]
    ) -> None:
        """retry 前台同步段折：本轮 attempt raw → 一条 TASK_COMPACT_SUMMARY（复用 act_recap），
        supersede 本轮全部 raw（keep_last=0），保 USER_PROMPT 锚 + 既往段摘要（累积）。
        protect TASK_COMPACT_SUMMARY → 多轮 retry 段摘要累积（不替换），由 L3 按 collapse_keep_last
        坍缩控界。无 keep_last 门、无额外 LLM。

        continue / waiting 都保留未完成任务的段摘要；
        其余 outcome 不折，no-op。

        act_recap 来自本轮 observation；空则不折、段保 raw（不写占位摘要）。
        短段免折（ObserveStep._is_short_segment，与段边界折叠同门）：当前段（末条 UP
        之后）active raw 低于 short_segment_token_threshold 时不折——recap 常比短原文更长，
        raw 原样留给下个 attempt 反而信息更全；attempt 之间无新 UP，raw 在段内跨 attempt
        累积，超阈值后下一次 retry 一并折叠。折叠带 since_last=USER_PROMPT（段作用域，
        2026-07-21）：交互任务中若有免折残留的前段 raw，不被跨段合折。
        """

        if verdict.task_outcome not in (VERDICT_CONTINUE, VERDICT_WAITING) or state.extra.get("observe_keep_raw"):
            return
        summary = (verdict.act_recap or "").strip()
        if not summary:
            logger.warning(
                "_fold_retry_segment: empty act_recap for task=%s; skip fold, segment kept raw",
                state.task.id,
            )
            return
        if await self._is_short_segment(state, ctx):
            logger.info(
                "_fold_retry_segment: short segment kept raw (task=%s); skip fold",
                state.task.id,
            )
            return
        # 与后台观察共用段作用域原子折叠，保护 user 回合与既有摘要。
        result = await ctx.memory.organize(
            ctx.provider_ctx, operation="fold_segment", summary=summary,
            record_ids=await _segment_input_ids(state, ctx),
        )
        events.append(make_event(state, EventType.MEMORY_COMPACTED, payload={
            "events_before": result.records_before,
            "events_after": result.records_after,
            "summary_event_id": (result.replacements[0].id if result.replacements else ""),
            "summary_length": len(summary),
            "layer": "task",
            "trigger": "observe_retry",
        }))

    async def _execute_background(self, state, ctx, boundary, judging):
        async with AsyncExitStack() as stack:
            options = _execution(ctx).options
            wait_for_park = (judging and options.get("waiting_hitl_id")
                             and not state.extra.get("observe_completion"))
            ready = None
            if boundary in CLOSE_BOUNDARIES or options.get("verdict") or wait_for_park:
                ready = await stack.enter_async_context(
                    StepDriver.for_context(ctx).executions.finish_notification(state.task.id))
            if options.get("verdict") or wait_for_park:
                from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT
                async def wake(event):
                    if event.task_id == state.task.id and (
                            event.type in TASK_STATUS_BY_EVENT or event.type == EventType.HITL_RESOLVED):
                        ready.set()
                subscription = ctx.event_bus.subscribe(None, wake)
                stack.push_async_callback(subscription.unsubscribe)
            if wait_for_park and not await wait_producer_park(state, ctx, ready):
                result = self._outcome(ObservationResult())
            else:
                result = await self._background_body(state, ctx, boundary, judging)
            if not state.extra.get("observe_completion"):
                await complete(state, ctx, ObservationResult(), accepted=False, action="keep_raw")
            completion = state.extra["observe_completion"]
            reason = ""
            if (boundary in CLOSE_BOUNDARIES and completion["action"] == "recap"
                    and result.state_patch["observation"].act_recap.strip()):
                reason = await apply_close_report(state, ctx, result.state_patch["observation"], ready) or ""
            if _execution(ctx).options.get("verdict"):
                await wait_disposition(state, ctx, ready)
            await applied(state, ctx, reason=reason)
            return result

    async def _background_body(self, state, ctx, boundary, judging):
        if boundary == "task_switch":
            ready = getattr(state.task, "switch_review_ready", None)
            if ready is not None:
                await ready.wait()
        segment_end_at = _execution(ctx).started_at
        completion = state.extra.get("observe_completion")
        short_segment = completion.get("short_segment", False) if completion else False
        async with StepDriver.for_context(ctx).executions.task_guard(state.task.id, "observe"):
            if not completion and boundary == "task_switch" and state.task.status != "REVIEWING":
                return self._outcome(ObservationResult())
            # 重跑幂等护栏（恢复重跑时才生效）：非 close 边界若该段已无 active raw，说明上次
            # 崩溃前已折叠（raw 被 supersede），再折会产冗余胶囊 → 跳过（finally 仍发 DONE）。
            # 正常运行时该段刚产生 raw、计数 > 0，护栏为 no-op。
            if not completion and boundary not in CLOSE_BOUNDARIES and boundary != "task_switch":
                from ctx_weft.protocols import MemoryKind
                view = await load_records(ctx.memory,
                    state.scope, MemoryScope.TASK, ctx.provider_ctx,
                    kinds=[MemoryKind.CONVERSATION_TURN],
                )
                # 旧口径 = LLM_RESPONSE 计数 = assistant 回合
                n_raw = sum(1 for r in view if r.role == "assistant")
                if n_raw == 0 and not judging:
                    logger.info(
                        "task recap re-fold guard: segment already folded (task=%s); skip",
                        state.task.id,
                    )
                    return self._outcome(ObservationResult())
                # 短段免折（_is_short_segment）：当前段（末条 UP 之后）active raw 低于阈值时
                # 跳过折叠——花一次后台 LLM 调用换一段常比原文还长的摘要不划算。跳过 = 该段
                # **永久**保 raw（与观察失败的降级同语义）：后续折叠带 since_last=USER_PROMPT
                # 只折各自的当前段，免折残留不会被跨段合折。
                short_segment = await self._is_short_segment(state, ctx, segment_end_at)
                if short_segment and not judging:
                    logger.info(
                        "short segment kept raw (task=%s boundary=%s); skip fold",
                        state.task.id, boundary,
                    )
                    return self._outcome(ObservationResult())
            if boundary == "task_switch":
                short_segment = False
            try:
                observation = (restore_result(completion) if completion else
                               await self._observe(state, ctx, [], judging, boundary))
                state.extra["observe_short_segment"] = short_segment
                meta = observation.metadata()
                act_recap = observation.act_recap.strip()
                if not act_recap and not judging:
                    # 无任何可用报告：与异常路径同语义——段保 raw，不写占位摘要、不动 finish 对。
                    logger.warning(
                        "background observe produced no usable report (task=%s boundary=%s); "
                        "segment kept raw", state.task.id, boundary,
                    )
                    return self._outcome(ObservationResult())
                # **判决先落地，折叠再决定**（2026-09-27）：判 success 会经带外 finalize 走
                # close（写 finish 对 + 折末段 raw），而那条路**刻意不产段摘要**——
                # `finalize._supersede_final_raw_segment` 的不变量是「末段 raw 与真实 Process
                # Report 至少存其一」，段摘要与 finish 对同时存在就是把同一段记了两遍。所以
                # 「这一段归谁记账」只有拿到判决才知道，不能先折了再判。
                #
                # 判 waiting、判决缺失、仲裁被拒（人先开口）→ 没有 close，段摘要照产；
                # 人先开口那一支还多一层保障：段结束时间让折叠只认这次 launch 之前的记录，
                # 新那条 user 回合不会被卷进来。
                if not judging:
                    await complete(state, ctx, observation, action="recap")
                state.extra["observation_result"] = observation
                closed = await submit_verdict(state, ctx, meta) if judging else False

                usable_recap = bool(act_recap) and (not judging or
                    observation.verdict is not None and observation.verdict.reported)
                if (usable_recap and boundary not in CLOSE_BOUNDARIES and not short_segment
                        and not closed and not state.extra.get("observe_continued")
                        and not state.extra.get("observe_keep_raw")):
                    # v2 P3c：策展上移——段作用域折叠（护 user 回合与既有段摘要、锚点/
                    # 段尾语义，与 observe._fold_retry_segment 同门）由框架侧 segment_fold
                    # 执行原子 fold。旧 apply_compact 的 TypeError 协议错配特判随之消亡
                    # （segment_fold 是框架内函数，签名错配不再是运行时 provider 风险）。
                    #
                    # `short_segment` 为真时跳到这里之外：**免折不免判**（2026-09-22）——
                    # 判定必须跑（短回合恰恰是提问最典型的形态），但短段的 recap 常比原文
                    # 还长，折它是净亏，段保 raw。
                    async with StepDriver.for_context(ctx).executions.task_guard(state.task.id, "finish_report"):
                        # Cancellation owns the final memory once its durable task
                        # transition wins. A rejected old verdict must not fold raw.
                        if (state.task.status != "CANCELED" and not (
                                ctx.task_manager is not None and hasattr(ctx.task_manager, "is_task_tree_canceled")
                                and ctx.task_manager.is_task_tree_canceled(state.task.id))):
                            await ctx.memory.organize(
                                ctx.provider_ctx, operation="fold_segment", summary=act_recap,
                                record_ids=await _segment_input_ids(state, ctx),
                            )
                return self._outcome(observation)
            except Exception:
                if state.extra.get("observe_completion"):
                    raise
                logger.exception("background observe failed (ignored); segment kept raw")
                return self._outcome(ObservationResult())


async def wait_producer_park(state, ctx, ready):
    """Do not arbitrate a fresh parked verdict before its producer has settled."""
    seconds = getattr(getattr(ctx, "config", None), "observe_llm_idle_timeout_sec",
                      RuntimeConfig.observe_llm_idle_timeout_sec)
    try:
        async with asyncio.timeout(seconds):
            while True:
                ready.clear()
                req = ctx.hitl.registry.get(_execution(ctx).options["waiting_hitl_id"])
                # Subscribed before reading: a past transition is already visible,
                # and a later transition wakes us. A user reply makes the old
                # verdict obsolete, so let the existing arbitration reject it.
                if (state.task.status != "ACTIVE" or req is None
                        or req.effective_decision is not None):
                    return True
                await ready.wait()
    except TimeoutError:
        logger.warning("observe producer did not park within %g seconds; keeping raw history", seconds)
        return False


async def await_foreground(work, state, ctx):
    """Stop cancellable observation work, joining transaction cleanup before handoff."""
    from ctx_weft.core.loop.steps.act import _await_tool_or_stop

    pending = asyncio.create_task(work)
    try:
        await _await_tool_or_stop(pending, ctx)
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    # A failed durable operation must propagate its original error before any
    # stop handoff tries to write more events to a possibly isolated session.
    result = pending.result() if not pending.cancelled() else None
    # Check even if the operation completed concurrently with the stop signal.
    await foreground_stop_checkpoint(state, ctx)
    if pending.cancelled():
        raise asyncio.CancelledError
    return result


async def foreground_stop_checkpoint(state, ctx):
    """Retire the old decision before parking; cancellation settles it after task cleanup."""
    identity = getattr(state, "extra", {}).get("observe_id")
    if not identity:
        return
    tok = getattr(ctx, "cancel_token", None)
    if tok is not None and tok.is_cancelled:
        await complete(state, ctx, ObservationResult(), accepted=False, action="keep_raw")
        # Runtime carries this identity through the canceled RunOutcome. Applied
        # must follow TaskCanceled and its memory/HITL finalization, not precede it.
        tok.raise_if_cancelled()
    pause = getattr(ctx, "pause_token", None)
    if pause is None or not pause.is_paused or getattr(ctx, "hitl", None) is None:
        return
    from ctx_weft.core.loop.steps.act import _park_for_interrupt

    closure = []
    if not state.extra.get("observe_completion"):
        closure.append(make_event(state, EventType.OBSERVE_COMPLETED, {
            "task_id": state.task.id, "observe_id": identity,
            "accepted": False, "action": "keep_raw",
            "result": dataclasses.asdict(ObservationResult()),
            "completed_at": now_utc().isoformat(), "short_segment": False}))
    # A persisted Completed remains immutable. Atomically retire it with the new
    # parked request, so recovery cannot replay its old continue disposition.
    closure.append(make_event(state, EventType.OBSERVE_APPLIED, {
        "task_id": state.task.id, "observe_id": identity,
        "reason": "foreground_interrupted"}))
    await _park_for_interrupt(state, ctx, edit=True, preceding_events=tuple(closure),
                              reply_anchor=_execution(ctx).options.get("reply_anchor"),
                              replaces_observe_id=identity,
                              replace_user_turn_id=state.extra.get("waiting_hitl_id", ""))


_PARK_CLOSED_BY_VERDICT = runtime_prompt('park_closed_by_verdict')

async def submit_verdict(state: "LoopState", ctx: "LoopContext", meta: dict) -> bool:
    """把 observer 的判决交给 TaskManager 的带外入口。

    返回 **这次判决是否把 task close 掉了**（= 判 success / fail 且仲裁接受）。调用方据它决定
    这一段归谁记账：close 掉了就由带外 finalize 的 finish 对承载，**不能再产段摘要**
    （`finalize._supersede_final_raw_segment` 的不变量：同一段不记两遍）。

    判决由 observe 生成：有效 LLM 判决保留原值；观察失败时传入机械 success。
    本函数负责仲裁与落地，不再依据 ROLE 或收尾工具决定是否提交。

    带外入口自己做仲裁（task 是否仍在 `AWAITING_HUMAN`），这里不必先查一遍：人可能在
    这两行之间开口，查了也不作数。
    """
    completion = state.extra.get("observe_completion")
    if completion:
        if not completion["accepted"] or completion["action"] not in ("close", "continue"):
            return False
        tm = ctx.task_manager
        decision = completion["disposition"]
        verdict = completion["result"]["verdict"]
        outcome = RunOutcome(kind=RunOutcomeKind.COMPLETED,
            verdict=verdict["task_outcome"], outputs=observation_task(state).outputs,
            error=verdict.get("failure_reason") or "")
        accepted = await tm.resume_observe_decision(
            state.task.id, decision, observe_id=state.extra["observe_id"],
            outcome=outcome, settled=_execution(ctx).options.get("settled", False),
            finalize=partial(_out_of_band_continue if completion["action"] == "continue"
                             else _out_of_band_finalize, state, ctx, meta))
        return accepted is not False and completion["action"] == "close"

    raw = (meta.get(K.OBSERVER_OUTCOME) or "").strip()
    # 空 = 压根没判（工具没被调 / LLM 失败）→ 维持 park。非空才归一：`retry` 是别名，
    # 认不出的归 continue。**不能**把空串也喂进 normalize_verdict——那会把「没判决」
    # 变成「判了 continue」并意外驱动自动续跑。
    verdict = normalize_verdict(raw) if raw else ""
    if not verdict:
        logger.info(
            "background observe produced no verdict (task=%s); staying parked", state.task.id)
        return False
    tm = getattr(ctx, "task_manager", None)
    if tm is None or not hasattr(tm, "apply_out_of_band_verdict"):
        logger.warning(
            "background observe: no TaskManager to submit the verdict to (task=%s)",
            state.task.id)
        return False
    outcome = RunOutcome(
        kind=RunOutcomeKind.COMPLETED,
        verdict=verdict,
        summary=meta.get(K.OBSERVER_TASK_SUMMARY, "") or "",
        outputs=getattr(observation_task(state), "outputs", None),
        error=meta.get(K.OBSERVER_FAILURE_REASON, "") or "",
        observe_id=state.extra.get("observe_id", ""),
    )
    async def record(accepted, disposition):
        result = state.extra.get("observation_result")
        if result is None:
            return
        await complete(state, ctx, result, accepted=accepted, disposition=disposition,
                       action=("continue" if disposition and disposition.status == "PENDING"
                               else "close" if disposition else "recap"))
    waiting_id = _execution(ctx).options.get("waiting_hitl_id")
    def is_current():
        # A later user turn may already have parked the same task again.
        hitl = getattr(ctx, "hitl", None)
        req = hitl.registry.get(waiting_id) if hitl is not None and waiting_id else None
        return not waiting_id or req is not None and req.effective_decision is None
    accepted = await tm.apply_out_of_band_verdict(
        state.task.id, outcome,
        process_report=meta.get(K.OBSERVER_ACT_RECAP, "") or "",
        task_summary=meta.get(K.OBSERVER_TASK_SUMMARY, "") or "",
        next_step_hint=meta.get(K.OBSERVER_NEXT_STEP_HINT, "") or "",
        # 带外终态收尾：success / fail，跑在「状态已写定」与「转移已宣布」
        # 之间。见 `apply_out_of_band_verdict` 的 docstring 对那个窗口的三条约束。
        finalize=partial(_out_of_band_finalize, state, ctx, meta),
        resume=partial(_out_of_band_continue, state, ctx, meta),
        is_current=is_current,
        before_apply=record if state.extra.get("observe_id") else None,
    )
    logger.info(
        "background observe verdict '%s' for task %s: %s",
        verdict, state.task.id, "accepted" if accepted else "rejected (the user spoke first)")
    return bool(accepted) and verdict in (VERDICT_SUCCESS, VERDICT_FAIL)


async def _segment_input_ids(state, ctx):
    """Use frozen observe input; legacy synchronous callers retain a bounded read."""
    from ctx_weft.protocols import MemoryKind

    options = _execution(ctx).options
    fixed = options.get("context_record_ids")
    if fixed is not None:
        return list(fixed)
    end = options.get("segment_end_at")
    if isinstance(end, str):
        end = datetime.fromisoformat(end)
    return await ctx.memory.load_view_ids(
        state.scope, MemoryScope.TASK, ctx.provider_ctx,
        kinds=[MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY, MemoryKind.TOOL_AUDIT],
        end_at=end or _execution(ctx).started_at,
    )


async def _out_of_band_continue(state, ctx, meta):
    """Apply a durable continuation before another actor can consume its history."""
    completion = state.extra.get("observe_completion") or {}
    recap = (meta.get(K.OBSERVER_ACT_RECAP) or "").strip()
    if recap and not completion.get("short_segment") and not state.extra.get("observe_keep_raw"):
        async with StepDriver.for_context(ctx).executions.task_guard(state.task.id, "finish_report"):
            if ctx.task_manager.is_task_tree_canceled(state.task.id):
                return
            await ctx.memory.organize(
                ctx.provider_ctx, operation="fold_segment", summary=recap,
                record_ids=await _segment_input_ids(state, ctx))
    if ctx.task_manager.is_task_tree_canceled(state.task.id):
        return
    await _close_park_bubble(state, ctx, message="")
    if ctx.task_manager.is_task_tree_canceled(state.task.id):
        return
    state.task.outputs = None
    state.task.error = meta.get(K.OBSERVER_FAILURE_REASON) or None
    from ctx_weft.core.loop.steps.finalize import task_finalized_event
    await emit_effect(state, ctx, task_finalized_event(state, state.task, outcome=VERDICT_CONTINUE))
    state.extra["observe_continued"] = True


async def _out_of_band_finalize(
    state: "LoopState", ctx: "LoopContext", meta: dict,
) -> None:
    """判 success 终结了这个 task 之后的收尾——**`FinalizeStep` 在这条路上没跑过**。

    park 在 `ActStep` 里抛 `HitlPark`，driver 到不了 observe/finalize。于是 S5/S6 之后每个
    「park → 后台判 success」的 task 都跳过了整个 `FinalizeStep`，而它是三件事的唯一发生地：

    - `finalize_task_memory` → `_close_one`：bubble 给 parent、同 agent 派发 ack 终态化、
      写 finish 对、折末段 raw。**子任务漏得最重**——parent 醒来只看到「派发框 + 停在
      running 的 ack」，拿不到任何产出。
    - `BLACKBOARD_PUBLISHED`（topic=task.id）：parent `recall_topic(task.id)` 的正路。
    - `TASK_FINALIZED`：host 的 `tasks.outputs_json` / `error` 两列。S5 起一直是空的。

    所以这里把 `apply_task_close` + `task_finalized_event` 照原样跑一遍，两个调用方共用同
    一份实现（不是在这里另写一份镜像——`synthesize_cancel_closure` 就是那样来的，它的
    docstring 自己承认在镜像 `_close_one`，两份必然漂）。

    The independent observation run carries the task scope and final-output anchors.

    `has_llm_summary=True`：判决方就是 LLM，`act_recap` 是真摘要，close 即折末段 raw，不走
    「占位 + `replace_finish_report` 替换」那条为「finalize 先跑、bg 后到」设计的延迟路径
    ——带外路径顺序正好反过来。

    事件用这份快照发，于是带着 `LOOP_BACKGROUND_OBSERVE` 的 origin。这是对的：host 按
    origin 把后台事件挡在**会话状态折叠**和**对话流**之外（两者都不该被一段后台工作改写），
    而投影更新器的 `TASK_FINALIZED` 支不按 origin 过滤，`outputs_json` 照样落库。

    持久化 observe 的收尾失败向外传播，保留 Completed 待办，不宣布终态或释放依赖；
    恢复后按同一份决定继续。旧的直接调用入口仍保留原有 best-effort 行为。
    """
    from ctx_weft.core.loop.steps.finalize import apply_task_close, task_finalized_event

    result = state.extra.get("observation_result")
    if result is None and state.extra.get("observe_completion"):
        result = restore_result(state.extra["observe_completion"])
    has_summary = bool(meta.get(K.OBSERVER_ACT_RECAP)) and (
        result.verdict.reported if result is not None and result.verdict is not None else True)
    events = []
    if "TaskFinalized:" not in state.extra.get("observe_effects", []):
        events = await apply_task_close(
            state, state.task, ctx,
            outcome=meta.get(K.OBSERVER_OUTCOME) or VERDICT_SUCCESS,
            act_recap=meta.get(K.OBSERVER_ACT_RECAP, "") or "",
            task_summary=meta.get(K.OBSERVER_TASK_SUMMARY, "") or "",
            has_llm_summary=has_summary,
        )
        events.append(task_finalized_event(state, state.task, outcome=meta.get(K.OBSERVER_OUTCOME) or VERDICT_SUCCESS))
    for event in events:
        await emit_effect(state, ctx, event)
    await _close_park_bubble(state, ctx, message=(
        "" if meta.get(K.OBSERVER_OUTCOME) == VERDICT_FAIL else _PARK_CLOSED_BY_VERDICT))


async def _close_park_bubble(
    state: "LoopState", ctx: "LoopContext", *, message: str = _PARK_CLOSED_BY_VERDICT,
) -> None:
    """Retire this observation's temporary user-turn entrance.

    success/fail close it when ending the task; continue closes it before automatic
    execution resumes. waiting never calls this function. Match the recorded HITL
    identity so a delayed observer cannot retire a newer turn's wait or a tool's
    approval request. Failed cancellation leaves the durable observation unapplied.
    """
    from ctx_weft.protocols.hitl import UserTurnDelivery

    task = state.task
    hitl = getattr(ctx, "hitl", None)
    if hitl is None or getattr(hitl, "registry", None) is None:
        return
    for req in hitl.registry.list_pending(state.session.id):
        if req.task_id != task.id or not isinstance(req.delivery, UserTurnDelivery):
            continue
        waiting_id = _execution(ctx).options.get("waiting_hitl_id")
        if waiting_id and req.id != waiting_id:
            continue
        await hitl.cancel(req.id, message=message)
        logger.info(
            "closed the park bubble %s of task %s after observation", req.id, task.id)


EFFECT_TYPES = {EventType.TASK_FINALIZED, EventType.BLACKBOARD_PUBLISHED,
                EventType.MEMORY_INGESTED, EventType.MEMORY_COMPACTED}


def effect_key(event):
    return f"{event.type}:{event.payload.get('source', '')}"


async def emit_effect(state, ctx, event):
    if event.type == EventType.TASK_FINALIZED:
        from ctx_weft.core.fork_checkpoint import prepare_checkpoint
        draft = await prepare_checkpoint(state, ctx)
        if draft is not None and getattr(state, "run_outcome", None) is not None:
            state.run_outcome = dataclasses.replace(state.run_outcome, fork_checkpoint=draft)
    observed_effect = event.type in EFFECT_TYPES and state.extra.get("observe_id")
    if observed_effect:
        event.causation_id = state.extra["observe_id"]
        key = effect_key(event)
        if key in state.extra.get("observe_effects", []):
            return
    protect = getattr(getattr(ctx, "task_manager", None), "finalized_write", None)
    guard = nullcontext()
    if event.type == EventType.TASK_FINALIZED:
        if _execution(ctx).mode == ExecutionMode.FOREGROUND:
            # Honor signals accepted before this write. No await separates this
            # checkpoint from closing stop admission for the publication itself.
            await foreground_stop_checkpoint(state, ctx)
        if callable(protect):
            guard = protect(state.task.id)
    with guard:
        if event.type == EventType.TASK_FINALIZED:
            def publish_finalized():
                state.task.finalized = True
                value = event.payload.get("finished_at")
                state.task.finished_at = datetime.fromisoformat(value) if value else event.timestamp

            await ctx.event_bus.emit_batch((event,), on_committed=publish_finalized)
        else:
            await ctx.event_bus.emit(event)
    if observed_effect:
        state.extra.setdefault("observe_effects", []).append(key)
        if event.type == EventType.TASK_FINALIZED:
            StepDriver.for_context(ctx).executions.notify_finish(state.task.id)


def observation_task(state):
    """The original producer's task snapshot, shared by prompting and judgement."""
    return getattr(state, "extra", {}).get("observe_context_task", state.task)


async def ensure_waiting(state, ctx):
    """Open a real, idempotent user-turn entrance for a foreground waiting verdict."""
    from ctx_weft.core.hitl.registry import HITL_STAGE_TOOL
    from ctx_weft.protocols.hitl import HITL_FORM_WAIT, PREFACE_NORMAL, HitlAsk, UserTurnDelivery

    for req in ctx.hitl.registry.list_pending(state.session.id):
        if req.task_id == state.task.id and isinstance(req.delivery, UserTurnDelivery):
            state.extra["waiting_hitl_id"] = req.id
            return
    req = await ctx.hitl.open(
        HitlAsk(form=HITL_FORM_WAIT,
                delivery=UserTurnDelivery(task_id=state.task.id, preface=PREFACE_NORMAL)),
        session_id=state.session.id, task_id=state.task.id, agent_id=state.agent.id,
        tool_call_id=state.extra["observe_id"], stage=HITL_STAGE_TOOL,
        unattended=state.task.unattended, tenant_id=state.session.tenant_id)
    state.extra["waiting_hitl_id"] = req.id


def apply_foreground_assessment(state, result, boundary):
    """Apply verdict fields without changing task status or executing observation."""
    verdict = result.verdict
    if verdict is None or state.task.status == "CANCELED":
        return
    extra = getattr(state, "extra", {})
    decision = (extra.get("observe_completion") or extra.get("observe_decision") or {}).get("disposition")
    if (decision and state.task.status in {"FINISHED", "FAILED"}
            and state.task.status != decision["status"]):
        return
    state.task.outputs = observation_task(state).outputs
    if verdict.reported:
        ObserveStep._apply_assessment(state.task, verdict)
    else:
        state.task.observer_outcome = verdict.task_outcome
        state.task.actor_done = True
    if verdict.next_step_hint:
        state.task.next_step_hint = verdict.next_step_hint


async def prepare_foreground_observe(state, ctx, boundary, judging=True):
    """Persist the foreground observation request before running its LLM review."""
    await request(state, ctx, boundary, judging)


async def request(state, ctx, boundary, judging):
    execution = getattr(ctx, "execution", StepExecution())
    ctx.execution = execution
    options = execution.options
    mode = options.get("execution_mode", execution.mode.value)
    if not options.get("observe_id"):
        options.update(observe_id=generate_id("obs"), segment_end_at=now_utc().isoformat(),
                       execution_mode=mode, boundary=boundary, judge_task=judging)
        if boundary == "max_turns":
            options["act_turns"] = act_turns(state)
        await capture_context(state, ctx, options)
        await ctx.event_bus.emit(make_event(state, EventType.OBSERVE_REQUESTED, {
            "task_id": state.task.id, "agent_id": state.agent.id, **options}))
    else:
        # Recovered requests may have pinned an incomplete pre-repair view.
        await capture_context(state, ctx, options)


async def begin(state, ctx, boundary, judging):
    await request(state, ctx, boundary, judging)
    execution = ctx.execution
    options = execution.options
    mode = options.get("execution_mode", execution.mode.value)
    if execution.started_at is None:
        ctx.execution = dataclasses.replace(execution, started_at=(
            datetime.fromisoformat(options["segment_end_at"]) if options.get("segment_end_at") else now_utc()))
    if options.get("verdict"):
        state.extra["observe_decision"] = options
    else:
        state.extra.pop("observe_decision", None)
    if state.extra.get("observe_id") != options["observe_id"]:
        state.extra.pop("fork_checkpoint", None)
        state.extra.pop("fork_checkpoint_prepared_event", None)
    state.extra.update(observe_id=options["observe_id"],
                       observe_effects=list(options.get("effects", [])))
    state.extra["observe_fork_options"] = options
    if options.get("fork_checkpoint"):
        state.extra["fork_checkpoint"] = dict(options["fork_checkpoint"])
    state.extra.pop("observe_completion", None)
    if options.get("completion"):
        state.extra["observe_completion"] = options["completion"]
    # Serialise ordinal allocation with commit publication, including concurrent
    # foreground/background reviews of the same task. Never hold a task-manager
    # lock while delivering events (subscribers can call back into that manager).
    executions = StepDriver.for_context(ctx).executions
    async with executions.task_guard(state.task.id, "observe_start"):
        pending = executions.observe_start_commits.get(state.task.id)
        if pending is not None:
            # A different observer may arrive after an uncertain start. Confirm
            # that original batch before allocating the next task ordinal.
            event, publish = pending
            await ctx.event_bus.emit_batch((event,), on_committed=publish)
            executions.observe_start_commits.pop(state.task.id, None)
        if options.get("started"):
            return
        get_task = getattr(ctx.task_manager, "get_task", None)
        task = (get_task(state.task.id) if callable(get_task) else None) or state.task
        if judging:
            options.setdefault("observe_index", task.observe_count + 1)
        index = options.get("observe_index")
        event = make_event(state, EventType.OBSERVE_STARTED, {
            "task_id": state.task.id, "agent_id": state.agent.id,
            "observe_id": options["observe_id"], "boundary": boundary,
            "segment_end_at": options.get("segment_end_at"), "execution_mode": mode,
            "judge_task": judging,
            **({"observe_index": index} if judging else {})})

        def publish_started():
            if judging:
                task.observe_count = max(task.observe_count, index)
            options["started"] = True

        executions.observe_start_commits[state.task.id] = (event, publish_started)
        await ctx.event_bus.emit_batch((event,), on_committed=publish_started)
        executions.observe_start_commits.pop(state.task.id, None)


def restore_result(completion):
    data = dict(completion["result"])
    if data.get("verdict"):
        data["verdict"] = Verdict(**data["verdict"])
    return ObservationResult(**data)


async def complete(state, ctx, result, *, accepted=True, disposition=None, action="recap"):
    if state.extra.get("observe_completion"):
        return state.extra["observe_completion"]
    payload = {"task_id": state.task.id, "observe_id": state.extra["observe_id"],
               "accepted": accepted, "action": action, "result": dataclasses.asdict(result),
               "completed_at": now_utc().isoformat(),
               "short_segment": state.extra.get("observe_short_segment", False)}
    if disposition:
        data = dataclasses.asdict(disposition)
        data["payload"].pop("outputs", None)  # actor response facts remain the output source
        payload["disposition"] = data
    # Freeze memory formatting before any close can archive its inputs.
    from ctx_weft.core.loop.steps.finalize import _descendant_task_ids, _is_short_leaf
    if action == "close" or (action == "foreground" and disposition and
                             disposition.status in {"FINISHED", "FAILED"}):
        payload["short_leaf"] = await _is_short_leaf(
            ctx.memory, state.scope, state.task, state.agent.loop_config, ctx,
            bool(_descendant_task_ids(state.task.id, ctx.task_manager)))
    await ctx.event_bus.emit(make_event(state, EventType.OBSERVE_COMPLETED, payload))
    state.extra["observe_completion"] = payload
    return payload


def planned_disposition(state, result):
    verdict = result.verdict
    return disposition_for(RunOutcome(kind=RunOutcomeKind.COMPLETED,
        verdict=verdict.task_outcome, summary=verdict.act_recap, outputs=state.task.outputs,
        error=(verdict.failure_reason if verdict.reported else state.task.error) or "",
        hitl_id=state.extra.get("waiting_hitl_id", "")),
        retry_count=state.task.retry_count,
        max_retries=state.task.max_retries)


async def applied(state, ctx, *, reason=""):
    if state.task.status == "CANCELED" and state.task.finished_at:
        from ctx_weft.core.loop.steps.finalize import synthesize_cancel_closure
        async with StepDriver.for_context(ctx).executions.task_guard(state.task.id, "finish_report"):
            # Applied releases recovery retention; cancellation closure must be
            # complete first, even if the canceling caller died after TaskCanceled.
            await synthesize_cancel_closure(ctx.memory, state.session.id, state.task,
                                            ctx.provider_ctx, state.task.error or "canceled")
            hitl = getattr(ctx, "hitl", None)
            if hitl is not None:
                from ctx_weft.core.models.discriminators import CancelReason
                for req in hitl.registry.list_pending(state.session.id):
                    if req.task_id == state.task.id:
                        await hitl.cancel(req.id, message=CancelReason.USER_CANCEL)
    from ctx_weft.core.fork_checkpoint import prepare_checkpoint, applied_event
    draft = await prepare_checkpoint(state, ctx) if not reason else None
    event = applied_event(state, ctx, draft, reason=reason)
    if draft is None:
        await ctx.event_bus.emit(event)
    else:
        await ctx.event_bus.emit_batch((event,))


async def capture_context(state, ctx, options):
    """Pin immutable input records before the request becomes visible to another turn."""
    from ctx_weft.core.loop.driver import assistant_memory_checkpoint
    checked = options.get("observe_id")
    if options.get("reply_anchor") is not None:
        from ctx_weft.core.fork_checkpoint import validate_reply_anchor
        options["reply_anchor"] = validate_reply_anchor(options["reply_anchor"])
    if checked and state.extra.get("assistant_context_checked") == checked:
        return
    fixed = options.get("context_record_ids")
    if fixed is not None and getattr(ctx, "assistant_memory_checkpoint", None) is None:
        return
    from ctx_weft.protocols import MemoryAddress, MemoryKind
    end = (datetime.fromisoformat(options["segment_end_at"])
           if options.get("segment_end_at") else _execution(ctx).started_at or now_utc())
    async with assistant_memory_checkpoint(state, ctx):
        record_ids = []
        for scope in (MemoryScope.TASK, MemoryScope.AGENT):
            record_ids.extend(await ctx.memory.load_view_ids(
                MemoryAddress(session_id=state.session.id, agent_id=state.scope.agent_id),
                scope, ctx.provider_ctx,
                kinds=([MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY, MemoryKind.TOOL_AUDIT]
                       if scope == MemoryScope.TASK else None), end_at=end))
        if fixed is not None:
            extra_ids = set(record_ids) - set(fixed)
            extra = await ctx.memory.read_records(list(extra_ids), ctx.provider_ctx) if extra_ids else []
            if any(r.scope == MemoryScope.TASK and r.kind == MemoryKind.CONVERSATION_TURN
                   and r.role == "assistant" for r in extra):
                # Never expand an old summary's fold pool to newly repaired raw.
                state.extra["observe_keep_raw"] = True
        else:
            if options.get("completion"):
                state.extra["observe_keep_raw"] = True  # Legacy summary has no proven input set.
            options["context_record_ids"] = record_ids
        if options.get("reply_anchor") is not None and "fork_base_record_ids" not in options:
            rows = await ctx.memory.read_records(options["context_record_ids"], ctx.provider_ctx)
            options["fork_base_record_ids"] = [r.id for r in rows if r.scope == MemoryScope.AGENT]
    state.extra["assistant_context_checked"] = checked
    from copy import copy
    # Recovery already reconstructed outputs from immutable actor facts; the
    # current task projection may still have outputs=None at the crash boundary.
    if "observe_context_task" not in state.extra or (fixed is None and not state.extra.get("observe_recovery")):
        state.extra["observe_context_task"] = copy(state.task)


async def context_records(state, ctx):
    options = _execution(ctx).options
    ids = options.get("context_record_ids")
    if ids is None:
        return None  # Legacy requests retain their bounded live-view fallback.
    key = (state.extra.get("observe_id"), tuple(ids))
    cached = state.extra.get("observe_record_cache")
    if cached is not None and cached[0] == key:
        return cached[1]
    records = await ctx.memory.read_records(ids, ctx.provider_ctx)
    if set(ids) != {r.id for r in records}:
        raise RuntimeError("observe context contains missing immutable memory records")
    state.extra["observe_record_cache"] = (key, records)
    return records


def act_turns(state):
    """Count the completed actor rounds in this segment, before history folding."""
    return state.extra.get(ACT_TURNS_USED_KEY, getattr(state.agent.loop_config, "max_turns_per_act", 0))


def make_observe_request(state, boundary, judging, *, mode="background", reply_anchor=None,
                         replaces_observe_id=""):
    """Build an intent, also embedded atomically in HitlOpened for park."""
    return {"observe_id": generate_id("obs"), "task_id": state.task.id,
            "agent_id": state.agent.id, "boundary": boundary,
            "segment_end_at": now_utc().isoformat(), "execution_mode": mode,
            "judge_task": judging,
            **({"reply_anchor": dict(reply_anchor)} if reply_anchor else {}),
            **({"replaces_observe_id": replaces_observe_id} if reply_anchor and replaces_observe_id else {}),
            **({"act_turns": act_turns(state)} if boundary == "max_turns" else {})}


async def request_background_observe(state, ctx, options):
    """Persist before the driver creates the background task; reuse parked intents."""
    if options.get("observe_id"):
        return
    boundary = options.get("boundary", "")
    judging = options.get("judge_task", boundary == "task_switch" or
                          judges(boundary))
    intent = make_observe_request(state, boundary, judging, reply_anchor=options.get("reply_anchor"))
    options.update({key: value for key, value in intent.items()
                    if key not in {"task_id", "agent_id"}})
    await capture_context(state, ctx, options)
    intent.update(context_record_ids=options["context_record_ids"])
    if "fork_base_record_ids" in options:
        intent["fork_base_record_ids"] = list(options["fork_base_record_ids"])
    await ctx.event_bus.emit(make_event(state, EventType.OBSERVE_REQUESTED, intent))


async def finish_observe(outcome, task_id, emit):
    """Called after foreground run/task settlement, using its original envelope emitter."""
    if outcome.observe_id and not outcome.observe_report_pending:
        await emit(EventType.OBSERVE_APPLIED, task_id=task_id,
                   payload={"task_id": task_id, "observe_id": outcome.observe_id,
                            **({"fork_checkpoint": outcome.fork_checkpoint} if outcome.fork_checkpoint else {})})


def blocks_actor_recovery(info):
    """An accepted close must finish before actor execution may be considered."""
    completion = info.get("completion") or {}
    return (info.get("execution_mode") == "foreground" or
            completion.get("accepted", False) and completion.get("action") in ("close", "continue"))


async def wait_disposition(state, ctx, ready):
    """Join report application to the committed task disposition, not a producer run."""
    options = _execution(ctx).options
    if not options.get("verdict") or options.get("settled"):
        return
    from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT
    while True:
        ready.clear()
        # Subscribe before reading: a disposition committed before subscription is
        # visible here; one committed during/after this read wakes the subscriber.
        rows = await ctx.event_store.read_range(state.session.id, include_types=(
            EventType.OBSERVE_REQUESTED, *TASK_STATUS_BY_EVENT))
        requested = False
        for row in rows:
            event = row.event
            if event.type == EventType.OBSERVE_REQUESTED and event.payload.get("observe_id") == state.extra["observe_id"]:
                requested = True
            elif (requested and event.task_id == state.task.id and event.type in TASK_STATUS_BY_EVENT
                  and event.payload.get("reason") != "crash_recovery"):
                options["settled"] = True
                return
        await ready.wait()


async def apply_close_report(state, ctx, observation, ready):
    """Only the owning observation may update its successful foreground closure."""
    await wait_disposition(state, ctx, ready)
    registry = StepDriver.for_context(ctx).executions
    if not _execution(ctx).options.get("verdict") and ready is not None:
        # Historical recap-only close intents may precede their finish slots.
        target = await find_finish_target(ctx.memory, ctx.provider_ctx, state.scope, state.task.id)
        running = getattr(ctx.task_manager, "running_task_ids", lambda: set())()
        if target is None and not ready.is_set() and (state.task.id in running or state.task.status == "ACTIVE"):
            await ready.wait()
    async with registry.task_guard(state.task.id, "finish_report"):
        if (state.task.status in {"CANCELED", "INTERRUPTED"} or
                ctx.task_manager is not None and hasattr(ctx.task_manager, "is_task_tree_canceled")
                and ctx.task_manager.is_task_tree_canceled(state.task.id)):
            return "task_not_completed"
        decision = _execution(ctx).options.get("disposition")
        if decision and state.task.status in {"FINISHED", "FAILED"} and state.task.status != decision["status"]:
            return "task_already_terminal"
        target = await find_finish_target(ctx.memory, ctx.provider_ctx, state.scope, state.task.id,
            legacy_outcome="fail" if state.task.status == "FAILED" else "success")
        if target is None:
            if _execution(ctx).options.get("verdict"):
                return "no_finish_report"  # Disposition has no successful closure; retain raw history.
            raise RuntimeError(f"Foreground close did not persist a finish pair: {state.task.id}")
        scope, tool_call_id, policy = target
        owner = policy.get("observe_id")
        if policy["outcome"] == "cancelled" or (owner and owner != state.extra["observe_id"]):
            return "finish_report_superseded"
        from ctx_weft.core.loop.steps.finalize import _output_text, _supersede_final_raw_segment
        await replace_finish_report(
            ctx.memory, ctx.provider_ctx, scope, state.task.id, tool_call_id,
            observation.act_recap, observation.task_summary, policy["outcome"], state.task.title or "",
            final_reply="" if policy["preserve_raw"] else _output_text(observation_task(state).outputs),
            operation_id=state.extra["observe_id"])
        if not policy["preserve_raw"] and not state.extra.get("observe_keep_raw"):
            await _supersede_final_raw_segment(ctx.memory, state.scope, ctx.provider_ctx)
