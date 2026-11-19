"""RecognizeIntentStep: single-shot step that fills task title/description and session goal.

Launched concurrently with ActStep (through StepDriver.start, after commit) on a snapshot of the loop
state, reusing the capability set PrepareStep already bound. Assembles with
purpose="recognize_intent"; CapabilitySource auto-filters tools to update_task_metadata.
Skips entirely if the title is already set or no recognize_intent tool is bound.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from contextlib import aclosing
from typing import Any

from ctx_weft.core.loop.driver import (
    LoopContext,
    LoopState,
    Step,
    StepDriver,
    StepOutcome,
    make_event,
)
from ctx_weft.core.loop.llm_gateway import (
    build_llm_request,
    llm_request_id,
    resolve_llm_identity,
    stream_llm_resilient,
)
from ctx_weft.core.loop.llm_stream import ensure_collected_stream, stream_until_cancel
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.protocols.events import EventType, PersistenceUnavailableError

logger = logging.getLogger(__name__)

def should_recognize_intent(task: Any) -> bool:
    """True for any untitled top-level task, including later user-created tasks."""
    return getattr(task, "parent_task_id", None) is None and not getattr(task, "title", "")


class RecognizeIntentStep(Step):
    """Single-shot LLM step that enriches task metadata and session goal."""

    name = "recognize_intent"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        token = getattr(ctx, "cancel_token", None)
        if token is not None:
            token.raise_if_cancelled()
        target_task = state.task
        target_task_id = target_task.id

        if target_task and target_task.title:
            await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_SKIPPED, payload={
                "task_id": target_task_id,
                "target_task_id": target_task_id,
            }))
            return StepOutcome(next_step=None)

        bound = list(state.extra.get("bound_capabilities", []))
        has_mf_tool = any(
            isinstance(c, ToolCapability) and "recognize_intent" in c.purposes for c in bound
        )
        if not has_mf_tool:
            logger.warning("RecognizeIntentStep: no recognize_intent tool bound for task %s", target_task_id)
            await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_SKIPPED, payload={
                "task_id": target_task_id,
                "target_task_id": target_task_id,
                "reason": "no_tools",
            }))
            return StepOutcome(next_step=None)

        # ── Assemble prompt via shared assembler (CapabilitySource filters by purpose) ──
        from ctx_weft.core.assembler import ContextRequest, MetadataIntent

        request = ContextRequest(
            purpose="recognize_intent",
            scope=state.scope,
            task=target_task,
            agent=state.agent,
            session=state.session,
            resolved_model=state.resolved_model,
            template=state.extra.get("template"),
            bound_capabilities=bound,
            token_counter=ctx.llm.tokenizer.count,
            intent=MetadataIntent(),
        )
        inputs = await ctx.assembler.assemble(request)
        if token is not None:
            token.raise_if_cancelled()

        await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_STARTED, payload={
            "task_id": target_task_id,
            "target_task_id": target_task_id,
        }))

        # req_id：与 gateway 内部用同一个确定性公式独立算出——两边都在「本次 LLM 调用的
        # 任何事件被发射之前」求值（此处在调用 stream_llm_resilient 之前；gateway 侧在其
        # while 重试循环、也就是第一次 emit 之前），故 state.sequence_counter 两处读到同一个
        # 值，天然一致（同 act.py::_run_llm_turn / compact.py::summarize_for_compact 的手法）。
        req_id = llm_request_id(state)
        model, llm_account = resolve_llm_identity(state)

        from ctx_weft.protocols import ToolCall

        # Compilation and budgeting happen before any LLM request events.
        try:
            llm_request = await build_llm_request(ctx, state, inputs)
        except Exception:
            logger.exception(
                "RecognizeIntentStep: request construction failed for task %s", target_task_id)
            return StepOutcome(next_step=None)

        stream = ensure_collected_stream(stream_llm_resilient(ctx, state, llm_request, inputs=inputs))
        registry = StepDriver.for_context(ctx).executions
        try:
            async with aclosing(stream_until_cancel(stream, token)) as chunks:
                async for _ in chunks:
                    stream.accept_current()
            if token is not None:
                token.raise_if_cancelled()
        except asyncio.CancelledError:
            if getattr(stream, "request_started", False):
                partial = stream.partial_result()
                await registry.emit_metadata_event(ctx.event_bus, make_event(state, EventType.LLM_RESPONSE_FINISHED,
                    payload={"request_id": req_id, "content": partial.text,
                             "reasoning": partial.reasoning, "tool_calls": [],
                             "usage": dataclasses.asdict(partial.usage),
                             "llm_model": model, "llm_account": llm_account,
                             "finish_reason": "cancelled"}))
            raise
        except PersistenceUnavailableError:
            raise  # The original response event is retained by the bus for confirmation.
        except Exception as exc:
            partial = stream.partial_result()
            if getattr(exc, "retriable", False):
                logger.warning("RecognizeIntentStep: LLM call failed for task %s: %s", target_task_id, exc)
            else:
                logger.exception("RecognizeIntentStep: LLM call failed for task %s", target_task_id)
            # 复审修复（task-6）：gateway 在进入重试循环**之前**就无条件发了
            # LLM_REQUEST_STARTED/LLM_PROMPT_SENT——任何退出路径都必须配对
            # LLM_RESPONSE_FINISHED，否则 host SSE 看到一条永远等不到收尾的挂死请求
            # （同成功路径下方那条收尾的理由）。finish_reason="error" 是本 step 自己的
            # 错误收尾语义，不对应任何 provider 原生 finish_reason 取值；content/reasoning/
            # usage 用异常发生前已累积到的部分值（可能全空）。
            #
            # 就地吞掉异常、不让它向上冒泡：这不是本次改动引入的行为，是 commit 920bb05
            # （总账 C5）已经明确裁定的既有设计——「正常路径（含内部已捕获、降级处理的
            # 失败）记 completed，只有真正逃出这段代码的未捕获异常才记 interrupted」。
            # recognize_intent 是尽力而为的元数据补全（失败不影响 task 主流程），outer
            # `StepDriver._run_background()` 因此仍会把这次失败报成
            # RUN_FINISHED(outcome=COMPLETED)——按该文档化的既定口径延续，未在本次改动中
            # 变更（复审记录见 task-6-report.md）。
            await registry.emit_metadata_event(ctx.event_bus, make_event(
                state, EventType.LLM_RESPONSE_FINISHED,
                payload={
                    "request_id": req_id, "content": partial.text, "reasoning": partial.reasoning,
                    "tool_calls": [],
                    "usage": dataclasses.asdict(partial.usage),
                    "llm_model": model, "llm_account": llm_account,
                    "finish_reason": "error"}))
            return StepOutcome(next_step=None)

        result = stream.get_final_result()
        text, reasoning, usage = result.text, result.reasoning, result.usage
        tool_call_obj: ToolCall | None = result.tool_calls[-1] if result.tool_calls else None
        tool_name = tool_call_obj.name if tool_call_obj is not None else ""
        tool_args = tool_call_obj.arguments if tool_call_obj is not None else {}

        # LLM_RESPONSE_FINISHED：gateway 只发流式侧 4 个事件，收尾事件由各调用方自己发
        # （同 act.py / compact.py 的既定分工）——否则 LLM_REQUEST_STARTED 有始无终，host
        # SSE 会看到一条永远等不到收尾的挂死请求。payload 结构照抄 act.py::_run_llm_turn。
        await registry.emit_metadata_event(ctx.event_bus, make_event(
            state, EventType.LLM_RESPONSE_FINISHED,
            payload={
                "request_id": req_id, "content": text, "reasoning": reasoning,
                "tool_calls": (
                    [{"id": tool_call_obj.id, "name": tool_call_obj.name,
                      "arguments": tool_call_obj.arguments}]
                    if tool_call_obj is not None else []
                ),
                "usage": dataclasses.asdict(usage),
                "llm_model": model, "llm_account": llm_account,
                "finish_reason": "tool_use" if tool_name else "stop"}))

        if tool_name and ctx.capability_gateway is not None:
            if token is not None:
                token.raise_if_cancelled()
            await ctx.capability_gateway.invoke(
                tool_name=tool_name,
                arguments=tool_args,
                state=state,
                ctx=ctx,
            )

        await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_TOOL_CALL, payload={
            "title": tool_args.get("title", ""),
            "description": tool_args.get("description", ""),
            "session_goal": tool_args.get("session_goal", ""),
        }))
        await ctx.event_bus.emit(make_event(state, EventType.RECOGNIZE_INTENT_COMPLETED, payload={
            "title": tool_args.get("title", ""),
            "description": tool_args.get("description", ""),
            "session_goal": tool_args.get("session_goal", ""),
            # 只透出不记账：意图识别 LLM 开销此前无账可查（spec 2026-07-16 §5.7）
            "usage": dataclasses.asdict(usage),
        }))

        return StepOutcome(next_step=None)
