"""ObserveStep：评估 actor transcript，产出 verdict。

miniAgents 对齐版：
- 有 ROLE 配置 + 非 root-normal 场景 → LLM 多轮 ReAct（用 report_task_outcome 工具）
- 其他情况 → **机械判决**：只从 transcript + exit_reason 定三态结局，**不产任何摘要**；
  摘要改由 background observe 异步产（用户裁定：不允许任何机械合成的摘要）。

走机械判决的场景：
  1. template 未配置 identity["observe"]（assigned agent 无 ROLE）
  2. root task（task.parent_task_id is None）——顶层任务无 parent 可上报，不需要 LLM observer
  3. LLM 调用失败，或耗尽轮次没调 report_task_outcome
  4. 本 run 已被取消——不再烧多轮 LLM。注意这是**选路径**，不是检查点：observe 是
     「整理现状」，取消到达时降级但仍走完并交出 verdict，绝不半途中止（用户裁定）。
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ctx_weft.core.assembler import ContextRequest
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols import LLMMessage, LLMRequest, LLMUsage, MemoryEventType, MemoryScope
from ctx_weft.core.loop.driver import LoopContext, LoopState, Step, StepOutcome, make_event
from ctx_weft.core.loop.llm_gateway import (
    request_prompt_estimate, resolve_llm_identity, stream_llm_resilient,
)
from ctx_weft.core.capabilities.control_tools import REPORT_TASK_OUTCOME_NAME, ControlResult
from ctx_weft.core.utils.ids import generate_id, mint_turn_call_ids

if TYPE_CHECKING:
    from ctx_weft.core.models.task import Task

logger = logging.getLogger(__name__)


# ── Shared ReAct helper ───────────────────────────────────────────────────────


async def run_observe_react(
    state: "Any",
    ctx: "Any",
    *,
    system: str,
    messages: "list[LLMMessage]",
    tools: "Any",
    max_rounds: int,
    terminal_tool_name: str,
) -> "tuple[ControlResult | None, str]":
    """共用 observe/background ReAct：跑多轮 LLM，指定 terminal_tool 被调用时返回其完整 ControlResult 终止。

    返回 (terminal_result, last_text)：
      terminal_result — terminal_tool_name 被调用时的完整 ControlResult（未调用则 None）。
      last_text       — 最后一轮的纯文本。
    非 terminal 控制工具（如 ask_user）只执行副作用，不终止循环。
    不解读 verdict、不写 task 状态（状态写是工具副作用，由调用方绑定的工具决定）。

    Task 5：不再有 event_types 间接层——「流式侧」4 种事件（REQUEST_STARTED/PROMPT_SENT/
    TOKEN_STREAMED/REASONING_STREAMED）统一由 llm_gateway.stream_llm_resilient 发射
    LLM_*，observe 前台与 background observe 之间只靠 state.origin
    （EventOrigin.LOOP_OBSERVE / LOOP_BACKGROUND_OBSERVE）区分——由调用方在调用本函数前
    设好 state.origin（observe 前台由 driver 按 step 名设好；background_observe 在
    launch 出的快照 state 上显式改写）。本函数自己只补发收尾的 LLM_RESPONSE_FINISHED
    （payload 结构对齐 act.py._run_llm_turn），否则 gateway 发的 REQUEST_STARTED 会等不到
    收尾（host SSE 侧的挂死请求，Task 4 在 compact 上踩过同样的坑）。
    """
    agent = state.agent
    model, llm_account = resolve_llm_identity(state)
    current_messages = list(messages)
    last_text = ""
    # 动态 max_tokens 的增量基线：上一轮实际发送条数（本轮 usage 对应的真实 prompt 基线）。
    baseline_msg_count: int | None = None

    for round_num in range(max_rounds):
        # req_id：与 llm_gateway.stream_llm_resilient 内部同一确定性公式独立算出（同
        # act.py._run_llm_turn 的手法）——两边都在本次 LLM 调用任何事件发射前求值，故
        # state.sequence_counter 两处读到同一个值，天然一致，不需要新增参数或跨函数传值。
        req_id = f"req_{agent.id}_{state.sequence_counter}"

        sent_msg_count = len(current_messages)  # 本轮发送条数（append 前）→ 下轮增量基线
        llm_request = LLMRequest(
            model=model,
            system=system,
            messages=list(current_messages),
            tools=tools,
        )
        llm_request.prompt_token_estimate = request_prompt_estimate(
            ctx.llm.tokenizer, llm_request, getattr(agent, "loop_guard", None), baseline_msg_count)
        # turn：gateway 从 metadata["turn"] 取（spec §9.4 统一口径：LLM_* 事件一律用 turn，
        # 不用 round）；round_num 仍是本函数内部的轮次计数局部变量，只是不再直接进 payload。
        llm_request.metadata["turn"] = round_num

        accumulated_text = ""
        reasoning_text = ""
        tool_calls = []
        usage = LLMUsage()

        async for chunk in stream_llm_resilient(ctx, state, llm_request):
            if chunk.kind == "token":
                accumulated_text += chunk.text
            elif chunk.kind == "reasoning":
                reasoning_text += chunk.text
            elif chunk.kind == "tool_call" and chunk.tool_call is not None:
                tool_calls.append(chunk.tool_call)
            elif chunk.kind == "usage" and chunk.usage is not None:
                usage = chunk.usage

        if accumulated_text:
            last_text = accumulated_text

        # 摄入前铸造内部调用标识（spec: conversation-integrity，与 act._run_llm_turn 同口径）：
        # observer 回合不入 task 层 memory（工具是 SILENT/控制面），锚仅取唯一性、不持久；
        # live 消息面（current_messages 的 tool_call↔result 配对）与事件 payload 用同一份值。
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
            current_messages.append(LLMMessage(
                role="assistant", content=accumulated_text or "(no reply)",
            ))
            current_messages.append(LLMMessage(
                role="user",
                content=(f"Submit the summary above by calling the `{terminal_tool_name}` "
                         "tool (put the content in the tool arguments); do not reply in "
                         "plain text."),
            ))
            continue

        current_messages.append(LLMMessage(
            role="assistant",
            content=accumulated_text,
            tool_calls=[{"id": tc.id, "name": tc.name, "input": tc.arguments} for tc in tool_calls],
        ))

        terminal_result = None
        for tc in tool_calls:
            if ctx.capability_gateway is not None:
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
                    terminal_result = result
            else:
                logger.warning(
                    "run_observe_react: no CapabilityGateway for tool '%s'", tc.name
                )
                content = f"[Error: CapabilityGateway not configured, tool '{tc.name}' skipped]"
            current_messages.append(LLMMessage(
                role="tool",
                content=content,
                tool_call_id=tc.id,
            ))

        if terminal_result is not None:
            return terminal_result, last_text

    return None, last_text


def _is_own_root(task) -> bool:
    """root task 判定（与 finalize._close_one 保持一致）。

    True  → session root（parent_task_id is None）或跨 agent own-root（cross_agent）。
    False → 同 agent 子任务（parent 非空且 creator==assigned）。
    非 root 不触发后台 observe（它们用 LLM observe 向 parent 上报，max_turns 同步 compact）。
    """
    same_agent = task.creator_agent_id == task.assigned_agent_id
    cross_agent = bool(task.parent_task_id) and not same_agent
    return task.parent_task_id is None or cross_agent


@dataclass
class Verdict:
    """Observer 输出（三态）。"""
    task_outcome: str   # "retry" | "success" | "fail"
    act_recap: str      # 诚实复述本段 act 做了什么 → finish 对 assistant；retry 作 Progress So Far
    task_summary: str = ""  # 整段综合总结（执行历程+结果）→ finish 对 tool 槽（仅终态有意义）
    reported: bool = False  # 本轮是否真的走成 report_task_outcome；压缩摘要据此取信


class ObserveStep(Step):
    name = "observe"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        events: list[Any] = []

        # 起点事件：先于下面任何分流决策（含取消 token 读取）——观测的开始时刻，
        # 与 ObserveCompleted 成对，同形于后台 TaskRecapStarted/Done。payload 只放
        # task_id：是否用 LLM 在起点还没定（取决于取消状态与 _should_use_llm），不猜。
        events.append(make_event(
            state, EventType.OBSERVE_STARTED,
            payload={"task_id": state.task.id},
        ))

        # 取消时不跑多轮 LLM observe：observe 是「整理现状」，**不中止**（用户裁定）——
        # 这里读 token 只为**选路径**，绝不是检查点：不 raise_if_cancelled、不提前 return，
        # observe 照常走完并交出 verdict（next_step="finalize"）。半途中止会留下既无判决、
        # 也没整理干净记忆的 task，比多等几轮更糟。但也不该在用户已按下取消后再烧几轮
        # LLM —— 降级走机械判决，摘要交后台 observe 产。
        tok = getattr(ctx, "cancel_token", None)
        cancelled = tok is not None and tok.is_cancelled

        verdict: Verdict | None = None
        if (not cancelled) and self._should_use_llm(state):
            try:
                verdict = await self._llm_observe(state, ctx, events)
            except Exception as exc:
                logger.warning(
                    "ObserveStep LLM call failed, degrading to mechanical verdict: %s", exc)
                verdict = None
        # used_llm = 判决**真的**出自 LLM observer（不是「尝试过 LLM」）：LLM 抛异常或
        # 耗尽轮次没调 report_task_outcome 时同样落机械判决，那份判决没有摘要，
        # 一律要转 background observe 补。
        used_llm = verdict is not None
        if verdict is None:
            verdict = self._mechanical_verdict(state)

        # 机械退出（max_turns/context_limit）：任务未完成、只是耗尽 turn/context，非终态——
        # 强制 retry 重排（覆盖 success/fail）；用 replace 保留 summary 与 reported 标记。
        if state.act_exit_reason in ("max_turns", "context_limit") and verdict.task_outcome != "retry":
            verdict = dataclasses.replace(verdict, task_outcome="retry")
            self._apply_assessment(state.task, verdict)

        # retry（三来源：max_turns/context_limit/observer-retry）→ 前台同步段折：
        # 本轮 attempt raw 折成一条 TASK_COMPACT_SUMMARY（复用 act_recap），删本轮 raw。
        # 「马上要重跑」故同步做好，下个 run 一进 prepare 即见折后段摘要。
        if verdict.task_outcome == "retry":
            await self._fold_retry_segment(state, ctx, verdict, events)

        # close 边界：root task 在 actor_done（finish_task 收尾 → boundary="finish"）或
        # normal（actor 产出最终文本正常结束 → boundary="normal"）时触发后台异步 observe，
        # 产段摘要 + 折 raw。两者均由 _mechanical_verdict 映射为 success/fail，属于 root 的
        # 单次终结点——task 只 close 一次，_close_report 槽写一次、弹一次，不存在乱序复用。
        # 注：纯文本暂停（plain_text 边界）由 act.py:_finish_plain_text_turn 单独触发，不经此处。
        # max_turns/context_limit 走同步 _fold_retry_segment；非 root 不触发（它们走 LLM observe）。
        launched = False
        if (state.act_exit_reason in ("normal", "actor_done")
                and verdict.task_outcome != "retry" and _is_own_root(state.task)):
            from ctx_weft.core.loop.steps.background_observe import launch_background_observe
            boundary = "finish" if state.act_exit_reason == "actor_done" else "normal"
            launch_background_observe(state, ctx, boundary=boundary)
            launched = True

        # 机械判决没有摘要（用户裁定：不允许任何机械合成的摘要）→ 交后台产真 recap。
        # 上面 close 边界那支已经 launch 过的不再重复：重复 launch 虽有 per-task 锁兜着，
        # 但会多发一对 TaskRecapStarted/Done。
        if not used_llm and not launched:
            from ctx_weft.core.loop.steps.background_observe import launch_background_observe
            launch_background_observe(state, ctx, boundary="mechanical")

        events.append(make_event(
            state, EventType.OBSERVE_COMPLETED,
            payload={
                "task_id": state.task.id,
                "outcome": verdict.task_outcome,
                "summary_length": len(verdict.act_recap),
                "used_llm": used_llm,
            },
        ))

        return StepOutcome(
            next_step="finalize",
            state_patch={"verdict": verdict},
            events=events,
        )

    # ── LLM 路径 ──────────────────────────────────────────────────────────────

    async def _llm_observe(
        self,
        state: LoopState,
        ctx: LoopContext,
        events: list[Any],
    ) -> Verdict | None:
        """LLM ReAct 循环：装配 observe prompt → 最多 max_turns_per_observe 轮 LLM 调用。

        每轮：若 LLM 调用 report_task_outcome → 立即返回 verdict；
              若 LLM 返回纯文本 → 继续下一轮（最多 max_turns_per_observe 次）。
        对齐 miniAgents Observer._llm_observe 多轮循环。
        """
        agent = state.agent
        max_rounds = agent.loop_config.max_turns_per_observe

        cache = ctx.capability_cache
        bound_caps = (
            cache.get(agent.id)
            if cache is not None and cache.has_agent(agent.id)
            else []
        )
        subtask_reviews: list[dict] = []
        tm = ctx.task_manager
        if tm is not None:
            for cid in tm.children_of(state.task.id):
                child = tm.get_task(cid)
                if child is None:
                    continue
                entry = {
                    "task_id": child.id,
                    "title": child.title or "",
                    "outcome": (child.status or "").lower(),
                }
                # spec: task-handoff——依赖阻塞取消的子任务带解释性 note（error_code
                # 区分于用户取消），父观察面据此知道「没跑是因为前序失败」。
                if child.status == "CANCELED" and child.error_code:
                    entry["note"] = child.error or child.error_code
                subtask_reviews.append(entry)
        request = ContextRequest(
            purpose="observe",
            scope=state.scope,
            task=state.task,
            agent=agent,
            session=state.session,
            template=state.extra.get("template"),
            bound_capabilities=bound_caps,
            token_counter=ctx.llm.tokenizer.count,
            extra={"subtask_reviews": subtask_reviews},
        )
        prompt = await ctx.assembler.assemble(request)

        terminal_result, last_text = await run_observe_react(
            state, ctx,
            system=prompt.system,
            messages=list(prompt.messages),
            tools=prompt.tools,
            max_rounds=max_rounds,
            terminal_tool_name=REPORT_TASK_OUTCOME_NAME,
        )

        if terminal_result is not None:
            # report_task_outcome 已写 task.observer_outcome / task.process_report / task.task_summary
            return Verdict(
                task_outcome=state.task.observer_outcome or "success",
                act_recap=state.task.process_report or last_text[:500],
                task_summary=state.task.task_summary or "",
                reported=True,
            )

        # 没走成 report_task_outcome → 本次 LLM observe 视为未产出判决，交给调用方
        # 落机械判决 + background observe（返回 None 而非在此合成，见 execute）。
        logger.warning(
            "ObserveStep: LLM did not call report_task_outcome in %d rounds, "
            "degrading to mechanical verdict", max_rounds)
        return None

    # ── 机械判决（无摘要）─────────────────────────────────────────────────────

    def _mechanical_verdict(self, state: LoopState) -> Verdict:
        """无可用 LLM observer 时的判决：**只定结局，不产摘要**。

        摘要由 background observe 异步产出（用户裁定：不允许任何机械合成的摘要）。
        act_recap 留空——`_fold_retry_segment` 对空摘要的口径是「不折、段保 raw」
        （见其 docstring），不会写占位。

        三条映射逐字对齐删除前的 `_rule_observe` 结局，故 task 终态不变：
          空 transcript                        → fail
          max_turns / context_limit（机械退出）→ retry
          normal / actor_done                  → success

        不写 task 状态：判决三态经 FinalizeStep 的 RunOutcome 交 TaskManager 处置
        （删掉的 `_rule_observe` 里那句 `_apply_assessment` 只写 observer_outcome /
        task_summary / actor_done，均无下游依赖——observer_outcome 只被 LLM 路径回读，
        actor_done 在下一轮 `TaskManager._run_task` 入口被重置为 False）。
        """
        if not state.transcript:
            return Verdict(task_outcome="fail", act_recap="")
        if state.act_exit_reason in ("max_turns", "context_limit"):
            return Verdict(task_outcome="retry", act_recap="")
        return Verdict(task_outcome="success", act_recap="")

    @staticmethod
    def _apply_assessment(task: Task, verdict: Verdict) -> None:
        """把规则 observe 的判决写进 task，对齐 report_task_outcome。

        **只写判决，不写状态**（Task 4）：三态 verdict 经 FinalizeStep 的 RunOutcome
        交给 TaskManager，由处置表决定 task 落 FINISHED / FAILED / PENDING。这里原本
        的 `task.status = ...` 是判决越界写状态，已删。
        """
        task.observer_outcome = verdict.task_outcome
        task.task_summary = verdict.task_summary
        task.actor_done = True

    # ── 条件判断 ──────────────────────────────────────────────────────────────

    def _should_use_llm(self, state: LoopState) -> bool:
        """规则降级条件（按 task 排除）：无 observe ROLE → 规则；root → 规则，
        但 max_turns/context_limit 机械退出强制 LLM（产出可信 act_recap 作压缩摘要）。"""
        # assigned agent 没有 observe ROLE → 规则降级（无可用 observer 装配）
        template = state.extra.get("template")
        if template is None or template.identity.get("observe") is None:
            return False
        # 机械退出（max_turns/context_limit）：即使 root 也要 LLM observe，产有质量 act_recap 作段摘要
        if state.act_exit_reason in ("max_turns", "context_limit"):
            return True
        # root task（无 parent）→ 规则降级；委派出的子任务才需要 LLM observer
        if state.task.parent_task_id is None:
            return False

        return True

    async def _fold_retry_segment(
        self, state: LoopState, ctx: LoopContext, verdict: Verdict, events: list[Any]
    ) -> None:
        """retry 前台同步段折：本轮 attempt raw → 一条 TASK_COMPACT_SUMMARY（复用 act_recap），
        supersede 本轮全部 raw（keep_last=0），保 USER_PROMPT 锚 + 既往段摘要（累积）。
        protect TASK_COMPACT_SUMMARY → 多轮 retry 段摘要累积（不替换），由 L3 按 collapse_keep_last
        坍缩控界。无 keep_last 门、无额外 LLM。

        仅当 verdict.task_outcome=="retry" 才折（三来源：max_turns/context_limit/observer-retry）；
        其余 outcome 不折，no-op。

        act_recap 来源：本轮真走成 report_task_outcome（reported）用其可信 report，否则用 verdict.act_recap
        （root 机械退出经 _should_use_llm 强制 LLM 已产出）。空则不折、段保 raw（不写占位摘要）。
        短段免折（background_observe.is_short_segment，与段边界折叠同门）：当前段（末条 UP
        之后）active raw 低于 short_segment_token_threshold 时不折——recap 常比短原文更长，
        raw 原样留给下个 attempt 反而信息更全；attempt 之间无新 UP，raw 在段内跨 attempt
        累积，超阈值后下一次 retry 一并折叠。折叠带 since_last=USER_PROMPT（段作用域，
        2026-07-21）：交互任务中若有免折残留的前段 raw，不被跨段合折。
        """
        from ctx_weft.core.loop.steps.background_observe import is_short_segment

        if verdict.task_outcome != "retry":
            return
        summary = (verdict.act_recap or "").strip()
        if not summary:
            logger.warning(
                "_fold_retry_segment: empty act_recap for task=%s; skip fold, segment kept raw",
                state.task.id,
            )
            return
        if await is_short_segment(state, ctx):
            logger.info(
                "_fold_retry_segment: short segment kept raw (task=%s); skip fold",
                state.task.id,
            )
            return
        # v2 P3c：策展上移——段作用域折叠（只折当前段、护 user 回合与既有摘要、
        # 锚点/段尾语义）由框架侧 segment_fold 执行原子 fold（与 bg 段折同门）。
        from ctx_weft.core.loop.steps.segment_fold import segment_fold
        result = await segment_fold(
            ctx.memory, state.scope, MemoryScope.TASK, summary, ctx.provider_ctx,
        )
        events.append(make_event(state, EventType.MEMORY_COMPACTED, payload={
            "events_before": result.events_before,
            "events_after": result.events_after,
            "summary_event_id": result.summary_event_id,
            "summary_length": len(summary),
            "layer": "task",
            "trigger": "observe_retry",
        }))
