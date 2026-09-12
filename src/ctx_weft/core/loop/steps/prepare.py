"""PrepareStep（资源装配）：能力解析/绑定 + token 估算 + compact 触发 + prompt 装配。

token 估算对齐 miniAgents Reasoner._fetch_base：
  - 有真实基线（loop_guard.context_tokens > 0）→ 增量估算：基线 + 新增消息估算
  - 无基线 → 用装配后的 prompt.token_count（全量估算）

compact 触发（纯预算，spec 2026-07-01 §3.6）：
  - token_estimate / context_limit >= compact_token_ratio
  命中后调 escalating_compact 升级式压缩，再完整重装配一次 prompt（Q4=c 校正）。
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from ctx_weft.core.assembler import ContextRequest
from ctx_weft.protocols.events import EventType
from ctx_weft.core.loop.driver import (
    RECOGNIZE_INTENT_PENDING_KEY,
    LoopContext, LoopState, Step, StepOutcome, make_event,
)
from ctx_weft.core.loop.steps._capabilities import resolve_and_bind
from ctx_weft.core.loop.steps.act_guidance import build_act_guidance, build_resume_cue
from ctx_weft.core.capabilities.skill_executor import (
    EXEC_SCRIPT_NAME,
    LIST_FILES_NAME,
    READ_FILE_NAME,
)
from ctx_weft.core.loop.steps.recognize_intent import (
    should_recognize_intent,
)
from ctx_weft.core.models.task import NormalTaskSettings
from ctx_weft.core.utils.estimate import effective_limit, estimate_content_tokens, estimate_tokens, estimate_tool_calls_tokens
from ctx_weft.protocols import MemoryEventType
from ctx_weft.protocols.capability import SkillCapability

logger = logging.getLogger(__name__)


_SKILL_SCRIPT_RUNTIME_NOTE = (
    "---\n"
    "**Runtime note (overrides any 'run python ...' wording above):** "
    "This skill's files are NOT in your working directory. To run ANY script the "
    f"instructions reference, you MUST call the `{EXEC_SCRIPT_NAME}` tool — e.g. "
    f"`{EXEC_SCRIPT_NAME}(script_path='scripts/foo.py', args='...')` — never run "
    "`python scripts/foo.py` yourself (the path will not resolve). To read a skill "
    f"file use `{READ_FILE_NAME}`; to list skill files use `{LIST_FILES_NAME}`. "
    "Do not build absolute paths by hand. Applies to local and remote skills alike."
)


def wrap_skill_instructions(instructions: str) -> str:
    """Append the exec_script runtime note after a non-empty SKILL.md body."""
    if not instructions:
        return ""
    return f"{instructions}\n\n{_SKILL_SCRIPT_RUNTIME_NOTE}"


def _estimate_record_tokens(r, count: Callable[[str], int] | None = None) -> int:
    """一条 memory 记录的 compact 估算：content + tool_calls 参数（在 metadata）+ reasoning。

    与 gateway._estimate_message_tokens 同口径（同 core.utils.estimate 计费项），使 prepare 的触发估算
    不再漏 tool_calls 参数/图片/framing（此前只 join content 文本、且丢弃非 str content）。

    count：文本费率经 count 回调走 tokenizer（通常是 ``ctx.llm.tokenizer.count``，已校准）；
    None 回退未校准启发式（纯单测/无 llm 场景）。
    """
    md = getattr(r, "metadata", None) or {}
    total = estimate_content_tokens(r.content, count=count) + estimate_tool_calls_tokens(
        md.get("tool_calls"), count=count)
    reasoning = md.get("reasoning")
    if reasoning:
        total += (count or estimate_tokens)(str(reasoning))
    return total


def _estimate_assembled_tokens(prompt, count: Callable[[str], int] | None = None) -> int:
    """无真实基线（首轮/一次性）时对整份装配 prompt 的估算：system + 每条消息（含 tool_calls
    参数/图片/framing）+ tools schema。

    比 composer 的 ``prompt.token_count``（纯文本、且不含 tools）更全，与 gateway 首次估算同口径——
    tools schema 是每个 act prompt 的固定占用，composer 完全没数，此处补上。

    count：文本费率经 count 回调走 tokenizer（通常是 ``ctx.llm.tokenizer.count``，已校准）；
    None 回退未校准启发式（纯单测/无 llm 场景）。
    """
    from ctx_weft.core.utils.estimate import estimate_tools_tokens

    c = count or estimate_tokens
    total = c(prompt.system or "")
    for m in prompt.messages:
        total += estimate_content_tokens(m.content, count=count) + estimate_tool_calls_tokens(
            m.tool_calls, count=count)
        if getattr(m, "reasoning_content", None):
            total += c(m.reasoning_content)
    total += estimate_tools_tokens(getattr(prompt, "tools", None), count)
    return total


class PrepareStep(Step):
    name = "prepare"

    async def execute(self, state: LoopState, ctx: LoopContext) -> StepOutcome:
        agent = state.agent
        session = state.session

        # ── 0. token budget guard ─────────────────────────────────────────────

        # ── 1. token 估算 ─────────────────────────────────────────────────────
        token_estimate, has_baseline = await self._estimate_tokens(state, ctx)

        # ── 2. capability 解析 ────────────────────────────────────────────────
        template = state.extra.get("template")
        bound_capabilities = await resolve_and_bind(state, ctx)
        # 供下游步骤复用（compact 内联、recognize_intent 并发）：单次解析，多步共享。
        state.extra["bound_capabilities"] = bound_capabilities

        # ── 3. skill instructions ─────────────────────────────────────────────
        settings = state.task.settings
        skill_name = settings.skill_name if isinstance(settings, NormalTaskSettings) else ""
        skill_instructions = await self._load_skill_instructions(state, ctx, skill_name)

        # ── 4. 装配 prompt ────────────────────────────────────────────────────
        purpose = settings.purpose if isinstance(settings, NormalTaskSettings) else "act"

        # 运行时态势文本（仅 act + 普通任务；authorship 见 act_guidance.py）：
        # guidance 经 extra → GuidanceSource → composer 恒拼末条 user 尾部；
        # resume cue 经 extra → composer 在历史以 assistant/tool 收尾时垫续跑回合。
        if purpose == "act" and isinstance(settings, NormalTaskSettings):
            act_guidance = build_act_guidance(state.task, ctx.task_manager)
            act_resume_cue = build_resume_cue(state.task, ctx.task_manager)
        else:
            act_guidance = ""
            act_resume_cue = ""

        def _assemble():
            return ctx.assembler.assemble(ContextRequest(
                purpose=purpose,
                scope=state.scope,
                task=state.task,
                agent=agent,
                session=session,
                template=template,
                bound_capabilities=bound_capabilities,
                token_counter=ctx.llm.tokenizer.count,
                extra={
                    "skill_instructions": skill_instructions,
                    "skill_name": skill_name,
                    "act_guidance": act_guidance,
                    "act_resume_cue": act_resume_cue,
                },
            ))

        prompt = await _assemble()
        if token_estimate == 0:
            token_estimate = _estimate_assembled_tokens(prompt, ctx.llm.tokenizer.count)

        # ── 5. compact 触发：命中则跑升级式 compact，再在压缩后 memory 上重装配一次（Q4=c 校正）──
        if await self._should_compact(state, ctx, token_estimate):
            from ctx_weft.core.loop.steps.compact import escalating_compact
            for ev in await escalating_compact(state, ctx, token_estimate=token_estimate, trigger="compact"):
                await ctx.event_bus.emit(ev)
            prompt = await _assemble()

        # ── 6. 会话意图识别：root task 首轮（无标题）判定在这里，**起飞在 act 的提交点** ──
        #
        # 判定留在这里是因为只有这里手握 `bound_capabilities`；起飞挪到提交点是因为
        # 旁路只该为**真的发生过**的那一轮花一次 LLM 调用。
        #
        # 别再把 `launch_` 挪回这一行。曾经挪回来过一次，理由是「第一句就按暂停的会话
        # 到不了提交点、会没有名字」——那个理由是**夸大的**：判据是「root task 且无
        # title」，那一轮夭折之后 title 仍是空的，下一轮 PrepareStep 照样判定成立、照样
        # 在那一轮的提交点起飞。代价只是标题晚一轮，不是永远没有。
        #
        # 而挪回来的代价是实打实的：root task 从**第二句**起每一轮都开未提交窗口
        # （消息注入既有 task 的那两条路 `begin_round(target.id, owns_task=False)`），
        # 只要它的 title 还是空的（旁路失败 / 模型没调 update_task_metadata / 上一轮的
        # 旁路还在飞），判据就成立——旁路于是跑在窗口里：这一轮被撤销时它的整串事件
        # 随缓冲一起丢掉，那次 LLM 调用白烧。
        if should_recognize_intent(state.task) and bound_capabilities:
            state.extra[RECOGNIZE_INTENT_PENDING_KEY] = True

        # ── 7. 发事件 + 返回 ─────────────────────────────────────────────────
        return StepOutcome(
            next_step="act",
            state_patch={"assembled_prompt": prompt},
            events=[
                make_event(state, EventType.CONTEXT_TOKENS_ESTIMATED, payload={
                    "estimated_tokens": token_estimate,
                    "assembled_tokens": prompt.token_count,
                    "has_baseline": has_baseline,
                }),
                make_event(state, EventType.CONTEXT_ASSEMBLED, payload={"token_count": prompt.token_count}),
                make_event(state, EventType.PREPARE_COMPLETED, payload={
                    "estimated_tokens": token_estimate,
                    "assembled_token_count": prompt.token_count,
                }),
            ],
        )

    # ── private helpers ───────────────────────────────────────────────────────

    async def _load_skill_instructions(
        self, state: LoopState, ctx: LoopContext, skill_name: str
    ) -> str:
        if not skill_name or not ctx.skill_provider_index:
            return ""
        skill_cap = (
            ctx.capability_cache.get_by_qualified_name(
                state.agent.id, skill_name, task_id=state.task.id)
            if ctx.capability_cache else None
        )
        if not isinstance(skill_cap, SkillCapability):
            return ""
        provider_name = skill_cap.id.rsplit(":", 1)[0]
        skill_provider = ctx.skill_provider_index.get(provider_name)
        if skill_provider is None:
            return ""
        try:
            defn = await skill_provider.load_definition(skill_cap.name, ctx.provider_ctx)
            if defn:
                # 原始 SKILL.md 正文 + exec_script 运行时说明（追加在正文之后，
                # recency 最强）；skill 名 + 标题层级由 composer 装配进
                # "## Instructions for the current task" 段。
                return wrap_skill_instructions(defn.instructions)
        except Exception:
            logger.exception("PrepareStep: failed to load skill definition for '%s'", skill_name)
        return ""

    async def _should_compact(
        self, state: LoopState, ctx: LoopContext, token_estimate: int
    ) -> bool:
        """纯预算触发（spec 2026-07-01 §3.6）：token 估算 / context_limit ≥ compact_token_ratio。
        消息条数门控（compact_message_delta）已废。"""
        loop_config = state.agent.loop_config
        loop_guard = state.agent.loop_guard
        context_limit = loop_guard.context_limit
        reserve = getattr(loop_guard, "reserved_output_tokens", 0)
        eff = effective_limit(context_limit, reserve)
        if eff > 0 and token_estimate > 0:
            return token_estimate / eff >= loop_config.compact_token_ratio
        return False

    async def _estimate_tokens(
        self, state: LoopState, ctx: LoopContext
    ) -> tuple[int, bool]:
        """增量 token 估算（对齐 miniAgents Reasoner._fetch_base）。

        返回 (token_estimate, has_baseline)。
        """
        guard = state.agent.loop_guard

        if guard.context_tokens > 0:
            try:
                from ctx_weft.protocols import MemoryKind, MemoryScope

                # v2 P3a：TASK 视图 + user/assistant 回合谓词（= 旧 USER_PROMPT+LLM_RESPONSE
                # 口径，与 act._account_tokens 的 context_message_count 对齐）。升序视图：
                # 基线计数之后的即新增记录。
                view = await ctx.memory.load_view(
                    state.scope, MemoryScope.TASK, ctx.provider_ctx)
                convo = [
                    r for r in view
                    if r.kind is MemoryKind.CONVERSATION_TURN and r.role in ("user", "assistant")
                ]
                new_records = convo[guard.context_message_count:]
                count = ctx.llm.tokenizer.count
                delta = sum(_estimate_record_tokens(r, count) for r in new_records)
                return guard.context_tokens + delta, True
            except Exception:
                pass

        return 0, False
