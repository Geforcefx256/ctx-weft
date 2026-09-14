"""CapabilityGateway：统一工具调用入口。

职责（对应 miniAgents ToolGateway）：
  1. 按名查 Capability 对象（CapabilityCache）
  2. 授权检查（Authorizer）
  3. 参数脱敏（headers 里的敏感 key）
  4. 执行（CapabilityProvider.invoke，流式）
  5. 发布审计事件（EventBus）
  6. Memory ingest（TOOL_INVOCATION + TOOL_RESULT）

ActStep 只调 gateway.invoke()，拿回 InvocationResult，不感知内部细节。
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jsonschema

from ctx_weft.core.utils.content import (
    content_with_prefix,
    content_with_suffix,
    legalize_tool_result_parts,
    normalize_content_parts,
    redact_content_for_event,
    split_for_tool_result,
)
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.protocols.events import EventBus
from ctx_weft.core.hitl.registry import HITL_STAGE_AUTHZ, HITL_STAGE_TOOL
from ctx_weft.protocols.hitl import HITL_OUTCOME_REJECTED, HitlDecision
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id, is_internal_call_id
from ctx_weft.protocols.capability import (
    AuthorizationDecision, Authorizer, CapabilityProvider, ToolCapabilityProvider, qualify,
)
from ctx_weft.protocols.context import ContentPart, TextPart
from ctx_weft.protocols.llm import RAW_ARGS_KEY
from ctx_weft.core.capabilities.control_tools import (
    ASK_USER_NAME,
    ASK_USER_UNATTENDED_RESULT,
    PROVIDER_NAME as CONTROL,
)
from ctx_weft.core.hitl.service import UnattendedHitl
from ctx_weft.protocols.filesystem import SpillSink
from ctx_weft.protocols.memory import MemoryEvent, MemoryEventType, MemoryScope, MemoryProvider, MemoryAddress
from ctx_weft.protocols.memory_compat import MemoryKind

if TYPE_CHECKING:
    from ctx_weft.core.loop.driver import LoopState, LoopContext
    from ctx_weft.protocols.hitl import HitlAsk
    from ctx_weft.protocols.memory import MemoryBlobStore

logger = logging.getLogger(__name__)

_REDACT_HEADERS = frozenset({"authorization", "cookie", "x-api-key", "x-auth-token"})

# 无人值守撞上 HITL 时的两句措辞（`ask_user` 例外，它的文本住在 control_tools）。
# 授权侧这一句被合成进一个「人拒绝了」的 `HitlDecision`，由发起方 authorizer 原样
# 转成 `AuthorizationDecision.message`，最终以 `[Blocked by human: ...]` 回灌 LLM。
_UNATTENDED_AUTHZ_NOTE = (
    "this task runs unattended in the background — there is nobody who could approve "
    "this tool call, so it is denied. Continue without it, or finish the task and say "
    "what you could not do."
)
_UNATTENDED_TOOL_NOTE = (
    "[No human available: this task runs unattended in the background, so nobody can "
    "respond. Continue with what you already know, or finish the task and state what "
    "blocked you.]"
)

# 畸形 {"_raw": ...} 报错里回吐原文的上限：畸形原文可能是大 write_file 的几 KB 内容，
# 整段回灌会炸 context，超长截断。
_RAW_ERROR_MAX_LEN = 800

# 派发型控制工具（spec 2026-06-28 §2.3）：其 tool_call 落 agent 层 delegate conversation turn
# （AGENT_CONVERSATION_TURN, assistant），即时 result 暂挂，由 child finalize 回填配对的 tool 回合
# （同 origin=delegating task）。普通工具仍走 task 层 TOOL_INVOCATION/RESULT。
DISPATCH_TOOLS = frozenset({
    qualify(f"{CONTROL}:delegate_task"),
    qualify(f"{CONTROL}:delegate_plan"),
})

# 计划型派发工具：除写 delegate conversation turn 外，还需写一条配对的 ack tool result，
# 避免该 plan 框悬挂（被 legalize 剥掉）。由 child finalize 补写的 result 仅针对 start_task 子框。
_PLAN_DISPATCH_TOOLS = frozenset({
    qualify(f"{CONTROL}:delegate_plan"),
})

# 编排/裁决型控制工具：其结果是状态信号、不入 task 对话——例如 report_task_outcome 的 HITL 回复
# 改由 finalize 以 role=user 注入。（ask_user 的人类答复是 actor 输入，仍写 task 层。）
# finish_task 同理：反转契约后它是无参收尾标记，最终答复即助手消息正文、canonical 出口是
# task.outputs（ActStep 收尾时合成），标记本身不入 task 对话。
# collect_process_report 是 background observe 的终止工具：result 由 run_observe_react 取出落
# close report 槽（→ Process Report），且 background observe 在 task close 后才跑，若入 task 对话
# 会污染已冻结的对话且不被 supersede（泄漏进后续 task prompt）。
SILENT_TOOLS = frozenset({
    qualify(f"{CONTROL}:report_task_outcome"),
    qualify(f"{CONTROL}:update_task_metadata"),
    qualify(f"{CONTROL}:finish_task"),
    qualify(f"{CONTROL}:collect_process_report"),
})


# 「工具返回非文本内容」的通用接缝（子设计 §4.2；2026-09-05 收成 content 单口径）。
#
# provider 把 `ImagePart` 直接放进 `CapabilityEvent(kind="result")` 的
# `payload["content"]`——它是 `str | list[ContentPart]`，**与三个执行入口、
# `HitlReply.message`、`AuthorizationDecision.message` 同一个联合类型**。曾经另有一条
# `metadata["content_parts"]` 侧信道，要求 provider 把文本与 part 分两处交；已删除：
# 同一件事两种写法，且与本仓其余所有内容口子都不一样。
#
# **通道是通用的，不认发布者**：`media:get_image`、MCP 的 `ImageContent`、将来的浏览器
# 截图/图表生成走同一条路，gateway 不做来源白名单。
#
# 「拆开」的活由 gateway 自己干（`_stream_events` 里一次 `split_for_tool_result`）：
# 落盘截断（`_maybe_spill`）、human note 拼接、事件 payload 脱敏这些既有加工全部只
# 作用于**文本部分**，parts 在它们之后才拼回去。provider 因此不需要知道这些加工存在。


def invocation_key(tool_name: str, arguments: dict[str, Any] | None) -> str:
    """一次**具体调用**的稳定指纹：工具名 + 原始参数。

    模型复用 tool_call id 是常态（`call_1` 这类短值），所以 `(session, tool_call_id,
    stage)` 三维并不能唯一标定「哪一次调用」——第 3 轮批准的 `call_1` 会替第 9 轮
    **另一次** `call_1` 开门，还把第 3 轮的 `modified_arguments` 一并带进去（复审 I3）。
    本键是决定缓存的第四维，把「同一次调用的合法重入」与「同 id 的另一次调用」分开。

    **必须用 `invoke()` 收到的原始 `arguments`**，不是改写之后的：冷路径由
    `ReconcileStep` 用对话里记着的那份 tool_call 参数原样重入，两侧只有原始参数才
    逐字节相同。dict 序不稳定 → `sort_keys`；非 JSON 值 → `default=str`（指纹不需要
    可逆，只需要确定）。
    """
    try:
        blob = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:                                   # pragma: no cover — 防御性
        blob = repr(arguments)
    return f"{tool_name}:{hashlib.sha256(blob.encode('utf-8', 'replace')).hexdigest()[:32]}"


# ── InvocationResult ──────────────────────────────────────────────────────────


@dataclass
class InvocationResult:
    """Gateway.invoke() 的结构化返回。ActStep 直接消费，不再处理原始事件流。"""

    invocation_id: str
    tool_name: str
    # 拼好的 result，追加进 LLM messages。默认是**文本**（与改造前逐字节相同）；
    # 仅当 provider 的 result content 里带了非文本 part（或人类备注带图）时才是
    # `list[ContentPart]`（形如 `[TextPart(文本), *parts]`）。
    # 读取方注意：对 list 做 `.strip()` / `join` / `content[:N]` 都是错的
    # （切片一个 list 不报错，但切出来的是前 N 个 part）——文本化请走
    # `core.utils.content` 的 `content_to_text` / `redact_content_for_event`。
    content: str | list[ContentPart]
    metadata: dict[str, Any] = field(default_factory=dict)  # control signals
    is_error: bool = False


@dataclass
class _ToolStream:
    """一次工具流（或一次人类答复）聚合出来的东西。

    `content` 是 ``str | list[ContentPart]``（与三个执行入口、`HitlReply.message`
    同一个联合类型），但 gateway 内部必须把**文本**单独拿在手上：它还要过 spill
    截断、还要在前面接 ``[Human note: …]``，两件事都只作用于文本。于是流一进来就用
    `split_for_tool_result` 拆成 `texts` / `parts` 两半，末尾再拼回去。

    多条 result 事件的文本按 ``
`` 累加（与改造前同），part 按到达顺序累加。
    """

    texts: list[str] = field(default_factory=list)
    parts: list[ContentPart] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    needs_human_ask: "HitlAsk | None" = None


# ── CapabilityGateway ─────────────────────────────────────────────────────────


async def converge_tool_output(
    full_text: str,
    invocation_id: str,
    spill_sink,
    provider_ctx,
    *,
    threshold: int,
    preview_chars: int,
    tail_chars: int,
) -> str:
    """收敛工具长输出（spec: tool-result-recovery）——唯一的收敛实现，四个入口共用。

    全文交给 `SpillSink`（宿主注册，可以是落盘的也可以是可回读的内存实现），拿回一句
    **取回说明**；上下文承载 = 说明 + 全长 + 头部预览 + **尾部预览**（错误与结论高发区）。

    无 sink 或 spill 抛错 → 显式标注「全文不可取回」，工具结果本身照常回灌——**不假装
    有个取不回来的引用**，那只会让模型白试一次。未超阈值原样返回。
    """
    if threshold <= 0 or len(full_text) <= threshold:
        return full_text

    original_length = len(full_text)
    head = full_text[:preview_chars]
    tail = full_text[len(full_text) - tail_chars:] if tail_chars > 0 else ""

    recovery = None
    if spill_sink is not None:
        try:
            recovery = await spill_sink.spill(
                full_text, provider_ctx, name_hint=invocation_id,
            )
        except Exception:
            logger.exception("CapabilityGateway: spill failed for %s", invocation_id)

    # 取回说明**整句**由 sink 给（SpillSink 契约），gateway 原样嵌入、不加框架词。
    # 只有存进去的那一方知道该用哪个工具、传什么参数取回来：落盘的点名
    # `fs__read_file` 并给出路径，可回读的给出可照抄的 `results__read_tool_output(...)`。
    # core 若统一套一句「full text at {ref}」，对两种 sink 都不准确——一个路径究竟是
    # 「宿主侧的文件」还是「我能读的东西」，模型只能猜，猜错就是白跑一轮工具。
    parts = [
        f"[Tool output truncated: {original_length} chars exceeded "
        f"{threshold}-char limit. "
        + (recovery or "Full text is NOT recoverable (no spill sink configured, or "
                       "the spill failed) — what follows is all that remains.")
        + "]"
    ]

    body = [f"--- preview (first {len(head)} chars) ---", head]
    if tail:
        body += [f"--- tail (last {len(tail)} chars) ---", tail]
    return "\n".join(parts) + "\n" + "\n".join(body)


class CapabilityGateway:
    """统一 capability 调用入口：授权 → 脱敏 → 执行 → 审计 → memory。"""

    def __init__(
        self,
        capability_cache: CapabilityCache,
        capability_providers: list[CapabilityProvider],
        memory: MemoryProvider,
        event_bus: EventBus,
        provider_authorizers: dict[str, Authorizer] | None = None,
        default_authorizer: Authorizer | None = None,
        spill_threshold: int = 4000,
        spill_preview_chars: int = 1000,
        spill_tail_chars: int = 1000,
        memory_blob_store: "MemoryBlobStore | None" = None,
        operation_store=None,
    ) -> None:
        self._cache = capability_cache
        self._providers = capability_providers
        self._provider_index: dict[str, ToolCapabilityProvider] = {
            p.name: p for p in capability_providers
            if isinstance(p, ToolCapabilityProvider)
        }
        self._memory = memory
        # provider 在 result content 里交上来的 inline 图片在这里外部化（见
        # `legalize_tool_result_parts`）。`None` = 不外部化，inline 原样跑——与「宿主
        # 没接 blob store」同一口径，也让不关心多模态的构造点（含全部既有测试）行为
        # 逐字节不变。
        self._memory_blob_store = memory_blob_store
        # spec: tool-operations（wp5）——操作账本（None = 调用方未接线，operation_id
        # 存在也旁路；runtime 构造期从 registry 解析注入）。
        self._operation_store = operation_store
        self._event_bus = event_bus
        self._provider_authorizers: dict[str, Authorizer] = provider_authorizers or {}
        if default_authorizer is None:
            from ctx_weft.providers.authorizer import AllowAllAuthorizer
            default_authorizer = AllowAllAuthorizer()
        self._default_authorizer: Authorizer = default_authorizer
        # 工具输出截断阈值（字符）：超出则全文入结果存储、上下文换收敛版
        # （引用 + 全长 + 头尾预览）。<=0 关闭。tail 预览是错误/结论高发区的立即止血。
        self._spill_threshold = spill_threshold
        self._spill_preview_chars = spill_preview_chars
        self._spill_tail_chars = max(0, spill_tail_chars)
        # 超长输出的去处只有一条：SpillSink（core 不碰文件系统、也不知道 workspace）。
        # 宿主注册哪种实现决定能力上限——落盘的宿主自己读，可回读的（
        # ResultsCapabilityProvider）模型能取回；都不注册则硬截断。
        self._spill_sink: SpillSink | None = next(
            (p for p in capability_providers if isinstance(p, SpillSink)),
            None,
        )

    async def invoke(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        state: "LoopState",
        ctx: LoopContext,
        tool_call_id: str = "",
    ) -> InvocationResult:
        """执行一次工具调用，返回结构化结果。

        tool_call_id：发起本次调用的 LLM tool_call id（spec/06 §5），透传给派发工具用于委派回填。
        编排：解析 → 授权 → 脱敏 → 执行(流式) → 记录；各步细节见私有 helper。
        """
        invocation_id = generate_id("inv")
        is_dispatch = tool_name in DISPATCH_TOOLS
        is_silent = tool_name in SILENT_TOOLS  # 不入 task 对话的编排/裁决工具
        # spec: tool-operations——账本键即本次调用的**内部 tool_call 标识**：摄入点铸造
        # 的 `tc_...` 已经唯一、跨重启稳定、随消息落库，直接拿来用即可。
        #
        # 非内部标识（裸 wire id：无铸造的测试替身、宿主直构 gateway）→ None → 账本
        # 全程旁路。判据放在这里而不是调用侧，是因为它同时消掉了旧设计里
        # `provider_ctx.operation_id` 的所有权转移——那是个共享可变字段，忘了清就会
        # 泄漏到后续无关 invoke（后台 observe 的 collect_process_report 曾因此命中别的
        # 操作的 completed 短路、回放错结果）。改成普通入参之后泄漏不可能发生。
        op_id = tool_call_id if is_internal_call_id(tool_call_id) else None

        # 1. Lookup capability（只处理 kind="tool"）。控制工具的全局可达性由 CapabilityCache 的
        # session 全局区保证（get_by_qualified_name 回退），gateway 无需特殊逻辑。
        cap = self._cache.get_by_qualified_name(state.agent.id, tool_name, state.task.id)
        if cap is None or cap.kind != "tool":
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id,
                f"[Error: unknown tool '{tool_name}']", is_dispatch, is_silent, tool_call_id,
            )

        # 2. Authorization：按 cap.id 前缀取 per-provider authorizer，无则用 default
        authorizer = self._get_authorizer(cap.id)
        # 决定缓存短路（冷路径重入 · 授权步）：registry 已有该 tool_call 的人工决定 →
        # 连 authorize() 都不调。工具步的同一短路在 `_resolve_human` 里（那时才知道要问人）。
        # 内存 pending（活的等待）不算「已答过」，registry.decision_for 已保证这点。
        # 第四维 `invocation_key`：**同一次调用**才算合法重入（复审 I3）。用原始参数，
        # 不是改写后的——冷路径 reconcile 拿的就是对话里记着的原始参数。
        inv_key = invocation_key(tool_name, arguments)
        cached = (
            ctx.hitl.registry.decision_for(
                ctx.provider_ctx.session_id, tool_call_id, HITL_STAGE_AUTHZ,
                invocation_key=inv_key)
            if ctx.hitl else None
        )
        if cached is not None:
            cached_decision, _resume_state = cached
            decision = await self._authz_after_human(
                authorizer, cap, ctx, arguments, tool_call_id, cached_decision)
        else:
            # 交出 ProviderContext（不是 loop 的 LoopContext）——授权契约只认 protocols 类型。
            decision = await authorizer.authorize(
                cap, ctx.provider_ctx, arguments, tool_call_id=tool_call_id,
            )
            if decision.needs_human is not None:
                # 等待权归 gateway：authorizer 只是**声明**需要人，不自己等。
                try:
                    _hitl_id, human = await self._resolve_human(
                        decision.needs_human, state, ctx, tool_call_id,
                        stage=HITL_STAGE_AUTHZ, invocation_key=inv_key)
                except UnattendedHitl:
                    # 无人值守：没有人能批准这次调用。**不抛给 agent loop**——把它合成
                    # 一个「人拒绝了」的决定，交回**发起方**去解释（与真人拒绝走同一条
                    # `on_decision` 路径，authorizer 因此不必认识 unattended 这个概念，
                    # 见 `providers/authorizer/human.py`）。它返回 allowed=False，
                    # 下面照常包成 `[Blocked by human: ...]` 回灌 LLM。
                    logger.info(
                        "Capability '%s' auto-denied: task %s runs unattended, "
                        "nobody can approve it", cap.id, state.task.id)
                    human = HitlDecision(
                        outcome=HITL_OUTCOME_REJECTED, message=_UNATTENDED_AUTHZ_NOTE)
                decision = await self._authz_after_human(
                    authorizer, cap, ctx, arguments, tool_call_id, human)
        if decision is None:
            # 契约违例：authorizer 声明了 NeedsHuman 却没实现 HumanGatedAuthorizer。
            # 收敛成一条工具结果错误，安全不变式仍然成立（绝不放行）。
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id,
                f"[Error: {type(authorizer).__name__} returned NeedsHuman but does not "
                f"implement HumanGatedAuthorizer]",
                is_dispatch, is_silent, tool_call_id,
            )
        if not decision.allowed:
            logger.warning("Capability '%s' blocked by authorizer for agent %s", cap.id, state.agent.id)
            # 走 content_with_prefix/suffix 而非 f-string：备注可能是 list[ContentPart]
            # （人类审批时贴的图），f-string 会把它拍成 repr。对 str 逐字节原样。
            # 先拆出非文本 part（split_for_tool_result）再对纯文本部分套前后缀，
            # 否则当 message 以图片收尾时，content_with_suffix 会把 "]" 拍到图片
            # 后面而非文本后面——两个前后缀必须都落在同一个 TextPart 里。
            if decision.message:
                note_text, note_parts = split_for_tool_result(decision.message)
                blocked_text = content_with_suffix(
                    content_with_prefix(note_text, "[Blocked by human: "), "]")
                content = (
                    normalize_content_parts([TextPart(text=blocked_text), *note_parts])
                    if note_parts else blocked_text
                )
            else:
                content = f"[Error: capability '{tool_name}' not authorized]"
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id, content, is_dispatch, is_silent, tool_call_id,
            )

        # 3. Sanitize arguments（改写参数生效，None → 原参；仍走脱敏）
        schema = getattr(cap, "input_schema", None)
        effective_args = decision.modified_arguments if decision.modified_arguments is not None else arguments
        effective_args = _coerce_args(effective_args, schema)
        # 兜底 {"_raw": <无法解析文本>}：adapter 对「参数没解析成 JSON」的哨兵（可解析的已在
        # finalize 解包）。给直白报错，别让 _validate_args 报误导性的「必填项缺失」——那会诱导
        # 模型把参数照抄进 _raw、陷入死循环（见 protocols.llm.RAW_ARGS_KEY）。
        if list(effective_args) == [RAW_ARGS_KEY]:
            # 带上畸形原文（截断防炸 context）：模型下轮读这条 tool_result 才看得到自己写错了什么
            # → 据此自纠。线上 arguments 那格已被降级成合法 "{}"（见 openai._dump_tool_arguments），
            # 原文只能靠这条 error 传回。
            raw = str(effective_args[RAW_ARGS_KEY])
            excerpt = raw if len(raw) <= _RAW_ERROR_MAX_LEN else raw[:_RAW_ERROR_MAX_LEN] + " …(truncated)"
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id,
                f"[Error: invalid arguments for '{tool_name}': arguments were not valid JSON "
                f"and could not be parsed. You sent: {excerpt} — re-send the call with a "
                f"well-formed JSON arguments object]",
                is_dispatch, is_silent, tool_call_id,
            )
        # 剥掉 schema 未声明的顶层键（对任意调用生效）。放在 _raw 兜底之后，避免把哨兵剥空
        # 而丢掉「参数非法」信号；放在 required 校验之前，使「只发了未知键」被剥空后照样触发 required。
        effective_args = _strip_unknown_keys(effective_args, schema)
        # 参数校验：放在 coerce 之后，看到的是收敛后的类型（3 而非 "3"），不会假阳性。
        # 只拦 required/type/enum（见 _validate_args），失败回灌 LLM 让其改参重试，与 unknown-tool 同出口。
        err = _validate_args(effective_args, schema)
        if err is not None:
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id,
                f"[Error: invalid arguments for '{tool_name}': {err}]",
                is_dispatch, is_silent, tool_call_id,
            )
        # 审计副本（spec: capability-gateway「执行参数与审计参数分离」）：只进事件与
        # TOOL_AUDIT，**不进执行通道**——Provider 收 effective_args（授权后未脱敏原值，
        # 含 HITL 改写）。执行与审计共用同一份脱敏对象会把 Authorization 等功能参数
        # 销毁成 '***' 后才交给 provider（上游方案 H4）。
        audit_args = _sanitize(effective_args)

        # 4. Find provider
        provider = self._find_provider(cap.id)
        if provider is None:
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id,
                f"[Error: no provider found for '{cap.id}']", is_dispatch, is_silent, tool_call_id,
            )

        # 5. 记录 invocation（事件 + TOOL_INVOCATION / delegate conversation turn 入 memory）——审计通道
        await self._record_invocation(state, ctx, tool_name, cap, invocation_id, audit_args, is_dispatch, is_silent, tool_call_id)

        # 6. 执行（流式）——执行通道：effective_args（未脱敏）。透传 invocation_id（provider 据此
        # 登记在途句柄，供 cancel 对应）与 tool_call_id（控制工具据此把 origin_tool_call_id 写到 child）。
        provider_ctx = dataclasses.replace(
            ctx.provider_ctx,
            invocation_id=invocation_id,
            extra={**ctx.provider_ctx.extra, "tool_call_id": tool_call_id},
        )

        # ── 操作账本（spec: tool-operations，wp5；方案 §5.3 执行序）──────────────
        # 步骤 ②③：prepare → CAS started。operation_id 由调用侧（act/reconcile）铸好放进
        # provider_ctx；**缺失（裸调）→ 账本全程旁路**——既有单测/宿主直构 gateway 零改动。
        # 已 completed → 短路复用账本结果（O-T08 前半：同逻辑调用重入不再打 provider）。
        ledger = self._operation_store
        ledger_record = None
        if ledger is not None and op_id:
            from ctx_weft.protocols.operations import (
                OperationRecord as _OpRec, OperationStatus as _OpSt,
                OperationUpdate as _Upd2, operation_memory_result_id,
            )
            existing = await ledger.get(op_id, ctx.provider_ctx)
            if existing is not None and existing.status == _OpSt.COMPLETED:
                # O-T08 前半：同逻辑调用重入——账本已有完整结局，不再打 provider、
                # 不重复写 memory（首次执行已写 TOOL_RESULT）。
                # spec: tool-result-recovery——重放统一收敛（D4/D6）：结果进入对话前过
                # converge（禁全文直灌）；引用身份 = 账本原执行 invocation_id（attempts
                # 尾项），store 逐出后以账本全文重新入库。
                result_text = existing.result if isinstance(existing.result, str) else (
                    "[Replayed completed operation result]")
                ref_inv = (existing.attempts[-1]
                           if getattr(existing, "attempts", None) else invocation_id)
                result_text = await self._converge_result(
                    result_text, ctx, ref_inv, tool_name, cap.spillable)
                logger.info(
                    "CapabilityGateway: operation %s completed in ledger — replaying "
                    "result, provider not re-invoked", op_id)
                return InvocationResult(
                    invocation_id=invocation_id, tool_name=tool_name,
                    content=result_text, is_error=False)
            try:
                # 已有记录（重入/恢复）→ 不再 prepare：op_id 即身份，reconcile 注入的
                # 记录身份字段来自原回合（gateway 的 extra 里未必带），prepare 的身份
                # 比对会误拒。直接沿用 existing；只有无记录时才铸新行。
                ledger_record = existing if existing is not None else await ledger.prepare(_OpRec(
                    operation_id=op_id,
                    tenant_id=state.session.tenant_id,
                    session_id=state.session.id,
                    agent_id=state.agent.id,
                    assistant_record_id=str(ctx.provider_ctx.extra.get("assistant_record_id", "")),
                    tool_ordinal=int(ctx.provider_ctx.extra.get("tool_ordinal", 0)),
                    tool_name=tool_name,
                    task_id=state.task.id if state.task is not None else "",
                    args_hash=inv_key,
                    memory_result_id=operation_memory_result_id(op_id),
                ), ctx.provider_ctx)
                if existing is not None and existing.status == _OpSt.STARTED:
                    # 重入（HITL 等待超时后同逻辑调用再执行）：状态已 started——不重复
                    # 转移（started→started 非法），只追加 attempt。
                    ledger_record = await ledger.compare_and_set(
                        op_id, existing.revision,
                        _Upd2(append_attempt=invocation_id), ctx.provider_ctx)
                else:
                    # waiting_human → started（人答了续跑）/ prepared → started（首启）
                    ledger_record = await ledger.compare_and_set(
                        op_id, ledger_record.revision,
                        _Upd2(status=_OpSt.STARTED, append_attempt=invocation_id),
                        ctx.provider_ctx)
            except Exception as _led_exc:
                # 账本写失败 → PersistenceUnavailableError（复用 WP3 隔离语义；D3：
                # 账本是 H3 恢复的依据，静默降级会重新制造「伪装成功」）
                from ctx_weft.protocols.events import PersistenceUnavailableError as _PUE
                raise _PUE(
                    f"operation ledger unavailable for {op_id!r}: {_led_exc}"
                ) from _led_exc

        # 执行期异常不在这里碰账本（spec: tool-operations，wp5）：记录**有意**留在
        # started——副作用可能已发生、结果不可判定，这正是 started 要表达的事实。
        # 把它改写成别的状态等于替 WP6 的恢复策略表下结论。
        streamed = await self._stream_tool(
            provider, cap.id, effective_args, provider_ctx, state, invocation_id,
        )

        # 6b. provider 让出了 needs_human：流已停在此处（其后 yield 的事件从未被消费，见
        # `_stream_events`）。等待权归 gateway——provider 只**声明**需要人。
        if streamed.needs_human_ask is not None:
            needs_human_ask = streamed.needs_human_ask
            try:
                needs_human_ask_id, human = await self._resolve_human(
                    needs_human_ask, state, ctx, tool_call_id, stage=HITL_STAGE_TOOL,
                    invocation_key=inv_key)
            except UnattendedHitl:
                # 无人值守：没有人可答。**不抛给 agent loop**——转成一条说得清楚的工具
                # 结果，让 actor 自己拿主意。`ask_user` 的措辞由它自己出（文本住在
                # control_tools，与该工具的语义配套）；其余 provider 走通用措辞。
                # 这一条 logger.info 是刻意的：后台作业「遇到问题自己拿了主意」是运维
                # 最需要在日志里看见的一幕，比任何指标都早。
                #
                # 不早退、不走 `_error_and_record`：这条结果与普通工具结果的记账义务
                # 完全一样（CapabilityFinished + 配对 TOOL_RESULT），落回下面的公共
                # 出口即可，也就不必再复制一遍那套记账。`is_error` 保持 False——「没人
                # 可问」不是工具出错，是这次调用得到的答复。
                logger.info(
                    "Tool '%s' asked for a human in unattended task %s — answering "
                    "'no human available' and letting the actor decide",
                    tool_name, state.task.id)
                streamed = _ToolStream(texts=[
                    ASK_USER_UNATTENDED_RESULT if tool_name == ASK_USER_NAME
                    else _UNATTENDED_TOOL_NOTE])
            except Exception as _park_exc:
                from ctx_weft.core.loop.park import HitlPark as _HP
                if not isinstance(_park_exc, _HP):
                    raise
                # spec: tool-operations——park 上抛前把账本标到 waiting_human
                # （started→waiting_human 合法）：冷/热恢复据此识别「同身份续跑」。
                if ledger is not None and op_id and ledger_record is not None:
                    from ctx_weft.protocols.operations import (
                        OperationStatus as _Wh, OperationUpdate as _Wu)
                    try:
                        await ledger.compare_and_set(
                            op_id, ledger_record.revision,
                            _Wu(status=_Wh.WAITING_HUMAN), ctx.provider_ctx)
                    except Exception:
                        logger.exception(
                            "ledger waiting_human mark failed for %s (park proceeds)",
                            op_id)
                raise
            else:
                # 拿到了真人的决定：原有两条路，逐字节未改。
                if needs_human_ask.reply_as_result:
                    # 答复即结果：重入根本不发生（`ask_user` 走这条）。
                    streamed = _human_reply_as_result(human, needs_human_ask)
                else:
                    from ctx_weft.protocols.capability import HumanResumable
                    if not isinstance(provider, HumanResumable):
                        return await self._error_and_record(
                            state, ctx, tool_name, invocation_id,
                            f"[Error: {type(provider).__name__} yielded needs_human but does "
                            f"not implement HumanResumable]",
                            is_dispatch, is_silent, tool_call_id,
                        )
                    # 重入是**新调用** resume（不是恢复挂起的生成器）——局部状态已随原生成器
                    # 关闭而消失，全靠 ask.resume_state 带回。
                    streamed = await self._stream_events_safe(
                        provider.resume(
                            needs_human_ask_id, human, needs_human_ask.resume_state,
                            provider_ctx,
                        ),
                        provider, provider_ctx, state, invocation_id,
                    )

        metadata, is_error = streamed.metadata, streamed.is_error
        text = "\n".join(streamed.texts)
        if not text:
            # 空文本但流里带了非文本 part（例如 ask_user 只回了一张图）时，别说
            # 「(no output)」——那会让模型以为真的什么都没拿到，图却已经在 content 里了。
            # 其余分支（真的什么都没有 / is_error）逐字节保留原行为。
            text = "" if is_error or streamed.parts else "(no output)"
        # spec: tool-result-recovery——全文先行，收敛后置：store put（失败显式标记）→
        # 账本 completed 持**收敛前全文**（修 tool-operations「完整规范化结果」的 spill
        # 截断偏离）→ 收敛 → human note / 事件 / memory 只见收敛版。
        full_text = text
        text = await self._converge_result(
            full_text, ctx, invocation_id, tool_name, spillable=cap.spillable)
        # 人类备注：文本前置进 text，备注里的图片 part 与工具结果的 part 一起进最终 content。
        # 顺序为「备注图 → 工具图」，与文本顺序一致（[Human note: …] 也在工具输出之前）。
        note_text, note_parts = split_for_tool_result(decision.message)
        if note_text or note_parts:
            text = f"[Human note: {note_text}]\n{text}"
        content: str | list[ContentPart] = text
        # provider 在 result content 里交上来的非文本 part 要补上入口那三件套
        # （校验 / 外部化），**不合格的换占位、恒不抛**——见 `legalize_tool_result_parts`。
        # 「一律」包括 `_human_reply_as_result` 带回来的人类答复：它与 provider 的产出
        # 走同一条路，gateway 在这里也分不出来，而这恰恰是想要的——宿主若没接
        # `set_content_normalizer`，人递进来的字节在入口一次都没被校验过，这里是它唯一
        # 的关口。已经过过入口的（ref 形态）在 `legalize_tool_result_parts` 里直接透传，
        # 不重复付 put。
        #
        # `note_parts` **不**过这一道：它来自 `AuthorizationDecision.message`，源头同样
        # 是 HITL；放它进来只会在「备注带图 + 无 blob store」时多解一次 base64，换不到
        # 任何新保障。
        parts = streamed.parts
        if parts:
            parts = await legalize_tool_result_parts(
                parts, blob_store=self._memory_blob_store, ctx=ctx.provider_ctx)
        if note_parts or parts:
            # 过归一层：宿主 provider 可能给 dict 形态的 part（JSON 往返），
            # 与 MemoryEvent / LLMMessage 的 __post_init__ 共用同一份归一。
            content = normalize_content_parts([TextPart(text=text), *note_parts, *parts])

        # ── 步骤⑤：账本 completed（持久确认）——先于 TOOL_RESULT 写与事件（spec §5.3：
        # completed 后 memory 写失败 → 恢复按 operation_memory_result_id(op_id) 幂等补写，
        # 不再执行工具）。完整结果入账本（非审计截断文本；parts/blob ref 原样）——
        # spec: tool-result-recovery 起 result 恒为**收敛前全文**（此前 spill 截断先于
        # 账本写入，实际入账的是截断文本，偏离既有条款）。
        if ledger is not None and op_id and ledger_record is not None:
            from ctx_weft.protocols.operations import OperationStatus as _St, OperationUpdate as _Upd2
            ledger_record = await ledger.compare_and_set(
                op_id, ledger_record.revision,
                _Upd2(status=_St.COMPLETED, result=full_text, result_set=True,
                     error=str(metadata.get("error", "")) or None if is_error else None),
                ctx.provider_ctx)

        # 7. 记录 result（事件 + TOOL_RESULT 入 memory）——审计通道（脱敏副本）
        await self._record_result(state, ctx, tool_name, invocation_id, audit_args, content, is_error, is_dispatch, is_silent, tool_call_id)

        return InvocationResult(
            invocation_id=invocation_id, tool_name=tool_name,
            content=content, metadata=metadata, is_error=is_error,
        )

    @staticmethod
    def _error_result(invocation_id: str, tool_name: str, content: str | list[ContentPart]) -> InvocationResult:
        return InvocationResult(invocation_id=invocation_id, tool_name=tool_name, content=content, is_error=True)

    async def _error_and_record(
        self, state, ctx, tool_name, invocation_id, content, is_dispatch, is_silent, tool_call_id,
    ) -> InvocationResult:
        """执行前错误出口（未知工具 / 未授权 / 非法参数 / 无 provider）：返回 error_result 的同时，
        补一条配对 TOOL_RESULT 入 task 对话，使 act 在派发前已落库的 assistant LLM_RESPONSE.tool_call
        不悬挂（spec/06 §4 无损重建）。否则纯从 memory 重组 prompt（observe at max_turns / resume 恢复）
        时会出现 assistant(tool_calls=[id]) 无配对 tool 消息 → provider 400。
        派发(submit_*)/SILENT 工具的 tool_call 本就不入 task 层 LLM_RESPONSE、不会悬挂，故跳过落库
        （与 _record_result 的 is_dispatch/is_silent 处理一致）。
        """
        if not is_dispatch and not is_silent:
            await self._memory.ingest(
                MemoryEvent(
                    kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                    address=_tool_scope(state),
                    content=content,
                    timestamp=now_utc(),
                    role="tool",
                    metadata={"invocation_id": invocation_id, "tool_name": tool_name,
                              "tool_call_id": tool_call_id, "is_error": True},
                ),
                ctx.provider_ctx,
            )
        return self._error_result(invocation_id, tool_name, content)

    async def _record_invocation(
        self, state, ctx, tool_name, cap, invocation_id, sanitized, is_dispatch, is_silent, tool_call_id,
    ) -> None:
        """发 CapabilityInvoked + ingest（派发→agent 层 delegate conversation turn；
        普通→task 层 TOOL_INVOCATION；SILENT 普通工具不入对话）。"""
        from ctx_weft.core.loop.driver import make_event
        await self._event_bus.emit(make_event(state, EventType.CAPABILITY_INVOKED, payload={
            "invocation_id": invocation_id,
            "capability_name": tool_name,
            "capability_id": cap.id,
            "arguments": sanitized,
            "tool_call_id": tool_call_id,
        }, origin=EventOrigin.LOOP_CAPABILITY_GATEWAY))
        if is_dispatch:
            # 派发（spec 2026-06-28 §2.3；2026-07-03 修订）：**只有 delegate_plan 的 envelope 框**
            # 在此 eager 写（plan 框 + 配对 ack，避免 plan 框悬挂被 legalize 剥掉）。
            # **delegate_task 不再 eager 写框**——eager 框只能带「派发时刻」，无法落在「任务开始执行」
            # 时间线上；改由 child finalize 的 _ensure_dispatch_frame 铸框，框与 result 同锚 task.started_at
            # → 二者严格相邻、且在 started_at 时间线上（并发多派发也各自成对、不再堆叠错序）。
            if tool_name in _PLAN_DISPATCH_TOOLS:
                await self._memory.ingest(
                    MemoryEvent(
                        kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
                        address=_tool_scope(state),
                        content="",
                        timestamp=now_utc(),
                        role="assistant",
                        metadata={"origin_task_id": state.task.id,
                                  "parent_task_id": state.task.parent_task_id,
                                  "tool_calls": [{"id": tool_call_id, "name": tool_name,
                                                  "input": sanitized}]},
                    ),
                    ctx.provider_ctx,
                )
                # 配对的 ack tool result 不在这里写——见 `_record_result` 的 plan envelope
                # 分支。此刻工具还没执行、子任务尚不存在，能写的只有一句不含任何 id 的
                # 常量；而那条才是被持久化、被此后每一次对话重建重放的版本。
        elif not is_silent:
            await self._memory.ingest(
                MemoryEvent(
                    kind=MemoryKind.TOOL_AUDIT, scope=MemoryScope.TASK,
                    address=_tool_scope(state),
                    content=f"{tool_name}({sanitized})",
                    timestamp=now_utc(),
                    role="assistant",
                    metadata={"invocation_id": invocation_id, "tool_name": tool_name,
                              "tool_call_id": tool_call_id},
                ),
                ctx.provider_ctx,
            )

    async def _stream_tool(
        self, provider, cap_id, execution_args, provider_ctx, state, invocation_id,
    ) -> "_ToolStream":
        """流式执行 provider.invoke，聚合 result/metadata/error（含 needs_human 让出的 ask）。

        ``execution_args`` 是执行通道参数（授权/HITL 修改 + schema 校验后的 effective 值，
        **未脱敏**——审计脱敏副本不进这里，见 invoke 第 5/6 步注释）。

        事件消费循环与错误/取消处理分别由 `_stream_events` / `_stream_events_safe` 承担，
        `resume` 复用同一对 helper——不重复写这段循环（spec §2 的编排约束）。
        """
        return await self._stream_events_safe(
            provider.invoke(cap_id, execution_args, provider_ctx), provider,
            provider_ctx, state, invocation_id)

    async def _stream_events_safe(
        self, events, provider, provider_ctx, state, invocation_id,
    ) -> "_ToolStream":
        """`_stream_events` 外面套一层取消/异常安全网，`invoke` 与 `resume` 两处调用点共用。

        CancelledError（在途被打断）→ 调 provider.cancel 作安全网后重抛（provider 自身的 finally，
        如 bash terminate_tree，已先杀进程树）。其它异常 → 收敛为错误 result，不让 loop 崩。
        """
        try:
            return await self._stream_events(events, state, invocation_id)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await provider.cancel(invocation_id, provider_ctx)
            raise
        except Exception as exc:
            logger.exception("CapabilityGateway: invoke failed for invocation %s", invocation_id)
            return _ToolStream(texts=[f"[Exception: {exc}]"], is_error=True)

    async def _stream_events(
        self, events, state, invocation_id,
    ) -> "_ToolStream":
        """消费一个 `CapabilityEvent` 流，聚合 result/metadata/error。

        **`result` 事件的 `payload["content"]` 是 `str | list[ContentPart]`**——与三个
        执行入口、`HitlReply.message` 同一个联合类型，provider 想返图就把 `ImagePart`
        直接放进 content，不必分两处交。这里用 `split_for_tool_result` 把它拆成文本与
        非文本两半（先过 `normalize_content_parts`，否则 dict 形态的文本 part 会被那个
        冻结判据误判成图片），文本按 ``\n`` 累加、part 按到达顺序累加。`str` 进来时
        拆分器返回同一对象与空列表，纯文本路径零开销、逐字节不变。

        **`needs_human` 是流的终点**（spec §2）：见到即 `break`，不再从 `events` 拉下一个
        事件——其后 provider 让出的任何东西都不可见。provider 的局部状态随之消失，这正是
        `HitlAsk.resume_state` 存在的理由。

        **`pin` 恰好相反，不终止流**：它只是把一批能力加进当前 task 的可用面，工具随后照常
        yield 自己的 result（见 `CapabilityEvent.kind` 的注释）。

        **提前退出时显式 `aclose()`，不把关闭寄给 GC**：第三方 provider 的 `invoke` 是个
        异步生成器，它的 `finally` 里可能要杀进程、关连接、释放锁。靠 GC 意味着那些清理在
        一个不确定的时刻发生（`aclose()` 是协程，GC 只能凑合地安排它），而契约文本对实现者
        承诺的是「让出即关闭」。正常跑完的流 `aclose()` 是 no-op。
        """
        from ctx_weft.core.loop.driver import make_event
        out = _ToolStream()
        try:
            async for ev in events:
                if ev.kind == "needs_human":
                    out.needs_human_ask = ev.payload.get("ask")
                    break
                if ev.kind in ("stdout", "progress"):
                    await self._event_bus.emit(make_event(
                        state, EventType.CAPABILITY_PROGRESS, payload={
                            "invocation_id": invocation_id,
                            "kind": ev.kind,
                            "data": ev.payload.get("data", "")[:500],
                        }, origin=EventOrigin.LOOP_CAPABILITY_GATEWAY))
                elif ev.kind == "result":
                    ev_text, ev_parts = split_for_tool_result(
                        normalize_content_parts(ev.payload.get("content", "")))
                    out.texts.append(ev_text)
                    out.parts.extend(ev_parts)
                    out.metadata.update(ev.payload.get("metadata", {}))
                elif ev.kind == "error":
                    out.is_error = True
                    out.texts.append(
                        f"[Error {ev.payload.get('code', 'ERR')}: "
                        f"{ev.payload.get('message', '')}]"
                    )
                elif ev.kind == "pin":
                    # 「把这批能力加进当前 task 的可用面」。**与 needs_human 相反，不 break**：
                    # pin 不是流的终点，工具随后照常 yield 自己的 result。落进 cache 即当轮
                    # 生效——AssembledPrompt.tools 每次读都问 cache，不是装配期的快照。
                    self._cache.pin(
                        state.agent.id, state.task.id, ev.payload.get("capabilities") or [])
        finally:
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                # provider 的 finally 自身出错不该盖掉已聚合好的结果 / 正在传播的取消。
                with contextlib.suppress(Exception):
                    await aclose()
        return out

    async def _record_result(
        self, state, ctx, tool_name, invocation_id, sanitized, content, is_error, is_dispatch, is_silent, tool_call_id,
    ) -> None:
        """发 CapabilityFinished + ingest TOOL_RESULT（派发暂挂 / SILENT 不入 / 普通写 task 层）。"""
        from ctx_weft.core.loop.driver import make_event
        # 事件 payload 必须先脱敏再截断：content 可能是 list[ContentPart]（工具返图时），
        # 直接 `content[:8000]` 切的是**前 8000 个 part**——不报错、语义完全错，且图片 part 的
        # base64 会随 repr 泄漏进事件库。`redact_content_for_event` 对 str 输入返回同一对象，
        # 纯文本路径逐字节不变。
        redacted = redact_content_for_event(content)
        await self._event_bus.emit(make_event(state, EventType.CAPABILITY_FINISHED, payload={
            "invocation_id": invocation_id,
            "capability_name": tool_name,
            "arguments": sanitized,
            "outcome": "error" if is_error else "success",
            "result": redacted[:8000],
            "result_length": len(redacted),
            "tool_call_id": tool_call_id,
        }, origin=EventOrigin.LOOP_CAPABILITY_GATEWAY))
        if is_dispatch and tool_name in _PLAN_DISPATCH_TOOLS:
            # envelope: 给 `_record_invocation` eager 写的 plan 框补配对 ack，避免该框
            # 悬挂（被 legalize 剥掉）。**写在执行之后**，`content` 就是工具的真回执
            # ——逐条带「标题 + id」，于是重建对话里「我派发了哪几个」有稳定句柄可循；
            # 改造前这里写的是不含 id 的 `_PLAN_DISPATCH_ACK` 常量，回执里那份 id 清单
            # 只活在当轮。窗口从「两次相邻 ingest」变成「一次工具调用」，而
            # delegate_plan 是纯进程内 staging（微秒级）；工具报错时落的是真错误，
            # 也好过一句假的「Plan created」。
            await self._memory.ingest(
                MemoryEvent(
                    kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.AGENT,
                    address=_tool_scope(state),
                    content=content,
                    timestamp=now_utc(),
                    role="tool",
                    metadata={"origin_task_id": state.task.id,
                              "parent_task_id": state.task.parent_task_id,
                              "tool_call_id": tool_call_id},
                ),
                ctx.provider_ctx,
            )
        elif not is_dispatch and not is_silent:
            await self._memory.ingest(
                MemoryEvent(
                    kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
                    address=_tool_scope(state),
                    content=content,
                    timestamp=now_utc(),
                    role="tool",
                    metadata={
                        "invocation_id": invocation_id,
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "is_error": is_error,
                    },
                ),
                ctx.provider_ctx,
            )

    def _find_provider(self, capability_id: str) -> ToolCapabilityProvider | None:
        prefix = capability_id.rsplit(":", 1)[0]
        return self._provider_index.get(prefix)

    def _get_authorizer(self, capability_id: str) -> Authorizer:
        if capability_id in self._provider_authorizers:
            return self._provider_authorizers[capability_id]
        prefix = capability_id.rsplit(":", 1)[0]
        return self._provider_authorizers.get(prefix, self._default_authorizer)

    async def _resolve_human(
        self, ask: "HitlAsk", state: "LoopState", ctx: "LoopContext", tool_call_id: str,
        *, stage: str, invocation_key: str = "",
    ) -> "tuple[str, HitlDecision]":
        """登记 → 热等 → 拿到决定；被驱逐则抛 `HitlPark`。

        **全仓唯一的登记+等待+抛 park 的地方。** 热路径与冷路径在此收敛：冷路径由
        reconcile 经 `invoke` 再入，命中上面的决定缓存短路，根本走不到这里。

        `stage`（`HITL_STAGE_AUTHZ` / `HITL_STAGE_TOOL`）是决定缓存键的第三维——同一
        `tool_call_id` 下授权步与工具步各自独立登记等待，互不偷答案（安全修复，见
        `HitlRegistry`）。调用方必须显式传入，无默认值。

        `invocation_key` 是缓存键的第四维（复审 I3）：同一 tool_call id 下的**另一次**
        调用不得复用上一次的记录与决定。见模块级 `invocation_key()`。

        返回 `(hitl_id, decision)`——id 供工具侧路径（`resume`）用；授权侧调用点只解构决定。
        """
        if ctx.hitl is None or ctx.waiter is None:
            raise RuntimeError("HITL requested but no HitlService/HitlWaiter wired")
        # 冷路径重入的决定缓存短路。授权步在 `invoke` 顶上已查过一次（为的是连
        # `authorizer.authorize()` 都不调）；**工具步只能在这里查**——provider 必须先跑
        # 到 yield needs_human，才知道这次调用要问人。少了这一查，reconcile 重跑一个
        # 已答过的 `ask_user` 会走 `open()`（幂等命中那条已终局的请求）→ `waiter.wait()`
        # 见 `resolved` 判为驱逐 → `HitlPark`：人给过的答案永远送不回模型，任务原地重挂。
        # 走 `find_for_tool_call` 而非 `decision_for`：这里还要那条记录的 id（工具侧
        # `resume` 要收 ask_id）。仍 pending（活的等待）时 `decision is None`，因此
        # 「活请求不算已答过」这条与 `decision_for` 同一判据。
        cached = ctx.hitl.registry.find_for_tool_call(
            ctx.provider_ctx.session_id, tool_call_id, stage,
            invocation_key=invocation_key or None)
        # 同 `HitlRegistry.decision_for_tool_call`：待终局的应答也算数，它就是本轮的
        # 那一条（两阶段，spec 2026-09-09）。
        if cached is not None and cached.effective_decision is not None:
            return cached.id, cached.effective_decision
        req = await ctx.hitl.open(
            ask,
            session_id=ctx.provider_ctx.session_id,
            task_id=state.task.id,
            agent_id=state.agent.id,
            tool_call_id=tool_call_id,
            stage=stage,
            # 无人值守 → `open()` 抛 `UnattendedHitl`（守卫在唯一登记入口一处堵死）。
            # 本方法**不接**它：两个调用点各自有贴合上下文的转译（授权侧翻成拒绝、
            # 工具侧翻成一条工具结果），在这里统一兜住只会把两者压成同一句废话。
            unattended=state.task.unattended,
            invocation_key=invocation_key,
            tenant_id=ctx.provider_ctx.tenant_id,
        )
        human = await ctx.waiter.wait(req.id)
        if human is None:
            # 热窗口被驱逐 → 不放行也不拒绝。守住安全不变式：绝不调 provider.invoke。
            from ctx_weft.core.loop.park import HitlPark
            raise HitlPark(hitl_id=req.id, tool_call_id=tool_call_id)
        return req.id, human

    @staticmethod
    async def _authz_after_human(
        authorizer, cap, ctx: "LoopContext", arguments, tool_call_id: str,
        human: "HitlDecision",
    ) -> "AuthorizationDecision | None":
        """把决定喂回发起方去解释。未实现可选接口 = 契约违例 → `None`（调用方出错误 result）。

        **契约违例在这里判定并就地收敛，不再靠调用方 `except TypeError` 兜**（复审）：
        那个 except 罩着整段授权，会把 host authorizer 内部一个货真价实的 `TypeError`
        （它自己的 bug）也翻译成一条温和的 tool-result 错误——真故障被静默吞掉，看起来
        只是「这次调用没被授权」。`on_decision` 自己抛的异常现在照旧向上传播。
        """
        from ctx_weft.protocols.capability import HumanGatedAuthorizer
        if not isinstance(authorizer, HumanGatedAuthorizer):
            logger.error(
                "%s returned NeedsHuman but does not implement HumanGatedAuthorizer",
                type(authorizer).__name__)
            return None
        return await authorizer.on_decision(
            cap, ctx.provider_ctx, arguments, tool_call_id, human)

    async def _converge_result(
        self,
        full_text: str,
        ctx: "LoopContext",
        invocation_id: str,
        tool_name: str,
        spillable: bool = True,
    ) -> str:
        """收敛出口（spec: tool-result-recovery）：`spillable=False` 或未超阈值原样返回；
        其余走 `converge_tool_output`（全文交 SpillSink，上下文持收敛版）。

        重放/补写入口走同一条——再 spill 一次即可：可回读的 sink 借此在逐出/重启后重新
        入库，落盘的 sink 重写同名文件无害。不需要 restore 这条分支。"""
        if not spillable:
            return full_text
        return await converge_tool_output(
            full_text, invocation_id, self._spill_sink, ctx.provider_ctx,
            threshold=self._spill_threshold, preview_chars=self._spill_preview_chars,
            tail_chars=self._spill_tail_chars,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _human_reply_as_result(
    human: "HitlDecision", ask: "HitlAsk",
) -> "_ToolStream":
    """把人的答复直接变成工具结果（`HitlAsk.reply_as_result=True` 的出口，如 `ask_user`）。

    返回的 `_ToolStream` 与一次真的 provider 流**同型**：文本与非文本 part 分放两处，
    invoke 的公共出口一视同仁地处理（拼 content、合法化、落库）。人的答复因此与
    provider 自己产出的图走完全同一条路——包括 gateway 的校验/外部化那一道。

    两条**非空的**语义，从旧 `control_capability` 的出口原样移过来（Task 10）：

    - **拒绝**：`outcome == rejected` 时给答复加「Human declined: 」前缀（无正文则用固定句）。
      不加的话模型只看到一段孤零零的备注，读不出「这是一次拒绝」——把「人不同意」降级成
      了「人说了句话」。前缀只套在**文本 part** 上（`split_for_tool_result` 已拆开），
      以图收尾时才不会把前缀拍到图片对象上。
    - **空答复**：既无文本又无 part 时回落 `ask.prompt`（工具自己的确认文案），而不是把
      空串当答案送回去——那会在上游变成 `(no output)`，读起来像工具坏了。
    """
    text, parts = split_for_tool_result(human.message)
    if human.outcome == HITL_OUTCOME_REJECTED:
        text = f"Human declined: {text}" if text else "Human rejected the request."
    elif not text and not parts:
        text = ask.prompt
    return _ToolStream(texts=[text] if text else [], parts=parts)


def _tool_scope(state: "LoopState") -> MemoryAddress:
    """工具调用的 memory scope（session/task/agent）。统一构造，避免重复。"""
    return MemoryAddress(session_id=state.session.id, task_id=state.task.id, agent_id=state.agent.id)


def _coerce_scalar(value: str, json_type: Any) -> Any:
    """把字符串 value 转成 json_type 声明的标量；转不动则原样返回（绝不抛）。"""
    try:
        if json_type == "integer":
            return int(value)
        if json_type == "number":
            return float(value)
        if json_type == "boolean":
            low = value.strip().lower()
            if low in ("true", "1", "yes"):
                return True
            if low in ("false", "0", "no"):
                return False
    except (ValueError, TypeError):
        return value
    return value


def _coerce_args(arguments: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """按 input_schema 把字符串入参收敛到声明的标量类型。

    防御纵深：即便 schema 正确，模型仍可能给整型参数回传 "3"。这里据 schema 把它转成 int，
    免得工具做算术时崩。未知 key / 非字符串值 / 转不动的值一律原样保留。
    """
    props = (schema or {}).get("properties") or {}
    out = dict(arguments)
    for key, value in arguments.items():
        if not isinstance(value, str):
            continue
        decl = props.get(key)
        if isinstance(decl, dict):
            out[key] = _coerce_scalar(value, decl.get("type"))
    return out


def _declarable_props(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """剥键的资格判定：能明确「什么是已知键」时返回
    ``properties``，否则 ``None``（fail-open——不剥、也不拒）。

    仅当 schema 自身可判定时才生效：
      - schema 非 dict / 无 ``properties`` → 不可判定；
      - 含组合关键字 ``allOf/anyOf/oneOf/not`` 或顶层 ``$ref`` → 键可能由子 schema 声明；
      - ``additionalProperties`` 显式为 ``True`` 或子 schema（schema 主动允许附加属性）。
    """
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties")
    if not isinstance(props, dict) or not props:
        return None
    if any(k in schema for k in ("allOf", "anyOf", "oneOf", "not", "$ref")):
        return None
    ap = schema.get("additionalProperties")
    if ap is True or isinstance(ap, dict):
        return None
    return props


def _strip_unknown_keys(arguments: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    """丢弃 input_schema.properties 未声明的顶层键（对任意调用生效）。

    模型偶尔臆造 schema 里没有的键；畸形缓冲救援也可能抠出带杂键的对象（如把嵌套内层
    ``{"b": 2}`` 当参数）。剥掉它们，只把 schema 声明的参数交给工具，避免杂键流进工具实现，
    也避免错碎片被当成合法调用执行。

    仅剥顶层，不递归进嵌套对象（组合/``$ref`` 下递归易误删）。
    资格判定见 ``_declarable_props``。
    """
    props = _declarable_props(schema)
    if props is None:
        return arguments
    unknown = [k for k in arguments if k not in props]
    if not unknown:
        return arguments
    logger.info("CapabilityGateway: dropping arg keys not declared in schema: %s", unknown)
    return {k: v for k, v in arguments.items() if k in props}


# 只在这三类约束上拦截（spec B）：required 缺失 / type 不符 / enum 越界。
# additionalProperties / format / pattern 等故意忽略，避免未打磨的 schema 误伤现有工具。
_ENFORCED_KEYWORDS = frozenset({"required", "type", "enum"})


def _validate_args(arguments: dict[str, Any], schema: dict[str, Any] | None) -> str | None:
    """按 input_schema 校验入参，返回人读得懂的错误信息（可回灌 LLM）或 None（放行）。

    借 jsonschema 的成熟语义，但只对 required/type/enum 报错（见 _ENFORCED_KEYWORDS）；
    不开 format_checker，故 format 天然不查。schema 缺失/无 properties → 放行。
    任何校验自身异常（含畸形 schema）一律 fail-open，绝不让 loop 因校验崩。
    """
    if not schema or not schema.get("properties"):
        return None
    try:
        validator = jsonschema.Draft202012Validator(schema)
        messages = [
            err.message
            for err in validator.iter_errors(arguments)
            if err.validator in _ENFORCED_KEYWORDS
        ]
    except Exception:
        logger.exception("CapabilityGateway: arg validation crashed; allowing through")
        return None
    return "; ".join(messages) if messages else None


def _sanitize(arguments: dict[str, Any]) -> dict[str, Any]:
    """脱敏 headers 中的敏感 key。"""
    result = dict(arguments)
    if isinstance(result.get("headers"), dict):
        result["headers"] = {
            k: "***" if k.lower() in _REDACT_HEADERS else v
            for k, v in result["headers"].items()
        }
    return result
