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
from ctx_weft.core.hitl.registry import (
    HITL_STAGE_AUTHZ, HITL_STAGE_RERUN, HITL_STAGE_TOOL)
from ctx_weft.protocols.hitl import HITL_OUTCOME_REJECTED, HitlDecision
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id, is_internal_call_id
from ctx_weft.protocols.capability import (
    AuthorizationDecision, Authorizer, CapabilityProvider,
    RecoveryPolicy, RerunAuthorizer, ToolCapabilityProvider,
    RerunContext, normalize_recovery_policy, qualify,
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


def tool_audit_record_id(tool_call_id: str) -> str | None:
    """TOOL_AUDIT 的 memory 记录 id —— 与 `tool_result_record_id` 同一套派生，前缀换 `aud_`。

    确定性的理由与结果那条相同：同一次调用可能被重入（冷续跑、停机后窗口复用），随机 id
    会让审计记录一次次叠加。裸 wire id 不存在确定性派生 → `None` → provider 发自动 id。
    """
    return f"aud_{tool_call_id[3:]}" if is_internal_call_id(tool_call_id) else None


def tool_result_record_id(tool_call_id: str) -> str | None:
    """TOOL_RESULT 的 memory 记录 id —— **只对内部标识有定义**，裸 wire id 返回 None。

    内部标识形如 ``tc_{seq36}_{ord36}_{hash12}``，前缀等长，切掉 3 字符换 ``res_``。
    确定性派生的用处有二：`CapabilityFinished` 已发而 TOOL_RESULT 写入失败时，恢复按
    同一 id 幂等补写；dangling 判定据它认出「这次调用已有结果」。

    **返回 None 而不是硬切前缀**：本函数曾无条件 ``f"res_{tool_call_id[3:]}"``，于是
    ``""`` → ``"res_"``、``"call_1"`` → ``"res_l_1"``——两条裸 wire id 的 dangling 作结
    会撞同一个 ``"res_"``，后一条被 memory 的 ingest 幂等契约当 no-op 吞掉：结果从对话
    里消失，而那个 tool_call 从此永久悬挂。裸 id 本来就**不存在**确定性派生（同一个
    ``call_1`` 可以属于任意多个回合），那就说出来，让调用方显式面对。
    """
    return f"res_{tool_call_id[3:]}" if is_internal_call_id(tool_call_id) else None


# ── TASK 层 TOOL_RESULT 的唯一写入点（spec: tool-operations）────────────────────


async def ingest_tool_result(
    memory: "MemoryProvider",
    provider_ctx: "ProviderContext",
    scope: MemoryAddress,
    *,
    tool_call_id: str,
    content: "str | list[ContentPart]",
    is_error: bool = False,
    invocation_id: str = "",
    tool_name: str = "",
    via: str = "",
    extra_metadata: dict[str, Any] | None = None,
    task_manager: Any = None,
    blob_refs: "list[str] | None" = None,
) -> str | None:
    """把一条 TOOL_RESULT 写进 task 层对话，返回它的记录 id（自动 id 时为 None）。

    ## 为什么全仓只许这一处构造这种记录

    一次工具调用的结果可以从五个地方进 task 层对话：正常执行、执行前错误出口
    （未知工具/未授权/非法参数）、重跑作结、被打断的补位、账本完成而 memory 缺失的
    补写。它们此前各写各的 `MemoryEvent(...)`，于是「记录 id 由 tool_call_id 确定性
    派生」这条不变式**没有单一执行点**——五处里只有两处遵守，而唯一的验证者
    （`_dangling_tool_calls` 的 memory 通道）只认遵守的那种。实测复现过三条：

    1. 正常执行完的调用，若崩在下一次 LLM 请求之前，再进 reconcile 会被判成 dangling
       （它写的是自动 id），账本却说 COMPLETED → 触发补写 → **同一次调用两份结果**。
    2. 执行前错误出口写补位结果的全部目的就是「让 tool_call 不悬挂」，而它写完仍然
       悬挂 → 恢复时这次调用被重新交给 gateway，**再走一遍授权链**（authorizer 非
       确定时会真的执行）。
    3. 同回合两条裸 wire id 的 dangling 作结，旧的派生函数都产出 `"res_"` → 撞车 →
       后一条被 ingest 幂等契约吞掉，结果消失、tool_call 永久悬挂。

    记录 id 的决定因此收在这里一处（见下方 `record_id`），五个调用点不再有选择。
    `tests/unit/test_tool_result_ids.py` 有静态守卫盯着别处不再手搓。

    ## 不包含什么

    AGENT 层的两处 `role="tool"` 写入**不走这里**：`delegate_plan` 的配对 ack 与子任务
    finish 对的 ack。它们活在另一个平面——按 `tcall_...` 配对、由 finalize 的框/对机制
    管理、`_dangling_tool_calls` 从不扫描（它只读 TASK 层）。并进来只会长出一堆用不上
    的参数。

    `via` 只进 metadata 供诊断（`recovered_via`）；空则不写该键，正常执行路径的记录
    因此与改造前逐字节一致。
    """
    # 内部标识 → 确定性 `res_...`（dangling 判定据它认出「已有结果」，重复写入由 memory
    # 的同 id 幂等契约兜住）。裸 wire id → None → 由 memory provider 发自动 id：裸 id
    # 不存在确定性派生（同一个 `call_1` 可属于任意多个回合），硬造一个就是撞车。
    record_id = tool_result_record_id(tool_call_id)
    if record_id is None and tool_call_id:
        # 留痕：这条结果拿不到确定性 id，于是 dangling 判定认不出它（只能退回 wire
        # 配对，跨回合有歧义）。存量数据的既定形态，不是错误——但命中时要能查到。
        logger.info(
            "tool result for bare wire id %r gets a generated record id; dangling "
            "detection falls back to wire pairing for it", tool_call_id)
    metadata: dict[str, Any] = {"tool_call_id": tool_call_id, "is_error": is_error}
    if invocation_id:
        metadata["invocation_id"] = invocation_id
    if tool_name:
        metadata["tool_name"] = tool_name
    if via:
        metadata["recovered_via"] = via
    if extra_metadata:
        metadata.update(extra_metadata)
    # ``task_manager``：给了就经 `ingest_or_stage` 分流——这一轮还没算数时（典型：人刚给的
    # 答复回灌成的结果、拒绝授权写下的补位）暂存进窗口，提交时在 `HitlResolved` 之后落盘。
    from ctx_weft.core.loop.driver import ingest_or_stage
    await ingest_or_stage(
        memory,
        MemoryEvent(
            id=record_id,
            kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.TASK,
            address=scope,
            content=content,
            timestamp=now_utc(),
            role="tool",
            metadata=metadata,
            # ``blob_refs``：内容里若有**占位形态**的图（从事件补写时就是这一形态），
            # 它携带的 ref 必须显式声明——GC 的 mark 判据只看结构化字段、刻意绝不解析
            # 占位文案。不声明的话占位活着、字节却会在下一轮回收里被当孤儿删掉，模型
            # `media:get_image` 取回来的是「图片不可用」。同 `compact.collapse_task_layer`。
            blob_refs=list(blob_refs or []),
        ),
        provider_ctx,
        task_manager=task_manager, task_id=scope.task_id or "",
    )
    return record_id


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
        rerun_authorizers: "dict[str, RerunAuthorizer] | None" = None,
        spill_threshold: int = 4000,
        spill_preview_chars: int = 1000,
        spill_tail_chars: int = 1000,
        memory_blob_store: "MemoryBlobStore | None" = None,
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
        self._event_bus = event_bus
        self._provider_authorizers: dict[str, Authorizer] = provider_authorizers or {}
        if default_authorizer is None:
            from ctx_weft.providers.authorizer import AllowAllAuthorizer
            default_authorizer = AllowAllAuthorizer()
        self._default_authorizer: Authorizer = default_authorizer
        # spec: tool-operations——重跑授权（崩溃恢复时「还该不该再跑」）。与事前授权
        # **分开注册**、分开解析；**没有 default**——未注册即不重跑，由 core 代为作结。
        # 默认值必须是「不跑」，所以这里刻意不给兜底实现（对照 _default_authorizer）。
        self._rerun_authorizers: "dict[str, RerunAuthorizer]" = rerun_authorizers or {}
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
        reentry: bool = False,
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
        # 那个独立 `provider_ctx.operation_id` 字段的所有权转移——共享可变字段忘了清
        # 就会泄漏到后续无关 invoke（后台 observe 的 collect_process_report 曾因此命中
        # 别的操作的 completed 短路、回放错结果）。改成普通入参之后泄漏不可能发生。
        ledger_key = tool_call_id if is_internal_call_id(tool_call_id) else None

        # 1. Lookup capability（只处理 kind="tool"）。控制工具的全局可达性由 CapabilityCache 的
        # session 全局区保证（get_by_qualified_name 回退），gateway 无需特殊逻辑。
        cap = self._cache.get_by_qualified_name(state.agent.id, tool_name, state.task.id)
        if cap is None or cap.kind != "tool":
            return await self._error_and_record(
                state, ctx, tool_name, invocation_id,
                f"[Error: unknown tool '{tool_name}']", is_dispatch, is_silent, tool_call_id,
            )

        # 第四维 `invocation_key`：**同一次调用**才算合法重入（复审 I3）。用原始参数，
        # 不是改写后的——冷路径 reconcile 拿的就是对话里记着的原始参数。
        inv_key = invocation_key(tool_name, arguments)

        # 2. 重跑授权（spec: tool-operations）——**先于事前授权**。
        #
        # 账本里已经有一条 started 记录，意思是这次调用上一轮跑到过 provider，副作用
        # 可能已经发生；现在问的不是「该不该跑」而是「**还该不该再跑一遍**」。两个问题
        # 走两个 authorizer：默认 AllowAllAuthorizer 对后者会答 allowed=True，闷头重跑
        # 副作用——可靠性方案 H3 存在的意义就是堵这个。
        #
        # 顺序上重跑在前：注定不重跑的调用不该白跑一遍事前授权链（那可能还会 park 问人）。
        # 放行之后事前授权照常跑——「当初准跑」不等于「现在还准跑」，两者不互相顶替。
        #
        # 判据来自**事件流**，且**只在重入时读**（`reentry=True`）。热路径一次调用只会
        # 发生一次，折了也是空——实测五个场景里四个读到「没有」。重入只有两条来路：
        # ReconcileStep 与 HITL 冷续跑，两者都知道自己是重入，由调用侧告知即可。
        facts = None
        if reentry and ledger_key and ctx.event_store is not None:
            from ctx_weft.core.control.reducers import (
                CAP_FOLD_EVENT_TYPES, fold_operations, load_events_of_types)
            facts = fold_operations(await load_events_of_types(
                ctx.event_store, ctx.provider_ctx.session_id,
                CAP_FOLD_EVENT_TYPES)).get(ledger_key)

        # 2a. 同逻辑调用重入且已有结局 → 复用结果，**这里就 return**。
        #
        # 短路点必须在 `_record_invocation` 之前。放在它之后会先发一条
        # `CapabilityInvoked`，再从短路 return——不走 `_record_result`、不发
        # `CapabilityFinished`，于是事件流里留下一条**孤立的 INVOKED，而 provider 根本
        # 没被调用**（`test_capability_event_integrity.py` 钉住）。
        #
        # 连带：重放**不走事前授权**——重放什么都不执行，而 authorize 问的是「能不能
        # 跑」；顺带省掉一次可能 park 问人的授权。
        #
        # 事件里那份 result **就是当初进对话的收敛版**，直接用，不再收敛一遍（再来一次
        # 会把收敛说明自己当正文又切一刀）。唯一的例外是 `spillable=False`：那类输出
        # 不收敛，超过事件 payload 的上限就会被截断——而它们全是只读、可重新派生的
        # （`fs__read_file` / `results__read_tool_output` / skill 的 `list_files`，清一色
        # `side_effects=False`），正确动作是**重跑而非重放**，所以这里放它落下去。
        # 例外：结果就是人的答复（`HitlRegistry.result_is_human_reply`，即 `ask_user`）——恒按
        # HITL 决定重新生成，不重放事件里那份（可能是人重答之前的旧答复，且是脱敏截断版）。
        if facts is not None and facts.finished and facts.result is not None \
                and not facts.truncated and not (
                    ctx.hitl is not None and ctx.hitl.registry.result_is_human_reply(
                        ctx.provider_ctx.session_id, tool_call_id)):
            logger.info(
                "CapabilityGateway: %s already finished — replaying recorded result, "
                "provider not re-invoked", ledger_key)
            return InvocationResult(
                invocation_id=invocation_id, tool_name=tool_name,
                content=facts.result, is_error=(facts.outcome == "error"))

        if facts is not None and facts.invoked and not facts.finished:
            if normalize_recovery_policy(
                    getattr(cap, "recovery_policy", None)) is RecoveryPolicy.IDEMPOTENT:
                logger.info(
                    "CapabilityGateway: %s is idempotent — rerunning %s without review",
                    tool_name, ledger_key)
            else:
                verdict = await self._authorize_rerun(
                    cap, facts, state, ctx, arguments, tool_call_id, inv_key)
                if not verdict.allowed:
                    return await self._conclude_without_rerun(
                        state, ctx, cap, facts, verdict,
                        tool_name, invocation_id, ledger_key,
                        tool_call_id=tool_call_id,
                        is_dispatch=is_dispatch, is_silent=is_silent)

        # 3. Authorization：按 cap.id 前缀取 per-provider authorizer，无则用 default
        authorizer = self._get_authorizer(cap.id)
        # 决定缓存短路（冷路径重入 · 授权步）：registry 已有该 tool_call 的人工决定 →
        # 连 authorize() 都不调。工具步的同一短路在 `_resolve_human` 里（那时才知道要问人）。
        # 内存 pending（活的等待）不算「已答过」，registry.decision_for 已保证这点。
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

        # 5. 关窗：这一轮已经不可撤销了（spec: event-commit / tool-operations）。
        #
        # 未提交窗口的前提是「这一轮还什么不可逆的事都没发生，所以可以当没发生过」。
        # 关窗点此前只认一件事——act 收到第一个 chunk（用户看见输出了）。但**调用一个
        # 工具**是同一类不可逆动作，而 `begin_round` 的三个调用点里有两个是消息注入 /
        # HITL 冷续跑，ReconcileStep 正跑在那条路上、且在任何 chunk 之前：于是它的
        # invoke 落在开着的窗口里。实测后果是 `discard_round` 之后事件流**否认一次已经
        # 发生了副作用的调用存在过**——日志在撒谎。
        #
        # 不按 `side_effects` 或 `recovery_policy` 挑：前者的声明可能不完整（MCP / 旧
        # provider，方案自己说的），后者的 `idempotent` 说的是「重跑安全」而非「撤销
        # 安全」——一个幂等写工具重跑无害，但撤销之后事件流否认它写过外部系统，同样是
        # 撒谎。判据越简单越不会漂：**调了工具就关窗**。
        #
        # 代价是「窗口里跑过工具的那一轮不再可撤销」，那正是正确的语义。幂等（act 的
        # 每个 chunk 都调同一个函数），无 task_manager 时是 no-op。
        #
        # **唯一的例外是 `ask_user`**：它的 provider 只声明「要问人」，什么都不做。重入它
        # 是在回放一条已经在案、尚未终局的答复（reconcile 冷续跑），而那条答复在 LLM 开口
        # 之前必须能撤回——在这里关窗就把撤回的机会提前掐掉了（答复连同回灌出来的工具结果
        # 此刻还在窗口的暂存区里）。真要**新开**一个问题时，`_resolve_human` 会在登记之前
        # 关窗（问题必须看得见）。
        from ctx_weft.core.loop.steps.act import _commit_round
        if tool_name != ASK_USER_NAME:
            await _commit_round(state, ctx)

        # 6. 记录 invocation（事件 + TOOL_INVOCATION / delegate conversation turn 入 memory）——审计通道
        await self._record_invocation(state, ctx, tool_name, cap, invocation_id, audit_args, is_dispatch, is_silent, tool_call_id)

        # 6. 执行（流式）——执行通道：effective_args（未脱敏）。透传 invocation_id（provider 据此
        # 登记在途句柄，供 cancel 对应）与 tool_call_id（控制工具据此把 origin_tool_call_id 写到 child）。
        provider_ctx = dataclasses.replace(
            ctx.provider_ctx,
            invocation_id=invocation_id,
            extra={**ctx.provider_ctx.extra, "tool_call_id": tool_call_id},
        )

        # 执行期异常不在这里补任何「结束」事件：`CapabilityInvoked` 已发而
        # `CapabilityFinished` 未发，**正是**「副作用可能已发生、结果不可判定」这个
        # 事实本身。补一条就等于替恢复策略下结论。
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
                # 停在人那儿这件事由 **HITL 自己的账**表达（registry 的未决请求 +
                # HitlOpened 事件），恢复时 `ReconcileStep._waiting_human` 据此判定。
                # 不再另记一份。
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
            await ingest_tool_result(
                self._memory, ctx.provider_ctx, _tool_scope(state),
                tool_call_id=tool_call_id, content=content, is_error=True,
                invocation_id=invocation_id, tool_name=tool_name,
                task_manager=getattr(ctx, "task_manager", None))
        # 未授权出口同样要盖章。这条路**一条 capability 事件都不发**（「没有 INVOKED 就是没
        # 跑过」），所以从 capability 事件推退役的做法在这里是个真空：一条「人拒绝了」的决定
        # 永远等不到销账。由消费方盖章就没有这个洞——拒绝也是一次消费。
        if ctx.hitl is not None:
            await ctx.hitl.close_for_tool_call(ctx.provider_ctx.session_id, tool_call_id)
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
        # 授权类决定到此花掉：审批的作用是放行**这一次**调用，`CapabilityInvoked` 落库就是
        # 「门已经过了」。留着它等结果，等于让一份用过的批准在冷重入时还能开门（`:507` 的重跑
        # 分支之后紧接着就是 `:524` 的授权步，而决定缓存第四维 `invocation_key` 跨重启逐字节
        # 相同、**会**命中）。重入本该重新问：那时情况变了（可能已经跑过一半），「准不准重跑」
        # 与「准不准跑」是两个问题，各答各的。
        #
        # 能盖在这里，前提是审批**不进两阶段**（`reply_to_hitl` 对 authz/rerun 不开窗、
        # `defer=False`），所以此刻它已经终局。否则 `close()` 会因待终局跳过，而强行盖会让这枚
        # 章排在 `HitlResolved` 之前，折叠先摘 `opened` 再撞上终局找不到请求 → 终局丢掉。
        #
        # 只盖 authz/rerun：`tool` 那一类（`ask_user`）的答复要等它变成工具结果才算被消费。
        if ctx.hitl is not None:
            await ctx.hitl.close_for_tool_call(
                ctx.provider_ctx.session_id, tool_call_id,
                stages=(HITL_STAGE_AUTHZ, HITL_STAGE_RERUN))
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
            # 走 `ingest_or_stage`：`ask_user` 重入时这一轮还没提交（step 5 对它不关窗），
            # 审计记录与它配对的结果同进同退。
            from ctx_weft.core.loop.driver import ingest_or_stage
            await ingest_or_stage(
                self._memory,
                MemoryEvent(
                    id=tool_audit_record_id(tool_call_id),
                    kind=MemoryKind.TOOL_AUDIT, scope=MemoryScope.TASK,
                    address=_tool_scope(state),
                    content=f"{tool_name}({sanitized})",
                    timestamp=now_utc(),
                    role="assistant",
                    metadata={"invocation_id": invocation_id, "tool_name": tool_name,
                              "tool_call_id": tool_call_id},
                ),
                ctx.provider_ctx,
                task_manager=getattr(ctx, "task_manager", None), task_id=state.task.id,
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
            await ingest_tool_result(
                self._memory, ctx.provider_ctx, _tool_scope(state),
                tool_call_id=tool_call_id, content=content, is_error=is_error,
                invocation_id=invocation_id, tool_name=tool_name,
                task_manager=getattr(ctx, "task_manager", None))
        # 这次调用牵到的 HITL 决定（事前审核 / provider 自问 / 准重跑）到此全部失效——结果已
        # 经落进 memory，谁也不会再问它们。**盖在 ingest 之后**，顺序纪律见
        # `EventType.HITL_CLOSED`：提前盖会让折叠不再补，而效果其实还没落。
        #
        # 派发 / SILENT 工具不入对话，但它们同样可能被门控过，所以盖章不跟着上面的
        # `is_dispatch/is_silent` 分支——那两个分支管的是「写不写对话」，与「这条决定还
        # 要不要留着」无关。
        if ctx.hitl is not None:
            await ctx.hitl.close_for_tool_call(ctx.provider_ctx.session_id, tool_call_id)

    def _find_provider(self, capability_id: str) -> ToolCapabilityProvider | None:
        prefix = capability_id.rsplit(":", 1)[0]
        return self._provider_index.get(prefix)

    # ── 重跑授权（spec: tool-operations）──────────────────────────────────────

    def _get_rerun_authorizer(self, capability_id: str) -> "RerunAuthorizer | None":
        """三级解析，与 `_get_authorizer` 同构——**但没有 default**。

        缺省是「不重跑」，而不是某个兜底实现：重跑是危险决定，没人表过态时唯一安全的
        答案是不跑。逐工具粒度的用处在于给自己不控制的 provider 挂（`mcp-github:create_issue`
        知道怎么查证，同 provider 其它工具不受影响）。
        """
        if capability_id in self._rerun_authorizers:
            return self._rerun_authorizers[capability_id]
        prefix = capability_id.rsplit(":", 1)[0]
        return self._rerun_authorizers.get(prefix)

    async def _authorize_rerun(
        self, cap, facts, state: "LoopState", ctx: "LoopContext",
        arguments: dict | None, tool_call_id: str, inv_key: str,
    ) -> "AuthorizationDecision":
        """问「还该不该再跑」。**恒返回一个 decision**，不抛。

        没注册重跑授权、或授权实现自己抛了错 → core 代为作结「无从查证」，同样不重跑。
        没有注册**不是配置错误而是默认形态**：绝大多数 provider 无从查证外部真值。

        这里是 core 唯一还替人组织措辞的地方，但没有别的选择；它说的也是实话——
        「没人能查证」，而不是「查了查不到」。
        """
        unverified = AuthorizationDecision(
            allowed=False, result_is_error=False,
            message=(f"[Operation outcome unverified] 工具 {cap.name} 在执行中被中断，"
                     f"没有注册重跑授权，无法确定副作用是否已发生。不要重试这次调用。"))
        from ctx_weft.core.loop.park import HitlPark
        # 内部折叠出来的事实**不直接递给宿主**：`OperationFacts` 带着 result / outcome /
        # truncated 这些内核内务，宿主既不该读也不该据以分支。转成对外的只读视图。
        record = RerunContext(
            tool_call_id=tool_call_id,
            attempts=tuple(getattr(facts, "attempts", ()) or ()),
            last_attempt_at=getattr(facts, "last_attempt_at", None))
        rerun = self._get_rerun_authorizer(cap.id)
        if rerun is None:
            logger.info(
                "CapabilityGateway: no rerun authorizer for %s — concluding unverified", cap.id)
            return unverified
        try:
            # 决定缓存短路（冷路径重入 · 重跑步）：人已经答过就不再问第二遍。stage 与
            # 授权步分开——同一个 tool_call 的「准跑」不等于「准重跑」。
            cached = (
                ctx.hitl.registry.decision_for(
                    ctx.provider_ctx.session_id, tool_call_id, HITL_STAGE_RERUN,
                    invocation_key=inv_key)
                if ctx.hitl else None
            )
            if cached is not None:
                return await rerun.on_human_decision(
                    cap, record, ctx.provider_ctx, arguments, tool_call_id, cached[0])
            decision = await rerun.authorize_rerun(
                cap, record, ctx.provider_ctx, arguments, tool_call_id=tool_call_id)
            if decision.needs_human is None:
                return decision
            # 等待权归 gateway——重跑授权只**声明**需要人，不自己等（与事前授权同一纪律）。
            try:
                _hitl_id, human = await self._resolve_human(
                    decision.needs_human, state, ctx, tool_call_id,
                    stage=HITL_STAGE_RERUN, invocation_key=inv_key)
            except UnattendedHitl:
                # 无人值守：没有人能批准重跑 → 保守作结，**不**抛给 agent loop。
                logger.info("CapabilityGateway: unattended rerun ask for %s — concluding", cap.id)
                return unverified
            return await rerun.on_human_decision(
                cap, record, ctx.provider_ctx, arguments, tool_call_id, human)
        except HitlPark:
            # 冷路径：park 是控制流信号，必须穿出去。穿出前把账本标到 waiting_human
            # （started→waiting_human 合法），冷恢复据此识别「同身份停在人那儿」。
            raise
        except Exception:
            logger.exception(
                "CapabilityGateway: rerun authorization failed for %s — concluding unverified",
                cap.id)
            return unverified

    async def _conclude_without_rerun(
        self, state: "LoopState", ctx: "LoopContext", cap, facts,
        verdict: "AuthorizationDecision", tool_name: str, invocation_id: str,
        ledger_key: str, *, tool_call_id: str, is_dispatch: bool, is_silent: bool,
    ) -> InvocationResult:
        """作结一次不重跑的调用：把 verdict 的话写成这次调用的工具结果。

        「查到了真结果」与「谁也查不到」走的是**同一条路**——对 core 而言两者没有区别，
        差的只是 message 的内容。不改 task 状态、不停机：**结果不确定是一种工具结果，
        不是一种控制流**。agent 下一轮读到它，在任务上下文里决定怎么办。

        走 `_record_result`（发 `CapabilityFinished` + 写 TOOL_RESULT）。这里发 FINISHED
        **不是**孤立事件——恰恰相反：作结只发生在 `facts.invoked and not facts.finished`
        时，也就是上一轮那条 `CapabilityInvoked` 还悬着，这一发正好把它闭合。不发的话
        那次调用会永远停在「已调用未完成」，下一次恢复又要重问一遍重跑授权。

        事件用的是**原执行的** invocation_id（`facts.attempts` 尾项）而非这次重入新生成
        的，配对才对得上。
        """
        text, parts = split_for_tool_result(verdict.message)
        ref_inv = facts.attempts[-1] if getattr(facts, "attempts", None) else invocation_id
        # spec: tool-result-recovery——凡进对话的结果统一过收敛，禁全文直灌。
        text = await self._converge_result(text, ctx, ref_inv, tool_name, cap.spillable)
        content: "str | list[ContentPart]" = (
            normalize_content_parts([TextPart(text=text), *parts]) if parts else text)
        await self._record_result(
            state, ctx, tool_name, ref_inv, {}, content,
            verdict.result_is_error, is_dispatch, is_silent, tool_call_id)
        logger.info("CapabilityGateway: %s concluded without re-running", ledger_key)
        return InvocationResult(
            invocation_id=invocation_id, tool_name=tool_name,
            content=content, is_error=verdict.result_is_error)

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
        # 要**新开**一个问题：它必须立刻看得见。开着的未提交窗口会把 `HitlOpened` 挡在
        # 缓冲里，人就永远看不到这个问题（step 5 对 `ask_user` 不关窗，见那里）。
        from ctx_weft.core.loop.steps.act import _commit_round
        await _commit_round(state, ctx)
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
        _rearm_commit_point_after_hot_reply(state, ctx, req.id)
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


def _rearm_commit_point_after_hot_reply(
    state: "LoopState", ctx: "LoopContext", hitl_id: str,
) -> None:
    """热应答把协程叫醒之后，给「这一轮」重新武装提交点（spec 2026-09-09）。

    `reply_to_hitl` 为这条答复开了未提交窗口，决定也只是待终局。被叫醒的 run 早就过过
    一次提交点（`ROUND_COMMITTED_KEY` 是 run 级幂等标志），不清掉它，`_commit_round`
    永远是 no-op：窗口要到 run 结束才关，期间该 task 的事件全挡在缓冲里，host 看不见
    下一个问题——连续提问时「第二个问题答了没用、一直让人重答」就是这么来的。

    只在**这条答复自己开的窗**上做（`round_hitl_id` 对得上）。窗若是别的应答开的
    （例如冷续跑的一轮里，reconcile 重放的工具又热等了一次授权），那一轮本来就没提交，
    标志本就是 False，也轮不到这里决定怎么撤。

    同时记下 `HOT_REPLY_ROUND_KEY`：这一轮若在 LLM 开口前被暂停，act 要走**热撤销**
    （撤回答复后 park），而不是整轮 `RoundDiscarded`——这个 run 的 `RUN_STARTED` 早已
    落盘，把它的结尾连同缓冲一起丢掉，日志里就留下一个永不结束的 run。
    """
    tm = ctx.task_manager
    if tm is None or not tm.is_round_open(state.task.id):
        return
    if tm.round_hitl_id(state.task.id) != hitl_id:
        return
    from ctx_weft.core.loop.driver import HOT_REPLY_ROUND_KEY, ROUND_COMMITTED_KEY
    state.extra[ROUND_COMMITTED_KEY] = False
    state.extra[HOT_REPLY_ROUND_KEY] = hitl_id


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
