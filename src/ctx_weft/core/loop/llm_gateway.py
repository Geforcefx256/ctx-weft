"""LLM 请求发送的统一关口：集中所有「发送前合法化」方法 + 流式转发。

四个 step 通过 :func:`build_llm_request` 编译或复用 PromptContext，并构造本轮请求。
请求经 :func:`stream_llm` 发送，使「发送前合法化」成为**唯一**关口：装配层（composer）
只负责构造逻辑消息结构，不做合法化；连续同角色 / 孤立 tool result 一律在这里、发送前
统一处理。

合法化六条不变式（:func:`legalize_messages` 内顺序：剥离悬挂 tool_call → 把 tool result 挪到
其 tool_call 之后 → 删空 content → 丢前导非 user → 丢孤立 tool result → 合并连续同角色）：
  1. :func:`drop_dangling_tool_calls` —— 剥离无后继 tool_result 配对的 assistant tool_call。
     reconcile/act 正常已补齐 dangling（见模块尾注），此处是发送前最后**防御性**兜底：只删
     不补、命中即打 ERROR 日志（说明上游对账漏了）。Anthropic 对悬挂 tool_use 直接 400。
  2. :func:`reorder_tool_results_after_calls` —— 把每条 tool result 紧挪到其 assistant
     tool_call 之后。跨层按 timestamp 归并会把 user 回合插进 assistant↔result 之间（OpenAI
     兼容端点据此 400），集合级配对查不出这种错位——本步按 owner 重排消除中间夹角色。只搬不删。
  3. :func:`remove_empty_messages` —— 删除内容为空且无 tool_calls 的 user/assistant 消息
     （Anthropic 对空 content 块 400）。同为防御性兜底，命中打 ERROR 日志。① 把全悬挂的
     assistant 剥成空消息后，正好由此清理。tool 消息即便空也不在此删（删了会制造悬挂）。
  4. :func:`ensure_leading_user` —— 丢弃开头 role != "user" 的消息直到首条为 user（Anthropic
     硬规则，否则 400）。丢弃前导 assistant 后其配对 tool 会成孤儿，由下一步清理。
  5. :func:`drop_orphan_tool_results` —— 丢弃 tool_call_id 无前序 assistant tool_call
     配对的 ``role="tool"`` 消息。compact 丢弃 assistant 但保留 TOOL_RESULT、recall 窗口
     在 assistant↔result 之间截断、跨层按 timestamp 归并错位等都会产生此类孤儿，会被
     Anthropic / OpenAI 直接 400。
  6. :func:`merge_consecutive_messages` —— 合并连续同角色消息（多模态安全）。

本模块仅依赖 protocols，且只被本 loop 层的 step import；故置于 ``core/loop`` 下。若将来
装配等下层也需复用这些合法化函数，应改为下沉到 ``core`` 根，避免下层反向依赖 loop。

注：「悬挂 assistant tool_call（无后继 result）」的**正路**是上游对账补 TOOL_RESULT
（resume 前的 ReconcileStep 重跑工具、act 为被打断工具补 result），本关口的
:func:`drop_dangling_tool_calls` 只是发送前防 400 的兜底，命中打 ERROR 提示上游漏补。
adapter 不应重复实现以上不变式——已在 core 层集中保证。

另注：「prompt 必须以 user 收尾」不下沉到此处——那是 act 装配期的语义兜底（注入具体
文案、且 provider 并不强制），若在此每个 turn 强制，会在工具循环里于 tool result 之后误
插一条 user，篡改正常的工具调用流。该兜底保留在 composer._build_actor_messages。
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import logging
import random
from collections.abc import AsyncIterator, Sequence
from time import monotonic
from typing import TYPE_CHECKING, Any

from ctx_weft.core.loop.driver import make_event
from ctx_weft.core.loop.llm_stream import DelegatingLLMStream, stream_until_cancel
from ctx_weft.core.models.discriminators import InterruptReason
from ctx_weft.core.orchestrator.model import ResolvedModel
from ctx_weft.core.utils.content import redact_content_for_event
from ctx_weft.protocols.content import rehydrate_content
from ctx_weft.core.utils.estimate import (
    dynamic_max_tokens,
    estimate_message_tokens,
)
from ctx_weft.protocols import LLMMessage, LLMOutageError, TextPart
from ctx_weft.protocols.events import EventType

if TYPE_CHECKING:
    from ctx_weft.core.assembler.context import PromptContext
    from ctx_weft.protocols import ContentPart, LLMChunk, LLMClient, LLMRequest

logger = logging.getLogger(__name__)

# request.metadata 瞬态键：request_prompt_estimate 写入的估算基线（真实 context_tokens 或 0），
# 供 act 回喂时算「真实段 = usage.prompt_tokens − 基线」。
PROMPT_EST_BASE_KEY = "prompt_est_base"

# request.metadata 瞬态键：本次请求的「估算段」——增量路径为 delta、整份路径为 full，两者都是
# tokenizer.count 的直接产出（未经 max(..., context_tokens) 的 floor）。act 回喂必须用这个值而
# 非返回值：返回值在整份路径上可能被 floor 成上一轮的真实 context_tokens（当启发式低估、被
# 真实基线 floor 时），若拿 floor 后的返回值回喂，ratio≈1 的假样本会把伺服系统性拖向 1——
# floor 只应影响 max_tokens 的保守性，不应污染校准。
PROMPT_EST_SEG_KEY = "prompt_est_seg"



def resolve_llm_identity(state) -> tuple[str, str]:
    """本次 LLM 调用实际使用的 (model, account)。

    真值是 state.resolved_model —— 派发时由 AgentLifecycleManager 解出的那一个。
    此前读 session.llm_model 并两级兜底到 agent.runtime / "mock"，那两级
    永远命中不了（runtime 从不填 agent.runtime["llm_model"]），于是未配置
    时恒报 "mock"。ResolvedModel 永远是解析过的确定值，报不出假数据。
    """
    rm = state.resolved_model
    if rm is None:
        raise RuntimeError(
            "resolve_llm_identity: state.resolved_model must be set before dispatch"
        )
    return rm.model, rm.account


def drop_dangling_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """剥离无后继 tool_result 配对的 assistant tool_call（防御性，命中打 ERROR 日志）。

    正路是上游对账补齐（reconcile/act）；此处仅发送前兜底，只删不补——保证不 400，但打
    ERROR 提示上游漏补。多 tool_call 的 assistant 只剥悬挂的那几个，保留有 result 的；全悬挂
    则剥成空 tool_calls（随后由 :func:`remove_empty_messages` 清理）。绝不动有 result 的调用，
    故不会反向制造孤立 tool result。
    """
    resolved = {m.tool_call_id for m in messages if m.role == "tool" and m.tool_call_id}
    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            kept = [tc for tc in m.tool_calls if tc.get("id") in resolved]
            if len(kept) != len(m.tool_calls):
                dangling = [tc.get("id") for tc in m.tool_calls if tc.get("id") not in resolved]
                logger.error(
                    "stream_llm: dropping %d dangling tool_call(s) with no tool_result "
                    "(上游对账漏补？): %s",
                    len(dangling), dangling,
                )
                m = LLMMessage(
                    role=m.role, content=m.content, tool_calls=kept,
                    tool_call_id=m.tool_call_id, reasoning_content=m.reasoning_content,
                    provider_state=m.provider_state,
                )
        out.append(m)
    return out


def _is_empty_content(content: "str | list[ContentPart]") -> bool:
    """内容是否为「空」：空串/纯空白，或空列表/仅含空白 TextPart。非文本块（图片等）视为有内容。

    与 composer._is_blank_content 语义相近但**不同**——本函数对字符串/TextPart 文本都
    做 ``.strip()``（纯空白 "   " 判空），composer 那份不 strip（纯空白判非空）。两者共存
    今日安全仅因为 legalize_messages 链路最终经本函数把纯空白消息滤掉；composer 侧的
    "   " 不是本函数意义上的「空」不会传导成 bug，纯属两处判据本就不追求一致——但这是个
    没写下来的非局部不变式，改动任一处前请先看 tests/unit/test_composer_vs_gateway_blank_content.py
    钉住的那对语义。"""
    if isinstance(content, str):
        return not content.strip()
    if not content:
        return True
    for p in content:
        if isinstance(p, TextPart):
            if p.text and p.text.strip():
                return False
        else:
            return False
    return True


def remove_empty_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """删除内容为空且无 tool_calls 的 user/assistant 消息（防御性，命中打 ERROR 日志）。

    Anthropic 对空 content 块直接 400。只管 user/assistant：纯 tool_use 的 assistant（有
    tool_calls、无文本）合法须保留；tool 消息即便空也不删（删了会制造悬挂 tool_call）。
    """
    out: list[LLMMessage] = []
    for m in messages:
        if m.role in ("user", "assistant") and not m.tool_calls and _is_empty_content(m.content):
            logger.error(
                "stream_llm: dropping empty-content %s message (no text, no tool_calls)", m.role
            )
            continue
        out.append(m)
    return out


def drop_orphan_tool_results(messages: list[LLMMessage]) -> list[LLMMessage]:
    """丢弃 tool_call_id 无前序 assistant tool_call 配对的孤立 tool result 消息。"""
    seen: set[str] = set()
    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "assistant":
            seen.update(tc.get("id", "") for tc in m.tool_calls if tc.get("id"))
        elif m.role == "tool":
            if not m.tool_call_id or m.tool_call_id not in seen:
                continue  # 孤儿 → 丢弃
        out.append(m)
    return out


def _merge_message_content(
    prev: "str | list[ContentPart]", cur: "str | list[ContentPart]"
) -> "str | list[ContentPart]":
    """连接两条消息的内容。两侧皆为 str → 以空行拼接（保留原文本语义）；任一侧为多模态
    时 → 拼成 ContentPart 列表（str 侧转 TextPart），避免把图片拍扁丢弃。"""
    if isinstance(prev, str) and isinstance(cur, str):
        return f"{prev}\n\n{cur}"

    def as_parts(c: "str | list[ContentPart]") -> "list[ContentPart]":
        return [TextPart(text=c)] if isinstance(c, str) else list(c)

    return as_parts(prev) + as_parts(cur)


def ensure_leading_user(messages: list[LLMMessage]) -> list[LLMMessage]:
    """丢弃开头 role != "user" 的消息直到首条为 user（Anthropic 首条必须 user，否则 400）。

    只动头部、不注入文案——与「以 user 收尾」语义兜底（保留在 composer）不同，对正常工具
    循环（中段 tool result 之后无 user）无影响。前导 assistant 被丢后其配对 tool 会成孤儿，
    由随后的 drop_orphan_tool_results 清理（见 stream_llm 串联顺序）。
    """
    i = 0
    while i < len(messages) and messages[i].role != "user":
        i += 1
    return messages[i:] if i else messages


def reorder_tool_results_after_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """把每条 tool result 紧挪到其 assistant tool_call 之后，消除中间夹着的非 tool 消息。

    OpenAI 兼容端点要求 ``role="tool"`` 紧跟在 ``role="assistant"(tool_calls)`` 之后、按
    ``tool_call_id`` 配对，中间不得夹其它角色（夹了即 400「insufficient tool messages
    following tool_calls message」）。跨层历史按 timestamp 归并会把 user 回合插进
    assistant↔tool_result 之间，甚至让 result 落到 call 之前；集合级配对
    （:func:`drop_dangling_tool_calls` / :func:`drop_orphan_tool_results`）只校验「id 是否
    存在」查不出这种**错位**。本步按 owner 把每个 tool result 移到其 assistant tool_call 之后，
    并按该 assistant 的 ``tool_calls`` 顺序排列。无 owner（没有任何 assistant 调用该 id）的
    孤儿 tool 原地保留，交 :func:`drop_orphan_tool_results` 处理；只搬不删、不补。

    存量歧义留痕（spec: conversation-integrity）：新数据的 id 在摄入点铸为全局唯一，
    命中「同一 id 被多个 assistant 携带」只可能来自改造前的裸 wire id 记录——行为维持
    现状（结果在每个携带者后各发一遍），但 MUST 留痕不静默。
    """
    owner_counts: dict[str, int] = {}
    for m in messages:
        if m.role == "assistant":
            for tc in m.tool_calls:
                tcid = tc.get("id")
                if tcid:
                    owner_counts[tcid] = owner_counts.get(tcid, 0) + 1
    dup_ids = [tcid for tcid, n in owner_counts.items() if n > 1]
    if dup_ids:
        logger.error(
            "stream_llm: %d tool_call_id(s) each claimed by multiple assistant messages "
            "(legacy bare-wire ids? their results will be duplicated after every owner): %s",
            len(dup_ids), dup_ids,
        )
    owned_ids = {
        tc.get("id")
        for m in messages
        if m.role == "assistant"
        for tc in m.tool_calls
        if tc.get("id")
    }
    results_by_id: dict[str, list[LLMMessage]] = {}
    for m in messages:
        if m.role == "tool" and m.tool_call_id in owned_ids:
            results_by_id.setdefault(m.tool_call_id, []).append(m)

    out: list[LLMMessage] = []
    for m in messages:
        if m.role == "tool" and m.tool_call_id in owned_ids:
            continue  # 由其 owner assistant 之后统一发出（见下）
        out.append(m)
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                out.extend(results_by_id.get(tc.get("id"), ()))
    return out


def merge_consecutive_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """合并连续相同角色的消息（tool 消息因绑定 tool_call_id 不合并）。"""
    merged: list[LLMMessage] = []
    for msg in messages:
        if (
            merged
            and merged[-1].role == msg.role
            and msg.role != "tool"
        ):
            prev = merged[-1]
            merged[-1] = LLMMessage(
                role=prev.role,
                content=_merge_message_content(prev.content, msg.content),
                tool_calls=prev.tool_calls + msg.tool_calls,
                provider_state=_merge_provider_state(prev.provider_state, msg.provider_state),
            )
        else:
            merged.append(msg)
    return merged


def _merge_provider_state(left: dict, right: dict) -> dict:
    """Keep continuation metadata attached when legalizing consecutive assistants."""
    if not left or not right:
        return left or right
    merged = left | right
    if "anthropic" in left and "anthropic" in right:
        a, b = left["anthropic"], right["anthropic"]
        merged["anthropic"] = a | b
        for key in ("thinking_blocks", "tool_call_ids"):
            merged["anthropic"][key] = [*a.get(key, ()), *b.get(key, ())]
    return merged


def legalize_messages(messages: list[LLMMessage]) -> list[LLMMessage]:
    """发送前把消息序列规整为 provider 合法形态（六条不变式，见模块 docstring）。

    纯函数、不依赖 llm/网络，便于单测全链路。``stream_llm`` 在发送前调用一次。
    """
    return merge_consecutive_messages(
        drop_orphan_tool_results(
            ensure_leading_user(
                remove_empty_messages(
                    reorder_tool_results_after_calls(
                        drop_dangling_tool_calls(messages)
                    )
                )
            )
        )
    )


def _estimate_message_tokens(m: LLMMessage, count) -> int:
    """单条消息的 provider 计费估算（往大了估）：文本 content + 图片 part + framing +
    tool_calls 参数 + reasoning_content。

    计费项定义在 core.utils.estimate（``estimate_content_tokens`` / ``estimate_tool_calls_tokens``）——
    单一真源，prepare/composer 对 memory 记录/装配消息共用同口径。tool_calls 的 arguments、
    reasoning、图片此前都没计入，是"单轮新增里一坨数不到的东西 > margin"致 400 的洞。

    ``count``：文本费率回调，caller 传 ``tokenizer.count``（已校准）。
    """
    return estimate_message_tokens(m, count)


def _estimate_request_tokens(request: "LLMRequest", count) -> int:
    """估算整份待发 prompt 的 token（system + 全部 messages + tools schema）。

    每条消息经 :func:`_estimate_message_tokens` 计全部计费项（文本 + tool_calls 参数 +
    reasoning + 图片 + framing）。含本轮新加的 role="tool" result。``count``：文本费率
    回调，caller 传 ``tokenizer.count``（已校准）。tools 面费率走单一真源
    ``estimate_tools_tokens``（与 prepare 首估同口径）。
    """
    from ctx_weft.core.utils.estimate import estimate_tools_tokens

    total = count(request.system or "")
    for m in request.messages:
        total += _estimate_message_tokens(m, count)
    total += estimate_tools_tokens(request.tools, count)
    return total


def request_prompt_estimate(tokenizer, request: "LLMRequest", loop_guard, baseline_msg_count: "int | None") -> int:
    """算本请求真实 prompt token 的最佳估算，由 gateway 挂到 request。

    - **增量**（``baseline_msg_count`` 非 None 且有真实基线 ``context_tokens>0``）：
      真实基线 + 仅 ``messages[baseline_msg_count:]``（本轮新增尾段，如工具结果）的估算。
      基线前的历史采信 provider 真实测量值、不重估——避免 len//4 对 CJK 历史大头系统性低估
      （正是它把整份重估压到基线之下、漏掉本轮增量而导致 max_tokens 过大 400 的根因）。
      **工具面追踪（spec: tool-schema-budget）**：请求的工具面指纹（规范化定义哈希，
      覆盖 name+description+schema）与 guard 记录的上次指纹不同时，估算计入 schema 差值
      （新面估算 − 上次面估算）——运行期 pin 大 schema 后估算立即增长；工具面不变时
      增量行为与旧口径逐字节一致（纯消息增量）。首次记录（旧签名为空）不加减值——
      真实基线已含当时的工具面。
    - **首次 / 一次性**（无循环内基线）：``max(整份估算, context_tokens)``——整份估算打底，
      并不低于上一步真实测量值（更保守，永不 400；上步后若发生压缩，偏大只是少给输出）。

    估算全经 ``tokenizer.count``（已校准值）；不再有 raw/factor 概念——伺服校准下沉到
    adapter 的 ``LLMClient.tokenizer`` 内部（见 providers.llm.tokenizer.HeuristicTokenizer）。
    metadata 记两个瞬态键：:data:`PROMPT_EST_BASE_KEY`（估算基线）与 :data:`PROMPT_EST_SEG_KEY`
    （本次估算段，未经 floor 的 tokenizer.count 直接产出），供 usage 到达后 act 回喂
    ``tokenizer.observe`` 用——整份路径的返回值可能被 ``max(full, ctx_tokens)`` floor 成上一轮
    的真实值（正是本函数存在的校准动机场景：启发式低估、被真实基线兜住），若回喂用 floor 后
    的返回值会产出 ratio≈1 的假样本、把伺服系统性拖向 1；估算段是 floor 前的值，不受污染。
    """
    from ctx_weft.core.utils.estimate import estimate_tools_tokens, tools_signature

    ctx_tokens = getattr(loop_guard, "context_tokens", 0) if loop_guard is not None else 0

    def _track_tools() -> int:
        """指纹变化时返回 schema 差值（新 − 上次），并推进 guard 记录；不变返回 0。
        首次（旧签名为空）只记录不加值（真实基线已含当时的面）。guard 字段经
        getattr 读写——SimpleNamespace 测试替身与旧持久化快照都兼容。"""
        if loop_guard is None:
            return 0
        sig = tools_signature(request.tools)
        last_sig = getattr(loop_guard, "last_tools_signature", "")
        if sig == last_sig:
            return 0
        new_est = estimate_tools_tokens(request.tools, tokenizer.count)
        delta = 0
        if last_sig:
            delta = new_est - getattr(loop_guard, "last_tools_est", 0)
        try:
            loop_guard.last_tools_signature = sig
            loop_guard.last_tools_est = new_est
        except Exception:  # pragma: no cover —— 只读替身（防御，不影响估算本身）
            pass
        return delta

    if baseline_msg_count is not None and ctx_tokens > 0:
        delta = sum(
            _estimate_message_tokens(m, tokenizer.count)
            for m in request.messages[baseline_msg_count:]
        )
        delta += _track_tools()
        request.metadata[PROMPT_EST_BASE_KEY] = ctx_tokens
        request.metadata[PROMPT_EST_SEG_KEY] = delta
        return ctx_tokens + delta
    full = _estimate_request_tokens(request, tokenizer.count)
    _track_tools()
    request.metadata[PROMPT_EST_BASE_KEY] = 0
    request.metadata[PROMPT_EST_SEG_KEY] = full
    return max(full, ctx_tokens)


def apply_dynamic_max_tokens(ctx, request: "LLMRequest", resolved_model: ResolvedModel | None) -> None:
    """Allocate output using the same immutable resolved model as assembly/compact."""
    if resolved_model is None:
        return
    used = request.prompt_token_estimate
    hard_cap = resolved_model.output_ceiling or resolved_model.context_limit
    if used is None:
        if request.config.max_tokens is not None:
            request.config.max_tokens = min(request.config.max_tokens, hard_cap)
        return
    if used >= resolved_model.context_limit:
        from ctx_weft.core.models.errors import ContextOverflowError
        raise ContextOverflowError(
            "Prompt estimate leaves no space for model output",
            required=used + 1, effective_limit=resolved_model.effective_input_limit,
            context_limit=resolved_model.context_limit, reserved_output_tokens=resolved_model.reserved_output_tokens,
        )
    if used > resolved_model.effective_input_limit:
        logger.warning("prompt estimate %d exceeds effective window %d; tightening max_tokens",
                       used, resolved_model.effective_input_limit)
    if request.config.max_tokens is not None:
        request.config.max_tokens = min(request.config.max_tokens, hard_cap, resolved_model.context_limit - used)
        return
    policy = resolved_model.output_policy
    margin = max(policy.margin, int(policy.margin_ratio * used))
    soft_cap = max(1, int(policy.output_ratio * resolved_model.context_limit), policy.output_min)
    request.config.max_tokens = dynamic_max_tokens(
        resolved_model.context_limit, used, min(hard_cap, soft_cap), margin=margin, floor=max(1, policy.floor),
    )


def stream_llm(
    llm: "LLMClient", request: "LLMRequest", *, stream: bool = True,
    blob_store: "Any" = None, provider_ctx: "Any" = None,
) -> DelegatingLLMStream:
    return DelegatingLLMStream(lambda owner: _stream_llm(
        owner, llm, request, stream=stream, blob_store=blob_store, provider_ctx=provider_ctx))


async def _stream_llm(
    owner: DelegatingLLMStream,
    llm: "LLMClient", request: "LLMRequest", *, stream: bool = True,
    blob_store: "Any" = None, provider_ctx: "Any" = None,
) -> AsyncIterator["LLMChunk"]:
    """发送前合法化 ``request.messages``、把 blob ref 还原成 base64，再流式转发 chunk。

    ⚠️ **本函数不做任何模态门控**（spec 2026-08-28-multimodal-adapter-dispatch）：
    图片原样送到 ``LLMClient`` 面前，发多模态还是降级成占位由实现方决定。曾经这里
    有一个 ``_gate_tool_images``，只降 ``role == "tool"`` 的图——它的前提是「用户递的
    图在入口已被视觉门控拒掉」，入口门控删除后该前提不成立，覆盖面必须扩到所有角色，
    而那正是 adapter 的 ``_prepare_messages`` 在做的事。留在这里就是第二处会分叉的判据。

    已知代价：本函数对纯文本 adapter 一样会 rehydrate 每一个 blob ref（blob get +
    base64 编码，单图最大 5 MiB），rehydrate 完之后 adapter 的 ``_prepare_messages``
    才把它们降级成占位丢弃——这一趟读取白费了。这是「core 不判模态」换来的已知浪费；
    要消除它就得让本函数重新知道 adapter 的能力，那正是本设计明确拒绝重新引入的
    第二判据。不要在这里加判断。

    rehydrate 落在这里而非 adapter（架构裁定 T0）：adapter 的序列化链
    （``_build_payload`` / ``_serialize_messages`` / ``_parts_to_blocks``）全是同步
    函数，而 ``MemoryBlobStore.get`` 是 async。本函数是出网前最后一个 async 关口，一处
    覆盖三家 adapter。

    ``blob_store`` / ``provider_ctx`` 均**带默认值 None**：不传时整段 rehydrate 不
    执行，既有调用方与既有测试行为逐字节不变。刻意用显式参数而不是模块级单例——
    blob store 是 per-runtime 依赖，藏进全局状态会让测试互相污染。
    """
    request.messages = legalize_messages(request.messages)
    if blob_store is not None:
        request.messages = [
            dataclasses.replace(m, content=await rehydrate_content(
                m.content, blob_store=blob_store, ctx=provider_ctx))
            for m in request.messages
        ]
    source = owner.bind(llm.complete(request, stream=stream))
    try:
        async for chunk in source:
            yield chunk
    finally:
        await source.aclose()


# ── 自愈退避包装 ───────────────────────────────────────────────────────────────

# 默认值（ctx.config 缺省时回退；与 RuntimeConfig 默认一致）
_DEFAULT_MAX_ATTEMPTS = 8
_DEFAULT_MAX_DURATION_SEC = 300.0
_DEFAULT_BASE_DELAY_SEC = 2.0
_DEFAULT_MAX_INTERVAL_SEC = 60.0


def llm_request_id(state) -> str:
    """Identify a logical request across runs; call before emitting its events.

    The per-run sequence may restart at zero. Include the run identity so a
    resumed agent cannot inherit an interrupted request's presentation buffers.
    Inline retries intentionally retain the same logical request identity.
    """
    return f"req_{state.run_id}_{state.agent.id}_{state.sequence_counter}"


def _cfg_val(ctx, name: str, default: float) -> float:
    cfg = getattr(ctx, "config", None)
    return getattr(cfg, name, default) if cfg is not None else default


def _is_outage(e: BaseException) -> bool:
    return bool(getattr(e, "outage", False))


def _compute_delay(
    attempt: int,
    *,
    base: float,
    max_interval: float,
    retry_after: "float | None",
) -> float:
    """本地退避上限为 max_interval；服务端 Retry-After 不提前截短。"""
    if retry_after is not None and retry_after > 0:
        return retry_after
    return min(base * (2 ** (attempt - 1)), max_interval)


async def _sleep_cancellable(delay: float, tok) -> None:
    """Sleep up to *delay* seconds; if *tok* cancels first, raise CancelledError."""
    if tok is None:
        await asyncio.sleep(delay)
        return
    waiter = asyncio.ensure_future(tok.wait())
    try:
        await asyncio.wait({waiter}, timeout=delay)
    finally:
        if not waiter.done():
            waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
    tok.raise_if_cancelled()


async def _emit_retry(
    ctx,
    state,
    attempt: int,
    max_attempts: int,
    delay: float,
    error: BaseException,
    request_id: str | None = None,
) -> None:
    bus = getattr(ctx, "event_bus", None)
    if bus is None or state is None:
        return
    await bus.emit(make_event(state, EventType.LLM_RETRY_TRIGGERED, payload={
        "request_id": request_id,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "next_delay_sec": delay,
        "error_code": getattr(error, "status_code", 0),
        "error": str(error),
    }))


def prompt_event_inputs(prompt, messages, tool_refs=None):
    """Use the assembled data directly, adding this request's continuation."""
    inputs = getattr(prompt, "inputs", None)
    if inputs is None:
        return None
    return inputs.for_request(messages[len(prompt.messages):],
                              prompt.tool_refs if tool_refs is None else tool_refs).to_payload()


def select_request_tool_refs(inputs, ctx, state):
    """Select live tool references without compiling provider definitions."""
    cache = getattr(ctx, "capability_cache", None)
    if cache is None or inputs.purpose not in {"act", "observe"}:
        return None
    from ctx_weft.core.assembler.sources.capability import capability_ref

    return [capability_ref(cap) for cap in cache.available(state.agent.id, state.task.id)
            if cap.kind == "tool" and inputs.purpose in cap.purposes]


async def build_llm_request(ctx, state, inputs: PromptContext, *,
                            appended_messages: Sequence[LLMMessage] = (),
                            baseline_msg_count: int | None = None, turn: int = 0) -> LLMRequest:
    """Compile/reuse the fixed base and build this round's actual request.

    Continuations remain live LLMMessage objects, including provider reasoning
    state. Only the streaming gateway serializes their event representation.
    """
    from ctx_weft.protocols import LLMRequest

    tool_refs = select_request_tool_refs(inputs, ctx, state)
    prompt = await ctx.assembler.compile(
        inputs, template=state.extra.get("template"),
        token_counter=ctx.llm.tokenizer.count, reuse=True, trim=True, tool_refs=tool_refs)
    request = LLMRequest(
        model=resolve_llm_identity(state)[0], system=prompt.system,
        messages=[*prompt.messages, *appended_messages], tools=prompt.tools,
        metadata={"turn": turn},
        config=copy.deepcopy(state.resolved_model.config),
        tool_refs=tool_refs if tool_refs is not None else getattr(prompt, "tool_refs", []),
        capability_errors=list(getattr(prompt, "capability_errors", [])))
    request.prompt_token_estimate = request_prompt_estimate(
        ctx.llm.tokenizer, request,
        getattr(state.agent, "loop_guard", None) if inputs.purpose == "act" else None,
        baseline_msg_count if inputs.purpose == "act" else None)
    return request


async def externalize_prompt_images(inputs, *, event_blob_store, ctx):
    """Never introduce inline image bytes into a prompt event.

    Memory references stay references. Only unpersisted input content can still
    contain inline images; send those through the existing event blob boundary.
    """
    from ctx_weft.protocols.content import content_to_event_jsonable
    from ctx_weft.protocols.content import normalize_content_parts

    async def content(value):
        if not isinstance(value, list):
            return value
        result = []
        for part in value:
            if isinstance(part, dict) and part.get("type") == "image" and part.get("source_type", "base64") != "ref":
                parts = normalize_content_parts([part])
                if event_blob_store is not None and event_blob_store.can_externalize:
                    result.extend(await content_to_event_jsonable(
                        parts, event_blob_store=event_blob_store, ctx=ctx))
                else:
                    result.append({"type": "text", "text": redact_content_for_event(parts)})
            else:
                result.append(part)
        return result

    result = {**inputs, "task": dict(inputs["task"])}
    for key in ("user_prompt", "outputs"):
        if key in result["task"]:
            result["task"][key] = await content(result["task"][key])
    for group, key in (("appended_messages", "content"), ("references", "content")):
        result[group] = [{**item, key: await content(item[key])} if key in item else item
                         for item in inputs.get(group, [])]
    result["memory_blocks"] = [
        {**item, "record": {**item["record"], "content": await content(item["record"]["content"])}}
        if "record" in item else item for item in inputs["memory_blocks"]
    ]
    return result



def stream_llm_resilient(ctx, state, request, *, inputs: PromptContext | None = None,
                         appended_messages: Sequence[LLMMessage] = ()) -> DelegatingLLMStream:
    """Expose the current attempt's shared result without another collector."""
    appended_messages = tuple(appended_messages)
    return DelegatingLLMStream(
        lambda owner: _stream_llm_resilient(owner, ctx, state, request,
                                          inputs=inputs, appended_messages=appended_messages))


async def _stream_llm_resilient(owner, ctx, state, request, *, inputs=None,
                               appended_messages=()) -> AsyncIterator["LLMChunk"]:
    """退避自愈包装：把 retriable+outage 的 LLM 调用在进程内指数退避重试。

    - 中途 outage 先关闭旧流、废弃结果并发 RESPONSE_DISCARDED，再从空状态重试。
    - 所有重试共用 request_id；消费者按废弃事件清空临时输出，不能跨事件拼接。
    - 非 outage 的 retriable（如 _finalize 截断）与永久错 → 原样抛出（保留既有语义）。
    - max_attempts 包含首次调用；原生 adapter 每次只发送一个 HTTP 请求。
    - max_duration 包括首包前的请求与退避；首个 chunk 后不再限制总时长。
    - 首次中途断流另开 midstream_recovery 窗口，后续故障不刷新，次数仍共用。
    - 请求和退避期间均尊重 cancel_token；退出前关闭在途流。

    「流式侧」4 种 LLM_* 事件（REQUEST_STARTED / PROMPT_SENT / TOKEN_STREAMED /
    REASONING_STREAMED）在此发射（task-4 收敛，spec 2026-09-03 §9.5），对所有调用方
    无条件发射——不看 ``state.origin``。

    Task 4 曾在这里挂过一道临时门禁（``_STREAM_EVENT_ORIGINS``，只放行
    ``LOOP_ACT``/``LOOP_COMPACT``），因为当时 observe.py 的 ``run_observe_react``
    还在自己发一套同形事件（前台 LLM_*、后台 BACKGROUND_OBSERVE_*），放行会重复发射/
    串号。Task 5 删掉了 observe.py 那套自发射，改成所有调用方（act / compact /
    observe / background_observe）统一由本函数发 LLM_*，只靠 ``state.origin``
    （``EventOrigin.LOOP_OBSERVE`` / ``LOOP_BACKGROUND_OBSERVE`` 等）区分前台/后台
    （docs/events-v2.md §3.6：V2 之前这是靠 BackgroundObserve* 那族独立类型做的，
    合并后由 origin 承担）——门禁因此整个删除。

    ``LLM_RESPONSE_FINISHED`` 各调用方自己发射（依赖各自的收尾逻辑：act 依赖软打断
    决策，见 task-4 brief；compact 是单次调用直出；observe/background_observe 共用
    ``ObserveStep._run_react``，Task 5 起也在那里补发）。

    request_id：与 act.py 侧 ``_run_llm_turn`` 用同一个确定性公式
    ``llm_request_id(state)`` 独立算出——两边都在「本次 LLM
    调用的任何事件被发射之前」求值（act.py 在调用本函数之前；本函数在 while 重试
    循环、也就是第一次 emit 之前），因此 ``state.sequence_counter`` 两处读到的是
    同一个值，算出来天然相等，不需要新增参数或跨函数传值。

    Output allocation follows REQUEST_STARTED, so allocation errors retain the
    started/finished lifecycle contract. PROMPT_SENT follows allocation and records
    the request-local config handed to the adapter; saved agent/run settings are
    never changed. Retries copy this bounded request, not an adapter's mutations.
    """
    from ctx_weft.protocols import LLMCallError  # local to avoid re-export confusion

    bus = getattr(ctx, "event_bus", None)
    emit_stream_events = bus is not None and state is not None
    request_id: str | None = None
    if emit_stream_events:
        request_id = llm_request_id(state)
        model, llm_account = resolve_llm_identity(state)
        turn = request.metadata.get("turn", 0)
        await bus.emit(make_event(state, EventType.LLM_REQUEST_STARTED, payload={
            "request_id": request_id, "model": model, "llm_account": llm_account,
            "turn": turn}))
        owner.request_started = True
        prompt_payload = {
            "request_id": request_id, "turn": turn,
            "tool_names": [t.name for t in request.tools],
        }
        event_inputs = (inputs.for_request(appended_messages, request.tool_refs or []).to_payload()
                        if inputs is not None else request.prompt_inputs)
        if event_inputs is not None:
            event_inputs = await externalize_prompt_images(
                event_inputs, event_blob_store=getattr(ctx, "event_blob_store", None),
                ctx=ctx.provider_ctx)
            prompt_payload.update(event_inputs)
        else:
            # Custom callers without an assembler still use the legacy event.
            prompt_payload.update(system=request.system, messages=[
                {"role": m.role, "content": redact_content_for_event(m.content)}
                for m in request.messages
            ])

    # Direct gateway callers have no run state; snapshot their configured client once.
    resolved_model = (state.resolved_model if state is not None else
                      ResolvedModel.from_client(ctx.llm) if request.prompt_token_estimate is not None else None)
    apply_dynamic_max_tokens(ctx, request, resolved_model)
    if emit_stream_events:
        from ctx_weft.protocols.llm import serialize_call_config
        prompt_payload["llm_config"] = serialize_call_config(request.config)
        await bus.emit(make_event(state, EventType.LLM_PROMPT_SENT, payload=prompt_payload))

    max_attempts = max(1, int(_cfg_val(ctx, "llm_self_heal_max_attempts", _DEFAULT_MAX_ATTEMPTS)))
    max_duration = _cfg_val(ctx, "llm_self_heal_max_duration_sec", _DEFAULT_MAX_DURATION_SEC)
    base = _cfg_val(ctx, "llm_self_heal_base_delay_sec", _DEFAULT_BASE_DELAY_SEC)
    max_interval = _cfg_val(ctx, "llm_self_heal_max_interval_sec", _DEFAULT_MAX_INTERVAL_SEC)
    midstream_duration = _cfg_val(ctx, "llm_midstream_recovery_max_duration_sec", 300.0)

    deadline = monotonic() + max_duration
    midstream_deadline = None
    # Adapters/legalization may mutate a request. Every retry receives the same
    # original input, never a previous attempt's mutations or partial response.
    # Assembled inputs belong to the event; adapters/retries need no copy of them.
    original_request = copy.deepcopy(dataclasses.replace(
        request, prompt_inputs=None, tool_refs=None, capability_errors=[]))
    tok = getattr(ctx, "cancel_token", None)
    attempt = 0
    last_error = None
    while True:
        if tok is not None:
            tok.raise_if_cancelled()
        if monotonic() >= deadline:
            logger.warning("LLM recovery deadline exceeded after %d attempt(s)", attempt)
            raise LLMOutageError(
                str(last_error) if last_error is not None else "LLM recovery deadline exceeded",
                status_code=getattr(last_error, "status_code", 0),
            ) from last_error
        attempt += 1
        yielded_anything = False
        source = owner.bind(stream_llm(
            ctx.llm, copy.deepcopy(original_request),
            blob_store=getattr(ctx, "blob_store", None),
            provider_ctx=getattr(ctx, "provider_ctx", None),
        ))
        chunks = stream_until_cancel(source, tok)
        first_timeout = None
        try:
            try:
                while True:
                    try:
                        if yielded_anything:
                            chunk = await anext(chunks)
                        else:
                            async with asyncio.timeout(max(0.0, deadline - monotonic())) as first_timeout:
                                chunk = await anext(chunks)
                    except StopAsyncIteration:
                        break
                    if emit_stream_events:
                        if chunk.kind == "token":
                            await bus.emit(make_event(
                                state, EventType.LLM_TOKEN_STREAMED,
                                payload={"request_id": request_id, "delta": chunk.text}))
                        elif chunk.kind == "reasoning":
                            await bus.emit(make_event(
                                state, EventType.LLM_REASONING_STREAMED,
                                payload={"request_id": request_id, "delta": chunk.text}))
                    yielded_anything = True
                    yield chunk
            finally:
                try:
                    await chunks.aclose()
                finally:
                    await source.aclose()
            return  # success
        except TimeoutError as e:
            if first_timeout is None or not first_timeout.expired():
                raise
            logger.warning("LLM recovery deadline exceeded after %d attempt(s)", attempt)
            raise LLMOutageError(
                str(last_error) if last_error is not None else "LLM recovery deadline exceeded",
                status_code=getattr(last_error, "status_code", 0),
            ) from e
        except LLMCallError as e:
            if not getattr(e, "retriable", True) or not _is_outage(e):
                raise  # 永久错 / 非 outage retriable → 原样抛出
            if tok is not None:
                tok.raise_if_cancelled()
            last_error = e
            if yielded_anything and midstream_deadline is None:
                midstream_deadline = monotonic() + midstream_duration
                deadline = midstream_deadline
            remaining = deadline - monotonic()
            will_retry = attempt < max_attempts and remaining > 0
            if yielded_anything:
                usage = source.partial_result().usage
                session = getattr(state, "session", None)
                if session is not None and hasattr(session, "token_used"):
                    session.token_used += usage.prompt_tokens + usage.completion_tokens
                # The finally above has closed every underlying stream before
                # we revoke its output. Detach even if no further retry is allowed.
                owner.discard_result()
                if emit_stream_events:
                    await bus.emit(make_event(state, EventType.LLM_RESPONSE_DISCARDED, payload={
                        "request_id": request_id, "reason": InterruptReason.LLM_OUTAGE,
                        "error": str(e), "error_code": getattr(e, "status_code", 0),
                        "will_retry": will_retry, "usage": dataclasses.asdict(usage),
                        "usage_known": usage.total_tokens > 0 or usage.prompt_tokens > 0
                                       or usage.completion_tokens > 0,
                        "llm_model": model, "llm_account": llm_account,
                    }))
            if not will_retry:
                logger.warning("LLM retry budget exhausted after %d attempt(s): %s", attempt, e)
                raise LLMOutageError(
                    str(e),
                    status_code=getattr(e, "status_code", 0),
                ) from e
            delay = _compute_delay(
                attempt,
                base=base,
                max_interval=max_interval,
                retry_after=getattr(e, "retry_after_sec", None),
            )
            delay += random.uniform(0, delay * 0.1)  # jitter
            delay = min(delay, remaining)
            await _emit_retry(ctx, state, attempt, max_attempts, delay, e, request_id)
            await _sleep_cancellable(delay, tok)
