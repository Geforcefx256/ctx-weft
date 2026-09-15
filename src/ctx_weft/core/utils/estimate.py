"""token 估算：文本费率、窗口换算、消息与工具调用的估算。

**为什么留在顶层而不是下放**：消费者横跨 `assembler`（预算）、`loop`（动态
max_tokens / prepare）、`models.errors`（溢出文案）与 `providers/llm`（tokenizer、
输出预留）。放进其中任何一个包，都会逼另外三个反向引它。这一簇和 `core/models/`
一样，是真正的共享词表，不是某个包的私有工具。

图片的计量（`image_tokens` / `image_part_count` / 两个常量）不在这里，在
`core/content.py`——它们是对 content 结构的检查，与文本费率无关。本模块的
`estimate_content_tokens` 从那边引。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ctx_weft.core.utils.content import content_to_text, image_tokens

if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart


_CJK_RE = re.compile(
    "["
    "　-〿"      # CJK 标点
    "぀-ヿ"      # 平假名 + 片假名
    "㐀-䶿"      # CJK 扩展 A
    "一-鿿"      # CJK 统一表意
    "가-힯"      # 谚文音节
    "豈-﫿"      # CJK 兼容表意
    "＀-￯"      # 全角/半角形式
    "\U00020000-\U0002fa1f"  # CJK 扩展 B–F + 兼容补充
    "]"
)


# 高熵 ASCII 长串：URL 段/UUID/哈希/hex/base64 等"随机字符"实测（cl100k/o200k）约 0.51~0.54
# token/字符。费率取 0.6 而非 0.5：保证冷启动单边高估（校准系数常态 <1、只向下回收窗口浪费，
# 不向上追赶低估缺口——追赶的过渡期就是 400 风险窗口）。≥20 连续才算：正常英文单词/短标识符
# 达不到，散文不受影响；超长 snake_case 标识符会被略高估（方向安全）。
_DENSE_ASCII_RE = re.compile(r"[A-Za-z0-9+/=_-]{20,}")

# 非 ASCII 字符（CJK 之外落此桶：emoji/西里尔/阿拉伯文/组合符等）：真实 1~3 token/字，
# len/3 严重低估（emoji 混排实测 est/real=0.96）。按 1/字计，方向安全。
_NON_ASCII_RE = re.compile(r"[^\x00-\x7f]")


def estimate_tokens(text: str) -> int:
    """Token 粗估：按脚本/熵分四段、**保证单边高估**（估算只许偏高，低估会触发 provider 400；
    偏高的浪费由 tokenizer 校准系数向下回收——系数常态 <1，收敛前的误差方向恒安全）。

    费率：CJK 表意字/假名/谚文每字 ``ceil(1.5*n)``（真实约 0.6~1.3）；高熵 ASCII 长串
    （≥20 连续 [A-Za-z0-9+/=_-]，URL 段/哈希/base64）``ceil(0.6*n)``（真实约 0.51~0.54）；
    其余非 ASCII（emoji/西里尔等）每字 1（真实约 0.5~2，混排下 ≥1.1x）；ASCII 散文/标点
    ``ceil(len/3)``（真实约 0.25~0.33）。四段相加、非空至少 1。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    dense = sum(len(m) for m in _DENSE_ASCII_RE.findall(text))
    non_ascii_other = len(_NON_ASCII_RE.findall(text)) - cjk
    ascii_other = len(text) - cjk - dense - non_ascii_other
    return max(
        1,
        (3 * cjk + 1) // 2
        + (3 * dense + 4) // 5
        + non_ascii_other
        + (ascii_other + 2) // 3,
    )


def effective_limit(context_limit: int, reserved_output_tokens: int) -> int:
    """装配/压缩预算的有效上限：为 LLM 输出预留余量后的可用输入窗口。"""
    return max(0, context_limit - max(0, reserved_output_tokens))


def default_output_reserve(context_limit: int) -> int:
    """未显式配置时的输出预留默认 = max(context_limit // 16, 4096)。

    按窗口尺寸取（固定 6.25% 比例）而非固定值：小上下文模型不至被固定 8192 吞光
    （effective_limit 归零），大模型自动放大；固定比例 → compact 触发点在各模型上统一
    （约 75% 原始窗口）。下限 4096 与输出软顶 output_min 对齐。
    """
    return max(context_limit // 16, 4096)


def dynamic_max_tokens(
    context_limit: int,
    used: int,
    ceiling: int,
    *,
    margin: int = 8192,
    floor: int = 1024,
) -> int:
    """按当前窗口占用实时算请求 max_tokens：clamp(context_limit − used − margin, floor, ceiling)。

    used = caller 估算的本请求真实 prompt token（真实基线 + 本轮增量，见
    gateway.request_prompt_estimate）——纯算术在此，不做估算。ceiling 默认由调用方传
    context_limit（剩余窗口全给输出）；配小则作收紧上限（如 Anthropic 硬输出上限）。
    """
    remaining = context_limit - used - margin
    return max(floor, min(ceiling, remaining))


# 一条消息/记录里文本 content 之外的计费补偿项（都往大了取，堵低估致 400 的洞）。
# 供 gateway（LLMMessage）与 prepare/composer（memory 记录 / 装配消息）共用，单一真源。
_MSG_FRAMING_TOKENS = 4       # 每条消息的角色/分隔 framing 开销（provider 计费、文本之外）


def _dumps_for_estimate(obj: Any) -> str:
    """估算用的 JSON 序列化：非 str 一律 dumps，失败退 str()。"""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def estimate_content_tokens(content: "str | list[ContentPart] | None", *, count: Callable[[str], int] | None = None) -> int:
    """一条 content 的估算：文本 + 图片 part 固定常数 + 每条 framing 开销。往大了估。

    None/空同样容错（与 content_to_text 对齐）：契约上 content 应为 str | list，但个别路径
    （旧/导入的 memory 记录、None 工具结果等）可能透传 None，估算期须容错而非迭代 None 崩溃。

    count：文本费率经 count 回调走 tokenizer；None 回退未校准启发式（纯单测/无 llm 场景）。
    framing 常数不过回调。

    注意：本函数带 _MSG_FRAMING_TOKENS 补偿，**不是**纯文本恒等的。只在本就计入
    framing 的路径（prepare / llm_gateway）使用；装配与 compact 的估算点请改用
    「既有文本计数 + image_tokens(content)」，见 spec 2026-08-20-multimodal-design §6.5。
    """
    count = count or estimate_tokens
    return _MSG_FRAMING_TOKENS + count(content_to_text(content)) + image_tokens(content)


def estimate_tool_calls_tokens(tool_calls: "list[dict] | None", *, count: Callable[[str], int] | None = None) -> int:
    """tool_calls（[{name, arguments|input}]）的估算：名字 + 参数 JSON。往大了估。

    纯工具回合 content 常为空、体量全在 arguments 里——不数就会严重低估（致 400 / compact 欠触发）。

    count：文本费率经 count 回调走 tokenizer；None 回退未校准启发式（纯单测/无 llm 场景）。
    """
    count = count or estimate_tokens
    total = 0
    for tc in tool_calls or []:
        total += count(str(tc.get("name", "")))
        total += count(_dumps_for_estimate(tc.get("arguments", tc.get("input", {}))))
    return total


# ── 工具面（API tools 参数）费率与指纹（spec: tool-schema-budget）────────────


def _tool_field(t: Any, name: str) -> Any:
    """LLMTool（dataclass）与 dict 形态统一取字段。"""
    v = getattr(t, name, None)
    if v is None and isinstance(t, dict):
        v = t.get(name)
    return v


def _canonical_tools_dump(tools: "list[Any] | None") -> list[str]:
    """规范化工具定义转储（与估算/指纹同源）：按 name 排序，每工具
    name + description + input_schema（**key 排序** JSON——指纹对键序鲁棒，估算共用
    同一份转储）。指纹必须覆盖与估算完全相同的输入——仅哈希名称集合不足以感知同名
    工具的定义更新（schema 变大 = 成本变化）。"""
    if not tools:
        return []
    parts: list[str] = []
    for t in sorted(tools, key=lambda x: str(_tool_field(x, "name") or "")):
        parts.append(str(_tool_field(t, "name") or ""))
        parts.append(str(_tool_field(t, "description") or ""))
        schema = _tool_field(t, "input_schema") or {}
        try:
            parts.append(json.dumps(schema, ensure_ascii=False, sort_keys=True))
        except Exception:  # pragma: no cover —— 防御（不可序列化 schema）
            parts.append(str(schema))
    return parts


def estimate_tools_tokens(
    tools: "list[Any] | None", count: Callable[[str], int] | None = None,
) -> int:
    """工具面 token 估算（name + description + schema JSON）——单一真源。

    取代 prepare._estimate_assembled_tokens 与 llm_gateway._estimate_request_tokens
    里两份重复的内联公式（口径逐值一致）。"""
    count = count or estimate_tokens
    return sum(count(p) for p in _canonical_tools_dump(tools))


def tools_signature(tools: "list[Any] | None") -> str:
    """工具面指纹：规范化定义转储（name+description+schema，见 _canonical_tools_dump）
    的稳定哈希。空面 = 空串。工具面不变时与上次哈希比较即跳过估算增量。"""
    import hashlib

    dump = _canonical_tools_dump(tools)
    if not dump:
        return ""
    return hashlib.sha256("\x1f".join(dump).encode("utf-8", "replace")).hexdigest()[:16]
