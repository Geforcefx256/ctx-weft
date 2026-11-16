"""Engine content budgets, prompt rendering and tool-result policies.

Shared content types and transformations live in protocols.content.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ctx_weft.core.models.errors import InvalidContentError
from ctx_weft.protocols.content import (
    BLOB_REF_PREFIX,
    _is_text_part,
    extract_blob_refs,
    normalize_content,
    normalize_content_parts,
    validate_content,
)
from ctx_weft.protocols.content import content_from_jsonable as content_from_jsonable
from ctx_weft.protocols.content import content_has_image as content_has_image
from ctx_weft.protocols.content import content_to_event_jsonable as content_to_event_jsonable
from ctx_weft.protocols.content import content_to_jsonable as content_to_jsonable
from ctx_weft.protocols.content import content_to_text as content_to_text
from ctx_weft.protocols.content import content_with_prefix as content_with_prefix
from ctx_weft.protocols.content import content_with_suffix as content_with_suffix
from ctx_weft.protocols.content import downgrade_images_to_text as downgrade_images_to_text
from ctx_weft.protocols.content import hydrate_event_content as hydrate_event_content
from ctx_weft.protocols.content import rehydrate_content as rehydrate_content

if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart

logger = logging.getLogger(__name__)

__all__ = [
    "content_to_text",
    "content_with_prefix",
    "content_with_suffix",
    "content_to_jsonable",
    "content_to_event_jsonable",
    "content_from_jsonable",
    "normalize_content_parts",
    "redact_content_for_event",
    "validate_content",
    "content_has_image",
    "normalize_content",
    "rehydrate_content",
    "hydrate_event_content",
    "downgrade_images_to_text",
    "split_for_tool_result",
    "legalize_tool_result_parts",
    "extract_blob_refs",
    "collect_blob_refs",
]


# 一条消息/记录里文本 content 之外的计费补偿项（都往大了取，堵低估致 400 的洞）。
# 供 gateway（LLMMessage）与 prepare/composer（memory 记录 / 装配消息）共用，单一真源。
_MSG_FRAMING_TOKENS = 4  # 每条消息的角色/分隔 framing 开销（provider 计费、文本之外）
# 每张图的**地板**。历史上 image_tokens 就等于「张数 × 本常数」，Phase 3c Task D 起
# 它退化成下限：小于 _IMAGE_PART_TOKENS * _IMAGE_BYTES_PER_TOKEN（= 200 KiB）的图仍按
# 它计，更大的图按字节折算。保留地板的两个理由：① 模型侧对任意一张图的固定开销本就在
# 这个量级，往下折算会低估；② 体积未知（存量记录 / ref 且无 byte_size）时的回落值，
# 保证存量数据与不接 MemoryBlobStore 的宿主行为不劣化。
_IMAGE_PART_TOKENS = 1600

# ⚠️ 本系数建模的是**字节压力，不是计费 token**。provider 侧会把图降采样，单图真实计费
# 大约就封顶在 1600 附近——拿本口径去算成本是误用。但预算机制的职责是「判断这个请求能不能
# 发出去」，而那由**字节**决定（请求体超上限会被 provider 直接拒），所以估算必须建模字节。
#
# 标定（Phase 3c Task D，依据全部取自仓内实际取值）：
#   · Anthropic 单次请求体上限约 32 MB；base64 膨胀 4/3 → 可容纳原始字节约 24 MiB。
#   · 仓内典型窗口 context_limit = 180_000（core/state/models.py、core/control/types.py、
#     core/control/reducers.py 三处默认值），reserved_output_tokens = 8_192
#     → effective_limit = 171_808。
#   · 令「预算耗尽」与「请求体触顶」对齐：25_165_824 B / 171_808 tok ≈ 146.5 B/tok。
#   · 向下取到 2 的幂 128：取整方向使估算**偏高**（与 estimate_tokens 的「保证单边高估」
#     同向——低估会触发 provider 400，高估只浪费窗口），且 128 = 移位，纯整数运算。
# 校验（compact_token_ratio 默认 0.8，protocols/template.py:79）：
#   · 单张满额图（core/content.py 的 _MAX_IMAGE_BYTES = 5 MiB）= 40_960 tok。
#   · 4 张 = 163_840 tok = 95% of 171_808 → 早已越过 0.8 触发比，compact 必然先行介入。
#   · 5 张 = 204_800 tok > 171_808 → 地板仍超限时 budget.py 抛 ContextOverflowError。
#   · 满预算 171_808 tok = 21 MiB 原始字节 ≈ 28 MB base64，对 32 MB 硬顶留约 13% 余量。
# 即「若干张满额图即超出典型预算」，而旧口径下 5 张满额图账面才 8_000 tok（不到 5%）。
_IMAGE_BYTES_PER_TOKEN = 128


def image_part_count(content: "str | list[ContentPart] | None") -> int:
    """content 中图片（非文本 part）的个数。

    对 str / None / 空一律返回 0，与 image_tokens 同口径——两者共用同一判据，
    确保「token 补偿」与「图片数」永不互相矛盾（budget.py 报错文案据此报数，
    不得靠对 image_tokens 整除反推）。
    """
    if not content or isinstance(content, str):
        return 0
    return sum(1 for p in content if not hasattr(p, "text"))


def image_byte_size(part: Any) -> int | None:
    """一个图片 part 的原始（解码后）字节数；无从得知时返回 None。

    两条来源，按可靠度排序：
    1. ``byte_size`` 字段——由 ``core.utils.content`` 在**还有字节**的时候填上
       （validate 解码 inline base64 / normalize 外部化拿到 raw bytes），
       并经 ``content_to_jsonable`` 往返持久化。ref 形态只有这一条路。
    2. inline base64 的载荷长度反解：``len(data) * 3 // 4`` 减去 padding。
       刻意**不解码**——``image_tokens`` 在装配/压缩热路径上被逐条调用，
       为估算去 b64decode 一张 5 MiB 的图是不可接受的开销。

    两条都够不着（ref / url 且无 ``byte_size``，即存量记录与不接 MemoryBlobStore 的
    宿主）→ None，由调用方回落到 ``_IMAGE_PART_TOKENS``。
    """
    size = getattr(part, "byte_size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return size
    if getattr(part, "source_type", "base64") != "base64":
        return None  # ref / url：data 不是载荷，长度无意义
    data = getattr(part, "data", "") or ""
    if not isinstance(data, str) or not data:
        return None
    pad = 2 if data.endswith("==") else (1 if data.endswith("=") else 0)
    return max(0, (len(data) * 3) // 4 - pad)


def image_tokens(content: "str | list[ContentPart] | None") -> int:
    """content 中图片（非文本 part）的 token 补偿。不含文本、不含 framing。

    对 str / None / 空一律返回 0——这保证调用方在纯文本路径上是恒等变换，
    可以安全地加在既有的文本计数之后而不改变既有口径。**这条不变量不可破**：
    所有文本口径都建立在它之上。

    单张图 = ``max(_IMAGE_PART_TOKENS, 字节数 // _IMAGE_BYTES_PER_TOKEN)``，
    字节数未知时取 ``_IMAGE_PART_TOKENS``（等价于旧口径，故存量数据与不接
    MemoryBlobStore 的宿主行为不劣化）。

    **为什么按字节**（Phase 3c Task D，用户裁定 D2）：原口径是 ``1600 × 张数``，
    对唯一真正变化的维度——体积——毫无反应。5 MiB 截图与 50 KiB 缩略图同价，于是
    唯一能阻止请求体无限膨胀的机制（token 预算）对真实失败模式完全失明：几张大截图
    账面才几千 token（远不触发 compact），实际请求体已 30 MB+ 被 provider 拒，
    而每次 act 都重发全部历史图 → 会话永久卡死且无法自愈。按字节估算之后，
    「预算 → compact → fold → 图片离开视图 → 请求缩小」这条既有链自己就闭合了。

    ⚠️ **本函数建模的是字节压力，不是计费 token**——provider 会降采样，单图真实计费
    大约封顶在 1600。别拿它算成本，见 ``_IMAGE_BYTES_PER_TOKEN`` 的标定说明。

    判据与 ``image_part_count`` 同为 ``not hasattr(p, "text")``（spec §13 冻结），
    两者对「哪些 part 是图」永不互相矛盾；但**数值上不再是 1600 的整数倍**，
    ``budget.py`` 报图片数须继续走 ``image_part_count``，不得对本函数整除反推。

    刻意不接受 count 回调：图片不过 tokenizer。
    """
    if not content or isinstance(content, str):
        return 0
    total = 0
    for part in content:
        if hasattr(part, "text"):
            continue
        size = image_byte_size(part)
        if size is None:
            total += _IMAGE_PART_TOKENS
        else:
            total += max(_IMAGE_PART_TOKENS, size // _IMAGE_BYTES_PER_TOKEN)
    return total


# ── JSON 往返（事件 payload / 投影快照）─────────────────────────────────────


# ── 边界归一（Phase 3c Task E / E2）─────────────────────────────────────────


# ── 事件脱敏 ───────────────────────────────────────────────────────────────

_REDACT_DATA_PREVIEW = 12


def redact_content_for_event(content: "str | list[ContentPart] | None") -> str:
    """把内容渲染成适合进事件 payload 的字符串。**字节绝不上事件流**（spec §6.8）：
    一张图几万字符，直接进事件库会把它撑爆。

    **已经外部化成 ref 的图渲染成 L0.5 占位**（`media.refs.encode_image_placeholder`，
    全仓唯一真源）——那是一段**含完整 ref** 的文本，模型读到它可以
    `media:get_image(...)` 把图取回来。这一点是崩溃恢复的承重件：`CapabilityFinished`
    的 `result` 是「结果事件已落盘、memory 记录还没写」那个窗口里唯一幸存的一份，
    补写只能照它重建。本函数从前在这里自制了一个短标记（只留 ref 前 12 个字符），
    取不回来——同一个仓库里两套图片标记，而事件这边用的是残缺的那套。

    仍留短标记的只有**内联 base64**（宿主没接 memory blob store，压根没有 ref 可写）：
    没有可取回的东西，占位也编不出来。

    纯文本（`str`）**原样返回同一对象**，零开销、对无图会话逐字节无影响。
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    from ctx_weft.core.media.refs import encode_image_placeholder

    parts: list[str] = []
    for part in content:
        if _is_text_part(part):
            parts.append(part.text)
            continue
        media_type = getattr(part, "media_type", "?")
        refs = extract_blob_refs([part])
        if refs:
            try:
                parts.append(encode_image_placeholder(refs[0], media_type))
                continue
            except ValueError:  # ref 不是可解析 token：退回短标记，别产出坏占位
                logger.warning("redact_content_for_event: unusable blob ref %r", refs[0])
        data = getattr(part, "data", "") or ""
        src = getattr(part, "source_type", "base64")
        parts.append(f"[image {media_type} {src}:{data[:_REDACT_DATA_PREVIEW]}…]")
    return "".join(parts)


# ── 入口校验 ───────────────────────────────────────────────────────────────

# Anthropic 与 OpenAI 都接受的交集。扩这个集合前请先确认两家都支持。


# ── 入口外部化（Phase 3b）─────────────────────────────────────────────────


# ── 出网 rehydrate（Phase 3b）───────────────────────────────────────────────

# 本仓所有图片占位的清单见 `core/media/refs.py` 模块 docstring（L6 收口，裁定 R1）。
# 这一条单向渲染、永不回读，故留在归一层。


def collect_blob_refs(event: Any) -> list[str]:
    """GC 的 **mark 判据**：一个 MemoryEvent 引用了哪些 blob。provider 建引用边只走这里。

    两个来源的并集，都**只看结构化字段**：

    1. ``content`` 里的 ref part（``extract_blob_refs``，与 ``rehydrate_content``
       会去 get 的那批逐一对应）；
    2. ``event.blob_refs`` 的显式声明——L0.5 降级把 ``ImagePart(ref)`` 换成文本占位后，
       ref 只能靠这条传递（缺陷 2026-08-27）。

    **绝不解析占位文案。** 占位格式的唯一真源是 ``core/media/refs.py``；让 mark 判据
    去解析它，等于把文案格式变成 GC 正确性的一部分——文案一改，图就开始被误删，而且
    要到一个宽限期之后才看得出来。声明式采集把这个耦合彻底切断。

    本函数内部去重，返回时 content 里的在前、声明的在后。但**这不是可依赖的顺序
    承诺**：下游只按集合语义使用返回值（建引用边、并集判重），SQL provider 读侧
    的 `_declared_refs` 相减查询没有 `ORDER BY`，往返顺序由 SQL 行序决定——双重
    降级后可能读回 `[B, A]` 而非写入时的 `[A, B]`。若未来需要顺序稳定，应在
    `_declared_refs` 补 `ORDER BY`，而不是假设这里的返回顺序会被保留下去。
    """
    seen: dict[str, None] = {}
    for ref in extract_blob_refs(getattr(event, "content", None)):
        seen.setdefault(ref, None)
    for ref in getattr(event, "blob_refs", None) or ():
        text = str(ref or "")
        if text.startswith(BLOB_REF_PREFIX):
            seen.setdefault(text, None)
    return list(seen)


# ── per-purpose 降级（Phase 3b Task 4）──────────────────────────────────────

# 本仓所有图片占位的清单见 `core/media/refs.py` 模块 docstring（L6 收口，裁定 R1）。
# ⚠️ 这是 **per-purpose 降级**（不落库、只影响本次 prompt），与 L0.5 降级
# （`core/media` 重写 memory 记录、占位含 ref 可被 `get_image` 取回）**并存、
# 互不替代**，不要合并。单向渲染、永不回读。


def split_for_tool_result(
    content: "str | list[ContentPart] | None",
) -> "tuple[str, list[ContentPart]]":
    """拆成 ``(文本, 非文本 part)``。

    **不是 provider 要调的东西**——provider 把混合内容整个放进
    `CapabilityEvent(kind="result")` 的 ``payload["content"]`` 即可（那是
    ``str | list[ContentPart]``，与三个执行入口同一个联合类型）。本函数是 **gateway
    自己**的拆分器：文本还要过 spill 截断、还要在前面接 ``[Human note: …]``，两件事
    都只作用于文本，所以流一进来就拆、末尾再拼回去（`capability_gateway._ToolStream`）。

    ``str`` 进 → 返回**同一个对象**与空列表；``None`` / 空列表 → ``("", [])``。
    纯文本路径因此零开销、逐字节不变。

    非文本判据 ``not hasattr(p, "text")`` 与 ``utils.content_to_text`` /
    ``utils.image_part_count`` 同源（spec 2026-08-20 §13 冻结），不在此另写一份。
    **dict 形态先过 `normalize_content_parts`** 再进来——那个冻结判据对 dict 是瞎的，
    会把 ``{"type":"text",...}`` 判成非文本 part。
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    text = "".join(p.text for p in content if hasattr(p, "text"))
    parts = [p for p in content if not hasattr(p, "text")]
    return text, parts


# 本仓所有图片占位的清单在 `core/media/refs.py` 的模块 docstring（L6 收口，裁定 R1）。
# 这一条与 `_IMAGE_PLACEHOLDER_TMPL` / `_IMAGE_UNAVAILABLE_TMPL` 同类：单向渲染、永不回读。
_IMAGE_DROPPED_TMPL = "[image dropped: {reason}]"


async def legalize_tool_result_parts(
    parts: "list[ContentPart] | None",
    *,
    blob_store: "Any" = None,
    ctx: "Any" = None,
) -> "list[ContentPart]":
    """provider 在 result ``content`` 里交上来的非文本 part 的**合法化**——工具返图
    这条路在入口三件套（validate → 外部化）之外，这里是它唯一的补课点。

    三个入口（`start_session` / `run_single_task` / `send_message`）与 HITL 应答都走
    `runtime._validate_and_normalize_content`，字节在**进系统之前**就已校验并外部化。
    工具结果绕开了那条路：`_record_result` 只管 ingest。在 `media:get_image` 是唯一
    生产者的年代这没问题——它交出来的本就是 ``source_type="ref"``（字节早在 blob store
    里）。一旦第三方 provider（MCP 的 ``ImageContent``、将来的截图工具）开始交
    **inline base64**，缺的这一课就变成三个洞：白名单/尺寸上限一次都不跑、裸 base64
    直接落进 memory 记录、宿主注册了 blob store 也用不上。

    调用方只有 `capability_gateway.invoke` 一处（`_ToolStream.parts` 拆好之后）。

    ════════════════════════════════════════════════════════════════════════
    三条规则
    ════════════════════════════════════════════════════════════════════════

    **只管 ``source_type == "base64"`` 的图。** 文本 part 与 ``ref`` / ``url`` 形态
    原样透传：ref 形态只可能由仓内产出（`media:get_image` 从占位解出来的那份），字节
    进 blob store 之前已经过过一次 `validate_content`，在这里重跑白名单只会**误伤**
    ——占位里的 ``media_type`` 允许回落成 ``"image"``（`media/refs.py` 的
    ``_UNKNOWN_MEDIA_TYPE``），而那个值不在 `ALLOWED_IMAGE_MEDIA_TYPES` 里。

    **恒不抛。** 这条路在工具调用循环上，与 `media:get_image`「一律不抛」同一取向
    （见 `core/media/capability.py` 模块 docstring）：一张图不合格不该掀掉整次工具
    调用，更不该掀掉整个 loop。不合格的那张**换成确定性文本占位**，其余 part 照走。
    换占位而不是静默丢弃——模型得知道「这里本来有张图，没能给你」，否则它会以为工具
    什么都没返回（这正是改造前 MCP 那条路的病：`ImageContent` 连痕迹都不留）。

    **占位逐字节确定**（与 `downgrade_images_to_text` 同一硬约束）：``reason`` 只由
    ``media_type`` 与失败类别拼出，不含 sha / 随机 id / 时间戳——工具结果处在 prompt
    前缀里，每次不同会砸掉其后整段自动前缀缓存。

    ════════════════════════════════════════════════════════════════════════
    参数与失败姿态
    ════════════════════════════════════════════════════════════════════════

    ``blob_store`` = memory 侧 blob store（``None`` 或 ``can_externalize`` 为假 →
    **不外部化，inline 原样保留**）。这与「宿主没接 blob」的既定口径一致：不接 blob
    的会话本来就是 inline base64 跑（多模态设计 Phase 2 的形态），不因为多了这道
    合法化就把图丢掉。

    ``put`` 真的抛了则是另一回事——宿主**接了**store、意图明确是「字节不进记录行」，
    那就不能偷偷退回 inline（一张图几 MB，直接进 memory 行 / SQL 列）。此时降级成占位
    并 ``logger.error``，姿态与 `runtime._restore_task_prompts` / `_cold_hitl_decision`
    一致：绝不让不该流下去的字节流下去，但也绝不静默。

    **这里刻意不传 ``event_blob_store``**（即不跑 `validate_content` 的第二道门）：
    那道门管的是「携图内容进事件流」，而工具结果进事件 payload 走
    `redact_content_for_event`——恒渲染成短标记，字节根本不上事件流。在这里跑那道门
    会让「宿主没接 EventBlobStore」变成「工具永远返不了图」，与事实不符。
    """
    normalized = normalize_content_parts(parts) if parts else None
    # isinstance 那半今日不可达（入参是 list，归一层对 list 恒回 list），写出来是为了把
    # `str | list | None` 收窄成 `list`——否则下面每次 append 都得单独忽略类型。
    if not normalized or isinstance(normalized, str):
        return []
    from ctx_weft.protocols import TextPart

    can_externalize = bool(blob_store is not None and blob_store.can_externalize)
    out: "list[ContentPart]" = []
    for part in normalized:
        if _is_text_part(part) or getattr(part, "source_type", "base64") != "base64":
            out.append(part)  # 文本 / ref / url —— 见规则一
            continue
        # 占位用的 media_type 取原始值（不经 _normalize_media_type）：那个归一只服务于
        # 白名单匹配，占位要如实报出 provider 给的是什么，才诊断得动。
        media_type = str(getattr(part, "media_type", "") or "") or "image"
        try:
            validate_content([part])
        except InvalidContentError as exc:
            logger.warning(
                "工具结果里的图片未通过校验（media_type=%s），换成占位：%s", media_type, exc
            )
            out.append(
                TextPart(
                    text=_IMAGE_DROPPED_TMPL.format(
                        reason=f"{media_type} rejected by content validation"
                    )
                )
            )
            continue
        if not can_externalize:
            out.append(part)  # 不接 blob 的宿主：inline 原样跑
            continue
        try:
            # 逐 part 外部化而非整批：一张图 put 失败不该连累同一次结果里的其它图。
            done = await normalize_content([part], blob_store=blob_store, ctx=ctx)
            out.extend(done if isinstance(done, list) else [part])
        except Exception:
            logger.error(
                "工具结果里的图片外部化失败（media_type=%s），换成占位", media_type, exc_info=True
            )
            out.append(
                TextPart(
                    text=_IMAGE_DROPPED_TMPL.format(reason=f"{media_type} could not be stored")
                )
            )
    return out
