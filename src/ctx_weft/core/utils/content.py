"""内容形态归一层（多模态 Phase 1）。

全仓唯一允许做「内容形态转换」的地方：拼接、JSON 往返、事件脱敏。其余模块只调
本模块，不各自写 isinstance 分支——这是把改动从「N 处散点」收成「1 个模块 +
N 处替换」的关键（spec 2026-08-20-multimodal-design §3①）。

不变量：所有函数对 ``str`` 输入的行为与改造前的朴素字符串操作**逐字节相同**。
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from ctx_weft.core.models.errors import (
    BlobStoreRequiredError,
    InvalidContentError,
)
from ctx_weft.protocols.context import BLOB_REF_PREFIX

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


def content_to_text(content: "str | list[ContentPart] | None") -> str:
    """Render ContentPart list as plain text (images skipped).

    None/空按空文本处理（与 estimate_tokens 对齐）——契约上 content 应为 str | list，但
    个别 provider 可能透传 None（如无描述的能力块），装配期须容错而非 raise。
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if hasattr(item, "text"):
            parts.append(item.text)
    return "".join(parts)

# 一条消息/记录里文本 content 之外的计费补偿项（都往大了取，堵低估致 400 的洞）。
# 供 gateway（LLMMessage）与 prepare/composer（memory 记录 / 装配消息）共用，单一真源。
_MSG_FRAMING_TOKENS = 4       # 每条消息的角色/分隔 framing 开销（provider 计费、文本之外）
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
        return None                        # ref / url：data 不是载荷，长度无意义
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


def _is_text_part(part: Any) -> bool:
    """非文本 part 的判据与 utils.content_to_text / image_part_count 一致。

    刻意用 duck-type 而非 isinstance：测试桩与第三方 provider 可能给出等价的
    鸭子类型对象。（已知局限见 spec §13 的 dict-shaped part 隐患。）
    """
    return hasattr(part, "text")


def content_has_image(content: "str | list[ContentPart] | None") -> bool:
    """内容里是否含至少一个非文本（图片）part。

    归一层里判断「是否含图」的唯一入口——调用方不该自己写 isinstance/hasattr 分支
    散布内容形态知识（spec §3①）。判据与 validate_content / _is_text_part 一致：
    str / None / 空 list / 全 TextPart 均返回 False。

    终审 2026-08-25（缺陷 A）后**当前无生产调用方**：runtime.start_session 原先
    用它决定要不要提前解析 LLM 客户端，但该判据对 dict 形态纯文本会误判成
    「含图」，于是 dict 纯文本反而触发了本该只属于「真图片」路径的提前解析。
    该调用方已删除，保留本函数是因为：(1) 已有直测覆盖判据本身；(2) Phase 3b/4
    的 per-purpose 策略（例如"仅在需要展示缩略图时才判断是否含图"）大概率会用
    到它。删除前请先确认这两点仍成立。
    """
    if not content or isinstance(content, str):
        return False
    return any(not _is_text_part(p) for p in content)


def content_with_prefix(
    content: "str | list[ContentPart] | None", text: str
) -> "str | list[ContentPart]":
    """把 text 拼到内容开头。

    str → 朴素拼接（与改造前逐字节相同）。list → 返回**新列表**，并入首个
    TextPart；首个不是 TextPart 时新插一个。空 text 原样返回（含 list 的同一
    对象，供调用方的 no-op 判定）。None 按空文本处理。
    """
    if not text:
        return content if content is not None else ""
    if content is None:
        return text
    if isinstance(content, str):
        return f"{text}{content}"
    from ctx_weft.protocols import TextPart
    if content and _is_text_part(content[0]):
        head = dataclasses.replace(content[0], text=f"{text}{content[0].text}")
        return [head, *content[1:]]
    return [TextPart(text=text), *content]


def content_with_suffix(
    content: "str | list[ContentPart] | None", text: str
) -> "str | list[ContentPart]":
    """把 text 拼到内容末尾。语义与 content_with_prefix 对称。"""
    if not text:
        return content if content is not None else ""
    if content is None:
        return text
    if isinstance(content, str):
        return f"{content}{text}"
    from ctx_weft.protocols import TextPart
    if content and _is_text_part(content[-1]):
        tail = dataclasses.replace(content[-1], text=f"{content[-1].text}{text}")
        return [*content[:-1], tail]
    return [*content, TextPart(text=text)]


# ── JSON 往返（事件 payload / 投影快照）─────────────────────────────────────


def content_to_jsonable(
    content: "str | list[ContentPart] | None",
) -> "str | list[dict] | None":
    """把内容转成可 json.dumps 的形态。

    ContentPart 是普通 dataclass，直接进 json.dumps 会 TypeError——事件与投影
    落库前必须过这一层（spec §6.2）。str / None 原样返回，纯文本路径零成本。
    """
    if content is None or isinstance(content, str):
        return content
    out: list[dict] = []
    for part in content:
        if _is_text_part(part):
            out.append({"type": "text", "text": part.text})
        else:
            item = {
                "type": "image",
                "data": getattr(part, "data", ""),
                "media_type": getattr(part, "media_type", ""),
                "source_type": getattr(part, "source_type", "base64"),
            }
            # byte_size 只在**有值时**才写键（Phase 3c Task D）：不写 None，存量事件
            # 载荷与不接 MemoryBlobStore 的宿主逐字节不变。读侧缺键 → None → image_tokens 回落。
            byte_size = getattr(part, "byte_size", None)
            if byte_size is not None:
                item["byte_size"] = byte_size
            out.append(item)
    return out


async def content_to_event_jsonable(
    content: "str | list[ContentPart] | None",
    *,
    event_blob_store: "Any",
    ctx: "Any",
) -> "str | list[dict] | None":
    """事件载荷专用：**保 ref、绝不落字节**（spec 2026-08-27 dual-blob-store §6）。

    用在五个参与状态重建的事件发射点（`SESSION_CREATED` / `SESSION_RESUMED` /
    `TASK_CREATED` / `TASK_REQUEUED` / `HITL_*`）。规则逐 part 判定：

    - 文本 part → 原样；
    - ``source_type == "ref"`` 的图 → **降级成 ``[image {media_type}]`` 文本占位**
      并 ``logger.warning``（blob-store 解耦 Task 2）：已是 ref 意味着调用方喂进来
      的是**归一化之后**的内容，那份字节属于 memory 侧，event 侧既无权解读、也解不开
      （两个契约独立，ref 命名空间互不相通）——透传会在事件 payload 里留下一个
      event store 永远打不开的 ref。正确的喂法是把**归一化之前**的原始 content
      递进来（Task 3）；
    - ``source_type == "base64"`` 的图 → put 进 event blob → 换成 ref。
      memory 侧无 blob 时入口不外部化、content 里仍是 inline base64，**只要 event blob
      可用，事件侧仍能独立完成 ref 化**——这是「所有 base64 变引用」在 memory 无 blob
      时也成立的关键（两条路径互补，合起来覆盖 memory×event 可/不可外部化的全部四种
      组合）。
    - 其余图（当下即 ``url``，本仓尚不支持、格式校验层理应拦下——但本函数不能假设
      调用方一定先过了校验）→ **降级成 ``[image {media_type}]`` 文本占位**
      （`_IMAGE_PLACEHOLDER_TMPL`，占位清单见 `core/media/refs.py` 模块 docstring
      第 2 行），**不静默透传**。理由：``url`` 形态可以携带 ``data:`` URI，静默
      `content_to_jsonable` 序列化会把字节原样写进事件 payload，直接击穿本函数的
      核心不变量（事件库恒不含字节）；同 `content_to_jsonable_refs_only` 对非 ref
      图的处理口径一致（本函数取代了它）。

    ⚠️ **不再有「``can_externalize`` 为 False 时整段短路」这回事**（Task 4 收口）：
    携图内容能不能走到这里，由 `validate_content` 入口（spec §7）的第三道门控把关——
    没有可外部化的 EventBlobStore 时，携图内容在入口就被 `BlobStoreRequiredError`
    拒了，根本到不了本函数。本函数因此不必自己判 `can_externalize`：真有内容绕过
    入口跑到这里、而 event blob 又不可用，`NullEventBlobStore.put` 会响亮抛出
    `NotImplementedError`——那正是想要的信号（某处绕开了入口门控，需要被看见，
    而不是被这里的一条静默降级悄悄吞掉）。Task 4 落地前这里曾有一条「退回同步
    `content_to_jsonable`」的短路，专门服务绕过 `validate_content` 直接构造的裸单测；
    那类调用如今若真的递入携图内容，会在此处撞见
    `NotImplementedError`——同样正确：它们绕过的正是本设计要求必经的入口。

    ⚠️ 与 ``redact_content_for_event`` 的分工：那个产出**一整个 str**（含截断预览），
    用于纯观测事件（`LLM_PROMPT_SENT` / `CAPABILITY_FINISHED`）的调试展示，**不可
    回读**；本函数产出 **jsonable 结构**，ref 完整、可经 ``content_from_jsonable``
    还原，用于**参与状态重建**的事件。别把两者互换。

    ⚠️ **不读也不写 ``MemoryEvent.blob_refs``**：那是 memory 侧 GC 的 mark 输入
    （L0.5 引用缺陷修复引入的字段，服务于 `SqlMemoryProvider` 建引用边），与本函数
    产出的事件 payload 是完全不同的载体。两者都叫「blob ref」但不相干（spec §6.1）。

    ``str`` / ``None`` 原样返回同一对象，纯文本路径零成本。
    """
    if content is None or isinstance(content, str):
        return content
    # 挪到循环外（final review M6）：url 分支每命中一次就 import 一次没有必要——
    # 只挪到函数顶而不挪到模块顶，是跟随本文件既有惯例（多处函数体内 import，
    # 避免与 TYPE_CHECKING 下的类型专用导入混在一起）。
    from ctx_weft.protocols import TextPart
    prepared: list[Any] = []
    for part in content:
        if not _is_image_part(part):
            prepared.append(part)                                    # 文本 → 原样
            continue
        source_type = _part_field(part, "source_type", "base64")
        if source_type == "ref":
            # 已是 ref = 调用方喂的是**归一化之后**的内容，那份字节属于 memory 侧，
            # event 侧既无权解读、也解不开（两个契约独立，ref 命名空间互不相通）。
            # 透传会在事件 payload 里留下一个 event store 永远打不开的 ref，故降级。
            # 正确的喂法是把**归一化之前**的原始 content 递进来（见 Task 3）。
            media_type = str(_part_field(part, "media_type", "") or "") or "image"
            logger.warning(
                "content_to_event_jsonable 收到 source_type='ref' 的图片 part（ref=%r）："
                "调用方应递入归一化之前的原始 content。本 part 已降级为文本占位。",
                _part_field(part, "data", ""),
            )
            prepared.append(TextPart(text=_IMAGE_PLACEHOLDER_TMPL.format(
                media_type=media_type)))
        elif source_type == "base64":
            raw = base64.b64decode(str(_part_field(part, "data", "") or ""), validate=True)
            media_type = str(_part_field(part, "media_type", "") or "")
            ref = await event_blob_store.put(raw, media_type, ctx)
            prepared.append(dataclasses.replace(
                part, data=ref, source_type="ref", byte_size=len(raw)))
        else:
            # url（或未来的未知形态）：不静默透传——见函数 docstring「为什么不能透传」。
            media_type = str(_part_field(part, "media_type", "") or "") or "image"
            prepared.append(TextPart(text=_IMAGE_PLACEHOLDER_TMPL.format(
                media_type=media_type)))
    return content_to_jsonable(prepared)


def content_from_jsonable(
    raw: "str | list[dict] | None",
) -> "str | list[ContentPart] | None":
    """content_to_jsonable 的逆变换。

    未知 type 的元素**跳过而不抛**：将来新增 part 类型时，旧版本读到新数据应当
    降级而非崩溃（事件流是只增的，回放会遇到比自己新的数据）。
    """
    if raw is None or isinstance(raw, str):
        return raw
    from ctx_weft.protocols import ImagePart, TextPart
    out: list[ContentPart] = []
    for item in raw:
        kind = item.get("type")
        if kind == "text":
            out.append(TextPart(text=item.get("text", "")))
        elif kind == "image":
            out.append(ImagePart(
                data=item.get("data", ""),
                media_type=item.get("media_type", ""),
                source_type=item.get("source_type", "base64"),
                # 存量行没有该键 → None → image_tokens 回落到 _IMAGE_PART_TOKENS。
                byte_size=item.get("byte_size"),
            ))
        else:
            # 未知类型：跳过而不抛（见 docstring），但留个信号——静默丢弃数据不该完全无声。
            logger.warning("content_from_jsonable: dropping unknown content part type %r", kind)
    return out


# ── 边界归一（Phase 3c Task E / E2）─────────────────────────────────────────


def normalize_content_parts(
    content: "str | list[ContentPart] | None",
) -> "str | list[ContentPart] | None":
    """把 list 里 dict 形态的 part 归一回 ContentPart dataclass；不改原对象。

    **三处边界共用这一份**（``MemoryRecord`` / ``MemoryEvent`` / ``LLMMessage`` 的
    ``__post_init__``）。dict 形态是**协议违规**——这三个字段的类型声明都是
    ``str | list[ContentPart]``；但违规输入现实存在（JSON 往返的 memory provider、
    宿主直构 ``LLMMessage``），且后果全是**静默**的：``image_part_count`` 把 dict 文本
    计成图（多算 1600 token）、``content_to_text`` 返回空串（摘要器看不见）、
    adapter 的 ``_parts_to_blocks`` 把 dict 图片整个丢掉（Phase 3c Task E 后实测 → ``[]``）。

    修法落在**类型自己的边界**、且**只此一份**（spec §3①：形态转换收在归一层）：
    三处各写一遍 isinstance 分支正是该条要防的散点。判据
    ``not hasattr(p, "text")`` 因此保持冻结（用户裁定 D1），不为违规形态解冻。

    **不在 adapter 里 raise**：adapter 在同步出网主路径上，抛异常会掀掉整个 LLM 请求
    （同 Phase 3b 对 ``MemoryBlobStore.get`` 恒不抛的取向）。归一是正解。

    性能（本函数在热路径上，三处 ``__post_init__`` 都无条件调）：``str`` / ``None`` /
    空立即返回**同一对象**；已合规的 dataclass 列表只多一次 ``isinstance`` 扫描并返回
    **同一对象**，不重建。扫描刻意用 ``for/else`` 而非 ``any(genexpr)``——实测生成器
    创建开销比扫描本身还大（691.7ns vs 512.1ns）。

    未知 type 的 dict 交 ``content_from_jsonable`` 处理：跳过而不抛（事件流只增，
    旧版本会读到比自己新的数据），语义与事件重放侧一致。
    """
    if isinstance(content, str) or not content:
        return content                               # 快路径：str / None / 空
    for part in content:
        if isinstance(part, dict):
            break
    else:
        return content                               # 已合规：同一对象，不重建
    normalized: "list[ContentPart]" = []
    for part in content:
        if isinstance(part, dict):
            # 复用归一层唯一真源；未知 type 在那里被跳过（见 docstring）。
            normalized.extend(content_from_jsonable([part]) or [])
        else:
            normalized.append(part)
    return normalized                                # 新列表：不就地改写调用方的 list


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
            except ValueError:      # ref 不是可解析 token：退回短标记，别产出坏占位
                logger.warning("redact_content_for_event: unusable blob ref %r", refs[0])
        data = getattr(part, "data", "") or ""
        src = getattr(part, "source_type", "base64")
        parts.append(
            f"[image {media_type} {src}:{data[:_REDACT_DATA_PREVIEW]}…]"
        )
    return "".join(parts)


# ── 入口校验 ───────────────────────────────────────────────────────────────

# Anthropic 与 OpenAI 都接受的交集。扩这个集合前请先确认两家都支持。
ALLOWED_IMAGE_MEDIA_TYPES = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp",
})

_MAX_IMAGE_BYTES = 5 * 1024 * 1024      # Anthropic 单图约 5MB 上限


def _normalize_media_type(media_type: str) -> str:
    """M8：ALLOWED_IMAGE_MEDIA_TYPES 精确匹配过严——"IMAGE/PNG"（大小写）、
    "image/jpeg; charset=..."（带参数）、"image/jpg"（常见但非标准别名）都会被误拒。
    lookup 前统一归一：小写 + 丢弃 ``;`` 之后的参数部分 + 把 image/jpg 映射到
    image/jpeg。只影响 validate_content 的匹配判定，不改写传给 provider 的原始
    media_type（wire 序列化不在本函数职责内）。
    """
    mt = (media_type or "").strip().lower().split(";", 1)[0].strip()
    if mt == "image/jpg":
        mt = "image/jpeg"
    return mt


def validate_content(
    content: "str | list[ContentPart] | None",
    *,
    event_blob_store: "Any" = None,
) -> None:
    """入口内容校验。通过返回 None，否则抛。

    只作用于 ImagePart——纯文本（str / 全 TextPart / None / 空）零影响、恒通过。

    两道门控，顺序刻意是「格式校验 → event blob 门控」：格式畸形的内容必须报
    ``InvalidContentError``，不能被 blob 门控抢先拦成 ``BlobStoreRequiredError``
    ——那会掩盖真正的问题（终审 2026-08-25 缺陷 B）。

    ⚠️ **这里没有、也不该有「模型支不支持图片」这道门控**（spec
    2026-08-28-multimodal-adapter-dispatch）。模态能力是 ``LLMClient`` 实现方的
    性质，由「host 注册了哪个 adapter 类」表达；core 全程透传多模态内容，纯文本
    adapter 在出网时自行降级成占位并告警。原先那道门控读的是 duck-typed 的
    ``supports_vision``，host 自写的 adapter 几乎必然读不到 → 一律被误判为无视觉。

    刻意**不**校验 token 总量——单条消息塞太多图由装配期 ContextOverflowError
    兜底（spec §6.1 / 子设计 §3）。
    """
    if not content or isinstance(content, str):
        return
    images = [p for p in content if not _is_text_part(p)]
    if not images:
        return

    for img in images:
        raw_media_type = getattr(img, "media_type", "") or ""
        media_type = _normalize_media_type(raw_media_type)
        if media_type not in ALLOWED_IMAGE_MEDIA_TYPES:
            raise InvalidContentError(
                f"不支持的图片类型 {raw_media_type!r}；"
                f"允许：{sorted(ALLOWED_IMAGE_MEDIA_TYPES)}"
            )
        source_type = getattr(img, "source_type", "base64")
        if source_type == "ref":
            # Phase 3b：ref 是本仓 normalize_content 写进 blob store 后产出的形态。
            # 字节合法性与尺寸上限在 put 之前（即上一次 validate_content）已经把过关，
            # data 此时是 "blob:<sha>" 而非 base64——再解码一次必然失败。故跳过解码与
            # 尺寸校验，但 media_type 白名单仍逐条校验（它可能来自记录回放/宿主构造，
            # 不能假定必然合法）。
            continue
        if source_type != "base64":
            # M5：当下（Phase 3a）constraint 2 恒成立——source_type 恒为 "base64"，
            # 本分支今日不可达。原 `continue`（静默放行未经解码/尺寸校验的输入）在
            # Phase 3b 引入 ref/url 形态后就是一个真实的洞：非 base64 的图片会跳过
            # 全部尺寸/内容校验直接放行。改成 raise，强制 Phase 3b 到时必须显式
            # 处理该分支（新增校验逻辑），而不是继续沉默跳过。
            # Phase 3b 已显式处理 "ref"（见上一分支）；剩下的 "url" 本 Phase 不支持，
            # 维持 raise。
            raise InvalidContentError(
                f"不支持的图片来源类型 {source_type!r}；"
                "当前支持 base64 与 ref（url 形态未实现校验）"
            )
        try:
            raw = base64.b64decode(getattr(img, "data", "") or "", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidContentError(f"图片 base64 解码失败：{exc}") from exc
        if len(raw) > _MAX_IMAGE_BYTES:
            raise InvalidContentError(
                f"单张图片 {len(raw)} 字节超过上限 "
                f"{_MAX_IMAGE_BYTES}（5 MiB）"
            )
        # Phase 3c Task D 简报要求「在 validate 解码 inline base64 时顺手填 byte_size」，
        # **本实现刻意不做**——理由两条，见 task-D 报告：
        # ① 零收益：inline 形态的体积由 image_tokens 的 len(data)*3//4（含 padding
        #    修正）反解，与 len(raw) 逐字节相等，填不填算出的 token 数完全一样；真正
        #    需要 byte_size 的只有 ref 形态，而那由 normalize_content 填。
        # ② 有代价：本函数是**校验器**，签名返回 None、契约上不碰入参。填字段就是就地
        #    改写调用方的对象，实测会让 tests/unit/test_multimodal_entry.py 的
        #    「Task 承载全量内容」（task.user_prompt == 原 content）转红——因为
        #    dataclass 相等比较把 byte_size 也算进去。为一个派生字段破坏「入口校验不改
        #    内容」的不变量不划算。

    # 第二道：event blob 门控（spec §7）。放在最后，与格式校验同理——畸形内容
    # 不该因为「没有 blob store」而报一个误导性的错。
    # 严格默认：拿不到可外部化的 store 就拒绝。event_blob_store=None 的调用点
    # （未接线的旧调用方）不做此门控。
    if event_blob_store is not None and not event_blob_store.can_externalize:
        raise BlobStoreRequiredError(
            "携带图片的内容需要宿主注册 EventBlobStore（事件库恒不落字节）。"
            "请调用 ProviderRegistry.register_event_blob_store()。"
        )


# ── 入口外部化（Phase 3b）─────────────────────────────────────────────────


async def normalize_content(
    content: "str | list[ContentPart] | None",
    *,
    blob_store: "Any",
    ctx: "Any",
) -> "str | list[ContentPart] | None":
    """把 base64 图片外部化成 blob ref。返回新内容；**不改原对象**。

    blob_store 不能外部化（``NullMemoryBlobStore``）时**原样返回同一对象**——不接
    blob 的宿主行为与 Phase 3a 逐字节一致（本 Phase 最重要的兼容性约束）。
    判定走 ``blob_store.can_externalize`` 探询，**不**调用 put 再捕获
    NotImplementedError：后者会把「响亮失败」降级成控制流（Phase 1 终审契约）。

    只处理 ``source_type == "base64"`` 的图片 part；文本 / ``url`` / 已是
    ``ref`` 的 part 原样保留（不重复外部化）。

    **调用契约：必须在 ``validate_content`` 通过之后调用。** ``b64decode`` 这里
    刻意不再 try/except——两个入口都是「先 validate 后 normalize」，畸形 base64
    在 validate 阶段就已报 InvalidContentError，到不了这里；此处再包一层
    try/except 只会制造一条永不被执行、也永不被测试的分支。顺序若被后来者接反，
    这里抛出的 binascii.Error 正好是响亮的信号。

    **只写 memory 侧。** event 侧的外部化由 `content_to_event_jsonable` 在事件发射点
    独立完成，两者从同一份原始 content 各自取字节、各自 put、各自拿 ref，**core 不假设
    两个 ref 相同**（两个契约独立，ref 相同只在 host 偷懒用同一实例时才成立，那是实现
    层的巧合，不是协议层的前提）。
    """
    if not content or isinstance(content, str):
        return content
    if not blob_store.can_externalize:
        return content                      # 原样返回，零改动；event 侧也不碰（见 docstring）
    out: list["ContentPart"] = []
    for part in content:
        if _is_text_part(part) or getattr(part, "source_type", "base64") != "base64":
            out.append(part)                # 文本 / url / 已是 ref → 原样
            continue
        raw = base64.b64decode(getattr(part, "data", "") or "", validate=True)
        ref = await blob_store.put(raw, getattr(part, "media_type", ""), ctx)
        # byte_size 必须在这里记下来（Phase 3c Task D）：外部化之后 data 是
        # "blob:<sha>"（长度恒约 69），体积信息就此丢失，而 image_tokens 是同步的、
        # 不能回 MemoryBlobStore 做 IO 取回来。这是最后一个还握着 raw bytes 的地方。
        out.append(dataclasses.replace(
            part, data=ref, source_type="ref", byte_size=len(raw)))
    return out


# ── 出网 rehydrate（Phase 3b）───────────────────────────────────────────────

# 本仓所有图片占位的清单见 `core/media/refs.py` 模块 docstring（L6 收口，裁定 R1）。
# 这一条单向渲染、永不回读，故留在归一层。
_IMAGE_UNAVAILABLE_TMPL = "[image unavailable: {media_type}]"


def _part_field(part: Any, name: str, default: Any = None) -> Any:
    """读 part 的字段，dataclass（属性）与 dict（键）两种形态都支持。

    刻意只服务于 rehydrate 路径，**不动**被 spec §13 冻结的非文本判据
    ``not hasattr(p, "text")``——那个判据同时被 core/utils 的 content_to_text /
    image_part_count 共用，改它必须三处同改。
    """
    if isinstance(part, dict):
        return part.get(name, default)
    return getattr(part, name, default)


def _is_ref_part(part: Any) -> bool:
    """该 part 是否是一个「需要还原」的 blob ref。

    dict 形态必须显式处理：``getattr(dict, "source_type", "base64")`` 在 dict 上
    取不到属性、落回默认值 ``"base64"``，于是 dict 形态的 ref 会被静默当 base64
    塞进 wire payload——图片废掉且全程无任何报错。这是三种结局里最差的一种
    （不可观测的损坏 vs 可观测的降级），故此处走 dict-aware 取值。

    第二条判据（``data`` 以 ``blob:`` 开头）是零误判的兜底：base64 字母表不含
    ``:``，任何以 ``blob:`` 开头的 data 都不可能是合法 base64。有了它，
    「把 blob:<sha> 当 base64 发出网」对任何 part 形态都不可达，即使某个
    非一致性 provider 把 source_type 记错了也一样。
    """
    if _is_text_part(part):
        return False
    if isinstance(part, dict) and part.get("type") != "image":
        return False                     # dict 形态的文本/未知 part
    if _part_field(part, "source_type", "base64") == "ref":
        return True
    return str(_part_field(part, "data", "") or "").startswith(BLOB_REF_PREFIX)


def extract_blob_refs(content: "str | list[ContentPart] | None") -> list[str]:
    """内容里引用到的全部 blob ref（``blob:<sha>``），去重、保持首次出现顺序。

    **归一层是 ref 判据的唯一真源**（spec §3①）：持久化侧（`providers/memory/sql`
    的引用表）必须调本函数，不得各写一遍 isinstance——判据一旦分叉，
    「哪些 blob 还活着」就会和「出网时哪些 part 会被 rehydrate」对不上，
    而那正好是「回收删掉了还在用的图」的成因。

    判据复用 ``_is_ref_part``（dataclass / dict 两种形态都认，
    ``source_type == "ref"`` 或 ``data`` 以 ``blob:`` 开头），故与
    ``rehydrate_content`` 会去 ``MemoryBlobStore.get`` 的那批 part **逐一对应**。

    ``str`` / ``None`` / 无 ref 一律返回空列表（不接 MemoryBlobStore 的宿主永远走这条）。
    """
    if not content or isinstance(content, str):
        return []
    seen: dict[str, None] = {}
    for part in content:
        if not _is_ref_part(part):
            continue
        ref = str(_part_field(part, "data", "") or "")
        if ref.startswith(BLOB_REF_PREFIX):
            seen.setdefault(ref, None)
    return list(seen)


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


def _replace_part(part: Any, **changes: Any) -> Any:
    """产出改过字段的**新** part，保持原形态（dict 进 dict 出，dataclass 同理）。"""
    if isinstance(part, dict):
        return {**part, **changes}
    return dataclasses.replace(part, **changes)


def _unavailable_part(part: Any, media_type: str) -> Any:
    """取不到图时的文本占位。形态跟随入参，内容对同一张图**恒定**。

    占位文本不得含随机 id / 时间戳 / 计数器 / blob sha（用户裁定 D2 的硬约束）：
    降级发生在 compact 这类最需要命中 prompt cache 的时刻，占位每次不同会把该
    purpose 自己的缓存前缀砸掉。
    """
    text = _IMAGE_UNAVAILABLE_TMPL.format(media_type=media_type or "image")
    if isinstance(part, dict):
        return {"type": "text", "text": text}
    from ctx_weft.protocols import TextPart
    return TextPart(text=text)


async def rehydrate_content(
    content: "str | list[ContentPart] | None",
    *,
    blob_store: "Any",
    ctx: "Any",
) -> "str | list[ContentPart] | None":
    """把 blob ref 还原成 base64，供 adapter 拼 wire payload。返回新内容；**不改原对象**。

    落在 gateway 而非 adapter（架构裁定 T0）：``_parts_to_blocks`` /
    ``_serialize_messages`` / ``_build_payload`` 全是同步函数，而 ``MemoryBlobStore.get``
    是 async——同步函数里没法 await。``stream_llm`` 是出网前最后一个 async 关口，
    且一处覆盖三家 adapter。

    三条零开销短路：纯文本（str / None / 空）、不能外部化的 store
    （``NullMemoryBlobStore``）、内容里根本没有 ref——都**原样返回同一对象**，
    不接 MemoryBlobStore 的宿主行为与 Phase 3a 逐字节一致。

    ``get`` 返回 ``None`` 时**降级、不抛**（spec §5.1）：换成
    ``[image unavailable: <media_type>]`` 文本 part。blob 过期 / 宿主换机 /
    GC 误删都会发生，绝不能因取图失败中断整个 loop——rehydrate 在出网主路径上，
    这里抛异常会掀掉整个 LLM 请求。
    """
    if not content or isinstance(content, str):
        return content
    if not blob_store.can_externalize:
        # 存不进去的 store 也取不出来（NullMemoryBlobStore.get 恒 None）。此处短路而不是
        # 让它走 get→None→降级，是为了保住「不接 MemoryBlobStore 时逐字节不变」的约束。
        return content
    if not any(_is_ref_part(p) for p in content):
        return content                   # 没有 ref → 零开销
    out: list["ContentPart"] = []
    for part in content:
        if not _is_ref_part(part):
            out.append(part)
            continue
        ref = str(_part_field(part, "data", "") or "")
        media_type = str(_part_field(part, "media_type", "") or "")
        got = await blob_store.get(ref, ctx)
        if got is None:
            out.append(_unavailable_part(part, media_type))
            continue
        raw, stored_media_type = got
        out.append(_replace_part(
            part,
            data=base64.b64encode(raw).decode("ascii"),
            media_type=media_type or stored_media_type,
            source_type="base64",
        ))
    return out


async def hydrate_event_content(
    content: "str | list[ContentPart] | None",
    *,
    event_blob_store: "Any",
    ctx: "Any",
) -> "str | list[ContentPart] | None":
    """把事件里的 event ref 还原成 base64 part，供恢复路径重新走 memory 侧归一化。

    这是两个 blob 世界之间**唯一**的桥，且方向单一：event → 字节 → 调用方自己决定
    要不要再 put 进 memory。桥架在恢复路径这个交界处、由 event 侧发起，而不是藏在
    memory 的写路径里替 event 代劳（那正是本次解耦拆掉的入口双写）。

    与 `rehydrate_content` 的分工：那个取 `MemoryBlobStore`、产出给 adapter 拼 wire
    payload；本函数取 `EventBlobStore`、产出给 `normalize_content` 重新落 memory。
    两者取的 store 不同、ref 命名空间不同，**不可互换**。

    取不回字节 → `[image unavailable: {media_type}]` 文本占位，**不抛**：event blob
    的保留策略归 host（spec §9），取不到是预期内的正常降级，不该让恢复整个失败。

    **未注册可外部化的 `EventBlobStore`（`can_externalize` 为假）与「取不回字节」是
    同一种情形**，走同一条降级：宿主上次跑时注册了、这次重启没注册，重放出来的
    `source_type == "ref"` part 就再也解不开。此时**绝不能原样返回**——调用方
    （`Runtime._restore_task_prompts`）会把返回值当 memory 侧内容落进 task 字段，
    而 `normalize_content` 按设计不碰 ref part、也不会抛，于是一个 event 命名空间的
    ref 悄悄流进 memory 侧，之后被 `rehydrate_content` 拿去问 `MemoryBlobStore`。
    那是本次解耦要消灭的最后一条跨命名空间通路（终审 I1）。
    """
    if not content or isinstance(content, str):
        return content
    can_get = event_blob_store.can_externalize
    from ctx_weft.protocols import ImagePart
    out: list[Any] = []
    for part in content:
        if not _is_ref_part(part):
            out.append(part)
            continue
        ref = str(_part_field(part, "data", "") or "")
        media_type = str(_part_field(part, "media_type", "") or "")
        got = await event_blob_store.get(ref, ctx) if can_get else None
        if got is None:
            logger.warning(
                "hydrate_event_content: event blob 取不回 %r（%s），降级为占位",
                ref, "无此条目" if can_get else "未注册可外部化的 EventBlobStore")
            out.append(_unavailable_part(part, media_type))
            continue
        raw, got_media_type = got
        out.append(ImagePart(
            data=base64.b64encode(raw).decode(),
            media_type=media_type or got_media_type,
            source_type="base64",
            byte_size=len(raw),
        ))
    return out


# ── per-purpose 降级（Phase 3b Task 4）──────────────────────────────────────

# 本仓所有图片占位的清单见 `core/media/refs.py` 模块 docstring（L6 收口，裁定 R1）。
# ⚠️ 这是 **per-purpose 降级**（不落库、只影响本次 prompt），与 L0.5 降级
# （`core/media` 重写 memory 记录、占位含 ref 可被 `get_image` 取回）**并存、
# 互不替代**，不要合并。单向渲染、永不回读。
_IMAGE_PLACEHOLDER_TMPL = "[image {media_type}]"


def _is_image_part(part: Any) -> bool:
    """该 part 是不是图片。

    dataclass 形态沿用被 spec §13 冻结的判据（``not hasattr(p,"text")``），dict 形态
    **必须显式判 ``type``**：dict 上永远取不到 ``.text`` 属性，直接套冻结判据会把
    ``{"type":"text","text":"hi"}`` 也当成图片、把纯文本换成 ``[image image]``——
    那是实打实的内容损坏。判法与两家 adapter 的 ``_parts_to_blocks`` 一致。
    """
    if isinstance(part, dict):
        return part.get("type") == "image"
    return not _is_text_part(part)


def downgrade_images_to_text(
    content: "str | list[ContentPart] | None",
) -> "str | list[ContentPart] | None":
    """把图片 part 换成确定性文本占位 ``[image {media_type}]``；不改原对象。

    用于 composer 的 per-purpose 策略（用户裁定 D2）：只有 ``act`` 需要模型真看图，
    ``compact`` / ``observe`` / ``background_observe`` / ``recognize_intent`` 一律降级。
    其中 compaction 恰在上下文超预算时触发，不降级就等于「在最贵的时刻多打一发最大的
    请求」。

    **换占位而不是直接删**：保留「这里曾有一张图」的信息，摘要器才写得出「用户提供了
    一张图」而不是完全无感。

    **占位文本必须逐字节确定性**（裁定 D2 的硬约束）：不得含 blob sha / 随机 id /
    时间戳 / 跨调用计数器。各 purpose 发送不同的 tools 集合、tools 排在缓存前缀最前，
    故各 purpose 之间本就不共享缓存条目；但占位每次不同会砸掉**该 purpose 自己**的
    自动前缀缓存——而 compact 正是最需要命中缓存的那一刻。同一条消息内的多张图共用
    相同占位，不加序号（无需区分它们）。

    纯文本（``str`` / ``None`` / 空 / 全文本 part）**原样返回同一对象**，零开销、
    对无图会话逐字节无影响。

    占位一律产出 **dataclass ``TextPart``**，即便入参是 dict 形态的图片——刻意不
    「dict 进 dict 出」：``core/utils`` 的 ``content_to_text`` / ``image_part_count``
    是 dict-blind 的（spec §13 冻结判据），dict 形态的文本 part 会被渲染成空串、
    且仍被计成一张图的 token。若这里回吐 dict 文本，占位文本对摘要器不可见（信息白留）、
    token 也不会降下来（本任务的两个目的双双落空）。两家 adapter 的 ``_parts_to_blocks``
    逐 part 判形态，混合列表照常工作。
    """
    if not content or isinstance(content, str):
        return content
    if not any(_is_image_part(p) for p in content):
        return content
    out: list["ContentPart"] = []
    for part in content:
        if not _is_image_part(part):
            out.append(part)
            continue
        media_type = str(_part_field(part, "media_type", "") or "") or "image"
        from ctx_weft.protocols import TextPart
        out.append(TextPart(text=_IMAGE_PLACEHOLDER_TMPL.format(media_type=media_type)))
    return out


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
            out.append(part)                      # 文本 / ref / url —— 见规则一
            continue
        # 占位用的 media_type 取原始值（不经 _normalize_media_type）：那个归一只服务于
        # 白名单匹配，占位要如实报出 provider 给的是什么，才诊断得动。
        media_type = str(getattr(part, "media_type", "") or "") or "image"
        try:
            validate_content([part])
        except InvalidContentError as exc:
            logger.warning(
                "工具结果里的图片未通过校验（media_type=%s），换成占位：%s",
                media_type, exc)
            out.append(TextPart(text=_IMAGE_DROPPED_TMPL.format(
                reason=f"{media_type} rejected by content validation")))
            continue
        if not can_externalize:
            out.append(part)                      # 不接 blob 的宿主：inline 原样跑
            continue
        try:
            # 逐 part 外部化而非整批：一张图 put 失败不该连累同一次结果里的其它图。
            done = await normalize_content([part], blob_store=blob_store, ctx=ctx)
            out.extend(done if isinstance(done, list) else [part])
        except Exception:
            logger.error(
                "工具结果里的图片外部化失败（media_type=%s），换成占位", media_type,
                exc_info=True)
            out.append(TextPart(text=_IMAGE_DROPPED_TMPL.format(
                reason=f"{media_type} could not be stored")))
    return out
