"""Text/image content types and shared parsing, validation and blob operations.

Input parsing is strict. Historical normalization remains tolerant of unknown
part types. These helpers never create tasks or commit execution events.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Union

from .errors import CtxWeftError

if TYPE_CHECKING:
    from .context import ProviderContext
    from .events import EventBlobStore
    from .memory import MemoryBlobStore

logger = logging.getLogger(__name__)
BLOB_REF_PREFIX = "blob:"
ALLOWED_IMAGE_MEDIA_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_IMAGE_UNAVAILABLE_TMPL = "[image unavailable: {media_type}]"
_IMAGE_PLACEHOLDER_TMPL = "[image {media_type}]"

__all__ = [
    "TextPart",
    "ImagePart",
    "ContentPart",
    "BLOB_REF_PREFIX",
    "InvalidContentError",
    "BlobStoreRequiredError",
    "parse_content",
    "validate_content",
    "content_has_image",
    "content_to_text",
    "content_to_jsonable",
    "content_from_jsonable",
    "normalize_content_parts",
    "externalize_content",
]


@dataclass
class TextPart:
    """文本片段。"""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ImagePart:
    """图像片段。"""

    data: str  # base64 编码或 URL
    media_type: str  # 'image/png' / 'image/jpeg' / ...
    source_type: Literal["base64", "url", "ref"] = "base64"
    type: Literal["image"] = "image"
    # 原始（解码后）字节数。**可选、默认 None**——既有构造点一律不必改。
    #
    # 存在的理由（Phase 3c Task D）：``image_tokens`` 要按体积估算预算压力，而
    # 外部化之后 ``data`` 是 ``blob:<sha>``（长度恒约 69），与真实体积毫无关系；
    # ``image_tokens`` 是同步函数，不能去 MemoryBlobStore 做 IO 把字节取回来。故让体积
    # 在**还知道的时候**（validate 解码 inline base64 / normalize 外部化拿到 raw
    # bytes 时）被记下来带走，并经 ``content_to_jsonable`` 往返持久化。
    #
    # ``None`` = 不知道（存量记录、不接 blob 的宿主直构、协议外来源）——
    # ``image_tokens`` 对此有明确回落，见该函数。
    byte_size: int | None = None


ContentPart = Union[TextPart, ImagePart]


def parse_content(raw: Any) -> str | list[ContentPart] | None:
    """Parse new input without silently dropping unknown part types.

    JSON text/image dictionaries use the host's existing string defaults. Use
    normalize_content_parts instead when reading forward-compatible history.
    Parsing performs no blob I/O; validate_content checks image payloads.
    """
    if raw is None or isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        raise InvalidContentError("Content must be text or a list of text/image parts")
    result: list[ContentPart] = []
    for item in raw:
        if isinstance(item, str):
            result.append(TextPart(item))
        elif isinstance(item, (TextPart, ImagePart)):
            result.append(dataclasses.replace(item))
        elif isinstance(item, dict) and item.get("type") == "text":
            result.append(TextPart(str(item.get("text", "") or "")))
        elif isinstance(item, dict) and item.get("type") == "image":
            size = item.get("byte_size")
            if size is not None and (type(size) is not int or size < 0):
                raise InvalidContentError("Image byte_size must be a nonnegative integer")
            result.append(
                ImagePart(
                    data=str(item.get("data", "") or ""),
                    media_type=str(item.get("media_type", "") or ""),
                    source_type=item.get("source_type") or "base64",
                    byte_size=size,
                )
            )
        else:
            kind = item.get("type") if isinstance(item, dict) else type(item).__name__
            raise InvalidContentError(f"Unsupported content part type: {kind!r}")
    return result


class InvalidContentError(CtxWeftError):
    """入口内容格式非法：未知 media_type / base64 畸形 / 单图超限。

    入口即拒，不落库——比让畸形内容流到 provider 侧再 400 更早、更可诊断。
    """

    code = "INVALID_CONTENT"


class BlobStoreRequiredError(CtxWeftError):
    """携图内容要求宿主注册 EventBlobStore（spec 2026-08-27 双 blob store §7）。

    事件库的口径是**恒不含字节、恒可回读**，没有例外分支——没有 event blob store 就
    无处放字节，只能在入口拒绝。纯文本会话不受影响。
    """

    code = "BLOB_STORE_REQUIRED"


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
    if content and _is_text_part(content[-1]):
        tail = dataclasses.replace(content[-1], text=f"{content[-1].text}{text}")
        return [*content[:-1], tail]
    return [*content, TextPart(text=text)]


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
    prepared: list[Any] = []
    for part in content:
        if not _is_image_part(part):
            prepared.append(part)  # 文本 → 原样
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
            prepared.append(TextPart(text=_IMAGE_PLACEHOLDER_TMPL.format(media_type=media_type)))
        elif source_type == "base64":
            raw = base64.b64decode(str(_part_field(part, "data", "") or ""), validate=True)
            media_type = str(_part_field(part, "media_type", "") or "")
            ref = await event_blob_store.put(raw, media_type, ctx)
            prepared.append(
                dataclasses.replace(part, data=ref, source_type="ref", byte_size=len(raw))
            )
        else:
            # url（或未来的未知形态）：不静默透传——见函数 docstring「为什么不能透传」。
            media_type = str(_part_field(part, "media_type", "") or "") or "image"
            prepared.append(TextPart(text=_IMAGE_PLACEHOLDER_TMPL.format(media_type=media_type)))
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
    out: list[ContentPart] = []
    for item in raw:
        kind = item.get("type")
        if kind == "text":
            out.append(TextPart(text=item.get("text", "")))
        elif kind == "image":
            out.append(
                ImagePart(
                    data=item.get("data", ""),
                    media_type=item.get("media_type", ""),
                    source_type=item.get("source_type", "base64"),
                    # 存量行没有该键 → None → image_tokens 回落到 _IMAGE_PART_TOKENS。
                    byte_size=item.get("byte_size"),
                )
            )
        else:
            # 未知类型：跳过而不抛（见 docstring），但留个信号——静默丢弃数据不该完全无声。
            logger.warning("content_from_jsonable: dropping unknown content part type %r", kind)
    return out


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
        return content  # 快路径：str / None / 空
    for part in content:
        if isinstance(part, dict):
            break
    else:
        return content  # 已合规：同一对象，不重建
    normalized: "list[ContentPart]" = []
    for part in content:
        if isinstance(part, dict):
            # 复用归一层唯一真源；未知 type 在那里被跳过（见 docstring）。
            normalized.extend(content_from_jsonable([part]) or [])
        else:
            normalized.append(part)
    return normalized  # 新列表：不就地改写调用方的 list


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
                f"不支持的图片类型 {raw_media_type!r}；允许：{sorted(ALLOWED_IMAGE_MEDIA_TYPES)}"
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
        _validate_image_bytes(raw)
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


def _validate_image_bytes(raw: bytes) -> None:
    if len(raw) > _MAX_IMAGE_BYTES:
        raise InvalidContentError(f"单张图片 {len(raw)} 字节超过上限 {_MAX_IMAGE_BYTES}（5 MiB）")


async def externalize_content(
    content: str | list[ContentPart] | None,
    *,
    blob_store: EventBlobStore | MemoryBlobStore,
    ctx: ProviderContext,
    source_blob_store: EventBlobStore | MemoryBlobStore | None = None,
) -> str | list[ContentPart] | None:
    """Store image bytes for a caller-owned record, returning detached content.

    Existing references require an explicit source store. References are read
    there and copied to the target when the stores differ; no namespace search
    or execution command occurs. A successful return confirms only blob writes,
    not a task or event commit. The caller owns record retention and GC roots.
    """
    validate_content(content, event_blob_store=blob_store)
    if content is None or isinstance(content, str):
        return content
    if any(isinstance(p, ImagePart) and p.source_type == "ref" for p in content):
        if source_blob_store is None:
            raise InvalidContentError("Referenced content requires an explicit source blob store")
    out: list[ContentPart] = []
    for part in content:
        if isinstance(part, TextPart):
            out.append(dataclasses.replace(part))
            continue
        media_type = part.media_type
        if part.source_type == "ref":
            stored = await source_blob_store.get(part.data, ctx)
            if stored is None:
                raise InvalidContentError(f"Image reference is unavailable: {part.data}")
            raw, stored_type = stored
            media_type = stored_type or media_type
            if _normalize_media_type(media_type) not in ALLOWED_IMAGE_MEDIA_TYPES:
                raise InvalidContentError(f"Unsupported stored image type: {media_type!r}")
            _validate_image_bytes(raw)
            ref = (
                part.data
                if source_blob_store is blob_store
                else await blob_store.put(raw, media_type, ctx)
            )
        else:
            raw = base64.b64decode(part.data, validate=True)
            ref = await blob_store.put(raw, media_type, ctx)
        out.append(
            dataclasses.replace(
                part, data=ref, source_type="ref", media_type=media_type, byte_size=len(raw)
            )
        )
    return out


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
        return content  # 原样返回，零改动；event 侧也不碰（见 docstring）
    out: list["ContentPart"] = []
    for part in content:
        if _is_text_part(part) or getattr(part, "source_type", "base64") != "base64":
            out.append(part)  # 文本 / url / 已是 ref → 原样
            continue
        raw = base64.b64decode(getattr(part, "data", "") or "", validate=True)
        ref = await blob_store.put(raw, getattr(part, "media_type", ""), ctx)
        # byte_size 必须在这里记下来（Phase 3c Task D）：外部化之后 data 是
        # "blob:<sha>"（长度恒约 69），体积信息就此丢失，而 image_tokens 是同步的、
        # 不能回 MemoryBlobStore 做 IO 取回来。这是最后一个还握着 raw bytes 的地方。
        out.append(dataclasses.replace(part, data=ref, source_type="ref", byte_size=len(raw)))
    return out


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
        return False  # dict 形态的文本/未知 part
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
        return content  # 没有 ref → 零开销
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
        out.append(
            _replace_part(
                part,
                data=base64.b64encode(raw).decode("ascii"),
                media_type=media_type or stored_media_type,
                source_type="base64",
            )
        )
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
                ref,
                "无此条目" if can_get else "未注册可外部化的 EventBlobStore",
            )
            out.append(_unavailable_part(part, media_type))
            continue
        raw, got_media_type = got
        out.append(
            ImagePart(
                data=base64.b64encode(raw).decode(),
                media_type=media_type or got_media_type,
                source_type="base64",
                byte_size=len(raw),
            )
        )
    return out


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
    ``compact`` / ``observe`` / ``recognize_intent`` 一律降级。
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
        out.append(TextPart(text=_IMAGE_PLACEHOLDER_TMPL.format(media_type=media_type)))
    return out
