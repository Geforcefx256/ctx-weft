"""gateway 出网前 rehydrate：ref → base64（多模态 Phase 3b Task 3）。

闭合 Task 2 留下的破损：入口把图片外部化成 ``blob:<sha>``，而两家 adapter 的
``_parts_to_blocks`` 忽略 source_type、无条件把 ``p.data`` 当 base64 写进 wire。
rehydrate 落在 ``stream_llm``（架构裁定 T0：adapter 的序列化链是同步的，
``MemoryBlobStore.get`` 是 async）。

「没有发生某件事」的断言全部用计数器 stub 钉住——不接 MemoryBlobStore 时行为与
Task 2 之前逐字节一致是本 Phase 最重要的兼容性约束，不能靠"没报错"推断。
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from ctx_weft.protocols.content import rehydrate_content
from ctx_weft.core.loop.llm_gateway import stream_llm
from ctx_weft.protocols import (
    BLOB_REF_PREFIX,
    MemoryBlobStore,
    ImagePart,
    LLMChunk,
    LLMMessage,
    LLMRequest,
    NullMemoryBlobStore,
    ProviderContext,
    TextPart,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 100
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode()
_PNG_REF = f"{BLOB_REF_PREFIX}{hashlib.sha256(_PNG_BYTES).hexdigest()}"


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="ses-1", tenant_id="default")


class _CountingStore(MemoryBlobStore):
    """能外部化的 stub：内容寻址，get 计数并可配置为「取不到」。"""

    def __init__(self, *, found: bool = True) -> None:
        self.found = found
        self.get_calls: list[str] = []
        self.blobs: dict[str, tuple[bytes, str]] = {_PNG_REF: (_PNG_BYTES, "image/png")}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        return f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"

    async def get(self, ref: str, ctx: ProviderContext) -> "tuple[bytes, str] | None":
        self.get_calls.append(ref)
        if not self.found:
            return None
        return self.blobs.get(ref)


class _CountingNullStore(NullMemoryBlobStore):
    """NullMemoryBlobStore + get 计数器（get 契约不变：恒返回 None）。"""

    def __init__(self) -> None:
        self.get_calls: list[str] = []

    async def get(self, ref: str, ctx: ProviderContext) -> "tuple[bytes, str] | None":
        self.get_calls.append(ref)
        return await super().get(ref, ctx)


class _CapturingLLM:
    """记录 complete() 实际收到的 messages——即 adapter 将要序列化进 wire 的东西。"""

    def __init__(self) -> None:
        self.seen: list[LLMMessage] = []

    async def complete(self, request: LLMRequest, stream: bool = True):
        self.seen = list(request.messages)
        yield LLMChunk(kind="done")


def _request(*messages: LLMMessage) -> LLMRequest:
    return LLMRequest(model="m", system="s", messages=list(messages))


async def _drain(llm, request, **kw) -> None:
    async for _ in stream_llm(llm, request, **kw):
        pass


def _images(content) -> list:
    """内容里的图片 part。**先断言 list**：``not hasattr(p,"text")`` 在 str 上恒为
    True，少了这道守卫的「含图」断言是重言式。"""
    assert isinstance(content, list), f"expected parts list, got {type(content).__name__}"
    return [p for p in content if not hasattr(p, "text")]


# ── 1. 含 ref 的消息经 stream_llm 后，adapter 收到的是 base64 ──────────────────


@pytest.mark.asyncio
async def test_ref_reaches_llm_as_base64() -> None:
    store = _CountingStore()
    llm = _CapturingLLM()
    msg = LLMMessage(role="user", content=[
        TextPart(text="look"),
        ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref"),
    ])
    await _drain(llm, _request(msg), blob_store=store, provider_ctx=_ctx())

    imgs = _images(llm.seen[0].content)
    assert len(imgs) == 1
    assert imgs[0].source_type == "base64"
    assert imgs[0].data == _PNG_B64
    assert not imgs[0].data.startswith(BLOB_REF_PREFIX)
    assert store.get_calls == [_PNG_REF]


@pytest.mark.asyncio
async def test_rehydrate_does_not_mutate_original_part() -> None:
    store = _CountingStore()
    original = ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref")
    out = await rehydrate_content([original], blob_store=store, ctx=_ctx())
    assert original.data == _PNG_REF and original.source_type == "ref"
    assert _images(out)[0].data == _PNG_B64


# ── 2. get 返回 None → 降级成文本占位，不抛 ────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_blob_degrades_to_text_placeholder_without_raising() -> None:
    store = _CountingStore(found=False)
    llm = _CapturingLLM()
    msg = LLMMessage(role="user", content=[
        TextPart(text="look"),
        ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref"),
    ])
    await _drain(llm, _request(msg), blob_store=store, provider_ctx=_ctx())

    content = llm.seen[0].content
    assert isinstance(content, list)
    assert _images(content) == []                       # 图片没了，但没抛
    texts = [p.text for p in content if hasattr(p, "text")]
    assert "[image unavailable: image/png]" in texts


@pytest.mark.asyncio
async def test_missing_blob_placeholder_is_deterministic() -> None:
    """占位文本必须对同一张图恒定（用户裁定 D2 的硬约束）——不含 sha / 时间戳 / 计数器，
    否则该 purpose 自己的 prompt cache 前缀每次都变。"""
    store = _CountingStore(found=False)
    part = ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref")
    a = await rehydrate_content([part], blob_store=store, ctx=_ctx())
    b = await rehydrate_content([part], blob_store=store, ctx=_ctx())
    assert [p.text for p in a] == [p.text for p in b] == ["[image unavailable: image/png]"]
    assert _PNG_REF not in a[0].text and hashlib.sha256(_PNG_BYTES).hexdigest() not in a[0].text


# ── 3. NullMemoryBlobStore → 消息原样、get 未被调用 ─────────────────────────────────


@pytest.mark.asyncio
async def test_null_blob_store_leaves_content_untouched_and_never_gets() -> None:
    store = _CountingNullStore()
    llm = _CapturingLLM()
    part = ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref")
    msg = LLMMessage(role="user", content=[TextPart(text="look"), part])
    await _drain(llm, _request(msg), blob_store=store, provider_ctx=_ctx())

    imgs = _images(llm.seen[0].content)
    assert imgs == [part]                 # 同一对象，原样
    assert imgs[0].data == _PNG_REF and imgs[0].source_type == "ref"
    assert store.get_calls == []          # get 一次都没被调用


@pytest.mark.asyncio
async def test_no_blob_store_argument_is_a_noop() -> None:
    """既有调用方（不传 blob_store）行为完全不变——rehydrate 整段不执行。"""
    llm = _CapturingLLM()
    part = ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref")
    msg = LLMMessage(role="user", content=[part])
    await _drain(llm, _request(msg))
    assert _images(llm.seen[0].content) == [part]


# ── 4. 纯文本消息 → 逐字节不变、get 未被调用 ──────────────────────────────────


@pytest.mark.asyncio
async def test_plain_text_untouched_and_never_gets() -> None:
    store = _CountingStore()
    llm = _CapturingLLM()
    await _drain(
        llm, _request(LLMMessage(role="user", content="hello world")),
        blob_store=store, provider_ctx=_ctx(),
    )
    assert llm.seen[0].content == "hello world"
    assert store.get_calls == []


@pytest.mark.asyncio
async def test_text_parts_only_untouched_and_never_gets() -> None:
    store = _CountingStore()
    parts = [TextPart(text="a"), TextPart(text="b")]
    out = await rehydrate_content(parts, blob_store=store, ctx=_ctx())
    assert out is parts                   # 无 ref → 原样返回同一对象，零开销
    assert store.get_calls == []


@pytest.mark.asyncio
async def test_base64_image_untouched_and_never_gets() -> None:
    """已是 base64 的图片不该触发任何 blob 访问。"""
    store = _CountingStore()
    parts = [ImagePart(data=_PNG_B64, media_type="image/png", source_type="base64")]
    out = await rehydrate_content(parts, blob_store=store, ctx=_ctx())
    assert out is parts
    assert store.get_calls == []


# ── 5. rehydrate 字节稳定（prompt cache 前缀稳定的前提）─────────────────────────


@pytest.mark.asyncio
async def test_rehydrate_is_byte_stable_across_calls() -> None:
    store = _CountingStore()
    part = ImagePart(data=_PNG_REF, media_type="image/png", source_type="ref")
    a = await rehydrate_content([part], blob_store=store, ctx=_ctx())
    b = await rehydrate_content([part], blob_store=store, ctx=_ctx())
    assert _images(a)[0].data == _images(b)[0].data == _PNG_B64
    assert _images(a)[0].media_type == _images(b)[0].media_type == "image/png"


# ── 6. dict 形态 ref 的盲点（Task 1 读取方扫描抓到）─────────────────────────────


@pytest.mark.asyncio
async def test_dict_shaped_ref_is_not_emitted_as_base64() -> None:
    """裁定 (A)：dict 形态的 ref 绝不能带着 "blob:" 前缀出网。

    ``getattr(dict, "source_type", "base64")`` 在 dict 上落回默认值 "base64"，
    于是 dict 形态的 ref 会被静默当 base64 塞进 wire——图片废掉且全程无报错。
    这里钉住：dict ref 必须要么被还原成 base64、要么降级成占位。

    Phase 3c Task E2 后**这条路上 dict 已到不了 rehydrate**：``LLMMessage.__post_init__``
    在构造时就把 dict 归一成 ``ImagePart``（``dataclasses.replace`` 在 rehydrate 之后
    也会重跑它）。故此处断言的是归一后的 dataclass 形态。``rehydrate_content`` 本身
    仍是「dict 进 dict 出」——那条性质由下面两条直调 ``rehydrate_content`` 的用例钉住。
    """
    store = _CountingStore()
    llm = _CapturingLLM()
    msg = LLMMessage(role="user", content=[
        {"type": "image", "data": _PNG_REF, "media_type": "image/png", "source_type": "ref"},
    ])
    await _drain(llm, _request(msg), blob_store=store, provider_ctx=_ctx())

    content = llm.seen[0].content
    assert isinstance(content, list)
    part = content[0]
    assert isinstance(part, ImagePart)                  # E2：边界归一后 dict 已消失
    assert part.data == _PNG_B64
    assert part.source_type == "base64"
    assert not part.data.startswith(BLOB_REF_PREFIX)
    assert store.get_calls == [_PNG_REF], "归一不得丢掉 source_type='ref'（否则不会去取 blob）"


@pytest.mark.asyncio
async def test_dict_shaped_ref_missing_blob_degrades_to_dict_text() -> None:
    store = _CountingStore(found=False)
    out = await rehydrate_content(
        [{"type": "image", "data": _PNG_REF, "media_type": "image/png", "source_type": "ref"}],
        blob_store=store, ctx=_ctx(),
    )
    assert out == [{"type": "text", "text": "[image unavailable: image/png]"}]


@pytest.mark.asyncio
async def test_dict_shaped_text_part_untouched() -> None:
    store = _CountingStore()
    parts = [{"type": "text", "text": "hello"}]
    out = await rehydrate_content(parts, blob_store=store, ctx=_ctx())
    assert out is parts
    assert store.get_calls == []


# ── 7. 兜底：data 带 blob: 前缀但 source_type 撒谎，也不得当 base64 出网 ────────


@pytest.mark.asyncio
async def test_blob_prefixed_data_is_rehydrated_even_if_source_type_says_base64() -> None:
    """base64 字母表不含 ':'，故 "blob:" 前缀是零误判的 ref 判据。

    有了它，「把 blob:<sha> 当 base64 静默发出去」这个不可观测的损坏对任何 part
    形态都不可达——source_type 记错的记录回放同样被兜住。
    """
    store = _CountingStore()
    parts = [ImagePart(data=_PNG_REF, media_type="image/png", source_type="base64")]
    out = await rehydrate_content(parts, blob_store=store, ctx=_ctx())
    img = _images(out)[0]
    assert img.data == _PNG_B64 and img.source_type == "base64"
    assert store.get_calls == [_PNG_REF]
