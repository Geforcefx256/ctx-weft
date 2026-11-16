"""normalize_content：入口外部化（多模态 Phase 3b Task 2）。

要点：不接 MemoryBlobStore（NullMemoryBlobStore）时行为与 Phase 3a 逐字节一致——本 Phase
最重要的兼容性约束。因此「没有调用 blob store」这件事必须被真的断言，而不是
靠"没报错"推断：下面用计数器 stub 把每一次 put 记下来。
"""

import base64
import hashlib

import pytest

from ctx_weft.core.models.errors import InvalidContentError
from ctx_weft.protocols.content import normalize_content, validate_content
from ctx_weft.protocols import (
    BLOB_REF_PREFIX,
    ImagePart,
    MemoryBlobStore,
    NullMemoryBlobStore,
    ProviderContext,
    TextPart,
)
from tests._initialization import initialize_all

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 100
_PNG = base64.b64encode(_PNG_BYTES).decode()


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="ses-1", tenant_id="default")


class _CountingNullStore(NullMemoryBlobStore):
    """NullMemoryBlobStore + put 计数器。put 仍抛 NotImplementedError（契约不变）。"""

    def __init__(self) -> None:
        self.put_calls = 0

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        self.put_calls += 1
        return await super().put(data, media_type, ctx)


class _CountingEventStore:
    """携图路径的 event blob 门控（Task 4）要求宿主注册 EventBlobStore——最小可外部化桩。

    本文件测的是 memory 侧 normalize_content，event 侧只需「存在且可外部化」，
    不断言其调用次数，故不继承 EventBlobStore 也不需要计数器。
    """

    can_externalize = True

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        ref = f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


class _CountingStore(MemoryBlobStore):
    """能真正外部化的 stub store（内容寻址，与 SqlMemoryProvider 的 ref 形态一致）。"""

    def __init__(self) -> None:
        self.put_calls = 0
        self.seen: list[tuple[bytes, str]] = []
        self.ctx_session_ids: list[str] = []

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        self.put_calls += 1
        self.seen.append((data, media_type))
        self.ctx_session_ids.append(ctx.session_id)
        return f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"

    async def get(self, ref: str, ctx: ProviderContext) -> "tuple[bytes, str] | None":
        return None


# ── can_externalize 探询属性（先探询、再决定，不 try/except NotImplementedError）──


def test_null_blob_store_cannot_externalize():
    assert NullMemoryBlobStore().can_externalize is False


def test_real_blob_store_can_externalize_by_default():
    """基类默认 True——新增该属性不破坏任何既有 MemoryBlobStore 实现。"""
    assert _CountingStore().can_externalize is True


# ── 1. NullMemoryBlobStore → 原样返回且从未调用 store ────────────────────────────


@pytest.mark.asyncio
async def test_null_store_returns_content_unchanged_and_never_calls_put():
    store = _CountingNullStore()
    content = [TextPart(text="看"), ImagePart(data=_PNG, media_type="image/png")]

    out = await normalize_content(content, blob_store=store, ctx=_ctx())

    assert out is content, "不能外部化时必须原样返回同一对象（零改动）"
    assert store.put_calls == 0, (
        "必须先探询 can_externalize 再决定，绝不能调用 put 并捕获 NotImplementedError"
    )


# ── 2. 真 store → base64 图片变成 ref ──────────────────────────────────────


@pytest.mark.asyncio
async def test_real_store_externalizes_base64_image_to_ref():
    store = _CountingStore()
    content = [TextPart(text="看"), ImagePart(data=_PNG, media_type="image/png")]

    out = await normalize_content(content, blob_store=store, ctx=_ctx())

    assert isinstance(out, list)
    assert out[0] == TextPart(text="看"), "文本 part 原样保留"
    img = out[1]
    assert not hasattr(img, "text"), "第二个 part 仍是图片 part"
    assert img.source_type == "ref"
    assert img.data.startswith(BLOB_REF_PREFIX)
    assert img.media_type == "image/png", "media_type 原样保留"
    assert store.put_calls == 1
    assert store.seen == [(_PNG_BYTES, "image/png")], "put 收到的是解码后的原始字节"


# ── 3. 不改原列表 ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_original_list_is_not_mutated():
    store = _CountingStore()
    original = ImagePart(data=_PNG, media_type="image/png")
    content = [TextPart(text="看"), original]

    out = await normalize_content(content, blob_store=store, ctx=_ctx())

    assert out is not content
    assert content[1] is original
    assert original.data == _PNG, "原 ImagePart 不得被就地改写"
    assert original.source_type == "base64"


# ── 4. 纯文本 / None / 全 TextPart → 原样返回且不碰 blob store ───────────────


@pytest.mark.parametrize("content", ["hello", "", None, [], [TextPart(text="a"), TextPart(text="b")]])
@pytest.mark.asyncio
async def test_text_only_content_returned_unchanged_without_touching_store(content):
    store = _CountingStore()

    out = await normalize_content(content, blob_store=store, ctx=_ctx())

    assert out == content
    assert store.put_calls == 0, "纯文本路径不得触碰 blob store"


# ── 5. url / 已是 ref 的 part 原样保留，不重复外部化 ─────────────────────────


@pytest.mark.asyncio
async def test_url_and_ref_parts_are_left_alone():
    store = _CountingStore()
    url_part = ImagePart(data="https://x/y.png", media_type="image/png", source_type="url")
    ref_part = ImagePart(data=f"{BLOB_REF_PREFIX}deadbeef", media_type="image/png", source_type="ref")

    out = await normalize_content([url_part, ref_part], blob_store=store, ctx=_ctx())

    assert isinstance(out, list)
    assert out[0] is url_part
    assert out[1] is ref_part, "已是 ref 的 part 不得被重复外部化"
    assert store.put_calls == 0


@pytest.mark.asyncio
async def test_mixed_content_only_externalizes_base64_parts():
    store = _CountingStore()
    ref_part = ImagePart(data=f"{BLOB_REF_PREFIX}deadbeef", media_type="image/png", source_type="ref")
    content = [ref_part, ImagePart(data=_PNG, media_type="image/png")]

    out = await normalize_content(content, blob_store=store, ctx=_ctx())

    assert isinstance(out, list)
    assert out[0] is ref_part
    assert out[1].source_type == "ref" and out[1].data.startswith(BLOB_REF_PREFIX)
    assert store.put_calls == 1, "只有 base64 形态被外部化"


# ── 6. validate_content 对 ref 形态：跳过解码/尺寸，仍校验 media_type ─────────


def test_validate_content_accepts_ref_without_decoding():
    """ref 的字节与尺寸在 put 时已校验过；data 是 ref 字符串，不该被当 base64 解码。"""
    validate_content([ImagePart(data=f"{BLOB_REF_PREFIX}deadbeef", media_type="image/png",
                                source_type="ref")])


def test_validate_content_still_checks_media_type_for_ref():
    with pytest.raises(InvalidContentError) as ei:
        validate_content([ImagePart(data=f"{BLOB_REF_PREFIX}deadbeef", media_type="image/tiff",
                                    source_type="ref")])
    assert "image/tiff" in str(ei.value)


def test_validate_content_ref_skips_size_limit():
    """ref 字符串本身不该被解成字节去比 5MiB 上限——尺寸在 put 时已经把过关。"""
    validate_content([ImagePart(data=f"{BLOB_REF_PREFIX}{'a' * 64}", media_type="image/png",
                                source_type="ref")])


def test_validate_content_still_rejects_url_source():
    """url 形态本 Phase 不支持，维持 raise。"""
    with pytest.raises(InvalidContentError):
        validate_content([ImagePart(data="https://x/y.png", media_type="image/png",
                                    source_type="url")])


# ── 7. 入口接线：validate_content 先于 normalize_content ────────────────────


def _make_runtime_with_store(store):
    from types import SimpleNamespace

    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.llm.provider import _FixedModelClient
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    runtime = make_runtime(agent_provider=templates)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    runtime.providers.register_memory_blob_store(store)
    runtime.providers.register_event_blob_store(_CountingEventStore())

    def _client(account=None, model=None, *, ctx):
        return _FixedModelClient(
            MockLLMAdapter(responses=[MockResponse(text="Hello!")]),
            model or "mock-model", 128_000, 8_192,
            account=account or "acct-main",
        )

    runtime.providers.register_llm_provider(SimpleNamespace(get_client=_client))
    return runtime


@pytest.mark.asyncio
async def test_rejected_content_never_reaches_blob_store():
    """畸形内容被 validate_content 拒掉时，blob store 里不得留下垃圾。

    顺序反了（先 normalize 后 validate）会让被拒的内容也被写进 blob store。
    """
    store = _CountingStore()
    runtime = _make_runtime_with_store(store)

    with pytest.raises(InvalidContentError):
        await initialize_all(runtime)
        await runtime.run_single_task(
            template_id="agent:tpl_echo",
            user_prompt=[TextPart(text="看"),
                         ImagePart(data="not!valid!base64!", media_type="image/png")],
        )

    assert store.put_calls == 0, "校验失败的内容不得被写入 blob store"


@pytest.mark.asyncio
async def test_run_single_task_externalizes_valid_image():
    """入口确实接了 normalize_content——合法图片会被写入 blob store。"""
    store = _CountingStore()
    runtime = _make_runtime_with_store(store)

    await initialize_all(runtime)
    _handle, state = await runtime.run_single_task(
        template_id="agent:tpl_echo",
        user_prompt=[TextPart(text="看"), ImagePart(data=_PNG, media_type="image/png")],
    )

    assert store.put_calls == 1
    assert store.seen == [(_PNG_BYTES, "image/png")]
    prompt = state.task.user_prompt
    assert isinstance(prompt, list), "守卫：不是 list 的话下面的 hasattr 断言会是重言式"
    imgs = [p for p in prompt if not hasattr(p, "text")]
    assert len(imgs) == 1
    assert imgs[0].source_type == "ref" and imgs[0].data.startswith(BLOB_REF_PREFIX), (
        "落库的 Task.user_prompt 必须已是 ref——core 全程只见 ref"
    )


@pytest.mark.asyncio
async def test_run_single_task_plain_text_never_touches_store():
    """纯文本路径逐字节不变：即便注册了真 store 也一次都不该调用。"""
    store = _CountingStore()
    runtime = _make_runtime_with_store(store)

    await initialize_all(runtime)
    await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="hello")

    assert store.put_calls == 0


@pytest.mark.asyncio
async def test_start_session_externalizes_under_the_real_session_id():
    """第二个入口同样接了 normalize_content，且外部化锚定的 session 就是真正创建的那个。

    start_session 的 session_id 可以是 None（由 runtime 生成）。外部化必须先把它定下来
    并透传给 create_session，否则 blob 会写到一个与真实 session 无关的锚点上。
    """
    from ctx_weft.core.runtime import SessionStartParams

    store = _CountingStore()
    runtime = _make_runtime_with_store(store)

    await initialize_all(runtime)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo",
        user_prompt=[TextPart(text="看"), ImagePart(data=_PNG, media_type="image/png")],

    ))
    try:
        assert store.put_calls == 1
        assert store.ctx_session_ids == [handle.session_id]
    finally:
        await initialize_all(runtime)
        await runtime.cancel_session(handle.session_id)


@pytest.mark.asyncio
async def test_start_session_plain_text_never_touches_store():
    from ctx_weft.core.runtime import SessionStartParams

    store = _CountingStore()
    runtime = _make_runtime_with_store(store)

    await initialize_all(runtime)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo",
        user_prompt="hello",

    ))
    try:
        assert store.put_calls == 0
    finally:
        await initialize_all(runtime)
        await runtime.cancel_session(handle.session_id)


# ── 8. normalize_content 退回纯 memory 侧（blob-store 解耦 Task 1）───────────


class _RecordingStore:
    """记录 put 次数的 memory blob 桩。"""

    can_externalize = True

    def __init__(self) -> None:
        self.puts: list[bytes] = []

    async def put(self, data, media_type, ctx):
        self.puts.append(data)
        return f"{BLOB_REF_PREFIX}mem-{len(self.puts)}"

    async def get(self, ref, ctx):
        return None


@pytest.mark.asyncio
async def test_normalize_content_takes_no_event_blob_store():
    """normalize_content 只认 memory 侧——多传 event_blob_store 必须 TypeError。"""
    store = _RecordingStore()
    content = [ImagePart(data=_PNG, media_type="image/png")]
    with pytest.raises(TypeError):
        await normalize_content(
            content,
            blob_store=store,
            event_blob_store=store,        # 已删除的参数
            ctx=ProviderContext(session_id="s1"),
        )


@pytest.mark.asyncio
async def test_normalize_content_puts_once_into_memory_only():
    """一张图只 put 一次，产出的 ref 就是 memory store 给的那个。"""
    store = _RecordingStore()
    out = await normalize_content(
        [ImagePart(data=_PNG, media_type="image/png")],
        blob_store=store,
        ctx=ProviderContext(session_id="s1"),
    )
    assert len(store.puts) == 1
    assert out[0].data == f"{BLOB_REF_PREFIX}mem-1"
    assert out[0].source_type == "ref"
    assert out[0].byte_size == len(base64.b64decode(_PNG))
