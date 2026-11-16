
import base64

import pytest

from ctx_weft.core.models.errors import InvalidContentError
from ctx_weft.protocols.content import content_has_image, validate_content
from ctx_weft.protocols import ImagePart, TextPart
from tests._initialization import initialize_all

_PNG = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 100).decode()


# ── 纯文本：零影响 ────────────────────────────────────────────────────────

def test_plain_str_always_passes():
    validate_content("hello")                      # 不抛


def test_none_and_empty_pass():
    validate_content(None)
    validate_content("")
    validate_content([])


def test_text_parts_only_pass():
    validate_content([TextPart(text="a"), TextPart(text="b")])


# ── 格式校验 ─────────────────────────────────────────────────────────────

def test_valid_image_passes():
    validate_content([ImagePart(data=_PNG, media_type="image/png")])


def test_unknown_media_type_rejected():
    with pytest.raises(InvalidContentError) as ei:
        validate_content([ImagePart(data=_PNG, media_type="image/tiff")])
    assert "image/tiff" in str(ei.value)


def test_malformed_base64_rejected():
    with pytest.raises(InvalidContentError):
        validate_content([ImagePart(data="not!valid!base64!", media_type="image/png")])


def test_oversized_image_rejected():
    big = base64.b64encode(b"x" * (6 * 1024 * 1024)).decode()
    with pytest.raises(InvalidContentError) as ei:
        validate_content([ImagePart(data=big, media_type="image/png")])
    assert "5" in str(ei.value), "错误文案应报出上限，便于宿主自查"


# ── M8：media_type 归一（大小写/参数/image-jpg 别名）──────────────────────────

def test_media_type_uppercase_accepted():
    validate_content([ImagePart(data=_PNG, media_type="IMAGE/PNG")])


def test_media_type_with_params_accepted():
    validate_content([ImagePart(data=_PNG, media_type="image/jpeg; charset=binary")])


def test_media_type_jpg_alias_accepted():
    validate_content([ImagePart(data=_PNG, media_type="image/jpg")])


def test_media_type_still_rejects_unsupported_after_normalization():
    with pytest.raises(InvalidContentError):
        validate_content([ImagePart(data=_PNG, media_type="IMAGE/TIFF")])


# ── M5：非 base64 source_type 应 raise，不再静默 continue 跳过校验 ─────────────

def test_non_base64_source_type_raises_not_silently_skipped():
    """Phase 3a 恒 base64（constraint 2），本分支今日不可达；但原 `continue` 会让
    Phase 3b 引入 ref/url 形态后静默跳过尺寸/内容校验。改成 raise，强制显式处理。"""
    with pytest.raises(InvalidContentError):
        validate_content([ImagePart(data=_PNG, media_type="image/png", source_type="url")])


# ── 模态能力已回归 adapter ──────────────────────────────────────────────────

def test_no_vision_gating_any_more():
    """模态能力已回归 adapter（spec 2026-08-28）：入口对图片一律放行，
    只做格式校验。任何模型能力判断都不在这里。"""
    validate_content([ImagePart(data=_PNG, media_type="image/png")])


# ── content_has_image：判据本身的直测 ───────────────────────────────────────
#
# Task 3 复审指出该函数没有任何直测，全靠 validate_content / start_session 的
# 测试间接覆盖。这里逐类型直接断言其返回值，钉住判据本身的正确性（str / None /
# 空 list / 全 TextPart 均为 False，含至少一个非文本 part 为 True）。


def test_content_has_image_str_is_false():
    assert content_has_image("hello") is False


def test_content_has_image_none_is_false():
    assert content_has_image(None) is False


def test_content_has_image_empty_list_is_false():
    assert content_has_image([]) is False


def test_content_has_image_all_text_parts_is_false():
    assert content_has_image([TextPart(text="a"), TextPart(text="b")]) is False


def test_content_has_image_with_image_part_is_true():
    assert content_has_image([TextPart(text="看"), ImagePart(data=_PNG, media_type="image/png")]) is True


# ── 真实入口：入口即拒、不落库 ───────────────────────────────────────────────
#
# 只测 validate_content 本身证明不了它接对了位置——下面这条测试驱动真实的
# runtime.run_single_task() 入口。视觉门控删除后，携图内容仍会在任何持久化之前
# 被拒——这次触发的是仍然保留的第二道门控：本用例故意不注册 EventBlobStore，
# `_validate_and_normalize_content` 取到的是不可外部化的 NullEventBlobStore，
# `validate_content` 因此报 `BlobStoreRequiredError`。这条测试的价值不在于具体
# 是哪道门控触发，而在于证明 `validate_content` 确实接在 run_single_task 的
# 任何持久化（Session/Task/事件）之前——(1) 抛 BlobStoreRequiredError，
# (2) memory 中无任何记录（instantiate / Session / Task 均未落库）。


@pytest.mark.asyncio
async def test_run_single_task_rejects_image_before_persisting_anything_without_event_blob_store():
    from types import SimpleNamespace

    from ctx_weft.core.models.errors import BlobStoreRequiredError
    from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
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
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)
    runtime.providers.register_llm_provider(
        SimpleNamespace(
            get_client=lambda account=None, model=None, *, ctx: _FixedModelClient(
                MockLLMAdapter(responses=[]), model or "mock-model", 128_000, 8_192,
                account=account or "acct-main",
            ),
        )
    )

    with pytest.raises(BlobStoreRequiredError):
        await initialize_all(runtime)
        await runtime.run_single_task(
            template_id="agent:tpl_echo",
            user_prompt=[TextPart(text="看这张图"), ImagePart(data=_PNG, media_type="image/png")],
        )

    assert memory._events == [], (
        "校验必须在任何持久化动作之前拒绝——memory 中不应有任何记录"
    )


# ── 纯文本行为逐字节不变：start_session 不得因新增校验而提前解析 LLM ────────────
#
# 改动前 start_session 从不调用 _resolve_llm——LLM 解析完全推迟到任务真正执行时
# （_make_task_runner → _SessionTaskRunner）才异步发生。若纯文本路径也提前解析，
# "解析不出 LLM" 这件事会从"任务执行时才失败"变成"start_session 里同步失败"，
# 这是本 Phase 明令禁止的行为变化（纯文本逐字节不变）。这条测试锁死：没有注册
# 任何 LLM provider、也没有 llm= fallback 时，纯文本 start_session 仍应正常返回
# TurnHandle，而不是同步抛 RuntimeError("No LLM available...")。


@pytest.mark.asyncio
async def test_start_session_plain_text_does_not_eagerly_resolve_llm():
    from ctx_weft.core.runtime import SessionStartParams
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    # 故意不传 llm=、不 register_llm_provider——_resolve_llm 此刻必然抛
    # RuntimeError("No LLM available...")。纯文本 start_session 不该触发它。
    runtime = make_runtime(agent_provider=templates)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    await initialize_all(runtime)
    handle = await runtime.start_session(
        SessionStartParams.create(
            template_id="agent:tpl_echo",
            user_prompt="纯文本，没有图片",

        )
    )
    assert handle is not None, (
        "纯文本 start_session 应像改动前一样正常返回 TurnHandle——"
        "LLM 解析失败应推迟到任务执行时才发生，不应被新增校验提前触发"
    )


@pytest.mark.asyncio
async def test_start_session_plain_text_resolver_never_called():
    """上一条测试断言"整体不抛"；这条用计数器 stub 直接锁死 resolver 调用次数
    为 0——比"不抛异常"更强的证据，直接证明惰性解析确实惰性。"""
    from ctx_weft.core.runtime import SessionStartParams
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

    calls = []
    original_resolve = runtime._resolve_llm

    def _counting_resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return original_resolve(*args, **kwargs)

    runtime._resolve_llm = _counting_resolve

    await initialize_all(runtime)
    await runtime.start_session(
        SessionStartParams.create(
            template_id="agent:tpl_echo",
            user_prompt="纯文本，没有图片",

        )
    )
    assert calls == [], (
        "纯文本 start_session 不应触发 _resolve_llm——validate_content 对纯文本 "
        "零影响、恒通过，压根用不上 LLM 客户端"
    )


@pytest.mark.asyncio
async def test_start_session_dict_text_does_not_eagerly_resolve_llm():
    """缺陷 A 的端到端回归：dict 形态纯文本（{"type":"text","text":...}）经
    content_has_image 误判为「含图」，但 validate_content 对纯文本零影响、恒
    通过——不再是 RuntimeError("No LLM available...")，_resolve_llm 也从未被
    调用。"""
    from ctx_weft.core.models.errors import InvalidContentError
    from ctx_weft.core.runtime import SessionStartParams
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    # 故意不注册任何 LLM provider——_resolve_llm 若被调用必然抛
    # RuntimeError("No LLM available...")。这条测试要证明它压根没被调用到。
    runtime = make_runtime(agent_provider=templates)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    calls = []
    original_resolve = runtime._resolve_llm

    def _counting_resolve(*args, **kwargs):
        calls.append((args, kwargs))
        return original_resolve(*args, **kwargs)

    runtime._resolve_llm = _counting_resolve

    with pytest.raises(InvalidContentError):
        await initialize_all(runtime)
        await runtime.start_session(
            SessionStartParams.create(
                template_id="agent:tpl_echo",
                user_prompt=[{"type": "text", "text": "hello"}],

            )
        )
    assert calls == [], (
        "dict 形态纯文本应在格式校验阶段就被拒——resolver 不该被调用，"
        "更不该抛出掩盖了真正问题的 RuntimeError"
    )


# ── I3（评审 2026-08-24 fix wave，选项 B）：dict 形态 part 入口硬拒绝 ──────────
#
# _is_text_part 用 `hasattr(part, "text")` 判据（spec §13 冻结，Phase 3a 不得更改）。
# dict 永远不满足 hasattr，故 dict 形态的 part——即便是纯文本 dict
# {"type":"text","text":...}——也会被误判成"非文本"（图片）。评审给出两个修法选项：
# (A) 让归一层认识 Mapping；(B) 保持现状但钉住这个限制、更新 spec。这里选 (B)：
# _is_text_part 与 utils.content_to_text / image_part_count 共享同一判据字面量
# （content.py 顶部 docstring 明确要求两者一致），content_to_text 还被
# test_content_module.py::test_content_to_text_reexported 钉死为 utils 的同一个
# 对象——只在 content.py 本地扩展 Mapping 支持会让这条"一致性"断言出现分叉：
# validate_content/content_has_image 认得 dict-text，utils.content_to_text /
# image_part_count 仍不认得，两条路径对同一份 dict 内容产出不同判断。这个新分叉
# 比现状的"两处对称地不认识 dict"更难追踪，故选 (B)：不改判据本身，只把限制钉死
# 成测试 + 文档（spec §13），要求宿主在 ingest 前把 dict 形态 rehydrate 成
# ContentPart 对象。
#
# 下面两条测试锁死这个已知限制的具体表现——若将来实现选项 (A)（连带把 utils.py
# 一并改成 Mapping-aware），这两条测试需要同步更新为新的、正确的行为。


def test_dict_text_part_is_misclassified_as_image_known_limitation():
    """纯文本 dict part 被 content_has_image 误判为「含图」——已知限制（选项 B），
    非本 Phase 修复范围。终审 2026-08-25（缺陷 A）之后 content_has_image 已不再
    是 start_session 决定是否提前解析 LLM 的判据（该调用方已删除），所以这个
    误判不再连带让 start_session 提前解析 LLM——
    见 test_start_session_dict_text_does_not_eagerly_resolve_llm。"""
    assert content_has_image([{"type": "text", "text": "hello"}]) is True


def test_dict_text_part_rejected_by_validate_content_known_limitation():
    """纯文本 dict part 被 validate_content 当成「无媒体类型的图片」硬拒绝——
    已知限制（选项 B）。宿主必须在 ingest 前把 dict 形态 rehydrate 成
    ContentPart 对象（TextPart/ImagePart），不能直接喂 dict-shaped part 进入口。"""
    with pytest.raises(InvalidContentError) as ei:
        validate_content([{"type": "text", "text": "hello"}])
    assert "''" in str(ei.value)  # media_type 取不到值（dict 无 .media_type 属性），报出空字符串类型


# ── event blob 门控（Task 4）：第二道门控 ─────────────────────────────────────

async def test_image_requires_event_blob_store() -> None:
    """携图会话未注册 EventBlobStore → 入口即拒（spec §7）。

    口径统一：事件库恒不含字节、恒可回读，没有例外分支。
    """
    from ctx_weft.core.models.errors import BlobStoreRequiredError
    from ctx_weft.protocols.events import NullEventBlobStore

    with pytest.raises(BlobStoreRequiredError):
        validate_content(
            [ImagePart(data=_PNG, media_type="image/png")],
            event_blob_store=NullEventBlobStore(),
        )


def test_plain_text_unaffected_by_event_blob_gate() -> None:
    """纯文本在门控之前就已返回——这条不变量不可破。"""
    from ctx_weft.protocols.events import NullEventBlobStore

    validate_content("纯文本", event_blob_store=NullEventBlobStore())
    validate_content(None, event_blob_store=NullEventBlobStore())


def test_gate_order_format_before_blob() -> None:
    """两道门控的顺序：格式 → blob。

    畸形内容必须报 InvalidContentError，不能被 blob 门控抢先——那会掩盖真正的问题。
    """
    from ctx_weft.core.models.errors import InvalidContentError
    from ctx_weft.protocols.events import NullEventBlobStore

    with pytest.raises(InvalidContentError):
        validate_content(
            [ImagePart(data="!!!not-base64!!!", media_type="image/png")],
            event_blob_store=NullEventBlobStore(),
        )
