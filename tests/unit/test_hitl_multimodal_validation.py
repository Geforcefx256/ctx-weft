"""HITL 应答入口的内容校验与外部化（多模态 Phase 3c Task A；Task 10 迁到新契约）。

背景：`validate_content` / `normalize_content` 此前只接了 `run_single_task` 与
`start_session` 两个入口，而 HITL 应答接口在 Phase 1/2 就已放宽成多模态。人类经
HITL 递进来的图**全程不校验、不外部化**——无格式校验（叠加 `b64decode` 默认
`validate=False` 不抛、静默解出垃圾字节，后果是**静默损坏**）、无视觉门控、图以
inline base64 永久留在 memory 里。

**接线点现在是唯一的生产入口 `CtxWeftRuntime.reply_to_hitl`**：它经
`HitlService.resolve` → `ReplyIntake` → runtime 的 `_normalize_hitl_content`
（三入口共用的 `_validate_and_normalize_content` 的薄包装）。旧的
`set_content_normalizer` 那道后期接线已随 `HitlManager` 删除——本文件保留的价值正是
**盯住没人把这条管线悄悄拆掉**。

两条硬约束在下面被显式锁死：
1. **顺序恒为 validate → normalize**——被拒内容不得在 blob store 留垃圾，因此
   「没有写过 blob」这件事必须用计数器 stub 真的断言，而不是靠「没报错」推断；
   且校验失败**不得推进请求状态**（仍 pending、不发事实）。
2. 纯文本逐字节不变（同一对象），且不为解 tenant 去读事件日志。
"""

import base64
import hashlib
from types import SimpleNamespace

import pytest

from ctx_weft.core.models.errors import InvalidContentError
from ctx_weft.core.runtime import CtxWeftRuntime
from ctx_weft.protocols import (
    BLOB_REF_PREFIX,
    MemoryBlobStore,
    ImagePart,
    ProviderContext,
    TextPart,
)
from ctx_weft.protocols.events import EventBlobStore
from ctx_weft.protocols.hitl import (
    HitlAsk,
    HitlReply,
    NoResumeDelivery,
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"x" * 100
_PNG = base64.b64encode(_PNG_BYTES).decode()
# 关键：`base64.b64decode(..., validate=False)`（默认）对它**不抛**，静默解出垃圾字节。
# 必须靠 validate_content 的 validate=True 拦住。
_MALFORMED = "!!!这不是合法的 base64!!!"


class _CountingStore(MemoryBlobStore):
    """能真正外部化的 stub store + put 计数器（内容寻址，ref 形态与真实实现一致）。"""

    def __init__(self) -> None:
        self.put_calls = 0
        self.seen: list[tuple[bytes, str]] = []
        self.ctx_session_ids: list[str] = []
        self.ctx_tenant_ids: list[str] = []

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        self.put_calls += 1
        self.seen.append((data, media_type))
        self.ctx_session_ids.append(ctx.session_id)
        self.ctx_tenant_ids.append(ctx.tenant_id)
        return f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"

    async def get(self, ref: str, ctx: ProviderContext) -> "tuple[bytes, str] | None":
        for data, media_type in self.seen:
            if ref == f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}":
                return data, media_type
        return None


class _CountingEventStore(EventBlobStore):
    """event 侧的 `_CountingStore` 同形版——只记 put 时收到的 ctx（Task 3 review Finding 2）。"""

    def __init__(self) -> None:
        self.put_calls = 0
        self.seen: list[tuple[bytes, str]] = []
        self.ctx_session_ids: list[str] = []
        self.ctx_tenant_ids: list[str] = []

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        self.put_calls += 1
        self.seen.append((data, media_type))
        self.ctx_session_ids.append(ctx.session_id)
        self.ctx_tenant_ids.append(ctx.tenant_id)
        return f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"

    async def get(self, ref: str, ctx: ProviderContext) -> "tuple[bytes, str] | None":
        for data, media_type in self.seen:
            if ref == f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}":
                return data, media_type
        return None


class _VisionClient:
    context_limit = 128_000
    output_reserve = 8_192


class _TextOnlyClient:
    context_limit = 128_000
    output_reserve = 8_192


def _make_runtime(llm, store: "MemoryBlobStore | None" = None) -> CtxWeftRuntime:
    """构造一个最小 runtime（CtxWeftRuntime 硬要求至少一个 AgentCapabilityProvider）。

    Task 4：event blob 门控严格默认拒绝携图内容，本文件绝大多数用例测的是校验/
    视觉门控/tenant 锚定等与 event blob store 本身无关的行为，故默认注册一个可
    外部化的 event blob store 桩，避免这些用例被无关的第三道门控挡在门口。
    唯一需要精确控制 event 侧行为的用例（Finding 2 那条）自己另外
    `rt.providers.register_event_blob_store(...)` 覆盖它。
    """
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    rt = make_runtime(agent_provider=templates, llm=llm)
    if store is not None:
        rt.providers.register_memory_blob_store(store)
    rt.providers.register_event_blob_store(_CountingEventStore())
    return rt


async def _pending(rt: CtxWeftRuntime, session_id: str = "ses-hitl",
                   form: str = "wait") -> str:
    """开一个未决请求。`NoResumeDelivery` 是刻意的——本文件测的是应答**内容管线**，
    不该顺带触发一次 session 续跑。"""
    req = await rt.hitl.open(
        HitlAsk(form=form, delivery=NoResumeDelivery()),
        session_id=session_id, task_id="tsk-1", stage="tool", unattended=False,
    )
    return req.id


async def _reply(rt: CtxWeftRuntime, hid: str, message, *, outcome: str = "accepted",
                 modified_arguments=None):
    """经**唯一的生产应答入口**回话。`agent_id` 如实取自该请求的系统记录（`_pending`
    没传 `agent_id` 时那就是真实值 `""`），不是拍脑袋的占位值——防呆校验本身不是
    本文件的测试对象，但仍要让它按真实记录通过，而不是悄悄绕过。"""
    agent_id = rt.hitl_registry.get(hid).agent_id
    return await rt.reply_to_hitl(HitlReply(
        hitl_id=hid, outcome=outcome, agent_id=agent_id, message=message,
        modified_arguments=modified_arguments,
    ))


def _decision(rt: CtxWeftRuntime, hid: str):
    """终局决定（`HitlRequestView` 刻意不带 message——那是 core 内部的活记录才有的）。"""
    req = rt.hitl_registry.get(hid)
    assert req is not None
    return req.decision


# ── 1. 畸形 base64 → 被拒，且 blob store 未被写入 ──────────────────────────────


@pytest.mark.asyncio
async def test_hitl_answer_rejects_malformed_base64_without_touching_blob_store():
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt)

    with pytest.raises(InvalidContentError):
        await _reply(rt, hid,
                     [TextPart(text="看这张"), ImagePart(data=_MALFORMED, media_type="image/png")])

    assert store.put_calls == 0, (
        "顺序必须是 validate → normalize：被拒的内容不得在 blob store 留下垃圾"
    )
    req = rt.hitl_registry.get(hid)
    assert req is not None
    assert req.resolved is False, "校验失败不得把请求推进到终态"
    assert req.decision is None, "校验失败不得把畸形内容写进决定"


@pytest.mark.asyncio
async def test_hitl_reject_rejects_malformed_base64_without_touching_blob_store():
    """reject 路径与 answer 同一下层，同样不得漏校验。"""
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt)

    with pytest.raises(InvalidContentError):
        await _reply(rt, hid, [ImagePart(data=_MALFORMED, media_type="image/png")],
                     outcome="rejected")

    assert store.put_calls == 0
    assert rt.hitl_registry.get(hid).resolved is False


# ── 2. 格式校验在 HITL 路径同样生效 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_hitl_answer_rejects_disallowed_media_type():
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt)

    with pytest.raises(InvalidContentError):
        await _reply(rt, hid, [ImagePart(data=_PNG, media_type="image/bmp")])

    assert store.put_calls == 0


# ── 3. 合法图 + 真 blob store → 落进 req.message 的是 ref ─────────────────────


@pytest.mark.asyncio
async def test_hitl_answer_externalizes_valid_image_to_ref():
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt, session_id="ses-blob")

    view = await _reply(rt, hid,
                        [TextPart(text="看这张"), ImagePart(data=_PNG, media_type="image/png")])

    assert view is not None and view.outcome == "accepted"
    content = _decision(rt, hid).message
    # 陷阱守卫：`not hasattr(p, "text")` 在 str 上恒为 True，下面的断言才不是重言式。
    assert isinstance(content, list), f"应答内容必须仍是 parts 列表，实为 {type(content)!r}"
    images = [p for p in content if not hasattr(p, "text")]
    assert len(images) == 1
    assert images[0].source_type == "ref", "HITL 递进来的图必须被外部化成 ref"
    assert images[0].data.startswith(BLOB_REF_PREFIX)
    assert images[0].data != _PNG
    assert store.put_calls == 1
    assert store.seen[0] == (_PNG_BYTES, "image/png"), "写进 blob 的必须是解码后的原始字节"
    assert store.ctx_session_ids == ["ses-blob"], "外部化必须锚定该 HITL 请求所属 session"
    assert content[0].text == "看这张", "文本 part 原样保留"
    # 存下来的 ref 必须能解回原始字节（wire 上仍是可解码的 base64 之来源）。
    assert await store.get(images[0].data, ProviderContext(session_id="ses-blob")) == (
        _PNG_BYTES, "image/png",
    )


# ── 4.（已删）「未注入 normalizer 时」的三条：裸 `HitlManager()` 不复存在。
#
# 那三条锁的是「旧实现未接线时退化成恒等变换 / 携图直接炸」这组过渡行为。新契约里
# `ReplyIntake.__init__` **要求**显式传入 normalizer（无默认值），生产路径恒由
# `CtxWeftRuntime` 构造期一次性接好（没有 setter、没有半成品窗口）——「未注入」这个
# 状态在类型与构造上都不可达，因此不再有可测的行为。管线**确实**挂在共用真源上这一点，
# 由下面第 6 组的 `test_runtime_injects_helper_that_delegates_to_the_shared_source` 钉住。

# ── 5. 纯文本应答逐字节不变（即使已注入 normalizer）─────────────────────────


@pytest.mark.asyncio
async def test_plain_text_hitl_answer_is_byte_identical():
    store = _CountingStore()
    calls: list[int] = []
    rt = _make_runtime(SimpleNamespace(), store)
    _orig = rt._resolve_llm

    def _counting_resolve(*a, **k):
        calls.append(1)
        return _orig(*a, **k)

    rt._resolve_llm = _counting_resolve  # type: ignore[method-assign]
    hid = await _pending(rt)

    text = "就是一句纯文本"
    await _reply(rt, hid, text)

    assert _decision(rt, hid).message is text, "纯文本必须原样返回同一对象"
    assert store.put_calls == 0, "纯文本不得触碰 blob store"
    assert calls == [], "纯文本不得触发 LLM 解析（格式校验先行，纯文本走不到图片分支，自然用不上）"


@pytest.mark.asyncio
async def test_empty_message_reject_is_byte_identical():
    """reject 的默认 message 为空串——最常见的既有调用形态，不得因接线而改变。"""
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt)

    view = await _reply(rt, hid, "", outcome="rejected")

    assert view is not None and view.outcome == "rejected"
    assert _decision(rt, hid).message == ""
    assert store.put_calls == 0


# ── 6. 三个调用点共用同一真源，行为一致 ──────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_injects_helper_that_delegates_to_the_shared_source():
    """HITL 用的薄包装底下必须仍是那三处共用的同一个方法，不是各写一遍的副本。

    `ReplyIntake` 收的是 `_normalize_hitl_content`（它由 session_id 解 tenant）；这条
    断言它**确实转调**共用真源 `_validate_and_normalize_content`，而不是自己重写一遍
    validate → normalize。旧版断言的是 `set_content_normalizer` 注入了什么——那道
    setter 已随 `HitlManager` 删除，钉子改钉在构造期接好的 `ReplyIntake` 上。
    """
    rt = _make_runtime(_VisionClient())
    injected = rt.hitl._intake._normalizer
    assert injected is not None
    assert injected.__func__ is CtxWeftRuntime._normalize_hitl_content

    seen: list[tuple] = []

    async def _spy(content, session_id, **kw):
        seen.append((content, session_id, kw))
        # 共用真源返回**二元组** `(memory 侧内容, event 侧载荷)`（blob-store 解耦
        # Task 3），桩必须同形——只回 content 的话调用方会把它当二元组解包
        # （短字符串正好解成两个字符，静默把 message 变成 "h"）。
        return content, content

    rt._validate_and_normalize_content = _spy  # type: ignore[method-assign]
    hid = await _pending(rt, session_id="ses-delegate")
    await _reply(rt, hid, "hi")

    assert len(seen) == 1, "薄包装必须转调共用真源（_validate_and_normalize_content）"
    assert seen[0][0] == "hi"
    assert seen[0][1] == "ses-delegate"
    assert _decision(rt, hid).message == "hi", (
        "回调返回的二元组第一项才是内容，不得被当成序列拆开")


@pytest.mark.asyncio
async def test_all_three_entry_points_reject_malformed_base64_before_any_put():
    """run_single_task / start_session / HITL 应答——三处顺序与判据必须一致。"""
    from ctx_weft.core.runtime import SessionStartParams
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    bad = [TextPart(text="看"), ImagePart(data=_MALFORMED, media_type="image/png")]

    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    store = _CountingStore()
    rt = make_runtime(agent_provider=templates, llm=_VisionClient())
    rt.providers.register_memory_blob_store(store)
    memory = InMemoryMemoryProvider()
    rt.providers.register_memory(memory)

    with pytest.raises(InvalidContentError):
        await rt.run_single_task(template_id="agent:tpl_echo", user_prompt=bad)
    assert store.put_calls == 0
    assert memory._events == [], "入口即拒、不落库"

    with pytest.raises(InvalidContentError):
        await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt=bad, context_limit=128_000,
        ))
    assert store.put_calls == 0
    assert memory._events == []

    hid = await _pending(rt)
    with pytest.raises(InvalidContentError):
        await _reply(rt, hid, bad)
    assert store.put_calls == 0


# ══ Task A2：补 Task A 的三条缺口 ══════════════════════════════════════════════
#
# 三条都是 controller 简报之误（已逐条核实）：
#   缺口 1 `resolve_approve` 未接 normalizer——approve 的备注同样是多模态，同一个洞还开着；
#   缺口 2 视觉门控解析的是**默认** client，不是本次应答真正要用的模型（正确性缺陷）；
#   缺口 3 blob 的 tenant 锚点写死 "default"，多租户宿主下落错 tenant。
# 修法：回调签名从 (content, session_id) 扩成 (content, req)，runtime 侧由 req 取
# resume_llm_* 与 session_id。


class _RoutingLLMProvider:
    """按 (account, model) 分派 client 的 stub provider，并记下每次解析的实参。

    缺口 2 必须用它才能真的红：只有「默认 client 与指定 client 能力不同」时，
    「门控判的是哪一个」才可观测。
    """

    def __init__(self, default, by_key: dict) -> None:
        self.default = default
        self.by_key = by_key
        self.calls: list[tuple] = []

    def get_client(self, account=None, model=None):
        self.calls.append((account, model))
        return self.by_key.get((account, model), self.default)


def _make_routing_runtime(provider: _RoutingLLMProvider, store: "MemoryBlobStore | None" = None):
    """注册按账号/模型分派的 provider——_resolve_llm 优先走 registry。"""
    rt = _make_runtime(_VisionClient(), store)
    rt.providers.register_llm_provider(provider)
    return rt


# ── A2-1/2：approve 路径同样接上（缺口 1）───────────────────────────────────────


@pytest.mark.asyncio
async def test_hitl_approve_rejects_malformed_base64_without_touching_blob_store():
    """`approve(message=...)` 与 answer/reject 同为多模态入口，不得漏校验。"""
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt, session_id="ses-hitl", form="approval")

    with pytest.raises(InvalidContentError):
        await _reply(rt, hid,
                     [TextPart(text="备注"), ImagePart(data=_MALFORMED, media_type="image/png")],
                     modified_arguments={"command": "ls -la"})

    assert store.put_calls == 0, "被拒的内容不得在 blob store 留垃圾"
    req = rt.hitl_registry.get(hid)
    assert req is not None
    assert req.resolved is False, "校验失败不得把请求推进到终态"
    assert req.decision is None, (
        "校验失败不得写决定——message 与 modified_arguments 都不得落下")


@pytest.mark.asyncio
async def test_hitl_approve_externalizes_valid_image_to_ref():
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    hid = await _pending(rt, session_id="ses-appr", form="approval")

    view = await _reply(rt, hid,
                        [TextPart(text="放行，见图"), ImagePart(data=_PNG, media_type="image/png")])

    assert view is not None and view.outcome == "accepted"
    content = _decision(rt, hid).message
    # 陷阱守卫：`not hasattr(p, "text")` 在 str 上恒为 True。
    assert isinstance(content, list), f"应答内容必须仍是 parts 列表，实为 {type(content)!r}"
    images = [p for p in content if not hasattr(p, "text")]
    assert len(images) == 1
    assert images[0].source_type == "ref", "approve 备注里的图同样必须被外部化"
    assert images[0].data.startswith(BLOB_REF_PREFIX)
    assert images[0].data != _PNG
    assert store.put_calls == 1
    assert store.ctx_session_ids == ["ses-appr"]


# ── 模态能力已回归 adapter：HITL 路径携图不再受任何模型能力判断（spec 2026-08-28）──


@pytest.mark.asyncio
async def test_hitl_image_reaches_memory():
    """HITL 应答携带图片时，内容被外部化落库成功——模态能力判断已不在 core，
    应答不再携带模型选择，换模型是独立的 `set_agent_llm`/`set_session_llm` 命令。"""
    store = _CountingStore()
    provider = _RoutingLLMProvider(
        default=_TextOnlyClient(), by_key={("acct-v", "vision-model"): _VisionClient()},
    )
    rt = _make_routing_runtime(provider, store)
    hid = await _pending(rt, session_id="ses-named-vision")

    view = await _reply(rt, hid, [ImagePart(data=_PNG, media_type="image/png")])

    assert view is not None and view.outcome == "accepted"
    assert store.put_calls == 1
    content = _decision(rt, hid).message
    assert isinstance(content, list), f"应答内容必须仍是 parts 列表，实为 {type(content)!r}"
    assert content[0].source_type == "ref"


@pytest.mark.asyncio
async def test_plain_text_reply_never_touches_the_event_store():
    """纯文本应答不碰事件库。

    2026-09-19 之前这里防的是「为解 tenant 去读事件日志」；那套反查已经删掉（tenant 随请求
    带着走），所以现在它守的是更一般的一条：应答内容管线不该为了任何理由回头读事件流。
    """
    store = _CountingStore()
    rt = _make_runtime(_VisionClient(), store)
    reads: list[str] = []
    _orig = rt.event_store.read_by_session

    async def _counting(session_id):
        reads.append(session_id)
        return await _orig(session_id)

    rt.event_store.read_by_session = _counting  # type: ignore[method-assign]
    hid = await _pending(rt)

    await _reply(rt, hid, "纯文本")

    assert _decision(rt, hid).message == "纯文本"
    assert reads == [], "纯文本应答不得为了解 tenant 去读事件日志"


# ── Task 3 review Finding 2：event 侧的 tenant 必须是真实解析值，不是硬编码 default ──


@pytest.mark.asyncio
async def test_hitl_event_put_receives_the_real_session_tenant_not_default():
    """应答提交路径给 event 侧 `content_to_event_jsonable` 传的 ctx.tenant_id
    必须是**请求带来的**真实 tenant，而不是写死的 "default"。

    刻意只注册 EventBlobStore、**不注册 MemoryBlobStore**：这是「memory 不可外部化、
    event 可外部化」组合（spec §6：`normalize_content` 短路不碰 event 侧，ref 化改由
    `content_to_event_jsonable` 独立完成）——message 到 `_resolve` 时仍是 inline
    base64，必然真的调用 event_blob_store.put，断言才立得住（若 memory 也可外部化，
    入口的双写会先把内容变成 ref，提交时走 ref 直通分支，根本不会再 put，
    这条用例就验不到本次要修的 bug）。
    """
    event_store = _CountingEventStore()
    rt = _make_runtime(_VisionClient())
    rt.providers.register_event_blob_store(event_store)
    hid = await _pending_with_tenant(
        rt, session_id="ses-tenant-event", tenant_id="tenant-gamma")

    view = await _reply(rt, hid, [ImagePart(data=_PNG, media_type="image/png")])

    assert view is not None and view.outcome == "accepted"
    assert event_store.put_calls == 1, "message 应仍是 inline base64，必须真的外部化一次"
    assert event_store.ctx_tenant_ids == ["tenant-gamma"], (
        "event 侧 put 收到的 tenant 必须是该 session 的真实 tenant，而非硬编码 default"
    )
    assert event_store.ctx_session_ids == ["ses-tenant-event"]


# ── blob-store 解耦 Task 3：HITL_* 事件里的 ref 归 event store ──────────────────


class _PrefixedEventStore(_CountingEventStore):
    """ref 方案刻意不同于 memory 侧（多一段 ``evt-``），仍内容寻址、仍幂等。

    两侧同用 sha256 时「这个 ref 是谁家的」不可观测，跨命名空间引用就被
    「恰好相同」掩盖了——这就是本组用例要拆掉的那层掩盖。
    """

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        await super().put(data, media_type, ctx)
        return f"{BLOB_REF_PREFIX}evt-{hashlib.sha256(data).hexdigest()}"

    async def get(self, ref: str, ctx: ProviderContext) -> "tuple[bytes, str] | None":
        for data, media_type in self.seen:
            if ref == f"{BLOB_REF_PREFIX}evt-{hashlib.sha256(data).hexdigest()}":
                return data, media_type
        return None


@pytest.mark.asyncio
async def test_hitl_event_payload_carries_an_event_ref_not_the_memory_ref():
    """`HitlResolved` 的 message 里必须是 **event store** 的 ref，且字节取得回。

    这是 HITL 入口版的解耦验收：事件侧载荷必须由**归一化之前的原始**应答内容算出。
    若改由已归一化的 memory ref 重算，`content_to_event_jsonable` 会把它
    降级成文本占位——图片就在事件流里丢了，而 memory 侧看起来一切正常。
    """
    mem_store = _CountingStore()
    evt_store = _PrefixedEventStore()
    rt = _make_runtime(_VisionClient(), mem_store)
    rt.providers.register_event_blob_store(evt_store)
    hid = await _pending(rt, session_id="ses-two-stores")

    await _reply(rt, hid,
                 [TextPart(text="看这张"), ImagePart(data=_PNG, media_type="image/png")])

    events = await rt.event_store.read_by_session("ses-two-stores")
    answered = next(e for e in events if e.type == "HitlResolved")
    parts = answered.payload["message"]
    assert isinstance(parts, list), f"message 不得被拍扁，实为 {type(parts).__name__}"
    images = [p for p in parts if isinstance(p, dict) and p.get("type") == "image"]
    assert len(images) == 1, (
        f"事件里应恰有一张图（被降级成文本占位就会是 0 张），实为 {len(images)}"
    )
    ref = images[0]["data"]

    ctxp = ProviderContext(session_id="ses-two-stores")
    assert await evt_store.get(ref, ctxp) == (_PNG_BYTES, "image/png"), (
        "事件里的 ref 必须能由 event store 解回原始字节"
    )
    assert await mem_store.get(ref, ctxp) is None, (
        "事件 payload 里出现了 memory 侧的 ref——两个命名空间不相通"
    )
    assert _PNG not in str(answered.payload), "事件 payload 恒不含字节"
    # 两侧各写各的，各一次
    assert mem_store.put_calls == 1
    assert evt_store.put_calls == 1


# ── tenant 随请求带着走（2026-09-19）──────────────────────────────────────────
#
# tenant 是 host 侧的概念，core 只搬运：`ProviderContext` → `HitlService.open` →
# `PendingHitl.tenant_id` → 应答内容管线。上面那三条用例走的是**兜底**路径（请求上的
# tenant 是缺省 "default"，靠活 TM / 事件日志解出真值）；下面两条钉的是那条链本身。


async def _pending_with_tenant(rt, *, session_id: str, tenant_id: str) -> str:
    """开一个**带 tenant** 的未决请求——生产路径就是这么开的
    （`capability_gateway` 传 `ctx.provider_ctx.tenant_id`、`act` 传
    `state.session.tenant_id`）。"""
    req = await rt.hitl.open(
        HitlAsk(form="wait", delivery=NoResumeDelivery()),
        session_id=session_id, task_id="tsk-1", stage="tool", unattended=False,
        tenant_id=tenant_id,
    )
    return req.id


def _count_event_reads(rt) -> list[str]:
    """给**所有**读事件库的路装计数。

    应答内容管线**不该**为任何理由回头读事件流（tenant 随请求带着走，2026-09-19 起 core
    里连解析入口都没有了）。每条路都装上，免得将来有人换一条路读、断言还以为是绿的。

    ⚠️ 这个函数从前装的是 `rt._read_session_events_of_types` 与 `read_by_session`。前者
    2026-09-20 删了（src 里零调用者，而它正是「按类型读整条会话」那个要清的形状），于是那
    一条计数器守的是一个不存在的方法——**负向断言在这种情况下不会红，只会变得空洞**。现在
    改成装 store 的三个读原语，覆盖面反而更全。
    """
    hits: list[str] = []
    store = rt.event_store
    originals = {
        name: getattr(store, name)
        for name in ("read_by_session", "read_range", "read_session_events_of_types")
    }

    def _wrap(name, fn):
        async def spy(*a, **k):
            hits.append(name)
            return await fn(*a, **k)
        return spy

    for name, fn in originals.items():
        setattr(store, name, _wrap(name, fn))         # type: ignore[method-assign]
    return hits
