
from tests.model_testkit import model_snapshot
"""`InvocationResult.content` 放宽 + provider 用 result `content` 交非文本 part。

子设计 §4.2；2026-09-05 把曾经的 `metadata["content_parts"]` 侧信道收成 content 单口径
——`CapabilityEvent(kind="result")` 的 `payload["content"]` 是 `str | list[ContentPart]`，
与三个执行入口、`HitlReply.message` 同一个联合类型。要点：

- content 里有非文本 part → `InvocationResult.content` 变 `[TextPart(文本), *parts]`；
- content 是 str（或拆完没有非文本 part）→ 仍是 **str**，且与改造前**逐字节相同**
  （最重要的兼容性约束）；
- spill / human note 只作用于**文本部分**（gateway 拆开加工完再把 parts 拼回去）；
- 事件 payload 走 `redact_content_for_event`：不泄漏 base64、不对 list 做切片。
"""

from collections.abc import AsyncIterator
from types import SimpleNamespace

from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import ControlMetaKey as K
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState, StepDriver
from ctx_weft.core.loop.steps import observe as recap_mod
from ctx_weft.protocols import ImagePart, MemoryAddress, ProviderContext, TextPart, load_records
from ctx_weft.protocols.capability import (
    AuthorizationDecision,
    Authorizer,
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.filesystem import SpillSink
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

# 一段够长、能在 repr 里被一眼认出的假 base64。
FAKE_B64 = "QUJDREVG" * 200


def _img(data: str = FAKE_B64, source_type: str = "base64") -> ImagePart:
    return ImagePart(data=data, media_type="image/png", source_type=source_type)


# ── provider / 脚手架 ─────────────────────────────────────────────────────────


class _Prov(ToolCapabilityProvider):
    """按构造参数回吐一次 result 事件：文本（+ 可选非文本 part）+ 可选 metadata。"""

    name = "mcp:t"

    def __init__(self, text: str, parts: list | None = None,
                 metadata: dict | None = None, spillable: bool = True) -> None:
        self._text = text
        self._parts = parts
        self._metadata = metadata
        self._spillable = spillable

    def _cap(self) -> ToolCapability:
        return ToolCapability(id="mcp:t:go", name="go", description="d", spillable=self._spillable)

    async def list(self, ctx): return [self._cap()]
    async def retrieve(self, ctx): return [self._cap()]
    async def describe(self, ctx): return CapabilityProviderInfo(name=self.name, capability_count=1)
    async def cancel(self, invocation_id, ctx) -> None: return None

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        async def _run():
            # provider 把混合内容整个放进 content，不再分两处交。
            content = ([TextPart(text=self._text), *self._parts]
                       if self._parts is not None else self._text)
            payload: dict = {"content": content}
            if self._metadata is not None:
                payload["metadata"] = self._metadata
            yield CapabilityEvent(kind="result", payload=payload)
        return _run()


class _NoteAuthorizer(Authorizer):
    """放行但带 human note（走 `[Human note: ...]` 拼接分支）。"""

    async def authorize(self, capability, ctx, arguments=None, *, tool_call_id=""):
        return AuthorizationDecision(allowed=True, message="be careful")


class _SpySpill(SpillSink):
    def __init__(self) -> None:
        self.called = False
        self.spilled: str | None = None

    async def spill(self, content, ctx, *, name_hint="") -> str:
        self.called = True
        self.spilled = content
        return "/tmp/spilled.txt"


def _state_ctx():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1")
    state = LoopState(
        run_id="r1", session=SimpleNamespace(id="s1", tenant_id="default"),
        task=SimpleNamespace(id="tsk_1"), agent=SimpleNamespace(id="agt_1", template_id="t"),
        scope=scope,
        resolved_model=model_snapshot(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=mem, event_bus=InProcessEventBus(),
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                     task_id="tsk_1", agent_id="agt_1"),
    )
    return mem, state, ctx


def _gw(provider, mem, bus, *, spill=None, authorizer=None, spill_threshold=4000):
    cache = CapabilityCache()
    cache.put("agt_1", [provider._cap()])
    providers = [provider] + ([spill] if spill is not None else [])
    return CapabilityGateway(
        capability_cache=cache, capability_providers=providers,
        memory=mem, event_bus=bus,
        default_authorizer=authorizer,
        spill_threshold=spill_threshold,
    )


async def _run(provider, *, spill=None, authorizer=None, spill_threshold=4000):
    """跑一次 invoke，回 (result, CAPABILITY_FINISHED 事件 payload, memory provider)。"""
    mem, state, ctx = _state_ctx()
    bus = InProcessEventBus()
    seen: list = []

    async def _sink(ev):
        if ev.type == EventType.CAPABILITY_FINISHED:
            seen.append(ev.payload)

    bus.subscribe(None, _sink)
    gw = _gw(provider, mem, bus, spill=spill, authorizer=authorizer,
             spill_threshold=spill_threshold)
    res = await gw.invoke("mcp__t__go", {}, state, ctx)
    return res, (seen[0] if seen else {}), mem


# ── 1. content 里带非文本 part → [TextPart(文本), *parts] ─────────────────────


async def test_content_parts_are_appended_after_the_text_part():
    part = _img()
    res, _, _ = await _run(_Prov("here it is", [part]))

    assert isinstance(res.content, list), f"预期 list[ContentPart]，实得 {type(res.content)}"
    assert len(res.content) == 2
    head = res.content[0]
    assert isinstance(head, TextPart) and head.text == "here it is"
    assert res.content[1] is part


async def test_channel_is_generic_not_bound_to_the_media_provider():
    """不认发布者：任意 provider（这里叫 mcp:t，不是 media）都能在 content 里带 part。"""
    res, _, _ = await _run(_Prov("shot", [_img()]))
    assert isinstance(res.content, list)
    assert res.tool_name == "mcp__t__go"


async def test_multiple_parts_keep_provider_order():
    a, b = _img(data="AAAA"), _img(data="BBBB")
    res, _, _ = await _run(_Prov("two", [a, b]))
    assert [p.data for p in res.content[1:]] == ["AAAA", "BBBB"]


async def test_dict_shaped_parts_are_normalized_to_dataclasses():
    """宿主 provider 可能给 JSON 形态；gateway 过归一层，不把 dict 泄进 content。"""
    res, _, _ = await _run(_Prov("json shape", [
        {"type": "image", "data": "ZZZZ", "media_type": "image/png", "source_type": "base64"},
    ]))
    assert isinstance(res.content, list)
    assert not isinstance(res.content[1], dict), "dict 形态的 part 未被归一"
    assert res.content[1].data == "ZZZZ"


async def test_memory_record_keeps_the_parts():
    """TOOL_RESULT 落库带着 ImagePart —— §4.2 靠这条实现「跨重启天然成立」。"""
    _, _, mem = await _run(_Prov("kept", [_img(data="ref-ish",
                                                                  source_type="ref")]))
    ctxp = ProviderContext(session_id="s1", tenant_id="default", task_id="tsk_1", agent_id="agt_1")
    from ctx_weft.protocols import MemoryScope
    view = await load_records(mem,
        MemoryAddress(session_id="s1", task_id="tsk_1", agent_id="agt_1"),
        MemoryScope.TASK, ctxp,
    )
    tool_recs = [r for r in view if r.role == "tool"]
    assert tool_recs, "没有 tool 记录落库"
    stored = tool_recs[-1].content
    assert isinstance(stored, list), f"落库的 content 被拍扁成了 {type(stored)}"
    assert any(getattr(p, "data", None) == "ref-ish" for p in stored)


# ── 2. content 是 str → 仍是 str，且逐字节与改造前相同 ────────────────────────
#
# 「某件事没有发生」型断言：靠 `is` 同一性 + 精确逐字节相等 + 类型守卫三重钉死，
# 并由 Step 6 的变异 M1/M2 证明其非永真。


async def test_no_content_parts_means_plain_str_content():
    res, _, _ = await _run(_Prov("plain output"))
    assert type(res.content) is str, f"纯文本路径的 content 类型变了：{type(res.content)}"
    assert res.content == "plain output"


async def test_no_content_parts_byte_for_byte_no_output_sentinel():
    """空输出兜底 `(no output)` 未被改动。"""
    res, _, _ = await _run(_Prov(""))
    assert type(res.content) is str
    assert res.content == "(no output)"


async def test_image_only_result_does_not_claim_no_output():
    """provider 只回了图（result_parts 为空，metadata 里挂着 content_parts）时，
    不该说「(no output)」——模型会以为真的什么都没拿到，图却已经在 content 里了。"""
    res, _, _ = await _run(_Prov("", [_img()]))
    assert isinstance(res.content, list)
    assert res.content[0].text == "", f"文本槽不该被塞进 (no output)：{res.content[0].text!r}"
    assert res.content[1].data == FAKE_B64


async def test_all_text_content_list_does_not_switch_to_parts_mode():
    """content 是 list 但**拆完没有非文本 part** → 仍回落成 str。

    否则纯文本路径会因为 provider 换了个写法就悄悄换形态（下游 `.strip()` /
    `content[:N]` 全是按 str 写的）。空列表与 None 同理。
    """
    class _P(_Prov):
        def __init__(self, content) -> None:
            super().__init__("")
            self._raw = content

        def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
            async def _run():
                yield CapabilityEvent(kind="result", payload={"content": self._raw})
            return _run()

    for value, expect in (([TextPart(text="still text")], "still text"),
                          ([], "(no output)"),
                          (None, "(no output)")):
        res, _, _ = await _run(_P(value))
        assert type(res.content) is str, f"content={value!r} 时变成了 {type(res.content)}"
        assert res.content == expect


async def test_str_content_is_never_splatted_into_parts():
    """守卫：字符串是可迭代的。content 走 `split_for_tool_result` 的 str 早退分支，
    不会被当成 part 列表 splat 成一串裸 str 元素。"""
    res, _, _ = await _run(_Prov("ab"))
    assert type(res.content) is str
    assert res.content == "ab"


async def test_no_content_parts_keeps_other_metadata_intact():
    res, _, _ = await _run(_Prov("ok", metadata={"task_summary": "s"}))
    assert type(res.content) is str and res.content == "ok"
    assert res.metadata["task_summary"] == "s"


# ── 3. human note 与 spill 只作用于文本部分 ───────────────────────────────────


async def test_human_note_prefixes_only_the_text_part():
    part = _img()
    res, _, _ = await _run(
        _Prov("tool said this", [part]),
        authorizer=_NoteAuthorizer(),
    )
    assert isinstance(res.content, list)
    assert res.content[0].text == "[Human note: be careful]\ntool said this"
    # note 没有在别处再拼一遍，图片 part 原样透传
    assert res.content[1] is part
    assert len(res.content) == 2


async def test_spill_truncates_only_the_text_part_and_keeps_parts_whole():
    part = _img()
    spy = _SpySpill()
    long_text = "x" * 20_000
    res, _, _ = await _run(
        _Prov(long_text, [part]),
        spill=spy, spill_threshold=8000,
    )
    assert spy.called is True
    assert spy.spilled == long_text, "落盘的应当是纯文本，不含 parts"
    assert isinstance(res.content, list)
    assert "truncated" in res.content[0].text and "/tmp/spilled.txt" in res.content[0].text
    assert res.content[1] is part, "图片 part 被 spill 波及了"


async def test_spill_and_human_note_compose_in_order_before_parts():
    """两者叠加：note 前缀在 spill 提示之上，parts 仍在最后、且只有一个文本 part。"""
    spy = _SpySpill()
    res, _, _ = await _run(
        _Prov("y" * 20_000, [_img()]),
        spill=spy, authorizer=_NoteAuthorizer(), spill_threshold=8000,
    )
    text = res.content[0].text
    assert text.startswith("[Human note: be careful]\n[Tool output truncated:")
    assert sum(1 for p in res.content if hasattr(p, "text")) == 1


# ── 4. 事件 payload：不泄漏 base64、不对 list 做切片 ──────────────────────────


async def test_finished_event_does_not_leak_base64_for_parts_content():
    _, payload, _ = await _run(_Prov("see image", [_img()]))
    result = payload["result"]
    assert isinstance(result, str), f"事件 payload 的 result 应是字符串，实得 {type(result)}"
    assert FAKE_B64 not in result
    assert "ImagePart" not in result, "part 的 repr 泄漏进了事件"
    assert "see image" in result
    assert "[image image/png" in result


async def test_finished_event_result_length_counts_characters_not_parts():
    """`len(content)` 在 list 上会退化成 part 数（2）——脱敏后应是字符数。"""
    _, payload, _ = await _run(_Prov("see image", [_img()]))
    assert payload["result_length"] == len(payload["result"])
    assert payload["result_length"] > 20, "result_length 看起来是 part 数而不是字符数"


async def test_finished_event_is_unchanged_on_the_plain_text_path():
    text = "z" * 9000
    _, payload, _ = await _run(_Prov(text), spill_threshold=0)
    assert payload["result"] == text[:8000]
    assert payload["result_length"] == 9000


# ── 5. 读取方在 list 形态下的行为 ─────────────────────────────────────────────


async def test_reader_act_step_wraps_parts_into_an_llm_message():
    """act.py `_execute_tool_calls` → LLMMessage(role="tool", content=...)：
    LLMMessage.content 声明即 `str | list[ContentPart]`，list 原样保留。"""
    from ctx_weft.protocols.llm import LLMMessage
    res, _, _ = await _run(_Prov("t", [_img()]))
    msg = LLMMessage(role="tool", content=res.content, tool_call_id="tc1")
    assert isinstance(msg.content, list) and len(msg.content) == 2


async def test_reader_background_observe_recap_does_not_crash_on_parts(fake_state_ctx, monkeypatch):
    """background_observe 的 `act_recap` 原先对 content 直接 `.strip()`——list 会
    AttributeError，而该异常被 `_run_recap` 整段吞掉（只 log），于是
    **段摘要静默丢失、段保 raw**。所以断言必须钉在「segment_fold 真的被调用且拿到文本」，
    而不是「没抛异常」——后者在吞异常的函数里恒成立。"""
    from ctx_weft.providers.memory import organize as sf

    state, ctx = fake_state_ctx
    res, _, _ = await _run(_Prov("  recap text  ", [_img()],
                                 metadata={K.OBSERVER_ACT_RECAP: "  recap text  "}))
    assert isinstance(res.content, list)  # 前提守卫：确实是 parts 形态

    state.agent.loop_config.max_turns_per_observe = 2

    async def _fake_react(*a, **k):
        return res, ""

    async def _not_short(*a, **k):
        return False

    monkeypatch.setattr(recap_mod.ObserveStep, "_is_short_segment", _not_short)

    folded: list = []

    async def _spy_fold(memory, pctx, *, operation, summary=None, **kwargs):
        if operation == "fold_segment":
            folded.append(summary)

    monkeypatch.setattr("ctx_weft.core.loop.steps.observe.ObserveStep._run_react", _fake_react)
    monkeypatch.setattr(sf, "organize", _spy_fold)

    await StepDriver.for_context(ctx).start("observe", state, ctx, options={"boundary": "interrupt"})

    assert folded == ["recap text"], f"段摘要没落到 segment_fold（实得 {folded!r}）"
    assert FAKE_B64 not in folded[0]


async def test_reader_content_to_text_skips_images():
    from ctx_weft.protocols.content import content_to_text
    res, _, _ = await _run(_Prov("only text survives", [_img()]))
    assert content_to_text(res.content) == "only text survives"
