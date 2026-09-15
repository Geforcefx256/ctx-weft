"""Phase 4 Task 6：图片折叠与回放的**端到端**验收（子设计 §11）。

Task 1-5b 各自都有单元测试，但都是在打过桩的边界上验的：refs 只验编解码、fold 只验
`memory.fold` 的入参、`get_image` 只验返回值形状、L0.5 接线只验 `escalating_compact`
里的调用顺序。**没有任何一条证明这些零件真的连成了一条线。**

本文件从真实全链路上跑通它：

    start_session（真 runtime / 真 memory / 真 FsBlobStore MemoryBlobStore）
      → 归一层把 base64 外部化成 blob:<sha>
      → PrepareStep 预算触发 escalating_compact → L0.5 `demote_for_budget` 落库
      → 模型（stub LLM）**从 prompt 里读出占位里的 ref**，调 media__get_image
      → CapabilityGateway → MediaCapabilityProvider（图直接在 result content 里）
      → 工具结果作为普通 TOOL_RESULT 记录落库（图在对话尾部）
      → 下一轮 PrepareStep 装配把它带上
      → llm_gateway rehydrate ref→base64 → adapter 出网 wire payload

stub LLM 只做一件真模型也会做的事：**扫 prompt 文本找占位、把 ref 原样抄进工具参数**。
它不认识 memory、不认识 blob store，也拿不到测试的局部变量——所以「模型能取回图」这件事
是被链路本身证明的，不是被测试喂出来的。

**跨组件组合（2026-08-29）**：本文件里承载图片字节的 `blob_store` 现在是独立的
`FsBlobStore`（文件系统内容寻址），memory 侧仍是 `InMemoryMemoryProvider`——引用边
与字节从一开始就分居两处，靠 ref 串起来，这正是宿主真正会跑的形态（SqlMemoryProvider
维护引用边的场景另见 `test_l05_demotion_blob_lifecycle.py` / `test_blob_gc_integration.py`）。
`blob_store=None` 的用例（未注册 MemoryBlobStore）不受影响，继续验「不接 blob store
时一切保持 inline」那条支路。

⚠️ 断言口径（台账陷阱 4/5）：
- 「某件事没有发生」型断言（重定位的 user 消息不落 memory / 未注册 MemoryBlobStore 不降级）
  一律配一个「确实发生了」的正向对照，写在**同一个用例内**；
- base64 一律断言 **`b64decode(...) == 原始字节`**——`b64decode` 默认不抛、会静默解出
  垃圾字节，「不以 blob: 开头」是弱判据（Phase 3b 实测）。
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses as _dc
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

import pytest

from ctx_weft.core.utils.content import content_to_jsonable
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.core.loop.steps.segment_fold import segment_fold
from ctx_weft.core.media.refs import find_image_placeholders
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import (
    BLOB_REF_PREFIX,
    ImagePart,
    LLMChunk,
    LLMUsage,
    LoopConfig,
    MemoryScope,
    ProviderContext,
    TextPart,
    ToolCall,
)
from ctx_weft.protocols.capability import (
    CapabilityEvent,
    CapabilityProviderInfo,
    ToolCapability,
    ToolCapabilityProvider,
)
from ctx_weft.protocols.events import EventBlobStore
from ctx_weft.protocols.memory import MemoryBlobStore
from ctx_weft.providers.blob.fs import FsBlobStore
from ctx_weft.providers.llm.anthropic import AnthropicMultimodalAdapter
from ctx_weft.providers.llm.openai import _TOOL_IMAGE_NOTICE, OpenAIMultimodalAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

# 两张**不同**的真字节图。全链路上任何一处把 A 和 B 弄混、或把 ref 解成别的 blob，
# 逐字节断言都会当场炸——这正是不用「非空/不以 blob: 开头」做判据的理由。
_RAW_A = b"\x89PNG\r\n\x1a\n-A-" + bytes(range(96))
_RAW_B = b"\x89PNG\r\n\x1a\n-B-" + bytes(range(96, 192))
_B64_A = base64.b64encode(_RAW_A).decode()
_B64_B = base64.b64encode(_RAW_B).decode()


def _prompt() -> list:
    return [
        TextPart(text="look at these two screenshots"),
        ImagePart(data=_B64_A, media_type="image/png"),
        ImagePart(data=_B64_B, media_type="image/png"),
    ]


class _StubEventBlobStore:
    """Task 4 的第三道门控（event blob 门控）要求携图会话注册 EventBlobStore。

    本文件测的是 L0.5 折叠/回放/两家 adapter wire 形状等链路，与 event blob 存储
    本身无关，故用一个最小可外部化桩满足门控——不断言其调用细节。
    """

    can_externalize = True

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        import hashlib
        ref = f"blob:{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


# ── 宿主侧的纯文本工具（覆盖 5 的对照组）─────────────────────────────────────


_ECHO_ID = "probe:echo"
_ECHO_TEXT = "probe echo: nothing but text"


class _EchoToolProvider(ToolCapabilityProvider):
    """一个只返回文本的宿主工具。

    存在的理由有两个：(1) 覆盖 5 需要一条**纯文本**工具结果，用来钉住它的 wire 形态
    没有被工具返图那条路改掉；(2) 覆盖 4 需要「同批两个 tool call」，
    OpenAI 追加的那条 user 消息必须落在**整段 tool 之后**而不是夹在中间。
    """

    name = "probe"
    description = "test-only text tool"

    def capability(self) -> ToolCapability:
        return ToolCapability(
            id=_ECHO_ID, name="echo", description="Echo a fixed line of text.",
            input_schema={"type": "object", "properties": {}}, side_effects=False,
            spillable=False,
        )

    async def list(self, ctx: ProviderContext) -> list[ToolCapability]:
        return [self.capability()]

    async def describe(self, ctx: ProviderContext) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(
            name=self.name, capability_count=1, supports_streaming=False,
            supports_cancel=False, description=self.description,
        )

    async def cancel(self, invocation_id: str, ctx: ProviderContext) -> None:
        return None

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._handle()

    async def _handle(self) -> AsyncIterator[CapabilityEvent]:
        yield CapabilityEvent(kind="result", payload={"content": _ECHO_TEXT, "metadata": {}})


# ── wire 捕获 + 「会读占位」的 stub 模型 ──────────────────────────────────────


def _payload_texts(payload: dict) -> list[str]:
    """payload 里所有模型**看得到的文本**（含 tool_result 内部的文本 block）。

    两家 adapter 的形状不同，这里一并铺平——stub 模型就是靠它找占位的，和真模型看到的
    是同一批字符。
    """
    out: list[str] = [str(payload.get("system") or "")]
    for m in payload.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    out.append(str(b.get("text") or ""))
                elif b.get("type") == "tool_result":
                    inner = b.get("content")
                    if isinstance(inner, str):
                        out.append(inner)
                    elif isinstance(inner, list):
                        out.extend(str(x.get("text") or "")
                                   for x in inner if isinstance(x, dict))
    return out


def _tool_names(payload: dict) -> set[str]:
    """payload 里的工具名——两家 adapter 的 tools 形状不同。

    Anthropic：``{"name": ...}``；OpenAI：``{"type":"function","function":{"name":...}}``。
    只按 Anthropic 那一种取名字会让 OpenAI 那一路的 recognize_intent 快照被误判成 act
    回合（实测踩过：模型于是在旁路快照里就把图取了，主循环全乱）。
    """
    out: set[str] = set()
    for t in payload.get("tools") or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name") or (t.get("function") or {}).get("name")
        if name:
            out.add(str(name))
    return out


def _refs_offered(payload: dict) -> list[str]:
    """prompt 里出现过的 L0.5 占位 ref（保序去重）。"""
    seen: list[str] = []
    for text in _payload_texts(payload):
        for ref, _mt in find_image_placeholders(text):
            if ref not in seen:
                seen.append(ref)
    return seen


class _PlaceholderReadingLLM:
    """stub 模型的行为（与 adapter 无关，混入两家 adapter 各一份）。

    每次被调用：
      1. 若 prompt 里有**没取过**的 L0.5 占位 ref → 调 `media__get_image`（外加可选的
         `probe__echo`，用来构造「同批两个 tool call」）；
      2. 否则 → `control__finish_task` 收尾。

    它只读 payload 文本，不读 memory、不读 blob store、也拿不到测试里的 ref 变量。
    """

    batch_with_echo = False

    def _init_stub(self) -> None:
        self.captured_payloads: list[dict] = []
        self.calls: list[str] = []
        self.asked: set[str] = set()
        self._n = 0

    def _tid(self) -> str:
        self._n += 1
        return f"tc{self._n}"

    def _decide(self, payload: dict) -> tuple[str, list[ToolCall]]:
        if "control__update_task_metadata" in _tool_names(payload):  # recognize_intent 旁路快照
            self.calls.append("intent")
            return "", []
        pending = [r for r in _refs_offered(payload) if r not in self.asked]
        if pending:
            ref = pending[0]
            self.asked.add(ref)
            self.calls.append(f"get_image:{ref[:12]}")
            calls = [ToolCall(id=self._tid(), name="media__get_image",
                              arguments={"ref": ref})]
            if self.batch_with_echo:
                calls.append(ToolCall(id=self._tid(), name="probe__echo", arguments={}))
            return "let me look at that image again", calls
        self.calls.append("finish")
        return "done", [ToolCall(id=self._tid(), name="control__finish_task",
                                 arguments={"deliverables_summary": "described"})]

    async def _emit(self, text: str, tool_calls: list[ToolCall]) -> AsyncIterator[LLMChunk]:
        if text:
            yield LLMChunk(kind="token", text=text)
        for tc in tool_calls:
            yield LLMChunk(kind="tool_call", tool_call=tc)
        yield LLMChunk(kind="usage", usage=LLMUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2))
        yield LLMChunk(kind="done",
                       finish_reason="tool_use" if tool_calls else "stop")


class _WireCapturingAnthropicLLM(_PlaceholderReadingLLM, AnthropicMultimodalAdapter):
    """真 `AnthropicAdapter._build_payload` / `_serialize_messages`，不走网络。"""

    def __init__(self, **kw: Any) -> None:
        AnthropicMultimodalAdapter.__init__(self, api_key="test-key", **kw)
        self._init_stub()

    def complete(self, request, stream=True):
        payload = self._build_payload(request)
        self.captured_payloads.append(payload)
        return self._emit(*self._decide(payload))


class _WireCapturingOpenAILLM(_PlaceholderReadingLLM, OpenAIMultimodalAdapter):
    """真 `OpenAIAdapter._build_payload` / `_serialize_messages`，不走网络。"""

    def __init__(self, **kw: Any) -> None:
        OpenAIMultimodalAdapter.__init__(self, api_key="test-key", **kw)
        self._init_stub()

    def complete(self, request, stream=True):
        payload = self._build_payload(request)
        self.captured_payloads.append(payload)
        return self._emit(*self._decide(payload))


# ── 会话跑手 ──────────────────────────────────────────────────────────────────
#
# context_limit=6000 / reserved_output_tokens=0 / compact_token_ratio=0.5：
#   - 两张图约 2×1600 token，pin 成 priority-0 的 user_prompt 不可裁，故上限必须
#     容得下它（否则 budget.py 抛 ContextOverflowError，压根走不到 compact）——
#     spec: tool-schema-budget 起装配预算先扣**工具面预留**（控制工具 + 回读工具
#     的 schema 是真实请求占用），窗口要按「地板 + 预留」配，4300 已不够；
#   - 同时首轮装配量 / 6000 ≥ compact_token_ratio(0.5) → PrepareStep 真的触发
#     escalating_compact，L0.5 才有机会跑。
# compact_keep_recent_images=1：保住**最新**那张（取回后是取回的那张），确保被降的是
#   最老的 A，且下一轮不会把刚取回的图又降掉。


async def _wait_task_terminal(handle, *, timeout: float = 5.0):
    """等 task 到终态就返回，**不经** `handle.wait_for_finish`。

    `wait_for_finish`（2026-09-04 起）多了一层更强的保证：终态事件到达后，若 close
    边界的后台 observe（段折叠/胶囊化，spec 2026-07-20 延迟折叠）确有在途，还会继续
    等它把折叠落地才返回——这是 host「等 wait_for_finish 返回就去读 memory 渲染」这条
    真实用法需要的durability，不是可选项。

    但本文件的断言恰恰要看**折叠落地前**的中间状态（取回的图仍以一条普通记录挂在
    对话尾部，还没被 close 时的段折叠 supersede 掉）。这里直接复刻 `wait_for_finish`
    曾经的（也是现在 `TaskFailed`/`TaskCanceled` 仍然沿用的）「见到 task 终态事件就
    返回」判据，绕开新增的那层等待，只钉住「task 到终态」这一刻——断言内容不变，只是
    不再途经 `wait_for_finish` 这个更强的契约。"""
    from ctx_weft.core.control.reducers import TASK_STATUS_BY_EVENT
    from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES
    from ctx_weft.protocols.events import EventFilter
    terminal = {et for et, st in TASK_STATUS_BY_EVENT.items() if st in TERMINAL_TASK_STATUSES}
    async with asyncio.timeout(timeout):
        async for ev in handle.event_bus.stream(
            EventFilter(agent_id=handle.agent_id, task_id=handle.task_id)
        ):
            if ev.type in terminal:
                return handle._state
    return handle._state


async def _run_session(*, llm, blob_store, batch_with_echo: bool = False,
                       prompt=None):
    resolver = InlineAgentTemplateProvider()
    resolver.register(_dc.replace(
        make_echo_template(),
        loop_config=LoopConfig(compact_keep_recent_images=1, compact_token_ratio=0.5)))
    llm.batch_with_echo = batch_with_echo
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)
    runtime.providers.register_capability(_EchoToolProvider())
    runtime.providers.register_event_blob_store(_StubEventBlobStore())
    if blob_store is not None:
        runtime.providers.register_memory_blob_store(blob_store)

    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo",
        user_prompt=_prompt() if prompt is None else prompt,
        context_limit=6000, reserved_output_tokens=0))
    # 不经 wait_for_finish：本文件要看 close 边界后台 observe 折叠落地**前**的中间状态
    # （见 _wait_task_terminal docstring），故只等 task 到终态。
    state = await _wait_task_terminal(handle, timeout=20.0)
    assert state is not None
    assert state.task.status == "FINISHED", f"expected FINISHED, got {state.task.status}"
    return runtime, memory, state


def _pctx(state) -> ProviderContext:
    return ProviderContext(session_id=state.session.id, task_id=state.task.id,
                           agent_id=state.agent.id)


async def _view(memory, state) -> list:
    return await memory.load_view(state.scope, MemoryScope.TASK, _pctx(state))


async def _l05_events(runtime, state) -> list:
    events = await runtime.event_store.read_by_session(state.session.id)
    return [e for e in events
            if e.type == EventType.MEMORY_COMPACTED
            and e.payload.get("source") == "demote_images"]


def _image_parts(content: object) -> list:
    """content 里的 ImagePart。

    调用方**必须**先 `assert isinstance(content, list)`——content 是 str 时本函数返回
    空列表，拿它做「没有图」的断言会得到重言式（Phase 3b 已知陷阱）。
    """
    if not isinstance(content, list):
        return []
    return [p for p in content if getattr(p, "type", None) == "image"]


def _record_text(rec) -> str:
    content = rec.content
    if isinstance(content, str):
        return content
    return "\n".join(p.text for p in (content or []) if hasattr(p, "text"))


def _anthropic_tool_results(payload: dict) -> list[dict]:
    out: list[dict] = []
    for m in payload.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            out.extend(b for b in content
                       if isinstance(b, dict) and b.get("type") == "tool_result")
    return out


def _relocated_user_message(msgs: list[dict]) -> tuple[int, dict]:
    """定位 OpenAI adapter 追加的那条承载图片的 user 消息（§4.3）。

    判据是**结构位置**而不是「含 image_url 的 user 消息」——原始 user_prompt 里那张没被
    降级的图同样渲染成 user + image_url，按内容找会一次抓两条。这里要的是「紧跟在整段
    连续 tool 消息之后、内容清一色 image_url」的那条。
    """
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    assert tool_idx, "wire 上一条 tool 消息都没有——工具结果没进这一轮装配"
    assert tool_idx == list(range(tool_idx[0], tool_idx[0] + len(tool_idx))), (
        f"tool 消息不连续：{tool_idx}")
    nxt = tool_idx[-1] + 1
    assert nxt < len(msgs), "整段 tool 之后什么都没有——重定位的 user 消息没被追加"
    m = msgs[nxt]
    assert m.get("role") == "user" and isinstance(m.get("content"), list), (
        f"整段 tool 之后不是承载图片的 user 消息，而是 {m.get('role')!r}")
    kinds = [b.get("type") for b in m["content"]]
    assert kinds and set(kinds) == {"image_url"}, (
        f"重定位的 user 消息里混进了非图片 block：{kinds}")
    return nxt, m


def _decode_data_url(url: str) -> bytes:
    head, _, b64 = url.partition(",")
    assert head.startswith("data:") and head.endswith(";base64"), f"bad data url head {head!r}"
    return base64.b64decode(b64)


# ── 覆盖 1：完整往返 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_round_trip_demoted_image_comes_back_decodable_on_the_wire(tmp_path) -> None:
    """真图入库 → L0.5 降级 → 模型读占位调 media:get_image → 图回到对话尾部 →
    下一轮装配带上它 → wire 上是**能解码回原始字节**的 base64。

    这条把 Task 1/2/3/4/5 的全部接缝串成一条线。任何一处没接上——占位写错解不出 ref、
    provider 没把图放进 content、工具结果没落库、装配把 parts 拍扁、rehydrate 取错 blob
    ——都会在下面某一条断言上炸，而不是静默降级成「没有图」。
    """
    blobs = FsBlobStore(tmp_path / "blobs")
    llm = _WireCapturingAnthropicLLM()
    runtime, memory, state = await _run_session(llm=llm, blob_store=blobs)

    # ① L0.5 真的跑了，且真的降了图（不是「一张都没降也算通过」）
    l05 = await _l05_events(runtime, state)
    assert l05, "没有 source='demote_images' 的 MemoryCompacted 事件——L0.5 根本没跑"
    assert sum(e.payload["demoted_images"] for e in l05) == 1, (
        f"应恰好降 1 张（keep_recent=1 保住最新那张），实为 {[e.payload for e in l05]}")
    assert all(e.payload["freed_tokens"] > 0 for e in l05), (
        "freed_tokens 为 0 —— token 口径不认图片，L0.5 等于白跑")

    # ② memory 里：最老那张变成占位（含 ref），最新那张仍是真图（正向对照）
    view = await _view(memory, state)
    user_recs = [r for r in view if r.role == "user"]
    assert len(user_recs) == 1
    assert isinstance(user_recs[0].content, list), (
        f"USER_PROMPT 记录被拍扁成了 {type(user_recs[0].content).__name__}")
    placeholders = find_image_placeholders(_record_text(user_recs[0]))
    assert len(placeholders) == 1, f"user 记录里应恰好一个 L0.5 占位，实为 {placeholders}"
    ref_a, media_type = placeholders[0]
    assert media_type == "image/png"
    surviving = _image_parts(user_recs[0].content)
    assert len(surviving) == 1 and surviving[0].data != ref_a, (
        "对照组失败：另一张图也被降了 / 或一张都没剩，那本条的「降了一张」就没有意义")

    # 占位里的 ref 指向的确实是 A 的字节（不是随手编的 sha）
    blob = await blobs.get(ref_a, _pctx(state))
    assert blob is not None and blob[0] == _RAW_A, "占位里的 ref 取不回原始字节"

    # ③ 模型确实读着占位调了工具（stub 只从 prompt 文本里拿 ref）
    assert llm.asked == {ref_a}, f"模型没有照占位调 get_image：asked={llm.asked}"

    # ④ 图回到了**对话尾部**：最后一条带图的记录是 tool 角色的工具结果
    img_recs = [r for r in view if _image_parts(r.content)]
    restored = [r for r in img_recs if any(p.data == ref_a for p in _image_parts(r.content))]
    assert len(restored) == 1, "取回的图没有作为一条普通记录落库"
    assert restored[0].role == "tool"
    assert view.index(restored[0]) > view.index(user_recs[0]), (
        "取回的图没有排在原 user 回合之后——它应该在对话尾部")
    restored_text = _record_text(restored[0])
    assert ref_a in restored_text and "1st message in this task" in restored_text, (
        f"取回结果缺少位置信息文本，实为 {restored_text!r}")

    # ⑤ 下一轮装配把它带上，且 wire 上是**可解码回原始字节**的 base64
    idx = llm.calls.index("finish")
    finish_payload = llm.captured_payloads[idx]
    tool_results = _anthropic_tool_results(finish_payload)
    assert tool_results, "收尾那轮的 wire 里没有 tool_result —— 工具结果没进下一轮装配"
    images = [b for tr in tool_results if isinstance(tr.get("content"), list)
              for b in tr["content"] if b.get("type") == "image"]
    assert len(images) == 1, (
        f"取回的图没有出现在下一轮的 wire tool_result 里（找到 {len(images)} 个）")
    src = images[0]["source"]
    assert src["type"] == "base64" and src["media_type"] == "image/png"
    assert base64.b64decode(src["data"]) == _RAW_A, (
        "wire 上的 base64 解码后不等于原始图片字节——rehydrate 还原出了别的内容")
    assert base64.b64decode(src["data"]) != _RAW_B, "取回的是另一张图"


# ── 覆盖 2：OpenAI 重定位的 user 消息不得落 memory（§4.3 关键性质 1）────────────


@pytest.mark.asyncio
async def test_openai_relocated_user_message_is_wire_only_and_never_lands_in_memory(
    tmp_path,
) -> None:
    """OpenAI 的 `role="tool"` 不收图，adapter 在整段 tool 之后**追加一条 user 消息**
    承载图片。这条消息**只能存在于 wire payload**：`segment_fold.py:47` /
    `finalize.py:466` / `background_observe.py:87` 三处都用「最后一条 role=user 回合」
    划段，落库会打乱段折叠与胶囊范围。

    「memory 里没有它」是典型的「某件事没有发生」型断言，压根没产生任何消息时也成立。
    故本用例把正向对照写在同一处：**wire 上必须真的有那条 user 消息**（且它带的
    base64 解码回原始字节），memory 里 user 角色的记录才**仍然只有一条**。
    """
    blobs = FsBlobStore(tmp_path / "blobs")
    llm = _WireCapturingOpenAILLM()
    runtime, memory, state = await _run_session(llm=llm, blob_store=blobs)

    assert await _l05_events(runtime, state), "L0.5 没跑，本用例的前提不成立"
    assert llm.asked, "模型没调 get_image —— 后面的断言会全部落空"

    # ── 正向：wire 上确实有那条重定位的 user 消息，且载着真图 ──
    finish_payload = llm.captured_payloads[llm.calls.index("finish")]
    msgs = finish_payload["messages"]
    rel_idx, rel_msg = _relocated_user_message(msgs)
    assert _TOOL_IMAGE_NOTICE.strip() in msgs[rel_idx - 1]["content"], (
        "被搬空的那条 tool 消息里没有指向后一条消息的标记")
    urls = [b["image_url"]["url"] for b in rel_msg["content"]
            if b.get("type") == "image_url"]
    assert len(urls) == 1
    assert _decode_data_url(urls[0]) == _RAW_A, (
        "重定位的 user 消息里的 base64 解码后不是原始图片字节")

    # ── 反向：memory 里没有它 ──
    view = await _view(memory, state)
    user_recs = [r for r in view if r.role == "user"]
    assert len(user_recs) == 1, (
        f"task 视图里的 user 回合应仍只有那条 USER_PROMPT，实为 "
        f"{[(r.role, _record_text(r)[:40]) for r in user_recs]}——"
        "adapter 重定位的 user 消息落库了，段边界会被打乱")
    # 图在 memory 里活在 role="tool" 的工具结果记录上，不是 user 记录上
    img_recs = [r for r in view if _image_parts(r.content)]
    assert [r.role for r in img_recs if any(
        p.data in llm.asked for p in _image_parts(r.content))] == ["tool"]
    # 那条 wire 专属消息的标记文本一个字都没进 memory
    assert not any(_TOOL_IMAGE_NOTICE.strip() in _record_text(r) for r in view), (
        "wire 专属的图片重定位标记出现在了 memory 记录里")


# ── 覆盖 3：生命周期（§5）────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_restored_image_is_folded_at_segment_boundary_and_can_be_restored_again(
    tmp_path,
) -> None:
    """§5：取回天然是短时的——下一个段边界到来时被 `segment_fold` 折走（段内 raw 不受
    保护），而**占位仍在原位、ref 一直可见**，所以再取一次照样能拿到图。

    用真 `segment_fold`（生产函数，非桩）制造段边界，再经**真的 `MediaCapabilityProvider`
    实例**（runtime 构造期注册的那一个，走 registry 现解析 memory / blob store）取第二次。
    """
    blobs = FsBlobStore(tmp_path / "blobs")
    llm = _WireCapturingAnthropicLLM()
    runtime, memory, state = await _run_session(llm=llm, blob_store=blobs)
    ctxp = _pctx(state)
    (ref_a,) = tuple(llm.asked)

    before = await _view(memory, state)
    assert any(p.data == ref_a for r in before for p in _image_parts(r.content)), (
        "前提不成立：取回的图不在视图里")

    # 段边界：真 segment_fold 把末条 user 回合之后的 raw 折成一条段摘要
    result = await segment_fold(memory, state.scope, MemoryScope.TASK,
                                "the agent looked at the screenshot", ctxp)
    assert result.events_before > result.events_after, (
        f"段折叠什么都没折（{result.events_before} → {result.events_after}）")

    after = await _view(memory, state)
    assert not any(p.data == ref_a for r in after for p in _image_parts(r.content)), (
        "取回的图没有被段折叠收走——§5 的「取回是短时的」不成立")

    # 占位还在原位、ref 一直可见（user 回合受 `_protected` 保护）
    user_recs = [r for r in after if r.role == "user"]
    assert len(user_recs) == 1
    assert [ref for ref, _ in find_image_placeholders(_record_text(user_recs[0]))] == [ref_a]
    assert after.index(user_recs[0]) == before.index(
        next(r for r in before if r.role == "user")), "占位记录的位置变了"

    # 再取一次仍可用——走真的 provider 实例
    provider = next(p for p in runtime.providers.get_capability_providers()
                    if p.name == "media")
    events = [ev async for ev in provider.invoke("media:get_image", {"ref": ref_a}, ctxp)]
    assert [e.kind for e in events] == ["result"], f"provider 报错：{events}"
    payload = events[0].payload
    content = payload["content"]
    imgs = [p for p in (content if isinstance(content, list) else [])
            if getattr(p, "type", None) == "image"]
    assert len(imgs) == 1 and imgs[0].data == ref_a, (
        f"第二次取回没拿到图：content={content!r}")
    assert (await blobs.get(imgs[0].data, ctxp))[0] == _RAW_A


# ── 覆盖 4：两家 adapter 各自的形状 ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_both_adapters_carry_the_restored_image_in_their_own_wire_shape(
    tmp_path,
) -> None:
    """同一条 core 侧 `LLMMessage(role="tool", content=[TextPart, ImagePart])`，两家
    adapter 的出网形状不同（§4.3），差异全部关在 adapter 内：

    - Anthropic：`tool_result.content` 是 **block 列表**（text + image）；
    - OpenAI：tool 消息只发文本，图片改由**整段 tool 之后**追加的一条 user 消息承载。

    「整段之后」用**同批两个 tool call** 才验得出：追加的 user 消息若夹在两条 tool
    消息中间，真实 OpenAI 会回 `insufficient tool messages following tool_calls message`。
    """
    blobs_a = FsBlobStore(tmp_path / "blobs_a")
    anthropic = _WireCapturingAnthropicLLM()
    await _run_session(llm=anthropic, blob_store=blobs_a, batch_with_echo=True)
    blobs_o = FsBlobStore(tmp_path / "blobs_o")
    openai = _WireCapturingOpenAILLM()
    await _run_session(llm=openai, blob_store=blobs_o, batch_with_echo=True)

    # ── Anthropic：tool_result.content 是 block 列表 ──
    a_payload = anthropic.captured_payloads[anthropic.calls.index("finish")]
    a_results = _anthropic_tool_results(a_payload)
    assert len(a_results) == 2, f"同批两个 tool call 应产出两条 tool_result，实为 {len(a_results)}"
    with_image = [tr for tr in a_results
                  if isinstance(tr.get("content"), list)
                  and any(b.get("type") == "image" for b in tr["content"])]
    assert len(with_image) == 1
    kinds = [b.get("type") for b in with_image[0]["content"]]
    assert kinds == ["text", "image"], f"tool_result.content 的 block 顺序异常：{kinds}"
    assert base64.b64decode(with_image[0]["content"][1]["source"]["data"]) == _RAW_A
    # 纯文本那条**不**被改造成 block 列表（形态回归，见覆盖 5）
    text_only = next(tr for tr in a_results if tr is not with_image[0])
    assert text_only["content"] == _ECHO_TEXT, (
        f"纯文本 tool_result 的 wire 形态变了：{text_only['content']!r}")

    # ── OpenAI：追加的 user 消息在整段 tool 之后 ──
    o_payload = openai.captured_payloads[openai.calls.index("finish")]
    msgs = o_payload["messages"]
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    assert len(tool_idx) == 2, f"同批两个 tool call 应产出两条 tool 消息，实为 {len(tool_idx)}"
    # 关键性质：两条 tool 消息**相邻**，重定位的 user 消息在它们**整段之后**
    # （夹在中间 → 真实 OpenAI 回 insufficient tool messages following tool_calls message）
    rel_idx, rel_msg = _relocated_user_message(msgs)
    assert rel_idx == tool_idx[-1] + 1
    assert all(isinstance(msgs[i].get("content"), str) for i in tool_idx), (
        "OpenAI 的 tool 消息只接受文本，这里出现了非 str content")
    assert _TOOL_IMAGE_NOTICE.strip() in msgs[tool_idx[0]]["content"]
    assert _TOOL_IMAGE_NOTICE.strip() not in msgs[tool_idx[1]]["content"], (
        "纯文本那条 tool 消息也被加上了图片重定位标记")
    assert msgs[tool_idx[1]]["content"] == _ECHO_TEXT
    assert len(rel_msg["content"]) == 1
    assert _decode_data_url(rel_msg["content"][0]["image_url"]["url"]) == _RAW_A


# ── 覆盖 5：回归 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_regression_no_blob_store_keeps_everything_inline_and_text_tool_wire_intact(
) -> None:
    """回归两条：

    (a) **未注册 MemoryBlobStore 时全链路行为不变**——一次 L0.5 事件都没有、memory 里的图仍是
        **逐字节相同**的 inline base64、模型看不到任何占位（自然一次 `get_image` 都不调）、
        wire 上照常出网。主断言取**正向逐字节相等**而不是「没有出现 blob:」：后者对
        「图被整个丢掉」同样成立。
    (b) **纯文本工具结果的 wire 形态逐字节不变**——Task 3 放宽 `InvocationResult.content`
        之后，不带图的工具结果必须仍是一条 `str`：Anthropic 的
        `tool_result` 只有三个键且 `content` 是那段原文，OpenAI 的 tool 消息同理，
        且**其后不追加任何 user 消息**。
    """
    llm = _WireCapturingAnthropicLLM()
    llm.batch_with_echo = True
    # 没有 MemoryBlobStore → 没有占位可读 → stub 模型第一轮就 finish，probe__echo 调不到；
    # 故这一路单独用一个「先 echo 再 finish」的模型（见下）验 (b)。
    runtime, memory, state = await _run_session(llm=llm, blob_store=None)

    # (a) memory 与 wire 都还是 inline base64，逐字节相同
    view = await _view(memory, state)
    user_recs = [r for r in view if r.role == "user"]
    assert len(user_recs) == 1
    assert user_recs[0].content == _prompt(), (
        f"未注册 MemoryBlobStore 时 memory 记录必须与改造前逐字节一致，实为 {user_recs[0].content!r}")
    assert not await _l05_events(runtime, state), "未注册 MemoryBlobStore 却跑了 L0.5"
    assert llm.asked == set(), "未注册 MemoryBlobStore 时不该有占位可取"
    # 正向对照：图确实照常出网了（不是「什么都没发生」）
    wire_images = [b for p in llm.captured_payloads for m in p["messages"]
                   if isinstance(m.get("content"), list)
                   for b in m["content"] if isinstance(b, dict) and b.get("type") == "image"]
    assert wire_images, "未注册 MemoryBlobStore 时图片根本没出网——那不叫「行为不变」"
    decoded = {base64.b64decode(b["source"]["data"]) for b in wire_images}
    assert decoded == {_RAW_A, _RAW_B}, "inline base64 出网时内容变了"

    # (b) 纯文本工具结果：两家 adapter 的形态都不因「工具能返图」而改变
    await _assert_text_tool_wire_shape_unchanged()


class _EchoThenFinishLLM(_PlaceholderReadingLLM):
    """先调一次纯文本工具、再收尾——用来在 wire 上拿到一条**纯文本** tool_result。"""

    def _decide(self, payload: dict) -> tuple[str, list[ToolCall]]:
        if "control__update_task_metadata" in _tool_names(payload):
            self.calls.append("intent")
            return "", []
        if "echo" not in self.calls:
            self.calls.append("echo")
            return "calling echo", [ToolCall(id=self._tid(), name="probe__echo",
                                             arguments={})]
        self.calls.append("finish")
        return "done", [ToolCall(id=self._tid(), name="control__finish_task",
                                 arguments={"deliverables_summary": "d"})]


class _TextAnthropicLLM(_EchoThenFinishLLM, AnthropicMultimodalAdapter):
    def __init__(self) -> None:
        AnthropicMultimodalAdapter.__init__(self, api_key="test-key")
        self._init_stub()

    def complete(self, request, stream=True):
        payload = self._build_payload(request)
        self.captured_payloads.append(payload)
        return self._emit(*self._decide(payload))


class _TextOpenAILLM(_EchoThenFinishLLM, OpenAIMultimodalAdapter):
    def __init__(self) -> None:
        OpenAIMultimodalAdapter.__init__(self, api_key="test-key")
        self._init_stub()

    def complete(self, request, stream=True):
        payload = self._build_payload(request)
        self.captured_payloads.append(payload)
        return self._emit(*self._decide(payload))


async def _assert_text_tool_wire_shape_unchanged() -> None:
    a_llm = _TextAnthropicLLM()
    _rt, _mem, _st = await _run_session(llm=a_llm, blob_store=None,
                                        prompt=[TextPart(text="just text please")])
    a_payload = a_llm.captured_payloads[a_llm.calls.index("finish")]
    a_results = _anthropic_tool_results(a_payload)
    assert len(a_results) == 1, f"应恰好一条 tool_result，实为 {a_results}"
    tr = a_results[0]
    assert set(tr) == {"type", "tool_use_id", "content"}, f"tool_result 多/少了键：{sorted(tr)}"
    assert tr["content"] == _ECHO_TEXT, (
        f"纯文本 tool_result.content 不再是那段原文（也不再是 str）：{tr['content']!r}")

    o_llm = _TextOpenAILLM()
    _rt2, _mem2, _st2 = await _run_session(llm=o_llm, blob_store=None,
                                           prompt=[TextPart(text="just text please")])
    o_payload = o_llm.captured_payloads[o_llm.calls.index("finish")]
    msgs = o_payload["messages"]
    tool_msgs = [(i, m) for i, m in enumerate(msgs) if m.get("role") == "tool"]
    assert len(tool_msgs) == 1, f"应恰好一条 tool 消息，实为 {tool_msgs}"
    i, tm = tool_msgs[0]
    assert set(tm) == {"role", "tool_call_id", "content"}, f"tool 消息多/少了键：{sorted(tm)}"
    assert tm["content"] == _ECHO_TEXT, (
        f"纯文本 tool 消息的 content 变了：{tm['content']!r}")
    assert _TOOL_IMAGE_NOTICE.strip() not in tm["content"]
    following = msgs[i + 1:]
    assert not any(m.get("role") == "user" and isinstance(m.get("content"), list)
                   for m in following), (
        "纯文本工具结果之后 adapter 仍追加了一条重定位 user 消息")


# ── Task 4：恢复路径把 event ref 转回 memory ref ─────────────────────────────
#
# 事件 payload 里恒为 event ref；重放出来的 task.user_prompt 要被 driver ingest 进
# memory，必须先过 `Runtime._restore_task_prompts` 这座桥。本组用例直接构造事件流
# （同 `test_crash_recovery_reconcile.py` 的手法），绕开真实 LLM 跑一整轮的不确定性，
# 只钉住 `_recover_session_locked` 这一段的行为。
#
# 两个 blob store 用**互不相同的 ref 方案**（同 `test_multimodal_end_to_end.py` 的
# 机关）：入口从前把同一份字节双写进两个 store 并断言两边 ref 相同，任何「拿 event
# ref 去 memory 侧解」的隐藏跨命名空间引用都被「恰好相同」掩盖了；下面这组桩把 ref
# 方案刻意做得不同，掩盖不住。


class _Sha256MemoryBlobStore(MemoryBlobStore):
    """memory 侧：内容寻址的 ``blob:<sha256>``（与仓内真实实现同口径）。"""

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        import hashlib
        ref = f"{BLOB_REF_PREFIX}{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


class _PrefixedEventBlobStore(EventBlobStore):
    """event 侧：``blob:evt-<sha256>``——仍内容寻址（协议要求幂等），但命名空间独立。

    多出的 ``evt-`` 段是本组用例的全部机关：它让「这个 ref 是谁家的」变成可断言的
    事实，而不是靠两边碰巧算出同一个 sha。
    """

    def __init__(self) -> None:
        self.blobs: dict[str, tuple[bytes, str]] = {}

    async def put(self, data: bytes, media_type: str, ctx: ProviderContext) -> str:
        import hashlib
        ref = f"{BLOB_REF_PREFIX}evt-{hashlib.sha256(data).hexdigest()}"
        self.blobs[ref] = (data, media_type)
        return ref

    async def get(self, ref: str, ctx: ProviderContext):
        return self.blobs.get(ref)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="unused", tenant_id="default")


@pytest.fixture
def runtime_with_images():
    """runtime + 两个**不同实例、不同 ref 方案**的 blob store（Task 4 回归专用）。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=None, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    mem_store = _Sha256MemoryBlobStore()
    evt_store = _PrefixedEventBlobStore()
    runtime.providers.register_memory_blob_store(mem_store)
    runtime.providers.register_event_blob_store(evt_store)
    return runtime, mem_store, evt_store


def _seed_event(seq: int, sid: str, type_: EventType, *, task_id=None, ts=None, **payload) -> Event:
    return Event(
        id=f"evt_{seq:04d}", run_id="run_task4", sequence=seq, session_id=sid,
        type=type_, timestamp=ts or datetime(2026, 8, 28, tzinfo=timezone.utc),
        task_id=task_id, payload=payload,
    )


@pytest.mark.asyncio
async def test_recovery_converts_event_refs_into_memory_refs(runtime_with_images) -> None:
    """崩溃恢复后 task.user_prompt 必须是 **memory** ref——它会被 driver ingest 进 memory。

    重放直接拿到的是 event ref；不转换就等于把一个 memory 解不开的 ref 落进记忆。
    """
    runtime, mem_store, evt_store = runtime_with_images
    sid, tid, aid = "ses_t4a", "tsk_t4a", "agt_root"
    ref_a = await evt_store.put(_RAW_A, "image/png", _ctx())
    user_prompt_jsonable = content_to_jsonable([
        TextPart(text="look at this"),
        ImagePart(data=ref_a, media_type="image/png", source_type="ref"),
    ])

    events = [
        _seed_event(1, sid, EventType.SESSION_CREATED, user_prompt="look at this",
                    template_id="agent:tpl_echo", root_agent_id=aid),
        _seed_event(2, sid, EventType.RUN_STARTED),
        # status 刻意用非终态（ACTIVE）：终态 task 会让 recover_agent 判定「无可恢复
        # task」，同步走 finalize_idle_session 收尾并把 TaskManager 从
        # `runtime._task_managers` 里摘掉——本用例要在恢复**之后**立刻查 Task 对象，
        # 必须让它留在可恢复集合里、TM 保持挂着（`_register_and_drain` 用
        # `asyncio.create_task` 派发真正的执行，不 await 就不会被后台协程抢跑）。
        _seed_event(3, sid, EventType.TASK_CREATED, task={
            "id": tid, "status": "ACTIVE", "title": "T", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid,
            "user_prompt": user_prompt_jsonable}),
    ]
    for e in events:
        await runtime.event_store.append(e)

    await runtime.recover_agent(aid)

    tm = runtime._task_managers[sid]
    task = tm.get_task(tid)
    assert task is not None
    img = next(p for p in task.user_prompt if getattr(p, "source_type", "") == "ref")
    assert await mem_store.get(img.data, _ctx()) is not None      # memory 解得开
    assert await evt_store.get(img.data, _ctx()) is None          # 不是 event 的 ref


@pytest.mark.asyncio
async def test_recovery_populates_both_event_jsonable_fields_for_reopen(runtime_with_images) -> None:
    """恢复之后、任何 reopen 发生之前，两个 event jsonable 字段都必须已经被填好。

    `reopen_task` 首次 reopen 时会把 ``user_prompt_event_jsonable`` 快照进
    ``original_user_prompt_event_jsonable``；若恢复路径只填了其中一个、或一个都
    没填，被重开的携图任务其事件载荷会静默降级成纯文本，而不会有任何报错——这条
    用例直接钉住恢复后 `user_prompt_event_jsonable` **与**
    `original_user_prompt_event_jsonable` 都非 None，且内容是 **event 侧**形态
    （ref 归 evt_store 解，归 mem_store 解不开）。

    构造一个「崩在 reopen 之后」的持久态：TASK_CREATED（FINISHED）之后紧跟一条真实的
    TASK_REQUEUED（`user_prompt` / `original_user_prompt` 均携带 event ref 图片）——
    reducer 把 TASK_REQUEUED 的状态折成 PENDING（非终态），故不需要再补一条终态事件；
    留在可恢复集合里，TaskManager 才不会被 `finalize_idle_session` 同步收尾摘掉
    （摘掉后 `runtime._task_managers` 查不到、Task 对象也就无从断言）。本用例只验证
    `_restore_task_prompts` 这一段的落地结果，不依赖真实 LLM 跑完一轮
    （`recover_agent` 用 `asyncio.create_task` 派发真正的执行，不 await 就不会被
    后台协程抢跑）。
    """
    runtime, mem_store, evt_store = runtime_with_images
    sid, tid, aid = "ses_t4b", "tsk_t4b", "agt_root"
    ref_a = await evt_store.put(_RAW_A, "image/png", _ctx())
    original_jsonable = content_to_jsonable([
        TextPart(text="look at this"),
        ImagePart(data=ref_a, media_type="image/png", source_type="ref"),
    ])
    revised_jsonable = content_to_jsonable([
        TextPart(text="look at this"),
        ImagePart(data=ref_a, media_type="image/png", source_type="ref"),
        TextPart(text="\n\n## Revision required\nplease redo it"),
    ])

    events = [
        _seed_event(1, sid, EventType.SESSION_CREATED, user_prompt="look at this",
                    template_id="agent:tpl_echo", root_agent_id=aid),
        _seed_event(2, sid, EventType.RUN_STARTED),
        _seed_event(3, sid, EventType.TASK_CREATED, task={
            "id": tid, "status": "FINISHED", "title": "T", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid,
            "user_prompt": original_jsonable}),
        _seed_event(4, sid, EventType.TASK_REQUEUED, task_id=tid,
                    reason="observer_review_reopen",
                    user_prompt=revised_jsonable, original_user_prompt=original_jsonable),
    ]
    for e in events:
        await runtime.event_store.append(e)

    await runtime.recover_agent(aid)

    tm = runtime._task_managers[sid]
    task = tm.get_task(tid)
    assert task is not None

    for field_name in ("user_prompt_event_jsonable", "original_user_prompt_event_jsonable"):
        jsonable = getattr(task, field_name)
        assert jsonable is not None, f"{field_name} 恢复后必须非 None"
        images = [p for p in jsonable if isinstance(p, dict) and p.get("type") == "image"]
        assert images, f"{field_name} 里没有 image part：{jsonable}"
        ref = images[0]["data"]
        assert await evt_store.get(ref, _ctx()) is not None, (
            f"{field_name} 里的 ref 必须是 event 侧的、能被 evt_store 解开")
        assert await mem_store.get(ref, _ctx()) is None, (
            f"{field_name} 混进了 memory 侧 ref——它是事件侧载荷，命名空间不该相通")


# ── review 追加轮 1：hydrate_event_content 的「取不回字节」降级分支 ──────────────
#
# 这是 Task 4 新增的真实行为（恢复时取不回 event blob → 图变成确定性文本占位），
# 且是任务说明里点名要覆盖的边角。之前的两条用例都走「取得到字节」的主路径，
# 这条分支此前零覆盖。


@pytest.mark.asyncio
async def test_hydrate_event_content_degrades_to_placeholder_when_blob_missing() -> None:
    """event blob 取不回字节（过期 / 宿主换机 / GC 误删）→ 降级成确定性文本占位，不抛。"""
    from ctx_weft.core.utils.content import hydrate_event_content

    evt_store = _PrefixedEventBlobStore()
    # 刻意不 put：这个 ref 在 evt_store 里查无此物，get() 恒返回 None。
    missing_ref = f"{BLOB_REF_PREFIX}evt-does-not-exist"
    content = [
        TextPart(text="look at this"),
        ImagePart(data=missing_ref, media_type="image/png", source_type="ref"),
    ]

    out = await hydrate_event_content(content, event_blob_store=evt_store, ctx=_ctx())

    assert isinstance(out, list)
    images = [p for p in out if getattr(p, "source_type", "") == "ref"
              or getattr(p, "source_type", "") == "base64"]
    assert images == [], f"取不回字节时不该还留着图片 part：{out}"
    texts = [p.text for p in out if hasattr(p, "text")]
    assert "[image unavailable: image/png]" in texts, (
        f"应降级成确定性占位文本，实为：{texts}")


@pytest.mark.asyncio
async def test_hydrate_event_content_missing_blob_placeholder_is_deterministic() -> None:
    """占位文本对同一张图必须逐字节确定（用户裁定 D2 的硬约束），不得含 ref/sha。"""
    from ctx_weft.core.utils.content import hydrate_event_content

    evt_store = _PrefixedEventBlobStore()
    missing_ref = f"{BLOB_REF_PREFIX}evt-does-not-exist"
    part = ImagePart(data=missing_ref, media_type="image/png", source_type="ref")

    a = await hydrate_event_content([part], event_blob_store=evt_store, ctx=_ctx())
    b = await hydrate_event_content([part], event_blob_store=evt_store, ctx=_ctx())

    assert [p.text for p in a] == [p.text for p in b] == ["[image unavailable: image/png]"]
    assert missing_ref not in a[0].text


# ── review 追加轮 1：_restore_task_prompts 的 per-field 韧性 ────────────────────
#
# 裁定：per-task/per-field 转换失败必须只降级那一个字段，绝不中断整场恢复
# ——崩溃恢复恰是最不能再崩一次的地方。构造一个会让 `normalize_content` 抛出的
# task（畸形 base64，绕开 hydrate_event_content 只碰 ref part 的判据，直接从
# 「损坏的事件日志」这个角度进入 normalize_content 未被 try/except 保护的
# b64decode），验证：① 整场恢复仍然成功、② 该 task 的字段被降级成不含任何图片
# part 的纯文本（不让解不开的 ref 流进 memory）、③ 同一 session 里的另一个正常
# task 完好无损、④ 记了一条 error 日志点名 task id 与 field name。


@pytest.mark.asyncio
async def test_restore_task_prompts_isolates_one_bad_task_and_logs_error(
    runtime_with_images, caplog,
) -> None:
    runtime, mem_store, evt_store = runtime_with_images
    sid = "ses_t4c"
    good_tid, bad_tid, aid = "tsk_t4c_good", "tsk_t4c_bad", "agt_root"
    ref_a = await evt_store.put(_RAW_A, "image/png", _ctx())
    good_jsonable = content_to_jsonable([
        TextPart(text="good task"),
        ImagePart(data=ref_a, media_type="image/png", source_type="ref"),
    ])
    # 畸形事件载荷：source_type="base64" 却带着解不了的 data——正常入口（validate_content
    # 先行）不可能产出这种东西，这里刻意模拟「事件日志损坏 / EventBlobStore 有 bug」，
    # 绕开 hydrate_event_content（只处理 ref part），直接命中 normalize_content 那句
    # 刻意不做 try/except 的 b64decode(..., validate=True)。
    bad_jsonable = [
        {"type": "text", "text": "bad task"},
        {"type": "image", "data": "not-valid-base64!!", "media_type": "image/png",
         "source_type": "base64"},
    ]

    events = [
        _seed_event(1, sid, EventType.SESSION_CREATED, user_prompt="two tasks",
                    template_id="agent:tpl_echo", root_agent_id=aid),
        _seed_event(2, sid, EventType.RUN_STARTED),
        _seed_event(3, sid, EventType.TASK_CREATED, task={
            "id": good_tid, "status": "ACTIVE", "title": "Good", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid,
            "user_prompt": good_jsonable}),
        _seed_event(4, sid, EventType.TASK_CREATED, task={
            "id": bad_tid, "status": "ACTIVE", "title": "Bad", "kind": "reasoning",
            "assigned_agent_id": aid, "creator_agent_id": aid,
            "user_prompt": bad_jsonable}),
    ]
    for e in events:
        await runtime.event_store.append(e)

    import logging
    with caplog.at_level(logging.ERROR, logger="ctx_weft.core.runtime"):
        await runtime.recover_agent(aid)  # 必须不抛——整场恢复不能因一个 task 坏数据而死

    tm = runtime._task_managers[sid]

    good_task = tm.get_task(good_tid)
    assert good_task is not None
    good_img = next(p for p in good_task.user_prompt if getattr(p, "source_type", "") == "ref")
    assert await mem_store.get(good_img.data, _ctx()) is not None, (
        "正常 task 不该被同 session 里另一个坏 task 拖累")

    bad_task = tm.get_task(bad_tid)
    assert bad_task is not None
    assert isinstance(bad_task.user_prompt, list)
    bad_images = [p for p in bad_task.user_prompt
                  if getattr(p, "source_type", "") in ("ref", "base64")]
    assert bad_images == [], (
        f"降级后不该再留着任何图片 part（无论 ref 还是 base64）：{bad_task.user_prompt}")
    bad_texts = [p.text for p in bad_task.user_prompt if hasattr(p, "text")]
    assert any("[image image/png]" in t for t in bad_texts), (
        f"应降级成 downgrade_images_to_text 的确定性占位，实为：{bad_texts}")

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any(bad_tid in r.getMessage() and "user_prompt" in r.getMessage()
               for r in error_records), (
        f"必须有一条 error 日志点名坏 task 的 id 与字段名，实际记录：" +
        repr([r.getMessage() for r in error_records]))


# ── 全分支终审：C1 / I1 的跨任务接缝回归 ───────────────────────────────────────


class _CapturingBus:
    """只收事件、不落库的 event bus 桩：用来读 reopen 发出的 TASK_REQUEUED 载荷。"""

    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, event) -> None:  # noqa: ANN001
        self.events.append(event)


@pytest.mark.asyncio
async def test_recovery_then_reopen_preserves_text_only_prompt(runtime_with_images) -> None:
    """**纯文本** prompt 的 task：恢复之后被 reopen，原始指令不得从事件流里消失（C1）。

    `_restore_task_prompts` 曾对 ``isinstance(content, str)`` 直接 continue，导致
    ``user_prompt_event_jsonable`` 恢复后恒为 None；`reopen_task` 拿到 None 之后
    `_append_text_sections` 把它当「base 为空」，发出的 TASK_REQUEUED 只剩一句
    「## Revision required」——下一次重放据此重建 task，用户的原始指令就此蒸发。
    内存里当场看不出任何异常（`task.user_prompt` 仍是对的），所以断言必须落在
    **事件载荷**上，而不是 task 的 memory 侧字段。
    """
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task

    runtime, _mem_store, _evt_store = runtime_with_images
    sid, tid = "ses_c1", "tsk_c1"
    original = "写一份 Q3 周报，重点写风险项"
    task = Task(id=tid, session_id=sid, status="FINISHED", user_prompt=original)

    await runtime._restore_task_prompts([task], sid, "default")

    assert task.user_prompt_event_jsonable == original, (
        "纯文本 prompt 恢复后也必须有事件侧快照（str 往返即自身，零成本）")

    bus = _CapturingBus()
    tm = TaskManager(session_id=sid, event_bus=bus)
    tm.register_task(task)
    assert await tm.reopen_task(tid, reason="补上数据来源") is True

    requeued = next(e for e in bus.events if e.type == EventType.TASK_REQUEUED)
    assert requeued.payload["user_prompt"].startswith(original), (
        f"TASK_REQUEUED 丢了原始指令：{requeued.payload['user_prompt']!r}")
    assert "补上数据来源" in requeued.payload["user_prompt"]
    assert requeued.payload["original_user_prompt"] == original
    # 事件侧与 memory 侧对纯文本必须逐字节一致（重放重建出的 task 与在途 task 同形）。
    assert requeued.payload["user_prompt"] == task.user_prompt


@pytest.mark.asyncio
async def test_reopen_falls_back_to_original_prompt_when_jsonable_missing() -> None:
    """兜底：event jsonable 为 None 而 base 是非空 str 时，不得被当成「base 为空」（C1）。

    这是与上一条正交的第二道闸——即便日后又出现一条没填 `user_prompt_event_jsonable`
    的路径，「字段没填」也不该再伪装成「原始 prompt 是空的」。
    """
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task

    bus = _CapturingBus()
    tm = TaskManager(session_id="s_fb", event_bus=bus)
    task = Task(id="t_fb", session_id="s_fb", status="FINISHED",
                user_prompt="原始指令")
    assert task.user_prompt_event_jsonable is None  # 刻意不填
    tm.register_task(task)

    assert await tm.reopen_task("t_fb", reason="重做") is True

    requeued = next(e for e in bus.events if e.type == EventType.TASK_REQUEUED)
    assert requeued.payload["user_prompt"].startswith("原始指令")
    assert requeued.payload["original_user_prompt"] == "原始指令"


@pytest.mark.asyncio
async def test_recovery_degrades_event_refs_when_event_store_unregistered() -> None:
    """宿主重启后没再注册 EventBlobStore：event ref 必须降级，**不得**原样流进 memory 侧（I1）。

    这是解耦分支要消灭的最后一条跨命名空间通路：`hydrate_event_content` 早退、
    `normalize_content` 按设计不碰 ``source_type == "ref"``、也没有异常触发
    `_restore_task_prompts` 的降级分支，于是一个 event 命名空间的 ref 悄悄落进
    `task.user_prompt`，之后被 `rehydrate_content` 拿去问 `MemoryBlobStore`。
    """
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    runtime = make_runtime(llm=None, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    mem_store = _Sha256MemoryBlobStore()
    runtime.providers.register_memory_blob_store(mem_store)
    # 刻意不注册 EventBlobStore（NullEventBlobStore.can_externalize is False）

    from ctx_weft.core.models.task import Task

    evt_ref = f"{BLOB_REF_PREFIX}evt-orphan"
    task = Task(
        id="tsk_i1", session_id="ses_i1", status="ACTIVE",
        user_prompt=[
            TextPart(text="look at this"),
            ImagePart(data=evt_ref, media_type="image/png", source_type="ref"),
        ],
    )

    await runtime._restore_task_prompts([task], "ses_i1", "default")

    assert isinstance(task.user_prompt, list)
    leftovers = [p for p in task.user_prompt
                 if getattr(p, "source_type", "") in ("ref", "base64")]
    assert leftovers == [], (
        f"event ref 不得原样流进 memory 侧的 task 字段：{task.user_prompt}")
    texts = [p.text for p in task.user_prompt if hasattr(p, "text")]
    assert any("image/png" in t and t.startswith("[image") for t in texts), (
        f"应降级成确定性图片占位，实为：{texts}")


@pytest.mark.asyncio
async def test_hydrate_event_content_degrades_when_store_cannot_externalize(caplog) -> None:
    """`hydrate_event_content` 自身的口径：拿不到 store 与拿不回字节是同一种情形（I1）。"""
    import logging

    from ctx_weft.core.utils.content import hydrate_event_content
    from ctx_weft.protocols.events import NullEventBlobStore

    content = [
        TextPart(text="look at this"),
        ImagePart(data=f"{BLOB_REF_PREFIX}evt-x", media_type="image/png", source_type="ref"),
    ]
    with caplog.at_level(logging.WARNING, logger="ctx_weft.core.utils.content"):
        out = await hydrate_event_content(
            content, event_blob_store=NullEventBlobStore(), ctx=_ctx())

    assert isinstance(out, list)
    assert [p for p in out if getattr(p, "source_type", "") in ("ref", "base64")] == []
    assert any("[image unavailable: image/png]" == getattr(p, "text", None) for p in out)
    assert any(r.levelno >= logging.WARNING for r in caplog.records), (
        "降级必须是可见的、有日志的，而不是静默交接")
