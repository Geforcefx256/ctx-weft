"""ask_user（form=question）的答复带图时，图必须到达模型。

背景（提交 99afb15 / 6284929，「带图时不再静默丢图」）：此前 control_capability 用
`content_to_text` 展平人的答复、并把 metadata 硬写成 `{}`——非文本 part 被静默丢弃
（`content_to_text` 连占位都不留）。而带 part 的工具结果通道就在旁边，media:get_image
走的正是它。

**段 2 之后这条路换了主人**：`ask_user` 只 yield `needs_human`（`reply_as_result=True`），
等待与出口都归 `CapabilityGateway._human_reply_as_result`。本文件因此改钉在新出口上，
全程走真实路径：真 provider → 真 gateway → 真 HitlService → 出口 InvocationResult。
四条断言与旧版逐条对应：图存活、拒绝仍读得出是拒绝、纯文本逐字节不变、空答复回落
工具自己的确认文案（而不是变成 `(no output)`）。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.core.hitl.registry import HitlRegistry
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.hitl.service import HitlService
from ctx_weft.core.loop.capability_gateway import CapabilityGateway
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.hitl_waiter import HitlWaiter
from ctx_weft.core.capabilities.cache import CapabilityCache
from ctx_weft.core.capabilities.control_tools import (
    PROVIDER_NAME,
    ControlCapabilityProvider,
)
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.protocols import ImagePart, MemoryAddress, ProviderContext, TextPart
from ctx_weft.protocols.capability import ToolCapability
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

REF = "blob:" + "a" * 64
_AGENT_ID = "agt_1"
#: `ask_user` 自己的确认文案（`control_capability.ask_user` 的 `ControlResult.content`），
#: 经 `HitlAsk.prompt` 带到 gateway，作为空答复的兜底。
_TOOL_CONFIRMATION = "Human input requested (1 question(s))"


def _img() -> ImagePart:
    return ImagePart(data=REF, media_type="image/png", source_type="ref")


class _PassthroughNormalizer:
    async def __call__(self, content, session_id):
        return content, content


def _env():
    """真 provider + 真 gateway + 真 HITL 子系统。返回 (gateway, ctx, state, registry, service)。"""
    provider = ControlCapabilityProvider()
    session = Session(id="s1", tenant_id="default", user_prompt="do it", status="RUNNING")
    task = Task(id="tsk_1", session_id="s1", status="ACTIVE", title="T1")
    tm = SimpleNamespace(get_task=lambda tid: task)
    provider.register_session("s1", tm, session)

    registry = HitlRegistry()
    bus = InProcessEventBus()
    service = HitlService(registry=registry, event_bus=bus,
                          reply_intake=ReplyIntake(_PassthroughNormalizer()))
    cache = CapabilityCache()
    cache.put(_AGENT_ID, [ToolCapability(
        id=f"{PROVIDER_NAME}:ask_user", name="ask_user", kind="tool",
        description="Ask the user.")])
    gw = CapabilityGateway(
        capability_cache=cache, capability_providers=[provider],
        memory=InMemoryMemoryProvider(), event_bus=bus,
    )
    scope = MemoryAddress(session_id="s1", task_id="tsk_1", agent_id=_AGENT_ID)
    state = LoopState(
        run_id="r1", session=session, task=task,
        agent=SimpleNamespace(id=_AGENT_ID, template_id="tmpl_a", session_id="s1"),
        scope=scope,
        resolved_model=SimpleNamespace(model="mock", account=""),
    )
    ctx = LoopContext(
        assembler=None, llm=None, memory=InMemoryMemoryProvider(), event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default",
                                     task_id="tsk_1", agent_id=_AGENT_ID),
        hitl=service, waiter=HitlWaiter(registry),
    )
    return gw, ctx, state, registry, service


async def _ask_and_respond(reply_kwargs: dict):
    """跑一次真实的 ask_user，用 `reply_kwargs` 应答，返回 gateway 的 InvocationResult。"""
    gw, ctx, state, registry, service = _env()
    handle = asyncio.create_task(gw.invoke(
        f"{PROVIDER_NAME}__ask_user", {"questions": [{"question": "Which DB?"}]},
        state, ctx, tool_call_id="tc_1",
    ))
    for _ in range(200):
        pend = registry.list_pending()
        if pend:
            break
        await asyncio.sleep(0)
    else:
        raise AssertionError("no pending HITL request appeared")
    await service.resolve(HitlReply(hitl_id=pend[0].id, agent_id=pend[0].agent_id,
                                    **reply_kwargs))
    return await asyncio.wait_for(handle, timeout=1.0)


def _images(content) -> list:
    return [p for p in content if not hasattr(p, "text")] if isinstance(content, list) else []


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.text for p in content if hasattr(p, "text"))


async def test_accepted_answer_with_image_carries_the_part():
    """🔴 本文件存在的理由：图进最终 content，不再被展平掉。"""
    result = await _ask_and_respond(
        {"outcome": "accepted", "message": [TextPart(text="就是这个"), _img()]})
    assert result.is_error is False
    assert _text(result.content) == "就是这个"
    assert [p.data for p in _images(result.content)] == [REF]


async def test_rejected_answer_with_image_keeps_the_part_and_the_prefix():
    """拒绝路径的前缀必须并进首个 TextPart，不能用 f-string 把 parts 拍成 repr；
    也不能干脆不加前缀——那会把「人不同意」降级成「人说了句话」。"""
    result = await _ask_and_respond(
        {"outcome": "rejected", "message": [TextPart(text="不行"), _img()]})
    assert _text(result.content) == "Human declined: 不行"
    assert [p.data for p in _images(result.content)] == [REF]


async def test_plain_text_answer_is_byte_identical_and_has_no_parts_key():
    """纯文本路径逐字节不变（Global Constraint 第一条）。"""
    result = await _ask_and_respond({"outcome": "accepted", "message": "use postgres"})
    assert result.content == "use postgres"
    assert _images(result.content) == []


async def test_rejected_without_message_uses_default_sentence():
    result = await _ask_and_respond({"outcome": "rejected"})
    assert result.content == "Human rejected the request."
    assert _images(result.content) == []


async def test_accepted_empty_answer_falls_back_to_tool_confirmation_text():
    """空答复用工具自己的确认文案兜底，而不是把空字符串当答案送回模型
    （空串在 gateway 出口会变成 `(no output)`，读起来像工具坏了）。"""
    result = await _ask_and_respond({"outcome": "accepted", "message": ""})
    assert result.content == _TOOL_CONFIRMATION
    assert _images(result.content) == []
