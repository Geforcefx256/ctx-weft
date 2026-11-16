"""content_to_text 容忍 None/空 content（与 estimate_tokens 一致，防装配期崩溃）。

回归：某 Capability 的 description 为 None（如无描述的 sub-agent 模板）经 CapabilitySource
落成 ContextBlock.content=None，estimate_tokens(None) 静默返回 0、块照常生成，直到 composer
渲染 `content_to_text(b.content)` 才 `TypeError: 'NoneType' object is not iterable` 崩溃。
"""
from __future__ import annotations

from ctx_weft.protocols.content import content_to_text
from ctx_weft.protocols.capability import AgentCapability, Capability
from ctx_weft.protocols.content import TextPart


def test_str_passthrough():
    assert content_to_text("abc") == "abc"


def test_parts_joined():
    assert content_to_text([TextPart(text="a"), TextPart(text="b")]) == "ab"


def test_none_is_empty_string():
    # 契约：content 应为 str | list；None 视作空文本、绝不 raise
    assert content_to_text(None) == ""


def test_capability_description_none_normalized_to_empty():
    # 根因：dataclass 声明 description: str = ""；传入 None 应归一为 ""，
    # 使任何 provider 构造出的 Capability 都不把 None 泄进 ContextBlock.content。
    assert Capability(id="x:y", name="y", kind="tool", description=None).description == ""
    assert AgentCapability(id="agent:planner", name="planner",
                           template_name="planner", description=None).description == ""
