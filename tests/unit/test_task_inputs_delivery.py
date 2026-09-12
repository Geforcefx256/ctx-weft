"""spec: task-handoff——显式输入的上下文投递：metadata 载体 + Composer 双路径渲染。

四条执行路径在 composer 层收敛为两条渲染路径：fresh 构建（首次执行 / reopen 后
重跑——两者 user_prompt_in_memory 均为 False）与 in-memory 就地装饰（已有 memory
续跑 / 崩溃恢复重建执行）。恢复链上游（投影 → Task.inputs）由集成测试覆盖。
"""

from __future__ import annotations

from types import SimpleNamespace

from ctx_weft.core.assembler.assembler import ContextBlock, ContextRequest
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource

_INPUTS = {"file": "report.csv", "mode": "strict"}


def _task(**kw):
    base = dict(id="t1", title="", description="", user_prompt="",
                user_prompt_in_memory=False, process_report=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _req(task):
    return ContextRequest(
        purpose="act",
        scope=SimpleNamespace(session_id="s1", task_id=task.id, agent_id="a1"),
        task=task, agent=SimpleNamespace(id="a1"), session=SimpleNamespace(id="s1"),
        template=None, bound_capabilities=[], extra={},
    )


def _spec_block(inputs=None, **kw):
    md = {"task_id": "t1", "title": kw.get("title", ""), "description": kw.get("description", ""),
          "user_prompt": kw.get("user_prompt", "")}
    if inputs is not None:
        md["inputs"] = inputs
    return ContextBlock(
        id="ts1", source="task_spec", kind="task_spec", target="messages",
        content="snapshot-only", priority=0, token_estimate=1, metadata=md,
    )


def _hist_block():
    return ContextBlock(
        id="h1", source="agent_recall", kind="history", target="messages",
        content="the real user message", priority=3, token_estimate=4,
        metadata={"role": "user", "type": "user_prompt",
                  "timestamp": "2026-06-30T00:00:00+00:00"},
    )


def _text(msgs):
    return "\n".join(m.content for m in msgs if isinstance(m.content, str))


# ── Source：inputs 进 metadata（content 仍只是快照）──────────────────────────

async def test_task_spec_source_emits_inputs_in_metadata():
    task = _task(title="T", inputs=_INPUTS)
    blocks = [b async for b in TaskSpecSource().fetch(_req(task), None)]
    assert blocks[0].metadata["inputs"] == _INPUTS
    assert "## Inputs" in blocks[0].content          # 快照里也有（debug 可读）
    assert "report.csv" in blocks[0].content


async def test_task_spec_source_without_inputs_keeps_metadata_none():
    task = _task(title="T")
    blocks = [b async for b in TaskSpecSource().fetch(_req(task), None)]
    assert blocks[0].metadata["inputs"] is None
    assert "## Inputs" not in blocks[0].content


# ── 路径一：fresh 构建（首次执行 / reopen 重跑）──────────────────────────────

def test_fresh_path_renders_inputs_section_from_block():
    comp = DefaultComposer()
    task = _task(title="T", user_prompt_in_memory=False)
    blocks = [_spec_block(title="T", inputs=_INPUTS)]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    assert "## Inputs" in text
    assert "report.csv" in text and "strict" in text


def test_fresh_path_renders_truncation_marker_verbatim():
    # 规整发生在控制工具侧；composer 只负责把标记原样渲染出去
    from ctx_weft.core.capabilities.task_inputs import normalize_task_inputs
    inputs = normalize_task_inputs({"values": list(range(10_000))})
    comp = DefaultComposer()
    task = _task(title="T", user_prompt_in_memory=False)
    blocks = [_spec_block(title="T", inputs=inputs)]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    assert "## Inputs" in text
    assert "dropped" in text          # 裁剪哨兵原样可见
    assert "_truncated" in text       # 规整标记原样可见


def test_fresh_path_falls_back_to_task_inputs_without_block():
    comp = DefaultComposer()
    task = _task(title="T", inputs=_INPUTS, user_prompt_in_memory=False)
    text = _text(comp._build_actor_messages([], _req(task)))
    assert "## Inputs" in text and "report.csv" in text


def test_fresh_path_without_inputs_zero_change():
    comp = DefaultComposer()
    task = _task(title="T", user_prompt="msg", user_prompt_in_memory=False)
    blocks = [_spec_block(title="T")]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    assert "## Inputs" not in text


# ── 路径二：in-memory 就地装饰（已有 memory 续跑 / 恢复重建执行）────────────

def test_inmemory_path_prefixes_inputs_section():
    comp = DefaultComposer()
    task = _task(title="WRONG", user_prompt_in_memory=True)
    blocks = [_hist_block(), _spec_block(title="T", inputs=_INPUTS)]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    assert "## Inputs" in text and "report.csv" in text
    # 输入区块与任务框同前缀、钉在当前 task 的 user 回合上
    assert text.index("## Current Task") < text.index("## Inputs") < text.index("## Current Message")
    assert text.count("the real user message") == 1


def test_inmemory_path_without_inputs_zero_change():
    comp = DefaultComposer()
    task = _task(title="WRONG", user_prompt_in_memory=True)
    blocks = [_hist_block(), _spec_block(title="T")]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    assert "## Inputs" not in text
