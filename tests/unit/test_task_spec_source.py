"""TaskSpecSource → task_spec block (metadata 载体) + composer 读块装饰当前消息。

设计：spec(title/description/user_prompt) 经 TaskSpecSource 进 task_spec block 的 metadata；
composer 不把块当独立消息渲染，而是读 metadata 去装饰「当前 task」那条 user 回合。
无块时回退直读 request.task（向后兼容手构 blocks 的调用）。
"""
from types import SimpleNamespace

from ctx_weft.core.assembler.assembler import ContextBlock, ContextRequest
from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.task_spec import TaskSpecSource


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


def _spec_block(title="", description="", user_prompt=""):
    return ContextBlock(
        id="ts1", source="task_spec", kind="task_spec", target="messages",
        content="snapshot-only", priority=0, token_estimate=1,
        metadata={"task_id": "t1", "title": title, "description": description,
                  "user_prompt": user_prompt},
    )


def _text(msgs):
    return "\n".join(m.content for m in msgs if isinstance(m.content, str))


# ── Source ───────────────────────────────────────────────────────────────────

async def test_task_spec_source_emits_fields_in_metadata():
    task = _task(title="My Title", description="My Desc", user_prompt="hello")
    blocks = [b async for b in TaskSpecSource().fetch(_req(task), None)]
    assert len(blocks) == 1
    blk = blocks[0]
    assert blk.kind == "task_spec" and blk.target == "messages" and blk.priority == 0
    assert blk.metadata == {
        "task_id": "t1", "title": "My Title",
        "description": "My Desc", "user_prompt": "hello",
        # spec: task-handoff——inputs 也走 metadata（composer 两条渲染路径的读取面）
        "inputs": None,
    }


# ── Composer consumes the block (block wins over request.task) ────────────────

def test_fresh_path_reads_spec_from_block_not_task():
    # request.task carries WRONG values; the task_spec block must be the source of truth.
    comp = DefaultComposer()
    task = _task(title="TASK-WRONG", description="WRONG-DESC", user_prompt="WRONG-MSG",
                 user_prompt_in_memory=False)
    blocks = [_spec_block(title="BLOCK Title", description="BLOCK Desc",
                          user_prompt="BLOCK message")]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    assert "## Current Task\nBLOCK Title\nBLOCK Desc" in text
    assert "## Current Message\nBLOCK message" in text
    assert "WRONG" not in text  # nothing from request.task leaked


def test_inmemory_path_frames_existing_turn_with_block_title():
    comp = DefaultComposer()
    task = _task(title="WRONG", description="WRONG", user_prompt_in_memory=True)
    hist = ContextBlock(
        id="h1", source="agent_recall", kind="history", target="messages",
        content="the real user message", priority=3, token_estimate=4,
        metadata={"role": "user", "type": "user_prompt",
                  "timestamp": "2026-06-30T00:00:00+00:00"},
    )
    blocks = [hist, _spec_block(title="BLOCK Title", description="BLOCK Desc")]
    text = _text(comp._build_actor_messages(blocks, _req(task)))
    # in-place decoration uses the block's title + the history turn's raw message (once)
    assert "## Current Task\nBLOCK Title\nBLOCK Desc" in text
    assert "## Current Message\nthe real user message" in text
    assert text.count("the real user message") == 1
    assert "WRONG" not in text


# ── Backward compat: no block → fall back to request.task ─────────────────────

def test_fallback_to_task_when_no_spec_block():
    comp = DefaultComposer()
    task = _task(title="Fallback Title", description="Fallback Desc",
                 user_prompt="fallback msg", user_prompt_in_memory=False)
    text = _text(comp._build_actor_messages([], _req(task)))
    assert "## Current Task\nFallback Title\nFallback Desc" in text
    assert "## Current Message\nfallback msg" in text
