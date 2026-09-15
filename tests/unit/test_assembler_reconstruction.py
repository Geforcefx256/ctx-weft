"""Composer 对重建后 history blocks 的收尾整形。

- act purpose 的 actor prompt 必须以 user 结尾（history 以 assistant 收尾且无 Current Progress
  时垫续跑兜底 user；facet purpose 不垫，由各自 trailing cue 收尾）
- observer 复用 act 风格会话（含全部 task 轮次 + 派发日志），尾部追加 observe 指令消息

注：原 RecentMemorySource / AgentExperienceSource 的记录→回合重建单测已随两源删除；其活体
等价覆盖（tool_call 链、派发配对/隐去未配对、按 timestamp 归并）在 test_agent_recall_source.py。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import ContextBlock
from ctx_weft.core.assembler.composer import DefaultComposer

pytestmark = pytest.mark.asyncio


async def test_actor_messages_always_end_with_user() -> None:
    """act purpose 的 actor prompt 必须以 user 结尾——history 以 assistant 收尾且无 Current
    Progress 时垫续跑兜底 user（锚定任务 + 盘点已完成/只做剩余）。"""
    blocks = [
        ContextBlock(id="b1", source="x", kind="history", target="messages", content="hi",
                     priority=3, token_estimate=1, metadata={"role": "user", "timestamp": "1"}),
        ContextBlock(id="b2", source="x", kind="history", target="messages", content="thinking",
                     priority=3, token_estimate=1, metadata={"role": "assistant", "timestamp": "2"}),
    ]
    task = SimpleNamespace(user_prompt_in_memory=True, process_report=None,
                           title="X", description="", user_prompt="hi")
    request = SimpleNamespace(task=task, purpose="act")
    msgs = DefaultComposer()._build_actor_messages(blocks, request)
    assert msgs[-1].role == "user"
    assert "You are still working on the task: 'X'" in msgs[-1].content


async def test_observer_reuses_act_conversation_plus_observe_message() -> None:
    """观察者复用 act 风格会话（含全部 task 轮次 + 派发日志），尾部追加 observe 指令消息。"""
    def _blk(src, role, content, ts):
        return ContextBlock(id=f"b{ts}", source=src, kind="history", target="messages",
                            content=content, priority=3, token_estimate=1,
                            metadata={"role": role, "timestamp": ts})

    ident = ContextBlock(id="id", source="identity", kind="identity", target="system",
                         content="OBSERVER ROLE", priority=0, token_estimate=1, metadata={})
    blocks = [
        ident,
        _blk("task_conversation", "user", "round1 user", "1"),
        _blk("task_conversation", "assistant", "round1 reply", "2"),
        _blk("task_conversation", "assistant", "round2 reply", "4"),
        _blk("agent_experience", "assistant", "dispatch stuff", "5"),
    ]
    request = SimpleNamespace(
        task=SimpleNamespace(title="T", description="d", user_prompt="up",
                             user_prompt_in_memory=True, process_report=None),
        session=SimpleNamespace(user_prompt="up"),
    )
    msgs = DefaultComposer()._build_observer_messages(blocks, request)
    joined = "\n".join(m.content for m in msgs if isinstance(m.content, str))
    assert "round1 reply" in joined and "round2 reply" in joined  # 全部轮次
    assert "dispatch stuff" in joined                             # act 风格：派发日志一并复用
    assert "OBSERVER ROLE" in joined                              # 尾部 observe ROLE
    assert "report_task_outcome" in joined                        # 判定提示
