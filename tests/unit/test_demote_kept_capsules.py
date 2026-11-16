from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
import pytest

from tests.compaction_testkit import demote_kept_capsules
from ctx_weft.core.loop.task_memory_fold import task_record_kind
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import MemoryEvent, MemoryEventType as T, MemoryAddress, ProviderContext

_BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _pctx():
    return ProviderContext(session_id="s", tenant_id="tn")


@pytest.mark.asyncio
async def test_demote_drops_body_and_thins_finish_pair():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="root", agent_id="a")
    # task 层 body 的 metadata["task_id"] 由 provider 据 ingest 时的 scope.task_id 回填
    # （_to_record 用 stored.event.scope.task_id 覆盖），故子任务 c1 的 body 须用 c1 自己的 scope 灌入。
    c1_scope = MemoryAddress(session_id="s", task_id="c1", agent_id="a")
    # 同 agent 子任务 c1 的 rich 胶囊：task 层 body（含 c1 的 USER_PROMPT + 段摘要）
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=c1_scope, content="c1 请求",
                                 timestamp=_BASE, role="user", metadata={}), _pctx())
    await mem.ingest(MemoryEvent(type=T.TASK_COMPACT_SUMMARY, address=c1_scope, content="c1 段摘要",
                                 timestamp=_BASE + timedelta(seconds=1), role="assistant",
                                 metadata={}), _pctx())
    # agent 层 finish 对：assistant{act_recap + finish 调用} / tool{task_summary}
    await mem.ingest(MemoryEvent(type=T.AGENT_CONVERSATION_TURN, address=scope, content="c1 act_recap",
                                 timestamp=_BASE + timedelta(seconds=2), role="assistant",
                                 metadata={"origin_task_id": "c1", "parent_task_id": "root",
                                           "tool_calls": [{"id": "tc1", "name": "control:finish_task"}]}), _pctx())
    await mem.ingest(MemoryEvent(type=T.AGENT_CONVERSATION_TURN, address=scope, content="c1 综合总结",
                                 timestamp=_BASE + timedelta(seconds=2), role="tool",
                                 metadata={"origin_task_id": "c1", "parent_task_id": "root",
                                           "tool_call_id": "tc1"}), _pctx())

    state = SimpleNamespace(scope=scope, agent=SimpleNamespace())
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx())
    n = await demote_kept_capsules(state, ctx, {"c1"})
    assert n >= 2  # 至少折掉 task body 2 条 + assistant 槽

    body = await mem.recall_recent_by_agent(
        scope, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    assert not any(r.metadata.get("task_id") == "c1" for r in body)  # task body 全删

    turns = await mem.recall_recent(scope, [T.AGENT_CONVERSATION_TURN], 100, _pctx())
    c1_turns = [r for r in turns if r.metadata.get("origin_task_id") == "c1"]
    # 只剩一条 tool 回填（lean），assistant 段被折
    assert len(c1_turns) == 1 and task_record_kind(c1_turns[0]) == "finish"
    assert "c1 综合总结" in c1_turns[0].content
    assert "legacy_task_record" not in c1_turns[0].metadata
    assert c1_turns[0].content.startswith("<task_finish")
