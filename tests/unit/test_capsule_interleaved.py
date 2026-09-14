"""_synthesize_dispatch_pair 合成 finish 对（task-resident，spec 2026-06-28 §3.1）。

形态翻转（Task 1）：close **不再镜像** task 层 body 进 agent 层——只写 finish 对
（assistant finish_task tool_call + tool Process Report）。body 留各自 task 层。

本文件原「交错时间线镜像」相关测试（镜像顺序/原始 timestamp/LLM 与 TOOL 元数据保留/段摘要
role 映射）随 mirror 删除而移除——其验证的镜像机制已不存在。保留 finish-对结构/配对/result
切割/fail 前缀等仍适用于新模型的断言。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from ctx_weft.core.utils.estimate import estimate_tokens

import pytest

from ctx_weft.core.assembler.composer import DefaultComposer
from ctx_weft.core.assembler.sources.agent_recall import AgentRecallSource
from ctx_weft.core.loop.steps.finalize import _synthesize_dispatch_pair, _dispatch_ack
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import MemoryEvent, MemoryEventType, MemoryAddress, ProviderContext
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

pytestmark = pytest.mark.asyncio

T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _ctx() -> ProviderContext:
    return ProviderContext(session_id="s1", tenant_id="default")


def _task_scope(task_id="t1", agent="ag1") -> MemoryAddress:
    """task 层 scope（含 task_id）。"""
    return MemoryAddress(session_id="s1", task_id=task_id, agent_id=agent)


def _agent_scope(agent="ag1") -> MemoryAddress:
    """agent 层 scope（task_id=None）。"""
    return MemoryAddress(session_id="s1", task_id=None, agent_id=agent)


def _ev(type_, scope, content, t, role=None, **meta) -> MemoryEvent:
    return MemoryEvent(
        type=type_, address=scope, content=content,
        timestamp=_BASE + timedelta(seconds=t), role=role, metadata=meta,
    )


def _task(task_id="t1", agent="ag1", prompt="初始请求") -> Task:
    return Task(
        id=task_id, session_id="s1", status="FINISHED", tenant_id="default",
        assigned_agent_id=agent, creator_agent_id=agent, parent_task_id=None,
        title="测试任务", description="", user_prompt=prompt,
        settings=NormalTaskSettings(),
        outputs="最终答复",
    )


async def _caps(mem, agent_scope):
    """召回 agent 层 AGENT_CONVERSATION_TURN，时间序（oldest first）。"""
    recs = await mem.recall_recent(agent_scope, [T.AGENT_CONVERSATION_TURN], 100, _ctx())
    return list(reversed(recs))


# ── task-resident：合成只写 finish 对、不镜像 body ────────────────────────────

async def test_synthesize_writes_only_finish_pair():
    """task-resident：task 层有 UP/段摘要/LLM/TOOL 时，agent 层仍只写 finish 对（2 条）。
    body 不镜像、留 task 层。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope()
    asc = _agent_scope()

    # task 层：模拟后台 observe 已折好后的幸存事件（不会被镜像）
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "UP1原文", 1, role="user"), _ctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, tsc, "段①摘要", 2, role="assistant"), _ctx())
    await mem.ingest(_ev(T.USER_PROMPT, tsc, "HITL原文", 3, role="user"), _ctx())
    await mem.ingest(_ev(T.TASK_COMPACT_SUMMARY, tsc, "段②摘要", 4, role="assistant"), _ctx())

    task = _task(prompt="UP1原文")
    mem_content = "最终答复\n\nProcess Report: 过程报告"
    await _synthesize_dispatch_pair(mem, asc, task, "最终答复", "Process Report: 过程报告", "success", _ctx())

    caps = await _caps(mem, asc)

    # 仅 finish 对（assistant finish_task + tool Process Report）
    assert len(caps) == 2, f"expected only finish pair; got {[(c.role, c.content[:30]) for c in caps]}"
    assert caps[-2].role == "assistant"
    tool_calls = caps[-2].metadata.get("tool_calls", [])
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"].endswith("finish_task")
    # finish 对 assistant 槽 = act_recap（此处 4th 参传入 "最终答复"）；finish_task 无参标记
    assert caps[-2].content.startswith("最终答复")  # + PROCESS_RECAP_NOTE
    assert tool_calls[0]["input"] == {}
    assert caps[-1].role == "tool"
    # tool 槽 = `[task: '<title>' (<id>)] ` 归属前缀 + 过程报告
    assert caps[-1].content.startswith("[task: '测试任务' (t1)] Process Report:")
    assert all(c.metadata.get("origin_task_id") == "t1" for c in caps)

    # body 不镜像、留 task 层
    body = await mem.recall_recent(tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY], 100, _ctx())
    assert len(body) == 4, "task-layer body must stay (not mirrored/superseded)"


async def test_finish_pair_timestamp_anchors_close():
    """finish 对 timestamp 锚 close 时刻（now_utc），落在 task 层 body 之后。"""
    mem = InMemoryMemoryProvider()
    tsc = _task_scope()
    asc = _agent_scope()

    t2 = _BASE + timedelta(seconds=20)
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=tsc, content="q",
                                 timestamp=_BASE + timedelta(seconds=10), role="user"), _ctx())
    await mem.ingest(MemoryEvent(type=T.TASK_COMPACT_SUMMARY, address=tsc, content="s",
                                 timestamp=t2, role="assistant"), _ctx())

    task = _task(prompt="q")
    await _synthesize_dispatch_pair(mem, asc, task, "出了\n\nProcess Report: r", "", "success", _ctx())

    caps = await _caps(mem, asc)
    assert len(caps) == 2
    # finish pair timestamps >= t2 (now_utc at call time, after the task events)
    assert caps[-2].timestamp >= t2
    assert caps[-1].timestamp >= t2


# ── finish 对的 tool_call_id 配对 ─────────────────────────────────────────

async def test_finish_pair_tool_call_id_matches():
    """finish 对：assistant.tool_calls[0].id == tool.tool_call_id。"""
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task()

    await _synthesize_dispatch_pair(mem, asc, task, "ans\n\nProcess Report: rpt", "", "success", _ctx())

    caps = await _caps(mem, asc)
    finish_assistant = caps[-2]
    finish_tool = caps[-1]
    tc_id = finish_assistant.metadata["tool_calls"][0]["id"]
    assert finish_tool.metadata.get("tool_call_id") == tc_id


# ── fail 结局在 tool content 里有 [outcome=fail] 前缀 ─────────────────────

async def test_fail_outcome_prefix_in_tool_content():
    """outcome=fail 时 tool Process Report 内容带 [outcome=fail] 前缀。"""
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task()
    task.outputs = None

    await _synthesize_dispatch_pair(mem, asc, task, "失败报告", "", "fail", _ctx())

    caps = await _caps(mem, asc)
    tool_content = caps[-1].content
    assert "[outcome=fail]" in tool_content


# ── 同 agent 派发 ack：终态 + 内联导读 ─────────────────────────────────────

def test_dispatch_ack_carries_outcome_and_inline_pointer():
    """ack 在 close 时才写，已知 outcome → 让 delegate_task 这个调用解析出终态，
    而不是永远停在 'started'；并导读下方的内联执行。"""
    ack = _dispatch_ack("抽取 TokenStore", "success")
    assert "抽取 TokenStore" in ack
    assert "completed" in ack, f"ack must resolve the call to a terminal outcome; got {ack!r}"
    assert "inlined below" in ack, f"ack must point at the inline execution; got {ack!r}"
    assert "control__finish_task" in ack


def test_dispatch_ack_marks_failure():
    ack = _dispatch_ack("抽取 TokenStore", "fail")
    assert "FAILED" in ack, f"fail outcome must be explicit at the call site; got {ack!r}"
    assert "completed" not in ack


def test_dispatch_ack_carries_no_child_payload():
    """ack **不含**子任务产出：真实结果由内联 body + 嵌套 finish 对承载。
    塞进来既重复，又会因 reorder_tool_results_after_calls 排到 body 之前成倒叙。"""
    ack = _dispatch_ack("抽取 TokenStore", "success")
    assert len(ack) < 220, f"ack must stay a pointer, not carry the child's result; got {ack!r}"


# ── finish 对 tool 槽标明归属 task（消灭匿名 finish）───────────────────────

async def test_finish_tool_content_names_owning_task():
    """finish 对的 tool 槽带 `[task: '<title>' (<id>)]` 前缀标明归属。

    assistant 槽是无参收尾标记（`finish_task{}`，反转契约），自身不带任何归属信息；
    归属补在 tool 槽，经 tool_call_id 配对回其 assistant。
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task()

    await _synthesize_dispatch_pair(mem, asc, task, "过程复述", "综合总结", "success", _ctx())

    tool_content = (await _caps(mem, asc))[-1].content
    assert tool_content.startswith("[task: '测试任务' (t1)] "), (
        f"finish tool content must name its owning task; got {tool_content!r}"
    )
    assert "综合总结" in tool_content


async def test_finish_tool_content_names_task_before_fail_marker():
    """归属标记与 fail 标记共存，归属在前：`[task: 'X' (id)] [outcome=fail] …`。"""
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task()
    task.outputs = None

    await _synthesize_dispatch_pair(mem, asc, task, "失败报告", "", "fail", _ctx())

    tool_content = (await _caps(mem, asc))[-1].content
    assert tool_content.startswith("[task: '测试任务' (t1)] [outcome=fail] "), (
        f"expected task marker before fail marker; got {tool_content!r}"
    )


async def test_finish_tool_content_without_title_keeps_fail_marker():
    """title 为空（旧数据/未命名 task）→ 归属标记退化为裸 id，仍标明归属。

    改造前这里是「不产标记」；现在 id 恒在手上，印出来比留空有用——归属标记存在的
    理由就是消歧，而空标记消不了任何歧。
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task()
    task.title = ""
    task.outputs = None

    await _synthesize_dispatch_pair(mem, asc, task, "失败报告", "", "fail", _ctx())

    tool_content = (await _caps(mem, asc))[-1].content
    assert tool_content.startswith("[task: t1] [outcome=fail] "), (
        f"empty title must not emit an empty task marker; got {tool_content!r}"
    )


async def test_adjacent_finish_pairs_are_distinguishable():
    """嵌套（孙→子）相邻 finish 对：两条 tool 槽各自标明归属，读者可区分谁收的尾。

    这是本前缀存在的理由——同 agent 孙任务场景里父会连看到两组
    `[assistant finish_task{}][tool …]`，assistant 侧完全同形。
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    grand = _task(task_id="g1")
    grand.title = "补 TokenStore 单测"
    child = _task(task_id="c1")
    child.title = "抽取 TokenStore"

    await _synthesize_dispatch_pair(mem, asc, grand, "写了 8 个 case", "孙总结", "success", _ctx())
    await _synthesize_dispatch_pair(mem, asc, child, "抽出 TokenStore 类", "子总结", "success", _ctx())

    tools = [c for c in await _caps(mem, asc) if c.role == "tool"]
    assert len(tools) == 2, f"expected 2 finish tool slots; got {len(tools)}"
    assert "[task: '补 TokenStore 单测'" in tools[0].content
    assert "[task: '抽取 TokenStore'" in tools[1].content


# ── 无幸存 task 层记录时 finish 对仍写出 ─────────────────────────────────

async def test_no_task_records_still_writes_finish_pair():
    """task 层为空（清空或短 task 场景）时，仍写出 finish pair（2条）。"""
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task(prompt="")  # empty prompt means no UP mirror either
    task.outputs = "答"

    await _synthesize_dispatch_pair(mem, asc, task, "答\n\nProcess Report: r", "", "success", _ctx())

    caps = await _caps(mem, asc)
    assert len(caps) == 2, f"expected 2 (finish pair only), got {len(caps)}"
    assert caps[-2].role == "assistant"
    assert caps[-1].role == "tool"


# ── 回归：dict-list outputs 须正确提取进 blackboard mem_content（不得为空）─────────

async def test_dict_list_outputs_extracted_in_memory_content():
    """回归：task.outputs=[{"type":"text","text":...}] 时，_build_memory_content 须提取文本
    （blackboard 汇报给 parent 的内容），不能为空（旧 content_to_text 返回 "" 的 bug）。
    反转契约后答复不再进 finish 对，改由 blackboard mem_content / 内联 body 承载。"""
    from ctx_weft.core.loop.steps.finalize import _build_memory_content
    outputs = [{"type": "text", "text": "结构化答复"}]
    mc = _build_memory_content(outputs, "过程报告")
    assert mc.startswith("结构化答复"), (
        f"dict-list outputs must extract text into mem_content; got {mc!r}; "
        "old content_to_text bug would yield ''"
    )


async def test_embedded_process_report_in_outputs():
    """outputs 文本本身包含 "Process Report: " 时，必须用 full separator rsplit
    从最后一个分隔符切割，防止误切。

    示例：
      outputs = "I wrote a Process Report: draft"
      summary = "real report"
      mem_content = "I wrote a Process Report: draft\\n\\nProcess Report: real report"

    旧代码用 split("Process Report: ", 1)[-1] 取第一个分隔符后的内容：
      "draft\\n\\nProcess Report: real report"  (错误，report 被 outputs 污染)

    新代码用 rsplit("\\n\\nProcess Report: ", 1)[-1] 从最后分隔符切割：
      "real report"  (正确)
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope()
    task = _task()
    task.outputs = "I wrote a Process Report: draft"

    # _build_memory_content 会拼成：
    # "I wrote a Process Report: draft\\n\\nProcess Report: real report"
    mem_content = "I wrote a Process Report: draft\n\nProcess Report: real report"

    await _synthesize_dispatch_pair(mem, asc, task, "I wrote a Process Report: draft", "Process Report: real report", "success", _ctx())

    caps = await _caps(mem, asc)
    finish_tool = caps[-1]

    # 验证 tool content 从完整分隔符后切割，即只含 "real report"
    # 而非 "draft\\n\\nProcess Report: real report"（前缀为归属标记，见 _finish_report_prefix）
    tool_content = finish_tool.content
    assert tool_content == "[task: '测试任务' (t1)] Process Report: real report", (
        f"tool content should be 'Process Report: real report' (+ task marker) but got "
        f"{tool_content!r}; old split logic would include embedded separator content"
    )


# ── 端到端相邻性守护：同 agent 派发对 + 子任务胶囊 ────────────────────────────

async def test_e2e_same_agent_subtask_no_400() -> None:
    """端到端相邻性守护（spec 2026-06-30 §2.5）：parent delegate 同 agent 子任务，close 后经
    AgentRecallSource + DefaultComposer 归并，每个 tool message 紧跟其 assistant tool_call，
    且 delegate → _DISPATCH_ACK 严格相邻、子 body 排在 ack 之后、finish 对相邻。
    Guards the 400 fix end-to-end.
    """
    pctx = _ctx()
    mem = InMemoryMemoryProvider()

    def _ts(t):
        return _BASE + timedelta(seconds=t)

    p_tsc = _task_scope("P")     # parent task scope
    c_tsc = _task_scope("C")     # child task scope (same agent ag1)
    asc = _agent_scope()          # agent scope (task_id=None, agent_id=ag1)

    origin_tcid = "oc_adj_1"

    # T=1-2: Parent body (task layer)
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=p_tsc, content="parent task",
                                 timestamp=_ts(1), role="user"), pctx)
    await mem.ingest(MemoryEvent(type=T.LLM_RESPONSE, address=p_tsc, content="delegating to child",
                                 timestamp=_ts(2), role="assistant"), pctx)

    # T=3: Delegate assistant turn (agent layer, as written by gateway)
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=asc,
        content="", timestamp=_ts(3), role="assistant",
        metadata={
            "origin_task_id": "P", "parent_task_id": None,
            "tool_calls": [{"id": origin_tcid, "name": "control__delegate_task",
                            "input": {"title": "child task"}}],
        },
    ), pctx)

    # T=4-5: Child body (task layer - same agent, thus recalled by AgentRecallSource)
    await mem.ingest(MemoryEvent(type=T.USER_PROMPT, address=c_tsc, content="child task",
                                 timestamp=_ts(4), role="user"), pctx)
    await mem.ingest(MemoryEvent(type=T.LLM_RESPONSE, address=c_tsc, content="doing child work",
                                 timestamp=_ts(5), role="assistant"), pctx)

    # T=3 (back-dated): dispatch ack tool result, same timestamp as delegate → strictly adjacent
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=asc,
        content=_dispatch_ack("child task", "success"), timestamp=_ts(3), role="tool",
        metadata={
            "origin_task_id": "P",
            "tool_call_id": origin_tcid,
        },
    ), pctx)

    # T=6: Child finish pair (agent layer)
    finish_tcid = "ftcall_adj_1"
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=asc,
        content="child act recap", timestamp=_ts(6), role="assistant",
        metadata={
            "origin_task_id": "C", "parent_task_id": "P",
            "tool_calls": [{"id": finish_tcid, "name": "control__finish_task",
                            "input": {"result": "child done"}}],
        },
    ), pctx)
    await mem.ingest(MemoryEvent(
        type=T.AGENT_CONVERSATION_TURN, address=asc,
        content="child task summary", timestamp=_ts(6), role="tool",
        metadata={
            "origin_task_id": "C", "parent_task_id": "P",
            "tool_call_id": finish_tcid,
        },
    ), pctx)

    # Compose messages via real AgentRecallSource + DefaultComposer
    deps = SimpleNamespace(memory=mem, provider_ctx=pctx)
    req = SimpleNamespace(scope=asc, token_counter=estimate_tokens)
    blocks = [b async for b in AgentRecallSource().fetch(req, deps)]
    triples = DefaultComposer()._history_to_messages_with_sources(blocks)
    messages = [m for m, *_ in triples]

    # ── Helper: every tool message must immediately follow an assistant with matching tool_call ──
    def _assert_tool_follows_call(msgs):
        for i, m in enumerate(msgs):
            if m.role == "tool":
                prev = msgs[i - 1]
                assert prev.role == "assistant" and any(
                    tc["id"] == m.tool_call_id for tc in (prev.tool_calls or [])
                ), f"tool@{i} (id={m.tool_call_id!r}) 未紧跟其 assistant tool_call"

    _assert_tool_follows_call(messages)

    # ── delegate → _DISPATCH_ACK strictly adjacent ──
    delegate_idx = next(
        i for i, m in enumerate(messages)
        if m.role == "assistant" and any(
            tc.get("id") == origin_tcid for tc in (m.tool_calls or [])
        )
    )
    ack_idx = delegate_idx + 1
    assert messages[ack_idx].role == "tool", (
        f"expected tool (dispatch ack) at {ack_idx}, got role={messages[ack_idx].role!r}"
    )
    assert messages[ack_idx].content == _dispatch_ack("child task", "success"), (
        f"expected dispatch ack after delegate, got {messages[ack_idx].content!r}"
    )
    assert messages[ack_idx].tool_call_id == origin_tcid

    # ── child body comes AFTER the ack ──
    child_body_indices = [
        i for i, m in enumerate(messages)
        if m.role == "user" and "child" in (m.content or "")
    ]
    assert child_body_indices, "child body (user message) must appear in composed messages"
    assert all(ci > ack_idx for ci in child_body_indices), (
        f"child body must come after _DISPATCH_ACK @{ack_idx}; got child at {child_body_indices}"
    )

    # ── finish pair adjacent ──
    finish_call_idx = next(
        i for i, m in enumerate(messages)
        if m.role == "assistant" and any(
            tc.get("id") == finish_tcid for tc in (m.tool_calls or [])
        )
    )
    assert messages[finish_call_idx + 1].role == "tool", "finish pair tool must follow finish assistant"
    assert messages[finish_call_idx + 1].tool_call_id == finish_tcid, "finish pair tool_call_id must match"
