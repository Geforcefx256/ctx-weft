"""G1-G4：嵌套 + 异步回填语序 golden（spec 2026-06-28 §2/§7）。

本设计的中心论断（不变量 A）：**锚定 timestamp + 既有 `(timestamp, seq_no)` 归并，
能在摘要/finish 记录异步迟到写入（拿到高 seq_no）的情况下，仍还原正确的嵌套先序语序**——
只要这些记录携带「逻辑时刻」timestamp（回锚纪律），而非 wall-clock 写入时刻。

G1（crux）：父 P 委派同 agent 子 C；C 的段摘要 / finish 对以「晚到的 ingest 顺序（高 seq_no）+
早的逻辑 ts」回填（模拟后台 observe 在 close 后 ~18s 才回填）。断言装配 message 序 =
  [P body…][P delegate 对][C body…][C finish 对][P 续跑…][P finish 对]
且 C 子树严格落在 P delegate 对与 P 续跑之间（不混入 P 后续回合）、tool 配对相邻不被夹断。
若排序是 seq_no-primary 或某记录用了写入时刻，本测试必失败。

G2：场景中的短叶子 task 保留 raw body。
G3：长 task 的 body = [user 锚点 + 段摘要]，末 raw 段被 supersede。
G4：跨 agent 子（不同 agent_id）的 body 不进父 prompt（仅黑盒 dispatch result）。

驱动 REAL `AgentRecallSource` + REAL `PromptBuilder` 历史投影
（生产排序 + tool_call/tool_call_id 无损重建），与 test_open_closed_recall.py /
test_dispatch_fold_golden.py 同范式。

关于回锚：生产 helper（_synthesize_dispatch_pair / capability_gateway）在「写入时刻」用
now_utc()——这对**同步**写入即逻辑时刻、正确；而对**异步回填**则须锚逻辑时刻。本测试不调那些
helper，而是按生产记录形态直接 ingest，并显式给出逻辑锚定 ts——以隔离地证明「只要回锚正确，
归并即还原语序」这一论断，同时核查既有锚定点的纪律（见 test_anchoring_discipline_*）。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from ctx_weft.core.loop.steps.finalize import (
    _supersede_final_raw_segment,
    _synthesize_dispatch_pair,
)
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.utils.estimate import estimate_tokens
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryEventType, ProviderContext
from ctx_weft.protocols.capability import qualify
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.assembler_testkit import history_messages, recalled_blocks


T = MemoryEventType
_BASE = datetime(2026, 1, 1, tzinfo=UTC)
SESSION = "s1"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pctx() -> ProviderContext:
    return ProviderContext(session_id=SESSION, tenant_id="default")


def _task_scope(task_id: str, agent_id: str) -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=task_id, agent_id=agent_id)


def _agent_scope(agent_id: str) -> MemoryAddress:
    return MemoryAddress(session_id=SESSION, task_id=None, agent_id=agent_id)


def _ts(t: float) -> datetime:
    return _BASE + timedelta(seconds=t)


async def _ingest(mem, type_, scope, content, t, role=None, **meta):
    """Ingest one record with an explicit LOGICAL-anchor timestamp _ts(t)."""
    await mem.ingest(
        MemoryEvent(type=type_, address=scope, content=content,
                    timestamp=_ts(t), role=role, metadata=meta),
        _pctx(),
    )


async def _ingest_delegate_pair(mem, parent_scope, *, tool_call_id, child_title,
                                anchor_t):
    """同 agent 委派对：agent 层 TASK_DISPATCH（assistant）+ TASK_DISPATCH_RESULT="…scheduled"。

    回锚纪律：派发对锚**派发时刻**（spec §2.2）。composer 把这一对都安到 TASK_DISPATCH 的 ts
    上（见 agent_recall.fetch），故二者相邻、子树落其后。这里两条都给 anchor_t（派发时刻）。
    """
    await _ingest(
        mem, T.TASK_DISPATCH, parent_scope, "", anchor_t, role="assistant",
        tool_call_id=tool_call_id,
        tool_name=qualify("control:delegate_task"),
        arguments={"title": child_title},
    )
    await _ingest(
        mem, T.TASK_DISPATCH_RESULT, parent_scope,
        f"Sub-task '{child_title}' scheduled.", anchor_t, role="tool",
        tool_call_id=tool_call_id, child_task_id="C", title=child_title,
    )


async def _ingest_finish_pair(mem, scope, *, task_id, parent_task_id, outputs,
                              report, close_t):
    """agent 层 finish 对（assistant finish_task + tool Process Report），锚 close 时刻 close_t。

    形态严格对齐生产 _synthesize_dispatch_pair 的两条 AGENT_CONVERSATION_TURN，但 timestamp 由
    调用方给逻辑锚（这里 = close_t），以便精确控制嵌套语序 + 模拟异步回填。
    """
    tcid = generate_id("tcall")
    await _ingest(
        mem, T.AGENT_CONVERSATION_TURN, scope, "", close_t, role="assistant",
        origin_task_id=task_id, parent_task_id=parent_task_id,
        tool_calls=[{"id": tcid, "name": qualify("control:finish_task"),
                     "input": {"result": outputs}}],
    )
    await _ingest(
        mem, T.AGENT_CONVERSATION_TURN, scope, f"Process Report: {report}", close_t,
        role="tool", origin_task_id=task_id, parent_task_id=parent_task_id,
        tool_call_id=tcid,
    )


async def _compose_messages(mem, agent_scope):
    """REAL AgentRecallSource.fetch → REAL PromptBuilder 排序/投影 → list[LLMMessage]."""
    deps = SimpleNamespace(memory=mem, provider_ctx=_pctx())
    req = SimpleNamespace(scope=agent_scope, token_counter=estimate_tokens)
    blocks = await recalled_blocks(req, deps)
    return history_messages(blocks, task_id=agent_scope.task_id or "")


def _make_task(task_id, agent_id, *, parent_task_id=None, creator_agent_id=None,
               status="FINISHED", outputs="答复"):
    t = Task(
        id=task_id, session_id=SESSION, status=status, tenant_id="default",
        assigned_agent_id=agent_id, creator_agent_id=creator_agent_id or agent_id,
        parent_task_id=parent_task_id, title="测试任务", description="",
        user_prompt="初始请求", settings=NormalTaskSettings(), outputs=outputs,
    )
    return t


# ════════════════════════════════════════════════════════════════════════════
# G1：嵌套 + 异步回填语序 golden（the crux）
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_g1_nested_order_survives_async_backfill() -> None:
    """父 P 委派同 agent 子 C；C 段摘要 + finish 对以「晚 ingest（高 seq_no）+ 早逻辑 ts」回填。

    逻辑时间线（_ts 秒）：
      t=1   P body：USER_PROMPT
      t=2   P body：LLM_RESPONSE（P 决定委派）
      t=3   P delegate 对（TASK_DISPATCH + …scheduled），锚派发时刻 t=3
      t=4   C body：USER_PROMPT
      t=5   C body：LLM_RESPONSE 段 1
      t=6   C body：TASK_COMPACT_SUMMARY（段摘要，逻辑锚 t=6；**异步回填**）
      t=7   C body：LLM_RESPONSE 段 2
      t=8   C finish 对（close 时刻 t=8；**异步回填**）
      t=9   P 续跑：LLM_RESPONSE
      t=10  P finish 对（close 时刻 t=10）

    **异步回填模拟**：C 的段摘要(t=6)、C finish 对(t=8) 在 P 续跑(t=9)/P finish(t=10) 之后才
    ingest → 它们在各自 scope 拿到更高 seq_no，但携带更早的逻辑 ts。若排序 seq_no-primary 或
    用写入时刻，C 子树会被错误地排到 P 续跑/finish 之后。正确（timestamp-primary + 回锚）应内联。

    断言装配序 = [P UP][P LLM][P delegate 对][C UP][C 段摘要][C LLM][C finish 对][P 续跑][P finish 对]
    """
    mem = InMemoryMemoryProvider()
    p_tsc = _task_scope("P", "ag1")
    c_tsc = _task_scope("C", "ag1")  # 同 agent → AgentRecallSource 跨 task 拉回
    asc = _agent_scope("ag1")

    # ── 先 ingest「同步」部分（正常时序 ingest）：P body、delegate 对、C 段1、P 续跑 ──
    await _ingest(mem, T.USER_PROMPT, p_tsc, "把 auth 改成 JWT，必要时拆子任务", 1, role="user")
    await _ingest(mem, T.LLM_RESPONSE, p_tsc, "我来委派子任务处理迁移", 2, role="assistant")
    await _ingest_delegate_pair(mem, asc, tool_call_id="dc1", child_title="JWT 迁移", anchor_t=3)
    await _ingest(mem, T.USER_PROMPT, c_tsc, "子任务：迁移 JWT", 4, role="user")
    c_seg1_id = await mem.ingest(
        MemoryEvent(type=T.LLM_RESPONSE, address=c_tsc, content="段1：读现状",
                    timestamp=_ts(5), role="assistant", metadata={}), _pctx())
    await _ingest(mem, T.LLM_RESPONSE, c_tsc, "段2：改完", 7, role="assistant")
    # P 续跑 + P finish（在 C 的异步回填之前就 ingest——让 C 回填拿到更高 seq_no）
    await _ingest(mem, T.LLM_RESPONSE, p_tsc, "子任务完成，我来收尾", 9, role="assistant")
    await _ingest_finish_pair(
        mem, asc, task_id="P", parent_task_id=None,
        outputs="JWT 迁移完成", report="P 成功收尾", close_t=10,
    )

    # ── 现在「异步回填」C 的段摘要(t=6) 和 C finish 对(t=8)：晚 ingest → 高 seq_no，但早 ts ──
    # （这正是「后台 observe 在 close 后 18s 回填段摘要」的危险点）
    # 段摘要锚到被折段 1 的逻辑时刻(t=6)，并 supersede 段 1 raw（对齐 apply_compact）。
    await _ingest(mem, T.TASK_COMPACT_SUMMARY, c_tsc, "段1 摘要：已读现状", 6, role="assistant")
    await mem.supersede([c_seg1_id], _pctx())
    await _ingest_finish_pair(
        mem, asc, task_id="C", parent_task_id="P",
        outputs="JWT 迁移子任务完成", report="C 成功", close_t=8,
    )

    messages = await _compose_messages(mem, asc)

    # 用 content 标记还原序列骨架
    def _label(m):
        c = m.content
        if "<task_start" in c:
            return "P:DELEGATE_call"
        if "<task_result" in c:
            return "P:DELEGATE_result"
        if c == "JWT 迁移子任务完成":
            return "C:FINISH_call"
        if c == "JWT 迁移完成":
            return "P:FINISH_call"
        if "<task_finish" in c:
            return "C:FINISH_report" if 'task_id="C"' in c else "P:FINISH_report"
        if m.role == "user" and "<task_recap" in c and "段1 摘要" in c:
            return "C:SUMMARY"
        if m.role == "user":
            if "auth 改成 JWT" in m.content:
                return "P:UP"
            if "迁移 JWT" in m.content:
                return "C:UP"
        if m.role == "assistant":
            c = m.content
            if "委派子任务" in c:
                return "P:LLM"
            if "段2" in c:
                return "C:LLM2"
            if "段1：读" in c:
                return "C:LLM1"
            if "收尾" in c:
                return "P:RESUME"
        return f"?{m.role}:{m.content[:20]}"

    labels = [_label(m) for m in messages]

    expected = [
        "P:UP", "P:LLM",
        "P:DELEGATE_call",
        "C:UP", "C:SUMMARY", "C:LLM2",
        "C:FINISH_call",
        "C:FINISH_report",
        "P:DELEGATE_result",
        "P:RESUME",
        "P:FINISH_call",
        "P:FINISH_report",
    ]
    assert labels == expected, f"nested order broke under async backfill:\n got={labels}\nwant={expected}"

    # ── 关键不变量断言（独立于上面的精确序，显式断言「设计中心论断」） ──
    pos = {lbl: i for i, lbl in enumerate(labels)}

    # (1) C 子树严格落在 P delegate 对 与 P 续跑 之间 —— NOT 混入 P 后续回合
    c_subtree = [pos["C:UP"], pos["C:SUMMARY"], pos["C:LLM2"],
                 pos["C:FINISH_call"], pos["C:FINISH_report"]]
    assert pos["P:DELEGATE_call"] < min(c_subtree), "C subtree must start after its start label"
    assert max(c_subtree) < pos["P:RESUME"], (
        "C subtree must end BEFORE P's resume (would mis-order if seq_no-primary / write-time ts)"
    )

    # Terminal labels follow the inline execution; only real tools need adjacency.
    assert max(c_subtree) < pos["P:DELEGATE_result"] < pos["P:RESUME"]
    # Every full capsule ends with its finish report before the parent resumes.
    assert pos["C:FINISH_call"] < pos["C:FINISH_report"]
    assert pos["P:FINISH_call"] < pos["P:FINISH_report"]

    # (4) tool 邻接：C 子树不被插进任一 assistant-tool_call 与其 tool_result 之间
    #     即每个 assistant(带 tool_calls) 后紧跟其 tool 结果
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.tool_calls:
            nxt = messages[i + 1]
            assert nxt.role == "tool", f"tool_call at {i} ({labels[i]}) not immediately followed by tool result"
            assert nxt.tool_call_id == m.tool_calls[0]["id"], (
                f"tool result at {i+1} ({labels[i+1]}) does not match preceding tool_call id"
            )


@pytest.mark.asyncio
async def test_g1_naive_seqno_primary_would_misorder() -> None:
    """对照：证明本场景下「seq_no-primary」排序会 visibly mis-order（即测试有判别力）。

    用与 G1 相同的 ingest（C 回填晚 → 高 seq_no），但故意改用 seq_no-primary 排序，断言
    C 子树会被错误推到 P 续跑/finish 之后 —— 这正是 timestamp-primary + 回锚所避免的。
    """
    mem = InMemoryMemoryProvider()
    p_tsc = _task_scope("P", "ag1")
    c_tsc = _task_scope("C", "ag1")
    asc = _agent_scope("ag1")

    await _ingest(mem, T.USER_PROMPT, p_tsc, "P 起", 1, role="user")
    await _ingest_delegate_pair(mem, asc, tool_call_id="dc1", child_title="C", anchor_t=3)
    await _ingest(mem, T.USER_PROMPT, c_tsc, "C 起", 4, role="user")
    await _ingest(mem, T.LLM_RESPONSE, p_tsc, "P 续跑", 9, role="assistant")
    # 异步回填：C finish 对晚 ingest（高 seq_no）但早 ts
    await _ingest_finish_pair(mem, asc, task_id="C", parent_task_id="P",
                              outputs="C done", report="C ok", close_t=8)

    deps = SimpleNamespace(memory=mem, provider_ctx=_pctx())
    req = SimpleNamespace(scope=asc, token_counter=estimate_tokens)
    blocks = await recalled_blocks(req, deps)

    # 正确排序（timestamp-primary）：C finish 在 P 续跑(t=9) 之前
    correct = sorted(blocks, key=lambda b: (b.value.timestamp,
                                            b.metadata.get("sequence", 0)))
    c_finish_correct = [i for i, b in enumerate(correct)
                        if b.value.role == "assistant"
                        and any(tc.get("name", "").endswith("finish_task")
                                for tc in b.value.metadata.get("tool_calls", []))]
    p_resume_correct = [i for i, b in enumerate(correct)
                        if b.value.content == "P 续跑"]
    assert c_finish_correct and p_resume_correct
    assert c_finish_correct[0] < p_resume_correct[0], (
        "timestamp-primary: C finish (logical t=8) must precede P resume (t=9)"
    )

    # 错误排序（seq_no-primary，跨 scope 不可比）：C finish 晚 ingest → 被推后
    naive = sorted(blocks, key=lambda b: (b.metadata.get("sequence", 0),
                                          b.value.timestamp))
    c_finish_naive = [i for i, b in enumerate(naive)
                      if b.value.role == "assistant"
                      and any(tc.get("name", "").endswith("finish_task")
                              for tc in b.value.metadata.get("tool_calls", []))]
    p_resume_naive = [i for i, b in enumerate(naive) if b.value.content == "P 续跑"]
    assert c_finish_naive[0] > p_resume_naive[0], (
        "seq_no-primary WOULD mis-order C finish after P resume — confirms test discriminates "
        "the design's central ordering claim"
    )


# ════════════════════════════════════════════════════════════════════════════
# 回锚纪律核查：既有锚定点用逻辑时刻（同步写入即逻辑时刻）
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_anchoring_discipline_finish_pair_anchors_close_time() -> None:
    """核查 _synthesize_dispatch_pair：finish 对两条 timestamp 同锚 close base（= now_utc()，
    同步写入即逻辑时刻，符合不变量 A），且 assistant/tool 同 ts → 由 scope 内 seq_no 定相邻序。
    """
    mem = InMemoryMemoryProvider()
    asc = _agent_scope("ag1")
    task = _make_task("P", "ag1", outputs="完成")
    await _synthesize_dispatch_pair(
        mem, asc, task, "完成了任务", "整体执行总结：ok", "success", _pctx())

    recs = await mem.recall_recent(asc, [T.AGENT_CONVERSATION_TURN], 100, _pctx())
    assert len(recs) == 2, f"finish pair = 2 records; got {len(recs)}"
    ts = {r.timestamp for r in recs}
    assert max(ts) - min(ts) == timedelta(microseconds=1)


# ════════════════════════════════════════════════════════════════════════════
# G2：短叶子 task 保留 raw body（嵌套场景内）
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_g2_short_leaf_keeps_raw_body() -> None:
    """短叶子 task close 后 raw body 原样留 task 层（spec §3.2）。

    短任务 → finalize 不调 _supersede_final_raw_segment → LLM_RESPONSE/TOOL_RESULT raw 仍在。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("leaf", "ag1")
    await _ingest(mem, T.USER_PROMPT, tsc, "小任务：报个数", 1, role="user")
    await _ingest(mem, T.LLM_RESPONSE, tsc, "答：42", 2, role="assistant")

    # 短任务：不 supersede（模拟 _close_one 的 short 分支 → 跳过 _supersede_final_raw_segment）
    body = await mem.recall_recent(tsc, [T.USER_PROMPT, T.LLM_RESPONSE], 100, _pctx())
    assert any("答：42" in (r.content or "") for r in body), (
        "short leaf must keep raw LLM_RESPONSE body in task layer"
    )
    asc = _agent_scope("ag1")
    msgs = await _compose_messages(mem, asc)
    assert any("答：42" in m.content for m in msgs), "short leaf raw body must be recalled"


# ════════════════════════════════════════════════════════════════════════════
# G3：长 task body 压末段（嵌套场景内）
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_g3_long_task_body_is_compacted_anchors_only() -> None:
    """长 task close：supersede 末 raw 段（LLM_RESPONSE/TOOL_INVOCATION/TOOL_RESULT），
    保留 USER_PROMPT + TASK_COMPACT_SUMMARY 锚点（spec §3.2）。装配 body = [user][段摘要]。
    """
    mem = InMemoryMemoryProvider()
    tsc = _task_scope("long", "ag1")
    asc = _agent_scope("ag1")
    await _ingest(mem, T.USER_PROMPT, tsc, "大任务：重构模块", 1, role="user")
    await _ingest(mem, T.TASK_COMPACT_SUMMARY, tsc, "中间段摘要：拆出 3 个子模块", 2, role="assistant")
    await _ingest(mem, T.LLM_RESPONSE, tsc, "末段 raw：最后清理", 3, role="assistant")
    await _ingest(mem, T.TOOL_RESULT, tsc, "末段 raw：测试通过", 4, role="tool")

    # 长任务 close → supersede 末 raw 段（签名收 provider_ctx，spec 2026-07-20 起 bg 侧共用）
    await _supersede_final_raw_segment(mem, tsc, _pctx())

    body = await mem.recall_recent(
        tsc, [T.USER_PROMPT, T.TASK_COMPACT_SUMMARY, T.LLM_RESPONSE, T.TOOL_RESULT], 100, _pctx())
    types = [r.type for r in body]
    assert T.USER_PROMPT in types, "long task must keep USER_PROMPT anchor"
    assert T.TASK_COMPACT_SUMMARY in types, "long task must keep TASK_COMPACT_SUMMARY anchor"
    assert T.LLM_RESPONSE not in types, "long task final raw LLM_RESPONSE must be superseded"
    assert T.TOOL_RESULT not in types, "long task final raw TOOL_RESULT must be superseded"

    msgs = await _compose_messages(mem, asc)
    contents = " ".join(m.content for m in msgs)
    assert "重构模块" in contents and "中间段摘要" in contents
    assert "末段 raw" not in contents, "compacted final raw segment must not appear in prompt"


# ════════════════════════════════════════════════════════════════════════════
# G4：跨 agent 子 body 不进父 prompt（黑盒）
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_g4_cross_agent_child_body_not_in_parent_prompt() -> None:
    """跨 agent 子（不同 agent_id）的 body 在子 agent 的 task 层；父 prompt 仅见黑盒 dispatch
    result（mem_content），不见子 body（spec §3.1 / G4）。
    """
    mem = InMemoryMemoryProvider()
    parent_agent, child_agent = "ag_p", "ag_c"
    p_tsc = _task_scope("P", parent_agent)
    c_tsc = _task_scope("C", child_agent)
    p_asc = _agent_scope(parent_agent)

    await _ingest(mem, T.USER_PROMPT, p_tsc, "父：协调收集", 0, role="user")
    # 父在 agent 层有跨 agent 委派对：result = 子的 mem_content（黑盒）
    await _ingest(mem, T.TASK_DISPATCH, p_asc, "", 1, role="assistant",
                  tool_call_id="xc1", tool_name=qualify("control:delegate_task"),
                  arguments={"title": "采集"})
    await _ingest(mem, T.TASK_DISPATCH_RESULT, p_asc,
                  "采集完成\n\nProcess Report: 收集 5 份。", 1, role="tool",
                  tool_call_id="xc1", child_task_id="C")
    # 子 body 在 child_agent 的 task 层（父 agent_id 召不到）
    await _ingest(mem, T.USER_PROMPT, c_tsc, "子机密：采集 X 数据", 2, role="user")
    await _ingest(mem, T.TASK_COMPACT_SUMMARY, c_tsc, "子机密摘要：已采 5 份", 3, role="assistant")

    msgs = await _compose_messages(mem, p_asc)
    contents = " ".join(m.content for m in msgs)

    assert "父：协调收集" in contents, "parent's own UP must be recalled"
    # 黑盒 result 可见
    assert "收集 5 份" in contents, "cross-agent dispatch result (black box) must be visible to parent"
    # 子 body 不可见
    assert "子机密" not in contents, (
        "cross-agent child task-layer body must NOT appear in parent prompt"
    )
