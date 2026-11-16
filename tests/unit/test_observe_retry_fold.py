from datetime import datetime, timedelta, UTC
from types import SimpleNamespace


from ctx_weft.core.loop.steps.observe import ObserveStep
from ctx_weft.providers.llm.tokenizer import HeuristicTokenizer
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import MemoryEvent, MemoryEventType as T, MemoryAddress, ProviderContext


_BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _pctx():
    return ProviderContext(session_id="s", tenant_id="tn", task_id="t1", agent_id="a")


def _llm():
    return SimpleNamespace(tokenizer=HeuristicTokenizer())


async def _ingest(mem, scope, typ, content, i, role="user"):
    await mem.ingest(MemoryEvent(type=typ, address=scope, content=content,
                                 timestamp=_BASE + timedelta(seconds=i), role=role,
                                 metadata={"task_id": scope.task_id}), _pctx())


async def test_retry_folds_current_attempt_and_deletes_raw():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "原始请求", 0)
    await _ingest(mem, scope, T.LLM_RESPONSE, "本轮回复", 1, role="assistant")
    await _ingest(mem, scope, T.TOOL_RESULT, "本轮工具结果", 2, role="tool")

    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"),
                            agent=SimpleNamespace(id="a", loop_config=SimpleNamespace(compact_keep_last=6)),
                            session=SimpleNamespace(id="s", tenant_id="tn"),
                            run_id="r1", sequence_counter=0, extra={})
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(), llm=_llm())
    verdict = SimpleNamespace(task_outcome="continue", act_recap="本段摘要：调了工具X", reported=False)

    events = []
    await ObserveStep()._fold_retry_segment(state, ctx, verdict, events)

    recs = await mem.recall_recent(scope, [T.USER_PROMPT, T.LLM_RESPONSE,
                                           T.TOOL_RESULT, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    types = {r.type for r in recs}
    # 本轮 raw 被折掉，USER_PROMPT 锚保留，新增一条段摘要
    assert T.LLM_RESPONSE not in types and T.TOOL_RESULT not in types
    assert T.USER_PROMPT in types
    summ = [r for r in recs if r.type == T.TASK_COMPACT_SUMMARY]
    assert len(summ) == 1 and summ[0].content == "本段摘要：调了工具X" and summ[0].role == "assistant"


async def test_retry_accumulates_prior_segments():
    """多轮 retry：本轮折只删本轮 raw，之前的段摘要保留累积（protect TASK_COMPACT_SUMMARY）。
    N 轮后 task 层 = USER_PROMPT + N 条段摘要（L3 再据 collapse_keep_last 坍缩），不是替换成 1 条。"""
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "原始请求", 0)
    # 上一轮 retry 已折出的段摘要①（assistant role，与前台折产物同形）
    await _ingest(mem, scope, T.TASK_COMPACT_SUMMARY, "段摘要①", 1, role="assistant")
    # 本轮 attempt 的 raw
    await _ingest(mem, scope, T.LLM_RESPONSE, "本轮回复", 2, role="assistant")
    await _ingest(mem, scope, T.TOOL_RESULT, "本轮工具结果", 3, role="tool")

    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"),
                            agent=SimpleNamespace(id="a", loop_config=SimpleNamespace(compact_keep_last=6)),
                            session=SimpleNamespace(id="s", tenant_id="tn"),
                            run_id="r1", sequence_counter=0, extra={})
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(), llm=_llm())
    verdict = SimpleNamespace(task_outcome="continue", act_recap="段摘要②", reported=False)

    events = []
    await ObserveStep()._fold_retry_segment(state, ctx, verdict, events)

    recs = await mem.recall_recent(scope, [T.USER_PROMPT, T.LLM_RESPONSE,
                                           T.TOOL_RESULT, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    types = {r.type for r in recs}
    assert T.LLM_RESPONSE not in types and T.TOOL_RESULT not in types  # 本轮 raw 折掉
    assert T.USER_PROMPT in types                                       # 锚保留
    summ = sorted((r for r in recs if r.type == T.TASK_COMPACT_SUMMARY),
                  key=lambda r: r.timestamp)
    contents = [r.content for r in summ]
    assert contents == ["段摘要①", "段摘要②"], "旧段摘要须保留，新段摘要追加（累积而非替换）"


async def test_retry_short_segment_kept_raw():
    """短段免折（与段边界折叠同门 is_short_segment）：本轮 attempt 的 active raw 低于
    short_segment_token_threshold → 不折、不写段摘要，raw 原样留给下个 attempt
    （recap 常比短原文更长，原文信息反而更全）。"""
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "原始请求", 0)
    await _ingest(mem, scope, T.LLM_RESPONSE, "一句话回复", 1, role="assistant")

    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"),
                            agent=SimpleNamespace(id="a", loop_config=SimpleNamespace(
                                compact_keep_last=6, short_segment_token_threshold=400)),
                            session=SimpleNamespace(id="s", tenant_id="tn"),
                            run_id="r1", sequence_counter=0, extra={})
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(), llm=_llm())
    verdict = SimpleNamespace(task_outcome="continue", act_recap="段摘要：短段不该写我", reported=False)

    events = []
    await ObserveStep()._fold_retry_segment(state, ctx, verdict, events)

    recs = await mem.recall_recent(scope, [T.LLM_RESPONSE, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    assert any(r.type == T.LLM_RESPONSE for r in recs), "短段 raw 应保留（未被 supersede）"
    assert not any(r.type == T.TASK_COMPACT_SUMMARY for r in recs), "短段不应写段摘要"
    assert events == []  # 未折 → 不发 MEMORY_COMPACTED


async def test_retry_long_segment_still_folds_when_threshold_set():
    """超过阈值的 attempt 照常折（门只放行短段）。

    段内需 ≥2 条 LLM 回复：单回复段无条件免折（is_short_segment 的单回复门）。"""
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.USER_PROMPT, "原始请求", 0)
    await _ingest(mem, scope, T.LLM_RESPONSE, "本轮回复", 1, role="assistant")
    await _ingest(mem, scope, T.LLM_RESPONSE, "本轮回复二", 2, role="assistant")

    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"),
                            agent=SimpleNamespace(id="a", loop_config=SimpleNamespace(
                                compact_keep_last=6, short_segment_token_threshold=1)),
                            session=SimpleNamespace(id="s", tenant_id="tn"),
                            run_id="r1", sequence_counter=0, extra={})
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(), llm=_llm())
    verdict = SimpleNamespace(task_outcome="continue", act_recap="段摘要", reported=False)

    events = []
    await ObserveStep()._fold_retry_segment(state, ctx, verdict, events)

    recs = await mem.recall_recent(scope, [T.LLM_RESPONSE, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    assert not any(r.type == T.LLM_RESPONSE for r in recs), "超阈值段应照常折叠"
    assert any(r.type == T.TASK_COMPACT_SUMMARY for r in recs)


async def test_non_retry_outcome_does_not_fold():
    mem = InMemoryMemoryProvider()
    scope = MemoryAddress(session_id="s", task_id="t1", agent_id="a")
    await _ingest(mem, scope, T.LLM_RESPONSE, "回复", 1, role="assistant")
    state = SimpleNamespace(scope=scope, task=SimpleNamespace(id="t1"),
                            agent=SimpleNamespace(loop_config=SimpleNamespace(compact_keep_last=6)),
                            session=SimpleNamespace())
    ctx = SimpleNamespace(memory=mem, provider_ctx=_pctx(), llm=_llm())
    verdict = SimpleNamespace(task_outcome="success", act_recap="done", reported=True)
    events = []
    await ObserveStep()._fold_retry_segment(state, ctx, verdict, events)
    recs = await mem.recall_recent(scope, [T.LLM_RESPONSE, T.TASK_COMPACT_SUMMARY], 100, _pctx())
    assert any(r.type == T.LLM_RESPONSE for r in recs)  # 未折
    assert not any(r.type == T.TASK_COMPACT_SUMMARY for r in recs)


def test_should_use_llm_on_root_context_limit():
    """root + `context_limit` 走 LLM observe。

    结论没变，**理由换了**：2026-09-28 前这是一条特例（`_should_use_llm` 里那句「机械退出即使
    root 也强制 LLM」），存在只为绕开同一函数里的 root 降级门——机械退出要一份可信 act_recap
    去当段摘要。root 门删掉之后那条特例成了死码，一并删了；现在它绿是因为**根本没有 root
    降级可绕**。
    """
    template = SimpleNamespace(identity={"observe": SimpleNamespace(text="ROLE")})
    state = SimpleNamespace(
        extra={"template": template},
        act_exit_reason="context_limit",
        task=SimpleNamespace(parent_task_id=None))  # root
    assert ObserveStep()._should_use_llm(state) is True
