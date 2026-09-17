"""上下文恢复：act 越过停机线 → 回 PrepareStep 压缩 → **原地续跑**，不吃一次 retry。

改造前，act 越过停机线（旧口径：硬编码 0.8 × effective_limit）会直接退出本段，经 observe
强制改判 retry、由 `_fold_retry_segment` 把整段 raw 折掉（keep_last=0），再 TASK_REQUEUED
重排——一次上下文耗尽 = 一次重试预算，而 `max_retries` 只有 3，`failure_counter` 还会跟着
涨。于是长任务在 `max_turns_per_act=50` 之前就先撞上下文墙、连撞三次即被判成 FAILED。

现在这条路改成：工具结果全部落定 → 回 prepare 跑一次升级式 compact + 重装配 → act 在压缩后
的 prompt 上继续，同一个 run、不吃 retry。本文件钉住这条路的四件事：
  1. 续跑真的发生（同一个 run 内 prepare→act→prepare→act），且没有 TASK_REQUEUED；
  2. **进度不丢**：压缩摘要与原始消息都出现在续跑那次请求里，模型真看见了；
  3. **无悬挂 tool_call**：越线那一轮的工具照执行完，配对完整，`drop_dangling_tool_calls`
     这个防御性兜底一次都不该命中（它命中即意味着上游对账漏补）；
  4. 终止性：compact 压不动时立刻放弃恢复、退回老路，不在 prepare↔act 之间空转。
"""

from __future__ import annotations

import dataclasses as _dc
import logging

import pytest

from ctx_weft.core.loop.steps import compact as cm
from ctx_weft.core.loop.steps.compact import COLLAPSE_DELIM, ParsedDigest
from ctx_weft.protocols import (
    AgentTemplate, IdentityFacet, LoopConfig, MemoryConfig,
    MemoryEventType as T, ProviderContext, ToolCall,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_compact_flow_e2e import _NOOP_NAME, _NoopToolProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

pytestmark = pytest.mark.asyncio

#: 摘要正文里的哨兵：断言它出现在**续跑那次请求**里，即「模型真的看见了压缩后的进度」。
#: 放在 Remaining 小节——Goal/Done 丢了还能从原始消息猜回来，Remaining 丢了就是进度丢了。
_REMAINING_MARK = "REMAINING-SENTINEL-must-survive-compaction"
_DONE_MARK = "Called the noop probe once"
_DIGEST = (
    "## Goal\nFinish the long task.\n"
    "## Constraints\nNone.\n"
    f"## Done\n{_DONE_MARK}\n"
    f"## Remaining\n{_REMAINING_MARK}\n"
)

_USER_PROMPT = "do a very long task"
#: 窗口 3000 / 预留 0 → eff=3000。停机线 0.9×3000=2700 < 注入的 2800（必越线）；
#: compact 触发线 0.8×3000=2400 ≤ 真实基线 2800（恢复落到 prepare 时门必开）。
_CTX_LIMIT, _INFLATED = 3000, 2800


def _template(**loop_kwargs) -> AgentTemplate:
    """act-only（无 observe facet）→ 机械退出走规则 observe，不必脚本化 observe LLM。

    `collapse_keep_last=1`：L3 的守卫是「task 层视图条数 > keep」，而越线那一轮只留下
    USER_PROMPT + assistant + tool_result 三条，默认的 3 压不动（各级全 noop → 恢复被
    `COMPACT_NOOP_KEY` 当场否决，测不到续跑）。调到 1 让 L3 在这个最小时间线上真的折。
    """
    lc = dict(collapse_keep_last=1, short_segment_token_threshold=0)
    lc.update(loop_kwargs)
    return AgentTemplate(
        id="tpl_recover", name="recover", version="0.1.0",
        identity={"act": IdentityFacet(text="You are a worker. Keep working.")},
        description="context-recovery e2e", capability_refs=[],
        memory_config=MemoryConfig(), loop_config=LoopConfig(**lc),
    )


class _CrossThenSettleMock(MockLLMAdapter):
    """前 `inflate_calls` 次调用把 usage 抬到越线值，之后回落真实值。

    一直抬着是测不出续跑的：act 每次续跑的第一轮都会立刻再越线，看到的只会是配额耗尽。
    另外留存**每次**请求的消息文本（`MockLLMAdapter.last_request` 只保最后一次），供
    「续跑那次请求里能看到压缩摘要」这条断言消费。
    """

    def __init__(self, *args, inflate_calls: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._inflate_calls = inflate_calls
        self._calls = 0
        self.seen: list[str] = []

    async def complete(self, request, stream: bool = True):
        self._calls += 1
        inflate = self._calls <= self._inflate_calls
        self.seen.append("\n".join(
            m.content if isinstance(m.content, str) else str(m.content)
            for m in request.messages))
        async for chunk in super().complete(request, stream=stream):
            if inflate and chunk.kind == "usage" and chunk.usage is not None:
                u = chunk.usage
                u = _dc.replace(
                    u, prompt_tokens=_INFLATED,
                    total_tokens=_INFLATED + u.completion_tokens,
                    input_tokens=max(0, _INFLATED - u.cache_read_tokens - u.cache_write_tokens))
                chunk = _dc.replace(chunk, usage=u)
            yield chunk


def _working(i: int) -> MockResponse:
    """一个「还在干活」的回合：有 tool call → 不走纯文本收尾那一支，会真撞停机线。"""
    return MockResponse(text="working on it",
                        tool_calls=[ToolCall(id=f"tc_{i}", name=_NOOP_NAME, arguments={})])


def _working_then_done() -> list[MockResponse]:
    """① 还在干活（撞停机线）；② 续跑后收尾。"""
    return [_working(1), MockResponse(text="Here is the final answer.")]


def _wire(monkeypatch, *, digest=_DIGEST, responses=None, inflate_calls=1, **loop_kwargs):
    """装好一个 runtime：摘要打桩免真实 LLM、注册 noop 工具与 in-memory provider。"""
    if digest is not None:
        async def _fake_summ(state, ctx, *, scope="task"):
            return ParsedDigest(text=digest, degraded=False, sections={})
        monkeypatch.setattr(cm, "summarize_for_compact", _fake_summ)
    # 后台 observe 与本文件无关，挡掉以免多消耗 mock 响应。
    monkeypatch.setattr(
        "ctx_weft.core.loop.steps.background_observe.launch_background_observe",
        lambda state, ctx, *, boundary: None)

    resolver = InlineAgentTemplateProvider()
    resolver.register(_template(**loop_kwargs))
    llm = _CrossThenSettleMock(responses=responses or _working_then_done(),
                               inflate_calls=inflate_calls,
                               context_limit=_CTX_LIMIT, output_reserve=0)
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())
    runtime.providers.register_capability(_NoopToolProvider())

    seen: list = []

    async def _observe(event) -> None:
        etype = event.type if isinstance(event.type, str) else event.type.value
        seen.append((etype, event.payload or {}))

    runtime.event_bus.subscribe(None, _observe)
    return runtime, llm, seen


def _steps(seen) -> list:
    return [p.get("step_name", "") for t, p in seen if t == EventType.STEP_STARTED.value]


def _types(seen, event_type) -> list:
    return [p for t, p in seen if t == event_type.value]


async def test_recovers_in_place_without_spending_a_retry(monkeypatch, caplog):
    """主路径：越线 → 回 prepare 压缩 → 续跑收尾。不吃 retry，且无悬挂 tool_call。"""
    runtime, llm, seen = _wire(monkeypatch)
    with caplog.at_level(logging.ERROR, logger="ctx_weft.core.loop.llm_gateway"):
        handle, state = await runtime.run_single_task(
            template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    # ① 续跑真的发生在**同一个 run 内**：prepare 出现两次，且第二次在第一段 act 之后。
    steps = _steps(seen)
    assert steps.count("prepare") == 2, f"应恰好恢复一次（prepare 跑两次），实际 {steps}"
    assert steps[:4] == ["prepare", "act", "prepare", "act"], steps

    # ② 没有吃掉重试预算，也没有被重排
    assert state.task.retry_count == 0
    assert not _types(seen, EventType.TASK_REQUEUED), "恢复不得走重排"
    assert state.task.status == "FINISHED", state.task.status

    # ③ 压缩真的跑了，且是 compact 触发（不是 pre_dispatch）
    started = _types(seen, EventType.MEMORY_COMPACT_STARTED)
    assert started and started[0]["trigger"] == "compact"
    assert _types(seen, EventType.MEMORY_COMPACTED), "该有实际折叠"

    # ④ 进度不丢：续跑那次请求里，压缩摘要与原始消息**同时**在场
    assert len(llm.seen) >= 2, "应有续跑那一次请求"
    resumed = llm.seen[1]
    assert _REMAINING_MARK in resumed, "压缩摘要的 Remaining 节没进续跑的 prompt = 进度丢了"
    assert _USER_PROMPT in resumed, "原始消息节没进续跑的 prompt"

    # ⑤ 无悬挂 tool_call：越线那轮的工具照执行完了，配对完整。
    #    （L3 折掉 assistant 却保住 tool_result 造成的**孤立 tool result** 是压缩的既有
    #     后果、由 drop_orphan_tool_results 静默清理；这里只禁 dangling 那一种。）
    assert "dangling tool_call" not in caplog.text, caplog.text

    # ⑥ 记忆里留下了坍缩物，且原始节有界（没有把摘要嵌套进原始节）
    mem = runtime.providers.get_memory()
    pctx = ProviderContext(session_id=state.session.id, agent_id=state.agent.id)
    ups = await mem.recall_recent(state.scope, [T.USER_PROMPT], 100, pctx)
    collapsed = [u.content for u in ups if COLLAPSE_DELIM in u.content]
    assert collapsed, "L3 应坍缩出带 COLLAPSE_DELIM 的 USER_PROMPT"
    assert all(c.count(_USER_PROMPT) == 1 for c in collapsed), "原始节有界，不嵌套膨胀"


async def test_tool_result_is_persisted_before_compaction(monkeypatch):
    """越线那一轮的工具**照执行完**：结果必须已落定（改造前它一条都不会有）。

    这是「先把工具结果拿到再压缩」的落点：不执行就没有 result，memory 里留下一条悬挂的
    assistant tool_call；改造前它被 `_fold_retry_segment`（keep_last=0）连整段一起 supersede
    而掩盖，一旦改成原地续跑就会暴露成 `drop_dangling_tool_calls` 的 ERROR + 整条 assistant
    被剥空删掉（模型看不到自己调过这些工具）。
    """
    runtime, llm, seen = _wire(monkeypatch)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    # 越线那一轮的 ACT_TURN_COMPLETED 用 context_limit 作 reason，但它排在工具执行**之后**
    completed = _types(seen, EventType.ACT_TURN_COMPLETED)
    assert any(p.get("reason") == "context_limit" for p in completed), completed

    mem = runtime.providers.get_memory()
    pctx = ProviderContext(session_id=state.session.id, agent_id=state.agent.id)
    # TOOL_RESULT 记录仍在，或其痕迹随坍缩物活下来——两者任一即证明工具真跑过。
    results = await mem.recall_recent(state.scope, [T.TOOL_RESULT], 100, pctx)
    ups = await mem.recall_recent(state.scope, [T.USER_PROMPT], 100, pctx)
    assert results or any(_DONE_MARK in u.content for u in ups), \
        "越线那一轮的工具结果必须已落定（执行点排在停机判定之前）"


async def test_turn_budget_is_not_reset_by_recovery(monkeypatch):
    """轮数预算跨恢复累计：`max_turns_per_act=1` 时，恢复后不该白送第二轮。

    预算重置的话 act 会在续跑段重新获得一整份 max_turns_per_act，一个 run 的总轮数就没有
    上界了（恢复几次就是几倍）。这里把预算压到 1：第一段用掉唯一那一轮 → 恢复后 range 为空
    → 立刻 max_turns 退出，`MAX_TURNS_REACHED` 必须发出。
    """
    runtime, llm, seen = _wire(monkeypatch, max_turns_per_act=1)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    assert _steps(seen).count("prepare") == 2, "仍应恢复一次"
    assert _types(seen, EventType.MAX_TURNS_REACHED), \
        "恢复后轮数预算已耗尽，必须落到 max_turns 退出而不是白送一轮"
    # 第二段 act 一次 LLM 都不该发：预算为 0 时循环体根本不进
    assert len(llm.seen) == 1, f"续跑段不该再发 LLM 请求，实际发了 {len(llm.seen)} 次"


async def test_gives_up_when_compaction_cannot_free_anything(monkeypatch):
    """终止性：恢复驱动的那次 compact 一无所获 → **不再**恢复第二次，退回 observe/retry。

    没有这条否决时，prepare↔act 会一路空转到配额耗尽，每圈烧一次真实 act LLM + 一次摘要
    调用。`collapse_keep_last=99` 让 L3 的 `task_n > keep` 永假（L1/L2 本就无可折胶囊）→
    每次 compact 都零折叠。

    配额是 2，所以「恰好恢复一次」（prepare 跑两次）才能区分两种停下的理由：若 noop 否决
    失效，配额会允许第二次恢复 → prepare 跑三次。usage 全程抬着，保证每段 act 都真越线。
    """
    runtime, llm, seen = _wire(monkeypatch, collapse_keep_last=99,
                               responses=[_working(i) for i in range(4)], inflate_calls=4)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    steps = _steps(seen)
    assert steps.count("prepare") == 2, f"应在第一次恢复无果后就放弃，实际 {steps}"
    assert "observe" in steps, steps
    # 退回老路：机械退出 → 强制 retry
    assert state.verdict is not None and state.verdict.task_outcome == "retry"
    # 只看 escalating_compact 那一类折叠（trigger="compact"）。退回老路后 observe 的
    # `_fold_retry_segment` 也会发 MEMORY_COMPACTED（trigger="observe_retry"）——那是 retry
    # 路径本来的动作，不是「压缩有效」的证据。本分支上它恰好不发（机械判决的 act_recap 为空
    # → 段保 raw），但断言不该依赖这个巧合：换个分支或改了机械判决就会误红。
    assert not [p for p in _types(seen, EventType.MEMORY_COMPACTED)
                if p.get("trigger") == "compact"], "恢复驱动的压缩本例应一无所获"


async def test_plain_text_finish_wins_over_the_stop_line(monkeypatch):
    """越线 + **纯文本**回合 → 按正常收尾处理，不当机械退出。

    改造前 context_limit 抢在纯文本那一支之前，代价是三重的：模型刚交出的答复被
    `_compose_final_outputs` 跳过而丢弃、observe 强制改判 retry、再白吃一次重试预算。
    纯文本意味着模型已经停止行动，压缩救不了「它不想继续」，该不该算做完是 observer 的事。
    """
    runtime, llm, seen = _wire(
        monkeypatch, responses=[MockResponse(text="Here is the final answer.")],
        inflate_calls=1)   # 唯一这一轮就越线
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    assert _steps(seen).count("prepare") == 1, "纯文本收尾不该触发恢复"
    assert state.task.status == "FINISHED", state.task.status
    assert not _types(seen, EventType.TASK_REQUEUED), "不得被改判成 retry 重排"
    assert "Here is the final answer." in (state.task.outputs or ""),         "越线不得吞掉模型刚交出的答复"


async def test_actor_done_wins_over_the_stop_line(monkeypatch):
    """越线 + **finish_task** 同一轮 → actor_done 胜出：任务已完成，没有可续的跑。"""
    runtime, llm, seen = _wire(monkeypatch, responses=[
        MockResponse(text="all done",
                     tool_calls=[ToolCall(id="tc_f", name="control__finish_task",
                                          arguments={"deliverables_summary": ""})]),
    ], inflate_calls=1)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    assert _steps(seen).count("prepare") == 1, "已收尾的 task 不该再压缩续跑"
    assert state.task.status == "FINISHED", state.task.status
    assert not _types(seen, EventType.TASK_REQUEUED)


async def test_suspend_wins_over_the_stop_line(monkeypatch):
    """越线 + **dispatch** 同一轮 → suspend 胜出：本 run 本来就要停在等子任务，没有续跑可言。

    父自身的上下文由 `_maybe_predispatch_compact` 那条独立的路负责（它排在 dispatch 执行
    之前，好让子 agent 继承到压缩后的快照）——不是这里。若恢复抢在 suspend 前面，父会先被
    拉回 prepare 压一遍再回 act，而 act 只会立刻又走到同一个挂起点，纯属多烧一次摘要。
    """
    runtime, llm, seen = _wire(monkeypatch, responses=[
        MockResponse(text="delegating",
                     tool_calls=[ToolCall(id="tc_d", name="control__delegate_task",
                                          arguments={"title": "sub", "description": "do part",
                                                     "task_prompt": "do part"})]),
    ], inflate_calls=1)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    steps = _steps(seen)
    assert steps.count("prepare") == 1, f"挂起的 run 不该被拉回 prepare 压缩，实际 {steps}"
    assert "suspend" in steps, steps
    assert "observe" not in steps, steps


async def test_quota_exhaustion_falls_back_to_the_retry_path(monkeypatch):
    """配额耗尽（每次压缩都有效、但每段都再越线）→ 第三次越线退回 observe → retry。

    与 `test_gives_up_when_compaction_cannot_free_anything` 成对：那条是「压不动」提前
    否决（恢复一次就停），这条是压得动但救不住、一路用满 `max_context_recoveries=2`
    （恢复两次 → prepare 跑三次）。两条走的是 `_can_recover_context` 的不同否决分支。
    """
    runtime, llm, seen = _wire(monkeypatch, max_context_recoveries=2,
                               responses=[_working(i) for i in range(6)], inflate_calls=6)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    assert _steps(seen).count("prepare") == 3, f"配额 2 → 恰好恢复两次，实际 {_steps(seen)}"
    assert [p for p in _types(seen, EventType.MEMORY_COMPACTED)
            if p.get("trigger") == "compact"], "本例的压缩是有效的（与 give-up 那条相区分）"
    assert state.verdict is not None and state.verdict.task_outcome == "retry"


async def test_recovery_disabled_keeps_legacy_retry_path(monkeypatch):
    """`max_context_recoveries=0` → 行为回到改造前：context_limit 直接退 observe → retry。"""
    runtime, llm, seen = _wire(monkeypatch, max_context_recoveries=0)
    handle, state = await runtime.run_single_task(
        template_id="agent:tpl_recover", user_prompt=_USER_PROMPT)

    assert _steps(seen).count("prepare") == 1
    assert state.verdict is not None and state.verdict.task_outcome == "retry"
