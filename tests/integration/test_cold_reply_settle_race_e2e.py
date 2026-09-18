"""端到端：冷应答落在 run 的收尾窗口里（审查文档 M5）。

park 信号要一路上抛到 `_run_task` 才被 TaskManager 看见，而运行槽位要到 `_settle` 才释放
（提前释放会让同一个 task 被派发两次）。落在这段收尾里的冷应答，其 `resume_task` 会被
「它还在跑」挡掉——答复还在，会话却停摆到下一次 `/resume`。

修法是把唤醒从「一次性推送」改成「可从状态推导」：收尾末尾自问一句「我等的那个问题有
答复了吗」，有就自己补一次重排。

两种结局按应答落点区分，两条都要恢复：
  · 落在兜底提交**之前** → 这一轮被提交，但 task 停在 AWAITING_HUMAN 永不重跑；
  · 落在兜底提交**之后** → 答复停在待终局，它开的那扇窗没人关，该 task 的事件全被挡住。

断言的是**普通订阅者**（host 的 SSE 就是这种，收不到未提交窗口里的事件）最终收到了什么，
而不只是内存状态——「前端有没有反应」才是这条 bug 的真实面目。
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols.events import EventFilter, EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_hitl_e2e_v2 import _all_request_text, _finish_call, _poll
from tests.integration.test_hitl_hot_reply_round_window_e2e import (
    _ask, _ask_user_results, _next_question, _ScriptedLLM, _stored_types,
)
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider, make_echo_template, make_runtime,
)

pytestmark = pytest.mark.asyncio


class _Gate:
    """卡住 run 收尾序列里的某一步，把竞态变成确定性的。"""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.used = False

    def wrap(self, real):
        async def gated(self_tm, task_id, *a, **k):
            if not self.used:
                self.used = True
                self.entered.set()
                await self.release.wait()
            return await real(self_tm, task_id, *a, **k)

        return gated


def _runtime(llm):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(hitl_timeout_sec=0))   # 零热窗 → 恒走冷路径
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


@pytest.mark.parametrize("gated_step", ["_commit_round_or_halt", "_flush_staged"],
                         ids=["before-commit", "after-commit"])
async def test_reply_during_the_settle_window_still_wakes_the_task(gated_step) -> None:
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), _finish_call()])
    rt = _runtime(llm)
    gate = _Gate()

    with mock.patch.object(TaskManager, gated_step,
                           gate.wrap(getattr(TaskManager, gated_step))):
        handle = await rt.start_session(SessionStartParams.create(
            template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
        sid = handle.session_id

        seen: list[tuple[str, str]] = []

        async def _subscribe() -> None:
            async for ev in handle.event_bus.stream(EventFilter()):
                if ev.session_id == sid:
                    seen.append((ev.type, (ev.payload or {}).get("hitl_id", "")))

        sub = asyncio.create_task(_subscribe())
        try:
            # run 已经 park（问题已开出），但还卡在收尾序列里 —— 正是那道竞态窗口。
            await asyncio.wait_for(gate.entered.wait(), timeout=5.0)
            q1 = await _next_question(rt, sid)
            view = await rt.reply_to_hitl(HitlReply(
                hitl_id=q1.id, outcome="accepted", agent_id=q1.agent_id, message="ANSWER"))
            assert view is not None
            tm = rt._task_managers[sid]
            assert q1.task_id in tm._running_tasks, (
                "这条应答必须落在「槽位还占着」的窗口里，否则这条测试没测到 M5")

            gate.release.set()

            # 修复前：这里永远等不到——重排被「还在跑」挡掉，没有第二个 run。
            await _poll(lambda: tm.get_task(q1.task_id).status == "FINISHED" or None,
                        timeout=10.0)
        finally:
            gate.release.set()
            sub.cancel()

    # ① 答复真的送到了模型
    assert "ANSWER" in _all_request_text(llm.act_requests[-1])
    results = await _ask_user_results(rt, sid, q1)
    assert len(results) == 1 and "ANSWER" in results[0], results

    # ② 普通订阅者（= host 的 SSE）最终收到了这一轮的事实
    types = [t for t, _ in seen]
    assert (EventType.HITL_RESOLVED, q1.id) in seen, "host 必须看到这条应答被终局"
    assert EventType.TASK_HUMAN_RESOLVED in types, "host 必须看到 task 不再等人"
    assert types.count(EventType.RUN_STARTED) >= 2, "被应答唤醒的那一轮 run 必须可见"

    # ③ 配对事实一对一：TaskAwaitingHuman{hitl} ↔ TaskHumanResolved{hitl}
    stored = await _stored_types(rt, sid)
    assert stored.count(EventType.TASK_AWAITING_HUMAN) == \
        stored.count(EventType.TASK_HUMAN_RESOLVED)

    # ④ 没有留下开着的窗
    assert not rt._task_managers[sid].open_round_task_ids


async def test_parking_without_an_answer_is_not_requeued() -> None:
    """反例：没人应答的正常冷 park，收尾自检不得把它重排起来（否则判据退化成无条件重排）。"""
    llm = _ScriptedLLM([_ask("tc1", "Q1?"), _finish_call()])
    rt = _runtime(llm)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="go", context_limit=100_000))
    sid = handle.session_id

    q1 = await _next_question(rt, sid)
    await asyncio.sleep(0.6)

    tm = rt._task_managers[sid]
    assert tm.get_task(q1.task_id).status == "AWAITING_HUMAN"
    stored = await _stored_types(rt, sid)
    assert stored.count(EventType.TASK_STARTED) == 1, "没有答复就不该有第二次派发"
    assert EventType.TASK_HUMAN_RESOLVED not in stored
    assert len(llm.act_requests) == 1
