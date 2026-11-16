"""Per-session control-state reclamation (per-run token registry + TaskManager).

- B: when a session parks/suspends and goes idle, its per-run tokens are already gone
     (deregistered in `_SessionTaskRunner.execute`'s finally as the run itself ends),
     but the TaskManager is KEPT (still needed to cancel a paused session).
- A: cancelling an already-idle (parked) session reclaims the TaskManager too —
     cancel_all never fires the session-done callback, so it would otherwise linger.
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.providers.llm.adapters.mock import MockResponse
from tests._initialization import initialize_all
from tests.integration.test_outage_resume import (
    _FlakyLLM,
    _make_runtime,
    _new_params,
    _wait_for_run_finish,
)



@pytest.mark.asyncio
async def test_idle_park_reclaims_tokens_keeps_task_manager():
    # fail_for=1: the first act call raises LLMOutageError → task SUSPENDED → session idle.
    llm = _FlakyLLM(fail_for=1, responses=[], ri_responses=[MockResponse(text="")] * 3)
    rt = _make_runtime(llm)

    await initialize_all(rt)
    handle = await rt.start_session(_new_params())
    sid = handle.session_id
    await _wait_for_run_finish(handle)
    await asyncio.sleep(0.05)  # let the SUSPENDED branch + idle callback run

    # B — tokens reclaimed on idle-park; TaskManager retained for cancellability/resume query.
    assert sid not in rt._run_tokens, "per-run tokens deregister when the run parks"
    assert sid in rt._task_managers, "task_manager must be kept while the session is only paused"

    # A — 取消一个已空闲挂起的会话：清掉这一轮的控制信号，但**不逐出** TaskManager。
    #
    # 2026-09-08 生命周期改造前这里断言的是「cancel 必须回收 TaskManager」。现在回收
    # 只由显式 forget_session 触发：取消 = 这一轮不跑了，不等于这条会话不要了——用户
    # 多半还要看它的历史、甚至接着聊，而那时 TM 还在就不必重建一份。
    await initialize_all(rt)
    assert await rt.cancel_session(sid) is True
    assert sid not in rt._run_tokens, "per-run 控制信号照旧随取消清掉"
    assert sid in rt._task_managers, "取消不逐出：TaskManager 留到显式 forget_session"

    # A' — forget_session 才是回收入口，它把重对象一并拆掉。
    assert rt.forget_session(sid) is True, "已终结、已空闲 → 该能忘掉"
    assert sid not in rt._task_managers, "forget_session 必须回收 TaskManager"
    assert sid not in rt._run_tokens
    await initialize_all(rt)
    assert [item.data for item in await rt.list_agents(session_id=sid)] == [], "agent record 一并逐出"
