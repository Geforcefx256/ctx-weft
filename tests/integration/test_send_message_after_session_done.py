"""回归：会话真的跑完之后再 `send_message`，必须能开出新一轮——两种内存状态都要能。

这里锁的是 2026-09-08 生命周期改造的结果，覆盖两条路：

① **跑完不逐出**（新契约）：`on_task_finished` → `is_done()` 且无 parked task →
   `_fire_session_done` → `on_session_done` 现在只剩 `_release_round`（清这一轮的
   per-run 控制信号），TaskManager 与 agent record 都留着。所以「跑完再发」是纯粹的
   热路径，一次事件日志都不用读。

   改造前这条链是 `_release_session`：`_task_managers.pop` +
   `ALM.release_session`（摘掉该 session 全部 record），于是 `send_message` 在第一行
   `assert_can_receive` 就抛 `AgentNotFound: unknown agent`，而它下游
   `_start_task_for_agent` 里那套「TM 没了就 `recover_agent(keep_alive=True)`」的自愈
   永远执行不到。宿主侧表现：一条正常结束的会话再发消息 → 404，且宿主已乐观把 entry
   置成 RUNNING，此后每次发送都被自己的 409 闸门挡掉，会话就此钉死。

② **显式逐出之后仍能自愈**：持有方调 `forget_session`（host 的空闲淘汰会这么做）把
   会话踢出内存后再发消息，`send_message` 必须靠 `_hydrate_agent_for_send` 从事件
   日志把它装填回来，而不是抛 `AgentNotFound`。

与 `tests/unit/test_send_message_after_cold_recover.py` 的分工：那边手动把 task 置
FINISHED 来模拟"上一条消息已经跑完"，走的是纯内存路径；这里跑真实收尾链。
"""
from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import ToolCall
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)

pytestmark = pytest.mark.asyncio


class _FinishEveryTaskLLM(MockLLMAdapter):
    """每次 act 都 finish_task；observe/背景折叠各自给出最简合法答复。"""

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def complete(self, request, stream=True):
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            return self._stream(MockResponse(text=""), request)
        if "control__report_task_outcome" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("obs"), name="control__report_task_outcome",
                         arguments={"task_status": "success", "act_recap": "done",
                                    "task_summary": "done"}),
            ]), request)
        if "control__collect_process_report" in names:
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("bg"), name="control__collect_process_report",
                         arguments={"act_recap": "segment summary"}),
            ]), request)
        # 交付物 = 收尾回合正文（finish_task 不带未声明参数，spec: capability-gateway）
        return self._stream(MockResponse(text="done", tool_calls=[
            ToolCall(id=self._id("fin"), name="control__finish_task",
                     arguments={}),
        ]), request)


async def _wait_until(predicate, timeout=10.0, interval=0.02):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval)
    raise TimeoutError("condition not met within timeout")


def _make_runtime() -> CtxWeftRuntime:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=_FinishEveryTaskLLM(), agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def test_send_message_works_after_the_session_was_released():
    rt = _make_runtime()
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="first question", context_limit=100_000,
    ))
    sid, aid, first_task = handle.session_id, handle.agent_id, handle.task_id

    async def _first_done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(first_task)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_first_done)
    # 收尾链是异步的（_fire_session_done 先 gather 后台协程再回调）——给它跑完的机会，
    # 否则下面「没被拆掉」的断言会因为收尾还没发生而假绿。
    await asyncio.sleep(0.3)

    # ── 新契约：跑完**不**逐出。TM 与 agent record 都还在。 ────────────────────
    assert rt._agent_lifecycle_manager.has(aid), "跑完不该摘 agent record"
    assert sid in rt._task_managers, "跑完不该拆 TaskManager"
    assert [a.agent_id for a in rt.list_agents(session_id=sid)] == [aid]
    assert rt.get_agent(aid).status == "idle", "task 终态不是 agent 终态（spec 3.1）"

    # ── 核心断言：这一句改造前抛 AgentNotFound ────────────────────────────────
    second = await rt.send_message(aid, "second question", session_id=sid)

    assert second.agent_id == aid
    assert second.session_id == sid
    assert second.task_id and second.task_id != first_task, "必须开出新一轮，而不是并进旧 task"
    assert rt._task_managers[sid] is not None

    async def _second_done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(second.task_id)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_second_done)


async def test_send_message_self_heals_after_an_explicit_forget_session():
    """持有方显式逐出（host 的空闲淘汰）之后再发消息，必须从事件日志自愈装填回来。

    这是 `_hydrate_agent_for_send` 在新生命周期下的**主要**理由：跑完不再自动逐出，
    但持有方会主动收，收完之后这条路必须还通。
    """
    rt = _make_runtime()
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="first question", context_limit=100_000,
    ))
    sid, aid, first_task = handle.session_id, handle.agent_id, handle.task_id

    async def _first_done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(first_task)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_first_done)
    await asyncio.sleep(0.3)

    rt.forget_session(sid)
    assert not rt._agent_lifecycle_manager.has(aid)
    assert sid not in rt._task_managers
    assert rt.list_agents(session_id=sid) == []

    second = await rt.send_message(aid, "second question", session_id=sid)
    assert second.task_id and second.task_id != first_task
    assert rt._agent_lifecycle_manager.has(aid), "自愈必须把 record 装填回来"
    assert sid in rt._task_managers, "TM 那一半由 _start_task_for_agent 的探测接住"


async def test_send_message_without_session_id_still_self_heals():
    """没有 session 语境的调用方回落 `rebuild_agent` 的全量 sweep——慢，但不该失败。"""
    rt = _make_runtime()
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="first question", context_limit=100_000,
    ))
    sid, aid, first_task = handle.session_id, handle.agent_id, handle.task_id

    async def _first_done() -> bool:
        view = await rebuild_view(rt.event_store, sid)
        t = view.tasks.get(first_task)
        return t is not None and t.status == "FINISHED"

    await _wait_until(_first_done)
    await asyncio.sleep(0.3)
    rt.forget_session(sid)

    second = await rt.send_message(aid, "second question")   # 不传 session_id
    assert second.task_id and second.task_id != first_task


async def test_truly_unknown_agent_still_raises_agent_not_found():
    """自愈是「还没喂进来」的补救，不是把 `AgentNotFound` 废掉：事件日志里也没有的
    agent_id 照常抛——`send_message` 的既有契约（upgrade doc §2）不变。"""
    from ctx_weft.core.models.errors import AgentNotFound

    rt = _make_runtime()
    with pytest.raises(AgentNotFound):
        await rt.send_message("agt_does_not_exist", "hello")
    with pytest.raises(AgentNotFound):
        await rt.send_message("agt_does_not_exist", "hello", session_id="ses_nope")
