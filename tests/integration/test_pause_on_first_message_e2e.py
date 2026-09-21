"""边界：会话的**第一句**就按暂停（spec 2026-09-09 的范围边界）。

第一句走的是 root task —— 它由 `SessionRegistry._make_root_task_manager` 在
`start_session` 里建出来，**不经 `_start_task_for_agent`，因此没有未提交窗口**
（A 类那条只覆盖「上一轮已终态之后新开的 task」）。

于是第一句的暂停走的是改造前那条路：act 在检查点 park 出 wait 气泡。这份文件把
「那条路在新代码下仍然完好」钉住 —— 它是本次改造里唯一**没有**被窗口覆盖的用户消息
入口，最容易在别处改动时被顺手带坏。

要验的四件事：
1. 暂停在 TTFT 窗口里**当场**生效（这是新的：改造前要等首个 chunk 才响应）；
2. 会话落 PAUSED、有 wait 气泡可续；
3. `recognize_intent` 的起飞被挪到了提交点，而这一轮根本没到提交点 —— 会话标题不能
   因此永久没有；
4. 下一条消息照常跑起来。
"""

from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import LLMChunk
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)
from tests._event_helpers import all_events

pytestmark = pytest.mark.asyncio

_MOCK_CONTEXT_LIMIT = 100_000


class _StallsFirstActLLM(MockLLMAdapter):
    """第一次 act 调用卡在首个 chunk 之前；`stalling` 关掉后一切照常。

    `recognize_intent` 那一路不卡——它是旁路快照，本用例正要看它有没有起飞。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stalling = True
        self.stalled = asyncio.Event()
        self.sidecar_calls = 0

    @staticmethod
    def _is_sidecar(request) -> bool:
        tools = getattr(request, "tools", None) or []
        return any(getattr(t, "name", "") == "control__update_task_metadata" for t in tools)

    def complete(self, request, stream=True):
        self.last_request = request
        if self._is_sidecar(request):
            self.sidecar_calls += 1
            return super().complete(request, stream=stream)
        if not self.stalling:
            return super().complete(request, stream=stream)

        async def _gen():
            self.stalled.set()
            await asyncio.Event().wait()
            yield LLMChunk(kind="token", text="unreachable")  # pragma: no cover

        return _gen()


def _runtime(llm):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=resolver,
                      config=RuntimeConfig(llm_self_heal_max_attempts=1))
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _wait_for_wait_bubble(rt, session_id: str, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        pending = [v for v in rt.list_pending_hitl(session_id=session_id) if v.form == "wait"]
        if pending:
            return pending[0]
        await asyncio.sleep(0.02)
    raise TimeoutError("暂停之后没有出现 wait 气泡——会话没有续跑点，卡死了")


async def test_pause_on_the_very_first_message_parks_and_stays_usable() -> None:
    llm = _StallsFirstActLLM(
        responses=[MockResponse(text=f"answer {i}") for i in range(8)],
        context_limit=_MOCK_CONTEXT_LIMIT)
    rt = _runtime(llm)

    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="第一句就想撤回",
        context_limit=_MOCK_CONTEXT_LIMIT))
    sid, aid = handle.session_id, handle.agent_id

    # ① 请求已发出、还没有任何 chunk
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    assert rt._agent_lifecycle_manager.record_of(aid).status == "running"

    tm = rt._task_managers[sid]
    root_task_id = rt._agent_lifecycle_manager.record_of(aid).current_task_id
    assert not tm.is_round_open(root_task_id), (
        "root task 不开未提交窗口——它不经 _start_task_for_agent，本用例的前提就是这个"
    )

    # ② 按暂停：必须**当场**生效（TTFT 窗口里；改造前这里要等首个 chunk）
    assert await rt.pause_session(sid) is True
    bubble = await _wait_for_wait_bubble(rt, sid)
    assert bubble.task_id == root_task_id

    # ③ agent 停在 waiting_human，task 停在 AWAITING_HUMAN —— 会话有续跑点，没卡死
    assert rt._agent_lifecycle_manager.record_of(aid).status == "waiting_human"
    assert tm.get_task(root_task_id).status == "AWAITING_HUMAN"

    # ④ 旁路**没有**起飞——它的起飞点在 act 的提交点，而这一轮压根没到提交点。
    #
    # 这是对的：`recognize_intent` 只该为**真的发生过**的那一轮花一次 LLM 调用。
    # 代价是标题晚一轮，**不是永远没有**——见 ⑥：判据是「root task 且无 title」，
    # 这一轮夭折之后 title 仍是空的，下一轮照样判定成立、照样在那一轮的提交点起飞。
    assert llm.sidecar_calls == 0, "这一轮没到提交点，旁路不该已经烧掉一次 LLM 调用"
    types = [e.type for e in await all_events(rt.event_store, sid)]
    assert EventType.RECOGNIZE_INTENT_STARTED not in types

    # ⑤ 回答那个气泡：会话继续
    llm.stalling = False
    view = await rt.reply_to_hitl(HitlReply(
        hitl_id=bubble.id, outcome="accepted", agent_id=bubble.agent_id,
        message="算了，换个问题"))
    assert view is not None
    await asyncio.sleep(1.0)

    types = [e.type for e in await all_events(rt.event_store, sid)]
    assert EventType.HITL_RESOLVED in types, "续跑没起来——这一句的答复没能驱动下一轮"

    # ⑥ **标题在下一轮补上。** 这一条是 ④ 的另一半：夭折那一轮不起飞是对的，但不能
    #    就此永远无名。判据「root task 且无 title」在下一轮仍然成立，于是在**那一轮的
    #    提交点**起飞。晚一轮，不是没有。
    #
    #    不断言"只跑一次"：这里的 mock 从不调 `update_task_metadata`，title 一直是空的、
    #    还会再判定成立。那是桩的行为，不是产品的。
    assert llm.sidecar_calls >= 1, (
        "上一轮夭折之后，下一轮必须把名字补上——否则第一句按停的会话就真的永远无名了"
    )


async def test_pause_immediately_after_start_before_the_run_is_dispatched() -> None:
    """更早的时刻：`start_session` 刚返回、run 还没派发就按停。

    前端上就是「回车之后立刻点停止」。这一刻 `_run_tokens` 可能还是空的，
    `pause_session` 于是没有在途 run 可发信号——它靠的是 `abandon_pending(keep_agent=root)`
    把 root 已入队未派发的那一条**保留**下来、再补一次 `drain()`，让它出生即 paused、
    在 act 的首个检查点 park 出唯一续跑点。

    这条路径最容易坏的形状是「既没有在途 run、又没有排队条目、也没有气泡」——会话
    就此永久滞留 RUNNING，界面上按了停毫无反应且再也发不出消息（409 闸门）。
    """
    llm = _StallsFirstActLLM(
        responses=[MockResponse(text=f"answer {i}") for i in range(8)],
        context_limit=_MOCK_CONTEXT_LIMIT)
    rt = _runtime(llm)

    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="立刻就撤",
        context_limit=_MOCK_CONTEXT_LIMIT))
    sid, aid = handle.session_id, handle.agent_id

    # **不等** llm.stalled：抢在 run 派发之前/之中按停。
    paused = await rt.pause_session(sid)
    assert paused is True, "pause_session 返回 False = 什么都没做，会话会永久滞留 RUNNING"

    bubble = await _wait_for_wait_bubble(rt, sid)
    assert rt._agent_lifecycle_manager.record_of(aid).status == "waiting_human"

    # 闩锁必须已经清掉，否则下一轮出生即取消（"答了没反应"）。
    assert sid not in rt._pausing, "_pausing 闩锁残留 → 下一轮 born-cancel"
    assert sid not in rt._pause_claimed, "_pause_claimed 闩锁残留 → 下一轮 born-cancel"

    # 续跑：回答那个气泡，会话必须真的跑起来。
    llm.stalling = False
    assert await rt.reply_to_hitl(HitlReply(
        hitl_id=bubble.id, outcome="accepted", agent_id=bubble.agent_id,
        message="继续")) is not None
    await asyncio.sleep(1.0)
    types = [e.type for e in await all_events(rt.event_store, sid)]
    assert EventType.HITL_RESOLVED in types, "续跑没起来"


async def test_new_task_branch_does_not_re_run_recognize_intent() -> None:
    """会话只命名一次：root task 终态之后开的新 task 不再跑 `recognize_intent`。

    ⚠ **本用例只覆盖「新建 task」那一条分支，不是什么普遍不变式。**
    曾经把它写成「recognize_intent 从不跑在开着未提交窗口的 task 上」——那是**假的**：
    交互式会话第二句往后走的是「消息注入既有 task」，而那个既有 task 就是 root task，
    每一轮都开窗；root task 的 title 若还是空的（旁路失败 / 模型没调
    `update_task_metadata` / 上一轮的旁路还在飞），判据成立，它就**会**在窗口里起飞。
    那不是缺陷，只是「窗口里跑的东西撤销时要能退回去」——`RoundSnapshot` 因此也存
    `title` / `description`（见那里）。

    这里真正钉住的是：新建分支的 task 带着 `title="User Message"` 这个**占位字符串**，
    判据的后半截才不成立。那个占位一旦被改成空（"让 recognize_intent 给每个新话题起名"
    是个很自然的想法），每条新消息都会多烧一次旁路 LLM 调用，而且不会有任何地方报错。
    """
    llm = _StallsFirstActLLM(
        responses=[MockResponse(text=f"answer {i}") for i in range(12)],
        context_limit=_MOCK_CONTEXT_LIMIT)
    llm.stalling = False
    rt = _runtime(llm)

    # `unattended=True` → root task 的纯文本回合即产出，跑完落终态。
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="第一句",
        context_limit=_MOCK_CONTEXT_LIMIT, unattended=True))
    sid, aid = handle.session_id, handle.agent_id
    await asyncio.sleep(0.8)

    tm = rt._task_managers[sid]
    root_task_id = rt._agent_lifecycle_manager.record_of(aid).current_task_id
    assert tm.get_task(root_task_id).status == "FINISHED", "前提：root task 已终态"
    after_first_round = llm.sidecar_calls
    assert after_first_round >= 1, "第一轮该给会话起名"

    # root 已终态 → 这条新消息走「新建 task」那条分支，**会开未提交窗口**。
    turn = await rt.send_message(aid, "第二句", session_id=sid)
    new_task = tm.get_task(turn.task_id)
    assert new_task is not None and new_task.parent_task_id is None, (
        "新建分支的 task 没有 parent —— 判据的前半截（root task）对它是成立的，"
        "拦住它的只有 title 那个占位"
    )
    assert new_task.title, (
        "新建分支的 task 必须带一个非空占位标题；为空会让每条新消息都多烧一次旁路调用"
    )
    await asyncio.sleep(0.8)
    assert llm.sidecar_calls == after_first_round, (
        f"新建分支不该再跑一次 recognize_intent：{after_first_round} → {llm.sidecar_calls}"
    )
