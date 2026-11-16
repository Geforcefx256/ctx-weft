"""端到端 · D 类：回答了 wait 气泡、LLM 还没开口时按暂停 → 整轮撤销（spec 2026-09-09）。

这是交互式会话里**最常走的那条路**：root task 是 `interactive`，每说完一段纯文本就
park 在一个 wait 气泡上，用户的下一句话经 `reply_to_hitl` 回答那个气泡（前端在
`status === 'PAUSED'` 时走的正是这条，见 ChatPanel 的 `waitReplyMut`）。

A 类（`send_message` 新建 task）那条在 `test_round_discarded_before_first_chunk_e2e.py`。
两者的差别是这份文件存在的理由：

- A 类的 task 是这条消息开出来的 → 丢弃时连 task 一起摘掉，agent 回 `idle`；
- D 类的 task 早就存在、是一段正在进行的对话 → 丢弃只把它退回开窗前的样子，
  agent 回 `waiting_human`，**而那个被回答掉的气泡要回到 pending**。

最后那一条是 (a) 方案的全部意义：`reply_to_hitl` 的终局被推迟成两阶段
（`pending_decision` → 提交点才 `HitlResolved`），所以撤销之后 RAM 与事件日志**都**
停在「人还没回答」，会话干干净净地回到 `PAUSED`，用户重答一次即可。
"""

from __future__ import annotations

from tests._recovery import load_pending_fixture

import asyncio

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import LLMChunk, MemoryAddress, MemoryScope, ProviderContext, load_records
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import HitlReply
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._event_helpers import all_events
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


_MOCK_CONTEXT_LIMIT = 100_000
_RETRACTED = "actually never mind, forget I asked"


class _StallsBeforeFirstChunkLLM(MockLLMAdapter):
    """`stalling` 打开后卡在首个 chunk 之前不动（模拟 TTFT）。旁路调用不受影响。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stalling = False
        self.stalled = asyncio.Event()

    @staticmethod
    def _is_sidecar(request) -> bool:
        """旁路调用——不该被 TTFT 卡住的那些。

        除 recognize_intent，还包括**判定回合**（2026-09-27）：S-b 起 root 的 `finish_task`
        也 park + 后台判定，卡住它等于让 verdict 永不抵达（≡ retry），task 永远停在 park。
        这里要模拟的是 act 那一轮的 TTFT，不是观察。
        """
        tools = getattr(request, "tools", None) or []
        names = {getattr(t, "name", "") for t in tools}
        return bool(names & {"control__update_task_metadata", "control__report_observation"})

    def complete(self, request, stream=True):
        self.last_request = request
        if not self.stalling or self._is_sidecar(request):
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


async def _stored_types(rt, session_id: str) -> list[str]:
    return [e.type for e in await all_events(rt.event_store, session_id)]


async def _user_texts(rt, session_id: str, agent_id: str, task_id: str) -> list[str]:
    from ctx_weft.protocols.content import content_to_text
    view = await load_records(rt.providers.get_memory(),
        MemoryAddress(session_id=session_id, task_id=task_id, agent_id=agent_id),
        MemoryScope.TASK,
        ProviderContext(session_id=session_id, tenant_id="default",
                        task_id=task_id, agent_id=agent_id),
    )
    return [content_to_text(r.content) for r in view if r.role == "user"]


async def _wait_for_wait_bubble(rt, session_id: str, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        await initialize_all(rt)
        pending = [v for v in [item.data for item in await rt.list_pending_hitl(session_id=session_id)] if v.form == "wait"]
        if pending:
            return pending[0]
        await asyncio.sleep(0.02)
    raise TimeoutError("wait bubble never appeared")


@pytest.mark.asyncio
async def test_pause_before_first_chunk_puts_the_answered_bubble_back() -> None:
    llm = _StallsBeforeFirstChunkLLM(
        responses=[MockResponse(text=f"answer {i}") for i in range(8)],
        context_limit=_MOCK_CONTEXT_LIMIT, observer_verdict="waiting")
    rt = _runtime(llm)

    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi"))
    sid, aid = handle.session_id, handle.agent_id

    # interactive 的 root task 说完一段纯文本就 park 出一个 wait 气泡 —— 这就是 D 类的起点。
    bubble = await _wait_for_wait_bubble(rt, sid)
    task_id = bubble.task_id
    events_before = await all_events(rt.event_store, sid)
    ids_before = {e.id for e in events_before}

    # ── 用户回答那个气泡，LLM 卡在首个 chunk 之前 ────────────────────────────
    llm.stalling = True
    await initialize_all(rt)
    view = await rt.reply_to_hitl(HitlReply(
        hitl_id=bubble.id, outcome="accepted", agent_id=bubble.agent_id,
        message=_RETRACTED))
    # 调用方视角：这次应答**被收下了**（两阶段之下「收下」与「落盘」不再同刻）。
    assert view is not None and view.outcome == "accepted"
    assert rt.hitl_registry.get(bubble.id).claim_pending is True
    assert rt.hitl_registry.get(bubble.id).resolved is False
    # 收下之后它就不该再出现在未决列表里 —— 否则 host 会让人对着刚答过的问题再答一次。
    await initialize_all(rt)
    assert bubble.id not in [v.id for v in [item.data for item in await rt.list_pending_hitl(session_id=sid)]]

    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    assert rt._agent_lifecycle_manager.record_of(aid).status == "running"
    # Prior actors, metadata and recap may finish while this round is provisional.
    # Only the replying run and the withdrawn input belong to the new window.
    replying_run_id = rt._task_managers[sid]._rounds[task_id].run_id
    during = [e for e in await all_events(rt.event_store, sid) if e.id not in ids_before]
    assert not any(e.run_id == replying_run_id or _RETRACTED in str(e.payload) for e in during), during

    # ── 按下暂停 ────────────────────────────────────────────────────────────
    await initialize_all(rt)
    assert await rt.pause_session(sid) is True
    await asyncio.sleep(0.5)

    # ① 日志只多**一条**：`HitlReplyRetracted`。
    #
    # `HitlResolved` 从来没发过 —— 被撤回的那句话不在日志里，这是两阶段买到的东西。
    # 而撤回这件事本身要留痕：不留的话「这条气泡被答过又撤了」跨不过重启，重答时
    # memory 的幂等键就会撞上上一次留下的 superseded 记录（见 ⑦ 与下一个用例）。
    after = [e for e in await all_events(rt.event_store, sid) if e.id not in ids_before
             and (e.run_id == replying_run_id or e.type == EventType.HITL_REPLY_RETRACTED)]
    assert not any(e.type == EventType.HITL_RESOLVED and e.payload.get("hitl_id") == bubble.id
                   for e in await all_events(rt.event_store, sid))
    assert [e.type for e in after] == [EventType.HITL_REPLY_RETRACTED]
    retraction = next(e for e in after if e.type == EventType.HITL_REPLY_RETRACTED)
    assert retraction.payload == {"hitl_id": bubble.id}, (
        f"撤回事件不得带正文——被撤回的话留在日志里就白撤了：{retraction.payload}"
    )
    assert not retraction.task_id, (
        "带上 task_id 会被未提交窗口挡住、随缓冲一起丢掉，等于没发"
    )

    # ② 气泡回到 pending —— (a) 方案的核心：撤销之后 RAM 与日志都停在「人还没回答」
    req = rt.hitl_registry.get(bubble.id)
    assert req.claim_pending is False and req.resolved is False
    await initialize_all(rt)
    assert bubble.id in [v.id for v in [item.data for item in await rt.list_pending_hitl(session_id=sid)]], (
        "被撤销的那次应答之后，气泡必须重新出现在未决列表里，人才能重答"
    )

    # ③ 用户那句话被 fold 掉了
    texts = await _user_texts(rt, sid, aid, task_id)
    assert all(_RETRACTED not in t for t in texts), f"撤销的话仍在 memory 视图里：{texts}"

    # ④ agent 回 waiting_human（**不是** idle —— task 还在，只是重新等人）
    assert rt._agent_lifecycle_manager.record_of(aid).status == "waiting_human"

    # ⑤ task 还在，且退回了 AWAITING_HUMAN
    tm = rt._task_managers[sid]
    task = tm.get_task(task_id)
    assert task is not None and task.status == "AWAITING_HUMAN"

    # ⑥ 重答一次照常跑起来（会话没被这次撤销搞坏）
    llm.stalling = False
    retyped = "ok, a different question then"
    await initialize_all(rt)
    view2 = await rt.reply_to_hitl(HitlReply(
        hitl_id=bubble.id, outcome="accepted", agent_id=bubble.agent_id,
        message=retyped))
    assert view2 is not None
    await asyncio.sleep(1.5)
    assert EventType.HITL_RESOLVED in await _stored_types(rt, sid), (
        "重答之后这一轮真的跑起来了，HitlResolved 应当在提交点发出"
    )

    # ⑦ **重打的那句话必须真的进 memory。**
    #
    # 这条不是锦上添花：memory 的 record-id 契约是「已存在的 id（**含已 superseded**）
    # = no-op，不比对内容、不重复写入」，而两次应答的幂等键都是从同一个 hitl_id 派生的。
    # 第一次的记录被撤销时 fold 成了 superseded——它**仍然占着那个 id**。所以只要键不带
    # 「第几次」这一维，用户重打的话就会被静默吞掉：界面上消息在、模型永远看不见。
    texts2 = await _user_texts(rt, sid, aid, task_id)
    assert any(retyped in t for t in texts2), (
        f"重答的内容没能进 memory（被 superseded 记录的幂等键吞了）：{texts2}"
    )


@pytest.mark.asyncio
async def test_retyped_reply_survives_a_restart_in_the_discard_window() -> None:
    """撤销之后**重启**，再重答同一个气泡——那句话仍然必须进 memory。

    `reply_attempt` 是内存态（`PendingHitl` 上的计数，`release_claim` 时 +1）。撤销
    这条曾经真的坏过，而且修错过一次。

    第一版修法是给 `PendingHitl` 加一个「这是第几次应答」的计数，让幂等键带上那一维。
    它在同一个进程里成立，**跨重启不成立**：撤销刻意不发任何事件，所以「这条气泡被答过
    又撤了」在事件日志里没有痕迹，重启后从事件折出来的记录计数必然归零，键必然撞回那条
    superseded 记录。任何内存计数器都救不了这个形状。

    第二版（现行）把「注没注入过」的判据从 id 约定挪到记录的
    `metadata[REPLY_HITL_ID_KEY]`，写入侧每次用新 id —— 那份判据跨重启存活，而 id 不必
    再复用，也就不会撞。

    窗口很窄（要正好在"撤销之后、重答之前"重启），但后果是**静默丢消息**：用户重打
    的那句话进不了 memory，模型永远看不见，而界面上一切正常。
    """
    llm = _StallsBeforeFirstChunkLLM(
        responses=[MockResponse(text=f"answer {i}") for i in range(8)],
        context_limit=_MOCK_CONTEXT_LIMIT, observer_verdict="waiting")
    rt = _runtime(llm)

    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi"))
    sid, aid = handle.session_id, handle.agent_id
    bubble = await _wait_for_wait_bubble(rt, sid)
    task_id = bubble.task_id

    llm.stalling = True
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(hitl_id=bubble.id, outcome="accepted",
                                     agent_id=bubble.agent_id, message=_RETRACTED))
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)
    await initialize_all(rt)
    await rt.pause_session(sid)
    await asyncio.sleep(0.5)

    # ── 模拟重启：HITL 内存态从事件日志重建 ──────────────────────────────────
    # 真正的重启 = **注册表先空掉**再从事件重建。只调 `rebuild_hitl` 不够：它对内存里
    # 已有的活记录是保留的（`load_snapshot` 的既有纪律），那样测到的还是同一个内存对象
    # ——这条测试曾经因此假绿过一次。
    rt._evict_session_memory(sid)
    rt.hitl_registry.forget_session(sid)
    await initialize_all(rt)
    await load_pending_fixture(rt, sid)
    rebuilt = rt.hitl_registry.get(bubble.id)
    assert rebuilt is not None, "重建之后这个气泡必须还在（撤销不发事件，日志里它仍是未决）"
    assert rebuilt.claim_pending is False and rebuilt.pending_decision is None, (
        "重建出来的必须是干净的 pending —— 它是从事件折的，而那次应答从没进过日志"
    )

    llm.stalling = False
    retyped = "restart then retype"
    await initialize_all(rt)
    assert await rt.reply_to_hitl(HitlReply(
        hitl_id=bubble.id, outcome="accepted", agent_id=bubble.agent_id,
        message=retyped)) is not None
    await asyncio.sleep(1.5)

    texts = await _user_texts(rt, sid, aid, task_id)
    assert any(retyped in t for t in texts), (
        f"重启之后重答的内容没能进 memory —— 幂等键被上一次的 superseded 记录占着：{texts}"
    )
