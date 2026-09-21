"""端到端：LLM 还没开口时按暂停 → 这一轮当作没发生过（spec 2026-09-09）。

改造前的形状：`pause_session` 只在流循环体里被检查，请求发出到首个 chunk 之间
（TTFT，思考模型 / provider 排队能到几十秒）协程挂在 `__anext__` 上，`pause_token`
被置位也没人读——用户按下暂停毫无反应，等首个 token 回来才 park，还留下一个半截回合：
一个 `TASK_CREATED`、一条没人应答的 user 记录、一个回合号的洞。

改造后：
- 暂停在 TTFT 窗口里**当场**生效（`_stream_until_stop` 把 `__anext__` 和停止信号赛跑）；
- 这一轮的全部事件攒在总线的未提交窗口里，一条都没落盘 → 整体丢弃；
- 用户那条消息用 `memory.fold` 纯遗忘掉；
- agent 回 `idle`，上一轮的终态原封不动。
"""

from __future__ import annotations

import asyncio

import pytest

from ctx_weft.core.models.config import RuntimeConfig
from ctx_weft.core.runtime import SessionStartParams
from ctx_weft.protocols import LLMChunk, MemoryAddress, MemoryScope, ProviderContext
from ctx_weft.protocols.events import EventType
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
_DISCARDED_PROMPT = "this question gets taken back"


class _StallsBeforeFirstChunkLLM(MockLLMAdapter):
    """第一轮正常回答；**第二轮起**卡在首个 chunk 之前不动（模拟 TTFT）。

    `recognize_intent` 那一路照常走 mock 响应——它是旁路快照，不该被这条卡住。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stalling = False
        self.stalled = asyncio.Event()      # 已经开始等首个 chunk 了

    @staticmethod
    def _is_recognize_intent(request) -> bool:
        tools = getattr(request, "tools", None) or []
        return any(getattr(t, "name", "") == "control__update_task_metadata" for t in tools)

    def complete(self, request, stream=True):
        self.last_request = request
        if not self.stalling or self._is_recognize_intent(request):
            return super().complete(request, stream=stream)

        async def _gen():
            self.stalled.set()
            await asyncio.Event().wait()     # 永不产出首个 chunk
            yield LLMChunk(kind="token", text="unreachable")  # pragma: no cover

        return _gen()


def _runtime(llm):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    rt = make_runtime(
        llm=llm, agent_provider=resolver,
        config=RuntimeConfig(llm_self_heal_max_attempts=1),
    )
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _stored_types(rt, session_id: str) -> list[str]:
    return [e.type for e in await all_events(rt.event_store, session_id)]


async def _user_texts(rt, session_id: str, agent_id: str, task_id: str) -> list[str]:
    view = await rt.providers.get_memory().load_view(
        MemoryAddress(session_id=session_id, task_id=task_id, agent_id=agent_id),
        MemoryScope.TASK,
        ProviderContext(session_id=session_id, tenant_id="default",
                        task_id=task_id, agent_id=agent_id),
    )
    from ctx_weft.core.utils.content import content_to_text
    return [content_to_text(r.content) for r in view if r.role == "user"]


async def test_pause_before_first_chunk_discards_the_whole_round() -> None:
    # 备足响应：第一轮 act + observe/recognize_intent 旁路 + 第⑤步那一轮，各自可能
    # 多调一次。数量不是本用例要盯的东西，给够即可。
    llm = _StallsBeforeFirstChunkLLM(
        responses=[MockResponse(text=f"answer {i}") for i in range(8)],
        context_limit=_MOCK_CONTEXT_LIMIT)
    rt = _runtime(llm)

    # `unattended=True` 把 root task 压成 `interaction_mode="auto"`：第一轮的纯文本回合
    # 就是任务产出、task 落 FINISHED。**这是本用例的前提**——只有上一轮已终态，下一条
    # `send_message` 才走「新建 task」那条分支（`_start_task_for_agent`），也才有未提交
    # 窗口可言。interactive 的 root task 会 park 在 wait 气泡上，下一条消息注入的是同一个
    # 既有 task，不开窗口（那条路径的暂停仍然照常 park，只是现在**当场**生效了）。
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi",
        context_limit=_MOCK_CONTEXT_LIMIT, unattended=True))
    sid = handle.session_id
    aid = handle.agent_id
    await asyncio.sleep(0.5)                        # 让第一轮跑完并落终态

    types_before = await _stored_types(rt, sid)
    created_before = types_before.count(EventType.TASK_CREATED)

    # ── 第二轮：新建 task（未提交窗口），LLM 卡在首个 chunk 之前 ────────────────
    llm.stalling = True
    turn = await rt.send_message(aid, _DISCARDED_PROMPT, session_id=sid)
    await asyncio.wait_for(llm.stalled.wait(), timeout=5.0)

    # 窗口开着：agent 必须已经是 running（否则 pause_agent 会拒、host 折叠会翻车），
    # 但这一轮的事件一条都还没落盘。
    assert rt._agent_lifecycle_manager.record_of(aid).status == "running", (
        "未提交窗口里 agent 必须是 running —— 这正是 provisional 订阅者存在的理由"
    )
    assert await _stored_types(rt, sid) == types_before, (
        "首个 chunk 之前，这一轮不得往事件日志里写任何东西"
    )

    # ── 按下暂停 ────────────────────────────────────────────────────────────────
    assert await rt.pause_session(sid) is True
    await asyncio.sleep(0.5)

    # ① 事件日志：这一轮一条都没留，连 TASK_CANCELED 都没有
    types_after = await _stored_types(rt, sid)
    assert types_after.count(EventType.TASK_CREATED) == created_before, (
        f"被丢弃的 task 仍然留下了 TASK_CREATED：{types_after}"
    )
    assert types_after == types_before, (
        f"这一轮当作没发生过，日志应当逐条不变；多出来的是 {types_after[len(types_before):]}"
    )

    # ② agent 回 idle —— 靠的是窗口内那条 TASK_CANCELED（日志上看不到，内存里生效了）
    assert rt._agent_lifecycle_manager.record_of(aid).status == "idle"

    # ③ 用户那条消息被 fold 掉了，不在视图里
    texts = await _user_texts(rt, sid, aid, turn.task_id)
    assert all(_DISCARDED_PROMPT not in t for t in texts), (
        f"被撤销的用户消息仍留在 memory 视图里：{texts}"
    )

    # ④ task 本身也不在登记里了
    tm = rt._task_managers.get(sid)
    assert tm is not None and tm.get_task(turn.task_id) is None

    # ⑤ 下一条消息照常开新一轮（会话没被这次丢弃搞坏）
    llm.stalling = False
    again = await rt.send_message(aid, "ok, different question", session_id=sid)
    assert again.task_id != turn.task_id
    await asyncio.sleep(0.5)
    assert EventType.TASK_CREATED in await _stored_types(rt, sid)
