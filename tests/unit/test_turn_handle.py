"""TurnHandle：句柄以 agent + task 为轴（2026-09-04 spec §3）。

run 是引擎内部一轮循环的相关性 id，句柄的职责是「指着一个外部可寻址的对象」。
agent + task 已经够定位，events() 与 wait_for_finish() 都不需要 run_id——
把它放进句柄只会多一个无法诚实填写的字段。
"""

from __future__ import annotations

from ctx_weft.core.loop.task_memory_fold import task_record_kind

import asyncio
from datetime import UTC, datetime

import pytest

from ctx_weft.core.runtime import SessionStartParams, TurnHandle
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEventType,
    MemoryKind,
    MemoryScope,
    ProviderContext,
    ToolCall,
    load_records,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events.bus.in_process.bus import InProcessEventBus
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



def _ev(**kw) -> Event:
    base = dict(
        id="evt_1", run_id="run_1", sequence=0, session_id="s1",
        type="TaskStarted", timestamp=datetime.now(UTC), origin="runtime",
        agent_id="agt_1", task_id="tsk_1",
    )
    base.update(kw)
    return Event(**base)


@pytest.mark.asyncio
async def test_handle_has_no_run_id_field():
    """结构性守卫：run_id 不在对外句柄上。"""
    assert "run_id" not in TurnHandle.__dataclass_fields__


@pytest.mark.asyncio
async def test_events_filters_by_agent_and_task():
    bus = InProcessEventBus()
    h = TurnHandle(session_id="s1", agent_id="agt_1", task_id="tsk_1",
                   template_id="tpl", event_bus=bus)
    got: list[str] = []

    async def _consume():
        async for ev in h.events():
            got.append(ev.id)
            if len(got) == 2:
                return

    t = asyncio.create_task(_consume())
    await asyncio.sleep(0)
    await bus.emit(_ev(id="e1"))
    await bus.emit(_ev(id="e2", agent_id="agt_2"))          # 别的 agent
    await bus.emit(_ev(id="e3", task_id="tsk_2"))           # 同 agent 别的 task
    await bus.emit(_ev(id="e4"))
    await asyncio.wait_for(t, timeout=2.0)

    assert got == ["e1", "e4"]


@pytest.mark.asyncio
async def test_wait_for_finish_returns_on_task_terminal_event():
    bus = InProcessEventBus()
    h = TurnHandle(session_id="s1", agent_id="agt_1", task_id="tsk_1",
                   template_id="tpl", event_bus=bus, _state=None)
    waiter = asyncio.create_task(h.wait_for_finish(timeout=2.0))
    await asyncio.sleep(0)
    await bus.emit(_ev(id="e1", type="TaskFinished"))
    await asyncio.wait_for(waiter, timeout=2.0)


@pytest.mark.asyncio
async def test_wait_for_finish_ignores_run_finished():
    """RunFinished 不再是判据——一轮 run 结束不等于这条 task 结束（可能还要 finalize）。"""
    bus = InProcessEventBus()
    h = TurnHandle(session_id="s1", agent_id="agt_1", task_id="tsk_1",
                   template_id="tpl", event_bus=bus)
    waiter = asyncio.create_task(h.wait_for_finish(timeout=0.3))
    await asyncio.sleep(0)
    await bus.emit(_ev(id="e1", type="RunFinished"))
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(waiter, timeout=2.0)


@pytest.mark.asyncio
async def test_wait_for_finish_ignores_an_earlier_overlapping_launchs_recap_done():
    """二轮修复的回归守卫（2026-09-04）：`_task_pending` 只挂最新一次 launch，但同一个
    task 一生里可能有多次 fire-and-forget 后台 observe（`interrupt`/`mechanical`/
    `dispatch`/`finish` 等不同 boundary，跨 suspend/resume、重试多轮发生），不重叠只是
    **通常**情况——上一轮（例如 `interrupt`/`mechanical` 边界）launch 的 `ObserveApplied`
    完全可能在这一轮终态事件之后、这一轮（close 边界）自己的 `ObserveApplied` 之前才
    姗姗来迟地送达。`wait_for_finish` 若只按事件类型匹配，会把这条迟到的、不相干的
    `ObserveApplied` 误当成「这次终态触发的那次折叠」，提前返回——同一个 bug 在更窄的
    窗口里复现。

    这里不跑真实 runtime（两次真实 launch 重叠是一场难以确定性复现的调度竞态），直接
    摆弄 `background_observe` 的模块级登记表，模拟「更早一次 launch 仍在途、这次终态
    事件之后才等到它迟到的 ObserveApplied」这个精确场景——与
    `test_wait_for_finish_returns_after_deferred_fold_lands` 互补：那条测真实折叠
    落地，这条测「等的必须是对的那次」。"""

    class _NeverDoneTask:
        """占位：只需要 `.done()` 恒为 False，不需要真的是 asyncio.Task。"""

        def done(self) -> bool:
            return False

    bus = InProcessEventBus()
    h = TurnHandle(session_id="s1", agent_id="agt_1", task_id="tsk_1",
                   template_id="tpl", event_bus=bus)

    # 登记「这次（close 边界）launch」为当前在途——`_task_pending`/`_task_pending_run_id`
    # 本就总是相邻一起写（见 launch_recap），这里手动摆出同样的形状。
    h._pending_observe = lambda sid, tid: "run_close_launch"
    try:
        waiter = asyncio.create_task(h.wait_for_finish(timeout=2.0))
        for _ in range(5):
            await asyncio.sleep(0)
        await bus.emit(_ev(id="e1", type="TaskFinished"))
        for _ in range(5):
            await asyncio.sleep(0)

        # 更早一次（interrupt/mechanical 边界）launch 的 ObserveApplied 迟到——run_id
        # 对不上这次登记的 "run_close_launch"。
        await bus.emit(_ev(id="e2", type=EventType.OBSERVE_APPLIED, run_id="run_earlier_launch"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not waiter.done(), (
            "wait_for_finish 不该被更早一次 launch 的 ObserveApplied 提前放行"
        )

        # 这次 launch 自己的 ObserveApplied 才是真正该等的那个。
        await bus.emit(_ev(id="e3", type=EventType.OBSERVE_APPLIED, run_id="run_close_launch"))
        await asyncio.wait_for(waiter, timeout=2.0)
    finally:
        h._pending_observe = None


@pytest.mark.asyncio
async def test_wait_for_finish_matches_observe_applied_by_value_not_identity():
    """三轮修复的回归守卫（2026-09-04）：`ev2.type` 与 `EventType.OBSERVE_APPLIED` 的
    匹配必须用 `==`（值相等），不能用 `is`（身份相等）。

    `EventBus` 是宿主可自行实现的协议（docs 明确把换成 Redis Streams 当作支持的范例）；
    序列化往返一趟的宿主总线，投出来的 `.type` 会是裸 `str`，不再是 `EventType` 枚举
    成员实例——`EventType` 是 `StrEnum`，`EventType.OBSERVE_APPLIED == "ObserveApplied"`
    恒真，但两者 `is` 恒假（裸 str 与枚举成员永远不是同一个对象）。若判据用 `is`，这
    条真正该等的 `ObserveApplied`（`run_id` 对得上）会被裸字符串这个事实悄无声息地
    滤掉——`wait_for_finish` 不抛异常，会耗光整个 `timeout` 才落到超时兜底返回，宿主
    体感就是「挂住了」。

    这里构造的 `ObserveApplied` 事件 `type` 字段是裸字符串 `"ObserveApplied"`（模拟一个
    序列化往返的宿主总线投出来的事件），`run_id` 与登记的在途 launch 一致——断言
    `wait_for_finish` 依然认得出它、正常返回，而不是耗光 `timeout` 才靠超时兜底返回。"""

    class _NeverDoneTask:
        def done(self) -> bool:
            return False

    bus = InProcessEventBus()
    h = TurnHandle(session_id="s1", agent_id="agt_1", task_id="tsk_1",
                   template_id="tpl", event_bus=bus)

    h._pending_observe = lambda sid, tid: "run_close_launch"
    try:
        # 短超时：判据若退化回 `is`，这条裸字符串事件永远不匹配，测试会在这个超时
        # 上原地耗尽——短超时让那种失败又快又好认，而不是拖到默认的几百秒。
        waiter = asyncio.create_task(h.wait_for_finish(timeout=0.3))
        for _ in range(5):
            await asyncio.sleep(0)
        await bus.emit(_ev(id="e1", type="TaskFinished"))
        for _ in range(5):
            await asyncio.sleep(0)

        # 裸字符串 type（不是 EventType 枚举成员）——模拟序列化往返的宿主总线；
        # run_id 与登记的在途 launch 一致，`wait_for_finish` 该认得出它。
        t0 = asyncio.get_running_loop().time()
        await bus.emit(_ev(id="e2", type="ObserveApplied", run_id="run_close_launch"))

        await asyncio.wait_for(waiter, timeout=2.0)
        elapsed = asyncio.get_running_loop().time() - t0
        # 关键断言：必须是被这条事件放行的，不是耗光 0.3s 的 timeout 兜底才返回——
        # `state is None` 本身在两条路径下都成立（这个裸 TurnHandle 没喂真实
        # LoopState），区分不出「匹配上了」和「超时了」，只有耗时能区分。
        assert elapsed < 0.2, (
            f"wait_for_finish 耗时 {elapsed:.3f}s——像是靠 0.3s 超时兜底返回的，"
            "说明 ObserveApplied 没被裸字符串 type 认出来"
        )
    finally:
        h._pending_observe = None


@pytest.mark.asyncio
async def test_wait_for_finish_does_not_hang_when_recap_done_arrives_before_terminal():
    """终审 IMPORTANT 3 回归：`ObserveApplied` 先于终态事件到达时不能挂住。

    原实现的外层循环只在 `ev.type in terminal` 时才有反应；先到的 `ObserveApplied`
    被当成"不认识的事件"直接丢弃（流是单向消费的，丢过去的事件读不回来）。等终态
    事件终于到达、内层循环才开始等那个 `run_id`——它已经过去了，`pending.done()`
    此刻仍是 `False`（后台任务是否跑完与它的 `ObserveApplied` 是否已送达是两回事），
    内层循环于是无休止地等一条不会再来的事件，直到 `timeout` 耗尽才靠兜底返回——
    不抛异常，只是把调用方晾到超时（这正是最坏的失败形态）。

    本测试给 `wait_for_finish` 一个短 `timeout`：修复后必须远早于这个超时返回
    （证明是被"提前到账"的 `ObserveApplied` 放行的，不是靠超时兜底）；未修复时会
    原地耗尽这个 `timeout` 才返回——用耗时区分两者，与
    `test_wait_for_finish_matches_observe_applied_by_value_not_identity` 同一手法。
    """

    class _NeverDoneTask:
        def done(self) -> bool:
            return False

    bus = InProcessEventBus()
    h = TurnHandle(session_id="s1", agent_id="agt_1", task_id="tsk_1",
                   template_id="tpl", event_bus=bus)

    h._pending_observe = lambda sid, tid: "run_close_launch"
    try:
        waiter = asyncio.create_task(h.wait_for_finish(timeout=0.3))
        for _ in range(5):
            await asyncio.sleep(0)

        t0 = asyncio.get_running_loop().time()
        # ObserveApplied 先到账——这次终态触发的那次折叠先落地，事件先送达。
        await bus.emit(_ev(id="e1", type=EventType.OBSERVE_APPLIED, run_id="run_close_launch"))
        for _ in range(5):
            await asyncio.sleep(0)
        # 终态事件随后才到。
        await bus.emit(_ev(id="e2", type="TaskFinished"))

        await asyncio.wait_for(waiter, timeout=2.0)
        elapsed = asyncio.get_running_loop().time() - t0
        assert elapsed < 0.2, (
            f"wait_for_finish 耗时 {elapsed:.3f}s——像是靠 0.3s 超时兜底返回的，"
            "说明先到账的 ObserveApplied 没被记住，白等了一条已经过去的事件"
        )
    finally:
        h._pending_observe = None


class _CloseFoldLLM(MockLLMAdapter):
    """单 root task：finish_task 收尾触发 close 边界后台 observe（launch_recap
    fire-and-forget），验证 `wait_for_finish` 返回**那一刻**该次折叠已经落地——不是靠
    调用方碰巧多等了一会儿。按 request.tools 路由：recognize_intent → 空文本；
    background observe（`control__report_observation`）→ 折叠摘要；其余（act）→
    finish_task 收尾。"""

    def __init__(self, **kw) -> None:
        super().__init__(responses=[], **kw)
        self._n = 0

    def _id(self, kind: str) -> str:
        self._n += 1
        return f"{kind}{self._n}"

    def complete(self, request, stream=True):
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:  # recognize_intent
            return self._stream(MockResponse(text=""), request)
        # 后台 observe 的 terminal tool 自 2026-09-22 起与前台同一个
        # （`report_observation`）；这里靠「没有 act 的工具面」把它与 act 区分开。
        if "control__report_observation" in names:  # observe（前台或后台）
            return self._stream(MockResponse(tool_calls=[
                ToolCall(id=self._id("bg"), name="control__report_observation",
                         arguments={"task_status": "success", "act_recap": "CLOSE段摘要",
                                    "task_summary": "CLOSE小结"}),
            ]), request)
        return self._stream(MockResponse(text="done", tool_calls=[]), request)


@pytest.mark.asyncio
async def test_wait_for_finish_returns_after_deferred_fold_lands():
    """回归守卫（2026-09-04）：root task 收尾会 fire-and-forget 一个后台 observe 去折叠/
    胶囊化本段 raw（spec 2026-07-20 延迟折叠）。这个折叠任务的登记先于 TaskFinished 发出
    （同一协程内、无 await 间隔），但它的执行体何时真正跑完，相对 TaskFinished 谁先谁后
    是一场纯粹的 asyncio 调度竞态——`wait_for_finish` 若只等 TaskFinished 就返回，会在
    折叠完工前把控制权交还调用方（`test_dispatch_boundary_recap_e2e` 曾经~1/3 概率
    因此失败：断言时看到本该被折掉的 raw）。

    本测试不靠任何额外 `sleep` 或轮询掩盖窗口：`wait_for_finish` 一返回就立刻查 memory，
    折叠结果必须已经可见。"""
    llm = _CloseFoldLLM()
    resolver = InlineAgentTemplateProvider()
    template = make_echo_template()
    # 关短任务/短段免折两道门（_is_short_leaf 用 short_task_token_threshold，close 边界
    # 走这条；is_short_segment 用 short_segment_token_threshold，非 close 边界走那条）——
    # 都关掉才能强制走延迟折叠，而不是「短 → 原文即胶囊」的免折捷径。
    template.loop_config.short_task_token_threshold = 0
    template.loop_config.short_segment_token_threshold = 0
    resolver.register(template)
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    memory = InMemoryMemoryProvider()
    runtime.providers.register_memory(memory)

    await initialize_all(runtime)
    handle = await runtime.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="say hello",
    ))
    await handle.wait_for_finish(timeout=8.0)

    # close 边界的后台 observe 不写 TASK_COMPACT_SUMMARY（那是非 close 边界 segment_fold 的
    # 产物）——它把真报告替换进 finish 对（AGENT scope 的 conversation turn，见
    # background_observe._replace_finish_report / finalize.build_finish_slots），随后
    # 补删 task scope 的末段 raw（_supersede_final_raw_segment）。两处都得在 wait_for_finish
    # 返回那一刻已经落地，而不是靠调用方之后又多等了一会儿才凑巧看到。
    task_scope = MemoryAddress(session_id=handle.session_id, task_id=handle.task_id,
                               agent_id=handle.agent_id)
    agent_scope = MemoryAddress(session_id=handle.session_id, agent_id=handle.agent_id)
    pctx = ProviderContext(session_id=handle.session_id, tenant_id="default",
                           task_id=handle.task_id, agent_id=handle.agent_id)

    turns = await load_records(memory,
        agent_scope, MemoryScope.AGENT, pctx, kinds=[MemoryKind.CONVERSATION_TURN])
    recap = [r for r in turns if task_record_kind(r) == "recap"
             and r.metadata.get("origin_task_id") == handle.task_id
             and not r.metadata.get("final_reply")]
    assert any("CLOSE段摘要" in r.content for r in recap), \
        ("wait_for_finish 返回时，close 边界后台 observe 的真报告必须已经替换进 finish 对，"
         f"实得 {[r.content for r in recap]}")

    raws = await memory.recall_recent(task_scope, [MemoryEventType.LLM_RESPONSE], 100, pctx)
    assert raws == [], \
        f"wait_for_finish 返回时，本段 raw 应已被折掉/胶囊化，实得 {[r.content for r in raws]}"


@pytest.mark.asyncio
async def test_start_session_returns_turn_handle():
    """裁定 A（控制方 2026-09-04）：brief 原版 setup（`template_id="echo"` + 未注册
    memory provider）在本仓库跑不起来——`start_session` 会在模板解析阶段先炸。改用
    `test_runtime_agent_api.py::test_start_session_agent_id_is_addressable_root_agent`
    验证过的搭台手法：`template_id="agent:tpl_echo"`、注册 memory provider、LLM 回
    `control__finish_task` 让 run 能终结。
    """
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(observer_verdict="success", responses=[MockResponse(text="done", tool_calls=[])])
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())

    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi",
    ))
    await handle.wait_for_finish(timeout=5.0)

    assert isinstance(handle, TurnHandle)
    assert handle.agent_id and handle.task_id and handle.session_id and handle.template_id
