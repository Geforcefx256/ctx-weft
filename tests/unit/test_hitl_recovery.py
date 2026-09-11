"""HITL 跨重启恢复：装填 registry 后 task 的 park / 重排判定（spec/07 §9）。

`rebuild_pending` 那两条（重建不带 future / 重建后应答走冷）已删——装填与「装填出来
的项恒无等待槽」由 `test_hitl_registry_load.py` 覆盖，冷应答由
`test_runtime_hitl_wiring.py` 覆盖。
"""

from __future__ import annotations

import pytest

from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
from ctx_weft.core.orchestrator.task.disposition import RunOutcome, RunOutcomeKind
from tests.unit._stub_runner import StubRunner
from tests.unit._legacy_recover import rebuild_all_active

pytestmark = pytest.mark.asyncio


def test_restore_keeps_hitl_parked_task_suspended() -> None:
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task
    tm = TaskManager(session_id="s1")
    parked = Task(id="t1", session_id="s1", status="SUSPENDED")
    tm.restore([parked], terminal_ids=set(), parked_task_ids={"t1"})
    assert tm.get_task("t1").status == "SUSPENDED"
    assert not tm._queue.has_pending()


def test_restore_requeues_suspended_on_children_when_all_terminal() -> None:
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task
    tm = TaskManager(session_id="s1")
    parent = Task(id="p", session_id="s1", status="SUSPENDED")
    child = Task(id="c", session_id="s1", status="FINISHED", parent_task_id="p")
    tm.restore([parent, child], terminal_ids={"c"}, parked_task_ids=set())
    assert tm.get_task("p").status == "PENDING"


def test_restore_parked_ids_default_none_is_old_behavior() -> None:
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task
    tm = TaskManager(session_id="s1")
    parent = Task(id="p", session_id="s1", status="SUSPENDED")
    tm.restore([parent], terminal_ids=set())   # no parked_task_ids → old behavior: requeue
    assert tm.get_task("p").status == "PENDING"


async def test_recover_agent_rebuilds_pending_hitl_and_parks() -> None:
    import asyncio
    from datetime import datetime, timezone
    from ctx_weft.core import CtxWeftRuntime
    from ctx_weft.protocols.events import Event, EventType
    from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="should not run")])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    ts = datetime(2026, 6, 12, tzinfo=timezone.utc)
    def ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"evt_{seq:04d}", run_id="run_1", sequence=seq, session_id="ses_1",
                     type=type_, timestamp=ts, task_id=task_id, payload=payload)

    seed = [
        ev(1, EventType.SESSION_CREATED, user_prompt="do it", template_id="agent:tpl_echo",
           root_agent_id="agt_root"),
        ev(2, EventType.RUN_STARTED),
        ev(3, EventType.TASK_CREATED, task={
            "id": "tsk_1", "status": "PENDING", "title": "T1",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root"}),
        ev(4, EventType.TASK_STARTED, task_id="tsk_1", assigned_agent_id="agt_root"),
        ev(5, EventType.HITL_REQUIRED, task_id="tsk_1", hitl_id="hit_1", form="question",
           capability_id="control:ask_user", tool_call_id="tcA", question="Which DB?"),
        ev(6, EventType.TASK_SUSPENDED, task_id="tsk_1"),
    ]
    for e in seed:
        await runtime.event_store.append(e)

    await runtime.recover_agent("agt_root")
    await asyncio.sleep(0)

    pend = runtime.hitl_registry.list_pending(session_id="ses_1")
    assert len(pend) == 1 and pend[0].tool_call_id == "tcA"
    assert llm.last_request is None             # parked task did not run


def test_restore_keeps_active_parked_task_out_of_queue() -> None:
    """缺陷 A：审批热等的任务恒为 ACTIVE；有未决 HITL 时 restore 必须保持挂起、不重排入队。

    park 判据应看"有无未决 HITL"（parked_task_ids），而非 task.status==SUSPENDED。
    """
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.task import Task
    tm = TaskManager(session_id="s1")
    parked = Task(id="t1", session_id="s1", status="ACTIVE")   # 审批热等 → ACTIVE
    tm.restore([parked], terminal_ids=set(), parked_task_ids={"t1"})
    assert not tm._queue.has_pending(), "ACTIVE+parked 任务不应被重排入队"
    assert tm.get_task("t1").status == "ACTIVE", "parked 任务状态不应被改成 PENDING"


def _tm_with_hooks():
    """一个装好 on_session_idle/on_session_done 回调探针的 TaskManager（本文件两条
    会话收尾用例共用）。

    2026-09-04（Task 12）起 `announce_queue_state`/`TaskQueueBlocked`/`TaskQueueDrained`
    已停发（其消费者——会话状态机——早已退役，events-v2 §5）：「会话终不终结」不再
    有事件可观测，只能从 `_fire_session_idle`/`_fire_session_done` 有没有被调、
    调用后 `session.status` 落到了什么值来判断。不再需要真实 event bus——
    `TaskManager._emit` 对 `event_bus=None` 是 no-op（既有语义），本文件这两条用例
    从来不关心事件本身，只关心收尾走哪条分支。
    """
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.orchestrator.task.hooks import TaskManagerHooks
    from ctx_weft.core.models.session import Session

    session = Session(id="s1", tenant_id="default", user_prompt="x", status="RUNNING", token_budget=0)
    tm = TaskManager(session_id="s1", max_concurrent=1)
    tm.set_session(session)

    idle_calls: list = []
    done_calls: list = []

    async def _on_idle():
        idle_calls.append(1)

    async def _on_done():
        done_calls.append(1)

    tm.set_hooks(TaskManagerHooks(on_session_idle=_on_idle, on_session_done=_on_done))

    async def _noop_runner(_sid, _tid):
        return None

    tm.set_runner(StubRunner(tm, _noop_runner))
    return tm, session, idle_calls, done_calls


async def test_session_not_finished_while_a_task_parked_on_hitl() -> None:
    """多任务：一个完成、另一个仍在等人 → 走 idle 收尾，不落终态。

    Task 6：「还有人在等」不再靠注入的 pending-HITL 谓词，而是由 AWAITING_HUMAN 的
    任务自己表达。2026-09-04（Task 12）起「会话终不终结」不再由 SM 据 TaskQueueBlocked/
    TaskQueueDrained 判定（那条信号已停发）——直接看走的是 `_fire_session_idle`
    （on_session_idle 回调）还是 `_fire_session_done`（on_session_done 回调 +
    落定终态）。
    """
    from ctx_weft.core.models.task import NormalTaskSettings, Task

    tm, session, idle_calls, done_calls = _tm_with_hooks()
    tm.register_task(Task(id="A", session_id="s1", status="ACTIVE", settings=NormalTaskSettings()))
    tm.register_task(Task(id="B", session_id="s1", status="AWAITING_HUMAN",
                          settings=NormalTaskSettings()))
    tm._running_tasks.add("A")
    await tm.on_task_finished("A", status="FINISHED")

    assert idle_calls == [1], "仍有等人的任务时应走 idle 收尾"
    assert done_calls == [], "仍有等人的任务时不得报「跑完了」"
    assert session.status == "RUNNING", "会话未终结，状态不应被改写"


async def test_session_finishes_when_no_pending_hitl() -> None:
    """对照：没有任何等人/中断的任务时，走终态收尾，session.status 落定 SUCCEEDED。"""
    from ctx_weft.core.models.task import NormalTaskSettings, Task

    tm, session, idle_calls, done_calls = _tm_with_hooks()
    tm.register_task(Task(id="A", session_id="s1", status="ACTIVE", settings=NormalTaskSettings()))
    tm._running_tasks.add("A")
    await tm.on_task_finished("A", status="FINISHED")

    assert done_calls == [1]
    assert idle_calls == []
    assert session.status == "SUCCEEDED"


async def test_recover_emits_paused_hitl_for_pending_session() -> None:
    """缺陷 C：启动恢复时，有 pending HITL 的会话必须如实反映「在等人」，
    而非停在崩溃前的 RUNNING。

    Task 16 起：SM 那层「代 TM 发 TaskQueueBlocked → 译成 SessionWaiting」的翻译
    整体退役（会话状态机随 SessionRegistry 降格一并删除）。2026-09-04（Task 12，
    events-v2 §5）起 `TaskQueueBlocked` 本身也停发——`recover()` 不再代 TM 合成
    这条会话级信号（其消费者早已不存在）。「等的是审批面板（PAUSED_HITL）还是
    一句话（PAUSED）」这条区分本就不靠它——那是 delivery 的性质，2026-09-04
    Task 14 起 core 不再替 host 派生这个串；host 自己拿 `list_pending_hitl(...)`
    的 `delivery` 字段判就是了，本测试因此只钉这一个原始事实；
    `_no_stray_session_status_changed` 仍守着「已退役的通用 setter 别再复活」
    这条不相关的回归线。
    """
    from ctx_weft.protocols.events import EventType
    from ctx_weft.protocols.hitl import ToolResultDelivery

    runtime, signals = _recover_runtime_with_regression_guard()
    seed = [
        _mk_ev(1, EventType.SESSION_CREATED, user_prompt="x", template_id="tpl", root_agent_id="agt"),
        _mk_ev(2, EventType.RUN_STARTED),
        _mk_ev(3, EventType.TASK_STARTED, task_id="t1", assigned_agent_id="agt"),
        _mk_ev(4, EventType.HITL_REQUIRED, task_id="t1", hitl_id="h1", form="approval",
               capability_id="fs:bash_exec", tool_call_id="tc1", question="ok?"),
    ]
    for e in seed:
        await runtime.event_store.append(e)

    await rebuild_all_active(runtime)
    assert signals == [], f"已退役的 SessionStatusChanged 不应重新出现: {signals}"
    # 面板 vs 一句话的区分仍在，只是原始事实（delivery）搬去了 host 的只读入口。
    pending = runtime.list_pending_hitl(session_id="ses_1")
    assert isinstance(pending[0].delivery, ToolResultDelivery)


def _recover_runtime_with_regression_guard():
    """构造一个会捕获已退役 `SessionStatusChanged` 的 runtime（recover 状态语义测试共用）。

    Task 16 前这里还捕 SM 译出的会话级事件（`SessionWaiting`/`SessionInterrupted`），
    Task 12（2026-09-04，events-v2 §5）前还捕 `TaskManager` 代发的队列聚合信号
    （`TaskQueueBlocked`/`TaskQueueInterrupted`）——两者现已先后停发，本文件不再
    钉着它们（`list_pending_hitl(...)` 的 `delivery` 字段才是这几条测试的真实
    观测点，见各测试 docstring；2026-09-04 Task 14 起 core 不再替 host 派生
    `session_status_after_recover` 这个串）。仍然保留 `SessionStatusChanged`：
    那是更早一轮退役的通用 setter，一旦重新出现也要当场被抓到。
    """
    from ctx_weft.protocols.events import EventType
    from ctx_weft.providers.llm.mock import MockLLMAdapter
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=InlineAgentTemplateProvider())
    signals: list = []

    async def _cap(ev):
        if ev.type == EventType.SESSION_STATUS_CHANGED:
            signals.append(ev)

    runtime.event_bus.subscribe(None, _cap)
    return runtime, signals


def _mk_ev(seq, type_, **payload):
    from datetime import datetime, timezone
    from ctx_weft.protocols.events import Event
    task_id = payload.pop("task_id", None)
    return Event(id=f"e{seq}", run_id="r1", sequence=seq, session_id="ses_1", type=type_,
                 timestamp=datetime(2026, 6, 12, tzinfo=timezone.utc), task_id=task_id, payload=payload)


async def test_recover_emits_paused_for_wait_only_pending() -> None:
    """wait-only pending（纯文本软待命）：会话判 WAITING，host 侧从 `delivery` 判出
    「没有面板要答」（旧串是 PAUSED）——与 SESSION_PAUSED_HITL 的 reducer/投影语义
    一致（form=wait → 无面板）。

    2026-09-04（Task 12）起不再断言 TaskQueueBlocked（已停发，见上一条测试
    docstring）；2026-09-04（Task 14）起 PAUSED vs PAUSED_HITL 这条派生串本身
    也已删除，不靠这条信号分流表单类型的事实改由 `delivery` 类型直接钉。
    """
    from ctx_weft.protocols.events import EventType
    from ctx_weft.protocols.hitl import UserTurnDelivery

    runtime, signals = _recover_runtime_with_regression_guard()
    seed = [
        _mk_ev(1, EventType.SESSION_CREATED, user_prompt="x", template_id="tpl", root_agent_id="agt"),
        _mk_ev(2, EventType.RUN_STARTED),
        _mk_ev(3, EventType.TASK_STARTED, task_id="t1", assigned_agent_id="agt"),
        _mk_ev(4, EventType.HITL_REQUIRED, task_id="t1", hitl_id="h1", form="wait",
               capability_id="control:wait_for_user", tool_call_id="tc1"),
    ]
    for e in seed:
        await runtime.event_store.append(e)

    await rebuild_all_active(runtime)
    assert signals == [], f"已退役的 SessionStatusChanged 不应重新出现: {signals}"
    # wait-only 不应误标「有面板」——前端会等一个不存在的面板。
    pending = runtime.list_pending_hitl(session_id="ses_1")
    assert isinstance(pending[0].delivery, UserTurnDelivery)


async def test_recover_emits_paused_hitl_when_wait_mixed_with_question() -> None:
    """混合 pending（wait + question/approval）：会话仍判 WAITING，host 侧从
    `delivery` 判出「有面板可答」（旧串是 PAUSED_HITL）——wait 那条是
    `UserTurnDelivery`，question 那条带 `tool_call_id`,是 `ToolResultDelivery`。"""
    from ctx_weft.protocols.events import EventType
    from ctx_weft.protocols.hitl import ToolResultDelivery, UserTurnDelivery

    runtime, signals = _recover_runtime_with_regression_guard()
    seed = [
        _mk_ev(1, EventType.SESSION_CREATED, user_prompt="x", template_id="tpl", root_agent_id="agt"),
        _mk_ev(2, EventType.RUN_STARTED),
        _mk_ev(3, EventType.TASK_STARTED, task_id="t1", assigned_agent_id="agt"),
        _mk_ev(4, EventType.HITL_REQUIRED, task_id="t1", hitl_id="h1", form="wait",
               capability_id="control:wait_for_user", tool_call_id="tc1"),
        _mk_ev(5, EventType.HITL_REQUIRED, task_id="t1", hitl_id="h2", form="question",
               capability_id="control:ask_user", tool_call_id="tc2", question="which?"),
    ]
    for e in seed:
        await runtime.event_store.append(e)

    await rebuild_all_active(runtime)
    assert signals == [], f"已退役的 SessionStatusChanged 不应重新出现: {signals}"
    pending = {r.id: r for r in runtime.list_pending_hitl(session_id="ses_1")}
    assert isinstance(pending["h1"].delivery, UserTurnDelivery)
    assert isinstance(pending["h2"].delivery, ToolResultDelivery)


async def test_recover_does_not_redispatch_task_running_in_live_tm() -> None:
    """单例：已有活 TM 正在跑 X 时，`/resume` 就地复用它——不另建 TM，也不重排 X。

    构造：活 owner 正在跑 X，X 在事件里是 ACTIVE、非 parked、非终态。从前 recover 会
    建一个新 TM 顶替它，靠 inflight 过滤避开 X；现在根本不建第二个 TM，而活 owner 的
    `requeue_resumable` 本就跳过在跑的 task。任一处破了，X 都会被再派发一次（echo
    模板 → MockLLM 被调用），同一个 agent 上并发出两个 run。
    """
    import asyncio
    from datetime import datetime, timezone
    from ctx_weft.core import CtxWeftRuntime
    from ctx_weft.protocols.events import Event, EventType
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import NormalTaskSettings, Task
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.providers.llm.mock import MockLLMAdapter, MockResponse
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[MockResponse(text="should not run")])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    # 活 owner：经 runtime 真实 wiring 登记，正在跑 X
    old_tm = TaskManager(session_id="ses_1", event_bus=runtime.event_bus, max_concurrent=1)
    old_tm.set_session(Session(id="ses_1", tenant_id="default", user_prompt="do it",
                               status="RUNNING", root_agent_id="agt_root", token_budget=0))
    old_tm.set_runner(StubRunner(old_tm, session_id="ses_1"))
    old_tm.register_task(Task(id="tsk_X", session_id="ses_1", status="ACTIVE",
                              assigned_agent_id="agt_root", creator_agent_id="agt_root",
                              settings=NormalTaskSettings()))
    old_tm._running_tasks.add("tsk_X")
    runtime._register_and_drain(old_tm.session, old_tm)

    ts = datetime(2026, 6, 12, tzinfo=timezone.utc)

    def ev(seq, type_, **payload):
        task_id = payload.pop("task_id", None)
        return Event(id=f"e{seq}", run_id="r1", sequence=seq, session_id="ses_1", type=type_,
                     timestamp=ts, task_id=task_id, payload=payload)

    seed = [
        ev(1, EventType.SESSION_CREATED, user_prompt="do it", template_id="agent:tpl_echo", root_agent_id="agt_root"),
        ev(2, EventType.RUN_STARTED),
        ev(3, EventType.TASK_CREATED, task={
            "id": "tsk_X", "status": "ACTIVE", "title": "X",
            "assigned_agent_id": "agt_root", "creator_agent_id": "agt_root"}),
        ev(4, EventType.TASK_STARTED, task_id="tsk_X", assigned_agent_id="agt_root"),
    ]
    for e in seed:
        await runtime.event_store.append(e)

    await runtime.recover_agent("agt_root")
    await asyncio.sleep(0)

    assert runtime._task_managers["ses_1"] is old_tm, "活 owner 不得被新建的 TM 顶替"
    assert llm.last_request is None, "X 正被活 owner 执行，不应被重复派发"
    assert old_tm.running_task_ids() == {"tsk_X"} and not old_tm._queue.peek_all()


async def test_cold_answer_reuses_live_owner_instead_of_rebuilding(monkeypatch) -> None:
    """单 owner 架构：冷 HITL 应答且存活 owner 拥有该 task → 就地重驱、**不重建 TM**。

    验证：(1) rebuild_view 未被调用(走复用而非重建)；(2) 被应答的 task 重新派发；
    (3) owner TM 未被顶替。**不再验证**「model 写回 session」——`recover_agent`
    不再接受 llm_account/llm_model，换模型走 `set_agent_llm`/
    `set_session_llm` 两条独立命令（Task 9）。
    """
    import asyncio
    from ctx_weft.core import CtxWeftRuntime
    from ctx_weft.core.orchestrator.task.manager import TaskManager
    from ctx_weft.core.models.session import Session
    from ctx_weft.core.models.task import NormalTaskSettings, Task
    from ctx_weft.providers.llm.mock import MockLLMAdapter
    from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime

    runtime = make_runtime(llm=MockLLMAdapter(responses=[]), agent_provider=InlineAgentTemplateProvider())

    rebuild_calls: list = []
    import ctx_weft.core.control.reducers as _reducers
    _orig = _reducers.rebuild_view

    async def _spy(*a, **k):
        rebuild_calls.append(1)
        return await _orig(*a, **k)

    monkeypatch.setattr(_reducers, "rebuild_view", _spy)

    session = Session(id="ses_1", tenant_id="default", user_prompt="x", status="RUNNING",
                      root_agent_id="agt", llm_provider="acct1", llm_model="m1", token_budget=0)
    # `recover_agent` 的主键是 agent_id，session_id 由 ALM 记录反查——这个测试从不
    # 经事件回放建 agent 记录（直接手搭 Session/TaskManager），故种一条最小记录，
    # 绕开 instantiate 的模板依赖（同 `test_agent_lifecycle.py::_plant` 的口径）。
    from ctx_weft.core.orchestrator.lifecycle.agent_manager import _AgentRecord
    runtime._agent_lifecycle_manager._agents["agt"] = _AgentRecord(
        session_id="ses_1", tenant_id="default", template_id="tpl",
        parent_agent_id=None, spawn_depth=0, memory_config=None, loop_config=None,
    )
    tm = TaskManager(session_id="ses_1", event_bus=runtime.event_bus, max_concurrent=1)
    tm.set_session(session)
    tm.set_hooks(TaskManagerHooks(
        is_current=lambda: runtime._task_managers.get("ses_1") is tm))

    ran: list = []

    async def _runner(_sid, tid):
        ran.append(tid)
        # 重新 park，避免 stub 无限重排；owner 保持存活。Task 4 起「这次 run 停在哪」
        # 走返回值（RunOutcome），不再由 runner 直接写 task.status。
        ran_outcome = RunOutcome(kind=RunOutcomeKind.SUSPENDED_ON_CHILDREN)
        return ran_outcome

    tm.set_runner(StubRunner(tm, _runner))
    tm.register_task(Task(id="tsk_A", session_id="ses_1", status="SUSPENDED", settings=NormalTaskSettings()))
    runtime._task_managers["ses_1"] = tm

    await runtime.recover_agent("agt", resumed_task_id="tsk_A")
    for _ in range(10):
        await asyncio.sleep(0)

    assert rebuild_calls == [], "复用路径不应调用 rebuild_view（未重建 TM）"
    assert ran == ["tsk_A"], "被应答的 task 应就地重新派发"
    assert runtime._task_managers["ses_1"] is tm, "owner TM 不应被顶替/替换"


async def test_crash_mid_batch_routes_to_reconcile() -> None:
    from datetime import datetime, timezone, timedelta
    from ctx_weft.core.runtime import _task_has_dangling_tool_call
    from ctx_weft.protocols import MemoryEventType, MemoryAddress, ProviderContext
    from ctx_weft.protocols.memory import MemoryEvent
    from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider

    base = datetime(2026, 6, 12, tzinfo=timezone.utc)
    mem = InMemoryMemoryProvider()
    pctx = ProviderContext(session_id="s1", tenant_id="default")
    sc = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    await mem.ingest(MemoryEvent(type=MemoryEventType.LLM_RESPONSE, address=sc, content="",
        timestamp=base + timedelta(seconds=1), role="assistant",
        metadata={"tool_calls": [{"id": "x1", "name": "web", "input": {}},
                                 {"id": "x2", "name": "web", "input": {}}]}), pctx)
    await mem.ingest(MemoryEvent(type=MemoryEventType.TOOL_RESULT, address=sc, content="r1",
        timestamp=base + timedelta(seconds=2), role="tool", metadata={"tool_call_id": "x1"}), pctx)
    assert await _task_has_dangling_tool_call(mem, sc, pctx) is True   # x2 dangling → reconcile
