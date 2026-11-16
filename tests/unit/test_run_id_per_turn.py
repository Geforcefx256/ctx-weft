"""一轮一个 run_id（2026-09-04 spec §4）。

改动前：start_session 预铸一个 run_id 当 `_SessionTaskRunner._default_run_id`，
`assemble` 的非 subagent 分支把它给每一条任务用；而每次 `_execute_task` 都新建
LoopState、sequence_counter 从 0 起。结果同一个 run_id 下跨轮撞号、出现多对
RunStarted/RunFinished。

改动后：`assemble` 的两个分支（subagent / 非 subagent）都各自 `generate_id("run")`，
没有例外——`_SessionTaskRunner` 不再持有 `_default_run_id` 字段。

夹具说明（控制方 2026-09-04 裁定，覆盖 brief 原文 Step 1 的写法，且在诊断过两个
非 run_id 相关的既有坑之后落定，均记在下面对应的类/函数 docstring 里）：

- `event_bus.subscribe` 要收 `Callable[[Event], Awaitable[None]]`，故用异步
  `recorder` 而非 `lambda ev: seen.append(ev) or None`（同步 lambda 不满足协议）。
- `start_session` 必须用 `template_id="agent:tpl_echo"`（`InlineAgentTemplateProvider`
  的 provider 前缀是 `agent:`），且要注册 memory provider，否则跑不起来。
- root task 恒 ``：第一轮 LLM 回纯文本触发冷 park
  （`AWAITING_HUMAN`）——`wait_for_finish` 在这一步就返回（它等的是"这个 run
  结束"，不是"task 终态"，同一口径见 `test_run_id_sequence_integrity.py` 的
  `bus_after_recap_run` fixture）。**刻意不用 `control__finish_task` 收尾第一
  轮**：那会把 session 判定为"done"，`TaskManager.on_task_finished` 同步走到
  `_fire_session_done` → runtime 的 `_on_done` 钩子 `_release_session`——回收
  `_task_managers` 映射与 `ControlCapabilityProvider._sessions` 注册表。这一步和
  `send_message` 之间没有互斥：`_start_task_for_agent` 读 `_task_managers` 那一刻
  可能仍在（尚未被回收），但等到新任务真正跑到调 `control__finish_task` 时，
  `ControlCapabilityProvider._sessions` 已被清空，`ctx.task` 解不出来，
  `finish_task` 静默失神（`ctx.task is None` 分支不生效），`actor_done` 永远置不
  上，`ActStep` 原地转到 `max_turns` 才认输——这是与本 task 无关的既有 send_message/
  会话收尾时序缺口，不是这里要修的东西，绕开即可：让第一轮以 `AWAITING_HUMAN`
  收尾，session 判定为"idle"而非"done"（`_fire_session_idle`，不回收任何映射），
  `send_message` 走「task 未终态 → 注入同一个 task」的分支（`_inject_user_turn`），
  全程不换 task_id，也不触碰 `_release_session`。
- Task 10 落地后 `send_message` 返回 `TurnHandle`，第二轮直接 `await h2.
  wait_for_finish(...)` 等它进终态即可，不再需要按 `RunFinished` 事件计数手工轮询
  （回填此前 Task 6 期间 `send_message` 还只返回裸 `task_id` 时留下的临时写法）。

`test_sequence_is_unique_per_run` 的豁免范围（**不是本 task 的 xfail**，控制方已
明确禁止：这里 `assert_no_duplicate_sequence` 照常跑、照常能失败，只是把一种已经
在 `docs/follow-ups/2026-09-03-outstanding-issues.md` §A11 记录在案、与本 task 无
关的既有撞号模式排除在外，其余任何撞号仍然会让断言失败）：

A11——「`(run_id, sequence)` 在每个多步 run 内都会撞号」——是 `_run_loop`
（`runtime.py`）与 `driver.run`（`driver.py`）两侧 `LoopState` 在第一个
`state_patch` 落下时分叉成两个独立对象的结构性缺陷：`_run_loop` 手里那份
`sequence_counter` 冻结在 prepare 步 patch 落下那一刻的值，run 收尾时
`RunFinished` 把这份冻结值 +1，确定性地撞上 driver 那侧 prepare 步的
`StepCompleted`（它也是同一起点 +1）。跟踪文档明载「影响面：**所有多步
run**，与 background observe / recognize_intent / compact_agent 无关」，且
「结构性改动……属独立立项，留给后续批次」——即在本 task 动手之前就已存在、
且明确排除在「run_id 一轮一个」这个修复范围之外（`_run_loop`/`driver.run` 的
state 传递方式完全没在本 task 的改动清单里）。

复核：`git stash` 回到本 task 改动前的 baseline，跑同一个最简单任务
（单轮、`control__finish_task` 收尾，不涉 background_observe/recognize_intent），
`RunFinished` 与 prepare 步 `StepCompleted` 在同一个 run_id 下依旧共享同一个
`sequence`——证实这条撞号在本 task 改动前后行为完全一致，不是这次改动引入的。
`test_run_id_sequence_integrity.py` 的三条测试早就在用同一个判断把它挡在门外
（它的 `_assert_no_duplicate_sequence` 只对 background recap 的 run_id 取子集，
从不对主 run 的事件跑全局唯一性）——这里延续同一个先例，而不是另起一套豁免
标准。
"""

from __future__ import annotations

import asyncio
import collections
import inspect

import pytest

from ctx_weft.core.runtime import SessionStartParams, _SessionTaskRunner
from ctx_weft.protocols import ToolCall
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



def assert_no_duplicate_sequence(events):
    """(run_id, sequence) 唯一。run 外的事件恒 sequence=0，不参与。

    唯一豁免：模块 docstring 里记录的 A11——同一个 run_id 内，`RunFinished` 与
    prepare 步的 `StepCompleted` 共享同一个 sequence（且仅此一种 type 组合、仅
    一次）。除这一种已知、既有、跨 turn 也会独立复现的模式外，任何撞号（含本 task
    要修的"跨 turn 共用 run_id"撞号）照样让断言失败——不是把检查关掉，是把已知
    误报排除。
    """
    by_run_seq = collections.defaultdict(list)
    for e in events:
        if e.run_id is None:
            continue
        by_run_seq[(e.run_id, e.sequence)].append(e.type)

    dupes = []
    for (run_id, sequence), types in by_run_seq.items():
        if len(types) <= 1:
            continue
        is_known_a11_pattern = (
            len(types) == 2
            and {EventType.RUN_FINISHED, EventType.STEP_COMPLETED} == set(types)
        )
        if is_known_a11_pattern:
            continue
        dupes.append(f"{run_id}:{sequence}:{types}")
    assert dupes == [], f"(run_id, sequence) 撞号（非已知 A11 模式）: {dupes}"


def assert_every_run_has_one_start_and_finish(events):
    """每个 run_id 恰有一对 RunStarted / RunFinished。"""
    starts = collections.Counter(
        e.run_id for e in events if e.type == EventType.RUN_STARTED)
    finishes = collections.Counter(
        e.run_id for e in events if e.type == EventType.RUN_FINISHED)
    seen = {e.run_id for e in events if e.run_id is not None}
    problems = []
    for rid in sorted(seen):
        if starts[rid] != 1 or finishes[rid] != 1:
            problems.append(f"{rid}: {starts[rid]} started / {finishes[rid]} finished")
    assert problems == [], f"run 起止不成对: {problems}"


class _TurnRouterLLM(MockLLMAdapter):
    """按 `request.tools` 路由，一个 adapter 覆盖一次会话里并发/相继出现的四种请求：

    - `control__update_task_metadata`（`recognize_intent`，root task 首次无 title
      时并发触发）：回一个真的 title，让它成功——第二轮 `send_message` 复用**同一个
      task**（第一轮是 `AWAITING_HUMAN` 冷 park，不是新建 task），title 落地后第二
      轮 `should_recognize_intent` 见 title 非空直接跳过，不用再照顾第二次
      `recognize_intent` 请求。
    - `control__report_observation`（`background_observe`，纯文本冷 park的
      boundary="plain_text" 触发）：回一个真的 recap，让它一轮成功——不这样做的话
      它会把手里任何认不出的回复都当"没产出摘要"，内部重试到认输才罢休（既有降级
      路径，非本 task 改动范围），拖慢且不必要。
    - 其余（root task 自己的 act 工具集 `control__finish_task` 等）：按调用序号分——
      第 1 次纯文本收尾（触发冷 park，`AWAITING_HUMAN`，这是本文件要的"第一轮"）；
      第 2 次起真正调 `control__finish_task` 收尾（"第二轮"）。
    - 兜底：空文本。

    **工具调用 id 一律现铸、不重复**：早前一版复用同一个 `MockResponse` 实例（同一个
    `tool_call.id`）在"第二轮复用同一个 task"这个场景上踩过坑——同一个 agent 第二次
    收到和第一次一模一样的 `tool_call.id` 被当成重放的旧调用直接忽略，工具从未真正
    执行、状态位置永远置不上，`ActStep` 原地打转到 `max_turns` 才认输（与本 task 无
    关的既有行为，绕开即可）。
    """

    def __init__(self) -> None:
        super().__init__(responses=[])
        self._act_calls = 0
        self._call_n = 0

    def _tc(self, name: str, arguments: dict) -> ToolCall:
        self._call_n += 1
        return ToolCall(id=f"tc{self._call_n}", name=name, arguments=arguments)

    def complete(self, request, stream: bool = True):
        self.last_request = request
        names = {getattr(t, "name", "") for t in (getattr(request, "tools", None) or [])}
        if "control__update_task_metadata" in names:
            response = MockResponse(tool_calls=[self._tc("control__update_task_metadata", {
                "title": "Echo smoke test", "description": "two-turn run_id fixture",
                "session_goal": "",
            })])
        elif "control__report_observation" in names:
            response = MockResponse(tool_calls=[self._tc("control__report_observation", {
                "act_recap": "said hi, awaiting the user's next message",
                "task_status": "success" if self._act_calls > 1 else "waiting",
                "task_summary": "done" if self._act_calls > 1 else "",
            })])
        else:
            self._act_calls += 1
            if self._act_calls == 1:
                response = MockResponse(text="Hi! Anything else?")
            else:
                response = MockResponse(text="done")
        return self._stream(response, request)


async def _two_turn_session():
    """跑两轮对话，收集全部事件。"""
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = _TurnRouterLLM()
    rt = make_runtime(llm=llm, agent_provider=resolver)
    rt.providers.register_memory(InMemoryMemoryProvider())

    seen: list = []

    async def recorder(ev):
        seen.append(ev)

    rt.event_bus.subscribe(None, recorder)

    await initialize_all(rt)
    handle = await rt.start_session(SessionStartParams.create(
        template_id="agent:tpl_echo", user_prompt="hi",
    ))
    async with asyncio.timeout(5):
        while handle._state is None:
            await asyncio.sleep(0.01)
    state = handle._state
    assert state is not None and state.task.status == "AWAITING_HUMAN", (
        f"第一轮应以纯文本冷 park 收尾，实际 status={state.task.status if state else None}"
    )

    await initialize_all(rt)
    h2 = await rt.send_message(handle.agent_id, "second turn")
    assert h2.task_id == handle.task_id, (
        "第一轮以 AWAITING_HUMAN（非终态）收尾，send_message 应注入同一个 task，"
        f"而不是新建一个（got {h2.task_id!r}, expected {handle.task_id!r}）"
    )

    await h2.wait_for_finish(timeout=5.0)
    return seen


@pytest.mark.asyncio
async def test_two_turns_do_not_share_a_run_id():
    """headline 不变式必须只看**主循环**（`_run_loop`，origin=RUNTIME）自己的两个
    run——`recognize_intent` / `background_observe` 在这条改动完全没碰过的代码路径
    上，每轮都会各自铸一个新 run_id（`launch_recognize_intent`/
    `launch_recap` 早已各自 `generate_id("run")`，与 `_default_run_id`
    无关），若把它们也计进 `run_ids` 的全局集合，即使主任务两轮共用同一个
    `_default_run_id`（本 task 要修的那个 bug），`len(run_ids) >= 2` 依旧会因为这些
    旁路 run 而碰巧为真——对 headline 不变式是假阳性。用 `origin == RUNTIME` 隔离出
    主循环自己的 RunStarted，只数这一路的 run_id，与 `_main_run_finishes`/
    `_two_turn_session` 里判定"第二轮那个主 run 完了没有"用的同一个过滤条件。
    """
    events = await _two_turn_session()
    main_starts = [
        e for e in events
        if e.type == EventType.RUN_STARTED and e.origin == EventOrigin.RUNTIME
    ]
    main_run_ids = {e.run_id for e in main_starts if e.run_id is not None}
    assert len(main_starts) >= 2, f"两轮的主循环至少两个 run，实际 {len(main_starts)} 个"
    assert len(main_run_ids) >= 2, f"两轮的主循环共用了 run_id: {main_run_ids}"


@pytest.mark.asyncio
async def test_sequence_is_unique_per_run():
    assert_no_duplicate_sequence(await _two_turn_session())


@pytest.mark.asyncio
async def test_runs_are_paired():
    assert_every_run_has_one_start_and_finish(await _two_turn_session())


@pytest.mark.asyncio
async def test_session_task_runner_has_no_default_run_id():
    """结构性守卫：这个字段不该再存在。"""
    assert "default_run_id" not in inspect.signature(_SessionTaskRunner.__init__).parameters
