"""纯文本回合的两种归宿（有人在场 park / 无人值守即产出）+ finish_task + 临时 guidance 注入。

判据 2026-09-22 起是 `Task.unattended`——`interaction_mode` 已删（它是个 task 级静态
配置，却在回答一个逐回合的语义）。文件名沿用旧称，内容测的是同一组语义。

- interactive 任务 actor 纯文本 → HITL input 冷 park（等用户），不产出、不完成。
- auto 任务 actor 纯文本 → 文本即 outputs，路由 observe。
- finish_task 收尾标记 → 答复正文即 outputs + 完成（端到端经 gateway/observe/finalize）。
- 运行时 guidance 现由装配管线注入（PrepareStep → extra["act_guidance"] →
  GuidanceSource → composer 末条 user 尾部）；本文件 fixture 按该形态预拼进
  assembled prompt，验证 ActStep 原样发送、guidance 不入 memory。
"""

from __future__ import annotations

from tests.model_testkit import model_snapshot

from types import SimpleNamespace

import pytest

from ctx_weft.core.assembler.assembler import AssembledPrompt
from ctx_weft.core.loop.driver import LoopContext, LoopState
from ctx_weft.core.loop.signals import HitlPark
from ctx_weft.core.loop.steps.act import ActStep
from ctx_weft.core.models.agent import Agent
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.protocols import (
    LLMMessage,
    MemoryAddress,
    MemoryEventType,
    ProviderContext,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import PREFACE_NORMAL, UserTurnDelivery
from ctx_weft.providers.events import InProcessEventBus
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.assembler_testkit import compiled_test_inputs, scripted_inputs
from tests.hitl_env import make_hitl
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)



def _template(*, observe_role: bool):
    """最小 template。``observe_role`` = 这个 agent 有没有 observer（ROLE facet）。

    没有 observer 的模板（2026-09-28 起 ROLE 不再从 default 借）在三条路上降级：前台
    observe 走机械判决、后台判定档降成只摘要档、`finish_task` 不再让位。
    """
    from ctx_weft.protocols.template import (
        AgentTemplate,
        IdentityFacet,
        LoopConfig,
        MemoryConfig,
    )

    identity = {"act": IdentityFacet(text="SOUL")}
    if observe_role:
        identity["observe"] = IdentityFacet(text="ROLE")
    return AgentTemplate(
        id="t", name="t", version="1.0.0", identity=identity,
        capability_refs=[], memory_config=MemoryConfig(), loop_config=LoopConfig(),
    )


def _act_state_ctx(unattended: bool, llm: MockLLMAdapter, *,
                   parent_task_id: str | None = None, gateway=None,
                   observe_role: bool = True):
    """返回 (state, ctx, task, registry, mem)。第 4 项是 `HitlRegistry`——`list_pending`
    的持有者从 `HitlManager` 换成了它，断言口径不变（仍是「这个 session 上挂着哪些未决
    请求」）。

    参数 2026-09-22 从 `interaction_mode: str` 换成 `unattended: bool`：纯文本回合的
    归宿判据改成了「有没有人在」（`interaction_mode` 随 S7 删除）。映射是逐字的——
    旧的 `"auto"`（自治、纯文本即产出）≡ `unattended=True`，旧的 `"interactive"`
    （让位等人）≡ `unattended=False`。

    ``parent_task_id`` / ``gateway`` 是 2026-09-27 加的（S-b）：root 的 `finish_task` 现在也
    让位，而「是不是 root」与「工具真被调过」正是那条路由的两个判据——前者要能摆出子任务，
    后者要一个能把 `task.actor_done` 置上的 gateway（见
    `tests/unit/test_finish_task_parks_on_root.py`）。两者都默认不变。
    """
    bus = InProcessEventBus()
    mem = InMemoryMemoryProvider()
    hitl_service, hitl = make_hitl(bus)
    session = Session(id="s1", tenant_id="default", user_prompt="hi")
    task = Task(
        id="t1", session_id="s1", status="ACTIVE",
        title="Greet", description="say hi politely",
        unattended=unattended, settings=NormalTaskSettings(),
        parent_task_id=parent_task_id,
    )
    agent = Agent(id="ag1", session_id="s1", template_id="t")
    scope = MemoryAddress(session_id="s1", task_id="t1", agent_id="ag1")
    # Assembler 责任不在这个 ActStep fixture 内；用一段已渲染的最小 guidance
    # 保留「只出现在发送 prompt，不落 memory」这个边界条件。
    guidance = (
        "Write the actual final answer as normal message text."
    )
    prompt = AssembledPrompt(
        system="", messages=[LLMMessage(role="user", content=f"hi\n\n{guidance}")],
        tools=[], token_count=1,
    )
    state = LoopState(
        run_id="r1", session=session, task=task, agent=agent, scope=scope, prompt_inputs=scripted_inputs(prompt),
        resolved_model=model_snapshot(model="mock", account=""),
        extra={"template": _template(observe_role=observe_role)},
    )
    ctx = LoopContext(
        assembler=SimpleNamespace(compile=compiled_test_inputs), llm=llm, memory=mem, event_bus=bus,
        provider_ctx=ProviderContext(session_id="s1", tenant_id="default", task_id="t1", agent_id="ag1"),
        hitl=hitl_service,
        capability_gateway=gateway,
    )
    return state, ctx, task, hitl, mem


@pytest.mark.asyncio
async def test_interactive_plain_text_parks_for_user() -> None:
    llm = MockLLMAdapter(responses=[MockResponse(text="Hi! Anything else?")])
    state, ctx, task, hitl, mem = _act_state_ctx(False, llm)

    with pytest.raises(HitlPark):
        await ActStep().execute(state, ctx)

    # park 不写 task 状态（Task 4）：AWAITING_HUMAN 由 TaskManager 据 RunOutcome 落。
    assert task.status == "ACTIVE"
    # 纯文本**就是**这一段的产出（2026-09-24 修）：park 之前必须先合成 `outputs`，
    # 否则 `report_observation` 的 success-without-outputs 护栏会把 observer 的
    # success 改判 retry，park 的 task 永远终结不了。见
    # `tests/unit/test_outputs_synthesized_before_park.py`。
    assert task.outputs == "Hi! Anything else?"
    pend = hitl.list_pending("s1")
    assert len(pend) == 1
    assert pend[0].form == "wait"
    # 续跑方式由 delivery 显式声明——取代旧的 `capability_id` sentinel（spec §5）。
    assert pend[0].delivery == UserTurnDelivery(task_id="t1", preface=PREFACE_NORMAL)

    # guidance 只在发送的 prompt（装配期已拼入），不入 memory
    sent = llm.last_act_request.messages[-1].content
    assert "actual final answer" in sent and "finish_task" not in sent
    recs = await mem.recall_recent(state.scope, [MemoryEventType.LLM_RESPONSE], 10, ctx.provider_ctx)
    assert recs and all("final reply to the user" not in (r.content or "") for r in recs)


@pytest.mark.asyncio
async def test_auto_plain_text_completes() -> None:
    llm = MockLLMAdapter(responses=[MockResponse(text="Hi! Anything else?")])
    state, ctx, task, hitl, _mem = _act_state_ctx(True, llm)

    outcome = await ActStep().execute(state, ctx)

    assert outcome.next_step == "observe"
    assert task.status != "SUSPENDED"
    assert task.outputs == "Hi! Anything else?"        # auto: 文本即产出（旧行为）
    assert hitl.list_pending("s1") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("unattended", [False, True])
async def test_reply_links_memory_to_its_prompt(unattended) -> None:
    llm = MockLLMAdapter(responses=[MockResponse(text="linked reply")])
    state, ctx, _task, _hitl, mem = _act_state_ctx(unattended, llm)
    events = []

    async def collect(event):
        events.append(event)

    ctx.event_bus.subscribe(None, collect)
    if unattended:
        await ActStep().execute(state, ctx)
    else:
        with pytest.raises(HitlPark):
            await ActStep().execute(state, ctx)

    reply, = [event for event in events if event.type == EventType.LLM_RESPONSE_FINISHED]
    prompt, = [event for event in events if event.type == EventType.LLM_PROMPT_SENT
               and event.payload["request_id"] == reply.payload["request_id"]]
    record, = await mem.read_records([reply.payload["memory_id"]], ctx.provider_ctx)
    assert record.role == "assistant"
    assert record.content == reply.payload["content"] == "linked reply"
    assert "request_id" not in record.metadata
    assert prompt.task_id == reply.task_id == record.address.task_id


@pytest.mark.asyncio
async def test_finish_task_finishes_task_end_to_end() -> None:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    # 反转契约：答复写在消息正文，finish_task 无参收尾标记 → outputs = 正文
    llm = MockLLMAdapter(responses=[
        MockResponse(text="computed: 42", tool_calls=[]),
    ])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    runtime.providers.register_memory(InMemoryMemoryProvider())

    await initialize_all(runtime)
    _handle, state = await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="compute")

    assert state.task.status == "FINISHED"
    assert state.task.outputs == "computed: 42"


@pytest.mark.asyncio
async def test_guidance_injected_into_prompt_not_memory() -> None:
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    llm = MockLLMAdapter(responses=[
        MockResponse(text="ok", tool_calls=[]),
    ])
    runtime = make_runtime(llm=llm, agent_provider=resolver)
    mem = InMemoryMemoryProvider()
    runtime.providers.register_memory(mem)

    await initialize_all(runtime)
    _handle, state = await runtime.run_single_task(template_id="agent:tpl_echo", user_prompt="hello")

    # 发送的 prompt 含 guidance
    # `last_act_request` 而非 `last_request`：2026-09-28 起 root 的收尾也过前台 observe
    # （`_should_use_llm` 删掉了 root 降级），一次 run 的最后一个 request 是观察者的。
    sent = llm.last_act_request.messages[-1].content
    assert "actual final answer" in sent and "finish_task" not in sent
    # memory 的 USER_PROMPT 不含 guidance（仅原始用户输入）
    ctxp = ProviderContext(session_id=state.session.id, agent_id=state.agent.id)
    user_recs = await mem.recall_recent(
        state.scope, [MemoryEventType.USER_PROMPT], 10, ctxp,
    )
    assert user_recs and all("finish_task" not in (r.content or "") for r in user_recs)
