"""模板跟着**执行 agent**，不跟着 session（2026-09-30）。

## bug 的形状

`_SessionTaskRunner` 曾持有一对 `_template` / `_template_id`，装的是建 runner 那一刻按
「会话的模板」解出来的那份，而 `assemble` 的非 subagent 分支（`case _`）直接把它配给执行
agent。runner 是 per-session 的，执行 agent 不是——`send_message` 可以把新一轮开在任意一个
park 中的子 agent 上，而 `_start_task_for_agent` 建的 task 正是 `use_subagent=False`，恰好
落那条分支。于是那个子 agent 跑 root 的 identity、root 的 capability 授权（它自己
`capability_refs` 里的 forbidden 一并失效）。

这个错误还是**自洽**的：同一次 run 的 `provider_ctx.agent_template_id` 报的是 agent 自己的
模板，`state.extra["template"]` 却是 root 的，两处谁也不检查谁。宿主侧的表现是「回子 agent
的第二句话开始它换了个人格」——第一轮（委派时新建，走 subagent 分支）反而是对的。

现在模板的唯一真相是 agent record 的 `template_id`（`AgentInstantiated` 钉死的出身），每次
装配现解，见 `_SessionTaskRunner._template_of`。

## 兜底

解不出来时的兜底只有一处：`RuntimeConfig.fallback_template_ref`（host 声明的母版），由
`TemplateLookup.get_template` 内部消化。registry **不再**在 agent 的 template_id 缺失时借
会话那份——见本文件最后两个用例。
"""

from __future__ import annotations

import dataclasses

import pytest

from ctx_weft import RuntimeConfig
from ctx_weft.core.control.types import AgentView
from ctx_weft.core.models.errors import TemplateNotFoundError
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.runtime import _SessionTaskRunner
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


SID = "ses_1"
ROOT, SUB = "agt_root", "agt_sub"


class _StubTaskManager:
    """assemble 只碰 `get_task`；execute 那半不在本文件范围内。"""

    def __init__(self, task: Task) -> None:
        self._task = task

    def get_task(self, task_id):
        return self._task if task_id == self._task.id else None


def _runtime(*, fallback_ref: str = ""):
    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())                                  # id="tpl_echo"
    resolver.register(dataclasses.replace(make_echo_template(), id="tpl_sub"))
    return make_runtime(
        llm=MockLLMAdapter(responses=[]), agent_provider=resolver,
        config=RuntimeConfig(fallback_template_ref=fallback_ref),
    )


def _session() -> Session:
    return Session(id=SID, tenant_id="default", user_prompt="x",
                    root_agent_id=ROOT)


async def _runner(rt, task: Task) -> _SessionTaskRunner:
    memory = InMemoryMemoryProvider()
    rt.providers.register_memory(memory)
    return _SessionTaskRunner(
        runtime=rt, session=_session(), lm=rt._agent_lifecycle_manager,
        memory=memory, task_manager=_StubTaskManager(task),
    )


def _user_message_task(agent_id: str) -> Task:
    """`send_message` 新建分支造出来的那种 task：非 subagent，挂在某个 agent 名下。"""
    return Task(
        id="tsk_1", session_id=SID, status="ACTIVE", tenant_id="default",
        assigned_agent_id=agent_id, creator_agent_id=agent_id,
        settings=NormalTaskSettings(),
    )


async def _load_two_agents(rt, *, sub_template_id: str = "agent:tpl_sub") -> None:
    await rt._agent_lifecycle_manager.load(
        {
            ROOT: AgentView(id=ROOT, spawn_depth=0, parent_agent_id=None,
                            template_id="agent:tpl_echo"),
            SUB: AgentView(id=SUB, spawn_depth=1, parent_agent_id=ROOT,
                           template_id=sub_template_id),
        },
        session_id=SID, tenant_id="default",
    )


@pytest.mark.asyncio
async def test_a_user_message_to_a_sub_agent_uses_the_sub_agents_template():
    """这就是 bug 本身：从前这里拿到的是 root 的 `tpl_echo`。"""
    rt = _runtime()
    await _load_two_agents(rt)
    task = _user_message_task(SUB)
    runner = await _runner(rt, task)

    binding = await runner.assemble(task.id)

    assert binding is not None
    assert binding.agent_id == SUB
    assert binding.template.id == "agent:tpl_sub"    # 它自己的出身，不是 root 的
    assert binding.agent.template_id == "agent:tpl_sub"   # 与 record 一致，两处不再打架


@pytest.mark.asyncio
async def test_a_user_message_to_the_root_agent_still_uses_the_root_template():
    """反向那一半：修完不能把 root 也搞错（root task 是这条分支上最常走的一支）。"""
    rt = _runtime()
    await _load_two_agents(rt)
    task = _user_message_task(ROOT)
    runner = await _runner(rt, task)

    binding = await runner.assemble(task.id)

    assert binding is not None
    assert binding.agent_id == ROOT
    assert binding.template.id == "agent:tpl_echo"


@pytest.mark.asyncio
async def test_a_missing_template_id_falls_back_to_the_hosts_master_template(caplog):
    """agent 没有自己的模板（存量事件流没发过 AgentInstantiated）→ 用 host 声明的母版。

    **不是** root 的模板：兜底来源是配置，不是这个会话碰巧用了什么。
    """
    import logging
    rt = _runtime(fallback_ref="agent:tpl_echo")
    await _load_two_agents(rt, sub_template_id="")
    task = _user_message_task(SUB)
    runner = await _runner(rt, task)

    with caplog.at_level(logging.WARNING):
        binding = await runner.assemble(task.id)

    assert binding is not None
    assert binding.template.id == "agent:tpl_echo"   # 母版，恰好与 root 同名（配置如此）
    assert binding.agent.template_id == ""           # record 仍然诚实地说「我不知道」
    assert "解析不出来" in caplog.text                # 每次装配都留痕，不静默


@pytest.mark.asyncio
async def test_without_a_fallback_a_missing_template_id_raises_instead_of_borrowing():
    """没配母版就抛，**不借会话的模板**。

    这是 2026-09-30 的行为变化，也是删掉 registry 那条借道的代价：从前这里会静默地让
    子 agent 冒充 root 跑下去。抛出来是刻意的——一个身份不明的 agent 该被看见。
    """
    rt = _runtime()                       # fallback_template_ref 留空 = 历史严格模式
    await _load_two_agents(rt, sub_template_id="")
    task = _user_message_task(SUB)
    runner = await _runner(rt, task)

    with pytest.raises(TemplateNotFoundError):
        await runner.assemble(task.id)
