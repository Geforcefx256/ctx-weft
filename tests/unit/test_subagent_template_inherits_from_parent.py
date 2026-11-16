"""`delegate_task(use_subagent=True)` 不填模板 → 派生一个和自己同型号的子 agent（2026-10-07）。

## 从前是什么

空 `subagent_template` 一路传到 `instantiate(template_id="")`，`TemplateLookup.get_template("")`
四段全落空，于是掉进 `RuntimeConfig.fallback_template_ref`——那是「模板解析不出来」的**故障
兜底**，被当成「没说要造什么」的答案用。后果：一个领域套件的 actor 派出来的无名子 agent 跑的
是宿主母版，没有父模板的领域红线，父 `capability_refs` 里的 forbidden 一并失效。没配兜底的部署
更直接：`TemplateNotFoundError('')`，派一个子任务就炸。

## 现在是什么

空值 = 继承**派生它的那个 agent** 的 `record.template_id`，与 `instantiate` 对 `llm` 的既有规则
同一条（省略→继承父，无父→系统默认）。无父（root）时仍落兜底——那时没有可继承的东西。

本文件两头都钉：`instantiate` 那一层的规则本身，以及 `assemble` 那条真实链路（settings 里
`subagent_template=""` 的子任务派发出来拿到的是父的模板）。
"""

from __future__ import annotations

import dataclasses

import pytest

from ctx_weft import RuntimeConfig
from ctx_weft.core.control.types import AgentView
from ctx_weft.core.models.errors import TemplateNotFoundError
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import NormalTaskSettings, Task
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.registry import ProviderRegistry
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
#: 父用的那份。子若自己没模板，就该拿到这个。
PARENT_REF = "agent:tpl_parent"
#: 显式指定时用的那份——与父不同，才看得出「有没有被继承覆盖」。
OTHER_REF = "agent:tpl_other"


class _Client:
    account, model = "acct", "mdl"
    context_limit, output_reserve = 200_000, 8192


class _Bus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, ev) -> None:
        self.events.append(ev)


def _provider() -> InlineAgentTemplateProvider:
    p = InlineAgentTemplateProvider()
    p.register(dataclasses.replace(make_echo_template(), id="tpl_parent"))
    p.register(dataclasses.replace(make_echo_template(), id="tpl_other"))
    return p


def _reg(*, fallback_ref: str = "") -> AgentLifecycleManager:
    providers = ProviderRegistry()
    providers.register_capability(_provider())
    reg = AgentLifecycleManager(
        template_lookup=TemplateLookup(providers=providers, fallback_ref=fallback_ref),
        event_bus=_Bus(),
        model_resolver=lambda a, m, **_context: _Client(),
    )
    reg.register_session(SID, tenant_id="default")
    return reg


async def _parent(reg: AgentLifecycleManager, *, template_id: str = PARENT_REF):
    agent, _ = await reg.instantiate(
        template_id=template_id, session_id=SID, tenant_id="default")
    return agent


# ── instantiate 那一层的规则 ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_empty_template_id_inherits_the_parents():
    """规则本体：没说要造什么型号 → 造一个和派生它的那个 agent 同型号的。"""
    reg = _reg()
    parent = await _parent(reg)

    child, template = await reg.instantiate(
        template_id="", session_id=SID, tenant_id="default",
        parent_agent_id=parent.id,
    )

    assert child.template_id == PARENT_REF          # Agent 对象
    assert reg.record_of(child.id).template_id == PARENT_REF   # 以及 record（恢复期的真相源）
    assert template.id == PARENT_REF                # 解出来的也是父那份


@pytest.mark.asyncio
async def test_an_explicit_template_id_is_not_overwritten_by_inheritance():
    """反向那一半：显式说了要造什么，继承不许插手。"""
    reg = _reg()
    parent = await _parent(reg)

    child, template = await reg.instantiate(
        template_id=OTHER_REF, session_id=SID, tenant_id="default",
        parent_agent_id=parent.id,
    )

    assert child.template_id == OTHER_REF
    assert template.id == OTHER_REF


@pytest.mark.asyncio
async def test_inheritance_needs_a_parent_so_a_root_still_falls_back():
    """无父（root）时空值没有可继承的东西 → 原样落 host 声明的母版。"""
    reg = _reg(fallback_ref=OTHER_REF)

    agent, template = await reg.instantiate(
        template_id="", session_id=SID, tenant_id="default")

    assert template.id == OTHER_REF                 # 母版，不是「某个父」
    assert agent.parent_agent_id is None


@pytest.mark.asyncio
async def test_without_a_fallback_a_parentless_empty_template_id_still_raises():
    """同上但没配母版 → 照旧抛。继承不是一条新的兜底，它只搬了「有父」那一格。"""
    reg = _reg()

    with pytest.raises(TemplateNotFoundError):
        await reg.instantiate(template_id="", session_id=SID, tenant_id="default")


@pytest.mark.asyncio
async def test_a_parent_that_does_not_know_its_own_template_hands_down_nothing():
    """父自己的 `template_id` 为空（恢复期缺口补出来的占位 record）→ 继承到空，落母版。

    刻意不特判：一个不知道自己是谁的父，派生出来的子不该假装知道。
    """
    reg = _reg(fallback_ref=OTHER_REF)
    await reg.load(
        {ROOT: AgentView(id=ROOT, spawn_depth=0, parent_agent_id=None, template_id="")},
        session_id=SID, tenant_id="default",
    )

    child, template = await reg.instantiate(
        template_id="", session_id=SID, tenant_id="default", parent_agent_id=ROOT)

    assert template.id == OTHER_REF                 # 兜底在解析那一层生效，不是父那份
    # record 存的是**母版自己的 id，规范形态**：空 `template_id` 让
    # `routable_template_id = template_id or template.id` 取了 `template.id`，而它自
    # 2026-10-07 起就是 `provider:name`（provider 在 `get_template` 出口贴的归属前缀）。
    # 从前这里存的是裸 `tpl_other`——`_normalise_template_id` 管的是 `__`→`:`、对裸 id
    # 归一不出前缀，于是空值路径会把一个在多 provider 部署下不唯一的 id 写进身份。
    assert child.template_id == OTHER_REF
    assert ":" in child.template_id                 # 没有任何一处还该看见裸 id


@pytest.mark.asyncio
async def test_inheritance_does_not_disturb_the_llm_rule_next_door():
    """同一处的两条继承规则互不干扰：模板继承了，`llm` 也还是照父继承。"""
    from ctx_weft.core.orchestrator.model import ModelChoice

    reg = _reg()
    parent, _ = await reg.instantiate(
        template_id=PARENT_REF, session_id=SID, tenant_id="default",
        llm=ModelChoice(account="acct_parent", model="mdl_parent"),
    )

    child, _ = await reg.instantiate(
        template_id="", session_id=SID, tenant_id="default", parent_agent_id=parent.id)

    assert reg.record_of(child.id).template_id == PARENT_REF
    assert reg._agents[child.id].llm == ModelChoice(
        account="acct_parent", model="mdl_parent")


# ── assemble 那条真实链路（delegate_task 不填模板派出来的子任务）─────────────────


class _StubTaskManager:
    """assemble 只碰 `get_task` 与 `all_tasks`；execute 那半不在本文件范围内。

    `all_tasks` 是 `inherit_memory` 那条路要的（`_latest_prior_root_task` 找继承源）。
    刻意保留 `inherit_memory=True`——那是 `delegate_task` 的默认值，task 形状要跟真的
    一样。本会话只有这一条 task 且 `created_at` 为 None，所以筛不出继承源、拷贝跳过，
    正好不把记忆那一摊拖进本文件。
    """

    def __init__(self, task: Task) -> None:
        self._task = task

    def get_task(self, task_id):
        return self._task if task_id == self._task.id else None

    def all_tasks(self):
        return [self._task]


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


@pytest.mark.asyncio
async def test_a_delegated_task_without_a_template_runs_on_the_parents_template():
    """`delegate_task(use_subagent=True)` 留空 `subagent_template` 的那条链路。

    从前这里拿到的是 `fallback_template_ref`（宿主母版）。
    """
    rt = make_runtime(
        llm=MockLLMAdapter(responses=[]), agent_provider=_provider(),
        # 母版配成**另一份**：继承生效时它必须一次都不被用到。
        config=RuntimeConfig(fallback_template_ref=OTHER_REF),
    )
    await rt._agent_lifecycle_manager.load(
        {ROOT: AgentView(id=ROOT, spawn_depth=0, parent_agent_id=None,
                         template_id=PARENT_REF)},
        session_id=SID, tenant_id="default",
    )
    task = Task(
        id="tsk_1", session_id=SID, status="ACTIVE", tenant_id="default",
        creator_agent_id=ROOT, title="Sub-task",
        # delegate_task 不填模板时 staged 出来的正是这个形状。
        settings=NormalTaskSettings(use_subagent=True, subagent_template=""),
    )
    runner = await _runner(rt, task)

    binding = await runner.assemble(task.id)

    assert binding is not None
    assert binding.template.id == PARENT_REF                # 父那份，不是母版
    assert binding.agent.template_id == PARENT_REF          # record 一并钉住
    assert binding.agent.parent_agent_id == ROOT
