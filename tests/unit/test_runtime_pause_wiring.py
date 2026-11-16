"""Runtime wires a PauseToken into LoopContext and tracks task managers."""

from ctx_weft.protocols.context import ProviderContext as _ProviderContext
import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control.tokens import CancelToken, PauseToken
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.event import emit_event
from ctx_weft.protocols import ProviderContext
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime
from tests.unit.test_runtime_agent_api import _plant


_PROVIDER_CONTEXT = _ProviderContext(session_id="test")


def _rt():
    return make_runtime(llm=MockLLMAdapter(responses=[]),
                           agent_provider=InlineAgentTemplateProvider())


def test_runtime_has_pause_and_task_manager_maps():
    rt = _rt()
    assert rt._run_tokens == {}
    assert rt._task_managers == {}


def test_build_loop_ctx_wires_pause_token():
    rt = _rt()
    pause = PauseToken()
    ctx = rt._build_loop_ctx(
        assembler=None, llm=rt._resolve_llm(None, None, ctx=_PROVIDER_CONTEXT), memory=None,
        provider_ctx=ProviderContext(session_id="session"), gateway=None,
        cancel_token=CancelToken(), task_manager=None, pause_token=pause,
    )
    assert ctx.pause_token is pause


def test_step_execution_registry_is_shared_only_within_a_session():
    rt = _rt()

    def build(session_id):
        return rt._build_loop_ctx(
            assembler=None, llm=rt._resolve_llm(None, None, ctx=_PROVIDER_CONTEXT), memory=None,
            provider_ctx=ProviderContext(session_id=session_id),
            gateway=None, cancel_token=None, task_manager=None)

    first, second, other = build("s1"), build("s1"), build("s2")
    assert first.steps is not second.steps
    assert first.steps.executions is second.steps.executions
    assert first.steps.executions is not other.steps.executions
    assert first.steps.executions is rt._step_executions["s1"]
    assert {"prepare", "act", "observe", "recognize_intent"} <= first.steps.steps.keys()


# ── 2026-09-04 spec §7.1 / §8：pause_task 内部化，pause_session 语义不变 ──


def test_pause_task_is_no_longer_public():
    assert not hasattr(CtxWeftRuntime, "pause_task")
    assert hasattr(CtxWeftRuntime, "_pause_task")


class _StubRunner:
    async def assemble(self, task_id):
        return None

    async def execute(self, binding, task_id):
        return None


def _task(tid: str, agent: str) -> Task:
    return Task(
        id=tid, session_id="s1", status="ACTIVE", tenant_id="default",
        assigned_agent_id=agent, creator_agent_id="root",
        title=tid, description="", user_prompt="x", created_at=now_utc(),
    )


@pytest.mark.asyncio
async def test_pause_session_leaves_non_root_agents_alive():
    """回归护栏：非 root agent 被取消的是 run，不是 agent 本身。

    这条在改动前就该是绿的——它锁死的正是「不要顺手改成 cancel_agent」。
    """
    rt = _rt()
    root_id, child_id = "root", "child"
    _plant(rt, root_id, None, session_id="s1", status="running")
    _plant(rt, child_id, root_id, session_id="s1", status="running")
    reg = rt._agent_lifecycle_manager
    reg._agents[root_id].current_task_id = "t_root"
    reg._agents[child_id].current_task_id = "t_child"

    tm = TaskManager(session_id="s1", event_bus=rt._event_bus)
    sess = Session(id="s1", tenant_id="default", user_prompt="x",
                     root_agent_id=root_id)
    tm.set_session(sess)
    tm.set_runner(_StubRunner())
    for tid, agent in (("t_root", root_id), ("t_child", child_id)):
        tm.register_task(_task(tid, agent))
        tm._running_tasks.add(tid)
        tm._running_agents[tid] = agent
    rt._task_managers["s1"] = tm
    root_tokens = rt._register_run_tokens("s1", "t_root")
    child_tokens = rt._register_run_tokens("s1", "t_child")

    await initialize_all(rt)
    assert await rt.pause_session("s1") is True

    # root：pause（唯一续跑点）；非 root：cancel 掉它的在途 run，agent 本身原封不动。
    assert root_tokens.pause.is_paused is True
    assert child_tokens.cancel.is_cancelled is True
    assert reg.status_of(child_id) == "running"          # 没被直接拍成 terminated
    await initialize_all(rt)
    assert (await rt.get_agent(child_id)).data.state.status != "terminated"

    # 真实世界里这条被 cancel 的 run 会在自己的收尾点发一条 TASK_CANCELED，经 ALM
    # （_INPUT_BY_EVENT[TASK_CANCELED] = SETTLED）落回 idle——这里补一条同形的事件，
    # 落地那条早已成立的下游事实（R23 / test_cancel_end_to_end.py），核实的是本测试
    # 关心的那一半：agent 落地之后仍然「活着」、可被 send_message 寻址，不是 terminated。
    await emit_event(
        rt._event_bus, EventType.TASK_CANCELED,
        session_id="s1", tenant_id="default", origin="test",
        agent_id=child_id, task_id="t_child",
    )
    assert reg.status_of(child_id) == "idle"
    reg.assert_can_receive(child_id)   # 不 raise：terminated 的 agent 会被这里拒绝
