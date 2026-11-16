
from tests._recovery import recover_fixture_session
"""Public instance shutdown preserves restart semantics and task ownership."""
import asyncio

import pytest

from ctx_weft import NormalTaskSettings, RuntimeClosedError, RuntimeShutdownError, SessionStartParams
from ctx_weft.protocols import LLMChunk
from ctx_weft.protocols.events import EventType
from ctx_weft.providers.events import InMemoryEventStore
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_echo_template, make_runtime
from tests.integration.test_send_message_after_session_done import _FinishEveryTaskLLM


class BlockingLLM(_FinishEveryTaskLLM):
    def __init__(self, *, committed=True, resist=False):
        super().__init__()
        self.started, self.stopped, self.release = (asyncio.Event() for _ in range(3))
        self.committed, self.resist = committed, resist

    def complete(self, request, stream=True):
        if any(t.name in {"control__update_task_metadata", "control__report_observation"}
               for t in request.tools):
            return super().complete(request, stream=stream)

        async def tokens():
            try:
                if self.committed:
                    yield LLMChunk(kind="token", text="working")
                self.started.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    if self.resist:
                        await self.release.wait()
                    raise
            finally:
                self.stopped.set()
        return tokens()


def runtime(llm, store=None):
    templates = InlineAgentTemplateProvider()
    templates.register(make_echo_template())
    rt = make_runtime(llm=llm, agent_provider=templates, event_store=store)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def start(rt, llm, sid="close-session"):
    handle = await rt.start_session(SessionStartParams(
        initial_task_settings=NormalTaskSettings(), session_id=sid, template_id="agent:tpl_echo", user_prompt="keep original input"))
    await asyncio.wait_for(llm.started.wait(), 5)
    return handle


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", [True, False])
async def test_close_interrupts_only_committed_work_and_allows_fresh_recovery(committed):
    llm = BlockingLLM(committed=committed)
    rt = runtime(llm)
    handle = await start(rt, llm)
    report, other = await asyncio.gather(rt.close(), rt.close())
    assert report == other and report.quiescent and report.durable
    assert llm.stopped.is_set()
    head = await rt.event_store.committed_head("close-session")
    events = [r.event for r in await rt.event_store.read_range("close-session")]
    assert not any(e.type in {EventType.TASK_CANCELED, EventType.RUN_CANCELED} for e in events)
    if committed:
        assert (await rt.get_task("close-session", handle.task_id)).data.status == "INTERRUPTED"
    await asyncio.sleep(.02)
    assert await rt.event_store.committed_head("close-session") == head
    with pytest.raises(RuntimeClosedError):
        await rt.send_message("unused", "new input", session_id="close-session")
    fresh = runtime(_FinishEveryTaskLLM(), rt.event_store)
    fresh.providers.register_memory(rt.providers.get_memory())
    try:
        await fresh.initialize("close-session")
        assert (await fresh.get_session("close-session")).data.session_id == "close-session"
        if committed:
            await fresh.rebuild_session("close-session", strict=True)
            agent = (await fresh.get_agent(handle.agent_id, session_id="close-session")).data
            await recover_fixture_session(fresh, agent.agent_id)
            async with asyncio.timeout(5):
                while (await fresh.get_task("close-session", handle.task_id)).data.status != "FINISHED":
                    await asyncio.sleep(.01)
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_timeout_and_cancelled_waiter_do_not_abandon_shutdown():
    llm = BlockingLLM(resist=True)
    rt = runtime(llm)
    await start(rt, llm)
    with pytest.raises(RuntimeShutdownError) as error:
        await rt.close(timeout=.01)
    assert not error.value.report.quiescent
    waiter = asyncio.create_task(rt.close())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    llm.release.set()
    assert (await rt.close()).durable


@pytest.mark.asyncio
async def test_close_is_instance_scoped_and_preserves_pending_hitl():
    llm1, llm2 = BlockingLLM(), BlockingLLM()
    first, second = runtime(llm1), runtime(llm2)
    h1, h2 = await start(first, llm1, "first"), await start(second, llm2, "second")
    try:
        await first.pause_session("first")
        async with asyncio.timeout(5):
            while not (pending := [r.data for r in await first.list_pending_hitl(session_id="first")]):
                await asyncio.sleep(.01)
        await first.close()
        assert [r.data for r in await first.list_pending_hitl(session_id="first")] == pending
        assert (await first.get_task("first", h1.task_id)).data.status == "AWAITING_HUMAN"
        assert not llm2.stopped.is_set()
        assert (await second.get_task("second", h2.task_id)).data.status == "ACTIVE"
    finally:
        await first.close()
        await second.close()


class FailingStore:
    def __init__(self):
        self.inner = InMemoryEventStore()
        self.fail = False

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def read_range(self, *args, **kwargs):
        return await self.inner.read_range(*args, **kwargs)

    async def committed_head(self, *args, **kwargs):
        return await self.inner.committed_head(*args, **kwargs)

    async def append_batch(self, *args, **kwargs):
        if self.fail:
            raise OSError("disk unavailable")
        return await self.inner.append_batch(*args, **kwargs)


@pytest.mark.asyncio
async def test_storage_failure_reports_quiescence_without_claiming_durability():
    llm, store = BlockingLLM(), FailingStore()
    rt = runtime(llm, store)
    await start(rt, llm)
    store.fail = True
    with pytest.raises(RuntimeShutdownError) as error:
        await rt.close()
    assert error.value.report.quiescent and not error.value.report.durable
    assert llm.stopped.is_set()


@pytest.mark.asyncio
async def test_close_during_observation_retains_recoverable_task():
    from tests.integration._user_control_journey import runtime as control_runtime, SID

    memory, store = InMemoryMemoryProvider(), InMemoryEventStore()
    rt, llm, _, _ = control_runtime(store, memory, "resume", "initial", "observe")
    handle = await rt.start_session(SessionStartParams.create(
        session_id=SID, template_id="agent:tpl_echo", user_prompt="prepare release"))
    await asyncio.wait_for(llm.started.wait(), 5)
    await rt.close()
    assert llm.closed.is_set()
    assert (await rt.get_task(SID, handle.task_id)).data.status in {"INTERRUPTED", "AWAITING_HUMAN"}
    assert not rt._step_executions
    fresh, _, _, _ = control_runtime(store, memory, "resume", "restored", "observe")
    try:
        await fresh.rebuild_session(SID, strict=True)
        await recover_fixture_session(fresh, handle.agent_id)
        async with asyncio.timeout(5):
            while (await fresh.get_task(SID, handle.task_id)).data.status != "FINISHED":
                await asyncio.sleep(.01)
    finally:
        await fresh.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [True, False])
async def test_external_cancellation_joins_inflight_tool_before_returning(tokens):
    from types import SimpleNamespace
    from ctx_weft.core.control.tokens import CancelToken, PauseToken
    from ctx_weft.core.loop.steps.act import _await_tool_or_stop

    entered, cancelled, release, stopped = (asyncio.Event() for _ in range(4))
    async def tool():
        try:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()  # Transaction rollback or subprocess cleanup.
                raise
        finally:
            stopped.set()
    child = asyncio.create_task(tool())
    parent = asyncio.create_task(_await_tool_or_stop(child, SimpleNamespace(
        cancel_token=CancelToken() if tokens else None,
        pause_token=PauseToken() if tokens else None)))
    await entered.wait()
    parent.cancel()
    await cancelled.wait()
    assert not parent.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert child.done() and stopped.is_set()


@pytest.mark.asyncio
async def test_close_stops_autonomous_backoff_without_requeue_or_cancel_outcome():
    from tests.unit.test_autonomous_interrupt_recovery import _tm, _autonomous, _interrupt

    rt = runtime(_FinishEveryTaskLLM())
    manager = _tm(base=.05)
    task = _autonomous()
    await _interrupt(manager, task)
    timer = manager._autonomous_requeue_timers[task.id]
    rt._bind_task_manager("s1", manager)
    await rt.close()
    await asyncio.sleep(.08)
    assert timer.done()
    assert task.status == "INTERRUPTED" and task.interrupt_requeue_count == 0
    assert not manager._runner.dispatched
