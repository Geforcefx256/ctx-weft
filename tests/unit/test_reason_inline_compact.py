"""PrepareStep compacts inline (calls escalating_compact directly) and still routes to act."""

from __future__ import annotations

from types import SimpleNamespace

from ctx_weft.core.loop.steps.prepare import PrepareStep
from ctx_weft.core.utils.estimate import estimate_tokens


class _SpyEscalatingCompact:
    def __init__(self):
        self.called = False
        self.kwargs = None

    async def __call__(self, state, ctx, **kwargs):
        self.order = list(getattr(ctx.task_manager, "committed", []))
        self.called = True
        self.kwargs = kwargs
        return []


async def test_reason_runs_compact_inline_and_routes_to_act(monkeypatch):
    spy = _SpyEscalatingCompact()
    monkeypatch.setattr("ctx_weft.core.loop.steps.compact.escalating_compact", spy)

    pushed = []

    rs = PrepareStep()
    async def _est(state, ctx):
        return 100000, True
    async def _resolve(state, ctx):
        return []
    async def _skill(state, ctx, name):
        return ""
    async def _should(state, ctx, est):
        return True

    monkeypatch.setattr(rs, "_estimate_tokens", _est)
    monkeypatch.setattr("ctx_weft.core.loop.steps.prepare.resolve_and_bind", _resolve)
    monkeypatch.setattr(rs, "_load_skill_instructions", _skill)
    monkeypatch.setattr(rs, "_should_compact", _should)

    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(compact_keep_last=2, compact_token_ratio=0.8,
                                    compact_message_delta=0),
        loop_guard=SimpleNamespace(context_limit=1000, context_tokens=0,
                                   context_message_count=0),
    )
    session = SimpleNamespace(id="s1", tenant_id="te1", token_budget=0, token_used=0,
                              context_limit=1000)
    task = SimpleNamespace(id="t1", parent_task_id="p", title="T",
                           settings=SimpleNamespace(skill_name="", purpose="act"))
    state = SimpleNamespace(agent=agent, session=session, task=task,
                            scope=SimpleNamespace(), extra={"template": None},
                            run_id="r1", sequence_counter=0)

    class _Assembler:
        def __init__(self):
            self.calls = 0
        async def assemble(self, request):
            self.calls += 1
            return SimpleNamespace(token_count=10, system="", messages=[], tools=[])

    class _TM:
        def __init__(self):
            self.committed = []

        async def push_task(self, *a, **k):
            pushed.append(a)

        async def commit_round(self, task_id):
            self.committed.append(task_id)

    class _Bus:
        async def emit(self, ev):
            pass

    asm = _Assembler()
    ctx = SimpleNamespace(assembler=asm, task_manager=_TM(), event_bus=_Bus(),
                          memory=SimpleNamespace(), provider_ctx=SimpleNamespace(),
                          llm=SimpleNamespace(tokenizer=SimpleNamespace(count=estimate_tokens)))

    outcome = await rs.execute(state, ctx)
    assert spy.called is True          # escalating_compact ran inline
    assert spy.kwargs["trigger"] == "compact"
    assert pushed == []                # no compact Task pushed
    assert outcome.next_step == "act"  # still proceeds to act
    assert asm.calls == 2              # assembled once, then re-assembled after compaction
    # 压缩只折 memory，够不着未提交窗口里暂存的记录 → 压缩之前必须先提交这一轮。
    assert spy.order == ["t1"]


async def test_reason_stashes_bound_capabilities(monkeypatch):
    rs = PrepareStep()
    sentinel = [object()]

    async def _est(state, ctx):
        return 10, True

    async def _resolve(state, ctx):
        return sentinel

    async def _skill(state, ctx, name):
        return ""

    async def _should(state, ctx, est):
        return False  # no compaction this run

    monkeypatch.setattr(rs, "_estimate_tokens", _est)
    monkeypatch.setattr("ctx_weft.core.loop.steps.prepare.resolve_and_bind", _resolve)
    monkeypatch.setattr(rs, "_load_skill_instructions", _skill)
    monkeypatch.setattr(rs, "_should_compact", _should)

    agent = SimpleNamespace(
        id="a1",
        loop_config=SimpleNamespace(compact_keep_last=2, compact_token_ratio=0.8,
                                    compact_message_delta=0),
        loop_guard=SimpleNamespace(context_limit=1000, context_tokens=0,
                                   context_message_count=0),
    )
    session = SimpleNamespace(id="s1", tenant_id="te1", token_budget=0, token_used=0,
                              context_limit=1000)
    task = SimpleNamespace(id="t1", parent_task_id="p", title="T",
                           settings=SimpleNamespace(skill_name="", purpose="act"))
    state = SimpleNamespace(agent=agent, session=session, task=task,
                            scope=SimpleNamespace(), extra={"template": None},
                            run_id="r1", sequence_counter=0)

    class _Assembler:
        async def assemble(self, request):
            return SimpleNamespace(token_count=10, system="", messages=[], tools=[])

    class _Bus:
        async def emit(self, ev):
            pass

    ctx = SimpleNamespace(assembler=_Assembler(), task_manager=None, event_bus=_Bus(),
                          memory=SimpleNamespace(), provider_ctx=SimpleNamespace(),
                          llm=SimpleNamespace(tokenizer=SimpleNamespace(count=estimate_tokens)))

    await rs.execute(state, ctx)
    assert state.extra["bound_capabilities"] is sentinel
