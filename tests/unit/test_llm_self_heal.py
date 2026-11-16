
from tests.model_testkit import model_snapshot
"""Unit tests for stream_llm_resilient backoff self-heal."""
import asyncio
from types import SimpleNamespace

import pytest

from ctx_weft.protocols import LLMCallError, LLMChunk, LLMOutageError, LLMRequest
from ctx_weft.core.loop import llm_gateway
from ctx_weft.core.loop.llm_gateway import stream_llm_resilient, _sleep_cancellable as _real_sleep
from ctx_weft.core.control.tokens import CancelToken
from ctx_weft.protocols.events import EventType



class _FakeLLM:
    """Scripted LLMClient. Each `complete()` consumes one action."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    @property
    def context_limit(self): return 1000
    @property
    def output_reserve(self): return 100
    @property
    def supports_tool_calling(self): return True

    def complete(self, request, stream=True):
        action = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return self._run(action)

    async def _run(self, action):
        kind = action["kind"]
        if kind == "ok":
            for t in action["tokens"]:
                yield LLMChunk(kind="token", text=t)
            yield LLMChunk(kind="done", finish_reason="stop")
        elif kind == "fail_pre":
            raise LLMCallError(
                action.get("msg", "pre"),
                status_code=action.get("code", 503),
                retriable=action.get("retriable", True),
                outage=action.get("outage", True),
            )
        elif kind == "fail_mid":
            yield LLMChunk(kind="token", text="partial")
            raise LLMCallError("mid-flight", status_code=0, retriable=True, outage=True)


class _Bus:
    def __init__(self): self.events = []
    async def emit(self, ev): self.events.append(ev)


def _ctx(llm, *, cancel_token=None, bus=None, config=None):
    return SimpleNamespace(llm=llm, cancel_token=cancel_token, event_bus=bus, config=config)


def _cfg(**kw):
    base = dict(
        llm_self_heal_max_attempts=8,
        llm_self_heal_max_duration_sec=300.0,
        llm_self_heal_base_delay_sec=2.0,
        llm_self_heal_max_interval_sec=60.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


async def _drain(agen):
    out = []
    async for c in agen:
        out.append(c)
    return out


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def _fake_sleep(delay, tok):
        if tok is not None:
            tok.raise_if_cancelled()
    monkeypatch.setattr(llm_gateway, "_sleep_cancellable", _fake_sleep)


@pytest.fixture(autouse=True)
def _stub_make_event(monkeypatch):
    def _fake(state, type, payload=None, **kw):
        return {"type": type, "payload": payload or {}}
    monkeypatch.setattr(llm_gateway, "make_event", _fake)


def _req():
    return LLMRequest(model="m", system="s", messages=[])


def _state():
    """最小 state：足以让 stream_llm_resilient 的流式侧事件真正跑（Task 5 删掉了
    origin 门禁后，gateway 对任何非 None 的 state 都无条件发射，需要 .agent /
    .resolved_model / .sequence_counter 才不会在 resolve_llm_identity/req_id 处炸。"""
    return SimpleNamespace(
        agent=SimpleNamespace(id="a1", loop_guard=None),
        resolved_model=model_snapshot(model="m", account="acct"),
        run_id="run-test", sequence_counter=0,
        origin="",
    )


@pytest.mark.asyncio
async def test_self_heals_then_succeeds_no_duplicate():
    llm = _FakeLLM([
        {"kind": "fail_pre"},
        {"kind": "fail_pre"},
        {"kind": "ok", "tokens": ["Hel", "lo"]},
    ])
    bus = _Bus()
    chunks = await _drain(stream_llm_resilient(_ctx(llm, bus=bus, config=_cfg()), _state(), _req()))
    text = "".join(c.text for c in chunks if c.kind == "token")
    assert text == "Hello"          # only the successful attempt's tokens, no dup
    assert llm.calls == 3
    retries = [e for e in bus.events if e["type"] == EventType.LLM_RETRY_TRIGGERED]
    assert len(retries) == 2
    assert retries[0]["payload"]["attempt"] == 1


@pytest.mark.asyncio
async def test_exhaust_max_attempts_raises_outage():
    llm = _FakeLLM([{"kind": "fail_pre"}])  # always fails
    with pytest.raises(LLMOutageError):
        await _drain(stream_llm_resilient(
            _ctx(llm, bus=_Bus(), config=_cfg(llm_self_heal_max_attempts=3)), None, _req()))
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_exhaust_max_duration_raises_outage(monkeypatch):
    clock = iter([0.0, 0.0, 0.0, 1000.0])
    monkeypatch.setattr(llm_gateway, "monotonic", lambda: next(clock))
    llm = _FakeLLM([{"kind": "fail_pre"}])
    with pytest.raises(LLMOutageError):
        await _drain(stream_llm_resilient(
            _ctx(llm, bus=_Bus(), config=_cfg(llm_self_heal_max_attempts=100)), None, _req()))
    assert llm.calls == 1   # deadline already passed after first failure


@pytest.mark.asyncio
async def test_midflight_outage_retries_with_discard_boundary():
    llm = _FakeLLM([{"kind": "fail_mid"}, {"kind": "ok", "tokens": ["x"]}])
    got = []
    async with (stream := stream_llm_resilient(_ctx(llm, bus=_Bus(), config=_cfg()), None, _req())):
        async for c in stream:
            got.append(c)
            stream.accept_current()
    assert [c.text for c in got if c.kind == "token"] == ["partial", "x"]
    assert stream.get_final_result().text == "x"
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_non_outage_retriable_passthrough():
    # _finalize-style truncation: retriable but NOT outage → re-raised as-is, not LLMOutageError
    llm = _FakeLLM([{"kind": "fail_pre", "outage": False, "retriable": True}])
    with pytest.raises(LLMCallError) as ei:
        await _drain(stream_llm_resilient(_ctx(llm, bus=_Bus(), config=_cfg()), None, _req()))
    assert not isinstance(ei.value, LLMOutageError)
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_permanent_error_passthrough():
    llm = _FakeLLM([{"kind": "fail_pre", "retriable": False, "outage": False, "code": 401}])
    with pytest.raises(LLMCallError) as ei:
        await _drain(stream_llm_resilient(_ctx(llm, bus=_Bus(), config=_cfg()), None, _req()))
    assert not isinstance(ei.value, LLMOutageError)
    assert ei.value.status_code == 401
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_cancel_pierces_backoff():
    tok = CancelToken()
    tok.cancel()
    llm = _FakeLLM([{"kind": "fail_pre"}, {"kind": "ok", "tokens": ["x"]}])
    with pytest.raises(asyncio.CancelledError):
        await _drain(stream_llm_resilient(
            _ctx(llm, cancel_token=tok, bus=_Bus(), config=_cfg()), None, _req()))
    assert llm.calls == 0   # cancelled before issuing a request


# ── Direct _sleep_cancellable tests (use the real function, bypassing the autouse stub) ──


@pytest.mark.asyncio
async def test_sleep_cancellable_none_token_completes():
    """No cancel token — sleep completes normally."""
    await _real_sleep(0.01, None)


@pytest.mark.asyncio
async def test_sleep_cancellable_precancelled_raises_fast():
    """Pre-cancelled token — should raise CancelledError well before the 5s delay."""
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(_real_sleep(5.0, tok), timeout=1.0)


@pytest.mark.asyncio
async def test_sleep_cancellable_cancel_midsleep_wakes():
    """Cancel fires mid-sleep — task wakes promptly and raises CancelledError."""
    tok = CancelToken()
    task = asyncio.create_task(_real_sleep(5.0, tok))
    await asyncio.sleep(0.02)
    tok.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_sleep_cancellable_not_cancelled_completes():
    """Fresh non-cancelled token — short sleep completes without raising."""
    tok = CancelToken()
    await _real_sleep(0.01, tok)
