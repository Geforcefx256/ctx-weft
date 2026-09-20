"""gateway 的两个落库出口都要给这次调用牵到的 HITL 决定盖 `HitlClosed`。

一次工具调用最多牵三条请求——`authz`（事前审核）、`tool`（provider 自问，如 `ask_user`）、
`rerun`（准重跑）。调用一结束它们**同时**失效，所以盖章不分 stage。

这两个出口的存在本身就是「统一由消费方盖章」胜过「从 capability 事件推退役」的理由：

1. `_record_result`：`CapabilityFinished` 发在 memory ingest **之前**，按它退役就有一条缝
   （日志说结果有了、对话里却没有）；而 `ask_user` 那类 reconcile 刻意不信事件通道，正需要
   那条冷决定还在。盖章在 ingest 之后，两者都不成问题。
2. `_error_and_record`（未授权出口）：这条路**一条 capability 事件都不发**，按事件推就是个
   真空——「人拒绝了」的决定永远等不到销账。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.hitl.registry import (
    HITL_STAGE_AUTHZ,
    HITL_STAGE_RERUN,
    HITL_STAGE_TOOL,
    HitlRegistry,
    PendingHitl,
)
from ctx_weft.protocols.events import EventType
from ctx_weft.protocols.hitl import HitlDecision, ToolResultDelivery

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID, _TCID = "s1", "call_7"


class _Bus:
    def __init__(self) -> None:
        self.events: list = []

    async def emit(self, ev) -> None:
        self.events.append(ev)


def _service(*stages, resolved=True):
    """一个 registry，里面按给定 stage 各挂一条**同 tool_call** 的请求。"""
    from ctx_weft.core.hitl.service import HitlService

    reg = HitlRegistry()
    for i, stage in enumerate(stages):
        req = PendingHitl(
            id=f"hit_{stage}", form="approval", session_id=_SID, task_id="t1",
            agent_id="ag1", delivery=ToolResultDelivery(tool_call_id=_TCID),
            created_at=_T0, tenant_id="acme", tool_call_id=_TCID, stage=stage,
        )
        if resolved:
            req.decision = HitlDecision(outcome="accepted", message="ok")
            req.resolved_at = _T0
        reg._requests[req.id] = req
    bus = _Bus()
    svc = HitlService(registry=reg, event_bus=bus, reply_intake=None)
    return svc, bus


def _closed_ids(bus) -> set[str]:
    return {e.payload["hitl_id"] for e in bus.events
            if e.type == EventType.HITL_CLOSED}


async def test_close_for_tool_call_covers_every_stage():
    """三个 stage 一起盖——逐个 stage 去盖要调用方记住有哪几个，漏掉的那条就永不销账。"""
    svc, bus = _service(HITL_STAGE_AUTHZ, HITL_STAGE_TOOL, HITL_STAGE_RERUN)

    n = await svc.close_for_tool_call(_SID, _TCID)

    assert n == 3
    assert _closed_ids(bus) == {
        f"hit_{HITL_STAGE_AUTHZ}", f"hit_{HITL_STAGE_TOOL}", f"hit_{HITL_STAGE_RERUN}"}


async def test_close_skips_requests_that_are_still_pending(caplog):
    """未决的不盖：人还没答，盖章等于宣布一件没发生的事。"""
    svc, bus = _service(HITL_STAGE_AUTHZ, resolved=False)

    n = await svc.close_for_tool_call(_SID, _TCID)

    assert (n, _closed_ids(bus)) == (0, set())


async def test_close_does_not_leak_across_sessions():
    """按 (session, tool_call) 取数——LLM 的 tool_call id 常是 `call_1` 这类短值，
    只按 id 会把另一个会话的决定一并了结掉。"""
    svc, bus = _service(HITL_STAGE_AUTHZ)

    assert await svc.close_for_tool_call("s_other", _TCID) == 0
    assert _closed_ids(bus) == set()


async def test_empty_tool_call_id_is_a_no_op():
    """UserTurn park 没有 tool_call_id（`_cold_park`）——空 id 不得匹配到任何东西。"""
    svc, bus = _service(HITL_STAGE_AUTHZ)

    assert await svc.close_for_tool_call(_SID, "") == 0
    assert _closed_ids(bus) == set()


# ── gateway 两个出口都真的调了它 ───────────────────────────────────────────────


def _gateway_source(name: str) -> str:
    import inspect

    from ctx_weft.core.loop.capability_gateway import CapabilityGateway

    return inspect.getsource(getattr(CapabilityGateway, name))


def test_record_result_closes_after_the_ingest():
    """`_record_result` 里盖章必须在 `ingest_tool_result` **之后**。

    源码顺序检查而非行为检查：这条出口要搭起真 provider + 真 memory 才跑得到，而要钉的只是
    「哪一句在哪一句之后」。顺序反了就留下「日志说了结、对话里没结果」，那条决定再也不会被补。
    """
    src = _gateway_source("_record_result")
    assert "close_for_tool_call" in src, "结果落库后没盖章——那些决定会永远留在清单里"
    assert src.index("ingest_tool_result") < src.index("close_for_tool_call"), \
        "盖章必须在 ingest 之后（见 EventType.HITL_CLOSED 的顺序纪律）"


def test_error_exit_closes_too():
    """未授权出口同样要盖——它一条 capability 事件都不发，按事件推退役在这里是真空。"""
    src = _gateway_source("_error_and_record")
    assert "close_for_tool_call" in src, "拒绝也是一次消费，那条决定同样要销账"
    assert src.index("ingest_tool_result") < src.index("close_for_tool_call")


def test_closing_is_not_gated_on_dispatch_or_silent():
    """盖章不跟着 `is_dispatch` / `is_silent` 分支走。

    那两个分支管的是「写不写对话」，与「这条决定还要不要留着」无关——派发 / SILENT 工具同样
    可能被门控过。挂在分支里面，它们的决定就永不销账。
    """
    src = _gateway_source("_record_result")
    tail = src[src.index("close_for_tool_call"):]
    head = src[:src.index("close_for_tool_call")]
    # 盖章那句所在的 if 只判 ctx.hitl，不判 is_dispatch/is_silent
    guard = head.rsplit("if ", 1)[1].split(":")[0]
    assert "is_dispatch" not in guard and "is_silent" not in guard, \
        f"盖章被 dispatch/silent 分支挡住了：if {guard}"
    assert tail  # 形态完整
