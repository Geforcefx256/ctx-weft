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


# ── 取消 / 销毁路径 ───────────────────────────────────────────────────────────


async def test_close_resolved_stamps_the_whole_session():
    """取消之后没人会再消费那些决定——任务不会再跑，补注入对终态 task 本就跳过。

    不盖章它们就永远留在折叠的清单里，而这条会话的事件日志还在（`purge_session` 明确不删
    日志）。这是这条路存在的全部理由。
    """
    svc, bus = _service(HITL_STAGE_AUTHZ, HITL_STAGE_TOOL)

    n = await svc.close_resolved(_SID)

    assert n == 2
    assert _closed_ids(bus) == {f"hit_{HITL_STAGE_AUTHZ}", f"hit_{HITL_STAGE_TOOL}"}


async def test_close_resolved_is_idempotent():
    """取消 → 销毁是前后相继的两条路，同一条请求不得发两遍章。

    幂等靠内存 `PendingHitl.closed`；`purge_session` 里那次正常路径下就是 no-op。
    """
    svc, bus = _service(HITL_STAGE_AUTHZ)

    first = await svc.close_resolved(_SID)
    second = await svc.close_resolved(_SID)

    assert (first, second) == (1, 0)
    assert len([e for e in bus.events if e.type == EventType.HITL_CLOSED]) == 1


async def test_close_resolved_can_narrow_to_one_agent():
    """agent 粒度：不得误伤同一 session 里别的 agent 仍然有用的决定。"""
    svc, bus = _service(HITL_STAGE_AUTHZ)
    other = PendingHitl(
        id="hit_other", form="approval", session_id=_SID, task_id="t2",
        agent_id="ag_other", delivery=ToolResultDelivery(tool_call_id="call_x"),
        created_at=_T0, tenant_id="acme", tool_call_id="call_x", stage=HITL_STAGE_AUTHZ)
    other.decision = HitlDecision(outcome="accepted", message="ok")
    other.resolved_at = _T0
    svc.registry._requests[other.id] = other

    n = await svc.close_resolved(_SID, agent_id="ag1")

    assert n == 1
    assert _closed_ids(bus) == {f"hit_{HITL_STAGE_AUTHZ}"}


async def test_closed_records_leave_the_re_injection_list():
    """盖过章的记录不再出现在 `resolved_for_session()`。

    这条不变式写在那个方法的 docstring 里：它返回的集合必须与「折叠装填出来的集合」一致，
    而折叠遇到 `HitlClosed` 会把那条整个摘掉。两边不一致，恢复期兜底就会对着一份幻影清单
    反复做无用功。
    """
    svc, bus = _service(HITL_STAGE_AUTHZ)
    assert len(svc.registry.resolved_for_session(_SID)) == 1

    await svc.close_resolved(_SID)

    assert svc.registry.resolved_for_session(_SID) == []


async def test_stamp_is_emitted_before_the_memory_mark():
    """**先发事件、后标内存**。

    反了的话「标了但没发出去」会让这条在内存里消失、日志里又没有章——恢复期的清单两头都
    看不见它，那是真丢，比多发一次章严重得多。
    """
    svc, bus = _service(HITL_STAGE_AUTHZ)
    req = svc.registry.get(f"hit_{HITL_STAGE_AUTHZ}")

    async def _boom(ev):
        raise RuntimeError("bus down")

    svc._bus.emit = _boom

    assert await svc.close(req) is False
    assert req.closed is False, "发不出去就不该标——否则它从两边同时消失"


def test_cancel_and_purge_both_close():
    """两条真终结路径都要盖章，而且 purge 必须在 `forget_session` **之前**盖。

    摘掉之后就没有请求对象可盖了。源码顺序检查——真跑通这两条要搭整个 runtime，而要钉的
    只是「哪句在哪句之前」。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    cancel_src = inspect.getsource(CtxWeftRuntime.cancel_session)
    assert "close_resolved" in cancel_src, "取消之后那些决定没人消费了，要盖章"

    purge_src = inspect.getsource(CtxWeftRuntime.purge_session)
    # 按**实际调用串**比，不按裸名字——docstring 里也提到了 forget_session。
    stamp = "await self.hitl.close_resolved("
    forget = "self.hitl_registry.forget_session("
    assert stamp in purge_src and forget in purge_src
    assert purge_src.index(stamp) < purge_src.index(forget), (
        "摘掉记录之后就盖不了章了")


def test_close_resolved_is_not_wired_into_the_shared_cancel_helper():
    """**不得**挂进 `_cancel_pending_hitl_of`。

    那个 helper 有一个活路径调用方（`_inject_user_turn` 传 `defer=True`），那条路上的请求
    正等着被注入/被 gateway 重放。在那里盖章会让它们从清单里消失，此后崩一次就永久丢。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime._cancel_pending_hitl_of)
    assert "close_resolved" not in src, (
        "这个 helper 被活路径共用，盖章会把还要用的决定销掉")


# ── 两阶段：章只能在提交点落下 ─────────────────────────────────────────────────
#
# 这一段是为一个真实缺陷补的。它一度被「替身比真实对象更终局」掩盖：早先的用例直接给
# `req.decision` 赋值造出一个已终局的请求，于是盖章顺利——而活路径上冷应答先落成
# `pending_decision`（`claim`），`req.resolved` 为 False，章根本没盖上。


def _two_phase_service():
    """registry + service，里面一条 UserTurn 请求，**用真实的 claim 走两阶段**。"""
    from ctx_weft.core.hitl.service import HitlService
    from ctx_weft.protocols.hitl import UserTurnDelivery

    reg = HitlRegistry()
    req = PendingHitl(
        id="hit_u", form="wait", session_id=_SID, task_id="t1", agent_id="ag1",
        delivery=UserTurnDelivery(task_id="t1", preface="normal"),
        created_at=_T0, tenant_id="acme", stage=HITL_STAGE_TOOL)
    reg._requests[req.id] = req
    bus = _Bus()
    svc = HitlService(registry=reg, event_bus=bus, reply_intake=None)
    reg.claim(req.id, HitlDecision(outcome="accepted", message="我的答复"), "我的答复")
    return svc, bus, req


async def test_claim_pending_is_never_stamped(caplog):
    """待终局的请求**不许**盖章，而且这不是调用点错误、不该报 WARNING。

    不许盖，是因为这一轮还能被撤销（`release`），而且它的 memory 写入此刻只在暂存区。
    不报 WARNING，是因为这正是两阶段的常态——把常态写成告警等于让告警失去意义。
    """
    import logging

    svc, bus, req = _two_phase_service()
    assert req.claim_pending and not req.resolved, "前提：处在待终局"

    with caplog.at_level(logging.WARNING):
        ok = await svc.close(req)

    assert ok is False
    assert _closed_ids(bus) == set(), "待终局就盖章 = 宣布一件还能被撤销的事"
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


async def test_commit_point_is_what_stamps_it():
    """章由提交点补，而且必须排在 `HitlResolved` **之后**。

    顺序是承重的：折叠遇到 `HitlClosed` 会摘掉 `opened`，若它排在前面，紧随其后的
    `HitlResolved` 就找不到请求（那个分支 `req is None` 直接 `continue`）——整条终局丢掉。
    """
    svc, bus, req = _two_phase_service()

    done = await svc.commit(req.id)                  # 第一阶段 → 第二阶段
    assert done is not None and done.resolved
    assert await svc.close(done) is True

    kinds = [e.type for e in bus.events
             if e.type in (EventType.HITL_RESOLVED, EventType.HITL_CLOSED)]
    assert kinds == [EventType.HITL_RESOLVED, EventType.HITL_CLOSED], kinds


async def test_release_before_commit_leaves_no_stamp():
    """这一轮被撤销 → 只发 `HitlReplyRetracted`，绝不发章。

    章发了就等于宣布「用掉了」，而那条答复恰恰是被收回的；配合折叠摘 `opened`，气泡再也
    回不到 pending。
    """
    svc, bus, req = _two_phase_service()

    released = await svc.release(req.id)

    assert released is not None
    assert _closed_ids(bus) == set()
    assert any(e.type == EventType.HITL_REPLY_RETRACTED for e in bus.events)


def test_commit_round_hook_stamps_after_both_steps():
    """`Runtime._commit_round_writes` 的三步顺序：终局 → 落盘暂存 → 盖章。

    源码顺序检查。前两步的相对顺序是既有纪律（先有终局事实，答复才进 memory）；第三步
    必须在两者之后，那是「决定已终局」与「效果已持久」同时成立的唯一位置。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime._commit_round_writes)
    i_commit = src.index("await self.hitl.commit(")
    i_ingest = src.index("await memory.ingest(")
    i_close = src.index("await self.hitl.close(")
    assert i_commit < i_ingest < i_close, (
        f"三步顺序错了：commit={i_commit} ingest={i_ingest} close={i_close}")


def test_reply_turn_writer_does_not_stamp_by_itself():
    """写对话那一步**不自己盖章**——它不知道自己这次写是暂存还是直落。

    `_ingest_user_turn` 走 `ingest_or_stage`：task 开着未提交窗口就进暂存区。所以「效果已
    持久」只有调用方知道，章归调用方。恢复期补注入那条路上请求本就已终局、也没有开窗，
    所以它那次 `close()` 照旧盖得上。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime._write_hitl_reply_turn)
    assert "ingest_or_stage" not in src   # 形态：它自己不判暂存
    assert src.count("self.hitl.close(") <= 1, (
        "这里最多只该有那一次（对已终局请求的补注入路径）")


# ── 审批不可撤销：门一过就花掉 ─────────────────────────────────────────────────


def test_authz_stages_do_not_open_an_uncommitted_round():
    """授权类应答（`authz` / `rerun`）不开未提交窗口 → `defer=False` → 当场终局。

    批准一个危险工具之后再「撤回」语义上不成立：工具可能已经跑了，撤掉的只是记录、不是世界。
    而今天那条路更糟——`abandon_round` 明文丢掉「那条答复带出来的 `CapabilityFinished` 之类」
    与暂存的工具结果，连 `CapabilityInvoked` 一起丢，于是重入时 `facts` 为空、gateway 以为它
    没跑过，连「准不准重跑」都不问就再跑一遍。

    判据是 **stage 而非 form**：stage 是 gateway 开气泡时显式传的授权语境，form 只是展示形态。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime.reply_to_hitl)
    assert "HITL_STAGE_AUTHZ" in src and "HITL_STAGE_RERUN" in src
    assert "and not _is_authz" in src, "授权类必须被排除在 _resumable 之外"
    # 判据不能退回 form：那会把 host 用 approval form 问的非授权问题一起变成不可撤销
    assert 'form == "approval"' not in src


def test_tool_stage_stays_retractable():
    """`ask_user`（`stage=tool`）**保持**可撤销——那是人打的字。

    2026-09-09 那个撤销窗口正是为它引入的：用户在 LLM 开口之前反悔，那句话当作没说过。
    这条和上一条一起，才说明这次改的是「授权」而不是「所有 HITL」。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime.reply_to_hitl)
    stages = src[src.index("_is_authz = pending.stage in"):]
    stages = stages[:stages.index(")")]
    assert "HITL_STAGE_TOOL" not in stages, "工具阶段不该被一起变成不可撤销"


def test_invocation_closes_only_the_authorization_stages():
    """`CapabilityInvoked` 之后只盖 `authz` / `rerun` 的章,不盖 `tool`。

    盖 `tool` 是错的：`ask_user` 的答复要等它变成工具结果才算被消费,在调用开始时盖等于宣布
    一件还没发生的事。
    """
    import inspect

    from ctx_weft.core.loop.capability_gateway import CapabilityGateway

    src = inspect.getsource(CapabilityGateway._record_invocation)
    assert "close_for_tool_call" in src, "门一过就该盖章"
    i_emit = src.index("CAPABILITY_INVOKED")
    i_close = src.index("close_for_tool_call")
    assert i_emit < i_close, "章要在事件之后"
    scope = src[i_close:i_close + 300]
    assert "HITL_STAGE_AUTHZ" in scope and "HITL_STAGE_RERUN" in scope
    assert "HITL_STAGE_TOOL" not in scope


async def test_close_for_tool_call_honours_the_stage_filter():
    """`stages` 过滤真的生效：只盖授权那两类,`tool` 那条留着等结果。"""
    svc, bus = _service(HITL_STAGE_AUTHZ, HITL_STAGE_TOOL, HITL_STAGE_RERUN)

    n = await svc.close_for_tool_call(
        _SID, _TCID, stages=(HITL_STAGE_AUTHZ, HITL_STAGE_RERUN))

    assert n == 2
    assert _closed_ids(bus) == {f"hit_{HITL_STAGE_AUTHZ}", f"hit_{HITL_STAGE_RERUN}"}
    # 结果落库那一步（不带 stages）再把剩下那条收掉
    assert await svc.close_for_tool_call(_SID, _TCID) == 1
    assert f"hit_{HITL_STAGE_TOOL}" in _closed_ids(bus)


async def test_result_exit_does_not_re_stamp_the_authorization_ones():
    """结果落库那一步不分 stage,但已盖过的按内存幂等跳过——不会发重复的章。"""
    svc, bus = _service(HITL_STAGE_AUTHZ, HITL_STAGE_TOOL)

    await svc.close_for_tool_call(_SID, _TCID, stages=(HITL_STAGE_AUTHZ,))
    await svc.close_for_tool_call(_SID, _TCID)                    # 兜底那次

    ids = [e.payload["hitl_id"] for e in bus.events
           if e.type == EventType.HITL_CLOSED]
    assert sorted(ids) == [f"hit_{HITL_STAGE_AUTHZ}", f"hit_{HITL_STAGE_TOOL}"], ids


# ── task 落终态：最通用的那条退役判据 ────────────────────────────────────────


async def test_close_resolved_can_narrow_to_one_task():
    """按 task 收窄——不得误伤同一 session 里别的 task 还有用的决定。"""
    svc, bus = _service(HITL_STAGE_AUTHZ)
    other = PendingHitl(
        id="hit_other", form="approval", session_id=_SID, task_id="t_other",
        agent_id="ag1", delivery=ToolResultDelivery(tool_call_id="call_x"),
        created_at=_T0, tenant_id="acme", tool_call_id="call_x", stage=HITL_STAGE_AUTHZ)
    other.decision = HitlDecision(outcome="accepted", message="ok")
    other.resolved_at = _T0
    svc.registry._requests[other.id] = other

    n = await svc.close_resolved(_SID, task_id="t1")

    assert n == 1
    assert _closed_ids(bus) == {f"hit_{HITL_STAGE_AUTHZ}"}


def test_task_terminal_hook_retires_that_tasks_decisions():
    """task 落终态是**最通用**的退役判据，一次覆盖 cancel / fail / finish。

    终态 task 不会再跑：`_inject_resolved_user_turns` 对它本就直接跳过
    （`target.status in TERMINAL_TASK_STATUSES`），gateway 也不会再为它求批。所以挂在这一个
    钩子上，比逐条去堵取消路径干净，也不会漏掉**正常结束**那一类——那类同样可能留下已终局
    未消费的决定，只是比取消罕见。
    """
    import inspect

    from ctx_weft.core.runtime import CtxWeftRuntime

    src = inspect.getsource(CtxWeftRuntime._on_task_terminal)
    assert "close_resolved" in src and "task_id=task_id" in src
    assert "clear_pins" in src, "原有的 pin 回收不能丢"

    # 钩子确实接上了这个方法（而不是仍然只接 clear_pins 的 lambda）
    assert "self._on_task_terminal(" in inspect.getsource(CtxWeftRuntime)


def test_task_terminal_hook_is_awaited():
    """钩子必须被 `await`——盖章是 async，同步调用会留下一个从不执行的协程。

    协程不 await 只会打一条 RuntimeWarning，测试照常绿，而那些决定一条都不会被销账。
    """
    import inspect

    from ctx_weft.core.orchestrator.task.manager import TaskManager

    src = inspect.getsource(TaskManager.on_task_finished)
    assert "await self._hooks.on_task_terminal(" in src


def test_retry_does_not_reach_the_terminal_hook():
    """重试**不**走终态钩子，所以「还要再跑一轮」的 task 的决定不会被提前销掉。

    与 CapabilityCache 的 pin 跨重试保住是同一个理由。这条把那个理由钉在测试里——它是新加的
    盖章能不能挂在这个钩子上的前提：挂错了，一个只是本轮没做完的 task 的批准会被销账，下一轮
    重跑时人要重新批一遍。
    """
    import inspect

    from ctx_weft.core.orchestrator.task.manager import TaskManager

    src = inspect.getsource(TaskManager._settle)
    i_retry = src.index('if status == "PENDING":')
    i_finished = src.index("await self.on_task_finished(")
    assert i_retry < i_finished, "retry 判据必须在收尾之前"
    # retry 分支里必须有 return，否则会穿下去落终态
    retry_branch = src[i_retry:i_finished]
    assert "return" in retry_branch, "retry 分支不 return 就会穿到终态收尾"
    assert "on_task_finished" not in retry_branch
