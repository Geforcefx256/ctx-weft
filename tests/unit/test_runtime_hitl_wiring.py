"""Runtime 的 HITL 接线与应答入口（段 2 · Task 8）。"""

from __future__ import annotations

import inspect

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.models.errors import InvalidContentError
from ctx_weft.protocols import ImagePart
from ctx_weft.protocols.hitl import (
    HitlAsk,
    HitlReply,
    NoResumeDelivery,
    ToolResultDelivery,
    UserTurnDelivery,
)
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from tests._initialization import initialize_all
from tests.integration.test_minimal_loop import (
    InlineAgentTemplateProvider,
    make_echo_template,
    make_runtime,
)


def _runtime() -> CtxWeftRuntime:
    provider = InlineAgentTemplateProvider()
    provider.register(make_echo_template())
    rt = make_runtime(agent_provider=provider)
    rt.providers.register_memory(InMemoryMemoryProvider())
    return rt


async def _warm(rt, session_id):
    await initialize_all(rt)
    if session_id not in rt._task_managers:
        await rt.create_session(template_id="agent:tpl_echo", session_id=session_id)


def _runtime_with_recorded_resume() -> tuple[CtxWeftRuntime, list[tuple]]:
    """构造一个 runtime，把 `recover_agent` / 用户回合注入替换成记录桩。

    2026-09-04（Task 15）起主键换成 agent：`_resume_after_hitl` 传的是
    ``req.agent_id``，不再是 ``req.session_id``——桩记的第二个字段跟着换轴。

    记录的元组形态：
    - ``("recover_session", agent_id, resumed_task_id)`` —— ToolResultDelivery 续跑；
    - ``("inject_user_turn", agent_id, task_id)`` —— UserTurnDelivery 续跑。

    `recover_agent` 不再接受 llm_account/llm_model（换模型走
    `set_agent_llm`/`set_session_llm` 两条独立命令），桩签名同步收紧。
    """
    rt = _runtime()
    calls: list[tuple] = []

    async def fake_recover_agent(agent_id, *, resumed_task_id=None, user_reply=None,
                                  hitl_id=""):
        if user_reply is not None:
            calls.append(("inject_user_turn", agent_id, resumed_task_id))
        else:
            calls.append(("recover_session", agent_id, resumed_task_id))

    rt._recover_session_locked = fake_recover_agent  # type: ignore[method-assign]
    return rt, calls


def _ask_tool_result(tool_call_id: str) -> HitlAsk:
    return HitlAsk(form="approval", delivery=ToolResultDelivery(tool_call_id=tool_call_id))


def _ask_user_turn(task_id: str) -> HitlAsk:
    return HitlAsk(form="wait", delivery=UserTurnDelivery(task_id=task_id))


def _ask_no_resume() -> HitlAsk:
    return HitlAsk(form="wait", delivery=NoResumeDelivery())


class _AcceptingSlot:
    """一个总是宣称已消费投递的等待槽——模拟「热等待方还在」。"""

    def deliver(self, decision) -> bool:
        return True


async def test_no_setter_injection_remains():
    """三个 setter 全部消失——构造完即可用，没有半成品窗口。"""
    import ctx_weft.core.runtime as mod

    src = inspect.getsource(mod)
    for name in ("set_cold_resolve_handler", "set_cold_decision_lookup",
                 "set_content_normalizer"):
        assert name not in src


async def test_hitl_service_is_usable_immediately_after_construction():
    rt = _runtime()
    assert rt.hitl is not None and rt.hitl_registry is not None


async def test_reply_returns_the_view_and_drives_resume_by_delivery():
    """冷续跑由**返回值**驱动，不挂总线订阅（spec §7.3 订正）。"""
    rt, calls = _runtime_with_recorded_resume()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)
    await initialize_all(rt)
    view = await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="accepted",
                                            agent_id=req.agent_id))
    assert view is not None and view.outcome == "accepted"
    # `_resume_after_hitl` 按 delivery 路由，参数原样透传给 `recover_agent`——真实生产
    # 路径的 `hitl.open()` 调用方（`capability_gateway.py`/`act.py`）恒传非空 `agent_id`，
    # 这里种一个真实形态的请求验证这条透传，而不是巧合地依赖签名默认值 `agent_id=""`
    # 那个专供「legacy 折叠、真没有 agent_id 可传」场景的回退分支
    # （见 `tests/unit/test_cold_hitl_legacy_agent_id_resume.py`，终审 CRITICAL 2）。
    assert calls == [("recover_session", "s1", "t1")]


async def test_user_turn_delivery_injects_instead_of_reconciling():
    rt, calls = _runtime_with_recorded_resume()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_user_turn("t1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", stage="tool", unattended=False)
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="accepted",
                                     agent_id=req.agent_id, message="继续"))
    assert calls[0][0] == "inject_user_turn"


async def test_no_resume_delivery_triggers_nothing():
    rt, calls = _runtime_with_recorded_resume()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_no_resume(), session_id="s1", task_id="t1",
                             agent_id="agent-a", stage="tool", unattended=False)
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="cancelled", agent_id=req.agent_id))
    assert calls == []


async def test_a_claimed_hot_reply_does_not_trigger_cold_resume():
    """热投递已就地续跑，再触发一次冷续跑就是双投。"""
    rt, calls = _runtime_with_recorded_resume()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)
    rt.hitl_registry.attach_slot(req.id, _AcceptingSlot())
    await initialize_all(rt)
    view = await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="accepted",
                                            agent_id=req.agent_id))
    # 不能只看 calls == []——resolve() 失败（未知 id / 已终局）时同样不产生任何 call，
    # 两种情况必须区分开：这里断言的是「已终局且真被消费」，不是「resolve 失败了」。
    assert view is not None and view.outcome == "accepted"
    assert calls == []


async def test_replying_twice_resumes_at_most_once():
    """应答入口可能被重试（host 超时重发 / 用户连点）——第二次是 no-op。"""
    rt, calls = _runtime_with_recorded_resume()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)
    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="accepted", agent_id=req.agent_id))
    await initialize_all(rt)
    assert await rt.reply_to_hitl(
        HitlReply(hitl_id=req.id, outcome="accepted", agent_id=req.agent_id)
    ) is None
    assert len(calls) == 1


async def test_empty_agent_id_falls_back_to_session_id_not_recover_agent():
    """终审 CRITICAL 2 的单元级接线守卫：legacy 折叠出的空 `agent_id` 必须走
    `_recover_session_locked(session_id, ...)` 的回退，绝不能再原样喂给
    `recover_agent("")`（那必然自愈失败 → `AgentNotFound`，见
    `tests/unit/test_cold_hitl_legacy_agent_id_resume.py` 的完整端到端复现）。
    这里只钉「路由选了哪条分支」这个快速的接线事实，不跑真实事件日志重建。
    """
    rt = _runtime()
    recover_agent_calls: list[str] = []
    session_calls: list[tuple] = []

    async def fake_recover_agent(agent_id, **kw):
        recover_agent_calls.append(agent_id)

    async def fake_recover_session_locked(session_id, **kw):
        session_calls.append((session_id, kw.get("resumed_task_id")))

    rt._recover_session_locked = fake_recover_agent  # type: ignore[method-assign]
    rt._recover_session_locked = fake_recover_session_locked  # type: ignore[method-assign]

    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             tool_call_id="call_1", stage="tool", unattended=False)  # 无 agent_id -> 落到默认空串
    assert req.agent_id == ""

    await initialize_all(rt)
    await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="accepted", agent_id=""))

    assert recover_agent_calls == [], "空 agent_id 不该再喂给 recover_agent"
    assert session_calls == [("s1", "t1")], "必须回退到 session_id 精确装填"


async def test_hitl_reply_intake_is_wired_to_the_runtime_shared_normalizer():
    """`ReplyIntake` 必须挂着 `_normalize_hitl_content`——不是某个桩、不是恒等变换。

    这是与旧 `set_content_normalizer` 那道接线等价的钉子：以前 `test_hitl_multimodal_
    validation.py` 直接断言 legacy manager 的 normalizer 就是这个方法；新路径下没有
    setter 可断言了，改断言 `ReplyIntake` 构造时收到的就是它，防止有人以后悄悄拿掉。
    """
    rt = _runtime()
    assert rt.hitl._intake._normalizer == rt._normalize_hitl_content


async def test_reply_with_a_disallowed_image_media_type_is_rejected_and_stays_pending():
    """校验能力在唯一的生产路径（`reply_to_hitl`）上是活的，不是被静默拆掉的接线。

    白名单外的 media type 必须在 `validate_content` 就地拒绝——不落库、不发事实、
    请求原样保持未决，供人类重新提交一个合法答复。
    """
    rt = _runtime()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             tool_call_id="call_1", stage="tool", unattended=False)
    events: list = []

    async def _record(ev):
        events.append(ev)

    rt._event_bus.subscribe(None, _record)  # type: ignore[attr-defined]

    with pytest.raises(InvalidContentError):
        await initialize_all(rt)
        await rt.reply_to_hitl(HitlReply(
            hitl_id=req.id, outcome="accepted", agent_id=req.agent_id,
            message=[ImagePart(data="AAAA", media_type="image/bmp")],
        ))

    pending = rt.hitl_registry.get(req.id)
    assert pending is not None and pending.resolved is False
    assert events == []


async def test_a_failed_cold_resume_after_commit_is_logged_loudly_and_still_raises(
    caplog,
):
    """resolve() 已经不可逆地提交——续跑失败没有第二次机会,必须在日志里带上 hitl_id
    响亮地留痕,而不是只悄悄传给 host（复审 cheap fix）。"""
    rt = _runtime()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)

    async def _boom(*a, **kw):
        raise RuntimeError("owner TM rebuild exploded")

    rt._recover_session_locked = _boom  # type: ignore[method-assign]

    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="owner TM rebuild exploded"):
            await initialize_all(rt)
            await rt.reply_to_hitl(HitlReply(hitl_id=req.id, outcome="accepted",
                                             agent_id=req.agent_id))

    assert req.id in caplog.text
    # 应答本身已经不可逆地提交——即便续跑失败，请求也真的终局了。
    assert rt.hitl_registry.get(req.id).resolved is True


# ── host 面向 HITL 的读入口（复审 I5a）───────────────────────────────────────


async def test_list_pending_hitl_returns_views_not_core_records():
    """host 不该被迫走 `runtime.hitl_registry.list_pending()`——那返回的 `PendingHitl`
    自己的 docstring 就写着「不出 core」。"""
    from ctx_weft.protocols.hitl import HitlRequestView

    rt = _runtime()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             tool_call_id="call_1", stage="tool", unattended=False)
    await _warm(rt, "s2")
    await rt.hitl.open(_ask_tool_result("call_2"), session_id="s2", task_id="t9",
                       tool_call_id="call_2", stage="tool", unattended=False)

    await initialize_all(rt)
    all_pending = [item.data for item in await rt.list_pending_hitl()]
    assert len(all_pending) == 2
    assert all(isinstance(v, HitlRequestView) for v in all_pending)
    assert not any(hasattr(v, "tool_call_id") or hasattr(v, "slot") for v in all_pending)

    await initialize_all(rt)
    only_s1 = [item.data for item in await rt.list_pending_hitl(session_id="s1")]
    assert [v.id for v in only_s1] == [req.id]

    # 经 service 终局（不走 reply_to_hitl，那会去事件库找一个本测试没建的会话）
    await rt.hitl.resolve(HitlReply(hitl_id=req.id, outcome="accepted", agent_id=req.agent_id))
    await initialize_all(rt)
    assert [item.data for item in await rt.list_pending_hitl(session_id="s1")] == []


# ── agent_id 防呆校验（Task 21，spec 4.3）───────────────────────────────────


async def test_reply_to_hitl_rejects_agent_id_mismatch():
    """调用方声明的 agent 与系统记录不符 → 拒绝，不静默按 hitl_id 走掉。"""
    rt = _runtime()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)
    with pytest.raises(ValueError):
        await initialize_all(rt)
        await rt.reply_to_hitl(
            HitlReply(hitl_id=req.id, outcome="accepted", agent_id="wrong-agent")
        )


async def test_reply_to_hitl_agent_id_mismatch_rejected_before_any_side_effect():
    """拒绝必须发生在任何副作用之前——不能先把回复写进去再报错。"""
    rt = _runtime()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)
    events: list = []

    async def _record(ev):
        events.append(ev)

    rt._event_bus.subscribe(None, _record)  # type: ignore[attr-defined]

    with pytest.raises(ValueError):
        await initialize_all(rt)
        await rt.reply_to_hitl(
            HitlReply(hitl_id=req.id, outcome="accepted", agent_id="wrong-agent")
        )

    pending = rt.hitl_registry.get(req.id)
    assert pending is not None and pending.resolved is False
    assert events == []


async def test_reply_to_hitl_agent_id_match_proceeds_normally():
    """agent_id 与记录相符 → 正常终局并续跑，新增校验不影响正路。"""
    rt, calls = _runtime_with_recorded_resume()
    await _warm(rt, "s1")
    req = await rt.hitl.open(_ask_tool_result("call_1"), session_id="s1", task_id="t1",
                             agent_id="agent-a", tool_call_id="call_1", stage="tool", unattended=False)
    await initialize_all(rt)
    view = await rt.reply_to_hitl(
        HitlReply(hitl_id=req.id, outcome="accepted", agent_id="agent-a")
    )
    assert view is not None and view.outcome == "accepted"
    assert calls == [("recover_session", "s1", "t1")]


def test_hitl_reply_requires_agent_id():
    import dataclasses

    f = {x.name: x for x in dataclasses.fields(HitlReply)}
    assert "agent_id" in f
    assert f["agent_id"].default is dataclasses.MISSING, "agent_id 必填"
