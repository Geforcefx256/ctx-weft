"""HitlService：HITL 的唯一漏斗。

只做三件事：登记（open）、终局（resolve / cancel）、**发事实**。它不认识 Runtime、
不认识协程栈、不持久化任何东西——耐久性是 event store provider 的事。

**冷续跑不是订阅出来的**（spec §7.3 订正）：早期草案让一个 `ResumeCoordinator` 订阅
`HitlResolved` 去驱动冷续跑，那个设计已被推翻——总线 handler 在 `emit()` 内同步 drain
且背压下丢事件，把控制流的关键信号挂上去，「人答了但会话永不续跑」就成了可能。现行
唯一驱动方是 `CtxWeftRuntime.reply_to_hitl` 的**返回值**：它按 `resolved.claimed` 分流，
未被热投递消费的才触发冷续跑。本模块只发事实，不认识续跑。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ctx_weft.core.hitl.registry import HitlRegistry, PendingHitl
from ctx_weft.core.hitl.reply_intake import ReplyIntake
from ctx_weft.core.utils.event import emit_event
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.events import EventOrigin, EventType
from ctx_weft.protocols.hitl import (
    HITL_OUTCOME_CANCELLED,
    Delivery,
    HitlAsk,
    HitlDecision,
    HitlReply,
    NoResumeDelivery,
    ToolResultDelivery,
    UserTurnDelivery,
)

if TYPE_CHECKING:
    from ctx_weft.protocols.events import EventBus
    from ctx_weft.protocols import ContentPart

logger = logging.getLogger(__name__)

_ORIGIN = EventOrigin.HITL_SERVICE


def delivery_to_payload(delivery: Delivery) -> dict[str, Any]:
    """Delivery → 事件载荷。**封闭值域**，故穷举即完备。"""
    if isinstance(delivery, ToolResultDelivery):
        return {"kind": "tool_result", "tool_call_id": delivery.tool_call_id}
    if isinstance(delivery, UserTurnDelivery):
        return {"kind": "user_turn", "task_id": delivery.task_id,
                "preface": delivery.preface}
    if isinstance(delivery, NoResumeDelivery):
        return {"kind": "no_resume"}
    raise ValueError(f"Unknown delivery: {delivery!r}")


class UnattendedHitl(Exception):
    """无人值守的 task 里发起了 HITL。

    这是控制流信号，不是故障——调用方必须 catch 并转成贴合上下文的工具结果，
    **绝不能让它逸出到 agent loop**：agent 该收到一个说得清楚的结果，而不是一次 run 失败。

    与 `HitlPark` 同一摆法（住在抛它的那个模块里，而不是集中式的 `models/errors.py`）：
    两者都是 HITL 控制流的信号类型，只有直接调用方需要认识它们。
    """

    def __init__(self, form: str = "", subject_id: str = "", *,
                 session_id: str = "", task_id: str = "") -> None:
        self.form = form
        self.subject_id = subject_id
        self.session_id = session_id
        self.task_id = task_id
        super().__init__(
            f"HITL requested in an unattended task: form={form!r} subject={subject_id!r} "
            f"(session={session_id!r} task={task_id!r}) — nobody is there to answer"
        )


class HitlService:
    def __init__(
        self,
        registry: HitlRegistry,
        event_bus: "EventBus",
        reply_intake: ReplyIntake,
        *,
        id_factory: Callable[[], str] = lambda: generate_id("hit"),
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        self.registry = registry
        self._bus = event_bus
        self._intake = reply_intake
        self._new_id = id_factory
        self._now = clock

    async def open(
        self,
        ask: HitlAsk,
        *,
        session_id: str,
        task_id: str,
        agent_id: str = "",
        tool_call_id: str = "",
        stage: str,
        unattended: bool,
        invocation_key: str = "",
        tenant_id: str = "default",
    ) -> PendingHitl:
        """登记一个请求并发 `HitlOpened`。同 `(session_id, tool_call_id, stage,
        invocation_key)` 复用既有请求且**不重发事实**。

        `invocation_key` 见 `PendingHitl.invocation_key`：同一 tool_call id 下的**另一次**
        调用不得复用上一次的记录/决定（复审 I3）。

        `tenant_id`：调用方从其上下文（`ProviderContext.tenant_id` / `Session.tenant_id`）
        传入——本类自己不持有、也不去解——存进 `PendingHitl.tenant_id`，供 `_emit` 与
        之后 `resolve`/`cancel` 时同一个 `req` 复用（总账 A5：漏填时事件落到 `Event` 的
        默认值 `"default"`，非 default 租户的投影租户就错了）。

        `unattended`：发起方那个 task 的 `Task.unattended`。**必填 keyword-only、无默认值**
        （同上面的 `stage`）：这是全仓唯一的 HITL 登记入口，也就是唯一能把「没有人会来
        应答」这件事一处堵死的地方；给它一个默认值，等于把守卫交给下一个调用点的记性。
        为真时抛 `UnattendedHitl`，由调用方转成贴合上下文的工具结果。
        """
        if unattended:
            # **排在幂等复用之前**：无人值守的 task 本就不该存在任何「等人回答」的记录，
            # 把同键的旧记录当答案返回，等于让一条它根本不该有的 pending 复活。
            raise UnattendedHitl(
                ask.form, ask.subject_id, session_id=session_id, task_id=task_id)
        existing = self.registry.find_for_tool_call(
            session_id, tool_call_id, stage, invocation_key=invocation_key or None)
        if existing is not None:
            return existing
        req = self.registry.open(
            ask, hitl_id=self._new_id(), session_id=session_id, task_id=task_id,
            agent_id=agent_id, tool_call_id=tool_call_id, stage=stage, created_at=self._now(),
            invocation_key=invocation_key, tenant_id=tenant_id,
        )
        logger.info("HITL opened [%s]: %s (%s)", req.form, req.id, req.prompt[:80])
        await self._emit(EventType.HITL_OPENED, req, {
            "hitl_id": req.id,
            "form": req.form,
            "delivery": delivery_to_payload(req.delivery),
            "subject_id": req.subject_id,
            "prompt": req.prompt,
            "detail": req.detail,
            "fields": list(req.fields),
            "proposal": req.proposal,
            "tool_call_id": req.tool_call_id,
            "stage": req.stage,
            "invocation_key": req.invocation_key,
            "agent_id": req.agent_id,
            "resume_state": req.resume_state,
            "reply_as_result": req.reply_as_result,
        })
        return req

    async def resolve(self, reply: HitlReply, *, defer: bool = False) -> PendingHitl | None:
        """终局一个请求。已终局 / 已有待终局答复 → `None`（幂等 no-op，不重发事实）；
        未知 id → `KeyError`。

        ``defer=True``（两阶段，spec 2026-09-09）：**冷**应答只登记 `pending_decision`，
        不发 `HitlResolved`、不 gc——那条答复会开出新的一轮，而一轮在 LLM 真的开口之前
        不算发生。真终局由 `commit(hitl_id)` 在 act 的提交点完成，`release(hitl_id)` 则
        把它退回 pending（用户在 TTFT 窗口里按了暂停）。

        **热投递同样推迟**：有活等待槽时决定照常就地递给那个协程（`claimed=True`），但
        终局一样留到提交点——人答完到 LLM 开口之间按暂停，这条答复要能撤回。
        """
        req = self.registry.get(reply.hitl_id)
        if req is None:
            raise KeyError(f"No HITL request found: {reply.hitl_id}")
        # 校验/外部化**先于**任何状态改动：被拒的内容不得写进 decision、不得发事实。
        message, event_payload = await self._intake.normalize(reply.message, req)
        decision = HitlDecision(outcome=reply.outcome, message=message,
                                modified_arguments=reply.modified_arguments)
        return await self._commit(req, decision, event_payload, defer=defer)

    async def cancel(self, hitl_id: str, *, message: "str | list[ContentPart]" = "",
                     defer: bool = False) -> PendingHitl | None:
        """收口一个悬挂 pending（会话关闭 / 熔断）。终态、不 requeue；已终局则 no-op。

        message 与 resolve 同走 `ReplyIntake`——不走同一条路就会发出「message 为真、
        载荷为 None」的事实，把「为什么被取消」从重放流里抹掉。

        ``defer`` 同 `resolve`：`send_message` 注入一条新消息时对旧气泡的收口要跟着
        那一轮走（撤销时旧气泡得回来）；会话销毁 / 熔断那些调用方必须用默认的 False。
        """
        req = self.registry.get(hitl_id)
        if req is None:
            raise KeyError(f"No HITL request found: {hitl_id}")
        normalized, event_payload = await self._intake.normalize(message, req)
        decision = HitlDecision(outcome=HITL_OUTCOME_CANCELLED, message=normalized)
        return await self._commit(req, decision, event_payload, defer=defer)

    async def commit(self, hitl_id: str) -> PendingHitl | None:
        """两阶段的第二阶段：待终局 → 终局 + 发 `HitlResolved`。

        由 act 的提交点调用（这一轮的 LLM 真的开口了）。无待终局答复 / 已终局 → `None`。
        """
        payload_carrier = self.registry.get(hitl_id)
        event_payload = payload_carrier.pending_event_payload if payload_carrier else None
        resolved = self.registry.commit_claim(hitl_id, self._now())
        if resolved is None:
            return None
        try:
            await self._emit_resolved(resolved, resolved.decision, event_payload,
                                      claimed=resolved.claimed)
        except BaseException:
            # 事实没落盘 → 内存里也不能算终局。退回待终局（答复与载荷原样保留），让
            # 调用方的提交整体失败、稍后重试；不退回的话，内存说「答完了」、日志说「还悬着」，
            # 而提交钩子若照常把答复写进 memory，重启后人重答会被 id 幂等静默吞掉。
            self.registry.uncommit_claim(hitl_id)
            raise
        resolved.pending_event_payload = None
        self.registry.gc()
        return resolved

    async def release(self, hitl_id: str) -> PendingHitl | None:
        """两阶段的回退：待终局 → 回 pending。这一轮被丢弃，那条答复当作没说过。

        **`HitlResolved` 从来没发过**，所以被撤回的那句话不会留在日志里；`HitlOpened`
        还在原地，折出来的状态就是撤销之前的 pending，会话状态因此自然回到 `PAUSED`。

        但要发一条 `HitlReplyRetracted`（**不含正文**）。理由是「这条气泡被答过又撤了」
        必须**可还原**：撤销之后重启，内存里什么都没有，而重答时 memory 的幂等键要靠
        「这是第几次」才不会撞上上一次留下的 superseded 记录（见 `reply_memory_id`）。
        不发这条事件，那个数就只能是内存计数器，而内存计数器跨不过重启——那正是
        `test_retyped_reply_survives_a_restart_in_the_discard_window` 钉住的形状。

        ⚠ **发的时候不带 `task_id`**。撤销发生在未提交窗口关闭**之前**（那是丢弃路径
        的顺序纪律），而那道闸正是按 task_id 定的——带上 task_id 这条事件就会落进缓冲、
        随之被一起丢掉，等于没发。与 `RoundCommitted` / `RoundDiscarded` 同一处理。
        """
        released = self.registry.release_claim(hitl_id)
        if released is None:
            return None
        await emit_event(
            self._bus,
            EventType.HITL_REPLY_RETRACTED,
            session_id=released.session_id,
            tenant_id=released.tenant_id,
            origin=_ORIGIN,
            task_id=None,                    # 见 docstring：带上就会被未提交窗口挡住
            agent_id=released.agent_id or None,
            payload={"hitl_id": released.id},
            timestamp=self._now(),
        )
        return released

    # ── internals ─────────────────────────────────────────────────────────────

    async def _commit(
        self,
        req: PendingHitl,
        decision: HitlDecision,
        message_event_payload: "str | list[dict] | None",
        *,
        defer: bool = False,
    ) -> PendingHitl | None:
        """状态转移 → 取槽 → 投递 → 发事实。

        转移与取槽在 `registry.resolve()` / `registry.claim()` 里同步完成
        （无 await ⟹ 原子），因此「热投递」与「冷续跑」互斥、不双投。投递与发事实
        在其后，不占原子段。

        ``defer=True`` 时走 `claim()`：冷热都只登记待终局。热投递的协程拿到的是
        `pending_decision`，提交点由 gateway 在它醒来时重新武装，所以不会「拿着一份
        永远不终局的答复继续跑」。
        """
        transferred = (
            self.registry.claim(req.id, decision, message_event_payload) if defer
            else self.registry.resolve(req.id, decision, self._now())
        )
        if transferred is None:
            return None                              # 已终局 / 已待终局：幂等 no-op
        resolved, slot = transferred
        claimed = False
        if slot is not None:
            # deliver 声明为不抛（-> bool），但对一个已完成的 future 再次 set 会抛
            # InvalidStateError。转移已不可逆——这里若真抛出且不接住，请求就停在
            # 「已终局」却没有 HitlResolved 事实，跨重启无法恢复。发事实的义务优先于
            # 让这个异常继续传播。
            try:
                claimed = bool(slot.deliver(decision))
            except Exception:
                logger.exception(
                    "HitlService._commit: slot.deliver raised for hitl_id=%s; "
                    "treating as unclaimed and still emitting HitlResolved", resolved.id)
                claimed = False
        resolved.claimed = claimed
        if defer:
            # 冷热同一口径：待终局，事件留到 act 的提交点再发（`commit`）。
            #
            # 热投递曾经在这里就地终局，理由是「协程就地醒来，没有新一轮可撤销」。那不对：
            # 人答完之后到 LLM 真的开口之前，用户同样可能按暂停，而那时整轮（这条答复、
            # 它回灌出来的工具结果）都还没发生。被叫醒的协程由 gateway 重新武装提交点
            # （`_resolve_human`），暂停则由 act 走热撤销（`_discard_round_if_uncommitted`）。
            return resolved
        await self._emit_resolved(resolved, decision, message_event_payload, claimed=claimed)
        self.registry.gc()
        return resolved

    async def _emit_resolved(
        self,
        resolved: PendingHitl,
        decision: HitlDecision,
        message_event_payload: "str | list[dict] | None",
        *,
        claimed: bool,
    ) -> None:
        """发 `HitlResolved`。一步终局与两阶段提交共用，载荷口径只此一份。"""
        payload: dict[str, Any] = {
            "hitl_id": resolved.id,
            "outcome": decision.outcome,
            "claimed": claimed,
        }
        # 事件载荷由**原始**内容一步之前算好、顺参数递进来——不在这里拿
        # decision.message 重算：那份内容已是 memory 侧的 ref，event store 解不开。
        if message_event_payload:
            payload["message"] = message_event_payload
        if decision.modified_arguments is not None:
            payload["modified_arguments"] = decision.modified_arguments
        await self._emit(EventType.HITL_RESOLVED, resolved, payload)

    async def close(self, req: PendingHitl) -> bool:
        """给一条**已终局**的请求盖「已了结」章：发 `HitlClosed`。返回有没有真盖上。

        语义见 `EventType.HITL_CLOSED`：它说的是「这条决定已经被消费掉，再也不会被问」。
        **调用点必须在持久效果落地之后**——答复已进对话、工具结果已进 memory。提前盖章会让
        折叠不再把它列进补注入清单，而那份效果其实还没落，那句话就永久消失了。

        只给**已终局**的盖。未决 / 待终局一律安静跳过：

        - 未决：人还没答，没有决定可谈。
        - **待终局**（两阶段的常态）：答复收下了但这一轮还能被撤销（`release`），而且它的
          memory 写入此刻还在暂存区（`ingest_or_stage`）。这种请求的章由提交点补
          （`Runtime._commit_round_writes` 的尾巴）——那里 `HitlResolved` 刚发完、暂存的写入刚落盘，
          是两个条件同时成立的唯一位置。**这道跳过因此是承重的**：在它之前盖章，事件会进
          这一轮的缓冲、补投时排在 `HitlResolved` **之前**，折叠会先摘掉 `opened` 再撞上
          `HitlResolved` 找不到请求，那条终局就整个丢了。

        **收请求对象而不是 id**：调用方手上本来就有它，而按 id 回查 registry 会凭空多一个
        失败模式——`gc()` 会按 `_max_resolved` 逐出最旧的已终局项，被逐出的那条就永远盖不上
        章、永远留在折叠的清单里。盖章不该依赖内存里还留着它。

        **不抛**：这是一枚事后的章，它失败不该回滚已经成功的消费。发不出去的后果只是退回
        本事件引入之前的行为（那条记录继续留在清单里、下次恢复幂等补一遍），不是数据损坏。
        """
        if req.closed:
            return False                     # 已经盖过——取消 + 销毁这类相继路径不重复发
        if not req.resolved:
            # **待终局不是错误，是两阶段的常态。** 冷应答先落成 `pending_decision`，而把它
            # 注入对话、被 gateway 重放的那些代码都跑在提交点**之前**（见
            # `PendingHitl.effective_decision`）。此刻盖章会宣布一件还能被撤销的事，所以
            # 这里安静跳过——那一枚章由提交点补（`Runtime._commit_round_writes` 的尾巴），它是
            # 「决定已终局」与「效果已持久」同时成立的唯一位置。
            logger.debug(
                "HitlService.close(%s): 待终局，章交给提交点补", req.id)
            return False
        try:
            await self._emit(EventType.HITL_CLOSED, req, {"hitl_id": req.id})
        except Exception:
            logger.exception(
                "HitlService.close(%s): HitlClosed 发射失败（消费已经成功，"
                "下次恢复会幂等补一遍）", req.id)
            return False
        # **先发事件、后标内存**：反了的话「标了但没发出去」会让这条在内存里消失、日志里
        # 又没有章，恢复期的清单两头都看不见它——那是真丢，比多发一次章严重得多。
        self.registry.mark_closed(req.id)
        return True

    async def close_resolved(
        self, session_id: str, *,
        agent_id: str | None = None, task_id: str | None = None,
    ) -> int:
        """把该 session（可按 agent / task 收窄）全部**已终局**请求一并了结，返回盖上几条。

        用在**取消 / 销毁**路径上：那些决定的消费者已经不存在了——任务不会再跑，
        `_inject_resolved_user_turns` 对终态 task 本就直接跳过，gateway 也不会再为它求批。
        不盖章它们就永远留在折叠的清单里，而那条会话的事件日志还在（`purge_session` 明确
        **不删日志**，删不删是 host 的步骤）。

        ⚠️ **只该由真终结的路径调**。它不区分「这条决定马上就要被消费」——活路径上那些
        请求正等着被注入/被 gateway 重放，盖章会让它们从清单里消失，此后崩一次就永久丢。
        所以不要把它挂进 `_cancel_pending_hitl_of` 那类共享 helper（它有一个 `defer=True`
        的活路径调用方 `_inject_user_turn`）。

        `task_id` 收窄用在 **task 落终态**那条路上（`on_task_terminal`）。那是最通用的退役
        判据：终态 task 不会再跑，`_inject_resolved_user_turns` 对它本就直接跳过
        （`target.status in TERMINAL_TASK_STATUSES`），gateway 也不会再为它求批。一次覆盖
        cancel / fail / finish 三种，不必逐条堵取消路径。
        """
        n = 0
        for req in self.registry.resolved_for_session(session_id):
            if agent_id is not None and req.agent_id != agent_id:
                continue
            if task_id is not None and req.task_id != task_id:
                continue
            if await self.close(req):
                n += 1
        return n

    async def close_for_tool_call(
        self, session_id: str, tool_call_id: str, *,
        stages: "tuple[str, ...] | None" = None,
    ) -> int:
        """把该 tool_call 的已终局请求了结，返回盖上几条。`stages=None` = 不分 stage。

        **默认不分 stage**：结果落库那一刻，这次调用牵到的三类请求（`authz` 事前审核、
        `tool` provider 自问、`rerun` 准重跑）同时失效；逐个 stage 去盖要调用方记住有哪几个，
        那种知识迟早漏掉一个。

        `stages` 只在一个地方收窄：**授权类决定在调用开始时就了结**（`_record_invocation`
        发完 `CapabilityInvoked` 之后，只盖 `authz` / `rerun`）。审批的作用是放行这一次调用,
        门一过它就花掉了；留着它等结果，等于让一份花掉的批准在重入时还能开门。`tool` 那一类
        不能这么早盖——`ask_user` 的答复要等它变成工具结果才算被消费。

        调用点一律在持久效果之后（事件已发 / 结果已进 memory），见 `close` 的顺序纪律。
        """
        if not tool_call_id:
            return 0
        n = 0
        for req in self.registry.resolved_for_tool_call(session_id, tool_call_id):
            if stages is not None and req.stage not in stages:
                continue
            if await self.close(req):
                n += 1
        return n

    async def _emit(self, event_type: EventType, req: PendingHitl, payload: dict) -> None:
        await emit_event(
            self._bus,
            event_type,
            session_id=req.session_id,
            tenant_id=req.tenant_id,
            origin=_ORIGIN,
            task_id=req.task_id or None,
            agent_id=req.agent_id or None,
            payload=payload,
            # **显式传**：HitlService 持有一个可注入的时钟（单测靠它冻结时间），
            # 丢掉它会让时间源静默换成 emit_event 内部的 now_utc()。
            timestamp=self._now(),
        )
