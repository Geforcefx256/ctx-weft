"""HitlRegistry：HITL 的纯内存状态机。

**全同步、无 await。** 单线程 asyncio 下，一段没有 await 的代码原子执行，因此
「状态转移 + 取走等待槽」天然互斥，不需要锁——热投递与冷续跑的单一权威转移由
此保证（spec §6）。所有 I/O（发事实、外部化内容）归 `HitlService`。

**完备即构造**：本类的一切查询只读自己内存，绝不回落去查存储。恢复期的完备性
由装填（`load_snapshot`，Task 6）承担（spec §3.1）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ctx_weft.protocols.hitl import (
    Delivery,
    HitlAsk,
    HitlDecision,
    HitlRequestView,
    NoResumeDelivery,
)

if TYPE_CHECKING:
    from ctx_weft.core.hitl.snapshot import HitlSnapshot

logger = logging.getLogger(__name__)

#: 装填决定时的占位创建时间——占位项只为回答 `decision_for`，永不出现在 pending 列表里，
#: 故取最小值即可（GC 排序用 resolved_at，装填项无 resolved_at 时回落到它）。
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

#: 决定缓存键的第三维——**core 内部键，不进 protocols**：host 不需要知道「谁问的」，
#: 只有 gateway（授权步）与工具 provider（工具步）自己需要区分（Task 4.5）。
HITL_STAGE_AUTHZ = "authz"
HITL_STAGE_TOOL = "tool"
#: 重跑授权（spec: tool-operations）——崩溃恢复时问「这次还该不该再跑」。
#: 必须与 AUTHZ 分开：同一个 tool_call 的事前授权决定与重跑决定是两件事，
#: 共用缓存键会让前者的「批准」被当成后者的「可以重跑」。
HITL_STAGE_RERUN = "rerun"


class WaitSlot(Protocol):
    """热等待的会合槽——**只有一个操作**的不透明句柄。

    刻意保持最小：registry 因此只接触一个并发原语，不接触任何 loop 类型，箭头
    仍然朝下（spec §3.2）。一旦往槽里塞更丰富的 loop 对象，反向依赖就回来了。

    返回 True = 已被热投递消费（`claimed`）；False = 投递未被接受（等待方已放弃）。
    """

    def deliver(self, decision: HitlDecision) -> bool: ...


@dataclass
class PendingHitl:
    """core 内部的活记录。**不出 core**——对外只经 `to_view()` 投影。"""

    id: str
    form: str
    session_id: str
    task_id: str
    agent_id: str
    delivery: Delivery
    created_at: datetime
    #: 开出这条请求时那次调用所在的租户——`HitlService._emit` 据此发 `HitlOpened` /
    #: `HitlResolved`（同一个 `req` 对象终局时沿用开局时的值，见 `_commit`）。不出
    #: `to_view()`：host 不需要这个维度，只有事件流的 `Event.tenant_id` 需要（总账 A5）。
    tenant_id: str = "default"
    subject_id: str = ""
    prompt: str = ""
    detail: str = ""
    fields: list[dict[str, Any]] = field(default_factory=list)
    proposal: dict[str, Any] | None = None
    tool_call_id: str = ""                       # 幂等键 + 决定缓存键
    #: 决定缓存键的第三维（连同 session_id、tool_call_id）：`HITL_STAGE_AUTHZ` /
    #: `HITL_STAGE_TOOL`。**只按 tool_call_id 查会让授权步吃掉工具阶段的答复**，见
    #: `find_for_tool_call` docstring。core 内部键，不出 `to_view()`。
    stage: str = ""
    #: 决定缓存键的**第四维**：这条决定授权/回答的是**哪一次调用**（工具名 + 原始参数的
    #: 摘要，见 `capability_gateway.invocation_key`）。模型复用 tool_call id 是常态
    #: （`call_1` 这类短值），只按前三维查会让第 3 轮的批准替第 9 轮的**另一次**调用开门，
    #: 并把第 3 轮的 `modified_arguments` 一并带进去（复审 I3）。
    #:
    #: `""` = 未键控，**通配**：旧模型事件折出来的记录、以及 host 直接喂的快照没有这一维，
    #: 让它们照旧命中（迁移期逐条同构）。活路径开出来的记录恒有非空 key。
    invocation_key: str = ""
    resume_state: dict[str, Any] | None = None   # 不透明，core 永不解读
    reply_as_result: bool = False
    #: 终局决定。**唯一的结局存储**——`resolved` 由它推导，不存第二份。
    decision: HitlDecision | None = None
    #: **已收到、尚未终局**的答复（两阶段终局，spec 2026-09-09）。
    #:
    #: 一条冷应答会开出新的一轮（注入对话 + 重排 + 跑 act）。那一轮在 LLM 真的开口
    #: 之前不算发生，所以这条答复也不能先终局：用户在 TTFT 窗口里按暂停要能把整轮撤掉，
    #: 而「气泡已经收口」是撤不回来的。于是先落在这里，act 的提交点才挪进 `decision`
    #: （`commit_claim`），丢弃则原样扔掉、气泡回到 pending（`release_claim`）。
    #:
    #: **`resolved` 仍只看 `decision`** —— 结局的唯一存储没有变成两份。带着
    #: `pending_decision` 的请求在事件日志里也仍然只有 `HitlOpened`，所以崩溃重启后
    #: 它如实地回到 pending，人重答一次即可。那正是要的语义。
    #:
    #: 热投递（有活等待槽）不走这条：那条路上没有「新一轮」，`HitlService` 就地提交。
    pending_decision: HitlDecision | None = None
    #: `pending_decision` 配套的事件载荷（`ReplyIntake.normalize` 的产物）。提交时才
    #: 发 `HitlResolved`，那时不能拿 `decision.message` 重算——它已是 memory 侧的 ref。
    pending_event_payload: "str | list[dict] | None" = None
    #: 这条气泡被撤回过几次答复。**从事件日志折出来**（`HitlReplyRetracted` 的条数，
    #: 见 `fold_hitl_snapshot`），不是内存计数器——撤销之后重启，只有日志说得清。
    #: 唯一用途是 memory 幂等键的第二维，见 `reply_memory_id`。
    reply_attempt: int = 0
    resolved_at: datetime | None = None
    slot: WaitSlot | None = None
    #: 本次终局是否被一个活等待槽热消费（`HitlService._commit` 在取槽的同一原子段里
    #: 判定并写入）。默认 `False`——快照装填出来的项永远是冷的（重启后一切皆冷），
    #: 这正是我们要的默认值，不需要装填路径另外清它。调用方（`reply_to_hitl`）据此
    #: 决定要不要触发冷续跑：已被热消费的不得再触发一次，否则同一次应答驱动两跑。
    claimed: bool = False
    #: 这条记录是从**旧模型事件**（`HITL_REQUIRED` + 各旧终态）折出来的。
    #:
    #: 恢复期的 `UserTurn` 补写（`Runtime._inject_resolved_user_turns`）据此跳过它：
    #: 补写关的是**新模型**的崩溃窗口，而旧路径注入的记忆记录不带
    #: `hitlreply:{hitl_id}` 幂等键、去重不了，补一次就凭空多一轮用户发言。
    #: 活路径开出来的请求恒为 False。**core 内部字段，不出 `to_view()`。**
    legacy_origin: bool = False
    #: 这条决定**已了结**：消费完成了，`HitlClosed` 已经发出（见那个事件类型的说明）。
    #:
    #: 纯内存、不入事件也不入快照：折叠遇到 `HitlClosed` 直接把这条从 `resolved` /
    #: `decisions_for` 摘掉，所以装填回来的记录里根本不会有 closed=True 的。它在内存里
    #: 存在只为两件事——① 让 `resolved_for_session()` 与「折叠装填出来的集合」保持一致
    #: （那条不变式写在那个方法的 docstring 里，是恢复期兜底的承重点）；② 让盖章幂等，
    #: 取消 + 销毁这类前后相继的路径不会给同一条请求发两遍章。
    closed: bool = False

    @property
    def resolved(self) -> bool:
        return self.decision is not None

    @property
    def effective_decision(self) -> "HitlDecision | None":
        """人**已经给出**的那个决定，不管它终局了没有。

        续跑路径要读的是这一份：两阶段之下（spec 2026-09-09）冷应答先落成
        `pending_decision`，而把它注入对话、按 outcome 分流的那些代码跑在提交点
        **之前** —— 读 `decision` 会拿到 None，用户说的那句话就静默变成空串。

        判「终局了没有」仍然只看 `decision`（`resolved`），两个问题各答各的。
        """
        return self.decision or self.pending_decision

    @property
    def claim_pending(self) -> bool:
        """有一条已收到但尚未终局的答复（见 `pending_decision`）。

        这样的请求**不该出现在 `list_pending` 里**：host 拿它去渲染面板，会让人对着
        一个自己刚答过的问题再答一次。但它在事件日志里仍是 pending —— 内存与日志的
        这处分歧是故意的，见 `pending_decision` 的说明。
        """
        return self.pending_decision is not None

    def to_view(self) -> HitlRequestView:
        return HitlRequestView(
            id=self.id, form=self.form, session_id=self.session_id, task_id=self.task_id,
            created_at=self.created_at, agent_id=self.agent_id, subject_id=self.subject_id,
            prompt=self.prompt, detail=self.detail, fields=list(self.fields),
            proposal=self.proposal,
            # 待终局的答复也要如实回报：调用方（host）刚把人的决定交进来，它读这个字段
            # 是为了确认「我这次应答被收下了」，而不是问「事实落盘了没有」。两阶段之下
            # 收下与落盘不再是同一刻，但「收下」仍然是真的。落盘时刻另有 `resolved_at`，
            # 它在提交之前保持 None —— 两个字段各答各的问题。
            outcome=self.effective_decision.outcome if self.effective_decision else "",
            delivery=self.delivery,
            resolved_at=self.resolved_at,
        )


def reply_memory_id(req: "PendingHitl") -> str:
    """这条应答落 memory 时的幂等键。**唯一派生点**（写入侧与恢复期补写侧共用）。

    第一次应答恒是 `hitlreply:{hitl_id}`；被撤销过再重答的带上 `:{n}`，`n` 取自
    `reply_attempt`（从事件日志里 `HitlReplyRetracted` 的条数折出来，跨重启成立）。

    ⚠ **这一维如今是历史包袱，不再承重**。它当初的理由是：撤销一轮要把已经写进 memory 的
    那条答复 `fold` 成 superseded，而 memory 的 id 契约是「已存在的 id（含已 superseded）
    = no-op」，换句话说旧键还占着，重答不换键就会被静默吞掉。改成**暂存**之后
    （`TaskManager.stage_memory`），被撤销的答复根本没进过 memory，也就没有占着的旧键。
    留着只是不动存量数据与既有日志协议；要清理的话，`reply_attempt` 与
    `HitlReplyRetracted` 的折叠会一起失去唯一的用途。
    """
    base = f"hitlreply:{req.id}"
    return base if req.reply_attempt == 0 else f"{base}:{req.reply_attempt}"


class HitlRegistry:
    """登记 / 幂等 / 决定缓存 / 等待槽 / GC。"""

    def __init__(self, max_resolved: int = 1000) -> None:
        #: 已终局项的保留上限：决定缓存只需近期的，超限裁剪最旧者，防止长跑进程无界增长。
        #: pending 永不裁剪。
        self._max_resolved = max_resolved
        self._requests: dict[str, PendingHitl] = {}

    # ── 写 ────────────────────────────────────────────────────────────────────

    def open(
        self,
        ask: HitlAsk,
        *,
        hitl_id: str,
        session_id: str,
        task_id: str,
        agent_id: str = "",
        tool_call_id: str = "",
        stage: str,
        created_at: datetime,
        invocation_key: str = "",
        tenant_id: str = "default",
    ) -> PendingHitl:
        """登记一个请求。同 `(session_id, tool_call_id, stage)` 已有记录 → **复用**，不新建
        （幂等，spec §10）。

        空 `tool_call_id` 不作幂等键——`UserTurn` 的冷 park 本就没有 tool_call。
        """
        existing = self.find_for_tool_call(
            session_id, tool_call_id, stage, invocation_key=invocation_key or None)
        if existing is not None:
            return existing
        req = PendingHitl(
            id=hitl_id, form=ask.form, session_id=session_id, task_id=task_id,
            agent_id=agent_id, delivery=ask.delivery, created_at=created_at,
            tenant_id=tenant_id,
            subject_id=ask.subject_id, prompt=ask.prompt, detail=ask.detail,
            fields=list(ask.fields), proposal=ask.proposal, tool_call_id=tool_call_id,
            stage=stage, invocation_key=invocation_key,
            resume_state=ask.resume_state, reply_as_result=ask.reply_as_result,
        )
        self._requests[hitl_id] = req
        return req

    def attach_slot(self, hitl_id: str, slot: WaitSlot) -> None:
        req = self._requests.get(hitl_id)
        if req is not None:
            req.slot = slot

    def detach_slot(self, hitl_id: str) -> None:
        """驱逐等待槽（热→冷降级）。**驱逐本身永不触发续跑**，唯有应答才触发。"""
        req = self._requests.get(hitl_id)
        if req is not None:
            req.slot = None

    def resolve(
        self, hitl_id: str, decision: HitlDecision, resolved_at: datetime,
    ) -> tuple[PendingHitl, WaitSlot | None] | None:
        """终局转移 + **原子地**取走等待槽。

        返回 `(req, slot)`；已终局或未知 id → `None`（调用方据此保持幂等：不重发事实）。
        整段无 await，故转移与取槽不可能被别的协程插入。
        """
        req = self._requests.get(hitl_id)
        if req is None or req.resolved:
            return None
        req.decision = decision
        req.resolved_at = resolved_at
        slot, req.slot = req.slot, None
        return req, slot

    def claim(
        self, hitl_id: str, decision: HitlDecision,
        event_payload: "str | list[dict] | None" = None,
    ) -> "tuple[PendingHitl, WaitSlot | None] | None":
        """**待终局**转移 + 原子地取走等待槽（两阶段的第一阶段，spec 2026-09-09）。

        与 `resolve` 只差一处：写的是 `pending_decision` 而不是 `decision`，于是
        `resolved` 仍为 False、这条请求在事件日志里仍然只有 `HitlOpened`。取槽仍在
        同一段无 await 的代码里完成，「热投递」与「冷续跑」的互斥不受影响。

        已终局 / 已有待终局答复 / 未知 id → `None`（调用方据此保持幂等：不重发事实）。
        """
        req = self._requests.get(hitl_id)
        if req is None or req.resolved or req.claim_pending:
            return None
        req.pending_decision = decision
        req.pending_event_payload = event_payload
        slot, req.slot = req.slot, None
        return req, slot

    def commit_claim(self, hitl_id: str, resolved_at: datetime) -> "PendingHitl | None":
        """待终局 → 终局。无待终局答复 / 已终局 / 未知 id → `None`。"""
        req = self._requests.get(hitl_id)
        if req is None or req.resolved or not req.claim_pending:
            return None
        req.decision = req.pending_decision
        req.pending_decision = None
        req.resolved_at = resolved_at
        return req

    def uncommit_claim(self, hitl_id: str) -> "PendingHitl | None":
        """`commit_claim` 的逆：终局 → 回待终局。只用于「终局事实没发出去」的回滚。

        答复（`decision` → `pending_decision`）与事件载荷（`pending_event_payload`，
        提交成功前没有清）原样保留，重试的 `commit` 拿到的仍是同一份。
        """
        req = self._requests.get(hitl_id)
        if req is None or not req.resolved or req.claim_pending:
            return None
        req.pending_decision = req.decision
        req.decision = None
        req.resolved_at = None
        return req

    def release_claim(self, hitl_id: str) -> "PendingHitl | None":
        """待终局 → 回 pending：这一轮被丢弃了，那条答复当作没说过。

        **等待槽不还**：`claim` 取走它的那一刻，热投递要么已经发生（那条路根本不会
        走到本方法）、要么这个槽本就不存在（冷路径）。凭空造一个槽回去只会让下一次
        应答误以为有人在同步等着。
        """
        req = self._requests.get(hitl_id)
        if req is None or req.resolved or not req.claim_pending:
            return None
        req.pending_decision = None
        req.pending_event_payload = None
        # 撤回之后没有任何协程在等它了（热投递的那个 run 已经 park 掉），下一次应答
        # 必须按冷路径分流。
        req.claimed = False
        # 与 `HitlReplyRetracted` 折出来的口径保持一致：内存里也 +1，好让**同一进程内**
        # 紧接着的重答不必等重启折叠就拿到对的键。日志是真相源，这里只是同步。
        req.reply_attempt += 1
        return req

    def forget_session(self, session_id: str) -> int:
        """把该 session 的**全部**记录（未决 + 已终局）摘出内存，返回摘掉的条数。

        `gc()` 的定向版本：那个按 `_max_resolved` 裁剪最旧的已终局项、且 pending 永不
        裁剪——它管的是"长跑进程别无界增长"。这个管的是"这条会话不存在了"，所以连
        pending 一起摘，也不看年龄。

        **只在会话被销毁时调**（`CtxWeftRuntime.purge_session`）。普通的内存逐出
        （`forget_session` 那条路）**不该**调它：那条路上会话随时能从事件日志装填回来，
        而已终局记录还有用——`resolved_for_session()` 是恢复期的崩溃窗口兜底（决定已
        落盘、进程在续跑前就死了；对 `UserTurn` 那一类，续跑动作是"把人的答复注入进
        对话"，不补的话人说的那句话静默消失）。摘了它，那条兜底就失效。

        **纯机制**：不发事件、不投递决定、不管谁在等。摘掉一条**未决**记录等于让它的
        热等待者（若有）永远醒不过来，所以调用方必须先把它们终局掉
        （`purge_session` 的上一步 `cancel_session` 就是干这个的）。真摘到未决的会记一条
        WARNING——那说明上一步没做干净，是要查的，不是要忍的。
        """
        doomed = [r for r in self._requests.values() if r.session_id == session_id]
        unresolved = [r.id for r in doomed if not r.resolved]
        if unresolved:
            logger.warning(
                "HitlRegistry.forget_session(%s): 摘掉了 %d 条**未决**请求 %s——"
                "调用方本应先终局它们（热等待者会就此永远挂着）",
                session_id, len(unresolved), unresolved[:5],
            )
        for r in doomed:
            self._requests.pop(r.id, None)
        return len(doomed)

    def gc(self) -> None:
        """裁剪已终局项，pending 永不裁剪。"""
        resolved = [r for r in self._requests.values() if r.resolved]
        if len(resolved) <= self._max_resolved:
            return
        resolved.sort(key=lambda r: r.resolved_at or r.created_at)
        for r in resolved[: len(resolved) - self._max_resolved]:
            self._requests.pop(r.id, None)

    def load_snapshot(self, snapshot: HitlSnapshot) -> int:
        """把折叠结果装填进内存，返回 pending 条数。

        **完备即构造**：装填之后 core 的一切查询只读内存，绝不回落去 scan 事件日志。
        装填的完备性因此是恢复路径的责任（spec §3.1）。

        两条规则：
        - 装填出来的 pending **不带等待槽**——重启后一切皆冷（spec §10）。
        - 已有的**活 pending 优先**：日志里的旧决定不得盖掉一个正在等人的请求，
          否则会把活请求判成「已答过」而跳过。

        **调用方在传入 `snapshot` 之前须先处理 event 侧 blob ref**：
        `snapshot.decisions_for[*][0].message` 仍是事件 blob store 命名空间下的引用，
        见 `HitlSnapshot` docstring 与 `fold_hitl_snapshot` docstring（spec §12.3.3，
        参照 `runtime.py:1943` 的 hydrate + normalize 实现）——本方法不做这一步。
        """
        for hitl_id, req in snapshot.pending.items():
            req.slot = None
            self._requests.setdefault(hitl_id, req)

        # ① 已终局的**请求本体**。装的是完整记录（`task_id` / `delivery` / `form` /
        #    `created_at` 都在），恢复期的两个消费方——`resolved_for_session()` 与
        #    `decision_for()`——由此都只读内存。**包括没有 tool_call_id 的 `UserTurn`
        #    park**：它进不了 `decisions_for`（那是按 tool_call 建的决定缓存），但恢复
        #    期必须看得见它，否则人答过的那句话在崩溃窗口里静默消失（复审 Finding 2）。
        for hitl_id, req in snapshot.resolved.items():
            live = self.find_for_tool_call(
                req.session_id, req.tool_call_id, req.stage,
                invocation_key=req.invocation_key or None)
            if live is not None and not live.resolved:
                continue                       # 活 pending 优先，旧决定不得盖掉活等待
            req.slot = None
            if req.resolved_at is None:
                req.resolved_at = _EPOCH
            self._requests.setdefault(hitl_id, req)

        # ② 只有决定、没有请求本体的快照（手工构造 / host 直接喂）：退回占位项。
        #    ① 已装过的会在 `find_for_tool_call` 处被认出来，不重复装。
        for (session_id, tool_call_id, stage), (decision, resume_state) in (
            snapshot.decisions_for.items()
        ):
            live = self.find_for_tool_call(session_id, tool_call_id, stage)
            if live is not None:
                continue                       # 活 pending 或已装填的决定，均不覆盖
            placeholder = PendingHitl(
                id=f"loaded:{session_id}:{stage}:{tool_call_id}", form="",
                session_id=session_id, task_id="",
                agent_id="", delivery=NoResumeDelivery(), created_at=_EPOCH,
                tool_call_id=tool_call_id, stage=stage, resume_state=resume_state,
                decision=decision,
            )
            self._requests[placeholder.id] = placeholder
        return len(snapshot.pending)

    # ── 读 ────────────────────────────────────────────────────────────────────

    def get(self, hitl_id: str) -> PendingHitl | None:
        return self._requests.get(hitl_id)

    def find_for_tool_call(
        self, session_id: str, tool_call_id: str, stage: str,
        invocation_key: str | None = None,
    ) -> PendingHitl | None:
        """按 (session, tool_call, stage[, invocation_key]) 取最近一条；空 tool_call_id → None。

        **三维缺一不可**：只按 tool_call_id 查会让 A 会话的批准替 B 会话里同名 id 的调用
        开门（LLM 的 tool_call id 常是 `call_1` 这类短值——跨会话授权绕过），也会让授权步
        吃掉工具阶段的答复、把它的 modified_arguments 当成工具参数送进 provider（跨阶段
        混淆，Task 4.5）。

        **第四维 `invocation_key`**（复审 I3）：`None` = 不按调用区分（登记本体的查询、
        host 的只读查询）；非 `None` = 只认同一次调用的记录——同 session 同 id 的**另一次**
        调用不得复用它的决定。记录侧的 `""` 通配（旧事件折出来的没有这一维，见
        `PendingHitl.invocation_key`）。
        """
        if not tool_call_id:
            return None
        matches = [r for r in self._requests.values()
                   if r.tool_call_id == tool_call_id
                   and r.session_id == session_id
                   and r.stage == stage
                   and (invocation_key is None
                        or not r.invocation_key
                        or r.invocation_key == invocation_key)]
        return max(matches, key=lambda r: r.created_at) if matches else None

    def mark_closed(self, hitl_id: str) -> bool:
        """把一条记录标成**已了结**。返回这次是否真标上（已标过 / 找不到 → False）。

        纯机制：不发事件、不判顺序。盖章那一步在 `HitlService.close`，它先发事件再来标——
        标了却没发出去，这条就永远不在清单里了，那才是真丢。
        """
        req = self._requests.get(hitl_id)
        if req is None or req.closed:
            return False
        req.closed = True
        return True

    def resolved_for_tool_call(self, session_id: str, tool_call_id: str) -> list[PendingHitl]:
        """该 tool_call 的全部**已终局**请求，三个 stage 都算。空 tool_call_id → 空表。

        `HitlService.close_for_tool_call` 的取数口——一次调用结束时那几条一起失效，所以这里
        **不按 stage 过滤**，也**不带 `invocation_key`**：同 id 另一次调用的记录也该跟着了结，
        它同样不会再被问（决定缓存按 `invocation_key` 认人，对不上就不会命中）。
        """
        if not tool_call_id:
            return []
        return [r for r in self._requests.values()
                if r.tool_call_id == tool_call_id
                and r.session_id == session_id
                and r.resolved and not r.closed]

    def result_is_human_reply(self, session_id: str, tool_call_id: str) -> bool:
        """这次调用的结果**就是人的答复**：工具阶段有一个 `reply_as_result` 的请求（`ask_user`）。

        用途：判「事件流里记着的那份工具结果能不能拿来补写 / 重放」。答案是不能——
        它的 `CapabilityFinished` 与 `HitlResolved` 在同一轮提交里先后落盘，崩在两者之间就是
        「日志里有结果、问题还悬着」；重启后人再答一次，拿事件里那份补写，新答复就被静默
        丢弃（又一次「答了没用」）。而且事件载荷是脱敏 + 截断版，答复里的图只剩文本标记。

        这类结果是答复的纯函数、不碰任何副作用，所以正确做法恒为「按 HITL 决定重新生成」：
        问题还悬着 → 继续等人；已有决定（待终局或已终局）→ 交给 gateway 重入，命中决定缓存。
        `HumanResumable` 的工具阶段请求（`reply_as_result=False`）**不在此列**——那条路会
        重新调 `provider.resume`，可能重放副作用。
        """
        req = self.find_for_tool_call(session_id, tool_call_id, HITL_STAGE_TOOL)
        return req is not None and req.reply_as_result

    def decision_for(
        self, session_id: str, tool_call_id: str, stage: str,
        invocation_key: str | None = None,
    ) -> tuple[HitlDecision, dict[str, Any] | None] | None:
        """决定缓存查询：`(decision, resume_state)` 成对返回。

        成对是硬要求——冷路径重入调的是 `resume(ask_id, decision, resume_state, ctx)`，
        丢掉 `resume_state` 就等于要求 provider 重做让出前的工作（spec §7.2）。

        仍 pending（活的等待）→ None：不得把它当成「已答过」。

        `invocation_key` 见 `find_for_tool_call`：授权步的短路**必须**传它，否则模型复用
        tool_call id 时旧决定会替新调用开门（复审 I3）。
        """
        req = self.find_for_tool_call(session_id, tool_call_id, stage, invocation_key)
        # `effective_decision`：批准这次调用的那条应答可能还停在待终局（两阶段，
        # spec 2026-09-09）——它正是**这一轮**的应答，而 gateway 重放被批准的
        # tool_call 恰恰发生在这一轮的 act 里、在提交点前后都可能。读 `decision`
        # 会查不到，gateway 于是再问一次人，会话就此卡死。这一轮若被丢弃，
        # `release_claim` 会把它一并撤掉，缓存不会留下幽灵批准。
        eff = req.effective_decision if req is not None else None
        if eff is None:
            return None
        return eff, req.resume_state

    def list_pending(
        self, session_id: str | None = None, *, agent_id: str | None = None,
    ) -> list[PendingHitl]:
        return [
            r for r in self._requests.values()
            if not r.resolved and not r.claim_pending
            and (session_id is None or r.session_id == session_id)
            and (agent_id is None or r.agent_id == agent_id)
        ]

    def claim_pending_for_session(self, session_id: str) -> list[PendingHitl]:
        """该 session 上**已收到答复、尚未终局**的请求（不分 task）。

        两个消费方都在收尾路径上：`cancel_session` 要把它们退回待答再收口（它们不在
        `list_pending` 里，直接遍历那份列表会把它们漏掉、永远卡在待终局）；
        `session_is_quiescent` 要据此判「这条会话还不能被逐出」。
        """
        return [r for r in self._requests.values()
                if r.claim_pending and r.session_id == session_id]

    def claim_pending_for_task(self, session_id: str, task_id: str) -> list[PendingHitl]:
        """该 task 上**已收到答复但尚未终局**的请求（两阶段，见 `PendingHitl.pending_decision`）。

        `_revert_round` 用它把这一轮收口掉的气泡全部退回 pending —— 按 task 查而不是
        记一个 hitl_id，是因为 `_cancel_pending_hitl_of` 收口的是该 agent 名下**全部**
        未决请求，可能不止一条。
        """
        return [
            r for r in self._requests.values()
            if r.claim_pending and r.session_id == session_id and r.task_id == task_id
        ]

    def resolved_for_session(self, session_id: str) -> list[PendingHitl]:
        """该 session **已终局**的请求。`list_pending` 的镜像：纯内存、同步、不查存储。

        用途是恢复期的崩溃窗口兜底（Task 9）：决定已落盘、但进程在续跑之前就死了。
        对 `UserTurn` 这一类，续跑动作是「把人的答复注入进对话」，而任务重排本身并不
        带上这一步——不补，人说的那句话就静默消失（复审 Finding 2）。

        **口径**：返回的集合与 `load_snapshot` 装填的已终局集合完全一致——装填漏掉的
        请求这里就看不见，兜底也随之失效。这就是「装填的完备性是恢复路径的责任」
        （spec §3.1）在这里第二次成为承重点。
        """
        return [
            r for r in self._requests.values()
            if r.resolved and not r.closed and r.session_id == session_id
        ]
