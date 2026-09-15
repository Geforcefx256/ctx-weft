"""SessionRegistry：session 创建 + root agent 启动。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from ctx_weft.core.models.errors import UnfinishedTasksError
from ctx_weft.protocols.events import Event, EventBus, EventOrigin, EventType
from ctx_weft.core.models.status import TERMINAL_TASK_STATUSES
from ctx_weft.core.utils.event import emit_event
from ctx_weft.core.orchestrator.lifecycle.agent_manager import AgentLifecycleManager
from ctx_weft.core.orchestrator.model import ModelChoice
from ctx_weft.core.orchestrator.task.manager import TaskManager
from ctx_weft.core.models.session import Session
from ctx_weft.core.models.task import Task
from ctx_weft.core.models.task import (
    CompactTaskSettings, MetadataFillerTaskSettings, NormalTaskSettings,
)
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols.context import ProviderContext



if TYPE_CHECKING:
    from ctx_weft.protocols import ContentPart

logger = logging.getLogger(__name__)

_ORIGIN = EventOrigin.ORCHESTRATOR_SESSION_REGISTRY


def _event_jsonable_or_fallback(
    user_prompt: "str | list[ContentPart]",
    user_prompt_event_jsonable: "str | list[dict] | None",
    where: str,
) -> "str | list[dict] | None":
    """事件侧载荷缺席时的判据，与 `TaskManager.push_task` 完全同形。

    新参数默认 None 是为了不打断既有的纯文本调用方；但「默认 None」若直接写进
    payload，任何忘记传参的调用方都会静默发出 ``user_prompt: None`` 的
    SESSION_CREATED / SESSION_RESUMED——prompt 就此从重放流里消失，正是本次解耦要
    消灭的那类静默降级。故：纯文本回退用它自己（`str`/`None` 本身即 jsonable，与
    `content_to_event_jsonable` 的快路逐字节一致），part 列表则**响亮拒绝**——这里
    没有原始字节可用（`user_prompt` 到这里可能已是 memory ref），拒绝是唯一诚实的选择。
    """
    if user_prompt_event_jsonable is not None:
        return user_prompt_event_jsonable
    if isinstance(user_prompt, list):
        raise ValueError(
            f"{where}: 非纯文本 user_prompt 必须由调用方传 user_prompt_event_jsonable"
            "（由归一化之前的原始 content 算出）——事件库恒不含字节，这里没有原始字节可用"
        )
    return user_prompt


@dataclass
class _SessionState:
    """会话容器：只记 tenant 归属与成员 agent。

    状态整体挪到 agent 身上（spec 2）——外部若想知道「这个 session 整体闲不闲」，
    自己聚合该 session 下所有 AgentSummary.status，不由系统预先算好广播。
    """

    tenant_id: str = "default"
    agent_ids: set[str] = field(default_factory=set)


@dataclass
class SessionRegistry:
    """Coordinates session creation and root task scheduling; owns session membership.

    从前是每次调用 new 一个、用完就扔的临时对象（无状态、不订阅事件）——会话状态因此
    无处可放，被 TaskManager / runtime / reducer 各写一份。现在是 runtime 级长生命周期
    组件，但它只是「这个 session 里注册了哪些 agent」的登记表——状态本身住在各 agent
    自己身上（docs/events-v2.md §2.1.1）。
    """

    agent_lifecycle_manager: AgentLifecycleManager
    event_bus: EventBus
    task_max_concurrent: int = 4
    task_max_retries: int = 3

    #: session_id → 容器状态（tenant + 成员 agent 集合）。
    _states: dict[str, _SessionState] = field(default_factory=dict, init=False, repr=False)

    # ── 登记 ─────────────────────────────────────────────────────────────

    def register_session(self, session_id: str, *, tenant_id: str = "default") -> None:
        """纳入管理。已存在则保留原状态（重入安全）。"""
        self._states.setdefault(session_id, _SessionState(tenant_id=tenant_id))

    def forget_session(self, session_id: str) -> None:
        """会话彻底收口后释放内存。此后 `agent_ids_of` 返回空集，调用方须先读后忘。"""
        self._states.pop(session_id, None)

    # ── 输入：总线事件 ────────────────────────────────────────────────────

    #: SM 现在只关心「谁加入了这个 session」——root 落地（AGENT_INSTANTIATED）
    #: 和子 agent 生成（AGENT_SPAWNED）。其余事件（task 队列信号、run 级事件……）
    #: 一概不消费：状态已经整体搬到 agent 自己身上，SM 不再聚合、不再广播。
    _MEMBER_EVENTS: ClassVar[frozenset[str]] = frozenset({
        EventType.AGENT_INSTANTIATED,
        EventType.AGENT_SPAWNED,
    })

    def attach_to_bus(self) -> None:
        """订阅。runtime 构造期调一次。

        ``provisional=True`` 与 ALM 同理：这是进程内登记表，成员集合要反映当下的真实，
        不受未提交窗口影响。
        """
        self.event_bus.subscribe(None, self.handle_event, provisional=True, required=True)

    async def handle_event(self, ev: Event) -> None:
        """总线回调。把新登场的 agent 收进该 session 的成员集合，别的一概不管。"""
        if ev.type not in self._MEMBER_EVENTS:
            return
        st = self._states.get(ev.session_id)
        if st is None or not ev.agent_id:
            return
        st.agent_ids.add(ev.agent_id)

    def agent_ids_of(self, session_id: str) -> set[str]:
        """该 session 下已登记的全部 agent id；未知 session 返回空集。"""
        st = self._states.get(session_id)
        return set(st.agent_ids) if st is not None else set()

    async def create_session(
        self,
        template_id: str,
        user_prompt: "str | list[ContentPart]",
        context_limit: int,
        tenant_id: str = "default",
        llm_model: str | None = None,
        llm_account: str | None = None,
        token_budget: int = 200_000,
        reserved_output_tokens: int = 8192,
        session_id: str | None = None,
        initial_task_settings: NormalTaskSettings | None = None,
        user_prompt_event_jsonable: "str | list[dict] | None" = None,
        unattended: bool = False,
    ) -> tuple[Session, Task, TaskManager]:
        """Create a new session, instantiate root agent, push initial task.

        ``unattended``：这一轮没有人看顾（后台自治作业）。原样落到 root task，并由
        `_make_root_task_manager` 强制 `interaction_mode="auto"`。见 `Task.unattended`。

        ``user_prompt_event_jsonable``：调用方（`CtxWeftRuntime.start_session`）由
        **归一化之前的原始** user_prompt 算好的 event 侧载荷（见
        `_validate_and_normalize_content`）。本类不自己算——它手上的 ``user_prompt``
        已经是 memory 侧归一化过的内容，图片 part 是 memory ref，再算一次只会把一个
        event store 打不开的引用写进事件（blob-store 解耦 Task 3）。
        """
        sid = session_id or generate_id("ses")
        ctx = ProviderContext(session_id=sid, tenant_id=tenant_id)
        # 先于 agent_lifecycle_manager.instantiate / 任何 emit：被拒的入参不该留下半个 session
        # （同「入口即拒、不落库」）。
        user_prompt_event_jsonable = _event_jsonable_or_fallback(
            user_prompt, user_prompt_event_jsonable, "create_session",
        )

        # 同样是「入口即拒、不落库」：template 解析失败必须在任何 emit 之前抛出，
        # 不能让 SESSION_CREATED 已经落库、随后才发现 template_id 是坏的。
        # 只解析这一次——LocalAgentTemplateProvider 每次 get_template 都重新扫盘
        # （热更新友好），解析两次会在 SESSION_CREATED 落库之后再开一个 TOCTOU
        # 窗口（模板可能已被热更新/删除），把「入口即拒、不落库」击穿。这里解析出的
        # template 对象直接传给下面的 instantiate(template=...)，它不会再自己解析。
        template = await self.agent_lifecycle_manager.template_lookup.get_template(
            template_id, None, ctx=ctx,
        )

        # root agent id 得在 SESSION_CREATED 之前铸出来：事件因果序必须是
        # Session → Agent → Task（host 侧 agents 表对 sessions.id 有 FK，
        # AgentInstantiated 投影早于 SESSION_CREATED 会炸），但 SESSION_CREATED
        # 的 payload 又需要 root_agent_id——所以先铸 id、后建 Session/emit，
        # 真正的 registry 登记 + AgentInstantiated 发射留给下面的 instantiate()。
        agent_id = generate_id("agt")

        session = Session(
            id=sid,
            user_prompt=user_prompt,
            status="RUNNING",
            tenant_id=tenant_id,
            root_agent_id=agent_id,
            llm_provider=llm_account or "",
            llm_model=llm_model or "",
            context_limit=context_limit,
            reserved_output_tokens=reserved_output_tokens,
            created_at=now_utc(),
        )
        logger.info("Session %s created (template=%s, agent=%s)", sid, template_id, agent_id)

        # Emit in causal order Session → Agent → Task. AGENT_INSTANTIATED (from
        # agent_lifecycle_manager.instantiate, below) and TASK_CREATED (from
        # _make_root_task_manager → push_task) MUST follow SESSION_CREATED: the host
        # projection inserts the agent/task row with a FK on session_id → sessions.id,
        # so the session row must be projected first.
        ts = now_utc()
        # 保 ref、不落字节、不拍扁（裁定 2026-08-27）——本事件参与状态重建（reducers
        # 的 SESSION_CREATED 分支），拍扁会让重放后「曾有一张图」无痕。
        await self._emit(EventType.SESSION_CREATED, sid, tenant_id, timestamp=ts, payload={
            "template_id": template_id,
            "user_prompt": user_prompt_event_jsonable,
            "root_agent_id": agent_id,
            "llm_model": llm_model or "",
            "llm_account": llm_account or "",
            "tenant_id": tenant_id,
            "token_budget": token_budget,
            "context_limit": context_limit,
            "reserved_output_tokens": reserved_output_tokens,
        })
        # 登记 record + 发 AgentInstantiated（root 没有 parent_agent_id，
        # AgentLifecycleManager.instantiate 内部只发这一条，不发 AgentSpawned）。
        # template=template：复用上面已经解析过的对象，全程只解析一次。
        # llm=：host 这次的选择（可空 = 跟随账号默认），住进 root agent 的 record——
        # 不再靠 session.llm_model/llm_provider 兼职当真值（批次 B）。
        await self.agent_lifecycle_manager.instantiate(
            template_id=template_id, session_id=sid, tenant_id=tenant_id,
            agent_id=agent_id, template=template, ctx=ctx,
            llm=ModelChoice(account=llm_account or "", model=llm_model or ""),
        )
        self.register_session(sid, tenant_id=tenant_id)

        root_task, task_manager = await self._make_root_task_manager(
            session, user_prompt, initial_task_settings, user_prompt_event_jsonable,
            unattended=unattended,
        )

        return session, root_task, task_manager

    async def resume_session(
        self,
        session_id: str,
        event_store: Any,
        user_prompt: "str | list[ContentPart]",
        tenant_id: str = "default",
        llm_model: str | None = None,
        llm_account: str | None = None,
        initial_task_settings: NormalTaskSettings | None = None,
        user_prompt_event_jsonable: "str | list[dict] | None" = None,
        unattended: bool = False,
        task_manager: TaskManager | None = None,
    ) -> tuple[Session, Task, TaskManager]:
        """Resume an existing session: recover root_agent_id from event store, push a new root task.

        ``user_prompt_event_jsonable`` 同 `create_session`：由调用方从原始 content 算好。

        ``task_manager``：该 session **活着的 owner TM**（调用方从 runtime 取；没有就传
        None）。session 的 TM 是单例——有活 owner 时新 root task 就推进它，不另建一个
        去顶替它；没有才新建。活 owner 的内存状态也一并参与下面的「弃轮禁止」判定：
        未提交窗口里的 task（spec 2026-09-09）还没落盘，只看事件视图会漏掉它们。"""
        from ctx_weft.core.control.reducers import rebuild_view
        user_prompt_event_jsonable = _event_jsonable_or_fallback(
            user_prompt, user_prompt_event_jsonable, "resume_session",
        )
        view = await rebuild_view(event_store, session_id)
        sess_proj = view.sessions.get(session_id)
        if not sess_proj or not sess_proj.root_agent_id:
            raise RuntimeError(
                f"Cannot resume session {session_id!r}: "
                "no SessionCreated event found in event store."
            )

        # 弃轮禁止：仍有未终结任务时不许开新轮（本方法只推一个新 root task，不续跑滞留
        # 任务——它们会被无声遗弃,之后 recover_agent 全量重建又把它们复活重跑）。调用方应走
        # 恢复路径续跑/收尾。辅助任务（compact/metadata）豁免——restore 也从不重排它们,
        # 阻塞会把会话永久锁死（与 TaskManager.restore 的跳过口径一致）。
        unfinished = [
            tid for tid, t in view.tasks.items()
            if t.status not in TERMINAL_TASK_STATUSES
            and (t.settings_raw or {}).get("_type") not in (
                "CompactTaskSettings", "MetadataFillerTaskSettings")
        ]
        if task_manager is not None:
            unfinished += [
                t.id for t in task_manager.all_tasks()
                if t.status not in TERMINAL_TASK_STATUSES
                and not isinstance(t.settings, (CompactTaskSettings, MetadataFillerTaskSettings))
                and t.id not in unfinished
            ]
        if unfinished:
            raise UnfinishedTasksError(session_id, unfinished)

        session = Session(
            id=session_id,
            user_prompt=user_prompt,
            status="RUNNING",
            tenant_id=tenant_id,
            root_agent_id=sess_proj.root_agent_id,
            llm_provider=llm_account or "",
            llm_model=llm_model or "",
            context_limit=sess_proj.context_limit,
            reserved_output_tokens=getattr(sess_proj, "reserved_output_tokens", 8192),
            created_at=now_utc(),
        )
        root_task, task_manager = await self._make_root_task_manager(
            session, user_prompt, initial_task_settings, user_prompt_event_jsonable,
            unattended=unattended, task_manager=task_manager,
        )

        logger.info("Session %s resumed (agent=%s)", session_id, sess_proj.root_agent_id)

        # 同 SESSION_CREATED：保 ref、不落字节、不拍扁。
        await self._emit(EventType.SESSION_RESUMED, session_id, tenant_id, payload={
            "user_prompt": user_prompt_event_jsonable,
            "root_agent_id": sess_proj.root_agent_id,
            "llm_model": llm_model or "",
            "llm_account": llm_account or "",
        })
        self.register_session(session_id, tenant_id=tenant_id)

        return session, root_task, task_manager

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _make_root_task_manager(
        self,
        session: Session,
        user_prompt: "str | list[ContentPart]",
        settings: NormalTaskSettings | None,
        user_prompt_event_jsonable: "str | list[dict] | None" = None,
        *,
        unattended: bool = False,
        task_manager: TaskManager | None = None,
    ) -> tuple[Task, TaskManager]:
        """建 root task 并推进 TaskManager。``task_manager`` 给了就用它（该 session 的
        活 owner，单例），否则新建一个。"""
        task = Task(
            id=generate_id("tsk"),
            session_id=session.id,
            status="ACTIVE",
            tenant_id=session.tenant_id,
            assigned_agent_id=session.root_agent_id,
            creator_agent_id=session.root_agent_id,
            title="",
            description="",
            user_prompt=user_prompt,
            settings=settings or NormalTaskSettings(),
            unattended=unattended,
            # root task = 用户对话：actor 纯文本即暂停等下一条用户消息（非自动完成）。
            # **无人值守时强制 auto**（不变式 `unattended ⟹ auto`，见 `Task.unattended`）：
            # 后台作业没有人会发下一条消息，interactive 的纯文本 park 就是永久挂起——
            # 没有人来应答，那条 HITL 也永远不会被终局。
            interaction_mode="auto" if unattended else "interactive",
            created_at=now_utc(),
        )
        if task_manager is None:
            task_manager = TaskManager(
                session_id=session.id,
                event_bus=self.event_bus,
                max_concurrent=self.task_max_concurrent,
                task_max_retries=self.task_max_retries,
            )
        # `push_task` 之前必须先注入 session：`TaskManager._emit` 取
        # `self._session.tenant_id if self._session else "default"`，晚注入会让 root
        # task 的 TaskCreated 落到 default 租户（总账 A5）。runtime 侧后续仍会再调一次
        # `set_session`（`start_session`/`recover_agent` 里另有用途——注入 llm 参数复用等），
        # 幂等、原样保留。
        task_manager.set_session(session)
        # root task 的 user_prompt 与 SESSION_CREATED 是同一份内容，故 event 侧载荷
        # 也是同一份——同样由调用方从原始 content 算好，不在这里重算（Task 3）。
        await task_manager.push_task(
            task, user_prompt_event_jsonable=user_prompt_event_jsonable,
        )
        return task, task_manager

    async def _emit(
        self,
        event_type: EventType,
        session_id: str,
        tenant_id: str,
        payload: dict,
        agent_id: str | None = None,
        timestamp=None,
    ) -> None:
        await emit_event(
            self.event_bus,
            event_type,
            session_id=session_id,
            tenant_id=tenant_id,
            origin=_ORIGIN,
            agent_id=agent_id,
            payload=payload,
            timestamp=timestamp,
        )
