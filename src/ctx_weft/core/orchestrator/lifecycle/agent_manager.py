"""AgentLifecycleManager：Agent 实例化 + spawn 深度检查。

Capability 解析已移至 PrepareStep（CapabilityResolver），
此处只负责从 template 创建 Agent 对象。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Protocol

from ctx_weft.core.control.types import AgentView
from ctx_weft.core.utils.event import emit_event
from ctx_weft.core.models.errors import AgentBusyError, AgentNotFound, AgentTerminatedError, CtxWeftError
from ctx_weft.core.orchestrator.lifecycle.agent_state import AgentInput, next_agent_transition
from ctx_weft.core.orchestrator.model import ModelChoice, ModelResolver, ResolvedModel
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.models.agent import Agent, LoopGuard
from ctx_weft.core.utils.clock import now_utc
from ctx_weft.core.utils.ids import generate_id
from ctx_weft.protocols import LLMClient, LoopConfig, MemoryConfig
from ctx_weft.protocols.context import ProviderContext
from ctx_weft.protocols.events import Event, EventBus, EventOrigin, EventType
from ctx_weft.protocols.template import AgentTemplate

logger = logging.getLogger(__name__)

_ORIGIN = EventOrigin.RUNTIME

#: TASK_* 终态 -> SETTLED 转移的 reason 缺省值（payload 没带显式 reason 时用）。
#: 只覆盖六种 task 终态事件——它们都回 idle（spec 3.1：task 终态不是 agent 终态）。
_SETTLE_REASON: dict[str, str] = {
    EventType.TASK_FINISHED: "task_finished",
    EventType.TASK_FAILED: "task_failed",
    EventType.TASK_CANCELED: "task_canceled",
    EventType.TASK_FINALIZED: "task_finalized",
    EventType.TASK_REQUEUED: "task_requeued",
    EventType.TASK_SUSPENDED: "task_suspended",
}


# LoopGuard() 的字段默认值——instantiate() 末尾构造 Agent 时刻意不解模型：会话创建
# （SessionRegistry.create_session）必须不碰 LLM，惰性解析要留到派发时才发生
# （test_content_validation.py 钉死这条：纯文本 start_session 不注册 LLM provider
# 也必须成功返回，_resolve_llm 一次都不许被调用）。真正生效的窗口由派发点的
# materialize()/resolve_model() 按 agent record 的 ModelChoice 现解、stamp 进 loop_guard。
_DEFAULT_CONTEXT_LIMIT = LoopGuard().context_limit
_DEFAULT_RESERVED_OUTPUT_TOKENS = LoopGuard().reserved_output_tokens


class UnknownCapabilityError(CtxWeftError):
    pass


class SpawnDepthExceeded(CtxWeftError):
    pass


class DuplicateAgentId(CtxWeftError):
    """instantiate(agent_id=...) 撞上已登记的 id——编程错误，不是「水合」。

    这个参数的含义是「这个*新* agent 的 id」（今天只有 SessionRegistry.create_session
    的 root 分支在用：它得先铸好 id 才能把 root_agent_id 塞进 SESSION_CREATED
    payload，而 SESSION_CREATED 必须先于 AgentInstantiated——见该处调用注释）。
    传一个已存在的 id 不是「可能是水合」的旧 existing_agent_id 语义（那个歧义已被
    Task 3 消灭，水合走 materialize()）——撞上就是调用方算错了 id，直接抛错。
    """


@dataclass
class _SessionDefaults:
    tenant_id: str
    fallback_template_id: str


@dataclass
class _AgentRecord:
    session_id: str
    tenant_id: str
    template_id: str
    parent_agent_id: str | None
    spawn_depth: int
    memory_config: MemoryConfig
    loop_config: LoopConfig
    llm: ModelChoice = field(default_factory=ModelChoice)
    status: str = "idle"                     # spec 3.1 五态机的当前值
    current_task_id: str | None = None       # 消息路由据此判断新建还是复用 task
    #: 实例化时刻。恢复路径（`load`）留 None——`AgentView` 不带该字段，
    #: 硬造一个「恢复时刻」当创建时刻是在撒谎。host 侧要精确值可读
    #: AgentInstantiated 事件的 timestamp。
    created_at: datetime | None = None
    #: 由 `_register_fallback` 就地补出来的占位记录（恢复期缺口：id 还没被装填就先被
    #: 用到了，session 归属是**猜**的）。`load()` 装填真记录时允许覆盖它——它不代表
    #: 一个已确认归属的 agent，不参与 agent_id 唯一性判定。
    placeholder: bool = False


@dataclass(frozen=True)
class AgentRecordView:
    """`_AgentRecord` 的只读快照——ALM 对外（core 内部）唯一的记录读法。

    是**快照**不是引用：取出之后 registry 再变也不影响手上这份，调用方不会拿到
    一个会自己变的「只读」对象。要最新值就再调一次 `record_of`。
    """

    agent_id: str
    session_id: str
    tenant_id: str
    template_id: str
    parent_agent_id: str | None
    spawn_depth: int
    status: str
    current_task_id: str | None
    created_at: datetime | None


@dataclass
class AgentLifecycleManager:
    """Agent 实例化 + 注册表。

    从前是「runtime.py 里 new 五次、用完即弃的无状态 dataclass」，现在是
    runtime 级长生命周期组件，`_agents` 是 agent 身份与配置的唯一住所——
    与 SessionRegistry 在 2026-09-02 做过的那次晋升同形（docs/events-v2.md §2.1.1）。
    Capability 解析已移至 PrepareStep（CapabilityResolver），此处只负责从
    template 创建 Agent 对象并登记。
    """

    template_lookup: "TemplateLookup"
    event_bus: EventBus
    model_resolver: ModelResolver
    _agents: dict[str, _AgentRecord] = field(default_factory=dict)
    _sessions: dict[str, _SessionDefaults] = field(default_factory=dict)
    # parent_agent_id -> {child_agent_id, ...}。只在 instantiate 落 record 后维护；
    # forget_session / forget_agent 必须把摘除的 agent 从键和所有值集合里都清掉，否则
    # descendants_of 会经由悬垂引用「复活」已逐出的 agent（见二者内注释）。
    _children: dict[str, set[str]] = field(default_factory=dict)

    # ── ALM：TASK_* 驱动五态机，发 AGENT_* ─────────────────────────────────

    #: ALM 的全部事件输入 → 状态机输入。**只含 TASK_***：`AgentLifecycleManager.apply_input`
    #: 发出的是 AGENT_*，若这张表也收 AGENT_*，内置 InProcessEventBus 的同步 drain
    #: 会在 `emit()` 返回前把刚发的事件回流给 `handle_event` 自己，一次转移变成
    #: 递归——与 `SessionRegistry._INPUT_BY_EVENT` 当初避开同一坑同一口径（该文件
    #: docstring 原话）。
    _INPUT_BY_EVENT: ClassVar[dict[str, AgentInput]] = {
        EventType.TASK_STARTED: AgentInput.TASK_STARTED,
        EventType.TASK_AWAITING_HUMAN: AgentInput.AWAITING_HUMAN,
        EventType.TASK_HUMAN_RESOLVED: AgentInput.HUMAN_RESOLVED,
        EventType.TASK_INTERRUPTED: AgentInput.INTERRUPTED,
        EventType.TASK_RESUMED: AgentInput.RESUMED,
        # 以下六种一律回 idle——task 终态不是 agent 终态（spec 3.1）。
        EventType.TASK_FINISHED: AgentInput.SETTLED,
        EventType.TASK_FAILED: AgentInput.SETTLED,
        EventType.TASK_CANCELED: AgentInput.SETTLED,
        EventType.TASK_FINALIZED: AgentInput.SETTLED,
        EventType.TASK_REQUEUED: AgentInput.SETTLED,
        EventType.TASK_SUSPENDED: AgentInput.SETTLED,
    }

    #: 恢复期现状广播的状态 → 事件映射。`terminated` 刻意不在表里（粘滞终态，重发无信息）；
    #: `running` 也不在——进程刚起来没有任何 run 在跑，把折出来的 `running` 照发会让 host
    #: 以为有活在跑。它在事件流里的真实含义是「崩溃时正在跑」，恢复后等 /resume 重新派发。
    _RECOVERY_BROADCAST_BY_STATUS: ClassVar[dict[str, EventType]] = {
        "idle": EventType.AGENT_IDLE,
        "waiting_human": EventType.AGENT_WAITING_HUMAN,
        "interrupted": EventType.AGENT_INTERRUPTED,
    }

    def attach_to_bus(self) -> None:
        """订阅。runtime 构造期调一次。

        ``provisional=True``：未提交窗口（见 `EventBus` 类 docstring）里的 `TASK_STARTED`
        也要照收。ALM 反映的是「现在真实发生了什么」，不是「哪一轮算数」——收不到就意味着
        派发出去的 agent 停在 `idle`，`pause_agent` 会拿 `AgentNotRunningError` 拒掉那个
        窗口里的暂停请求、host 的会话状态折叠会把会话判成上一轮终态，并发闸门一并失效。
        """
        # required=True（spec: event-commit）：必要状态消费者——异常穿出 emit，
        # 不再被 bus 吞掉；推测态照看（provisional）不变。
        self.event_bus.subscribe(None, self.handle_event, provisional=True, required=True)

    async def handle_event(self, ev: Event) -> None:
        """总线回调。**只读事件、只喂状态机**，不碰其他组件。

        agent_id 取法：优先信封 `ev.agent_id`（Task 8 已把 `_running_agents` 优先于
        `Task.assigned_agent_id` 填好），信封缺失（存量/边缘事件）才回落 payload 里的
        `assigned_agent_id`。不在 `_agents` 里登记的 agent_id（幽灵/未知）直接忽略——
        与 `template_id_of` 等既有查询方法「不存在就不该往下走」的口径一致，也是
        brief 明确要求的过滤。

        没有再加别的过滤：见 `apply_input` 与本方法上方关于「要不要防语义误用」的
        分析（agent_lifecycle_manager 模块 docstring 之外，写在本任务的报告里）——结论是
        TaskManager 的派发本身保证「同一 agent 同时只挂一个在跑 task」（busy_agents
        校验，task_manager.py `abandon_pending` 等处可见同一不变量），ALM 没有独立
        证据表明还有别的「语义上不该发生」的组合需要在这一层补挡；再加会变成
        replicate TaskManager 的不变量、猜它可能错在哪，属于过度防御。
        """
        inp = self._INPUT_BY_EVENT.get(ev.type)
        if inp is None:
            return
        agent_id = ev.agent_id or (ev.payload or {}).get("assigned_agent_id")
        if not agent_id or agent_id not in self._agents:
            return
        if ev.task_id:
            self._agents[agent_id].current_task_id = ev.task_id
        p = ev.payload or {}
        await self.apply_input(
            agent_id,
            inp,
            task_id=ev.task_id,
            hitl_id=str(p.get("hitl_id", "")),
            reason=str(p.get("reason", "")) or _SETTLE_REASON.get(ev.type, ""),
        )

    async def apply_input(
        self,
        agent_id: str,
        inp: AgentInput,
        *,
        task_id: str | None = None,
        hitl_id: str = "",
        reason: str = "",
        cascaded_from: str | None = None,
    ) -> bool:
        """状态转移与事件发射的**唯一入口**。返回是否真的发生了转移。

        `handle_event` 走它，Phase F 的 cancel/pause/resume（Task 19/20）也走它——
        任何要驱动五态机的调用方都不得绕过本方法直接改 `rec.status` 或直接
        `event_bus.emit`，否则「转移即发事件」这条不变量会被绕开一半。
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            return False
        tr = next_agent_transition(
            rec.status, inp,
            task_id=task_id, hitl_id=hitl_id, reason=reason, cascaded_from=cascaded_from,
        )
        if tr is None:
            return False
        rec.status = tr.status
        await emit_event(
            self.event_bus, tr.event_type,
            session_id=rec.session_id,
            tenant_id=rec.tenant_id,
            origin=_ORIGIN,
            task_id=task_id,
            agent_id=agent_id,
            payload=dict(tr.payload),
        )
        return True

    # ── 登记 ─────────────────────────────────────────────────────────────

    def register_session(
        self, session_id: str, *, tenant_id: str, fallback_template_id: str,
    ) -> None:
        """纳入管理。已存在则保留原状态（重入安全），与 SessionRegistry 同口径。"""
        self._sessions.setdefault(
            session_id, _SessionDefaults(tenant_id=tenant_id, fallback_template_id=fallback_template_id),
        )

    def _drop(self, ids: set[str]) -> int:
        """把这批 agent 从 `_agents` 与 `_children` 索引里摘干净，返回真正摘掉的条数。
        `forget_session` / `forget_agent` 共用——索引清理只写一份。

        `_children` 的**两侧**都要清：既摘掉被移除 agent 自己的键（它的子列表跟着它
        一起消失），也把它们从其它 agent（多半是它们自己的父）的值集合里摘掉。只清一侧
        的话，父的 `children_of` 会指向一个 `_agents` 里已经不存在的 id，
        `descendants_of` 遍历到它时仍会把它当「活着」吐出来——级联 cancel/pause 会去
        操作一个幽灵。这就是「悬垂引用」的具体后果。
        """
        removed = {aid for aid in ids if self._agents.pop(aid, None) is not None}
        for aid in ids:
            self._children.pop(aid, None)
        if removed:
            for children in self._children.values():
                children -= removed
        return len(removed)

    def forget_session(self, session_id: str) -> int:
        """把该 session 的**全部** agent record 逐出内存，返回逐出条数。

        **只由显式逐出调用，不由「会话跑完」触发**（2026-09-08 生命周期改造）。从前
        它挂在 `_fire_session_done` -> `_release_session` 上：任务一跑完 record 就没了，
        而 agent 在概念上只是回到 `idle`（spec 3.1：task 终态不是 agent 终态），
        `send_message` / `list_agents` / `get_agent` 随即全部瞎掉。现在 registry 是
        「只增不删」的缓存。

        **纯机制**：同 `forget_agent`，本方法不判断这条会话该不该被忘掉——那要看 task
        队列、未决 HITL 这些 ALM 不认识的东西。判据在 `CtxWeftRuntime.forget_session`。

        真相源始终是事件日志：逐出之后任何入口都能用 `rebuild_session` /
        `rebuild_agent` 把它装填回来，所以这里删得干净不必手软。
        """
        n = self._drop(set(self.agent_ids_of_session(session_id)))
        self._sessions.pop(session_id, None)
        return n

    def forget_agent(self, agent_id: str) -> bool:
        """逐出**单个** agent record。返回是否真的删了（不在册 → False）。

        `forget_session` 的单点版本。`_children` 的两侧都要清，见 `_drop`。

        **纯机制，不判断该不该逐**：「这个 agent 现在能不能被忘掉」要同时看它的 task
        队列、未决 HITL、以及同 session 其他 agent 的状态——ALM 一样都不认识。判据在
        `CtxWeftRuntime.forget_agent` / `forget_session`（组合根，那里才看得全）。
        本类只负责"摘干净"这一件事。

        `_sessions[session_id]` 的默认值**不动**：同 session 可能还有别的 agent 在册，
        而它是 `materialize()` 回落时要用的 session 语境。整条会话的清理走
        `forget_session`。
        """
        if agent_id not in self._agents:
            return False
        self._drop({agent_id})
        return True

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents

    def template_id_of(self, agent_id: str) -> str:
        return self._agents[agent_id].template_id

    def status_of(self, agent_id: str) -> str:
        """当前五态机状态。未登记的 agent_id 抛 `AgentNotFound`（R18 收口：此前是
        裸 `KeyError`，与 `template_id_of` 同一口径的「按 id 查已知 record 的字段
        历来是裸下标」——但 `assert_can_receive` 现在复用本方法做存在性校验，
        runtime 级 API 需要一个有意义的领域异常类型往外传播，不能让实现细节的
        `KeyError` 漏出去）。全仓核实过：本方法此前零调用点，收口不影响既有控制流。

        没有采用 brief 草案里「未登记 -> 回落 'terminated'」的写法：'terminated'
        是五态机里一个**真实、有意义**的终态（只由外部显式 cancel 触发，见
        agent_state.py 顶部 docstring），把它复用成「查无此 agent」的哨兵值，
        会让调用方没法区分「这个 agent 曾经存在、现在已终止」和「这个 id
        压根没登记过」——前者是合法的终态查询，后者多半是调用方自己算错了 id
        或者查早了（agent 还没 instantiate）。两者背后要做的事不一样：前者可能
        要继续走「已终态，忽略」的分支，后者是编程错误，应该尽早炸出来，而不是
        被误判成「已终止」悄悄放过。
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            raise AgentNotFound(f"unknown agent: {agent_id}")
        return rec.status

    def record_of(self, agent_id: str) -> AgentRecordView | None:
        """该 agent 的只读记录快照；未登记返回 None。

        取代 runtime 对 `_agents` 的私有穿透（2026-09-04 spec §5.3）。返回 None
        而不是抛错：调用方各有各的报错口径（`send_message` 抛 `AgentNotFound`、
        `_task_is_terminal` 静默降级），由它们自己决定。
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            return None
        return AgentRecordView(
            agent_id=agent_id,
            session_id=rec.session_id,
            tenant_id=rec.tenant_id,
            template_id=rec.template_id,
            parent_agent_id=rec.parent_agent_id,
            spawn_depth=rec.spawn_depth,
            status=rec.status,
            current_task_id=rec.current_task_id,
            created_at=rec.created_at,
        )

    def set_current_task(self, agent_id: str, task_id: str | None) -> None:
        """外部消息新建 task 时同步 `current_task_id`——`send_message` 的路由依据。

        运行期这个字段由 `AGENT_*` 事件的折叠维护；`_start_task_for_agent` 是唯一
        「task 还没派发、但路由已经必须认它」的时刻，故留这一个显式写口。
        """
        rec = self._agents.get(agent_id)
        if rec is not None:
            rec.current_task_id = task_id

    def assert_can_receive(self, agent_id: str) -> None:
        """外部消息投递前的同步守卫（spec §3.5 / §4.1）。判断逻辑收敛在此一处，
        不散落到 runtime.py 各个 API 方法里。

        - 不存在 -> `AgentNotFound`（复用 `status_of` 的存在性校验）
        - `terminated`（已被显式 cancel）-> `AgentTerminatedError`
        - `running` -> `AgentBusyError`（忙碌直接拒绝，不排队；调用方自行重试，
          或先 pause/cancel）
        - `idle` / `waiting_human` / `interrupted`（`interrupted` 是可恢复态，
          resume 后能继续处理）-> 放行

        同步方法：投递前的守卫不该引入 await 点。
        """
        status = self.status_of(agent_id)
        if status == "terminated":
            raise AgentTerminatedError(f"agent {agent_id} already terminated")
        if status == "running":
            raise AgentBusyError(
                f"agent {agent_id} is running; retry later or pause/cancel it first"
            )

    def children_of(self, agent_id: str) -> set[str]:
        return set(self._children.get(agent_id, ()))

    def descendants_of(self, agent_id: str) -> list[str]:
        """深度优先展开全部子孙，不含自己。带 seen 集防御索引成环。

        父子关系理论上无环（每个 agent 只在 instantiate 时认一次父），但索引
        一旦因为 bug 损坏成环，宁可返回一份不完整的列表也不要死循环卡死调用方。
        """
        out: list[str] = []
        seen: set[str] = {agent_id}
        stack = list(self._children.get(agent_id, ()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(self._children.get(cur, ()))
        return out

    def agent_ids_of_session(self, session_id: str) -> list[str]:
        return [k for k, r in self._agents.items() if r.session_id == session_id]

    def all_agent_ids(self) -> list[str]:
        """全部已登记 agent 的 id——`list_agents()` 不传 session_id 时的数据源。"""
        return list(self._agents)

    async def load(
        self,
        agent_views: dict[str, AgentView],
        *,
        session_id: str,
        tenant_id: str,
        fallback_template_id: str,
        protected_current_tasks: "set[str] | None" = None,
    ) -> int:
        """恢复期喂入：把 reducer 折出的 `AgentView` 逐条装填进 registry。

        「喂进来，不是查回去」——registry 不订阅 reducer、不订阅总线，装填之后
        只读自己内存，绝不回落 scan 事件（与 `rebuild_hitl` 同一条纪律）。

        `AgentView` 只有 id/spawn_depth/parent_agent_id/template_id 四个字段，
        没有 memory_config/loop_config——这里用 `template_lookup` 重新解析出
        真正的 template 配置（取代 `agents_from_projection` 留下的 dataclass
        默认值，那是本 Task 有意修正的行为）。

        `view.template_id` 为空（存量事件流子 agent 没发过 AgentInstantiated）
        → 回落 `fallback_template_id`；模板解析失败 → `logger.warning` 后用
        `MemoryConfig()`/`LoopConfig()` 默认值继续。**绝不抛**：这条路要在
        恢复路径的每个 session 上都跑通，抛一次就卡住整条恢复链——回落而非
        报错的口径与 `_register_fallback` 一致。唯一的例外是**身份冲突**：某个 id 已被
        另一个 session 的真记录占着 → `DuplicateAgentId`，整批不装（见方法体开头）。

        装填完按每个 `AgentView` 折出来的现状发一条 `AGENT_*`（idle/waiting_human/
        interrupted；`terminated`/`running` 不发，见 `_RECOVERY_BROADCAST_BY_STATUS`
        旁的注释），host 的投影因此不会停在崩溃前的状态。这是广播已成立的事实，
        不是新状态转移，不经 `apply_input`。

        **只对这次调用之前 registry 里还没有的 agent 广播**（见方法体 `cold_ids`
        旁的注释）：`load()` 不止被 `rebuild_session` 的按需装填调用，`_recover_session_locked`
        每次 `/resume`、每次冷 HITL 应答都会重新 load 一遍已经在内存里的 agent——
        那种情形不该每次都把「现状」再回声一遍，真实转移各自已经发过自己的
        `AGENT_*`，重复广播会让下游把「装填回声」误当成「刚发生了一次新转移」。

        ``protected_current_tasks``：这些 task 正处在**未提交窗口**里（spec 2026-09-09），
        它们的 `TASK_CREATED` 还没落盘，因此**折不进 `AgentView`**。此时按 view 覆盖
        `current_task_id` 就会把路由判据倒回上一个已终态的 task——下一条消息于是凭空
        再开一个 task，同一个 agent 挂两个（`test_agent_current_task_fold` 正是为这个
        形状写的回归）。内存里那份才是真的：命中就保住它，不被日志折出来的旧值盖掉。
        """
        # agent_id 全局唯一：一个 id 只属于一个 session。另一个 session 已经登记着同一个
        # id（非占位记录）时，这份日志与内存里的那份**必有一份是错的**——静默覆盖会把那个
        # agent 连同它的状态搬进这个 session，而 CapabilityCache 等按 agent_id 存的运行期
        # 状态也会从此被两个 session 共用。数据完整性问题要响亮：在改动任何记录之前整批
        # 检查、整批拒绝。这是本方法「绝不抛」纪律唯一的例外——那条纪律防的是恢复期的
        # 配置缺口（模板解析失败），不是身份冲突。
        for av in agent_views.values():
            live = self._agents.get(av.id)
            if live is not None and not live.placeholder and live.session_id != session_id:
                raise DuplicateAgentId(
                    f"agent_id {av.id} is registered to session {live.session_id!r}, "
                    f"but the event log of session {session_id!r} claims it too"
                )
        self.register_session(
            session_id, tenant_id=tenant_id, fallback_template_id=fallback_template_id,
        )
        ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id)
        n = 0
        # 广播只认「这次调用之前 registry 里确实还没有这条记录」的 agent——记在覆盖
        # `self._agents[av.id]` 之前，覆盖后 membership 判据就没了。这是真正的**冷**
        # 装填（重启后对这条会话的首次装填）与**热**重装（`_recover_session_locked`
        # 每次 `/resume`、每次冷 HITL 应答都会重新调一次 `load()`，把已经在内存里的
        # record 幂等覆盖一遍，这不是本 spec 要补的缺口）之间唯一站得住脚的分界。
        #
        # 不分界、每次 `load()` 都无条件广播的后果是真实的、已经复现过：
        # `tests/integration/test_outage_resume.py::test_idempotent_outage_resume_completes`
        # 的 `_wait_for_interrupted_event` 把「看到一条新 AgentInterrupted」当成
        # 「刚发生了一次新的中断」，若 `recover_agent()` 每次都在真实转移之外
        # 额外回声一遍现状，这个假设就被推翻——测试会在上一轮 drain 还没跑完时就
        # 发起下一轮 `recover_agent()`，两个 TaskManager 竞争同一个 session，
        # 任务卡死在 INTERRUPTED 永不完成（不是这个测试脆弱，是这类消费者对「事件
        # 到达 = 新事实发生」的假设本就合理，广播不该在没有新事实时重复兑现它）。
        cold_ids = {av.id for av in agent_views.values() if av.id not in self._agents}
        for av in agent_views.values():
            template_id = av.template_id or fallback_template_id
            try:
                template = await self.template_lookup.get_template(template_id, None, ctx=ctx)
                resolved_template_id = template.id
                memory_config = template.memory_config
                loop_config = template.loop_config
            except Exception:
                logger.warning(
                    "AgentLifecycleManager.load: template %r unresolvable for agent %s; "
                    "using default configs (recovery-time gap, degrading not crashing)",
                    template_id, av.id,
                )
                resolved_template_id = template_id
                memory_config = MemoryConfig()
                loop_config = LoopConfig()
            # 未提交窗口里的路由判据不得被日志折出来的旧值倒回（见 docstring）。
            current_task_id = av.current_task_id
            if protected_current_tasks:
                live = self._agents.get(av.id)
                if (live is not None and live.current_task_id
                        and live.current_task_id in protected_current_tasks):
                    current_task_id = live.current_task_id
            self._agents[av.id] = _AgentRecord(
                session_id=session_id,
                tenant_id=tenant_id,
                template_id=resolved_template_id,
                parent_agent_id=av.parent_agent_id,
                spawn_depth=av.spawn_depth,
                memory_config=memory_config,
                loop_config=loop_config,
                # D1 修复：模型选择从 AgentView 读回，跨重启存活——不再是
                # ModelChoice() 默认值（那是修复前 recover_agent 静默降级的根因）。
                llm=ModelChoice(account=av.llm_account, model=av.llm_model),
                # Task 14：五态机状态从 AgentView 读回，跨重启存活——不这样做的话
                # 冷恢复后每个 agent 都会被 `_AgentRecord.status` 的字段默认值
                # 重置成 idle，即使它重启前正处于 waiting_human（有未决 HITL 挂着），
                # 导致 `assert_can_receive` 错误放行、`send_message` 的路由判断失准。
                status=av.status,
                current_task_id=current_task_id,
            )
            if av.parent_agent_id is not None:
                # 重建 _children——冷恢复必须让级联 cancel/pause（Task 19/20）在
                # 重启后依旧可用，不能只在 instantiate() 这条热路径上维护索引。
                # 与 instantiate() 同一条件（parent 非 None 就落边），不额外要求
                # 父已经在本次 agent_views 批次里出现：`set.add` 天然幂等，
                # `load()` 被多次调用或与既有索引共存时不会重复计数、也不会覆盖；
                # 父若属于尚未 load 的另一个 session，边先落在这里，等那个 session
                # 也 load() 完，_children[parent] 已经是齐的，不依赖跨 session 的
                # 加载顺序。父若是彻底的幽灵（事件流损坏、永远不会被登记）——
                # 边挂在一个未注册的 id 下，无害：没有已知调用路径会拿一个未
                # `has()` 通过的 id 去发起级联遍历，`children_of`/`descendants_of`
                # 该 id 之外的查询结果不受影响；`forget_session` 逐出这个子
                # agent 所在的 session 时，会把它从这条边的值集合里摘掉（见
                # forget_session 对 `self._children.values()` 的全量清理），
                # 不会留下指向「已被移除的 agent」的悬垂值。
                self._children.setdefault(av.parent_agent_id, set()).add(av.id)
            n += 1

        # 恢复期现状广播（2026-09-04 spec §6.4）：装填完把折出来的状态照实发一遍，
        # 取代此前由 runtime 代 TaskManager 发的队列聚合信号（那条信号的消费者
        # SessionRegistry 早已不订阅它）。只对上面记下的 `cold_ids` 发——见那段注释。
        #
        # 「恢复不是一种状态」：这里发的是**现状**不是新状态，不引入 RECOVERING
        # 之类的值域，也不经状态机——`apply_input` 是给真实转移用的，这里是把已经
        # 成立的事实广播出去，走同一个发射点（ALM 是 AGENT_* 的唯一发射者）。
        #
        # `terminated` 不发：它是粘滞终态，host 投影里本就已经是终态，重发没有信息。
        for av in agent_views.values():
            if av.id not in cold_ids:
                continue
            event_type = self._RECOVERY_BROADCAST_BY_STATUS.get(av.status)
            if event_type is None:
                continue
            await emit_event(
                self.event_bus, event_type,
                session_id=session_id, tenant_id=tenant_id,
                origin=_ORIGIN, task_id=av.current_task_id, agent_id=av.id,
                payload={"reason": "recovered"},
            )

        return n

    async def instantiate(
        self,
        *,
        template_id: str,
        session_id: str,
        tenant_id: str,
        parent_agent_id: str | None = None,
        task_id: str | None = None,
        agent_id: str | None = None,
        template: AgentTemplate | None = None,
        ctx: ProviderContext | None = None,
        llm: ModelChoice | None = None,
    ) -> tuple[Agent, AgentTemplate]:
        """真新建：解析 template（或用调用方预解析的），生成新 id（或用调用方预铸的），
        登记 record，发出身事件。

        template_id 须为规范形式 provider:name；裸 id 由 TemplateLookup 抛 TemplateNotFoundError。
        深度超限发 SpawnRejected 并抛 SpawnDepthExceeded。

        ★ 无 existing_agent_id 参数——水合走 materialize()，两件事不再共用一个入口。

        llm：这个 agent 的 `(account, model)` 选择。省略（None）→ 继承**派生它的那个
        agent**（`parent_agent_id` 的 record.llm）；无父（root）则 `ModelChoice()`——
        跟随账号默认。是 root 时也允许显式传，不必是空。

        task_id：AgentSpawned / SpawnRejected 的 envelope 需要——两条事件的主语都是
        「围绕这次 spawn 尝试」，task_id 标的是被 spawn 出来要跑的那个子任务。root
        agent 实例化没有 task（session 尚未建 root task），传 None 即可。

        agent_id：调用方预先铸好的新 agent id，省略则内部照旧 generate_id("agt")。
        目前只有 SessionRegistry.create_session 的 root 分支会传——它得先知道 id
        才能把 root_agent_id 塞进 SESSION_CREATED payload，而 SESSION_CREATED 必须
        先于这里发出的 AgentInstantiated（因果序 Session → Agent → Task）。传入的
        id 若已登记过 → DuplicateAgentId：见该异常 docstring，这不是「水合」。

        template：调用方已经解析过的 template 对象，传了就直接用，不再自己
        `get_template`。目前只有 SessionRegistry.create_session 会传——它得先解析
        一次校验 template_id（入口即拒、不落库，必须在任何 emit 之前完成），若这
        里再解析第二次，`LocalAgentTemplateProvider` 每次都重新扫盘（docstring
        明写「热更新友好」），两次调用之间存在真实 TOCTOU 窗口：SESSION_CREATED
        已经落库后，第二次解析可能拿到热更新后的不同版本甚至 TemplateNotFoundError，
        击穿「入口即拒、不落库」。传预解析对象消灭这个窗口——全程只解析一次。
        省略则照旧自己解析（子 agent 路径不受影响，它没有相同的两阶段需求）。
        """
        # 先定 id、再判重：调用方预铸的与这里现铸的一视同仁（ULID 撞车在实践中不会
        # 发生，但「agent_id 唯一」是不变量，不该靠概率成立）。判重先于模板解析与任何
        # emit——撞上了不留半个 agent。
        agent_id = agent_id or generate_id("agt")
        if agent_id in self._agents:
            raise DuplicateAgentId(f"agent_id {agent_id} already registered")
        if template is None:
            resolve_ctx = ctx or ProviderContext(session_id=session_id, tenant_id=tenant_id)
            template = await self.template_lookup.get_template(
                template_id, None, ctx=resolve_ctx,
            )

        spawn_depth = 0
        if parent_agent_id is not None:
            # .get() + 回落而非裸下标：父 agent 若因跨重启未被本进程重新登记
            # （旧调用方直接递 Agent 对象、绕过了 registry），裸下标会把一次
            # 「登记缺口」升级成一次崩溃——与 materialize 的既有口径一致。这里手上
            # 就有本次 instantiate 调用自己的 session_id/tenant_id——子 agent 的父
            # 几乎必然在同一 session，传进去比 `_register_fallback` 内部「猜最近一次
            # register_session 的会话」精确，消掉多 session 并发恢复时猜错 tenant
            # 的那类风险（Task 3 评审 deferred minor，Task 5 顺手收紧这一个调用点）。
            parent_rec = self._agents.get(parent_agent_id) or self._register_fallback(
                parent_agent_id, session_id=session_id, tenant_id=tenant_id,
            )
            spawn_depth = parent_rec.spawn_depth + 1
            if spawn_depth > template.loop_config.max_spawn_depth:
                # SpawnRejected 是 AgentSpawned 的另一半：被拒的 spawn 根本不会有
                # agent 诞生，AgentInstantiated 覆盖不到，不发这条则事件流里看不出
                # 「有人想 spawn 但被挡了」。envelope 的 agent_id 填**父**——子 agent
                # 没诞生，没有 id 可填，而这条事件的主语正是发起 spawn 的那个 agent。
                await emit_event(
                    self.event_bus, EventType.SPAWN_REJECTED,
                    session_id=session_id,
                    tenant_id=tenant_id,
                    origin=_ORIGIN,
                    task_id=task_id,
                    agent_id=parent_agent_id or None,
                    payload={
                    "reason": "depth_limit",
                    # 恒 False：spawn 被拒后降级为 inline 执行的能力今天不存在，
                    # 如实反映现状而不是留一个骗人的 True。
                    "fallback_to_inline": False,
                    "attempted_subtask_id": task_id,
                    },
                )
                raise SpawnDepthExceeded(
                    f"Max spawn depth {template.loop_config.max_spawn_depth} exceeded "
                    f"(current: {spawn_depth})"
                )

        if llm is None:
            # 继承**派生它的那个 agent**（parent_rec，上面 depth 检查已解出），不是 root。
            llm = parent_rec.llm if parent_agent_id is not None else ModelChoice()

        self._agents[agent_id] = _AgentRecord(
            session_id=session_id,
            tenant_id=tenant_id,
            template_id=template.id,
            parent_agent_id=parent_agent_id,
            spawn_depth=spawn_depth,
            memory_config=template.memory_config,
            loop_config=template.loop_config,
            llm=llm,
            created_at=now_utc(),
        )
        if parent_agent_id is not None:
            self._children.setdefault(parent_agent_id, set()).add(agent_id)

        logger.info(
            "AgentLifecycleManager: instantiated agent %s (template=%s, depth=%d)",
            agent_id, template_id, spawn_depth,
        )
        # 不走 materialize()/resolve_model()：这里刻意不碰模型解析——instantiate 是
        # 会话/子 agent 创建路径，调用方（如 SessionRegistry.create_session）此刻常常
        # 还没有真正要用的窗口、也不该为了造一个返回值就触发 LLM 解析（惰性不变量，
        # 见上面 _DEFAULT_CONTEXT_LIMIT 的注释）。真实窗口留给派发时的 materialize()。
        agent = Agent(
            id=agent_id,
            session_id=session_id,
            tenant_id=tenant_id,
            template_id=template.id,
            parent_agent_id=parent_agent_id,
            spawn_depth=spawn_depth,
            memory_config=template.memory_config,
            loop_config=template.loop_config,
            loop_guard=LoopGuard(
                context_limit=_DEFAULT_CONTEXT_LIMIT,
                reserved_output_tokens=_DEFAULT_RESERVED_OUTPUT_TOKENS,
            ),
            created_at=now_utc(),
        )

        if parent_agent_id is not None:
            # 先记「这次 spawn 被准了」，再记「诞生的 agent 长这样」——读事件流的
            # 因果顺序。AgentSpawned 的主语是**父 agent 的一次 spawn 动作**（与
            # SpawnRejected 配对，构成对每次 spawn 尝试的完整审计）；下面那条的
            # 主语是这个 agent 自己的出身配置。
            await emit_event(
                self.event_bus, EventType.AGENT_SPAWNED,
                session_id=session_id,
                tenant_id=tenant_id,
                origin=_ORIGIN,
                task_id=task_id,
                agent_id=agent_id,
                payload={
                "parent_agent_id": parent_agent_id,
                "subtask_id": task_id,
                },
            )
        # 事件流里唯一记录「该 agent 用的哪个模板」的地方——`_rebuild_agents` 从
        # session/task 树推算 AgentView，推得出 parent/depth，推不出模板。root 与
        # 子 agent 现在共用同一条发射路径，不再分落 session_registry 与 runtime 两处。
        await emit_event(
            self.event_bus, EventType.AGENT_INSTANTIATED,
            session_id=session_id,
            tenant_id=tenant_id,
            origin=_ORIGIN,
            task_id=task_id,
            agent_id=agent_id,
            payload={
            "template_id": template.id,
            "template_version": template.version,
            "llm_account": llm.account,
            "llm_model": llm.model,
            },
        )

        return agent, template

    async def set_agent_llm(
        self, agent_id: str, choice: ModelChoice, *,
        reason: str, causation_id: str | None = None,
    ) -> bool:
        """纯赋值：改这一个 agent 的模型选择，发一条 `AGENT_LLM_CHANGED`。

        不入队、不改任何 task 状态、不触发调度——「换模型」和「让 task 跑起来」
        是两件事（spec §06 的三条命令）。未登记的 agent_id、或选择与现值相同
        → no-op，不发事件（host 重复点击不刷屏，也不会对幽灵 id 广播事实）。
        """
        rec = self._agents.get(agent_id)
        if rec is None or rec.llm == choice:
            return False
        rec.llm = choice
        await emit_event(
            self.event_bus, EventType.AGENT_LLM_CHANGED,
            session_id=rec.session_id,
            tenant_id=rec.tenant_id,
            origin=_ORIGIN,
            agent_id=agent_id,
            payload={
            "llm_account": choice.account,
            "llm_model": choice.model,
            "reason": reason,
            },
            causation_id=causation_id,
        )
        return True

    async def set_session_llm(self, session_id: str, choice: ModelChoice, *, reason: str) -> int:
        """作用于该 session 下 registry 持有的**全部** record。

        不去问 TaskManager「哪些还会被派发」——零查询依赖是 Registry 的设计属性。
        已跑完的 agent 改了也无害（不会再被派发），代价只是多几条事件。

        发 N 条 `AGENT_LLM_CHANGED`，不是一条会话级事件：真相源因此仍然唯一，
        reducer 不必处理「一条事件改 N 个实体」。N 条事件共享一个 causation_id，
        host 要展示「这是一次会话级切换」→ 按它聚合，不需要第三种事件类型。
        """
        cid = generate_id("cau")
        ids = self.agent_ids_of_session(session_id)
        n = 0
        for aid in ids:
            if await self.set_agent_llm(aid, choice, reason=reason, causation_id=cid):
                n += 1
        return n

    def resolve_model(self, agent_id: str) -> ResolvedModel:
        """现解，不缓存：client/身份/窗口是同一次解析的三面，一次算出、当次即弃。

        缓存 client 是 `LLMClientResolver` 的职责——Registry 再存一份就有第二个
        缓存和它自己的失效问题（host 换了账号凭据，陈旧 client 继续被用）。
        """
        choice = self._agents[agent_id].llm
        client = self.model_resolver(choice.account, choice.model)
        return ResolvedModel(
            client=client,
            account=choice.account or getattr(client, "account", ""),
            model=choice.model or getattr(client, "model", ""),
            context_limit=client.context_limit,
            reserved_output_tokens=client.output_reserve,
        )

    def materialize(self, agent_id: str) -> tuple[Agent, ResolvedModel]:
        """水合：按 id 从 record 造一个新的 Agent 对象，顺带解出这次要用的模型。零事件，永不抛。

        未登记的 id（恢复期缺口——比如跨重启后本进程的 registry 是空的）走「按
        session 的 fallback_template_id 就地补登记 + WARNING」而不是 KeyError：
        回落而非报错是刻意的，与 `load()` 装填时模板解析失败的口径一致——
        授权按模板做策略，重启后把未知模板判成「无权限」会让老会话直接跑不动，
        把恢复期的一个缺口变成崩溃是净损失。

        每次调用产出一个新实例（不是共享引用）：Agent 带一次 run 的可变量
        （loop_guard.context_tokens 由 act.py 改写），派发时各自持有自己的份是对的。

        窗口不再由调用方传入——`LLMClient` 协议本就把 context_limit / output_reserve
        定义成抽象属性（protocols/llm.py），永远从这次解出的 client 现读，没有
        「换模型后对齐窗口」这个动作要做，因为窗口从没存过、一直跟着 client 走。
        """
        rec = self._agents.get(agent_id)
        if rec is None:
            rec = self._register_fallback(agent_id)
        rm = self.resolve_model(agent_id)
        agent = Agent(
            id=agent_id,
            session_id=rec.session_id,
            tenant_id=rec.tenant_id,
            template_id=rec.template_id,
            parent_agent_id=rec.parent_agent_id,
            spawn_depth=rec.spawn_depth,
            memory_config=rec.memory_config,
            loop_config=rec.loop_config,
            loop_guard=LoopGuard(
                context_limit=rm.context_limit,
                reserved_output_tokens=rm.reserved_output_tokens,
            ),
            created_at=now_utc(),
        )
        return agent, rm

    def _register_fallback(
        self, agent_id: str, *, session_id: str | None = None, tenant_id: str | None = None,
    ) -> _AgentRecord:
        """未登记 id 的一次性补登记（materialize 的降级路径，instantiate 的父查找也借用它）。

        本方法**不**代表「除 instantiate 外还有人改已存在的 record」——它只在
        record 从未存在过时创建一条，且创建后立刻幂等（第二次直接命中 `_agents.get`，
        不再触发警告），不会覆盖任何已登记的真实状态。

        `session_id`/`tenant_id`：调用方若手上已经有确凿的 session 语境（比如
        instantiate 的父查找——子 agent 的父几乎必然在同一次 instantiate 调用
        的 session 里），传进来直接用，不猜。省略时才回落「最近一次
        register_session 的会话」这个近似——materialize() 签名里没有 session_id
        参数，是那条路径专属的容忍度（恢复期的一次缺口容忍度本就高于严格授权
        路径），不是本方法本身必须用猜的。
        """
        logger.warning(
            "AgentLifecycleManager: unregistered agent %s; "
            "falling back to a session default (recovery-time gap, degrading not crashing)",
            agent_id,
        )
        if session_id is not None and session_id in self._sessions:
            _sid, defaults = session_id, self._sessions[session_id]
        elif session_id is not None:
            _sid = session_id
            defaults = _SessionDefaults(
                tenant_id=tenant_id or "default", fallback_template_id="",
            )
        elif self._sessions:
            _sid, defaults = next(reversed(self._sessions.items()))
        else:
            defaults = _SessionDefaults(tenant_id="default", fallback_template_id="")
            _sid = ""
        rec = _AgentRecord(
            session_id=_sid,
            tenant_id=defaults.tenant_id,
            template_id=defaults.fallback_template_id,
            parent_agent_id=None,
            spawn_depth=0,
            memory_config=MemoryConfig(),
            loop_config=LoopConfig(),
            placeholder=True,
        )
        self._agents[agent_id] = rec
        return rec
