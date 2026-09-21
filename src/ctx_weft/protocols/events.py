"""Event 领域的 host-facing 契约：事件数据类型 + EventBus + EventStore。

划界判据（spec 2026-08-27-protocols-layer-event-contracts-design §2，
用户 2026-08-28 裁定改为三层）：
**契约进 `protocols/`，实现进 `providers/`，`core/` 只留编排。**

故本模块装：`Event` / `EventFilter` / `EventType` 与两个常量集（host 要构造事件、
要持久化、要按类型分派）、`EventBus` 协议（README 明说 host 可换 Redis Streams）、
`EventStore` 协议（必需面：append / append_batch / read_range / committed_head /
read_last_of_type / read_session_events_of_types——快照不在其中，它是一种事件）。

**不装**：`InProcessEventBus` / `InMemoryEventStore`（内置实现，在
`providers/events/bus/in_process/bus.py` / `providers/events/store/in_memory/store.py`）、
`TASK_STATUS_BY_EVENT`（core 的投影逻辑，且依赖 core 的 `TaskStatus`）。

⚠️ **本模块不得 import `ctx_weft.core` 的任何东西。** protocols 是比 core 低的层；
反向依赖会让 `protocols/context.py` 那个刻意的惰性绑定失去意义，并在某些 import
顺序下变成真实的循环导入。`tests/unit/test_protocols_events_relocation.py` 钉住这条。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # 仅类型注解用；`protocols/events.py` 运行时只依赖 stdlib 的现状不变
    # （同层 import 不违反层序 ast 守卫，但保持现状更稳，见文件顶部说明）。
    from ctx_weft.protocols.context import ProviderContext


@dataclass
class Event:
    """事件基类。所有事件共享此结构；payload 字段按事件类型而异（详见 §9.4-§9.6）。"""

    id: str  # evt_ULID
    run_id: str | None  # 一次 loop run 的标识；某些 session 级事件可为 None
    sequence: int  # 在同一 run_id 内单调递增
    session_id: str
    type: str  # 见 EVENT_TYPES
    timestamp: datetime
    tenant_id: str = "default"
    task_id: str | None = None
    agent_id: str | None = None
    origin: str = ""  # V2 新增：哪个组件发出的，见 docs/events-v2.md §4。存量事件读出空串
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    causation_id: str | None = None
    schema_version: int = 1  # payload 版本；reducer 据此分支


@dataclass
class EventFilter:
    """订阅过滤器。"""

    session_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    # agent 维度：信封的 agent_id 是 agent-centric 下 host 最常用的订阅轴
    # （只渲染某一个 agent 的事件流）。事件 agent_id 为 None 时不匹配任何
    # 具体 agent_id——「没有归属」不等于「属于你要的那个」。
    agent_id: str | None = None
    types: list[str] | None = None  # None=全部


# ── V1 冻结的事件类型清单（详见 §9.3）────────────────────────────────────────────


class EventType(StrEnum):
    """V1 冻结的事件类型枚举。

    继承 StrEnum：成员即字符串（`EventType.TASK_CREATED == "TaskCreated"`、
    `json.dumps` / `str()` 都得到 `"TaskCreated"`），因此对所有按字符串比较 /
    序列化的消费端完全向后兼容。业务代码应优先引用枚举成员而非字面量。
    """

    # ── Session / Run / Step 域 ──
    SESSION_CREATED = "SessionCreated"
    SESSION_RESUMED = "SessionResumed"     # recover_agent 续跑被打断的 session 时发
    SESSION_STATUS_CHANGED = "SessionStatusChanged"
    SESSION_FINISHED = "SessionFinished"   # TaskManager 确定 session 真正结束时发（含 final_status）
    SESSION_PAUSED_HITL = "SessionPausedHitl"
    # 一轮对话的两个结局信号（spec 2026-09-09）。**会话级、不带 task_id、不含正文**：
    # 它们要绕过未提交窗口（那道闸按 task_id 定）才能到达 host——host 正是靠它们决定
    # 那条攒着的用户消息帧是 flush 还是丢弃。
    #
    # 为什么需要显式的两条，而不是让 host「看到 TaskStarted 就 flush」：后者是隐式契约，
    # 加一个新事件类型、或哪天窗口里事件的顺序变了，它就会静默失准。
    ROUND_COMMITTED = "RoundCommitted"      # payload: {task_id}
    # payload: {task_id, reason} —— reason 目前恒为 discarded_before_first_chunk
    ROUND_DISCARDED = "RoundDiscarded"
    RUN_STARTED = "RunStarted"
    RUN_CANCELED = "RunCanceled"
    RUN_FINISHED = "RunFinished"
    STEP_STARTED = "StepStarted"
    STEP_COMPLETED = "StepCompleted"
    STEP_FAILED = "StepFailed"
    # ── Task 域 ──
    TASK_CREATED = "TaskCreated"
    TASK_STARTED = "TaskStarted"
    TASK_SUSPENDED = "TaskSuspended"
    # ── task/run 层「为什么停」（2026-09-02 所有权重构）──
    # 从前三件事都压在 TASK_SUSPENDED 的 reason 字面量里，消费方只能匹配字符串。
    # 现在各有类型：TASK_SUSPENDED（等子任务）/ TASK_AWAITING_HUMAN（等人）/
    # TASK_INTERRUPTED（被外部打断）。判据是类型，不是 payload。
    TASK_AWAITING_HUMAN = "TaskAwaitingHuman"   # payload: {hitl_id}
    # task 域：这个 task 停在 INTERRUPTED，等 /resume。**发在重试判定之后**——崩溃后
    # 还能原地重试的那一支发的是 TASK_REQUEUED，不是这条（docs/events-v2.md §2.3）。
    # payload: {reason, error_code?, error_message?, retry_count}
    TASK_INTERRUPTED = "TaskInterrupted"
    # run 域：这次执行被外部原因打断了。**只由 _run_loop 发**（run 域的四条事实同源），
    # 且不写 task 状态——task 停在哪由 TASK_INTERRUPTED 说（docs/events-v2.md §2.4）。
    RUN_INTERRUPTED = "RunInterrupted"          # payload: {reason, error_code?, error_message?}
    TASK_RESUMED = "TaskResumed"
    # task 域：TaskAwaitingHuman{hitl_id} 的配对解除事件——「人已经答复/放行，这个 task
    # 不再等人了」。同一个 hitl_id 把被挡住的区间括起来（D4）。**不复用 TaskRequeued**：
    # 后者是「这个 task 要重跑一遍」（retry / observer 重排），语义不同（docs/events-v2.md §2.3）。
    # → PENDING，且清旧产出（与 TaskRequeued 效果相同，但类型不同）。
    TASK_HUMAN_RESOLVED = "TaskHumanResolved"   # payload: {hitl_id}
    TASK_FINISHED = "TaskFinished"
    TASK_FAILED = "TaskFailed"
    TASK_CANCELED = "TaskCanceled"
    TASK_FINALIZED = "TaskFinalized"
    TASK_REQUEUED = "TaskRequeued"
    # task 域：一条用户消息被并进了这个 task 的对话（`send_message` 的注入分支）。**只记
    # 事实、不改状态**——它存在的唯一理由是让这条消息的正文在事件日志里有一份：消息写进
    # memory 要等这一轮提交（暂存区在事件补投**之后**才落盘），崩在两者之间时，恢复期据此
    # 把消息补回 memory（`Runtime._restore_appended_messages`）。
    # payload: {memory_id, agent_id, content（event 侧形态，图走 event blob ref）, source, timestamp}
    TASK_MESSAGE_APPENDED = "TaskMessageAppended"
    BLACKBOARD_PUBLISHED = "BlackboardPublished"
    # ── Agent 域 ──
    AGENT_INSTANTIATED = "AgentInstantiated"
    AGENT_SPAWNED = "AgentSpawned"
    # agent 的模型选择变了（D1 修复：跨重启存活）。纯赋值，不碰 task / session 状态——
    # 「换模型」和「让 task 跑起来」是两件事（docs/events-v2.md 三条命令，见 spec §06）。
    AGENT_LLM_CHANGED = "AgentLlmChanged"   # payload: {llm_account, llm_model, reason}
    # agent 生命周期状态（spec 3.3）。ALM 是唯一发射者。
    AGENT_RUNNING = "AgentRunning"
    AGENT_IDLE = "AgentIdle"
    AGENT_WAITING_HUMAN = "AgentWaitingHuman"
    AGENT_INTERRUPTED = "AgentInterrupted"
    AGENT_TERMINATED = "AgentTerminated"
    SPAWN_REJECTED = "SpawnRejected"
    # ── Context 域 ──
    PREPARE_COMPLETED = "PrepareCompleted"
    CONTEXT_TOKENS_ESTIMATED = "ContextTokensEstimated"
    CONTEXT_ASSEMBLED = "ContextAssembled"
    # ── LLM 域 ──
    LLM_REQUEST_STARTED = "LLMRequestStarted"
    LLM_PROMPT_SENT = "LLMPromptSent"           # 完整 prompt（system + messages + tools），供调试
    LLM_TOKEN_STREAMED = "LLMTokenStreamed"
    LLM_REASONING_STREAMED = "LLMReasoningStreamed"    # extended thinking delta
    LLM_RESPONSE_FINISHED = "LLMResponseFinished"
    LLM_RETRY_TRIGGERED = "LLMRetryTriggered"
    # ── Capability 域 ──
    CAPABILITY_INVOKED = "CapabilityInvoked"
    CAPABILITY_PROGRESS = "CapabilityProgress"
    CAPABILITY_FINISHED = "CapabilityFinished"
    # ── ActStep / ObserveStep 子事件 ──
    ACT_TURN_STARTED = "ActTurnStarted"
    ACT_TURN_COMPLETED = "ActTurnCompleted"
    MAX_TURNS_REACHED = "MaxTurnsReached"
    OBSERVE_STARTED = "ObserveStarted"                 # observe 起点（与 Completed 成对）
    OBSERVE_COMPLETED = "ObserveCompleted"
    # ── Memory 域 ──
    MEMORY_INGESTED = "MemoryIngested"
    MEMORY_COMPACT_STARTED = "MemoryCompactStarted"
    MEMORY_COMPACTED = "MemoryCompacted"
    MEMORY_COMPACT_FINISHED = "MemoryCompactFinished"   # 一轮压缩收尾聚合（总折叠数/省 token/各级），供前端落一条持久标记
    # ── HITL 域 ──
    HITL_REQUIRED = "HitlRequired"
    HITL_APPROVED = "HitlApproved"       # approval kind 放行（无改参）
    HITL_ANSWERED = "HitlAnswered"       # input kind 取得人类文字答复
    HITL_REJECTED = "HitlRejected"
    HITL_MODIFIED = "HitlModified"       # approval kind 放行（带改参）
    HITL_CANCELLED = "HitlCancelled"   # session 关闭 / interrupt / GC：收口悬挂 pending，不 requeue
    # ── HITL v2（2026-09-01 重设计）──
    # outcome 是事实本身，不再由事件类型编码结局：approved vs modified 由
    # payload 有无 modified_arguments 推出，其余由 outcome 推出。host 自定义
    # outcome 因此无需新增事件类型。上方 6 个 legacy HITL 事件在段 3 才退役。
    # 一次已收下的答复被收回了（spec 2026-09-09）：这一轮在 LLM 开口之前被撤销，
    # 那条答复当作没说过，气泡回到未决。
    #
    # **不含正文**——被撤回的那句话不该留在日志里，而这条事件也不需要它：折叠只用它
    # 把气泡放回 pending、并数出「这条气泡被撤过几次」。后者是 memory 幂等键的第二维
    # （见 `HitlRegistry.reply_memory_id`）：撤销后重答会写一条新记录，而上一条已被
    # `fold` 成 superseded、**仍占着旧键**，键不带这一维就会被静默吞掉。
    #
    # 那个计数**必须从日志折出来**，不能是内存计数器：撤销之后重启，内存里什么都没有，
    # 键必然撞回去。这条事件的存在就是为了让它可还原。
    HITL_REPLY_RETRACTED = "HitlReplyRetracted"   # payload: {hitl_id}
    # 这条请求**已了结**：它的决定已经被消费掉，再也不会被问。
    #
    # **「已终局」不等于「已了结」。** `HitlResolved` 只说人答了；系统把那个答复/批准用掉
    # 是**之后**的一步，而两者之间有个真实的崩溃窗口。留着那些已终局记录就是为了补它：
    # UserTurn 要补「把答复注入进对话」（不补，人说的那句话静默消失），工具审核要补「这次
    # 被门控的调用还没跑完」（不补，冷重入会把同一个工具重新求批一遍）。代价是折叠只能留下
    # 该会话**全部**已终局请求——交互式会话里每条用户消息都是一次 UserTurn HITL、每次受管
    # 工具调用又是一次审核，那个集合随会话线性增长，而 `rebuild_hitl` 每次冷应答重折一遍。
    # 这条事件把两者分开，于是集合只剩「答了但还没用掉」的那几条：自己销账、有界。
    #
    # **一条规则贯穿所有形态：谁消费了这条决定，谁在持久效果落地之后盖章。**
    # 四个发射点（UserTurn 的答复注入、工具跑完、`ask_user` 出结果、未授权出口）都遵循它。
    # 这不是风格，是那两个崩溃方向的正确性来源：
    #
    #   崩在持久效果之前   → 没有本事件 → 记录留着 → 下次补 ✓
    #   崩在持久效果之后、本事件之前 → 同样没有本事件 → 下次重做，而那些写入都按确定性 id
    #                       幂等（`hitlreply:{hitl_id}` / `tool_result_record_id`），不会写重 ✓
    #
    # 失败方向恒朝「多留一条、多做一次幂等写」，不朝「少救一条答复」。
    #
    # **为什么不从 capability 事件推。** 那是这条事件的前身设计，两种形态各推一套：工具那半
    # 从 `CapabilityFinished` + `truncated` + 「结果是不是人的答复」三个条件推。三处都不成立——
    # ① `CapabilityInvoked` 不行：冷重入时 `facts.invoked` 已为真却**还会**再走一次授权
    #    （gateway :507 的重跑分支之后就是 :524 的授权步），而决定缓存第四维
    #    `invocation_key = tool_name + sha256(原始参数)` 跨重启逐字节相同、**会**命中；
    # ② `truncated` 只影响事件载荷那份拷贝，memory 里是完整的——按事件推才需要关心它；
    # ③ `ask_user` 的 `CapabilityFinished` 与 `HitlResolved` 同一轮提交里先后落盘，崩在中间
    #    就是「日志里有结果、问题还悬着」，所以 reconcile 刻意不信那个通道。
    # 而且未授权出口（`_error_and_record`）根本不发任何 capability 事件，那条拒绝的决定永远
    # 等不到销账。统一由消费方盖章之后，①②③ 与那个真空一起消失，也不必再维护「退役条件必须
    # 和 gateway 的重放短路同源」这种没人会记得的不变式。
    #
    # **不含正文**：该落的都落了，本事件只是一枚章。
    HITL_CLOSED = "HitlClosed"                    # payload: {hitl_id}
    HITL_OPENED = "HitlOpened"
    HITL_RESOLVED = "HitlResolved"
    # ── 状态快照 ──
    # 一张状态快照**就是一条事件**，载荷里直接带 blob（裁剪后实测 2.2 KB——
    # `prune_view_for_snapshot` 只留活闭包，946 KB → 2.2 KB，436×）。
    #
    # 换来两件事：
    #
    #   · **「哪张最新」= position 最大。** 从前要靠协议规定一套快照专属口径
    #     （`snapshot_at` 最大、相同则 `id` 最大，因为写入序 ≠ 时间序）。现在和其它一切
    #     共用同一个序，不存在第二种「最新」，也就不存在两种口径不一致。
    #   · **那三个「必须原样往返」的字段进了 payload**（切面位置 / `projection_version` /
    #     `chain_depth`），而 payload 对 store 是**整体**不透明的——它没法只丢其中一个。
    #     从前它们是列，store 可能忘了存：m020 补的就是那几列，而丢了它们
    #     `snapshot_is_usable` 恒判不可用，恢复永远全量回放且**不报错**。
    #
    # ⚠️ 切面位置仍要**显式**写进 payload，不能用「这条事件自己的 position 减一」——并发追加
    # 时快照事件拿到的是 `head+k`，k 不定。
    #
    # ⚠️ **store 对这个类型不做任何特殊处理**：它是 core 的词表，不是存储契约的一部分。
    # 需要「排除快照」的地方（全量重放）由 core 把类型传给 `read_range(exclude_types=...)` /
    # `read_range(exclude_types=...)`，不写死在协议里。
    #
    # ⚠️ **不经 EventBus，由写入侧直接 `append_batch`。** 三个理由，第三个是决定性的：它不是
    # 任何人该反应的领域事实；走 bus 会投给 SSE 转译 / 投影更新器 / 分析订阅者，那些都没理由
    # 看见它；而**写入者自己就是 bus 订阅者**，走 bus 会递归调回它自己。
    STATE_SNAPSHOT = "StateSnapshot"
    # ── Guard 域 ──
    FAILURE_THRESHOLD_HIT = "FailureThresholdHit"
    # ── RecognizeIntent 域 ──
    RECOGNIZE_INTENT_STARTED = "RecognizeIntentStarted"
    RECOGNIZE_INTENT_LLM_PROMPT = "RecognizeIntentLLMPrompt"
    RECOGNIZE_INTENT_COMPLETED = "RecognizeIntentCompleted"
    RECOGNIZE_INTENT_TOOL_CALL = "RecognizeIntentToolCall"
    RECOGNIZE_INTENT_SKIPPED = "RecognizeIntentSkipped"
    # ── BackgroundObserve 域（root 后台异步 observe 的 LLM 交互；与 LLM_* 同形但独立类型，
    #     host 据此区分前端是否渲染——core 不感知前端可见性，只发不同类型）──
    BACKGROUND_OBSERVE_REQUEST_STARTED = "BackgroundObserveRequestStarted"
    BACKGROUND_OBSERVE_PROMPT_SENT = "BackgroundObservePromptSent"
    BACKGROUND_OBSERVE_TOKEN_STREAMED = "BackgroundObserveTokenStreamed"
    BACKGROUND_OBSERVE_RESPONSE_FINISHED = "BackgroundObserveResponseFinished"
    # ── TaskRecap 域（background observe 的持久化生命周期标记；崩溃恢复据此重跑，
    #     与逐轮 BACKGROUND_OBSERVE_* 流式事件不同——这两条是"整段 recap 起/止"的记账）──
    TASK_RECAP_STARTED = "TaskRecapStarted"   # payload: {task_id, boundary, agent_id}
    TASK_RECAP_DONE = "TaskRecapDone"         # payload: {task_id}
    # ── 观察者背压（spec: event-commit）：观察者队列溢出丢弃的实时通报。transient
    #     不落库——补读走持久日志（payload 带 position 区间，read_range 可回放）──
    EVENTS_DROPPED = "EventsDropped"          # payload: {subscriber_id, position, dropped}


# 向后兼容：保持 `EVENT_TYPES` 为字符串 frozenset，供 `type not in EVENT_TYPES` 校验。
# StrEnum 成员即字符串，故对原有 `"TaskCreated" in EVENT_TYPES` 用法等价。
EVENT_TYPES: frozenset[str] = frozenset(EventType)


class EventOrigin:
    """`origin` 的 17 个取值（docs/events-v2.md §4）。

    两级点号是为了让 host 能前缀匹配：`loop.` 取全部循环内事件，
    `loop.background_observe` 精确排除后台观察的渲染。
    分隔符用 `.` 不用 `:`——`:` 留给可路由的 capability id（`provider:tool`）。
    """

    ORCHESTRATOR_SESSION_REGISTRY = "orchestrator.session_registry"
    ORCHESTRATOR_TASK_MANAGER = "orchestrator.task_manager"
    LOOP_DRIVER = "loop.driver"
    LOOP_PREPARE = "loop.prepare"
    LOOP_ACT = "loop.act"
    LOOP_OBSERVE = "loop.observe"
    LOOP_BACKGROUND_OBSERVE = "loop.background_observe"
    LOOP_RECOGNIZE_INTENT = "loop.recognize_intent"
    LOOP_COMPACT = "loop.compact"
    LOOP_FINALIZE = "loop.finalize"
    LOOP_SUSPEND = "loop.suspend"
    LOOP_RECONCILE = "loop.reconcile"
    LOOP_CAPABILITY_GATEWAY = "loop.capability_gateway"
    LOOP_LLM_GATEWAY = "loop.llm_gateway"
    HITL_SERVICE = "hitl.service"
    RUNTIME = "runtime"
    PERSISTENCE_SNAPSHOT_WRITER = "persistence.snapshot_writer"

    @classmethod
    def all(cls) -> frozenset[str]:
        return frozenset(
            v for k, v in vars(cls).items()
            if not k.startswith("_") and isinstance(v, str)
        )


# 瞬态事件：高频流式 delta，仅供实时订阅（SSE）消费，**不进任何持久化 / 投影 / 快照路径**。
# 单一真相由 LLMResponseFinished（含完整文本）承载，reducer 也不消费这些 delta，
# 故跳过它们不影响回放、投影与崩溃恢复，只是不再把每个 token 写进事件存储与 DB。
# 持久化/投影/快照各订阅者统一引用此集合（此前 host 以字符串字面量各维护一份）。
TRANSIENT_EVENT_TYPES: frozenset[str] = frozenset({
    EventType.EVENTS_DROPPED,
    EventType.LLM_TOKEN_STREAMED,
    EventType.LLM_REASONING_STREAMED,
    EventType.LLM_RETRY_TRIGGERED,
    EventType.BACKGROUND_OBSERVE_TOKEN_STREAMED,
    # 一轮的两个结局信号（spec 2026-09-09）：给 host 的**实时**指令（把攒着的用户消息
    # 帧 flush 还是丢掉），不是事实。落库会破掉这套设计要保的那条不变式——「丢弃之后
    # 事件日志逐条不变」；而重放也不需要它们：host 的待发帧是内存态，重放时该在的帧
    # 早已在帧日志里、该没有的从来没进去过。
    EventType.ROUND_COMMITTED,
    EventType.ROUND_DISCARDED,
})


# L 档：已停止发射，但 reducer 仍读它们以重放存量日志。删除须过退役闸门。
# 这是「EventType 全集 ≡ 实际发射 ∪ L 档」这条不变式的唯一真相源——
# `tests/unit/test_no_dead_event_types.py` 从这里 import，不再自建平行注册表。
L_TIER_EVENT_TYPES: frozenset[str] = frozenset({
    "SessionStatusChanged", "SessionPausedHitl",
    "HitlRequired", "HitlApproved", "HitlAnswered", "HitlRejected",
    "HitlModified", "HitlCancelled",
    # Task 5（2026-09-03-agent-centric-interaction）：observe.py 的 ReactEventTypes 间接层
    # 删除后，run_observe_react 统一发 LLM_*（靠 state.origin 区分前台/后台），这 4 个
    # BackgroundObserve* 类型停止发射。枚举成员本身按控制方裁定暂不删除（退役闸门是
    # 后续任务的事）——就地登记 L 档，让「EventType 全集 ≡ 实际发射 ∪ L 档」这条不变式
    # 在本 commit 就恢复成立，不把红灯留给下一个任务。
    "BackgroundObserveRequestStarted", "BackgroundObservePromptSent",
    "BackgroundObserveTokenStreamed", "BackgroundObserveResponseFinished",
    # Task 6（2026-09-03-agent-centric-interaction）：recognize_intent.py 切到
    # stream_llm_resilient 后，通用 LLM_PROMPT_SENT（由 gateway 发，origin=
    # loop.recognize_intent）取代了这条 step 专属的镜像事件，停止发射。就地登记 L 档，
    # 不新建常量、不搬到 events.py（那是下一个任务的事），不从 EventType 枚举里删除。
    "RecognizeIntentLLMPrompt",
    # Task 16（2026-09-03-agent-centric-interaction）：会话状态机（session_state.py）
    # 随 SessionRegistry 降格（Task 15）一并退役，这条不再有发射点。reducer 分支原样
    # 保留（存量日志重放靠它）。
    #
    # 同批退役的 `SessionRunning` / `SessionWaiting` / `SessionInterrupted` 与
    # 2026-09-04 停发的 `TaskQueueBlocked` / `TaskQueueInterrupted` /
    # `TaskQueueDrained` **已于 2026-09-05 连枚举一并删除**，不在本档内：这 6 个类型
    # 生于 2026-09-02、死于 09-03/09-04，全部在 `master` 之后的分支内部，`master`
    # 的 `EventType` 里从来没有它们——任何从 master 迁移来的事件流都不可能含有这些
    # 字符串，docs/events-v2.md §5 的退役闸门第 2 级（「确认没有任何回放会碰到」）
    # 因此天然成立，无需等归档周期。
    "SessionFinished",
})


# ── Subscription handle ───────────────────────────────────────────────────────


@dataclass
class SubscriptionHandle:
    """订阅句柄，用于 unsubscribe。"""

    subscriber_id: str
    _bus: "EventBus"

    async def unsubscribe(self) -> None:
        await self._bus._unsubscribe(self.subscriber_id)


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class EventBus(Protocol):
    """事件总线。

    ## 未提交窗口（provisional gate）

    一轮对话在 LLM 真的开口之前不算发生（spec 2026-09-09）：那之前的 `TASK_CREATED` /
    `TASK_STARTED` / `RUN_STARTED` / `LLM_PROMPT_SENT` 都不该进事件日志、也不该到达
    host，否则用户一按暂停就留下一个半截回合。但它们**必须**立刻到达进程内的状态机
    （`AgentLifecycleManager`），否则 agent 停在 `idle`：`pause_agent` 会以
    `AgentNotRunningError` 拒绝（那恰好正是要暂停的那个窗口）、host 的会话状态折叠会把
    会话判成上一轮终态、并发闸门一并失效。

    两个诉求的分野不在「发不发」，而在**发给谁**：

    - `subscribe(..., provisional=True)` 的订阅者恒收全量——进程内状态机反映「现在真实
      发生了什么」；
    - 其余订阅者（`EventPersister`、host 的消费者）在窗口关闭前收不到该 task 的任何
      事件——事件日志只记录「哪一轮算数」。

    窗口由 `TaskManager` 开合（它是 task 生命周期的所有者），见
    `begin_provisional` / `commit_provisional` / `discard_provisional`。

    **不实现这三个方法的总线**（外部 Redis Streams 等）拿到的是默认实现：窗口是
    no-op，事件照常全量投递。行为退化成改造之前——夭折的回合仍会留痕，但不会出错。
    """

    @abstractmethod
    async def emit(self, event: Event) -> None: ...

    @abstractmethod
    def subscribe(
        self,
        event_type: str | None,
        handler: Callable[[Event], Awaitable[None]],
        *,
        provisional: bool = False,
        required: bool = False,
    ) -> SubscriptionHandle: ...

    # ── 提交门（spec: event-commit；默认不支持——required 模式构造期据此显式失败）──

    def attach_commit_gate(self, gate: "object") -> None:
        """接入提交门：emit 在 fanout 之前先经 gate 确认存储提交（required 语义）。

        不支持的总线不实现/继承本默认实现：required 模式下 Runtime 构造期探测到不支持
        即失败并附适配说明，**不静默退化**为观察者路径。实现方契约见
        ``providers/events/bus/in_process/bus.py``。
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support attach_commit_gate; "
            "implement it (commit before fanout in emit/commit_provisional) or "
            "configure event_commit_policy='best_effort' explicitly.")

    # ── 未提交窗口（默认 no-op，见类 docstring）────────────────────────────────

    def begin_provisional(self, task_id: str) -> None:
        """开窗：此后该 task 的事件只投给 provisional 订阅者，其余按序缓冲。

        幂等——重复开窗不清空已有缓冲（`retry` 重排会重进同一条路径）。
        """
        return None

    async def commit_provisional(self, task_id: str) -> None:
        """关窗并**按发生顺序**把缓冲补投给其余订阅者。未开窗时 no-op。"""
        return None

    def discard_provisional(self, task_id: str) -> None:
        """关窗并丢弃缓冲——这一轮当作没发生过。未开窗时 no-op。"""
        return None

    @abstractmethod
    def stream(
        self,
        filter: EventFilter,
    ) -> AsyncIterator[Event]: ...

    @abstractmethod
    async def _unsubscribe(self, subscriber_id: str) -> None: ...


# ── 提交位置、批次与存储错误（spec: event-log / event-commit）──────────────────


@dataclass(frozen=True)
class StoredEvent:
    """事件 + 其在本会话日志中的提交位置。

    `position` 由存储层在提交时分配：同会话内唯一、严格递增，**与事件 ID（铸造序）
    无关**——延迟提交的旧 ID 会拿到较大的 position，这正是快照恢复改用 position 截断
    的原因（可靠性方案 H2）。
    """

    event: Event
    position: int


@dataclass(frozen=True)
class CommitReceipt:
    """一次批次提交的确认：batch_id + 按提交顺序排列的 StoredEvent。

    调用方收到 receipt 即意味着该批已获存储确认（required 语义的依据，WP3 提交门）。
    重试幂等：同 batch_id 同内容重复提交返回原 receipt（position 不变）。
    """

    batch_id: str
    records: tuple[StoredEvent, ...]


class EventConflictError(Exception):
    """相同 batch_id 以不同内容重放，或事件 ID 已被另一批次提交。

    幂等重试必须**原样**（逐字段一致，忽略 position）；顶替/改写已提交批次不被允许。
    调用方不得换一个新 batch_id 猜测性重发——那是双写，不是重试。
    """


class PersistenceUnavailableError(Exception):
    """事件提交未获存储确认（required 模式下 emit 抛出；spec: event-commit）。

    语义：调用方不得把本错误当作普通可重试失败——会话已进入 ``storage_unavailable``
    隔离（停止调度新副作用），恢复前须以 batch_id 确认未知提交（幂等重试拿原 receipt）。
    """


# ── EventStore Protocol ───────────────────────────────────────────────────────


@runtime_checkable
class EventStore(Protocol):
    """事件流持久化抽象。host 提供具体实现（Postgres / SQLite / in-memory）。

    ## 全部必须实现，没有可选扩展

    `append` / `append_batch` / `read_range` / `committed_head` / `read_last_of_type` /
    `read_session_events_of_types`。**有序提交是底线，不是可选项**——理由见下节。后两个是
    恢复路径无条件要走的（取最新快照 / 按类型收窄折叠），做成可选就又要在 core 里探一次能力。

    「可选扩展（未实现就抛 `NotImplementedError`，调用方降级）」那一档**已经没有了**
    （2026-09-21）。它每多一个成员，core 里就多一处能力探测、两个分支、两种失败形态——这个
    仓为此付过四次（`replay` / `read_by_session_after` / 快照 / `read_session_events_of_types`）。
    最后一个成员 `list_active_session_ids` 连同它唯一的消费者一起删掉了：会话清单是 host
    自己的数据，不该由 core 从事件流反推。

    读取一律按 position：**不存在**「按事件 ID 取增量」的 API。ID 铸造序 ≠ 提交序，
    那种游标正是 H2 的根因（详见下节）。

    ## 有序提交为何必选（spec: event-log；WP3 提交门 / WP4 快照切面的地基）

    没有提交位置就没有正确的恢复：ID 在事件构造时铸造，而提交可以更晚发生（未提交
    窗口 `begin_provisional` / `commit_provisional`），「后提交的小 ID」真实可达。
    按 ID 当快照游标，延迟提交的事件会**永久**落在游标之外——两条恢复路径（全量回放
    vs 快照+增量）从此给出不同的世界，且不报错、不留痕（可靠性方案 H2）。

    所以这里**不提供 legacy 回落**：一个按 ID 排序的 store 不是「功能少一点」，而是
    「恢复语义是错的」。宁可在构造期响亮拒绝，也不要让它静默跑出两个世界。

    - `append_batch` 是**原子**提交：整批要么全部落库（各自分配连续递增 position）、
      要么全部不存在；批内事件必须同属一个 session。
    - `batch_id` 在第一次提交前生成、重试不换；相同 batch_id + 相同内容（忽略
      position）→ 原 receipt；不同内容 → `EventConflictError`。
    - `committed_head` 返回该会话最新已确认提交的 position；无提交时为 0。
    - `read_range` 按 position 升序只读已提交事件，`through_position` 含端点。

    存储实现**不得**用无锁 `MAX(position)+1` 分配——必须经会话 head 行（同事务
    UPDATE）串行化（SQLite `BEGIN IMMEDIATE` 等价路径 / PostgreSQL 行锁）。

    Protocol 的 `@abstractmethod` 拦不住鸭子类型的 store（不继承本协议就没有 ABC
    检查），所以 Runtime 在构造期用 `supports_ordered_commit(store)` 兜一道，缺一个
    方法就拒绝启动。

    ## 排序口径

    `read_session_events_of_types` 按 **position（提交序）** 排序。
    `append(event)` 由单事件 `append_batch` 实现（batch_id 确定性取 event.id），
    既有调用方语义不变。
    """

    # **这里没有 `append(event)`**（2026-09-21 删）。它在两个实现里逐字相同——
    # `append_batch(event.session_id, event.id, [event])`——而 src 里只有一个调用者
    # （`EventPersister`）。
    #
    # 删它不是为了少一个便利方法，是因为 `batch_id = event.id` 那一句是**策略**：确定性取值，
    # 好让原样重试撞上 batch 幂等账而不是写出第二份事件。让每个实现各写一遍，等于把一条正确性
    # 策略复制 N 份——某个实现写成 `generate_id()`，重试就会写出第二份，而且不报错。本仓已经
    # 为同一个形状付过一次（`PROJECTION_VERSION` 写侧读侧各一份字面量，改一个的后果是所有快照
    # 永远判不可用、O(delta) 退回 O(n)、不报任何错）。
    #
    # 调用方直接用 `append_batch`；测试夹具里那个动作收在 `tests/_event_helpers.append_one`。

    # ── 有序提交（必须实现；spec: event-log）──────────────────────────────────

    @abstractmethod
    async def append_batch(
        self, session_id: str, batch_id: str, events: "list[Event]",
    ) -> CommitReceipt:
        """原子提交一批事件，返回带提交位置的 receipt。"""
        ...

    @abstractmethod
    async def read_range(
        self,
        session_id: str,
        *,
        after_position: int = 0,
        through_position: int | None = None,
        include_types: "tuple[str, ...]" = (),
        exclude_types: "tuple[str, ...]" = (),
        task_id: str = "",
    ) -> "list[StoredEvent]":
        """按 position 升序读取 (after_position, through_position] 的已提交事件。

        **本协议唯一的区间读**。四个过滤器都是可选的，都作用在同一个 position 区间上：

        - `include_types` 非空 → 只要这些类型。空 = 不限。
        - `exclude_types` 非空 → 跳过这些类型。全量重放用它排掉状态快照事件（那些事件的存在
          是为了省重放，读回来反而更贵）。两者同时给时 exclude 胜出。
        - `task_id` 非空 → **只排除明确属于别的 task 的行**。`task_id` 为 NULL / 空串的照样
          取回，因为那一档是「无从归属」而不是「属于别人」。判错方向的代价不对称：把一条没
          归属的 `CapabilityInvoked` 漏掉，capability gateway 就以为那次调用没跑过 → 静默重跑
          一个有副作用的工具；而多取回几条只是多折几下。**这条语义两个实现必须逐字一致**，
          由 `tests/unit/test_read_primitives_conformance.py` 对两者跑同一套用例钉住。

        **通用读原语，不带任何特定类型的语义**——调用方（core）自己填要读/要排除什么。

        ## 为什么没有单独的 `read_session_events_of_types`（2026-09-21 并进来）

        从前有。它和本方法是同一个查询换了过滤条件，差三处：include 而非 exclude、多一个
        `task_id`、**少了位置区间**。第三处是它的问题所在：capability 折叠靠它回答「这个
        dangling tool_call 跑过没有」，而它读的是整条会话——按 task 收窄之后主干收住了，但那条
        「无从归属」的安全阀仍然是全会话的，于是上界里始终留着一项随会话长度增长的成分。并进
        来之后它自动获得 `after_position`，那一项就关掉了。

        顺带清掉的另一个残留：它的排序口径带着
        `ORDER BY CASE WHEN position IS NULL THEN 0 ELSE 1 END, position, id`，为的是让未回填的
        存量行排在前面——而那个状态在 2026-09-20/21 之后没有读者了（`open_sqlite_event_store`
        拒绝打开未回填的库、宿主侧 m020 在任何读之前回填完）。那个 CASE 表达式同时也是它必须
        走 temp b-tree 的直接原因。
        """
        ...

    @abstractmethod
    async def committed_head(self, session_id: str) -> int:
        """该会话最新已确认提交的 position；无提交时为 0。"""
        ...

    # **这里没有 `replay` / `REPLAY_BATCH`**（2026-09-21 搬走）。「按 position 区间分批重放
    # 整条会话」现在是 `core.control.reducers.replay_session`——它只用 `read_range` 与
    # `committed_head` 两个必需读原语，所以任何满足本协议的 store 都能被重放，**不需要任何
    # 能力探测**。搬走的三条理由（没有 store 覆盖过它、它反而引入了一次构造期探测
    # `supports_replay`、那层抽象两天内失效两次导致同一个循环抄了三份）写在那个函数的
    # docstring 里。
    #
    # 留在这里的教训：给协议加一个「能用的默认实现」等于宣称「实现方可以换掉它」。那份自由
    # 一次没被行使，而它的代价是实打实的——鸭子类型的 store 拿不到默认实现，于是必须补一道门
    # 来兜；一个只由必需原语组合而成的算法，属于调用方。


    # 这里从前有 `list_active_session_ids()`——从事件流反推「有 SessionCreated 但无终态
    # 事件」的会话清单。2026-09-21 删，因为**这个问题 core 不该回答**：会话清单是 host
    # 自己的数据（它建的会话、它的 sessions 表、它的状态列），core 从事件流把它重新推一遍
    # 是职责倒置，而且推得更差——那台状态机的两条 discard 依据（`SessionFinished` /
    # `SessionStatusChanged`）在 src 下没有任何 emit 调用点，判据恒真，返回的实际是「这个
    # 库里出现过的全部会话」。host 侧同样的问题是一条带索引的 status 查询。
    #
    # 它唯一的消费者是 `Runtime.rebuild_all_pending_hitl`（同日删除）。要按会话装填，host
    # 拿自己的清单逐条调 `Runtime.rebuild_hitl(session_id)` / `rebuild_session(session_id)`
    # ——那两条都走快照 + 增量（`rebuild_view`），O(delta)。

    @abstractmethod
    async def read_last_of_type(
        self, session_id: str, type_: str,
    ) -> "StoredEvent | None":
        """按提交序取该会话**最后一条**指定类型的事件；没有则 None。

        通用读原语，同样不带特定类型的语义。存在的理由是「取最新那一条」必须只读**一条**
        ——用 `read_session_events_of_types` 会把全部同类事件连载荷一起捞回来，对状态快照
        这种一条一条攒下来的类型就是 O(快照张数) 的浪费。

        「最后一条」= position 最大。与 `read_range` / `committed_head` 同一个序，不引入
        第二种「最新」口径（从前快照那条口径要靠 `snapshot_at` + `id` 兜，因为写入序 ≠
        时间序）。

        **必需**，不是可选扩展：恢复路径无条件靠它取快照（`rebuild_view`）。做成可选就又要在
        core 里探一次「这个 store 支不支持」，那是同一个坏设计的第三次——`replay` 那次已经
        证过，一个能力探测生出两个分支与两种失败形态。想不做快照的 store 返回 None 即可，
        那是合法配置（恢复退化为全量重放，仍然正确）。
        """
        ...

    # **这里没有 `read_session_events_of_types`**（2026-09-21 并进 `read_range`）。它和那个
    # 方法是同一个查询换过滤条件，而缺的那一样（位置区间）正是它的问题：capability 折叠靠它
    # 回答「这个 dangling tool_call 跑过没有」，「无从归属」那条安全阀让它的上界始终留着一项
    # 随会话长度增长的成分。合并的完整理由见 `read_range` 的 docstring。

    # ── 快照 ────────────────────────────────────────────────────────────────
    # **协议不提及快照。** 一张状态快照就是日志里的一条 `EventType.STATE_SNAPSHOT` 事件，
    # 读写都走通用原语（`append_batch` / `read_last_of_type` /
    # `read_range(exclude_types=)`），store 对那个类型不做任何特殊处理——它是 core 的词表，
    # 不是存储契约的一部分。
    #
    # 从前这里有 `save_snapshot` / `load_latest_snapshot` 与 `RunSnapshot`，2026-09-20 删除。
    # 删掉换来的不只是协议变小：
    #
    #   · 「哪张最新」不再需要一套快照专属口径。从前必须规定「按 `snapshot_at` 取最大，相同
    #     时按 `id` 取最大」——因为写入顺序不保证与时间顺序一致（并发写、重试补写都可能乱序），
    #     而「哪次调用最后执行」不是可跨实现定义的排序键。现在 = position 最大，与
    #     `read_range` / `committed_head` 同一个序，不存在第二种「最新」。
    #   · 切面位置 / `projection_version` / `chain_depth` 从**列**变成 payload 里的键，而
    #     payload 对 store 是整体不透明的——它没法只丢其中一个。m020 补的就是那几列，而丢了
    #     它们 `snapshot_is_usable` 恒判不可用，恢复永远全量回放且**不报错**。


#: 有序提交的三个必需方法——`supports_ordered_commit` 要求**全部**落地
#: （部分实现会让提交门与快照切面处于半可用状态，比完全没有更难诊断）。
_ORDERED_COMMIT_METHODS = ("append_batch", "read_range", "committed_head")


def supports_ordered_commit(store: object) -> bool:
    """store 是否真的实现了有序提交（spec: event-log）。

    有序提交是 `EventStore` 的**必需**部分，所以本函数不是能力协商、而是**契约校验**：
    Runtime 构造期调用它，缺一个方法就拒绝启动。之所以还需要运行时校验，是因为
    `@abstractmethod` 只对显式继承 `EventStore` 的实现生效——鸭子类型的 store 不经过
    ABC 检查，少一个方法要到第一次提交才炸。

    **不要用 `hasattr` 代替本函数**：继承 `EventStore` 的 store 恒有这些属性名，
    `hasattr` 会把只继承了抽象桩的 store 误判为已实现。判据是「这个属性是不是协议
    自带的那个桩」：鸭子类型自带实现 → 真；继承协议但未覆盖 → 假；没有该属性 → 假。
    """
    for name in _ORDERED_COMMIT_METHODS:
        impl = getattr(type(store), name, None)
        if impl is None or impl is getattr(EventStore, name, None):
            return False
    return True


# **这里没有 `supports_replay`**（2026-09-21 随 `replay` 一起删）。它存在的唯一原因是「协议的
# 默认实现只白送给显式继承的 store，鸭子类型拿不到」——判据还得写明「与
# `supports_ordered_commit` 相反，这不是笔误」。`replay` 变成 core 的函数
# （`reducers.replay_session`，只用 `read_range` + `committed_head`）之后，任何满足本协议的
# store 都能被重放，这道门连带那条解释一起没有了。
#
# `supports_ordered_commit` 留着：它兜的是**抽象桩**（鸭子类型 store 压根没有那些方法），
# 与「默认实现继承不到」是两回事。


# ── Blob 存储（事件流的字节侧）─────────────────────────────────────────────────


class EventBlobStore(ABC):
    """事件流侧的「二进制 sink」：存取图片等二进制内容，事件库里只留 ref。

    与 `protocols.memory.MemoryBlobStore` **同形但类型无关**（spec §3）。不做成子类型、
    也不共用一个 ABC，理由是两侧语义会各自演进——最明显的是**回收锚点不同**：memory 侧
    是记录 `is_superseded`，event 侧是事件保留策略。今天同形不代表明天同形。

    host 要共用就一个类同时实现两者，注册两次：

        class MyBlobStore(MemoryBlobStore, EventBlobStore): ...

    **ref 前缀取自 `protocols.context.BLOB_REF_PREFIX`**，与 memory 侧同一个常量——
    但**仅此而已**：两侧的 ref 是**两个独立的命名空间**，core 从不比较、也从不拿
    一侧的 ref 去另一侧解。host 用同一实例时两个 ref 恰好相同，那是实现层的巧合，
    不是任何代码可以依赖的前提。

    ⚠️ **回收策略由 host 定，core 不规定。** 事件流里的 ref 能否取回字节，完全取决于
    host 让 event blob 活多久：想让事件流永远可重建，就让回收与事件保留策略对齐
    （例如永不回收，或按事件 TTL）。**共用一个实例时尤其当心**——该实现要同时看两侧的
    引用才能安全回收，仅把 `MemoryProvider.live_blob_refs()`（memory 侧的活引用集合）
    喂给 `FsBlobStore.collect` 之类的 sweep，会把事件流仍需要的字节当孤儿删掉
    （spec §9）。
    """

    @property
    def can_externalize(self) -> bool:
        """本 store 是否真的能存——`NullEventBlobStore` 返回 False。

        调用方据此**先探询、再决定**，而不是调用 put 并捕获 NotImplementedError：
        后者会把「响亮失败」降级成控制流，让真正的接线错误也被静默吞掉。
        基类默认 True，既有实现无需改动。
        """
        return True

    @abstractmethod
    async def put(self, data: bytes, media_type: str, ctx: "ProviderContext") -> str:
        """存字节，返回 ref。必须**内容寻址且幂等**：同样的 data 返回同样的 ref。

        这同时给到三件事：写入端去重、重放安全、以及 rehydrate 字节稳定——同一 ref
        每次还原出的 base64 完全一致，prompt cache 前缀不会被打碎。
        """

    @abstractmethod
    async def get(self, ref: str, ctx: "ProviderContext") -> "tuple[bytes, str] | None":
        """取字节。对不存在 / 已回收的 ref 返回 `None`，**不得 raise**。

        blob 过期、宿主换机、GC 误删都会发生，调用方据此降级为文本占位，
        绝不因取图失败中断 loop。
        """


class NullEventBlobStore(EventBlobStore):
    """未注册 `EventBlobStore` 时的默认实现。

    `put` 刻意抛错而不是静默产出假 ref：调用方（`core.utils.content`）先探询
    `can_externalize` 决定是否外部化，**不**捕获这里的 NotImplementedError——
    它仍是接线错误的响亮信号。
    """

    @property
    def can_externalize(self) -> bool:
        return False

    async def put(self, data: bytes, media_type: str, ctx: "ProviderContext") -> str:
        raise NotImplementedError(
            "No EventBlobStore registered; register one via "
            "ProviderRegistry.register_event_blob_store() before externalizing content."
        )

    async def get(self, ref: str, ctx: "ProviderContext") -> "tuple[bytes, str] | None":
        return None


