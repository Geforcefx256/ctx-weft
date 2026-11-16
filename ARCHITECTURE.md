# ctx-weft 内部架构与改动边界

> 核对日期：2026-11-16。本文描述当前工作区实现，供维护引擎与后续 agent 定位职责。
> SDK 调用参考见 [README.md](./README.md)，公开契约以 `src/ctx_weft/__init__.py`、`protocols/` 和对应回归测试交叉核对。
> 文档使用模块路径和符号定位，不保存易过期的源码行号或分支名。

## 0. Agent 修改前必读

- 先读本仓 `AGENTS.md`、本文件及待改模块的测试，再检查 `git status --short`。工作区可能包含进行中的重构；未提交文件和新增模块都是当前实现的一部分，不得用 HEAD 覆盖。
- `ctx-weft` 拥有执行语义；`NetliveCoworkPy` 拥有产品路由、身份与资源权限、HTTP/SSE 和展示投影。修复应落在实际拥有该职责的一侧。
- 不为适配旧文档恢复已删除的入口、全局状态、第二份投影或状态推导。历史 spec、plans、upgrade 文档用于理解决策背景；与当前实现冲突时，先核对具体测试和迁移记录，不能仅凭旧方案改回代码。
- 改事件、状态、记忆折叠、恢复或公开协议时，必须检查正常执行、暂停/取消、冷恢复及提交失败路径。不能只让当前热路径通过。
- 测试中的边界约束也是设计依据。不得为让变更通过而随意扩大 import 白名单、删除隔离测试或伪造提交成功。

## 1. 架构总览与数据流

```text
host / 自定义 Provider
  -> CtxWeftRuntime（公开命令、会话锁、生命周期与关闭）
       -> SessionRegistry / AgentLifecycleManager
       -> 每会话一个 TaskManager
            -> TaskRunner.assemble -> AgentBinding（Agent / 模板 / 本次模型）
            -> TaskRunner.execute -> StepDriver -> RunOutcome
            -> disposition_for -> TaskManager 落任务状态、调度队列

Step -> ContextAssembler -> PromptContext -> Jinja + PromptBuilder -> LLMRequest
     -> llm_gateway / llm_stream -> LLMClient
     -> CapabilityGateway -> Authorizer / HITL -> CapabilityProvider

持久事件 -> EventBus -> CommitGate -> EventStore 确认 -> 订阅者
恢复读取 -> committed position / 快照 / reducers -> 初始化或按需装填
```

`TaskRunner.execute` 的返回值是任务处置的数据通道。TaskManager 不订阅 RunFinished 来重新调度：进程内总线同步投递，订阅回调内再 drain 会导致重入。`RunFinished` 是执行事实，业务任务状态由 TaskManager 按处置表落地。

## 2. 核心组件职责

以下路径相对 `src/ctx_weft/`。

| 组件 / 源码 | 所有权与边界 |
| --- | --- |
| `core/runtime.py` / `CtxWeftRuntime` | 命令入口、会话锁、构造接线、初始化、装填、停止、查询和实例关闭；宿主通过公开入口调用 |
| `core/registry.py` / `ProviderRegistry` | Memory 单实例、Capability 集合、按优先级的 Knowledge、LLM resolver、鉴权器和两类 blob store |
| `core/orchestrator/lifecycle/agent_manager.py` / ALM | Agent 身份、状态、当前任务、模型选择；instantiate 创建并发事件，materialize 构造执行对象，load 重放已有事实 |
| `core/orchestrator/lifecycle/template_lookup.py` | 模板限定 ID 的解析；模板只经 AgentCapabilityProvider 进入引擎 |
| `core/orchestrator/lifecycle/session_registry.py` | 会话成员与 TaskManager 工厂；不持有聊天产品展示状态 |
| `core/orchestrator/task/manager.py` | 任务队列、同 Agent 串行、恢复与结果应用；一个已装填会话只有一个 owner |
| `core/orchestrator/task/disposition.py` | 纯函数 `disposition_for`：RunOutcome + 重试预算 -> 任务处置；无 bus、队列或 I/O |
| `core/orchestrator/model.py` | ModelChoice / ResolvedModel 等执行侧模型值对象 |
| `core/loop/driver.py` | StepDriver、LoopState / LoopContext、前后台 Step 执行与 ExecutionRegistry |
| `core/assembler/` | Source 取事实、purpose 装配、Jinja 提示词程序及预算执行；不改变任务状态 |
| `core/loop/capability_gateway.py` | 工具解析、鉴权、HITL、逻辑调用身份、执行与恢复对账 |
| `core/capabilities/cache.py` | 按会话共享的能力缓存和 pin 生命周期；不在每个 Step 内另建缓存 |
| `core/hitl/` | Registry 决定缓存、Service 请求生命周期、ReplyIntake 答复归一；协程等待在 loop/hitl_waiter.py |
| `core/control/` | 持久事件折叠、查询视图、初始化校准与 SnapshotWriter |
| `core/events/commit_gate.py` | 原批次提交确认、未知提交重试与存储隔离 |
| `protocols/` | host、core、Provider 共用协议和 DTO；外部调用者不导入 core 实现 |
| `providers/` | 可替换默认实现；不得反向依赖 core 的调度/恢复域逻辑，有限纯工具例外由边界测试明确列出 |

构造期自动注册 `ControlCapabilityProvider`、`SkillExecutorCapabilityProvider`、`MediaCapabilityProvider`，并要求注册至少一个 `AgentCapabilityProvider`。构造同步完成公共接线，不读取历史或发起恢复。

### 公开内容协议

`protocols/content.py` 定义 TextPart、ImagePart、ContentPart 和 blob 引用前缀，并实现
共享的解析、校验、JSON 转换、文本提取及 blob 外部化。协议模块不导入 core；
`context.py` 只保留调用上下文和 Citation。共同异常基类在 `protocols/errors.py`，
内容异常在 content 协议中定义，顶层也导出同一异常类型。

新输入用 `parse_content` 严格解析，未知 part 不静默丢弃；历史读取仍用
`normalize_content_parts` 的容错行为。`externalize_content` 只处理调用方明确指定的
存储和 ProviderContext：已有引用要求 source_blob_store，跨存储读取后写入目标，
缺失引用明确失败。返回成功只确认 blob 准备，不表示事件提交或任务派发。
引擎的事件侧／记忆侧仍从原始输入各自外部化；已有事件恢复降级规则保持。
`core/utils/content.py` 保留 token 预算、提示词渲染及工具结果策略，引用协议的共同实现。

## 3. Loop 引擎与 Step 流水线

`CtxWeftRuntime._build_step_driver` 注册八个 Step，源码均在 `core/loop/steps/`。当前没有独立的 reason Step。

| Step | 职责与接续 |
| --- | --- |
| `prepare` | 解析能力、估算预算、装配输入；进入 act，或 compact 后重装配 |
| `act` | 模型/工具循环与回合提交；按退出原因进入 suspend、prepare、observe，或交接后台观察 |
| `observe` | 判决及段报告的统一实现；前后台由执行模式和 judge_task 控制，前台判决后进入 finalize |
| `finalize` | finish 记录、父 scope 回填、blackboard、TaskFinalized 和 RunOutcome；不自行写任务终态 |
| `suspend` | 返回等待子任务的 RunOutcome，由 TaskManager 挂起及唤醒 |
| `compact` | 统一前台压缩；由 options.after 指定重装配或结束维护 run |
| `recognize_intent` | 旁路元信息生成，回填 title / description / session goal |
| `reconcile` | 在模型执行前对账 dangling tool calls，再进入 prepare |

`StepDriver.run` 先确保子任务派发框，再持久化/暂存 raw user prompt，然后按 next_step 执行。`## Current Task` / `## Current Message` 是渲染框架，不存为用户原文。每步的事件、state_patch 和 StepStarted/Completed/Failed 由 driver 统一处理。

有未配对工具调用时从 reconcile 开始，否则从 prepare 开始。软暂停由协作检查点 park；硬取消在 Step 边界中止，不能把暂停直接替换为取消协程。

### 压缩与接续

**step 参数与 compact 的三个入口**：

`StepOutcome.next_step_options` 指定下一次 step 的调用参数，driver 复制后写入该次
`ctx.execution.options`，不会自动继承到后续步骤。跨 step 的执行结果仍通过
`state_patch` / `LoopState` 传递。前台 `act → observe` 显式传 `boundary` 与 `judge_task`；
后台 Observe 仍经 `start_durable(..., options=...)` 发起，保留持久化请求与恢复语义。

- **Prepare 预算触发**：`prepare → compact → prepare → act`。Compact 接收
  `trigger/token_estimate/force/after`；`after="prepare"` 时返回
  `next_step_options={"after_compact": True}`。第二次 Prepare 从 source 重新装配，
  不再检查压缩阈值，避免重复触发；一次性 hint 在最终进入 Act 前消费。
- **Act 派发前触发**：`act → compact → prepare → act`。已持久化、已记账但尚未执行工具的
  回合存入 `LoopState.pending_act_turn`；压缩保护该回合并从摘要输入排除。Act 重入后先恢复
  原工具调用，不重问模型、不重复记账，不消耗新的模型回合；工具结果落定后清除接续状态。
- **runtime 手动触发**：`compact_agent` 保留 idle guard 与 `CompactReceipt`，通过 driver 的
  前台执行入口运行 Compact，`force=True, after=None`。仅结束维护 run，不启动 Act 或终结业务 task。

Compact 在可能改写记忆前清除旧的增量 token 基线；重装配使用完整估算，后续真实 Act usage
重建基线。压缩 noop 仍只回 Prepare 一次，并保留上下文恢复次数与压不动时的止损规则。

#### CompactStep 与 memory 整理边界

`CompactStep` 在 `core/loop/steps/compact.py` 内维护 L0.5 → L1 → L2 → L3、
模型预算、tokenizer、摘要调用及事件。普通模式达标即停，force 持续到适用级别结束。
进入该 step 后先通过会话的 `ExecutionRegistry` 等待同 agent 全部后台 Observe 收尾，
包括同 task 较早的运行和恢复启动的运行；不等待其他 agent 或元信息生成。等待前发出
`MemoryCompactStarted(phase="waiting_observe")`，等待不持有 memory/任务锁，取消等待不
取消 Observe。等待后重新装配未裁剪的当前任务 prompt 并计算完整预算，再决定是否压缩。

`MemoryProvider.organize` 在既有 memory 协议中定义五种操作：`demote_images`、
`reduce_history`、`fold_history`、`fold_current`、`fold_segment`。默认实现位于
`providers/memory/organize.py`，不依赖 loop、TaskManager、Agent 事件、LLM 或预算。
两个存储后端在各自事务/锁内实现条件检查和替换；`fold` 的公开契约不变。

Compact 通过 `recall_for_organize` 取得普通 MemoryRecord，将现成材料交给 assembler；
Source 不重复召回。摘要成功后以相同输入 ID 调用 organize，不把旧摘要用于新选的范围。
L3 只摘要实际前缀，近期窗口不进入摘要输入。材料召回只读；折区残留图片由 Compact
显式调用 demote_images 预处理，返回新 ID 后使用新记录。

Observe 继续决定是否折段并生成报告，以固定观察输入调用 `fold_segment`。该操作保留
用户边界与已有摘要；它和跨多段的 `fold_current` 是不同操作。收尾创建和报告回填仍在
现有 `task_memory_fold.py` / Finalize 路径，不在这次迁移中顺带改造。

运行时已有 LoopContext.memory；传入 provider 的 ProviderContext 仅携带身份和作用域。
没有新 Organizer 注册、公开 Plan/prepare/apply、ContextVar 或 CompactSnapshot。
默认实现复用有界的不可变正文缓存，始终重新检查活动成员，按租户/会话隔离缓存键。
普通工具通过调用/结果关联识别完成状态；被排除出对话协议的派发/silent 调用通过非渲染
的执行关联及完成审计记录提供相同事实，不再由 Compact 传递 protected_ids。

## 4. LoopState、执行上下文与模型配置

`LoopState` 保存一次执行推进的数据（task、agent、session、本次 ResolvedModel、prompt、transcript、verdict、pending_act_turn 等）。`LoopContext` 提供依赖与控制令牌；`StepExecution` 指定本次前台/后台模式和 options。`StepOutcome` 只表达接续、补丁与事件，不能变成另一份任务状态机。

ALM 保存 `ModelChoice(account, model, config)`，派发时解析为 `ResolvedModel` 并经 AgentBinding 交给执行。client、实际账号/模型、上下文窗口、输出预留与 output policy 必须来自同一次解析，不在 Step 中分别猜测。

`protocols/llm.py` 中的 `ModelCallConfig` 保存 reasoning_effort、temperature、top_p、max_tokens、stop 等调用设置，支持显式序列化 codec。Agent 选择、run、请求及重试持有独立副本；修改查询返回值不能修改实际 Agent。模型切换和配置替换通过 `set_agent_llm` / `set_session_llm` 写 AgentLLMChanged，恢复重放保留完整配置。

`core/loop/llm_gateway.py` 负责请求构造、消息合法化、动态输出预算和请求自愈；`core/loop/llm_stream.py` 拥有流的收集。Adapter 只解释自己的 wire 格式，经 protocols 接入，不能把收集器搬回默认 Provider。网络自愈不能重做已执行的工具副作用；LLM outage 最终走可恢复中断，不按普通任务失败消耗原地重试预算。

## 5. 上下文装配与提示词职责

`ContextRequest` 用 purpose 选择 act / observe / compact / recognize_intent，并用匹配的 Intent 携带参数。Source 从 Memory、Knowledge、能力和模板取事实，装配成 `PromptContext`；`prompts.py` 执行 `prompt_assets/steps/*.jinja`，`PromptBuilder` 执行预算、原子分组裁剪和合法消息输出。

- 模板 identity facets（SOUL/ROLE 等）表达身份与质量原则；Jinja 表达提示词内容、顺序、调用指令和裁剪优先级；Python 负责事实、参数与机械预算。不要在 Source、Step、工具 schema 中复制另一套提示词程序。
- 工具定义有独立预留预算，不作为普通文本任意裁掉。当前任务和工具调用/结果等受保护分组必须保持完整。
- 已关闭任务的 full / lean 胶囊是装配时的可选视图。预算选择不等于持久折叠；真正 demotion 才修改记录。无可用摘要时保留 raw 证据，不造空胶囊。
- 更改提示词先读 `core/assembler/prompt_assets/README.md`，同时核对 en / zh 资源及渲染、预算和协议测试。

## 6. 能力调用、鉴权与恢复

Provider 声明 capability，resolver 按引用绑定，gateway 负责解析及调用。鉴权器与 Provider 通过 registry 接线；人工审批通过统一 HITL 服务，不能在宿主或工具中另外维护审批决定缓存。

控制工具以 `core/capabilities/control_tools.py` 的实际注册为准。当前使用 `switch_task`、`delegate_task`、`delegate_plan`、`ask_user`、`report_observation`、`update_task_metadata`。`finish_task` 仅保留旧事件/恢复名称，不是当前注册的可调用工具；不得根据历史文档重新暴露。`report_observation` 返回结构化观察结果，由 ObserveStep 应用。

恢复以逻辑操作身份与已确认结果为准，不把模型 wire tool_call_id 当全局唯一调用。Reconcile 先复用已完成结果；未完成调用按活 capability 的 RecoveryPolicy 与已存执行事实处理。不能把“没有 TOOL_RESULT”直接等同为“从没执行”，更不能在未知提交时重跑有副作用的操作。

ProviderContext 携带 tenant / session / agent 等身份和作用域；产品的用户、套件与 namespace 规则由宿主 Provider 解释，不渗入通用引擎。

## 7. 任务编排、Agent 与观察生命周期

### 任务与 Agent

任务先 assemble 得到真实执行 Agent，再发 TaskStarted 并执行；同 Agent 任务串行，不同 Agent 可在配置并发范围内运行。`effective_agent_id` 是调度串行键和装配共用的归属规则，不能另写“当前活跃 Agent”猜测器。

`send_message(agent_id, content, session_id=...)` 按 Agent 定址：空闲/终态当前任务可开新任务，已有任务按其状态注入或重排，等待子任务时保留父任务等待语义。running / terminated Agent 有明确守卫，不隐式排队。会话级聊天路由归宿主。

ALM 是 Agent 状态和模型选择的唯一住所；TaskManager 应用任务处置，ALM 消费任务事件推进 Agent。新建 instantiate 和恢复 load/materialize 必须分开，重放不能再次发 AgentSpawned / AgentInstantiated。

### RunOutcome 处置

| 执行结果 | TaskManager 处置 |
| --- | --- |
| success / fail 判决 | FINISHED / FAILED |
| continue 判决 | PENDING，自动重排，不消耗运行异常重试预算 |
| waiting / HITL park | AWAITING_HUMAN，保留可应答入口 |
| 等子任务 | SUSPENDED，由依赖完成唤醒 |
| task switch | REVIEWING，沿任务切换协议收尾 |
| 可重试运行异常 | 预算内重排，耗尽后 INTERRUPTED |
| outage / 不可原地重试的中断 | INTERRUPTED，等显式恢复 |
| 硬取消 | CANCELED，先确认在途 run 收口 |

### Observe 持久协议与崩溃边界

前后台 observe 共用 `ObserveRequested → ObserveStarted → ObserveCompleted → ObserveApplied`。

`report_observation.task_status` 使用四态：`success` / `fail` 结束任务，`continue`
一律自动重新入队执行，`waiting` 停在 `AWAITING_HUMAN` 等待用户回复。`continue`
不消耗运行异常重试预算，也不兼容为旧的等待语义。交互任务的纯文本回合先临时 park，
后台判 `continue` 后先完成段摘要、收口本轮等待入口，再发布 `TaskRequeued` 并启动下一轮；
判 `waiting` 则保留入口。前台判 `waiting` 同样建立可应答的 HITL 入口。
后台自动续跑以 `ObserveCompleted.action="continue"` 持久化，恢复时完成原决定而不重新判决。
`max_turns` 复核会展示本段已执行轮数：明确超出能力范围可判 `fail`，其余结果
由代码收敛为 `continue`，不允许 `waiting` 或 `success`，观察失败也继续执行；其余场景
观察没有可用判决时的机械 `success` 兜底保持不变。

当前前台 observe 先持久化请求，再执行 ObserveStep 的 LLM 复核；不再按 root、跨 Agent
或 ROLE 是否存在跳过复核。`_mechanical_verdict` 只在没有可用判决时兜底，不能改回
“前台先机械成功、后台补报告”。前后台执行模式与是否判决是两个维度，`judge_task`
控制判决，report-only 观察仍可生成段报告。旧日志中预存的 verdict/disposition 仅按既有
恢复路径兑现，不是新请求的默认生成策略。
前台收尾的崩溃恢复由 runtime 负责，observe 内不手动推进 FinalizeStep。
`ObserveCompleted` 保存报告及确定的判决/处置；`ObserveApplied` 必须晚于任务处置与
报告应用。报告等待任务状态通知后仍读取持久化事实，晚订阅不依赖历史通知；应用时
校验 finish 记录的 observe 归属，并与取消收尾串行。
执行逻辑和四个事件的组装、发送统一位于 `core/loop/steps/observe.py`；driver 在调度前调用意图持久化入口，TaskManager 在仲裁和结算位置调用 observe 提供的回调。
Requested 在执行前持久化（park 时嵌入 `HitlOpened`），Started 在观察前立即写入；Completed
保存报告及执行后确定的判决。Completed 先于判决应用和相应折叠；Applied 只在结果应用完成后销账。
`TaskRecapRequested/Started/Done` 已停发，仅供旧日志重放。快照保存 `pending_observe` 的结果、
已完成副作用与任务状态提交标记。冷恢复已有 Completed 时不再调用模型，沿原决定幂等补完。
前台仍由 TaskManager 在 RunFinished 后落任务状态；恢复沿原 observe_id 补完，不新建一份判决或报告。
收尾报告交接以完整持久化的 finish 对为准：其 metadata 保存原文保留策略和槽位数量；后台在真实报告写入、必要折叠完成后才发送 Applied。前后台先后次序仅影响等待，进程内通知只唤醒，不承载报告或恢复决定。
前台 observe 可协作暂停/取消；暂停时旧 Completed（放弃判决）、旧 Applied 与携带新
recap 请求的 HitlOpened 原子提交。气泡提交确认后才可回复；失败重试复用原事件封套。
热恢复先确认停止 run 的原始起止事件，再按已提交交接事实补齐等待状态、槽位释放和 recap。
未完成判决的冷恢复由 TaskManager 持有前台槽位和 dispatch，注册真实暂停/取消令牌；
已接受的判决只恢复收尾。root、跨 agent 和同 agent 子任务的暂停均安排 report-only recap。
observe 请求通过 `context_record_ids` 固定输入记录；memory 的 `read_records` 按原 ID
读取含 superseded 的记录，并保持租户/会话隔离。装配、短段门与折叠共用这份范围；
后续收尾已经接管的段不再追加旧摘要。旧快照遗漏引用时，从原请求事件补齐。
新 user 回合的 HitlResolved 与消息回合在同一批提交，确认后才发布 claim 状态、写入暂存
memory；首字前撤销仍丢弃整批，并恢复原气泡。
熔断的 `FailureThresholdHit` 先持久化 root 失败及清场目标；在途 root 的 `TaskFailed`
必须晚于其前台 `RunFinished`。启动校准独立闭合所有遗留 run，再兑现尚未完成的熔断决定。

## 8. 记忆、回合与收尾

MemoryRecord 的 kind / scope / role 和关联 ID 是事实模型；渲染标题不是恢复协议。InMemory 与 SQL Provider 必须遵守同一归一、scope、序号、fold 和 organize 契约。

用户原文、多模态 part 与工具结果通过统一路径写入。新用户回合在 provisional 窗口暂存，装配可见；确认后才写入持久 memory 和普通事件流，弃轮丢掉暂存。不能为简化 UI 提前确认，也不能把独立后台观察误归进新回合而一起丢弃。

`task_memory_fold.py` 和 Finalize 统一负责派发框、finish 对、报告替换与父 scope 回填。观察请求固定 `context_record_ids`；折段只覆盖该段，不吞并新用户输入。晚到报告按已持久化的 finish_policy / 槽位更新，不复活已降级的正文。图片降级与 blob 引用保留各自生命周期，不能把展示文本当成图片原件。

记忆压缩的模型和预算决策在 CompactStep，存储侧原子整理在 MemoryProvider.organize；详细边界见第 3 节。

## 9. Blackboard / Topic

成功收尾按 task.id 发布 blackboard 记录，供依赖方经 `recall_topic` 拉取。BlackboardSource 读取会话订阅和任务关系对应的内容。发布、订阅声明和读取仍存在，但 driver 的自动订阅接线已是 no-op；不要恢复旧的事件通知驱动 review/reopen 旁路。

变更前核对 `core/assembler/sources/blackboard.py`、`core/loop/steps/finalize.py` 和 `test_no_blackboard_subscriptions.py`，不能仅依据早期 blackboard spec 接回旧状态机。

## 10. 初始化、装填、查询与关闭

| 入口 | 效果与禁止混入的工作 |
| --- | --- |
| 构造 | 同步公共接线；不枚举、初始化或装填历史 |
| `initialize(session_id)` | 快照 + 增量校准崩溃事实；不装填执行对象，不启动模型/任务 |
| `get_session/get_task/list_tasks/get_agent/list_agents/list_pending_hitl` | 异步 QueryResult 查询；内存优先、事件回退，不触发初始化或执行 |
| `rebuild_session` | 已有会话唯一装填入口，装填 Session / Agent / HITL / TaskManager；不 drain |
| `recover_session` | 在已校准、装填基础上恢复已有工作；输入先装入，再激活 deferred recovery |
| `forget_session` | 仅在 quiescent 时逐出运行缓存，保留历史和初始化标记 |
| `purge_session` | 显式删除执行状态并清初始化标记；与逐出缓存语义不同 |
| `close` | 拒绝新命令，停止并等待本实例执行，保存可恢复事实，返回 ShutdownReport；不替宿主关闭外部 Providers/数据库 |

会话详情与初始化报告共用 `SessionStateSummary`：goal、计数、熔断和完整公开查询对象来自同一次查询。`root_agent/agents` 使用既有 `AgentDetail`，`root_task` 和 `AgentState.current_task` 使用既有 `TaskDetail`，`pending_hitl` 使用既有 `HitlRequestView`。不另建状态包装类型；SDK 不选择某个中断原因作为产品主提示。启动校准仍使用轻量控制投影，返回时在同一提交位置按需读取任务及 HITL 正文，不重放整个会话历史。HITL 正文只从活账对应的开出事件或来源快照按 position 精确读取；启动投影不携带答复正文，也不另扫已关闭的 HITL 历史。

同一会话 initialize 共享 Task，取消一个等待者不取消校准；成功后仅保留 ID。首次及并发调用获得独立初始化结果，后续返回 None；失败可重试。宿主若写投影失败，需保存并重试那份报告。初始化缓存以单进程独占存储范围为契约，不提供跨实例分布式执行锁。

`_bind_task_manager` 保持唯一 owner，构造统一复用 SessionRegistry 工厂；会话锁内确保装填，真正执行前才激活 deferred recovery。`recover_agent`、`rebuild_hitl`、`prepare_agent` 已删除，不增加兼容回入口或全会话扫描。

纯查询返回可独立修改的副本；未装填历史任务可从日志补读。无 session_id 的列表只列已装填对象；冷 Agent/HITL 访问必须明确会话 scope。

`TurnHandle.wait_for_finish` 等任务终态及相关收尾完成，人工等待不算完成。关闭超时和持久化失败通过公开报告/异常表达，外部资源必须在执行真正静止后才能释放。

## 11. 事件、提交与快照

- 当前事件与瞬态集合定义在 `protocols/events.py`。required 策略下先由 CommitGate 确认持久化，再 fanout；best_effort 是显式策略，不能因存储失败自动降级。
- `append_batch` 同会话原子提交；batch_id 和事件封套在重试中保持原样。同 batch 不同内容应冲突，不换 ID 盲目补发。存储隔离解除前不恢复业务副作用。
- 存储按会话分配严格递增 position；事件 ID / timestamp / run sequence 都不能替代提交游标。恢复、投影及快照读取固定 committed head 切面。
- EventStore 的区间读统一为 `read_range`，支持区间、类型、任务关联及字段投影；`read_last_of_type` 读取最近类型事件，`list_session_ids` 枚举日志分区供启动校准。协议没有 append / replay / read_session_events_of_types 兼容入口。构造时的三方法有序提交检查不代表只需实现那三个方法。
- `EventReadProjection` 返回明确不完整的 ProjectedEvent，筛行在原事件上完成；不能把投影 DTO 当完整事件写回。字段裁剪用于轻量恢复/查询，不能在 Python 先读全正文再假装有界读取。
- `core/control/reducers.py` 拥有重放语义和快照版本，`snapshot_writer.py` 拥有快照时机和切面；Provider 只执行存储，不复制版本常量或判定策略。
- RoundCommitted / RoundDiscarded 是瞬态通知；SessionStatusChanged 与 TaskRecapRequested/Started/Done 仅兼容旧日志，不恢复发送。会话展示状态归宿主。
- 快照保留未结束 run 的 task、agent 与最大 sequence，RunFinished 销账；裁掉终态任务不能裁掉恢复所需执行事实。旧快照按版本/缺项规则回退，不猜测完成。

## 12. 修改定位与验证

在本仓根目录用 `.venv/Scripts/python.exe -m pytest <对应测试>`；也可使用已同步的 uv 环境。按修改范围选择现有用例，不为纯文档修改运行完整业务集。

| 修改范围 | 优先核对的回归测试（相对仓根） |
| --- | --- |
| 公开接口与分层 | `tests/unit/test_public_exports.py`、`test_runtime_public_api.py`、`test_provider_layer_boundary.py`、`test_protocols_imports.py` |
| 初始化、查询、恢复 | `tests/unit/test_runtime_initialization_queries.py`、`test_public_session_recovery.py`、`test_session_initialization.py`；`tests/integration/test_session_lifecycle_forget_rebuild.py` |
| 模型配置与流 | `tests/unit/test_model_call_config.py`、`test_agent_llm_replay.py`、`test_llm_stream_layering.py`、`test_llm_request_identity.py` |
| 任务/Agent 状态 | `tests/unit/test_task_disposition.py`、`test_same_agent_serialization.py`、`test_agent_state_machine.py` |
| 提交、position、恢复副作用 | `tests/unit/test_ordered_event_store_conformance.py`、`test_read_primitives_conformance.py`、`test_operation_recovery_policy.py`；`tests/integration/test_tool_outcome_not_rerun.py` |
| 提示词与压缩 | `tests/unit/test_prompt_responsibility_boundaries.py`、`test_prompt_builder.py`、`test_compact_step_transitions.py`、`test_compact_wait_observe.py` |
| Observe / HITL / 停止 | `tests/unit/test_observe_lifecycle.py`、`test_observe_recovery_ownership.py`；`tests/integration/test_foreground_observe_interrupt.py`、`test_cold_reply_settle_race_e2e.py` |

跨仓修改执行协议时，先在本仓修复并验证，再按宿主的 `dev/sync_ctx_weft.py` 流程更新 wheel 和运行环境，最后跑宿主对应回归；只改源码仓不会更新宿主已安装副本。纯文档修改无需重建 wheel。
