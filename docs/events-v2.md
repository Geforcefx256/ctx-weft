# 事件体系 V2 · 事件清单

> 这是 V2 定案后的**事件全集**：留哪些、每个长什么样、代表什么。
> 状态：**会话状态所有权那一段（§2.1 / §2.3 / §2.4 / §3.3 / §5.2）已于 2026-09-02
> 落地**——`SessionStatusChanged` 停发进 L 档，`Session*` 四条、`TaskAwaitingHuman`、
> `RunInterrupted`、三条 `TaskQueue*` 均已接线；`SessionStatus` 值域已收敛为 6 个。
> host 侧的破坏性变更见 `docs/upgrade/2026-09-02-session-status-ownership.md`。
> 本文其余各节仍是**已定案、未实施**的改名/合并。
> 表中「存量名」列非空者，代码里现在还叫旧名字。
>
> 真相源：`src/ctx_weft/protocols/events.py`（类型）、各发射点（payload）、
> `src/ctx_weft/core/control/reducers.py`（S 档的状态含义）。
> 决策依据与迁移工序见 `docs/events-v2-taxonomy-change-record.md`。
>
> **已知过期（2026-09-03 agent-centric 改造，未随本次重命名一并补齐，记账用）**：
> §2.1（`Session*` 四条状态事件：`SessionInterrupted`/`SessionWaiting`/
> `SessionRunning`/`SessionFinished`）随会话状态机（`session_state.py`）整体退役
> **全部停发**，已转入 L 档（`EventType.L_TIER_EVENT_TYPES`，见
> `src/ctx_weft/protocols/events.py` Task 16 处的内联注释）——§2.1.1 的「唯一持有者」
> 表述已随之作废（见该节的更正说明）；这 4 个已随 2026-09-04 的 L 档核账一并补进
> §5（见 §5.5）；§2.5（`Agent`）仍**尚未**补上新增的 5 个 `AGENT_*` 状态事件
> （`AgentRunning`/`AgentIdle`/`AgentWaitingHuman`/`AgentInterrupted`/
> `AgentTerminated`，`AgentLifecycleManager.apply_input` 发出）——这是本文档自身
> 滞后于代码的已知缺口，不是本次重命名或 2026-09-04 核账的范围。当前权威状态见
> `docs/upgrade/2026-09-03-agent-centric-interaction.md` 第 5 节。
>
> **2026-09-05 · 退役闸门第 2 级已对 6 个类型执行**：`SessionRunning` /
> `SessionWaiting` / `SessionInterrupted`（§2.1.2、§5.5）与 `TaskQueueBlocked` /
> `TaskQueueInterrupted` / `TaskQueueDrained`（§3.3、§5.6）**已连枚举一并删除**，
> 不再是 L 档。判据：这 6 个类型生于 2026-09-02、死于 09-03/09-04，全部落在 `master`
> 之后的分支内部——`master` 的 `EventType` 里从来没有它们，任何从 master 迁移来的
> 事件流都不可能含有这些字符串，§5.7 的闸门条件因此天然成立，无需等归档周期。
> 受影响的正文段落（§2.1.2 的三行、§2.1.3 的拆解表、§2.1.4 的转移表、§3.3 的三行）
> 各自就地标注，**未逐段重写**——它们记录的是那次重构的推理，仍有阅读价值。

**总量：55 个在用 + 14 个只读存量（L 档）——上面这条 §2.5 缺口未反映在这个数字里。**

---

## 0. 信封：所有事件共有的字段

`payload` 之外的字段对所有类型一律相同，**不因类型而异**。

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | `str` | `evt_<ULID>` |
| `run_id` | `str \| None` | 一次 loop run；session 级事件为 `None` |
| `sequence` | `int` | 同一 run 内单调递增；run 外的事件为 `0` |
| `session_id` | `str` | |
| `type` | `str` | 本文第 2、3 节的取值之一 |
| `timestamp` | `datetime` | |
| `tenant_id` | `str` | 默认 `"default"` |
| `task_id` | `str \| None` | |
| `agent_id` | `str \| None` | |
| **`origin`** | `str` | **V2 新增**：哪个组件发出的，见 §4。存量事件读出空串 |
| `payload` | `dict` | 按类型而异，即下面各表的「payload」列 |
| `metadata` | `dict` | |
| `causation_id` | `str \| None` | |
| `schema_version` | `int` | payload 版本，默认 `1` |

**envelope 管身份，payload 管内容。** payload 里不再重复 `task_id` / `agent_id`
（V2 的清理动作之一）。唯一例外是 `TaskRecapStarted/Completed`——它的 `task_id`
是**被折叠的那个 task**，未必等于 envelope 的 `task_id`。

---

## 1. 分档：三个档 + 一个尾巴

| 档 | 判据 | 可改动程度 |
|---|---|---|
| **S** · 30 个 | 有状态消费者（reducer / host 投影）折叠它 | 语义冻结；改名须走别名表；payload 只可**加**字段 |
| **O** · 25 个 | 已发射、无状态消费者，纯观测/展示 | 可重命名 / 合并 / 删除，与 host SSE 同步即可 |
| **L** · 14 个 | 曾发射、现已停发、重放仍须认识 | **只读**。不得再发射；删除须过退役闸门（§5） |
| X · 0 个 | 从未发射 | V2 已清空——定义即必须发射 |

其中 3 个是 **TRANSIENT**（⊂ O）：`LLMTokenStreamed` / `LLMReasoningStreamed` /
`LLMRetryTriggered`。高频流式 delta，只进实时订阅，**不进持久化 / 投影 / 快照**；
单一真相由 `LLMResponseFinished` 承载。

---

## 2. S 档 · 30 个（承载状态）

### 2.1 Session · 6

#### 2.1.1 所有权：本节描述的模型已被 2026-09-03 agent-centric 改造取代

> **过期提醒**：以下描述的是 2026-09-02 落地、2026-09-03 又整体推翻的架构——
> `session_state.py` 连同它的状态机已被删除，`SessionManager` 随之降格为
> `SessionRegistry`（`src/ctx_weft/core/orchestrator/session_registry.py`），
> **不再持有、也不再改写任何会话状态**，只剩「这个 session 里登记了哪些
> agent」这一件事（`register_session`/`agent_ids_of`）。「会话状态」这个概念
> 本身不再存在：状态整体搬到了各 agent 自己身上，由 `AgentLifecycleManager`
> 的五态机（idle/running/waiting_human/interrupted/terminated）持有。host 若
> 需要「这个会话整体是什么状况」，自己聚合该 session 下各 `AgentSummary.status`
> （`CtxWeftRuntime.list_agents`），系统不再预先算好广播。下文表格与流程图仍
> 保留供历史对照，但**不代表当前行为**；当前权威说明见
> `docs/upgrade/2026-09-03-agent-centric-interaction.md` 第 2、5 节。

以下是取代前（2026-09-02 版）的模型，仅供理解演进脉络：

**`SessionRegistry`（原 `SessionManager`）曾是会话状态的唯一持有者与唯一改写者。**
其余组件一概不写会话状态，它们只发自己领域的事实。

事件因此分成两类，**这是本节（历史版本）的全部要点**：

| | 谁发 | 例子 | 对会话状态 |
|---|---|---|---|
| **领域事实** | 各组件发自己领域的事 | `HitlOpened` `TaskAwaitingHuman` `RunInterrupted` `TaskStarted` `TaskFinished` `RunFinished` … | **一概不写**。reducer 里这些事件的 `session_status` 写入全部删除 |
| **会话状态事件** | 只有 `SessionRegistry` | 本节表里这 6 条 | 唯一的写入者 |

#### 严格分层：每层只跟下一层说话

```
loop   ──►  这次执行被打断了              RunInterrupted{reason}          ← run 域
              │  RunOutcome（发生了什么，不含 task 该变成什么）
              ▼
TM     ──►  task 挂起了、在等人           TaskAwaitingHuman{hitl_id}      ← task 域
TM     ──►  这个 task 停在 INTERRUPTED    TaskInterrupted{reason, …}      ← task 域
              │
TM     ──►  我这边没有能跑的了，因为 X     TaskQueueBlocked / …Interrupted / …Drained
              │
SM     ──►  会话状态                      SessionWaiting / SessionInterrupted / …
```

> **task 状态事件（`TaskAwaitingHuman` / `TaskInterrupted` 及本节其余 9 条）现在只从
> `TaskManager` 发**（task 状态所有权重构，2026-09-02）。loop 五个结局点只交出
> `RunOutcome`，`TaskManager.apply_run_outcome` 据 `disposition_for` 判定后才写状态、
> 发事件——发生在 run 返回之后，故这些事件不带 `run_id`，且相对 `RunFinished` 的顺序
> 是「之后」不是「之前」。

> **`RunInterrupted` 与 `TaskInterrupted` 是两件事，故是两条事件**（2026-09-02 收尾修正）。
> 「这次执行死了」是 run 域的事实，无条件发；「这个 task 停在 `INTERRUPTED` 等 `/resume`」
> 是 task 域的结论，**发在重试判定之后**——崩溃后还能原地重试的那一支发的是
> `TaskRequeued`（→ `PENDING`）。一条事件做两份工作，正是本次重构从 `RunFinished`
> 身上拆掉的那个模式。

**HITL 不决定会话状态，它决定 task 状态**——一个 task 被挂起、需要人来解决。
**SM 只看 TaskManager 的信号**，不订阅 HITL 事件、不读 `HitlRegistry`。

于是 SM 的**查询依赖为零**：它不问任何人任何事，需要的结论都由 TM 的信号带过来。

**没有任何组件调用 SM 改状态。** 组件对 SM 只有查询依赖（`status_of`，用于自己的准入
判断）；改状态一律走「发事实 → 总线 → SM 判定」。SM 的方法调用只剩真正的外部命令
（`create` / `resume` / `cancel` / `recovered`）——那是 host 经 runtime 进来的请求，
不是组件在驱动 SM。

这条查询是重构前那堆散落守卫的正确归宿，**已一并收掉**：

| 重构前在哪 | 手工做的判断 |
|---|---|
| `task_manager.py` | `if self._session.status not in ("FAILED", "CANCELED")` |
| `task_manager.py` | `if self._session_done_fired: return` |
| `task_manager.py` | `if not self._is_current(): return`（被顶替的旧 TM 不得代表会话发言） |
| `reducers.py` | `if view.session_status in PAUSED_STATUSES`（`HitlResolved` 的「不覆盖已到终态」） |
| host 投影 | 「不得覆盖已落终态」的守卫 |

它们各自维护一份对会话状态的判断，口径不同、位置分散——这正是通用 setter
`SessionStatusChanged` 存在的土壤。收进 SM 之后，「已终态就不再转移」写在状态机里一次。

> **前置改造已完成**：`SessionRegistry` 从前是 `runtime.py` 里 `new` 出来用完即弃的
> dataclass，无状态、不订阅事件；现在是 runtime 级的长生命周期组件，订阅 TM 的四类信号。

#### 2.1.2 会话状态事件 · 6

| 事件 | 存量名 | payload | 含义 · 状态效果 |
|---|---|---|---|
| `SessionCreated` | | `template_id` `user_prompt`（jsonable，保 ref 不落字节） `root_agent_id` `llm_model` `llm_account` `tenant_id` `token_budget` `context_limit` `reserved_output_tokens` | 会话诞生。建 `SessionView`，`status=RUNNING` |
| `SessionResumed` | | `user_prompt` `root_agent_id` `llm_model` `llm_account` | 在已有会话上**开新一轮**：带新的 user_prompt 建新 root task。前置拒绝有未终结任务的会话。`status` 回 `RUNNING`，并把 `SessionView.user_prompt` 覆写成本轮的 |
| ~~`SessionWaiting`~~ | | （空） | **已删除（2026-09-05）**。原义：会话停着但**正常地停**——所有任务都在等人 / 等外部输入 → `WAITING` |
| ~~`SessionInterrupted`~~ | | `reason` | **已删除（2026-09-05）**。原义：会话停着，**异常**——系统故障，等 `/resume` → `INTERRUPTED` |
| ~~`SessionRunning`~~ | | `reason` | **已删除（2026-09-05）**。原义：会话（重新）开跑 → `RUNNING` |

> **上面三行已连枚举一并删除**（见文首 2026-09-05 记账）：会话状态机随
> `SessionRegistry` 降格整体退役后它们就无发射点，而 `master` 从未有过这三个类型，
> 存量流里也不会有。会话级别的展示状态现在不由 core 预先算好广播，由 host 按 agent
> 状态（`AGENT_*`）自行聚合推导。**`SessionCreated` / `SessionResumed` /
> `SessionFinished` 三条不受影响**，其中 `SessionFinished` 仍在 L 档。
| `SessionFinished` | | `final_status` | **唯一的会话终态事件**。全部任务终态，或 `cancel_all` 硬取消。→ `final_status` |

> **`SessionStatusChanged` 已删除**（进 L 档，§5）。它是事实流里唯一的命令式事件——别的都说「发生了什么」，只有它说「把状态写成 X」，于是 6 个发射点混着三类完全不同的东西。拆解如下：
>
> | 原发射点 | 现在 |
> |---|---|
> | run 崩溃 → `INTERRUPTED` | loop 发 `RunInterrupted`，TM 聚合后发 `TaskQueueInterrupted`，SM 发 `SessionInterrupted` |
> | LLM outage → `INTERRUPTED` | 同上，`reason="llm_outage"` |
> | 恢复期有未决 HITL → `PAUSED` / `PAUSED_HITL` | TM 重建后发 `TaskQueueBlocked` → `SessionWaiting` |
> | 恢复期无未决 HITL → `INTERRUPTED` | TM 重建后发 `TaskQueueInterrupted(reason="process_restart")` |
> | `cancel_all` → `CANCELED` | `SessionFinished(CANCELED)` |
> | 会话完成 / 熔断时**早报终态** | **删除，不替换**（见下） |
>
> **删掉「早报终态」这个功能。** 原来 `on_task_finished` 和 `_trip_failure_threshold` 会先发一条
> `SessionStatusChanged(final_status)`，再 `await gather(后台任务)`，最后才发载荷完全相同的
> `SessionFinished(final_status)`——目的是让前端立刻看到终态、同时 SSE 保持开放等后台协程收尾。
> 代价是同一个事实在流里出现两遍，且事件模型里看不出谁是早报、谁是收尾。
>
> **前端改为监听 act step 结束或 task 结束**来提前反映「这一轮跑完了」，不再依赖一条抢跑的会话级事件。

> 「复活」不是一种状态，是一个动作，所以**没有 `SessionRecovered` 这个类型**。
> 复活时 TM 重建任务集合后照常发它那三条信号之一，SM 收到什么就转到什么——
> 恢复路径和正常路径走同一条链，不是两套。

#### 2.1.3 状态机

**会话状态回答的不是「卡在哪」，而是「这个会话现在健康吗、还能不能自己往前走」。**
「卡在哪」是 task 的事——会话有多个 task，其中一个在等人不等于会话在等人。

| 状态 | 含义 | 谁能让它动 |
|---|---|---|
| `RUNNING` | 有 task 在跑 | 自己 |
| `WAITING` | 停着，但是**正常地**停——在等人、等外部输入 | 用户 |
| `INTERRUPTED` | 停着，**异常**——系统故障 | 运维（`/resume`） |
| `SUCCEEDED` / `FAILED` / `CANCELED` | 全部 task 终态 | — |

`WAITING` 与 `INTERRUPTED` 的区分留在会话级，因为**「正常地停」和「异常地停」是会话级判断**
（这个会话是不是出事了，监控与告警看它）。而「等的是审批面板还是一句话」不是——那个信息的
源头是 `HitlOpened.delivery`，前端渲染面板时已经拿到，会话状态再复制一份只会制造第二个副本。

SM 的输入只有四类，全部来自 TaskManager，每一类都是一个独立类型——
收到哪条就转到哪个状态，**没有任何判据、不读任何字面量**。

| 当前状态 | 收到 | 新状态 | SM 发出 |
|---|---|---|---|
| （无） | `create_session` 命令 | `RUNNING` | `SessionCreated` |
| 任一非终态 | `TaskQueueBlocked{count}` | `WAITING` | `SessionWaiting`（payload 空） |
| 任一非终态 | `TaskQueueInterrupted{reason}` | `INTERRUPTED` | `SessionInterrupted(reason)` |
| `RUNNING` | `TaskQueueDrained{final_status}` | `final_status` | `SessionFinished(final_status)` |
| `WAITING` | `TaskStarted` | `RUNNING` | `SessionRunning("human_replied")` |
| `INTERRUPTED` | `TaskStarted` | `RUNNING` | `SessionRunning("resumed")` |
| `INTERRUPTED` | `resume_session` 命令（新一轮） | `RUNNING` | `SessionResumed` |
| 非终态 | `cancel_session` 命令 | `CANCELED` | `SessionFinished("CANCELED")` |
| `WAITING` / `INTERRUPTED` | `TaskQueueDrained` | 不变 | 无 |
| 任一终态 | **任何输入** | 不变 | 无 |

> **「重新开跑」用已有的 `TaskStarted`，不新造类型。** 有 task 开始跑，会话就在跑——
> 这是结构性的，不需要 TM 再发一条「我又有活了」。

> **会话状态事件不带展示数据。** `TaskQueueBlocked` 的 `count` 停在 TM 那一层，
> 不穿过状态机——让展示需求进入状态判据，正是本次删掉 `needs_panel` 的那条理由。
> host 要在徽标上显示「几个任务在等你」，读 `TaskQueueBlocked{count}`：它比会话事件
> 更早到达，且是那个数字的原产地。

> **`SessionRunning.reason` 由「从哪个状态回来」定**（`WAITING` → `human_replied`，
> `INTERRUPTED` → `resumed`）。这是溯源，不是判据——SM 转移到 `RUNNING` 这件事本身
> 不依赖它。

> **`needs_panel` 整条传递链不存在了。** 早先的方案让 `TaskAwaitingHuman` 带
> `needs_panel`、`TaskManager` 对全部等人任务取或、再随信号上到 SM——三层传递一个
> 纯 UI 事实，而每一跳都是一次可能失步的复制。删掉它同时消掉了一个真实缺陷：
> `task.awaiting_needs_panel` 是单个 bool，同一 task 若 park 两次会**后写覆盖前面的**，
> 等于把「降级污染」从会话层搬到了 task 层。字段没了，缺陷无从谈起。

> **`SessionStatus` 已收敛为 6 个**（2026-09-02）：`RUNNING` / `WAITING` /
> `INTERRUPTED` / `SUCCEEDED` / `FAILED` / `CANCELED`。删掉的四个：`QUEUED` 和
> `TIMEOUT` 是死值——定义在 `models.py`、全仓从未被赋值过（`grep` 只命中 capability
> provider 的错误码 `"TIMEOUT"`，是另一回事）；`PAUSED` / `PAUSED_HITL` 合并成
> `WAITING`。状态机是判据——状态机表里没有的状态就不该在值域里，同 §6 不变式 1 对
> 事件的要求。值域由 `tests/unit/test_session_status_domain.py` 钉住。
>
> **`TaskStatus` 相反，是不够用**：一个 `SUSPENDED` 盖住「等子任务」/「等人」/「被打断」
> 三件事，逼得消费方去匹配 `TaskSuspended.reason`。已拆成
> `SUSPENDED` / `AWAITING_HUMAN` / `INTERRUPTED` 三个值（§2.3）。

> **并行 task 状态不同时怎么办**（TaskManager 侧的两层判断）：
> 1. **有活在跑就什么都不报**——task_1 在跑、task_2 在等人 → 会话保持 `RUNNING`。
>    会话确实在推进。代价是那个停着的 task 在**会话级看不见**，直到全都停下来；
>    它的 task 级状态前端照样看得见。
> 2. **全停下来后按「解开它需要谁」排优先级**：`INTERRUPTED`（运维）>
>    `AWAITING_HUMAN` / `SUSPENDED`（用户）> 全终态。一个 task 断了、另一个在等人，
>    先报「断了」——人答完了那个断的还是断的。

### 2.2 Step · 2

| 事件 | payload | 含义 · 状态效果 |
|---|---|---|
| `StepStarted` | `step_name` | 进入某个 step。`view.current_step = step_name` |
| `StepCompleted` | `step_name` `next_step` | 该 step 结束并给出下一步。`view.current_step = next_step` |

> `step_name` 与新增的 `origin` 语义重复，但 reducer 在读它，**保留不动**。

### 2.3 Task · 12

| 事件 | 存量名 | payload | 含义 · 状态效果 |
|---|---|---|---|
| `TaskCreated` | | `task: {id, session_id, status, title, description, creator_agent_id, assigned_agent_id, parent_task_id, user_prompt, priority, max_retries, timeout_ms, dag_deps, interaction_mode, unattended, origin_tool_call_id, origin_tool_name, settings}` | 任务入队。建 `TaskView` |
| `TaskStarted` | | `assigned_agent_id` | → `ACTIVE`，并回填 `assigned_agent_id` |
| `TaskSuspended` | | `summary` `spawn_titles` | **只剩「等子任务完成」这一个语义**。→ `SUSPENDED`（非终态） |
| `TaskAwaitingHuman` | | `hitl_id` | 这个 task 被 HITL 挂起、需要人来解决。→ `AWAITING_HUMAN` |
| `TaskInterrupted` | | `reason` `error_code` `error_message` `retry_count` | 这个 task 停在 `INTERRUPTED`，等 `/resume`。→ `INTERRUPTED` |
| `TaskResumed` | | `{}` | 阻塞的子任务全部终态，父任务解除挂起 → `PENDING` |
| `TaskHumanResolved` | | `hitl_id` | 挡住这个 task 的那个 HITL 有了结果，与 `TaskAwaitingHuman{hitl_id}` 配对，同一个 `hitl_id` 括起被挡住的区间 → `PENDING` |
| `TaskFinished` | | `outcome="success"` `summary` `outputs` | → `FINISHED`，并把会话的 `failure_counter` 清零 |
| `TaskFailed` | | `error_code` `error_message` `retry_count` | → `FAILED`，`failure_counter += 1`。`error_code=TASK_FAILED_BY_THRESHOLD` 是熔断的聚合结果，**不计数** |
| `TaskCanceled` | | `reason` | → `CANCELED` |
| `TaskRequeued` | | 两种形状：<br>· observer 判重试：`outcome="retry"` `summary` `retry_count`<br>· reopen：`reason` `user_prompt` `original_user_prompt` | → `PENDING`，清空 `outputs`；reopen 还会覆写 `user_prompt` |
| `TaskMessageAppended` | | `memory_id` `agent_id` `content` `source` `timestamp` | **只记事实、不改状态**。一条用户消息被并进这个 task 的对话（`send_message` 的注入分支）；`content` 是 event 侧形态（图走 event blob ref）。消息进 memory 要等这一轮提交，而暂存区在事件补投**之后**才落盘——崩在两者之间时，恢复期据此把消息补回 memory |
| `TaskOutcomeRecorded` | `TaskFinalized` | `task_id` `outcome` `outputs: {output, summary}` `error` | **只记结果，不改状态**：状态转移是 `TaskFinished` / `TaskFailed` 的事。交付物两段分开送（2026-09-05）：`output` = **答复正文本身**（收尾回合的 assistant 正文），`summary` = agent 调 `finish_task` 时给的 `deliverables_summary`，即写给 reviewer 的交付物自评清单。host 打印「最终答复」只该取 `output`——两段混成一串正是自评清单被当答案打出来的成因。判据以 `task.outputs` 为准绳：retry 驳回把它置空时两段一起归空，不送已被驳回的废稿。`output` 恒为**纯文本**（多模态交付物走 blob store，不进事件载荷）。reducer 侧仍不折这两个字段，投影的 `outputs`/`error` 只认 `TaskFinished`/`TaskFailed` 一个来源；本字段的消费者是 host 的 `tasks` 表与 CLI |

> **`TaskSuspended` 从三义收窄到一义。** 它从前靠 `reason` 字面量区分「等子任务」/
> 「在等人」/「崩了」三件语义完全不同的事，消费方只能匹配字符串分流——那正是 HITL 旧实现
> 按 `form == "wait"` 判暂停态的同一种病。现在三者各有类型：`TaskSuspended`（等子任务）、
> `TaskAwaitingHuman`（等人）、`TaskInterrupted`（被打断；run 域那一半是 `RunInterrupted`，见 §2.4）。
>
> **`TaskInterrupted` 的发射时机是「重试判定之后」，且只从 `TaskManager` 发**（task 状态
> 所有权重构，2026-09-02）。LLM outage 与 run 崩溃两条异常路径都不在 `runtime._run_loop`
> 里发它：`_run_loop` 只无条件发 run 域的 `RunInterrupted`（§2.4），随后把
> `RunOutcome(INTERRUPTED, retriable=...)` 交回（outage 硬编码 `retriable=False`，崩溃取
> `getattr(exc, "retriable", True)`，两者不同源、不许合并）——`TaskManager.apply_run_outcome`
> 用同一张 `disposition_for` 表判定：`retriable` 且预算未尽 → `TaskRequeued`（→ `PENDING`）；
> 否则 → 本条（→ `INTERRUPTED`）。发早了，走重试的 task 会先被打成 `INTERRUPTED` 再翻回
> `PENDING`，中间那一下是假的——这正是两支路径都要等 `disposition_for` 判完才发的原因。
> `TaskManager._handle_task_failure` / `_suspend_task_interrupted` 仍存在，但只服务**装配
> 失败**这一条路（装配阶段没有 run、没有 `RunOutcome` 可喂）；执行阶段的崩溃自 Task 4 起
> 不再经过它们。
>
> 同理，`TaskStatus` 值域也从一个 `SUSPENDED` 拆成三个：
> `SUSPENDED` / `AWAITING_HUMAN` / `INTERRUPTED`。**状态值域的过载是事件过载的根**——
> 只拆事件不拆状态，消费方仍然分不清。

> **一个 task 同时最多一个「挡住它」的 HITL**，由 `act.py` 的 tool call 循环保证：
> 它是串行的（`for i, tc in enumerate(tool_calls)`，逐个 `await`），第一个 park 就
> `raise HitlPark` unwind 整个 run，后面的 tool call 根本没执行。
> `TaskAwaitingHuman` 因此描述的是**挡住这个 task 的那一个请求**——`HitlPark.hitl_id`
> 就是它，唯一确定。
>
> 同一 task 在 `HitlRegistry` 里仍可能有**多条未决**（热等待中的、background observe
> 并发开的、跨升级边界的存量），但它们**不挡这个 task**，不影响 task 状态。
>
> ⚠️ 这条不变式是**结构性巧合**：哪天有人把 tool call 改成 `asyncio.gather` 并发执行，
> 它就静默破坏。必须有测试直接钉住「两个都要审批的 tool call → 只有第一个执行」。

> ⚠️ **`TaskOutcomeRecorded` 有一处发射侧与消费侧对不上**：reducer 读
> `payload["outputs"]` / `payload["error"]`，而 `finalize.py` 只发 `{task_id, outcome}`，
> 于是回放后 `TaskView.outputs` / `.error` 恒为 `None`（只有 `finished_at` 落到了）。
> 改名接线时一并修：要么发射侧补这两个字段，要么消费侧改读 `TaskFinished.outputs`。

### 2.4 Run · 1

| 事件 | payload | 含义 · 状态效果 |
|---|---|---|
| `RunInterrupted` | `reason` `error_code?` `error_message?` | 这次执行被**外部原因**打断（`llm_outage` / `run_crash`），不是任务逻辑决定的停。**不写 task 状态** |

> **这是 run 唯一真正拥有的事实。** §3.2 说 run 没有自己的状态——那是对的，
> `RunStarted` / `RunFinished` 一直在替 task 和 session 说话。但「这次执行非正常终止」
> 确实是 run 层的事：task 没有选择停下，是执行环境塌了。
>
> `RunFinished` 照发（关流，`outcome="interrupted"`）；`RunInterrupted` 说**为什么**。
> 消费方靠**类型存在与否**判断，不读 `reason`——`reason` 只是溯源。
>
> **`RunFinished.outcome`（Task 3，2026-09-02）是权威字段**，值 ∈ run 词表五值
> （`completed` / `awaiting_human` / `suspended_on_children` / `interrupted` / `canceled`，
> `RunOutcomeKind`）——说的是**这次执行自己**怎么收场，不是 task 状态。旧键
> `final_status`（装 `task.status`）**已废弃，保留一个发布周期**供旧断言过渡，
> 下个周期随其余旧路径一并删除（Task 4）；host 应改读 `outcome`。
>
> **它只由 `runtime._run_loop` 发**，两支（LLM outage / run 崩溃）各发一次，**无条件**：
> 那次 run 确实死了，与 task 后续是重试还是挂起无关。run 域另外三条
> （`RunStarted` / `RunCanceled` / `RunFinished`）同源。从前它还有一个发射点在
> `TaskManager._suspend_task_interrupted`——那在 run 外面，TaskManager 不拥有 run、
> 连 `run_id` 都拿不到，发出来的是一条自称属于某次 run 却没有 `run_id` 的事件；
> 已改发 task 域的 `TaskInterrupted`（§2.3）。
>
> **它也不再写 task 状态**：已从 `TASK_STATUS_BY_EVENT` 移出，那份工作归
> `TaskInterrupted`。理由见 §2.3——run 死了不等于 task 停在 `INTERRUPTED`。

### 2.5 Agent · 2

| 事件 | payload | 含义 · 状态效果 |
|---|---|---|
| `AgentInstantiated` | `template_id` `template_version` `llm_account` `llm_model` | **事件流里唯一记录「该 agent 用哪个模板」的地方**——树形结构推算不出模板，而授权按模板做策略。冷 resume 拿不到它，子 agent 就会顶着 root 的模板身份 |
| `AgentLlmChanged` | `llm_account` `llm_model` `reason` | 换模型：写 `AgentView` 的 `llm_account` / `llm_model` 两个字段，不碰任何 task / session 状态 |

### 2.6 Context · 1

| 事件 | 存量名 | payload | 含义 · 状态效果 |
|---|---|---|---|
| `PromptAssembled` | `PrepareCompleted` | `estimated_tokens` `assembled_token_count` | 本轮 prompt 装配完成的 token 数 → `view.assembled_prompt_tokens`。阶段标记职责已由 `StepCompleted(origin=loop.prepare)` 接走，它现在纯粹是这个数的载体 |

### 2.7 Act · 1

| 事件 | payload | 含义 · 状态效果 |
|---|---|---|
| `ActTurnCompleted` | `turn` `reason` | 一轮 act 结束 → `view.transcript_turns = turn`。`reason` ∈ `stop`（纯文本收尾）/ `tool_calls_processed` / `await_user`（纯文本暂停等人）/ `context_limit` |

### 2.8 HITL · 2

8 个旧事件收敛到这 2 个。**`outcome` 是事实本身，不再由事件类型编码结局**——
host 自定义结局因此不必新增事件类型。

| 事件 | payload | 含义 · 状态效果 |
|---|---|---|
| `HitlOpened` | `hitl_id` `form` `delivery` `subject_id` `prompt` `detail` `fields` `proposal` `tool_call_id` `stage` `invocation_key` `agent_id` `resume_state` `reply_as_result` | 一个请求**进入**未决集合 |
| `HitlResolved` | `hitl_id` `outcome` `claimed` `message?` `modified_arguments?` | 一个请求**离开**未决集合（终局） |

> **HITL 不决定会话状态，它决定 task 状态**（§2.1.1）。这两条事件的职责到「哪个 session
> 有哪些未决请求」为止；一个请求真的挡住了任务，是 park 时发的 `TaskAwaitingHuman`
> 说的事；会话在不在推进，是 TM 聚合后的 `TaskQueueBlocked` 说的事。
> **SM 不订阅这两条，也不读 `HitlRegistry`。**
>
> 它们仍是 S 档——`fold_hitl_snapshot` 靠它们重建未决集合，那是崩溃恢复的真相源。

`HitlOpened` 各字段：

| 字段 | 取值 / 含义 |
|---|---|
| `form` | `approval` / `question` / `wait`，**开放 `str`**，host 可自定义 |
| `delivery` | 决定人的答复怎么回灌，**封闭三选一**：<br>`{kind:"tool_result", tool_call_id}`——回灌成工具结果<br>`{kind:"user_turn", task_id, preface}`——回灌成一轮用户发言，`preface` ∈ `normal`/`interrupt`/`interrupt_edit`<br>`{kind:"no_resume"}`——不回灌任何对话 |
| `stage` | 请求发生在哪一段（授权阶段 / 工具执行中） |
| `invocation_key` | 同一 `tool_call_id` 下**这一次**调用的标识。少了它，同 id 的另一次调用会被上一次的批准放行 |
| `resume_state` | provider 自带的续跑状态。**只省掉热重入**——冷路径会重调 `invoke` |
| `reply_as_result` | 人的答复直接就是结果（`ask_user` 那一类），不发生重入 |
| `subject_id` / `prompt` / `detail` / `fields` / `proposal` | 问的是谁 / 问什么 / 补充说明 / 要人填的字段 / 待批准的参数 |

`HitlResolved` 各字段：

| 字段 | 取值 / 含义 |
|---|---|
| `outcome` | `accepted` / `rejected` / `cancelled`，**开放 `str`**。旧的 approved vs modified 之分改由「有没有 `modified_arguments`」推出 |
| `claimed` | 决定是否被一个**还活着的热等待者**接走。`false` = 走冷续跑。这是权威判据，不是「已终局」 |
| `message` | 人写的话。内容已外部化为 memory ref 时，事件里存的是**原始**内容而非 ref（event store 解不开 ref） |

**「要不要出审批面板」是 `delivery` 的性质，只有前端需要它，不上升到任何状态：**

| delivery | 前端 | 场景 |
|---|---|---|
| `user_turn` | 输入框，无审批面板 | 只是等用户说话 |
| `tool_result` / `no_resume` | 审批面板 | 有一个决定悬着 |

前端渲染面板时读的就是这条事件的 `delivery`。**会话状态不复制它**——早先的方案让它
经 `TaskAwaitingHuman` → `TaskQueueBlockedOnHuman` → `SessionAwaitingHuman` 三层传递，
那是同一份信息的三个副本，每一跳都可能失步（§2.1.3）。

> 旧实现按 `form == "wait"` 字面量判定暂停态，host 自定义的 form 一律落错。
> 新模型里**会话状态**这一侧的判定整个不存在了：`PAUSED` / `PAUSED_HITL` 已从
> `SessionStatus` 值域删除，合并成单一 `WAITING`。
>
> **2026-09-04（runtime 对外面 agent-centric 对齐 Task 14）起，两者都已删除**：
> `core/hitl/status.py::paused_status_for` 与它唯一的消费方
> `CtxWeftRuntime.session_status_after_recover` 一并删掉了（上面那条"改名待办"因此
> 也一并消失，不再需要）。它们原来答的是**面板提示**（panel hint：按 `delivery`
> 判、不按 form，**不是 `SessionStatus` 值**）这个问题，现在由 host 直接读
> `HitlRequestView.delivery` 自判——同一份原始事实，少一层会漂移的派生。

> **热等待窗口期间会话仍是 `RUNNING`**（2026-09-02 定案的行为变更）。热等待时 task
> 真的还在跑——阻塞在一个 `await` 里，和阻塞在一次 LLM 调用上没有区别，没有 park、
> 没有 `TaskAwaitingHuman`、TM 什么都不知道。只有热窗口耗尽降级成冷 park 时会话才变
> `WAITING`。旧行为是 `HitlOpened` 一到就立刻翻成 `PAUSED_HITL`。
>
> 前端的审批面板由 `HitlOpened` 驱动，**不受影响**；变的只是会话徽标会晚一点。

### 2.9 Intent · 1

| 事件 | 存量名 | payload | 含义 · 状态效果 |
|---|---|---|---|
| `IntentRecognized` | `RecognizeIntentToolCall`（并入 `RecognizeIntentCompleted` 的 `usage`） | `title` `description` `session_goal` `usage` | 意图识别的产出 → `session.goal`、root task 的 `title` / `description`（root task 创建时无名，靠它并发补填）。**空值不覆盖已有值** |

### 2.10 TaskRecap · 2

| 事件 | 存量名 | payload | 含义 · 状态效果 |
|---|---|---|---|
| `TaskRecapStarted` | | `task_id` `boundary` `agent_id` | 后台 recap 开始。**有 Started 而无 Completed = 该段 memory 写没落完**（崩在中途），恢复据此重跑 |
| `TaskRecapCompleted` | `TaskRecapDone` | `task_id` | 该段 recap 收尾。与逐轮流式事件不同，这两条是「整段起止」的持久记账 |

---

## 3. O 档 · 29 个（纯观测 / 展示）

删改只须与 host SSE 同步，不影响重放与投影。

### 3.1 Step · 2

| 事件 | payload | 含义 |
|---|---|---|
| `StepFailed` | `step_name` `error_code` `error_message` | step 抛异常（异常随后原样上抛，不吞） |
| `StepSkipped` **新增** | `step_name` `reason?` | 步骤被跳过。把「跳过」变成通用概念，下一个需要它的 step 不必再自造类型 |

### 3.2 Run · 2

`RunStarted` 旧模型里写的是 `task_status` 与 `session_status`——这两条事件从来没描述过
task/session 自己，一直在替它们说话。剥掉那些越界的写入之后，reducer 里什么都不剩，
判据上就落在 O 档。**`RunFinished` 现在确实说了 run 自己的事**——`outcome`
（Task 3，见 §2.4）是 run 词表五值之一，不再是 `task.status` 的镜像；旧键
`final_status`（仍装 `task.status`，`runtime.py:2578` 附近）保留一个发布周期后随
Task 4 删除。

run 真正的用处是**事件流的分段与 SSE 的开关**：`run_id` 把一次执行的事件聚成一组，
`sequence` 在组内单调递增。

> **例外（Task 4 起）**：七条 task 状态事件由 `TaskManager` 在 run **之外**发出
> （`TaskFinished` / `TaskFailed` / `TaskRequeued` / `TaskSuspended` /
> `TaskAwaitingHuman` / `TaskInterrupted` / `TaskCanceled`，另加 `TaskStarted` /
> `TaskResumed` 这两条本来就由 TM 发的）。TM 不属于任何一次 run，拿不到 run_id、
> 也不持有 run 的 `sequence_counter`，故它发的事件 `run_id=None`、`sequence=0`
> （`TaskManager._emit`）。**这些事件既不进任何 run 组，也不参与组内单调递增**——
> 按 `sequence` 排序或去重的 host 必须改按到达顺序 / `timestamp`，否则这一整批会被
> 排到最前面或被当成重复项丢掉。详见 docs/upgrade/2026-09-02-task-status-ownership.md。

| 事件 | payload | 含义 |
|---|---|---|
| `RunStarted` | `run_id` `initial_step` | 一次 step 链执行开始。一个 task 可以有多个 run（重试 / 重排 / 挂起后恢复各一个新 run），一个 session 可以同时有多个 run 在跑 |
| `RunFinished` | `outcome` `final_status`（废弃） `will_retry` `total_events` `total_turns` `error` `error_type` | 一次执行结束，**无论成败必发**——host 靠它关 SSE。`outcome` 是**run 自己**的结局（run 词表五值，见 §2.4）；`final_status` 已废弃（仍装 `task.status`，保留一个发布周期供旧断言过渡）；`will_retry=true` 时 host 先别关流 |

### 3.3 TaskManager 信号 · 0（原 3 个，已停发，见 §5.6）

> **2026-09-04（runtime 对外面 agent-centric 对齐 Task 12）起停发**：这三个类型
> 唯一的消费者是 SM（`SessionRegistry` 的状态机），它自 2026-09-03 起已经降格、
> 只订阅 `AGENT_INSTANTIATED` / `AGENT_SPAWNED`，不再消费队列信号——`TaskManager.
> announce_queue_state` 与 `runtime._announce_queue_state_as_tm_proxy`（后者是
> "代 TaskManager 给 SessionRegistry 发它唯一的输入"这层代理，消费者已不存在，
> 整个方法一并删除）两处发射点都已停发。枚举成员与下面的 reducer 分支按 §5.6
> 保留（存量日志重放靠它们）。前端要的「这一轮跑完了」不再靠 `TaskQueueDrained`
> 提前收到信号——改订 `AgentLifecycleManager` 发的 `AGENT_IDLE`（正常收工，无未决
> HITL）/ `AGENT_WAITING_HUMAN`（挂在人身上）/ `AGENT_INTERRUPTED`（被进程重启
> 打断），这本就是 agent 该承担的信号；恢复期的可观测性也由同一套广播承担
> （`recover()` 装填 `AgentLifecycleManager` 后按折出来的现状重发，不再需要
> 代 TM 合成一条会话级队列信号）。下表描述的是停发前的行为，留作存量事件的读法
> 参考。

TM 对它手上全部任务的聚合结论曾是 **SM 的唯一输入**。三个独立类型而不是一个带
discriminator 的类型——SM 收到哪条就转到哪个状态，不读任何字面量。

**有活在跑时一条都不发**（`if self._queue or self._running_tasks: return`）。
全停下来后按「解开它需要谁」挑一条：`Interrupted`（运维）> `Blocked`（用户）> `Drained`（无需）。
TM 区分前两者靠的是 **task 状态**，那本来就是它自己的领域，不需要知道 HITL 的任何事。

> **这三个类型已于 2026-09-05 连枚举一并删除**（§5.8）：唯一消费者（会话状态机）
> 早已降格，两处发射点在 2026-09-04 停发；`master` 从未有过它们，存量流里不会出现。
> 下表保留的是那次设计的推理，不是现状。

| 事件 | payload | 含义 |
|---|---|---|
| ~~`TaskQueueBlocked`~~ | `count` | 没有能跑的任务了，剩下的都**正常地**停着（`AWAITING_HUMAN` / `SUSPENDED`）。`count` 供展示，SM 不读 |
| ~~`TaskQueueInterrupted`~~ | `reason` | 没有能跑的了，且**有任务被打断**（等 `/resume`）。优先于 `Blocked`——解开它需要运维 |
| ~~`TaskQueueDrained`~~ | `final_status` | 全部任务终态，可以收工 |

### 3.4 Task / Agent · 3

| 事件 | payload | 含义 |
|---|---|---|
| `BlackboardPublished` | `topic`（= task_id） `content_length` `parent_task_id` | 任务成功产出已发布到黑板，供任何 agent 按 task_id 精确召回 |
| `AgentSpawned` | `parent_agent_id` `subtask_id` | 为子任务派生了 agent |
| `SpawnRejected` | `reason` `fallback_to_inline` `attempted_subtask_id` | 派生被拒（今天只有 `depth_limit`）。`fallback_to_inline` 恒 `false`——降级 inline 执行的能力不存在 |

### 3.5 Context · 1

| 事件 | payload | 含义 |
|---|---|---|
| `ContextTokensEstimated` | `estimated_tokens` `assembled_tokens` `has_baseline` | 装配前的估算 vs 装配后的实际 |

### 3.6 LLM · 6

`origin` 区分是哪个子循环在跑：`loop.act` / `loop.observe` / `loop.background_observe` /
`loop.recognize_intent`。V2 之前这是靠**四套同形的独立类型**做的（`BackgroundObserve*`
那一族），合并后由 `origin` 承担。

| 事件 | payload | 含义 |
|---|---|---|
| `LLMRequestStarted` | `request_id` `model` `llm_account` `turn`/`round` | 本轮 LLM 调用开始 |
| `LLMPromptSent` | `request_id` `turn`/`round` `system` `messages:[{role, content}]` `tool_names` | 完整 prompt，供调试。`content` 经 `redact_content_for_event` 变成**带截断的预览串，不可回读** |
| `LLMTokenStreamed` ⚡ | `request_id` `delta` | 正文 token delta |
| `LLMReasoningStreamed` ⚡ | `request_id` `delta` | extended thinking delta |
| `LLMResponseFinished` | `request_id` `content` `reasoning` `tool_calls:[{id,name,arguments}]` `usage` `llm_model` `llm_account` `finish_reason` `turn`/`round` | 本轮完整结果。`llm_model`/`llm_account` 是**实际用的**那个（host 计账按此，不受切换竞态影响） |
| `LLMRetryTriggered` ⚡ | `attempt` `max_attempts` `next_delay_sec` `error_code` `error` | 退避重试。仅 outage 类可重试，且仅在本次尝试尚未吐出任何 chunk 时 |

⚡ = TRANSIENT，不持久化。

### 3.7 Capability · 3

| 事件 | payload | 含义 |
|---|---|---|
| `CapabilityInvoked` | `invocation_id` `capability_name` `capability_id` `arguments`（已脱敏） `tool_call_id` | 工具开始执行 |
| `CapabilityProgress` | `invocation_id` `kind` `data`（截 500） | 执行中的进度事件 |
| `CapabilityFinished` | `invocation_id` `capability_name` `arguments` `outcome`（`success`/`error`） `result`（截 8000） `result_length` `tool_call_id` | 工具结束。**失败也走这条**（`outcome=error`），没有独立的失败事件 |

### 3.8 Act · 2

| 事件 | payload | 含义 |
|---|---|---|
| `ActTurnStarted` | `turn` | 一轮 act 开始 |
| `MaxTurnsReached` | `max_turns` | act 因轮次上限退出。**唯一记录这件事的地方**——`act_exit_reason` 只进内存态，从不进事件流 |

### 3.9 Observe · 2

| 事件 | payload | 含义 |
|---|---|---|
| `ObserveStarted` **新增** | `task_id` | observe 的**起点**，与 `ObserveCompleted` 成对（同形于后台 recap 的 `TaskRecapStarted`/`TaskRecapDone`）。发在任何分流决策之前，故 payload 里**没有** `used_llm`——那时还没定。`execute` 无提前 return，取消路径下也照样成对 |
| `ObserveCompleted` | `outcome` `summary_length` `used_llm` | observer 的**结论**（`used_llm` 自 2026-09-03 起表示「判决**真的出自** LLM」，而非「尝试过 LLM」），不是 observe 步骤的阶段标记（阶段由 `StepCompleted(origin=loop.observe)` 管）。`outcome` 含 `retry` / `needs_user_input` 等**非终态**取值——重试 3 轮的 task 会有 3 条，而 task 状态事件前两轮什么都不说 |

### 3.10 Memory · 4

| 事件 | payload | 含义 |
|---|---|---|
| `MemoryIngested` | `memory_event_type` `source` `content_length` | 一条记忆写入。`source` 如 `dispatch_result` / `root_finish_pair` |
| `MemoryCompactStarted` | `trigger` `token_estimate` `target_tokens` | 一轮压缩开始。**必须 live 发**，否则前端错过整个「压缩中」窗口 |
| `MemoryCompacted` | `superseded_count` `layer` `source` `trigger` `freed_tokens`（按路径另有 `demoted_images` / `summary_event_id` / `events_before` / `events_after`） | 一**级**折叠完成。`source` ∈ `demote_images` / `root_experience` / `demote_lean` / `collapse` / `observe_retry` |
| `MemoryCompactFinished` | `trigger` `total_superseded` `freed_tokens` `levels` `est_before` `est_after` `target_tokens` | 一轮压缩收尾聚合。多数字段是各 `MemoryCompacted` 的加总，但 `est_after` 独有 |

### 3.11 Guard · 1

| 事件 | payload | 含义 |
|---|---|---|
| `FailureThresholdHit` | `failure_counter` `threshold` `failures:[{title, reason}]` | 连败达阈值、会话熔断。**只是记录**——真正的终结由随后的 `TaskCanceled` / `TaskFailed` / `SessionStatusChanged(FAILED)` 承载 |

---

## 4. `origin` · 17 个取值

「**哪个组件发出了这条事件**」。所有事件都有；存量事件读出空串，**不做反推**
（`LLMPromptSent` 这类本就有多个发射者，反推会造出看起来精确、其实是猜的数据）。

```
orchestrator.session_manager     orchestrator.task_manager
loop.driver     loop.prepare     loop.act        loop.observe
loop.background_observe          loop.recognize_intent            loop.compact
loop.finalize   loop.suspend     loop.reconcile
loop.capability_gateway          loop.llm_gateway
hitl.service
runtime
persistence.snapshot_writer
```

两级点号是为了让 host 能**前缀匹配**：`loop.` 取全部循环内事件，
`loop.background_observe` 精确排除后台观察的渲染。分隔符用 `.` 不用 `:`，
`:` 留给可路由的 capability id（`provider:tool`）。

**怎么填**（60+ 个发射点靠手填必然有人忘，所以结构性填充）：

1. `LoopState.origin`，driver 在每步开始前写入 → `make_event` 默认从 `state.origin` 取，
   **40+ 个循环内发射点零改动**；
2. `make_event(..., origin=...)` 显式覆盖，给 background observe 这类脱离主 driver 序列
   异步跑的场景；
3. 循环外的发射者（`session_manager` / `task_manager` / `hitl.service` / `runtime` /
   `snapshot_writer`）各持一个模块常量，在自己的 `_emit` 里填。

---

## 5. L 档 · 14 个（只读存量，不得再发射）

共同点：**还在 `EventType` 里、新流量里不再出现、重放存量日志时仍会被读到、不进别名表**
（别名表的语义是「旧名折进新分支」，这些没有对应新分支，用的是自己的旧分支）。
真相源是 `EventType.L_TIER_EVENT_TYPES`（`src/ctx_weft/protocols/events.py`）；下面按
退役批次分列，每批的退役条件不同。

### 5.1 HITL 那批 · 7 个

`HitlRequired` `HitlApproved` `HitlModified` `HitlAnswered` `HitlRejected`
`HitlCancelled` `SessionPausedHitl`

`fold_hitl_snapshot` 新旧双读，`_apply` 保留旧分支。旧事件折出来的 HITL 记录
`invocation_key` 为 `""`（通配），行为与升级前逐条同构。

**退役闸门（两级，缺一即停）：**

1. **删双读折叠**：升级点之前产生的 `HitlRequired` 全部已有对应终态事件（无悬挂未决），
   且这些 session 均已归档 / 超出最长存活期。
2. **删枚举值**（更严）：确认没有任何回放会碰到这 7 个字符串。通常要等一整个归档周期。

### 5.2 `SessionStatusChanged` · 1 个

拆成 `SessionInterrupted` / `SessionWaiting` / `SessionRunning` / `SessionFinished`
四条具体事实（§2.1.2）。
`_apply` 保留原分支——它读 `payload["new_status"]` 直接写会话状态，存量日志靠它才能重建。

**退役闸门比 HITL 那批简单**：它没有双读折叠，只有 `_apply` 的一个分支；条件只有一条——
确认没有任何回放会碰到这个字符串（同 5.1 的第 2 级）。

### 5.3 BackgroundObserve 那批 · 4 个（2026-09-03 起停发）

`BackgroundObserveRequestStarted` `BackgroundObservePromptSent`
`BackgroundObserveTokenStreamed` `BackgroundObserveResponseFinished`

`observe.py` 的 `ReactEventTypes` 间接层删除后，`run_observe_react` 统一发
`LLM_*`（靠 `state.origin` 区分前台/后台），这 4 个专属镜像类型停止发射。
`_apply` 分支原样保留，枚举成员按控制方裁定暂不删除（退役闸门是后续任务的事）。

### 5.4 `RecognizeIntentLLMPrompt` · 1 个（2026-09-03 起停发）

`recognize_intent.py` 切到 `stream_llm_resilient` 后，通用 `LLM_PROMPT_SENT`
（由 gateway 发，`origin=loop.recognize_intent`）取代了这条 step 专属的镜像事件。

### 5.5 `SessionFinished` · 1 个（2026-09-03 起停发）

会话状态机（`session_state.py`）随 `SessionRegistry` 降格一并退役，它不再有发射点。
`_apply` 分支原样保留（存量日志重放靠它——`master` 发过这条，宿主库里有）。

同批停发的 `SessionRunning` / `SessionWaiting` / `SessionInterrupted` **不在 L 档**：
见 §5.8。

### 5.6 `TaskQueue*` · 0 个（2026-09-04 停发，2026-09-05 删除）

`TaskQueueBlocked` `TaskQueueInterrupted` `TaskQueueDrained` 已不在 L 档，见 §5.8。

原委见 §3.3：唯一消费者（会话状态机）早已降格，两处发射点
（`TaskManager.announce_queue_state`、`runtime._announce_queue_state_as_tm_proxy`）
一并停发。恢复期的可观测性改由 `AgentLifecycleManager` 装填后的 `AGENT_*` 现状
广播承担。

### 5.7 共同的

5.3–5.5 这三批与 5.1/5.2 的退役条件不同：它们没有专属的双读折叠，`_apply`
分支原样保留即可支持存量日志重放，因此**没有"删双读折叠"这一级**，退役闸门只剩
5.1/5.2 式的第 2 级——确认没有任何回放会碰到这些字符串。

第 2 级闸门过了 L 档才清空。**在那之前 L 档非空是正常状态，不是待办积压。**

> 例外：host 的迁移脚本读的就是历史事件，**闸门再怎么过它都得留着旧字符串**。

### 5.8 已过闸门 · 6 个（2026-09-05 连枚举一并删除）

`SessionRunning` `SessionWaiting` `SessionInterrupted`
`TaskQueueBlocked` `TaskQueueInterrupted` `TaskQueueDrained`

**它们是「分支内部生死」的类型**：2026-09-02 随会话状态所有权重构上线，09-03（前三个）
与 09-04（后三个）停发，全程落在 `master` 之后的分支内部。`master` 的 `EventType`
里从来没有这 6 个名字，因此 §5.7 的第 2 级闸门（「确认没有任何回放会碰到这些字符串」）
对它们**天然成立**——任何从 master 迁移来的事件流都不可能含有它们，不必等一整个归档周期。

一并删除的：`EventType` 枚举成员、`L_TIER_EVENT_TYPES` 登记、`reducers.py` 的三个
`_apply` 分支、`providers/events/_lifecycle.py` 的三个活跃性分支与收窄元组项、
golden 用例 06/07 里的对应事件。

守卫：`tests/unit/test_agent_lifecycle.py::test_intra_branch_session_types_are_gone`、
`tests/unit/test_recover_loads_agents.py::test_task_queue_types_are_deleted_outright`、
`tests/unit/test_lifecycle_active_sessions.py::test_deleted_types_are_not_in_the_narrowing_tuple`。

---

## 6. 三条不变式

1. **`EventType` 全集 ≡ 实际发射集合 ∪ L 档。** 定义即必须发射；L 档是唯一例外，
   且必须是**显式白名单**，不是「测试没扫到就放过」。
2. **S / O / L 三个集合两两不交，并集 ≡ `EventType` 全集。** 加一个枚举值就得显式选边。
3. **`STATE_EVENT_TYPES` ≡ 两个状态消费者折叠集合的并集**——ctx-weft 的 `reducers.py`
   与 host 的 `projection_updater.py`。后者不共享 core 的 reducer，故须以 host 侧的
   对应测试补足；这是唯一测不严的地方。

外加两条：所有发射出的事件 `origin` 非空；L 档 ∩ 实际发射集合 = ∅。

---

## 7. 命名规则

**新名字绝不复用任何曾经发射过的字符串。**

反例：`PrepareCompleted` 本想改名叫 `ContextAssembled`（复用那个要删的、更好的名字）。
但存量流里真有 `ContextAssembled`、payload 是 `{token_count}`；改名后它会命中新分支去读
`assembled_token_count`，取不到得 0，把 `assembled_prompt_tokens` 覆盖成 0。
