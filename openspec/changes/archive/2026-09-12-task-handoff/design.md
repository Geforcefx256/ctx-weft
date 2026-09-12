# Design: task-handoff

## Context

见 proposal.md（Why）。代码现状锚点：

- `Task`（`core/models/task.py`）无任何输入字段；`delegate_task` 声明的 `inputs` 参数在 `TaskModel(...)` 构造中被丢弃（`core/capabilities/control_tools.py:187` 附近），`delegate_plan` 的 spec schema 未声明 inputs。
- 子任务审核按标题配对：`_collect_reviews` 的 `{t.title: t}` 字典（`control_tools.py:350`），后注册覆盖先注册；review 展示面（`core/assembler/composer.py:1103`-`1107`）**已打印 task_id**，仅指令行要求按 task_title 引用。
- 依赖解锁不分成败：`TaskQueue.mark_failed` 把失败任务塞进 `_completed`（`core/orchestrator/task/queue.py:94`-`96`，注释标明"treat failed as done for unblocking"是故意的）；`restore` 重建时同样只查 `terminal_ids`（`core/orchestrator/task/manager.py:205`）。
- 恢复链约束：事件流是单一事实源，TASK_CREATED 回放即重建 Task；「事件库恒不含字节」是硬规则。
- 控制工具严格校验（spec: capability-gateway）只查**顶层**参数：`control:` 前缀工具的未知顶层参数会被拒并回灌错误；嵌套结构（`delegate_plan` 的 `tasks` 列表项、`report_task_outcome` 的 `task_reviews` 条目）不受其约束——前者可用于自由扩展 spec 字段，后者意味着**嵌套层的合法性必须由工具实现显式校验**。
- Composer 对 task_spec block **只读 metadata 的 title / description / user_prompt 三字段**（`DefaultComposer._task_spec_fields`；fresh 实时构建与 `_frame_current_message` 就地装饰两条渲染路径皆然），block 的 content 仅作快照与 token 估算（`core/assembler/sources/task_spec.py:42` 注释明示此契约）。任何新字段要到达模型，必须同时接进这两条渲染路径。
- `on_task_finished` 的 CANCELED 分支会把 session 整体标为 CANCELED（`core/orchestrator/task/manager.py:1033`-`1039`），已有 `_pause_abandon` / `_threshold_tripped` 两种豁免先例。
- `TaskView` 无 error_code 字段，reducer 对 TASK_CANCELED 只落状态不读 payload 原因——回放后阻塞原因会丢失。

## Goals / Non-Goals

**Goals:**

- 输入通道：delegate 时声明 → 任务契约 → TASK_CREATED 持久化 → 上下文投递 → 恢复后可用，单一路径无分叉。
- ID 配对：模型可见的操作句柄全部换成稳定 id，标题退化为纯展示。
- 依赖条件：两种条件 + 缺省语义 + 永久阻塞善后，全部在现有调度组件内完成。

**Non-Goals:**

- 不做通用数据流/工件平台：inputs 是尺寸受限的 JSON 字典，无版本、无引用解析。
- 不做完整 DAG 表达语言：只有 on_success / on_any，无优先级组合、无条件表达式。
- 不引入 Goal/Plan/Acceptance 等新领域对象（属后续独立提案）。
- 不自动补救或重规划：阻塞善后只落终态并让父任务观察面可见，补救决策留给上层。

## Decisions

### D1 依赖缺省语义：写入时物化，新派发 `on_success`、回放 `on_any`

**决策**：缺省只在**写入时**发生一次。新代码铸造任务时，`dep_conditions` 全量显式落盘（每个 dep 都写明 `"success"` 或 `"any"`）；存量事件没有 `dep_conditions` 字段，读取侧读不到即按 `on_any` 解释。读取侧永远不需要猜缺省。

- 理由：完成标准要求交付链缺省不因前序失败误启动 → 新缺省必须 success；而回放是「重现已发生的状态」，历史事件未声明条件，按 any 解释才是保真。物化缺省消除了「同一缺项两种解释」的歧义。
- 备选（否决）：回放缺省也用 success——改变存量会话的恢复行为，违反恢复语义；新派发保持 any、仅显式声明 success——把正确性负担转嫁给模型每次显式声明，与缺陷修复目标相反。

**表示**：`Task.dag_deps: list[str]` 保持不变（兼容投影/事件负载），新增 `Task.dep_conditions: dict[str, str] | None`（dep_id → `"success" | "any"`；None = 存量/无条件）。`QueueEntry` 将 `blocked_by` 拆为两个集合：`blocked_any`（on_any 依赖）与 `blocked_success`（on_success 依赖）。`TaskQueue` 增加第二完成集 `_succeeded`（仅 FINISHED；`_completed` 保持「任一终态」语义不变）。解锁判据：

- `mark_complete`（FINISHED）：两集都释放（刷新逻辑现有，扩到两集）。
- `mark_failed` / 取消：只从 `blocked_any` 释放；`blocked_success` 中包含该 dep 的条目**不**释放。

**模型声明面**：`delegate_plan` 任务 spec 增 `run_if: "success" | "any"`（缺省 success）；`delegate_task` 单任务无前序依赖，不暴露该参数。

### D2 依赖取消的终态、会话语义与持久化投影

**决策**：on_success 依赖判明永不可满足（某依赖落 FAILED/CANCELED）时，后继任务落 `CANCELED` 终态、`error_code = "BLOCKED_BY_FAILED_DEP"`、error 文本指明阻塞源任务 id；级联处理（B 被取消后，on_success 依赖 B 的 C 同样判定）。

**与用户取消的区分（会话语义豁免）**：依赖取消经 `on_task_finished` 收尾时，MUST NOT 把 session 标为 CANCELED——在 CANCELED 分支按 `error_code == BLOCKED_BY_FAILED_DEP` 增加第三种豁免（与 `_pause_abandon` / `_threshold_tripped` 同型）。豁免**只作用于会话终态改写**，其余收尾语义全部保留：`_try_resume_parent` 照常（子任务全终态后父任务唤醒）、队列终态清理照常、drain 照常（on_any 后继照常派发）。被阻塞任务从未 start（`started_at` 为空），`cancel_finalizer` 胶囊闭合天然跳过（无派发框可闭）。也不计入 failure_counter（CANCELED 分支本就不计）。

- 理由：`TERMINAL_TASK_STATUSES` 已含 CANCELED，投影/恢复/终态事件族零新增。FAILED 不行——它隐含「运行过且失败」，且会计入 failure threshold（`failure_counter`/熔断），语义污染。
- 备选（否决）：新增 SKIPPED 状态——波及状态词表、投影、事件与全部下游判断，收益仅是命名精确；FAILED + error_code——见上，阈值计数是硬伤。

**阻塞原因的持久化投影**：TASK_CANCELED payload MUST 携带 `error_code` 与 `blocked_by_task_id`（阻塞源任务 id）；`TaskView` 增 `error_code` / `blocked_by_task_id` 字段，reducer 读 payload 回填；converter 回填 `Task.error` / `error_code`。快照随 TaskView 自动覆盖。这样重启回放后父任务观察面仍可解释「谁被谁阻塞而取消」。

**永久阻塞扫描的双入口**：

1. **运行期**：`mark_failed` / 协作取消路径之后由 TaskManager 扫描队列条目触发（定点迭代：处理一个被阻塞取消的任务后重扫，直至不动点），发取消终态事件并出队。
2. **恢复期**：`restore` 重建依赖条件之后、**首次 drain 之前**执行同一套扫描。理由：崩溃窗口——A 的 FAILED 已落盘、B 的级联取消尚未落盘时进程崩溃，恢复只重建依赖条件，A 的失败回调不会重放，没有恢复期扫描 B 将永久 PENDING。扫描幂等：已终态任务跳过、重复执行无副作用，运行期与恢复期可安全复用同一实现。

### D3 inputs：纯 JSON 字典、分级尺寸收敛、双路径投递、不入 memory

**决策**：

- `Task.inputs: dict | None`。控制工具侧先 `json.dumps` 校验（不可序列化 → 响亮报错、不派发，与「part 列表响亮拒绝」同一先例），再施加尺寸上限（序列化后 8 KiB，常量可调），计量单位为 **UTF-8 字节**（`json.dumps(..., ensure_ascii=False).encode("utf-8")` 的长度）。
- **分级收敛**（上限语义：规整后必达限内，否则拒绝）：
  1. 长字符串值截断；
  2. 容器裁剪：超限来自数组元素或字典键数量时（探针实测：一万元素数字数组 30,012 字节、九千字符键 9,007 字节——均无可截断的字符串值），数组保留前 N 项、字典按插入序保留前 N 键，裁剪处以计数标记标注（如 `"_truncated": {"dropped_items": 9700}`）；
  3. 复核：加标记后**重新计量**，仍超限（如单个巨大标量）→ 响亮拒绝。
- 持久化：inputs 随 TASK_CREATED payload 落盘（满足「事件库恒不含字节」——它本来就必须是纯 JSON）。
- **投递（双路径接线）**：`Task.inputs` → `TaskSpecSource` 将 inputs 写入 block 的 `metadata["inputs"]`（content 仍只是快照，遵守其 docstring 契约）→ **Composer 两条渲染路径**均渲染 `## Inputs` 小节（pretty JSON，截断标记原样可见）：fresh 实时构建分支（`composer.py:574`-`582` 的 parts 拼装）与 `_frame_current_message` 就地装饰分支（`:713` 起）。`_task_spec_fields` 相应扩为读取 inputs 字段。覆盖四条执行路径：首次执行、已有 memory 的续跑、崩溃恢复后的重建执行、reopen 后重跑。无 inputs 的任务零行为变化。
- **不**写 memory：单一事实源是事件 payload，prompt 已投递执行面，双写只会漂移。

### D4 ID 配对与回执：配对键换 id，条目级显式校验，回执必须带 id

**决策**：

- `_collect_reviews` 的字典键从 title 换 id；`report_task_outcome` 的 `task_reviews` 条目 schema 从 `task_title` 换 `task_id`（**直接移除旧字段**）。
- **条目级显式校验（不依赖顶层严格校验）**：`task_reviews` 是嵌套列表，gateway 的控制工具严格校验只查顶层参数，嵌套条目它不管——旧式 `task_title` 条目不会被自动拒绝。因此在 `_collect_reviews` 入口显式校验每条：缺 `task_id`、`task_id` 非字符串、或携带未知字段（含旧 `task_title`）→ **该条目拒绝**并在回执中说明原因，其余合法条目照常生效（与现有越权拒绝的回执形态一致）。同时补齐 `task_reviews` 的 schema items 描述（文档性约束；机器校验在工具实现内）。模型收到条目拒绝说明后同 run 重发正确条目——自愈，但靠的是条目级校验的显式错误，不是 gateway。
- review 面指令行（composer）从 "referencing the exact task_title" 改为 task_id（展示面本就打印 id）。
- 回执：`delegate_task` ack 带子任务 id；`delegate_plan` ack 带与 spec 顺序对应的 id 列表。`spawn_titles` 累积器保留（标题纯展示用途），必要时平行增 `spawn_ids`——操作引用一律 id，标题不再作为任何配对键。

## Risks / Trade-offs

- [ack 文案与 schema 变更破坏依赖旧文案/旧参数的测试与下游] → 实现时全仓检索断言点同步更新；旧式 `task_title` 条目靠条目级校验的显式拒绝说明引导模型重发。
- [on_success 缺省改变现有 delegate_plan 行为：此前失败后继续的链会中断] → 这正是修复目标；存量恢复按 any 不受影响；在 proposal 与本设计中均已显式声明该行为变更。
- [依赖取消误伤会话终态（把 session 标 CANCELED）] → `error_code == BLOCKED_BY_FAILED_DEP` 的会话语义豁免（同 `_pause_abandon` 先例）；集成测试断言：子任务被阻塞取消后父任务照常唤醒、会话不被标 CANCELED。
- [崩溃窗口漏掉级联取消（A FAILED 落盘、B 未落盘）导致 B 永久 PENDING] → 恢复期幂等扫描兜底（restore 后、首次 drain 前）；崩溃窗口专项测试。
- [阻塞善后的级联判定引入新的调度死角（如环状依赖误判）] → 定点迭代以不动点终止；菱形/链式/环状依赖各有单测。环状依赖本就属于既有死锁域（`all_blocked`），本变更不扩大处理范围。
- [inputs 裁剪让模型拿到不完整数据] → 裁剪标记（含丢弃计数）显式可见；无法收敛至上限的输入直接拒绝而非静默；上限是常量，实测后可调。
- [CANCELED 阻塞取消与用户主动取消在事件面上难区分] → payload 的 error_code + blocked_by_task_id 区分，且随投影持久化、恢复后可解释。

## Migration Plan

- **存量事件**：无 `inputs` / `dep_conditions` 字段 → 旧语义（无输入、依赖按 any）。无需回填脚本。
- **部署**：单版本切换，无灰度开关（行为变更即修复目标，兼容开关只会掩盖语义）。
- **回退**：回滚代码即可——旧代码忽略事件中的新字段（`deserialize_settings` 已有未知键过滤先例，Task 重建侧同样宽容），行为退回旧语义，无格式损坏。

## Open Questions

- 「三种 inputs」的第三条通道具体指什么（探针语境）：本设计按已坐实的两条通道（delegate_task 参数、delegate_plan spec）实现；第三条确认后在测试矩阵中补齐，不影响接口形状。
