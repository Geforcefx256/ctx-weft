# task-handoff

## Purpose

让任务之间的交接可靠：派发时声明的输入数据完整到达执行面并在恢复后可用，对子任务的操作以稳定 ID 配对不受标题干扰，任务依赖按前序成败区分放行条件。

## Requirements

### Requirement: 子任务输入的保存与传递

`delegate_task` 的 `inputs` 参数与 `delegate_plan` 每任务 spec 的 `inputs` 字段所声明的数据 MUST 写入子任务的任务契约，并 MUST 随任务创建事件（TASK_CREATED payload）持久化。输入 MUST 为纯 JSON 值：含不可 JSON 序列化内容的输入 MUST 使工具调用返回明确错误、不派发子任务，不得静默丢弃。尺寸上限按 UTF-8 字节计量；超过上限的输入 MUST 经分级收敛（截断长字符串值、裁剪数组元素与字典键数并保留计数标记、加标记后复核），收敛后仍超限的输入（如单个巨大标量）MUST 被明确拒绝。崩溃恢复（进程重启后重建）MUST 能从事件流原样取回子任务输入。

#### Scenario: delegate_task 输入到达子任务
- **WHEN** 模型调用 `delegate_task` 并传入 `inputs` 字典
- **THEN** 子任务契约持有该数据，TASK_CREATED payload 包含它，工具回执成功

#### Scenario: delegate_plan 每任务输入到达子任务
- **WHEN** 模型调用 `delegate_plan`，某任务 spec 含 `inputs` 字段
- **THEN** 该子任务契约与创建事件携带对应数据，其余任务不受影响

#### Scenario: 不可序列化输入被响亮拒绝
- **WHEN** `inputs` 含无法 JSON 序列化的内容
- **THEN** 工具返回明确错误（指明 inputs 必须为 JSON 数据），不创建子任务、不发创建事件

#### Scenario: 超限输入分级收敛
- **WHEN** `inputs` 序列化后超过尺寸上限，且超限来自长字符串值、数组元素数量或字典键数量
- **THEN** 持久化的输入经截断/裁剪收敛至上限内，裁剪处保留可识别的截断与计数标记，子任务正常派发

#### Scenario: 无法收敛的超限输入被拒绝
- **WHEN** `inputs` 含无法经分级收敛降到上限内的单个巨大标量
- **THEN** 工具返回明确错误，不派发子任务

#### Scenario: 崩溃恢复后输入仍可用
- **WHEN** 含输入的子任务已持久化、进程重启后恢复会话
- **THEN** 重建出的子任务输入与派发时声明的一致（截断/裁剪标记除外）

### Requirement: 子任务输入的上下文投递

已派发子任务的执行上下文 MUST 包含其声明的输入数据，以独立可识别的输入区块呈现，内容与任务契约一致。恢复后续跑（resume / reopen 后重跑）的执行上下文 MUST 同样包含该区块。

#### Scenario: 执行上下文包含输入区块
- **WHEN** 携带输入的子任务被派发执行
- **THEN** 装配出的模型请求中存在输入区块，内容与派发声明一致

#### Scenario: 恢复后续跑同样投递
- **WHEN** 子任务在崩溃恢复或重开后再次执行
- **THEN** 重建装配的执行上下文仍包含该输入区块

### Requirement: 任务操作以稳定 ID 配对

对子任务的审核操作（confirm / reopen / skip，经 `report_task_outcome` 的 `task_reviews`）MUST 以 `task_id` 配对目标；同名、标题修改、登记顺序变化 MUST NOT 改变操作对象。`task_reviews` 条目 MUST 以 `task_id` 引用目标；缺 `task_id`、`task_id` 非字符串或携带未知字段（含旧 `task_title`）的条目 MUST 被拒绝并在回执中说明原因，其余合法条目照常生效；引用非当前任务直接派生子任务的 id 时 MUST 拒绝并说明越权。`delegate_task` 与 `delegate_plan` 的成功回执 MUST 返回子任务 id（plan 为与 spec 顺序对应的 id 列表）。

#### Scenario: 同名子任务审核对象不漂移
- **WHEN** 存在两个同名子任务，审核条目指定其中之一的 `task_id`
- **THEN** 恰好对该子任务生效，另一个不受影响

#### Scenario: 改名不影响既定操作
- **WHEN** 子任务标题在派发后被修改，审核条目仍用原 id
- **THEN** 操作仍命中同一子任务

#### Scenario: 越权引用被拒绝
- **WHEN** 审核条目引用的 `task_id` 不是当前任务直接派生的子任务
- **THEN** 该条目被拒绝并在回执中说明，其余合法条目照常生效

#### Scenario: 旧式条目被显式拒绝
- **WHEN** 审核条目携带旧 `task_title` 字段或缺 `task_id`
- **THEN** 该条目被拒绝且回执说明须以 `task_id` 引用，其余合法条目照常生效

#### Scenario: 派发回执返回子任务 id
- **WHEN** `delegate_task` 或 `delegate_plan` 成功派发
- **THEN** 回执包含子任务 id（plan 为有序 id 列表），模型可据此在后续操作中引用

### Requirement: 依赖条件区分前序成败

任务依赖 MUST 支持两种条件：`on_success`（前序 FINISHED 才放行）与 `on_any`（前序到达任一终态即放行）。本变更之后新派发的依赖缺省条件 MUST 为 `on_success`；存量事件回放中未声明条件的依赖 MUST 按 `on_any` 解释（历史保真）。前序失败或被取消时，`on_success` 后继 MUST NOT 被派发，且 MUST 落 CANCELED 终态并附专用错误码（BLOCKED_BY_FAILED_DEP）与阻塞源任务标识；`on_any` 后继 MUST 照常派发。因依赖取消落终态的任务 MUST NOT 改写会话终态（不将会话标为 CANCELED），且 MUST 保留父任务唤醒与后续调度。永久阻塞判定 MUST 在运行期（前序失败/取消后）与恢复期（重建依赖后、首次调度前）各执行一次幂等扫描，确保崩溃窗口内漏落的级联取消在恢复后被补齐。阻塞取消的原因（错误码与阻塞源任务）MUST 随终态事件 payload 持久化并进入任务投影，恢复回放后仍可解释。因条件不满足而落终态的任务 MUST NOT 阻塞会话的完成判定。依赖条件 MUST 在任务链重开（reopen）重建后保持原语义。`delegate_plan` 的任务 spec MUST 支持显式声明条件（`run_if: "success" | "any"`，缺省 success）。

#### Scenario: 前序失败不放行成果依赖
- **WHEN** 任务 A 落 FAILED，任务 B 以 on_success 依赖 A
- **THEN** B 不被执行，落 CANCELED 终态且错误码为 BLOCKED_BY_FAILED_DEP

#### Scenario: 清理任务在前序失败后仍执行
- **WHEN** 任务 A 落 FAILED，任务 C 声明 on_any 依赖 A
- **THEN** C 被正常派发执行

#### Scenario: 前序成功时两种条件均放行
- **WHEN** 任务 A 落 FINISHED
- **THEN** on_success 与 on_any 后继均被放行

#### Scenario: 依赖取消不终结会话
- **WHEN** 某子任务因 on_success 依赖失败而落 CANCELED（BLOCKED_BY_FAILED_DEP）
- **THEN** 会话不被标记为 CANCELED，父任务在子任务全部终态后照常唤醒

#### Scenario: 崩溃窗口的恢复期补扫
- **WHEN** 任务 A 的 FAILED 已持久化、其 on_success 后继 B 的级联取消尚未持久化时进程崩溃，随后恢复
- **THEN** 恢复期的幂等扫描将 B 判定并落终态，B 不会永久滞留 PENDING

#### Scenario: 重启后阻塞原因可解释
- **WHEN** 含 BLOCKED_BY_FAILED_DEP 终态的会话经重启回放重建
- **THEN** 任务的阻塞原因与阻塞源任务标识仍可从投影读到，父任务观察面可解释

#### Scenario: 条件断裂不滞留会话
- **WHEN** 某计划的 on_success 后继因前序失败全部落终态
- **THEN** 不存在永久 PENDING 的任务，会话可到达完成/空闲判定

#### Scenario: 存量回放保真
- **WHEN** 回放本变更之前持久化的事件流，依赖未带条件字段
- **THEN** 依赖按 on_any 语义解锁，恢复行为与历史版本一致

#### Scenario: 重开后条件存活
- **WHEN** 审核触发任务链 reopen、依赖关系重建
- **THEN** 重建后的依赖保持原条件语义（on_success 仍 on_success）

#### Scenario: 清理步的显式声明
- **WHEN** 模型在 `delegate_plan` 的任务 spec 中声明 `run_if: "any"`
- **THEN** 该任务的依赖按 on_any 处理，其余未声明的任务按 on_success
