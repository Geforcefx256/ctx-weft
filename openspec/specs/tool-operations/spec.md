# tool-operations

## Purpose

定义工具执行的稳定身份、操作账本与恢复策略契约：跨重启的同一逻辑调用拥有稳定 operation_id，副作用是否已发生可判定，结果未知的操作交宿主显式处置——堵死「恢复盲重跑副作用」（H3）。

## Requirements

### Requirement: 稳定逻辑调用身份

每个工具调用 SHALL 拥有跨重启稳定的 `operation_id`，由 `(tenant_id, session_id, agent_id, assistant_record_id, tool_ordinal)` 确定性派生；`assistant_record_id` MUST 在执行前持久化（assistant 回合入 memory 的记录 id）。`invocation_id` 保留为单次执行尝试身份（取消用）；`tool_call_id` 仅为 LLM wire 配对字段，MUST NOT 用作恢复匹配依据。普通执行、热 HITL resume、冷恢复 SHALL 引用同一 operation_id；两条内容相同的合法调用 MUST 得到不同 operation_id（不误去重）；模型复用同一 tool_call_id MUST NOT 使旧结果覆盖新调用。silent/dispatch 类控制工具 SHALL 同样拥有账本身份（不入对话 ≠ 不入账本）。

#### Scenario: 同一调用跨重启身份不变

- **WHEN** 一次工具调用在崩溃后经冷恢复重入
- **THEN** 恢复路径读到与首次执行相同的 operation_id（同一 assistant_record_id + ordinal 派生）

#### Scenario: 两次合法同参调用不去重

- **WHEN** 两条 assistant 消息都用 call_1 且参数相同
- **THEN** 两个不同 operation_id（不同 record_id/ordinal），各自完整执行

#### Scenario: call_1 复用不串扰

- **WHEN** 后一轮 LLM 回合复用了前一轮的 tool_call_id
- **THEN** 两轮调用各自 operation_id 独立，前一轮的结果不配对到后一轮

### Requirement: 操作账本状态机

`OperationStore` SHALL 提供 `get / prepare / compare_and_set`，按状态机 `prepared → started → completed` 记录操作（另含 `waiting_human` / `unknown`）；OperationRecord MUST 含身份五元组、授权后参数指纹、恢复策略字段、attempt IDs、完整规范化结果或可持久读取的结果引用（非审计截断文本）、error、memory result ID 引用。执行顺序 SHALL 为：持久身份 → 授权校验 → prepared → CAS started（持久确认）→ 调 Provider → 保存 outcome completed（持久确认）→ 幂等写入 TOOL_RESULT memory → 发 CapabilityFinished。账本 completed 而 memory 写入失败时，恢复 SHALL 从账本重建 memory、不再次执行工具。memory result id SHALL 由 operation_id 确定性生成。OperationStore 写失败 SHALL 按存储不可用处理（会话隔离），不得静默降级。宿主未注册持久 OperationStore 时 runtime SHALL 默认提供内存实现并在声明跨进程恢复能力时如实报告缺失。

#### Scenario: completed 后 memory 写前崩溃

- **WHEN** 工具外部成功且账本已 completed，进程在 TOOL_RESULT 写入前退出，随后恢复
- **THEN** 自账本补写 memory 结果（id 由 operation_id 派生），外部副作用仍恰好一次

#### Scenario: CAS 串行化并发推进

- **WHEN** 两个路径并发对同一 operation 推进状态（prepare/started/completed）
- **THEN** revision 不匹配的一方被拒绝，状态机不出现倒退或跳跃

### Requirement: 恢复策略表

`ToolCapability` SHALL 声明 `recovery_policy`，取值**恰为两类**（`idempotent | reviewed`），默认 `reviewed`（不从 side_effects 推断安全）。分类判据是「core 要不要做决定」：`idempotent` = 重跑安全，core 同 operation_id 直接重跑；`reviewed` = core MUST NOT 自行重跑，交裁决链。

无法识别的取值 SHALL 在启动校验响亮失败，MUST NOT 静默降级为保守值。

裁决能力 SHALL 由 `isinstance(provider, OperationAdjudicator)` **发现**，MUST NOT 要求在 capability 上声明——该接口是 Provider 级的，用 capability 级字段声明会制造「声明了却没实现」这一类本不必存在的失败模式。

恢复时按状态×策略分派：`completed` 复用结果不重执行；`prepared` 且从未进入 `started` 可首次执行（仍先重新检查授权）；`started + idempotent` 以同 operation_id 重试；`started + reviewed` 走裁决链——Provider 实现 `OperationAdjudicator` 则调用之，`completed` 直接复用外部结果不再执行、`definitely_not_started`（仅权威否定）方可重跑、`unknown` 置 `unknown`；未实现裁决接口则直接置 `unknown`（默认形态，不是配置错误）。一切结果不定情形一律置 `unknown` 不自动执行；`waiting_human` 经既有 HumanResumable 协议恢复，不从头重新 invoke。Reconcile 的完成匹配 SHALL 以账本 operation_id 为判据，MUST NOT 再以 tool_call_id 集合判定（防 call_1 复用串扰）。无账本身份的存量 dangling SHALL 默认置 `unknown`，不得以随机生成的 id 自动执行副作用工具。控制工具单独核验：delegate 以 operation_id 找回已创建子任务（确认丢失重入不生成第二棵子树）、finish/metadata 同身份幂等、ask_user 复用已有请求；MUST NOT 对控制工具整体标记 `idempotent` 后省略验证。

#### Scenario: reviewed 且无裁决者时不重跑

- **WHEN** started 后崩溃且策略为 reviewed，Provider 未实现 OperationAdjudicator
- **THEN** 操作置 unknown 等宿主处置，恢复不得自动重执行；这是默认形态，不报配置错误

#### Scenario: idempotent 重跑恰好一次补全

- **WHEN** started 后崩溃且策略为 idempotent，恢复重入
- **THEN** 以同 operation_id 重试一次；外部副作用由 Provider 承诺幂等；账本与 memory 各落一次

#### Scenario: 裁决者的权威否定

- **WHEN** started 后崩溃且策略为 reviewed，Provider 的裁决返回 definitely_not_started
- **THEN** 以同 operation_id 重新执行；裁决返回 completed 则直接复用外部结果不再执行；返回 unknown 则置 unknown 落到人

#### Scenario: 非法取值响亮失败

- **WHEN** Provider 声明一个不在取值域内的 `recovery_policy`
- **THEN** 启动校验失败并指明合法取值；MUST NOT 静默降级为 `reviewed` 后照常启动

#### Scenario: 存量无身份 dangling 保守停住

- **WHEN** 恢复遇到 WP5 之前持久化的 dangling（无账本记录可查）
- **THEN** 默认置 unknown 并要求宿主处置，不自动执行

### Requirement: 结果未知的宿主处置

未知操作 SHALL 将 task 置 INTERRUPTED + `TOOL_OUTCOME_UNKNOWN` 错误码并发布 `OperationUncertain` 事件（payload 至少含 operation_id、工具名、revision、可用处置动作、脱敏摘要）；宿主经 `resolve_operation(operation_id, decision, expected_revision)` 处置：`supply_result`（宿主已核实外部结果，按 Provider 结果结构归一化后入账本并按确定性 id 补写 memory）/ `retry_confirmed`（宿主显式承担重复执行风险，持久记录决定后以原 operation_id 重排）/ `cancel_task`（停止该任务；不声称撤销已发生的外部动作）。revision 不匹配 SHALL 拒绝（双宿主并发处置恰好一个成功）；unknown 状态下普通 `recover_agent` MUST NOT 绕过决策重跑工具（闸门）。

#### Scenario: 两个宿主并发处置

- **WHEN** 两个请求以不同 decision 并发 resolve 同一 operation
- **THEN** 恰好一个成功，另一个因 revision 不匹配被拒绝

#### Scenario: supply_result 补写 memory 不重执行

- **WHEN** 宿主以 supply_result 提供核实过的外部结果
- **THEN** 账本 CAS completed、按 operation_id 确定性派生的 memory id 幂等补写 TOOL_RESULT；provider 不被调用

#### Scenario: unknown 闸门挡住普通恢复

- **WHEN** task 处于 unknown 中断且宿主未 resolve，调用 recover_agent
- **THEN** 恢复不重跑该工具，保持等待宿主处置
