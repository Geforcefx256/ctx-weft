# tool-operations

## Purpose

定义工具执行的稳定身份、操作账本与恢复策略契约：跨重启的同一逻辑调用拥有稳定 operation_id，副作用是否已发生可判定，结果未知的操作交宿主显式处置——堵死「恢复盲重跑副作用」（H3）。

## Requirements

### Requirement: 稳定逻辑调用身份

每个工具调用 SHALL 拥有跨重启稳定的 `operation_id`，**它即该调用在摄入点铸造的内部 tool_call 标识**（`tc_...`，见 capability `conversation-integrity`）——随 assistant 回合一同持久化，恢复时**读出**而非重算。SHALL NOT 另立第二条由 `(tenant, session, agent, record_id, ordinal)` 派生的身份：铸造落地之后消息里的 `tool_call_id` 本身已唯一且跨重启稳定，两条同源身份只会制造「必须保持一致」的隐式约束。`invocation_id` 保留为单次执行尝试身份（取消用）。未经铸造的裸 wire id（宿主直构 / 测试替身）MUST NOT 用作账本键——判据是「是否内部标识」，不匹配则账本全程旁路。普通执行、热 HITL resume、冷恢复 SHALL 引用同一 operation_id；两条内容相同的合法调用 MUST 得到不同 operation_id（不误去重）；模型复用同一 tool_call_id MUST NOT 使旧结果覆盖新调用。silent/dispatch 类控制工具 SHALL 同样拥有账本身份（不入对话 ≠ 不入账本）。

#### Scenario: 同一调用跨重启身份不变

- **WHEN** 一次工具调用在崩溃后经冷恢复重入
- **THEN** 恢复路径读到与首次执行相同的 operation_id（同一条 tool_call 的内部标识，随消息落库）

#### Scenario: 两次合法同参调用不去重

- **WHEN** 两条 assistant 消息都用 call_1 且参数相同
- **THEN** 两个不同 operation_id（铸造时 anchor/ordinal 不同），各自完整执行

#### Scenario: call_1 复用不串扰

- **WHEN** 后一轮 LLM 回合复用了前一轮的 tool_call_id
- **THEN** 两轮调用各自 operation_id 独立，前一轮的结果不配对到后一轮

### Requirement: 操作账本状态机

`OperationStore` SHALL 提供 `get / prepare / compare_and_set`，按状态机 `prepared → started → completed` 记录操作（另含 `waiting_human`）；MUST NOT 为「结果无法确定」另立状态——那是 `completed` 的一种 `result` 内容；OperationRecord MUST 含身份五元组、授权后参数指纹、恢复策略字段、attempt IDs、完整规范化结果或可持久读取的结果引用（非审计截断文本）、error、memory result ID 引用。执行顺序 SHALL 为：持久身份 → 授权校验 → prepared → CAS started（持久确认）→ 调 Provider → 保存 outcome completed（持久确认）→ 幂等写入 TOOL_RESULT memory → 发 CapabilityFinished。账本 completed 而 memory 写入失败时，恢复 SHALL 从账本重建 memory、不再次执行工具。memory result id SHALL 由 operation_id 确定性生成。OperationStore 写失败 SHALL 按存储不可用处理（会话隔离），不得静默降级。宿主未注册持久 OperationStore 时 runtime SHALL 默认提供内存实现并在声明跨进程恢复能力时如实报告缺失。

#### Scenario: completed 后 memory 写前崩溃

- **WHEN** 工具外部成功且账本已 completed，进程在 TOOL_RESULT 写入前退出，随后恢复
- **THEN** 自账本补写 memory 结果（id 由 operation_id 派生），外部副作用仍恰好一次

#### Scenario: CAS 串行化并发推进

- **WHEN** 两个路径并发对同一 operation 推进状态（prepare/started/completed）
- **THEN** revision 不匹配的一方被拒绝，状态机不出现倒退或跳跃

### Requirement: 恢复策略表

`ToolCapability` SHALL 声明 `recovery_policy`，取值**恰为两类**（`idempotent | reviewed`），默认 `reviewed`（不从 side_effects 推断安全）。分类判据是「core 要不要做决定」：`idempotent` = 重跑安全，core 同 operation_id 直接重跑；`reviewed` = core MUST NOT 自行重跑，交裁决链。

无法识别的取值 SHALL 在启动校验响亮失败，MUST NOT 静默降级为保守值。

重跑授权 SHALL 由**宿主注册**，MUST NOT 要求由 provider 对象自身实现——知道怎么查证外部真值的对象未必是提供工具的对象（MCP 工具尤其如此），绑在 provider 上则宿主无法为自己不控制的 provider 补上查证能力。控制工具 SHALL 声明 `idempotent`（core 自有的同身份幂等状态迁移），MUST NOT 在 gateway 里写成按名字前缀的特判。

恢复时按状态×策略分派：`completed` 复用结果不重执行；`prepared` 且从未进入 `started` 可首次执行；`started + idempotent` 以同 tool_call_id 重试；`started + reviewed` 问**重跑授权**（`RerunAuthorizer.authorize_rerun`），它 SHALL 返回 `AuthorizationDecision`：`allowed=True` 以同 tool_call_id 重跑；`allowed=False` 把 `message` 写成这次调用的工具结果并作结（`result_is_error` 由授权方按内容决定——查到外部真结果时 MUST NOT 标成错误）；`needs_human` 走既有 HITL park（账本落 `waiting_human`，决定缓存第三维 SHALL 是独立的 rerun stage，MUST NOT 与事前授权共用键）。未注册重跑授权时 core SHALL 代为作结「无从查证」，同样不重跑。

重跑授权 SHALL 与事前授权 `Authorizer` **分开注册、分开解析**（provider 级 + 逐工具级，无 default = 不重跑），MUST NOT 复用 `authorize` 回答重跑问题——默认放行型 authorizer 会对它答「允许」，等于自动重跑副作用。二者 SHALL 在同一次恢复 invoke 中依次执行：先重跑授权、放行后再走事前授权（「当初准跑」不等于「现在还准跑」），MUST NOT 互相顶替。

`prepared` 之前（**持久**账本查不到记录）SHALL 视为确定未启动而首次执行。该判定 MUST 以账本自陈的跨进程持久性（`OperationStore.durable`）为条件：非持久账本重启后为空，「没跑过」与「跑过但记录随进程消失」不可区分，此时 SHALL 保守作结。Runtime 在 `event_commit_policy="required"` 而账本不持久时 SHALL 启动告警，如实报告该能力缺失。

「结果无法确定」SHALL 表现为一种**工具结果**，MUST NOT 成为一种控制流：MUST NOT 因此停机、MUST NOT 要求宿主介入才能续跑、MUST NOT 为它设立专属错误码/事件类型/处置 API。裁决者判不了时用 `result` 的**文本**说明查到了什么、查不到什么——那是内容不是状态，core 不替它组织措辞，也不据此分支。重复调用的防护归 Provider 自理：core 的承诺止于「不自行重跑」，agent 下一轮主动再调是一次新的逻辑调用。一切结果不定情形一律不自动执行；`waiting_human` 经既有 HumanResumable 协议恢复，不从头重新 invoke。Reconcile 的完成匹配 SHALL 以账本 operation_id 为判据，MUST NOT 再以 tool_call_id 集合判定（防 call_1 复用串扰）。无账本身份的存量 dangling SHALL 同样作结「无账本记录，无从查证」，MUST NOT 以随机生成的 id 自动执行副作用工具。控制工具单独核验：delegate 以 operation_id 找回已创建子任务（确认丢失重入不生成第二棵子树）、finish/metadata 同身份幂等、ask_user 复用已有请求；MUST NOT 对控制工具整体标记 `idempotent` 后省略验证。

#### Scenario: reviewed 且无重跑授权时不重跑

- **WHEN** started 后崩溃且策略为 reviewed，宿主未注册 RerunAuthorizer
- **THEN** core 代为作结「无从查证」：账本 CAS completed、该说明文本成为这次调用的工具结果，恢复不重执行、会话照常续跑；这是默认形态，不报配置错误

#### Scenario: idempotent 重跑恰好一次补全

- **WHEN** started 后崩溃且策略为 idempotent，恢复重入
- **THEN** 以同 operation_id 重试一次；外部副作用由 Provider 承诺幂等；账本与 memory 各落一次

#### Scenario: 重跑授权的权威放行

- **WHEN** started 后崩溃且策略为 reviewed，重跑授权返回 `allowed=True`（权威判定「这次没跑成」）
- **THEN** 以同 tool_call_id 重新执行；其后仍走一次事前授权

#### Scenario: 判不了也只是一种结果

- **WHEN** 重跑授权查不到外部真值，返回 `allowed=False` + 说明文本
- **THEN** 不重跑，该文本成为这次调用的工具结果；task 状态不变、无专属事件、无需宿主介入，agent 下一轮据此自行决定

#### Scenario: 交给人

- **WHEN** 重跑授权返回 `needs_human`
- **THEN** 走既有 HITL park、账本落 waiting_human；人批准则以同 tool_call_id 重跑，拒绝则用人写的话作结；决定按 rerun stage 缓存，冷恢复复用不再问第二遍

#### Scenario: 重跑授权看得到真实执行参数

- **WHEN** 首次执行的参数经 HITL 改写过，崩溃后重跑授权被问到
- **THEN** 它从账本记录读到的是**授权与改写之后**交给 provider 的那份参数，而非对话里模型给的原始参数

#### Scenario: 非持久账本不得据「无记录」判为未启动

- **WHEN** 宿主未注册持久 OperationStore，进程真实退出后恢复，账本为空
- **THEN** 恢复保守作结、不重跑（外部副作用恰好一次）；启动时已就该能力缺失告警

#### Scenario: 非法取值响亮失败

- **WHEN** Provider 声明一个不在取值域内的 `recovery_policy`
- **THEN** 启动校验失败并指明合法取值；MUST NOT 静默降级为 `reviewed` 后照常启动

#### Scenario: 存量无身份 dangling 保守作结

- **WHEN** 恢复遇到账本建立之前持久化的 dangling（无账本记录可查）
- **THEN** 作结「无账本记录，无从查证」，不自动执行；不生成随机 id 去碰副作用工具
