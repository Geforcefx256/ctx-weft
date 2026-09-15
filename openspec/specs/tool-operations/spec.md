# tool-operations

## Purpose

定义工具执行的稳定身份与恢复判据：跨重启的同一逻辑调用拥有稳定标识，副作用是否已发生**可从 capability 事件流折出**，结果不确定表现为一种工具结果——堵死「恢复盲重跑副作用」（H3）。

## Requirements

### Requirement: 稳定逻辑调用身份

每个工具调用 SHALL 拥有跨重启稳定的身份——**即该调用在摄入点铸造的内部 tool_call 标识**（`tc_...`，见 capability `conversation-integrity`），随 assistant 回合一同持久化，也是 capability 事件 payload 的配对键；恢复时**读出**而非重算。SHALL NOT 另立第二条派生身份（曾有一个由 `(tenant, session, agent, record_id, ordinal)` 派生的 `operation_id`，铸造落地后它只是同一个值的第二个名字，连名字也已退场）。`invocation_id` 保留为单次执行尝试身份（取消用、结果存储键）。未经铸造的裸 wire id（宿主直构 / 测试替身 / 存量数据）MUST NOT 用作恢复判据键——跨回合复用会互相覆盖。普通执行、热 HITL resume、冷恢复 SHALL 引用同一标识；两条内容相同的合法调用 MUST 得到不同标识（不误去重）；模型复用同一 wire id MUST NOT 使旧结果覆盖新调用。silent/dispatch 类控制工具 SHALL 同样留下 capability 事件（不入对话 ≠ 不留痕）。

#### Scenario: 同一调用跨重启身份不变

- **WHEN** 一次工具调用在崩溃后经冷恢复重入
- **THEN** 恢复路径读到与首次执行相同的内部标识（随消息落库）

#### Scenario: 两次合法同参调用不去重

- **WHEN** 两条 assistant 消息都用 call_1 且参数相同
- **THEN** 两个不同的内部标识（铸造时 anchor/ordinal 不同），各自完整执行

#### Scenario: call_1 复用不串扰

- **WHEN** 后一轮 LLM 回合复用了前一轮的 tool_call_id
- **THEN** 两轮调用各自的内部标识独立，前一轮的结果不配对到后一轮

### Requirement: 恢复判据来自 capability 事件流

一次工具调用的执行记录 SHALL **就是它在事件流里留下的痕迹**；core MUST NOT 为此另设存储，protocols MUST NOT 为此暴露存储协议。

`CapabilityInvoked` SHALL 在调用 provider **之前**发出，且在 `event_commit_policy="required"` 下由提交门确认写入后才返回——这就是「我要动手了」的持久前置。`CapabilityFinished` SHALL 携带**进入对话的那份结果**（收敛版）与 `result_length`。据此折出的事实与旧账本一一对应：

| 旧账本状态 | 事件流 |
|---|---|
| prepared / 无记录 | 没有 `CapabilityInvoked`（授权拒绝与参数校验失败都排在它之前，故不留痕） |
| started | 有 INVOKED、无 FINISHED |
| completed | 有 FINISHED |
| waiting_human | HITL 自己的事件集与折叠 |

折叠 SHALL 是纯函数、**不落库、不进快照**（与常驻投影分开）；读取 SHALL 走按类型的轻查询，MUST NOT 全量回放。折叠结果 MUST NOT 直接递给宿主实现——`RerunAuthorizer` 收到的是只读视图 `RerunContext`（身份 + 尝试记录），不含 core 内务，也不含未脱敏执行参数。

调用工具 SHALL 关闭未提交窗口（与「首个 LLM chunk」同为不可逆动作）：否则 `discard_round` 会把一次**已经产生副作用**的调用从日志里抹掉。判据 SHALL 是「调了工具」本身，MUST NOT 依赖 `side_effects` 或 `recovery_policy`（前者声明可能不完整，后者的「重跑安全」不等于「撤销安全」）。

同逻辑调用重入且已有结局时 SHALL 复用事件里那份结果、不再调 provider，且短路 MUST 发生在发 `CapabilityInvoked` **之前**（否则留下一条孤立的 INVOKED）。事件里那份被截断时（`spillable=False` 的不收敛输出可能超 payload 上限）MUST NOT 用于重放——那类工具皆只读可重新派生，SHALL 重跑。

#### Scenario: 进程在 provider 执行中消失

- **WHEN** `CapabilityInvoked` 已提交、provider 执行中进程退出，随后恢复
- **THEN** 折出 invoked 而未 finished；副作用可能已发生，按恢复策略分派

#### Scenario: 授权拒绝不留痕

- **WHEN** 调用被 authorizer 拒绝或参数校验失败
- **THEN** 事件流中没有该调用的任何 capability 事件；恢复据此断定 provider 从未被调用，首执安全

#### Scenario: 已完成而 TOOL_RESULT 缺失

- **WHEN** `CapabilityFinished` 已发、TOOL_RESULT 写入前崩溃，随后恢复
- **THEN** 以事件里那份（**不重新收敛**）按确定性 id 幂等补写 memory，provider 不被重执行

#### Scenario: 窗口里的调用不被撤销抹掉

- **WHEN** 未提交窗口开着时执行了工具，其后该轮被 discard
- **THEN** 该次调用的 capability 事件仍在日志中——已发生的副作用不得被否认

### Requirement: 恢复策略表

`ToolCapability` SHALL 声明 `recovery_policy`，取值**恰为两类**（`idempotent | reviewed`），默认 `reviewed`（不从 side_effects 推断安全）。分类判据是「core 要不要做决定」：`idempotent` = 重跑安全，core 同一标识直接重跑；`reviewed` = core MUST NOT 自行重跑，交裁决链。

无法识别的取值 SHALL 在启动校验响亮失败，MUST NOT 静默降级为保守值。

重跑授权 SHALL 由**宿主注册**，MUST NOT 要求由 provider 对象自身实现——知道怎么查证外部真值的对象未必是提供工具的对象（MCP 工具尤其如此），绑在 provider 上则宿主无法为自己不控制的 provider 补上查证能力。控制工具 SHALL 声明 `idempotent`（core 自有的同身份幂等状态迁移），MUST NOT 在 gateway 里写成按名字前缀的特判。

恢复时按事实×策略分派：已 finished 复用结果不重执行；从未 invoked 可首次执行；invoked 未完成 + `idempotent` 以同 tool_call_id 重试；invoked 未完成 + `reviewed` 问**重跑授权**（`RerunAuthorizer.authorize_rerun`），它 SHALL 返回 `AuthorizationDecision`：`allowed=True` 以同 tool_call_id 重跑；`allowed=False` 把 `message` 写成这次调用的工具结果并作结（`result_is_error` 由授权方按内容决定——查到外部真结果时 MUST NOT 标成错误）；`needs_human` 走既有 HITL park（停在人那儿由 HITL 自己的账表达，决定缓存第三维 SHALL 是独立的 rerun stage，MUST NOT 与事前授权共用键）。未注册重跑授权时 core SHALL 代为作结「无从查证」，同样不重跑。

重跑授权 SHALL 与事前授权 `Authorizer` **分开注册、分开解析**（provider 级 + 逐工具级，无 default = 不重跑），MUST NOT 复用 `authorize` 回答重跑问题——默认放行型 authorizer 会对它答「允许」，等于自动重跑副作用。二者 SHALL 在同一次恢复 invoke 中依次执行：先重跑授权、放行后再走事前授权（「当初准跑」不等于「现在还准跑」），MUST NOT 互相顶替。

折得出事实而其中没有 `CapabilityInvoked` SHALL 视为确定未启动而首次执行——事件库在 required 模式下本来就必须持久（那是提交门的前提），不存在「没跑过」与「记录随进程消失」的歧义。折不出事实时（未接查询 / 裸 wire id）SHALL 保守作结。

「结果无法确定」SHALL 表现为一种**工具结果**，MUST NOT 成为一种控制流：MUST NOT 因此停机、MUST NOT 要求宿主介入才能续跑、MUST NOT 为它设立专属错误码/事件类型/处置 API。裁决者判不了时用 `result` 的**文本**说明查到了什么、查不到什么——那是内容不是状态，core 不替它组织措辞，也不据此分支。重复调用的防护归 Provider 自理：core 的承诺止于「不自行重跑」，agent 下一轮主动再调是一次新的逻辑调用。一切结果不定情形一律不自动执行；`waiting_human` 经既有 HumanResumable 协议恢复，不从头重新 invoke。Reconcile 的完成匹配 SHALL 以内部标识为判据（事件流的 finished ∨ 确定性 TOOL_RESULT 记录 id），MUST NOT 以 wire id 集合判定（防 call_1 复用串扰）。无内部标识的存量 dangling SHALL 同样作结「无从查证」，MUST NOT 以随机生成的 id 自动执行副作用工具。控制工具单独核验：delegate 以内部标识找回已创建子任务（确认丢失重入不生成第二棵子树）、finish/metadata 同身份幂等、ask_user 复用已有请求；MUST NOT 对控制工具整体标记 `idempotent` 后省略验证。

#### Scenario: reviewed 且无重跑授权时不重跑

- **WHEN** started 后崩溃且策略为 reviewed，宿主未注册 RerunAuthorizer
- **THEN** core 代为作结「无从查证」：该说明文本成为这次调用的工具结果、并补发 `CapabilityFinished` 闭合那条悬着的 INVOKED，恢复不重执行、会话照常续跑；这是默认形态，不报配置错误

#### Scenario: idempotent 重跑恰好一次补全

- **WHEN** started 后崩溃且策略为 idempotent，恢复重入
- **THEN** 以同一标识重试一次；外部副作用由 Provider 承诺幂等；事件流与 memory 各落一次

#### Scenario: 重跑授权的权威放行

- **WHEN** started 后崩溃且策略为 reviewed，重跑授权返回 `allowed=True`（权威判定「这次没跑成」）
- **THEN** 以同 tool_call_id 重新执行；其后仍走一次事前授权

#### Scenario: 判不了也只是一种结果

- **WHEN** 重跑授权查不到外部真值，返回 `allowed=False` + 说明文本
- **THEN** 不重跑，该文本成为这次调用的工具结果；task 状态不变、无专属事件、无需宿主介入，agent 下一轮据此自行决定

#### Scenario: 交给人

- **WHEN** 重跑授权返回 `needs_human`
- **THEN** 走既有 HITL park；人批准则以同一标识重跑，拒绝则用人写的话作结；决定按 rerun stage 缓存，冷恢复复用不再问第二遍

#### Scenario: 重跑授权拿不到凭据明文

- **WHEN** 重跑授权被问到
- **THEN** 它收到的 `RerunContext` 只含身份与尝试记录，**不含执行参数**——要按参数查证，由 provider 在执行时以 `ctx.extra["tool_call_id"]` 作幂等键自行留存；核心 MUST NOT 替宿主保管未脱敏参数

#### Scenario: 重入不留孤立的 INVOKED

- **WHEN** 同逻辑调用重入且已有结局
- **THEN** 复用事件里那份、不打 provider，且不新发 `CapabilityInvoked`；折出的尝试数不增加

#### Scenario: 非法取值响亮失败

- **WHEN** Provider 声明一个不在取值域内的 `recovery_policy`
- **THEN** 启动校验失败并指明合法取值；MUST NOT 静默降级为 `reviewed` 后照常启动

#### Scenario: 存量无身份 dangling 保守作结

- **WHEN** 恢复遇到裸 wire id 的 dangling（跨回合会互相覆盖，不能当判据）
- **THEN** 作结「无从查证」，不自动执行；不生成随机 id 去碰副作用工具
