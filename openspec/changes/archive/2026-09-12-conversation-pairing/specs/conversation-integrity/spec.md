# conversation-integrity

## Purpose

定义对话重建中工具调用与结果配对的完整性契约：调用标识在摄入点铸造为内部唯一值，live 对话与 memory 重建两个平面同口径，使跨轮次/跨任务的重复 wire id 不再造成错配、复制或孤弃——对话面的「tool_call_id 不作匹配依据」原则。

## ADDED Requirements

### Requirement: 摄入点铸造内部唯一调用标识

每个 assistant 回合被系统摄入（写入 memory 与追加活动消息列表，两者 SHALL 由同一次铸造供值）时，其携带的每个 tool_call 的 id SHALL 被替换为内部唯一标识：由回合锚（assistant 记录锚）与调用序号确定性派生，跨轮次、跨任务、跨会话全局唯一；其字符集与长度 MUST 满足主流 provider 对工具 id 的约束——适配层原样透传即合法，MUST NOT 依赖任何 provider 端的编码、截断或改写。原始 LLM wire id MUST 保留于记录 metadata（`raw_tool_call_id`）供审计与追溯。provider 对 id 值是黑盒——仅要求同一请求内 assistant.tool_call.id 与配对 tool.tool_call_id 一致，因此替换 MUST NOT 改变任何请求的合法性。后续对该调用的全部引用（gateway 执行、TOOL_RESULT 配对、HITL 登记与决定缓存键、delegate 回填的 origin_tool_call_id、事件 payload）SHALL 一律使用内部标识，形成单一平面。

#### Scenario: 铸出的 id 满足 provider 约束

- **WHEN** 任一内部标识经 openai / anthropic adapter 构建请求 payload
- **THEN** id 原样出现在 payload 中（无编码/截断），且符合两家 provider 公开的工具 id 字符集与长度限制

#### Scenario: 跨轮次重复 wire id 不错配

- **WHEN** 第 2 轮与第 9 轮 assistant 回合都携带 wire id `call_1`，会话经 memory 重建（observe 边界 / 恢复 / 压缩后）
- **THEN** 两条 result 各自紧邻其真正所属的 assistant 回合，预算裁剪与发送前合法化均不将第 2 轮的结果归到第 9 轮名下，也不产生重复发出的 result

#### Scenario: 跨任务召回混合历史不错配

- **WHEN** agent 层召回把多个已完成 task 的对话与当前 task 历史按时间归并，其中两个 task 存在同 wire id 的调用
- **THEN** 配对仅发生在同回合锚的内部标识之间，跨 task 不串扰

#### Scenario: 原始 wire id 可审计

- **WHEN** 查看任一新产生的 assistant 回合记录
- **THEN** 每个内部标识旁可取到对应的 `raw_tool_call_id` 原值

#### Scenario: 活动对话与重建同口径

- **WHEN** act 循环内下一轮 LLM 请求的 messages 与（若此刻崩溃恢复）从 memory 重建的 messages 含同一回合
- **THEN** 该回合的 tool_call id 在两个平面上逐字节一致

### Requirement: 发送前合法化不产生复制与孤弃（内部标识平面）

发送前合法化（剥离悬挂调用、重排结果紧邻其调用、丢弃孤立结果）SHALL 以内部标识为唯一配对依据；在内部标识平面上，合法化 MUST NOT 将同一结果重复发出到多个调用之后，也 MUST NOT 因他处存在同 id 结果而误判某调用已有配对。**存量裸 wire id 记录不在此保证范围内**：其参与重建时合法化行为维持改造前现状（重复 id 下可能出现复制），但此类歧义命中时 SHALL 留下可检索的留痕（日志指明涉及的重复 id 与受影响消息），MUST NOT 静默。

#### Scenario: 同 wire id 多携带者不复制结果（内部标识平面）

- **WHEN** 重建历史里两个 assistant 回合各自携带**内部标识**互不相同但源自相同 raw id 的调用（跨轮次复用 call_1 的正常新数据）
- **THEN** 每条 result 至多出现一次，且位于其记录所锚定的回合之后

#### Scenario: 存量重复 id 维持现状且留痕

- **WHEN** 重建历史里两个 assistant 回合携带相同的**存量裸 wire id**（改造前写入）
- **THEN** 合法化行为与改造前一致（不因此新增错配），且留痕指明该重复 id 歧义已被命中

### Requirement: 裁剪以配对单元同生共死

上下文预算裁剪 SHALL 以配对单元（assistant 调用回合与其全部结果）为最小丢弃单位。单元内成员 MUST NOT 被分开保留或分开丢弃；因配对歧义产生的孤立结果 MUST NOT 随无关单元幸存进入最终请求。

#### Scenario: 丢弃调用则其结果同命运

- **WHEN** 预算裁剪丢弃某个 assistant 调用回合
- **THEN** 该调用的全部 result 同批丢弃，最终请求不出现无调用对应的孤立 result

### Requirement: 存量数据兼容

本 change MUST NOT 要求迁移存量 memory 记录。存量裸 wire id 记录参与的重建，配对行为 MUST NOT 劣于现状；内部标识的形态（含分隔符的复合值）SHALL 保证不与任何存量裸 id 相等。

#### Scenario: 旧库重建行为不回退

- **WHEN** 会话含改造前写入的裸 id 记录，与改造后新写入的内部标识记录混合重建
- **THEN** 新记录配对正确；旧记录的配对结果与改造前一致（不因混合而新增错配）
