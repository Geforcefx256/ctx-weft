# event-log

## Purpose

定义事件日志的有序提交契约：同会话提交位置（position）唯一单调、批次原子提交、batch_id 幂等重试与冲突检测，为提交确认（H1）与按提交位置截断的快照恢复（H2）提供存储层地基。

## Requirements

### Requirement: 同会话提交位置唯一且单调

有序提交（`append_batch` / `read_range` / `committed_head`）是 `EventStore` 的**必需**部分，不是可选扩展、也不另立平行 Protocol：三者均为 `@abstractmethod`，Runtime 构造期再以 `supports_ordered_commit(store)` 校验一道（拦鸭子类型实现），缺一即拒绝启动。恢复路径 SHALL NOT 提供按事件 ID 排序的回落分支。EventStore SHALL 为同会话的每次成功提交分配 position：同会话内 SHALL 严格递增、永不重用、无空洞之外的无序分配；`committed_head(session_id)` SHALL 返回该会话最新已确认提交的 position。`event.id` 继续作为事件身份，SHALL NOT 被用作提交顺序依据；同 event.id MUST NOT 分属两个 session。

#### Scenario: 位置随批次递增

- **WHEN** 同会话先后提交两个批次
- **THEN** 第二个批次所有事件的 position 严格大于第一个批次的最大 position，`committed_head` 等于第二个批次的最大 position

#### Scenario: 同一事件不落两个会话

- **WHEN** 携带某 event.id 的批次已提交到会话 A，再向会话 B 提交同一 event.id
- **THEN** 提交被拒绝（冲突或唯一约束失败），不会在两个会话各存一份

### Requirement: 批次原子提交

`append_batch` 对一批事件 SHALL 全有或全无：批内任一写入失败时整批不存在（含部分写入回滚），调用方可以原样重试；批内所有事件 MUST 属于同一 session。

#### Scenario: 批中第 k 条失败整批回滚

- **WHEN** 一个 N 条事件的批次中第 k 条因约束失败
- **THEN** 该批次 0 条落库，已存在的其他批次不受影响，同 batch_id 重试成功后恰好落库 N 条

### Requirement: batch_id 幂等与冲突检测

相同 batch_id 的重复提交 SHALL 幂等：内容一致（事件 envelope/payload 逐字段，忽略存储分配的 position）时返回原 CommitReceipt；内容不一致时 SHALL 抛出 `EventConflictError`。不允许「相同 ID 任意内容都忽略」。batch_id 在第一次提交前生成，重试 MUST NOT 更换。

#### Scenario: 确认丢失后原样重试

- **WHEN** 一个批次已提交但调用方未收到 receipt，用同一 batch_id 与同一内容重试
- **THEN** 返回与首次提交一致的 receipt（含相同 position），无重复事件

#### Scenario: 同 batch_id 不同内容

- **WHEN** 用已存在的 batch_id 提交内容不同的批次
- **THEN** 抛出 `EventConflictError`，原批次不变

### Requirement: append 单事件兼容

既有 `append(event)` SHALL 由单事件 `append_batch` 实现，batch_id 确定性取 event.id；既有调用方（persister 等）行为与语义零变化。

#### Scenario: 逐条 append 与批次提交共存

- **WHEN** 同会话先逐条 append 两条、再 append_batch 两条
- **THEN** 四条事件按提交顺序获得递增 position，read_by_session 返回全部四条

### Requirement: 跨会话隔离

不同会话的提交 SHALL 互不阻塞，各自维护独立的 head 与 position 序列。

#### Scenario: 两会话并发提交

- **WHEN** 会话 A 与会话 B 并发 append_batch
- **THEN** 互不等待对方完成，各自 position 序列独立正确

### Requirement: 并发争用安全

两个连接争用同一会话 head 时 SHALL 串行化分配：不产生重复 position、不丢批次；验证 MUST 使用数据库级并发（两个独立连接/任务），不以单协程顺序调用代替。存储实现 MUST NOT 使用无锁的 `MAX(position)+1` 分配。

#### Scenario: 双连接同会话争用

- **WHEN** 两个独立连接并发向同一会话提交批次
- **THEN** 全部成功，position 无重复无空洞交错错乱，两次 committed_head 观测单调

### Requirement: 按位置读取

`read_range(session_id, after_position=0, through_position=None)` SHALL 返回 position 升序的 StoredEvent 列表，且只含已确认提交的事件；`through_position` 截断上界包含该位置本身。

#### Scenario: 增量读取边界

- **WHEN** 会话已提交 position 1..5，read_range(after_position=2, through_position=4)
- **THEN** 恰好返回 position 3、4 两条事件

### Requirement: legacy 迁移确定性

迁移工具 SHALL 默认 dry-run（实际迁移必须显式参数）；旧数据按 `(session_id, event.id)` 排序分配 position；SHALL 报告会话数、事件数与不完整事件；SHALL NOT 声称还原了历史实际提交顺序（旧契约没有记录它）。

#### Scenario: dry-run 默认与报告

- **WHEN** 不带显式迁移参数运行迁移工具
- **THEN** 不写入任何数据，输出会话数/事件数/不完整事件计数；带显式参数执行后，同会话事件获得按 (session_id, event.id) 排序的连续 position
