# snapshot-recovery

## Purpose

定义快照恢复的一致切面契约：快照必须恰好由不超过某已确认提交位置的事件生成，全量回放与快照+增量两条恢复路径产出等价领域状态——堵死「延迟提交的旧事件 ID 被快照增量跳过」（H2）。

## Requirements

### Requirement: 一致切面算法

快照创建 SHALL 固定为三步：`C = committed_head(session)` → 按 `read_range(after=S.cursor 或 0, through_position=C)` 取已提交事件并 apply（S 为最新可用快照，可为空）→ 保存携带 `last_commit_position=C` 与 `projection_version` 的快照。快照触发事件 SHALL 只作为「请求做快快照」的信号，MUST NOT 作为快照边界本身；实现 MUST NOT 取一次无上界的最新 view 再把较早事件的位置写成游标。获取 C 之后发生的并发提交 MUST NOT 混入该快照 blob（由下一次快照/增量应用一次）。

#### Scenario: 延迟提交不再丢失

- **WHEN** 事件 B 已提交并触发快照（C=较大 position），事件 A（较小 position、较晚提交）随后落库
- **THEN** 下一次快照/恢复按 position 截断，A 与 B 都出现在快照恢复结果中，与全量回放等价

#### Scenario: 取 head 后并发提交不混入

- **WHEN** 快照 writer 取得 C 后同会话又有新事件提交
- **THEN** 快照 blob 不含超过 C 的事件；新事件由其后的 delta 路径应用恰一次

### Requirement: 两路恢复等价

对任意会话，全量回放（read_range 全量 apply）与 snapshot(C)+delta(position ∈ (C_cursor, head]) SHALL 产出等价领域状态——比较范围至少覆盖 session/task/agent/HITL/outputs 业务字段，不只 task 集合；两条路径 MUST 使用同一排序语义（position）。

#### Scenario: 等价性对照

- **WHEN** 同一会话分别走全量回放与快照增量恢复
- **THEN** 归一化随机 ID/时间戳后，两份投影的业务字段逐项一致

### Requirement: 不可用快照降级

快照损坏、`projection_version` 与当前实现不匹配、或快照缺 `last_commit_position`（legacy）时，恢复 SHALL 忽略该快照并按 position 有序全量回放重建后重新创建快照；MUST NOT 猜测位置继续增量读取。快照创建失败 SHALL 表现为性能降级（记录错误、可从日志重建），不是数据丢失。

#### Scenario: legacy 快照忽略重建

- **WHEN** 持久化中的最新快照无提交位置或版本不匹配
- **THEN** 恢复走全量回放得到正确状态，不使用该快照也不报致命错；后续新快照携带 position 与版本

### Requirement: 折叠策略——常态增量，全量为重锚

快照创建在有可用基底时 SHALL 走增量（`read_range((S.cursor, C])` 应用到 S 的 blob 上），使写入代价为 `O(delta)`；MUST NOT 每次都从 0 全量折——写入路径内联于事件发射，`O(全部事件)` 的折叠会随会话增长阻塞主循环。

基底可用性判据 SHALL 与恢复路径共用同一实现（避免判据漂移导致 writer 基于 reader 不认的快照做增量）。

全量折 SHALL 保留为**重锚**，且至少在下列时机各触发一次：无可用基底（首张 / 缺 position / 版本不匹配 / 位置超前）、快照链深达到实现声明的上限、以及进程启动后首次为某会话写快照。重锚的作用是让快照周期性地重新对齐日志，纠正历史快照可能累积的偏差（如 blob 序列化往返的有损累积）。

两种模式下快照 blob SHALL 恒等于 `fold(0..C)`：投影的增量 apply 与全量 reduce 必须是同一个左折叠，等价性由结合律保证，不依赖折叠策略。

#### Scenario: 常态写入不重折全部事件

- **WHEN** 同一进程内已为某会话重锚过，且最新快照可用
- **THEN** 下一张快照只折 `(S.cursor, C]` 区间，其 blob 与「从日志全量折到 C」逐字段一致

#### Scenario: 链深到顶触发重锚

- **WHEN** 快照连续增量层数达到实现声明的上限
- **THEN** 下一张快照从日志全量重建，链深归零，其后继续增量

#### Scenario: 进程启动后首张为重锚

- **WHEN** 进程启动后首次为某会话写快照（无论持久层是否已有该会话的可用快照）
- **THEN** 该张快照从日志全量重建——上一个进程可能留下的偏差在此被纠正
