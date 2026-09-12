# tool-result-recovery

## Purpose

定义工具长输出的可回取性契约：全文持久于结果存储，上下文（对话与 memory）只承载收敛版（引用 + 全长 + 头尾预览），模型可经回读工具分页取回——消灭「输出超长即证据丢失」，并把账本/重放与上下文收敛分层。

## ADDED Requirements

### Requirement: 全文入结果存储且可窗口回读

工具输出超过收敛阈值时，执行链 SHALL 在丢弃任何内容之前把全文写入结果存储，键为本次执行身份（invocation_id），后续同键重写以最新执行为准。结果存储 SHALL 支持窗口读取：按 offset/limit 分页与从末尾直读（tail）。宿主未注册持久实现时 runtime SHALL 提供内存默认（会话内可回取），声明跨进程恢复能力时 SHALL 如实报告该默认的易失性。

#### Scenario: 大输出尾部证据可找回

- **WHEN** 工具返回远超阈值的输出，关键错误信息位于末尾
- **THEN** 进入上下文的收敛版含尾部预览直接呈现该信息；且经 `read_tool_output` 以 tail 模式可读到包含它的全文片段

#### Scenario: 分页回读完整覆盖

- **WHEN** 模型以 offset 逐页读取直至结束
- **THEN** 各页拼接与原始全文逐字节一致

### Requirement: 上下文收敛形态

进入对话消息与 memory TOOL_RESULT 的超阈值输出 SHALL 为收敛版：回取引用（执行身份 + 回读工具指引）、全文长度、头部预览、尾部预览。尾部预览 MUST 存在（错误与结论高发区）。收敛 MUST NOT 使 memory 记录成为全文唯一副本（全文在结果存储/账本）。`spillable=False` 的自分页工具输出 SHALL 维持不收敛。

#### Scenario: 崩溃恢复后对话仍为收敛版

- **WHEN** 超阈值工具结果已入对话，随后会话崩溃恢复、从 memory 重建对话
- **THEN** 重建出的该结果仍为收敛版，未被任何路径替换为全文

### Requirement: 回读工具自约束

`read_tool_output` SHALL 作为普通工具提供（act 可用），参数含执行身份与读取窗口（offset/limit 或 tail/limit）。其自身输出 MUST 受窗口参数约束并有界，MUST NOT 一次回灌全文；身份不存在或已逐出时 SHALL 返回明确的不可用信息（非空、可区分于空输出）。

#### Scenario: 回读自身不炸上下文

- **WHEN** 全文 1MB，模型以 limit=4000 调用 read_tool_output
- **THEN** 返回不超过所请求窗口的有界内容

### Requirement: 账本与重放的收敛分层

操作账本 completed 记录 SHALL 持收敛前全文（或对结果存储的持久引用），MUST NOT 持收敛版。凡结果进入对话上下文的重放/补写入口——completed 短路重放、恢复时「账本已完成而 memory 缺失」的补写、queryable 查询重放、宿主处置 `supply_result`——MUST 统一重走收敛，禁止全文直灌。重放产生的收敛版 MUST 引用可解析的执行身份：沿用账本记录的原执行 invocation_id（而非重放路径新生成的执行身份）；结果存储未命中该键时 SHALL 先以账本全文重新入库再收敛，重放入库失败时按存储失败显式标记。

#### Scenario: 账本存全文

- **WHEN** 输出超阈值被收敛
- **THEN** 账本 completed 记录中的结果为收敛前全文，长度与原始输出一致

#### Scenario: 重放不回灌全文

- **WHEN** 同逻辑调用命中账本 completed 短路重放
- **THEN** 进入对话的结果为收敛版（含回取引用），与首次执行进入对话的形态一致

#### Scenario: 清空存储后重放仍可回读

- **WHEN** 内存结果存储被清空（逐出/重启），同逻辑调用经持久账本 completed 短路重放
- **THEN** 重放以账本全文重新入库（键 = 账本记录的原执行 invocation_id），收敛版中的回取引用可实际取回内容

#### Scenario: 宿主补结果同样收敛

- **WHEN** 结果未知（unknown）的操作由宿主经 `supply_result` 补入完整结果
- **THEN** 该结果进入对话时为收敛版，非全文直灌

### Requirement: 存储失败显式标记

结果存储写入失败时，收敛版 MUST 含显式「全文不可用（存储写入失败）」标记；MUST NOT 在无标记的情况下呈现为「已截断且可回取」。SpillSink（宿主文件落盘）SHALL 作为可选增值并存（成功时可附带文件路径），全文可回取性 MUST NOT 依赖其存在。

#### Scenario: 存储失败不伪装可回取

- **WHEN** 结果存储写入抛错且 SpillSink 不可用
- **THEN** 上下文中的结果含显式不可用标记，模型据此知道无法回取而非徒劳重试
