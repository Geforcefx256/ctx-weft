# Proposal: tool-result-recovery

## Why

工具输出超过阈值且未配置 SpillSink（或落盘失败）时，全文被静默丢弃，上下文与 memory 只剩头部 1000 字符预览——尾部恰是错误与最终结论的高发区，证据随预览截断同亡；取回全文的唯一途径是重跑工具，对副作用工具是真实风险。核对执行顺序还发现一处 spec 偏离：spill 截断发生在操作账本写入**之前**，账本 completed 记录实际存入的是截断文本，违背 tool-operations「完整规范化结果（非审计截断文本）」条款。

## What Changes

- **结果存储**：超阈值的工具输出全文写入可回取的结果存储（键 = 本次执行身份 invocation_id），支持窗口读取（offset/limit 与尾部直读）；宿主未注册持久实现时 runtime 提供内存默认（与 OperationStore 内存默认同口径），跨进程恢复能力声明如实报告缺失。
- **上下文收敛**：进入对话与 memory TOOL_RESULT 的内容改为收敛版 = 回取引用 + 全长 + 头部预览 + **尾部预览**（尾部止血）；收敛阈值语义保留，`spillable=False` 的自分页工具维持原状。
- **回读工具**：新增 `read_tool_output(invocation_id, offset | tail, limit)` 普通工具，模型可分页/尾部回取全文；其自身输出受分页参数约束，不回灌全文。
- **账本全文语义修复**：账本 completed 改存收敛**前**全文（满足 tool-operations 既有条款）；四个重放/补写入口（completed 短路重放、恢复时「账本完成而 memory 缺失」补写分支【现状缺失，本 change 补齐】、queryable 查询重放、宿主 `supply_result`）进入对话时 MUST 统一重走收敛；重放引用沿用账本原执行 invocation_id 并在存储逐出后以账本全文重新入库（防「有回读指引却查不到内容」）。
- **失败显式**：结果存储写失败时收敛版 MUST 含显式「全文不可用」标记，不再静默 `full output dropped`。
- **SpillSink 降级为可选增值**：宿主可见文件保留，不再是全文唯一归宿；无 SpillSink 时回取能力不依赖它。

## Capabilities

### New Capabilities

- `tool-result-recovery`: 工具长输出的可回取性契约——全文入结果存储、上下文持收敛版（头尾预览+引用+失败显式）、回读工具分页/尾部读取、账本与重放的全收敛分层。

### Modified Capabilities

（无——tool-operations「完整规范化结果或可持久读取的结果引用」既有条款已容纳本修复方向，账本重放收敛要求落在新能力内；capability-gateway 既有需求不变。）

## Impact

- **代码**：capability_gateway（`_maybe_spill` 重构为收敛 + 存储写入、invoke 尾部顺序调整：全文先落 store/账本再收敛）、新 ToolResultStore 协议 + 内存实现 + runtime 注册、builtin/独立 provider 注册 `read_tool_output`、operations ledger 写入点取值改为收敛前全文。
- **行为面**：超阈值工具结果文本形态变化（原「预览 only」→「引用+头+尾」）；账本 result 字段内容变化（截断→全文）。事件审计通道的 `[:8000]` 截断维持不变。
- **依赖**：`compact-fidelity` 的 Evidence 引用回取依赖本 change 的 read_tool_output（该 change 声明依赖）。
- **顺序**：独立 change；与 conversation-pairing 无耦合，可并行。
