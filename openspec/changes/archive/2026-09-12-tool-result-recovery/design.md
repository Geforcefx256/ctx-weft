# Design: tool-result-recovery

## Context

现行收敛点 `CapabilityGateway._maybe_spill`（capability_gateway.py:979）是「截断 + 可选落盘」：无 SpillSink / spill 失败 → 保留头部 `spill_preview_chars`（默认 1000）预览，全文丢弃；执行顺序为 spill（invoke:580）→ 账本 completed 写入（:612-616）→ TOOL_RESULT 落库（:619），故账本与 memory 均只见到截断后文本。SpillSink 仅当宿主注册了 filesystem provider 才存在（:244-247）。MemoryProvider 恒在场，但没有适合「按窗口读取长文本」的既有通路；MemoryBlobStore 是多模态字节外部化（put/get 整体字节），get 无窗口语义、未注册时 `can_externalize=False`。

## Goals / Non-Goals

**Goals:**

- 全文在丢弃前先有归宿；上下文/重放/恢复三个消费面统一走收敛。
- 无宿主配置也有默认能力（内存 store），显式标记失败。
- 账本回到 tool-operations「完整结果」语义。

**Non-Goals:**

- 不做全文搜索（grep 语义）——窗口/尾部读取已满足验收标准主体，搜索列为可选后续。
- 不改事件审计通道的 `[:8000]` 截断。
- 不承诺结果存储的跨进程持久（内存默认即会话内），恢复能力声明如实报告。
- `spillable=False` 工具（read_file 等）的自分页语义不动。

## Decisions

### D1 载体：新建 `ToolResultStore` 协议（put 全文 / get 窗口），不用 MemoryBlobStore

- 窗口读取（offset/tail 切片）是服务端职责——blob store 整体 get 会让尾部读取重复搬运全文；且 blob store 未注册时恒 Null，无法承担默认通路。
- 接口：`put(invocation_id, text)` / `get(invocation_id, offset?, limit?, tail?)` / 有界 LRU 逐出（内存实现，按条数+总字节双上限）。
- runtime 构造期解析：显式注册 > 内存默认（与 OperationStore 同口径，见 capability_gateway 构造注释）。
- **备选（弃）**：memory 侧车记录（新 MemoryKind）——要动 memory 协议与两个 provider 实现，且 load_view 语义（视图/折叠）与「可逐出的原始字节」冲突。
- **备选（弃）**：复用 SpillSink 扩展接口——把宿主文件系统变成唯一真相源，无 fs host 全灭。

### D2 收敛形态与阈值语义

- 收敛版 = `[Tool output truncated: {N} chars; full text via read_tool_output({invocation_id}, offset=..., or tail=...)]` + 头部预览（沿用 `spill_preview_chars`）+ 尾部预览（新配置 `spill_tail_chars`，默认 1000）+ `--- tail (last K chars) ---` 分隔。
- 阈值沿用 `spill_threshold`（4000 字符）与 `cap.spillable` 门控；spill 成功时附带文件路径（增值信息并存）。
- 回读工具注册为普通 ToolCapability（builtin provider 一员，purposes 含 act），`spillable=False`，参数：`invocation_id`、`offset`/`limit` 或 `tail`/`limit`；limit 有硬上限（如 100_000 字符），自身输出再经一次有界包装。

### D3 执行顺序：全文先行，收敛后置

```
stream 完成 → 全文 put 结果存储（失败：显式标记位）
           → 账本 completed（result = 收敛前全文，维持 tool-operations 语义）
           → 收敛（含引用/长度/头/尾/标记）
           → human note 拼接 / 事件（审计 [:8000] 不变）/ TOOL_RESULT memory
```

- 账本持**全文**而非 store 引用：恢复补写路径（账本 completed → memory）只读账本单一来源，无需 store 与账本双源合一；store 只是 live 会话的低成本回取通路（逐出后仍可从账本恢复——重放收敛在 D4）。
- **备选（弃）**：账本持 store 引用——省账本体积，但 store 逐出即引用悬空，恢复语义脆。

### D4 统一收敛入口：四个重放/补写面全走 converge

现状核对（探针结论）：reconcile 遇账本 completed 目前是**直接 continue**——「账本已完成而 memory 缺失」的补写分支并不存在（tool-operations spec 要求的补写是缺失项，dangling 兜底在掩盖它）；宿主 `resolve_operation(supply_result)` 则把完整结果直写 memory，未过任何收敛。本 change 把 converge 定为唯一收敛入口，覆盖四个面：

1. **completed 短路重放**（gateway invoke 账本命中分支）；
2. **恢复补写**：补齐 reconcile 的「账本 completed 而 task 视图缺 result 记录」分支——以账本全文经 converge 补写 memory（同时兑现 tool-operations 的补写要求）；
3. **queryable 查询重放**（QueryOutcome 命中的复用结果）；
4. **宿主 `supply_result`**：补入结果落 memory/对话前过 converge。

无 store 旧数据（改造前账本存的截断文本）补写时原样通过、不二次收敛（长度低于阈值自然直通）。`converge(text, ref_invocation_id)` 帮助函数单点实现，四处调用。

### D5 失败语义

- `put` 抛错：置 `store_failed=True`，收敛版标记 `[full output unavailable: result store write failed]`，不中断调用链（工具结果本身仍回灌）；ERROR 日志留痕。
- `get` 未命中/逐出：返回 `[no stored output for invocation <id>: evicted or written before this session]`——可区分于空输出。

### D6 重放身份与重放入库

- **引用身份 = 账本记录的原执行 invocation_id**：gateway 每次进入（含重放）都会新铸 invocation_id——若收敛版引用重放路径的新 id，指向的是从未入库的键，模型拿着回读指引查不到内容。规则：账本 completed 记录 MUST 附带原执行 invocation_id（现有 OperationRecord 的 memory result 关联字段可承载）；converge 在重放面一律使用账本里的原执行 id。
- **重放先重新入库**：重放面 converge 前先 `store.get(原执行 id)`，未命中则 `put(账本全文)`——使「持久账本 + 易失内存 store」组合在重启/逐出后仍可回读；re-put 失败按 D5 存储失败显式标记（引用保留但注明当前不可回取）。
- 验收含「清空内存 store → 从持久账本（SQL 实现）重放 → 回读成功」端到端用例（tasks 3.3）。

## Risks / Trade-offs

- [内存默认 store 被超大输出灌爆] → LRU 双上限（条数 + 总字节，默认如 64 条 / 16MB，可配）；超限逐出最旧并如实返回未命中。
- [收敛版比原预览长（+尾部+引用），极端高频工具轻微增耗] → 增量 ~1KB 量级，预算估算按内容自然计入；阈值不变。
- [read_tool_output 被 LLM 滥用于反复全量读取] → limit 硬上限 + 窗口语义（无 offset/tail 时默认首页，不给全文）；与 read_file 同款分页提示措辞。
- [invocation_id 跨重试变化导致回取指向旧执行] → 语义即「按执行回取」：收敛版嵌的就是当次执行的 id；重试产生新全文与新收敛版，旧条目 LRU 逐出。
- [与 conversation-pairing 并行实施的文本格式互相踩] → 收敛发生在 gateway 出口，配对在摄入点，互不触碰；两 change 测试各自独立。

## Migration Plan

纯增量：store 缺省内存实现即生效，无开关、无数据迁移。回滚 = 还原代码（行为回到「预览 only」，无残留状态）。

## Open Questions

（无——载体、顺序、失败语义、回读约束均已定；全文搜索明确列为可选后续而非本 scope。）
