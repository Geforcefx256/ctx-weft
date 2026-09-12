# Design: tool-schema-budget

## Context

两处既有代码已含 tools 面费率，但只覆盖「无基线首估」：`prepare._estimate_assembled_tokens`（无基线时对整份 prompt 估，含 schema；注释自认「composer 完全没数，此处补上」）与 `llm_gateway._estimate_request_tokens`（整份路径）。有基线后的两条增量路径（prepare `_estimate_tokens`：基线+新 memory 记录；`request_prompt_estimate`：基线+新消息）均不含工具面。装配预算（`PriorityBudgetStrategy.apply`）只对 block 的 token_estimate 求和，capability 工具块估算= description，schema 只经 API tools 参数下发（`build_llm_tools` 明文「不受裁剪」）。工具面是活视图（`AssembledPrompt.tools` 读 cache 现算，pin 当轮可见），装配期快照会过期——这正是循环期也要追踪的原因。

## Goals / Non-Goals

**Goals:**

- 工具面成本在装配与循环两处入账，动态变化可感知。
- 单一费率真源，收敛两处既有重复公式。
- 发送前不硬拒；超限可观测。

**Non-Goals:**

- 不按 schema 体积裁剪/降级工具声明（声明完整性优先）。
- 不动 capability 块 prose 估算的既有口径（description vs name+sig 的不严格对应是独立小问题，预留只针对 API 工具面新增成本，避免双计）。
- 不引入精确 tokenizer 级 per-tool 缓存（估算成本可控，见 D1）。

## Decisions

### D1 费率单一真源：`estimate_tools_tokens(tools, count)` 落 core.utils.estimate

- 公式 = 现有两处的并集：`count(name) + count(description) + count(json.dumps(input_schema))`；prepare 与 gateway 的重复实现改为调用它。
- 成本：每装配/每请求一次 schema 序列化。MCP 大 schema 场景（几十工具 × 数 KB）估算耗时在毫秒级，可接受；如将来成为热点，按指纹缓存估算值（Open Question 留待实测）。

### D2 装配扣减点：assemble 内、budget.apply 前

```
tools_snapshot = build_llm_tools(cache.available(agent, task), purpose)   # compact 为空，天然零预留
reserved = estimate_tools_tokens(tools_snapshot, token_counter)
kept = budget.apply(blocks, effective_limit − reserved, request)
AssembledPrompt.metadata: tools_reserved_tokens = reserved,
                          tools_signature = 指纹
```

- **指纹口径 = 与估算同源的规范化工具定义**：对每个工具的 `name + description + input_schema` 规范化序列化（schema JSON 按 key 排序），按 name 排序后整体哈希。**只哈希名字不够**——审核指出同名工具的 description/schema 更新后指纹不变、增量路径会跳过成本调整；指纹必须覆盖与估算完全相同的输入（同一份规范化转储一次序列化、两用：喂估算 + 喂哈希）。
- cache 缺失（测试直构 assembler）时预留 0，行为与现状一致。
- **为什么扣减而非给 budget 传假 limit**：`ContextOverflowError` 的 `effective_limit` 字段要反映真窗口——报错信息用真值、裁剪用扣减值，两者分开传。

### D3 循环增量算法：`est = ctx_tokens + msg_delta + (new_tools − last_tools)`

- 真实基线 `ctx_tokens` 是上次发送的实测（含旧工具面），故增量必须用**差值**而非新面全量（全量会重复计入旧面）。
- `loop_guard` 新增两个瞬态字段：`last_tools_signature` / `last_tools_est`（发送成功回填 usage 时更新）。指纹变化 → delta += `estimate_tools_tokens(request.tools) − last_tools_est`；不变 → 维持纯消息增量。
- 指纹取 D2 同一算法（规范化工具定义哈希，**覆盖 name+description+schema**——名称不变而 schema 变大时指纹必须变化，这正是增量要捕捉的成本变化）；每次请求构建规范化转储与上次哈希比较，不变时零估算开销，变化时同一次转储喂估算。

### D4 发送前策略：不硬拒（明示决策）

- 拒绝项：硬拒（估算误差下误伤真实可发请求；provider 自身 400 仍是最终防线，且已有自愈退避）。
- 生效路径：`apply_dynamic_max_tokens` 的 used 已含工具面（D3 后估算正确）→ max_tokens 自动收紧；超限时新增 WARNING（含 used / eff / tools_signature）；压缩由既有 `compact_token_ratio` 机制在下一 prepare 触发。
- **备选（弃）**：发送前整份重估校验——每请求多一次全量估算开销，且与增量路径口径重叠；增量已含工具面后无增益。

### D5 与 context-evidence-delivery 的组合口径

- 两 change 皆落地时：`有效预算 = effective_limit − tools_reserved`（本 change），证据地板在其剩余内生效（彼 change D4）。组合用例各自侧一条。

## Risks / Trade-offs

- [估算偏高（schema 序列化形态与 provider 实际计费有差）挤掉本可保留的内容] → 偏差是双向的，tokenizer 校准回喂已存在；偏差留痕（R4）让漂移可见；预留只影响裁剪阈值不影响正确性。
- [last_tools_est 与真实基线不同源（估 vs 测）造成差值失真] → 差值只在指纹变化时介入，首变轮偏高（新面全按估）、随后被实测基线吸收——方向安全（估算只偏高）；留痕可观测。
- [loop_guard 新增字段与持久化格式耦合] → 字段为瞬态（恢复时重算），不进快照持久层；设计上放 guard 内存态。

## Migration Plan

默认生效，无配置开关（预留是正确性修复）；`AssembledPrompt.metadata` 新键为附加信息，消费方可忽略。回滚 = 还原代码，无状态残留。

## Open Questions

- schema 估算按指纹缓存是否必要（当前判断不需要，待大 MCP 场景实测 p95 后再议——不影响接口）。
