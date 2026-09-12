# Design: conversation-pairing

## Context

配对错乱的四个现场（见 proposal Why）共享同一根因：`LLMMessage` 平面上的 `tool_call_id` 是 LLM 自由复用的 wire 值，而所有配对点（`budget._coalesce_tool_pairs` 的 owner 字典、`llm_gateway` 的 reorder/dangling/orphan、`_history` 重建）都拿它当全局键。恢复面已由 tool-operations 用 `operation_id` 解决，但对话面没有等价物——flat `LLMMessage` 一旦离开摄入点就丢失回合元数据，**在配对点修补（方案 A）无法补干净**；直接用 `operation_id` 配对（方案 C）则要求重建路径先携带 record_id/ordinal 元数据，改造面反而更大。已选**方案 B：摄入边界唯一化**——重复只可能经「跨回合」进入，而跨回合必经摄入点，在一处根治。

现状约束：

- act 循环的消息平面：`current_messages`（live）与 memory 记录（重建）双轨并存，同回合必须同值（spec 场景「活动对话与重建同口径」）。
- `invocation_key`（HITL 决定缓存第四维）已把「同 id 的另一次调用」隔开——保持不动，作双保险。
- reconcile 重入、delegate 回填、`_error_and_record` 错误补写都以 `tool_call_id` 参数串联——它们都从 act 摄入点拿值，天然跟随。

## Goals / Non-Goals

**Goals:**

- 一处铸造、全平面（live messages / memory 记录 / gateway 透传 / 事件 / HITL）同值。
- 配对点零算法改动：`_coalesce_tool_pairs`、legalize 四函数、`_history` 原样保留。
- 存量零迁移，行为单调不劣化。

**Non-Goals:**

- 不改恢复面匹配（tool-operations 既有契约，reconcile 仍按 operation_id）。
- 不给存量裸 id 记录补铸标识（重建时旧配对行为维持现状：last-write-wins，不更糟）。
- 不改 `invocation_key` 四维键结构。

## Decisions

### D1 内部标识形态：provider 安全的确定性 ID——`tc_{seq36}_{ord36}_{hash12}`

- **字符集 `[A-Za-z0-9_]`、长度 ≤ 64**：Anthropic 对工具 id 有字符限制（冒号等非法），openai/anthropic adapter 均原样透传——复合值必须落在两家约束的交集内。铸 id 的纯正则与长度断言 + 两家 adapter payload 透传断言（tasks 1.2/3.2）。
- 构成：`tc_` 前缀 + base36(回合摄入序号) + `_` + base36(回合内调用序号) + `_` + sha256(anchor|ordinal|raw_id) 前 12 hex。前两段人可读（定位回合与调用），哈希段保证跨会话确定唯一；`raw_tool_call_id` 全量留在 metadata，追溯不依赖 id 形态本身。
- **确定性**而非随机：同一次摄入的两次供值（memory / current_messages）天然同值；崩溃重放同输入同 id。
- **与存量不撞**：`tc_` 前缀 + 无冒号形态不可能等于任何存量裸 wire id（provider 生成的 call/toolu 值不含该结构）。
- 铸造时在回合内做一次唯一性自检（同回合撞哈希即抛——12 hex 撞率工程上为零，自检是防实现错误的断言口）。
- **备选（弃）**：`{anchor}:{ordinal}:{raw_id}` 冒号复合形态——Anthropic 工具 id 字符集不接受，adapter 端再做编码映射等于把约束散进适配层，且映射可逆性是新的故障面。
- **备选（弃）**：直接复用 `operation_id`——语义重载（操作身份 vs 对话配对），且 ledger 不在场的路径（裸调 gateway 的测试/宿主直构）无值可用；作为 metadata 伴随字段同时写入（`operation_id` 若已铸）。

### D2 铸造点：assistant 回合摄入的公共函数，双平面一次供值

- act 收到 LLM 回合（含 tool_calls）后、**任何下游消费之前**完成铸造：memory 摄入与 `current_messages` 追加从同一次铸造取值。重入路径（reconcile、park 恢复）都汇聚到同一摄入函数，不出现第二铸造点。
- **为什么不在 gateway 内铸**：gateway 看到的是单个调用，不见回合锚；且 dispatch/SILENT 分支各有落库点，分散。
- **为什么不在发送前铸**：发送时信息已摊平，无法区分「同一回合的两个调用」与「两个回合的同 id 调用」。

### D3 事件与 HITL 值域变化按有意变更处理

- 事件 payload（CAPABILITY_INVOKED/FINISHED、TOOL_AUDIT metadata 等）与 HITL 请求的 `tool_call_id` 变为内部 id。HITL 决定缓存以 `(session, tool_call_id, stage, invocation_key)` 为键——键值变化只影响新旧混跑窗口（恢复中既有 pending 记录是裸 id，新执行是内部 id），该窗口内本就无跨值匹配需求（pending 属于旧调用，重入走 reconcile 的 operation_id 面）。文档（README 事件说明）标注值域变化。

### D4 存量兼容：新值不可能等于旧值 + 歧义留痕

- 不迁移、不双读。旧记录之间仍按现状 last-write-wins 配对（不劣化）；新记录 `tc_` 形态与之不相撞。混合历史的合法性兜底仍由发送前 legalize 承担。
- **存量歧义显式留痕**：`reorder_tool_results_after_calls` 命中「同一裸 id 被多个 assistant 携带」时打 ERROR（含 id 与受影响消息定位）——现状此处静默复制（实测 r1, r2, r1, r2），留痕不改变行为（spec 明示不在此保证范围），但把脏数据暴露给运维。这是 legalize 层唯一的存量适配，配对算法本身仍零改动。

## Risks / Trade-offs

- [事件/面板消费者按裸 id 匹配会失配] → 值域变化列入 proposal Impact 与 README 迁移说明；raw id 保留在 metadata，宿主可回显。
- [id 变长推高 token 估算与 prompt 体积] → 形态约 25–30 字符（`tc_` + 两段 base36 + 12 hex），`estimate_tool_calls_tokens` 已按内容计数自然覆盖；用基准脚本核对增量可忽略。
- [哈希截断理论撞车] → 铸造时回合内唯一性自检（撞即抛，见 D1）；12 hex 撞率工程上为零。
- [遗漏某个以 `tool_call_id` 串联的旁路（如 delegate 回填的 `origin_tool_call_id`）导致新旧值混用] → 任务组含全仓 `tool_call_id` 消费点清单核对；配对一致性的贯通测试（重复 id 回归组）兜底。
- [双平面铸造时机竞态（park/打断的半截回合）] → 摄入点在回合提交（`_commit_round` 语义）处，半截回合不携 tool_calls 时无铸造需求；携 tool_calls 的打断路径同样经摄入函数。

## Migration Plan

纯增量部署：新回合新值、旧回合旧值，无切换开关、无回滚数据动作。回滚 = 还原代码（新写入的内部 id 记录在旧代码下仍配对正确——其唯一性使 last-write-wins 永不误配）。

## Open Questions

（无——铸造形态、铸造点、值域变更处置均已定。）
