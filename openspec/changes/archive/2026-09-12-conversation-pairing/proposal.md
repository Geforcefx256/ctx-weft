# Proposal: conversation-pairing

## Why

对话重建中 assistant(tool_calls) 与 tool(result) 的配对只按 LLM 分配的 `tool_call_id` 裸字符串匹配：budget 裁剪的 owner 字典后写覆盖（`_coalesce_tool_pairs`）、发送前合法化的 reorder 按 id 聚合会在每个携带者后重复发出结果、dangling/orphan 判定按 id 集合。模型复用 `call_1` 这类短 id 是常态（`invocation_key` 注释自认），跨轮次/跨 task 召回重建后结果会错配到别的调用、被复制或随错误单元幸存——静默污染推理链：模型把旧结果当成自己这次调用的答复。恢复面已由 tool-operations 落实「tool_call_id MUST NOT 用作恢复匹配依据」，对话面缺同等机制。

## What Changes

- **摄入点铸造内部唯一调用标识**：assistant 回合摄入（memory 持久化 + 活动消息列表追加，两平面同一次铸造）把每个 tool_call 的 id 替换为内部唯一 id——**限定字符集与长度的确定性值**（满足主流 provider 对工具 id 的约束，由回合锚 + 调用序号 + 原始 id 确定性派生，形态见 design D1）；原始 id 降级为 metadata（`raw_tool_call_id`）。补 adapter 验收：铸出的 id 经 openai/anthropic adapter 原样透传且符合其 id 约束。
- **单平面透传**：gateway/HITL 登记/delegate 回填/事件 payload 携带的 `tool_call_id` 一律使用内部 id；HITL 决定缓存的四维键（含 `invocation_key`）结构不变，双保险保留。
- **配对面零算法改动**：budget 配对、legalize 四函数、`_history` 重建不改匹配逻辑——输入的 id 天然唯一后，错配/复制/孤弃自然消失。
- **存量数据不迁移**：旧记录仍是裸 id；完整性保证（不错配/不复制/不孤弃）**限定覆盖新记录平面**，存量重复 wire id 在合法化层维持现状行为（可能复制）但 MUST 留痕可观测，不再宣称对旧行为的改进。
- **可观察值域变化**：事件（CAPABILITY_INVOKED/FINISHED 等）与 HITL 请求中的 `tool_call_id` 值从裸 id 变为内部 id（更长、全局唯一）——宿主面板若按 id 展示/匹配需知悉，属有意变更。
- 补齐重复 id 回归测试：跨轮次、跨 task 召回混合历史、裁剪原子性（配对同生共死、无孤立 result 幸存）。

## Capabilities

### New Capabilities

- `conversation-integrity`: 对话重建配对完整性契约——摄入点铸造内部唯一调用标识、live 与 memory 双平面同口径、配对/裁剪/合法化不产生错配与孤弃、存量兼容。

### Modified Capabilities

（无——tool-operations 已要求恢复面不用 tool_call_id 匹配，本 change 在对话面落实同一原则，不改动其需求；capability-gateway 的既有需求不变。）

## Impact

- **代码**：act 回合摄入路径（assistant LLM_RESPONSE 落库与 current_messages 追加的公共点）、driver 的 assistant turn ingest、capability_gateway 的 tool_call_id 透传（仅取值来源变化）、assembler/sources/_history（metadata 透传 raw id）。
- **行为面**：事件与 HITL payload 中 `tool_call_id` 值域变化（见上）；LLM 在对话历史中看到的 call id 变为内部 id（provider 对 id 值是黑盒，仅要求 call↔result 一致，无兼容风险）。
- **测试**：新增重复 id 回归组；既有以裸 id 断言的测试需按新值域更新。
- **顺序**：独立 change，无在途依赖；建议五个上下文工程 change 中最先实施（其余改进建立在「结果对应正确的调用」之上）。
