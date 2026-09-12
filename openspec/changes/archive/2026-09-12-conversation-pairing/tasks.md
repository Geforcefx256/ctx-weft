# Tasks: conversation-pairing

## 1. 摄入点铸造

- [x] 1.1 盘点 `tool_call_id` 全部消费点（act / driver / capability_gateway / reconcile / delegate 回填 / 事件 / HITL / `_history` / budget / legalize），产出清单核对「单一供值源」覆盖完整性——清单落 `docs/` 或 PR 描述，无遗漏点
- [x] 1.2 在 assistant 回合摄入公共函数实现铸造：`tc_{seq36}_{ord36}_{hash12}`（字符集 `[A-Za-z0-9_]`、长度 ≤64，sha256(anchor|ordinal|raw) 截断，回合内唯一性自检撞即抛），metadata 写 `raw_tool_call_id`（有 operation_id 时一并伴随）；memory 摄入与 `current_messages` 追加同值供值——单测：同一回合两个平面 id 逐字节一致、raw id 可从 metadata 取回、全部铸出 id 匹配 provider 安全正则与长度上限
- [x] 1.3 gateway `invoke` 的 `tool_call_id` 参数、TOOL_RESULT/TOOL_AUDIT/派发框/错误补写落库、事件 payload、HITL 登记、delegate 的 `origin_tool_call_id` 全部改为透传摄入点铸造值（不改各自逻辑）——单测：这些出口出现的 id 均为内部形态

## 2. 配对面回归

- [x] 2.1 跨轮次重复 id 回归：两轮 assistant 均带 `call_1`，经 memory 重建（observe 边界/恢复路径）后断言各自 result 紧邻真正 owner、`legalize_messages` 输出中每条 result 恰出现一次
- [x] 2.2 跨 task 召回混合历史回归：agent_recall 归并多个 task（含同 wire id）后断言配对不串扰；budget `_coalesce_tool_pairs` 单测改为内部 id 输入下单元划分正确
- [x] 2.3 裁剪原子性回归：超限裁剪丢弃调用回合时其 result 同批消失，最终 prompt 无孤立 result（在既有 `test_budget_strategy` 配对用例上补重复 id 变体）
- [x] 2.4 存量混合回归：旧裸 id 记录 + 新内部 id 记录混合重建，断言新记录配对正确、旧记录行为与改造前一致（对照快照或等价断言）；`reorder_tool_results_after_calls` 命中存量重复 id 歧义时 ERROR 留痕（含 id 与消息定位）且行为不变——caplog 断言留痕可检索

## 3. 值域变更收尾

- [x] 3.1 既有测试中以裸 `call_*` 值断言的用例更新为新值域；README 事件/HITL 说明补 `tool_call_id` 值域迁移注记——`pytest` 全绿 + `ruff` / `mypy` 通过
- [x] 3.2 端到端贯通：mock LLM 多轮工具循环 + 中途压缩/恢复，全链路断言无「结果归错调用」「结果重复」「孤立 result」三类现象（可并入既有 verify 脚本或新增 `tests/` 集成用例）
- [x] 3.3 adapter 验收：openai / anthropic adapter 的 payload 构建单测断言铸出的内部 id 原样透传（无编码/截断/改写），并对照两家公开的工具 id 字符集与长度约束核对全部新 id 形态
