# Proposal: tool-schema-budget

## Why

工具 schema 是每个请求的固定占用，却与预算完全脱节：装配期 capability 块的 token 估算只算 description（schema 不计），预算裁剪对着全量有效窗口裁内容块，工具数组本身明文不受裁剪；循环内增量估算（有基线后）只算新增消息——运行期 pin 进一个大 schema 工具，估算值纹丝不动。结果是窗口实际超限风险（provider 400）与输出空间被无声挤占，且估算/实测偏差无观测。

## What Changes

- **装配期预留**：预算裁剪的有效上限 SHALL 先扣除当前工具面（名称 + 描述 + 入参 schema 的 token 估算，从 capability cache 快照现算）；预留值与工具面指纹写入 `AssembledPrompt.metadata`。工具集变化（动态加载/pin）后再次装配时预留随之增长。
- **循环期追踪**：act 循环内增量估算在工具面指纹变化时 SHALL 计入 schema 增量（新面估算 − 上次面估算，叠加在真实基线之上）；工具面不变时维持纯消息增量，行为与现状一致。
- **发送前策略（明示决策：不硬拒）**：估算超限 MUST NOT 拒发请求——超限信号转化为 max_tokens 自然收紧（动态输出计算已按含工具面的 used 值）+ 显式 WARNING 留痕 + 既有比例机制触发下轮压缩。理由：估算存在误差，硬拒会造成误伤；收敛路径（收紧 + 压缩）已齐备。
- **偏差观测**：估算与实际 usage 的回喂链路（tokenizer.observe 既有）之上，补工具面分项的偏差留痕。工具面指纹 SHALL 覆盖与估算一致的规范化工具定义（名称 + 描述 + schema）——仅名称集合不构成指纹，防同名工具定义更新后成本变化被跳过。
- 工具声明完整性语义不变：tools 数组仍不受裁剪，预算压力由预留扣减承担。

## Capabilities

### New Capabilities

- `context-budget`: 上下文预算口径契约——工具面成本在装配预留与循环增量两处入账、发送前不硬拒、超限可观测。

### Modified Capabilities

（无——预算/估算此前无 spec 覆盖，本 change 首立契约。）

## Impact

- **代码**：`core/utils/estimate.py`（新增 tools 面费率单一真源，与 prepare/gateway 两处既有公式收敛）、`assembler.py`（assemble 扣减 + metadata）、`llm_gateway.request_prompt_estimate`（增量路径扩 tools delta，loop_guard 新增上次工具面估算/指纹记录）、发送前 WARNING。
- **测试**：动态加载大 schema 后估算增长的回归；工具面不变时增量行为不变的对照；超限不硬拒断言。
- **协同**：与 context-evidence-delivery 独立；组合口径为 schema 预留先扣减、证据地板在其剩余内生效（两 design 互引）。
- **顺序**：独立 change，无在途依赖。
