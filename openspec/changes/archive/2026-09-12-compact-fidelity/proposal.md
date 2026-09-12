# Proposal: compact-fidelity

## Why

压缩机制（L0.5–L3 升级压缩、段折叠、act_recap）控制了体积，但摘要产物是纯自由文本：用户约束、未完成事项、失败原因是否被保留全凭模型发挥，无结构字段、无解析、无验收——多轮压缩后静默丢失约束或重做已完成工作是现实风险。fold 不可逆使「关键证据引用」在摘要中无处安放（引用了也取不回）。

## What Changes

- **结构化 digest 契约**：task 域（L3 坍缩）与 agent 域（L1 折叠）的压缩摘要 SHALL 按固定小节生成——`## Goal` / `## Constraints` / `## Done` / `## Remaining` / `## Evidence`；cue 同步改造（模板 + 逐节语义说明）。
- **宽松解析**：摘要落库前宽松解析小节；任一必需小节缺失时整文作为 digest 降级存档（标 `degraded`），MUST NOT 因格式不符中断压缩链路。
- **Evidence 引用可回取**：`## Evidence` 小节引用工具产出时 SHALL 使用可回取标识（invocation_id，经 tool-result-recovery 的 `read_tool_output` 解析）——**硬依赖 tool-result-recovery change**，其未落地时本项任务阻塞。
- **分层保真验收**：固定 mock 输出不随 cue 变化，不能冒充语义证明——验收分三层：管道层（CI，mock 证材料传递/解析/多轮落库/降级）、请求层（CI，断言 compact 请求的 cue 含五节契约，删改即失败）、语义质量层（非 CI，真实模型评测约束存活）；夹具作为回归资产入库。
- **不可逆边界契约化（Non-goal 显性化）**：对话文本原文在 fold 后 MUST NOT 承诺可恢复；可回取面 = 证据（结果存储）+ 图片（既有 ref 机制）。文档与 spec 明示，防验收标准定歪。

## Capabilities

### New Capabilities

- `compact-fidelity`: 压缩摘要的语义保真契约——结构化小节、宽松解析降级、证据引用可回取、多轮保真可验收、原文不可逆边界。

### Modified Capabilities

（无——压缩机制此前无 spec 覆盖，本 change 首立保真契约；体积机制语义不变。）

## Impact

- **代码**：composer 压缩 cue（task/agent 两域）、compact.py 摘要落库前的宽松解析与 degraded 标记、observe/background_observe 的 act_recap 不动（scope 控制，见 design D2）。
- **测试**：新增多轮压缩保真夹具（约束/待办/失败原因/Evidence/重做检测五维断言）；既有 compact 测试适配结构化输出。
- **依赖**：Evidence 回取硬依赖 `tool-result-recovery`（read_tool_output）；实施顺序应在彼之后，其余任务组不受阻塞。
- **顺序**：五个上下文工程 change 中最后实施。
