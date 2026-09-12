# compact-fidelity

## Purpose

定义压缩摘要的语义保真契约：摘要按固定小节结构生成（目标/约束/已完成/待完成/证据）、解析宽松可降级、证据引用可回取、多轮压缩后保真可验收——并明示原文不可逆边界。

## ADDED Requirements

### Requirement: 结构化 digest 小节契约

task 域与 agent 域的压缩摘要 SHALL 按固定小节组织：`## Goal`（目标）、`## Constraints`（用户约束与硬性限制）、`## Done`（已完成含结果要点）、`## Remaining`（未完成事项与失败原因）、`## Evidence`（关键工具产出引用）。压缩 cue SHALL 明确要求各小节语义。摘要落库前 SHALL 经宽松解析；必需小节缺失时整文作为 digest 降级存档并标记 `degraded`，解析失败 MUST NOT 中断压缩链路。

#### Scenario: 多轮压缩后约束与待办在场

- **WHEN** 含明确用户约束与未完成事项的任务经历连续多轮压缩（L1/L3 叠加）
- **THEN** 最终 digest 的 Constraints 与 Remaining 小节仍携带约束原文要点与未完成清单（夹具断言）

#### Scenario: 缺节降级不炸链路

- **WHEN** 摘要输出缺少任一必需小节（模型未遵守格式）
- **THEN** 整文作为 digest 存档且带 `degraded` 标记，压缩流程正常完成

### Requirement: Evidence 引用可回取

`## Evidence` 小节引用工具产出时 SHALL 使用可回取标识（该次执行的 invocation_id），使模型能经回读工具（tool-result-recovery 的 `read_tool_output`）取回被折叠原文中的关键输出。无工具产出可引用时该小节 SHALL 允许为空。

#### Scenario: 引用可解析

- **WHEN** digest 的 Evidence 小节引用某次大输出的 invocation_id，模型随后调用回读工具
- **THEN** 该引用可解析为实际存储的工具输出内容

### Requirement: 保真验收分层可执行

压缩链路 SHALL 配备可重复执行的分层验收，且各层守卫与其证明力匹配：

- **管道层（CI，脚本化 LLM）**：断言输入材料传递（约束/待办/失败原因/工具产出确实进入压缩请求的输入历史）、digest 解析与多轮落库（预排 digest 中的标记跨轮存活）、降级路径正确。
- **请求层（CI，请求断言）**：捕获 compact 的 LLM 请求，断言 cue 含五节契约的完整指示——cue 被删改（如去掉 Constraints 指示）时该层失败。
- **语义质量层（非 CI）**：真实模型是否按 cue 生成含约束的摘要，由真实模型评测承担（发布前人工/脚本评测），MUST NOT 由固定 mock 冒充证明。

夹具 SHALL 作为回归资产入库，随压缩相关改动运行。

#### Scenario: 请求层守住 cue 契约

- **WHEN** 压缩 cue 的 Constraints 指示被删改
- **THEN** CI 请求断言层失败（固定 mock 的摘要输出不随 cue 变化，语义存活断言不承担此守卫）

#### Scenario: 管道层守住材料与落库

- **WHEN** 压缩链路改动导致输入历史不再含约束材料，或预排 digest 的标记在多轮 collapse/fold 后丢失
- **THEN** CI 管道层以对应断言失败

### Requirement: 原文不可逆边界

本能力 MUST NOT 承诺对话文本原文在 fold 后可恢复；可回取面限定为：工具产出证据（经结果存储）与图片（经既有 ref 机制）。该边界 SHALL 在用户文档中明示。

#### Scenario: fold 不可逆语义不回退

- **WHEN** 折叠执行后检索被 supersede 的原始对话记录
- **THEN** 原始记录不再参与重建（与既有 fold 语义一致），digest 是唯一承载
