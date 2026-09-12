# context-evidence

## Purpose

定义证据供给契约：知识检索与语义召回的内容块必有渲染归宿（或显式留痕的丢弃）、渲染位置不打穿 prompt cache 前缀、预算裁剪在来源等级之上含相关性轴且证据保护有界——使命中的证据真正到达模型。

## ADDED Requirements

### Requirement: 证据渲染进入最终请求

purpose=act 的装配中，通过预算裁剪存活的 `reference`（知识检索）与 `summary`（语义召回）块 SHALL 被渲染进最终 LLM 请求的 messages，含内容与来源标识（provider/来源名与相关性 score）。渲染位置 SHALL 位于尾部动态区（末条 user 回合、guidance 之前），MUST NOT 注入当前 task 锚回合或其它 cache 前缀稳定区。被预算裁掉的块 MUST NOT 出现（裁剪优先于渲染，无幽灵内容）。无渲染路径的块 kind 在渲染层 SHALL 产生显式警告（不留静默盲区）。

#### Scenario: 命中证据到达模型

- **WHEN** 注册的 KnowledgeProvider / MemoryProvider 语义召回各自返回内容，预算允许
- **THEN** 最终 LLM 请求的 messages 同时含两类证据的内容与来源标识

#### Scenario: 检索变化不打穿缓存前缀

- **WHEN** interactive 连续两轮装配，新一轮检索/召回内容变化
- **THEN** 当前 task 锚回合（## Current Task/Capabilities 落点）在两轮间字节稳定，证据仅随尾部动态区变化

#### Scenario: 未渲染 kind 有警告

- **WHEN** 装配结果含渲染层不认识的块 kind（如 blackboard）
- **THEN** 产生显式 warning（含 kind 与 source），而非静默跳过

### Requirement: 裁剪含相关性轴且证据保护有界

预算裁剪 SHALL 保留来源等级静态阶梯，并在其上叠加证据动态提级：`reference`/`summary` 中**位列来源 top-K（默认 3，可配，K=0 关闭）且相关性 score 达标**（两条件同时满足）的条目，保护档提级至与当前任务内容同档；per-source top-K 为硬数量上界——同来源无论多少条目达标，提级条数 MUST NOT 超过 K。提级后的证据在丢弃顺序上 MUST 晚于已完成任务的胶囊与摘要。同档内 SHALL 低 score 先丢；无 score 的块（当前任务历史等）在同档内 SHALL 视为最受保护（最后丢）。未提级证据维持原最低档——防止证据无限挤占历史预算。

#### Scenario: 紧张预算下证据晚于陈旧胶囊被裁

- **WHEN** 同一模型与 token 预算下，高分证据与多个已完成任务的胶囊/摘要竞争
- **THEN** 丢弃顺序为先未提级证据、再陈旧胶囊与摘要、后提级证据——对照改造前（证据最先丢），提级证据存活率提升

#### Scenario: 证据保护有界

- **WHEN** 单一来源返回大量高分条目（如十条 score 均 0.9）且窗口紧张
- **THEN** 该来源仅排名前 K 条提级，其余按原最低档参与裁剪；多来源各自独立计 K；历史地板（identity/当前 user_prompt）不受侵蚀

#### Scenario: 无 score 历史块最后丢

- **WHEN** 提级证据与当前任务历史块处于同一保护档且需继续丢弃
- **THEN** 有 score 的证据先丢（低分先），无 score 的当前任务历史在同档内最后丢

### Requirement: 裁剪丢弃留痕

预算裁剪丢弃任何块时 SHALL 留下可机器辨识的记录（至少结构化日志：kind、source、token 估算、丢弃时所属保护档）；命中保护地板仍超限的既有溢出报错语义不变。

#### Scenario: 丢弃可审计

- **WHEN** 一次装配因超限丢弃若干块
- **THEN** 事后可从留痕还原「丢了什么、多大、当时什么档」，其中证据类丢弃可被单独检索
