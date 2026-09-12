# Tasks: compact-fidelity

## 1. 结构化 digest

- [x] 1.1 cue 改造：task/agent 两域压缩指令重写为五小节模板（Goal/Constraints/Done/Remaining/Evidence，每节一行起步、有则多写、无则写 none、Evidence 允许缺）——单测：cue 文本含五节名与指示
- [x] 1.2 宽松解析器：正则抓 `## <name>` 节；Goal/Constraints/Done/Remaining 任一缺失 → 整文作 digest + metadata `degraded: true`；纯函数独立单测（全节/缺一节/全散文/旧格式兼容四形态）
- [x] 1.3 接线：`summarize_for_compact` 产出经解析后落库（L3 坍缩 digest 节 / L1 AGENT 层 SUMMARY），降级路径不中断压缩——单测：缺节响应下 collapse/fold 正常完成且记录带 degraded 标记

## 2. Evidence 引用联通（依赖 tool-result-recovery 落地；未落地则本组阻塞并注明）

- [x] 2.1 Evidence 引用格式 `- read_tool_output(<invocation_id>): 说明` 进 cue 示例；工具结果收敛版中的 invocation_id 在 compact 装配历史里可见性核验——单测：含大输出工具调用的会话压缩后，digest 的 Evidence 节出现可解析引用（或显式为空不 degraded）
- [x] 2.2 回取贯通：夹具会话中 digest 引用的 invocation_id 经 `read_tool_output` 实际取回内容——集成断言引用→回读闭环

## 3. 分层保真验收

- [x] 3.1 管道层夹具（脚本化 mock，CI）：预排多轮响应（含埋唯一标记的约束/待办/失败原因/工具产出 + 预排结构化 digest），驱动 summarize + collapse/fold 循环；断言输入材料进入压缩请求历史、预排 digest 标记经解析落库并跨轮存活、Done/Remaining 归属不变、降级路径正确、digest 长度 ≤ 旧版 1.5×——夹具可重复执行入库 `tests/`
- [x] 3.2 请求层断言（CI）：捕获 compact 的 LLMRequest，断言 cue 含五节契约完整指示（节名 + 逐节语义句）；**删改 cue（去掉 Constraints 指示）时此层失败**——注意：不使用「固定 mock 摘要存活」来证 cue（其输出不随 cue 变化）
- [x] 3.3 语义质量层（非 CI）：真实模型评测脚本/流程（可承载于 `benchmarks/`）——对含约束样本集压缩，人工/脚本核对约束存活；MUST NOT 入 CI；交付评测操作说明
- [x] 3.4 回归守门验证：人为破坏管道（如落库丢 digest 节）→ 管道层对应断言失败；删改 cue → 请求层失败——两层守卫各证其位

## 4. 收尾

- [x] 4.1 既有 compact 测试适配结构化输出；README/ARCHITECTURE 补保真契约说明与**原文不可逆边界**明示（可回取面 = 证据 + 图片）——`pytest` 全绿 + `ruff` / `mypy` 通过
