# Agent 工作约定

先读 [ARCHITECTURE.md](./ARCHITECTURE.md)，再定位源码和对应测试。SDK 使用方式参考 [README.md](./README.md)。架构文档记录当前实现与职责边界，历史 plans/spec/upgrade 需与代码及回归测试交叉核对。

- 修改前检查 `git status --short` 和目标文件现有 diff；保留用户及其他任务未提交的内容，包括新增模块。不要用旧分支或 HEAD 覆盖当前实现。
- 本仓负责通用引擎执行、任务/Agent 生命周期、记忆、事件提交与恢复。HTTP/SSE、产品聊天路由、用户/套件权限和展示状态属于 NetliveCoworkPy。
- 沿已有职责实现：TaskManager 应用任务处置，ALM 管 Agent 与模型选择，StepDriver 管步骤执行，CommitGate 管持久提交确认，reducers/SnapshotWriter 管恢复切面。不要新增第二份状态机或复制判据。
- 外部契约放在顶层导出和 protocols；Provider 不反向依赖 core 的调度、恢复域逻辑。允许的纯工具依赖以 `test_provider_layer_boundary.py` 为准，不随意扩大白名单。
- 不恢复已删除的兼容入口，不把冷查询变成装填/执行，不全局扫描 Agent/HITL 来猜目标。消息显式按 Agent 定址，冷访问明确 session scope。
- 事件或工具提交结果未知时，确认原 batch/操作，不换 ID 重发或重跑副作用。position 是持久游标，事件 ID 不是提交顺序。
- 变更状态、恢复、HITL、压缩、提示词或模型配置时，按架构文档的验证表选择已有回归，覆盖相关冷恢复与失败边界；不要删除边界断言来迁就实现。
- 只改文档无需完整业务测试。交付说明修改了什么、验证了什么及未解决的问题；不自动提交、发布或顺带重构。
- 跨仓执行行为变化先验证本仓，再经宿主 `dev/sync_ctx_weft.py` 更新 wheel/lock/安装副本并验证宿主；不要直接修改宿主 vendor 或 site-packages 中的源码。
