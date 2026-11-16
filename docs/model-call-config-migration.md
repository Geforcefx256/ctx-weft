# 宿主调用配置迁移

调用配置只保存在 agent 的 `ModelChoice.config`。每次派发复制到
`ResolvedModel.config`，每次请求复制到 `LLMRequest.config`。不增加会话状态或事件，
不提供共享初始化默认值。本文列出 ctx-weft 的接口变化；Netlive Cowork 未在本次修改中更新。

## 请求与 adapter

```python
from ctx_weft.protocols import LLMRequest, ModelCallConfig

request = LLMRequest(
    model="my-model", system="...", messages=[],
    config=ModelCallConfig(reasoning_effort="high", temperature=0.5, max_tokens=4096),
)
```

- 删除 `LLMRequest.max_tokens` / `LLMRequest.temperature` 顶层字段和对应构造参数。
  改为 `request.config.max_tokens` / `request.config.temperature`。
- 其他标准配置：`reasoning_effort`、`top_p`、`stop`。默认全部为 `None`。
  显式 `temperature=1.0` 现在会发送，空配置保持原有不指定温度的出网行为。
- `LLMClient.complete(request, stream=True)`、`ModelResolver(account, model)` 和
  `LLMClientResolver.get_client(account, model)` 签名不变。
- 删除 `OpenAIAdapter(..., reasoning_effort_provider=...)`。配置直接从请求读取。
- adapter 可以修改请求副本，不能放宽网关计算的安全输出上限。重试从预算计算后的
  原始请求复制，不能依赖 adapter 实例上的当前会话配置。
- 内置 Anthropic 支持 `none`/`off` 关闭及 `low`/`medium`/`high` 预算映射。
  `ctx_weft.providers.llm.AnthropicCallConfig(thinking_budget_tokens=...)` 可覆盖预算。
  不支持的档位、不足的输出预算、缺少签名的 thinking 工具续接明确失败，不静默降档。

## 创建和设置 agent

以下入口新增 `llm_config: ModelCallConfig | None = None`：

- `SessionStartParams.create(...)` / `SessionStartParams`：仅用于创建 root agent。
- `runtime.create_session(...)`：用于容器会话的 root agent。
- `runtime.run_single_task(...)`：测试/便利入口的 root agent。
- `runtime.dispatch_task(...)`：仅在创建新 agent 时使用。
- `runtime.set_agent_llm(...)` / `runtime.set_session_llm(...)`：替换完整选择。

```python
await runtime.set_agent_llm(
    agent_id, llm_account=current_account, llm_model=current_model,
    llm_config=ModelCallConfig(reasoning_effort="high"),
)
await runtime.recover_session(session_id)
```

省略 `llm_config` 或传 `None` 的设置命令会清除配置覆盖。宿主若提供 PATCH API，应先
合并当前配置再提交。不要把 HTTP 的“字段未提供”直接转换成清空设置。
`SessionStartParams(resume=True)` 不接受 `llm_config`；恢复前用设置命令修改 agent。

`set_session_llm` 只修改当前已有 agent，不保存新 agent 的会话默认值。子 agent 默认复制
父选择；独立新树未显式提供选择时使用空配置。正在运行的 run 保持原配置，后续派发才用新值。

## 配置子类与持久化

`ModelCallConfig` 可被 adapter 的 dataclass 继承。创建或恢复 agent 前注册稳定类型 ID：

```python
from dataclasses import dataclass
from ctx_weft.protocols import ModelCallConfig, register_model_call_config

@dataclass
class GatewayConfig(ModelCallConfig):
    thinking_budget: int | None = None

register_model_call_config("company.gateway", GatewayConfig, version=1)
```

默认编码使用 `asdict` 转换成 JSON 值。注册时可传 `decode=` 处理自定义嵌套类型。
同一类型 ID/版本不能注册不同类；不同版本使用不同配置类和对应 decoder。

`AgentInstantiated` / `AgentLlmChanged` 新增 `llm_config`：

```json
{
  "type": "company.gateway",
  "version": 1,
  "values": {"reasoning_effort": "high", "thinking_budget": 4096}
}
```

`AgentView.llm_config` 及 agent 快照保留该载荷，包括未知类型。
`serialize_call_config` / `deserialize_call_config` 是公共编解码函数。
旧事件和旧快照缺字段时恢复空配置；未知类型/版本阻止执行恢复，不能降级为基础类型。
`SessionCreated` 不增加配置，`set_session_llm` 仍发共享 causation_id 的 N 条 agent 事件。
`LLMPromptSent.llm_config` 是网关计算预算后交给 adapter 的配置，用于诊断，不回写选择。

## Netlive Cowork 适配位置

1. 保留现有 HTTP `reasoning_effort` 字段即可；后端将它转换成 `ModelCallConfig`。
   创建、消息续跑、恢复、HITL（包括仅改强度）和定时任务统一走上述创建/设置入口。
2. 宿主 provider 停止传 `reasoning_effort_provider`，宿主 adapter mixin 改读
   `request.config`。保留宿主 SSL、网关专属开关及响应兼容逻辑。
3. 移除通过 `reasoning_effort_context` 在启动或恢复任务时注入参数的路径。
4. 保留宿主现有 `config_json.reasoning_effort` 的读写、字段省略和显式清空语义，
   不迁移存量数据或修改宿主表结构。冷恢复时先装填目标会话，再把已有宿主偏好合并进
   agent 选择；保留各 agent 的模型和其他配置。SDK 事件及快照负责保存完整 `llm_config`。
5. 更新 vendored ctx-weft wheel、锁文件及 adapter/恢复/定时任务回归测试。无需为此新增
   会话配置列或调整现有前端界面。
