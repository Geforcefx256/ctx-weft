# ctx-weft

**Protocol-based Agent Runtime Core** — 用于构建可控、可观测 AI Agent 应用的 Python SDK。

ctx-weft 只做一件事：把外部系统（知识库、记忆系统、能力系统、LLM、模板）通过协议接入，驱动 LLM 完成任务。核心运行时通过协议解耦具体实现，可选默认实现在 `providers` 中提供，由上层应用（host）配置与装配。

> 本文是**使用参考**。想了解循环引擎、任务编排、崩溃恢复等**内部实现逻辑**，见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

## Runtime 初始化与按需装填

构造同步完成公共接线，不枚举或读取历史。已有会话在执行前按 ID 初始化：

```python
runtime = CtxWeftRuntime(providers=providers, event_store=event_store)
result = await runtime.initialize(session_id)
```

`initialize(session_id)` 自主读取该会话的快照与轻量增量，补写崩溃中断事实；不装填执行对象、
不启动模型或任务。Host 可以只在启动时初始化需要立即校准的会话，其余延后到首次使用。
执行入口和 `rebuild_session` 也会确保初始化；新建成功的会话直接视为已初始化。
纯查询与历史浏览不触发初始化或恢复写入。

同一会话的并发调用共享 Task，取消单个等待者不取消恢复；不同会话可并行。
成功后只保留 session_id，Task 和报告不常驻。首次调用及其并发等待者返回各自独立的
`SessionInitializationResult`，后续调用返回 `None`；失败不记成功，允许重试。
缓存逐出保留初始化标记，`purge_session` 清除标记。该契约适用于单进程独占的存储范围。

结果包含会话校准前后的位置、轻量状态摘要与本次提交的 `startup_events`；没有新增修复时
首次调用仍返回摘要。报告是初始化时的固定切面，当前状态应通过查询接口获取。
Host 应在执行前同步恢复报告到自己的投影；投影失败后的重试由 Host 保留待完成的写入。

`get_session`、`get_task`、`list_tasks`、`get_agent`、`list_agents`、`list_pending_hitl`
均为异步查询。单条返回 `QueryResult`，列表返回 `list[QueryResult]`：

```python
result = await runtime.get_session(session_id)
print(result.data.state, result.source, result.loaded)
task = (await runtime.get_task(session_id, task_id)).data
agents = await runtime.list_agents(session_id=session_id)
```

已装填对象从内存读取；未装填或未保留的历史对象从事件流推断，查询不触发装填。
`source` 为 `memory` / `events`，事件结果带 `committed_position`。
不指定 Session 的列表只枚举已装填对象；冷 Agent 查询需要 `session_id` 定址。
恢复快照可能裁掉终态任务，历史任务查询会补读原始任务事件。
`get_session().data.state` 是完整会话查询快照：`root_agent` 和按 Agent ID 排序的 `agents` 复用 `AgentDetail`，`root_task` 和 `AgentState.current_task` 复用 `TaskDetail`，`pending_hitl` 复用完整 `HitlRequestView`；同时包含 `goal`、失败计数和熔断标记。任务包含正文、输出、错误码、错误正文、失败时重试次数、阻塞任务和收尾标记，无需再补查根任务。保留全部 Agent 的中断原因，宿主选择展示方式；原因未知为空字符串，旧日志没有任务记录时任务对象为 `None`。

`get_agent().data.state` 使用同一 `AgentState`；模型选择仍在详情的 `llm_account`、`llm_model`、`llm_config` 中。`initialize()` 返回同一完整 state，冷恢复与实时查询共用公开对象构造逻辑。`TaskDetail` 定义在 `ctx_weft.protocols.task`，也可从顶层包导入。旧的扁平根任务字段和 HITL 计数没有兼容别名。会话级聊天路由和状态到界面的映射由宿主实现。RoundCommitted/RoundDiscarded 仍是瞬态通知，不参与恢复重放。

`rebuild_session()` 是已有会话唯一装填入口，只装填、不执行。同一会话的 owner
TaskManager 只创建一份。`start_session(resume=True)`、`send_message`、`dispatch_task`、
`recover_session` 和需续跑的 HITL 应答会在会话锁内自动确保装填，不要求宿主提前装填。
冷 `reply_to_hitl(reply, session_id=...)` 和 `recover_session(session_id)`
需要 Session 定址信息。`recover_agent`、`rebuild_hitl`、`prepare_agent` 兼容入口已删除；装填统一使用 `rebuild_session`。

`TurnHandle.wait_for_finish()` 只在任务终态且相关收尾完成后正常返回，超时抛
`TimeoutError`；人工等待不是完成。稳定的任务状态和产出请通过 `get_task()` 读取。

---

## 目录

- [安装](#安装)
- [核心概念](#核心概念)
- [快速上手](#快速上手)
- [三大协议](#三大协议)
  - [MemoryProvider](#memoryprovider)
  - [CapabilityProvider](#capabilityprovider)
  - [KnowledgeProvider](#knowledgeprovider)
- [AgentTemplate 与 AgentCapabilityProvider](#agenttemplate-与-agentcapabilityprovider)
- [LLM 接入](#llm-接入)
- [CtxWeftRuntime API](#ctxweftruntime-api)
- [事件系统](#事件系统)
- [内置 Provider](#内置-provider)
- [控制面](#控制面)
- [自定义 Provider](#自定义-provider)
- [在测试中使用](#在测试中使用)
- [限制与约束](#限制与约束)

---

## 安装

```bash
# 核心（零额外依赖）
pip install ctx-weft

# 含内置工具（bash / http / file）
pip install "ctx-weft[builtin]"

# 含 MCP bridge
pip install "ctx-weft[mcp]"

# 含远程 skill 同步
pip install "ctx-weft[skills]"

# 开发环境（pytest + ruff + mypy）
pip install "ctx-weft[dev]"
```

Python ≥ 3.11 required（用到 `StrEnum` / `datetime.UTC` / `asyncio.timeout`）。

顶层包导出：

```python
from ctx_weft import (
    CtxWeftRuntime, ProviderRegistry, RunHandle, SessionStartParams, InMemoryEventStore,
    TaskSettings, NormalTaskSettings, CompactTaskSettings, MetadataFillerTaskSettings,
)
# 测试辅助单独放在 ctx_weft.testing（不污染生产 API）：
from ctx_weft.testing import MockLLMAdapter, MockResponse, ToolCall
```

---

## 核心概念

```
AgentTemplate        定义 Agent 的身份（SOUL/ROLE）和能力引用
       ↓
CtxWeftRuntime        顶层 API，连接所有组件
  ├── ProviderRegistry        注册 Memory / Capability / Knowledge / LLM
  ├── AgentCapabilityProvider 读取 AgentTemplate（由上层实现或用内置 LocalAgentTemplateProvider）
  └── LLMClient               LLM 调用接口（由上层实现）
       ↓
Loop Engine          reason → act → observe → finalize
       ↓
EventBus             事件总线，所有状态变更的唯一出口
```

> **证据投递（尚未实现）**：知识检索（KnowledgeProvider）与语义召回（recall_semantic）命中的内容目前**被装配、进预算、然后在渲染层丢弃**——`reference` / `summary` 两类块没有渲染路径。这是已知缺口，投递与相关性裁剪留待后续单独实现。在那之前装配层对它们发 warning（`no rendering path for block kind(s)`），不留静默盲区。
>
> **工具面预算口径（2026-09 起，spec: context-budget）**：装配预算先扣工具面预留（名称 + 描述 + 入参 schema，按能力快照现算，`AssembledPrompt.metadata.tools_reserved_tokens` 可读）——工具声明不参与裁剪，预算压力由预留承担；溢出报错仍报真窗口。act 循环内增量估算追踪工具面指纹（覆盖 name+description+schema 的规范化定义哈希——同名工具 schema 更新可感知）：运行期 pin 大 schema 后估算立即增长（`max_tokens` 随之收紧）；工具面不变时增量行为与旧口径一致。发送前超限**不硬拒**（明示决策）：收紧 max_tokens + WARNING 留痕 + 既有比例机制触发下轮压缩。估算与实际 usage 的偏差留工具面分项记录（指纹/估算/实测同现于回喂日志）。
>
> **压缩摘要保真（2026-09 起，spec: compact-fidelity）**：task 域（L3 坍缩）与 agent 域（L2 历史归并）的压缩 digest 按五个固定小节生成——`## Goal / ## Constraints / ## Done / ## Remaining / ## Evidence`（Evidence 可缺）；落库前宽松解析，缺必需小节整文降级存档（metadata `digest_degraded: true`）、不中断压缩。`## Evidence` 引用可回取的工具产出（`results__read_tool_output(invocation_id=...)`）。验收分三层：管道层与请求层（cue 契约）在 CI（`tests/unit/test_compact_fidelity.py`）；**语义质量层不入 CI**——真模型是否按 cue 保留约束，发布前人工运行 `python benchmarks/compact_fidelity_eval.py`（需真实 LLM key）。**不可逆边界**：对话文本原文在 fold 后不可恢复（既有 supersede 语义不变，本契约不承诺原文回取）；可回取面仅限工具产出（结果存储）与图片（既有 ref 机制）。段摘要（act_recap / Progress So Far）不在本契约范围（逐段生成、消费方为观察锚点，见 ARCHITECTURE）。

这些**协议**是对外的接入面。外部系统只需实现其中一个协议，ctx-weft 自动用上（LLM 详见
[LLM 接入](#llm-接入) 一节）：

| 协议 | 用途 | Context 落位 | 数量 |
|------|------|-------------|------|
| `MemoryProvider` | 记忆（短期对话 / 长期 / blackboard） | messages | **唯一，必需** |
| `CapabilityProvider` | Agent 可调用的动作（tool / skill / sub-agent） | 首条 user message 前缀 + LLM tools | 多个 |
| `KnowledgeProvider` | 动态检索的参考资料（RAG / Wiki） | messages | 多个 |
| `LLMClient` / `LLMClientResolver` | LLM 调用 adapter / 多账号 provider | — | **唯一，必需** |

全部协议（含数据类型）均从 `ctx_weft.protocols` 导出。

> 引擎如何把这些协议拼成 prompt、如何驱动 Step、如何编排子任务，见 [ARCHITECTURE.md](./ARCHITECTURE.md)。

---

## 快速上手

最小可运行示例（使用内置 `MockLLMAdapter`，不需要真实 LLM）：

```python
import asyncio
from ctx_weft import CtxWeftRuntime, ProviderRegistry
from ctx_weft.testing import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import (
    AgentCapability, AgentCapabilityProvider, AgentTemplate,
    CapabilityProviderInfo, IdentityFacet, LoopConfig, MemoryConfig,
)


# 1. 实现一个最简 AgentCapabilityProvider——模板进入 core 的唯一通道
#    （发现与加载同源，spec 2026-07-22）。真实项目通常直接用内置
#    ctx_weft.providers.agent_template_local.LocalAgentTemplateProvider 扫描模板目录。
class DictAgentTemplateProvider(AgentCapabilityProvider):
    name = "agent"

    def __init__(self, templates: dict[str, AgentTemplate]):
        self._t = templates

    async def list(self, ctx) -> list[AgentCapability]:
        return [
            AgentCapability(id=f"{self.name}:{t.id}", name=t.id, template_name=t.id)
            for t in self._t.values()
        ]

    async def get_template(self, template_id, version, ctx) -> AgentTemplate | None:
        return self._t.get(template_id)

    async def describe(self, ctx) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=len(self._t))


# 2. 定义 AgentTemplate（Agent 的身份蓝图）
template = AgentTemplate(
    id="my_agent", name="My Agent", version="1.0.0",
    identity={
        "act": IdentityFacet(text="You are a helpful assistant."),
        "observe": IdentityFacet(text="Evaluate if the task is complete."),
    },
    capability_refs=[],                 # 暂不绑定外部工具
    memory_config=MemoryConfig(),
    loop_config=LoopConfig(),
)

# 3. 组装 Runtime
providers = ProviderRegistry()
providers.register_memory(InMemoryMemoryProvider())
providers.register_capability(DictAgentTemplateProvider({"my_agent": template}))

runtime = CtxWeftRuntime(
    llm=MockLLMAdapter(responses=[MockResponse(text="The answer is 42.")]),
    providers=providers,
)

# 4. 运行一个任务：template_id 须为规范形式 provider:name——本 provider 注册的
#    前缀是 "agent"（DictAgentTemplateProvider.name），故 "my_agent" → "agent:my_agent"。
async def main():
    handle, state = await runtime.run_single_task(
        template_id="agent:my_agent",
        user_prompt="What is the meaning of life?",
    )
    print(state.verdict.summary)        # "The answer is 42."

asyncio.run(main())
```

---

## 三大协议

### MemoryProvider

**唯一的 memory 抽象**——承载短期对话窗口、长期记忆、blackboard 父子通信。

```python
from ctx_weft.protocols import (
    MemoryProvider, MemoryEvent, MemoryEventType,
    MemoryScope, MemoryRecord, ProviderContext,
)
```

核心接口（全部 `async`）：

```python
class MemoryProvider(Protocol):
    name: str

    # 写入（core 在每个关键节点自动调用）
    async def ingest(self, event: MemoryEvent, ctx: ProviderContext) -> str: ...

    # 读取（三种模式）
    async def recall_recent(self, scope, types, limit, ctx) -> list[MemoryRecord]: ...
    async def recall_topic(self, topic, since, ctx) -> tuple[list[MemoryRecord], int]: ...
    async def recall_semantic(self, query, scope, top_k, ctx) -> list[MemoryRecord]: ...

    # topic 订阅（跨 session 长期上下文）
    async def subscribe_topic(self, session_id, topic, intent, ctx) -> str: ...
    async def list_subscriptions(self, session_id, ctx) -> list[Subscription]: ...

    # 压缩（超出 context 时触发）
    async def apply_compact(self, scope, summary, keep_last, ctx) -> CompactResult: ...

    # 工具
    async def count_recent(self, scope, types, ctx) -> int: ...
    async def describe(self, ctx) -> MemoryProviderInfo: ...
```

**MemoryEventType** — core 自动 ingest 的事件类型：`USER_PROMPT` / `LLM_RESPONSE` /
`TOOL_INVOCATION` / `TOOL_RESULT` / `OBSERVER_SUMMARY` / `COMPACT_SUMMARY` / `BLACKBOARD_PUBLISH`
（每种类型由 core 在何时写入，见 [ARCHITECTURE.md §记忆写入时机](./ARCHITECTURE.md#记忆写入时机)）。

**注册方式**：

```python
providers.register_memory(InMemoryMemoryProvider())   # 唯一槽位，重复注册即覆盖
```

### CapabilityProvider

**Agent 的可调用动作**——tool、skill、sub-agent 三类。

```python
from ctx_weft.protocols.capability import (
    Capability, ToolCapability, SkillCapability, AgentCapability,
    CapabilityEvent, ToolCapabilityProvider,
)
```

`CapabilityProvider` 基类只声明三个方法；`ToolCapabilityProvider` 额外要求 `invoke()` / `cancel()`：

```python
class CapabilityProvider(ABC):
    name: str
    async def list(self, ctx) -> list[Capability]: ...
    async def retrieve(self, ctx) -> list[Capability]: ...   # 默认回落到 list()
    async def describe(self, ctx) -> CapabilityProviderInfo: ...

class ToolCapabilityProvider(CapabilityProvider):
    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]: ...
    async def cancel(self, invocation_id, ctx) -> None: ...
```

**Capability** 数据结构：

```python
@dataclass
class ToolCapability(Capability):
    id: str            # 全局唯一，如 "builtin:bash_exec" / "mcp:fs:read_file"
    name: str          # LLM tool_call 时用的短名
    kind: str = "tool"
    purposes: list[Purpose]   # ["act"] / ["observe"] / ["act", "compact"]
    description: str = ""
    input_schema: dict = {}   # JSON Schema
    side_effects: bool = False
```

**CapabilityEvent** — invoke 流式返回：`kind ∈ {"progress","stdout","stderr","result","error"}`；
最终结果放 `payload.content`，错误放 `payload.code` + `payload.message`。

**Purpose 过滤** — core 在每个阶段只向 LLM 暴露对应 purpose 的 capability：

| Purpose | 对应 Step | 典型工具 |
|---------|-----------|---------|
| `"act"` | ActStep | bash_exec / http_request / submit_task |
| `"observe"` | ObserveStep | submit_task_assessment / request_human_input |
| `"compact"` | CompactStep | read_file / glob |
| `"metadata_filler"` | MetadataFillerStep | update_task_metadata |

**注册方式**（可注册多个）：

```python
providers.register_capability(BuiltinToolsCapabilityProvider())
providers.register_capability(MCPCapabilityProvider(config))
providers.deregister_capability("builtin")   # 按 provider.name 注销
```

### KnowledgeProvider

**动态检索的参考资料**——RAG、Wiki、API 文档。只读，不接受写入。

```python
from ctx_weft.protocols import KnowledgeProvider, KnowledgeQuery, KnowledgeDoc

class KnowledgeProvider(Protocol):
    name: str
    async def retrieve(self, query: KnowledgeQuery, ctx) -> AsyncIterator[KnowledgeDoc]: ...
    async def describe(self, ctx) -> KnowledgeProviderInfo: ...
```

**KnowledgeQuery**：`text` / `intent`（"policy" / "reference" / "api_spec" / ...）/ `top_k=5` / `filters`。

**注册方式**（可注册多个，priority 升序，数字小=先查询）：

```python
providers.register_knowledge(MyRAGProvider(), priority=0)
providers.register_knowledge(WikiProvider(), priority=10)
```

---

## AgentTemplate 与 AgentCapabilityProvider

### AgentTemplate

Agent 实例化的蓝图，由上层（host 应用或用户代码）构建：

```python
from ctx_weft.protocols import (
    AgentTemplate, IdentityFacet, CapabilityRef, MemoryConfig, LoopConfig,
)

template = AgentTemplate(
    id="research_agent", name="Research Agent", version="2.1.0",  # semver

    # Identity：按 purpose 分面（act / observe / compact / metadata_filler）
    identity={
        "act":     IdentityFacet(text="You are a research agent. Be thorough and cite sources."),
        "observe": IdentityFacet(text="Evaluate whether the research task is complete."),
    },

    # 外部 capability 引用，mode: optional / required / forbidden
    capability_refs=[
        CapabilityRef(capability_id="builtin:http_request", mode="optional"),
        CapabilityRef(capability_id="control:submit_task",  mode="required"),
    ],

    memory_config=MemoryConfig(
        short_window_size=20,       # 最近多少条历史进 prompt
        summary_threshold=20,       # 超过多少条触发 compact
        use_long_term=True,
        subscribed_blackboard_topics=[],
    ),

    loop_config=LoopConfig(
        max_turns_per_act=10,       # 单个 ActStep 最多多少轮 LLM 调用
        max_turns_per_observe=5,
        failure_threshold=3,
        max_spawn_depth=4,          # 子 agent 最大嵌套深度
        compact_token_ratio=0.8,    # token 占 context_limit 比例超过则压缩
        compact_message_delta=20,   # 距上次 compact 累积多少条再触发
        compact_keep_last=6,        # L2 优先保留的最近 lean 历史数，预算不足时继续归并
    ),
)
```

### AgentCapabilityProvider

模板进入 core 的**唯一通道**是 `AgentCapabilityProvider`（spec 2026-07-22 方案 B）：
发现（`list`）与加载（`get_template`）同源，`register_capability` 直接注册即可，无需
额外适配器。

推荐直接使用内置的目录扫描实现——根目录下每个含 `SOUL.md` 的子目录即一个模板：

```python
from pathlib import Path
from ctx_weft.providers.agent_template_local import LocalAgentTemplateProvider

providers.register_capability(LocalAgentTemplateProvider(Path("resources/templates")))
runtime = CtxWeftRuntime(providers=providers)
```

也可以自己实现协议（比如从数据库/远端注册表读取）：

```python
from ctx_weft.protocols import AgentCapability, AgentCapabilityProvider, AgentTemplate, CapabilityProviderInfo

class MyAgentTemplateProvider(AgentCapabilityProvider):
    name = "agent"

    def __init__(self, store: dict[str, AgentTemplate]):
        self._store = store

    async def list(self, ctx) -> list[AgentCapability]:
        return [
            AgentCapability(id=f"{self.name}:{t.id}", name=t.id, template_name=t.id,
                            description=t.metadata.get("description", ""), version=t.version)
            for t in self._store.values()
        ]

    async def get_template(self, template_id, version, ctx) -> AgentTemplate | None:
        return self._store.get(template_id)

    async def describe(self, ctx) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=len(self._store))
```

> `AgentCapabilityProvider` 缺省 `retrieve()` 返回 `[]`：全目录只经 `list()` 暴露给
> required-ref 精确查找，永不自动召回——sub-agent 只经模板声明的 `subagents` 绑定。

---

## LLM 接入

ctx-weft 的运行时依赖 `LLMClient` 协议，并在 `providers.llm` 提供可选的默认 adapter。
安装 `ctx-weft[llm]` 可使用 OpenAI 官方 Python SDK 实现。协议与全部数据类型都在
`ctx_weft.protocols`（与 Memory / Capability / Knowledge 同层）：

```python
from ctx_weft.protocols import (
    # 写一个 LLM adapter 需要的全部类型：
    LLMClient,                         # 要实现的协议
    LLMRequest, LLMMessage, LLMTool,   # 读：complete() 入参
    LLMChunk, ToolCall, LLMUsage,      # 产：complete() yield 的流式 chunk
    LLMCallError,                      # 错误/重试契约
    # 仅多账号 provider 需要：
    LLMClientResolver,
)
```

有两种接法。

### 方式 A：直接传一个 LLMClient（`llm=`）

适合脚本/测试、单一模型：

```python
providers = ProviderRegistry()
providers.register_capability(my_agent_capability_provider)
runtime = CtxWeftRuntime(providers=providers, llm=my_adapter)
```

`LLMClient` 协议：

```python
class LLMClient(Protocol):
    @property
    def context_limit(self) -> int: ...
    @property
    def output_reserve(self) -> int: ...   # 输入侧输出预留（→ reserved_output_tokens）
    @property
    def supports_tool_calling(self) -> bool: ...
    def complete(self, request: LLMRequest, stream: bool = True) -> AsyncIterator[LLMChunk]: ...
    async def count_tokens(self, text: str) -> int: ...
```

`LLMChunk.kind ∈ {"token","reasoning","tool_call","tool_call_partial","usage","done"}`。
自定义 Adapter 见 [自定义 Provider](#自定义-provider)。

### 方式 B：多账号 LLMProvider（推荐用于服务端）

`LLMProvider` 实现 `LLMClientResolver`，管理多账号 + 多模型，支持持久化与环境变量自举：

```python
from ctx_weft.providers.llm import (
    LLMProvider, LLMAccount, ModelConfig, ModelLimits, ModelCapabilities,
)
# 真实 adapter（需 [llm] extra）/ 测试 adapter 也都从同一入口取：
# from ctx_weft.providers.llm import AnthropicAdapter, OpenAIAdapter, MockLLMAdapter

provider = LLMProvider(store)          # store 实现 LLMAccountStoreProtocol(save/delete/list_all)
provider.register_account(LLMAccount(
    name="claude",
    api_key="sk-...", base_url="https://api.anthropic.com",
    models=[ModelConfig(
        name="claude-sonnet-4-6", protocol="anthropic",
        limit=ModelLimits(context=200_000, output=32_000),
        capabilities=ModelCapabilities(input=["text", "image"], output=["text"]),
        output_reserve=None,  # ctx-weft 的输入预算预留，缺省按窗口尺寸计算
    )],
    default_model="claude-sonnet-4-6", timeout_sec=120,
))
provider.load_from_store()             # 从持久化恢复
runtime.providers.register_llm_provider(provider)
```

账号/模型管理：`register_account` / `delete_account` / `get_account` / `list_accounts` /
`is_registered` / `add_model(account, ModelConfig)` / `update_model(account, ModelConfig)` /
`remove_model` / `set_default_model` / `get_model_config(account, model)` / `get_client(account, model)`。

> SDK core 不读环境变量。从环境变量自举一个默认账号（`bootstrap_from_env`）由上层应用
> 在子类中实现（host 的 `LLMProvider` 子类读自己约定前缀的环境变量）。

> 内置 adapter 由 `LLMProvider` 按模型的 `protocol` 与输入能力自动构造。
> `OpenAIAdapter` 是组合官方 `openai.AsyncOpenAI` 的独立实现，图片能力由 `image_input` 控制；
> `OpenAICompatibleAdapter` / `OpenAICompatibleMultimodalAdapter` 保留原有兼容网关实现。
> `run_single_task` / `SessionStartParams` 里的 `llm_account` / `llm_model` 会透传给 `get_client()`。

`protocol` 支持以下值：

| 协议 | 默认实现 | 输出预算字段 |
| --- | --- | --- |
| `openai-official` | `OpenAIAdapter`，官方 Python SDK，Chat Completions | `max_completion_tokens` |
| `openai-compatible` | `OpenAICompatibleAdapter`，兼容网关 | `max_tokens` |
| `anthropic` | `AnthropicAdapter`，Messages API | `max_tokens` |

历史值 `openai` 继续使用兼容实现，已有账号、模型与默认选择不自动迁移。
两种 OpenAI 协议均使用 `/v1/chat/completions`；官方 adapter 关闭 SDK 自动重试，
由 runtime 统一决定恢复，并将 SDK 返回的文本、工具调用与用量转换为 `LLMChunk`。
直接创建官方 adapter 时，使用后调用 `await adapter.aclose()` 关闭 SDK 客户端。

输入能力为
`["text"]` 或含 `"image"` 的列表，输出目前支持 `["text"]`。同一账号下的模型可以使用不同协议和能力，
provider 按账号、协议和图片能力复用 adapter。纯文本 adapter 收到图片时会降级成 `[image ...]` 文本占位
并记一条 warning，**不中断会话**；core 对模态零判断，全程透传到 `LLMClient` 面前。
自写 adapter 时，`complete()` 收到的 `LLMMessage.content` 可能是
`list[ContentPart]`，如何处置完全由你决定。

---

## CtxWeftRuntime API

`CtxWeftRuntime` 是使用 ctx-weft 的唯一入口。

### 构造

```python
# 模板进入 core 的唯一通道是 AgentCapabilityProvider——构造前先注册（发现与加载同源）；
# registry 里一个 AgentCapabilityProvider 都没有会在构造期抛 ValueError（fail-fast）。
providers = providers or ProviderRegistry()
providers.register_capability(LocalAgentTemplateProvider(Path("resources/templates")))

runtime = CtxWeftRuntime(
    providers=providers,              # 必需含至少一个 AgentCapabilityProvider
    llm=my_llm_adapter,               # 可选，LLM 兜底（未注册 llm provider 时用）
    event_bus=None,                   # 可选，默认 InProcessEventBus；注入以接 Redis / NATS / 跨进程
    event_store=None,                 # 可选，默认 InMemoryEventStore（自动订阅 event_bus）
)
```

构造时 **ControlCapabilityProvider**（内置控制工具）与 **SkillExecutorCapabilityProvider**
会**自动注册**，无需手动添加。常用属性：`runtime.event_bus` / `runtime.event_store` /
`runtime.hitl` / `runtime.hitl_registry` / `runtime.providers`。

主要运行入口：

| 方法 | 用途 | 返回 |
|------|------|------|
| `run_single_task(...)` | 单任务便利入口 | `(TurnHandle, LoopState)` |
| `create_session(...)` | 新建空容器 | `SessionHandle` |
| `start_session(params)` | 新建会话并派发首条任务 | `TurnHandle` |
| `start_session(params)`（`resume=True`） | 在已有会话开启新 root 轮次 | `TurnHandle` |
| `send_message(agent_id, content, ...)` | 继续对话，按需自动装填 | `TurnHandle` |
| `dispatch_task(session_id, content, ...)` | 派发新任务，按需自动装填 | `TurnHandle` |
| `rebuild_session(id)` | 装填冷会话的 HITL、agent 与 TaskManager（**只装填，不跑**） | `int` |
| `recover_session(session_id)` | 复用准备好的 owner，激活任务并 drain（缺 owner 时先统一冷准备） | `None` |
| `cancel_session(id)` | 取消会话范围工作 | `bool` |

### run_single_task() — 单任务直跑

跑一个任务，**await 直到完成**再返回。适合脚本、CLI、单测：

```python
handle, state = await runtime.run_single_task(
    template_id="agent:my_agent",    # 必需；规范形式 provider:name（见下方 AgentCapabilityProvider）
    user_prompt="总结这份文档：...",   # 必需
    session_id=None,                 # 可选，不传则自动生成 ses_xxx
    tenant_id="default",
    llm_account=None, llm_model=None,
)
print(state.task.status)                     # "FINISHED" / "FAILED" / "CANCELED"
print(state.verdict.task_outcome)            # "success" / "failed"
print(state.verdict.summary)                 # Observer 总结
print(state.transcript[-1].assistant_text)   # 最后一轮 LLM 回复
```

### start_session() — 新建会话（完整模式）

支持多 agent 树、`submit_task` 派生子任务、suspend/resume、compact。入参封装在
**`SessionStartParams`**，用 `.create()` 构造；`start_session` **只接受这一个参数**：

```python
from ctx_weft import SessionStartParams

params = SessionStartParams.create(
    template_id="agent:planner_agent",  # 必需；规范形式 provider:name
    user_prompt="研究并撰写量子计算报告", # 必需
    session_id=None,                    # None=生成新 id；恢复需 resume=True
    initial_task=None,                  # 可选，dict → 反序列化为 TaskSettings（见下）
    tenant_id="default",
    llm_account=None, llm_model="claude-sonnet-4-6",
)
handle = await runtime.start_session(params)   # 立即返回，任务在后台 drain
final_state = await handle.wait_for_finish(timeout=300.0)
```

Token 预算全部由所选 LLM 配置提供：`limit.context` 是总窗口，`output_reserve` 是输入侧输出预留，
`limit.output` 是单次输出硬上限，`output_policy` 配置输出软上限和估算安全余量。
Session 和 `SessionStartParams` 不接受 `context_limit`、`reserved_output_tokens` 或 `token_budget`。
每次派发解析不可变的 `ResolvedModel`，直接携带 client、模型身份、窗口和输出策略；
prepare、assembler、compact、act 和 gateway 共用这份预算。
`effective_input_limit = context_limit - output_reserve`，压缩比例以它为分母。
`set_agent_llm` / `set_session_llm` 在下一次派发生效，当前 run（包括返回 prepare）继续使用原快照。
恢复旧会话时忽略事件与快照中遗留的 session 预算，新 run 使用当前 LLM 配置。

**恢复已有会话**：传已有 `session_id` 并设置 `resume=True`，runtime 从事件存储重建
`root_agent_id`，把新 `user_prompt` 作为后续输入续跑：

```python
params = SessionStartParams.create(
    template_id="agent:planner_agent",
    user_prompt="补充一节关于纠错码的内容",
    session_id="ses_01HXXXX",          # 复用已有 session
    resume=True,
)
handle = await runtime.start_session(params)
```

**TaskSettings**（`initial_task` dict 按 `_type` 反序列化，默认 `NormalTaskSettings`）：

```python
initial_task = {
    "_type": "NormalTaskSettings",   # 缺省即此类型
    "skill_name": "",                # 绑定某个 skill 作为本任务上下文
    "use_subagent": False,           # True → 用子 agent 执行
    "subagent_template": "",         # 子 agent 模板 id
    "inherit_memory": True,          # 子任务是否继承父 agent 近期记忆
    "purpose": "act",
}
```
`CompactTaskSettings` / `MetadataFillerTaskSettings` 仍保留为类型（反序列化兼容），但已不再作为独立 task 调度：
compact 由 ReasonStep 命中阈值后内联直调 CompactStep；metadata_filler 由后台协程直跑在 root task 上。

### 调用配置：agent 选择与每次请求

`ModelCallConfig` 是可由 adapter 继承的协议 dataclass。`ModelChoice.config` 保存 agent
选择；派发时复制到 `ResolvedModel.config`，每次请求再复制到 `LLMRequest.config`。
`max_tokens`、`temperature` 已从 `LLMRequest` 顶层迁入 config；`LLMClient.complete`
签名保持 `complete(request, stream=True)`，adapter 从 `request.config` 读取配置。

```python
from ctx_weft import SessionStartParams
from ctx_weft.protocols import ModelCallConfig

handle = await runtime.start_session(SessionStartParams.create(
    template_id="agent:planner_agent", user_prompt="审查方案",
    llm_config=ModelCallConfig(reasoning_effort="high", max_tokens=32768),
))
await runtime.set_agent_llm(
    handle.agent_id, llm_account="account-a", llm_model="model-b",
    llm_config=ModelCallConfig(reasoning_effort="low"),
)
```

设置命令替换完整选择；省略 `llm_config` 表示空配置，不是保留旧配置。字段为 `None`
表示不指定该参数；`max_tokens=None` 仍由网关计算安全输出上限。网关只修改本次请求的
`config.max_tokens`，不回写 agent/run。adapter 也只处理请求副本，重试重新复制原始请求。

`create_session`、`run_single_task`、`dispatch_task` 也接受 `llm_config`。`dispatch_task`
只在创建新 agent 时使用它；已有 agent 的配置通过设置命令修改。恢复已有会话不接收
新配置，应先设置 agent 选择再恢复。`set_session_llm` 只批量修改已有 agent，仍发 N 条
`AgentLlmChanged`，不保存会话默认值。新子 agent 复制父选择；新独立树未传配置时使用空配置。

只有 `AgentInstantiated` / `AgentLlmChanged` 和 agent 快照持久化选择，`SessionCreated`
不增加配置字段。`LLMPromptSent.llm_config` 记录预算计算后交给 adapter 的配置。
配置载荷为 `{type, version, values}`，旧事件/快照缺失时恢复空配置。自定义配置子类须在
创建或恢复 agent 前调用 `register_model_call_config` 注册稳定 ID；未知类型/版本不会
静默丢字段。注册可提供 `decode` 函数处理嵌套类型；默认编码使用 dataclass 的 JSON 字段。

```python
from dataclasses import dataclass
from ctx_weft.protocols import ModelCallConfig, register_model_call_config

@dataclass
class GatewayConfig(ModelCallConfig):
    thinking_budget: int | None = None

register_model_call_config("my-gateway", GatewayConfig, version=1)
```

内置 OpenAI 两条协议直接发送 `reasoning_effort`；Anthropic 将 `none`/`off` 转为关闭，
将 `low`/`medium`/`high` 映射为 1024/4096/16384 的 thinking 预算。
`from ctx_weft.providers.llm import AnthropicCallConfig` 可显式设置 `thinking_budget_tokens`。
预算不满足输出上限、未知档位或缺少必要签名历史时明确报不可重试错误，不静默降档。
宿主迁移接口清单见 [调用配置迁移说明](docs/model-call-config-migration.md)。

### 崩溃恢复

**恢复是用户驱动的：用到哪条会话，就装填哪条。** 启动时不做任何预热——从前那个扫「全部 active session」的
`runtime.recover()` 已于 2026-09-09 删除（它的活跃判据只增不减，开销与历史会话数线性增长且永不收敛）。

```python
# ① 装填：回放一次事件，恢复 HITL、agent 和 TaskManager；不入队、不执行任务。
#    冷启动残留的 running agent 会校准为 interrupted，并记录该状态转移。
#    strict=True 将准备失败直接交给调用方；已有活 owner 时复用，不重复回放。
n = await runtime.rebuild_session(session_id, strict=True)  # 返回装填的 agent 条数

# ② 之后才谈操作这条会话里的 agent：
await runtime.send_message(agent_id, "继续", session_id=session_id)   # 接着聊
await runtime.recover_session(session_id)                                 # 续跑被打断的
```

`rebuild_session` 中的 `TaskManager.restore(..., enqueue=False)` 保留持久化任务状态，
队列保持为空，不补注入答复、不启动 recap，也不调用模型。发送消息先用准备好的 owner
判断旧任务是否终态，交付输入后才激活并 drain；显式恢复也复用同一个 owner。
`waiting_human` 不会被冷重建改成 `interrupted`，未答复的 HITL 仍保持等待。

⚠️ **顺序不能反。** agent record 是只增不删的**缓存**，一次 miss 只说明「还没喂进来」，事实一直在事件日志里。
但 core 不会替你去找那条会话——事件按 session 分区存，`agent_id` 没有反向索引，「只有 agent_id」时唯一的找法
是扫全部会话。所以：

- `send_message` 撞上 miss 时，**带了 `session_id` 就自愈**（精确装填那一个 session），没带就抛 `AgentNotLoaded`。
- `recover_session(session_id)` 自行确保该会话已初始化、装填和存储提交确认，再恢复已有工作。它不创建新任务，也不替人回答 HITL。

`AgentNotLoaded` 是 `AgentNotFound` 的子类，既有的 `except AgentNotFound` 照样接住；想区分「没装填」与
「真不存在」的再按它分支。

宿主的投影表（`load_sessions_from_db` 那类）不再需要排在某个全局恢复步骤之后——按会话装填，各自独立。

恢复流程的内部细节见 [ARCHITECTURE.md §会话生命周期与崩溃恢复](./ARCHITECTURE.md#会话生命周期与崩溃恢复)。

### RunHandle

```python
@dataclass
class RunHandle:
    run_id: str
    session_id: str
    task_id: str
    agent_id: str
    template_id: str
    event_bus: EventBus

    async def events(self) -> AsyncIterator[Event]: ...        # 流式订阅本 run 全部事件
    async def wait_for_finish(self, timeout: float = 300.0) -> LoopState | None: ...
```

---

## 事件系统

> **对话配对完整性（2026-09 起，spec: conversation-integrity）**：assistant 回合摄入时把 LLM 分配的 tool_call wire id 替换为内部唯一标识（`tc_{seq36}_{ord36}_{hash12}`，`[a-z0-9_]`、≤64 字符）——模型复用 `call_1` 这类短 id 不再造成跨轮次/跨任务的结果错配。**值域迁移注记**：事件 payload（`CapabilityInvoked` / `CapabilityFinished` / `LLMResponseFinished.tool_calls`）与 HITL 请求中的 `tool_call_id` 从裸 wire id 变为内部标识；原始 id 保留在 assistant 记录 metadata（`tool_calls[].raw_id`，伴随 `op_id`）供追溯。宿主面板若按 id 展示/匹配需知悉；存量记录不迁移（配对行为不回退，重复歧义在发送前留 ERROR 痕迹）。
>
> **快照恢复（2026-09 起，spec: snapshot-recovery）**：快照边界 = 已确认提交位置（`committed_head` 一致切面），恢复增量按 position 区间；旧格式快照自动忽略并全量重建，无需手工迁移事件数据（存量库回填用 `scripts/migrate_event_positions.py`）。
>
> **执行时长/轮数限制（2026-09 起）**：core **不提供**任务级 deadline 或轮数上限——这是宿主的策略，不是 SDK 的职责。需要超时就在宿主侧计时并调 `cancel_task` / `cancel_agent`；需要 provider 级超时就在 provider 自己的工具实现里做。三个从未被执行的旧字段（`max_turns_per_agent` / `timeout_per_step_sec` / `Task.timeout_ms`）**已删除**——它们声明了限制却无任何运行时消费者，留着只会让宿主以为有防线。`max_turns_per_act` / `max_turns_per_observe` 等**真正生效**的循环上限不受影响。

Observe 的 LLM 请求另有无有效进展保护：`RuntimeConfig.observe_llm_idle_timeout_sec` 默认 300 秒，须为有限正数。文本、思考和工具调用分片会刷新计时，SSE 心跳及空 chunk 不刷新；正常持续生成可超过该时长。超时会关闭请求，并按观察失败的既有规则处理；仅生成报告的 observe 保留原始记忆并完成生命周期，释放任务锁。后台报告仍独立于 actor 的取消信号。

前台 observe 的暂停、取消也覆盖判决后的记忆折叠，并在 finalize 入口再次检查停止信号。停止时等待在途事务清理；已提交的折叠保留。暂停会原子闭合旧 observe 并登记新的人工等待，取消则在任务和记忆收尾完成后发送 `ObserveApplied`，保证完成等待可返回。

`TaskFinalized` 写入及提交通知期间不接收新的暂停、取消请求，沿用已确定的结局正常收尾。被跳过的 agent 接口请求返回空列表，会话接口返回 `False`；不产生取消标记或 agent 终止事实，也不把信号延迟到下一轮。会话级请求整次跳过，避免留下会话控制闩锁。写入完成或失败后立即解除保护，区间前已接受的停止仍按原逻辑处理。

公开停止接口在等待初始化、会话命令锁之前检查写入保护；初始化完成后及取得会话锁后再次核对。保护期间的请求立即返回，不排队到后续运行，也不影响同会话内其他 agent 的定向暂停。

带人工等待入口的新后台判决会先等原 run 提交停驻状态，避免判决过早到达而被拒绝；该交接等待同样受 `observe_llm_idle_timeout_sec` 约束。

> **提交策略（2026-09 起，spec: event-commit）**：默认 `event_commit_policy="required"`——事件先经提交门确认存储写入、再对外通知（存储失败显式抛 `PersistenceUnavailableError` 并隔离会话）。**自定义 EventBus 必须实现 `attach_commit_gate` 扩展**，否则 required 模式构造期失败；不接受该约束的宿主可显式配置 `RuntimeConfig(event_commit_policy="best_effort")` 退回旧的吞错路径（启动告警、不可靠恢复）。

临时存储故障按原 `batch_id`、原事件 ID 和不可变内容确认提交，恢复过程不重新执行工具。
默认原地提交最多 3 次；每会话最多 3 轮后台恢复，初始退避 0.25 秒，总恢复期限 120 秒，
分别由 `storage_commit_max_attempts`、`storage_recovery_max_attempts`、
`storage_recovery_base_delay_sec`、`storage_recovery_max_duration_sec` 配置。
所有未知批次、待通知事件和已完成工具/assistant 结果收口后才解除隔离。
预算耗尽仍保留原始提交，可在存储恢复后调用 `await runtime.recover_storage(session_id)`；
返回 `False` 时隔离继续，`clear_storage_isolation` 不会绕过未确认提交。
SQL provider 将锁等待、连接池超时及可恢复连接错误标记为 `TransientPersistenceError`；
其他 provider 可使用同一协议异常。自定义总线若要参与后台确认，还须实现
`confirm_pending_storage` / `has_pending_storage`；`attach_storage_recovery` 可让原调用等待
单批回执后原地继续。进程崩溃且结果从未落盘时，仍按副作用核对/授权策略处理，不承诺任意外部工具恰好一次。

所有状态变更都通过事件总线发布，append-only。

### Event 结构

```python
@dataclass
class Event:
    id: str               # evt_ULID
    run_id: str | None
    sequence: int         # 在同一 run_id 内单调递增
    session_id: str
    type: str             # 见 EVENT_TYPES（冻结清单）
    timestamp: datetime
    tenant_id: str = "default"
    task_id: str | None
    agent_id: str | None
    payload: dict         # 按 type 不同而异
    metadata: dict
    causation_id: str | None
    schema_version: int
```

### 常用事件类型

事件类型在 `ctx_weft.core.events.types.EVENT_TYPES` 中**冻结**（业务代码不得发未登记类型）：

- **Run/Step**：`RunStarted` `RunFinished` `RunCanceled` `RunPaused` `RunResumed` `StepStarted` `StepCompleted` `StepFailed`
- **Session**：`SessionCreated` `SessionResumed`；会话状态由 host 维护，`SessionStatusChanged`、`SessionFinished`、`SessionPausedHitl` 已停发，仅兼容旧日志。
- **LLM**：`LLMRequestStarted` `LLMTokenStreamed` `LLMResponseFinished`
- **Capability**：`CapabilityStarted` `CapabilityProgress` `CapabilityFinished`
- **Task**：`TaskStarted` `TaskFinished` `TaskFailed` `TaskSuspended` `TaskCanceled`
- **Observe**：`ObserveRequested` `ObserveStarted` `ObserveCompleted` `ObserveApplied`；前后台统一，旧 `TaskRecap*` 已停发。
- **Memory**：`MemoryIngested` `MemoryCompactionStarted` `MemoryCompacted`

`RunFinished.payload` 含 `final_status` / `will_retry` / `total_events` / `total_turns` / `error` / `error_type`。

### 订阅事件

```python
from ctx_weft.core.events.types import EventFilter

# 用 handle 订阅本 run
async for ev in handle.events():
    if ev.type == "LLMTokenStreamed":
        print(ev.payload.get("delta", ""), end="", flush=True)
    elif ev.type == "RunFinished":
        break

# 用 event_bus 自定义过滤（session / run / task / types）
async for ev in runtime.event_bus.stream(EventFilter(session_id="ses_xxx")):
    ...
```

---

## 内置 Provider

### 工具长输出的收敛与回取（spec: tool-result-recovery）

工具输出超过 `spill_threshold`（默认 4000 字符）时：全文交给 **`SpillSink`**（core 对「超长输出去哪」的唯一契约，靠 `isinstance` 从已注册 provider 里发现），上下文（对话与 memory）只承载**收敛版** = sink 返回的**取回说明**（整句由 sink 自己写，点名用哪个工具取回；gateway 原样嵌入）+ 全长 + 头部预览 + **尾部预览**（`spill_preview_chars` / `spill_tail_chars`）。

宿主注册哪种 sink 决定能力上限：`FilesystemToolsProvider` 落盘进 workspace，说明里点名 `fs__read_file(path=..., offset=1, limit=500)`；**`ResultsCapabilityProvider`** 落进内存 LRU 并自带 `results__read_tool_output(invocation_id, offset | tail, limit)` 工具，**模型能分页/尾部取回全文**（invocation_id 每次执行一枚，取最新收敛文本里的值）。该 provider 同时是 SpillSink 与 ToolCapabilityProvider，注册一次两个角色齐备。**都不注册**则超长输出硬截断。

四个重放/补写入口（账本 completed 短路重放、恢复补写、裁决作结、宿主补录）统一过收敛——重放时再 spill 一次即可：可回读的 sink 借此在逐出/重启后重新入库，落盘的 sink 重写同名文件无害。无 sink 或 spill 抛错 → 收敛版显式标注 `not recoverable`，不留取不回来的引用。

### InMemoryMemoryProvider

零依赖，用于开发/测试：

```python
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
providers.register_memory(InMemoryMemoryProvider())
# recall_recent ✓ / recall_topic ✓ / recall_semantic ✗（返空）；线程不安全，仅单进程/测试
```

### BuiltinToolsCapabilityProvider

需要 `pip install "ctx-weft[builtin]"`：

```python
from pathlib import Path
from ctx_weft.providers.capability_builtin import (
    BuiltinToolsCapabilityProvider, BuiltinToolsConfig,
)

providers.register_capability(BuiltinToolsCapabilityProvider(
    BuiltinToolsConfig(allowed_dirs=[Path("/workspace")])   # 文件操作限定目录；空列表=不限制
))
```

> `BuiltinToolsConfig` **只有 `allowed_dirs` 一个字段**。bash 黑名单、命令超时、HTTP SSRF
> 防护、响应大小上限等都是 provider 内置常量，不通过 config 开关。

提供的 capability：

| ID | Purpose | 描述 |
|----|---------|------|
| `builtin:bash_exec` | act | 执行 shell（黑名单含 rm/sudo/curl 等；30s 超时） |
| `builtin:http_request` | act | HTTP 请求（仅 http/https；内网地址 SSRF 拦截；剥离敏感 header） |
| `builtin:read_file` | act, compact | 读文件（≤500KB） |
| `builtin:write_file` | act | 写文件（自动建父目录） |
| `builtin:edit_file` | act | 原地替换（old_string→new_string；要求唯一匹配或 replace_all） |
| `builtin:glob` | act, compact | 列出匹配文件（按文件名） |
| `builtin:grep` | act, compact | 按内容正则搜索（ripgrep 优先，回退纯 Python；files/content 两种输出模式） |

### MCPCapabilityProvider

桥接任意 MCP server，需要 `pip install "ctx-weft[mcp]"`：

```python
from ctx_weft.providers.capability_mcp import MCPCapabilityProvider, MCPServerConfig

# stdio transport（启动子进程）
providers.register_capability(MCPCapabilityProvider(MCPServerConfig(
    name="filesystem", transport="stdio",
    command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
    env={}, default_purposes=["act"], timeout_per_call_sec=60, connect_timeout_sec=10,
)))

# http / streamable_http transport
providers.register_capability(MCPCapabilityProvider(MCPServerConfig(
    name="my_service", transport="http",
    url="http://localhost:3000", headers={"Authorization": "Bearer xxx"},
    capability_purpose_override={"some_tool": ["observe"]},  # 单工具 purpose 覆盖
)))
# provider.name = "mcp:<name>"，cap.id = "mcp:<name>:<tool>"
```

### LocalSkillCapabilityProvider

从本地目录加载 SKILL.md：

```python
from pathlib import Path
from ctx_weft.providers.capability_skill_local import LocalSkillCapabilityProvider
providers.register_capability(LocalSkillCapabilityProvider(Path("./skills")))
# 目录结构：skills_dir/<skill_name>/SKILL.md（+ 可选 references/、scripts/）
# capability id 前缀 local_skill:
```

### RemoteSkillCapabilityProvider

从 Git 仓库同步 SKILL.md，需要 `pip install "ctx-weft[skills]"`：

```python
from pathlib import Path
from ctx_weft.providers.capability_skill_remote import (
    RemoteSkillCapabilityProvider, GitSkillSyncer,
)

syncer = GitSkillSyncer(repo_url="https://github.com/org/skills",
                        cache_dir=Path("./skills_cache"), branch="main")
providers.register_capability(RemoteSkillCapabilityProvider(
    source_name="github",          # 必填：决定 provider.name 与 cap.id 前缀
    cache_dir=Path("./skills_cache"),
    syncer=syncer,
))
# name = "remote_skill_github"，cap.id = "remote_skill_github:<skill>"
# 首次 list()/load_definition() 触发 sync；reset_sync() 强制下次重新同步
```

### 内置控制 Capability（ControlCapabilityProvider）

由 `CtxWeftRuntime` 自动注册，无需手动添加。向 LLM 暴露：

| capability id | purpose | 描述 |
|--------------|---------|------|
| `control:submit_task` | act | 派生子 task，当前 agent 挂起等待 |
| `control:submit_plan` | act | 一次提交多个并行/串行子 task |
| `control:replan` | act | 修订当前计划 |
| `control:submit_task_assessment` | observe | Observer 报告 task 结果 |
| `control:update_task_metadata` | metadata_filler | 回填 title / description / session goal |
| `control:request_human_input` | act, observe | 请求人工介入（触发 HITL） |

---

## 控制面

### 取消正在运行的 session

```python
ok = runtime.interrupt_session("ses_01HXXXX")
# True = 找到并已发取消信号；False = 已结束 / 未注册（run_single_task 不注册 cancel token）
# 被取消时发 RunCanceled + TaskCanceled，任务状态置 CANCELED
```

### HITL 人工审批

当 `request_human_input` 等触发人工介入时，发 `SessionPausedHitl` 事件并暂停。上层通过
`runtime.hitl_manager` 应答：

```python
runtime.hitl_manager.approve(request_id, response="同意，继续")
runtime.hitl_manager.reject(request_id, reason="不允许该操作")
```

### 鉴权（Authorizer）

给敏感 capability 挂鉴权器，在 invoke 前拦截：

```python
from ctx_weft.protocols.capability import Authorizer
from ctx_weft.providers.authorizer import (
    AllowAllAuthorizer, AllowListAuthorizer, HumanConfirmationAuthorizer,
)

# 方式 ①：注册 capability 时传入
providers.register_capability(
    BuiltinToolsCapabilityProvider(),
    tool_authorizers={
        "builtin:bash_exec": HumanConfirmationAuthorizer(hitl_manager=runtime.hitl_manager),
    },
)

# 方式 ②：事后单独设置（key 为 provider_name 或完整 capability_id）
providers.set_capability_authorizer(
    "builtin:bash_exec",
    HumanConfirmationAuthorizer(hitl_manager=runtime.hitl_manager),
)
```

---

## 自定义 Provider

### 自定义 MemoryProvider

需实现全部抽象方法（`ingest` / `recall_recent` / `recall_topic` / `recall_semantic` /
`subscribe_topic` / `list_subscriptions` / `apply_compact` / `count_recent` / `describe`）：

```python
from ctx_weft.protocols import (
    MemoryProvider, MemoryEvent, MemoryEventType, MemoryScope, MemoryRecord,
    MemoryProviderInfo, CompactResult, Subscription, ProviderContext,
)

class MyMemoryProvider(MemoryProvider):
    name = "my_memory"

    async def ingest(self, event, ctx) -> str: ...
    async def recall_recent(self, scope, types, limit, ctx) -> list[MemoryRecord]: ...
    async def recall_topic(self, topic, since, ctx) -> tuple[list[MemoryRecord], int]: ...
    async def recall_semantic(self, query, scope, top_k, ctx) -> list[MemoryRecord]: ...
    async def subscribe_topic(self, session_id, topic, intent, ctx) -> str: ...
    async def list_subscriptions(self, session_id, ctx) -> list[Subscription]: ...
    async def apply_compact(self, scope, summary, keep_last, ctx) -> CompactResult: ...
    async def count_recent(self, scope, types, ctx) -> int: ...
    async def describe(self, ctx) -> MemoryProviderInfo:
        return MemoryProviderInfo(name=self.name, supports_semantic=True, supports_topic=True)
```

### 自定义 ToolCapabilityProvider

```python
from collections.abc import AsyncIterator
from ctx_weft.protocols.capability import (
    ToolCapabilityProvider, Capability, ToolCapability,
    CapabilityEvent, CapabilityProviderInfo,
)
from ctx_weft.protocols import ProviderContext

class DatabaseProvider(ToolCapabilityProvider):
    name = "database"

    async def list(self, ctx) -> list[Capability]:
        return [ToolCapability(
            id="database:query", name="database_query", purposes=["act"],
            description="Run a read-only SQL query.",
            input_schema={"type": "object",
                          "properties": {"sql": {"type": "string"}}, "required": ["sql"]},
            side_effects=False,
        )]

    def invoke(self, capability_id, arguments, ctx) -> AsyncIterator[CapabilityEvent]:
        return self._run(arguments)

    async def _run(self, args) -> AsyncIterator[CapabilityEvent]:
        try:
            rows = await self._db.execute(args["sql"])
            yield CapabilityEvent(kind="result",
                                  payload={"content": str(rows), "metadata": {"rows": len(rows)}})
        except Exception as e:
            yield CapabilityEvent(kind="error", payload={"code": "QUERY_FAILED", "message": str(e)})

    async def cancel(self, invocation_id, ctx) -> None: ...
    async def describe(self, ctx) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=1)
```

> 还可继承 `SkillCapabilityProvider`（`load_definition` / `list_files` / `load_resource` /
> `exec_script`）或 `AgentCapabilityProvider`（发现与加载同源：`list()` 返回 `AgentCapability`，
> `get_template()` 按 cap.id 前缀路由加载对应 `AgentTemplate`——模板进入 core 的唯一通道）。

### 自定义 KnowledgeProvider

```python
from collections.abc import AsyncIterator
from ctx_weft.protocols import (
    KnowledgeProvider, KnowledgeQuery, KnowledgeDoc, KnowledgeProviderInfo, ProviderContext,
)

class VectorSearchProvider(KnowledgeProvider):
    name = "vector_search"

    async def retrieve(self, query, ctx) -> AsyncIterator[KnowledgeDoc]:
        for r in await self._index.search(query.text, top_k=query.top_k):
            yield KnowledgeDoc(id=r.id, content=r.text, score=r.score, source=self.name)

    async def describe(self, ctx) -> KnowledgeProviderInfo:
        return KnowledgeProviderInfo(name=self.name)
```

### 自定义 LLMAdapter

```python
from collections.abc import AsyncIterator
from ctx_weft.protocols import LLMClient, LLMChunk, LLMRequest, LLMUsage

class MyLLMAdapter(LLMClient):
    @property
    def context_limit(self) -> int: return 128_000
    @property
    def output_reserve(self) -> int: return 4096
    @property
    def supports_tool_calling(self) -> bool: return True

    def complete(self, request, stream=True) -> AsyncIterator[LLMChunk]:
        return self._stream(request)

    async def _stream(self, request) -> AsyncIterator[LLMChunk]:
        async for c in my_api.stream(request.system, request.messages):
            yield LLMChunk(kind="token", text=c.text)
        yield LLMChunk(kind="usage", usage=LLMUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150))
        yield LLMChunk(kind="done", finish_reason="stop")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4
```

---

## 在测试中使用

### MockLLMAdapter

```python
from ctx_weft.testing import MockLLMAdapter, MockResponse, ToolCall

# 纯文本，按队列依次返回
llm = MockLLMAdapter(responses=[MockResponse(text="pong")])

# 返回 tool call，下一条是工具结果后的回复
llm = MockLLMAdapter(responses=[
    MockResponse(text="", tool_calls=[ToolCall(id="c1", name="bash_exec",
                                               arguments={"command": "ls"})]),
    MockResponse(text="done"),
])

# 断言收到的请求
_, state = await runtime.run_single_task(template_id="agent:echo", user_prompt="ping")
assert llm.last_request.system.startswith("You are")
```

### 完整集成测试示例

```python
import pytest
from ctx_weft import CtxWeftRuntime, ProviderRegistry
from ctx_weft.testing import MockLLMAdapter, MockResponse
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.protocols import (
    AgentCapability, AgentCapabilityProvider, AgentTemplate,
    CapabilityProviderInfo, IdentityFacet, LoopConfig, MemoryConfig,
)

@pytest.fixture
def echo_template():
    return AgentTemplate(
        id="echo", name="Echo", version="1.0.0",
        identity={"act": IdentityFacet(text="You are a helpful echo agent.")},
        capability_refs=[], memory_config=MemoryConfig(), loop_config=LoopConfig(),
    )

@pytest.fixture
def runtime(echo_template):
    class Provider(AgentCapabilityProvider):
        name = "agent"
        async def list(self, ctx) -> list[AgentCapability]: return []
        async def get_template(self, tid, version, ctx): return echo_template
        async def describe(self, ctx): return CapabilityProviderInfo(name=self.name)
    providers = ProviderRegistry()
    providers.register_memory(InMemoryMemoryProvider())
    providers.register_capability(Provider())
    return CtxWeftRuntime(
        llm=MockLLMAdapter([MockResponse(text="pong")]),
        providers=providers,
    )

@pytest.mark.asyncio
async def test_single_task(runtime):
    handle, state = await runtime.run_single_task(template_id="agent:echo", user_prompt="ping")
    assert state.verdict.task_outcome == "success"
    assert state.transcript[-1].assistant_text == "pong"
```

---

## 升级须知（模型配置）

- 账号移除 `style`，每个模型显式配置 `protocol="openai" | "anthropic"`。
- 模型的 `context_limit` 移到 `limit.context`，`output_ceiling` 移到 `limit.output`；
  `output_reserve` 仍在模型层，`None` 自动计算，`0` 表示不预留。
- `capabilities.input=["text", "image"]` 选择图片 adapter，`["text"]` 选择文本 adapter；
  `capabilities.output` 当前支持 `["text"]`。这份配置由 provider 消费，core loop 仍只依赖 client。
- 账号与模型配置类型集中在 `providers/llm/config.py`；Store 契约与 tokenizer 分别保留在
  `store.py` 和 `tokenizer.py`，`provider.py` 管理账号并返回绑定模型的 client。
  Store 应将外部配置转换成 `LLMAccount` / `ModelConfig`；旧账号级 style 和扁平种子的迁移
  由宿主读取入口负责。`dataclasses.asdict(account)` 可用于保存当前结构。
- 内置 adapter 与共用辅助实现位于 `providers/llm/adapters/`。
  仍可从 `ctx_weft.providers.llm` 导入公开 adapter 类；直接引用实现模块时请使用
  `ctx_weft.providers.llm.adapters.openai` / `.anthropic` / `.mock`。
  宿主共用的 `ReasoningCandidate` 从 `ctx_weft.providers.llm.adapters` 导入。
- `add_model`、`update_model` 接收完整的 `ModelConfig`。`get_client` 要求模型已登记；
  验证尚未保存的模型可调用 `verify_model_for_account(account, candidate_config)`。
- 拉取模型列表和账号连通性探测需显式指定 `protocol`：
  `fetch_models_for_account(account, protocol="openai")`、`ping_account(account, protocol="anthropic")`。
  内置 adapter 按服务根拼接 `/v1/{operation}`；宿主可通过子类重写 `_api_url`，统一适配消息调用与模型列表的地址。
- 运行时 client 的 `context_limit`、`output_ceiling`、`output_reserve` 属性保持原语义。

## 多模态行为

- **破坏性变更：`ModelConfig.supports_vision` 已删除。** 若你的账号配置里还带着
  `ModelConfig(..., supports_vision=True)`，升级后构造会直接失败——把这个参数
  从你的配置里删掉，改用 `capabilities.input`。core 入口不再
  做任何视觉门控（旧的 `VisionNotSupportedError` / `error_code=VISION_NOT_SUPPORTED`
  一并删除）。
- **能力由 provider 根据模型配置选择 adapter 类。** 用 `LLMProvider` 的宿主配置
  每个模型的 `protocol` 和 `capabilities.input`；直接 `llm=` 一个 `LLMClient`
  的宿主，则由你传的是 `AnthropicAdapter`/`OpenAIAdapter` 还是
  `AnthropicMultimodalAdapter`/`OpenAIMultimodalAdapter`（或自定义等价类）决定。
- **行为变化：图片不再在入口被拒绝。** 携带图片（`ImagePart`）的会话现在会被
  正常接受、落库，并按既有规则外部化到 `MemoryBlobStore`——不论当前挂的是哪种
  adapter。纯文本 adapter 会在**出网前**把图片降级成 `[image {media_type}]`
  文本占位并记一条 warning（每次请求最多一条），**不中断会话**。
  实际影响：即使当前账号绑定的是纯文本 adapter，图片依然会占用 memory / blob
  storage 与 token 预算——换来的是，日后把该账号切到对应的多模态 adapter，
  同一份既有历史立刻可以看图，不需要重新携带一次。
- 若你的宿主此前依赖入口拒绝携图请求（例如用 `VisionNotSupportedError` 做业务
  分支），升级后需要自己在应用层做等价判断——core 不再替你拒绝。

## 升级须知（双 blob store）

- **破坏性变更：携图会话默认失败，除非显式注册 `EventBlobStore`。** 事件库的口径是
  **恒不含字节、恒可回读**，没有例外分支——升级到本版本后，任何携带图片
  （`ImagePart`）的会话会在入口即以 `BlobStoreRequiredError`
  （`error_code=BLOB_STORE_REQUIRED`）失败，**除非**在 `ProviderRegistry` 上调用
  `register_event_blob_store()` 注册一个能外部化的 `EventBlobStore` 实现。这是刻意
  的严格默认：没有 event blob store 就无处安放字节，只能在入口拒绝，而不是让图片
  的原始字节悄悄进事件库、之后再无处收拾。纯文本会话不受影响，行为逐字节不变。
  若你的宿主此前携图跑过（不管是否接了 `MemoryBlobStore`），升级后需要额外注册
  `EventBlobStore` 才能继续工作；宿主若共用同一份存储服务两侧，同一个实现类可以
  同时满足 `MemoryBlobStore` 与 `EventBlobStore` 两个协议（本仓自带的
  `providers/blob/fs/FsBlobStore` 已经同时实现两者，注册两次即可；`SqlMemoryProvider`
  自 2026-08-29 起**不再**实现这两个协议，只维护引用边，见下面「blob 字节移出
  memory」一节）。
- **行为变化：入口内容里已经是 `source_type == "ref"` 的图片 part，不再进事件流。**
  典型场景是宿主把从 memory 读回来的内容（其中的图已被外部化成 memory ref）原样再
  提交一次。这类 part 的 ref 属于 **memory 命名空间**，event blob store 永远解不开，
  故事件侧会把它降级成 `[image {media_type}]` 文本占位并记一条 warning；**memory 侧
  照旧保留该图**。也就是说：进程内与后续 memory 读取都看得见图，但**崩溃恢复后从事件
  流重建出来的那份只剩文本占位**。宿主若有「回读—再提交」的用法，请改为提交原始
  base64（或自己先把字节取回来再提交），不要依赖 ref 原样透传。
- **回读有且仅有一条 core 内置路径：会话恢复。** `Runtime._restore_task_prompts`
  （`core/content.py::hydrate_event_content`）在 `_recover_session_locked` 里把
  重放出来的 event ref 过桥转回 memory ref，逐 task 逐字段包 try/except，失败时
  `logger.error(exc_info=True)` 并降级为占位，不拖垮整场恢复——这是两个 blob
  世界之间**唯一**的桥，方向单一，由 event 侧发起。除此之外 core 不会主动从
  事件流回读图片字节，读侧走的仍是 memory 侧的 `MemoryBlobStore`。其他场景
  （例如导入到一个新的 memory 实例、或做纯事件重放）想把图片字节取回来，
  仍需宿主自己调用 `EventBlobStore.get(ref, ctx)`。若重启后**没再注册**
  `EventBlobStore`，重放出来的 event ref 无从解开，走与「取不回字节」完全相同的降级
  （占位 + warning）——绝不会原样流进 memory 侧的 task 字段。
- **两侧的 ref 是两个独立的命名空间。** memory 侧与 event 侧各自 put、各自拿
  自己的 ref，core 从不比较两者、也从不拿一侧的 ref 去另一侧解。宿主自
  2026-08-28 起可以分开注册两个实现——`providers/blob/fs/FsBlobStore` 是可
  直接用的内容寻址示例，可以各建一个实例分别注册为 `MemoryBlobStore` 与
  `EventBlobStore`。共用同一个实例（如 `providers/blob/fs/FsBlobStore`）时，回收
  必须同时喂两侧的活引用——只喂 memory 侧的 `live_blob_refs()` 会把事件流仍需要
  的字节当孤儿删掉，而 `EventStore` 协议目前没有对应的活引用计算可用，分开部署
  两个实例可回避该陷阱（详见「blob 字节移出 memory」一节与
  `docs/host-migration-to-sql-memory.md`）。

## 升级须知（providers 目录重组）

**破坏性变更：provider 的 import 路径全部变了。** 按「领域 → 协议 → 变体」重组，
不留兼容 shim：

| 旧 | 新 |
|---|---|
| `ctx_weft.providers.memory_blackboard` | `ctx_weft.providers.memory.in_memory` |
| `ctx_weft.providers.memory_sql` | `ctx_weft.providers.memory.sql` |
| `ctx_weft.providers.blob_fs` | `ctx_weft.providers.blob.fs` |

`ctx_weft.providers.events.bus` / `ctx_weft.providers.events.store` **不在上表里**——
它们现在是子包而不是模块，但各自的 `__init__.py` 仍 re-export `InProcessEventBus` /
`InMemoryEventStore`，所以 `from ctx_weft.providers.events.bus import InProcessEventBus`
与 `from ctx_weft.providers.events.store import InMemoryEventStore` 照样能 import，
和改造前一样，`ctx_weft.providers.events` 顶层也一样。真正挪动的只是私有实现模块名
（`bus.py` → `bus/in_process/bus.py`、`store.py` → `store/in_memory/store.py`），
没人应该直接 import 那两个私有模块，因此不构成破坏性变更。

## 升级须知（blob 字节移出 memory）

- **破坏性变更：`SqlMemoryProvider` 不再实现 `MemoryBlobStore` / `EventBlobStore`。**
  它只维护 `memory_blob_refs` 引用边（那部分需要与 ingest 同事务），字节归 blob store。

  **强烈建议分开部署两个 blob store 实例**——一个只挂 `MemoryBlobStore`、一个只挂
  `EventBlobStore`，各自按各自的策略回收，不必互相当心：

  ```python
  memory_blobs = FsBlobStore(Path("/var/lib/app/memory-blobs"))
  event_blobs = FsBlobStore(Path("/var/lib/app/event-blobs"))
  registry.register_memory_blob_store(memory_blobs)
  registry.register_event_blob_store(event_blobs)
  ```

  共用同一个实例给两侧在类型上是允许的（`FsBlobStore` 恰好同时满足两个协议），但这是
  实现层的巧合，不是推荐做法——见下面「回收方式变了」。

- **`ProviderRegistry.get_memory_blob_store()` 不再自动回落到 memory provider。**
  现在与 `get_event_blob_store()` 一样只有两级：显式注册 > `NullMemoryBlobStore`。
  没注册就是「不接 blob」，携图会话会在入口被拒（`BlobStoreRequiredError`）。

- **回收方式变了。** 原先 `await memory.collect_blobs()` 一步搞定，现在是标准的
  mark-sweep 两步——mark 在 memory（它持有引用边），sweep 在 blob store（它持有字节）。
  **分开部署时**（推荐），mark 只需 memory 侧的活引用：

  ```python
  await memory_blobs.collect(await memory.live_blob_refs())
  ```

  ⚠️ **若你选择共用同一个 blob store 实例给两侧**，上面这行是**错的**——
  `live_blob_refs()` 只枚举 memory 侧的活引用，会把事件流仍需要的字节当孤儿删掉。
  `EventStore` 协议目前**没有**对应的活引用枚举接口可用于补全另一半，core 不提供
  现成方案；共用实例时必须自己另想办法算出两侧活引用的并集再喂给 `collect`，
  否则就只回收 memory 侧不再需要、且确定不在事件流里被引用的那部分。这正是
  上面建议分开部署的原因——分开之后这个陷阱不存在。

## 升级须知（HITL 契约两维化 + 多模态出口）

- **破坏性变更：`HitlRequest.status` 已删除，换成 `outcome`。**
  `status` 的四值 `Literal["pending","accepted","rejected","cancelled"]` 是按「批准一次
  工具调用」造的，却要同时服务 `question`（accepted = 人答了）与 `wait`（accepted = 人
  发了条消息）。现在是：

  ```python
  req.outcome     # str，开放值域。"" = 未决；"accepted"/"rejected"/"cancelled" 是内建值
  req.resolved    # 推导属性：bool(outcome)
  req.accepted    # 推导属性：outcome == "accepted"
  req.resolve(o)  # 终局的唯一写入点
  ```

  host 定义了自己的 `form`，现在也能定义自己的 `outcome`。空串是未决哨兵，不要占用。
  `HitlStatus` 已从 `ctx_weft.protocols` 移出，改用 `HitlOutcome` 与
  `HITL_OUTCOME_ACCEPTED` / `HITL_OUTCOME_REJECTED` / `HITL_OUTCOME_CANCELLED`。

- **破坏性变更：`AuthorizationDecision.message` 放宽为 `str | list[ContentPart]`。**
  自实现 Authorizer 只**写** str 的不受影响；**读** `decision.message` 的要处理 parts
  （用 `core.content.split_for_tool_result` 或 `core.utils.content_to_text`）。

- **破坏性变更：`Authorizer.filter()` 已删除。** 它零调用点，且对
  `HumanConfirmationAuthorizer` 会真的发出一个 HITL 请求并等人——把「列出可见工具」
  变成「逐个求批」。需要装配期可见性过滤请自行实现，并显式排除会挂起的 authorizer。

- **行为变更：`ask_user` 与 approval 备注现在会把人贴的图带给模型。** 此前这两条出口
  用 `content_to_text` 展平，非文本 part 被静默丢弃（连占位都不留）。工具结果因此可能
  是 `list[ContentPart]` 而非 `str`——`InvocationResult.content` 早已声明为该联合类型，
  host 通常无感。

- **不变**：事件类型、事件 payload 形状、`RunSnapshot` 序列化、golden 数据。
  旧事件日志与旧快照原样可读，无需迁移。

## 升级须知（事件持久化）

- **`InMemoryEventStore` 不再接受 `event_bus=` 参数，`append()` 也不再过滤瞬态事件。**
  订阅与过滤都归新的 `EventPersister`。若你此前直接调 `event_store.append()`，注意
  每 token 一个的流式 delta 现在会真的落库——改用 `EventPersister` 或自己加过滤。

- **接线改用 `CtxWeftRuntime(event_store=..., snapshot_every_n=...)`。**
  `attach_persistence` 现在由 `CtxWeftRuntime.__init__` 内部调用，接线的正确顺序是
  **先建 store，再把它传进 runtime 构造器**——不要先建 runtime、再自己额外调一次
  `attach_persistence(runtime.event_bus, my_store)`：那样默认的 `InMemoryEventStore`
  仍然是 runtime 内部已经接好的那份，`recover()` / `rebuild_view` 等所有读路径读的
  还是它，你自己接的 store 只会被写、永远不会被读，等于白写一份、事件还多存一遍。

  ```python
  my_store = InMemoryEventStore()  # 或任意自定义 EventStore
  runtime = CtxWeftRuntime(event_store=my_store, snapshot_every_n=50)
  # runtime.persistence 是 attach_persistence() 返回的 PersistenceHandle
  # ...
  await runtime.persistence.detach()
  ```

  `snapshot_every_n > 0` 时会一并接上 `SnapshotWriter`（**默认 0 = 不接**，现有行为
  零变化）。接上之后崩溃恢复从 O(全部事件) 全量回放退化成「最新快照 + 增量」。
  顺序（persister 必须先于 snapshot writer）由 `attach_persistence` 保证，宿主不需要
  也不应该自己分别 subscribe。想在 `CtxWeftRuntime` 之外单独使用事件总线（例如测试里
  只测 store 往返），仍可以直接调 `attach_persistence(bus, store, snapshot_every_n=N)`
  ——但那是脱离 runtime 的独立用法，不要在已经传了 `event_store=` 的 runtime 上重复调。

- **新增 SQL 事件持久化**（需 `ctx-weft[sql]`）：

  ```python
  from ctx_weft.providers.events.store.sql import open_sqlite_event_store

  async with open_sqlite_event_store("app-events.db") as store:
      runtime = CtxWeftRuntime(event_store=store, snapshot_every_n=50)
      # runtime 的 recover() / rebuild_view 等读路径现在读的就是这个 SQL store，
      # 重启后可以从它恢复。
  ```

## 原生发布 wheel（Cython）

日常开发仍使用普通源码 wheel。发布时可将同一份源码编译为原生扩展：

```powershell
uv run --no-sync --with Cython==3.2.4 --with setuptools --with wheel python scripts/build_cython_wheel.py
```

Windows 构建机需安装 MSVC C++ Build Tools 和 Windows SDK。产物位于 `dist/native/`，
例如 `ctx_weft-0.1.0-cp311-cp311-win_amd64.whl`，绑定构建时的 Python ABI 和平台。
编译在 `build/` 临时目录完成，不改写 `src/`；原 wheel 的依赖、extras、Jinja 资源均保留，
Python 模块（包括包初始化模块）替换为原生扩展。原生代码仍可逆向，Jinja 资源仍为明文。

也可以编译指定源码 wheel：

```powershell
uv run --no-sync --with Cython==3.2.4 --with setuptools --with wheel python scripts/build_cython_wheel.py --source-wheel dist/ctx_weft-0.1.0-py3-none-any.whl
```

源码 wheel 携带 `build_ctx_weft_cython.py` 构建脚本，宿主无需访问 core 仓库即可编译。
原生 wheel 中的 `_cython_imports.json` 记录 Python 导入候选，供宿主的 PyInstaller hook
收集编译后无法通过字节码分析发现的依赖。函数签名、类型注解和工具描述 docstring 保留。
固定使用 Cython 3.2.4；3.3.0 会对 `tuple[str, ...]` 中的 `StrEnum` 元素施加额外类型检查，
即使关闭注解类型推断也会改变事件存储的 Python 语义。

## 实例关闭与 prompt 诊断

实例关闭和命令准入由 `CtxWeftRuntime` 直接持有，不通过 lifecycle mixin 继承。
prompt 诊断保留在独立的内部模块，只接收事件存储、记忆、模板查询器和能力索引，
返回诊断数据及同一已提交切面的位置；公开入口由 runtime 包装成 `QueryResult`。

```python
from ctx_weft import RuntimeShutdownError

view = await runtime.inspect_prompt(session_id, prompt_event_id, trim=True)
print(view.data.system_prompt, view.data.messages)

try:
    report = await runtime.close(timeout=30.0)
except RuntimeShutdownError as exc:
    # quiescent=False：清理仍在运行，应继续 await close()，保留外部资源。
    # quiescent=True, durable=False：执行已退出，但恢复事实未全部确认落盘。
    print(exc.report)
    raise
```

`close()` 属于单个 runtime 实例，重复和并发调用共用一次关闭。它拒绝新命令、冻结派发与自治重排、
停止并等待在途执行（包括工具、observe、metadata），确认已开始的提交，丢弃未提交回合，
将剩余已提交工作记录为可恢复中断。已完成的终态和待处理 HITL 的 ID、内容保持不变。
超时或调用方取消不会放弃清理；关闭后的查询仍可读取已提交事件。
外部数据库、MCP、LLM 客户端等 Provider 资源由宿主在 SDK 停止、宿主投影排空后关闭。
`best_effort` 提交策略不能保证恢复持久性，会在有执行会话时报告关闭失败。

`inspect_prompt()` 只接受所属会话的已提交 `LlmPromptSent` 事件 ID，并校验输入版本和 scope。
它使用持久化租户身份读取模板、记忆和能力描述，返回脱敏的公开 DTO，不装填执行缓存、不调用模型或执行工具。
`source="recompiled"` 表示依据保存的输入和当前可用模板/Provider 重编译，不保证等于历史线上请求字节；
`trim=False` 可查看未按预算裁剪的版本。宿主保留 SSE 下标与旧版完整记录的展示兼容。

恢复时原始输入的核对和幂等补写由 SDK 的 `rebuild_session(..., strict=True)` 负责，
包括已有 HITL 回答但缺少原始提问、多模态输入与旧版宿主 `userprompt:<task_id>` 记录。
严格重建遇到记忆存储错误会失败，允许重试，不应在宿主重复实现补写。

## 限制与约束

- **单进程**：EventBus 是进程内实现，不跨进程。多进程需替换为 Redis Streams 等外部总线。
- **MemoryProvider 单实例**：同一个 `ProviderRegistry` 只能注册一个 MemoryProvider。
- **LLM 单解析器**：`register_llm_provider` 是唯一槽位；或用 `llm=` 兜底，二者皆无则运行时报错。
- **core 不做 I/O**：`ctx_weft` 内部无文件读写、无网络请求（providers 是外部的）。
- **默认单租户**：V1 默认 `tenant_id="default"`，多租户需在上层处理。
- **Python ≥ 3.11**。

---

> 更深入的内部实现（循环引擎、Step 流水线、上下文装配、任务编排、崩溃恢复）见 [ARCHITECTURE.md](./ARCHITECTURE.md)。


## 内容处理的公开接口

从 `ctx_weft.protocols.content` 导入 `TextPart`、`ImagePart`、`ContentPart` 及
`parse_content`、`validate_content`、`content_has_image`、`content_to_text`、
`content_to_jsonable`、`externalize_content`。输入解析拒绝未知 part；读取历史的
`normalize_content_parts` 继续容错未知类型。`InvalidContentError` 和
`BlobStoreRequiredError` 同时从该模块与 ctx_weft 顶层公开。

```python
from ctx_weft.protocols import ProviderContext
from ctx_weft.protocols.content import (
    parse_content, validate_content, externalize_content, content_to_jsonable,
)

content = parse_content(raw_input)
store = runtime.providers.get_event_blob_store()
validate_content(content, event_blob_store=store)  # 不写库、不修改输入
prepared = await externalize_content(
    content, blob_store=store, ctx=ProviderContext(session_id="s", tenant_id="tenant"),
)
record_content = content_to_jsonable(prepared)
```

输入已有图片引用时，必须明确传 `source_blob_store`；外部化会读取该来源，必要时
复制到目标存储，不能假设记忆和事件的引用空间相同。展示记录使用准备后的内容，
引擎执行仍传原始 content。外部化成功不代表执行已接受；调用方负责自己的记录
提交及 blob 保留策略。ModelCallConfig 和其 codec 的既有使用方式保持。
