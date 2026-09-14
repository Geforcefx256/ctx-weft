"""ProviderRegistry：宿主注册进来的四类 provider 的唯一住所。

**为什么单独成模块**：它此前住在 3465 行的 `runtime.py` 里，而 `runtime.py` 是
**装配器**，不是类型的家。后果是一条 import 环——`orchestrator` 的
`skill_executor_capability` / `template_lookup` 都要 `ProviderRegistry` 的类型，
只能 `TYPE_CHECKING` 里反向引 `core.runtime`；registry 这边通知 skill 索引重建时
又要惰性 import 回 orchestrator。两头各绕一次。

搬到这里 + 用 `SkillIndexConsumer` Protocol 取代对具体类的 isinstance 之后，本模块
只依赖 `protocols`，成为真正的叶子，那条环消失。

公共 API 不变：`from ctx_weft import ProviderRegistry` 与
`from ctx_weft.core import ProviderRegistry` 照旧。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ctx_weft.protocols import (
    Authorizer,
    CapabilityProvider,
    KnowledgeProvider,
    LLMClientResolver,
    MemoryProvider,
    SkillCapabilityProvider,
    SkillIndexConsumer,
)

if TYPE_CHECKING:
    from ctx_weft.protocols import MemoryBlobStore
    from ctx_weft.protocols.events import EventBlobStore


class ProviderRegistry:
    """Provider 注册表。

    四种 provider 类型：
      memory     — 唯一；重复注册覆盖
      knowledge  — 有序列表，按 priority 升序（数字小 = 优先级高）
      capability — 有序列表；SkillCapabilityProvider 注册/注销时通知 SkillExecutorCapabilityProvider
      llm        — 唯一；重复注册覆盖
    """

    def __init__(self) -> None:
        self._memory: MemoryProvider | None = None
        self._knowledge: list[tuple[int, KnowledgeProvider]] = []  # (priority, provider)
        self._capabilities: list[CapabilityProvider] = []
        self._capability_authorizers: dict[str, Authorizer] = {}  # provider_name or capability_id → Authorizer
        self._llm_provider: LLMClientResolver | None = None
        self._blob_store: "MemoryBlobStore | None" = None
        self._null_blob_store: "MemoryBlobStore | None" = None
        self._event_blob_store: "EventBlobStore | None" = None
        self._null_event_blob_store: "EventBlobStore | None" = None
        # spec: tool-operations（wp5）——操作账本：显式注册 > 内存默认（惰性单例）
        self._operation_store = None
        self._operation_store_registered = False
        self._in_memory_operation_store = None

    # ── Memory ────────────────────────────────────────────────────────────────

    def register_memory(self, provider: MemoryProvider) -> None:
        self._memory = provider

    def get_memory(self) -> MemoryProvider:
        if self._memory is None:
            raise RuntimeError("MemoryProvider not registered")
        return self._memory

    # ── Knowledge ─────────────────────────────────────────────────────────────

    def register_knowledge(self, provider: KnowledgeProvider, *, priority: int = 0) -> None:
        """注册知识源。priority 升序决定查询顺序（0 最高，数字越小越先查询）。"""
        self._knowledge.append((priority, provider))
        self._knowledge.sort(key=lambda t: t[0])

    def get_knowledge_providers(self) -> list[KnowledgeProvider]:
        return [p for _, p in self._knowledge]

    # ── Capability ────────────────────────────────────────────────────────────

    def register_capability(
        self,
        provider: CapabilityProvider,
        *,
        authorizer: Authorizer | None = None,
        tool_authorizers: dict[str, Authorizer] | None = None,
    ) -> None:
        self._capabilities.append(provider)
        if authorizer is not None:
            self._capability_authorizers[provider.name] = authorizer
        if tool_authorizers:
            self._capability_authorizers.update(tool_authorizers)
        # spec: tool-operations（wp6）——此处曾有 queryable 声明/实现的对齐校验。
        # 裁决能力改由 isinstance(provider, OperationAdjudicator) 发现之后，「声明了却
        # 没实现」这个失败模式不存在了，校验随之删除（且原实现依赖的 `_wp6_caps` 属性
        # 全仓无人设置，那层弱校验本就从未执行过）。recovery_policy 的**取值**校验在
        # resolver 的异步面做——那里能 await provider.list(ctx)。
        if isinstance(provider, SkillCapabilityProvider):
            self._notify_skill_executor_dirty()

    def deregister_capability(self, provider_name: str) -> bool:
        before = len(self._capabilities)
        removed = [p for p in self._capabilities if p.name == provider_name]
        self._capabilities = [p for p in self._capabilities if p.name != provider_name]
        prefix = provider_name + ":"
        for key in [k for k in self._capability_authorizers if k == provider_name or k.startswith(prefix)]:
            del self._capability_authorizers[key]
        if any(isinstance(p, SkillCapabilityProvider) for p in removed):
            self._notify_skill_executor_dirty()
        return len(self._capabilities) < before

    def get_capability_providers(self) -> list[CapabilityProvider]:
        return list(self._capabilities)

    def set_capability_authorizer(self, key: str, authorizer: Authorizer) -> None:
        """注册或覆盖单个 authorizer，key 可以是 provider_name 或完整 capability_id。"""
        self._capability_authorizers[key] = authorizer

    def get_capability_authorizers(self) -> dict[str, Authorizer]:
        """provider_name → Authorizer 映射，供 CapabilityGateway 使用。"""
        return dict(self._capability_authorizers)

    def _notify_skill_executor_dirty(self) -> None:
        """SkillCapabilityProvider 增减时通知持有 skill 索引的 provider 重建。

        判据是 `SkillIndexConsumer` 这个 Protocol，不是具体类：本模块因此完全不必
        知道 `SkillExecutorCapabilityProvider` 的存在。改造前这里对具体类做
        isinstance，而那个类住在 `core.orchestrator`、后者又要本类的类型，只能靠
        函数内惰性 import 绕环——现在环真正没了。
        """
        for p in self._capabilities:
            if isinstance(p, SkillIndexConsumer):
                p.mark_dirty()
                break

    # ── LLM ──────────────────────────────────────────────────────────────────

    def register_llm_provider(self, provider: LLMClientResolver) -> None:
        self._llm_provider = provider

    def get_llm_provider(self) -> LLMClientResolver:
        if self._llm_provider is None:
            raise RuntimeError("LLMProvider not registered")
        return self._llm_provider

    def has_llm_provider(self) -> bool:
        return self._llm_provider is not None

    # ── MemoryBlobStore ──────────────────────────────────────────────────────

    def register_memory_blob_store(self, store: "MemoryBlobStore") -> None:
        """注册二进制内容存储。未注册时 get_memory_blob_store() 返回 NullMemoryBlobStore。"""
        self._blob_store = store

    def get_memory_blob_store(self) -> "MemoryBlobStore":
        """取 memory 侧 blob store。**只有两级：显式注册 > NullMemoryBlobStore。**

        与 `get_event_blob_store()` 完全对称。曾经这里有第三级——「memory provider 自己
        实现了 MemoryBlobStore 且 can_externalize 就用它」——那一级的唯一服务对象是
        `SqlMemoryProvider` 的字节存储；字节已移出 RDBMS（spec 2026-08-29 §5），
        该级无对象可服务，一并删除。

        自动解析删掉之后，「字节放哪」这件事只由接线代码表达，不再藏在解析规则里：
        宿主要 blob 能力就显式 `register_memory_blob_store(FsBlobStore(...))`。
        不注册就是 `NullMemoryBlobStore`，行为与不接 blob 的宿主逐字节一致。

        `NullMemoryBlobStore` 实例只建一次，重复调用返回同一对象。
        """
        if self._blob_store is not None:
            return self._blob_store
        if self._null_blob_store is None:
            from ctx_weft.protocols import NullMemoryBlobStore
            self._null_blob_store = NullMemoryBlobStore()
        return self._null_blob_store

    # ── OperationStore（spec: tool-operations，wp5）───────────────────────────

    def register_operation_store(self, store: "object") -> None:
        """注册工具操作账本。未注册时 get_operation_store() 返回内存默认实现
        （进程内可用；跨进程恢复需宿主显式注入 SQL/持久实现——runtime 起动时据
        registered 标志如实报告能力差异）。"""
        self._operation_store = store
        self._operation_store_registered = True

    def get_operation_store(self):
        """取操作账本。两级：显式注册 > InMemoryOperationStore（惰性单例）。"""
        if self._operation_store is not None:
            return self._operation_store
        if self._in_memory_operation_store is None:
            from ctx_weft.providers.operations import InMemoryOperationStore
            self._in_memory_operation_store = InMemoryOperationStore()
        return self._in_memory_operation_store

    @property
    def operation_store_registered(self) -> bool:
        """宿主是否显式注册了持久账本（False = 内存默认，跨进程恢复能力缺失）。"""
        return self._operation_store_registered

    # ── EventBlobStore ───────────────────────────────────────────────────────

    def register_event_blob_store(self, store: "EventBlobStore") -> None:
        """注册事件流侧的二进制存储。未注册时 get_event_blob_store() 返回 NullEventBlobStore。"""
        self._event_blob_store = store

    def get_event_blob_store(self) -> "EventBlobStore":
        """取 event blob store。**只有两级：显式注册 > NullEventBlobStore。**

        与 `get_memory_blob_store()` 完全对称——两侧都**刻意不**自动解析到 memory
        provider（spec §4；memory 侧那条已删除的中间级见 spec 2026-08-29 §5.3）：
        自动解析会让「共用」成为隐式默认，而双 store 的出发点正是让两者**可分**。
        host 要共用就把同一个实例注册两次——意图写在接线代码里，而不是藏在解析规则里。

        「可分」的实质不在解析规则，而在 ref 命名空间：两个 store 的 ref 是两个
        独立的命名空间，即便 host 把同一个实例注册两次也不改变这一点——core 从不
        比较两侧的 ref，也从不拿一侧的 ref 去另一侧解析。

        `NullEventBlobStore` 实例只建一次，重复调用返回同一对象。
        """
        if self._event_blob_store is not None:
            return self._event_blob_store
        if self._null_event_blob_store is None:
            from ctx_weft.protocols.events import NullEventBlobStore
            self._null_event_blob_store = NullEventBlobStore()
        return self._null_event_blob_store
