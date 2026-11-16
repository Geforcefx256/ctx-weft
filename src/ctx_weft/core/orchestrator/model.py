"""LLM 选择与解析的三个值对象。`task/` 与 `lifecycle/` 共用的叶子。

**为什么单独成模块**：它们此前住在 `agent_lifecycle_manager.py` 里，只是因为 ALM
负责做解析。但它们跟 agent 生命周期并没有关系——外部消费者是
`core.loop.driver`、`core.loop.llm_gateway`、`core.control.types`、
`core.control.reducers`，没有一个跟 agent 生命周期相关。

具体的坏处是一条包内反向边：`task/runner.py` 的 `AgentBinding.model` 要
`ResolvedModel` 的类型（两阶段派发契约的一部分——assemble 期顺带解出的 LLM 随
binding 交给 execute），于是调度那半边不得不 `TYPE_CHECKING` 反向引生命周期那半边。
提到本模块之后，两个子包各自向下引它，边界真正单向。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Protocol

from ctx_weft.protocols import LLMClient, ModelCallConfig, OutputPolicy
from ctx_weft.protocols.context import ProviderContext

__all__ = ["ModelChoice", "ModelResolver", "ResolvedModel"]


@dataclass(frozen=True)
class ModelChoice:
    """host 要的 `(account, model)`——可以全空，空即「跟随账号默认」。

    这是三样东西里唯一住进 `_AgentRecord` 的一样：解析出的 client 与实际身份
    都不存，派发时从这个 choice 现解（见 `AgentLifecycleManager.resolve_model`）。
    """

    account: str = ""
    model: str = ""
    config: ModelCallConfig = field(default_factory=ModelCallConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.config, ModelCallConfig):
            raise TypeError("config must be a ModelCallConfig")
        copied = deepcopy(self.config)
        copied.validate()
        object.__setattr__(self, "config", copied)


@dataclass(frozen=True)
class ResolvedModel:
    """一次解析的产物：client + 实际身份 + 窗口。三者同源，一次算出。"""

    client: LLMClient
    account: str
    model: str
    context_limit: int
    reserved_output_tokens: int
    output_ceiling: int | None = None
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    config: ModelCallConfig = field(default_factory=ModelCallConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.config, ModelCallConfig):
            raise TypeError("config must be a ModelCallConfig")
        object.__setattr__(self, "config", deepcopy(self.config))
        self.config.validate()
        if type(self.context_limit) is not int or self.context_limit <= 0:
            raise ValueError("context_limit must be a positive integer")
        if (
            type(self.reserved_output_tokens) is not int
            or not 0 <= self.reserved_output_tokens < self.context_limit
        ):
            raise ValueError(
                "reserved_output_tokens must be non-negative and smaller than context_limit"
            )
        if self.output_ceiling is not None and (
            type(self.output_ceiling) is not int
            or not 0 < self.output_ceiling <= self.context_limit
        ):
            raise ValueError("output_ceiling must be positive and no larger than context_limit")
        if not isinstance(self.output_policy, OutputPolicy):
            raise ValueError("output_policy must be an OutputPolicy")

    @property
    def effective_input_limit(self) -> int:
        return self.context_limit - self.reserved_output_tokens

    @classmethod
    def from_client(cls, client: LLMClient, *, account: str = "", model: str = "",
                    config: ModelCallConfig | None = None) -> ResolvedModel:
        """Resolve identity and copy model configuration once for this run."""
        return cls(
            client=client,
            account=account or getattr(client, "account", ""),
            model=model or getattr(client, "model", ""),
            context_limit=client.context_limit,
            reserved_output_tokens=client.output_reserve,
            output_ceiling=getattr(client, "output_ceiling", None),
            output_policy=getattr(client, "output_policy", OutputPolicy()),
            config=config if config is not None else ModelCallConfig(),
        )


class ModelResolver(Protocol):
    def __call__(self, account: str, model: str, *, ctx: ProviderContext) -> LLMClient: ...
