"""Account connection settings and per-model API, capabilities and limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from ctx_weft.protocols.llm import OutputPolicy, deserialize_call_config

# ``openai`` is retained for persisted configurations from before the split.
# It must always resolve to the compatible adapter, never to the official SDK.
LLMProtocol = Literal["openai", "openai-compatible", "openai-official", "anthropic"]
SUPPORTED_PROTOCOLS = frozenset({"openai", "openai-compatible", "openai-official", "anthropic"})


@dataclass
class ModelLimits:
    context: int
    output: int | None = None

    def __post_init__(self) -> None:
        if type(self.context) is not int or self.context <= 0:
            raise ValueError("limit.context must be a positive integer")
        if self.output is not None and (type(self.output) is not int or self.output <= 0):
            raise ValueError("limit.output must be a positive integer or None")
        if self.output is not None and self.output > self.context:
            raise ValueError("limit.output must not exceed limit.context")


@dataclass
class ReasoningLevel:
    level_id: str
    settings: dict

    def __post_init__(self) -> None:
        if not isinstance(self.level_id, str) or not self.level_id.strip():
            raise ValueError("reasoning level_id must be non-empty")
        if not isinstance(self.settings, dict):
            raise ValueError("reasoning settings must be a serialized ModelCallConfig")
        deserialize_call_config(self.settings)


@dataclass
class ReasoningCapabilities:
    supported: bool = False
    effort: list[ReasoningLevel] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.supported) is not bool:
            raise ValueError("reasoning.supported must be boolean")
        self.effort = [ReasoningLevel(**v) if isinstance(v, dict) else v for v in self.effort]
        for level in self.effort:
            if not isinstance(level, ReasoningLevel):
                raise ValueError("reasoning effort must contain level descriptors")
            level.__post_init__()
        if len({v.level_id for v in self.effort}) != len(self.effort):
            raise ValueError("Duplicate reasoning level_id")
        if not self.supported and self.effort:
            raise ValueError("Unsupported reasoning cannot declare levels")


@dataclass
class ModelCapabilities:
    input: list[str] = field(default_factory=lambda: ["text"])
    output: list[str] = field(default_factory=lambda: ["text"])
    reasoning: ReasoningCapabilities | None = None

    def __post_init__(self) -> None:
        if not self.input or any(m not in {"text", "image"} for m in self.input):
            raise ValueError("capabilities.input must contain text and/or image")
        if self.output != ["text"]:
            raise ValueError("Only text output is currently supported")
        if isinstance(self.reasoning, dict):
            self.reasoning = ReasoningCapabilities(**self.reasoning)
        if self.reasoning is not None:
            if not isinstance(self.reasoning, ReasoningCapabilities):
                raise ValueError("Invalid reasoning capabilities")
            self.reasoning.__post_init__()


@dataclass
class ModelConfig:
    name: str
    protocol: LLMProtocol
    limit: ModelLimits
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    output_reserve: int | None = None
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Model name must not be empty")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol '{self.protocol}'")
        self.limit.__post_init__()
        self.capabilities.__post_init__()
        if self.output_reserve is not None and (
            type(self.output_reserve) is not int or self.output_reserve < 0
        ):
            raise ValueError("output_reserve must be a non-negative integer or None")
        if self.output_reserve is not None and self.output_reserve >= self.limit.context:
            raise ValueError("output_reserve must be smaller than limit.context")
        if not isinstance(self.output_policy, OutputPolicy):
            raise ValueError("output_policy must be an OutputPolicy")


@dataclass
class LLMAccount:
    name: str
    api_key: str = field(repr=False)
    base_url: str
    models: list[ModelConfig] = field(default_factory=list)
    default_model: str = ""
    timeout_sec: int = 120
    namespace: str = "global"
    resource_id: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.base_url.strip():
            raise ValueError("Account name and base_url must not be empty")
        if type(self.timeout_sec) is not int or self.timeout_sec <= 0:
            raise ValueError("timeout_sec must be a positive integer")
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("Account namespace must be non-empty")
        if not self.resource_id:
            self.resource_id = "llm_" + uuid5(NAMESPACE_URL, repr((self.namespace, self.name))).hex

    @property
    def key(self) -> tuple[str, str]:
        return self.namespace, self.resource_id


class LLMModelNotFoundError(ValueError):
    retriable = False
