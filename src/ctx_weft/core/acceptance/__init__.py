"""spec: delivery-acceptance——交付验收闭环模块。"""

from ctx_weft.core.acceptance.executor import AcceptanceExecutor
from ctx_weft.core.acceptance.protocol import (
    BUILTIN_STRUCTURE_ID,
    BUILTIN_STRUCTURE_VERSION,
    AcceptanceCheckResult,
    AcceptanceFinding,
    AcceptanceRegistry,
    InvalidAcceptanceSpec,
    acceptance_spec_version,
    normalize_acceptance_spec,
)

__all__ = [
    "AcceptanceExecutor",
    "AcceptanceRegistry",
    "AcceptanceCheckResult",
    "AcceptanceFinding",
    "BUILTIN_STRUCTURE_ID",
    "BUILTIN_STRUCTURE_VERSION",
    "InvalidAcceptanceSpec",
    "acceptance_spec_version",
    "normalize_acceptance_spec",
]
