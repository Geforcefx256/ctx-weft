"""ProviderContext + ContentPart 体系。

ProviderContext 在每次 provider 调用时由 core 注入，携带 session/task/agent 标识、
追踪信息、取消令牌。ContentPart 用于多模态内容（text + image）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union


# ── 多模态内容部件 ─────────────────────────────────────────────────────────────


@dataclass
class TextPart:
    """文本片段。"""

    text: str
    type: Literal["text"] = "text"


@dataclass
class ImagePart:
    """图像片段。"""

    data: str  # base64 编码或 URL
    media_type: str  # 'image/png' / 'image/jpeg' / ...
    source_type: Literal["base64", "url", "ref"] = "base64"
    type: Literal["image"] = "image"
    # 原始（解码后）字节数。**可选、默认 None**——既有构造点一律不必改。
    #
    # 存在的理由（Phase 3c Task D）：``image_tokens`` 要按体积估算预算压力，而
    # 外部化之后 ``data`` 是 ``blob:<sha>``（长度恒约 69），与真实体积毫无关系；
    # ``image_tokens`` 是同步函数，不能去 MemoryBlobStore 做 IO 把字节取回来。故让体积
    # 在**还知道的时候**（validate 解码 inline base64 / normalize 外部化拿到 raw
    # bytes 时）被记下来带走，并经 ``content_to_jsonable`` 往返持久化。
    #
    # ``None`` = 不知道（存量记录、不接 blob 的宿主直构、协议外来源）——
    # ``image_tokens`` 对此有明确回落，见该函数。
    byte_size: int | None = None


# blob ref 的前缀。放在这里而不是某个 blob 协议里：它是**内容形态**的一部分——
# `ImagePart.source_type == "ref"` 时 `data` 字段就长这样——而 `ImagePart` 定义在本模块。
# 两个 blob 协议（memory 侧与 event 侧）共用这同一个前缀常量，但这仅是形态上的共用——
# 两侧的 ref 是两个独立的命名空间，各自的 sha 口径互不相干，core 从不比较、也从不拿
# 一侧的 ref 去另一侧解。放在中立的 context 层，两边各自取，谁也不必 import 对方。
BLOB_REF_PREFIX = "blob:"


ContentPart = Union[TextPart, ImagePart]


# ── 边界归一的惰性绑定（Phase 3c Task E2）──────────────────────────────────────

_content_mod: Any = None


def normalize_content_parts(content: Any) -> Any:
    """转调 ``core.utils.content.normalize_content_parts``（dict 形态 part → dataclass）。

    **本函数只是绑定，不是实现**——归一实现全仓唯一一份，在 ``core/content.py``
    （spec §3①：形态转换收在归一层）。这里存在的理由只有两条：

    1. **层序**：protocols 是比 core 低的层，模块级 ``from ctx_weft.core.utils.content
       import ...`` 会把依赖反向（``core/content.py`` 已反向依赖
       ``protocols.filesystem``，今天不成环只是初始化顺序侥幸）。故惰性解析。
    2. **热路径**：三处边界（``LLMMessage`` / ``MemoryRecord`` / ``MemoryEvent``）的
       ``__post_init__`` 无条件调它。把 ``from X import Y`` 直接写在 ``__post_init__``
       里，**每次构造**都要跑一遍 ``__import__`` + ``_handle_fromlist``——实测该语句
       本身就要 361 ns（423.1 ns 的「import + 调用」减去 61.0 ns 的「模块属性 + 调用」），
       把 ``LLMMessage`` 的构造成本从 189 ns 抬到 682 ns。缓存**模块对象**后降到 232 ns。

    缓存的是**模块**而非函数：属性查找留在每次调用时，monkeypatch
    ``core.utils.content.normalize_content_parts`` 仍对本绑定生效（测试可观测性 + 不留
    陈旧绑定的坑）。
    """
    global _content_mod
    if _content_mod is None:
        from ctx_weft.core.utils import content as _m
        _content_mod = _m
    return _content_mod.normalize_content_parts(content)


# ── Citation（KnowledgeDoc 引用信息）────────────────────────────────────────────


@dataclass
class Citation:
    """知识检索结果的引用信息。"""

    url: str | None = None
    title: str | None = None
    anchor: str | None = None  # 段落锚点
    metadata: dict[str, Any] = field(default_factory=dict)


# ── ProviderContext ────────────────────────────────────────────────────────────


@dataclass
class ProviderContext:
    """Provider 调用上下文。core 在每次 provider 操作时构造并注入。

    携带：
    - 当前 session/task/agent 标识 + agent 模板 id（限定操作范围 / 授权维度）
    - tenant_id（多租户预留）
    - trace_id（全链路追踪）
    - invocation_id（capability invoke 时分配，用于 cancel）
    - 时间戳
    """

    session_id: str
    tenant_id: str = "default"
    task_id: str | None = None
    agent_id: str | None = None
    agent_template_id: str = ""  # 该 agent 的模板 id；授权按模板维度做策略（AllowListAuthorizer）
    trace_id: str | None = None
    invocation_id: str | None = None
    request_id: str | None = None
    timestamp: datetime | None = None  # None 时 provider 自填 now
    skill_name: str = ""  # 供 SkillExecutorCapabilityProvider 读取
    extra: dict[str, Any] = field(default_factory=dict)
