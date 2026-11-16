"""纯文本 adapter 的模态降级——两家共用一份（spec 2026-08-28-multimodal-adapter-dispatch §3）。

本包既有的私有辅助模块惯例（`_finalize.py` / `_schema.py`）：两家 adapter 的降级逻辑
逐字相同，各写一份就是两份会分叉的真源。
"""

from __future__ import annotations

import dataclasses
import logging

from ctx_weft.protocols.content import downgrade_images_to_text
from ctx_weft.core.utils.content import image_part_count
from ctx_weft.protocols import LLMMessage, LLMRequest

logger = logging.getLogger(__name__)

__all__ = ["downgrade_for_text_only"]


def downgrade_for_text_only(
    request: LLMRequest, *, adapter_hint: str
) -> list[LLMMessage]:
    """把请求里的图片 part 降级成确定性文本占位；返回新 messages 列表。

    ``adapter_hint`` 是对应多模态子类的类名，只用于 warning 文案——把修复路径直接
    写进日志，而不是让运维去翻文档。

    **不抛异常**：调用方在同步出网主路径上，raise 会掀掉整个 LLM 请求（同
    ``_parts_to_blocks`` 的 getattr 兜底、``rehydrate_content`` 的 get→None 降级、
    ``MemoryBlobStore.get`` 恒不抛）。

    **无图时返回 ``request.messages`` 同一对象、且不记任何日志**——纯文本会话逐字节
    不受影响（本计划 Global Constraints 第一条）。

    占位由 ``core.content.downgrade_images_to_text`` 产出（``[image {media_type}]``），
    逐字节确定，不砸 prompt 前缀缓存；图片判据 ``not hasattr(p, "text")`` 复用
    ``image_part_count``，不在本模块另写一份（spec §13 冻结判据）。
    """
    out: list[LLMMessage] = []
    dropped = 0
    for m in request.messages:
        downgraded = downgrade_images_to_text(m.content)
        if downgraded is not m.content:
            dropped += image_part_count(m.content)
            m = dataclasses.replace(m, content=downgraded)
        out.append(m)
    if not dropped:
        return request.messages
    logger.warning(
        "模型 %s 走的是纯文本 adapter，本次请求的 %d 张图片已降级成文本占位。"
        "若该模型确实支持图片，请改用 %s。",
        request.model, dropped, adapter_hint,
    )
    return out
