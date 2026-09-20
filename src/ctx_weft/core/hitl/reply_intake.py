"""ReplyIntake：应答内容的校验 + 双侧外部化。

**构造期注入，无默认值**——旧实现「未注入 normalizer 时退化为恒等变换」制造了
「单测跑的和生产跑的不是同一个东西」，本类因此不给缺省实现：调用方必须显式提供
一个 normalizer（生产是 Runtime 的三入口共用管线，测试是显式替身）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ctx_weft.core.hitl.registry import PendingHitl
    from ctx_weft.protocols import ContentPart


class ContentNormalizer(Protocol):
    """校验 + 双侧外部化：返回 `(memory 侧内容, event 侧载荷)`。

    两侧各写各的 blob store，两个 ref 不必相同——事件侧载荷必须由**原始**内容算出，
    不能拿 memory 侧的 ref 重算（那份 ref 事件库既无权解读也解不开）。

    `tenant_id` 是 blob 的落点锚点，**由调用方给**——与 `HitlService.open` 同一条纪律
    （「本类自己不持有、也不去解」）。它随请求一路带过来（`PendingHitl.tenant_id`），
    实现方不该拿 `session_id` 去事件日志里反查一个已经在手里的字段。
    """

    async def __call__(
        self, content: "str | list[ContentPart]", session_id: str, tenant_id: str,
    ) -> "tuple[str | list[ContentPart], str | list[dict] | None]": ...


class ReplyIntake:
    """把应答内容过一遍校验与外部化。校验失败**原样抛出**，由调用方保证不推进状态。"""

    def __init__(self, normalizer: ContentNormalizer) -> None:
        self._normalizer = normalizer

    async def normalize(
        self, content: "str | list[ContentPart]", req: "PendingHitl",
    ) -> "tuple[str | list[ContentPart], str | list[dict] | None]":
        """收整个 `PendingHitl` 而非零散字段：blob 的 tenant 锚点就在它身上
        （`PendingHitl.tenant_id`），将来再要别的字段也不必改签名。"""
        return await self._normalizer(content, req.session_id, req.tenant_id)
