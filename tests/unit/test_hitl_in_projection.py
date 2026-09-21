"""HITL 活账进投影：`rebuild_hitl` 不再为它发任何 HITL 查询。

这个判断没有年龄上界——三个月前开出、至今未决的请求今天仍必须被看见。所以它一度只能每次
冷应答把该会话**全部** HITL 事件读回来折一遍：实测 800 次人工确认的会话取回 1600 条、
读放大 1600×、218ms，而交互式会话里每条用户消息都是一次 UserTurn HITL，那个量只增不减。

截尾这条路始终是错的（任何按条数/时间的界都可能踩中那条很久以前开出、至今未决的请求：
`list_pending` 看不见它 ⟹ `parked_task_ids` 少一个 ⟹ 任务在人还没回答时就被重排跑起来）。
进投影**不是**截尾——是让那份账自己销账（`HitlClosed` / `outcome=cancelled`），所以它有界
而不是被截断。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ctx_weft.core.control.reducers import (
    deserialize_view,
    reduce_events,
    serialize_view,
)
from ctx_weft.protocols.events import Event, EventType
from ctx_weft.providers.events import InMemoryEventStore
from tests._event_helpers import append_one

pytestmark = pytest.mark.asyncio

_T0 = datetime(2026, 9, 20, tzinfo=UTC)
_SID = "s_long"


def _ev(n: int, type_: str, **payload) -> Event:
    return Event(
        id=f"evt_{n:08d}", run_id="r1", sequence=n, session_id=_SID, type=type_,
        timestamp=_T0, tenant_id="acme", task_id="t1", agent_id="ag1", payload=payload,
    )


def _opened(n: int, hitl_id: str, **over) -> Event:
    p = {
        "hitl_id": hitl_id, "form": "question",
        "delivery": {"kind": "tool_result", "tool_call_id": f"call_{hitl_id}"},
        "subject_id": "deploy:apply", "prompt": "确认部署？", "detail": "",
        "fields": [{"id": "q1", "text": "选哪个？", "options": ["A", "B"]}],
        "proposal": {"path": "/etc/x"}, "tool_call_id": f"call_{hitl_id}",
        "agent_id": "ag1", "resume_state": {"step": 3}, "reply_as_result": False,
        "stage": "tool", "invocation_key": "iv",
    }
    p.update(over)
    return _ev(n, EventType.HITL_OPENED, **p)


class _CountingStore(InMemoryEventStore):
    """记下读法——断言的是「发了哪些查询」，不只是结果。"""

    def __init__(self) -> None:
        super().__init__()
        self.typed_reads: list[tuple[str, ...]] = []
        #: (after_position, through_position, 是否按类型收窄)
        self.ranges: list[tuple[int, "int | None", bool]] = []

    async def read_range(self, session_id: str, **k):
        self.ranges.append((k.get("after_position", 0), k.get("through_position"),
                            bool(k.get("include_types"))))
        if k.get("include_types"):
            # 「按类型收窄的那种读」现在也是 read_range（2026-09-21 合并），所以这两笔账都记
            # 在这里。分别记是因为断言要区分「有没有按类型收窄」与「有没有无界地读」。
            self.typed_reads.append(tuple(str(t) for t in k["include_types"]))
        return await super().read_range(session_id, **k)

    @property
    def full_reads(self) -> int:
        """**无界且不收窄**的 read_range 次数（after=0、无上界、无类型过滤 = 整条会话）。

        「按类型收窄」的读 2026-09-21 也并进了 `read_range`，而那种读正是这条守卫要的**替代
        品**，不是它要防的东西——所以判据里必须把它排掉，否则一条正确的收窄读会被当成全量读。

        从前这里数的是 `read_by_session` 的调用次数。那个方法 2026-09-21 从协议删了
        （src 零调用者），而「方法不存在时调用 0 次」由语言保证、不需要测试。改数
        `read_range` 里无界的那一形状——那是删掉它之后**仅剩**的整条会话读法。
        """
        return sum(1 for after, through, narrowed in self.ranges
                   if after == 0 and through is None and not narrowed)


async def _seed(store: InMemoryEventStore, n_noise: int) -> None:
    await append_one(store, _ev(0, EventType.SESSION_CREATED, user_prompt="go",
                           template_id="agent:tpl", root_agent_id="ag1"))
    for i in range(1, n_noise + 1):
        await append_one(store, _ev(i, EventType.RUN_FINISHED, outcome="completed"))
    # 一条答了又了结的（该销账）+ 一条仍未决的（必须留着）
    await append_one(store, _opened(n_noise + 1, "h_done"))
    await append_one(store, _ev(n_noise + 2, EventType.HITL_RESOLVED,
                           hitl_id="h_done", outcome="accepted", claimed=False,
                           message="ok"))
    await append_one(store, _ev(n_noise + 3, EventType.HITL_CLOSED, hitl_id="h_done"))
    await append_one(store, _opened(n_noise + 4, "h_open"))


# ── 1. 折进投影，且不发 HITL 查询 ────────────────────────────────────────────


async def test_projection_carries_the_live_hitl_ledger() -> None:
    """重放就能折出活账——不为它单发类型查询。"""
    from ctx_weft.core.control.reducers import rebuild_view

    store = _CountingStore()
    await _seed(store, n_noise=5)

    view = await rebuild_view(store, _SID)

    assert store.typed_reads == [], f"不该为 HITL 单发类型查询，实际 {store.typed_reads}"
    assert set(view.hitl.pending) == {"h_open"}
    assert view.hitl.resolved == {}, "已了结的该销账"
    assert set(view.hitl.opened) == {"h_open"}, "工作账也要跟着销"


async def test_rebuild_hitl_reads_no_hitl_query_at_all() -> None:
    """真实 `rebuild_hitl` 路径：一条 HITL 查询都不发。

    ⚠️ 这条是整件事的**收益本身**。只钉 `full_reads == 0` 不够——按类型收窄的那一版同样
    满足它，而那一版的代价仍随会话长度线性增长。所以这里连类型查询一起钉。
    """
    from tests.integration.test_minimal_loop import (
        InlineAgentTemplateProvider,
        make_echo_template,
        make_runtime,
    )

    resolver = InlineAgentTemplateProvider()
    resolver.register(make_echo_template())
    store = _CountingStore()
    rt = make_runtime(agent_provider=resolver, event_store=store)
    await _seed(store, n_noise=200)

    n = await rt.rebuild_hitl(_SID)

    assert n == 1, f"应装填那条仍未决的，实际 {n}"
    assert store.full_reads == 0, "不该读整条事件流"
    hitl_reads = [t for t in store.typed_reads if any("Hitl" in x for x in t)]
    assert hitl_reads == [], f"HITL 活账在投影里，不该再单发查询，实际 {hitl_reads}"
    assert [r.id for r in rt.hitl_registry.list_pending(session_id=_SID)] == ["h_open"]


# ── 2. blob 往返 ──────────────────────────────────────────────────────────────


def test_pending_survives_the_blob_round_trip_field_by_field() -> None:
    """未决请求进 blob、读回来逐字段一致。

    掉字段不会报错——只会让恢复出来的气泡少了 `fields`（前端空表单）、少了 `resume_state`
    （provider 被要求重做让出前的工作）、或少了 `invocation_key`（决定缓存的第四维失效，
    同 id 的另一次调用会复用旧批准）。所以这条逐字段比，不抽样。
    """
    from ctx_weft.core.hitl.service import delivery_to_payload

    view = reduce_events([_opened(1, "h1")], run_id=_SID)
    back = deserialize_view(serialize_view(view))

    a, b = view.hitl.pending["h1"], back.hitl.pending["h1"]
    for f in ("id", "form", "session_id", "task_id", "agent_id", "tenant_id",
              "subject_id", "prompt", "detail", "fields", "proposal", "tool_call_id",
              "stage", "invocation_key", "resume_state", "reply_as_result",
              "reply_attempt", "legacy_origin", "closed", "created_at", "resolved_at"):
        assert getattr(a, f) == getattr(b, f), f"字段 {f} 没往返回来"
    assert delivery_to_payload(a.delivery) == delivery_to_payload(b.delivery)


def test_resolved_decision_and_cache_survive_the_round_trip() -> None:
    """已终局那半：决定本身 + 决定缓存（三维键）都要回来。

    `decisions_for` 丢了的后果是冷重入查不到决定 → 同一个工具重新求批一遍、同一个问题
    重新问一遍。
    """
    view = reduce_events([
        _opened(1, "h1"),
        _ev(2, EventType.HITL_RESOLVED, hitl_id="h1", outcome="accepted",
            claimed=False, message="go"),
    ], run_id=_SID)
    back = deserialize_view(serialize_view(view))

    assert set(back.hitl.resolved) == {"h1"}
    assert back.hitl.resolved["h1"].decision is not None
    assert back.hitl.resolved["h1"].decision.message == "go"
    key = (_SID, "call_h1", "tool")
    assert key in back.hitl.decisions_for, "决定缓存没重建出来"
    decision, resume_state = back.hitl.decisions_for[key]
    assert decision.outcome == "accepted"
    assert resume_state == {"step": 3}, "resume_state 成对丢了 → provider 要重做让出前的工作"
    assert back.hitl.decision_owner[key] == "h1", "主人账没重建 → 了结会误销别人的决定"


def test_derived_ledgers_are_not_stored_twice() -> None:
    """`opened` / `decisions_for` / `decision_owner` **不写进 blob**——它们可重建。

    存它们是把同一份信息写两遍，而两份会分叉。这条钉住「blob 里只有 pending / resolved」。
    """
    view = reduce_events([
        _opened(1, "h1"),
        _ev(2, EventType.HITL_RESOLVED, hitl_id="h1", outcome="accepted",
            claimed=False, message="go"),
        _opened(3, "h2"),
    ], run_id=_SID)

    blob = serialize_view(view)["hitl"]

    assert set(blob) == {"pending", "resolved"}, f"blob 里多存了东西：{sorted(blob)}"
    # 但读回来派生账必须齐
    back = deserialize_view(serialize_view(view)).hitl
    assert set(back.opened) == {"h1", "h2"}
    assert len(back.decisions_for) == 1


def test_transient_fields_never_reach_the_blob() -> None:
    """`slot` / `pending_decision` / `claimed` 不得进 blob。

    `pending_decision` 是承重的那一条：它对应的 `HitlResolved` 压根没落库，写进 blob 会让
    冷重建以为「人答过了」，而日志说还悬着——正是两阶段要消灭的那种分歧。
    """
    import json

    view = reduce_events([_opened(1, "h1")], run_id=_SID)
    raw = json.dumps(serialize_view(view)["hitl"], ensure_ascii=False)

    for forbidden in ("pending_decision", "pending_event_payload", "slot", "claimed"):
        assert forbidden not in raw, f"{forbidden} 不该进 blob"


def test_old_blob_without_the_key_deserializes_to_an_empty_ledger() -> None:
    """v3 及更早的 blob 无此键 → 空账，不是 KeyError。

    那类 blob 已因 `projection_version` 不匹配被判不可用、走全量重放（全量重放会正确折出
    它），所以这里只钉形态完整性——但它得**是**形态完整：空账而不是炸。
    """
    v = deserialize_view({"session_id": _SID})
    assert v.hitl.pending == {} and v.hitl.resolved == {}


def test_projection_version_was_bumped_for_this_field() -> None:
    """必须 bump：v3 的 blob 没有 `hitl`，拿它当增量基底会让**未决集合凭空为空**。

    后果不是性能退化——是 `parked_task_ids` 为空，于是「人还没答，任务却自己跑起来了」。
    这是这套版本号防的最严重的一种，所以单钉一条。
    """
    from ctx_weft.core.control import reducers

    assert reducers._PROJECTION_VERSION >= 4
