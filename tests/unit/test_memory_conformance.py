"""MemoryProvider 协议一致性测试套（Phase 3c Task C1）。

**这是面向协议、不面向实现的测试套。** 每条用例只经 `protocols/memory.py` 声明的
8 个方法（ingest / fold / load_view / recall_topic / recall_semantic /
subscribe_topic / list_subscriptions / describe）与 `protocols/memory_compat.py`
的公开词汇（MemoryKind / MemoryScope）操作 provider，**不碰任何实现内部字段**
（`_events` / `_subscriptions` / `_scope_key` …），也不调非协议的存量兼容方法
（`recall_recent` / `recall_recent_by_agent` / `count_recent` / `supersede`）。

═══ C2 接入点 ═══════════════════════════════════════════════════════════════
新 provider 接进来只需在 `_PROVIDER_FACTORIES` 里**加一行**：

    _PROVIDER_FACTORIES = {
        "in_memory": _make_in_memory,
        "sqlite": _make_sqlite,        # ← Task C2 加这一行 + 一个 6 行的工厂
    }

`memory` fixture 按 `_PROVIDER_FACTORIES` 的键自动参数化，整套用例随即对新
provider 全跑一遍。工厂签名 `(tmp_path) -> AsyncIterator[MemoryProvider]`
（asynccontextmanager），带临时目录是为了让需要落盘的 provider 有地方放文件。
═════════════════════════════════════════════════════════════════════════════

**能力差异一律用探测表达，绝不写 `if provider_name == "sqlite"`**——那种写法在
第三方 provider 接进来时立刻失效。本文件的三个探测：

- `_declared(m)`      → `describe()` 返回的 `MemoryProviderInfo`（supports_semantic /
                        supports_topic 据此分流）

（`_supports_blobs` 探测已于 2026-08-29 删除：字节离开 RDBMS 后，仓内没有任何
`MemoryProvider` 同时是 `MemoryBlobStore`，探测恒 False、无分辨力。blob 存取的
契约测试在 `test_blob_fs_store.py`，引用边在 `test_sql_blob_refs.py`。）

**覆盖不下降说明**：本套是**新增**的协议层覆盖，既有测试一条未删。
`tests/unit/test_memory_record_id.py` / `test_memory_layers.py` /
`test_blackboard.py` 里那些经 `recall_recent` / `supersede` / 直读
`m._subscriptions` 的用例**留在原处**——它们钉的是 in_memory 的**非协议**兼容面
（P4b-2 已把这些方法移出协议），迁进来反而会把实现细节写进 conformance 契约。
本套对同一批**协议行为**另起了等价用例（经 load_view / fold），故是重复覆盖而非替换。
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryEventType,
    MemoryProvider,
    MemoryProviderInfo,
    MemoryRecord,
    MemoryScope,
    ProviderContext,
    Subscription,
    load_records,
)
from ctx_weft.protocols.content import ImagePart, TextPart
from ctx_weft.protocols.memory_compat import MemoryKind
from ctx_weft.providers.memory.in_memory import InMemoryMemoryProvider
from ctx_weft.providers.memory.sql import open_sqlite_memory

# ── Provider 注册表（C2 接入点）────────────────────────────────────────────────


@asynccontextmanager
async def _make_in_memory(tmp_path: Any) -> AsyncIterator[MemoryProvider]:
    yield InMemoryMemoryProvider()


@asynccontextmanager
async def _make_sqlite(tmp_path: Any) -> AsyncIterator[MemoryProvider]:
    """SQLite 落盘的 SqlMemoryProvider（Task C2）。每条用例一个新库。"""
    async with open_sqlite_memory(tmp_path / "memory.db") as provider:
        yield provider


_PROVIDER_FACTORIES = {
    "in_memory": _make_in_memory,
    "sqlite": _make_sqlite,
}


@pytest.fixture(params=sorted(_PROVIDER_FACTORIES))
async def memory(request: pytest.FixtureRequest, tmp_path: Any) -> AsyncIterator[MemoryProvider]:
    """被测 provider。参数化自 `_PROVIDER_FACTORIES`，每个实现跑整套。"""
    async with _PROVIDER_FACTORIES[request.param](tmp_path) as provider:
        yield provider


# ── 能力探测（不得替换为 provider 名字判断）──────────────────────────────────


async def _declared(m: MemoryProvider) -> MemoryProviderInfo:
    """provider 自己声明的能力。分流依据是它，不是类名。"""
    return await m.describe(_ctx())


# ── 固定装置 ──────────────────────────────────────────────────────────────────

_BASE = datetime(2026, 1, 1, tzinfo=UTC)
_SESSION = "s1"
_AGENT = "ag1"
_PNG_B64 = base64.b64encode(bytes(range(64))).decode()


def _ctx(task_id: str | None = None, agent_id: str | None = None) -> ProviderContext:
    return ProviderContext(
        session_id=_SESSION, tenant_id="default", task_id=task_id, agent_id=agent_id
    )


def _addr(task_id: str | None = "t1", agent_id: str | None = _AGENT) -> MemoryAddress:
    return MemoryAddress(session_id=_SESSION, task_id=task_id, agent_id=agent_id)


def _turn(
    content: str | list[Any],
    *,
    t: int = 0,
    task_id: str | None = "t1",
    agent_id: str | None = _AGENT,
    role: str = "user",
    scope: MemoryScope = MemoryScope.TASK,
    kind: MemoryKind = MemoryKind.CONVERSATION_TURN,
    event_id: str | None = None,
    topic: str | None = None,
) -> MemoryEvent:
    """v2-native 事件构造（kind + scope 显式，type=None）。"""
    return MemoryEvent(
        kind=kind,
        scope=scope,
        address=MemoryAddress(session_id=_SESSION, task_id=task_id, agent_id=agent_id),
        content=content,
        timestamp=_BASE + timedelta(seconds=t),
        role=role,  # type: ignore[arg-type]
        id=event_id,
        topic=topic,
    )


def _publication(content: str | list[Any], topic: str, *, t: int = 0) -> MemoryEvent:
    return MemoryEvent(
        kind=MemoryKind.PUBLICATION,
        scope=MemoryScope.SESSION,
        address=MemoryAddress(session_id=_SESSION),
        content=content,
        timestamp=_BASE + timedelta(seconds=t),
        topic=topic,
        metadata={"title": "Report", "outcome": "success"},
    )


def _multimodal() -> list[Any]:
    return [
        TextPart(text="look at this"),
        ImagePart(data=_PNG_B64, media_type="image/png", byte_size=64),
    ]


async def _contents(m: MemoryProvider, **kw: Any) -> list[Any]:
    view = await load_records(m, _addr(), MemoryScope.TASK, _ctx(), **kw)
    return [r.content for r in view]


# ══════════════════════════════════════════════════════════════════════════════
# describe
# ══════════════════════════════════════════════════════════════════════════════


async def test_describe_returns_provider_info(memory: MemoryProvider) -> None:
    info = await memory.describe(_ctx())
    assert isinstance(info, MemoryProviderInfo)
    assert isinstance(info.name, str) and info.name
    assert isinstance(info.supports_semantic, bool)
    assert isinstance(info.supports_topic, bool)
    assert isinstance(info.archives_superseded, bool)


async def test_describe_name_matches_provider_name(memory: MemoryProvider) -> None:
    """协议要求 provider 有 `name` 属性；describe 的 name 必须是同一个身份。"""
    info = await memory.describe(_ctx())
    assert info.name == memory.name


async def test_provider_satisfies_runtime_protocol(memory: MemoryProvider) -> None:
    assert isinstance(memory, MemoryProvider)


# ══════════════════════════════════════════════════════════════════════════════
# ingest —— record-id 契约
# ══════════════════════════════════════════════════════════════════════════════


async def test_ingest_returns_id_and_record_becomes_visible(memory: MemoryProvider) -> None:
    rid = await memory.ingest(_turn("hello"), _ctx())
    assert isinstance(rid, str) and rid
    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert [r.id for r in view] == [rid]
    assert [r.content for r in view] == ["hello"]


async def test_ingest_without_id_generates_unique_ids(memory: MemoryProvider) -> None:
    id1 = await memory.ingest(_turn("one", t=0), _ctx())
    id2 = await memory.ingest(_turn("two", t=1), _ctx())
    assert id1 and id2 and id1 != id2


async def test_ingest_adopts_caller_supplied_id(memory: MemoryProvider) -> None:
    """契约：event.id 给定 → 必须采用并原样回显。"""
    returned = await memory.ingest(_turn("hello", event_id="mem_pre_001"), _ctx())
    assert returned == "mem_pre_001"
    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert [r.id for r in view] == ["mem_pre_001"]


async def test_ingest_same_id_twice_is_noop(memory: MemoryProvider) -> None:
    """按 id 幂等：重复 ingest 不比对内容、不重复写入——原内容保留。"""
    first = await memory.ingest(_turn("hello", event_id="mem_pre_001"), _ctx())
    second = await memory.ingest(_turn("changed content", event_id="mem_pre_001"), _ctx())
    assert first == second == "mem_pre_001"

    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert len(view) == 1
    assert view[0].content == "hello"


async def test_ingest_noop_does_not_advance_seq_counter(memory: MemoryProvider) -> None:
    """契约「不推进任何计数器」。

    seq_no 是可选的实现暴露面（协议未强制回显），故用探测：provider 若在
    metadata 里回显 seq_no，就断言它没被 no-op 推进；不回显则至少断言条数。
    """
    await memory.ingest(_turn("hello", event_id="mem_pre_001"), _ctx())
    await memory.ingest(_turn("dup", event_id="mem_pre_001"), _ctx())
    await memory.ingest(_turn("next", t=1), _ctx())

    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert [r.content for r in view] == ["hello", "next"]
    if all("seq_no" in r.metadata for r in view):
        assert sorted(r.metadata["seq_no"] for r in view) == [1, 2]


async def test_ingest_superseded_id_stays_noop(memory: MemoryProvider) -> None:
    """身份包含已 superseded 的记录：折叠后同 id 重放不得复活或重写。"""
    await memory.ingest(_turn("hello", event_id="mem_pre_001"), _ctx())
    await memory.fold(["mem_pre_001"], [], _ctx())

    returned = await memory.ingest(_turn("hello", event_id="mem_pre_001"), _ctx())
    assert returned == "mem_pre_001"
    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert view == []


# ══════════════════════════════════════════════════════════════════════════════
# fold —— 原子「遗忘 + 补偿」
# ══════════════════════════════════════════════════════════════════════════════


async def test_fold_supersedes_and_writes_replacement(memory: MemoryProvider) -> None:
    a = await memory.ingest(_turn("u", t=0), _ctx())
    b = await memory.ingest(_turn("a1", t=1, role="assistant"), _ctx())
    await memory.ingest(_turn("a2", t=2, role="assistant"), _ctx())

    new_ids = await memory.fold(
        [a, b],
        [_turn("SUMMARY", t=1, role="assistant", kind=MemoryKind.SUMMARY)],
        _ctx(),
    )
    assert len(new_ids) == 1 and isinstance(new_ids[0], str) and new_ids[0]

    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    # 两个效果必须同时可见：旧的没了、补偿的在了（时间正序）
    assert [r.content for r in view] == ["SUMMARY", "a2"]
    assert new_ids[0] in {r.id for r in view}


async def test_fold_with_no_replacements_is_pure_forget(memory: MemoryProvider) -> None:
    a = await memory.ingest(_turn("u", t=0), _ctx())
    await memory.ingest(_turn("keep", t=1), _ctx())

    new_ids = await memory.fold([a], [], _ctx())
    assert new_ids == []
    assert await _contents(memory) == ["keep"]


async def test_fold_skips_unknown_ids(memory: MemoryProvider) -> None:
    """不存在的 id 跳过（幂等），不抛、不影响其它记录。"""
    a = await memory.ingest(_turn("u", t=0), _ctx())
    await memory.fold([a, "no_such_id_at_all"], [], _ctx())
    assert await _contents(memory) == []


async def test_fold_is_idempotent_on_already_superseded(memory: MemoryProvider) -> None:
    """已 superseded 的 id 跳过；replacement 带 id 时按 record-id 契约幂等（重放安全）。"""
    a = await memory.ingest(_turn("u", t=0), _ctx())
    repl = _turn("SUMMARY", t=1, kind=MemoryKind.SUMMARY, role="assistant",
                 event_id="mem_sum_001")

    first = await memory.fold([a], [repl], _ctx())
    second = await memory.fold([a], [repl], _ctx())
    assert first == second == ["mem_sum_001"]
    assert await _contents(memory) == ["SUMMARY"]


async def test_fold_returns_ids_in_replacement_order(memory: MemoryProvider) -> None:
    new_ids = await memory.fold(
        [],
        [
            _turn("s1", t=1, kind=MemoryKind.SUMMARY, role="assistant"),
            _turn("s2", t=2, kind=MemoryKind.SUMMARY, role="assistant"),
        ],
        _ctx(),
    )
    assert len(new_ids) == 2 and len(set(new_ids)) == 2
    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert [r.id for r in view] == new_ids


async def test_fold_does_not_touch_other_scopes(memory: MemoryProvider) -> None:
    """归属分区隔离：折叠 TASK 分区不得波及 AGENT 分区。"""
    a = await memory.ingest(_turn("task-turn", t=0), _ctx())
    await memory.ingest(
        _turn("agent-exp", t=1, task_id=None, scope=MemoryScope.AGENT, role="assistant"),
        _ctx(),
    )
    await memory.fold([a], [], _ctx())

    assert await _contents(memory) == []
    agent_view = await load_records(memory,
        _addr(task_id=None), MemoryScope.AGENT, _ctx())
    assert [r.content for r in agent_view] == ["agent-exp"]


# ══════════════════════════════════════════════════════════════════════════════
# load_view —— 幸存视图 / 时间正序 / 半址过滤
# ══════════════════════════════════════════════════════════════════════════════


async def test_load_view_filters_superseded(memory: MemoryProvider) -> None:
    a = await memory.ingest(_turn("gone", t=0), _ctx())
    await memory.ingest(_turn("stays", t=1), _ctx())
    await memory.fold([a], [], _ctx())
    assert await _contents(memory) == ["stays"]


async def test_load_view_is_chronological_regardless_of_write_order(
    memory: MemoryProvider,
) -> None:
    """排序键是 timestamp 升序——写入顺序不得影响视图顺序。"""
    await memory.ingest(_turn("third", t=30), _ctx())
    await memory.ingest(_turn("first", t=10), _ctx())
    await memory.ingest(_turn("second", t=20), _ctx())
    assert await _contents(memory) == ["first", "second", "third"]


async def test_load_view_ties_broken_by_insertion_order(memory: MemoryProvider) -> None:
    """同 timestamp 时按 seq_no 升序，即写入顺序——「最近一条」取 [-1] 才稳定。"""
    for text in ("one", "two", "three"):
        await memory.ingest(_turn(text, t=5), _ctx())
    assert await _contents(memory) == ["one", "two", "three"]


async def test_load_view_default_kinds_are_turns_and_summaries(
    memory: MemoryProvider,
) -> None:
    await memory.ingest(_turn("turn", t=0), _ctx())
    await memory.ingest(_turn("sum", t=1, kind=MemoryKind.SUMMARY, role="assistant"), _ctx())
    await memory.ingest(_turn("audit", t=2, kind=MemoryKind.TOOL_AUDIT, role="tool"), _ctx())
    assert await _contents(memory) == ["turn", "sum"]  # TOOL_AUDIT 默认不进视图


async def test_load_view_explicit_kinds_include_tool_audit(memory: MemoryProvider) -> None:
    await memory.ingest(_turn("turn", t=0), _ctx())
    await memory.ingest(_turn("audit", t=1, kind=MemoryKind.TOOL_AUDIT, role="tool"), _ctx())
    got = await _contents(memory, kinds=[MemoryKind.TOOL_AUDIT])
    assert got == ["audit"]


async def test_load_view_task_scope_is_isolated_per_task(memory: MemoryProvider) -> None:
    await memory.ingest(_turn("promptA", t=0, task_id="tA"), _ctx())
    await memory.ingest(_turn("promptB", t=1, task_id="tB"), _ctx())

    view = await load_records(memory, _addr(task_id="tA"), MemoryScope.TASK, _ctx())
    assert [r.content for r in view] == ["promptA"]


async def test_load_view_task_scope_aggregates_across_tasks_by_agent(
    memory: MemoryProvider,
) -> None:
    """半址 TASK（task_id=None, agent_id 给定）= 跨 task 聚合。"""
    await memory.ingest(_turn("a", t=0, task_id="tA"), _ctx())
    await memory.ingest(_turn("b", t=1, task_id="tB"), _ctx())
    await memory.ingest(_turn("other", t=2, task_id="tC", agent_id="ag2"), _ctx())

    view = await load_records(memory,
        MemoryAddress(session_id=_SESSION, agent_id=_AGENT), MemoryScope.TASK, _ctx())
    assert [r.content for r in view] == ["a", "b"]


async def test_load_view_agent_scope_accumulates_across_tasks(
    memory: MemoryProvider,
) -> None:
    for i, text in enumerate(("r1", "r2")):
        await memory.ingest(
            _turn(text, t=i, task_id=None, scope=MemoryScope.AGENT, role="assistant"), _ctx())

    view = await load_records(memory,
        MemoryAddress(session_id=_SESSION, agent_id=_AGENT), MemoryScope.AGENT, _ctx())
    assert [r.content for r in view] == ["r1", "r2"]


async def test_load_view_session_scope_sees_only_session_partition(
    memory: MemoryProvider,
) -> None:
    await memory.ingest(_turn("task-turn", t=0), _ctx())
    await memory.ingest(
        MemoryEvent(
            kind=MemoryKind.SUMMARY,
            scope=MemoryScope.SESSION,
            address=MemoryAddress(session_id=_SESSION),
            content="session-note",
            timestamp=_BASE + timedelta(seconds=1),
        ),
        _ctx(),
    )
    view = await load_records(memory,
        MemoryAddress(session_id=_SESSION), MemoryScope.SESSION, _ctx())
    assert [r.content for r in view] == ["session-note"]


async def test_load_view_isolates_sessions(memory: MemoryProvider) -> None:
    await memory.ingest(_turn("mine", t=0), _ctx())
    await memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN, scope=MemoryScope.TASK,
            address=MemoryAddress(session_id="other", task_id="t1", agent_id=_AGENT),
            content="theirs", timestamp=_BASE, role="user",
        ),
        ProviderContext(session_id="other", tenant_id="default"),
    )
    assert await _contents(memory) == ["mine"]


@pytest.mark.parametrize(
    ("address", "scope"),
    [
        # TASK 半址：task_id / agent_id 至少其一
        (MemoryAddress(session_id=_SESSION), MemoryScope.TASK),
        # AGENT 半址：agent_id 必给
        (MemoryAddress(session_id=_SESSION), MemoryScope.AGENT),
        # AGENT 半址：禁带 task_id
        (MemoryAddress(session_id=_SESSION, task_id="t1", agent_id=_AGENT), MemoryScope.AGENT),
        # SESSION 半址：禁带 task_id / agent_id
        (MemoryAddress(session_id=_SESSION, task_id="t1"), MemoryScope.SESSION),
        (MemoryAddress(session_id=_SESSION, agent_id=_AGENT), MemoryScope.SESSION),
    ],
)
async def test_load_view_rejects_illegal_half_address(
    memory: MemoryProvider, address: MemoryAddress, scope: MemoryScope
) -> None:
    """非法非 None 字段必须 loud ValueError——静默漏召回是最难查的一类缺陷。"""
    with pytest.raises(ValueError):
        await load_records(memory, address, scope, _ctx())


async def test_load_view_records_carry_kind_scope_and_address(
    memory: MemoryProvider,
) -> None:
    await memory.ingest(_turn("hi", t=0), _ctx())
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert isinstance(rec, MemoryRecord)
    assert rec.kind is MemoryKind.CONVERSATION_TURN
    assert rec.scope is MemoryScope.TASK
    assert rec.address is not None
    assert (rec.address.session_id, rec.address.task_id, rec.address.agent_id) == (
        _SESSION, "t1", _AGENT)
    assert rec.role == "user"
    assert rec.timestamp == _BASE


async def test_load_view_on_empty_partition_returns_empty_list(
    memory: MemoryProvider,
) -> None:
    assert await load_records(memory, _addr(task_id="nope"), MemoryScope.TASK, _ctx()) == []


async def test_load_view_normalizes_legacy_typed_events(memory: MemoryProvider) -> None:
    """legacy `type` 构造的事件也必须出现在 v2 视图里（读侧归一，零数据迁移）。"""
    await memory.ingest(
        MemoryEvent(
            type=MemoryEventType.USER_PROMPT,
            address=_addr(),
            content="legacy prompt",
            timestamp=_BASE,
            role="user",
        ),
        _ctx(),
    )
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert rec.content == "legacy prompt"
    assert rec.kind is MemoryKind.CONVERSATION_TURN
    assert rec.scope is MemoryScope.TASK


# ══════════════════════════════════════════════════════════════════════════════
# recall_topic / subscribe_topic / list_subscriptions
# ══════════════════════════════════════════════════════════════════════════════


async def _skip_unless_topic(m: MemoryProvider) -> None:
    if not (await _declared(m)).supports_topic:
        pytest.skip("provider declares supports_topic=False")


async def test_recall_topic_returns_records_and_cursor(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    await memory.ingest(_publication("v1", "A", t=0), _ctx())

    recs, cursor = await memory.recall_topic("A", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["v1"]
    assert isinstance(cursor, int) and cursor > 0


async def test_recall_topic_since_cursor_excludes_already_read(
    memory: MemoryProvider,
) -> None:
    await _skip_unless_topic(memory)
    await memory.ingest(_turn("e1", t=0, kind=MemoryKind.SUMMARY, topic="log",
                              role="assistant"), _ctx())
    _, cursor = await memory.recall_topic("log", since=0, ctx=_ctx())
    await memory.ingest(_turn("e2", t=1, kind=MemoryKind.SUMMARY, topic="log",
                              role="assistant"), _ctx())

    recs, new_cursor = await memory.recall_topic("log", since=cursor, ctx=_ctx())
    assert [r.content for r in recs] == ["e2"]
    assert new_cursor > cursor


async def test_recall_topic_empty_keeps_cursor(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    recs, cursor = await memory.recall_topic("nothing_here", since=7, ctx=_ctx())
    assert recs == []
    assert cursor == 7


async def test_recall_topic_topics_are_independent(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    await memory.ingest(_publication("a-latest", "A", t=0), _ctx())
    await memory.ingest(_publication("b-latest", "B", t=1), _ctx())

    a_recs, _ = await memory.recall_topic("A", since=0, ctx=_ctx())
    b_recs, _ = await memory.recall_topic("B", since=0, ctx=_ctx())
    assert [r.content for r in a_recs] == ["a-latest"]
    assert [r.content for r in b_recs] == ["b-latest"]


async def test_publication_overwrites_same_topic(memory: MemoryProvider) -> None:
    """黑板覆盖语义：同 topic 的 PUBLICATION 只保留最新一条。

    注：协议 docstring 未写这条（见 C1 报告的「简报/协议缺口」一节），但 core 侧
    `BlackboardSource` 依赖它——provider 之间若不一致，黑板就换了语义。
    故先在 conformance 里钉住。
    """
    await _skip_unless_topic(memory)
    for i, text in enumerate(("v1", "v2", "v3")):
        await memory.ingest(_publication(text, "A", t=i), _ctx())

    recs, _ = await memory.recall_topic("A", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["v3"]
    assert recs[0].metadata.get("title") == "Report"


async def test_recall_topic_excludes_superseded(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    rid = await memory.ingest(_turn("e1", t=0, kind=MemoryKind.SUMMARY, topic="log",
                                    role="assistant"), _ctx())
    await memory.fold([rid], [], _ctx())
    recs, _ = await memory.recall_topic("log", since=0, ctx=_ctx())
    assert recs == []


async def test_non_publication_topic_events_accumulate(memory: MemoryProvider) -> None:
    """只有 PUBLICATION 触发覆盖；其它带 topic 的记录应累积保留。"""
    await _skip_unless_topic(memory)
    for i, text in enumerate(("entry1", "entry2")):
        await memory.ingest(
            _turn(text, t=i, kind=MemoryKind.SUMMARY, topic="proj_log", role="assistant"),
            _ctx(),
        )
    recs, _ = await memory.recall_topic("proj_log", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["entry1", "entry2"]


async def test_publication_overwrite_spares_non_publication_rows(
    memory: MemoryProvider,
) -> None:
    """覆盖只作用于 PUBLICATION 行，**同 topic 的其它 kind 必须幸存**。

    单写「非 PUBLICATION 会累积」是不够的——那条 topic 上从没发生过覆盖，覆盖分支
    根本没执行（实测：把覆盖条件放宽到「同 topic 任意行」时该断言仍绿）。必须让
    覆盖**真的发生一次**再看旁边的行还在不在。
    """
    await _skip_unless_topic(memory)
    await memory.ingest(
        _turn("audit-note", t=0, kind=MemoryKind.SUMMARY, topic="A", role="assistant"), _ctx())
    await memory.ingest(_publication("v1", "A", t=1), _ctx())
    await memory.ingest(_publication("v2", "A", t=2), _ctx())

    recs, _ = await memory.recall_topic("A", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["audit-note", "v2"]


async def test_subscribe_topic_returns_id_and_lists(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    sid = await memory.subscribe_topic(
        _SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    assert isinstance(sid, str) and sid

    subs = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="B")
    assert len(subs) == 1
    assert isinstance(subs[0], Subscription)
    assert (subs[0].session_id, subs[0].topic, subs[0].task_id, subs[0].intent) == (
        _SESSION, "A", "B", "subtask")
    assert subs[0].cursor == 0


async def test_subscriptions_are_task_scoped(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(_SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    await memory.subscribe_topic(_SESSION, topic="X", intent="subtask", ctx=_ctx(), task_id="C")

    b = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="B")
    c = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="C")
    assert sorted(s.topic for s in b) == ["A"]
    assert sorted(s.topic for s in c) == ["X"]


async def test_session_level_subscription_visible_to_every_task(
    memory: MemoryProvider,
) -> None:
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(
        _SESSION, topic="G", intent="long_term_background", ctx=_ctx(), task_id="")
    await memory.subscribe_topic(_SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")

    b = {s.topic for s in await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="B")}
    c = {s.topic for s in await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="C")}
    assert b == {"G", "A"}
    assert c == {"G"}


async def test_subscribe_topic_is_idempotent(memory: MemoryProvider) -> None:
    """同 (session, task, topic) 重复订阅不得产生第二条、不得重置游标。

    **协议面没有「推进游标」的方法**（见 C1 报告的协议缺口一节），所以「保留游标」
    只能在 provider 恰好把可写的 Subscription 暴露出来时才可观测。这里用**探测**：
    先改一次 `list_subscriptions` 返回的对象，再 re-list 看改动是否可见；不可见就
    跳过后半段（对 JSON/SQL 型 provider 是常态），可见则要求 re-subscribe 后游标仍在。
    只断言「恰一条」是不够的——重复订阅整条覆盖时条数同样是 1（实测该变异会存活）。
    """
    first = await memory.subscribe_topic(
        _SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    second = await memory.subscribe_topic(
        _SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    assert first == second

    subs = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="B")
    assert len(subs) == 1
    assert subs[0].cursor == 0

    subs[0].cursor = 7
    if [s.cursor for s in await memory.list_subscriptions(
            _SESSION, ctx=_ctx(), task_id="B")] != [7]:
        pytest.skip(
            "provider does not expose a writable cursor through list_subscriptions; "
            "cursor preservation is not observable through the protocol surface")

    await memory.subscribe_topic(
        _SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    again = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="B")
    assert len(again) == 1
    assert again[0].cursor == 7, "重复订阅必须保留游标，不得重置"


async def test_list_subscriptions_none_returns_all(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(_SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    await memory.subscribe_topic(_SESSION, topic="X", intent="subtask", ctx=_ctx(), task_id="C")
    allsubs = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id=None)
    assert sorted(s.topic for s in allsubs) == ["A", "X"]


async def test_list_subscriptions_isolates_sessions(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(_SESSION, topic="A", intent="subtask", ctx=_ctx(), task_id="B")
    other = await memory.list_subscriptions(
        "other_session", ctx=ProviderContext(session_id="other_session"), task_id=None)
    assert other == []


# ══════════════════════════════════════════════════════════════════════════════
# recall_semantic —— 可选能力，按 describe() 分流
# ══════════════════════════════════════════════════════════════════════════════


async def test_recall_semantic_matches_declared_capability(memory: MemoryProvider) -> None:
    await memory.ingest(_turn("the quick brown fox", t=0), _ctx())
    info = await _declared(memory)
    got = await memory.recall_semantic("fox", _addr(), 5, _ctx())
    assert isinstance(got, list)
    if not info.supports_semantic:
        assert got == [], "supports_semantic=False 的 provider 必须返空（core 默认行为）"
    else:
        assert all(isinstance(r, MemoryRecord) for r in got)
        assert len(got) <= 5, "必须尊重 top_k"
        # 声明 supports_semantic=True 就必须真的能召回：唯一一条记录里逐字含
        # 查询词，此时返空只能是「声明了却没实现」。不加这条断言的话
        # 「describe 谎报 supports_semantic」这个变异会存活（实测）。
        assert got, "supports_semantic=True 的 provider 必须能召回逐字命中的记录"


# ══════════════════════════════════════════════════════════════════════════════
# 多模态无损存取契约（协议 docstring 四条）
# ══════════════════════════════════════════════════════════════════════════════


def _assert_lossless(content: Any) -> None:
    """契约 1/2/4：原样出库、不得在持久化层拍扁成纯文本。

    先 `isinstance(content, list)`——`hasattr(p, "text")` 在 `str` 上是逐字符
    恒真的重言式（台账陷阱 3），不先守住类型的话整段断言是假阳性。
    """
    assert not isinstance(content, str), "契约 4：持久化层禁止拍扁成纯文本"
    assert isinstance(content, list)
    assert len(content) == 2
    text_part, image_part = content
    assert isinstance(text_part, TextPart)
    assert text_part.text == "look at this"
    assert isinstance(image_part, ImagePart)
    assert image_part.data == _PNG_B64
    assert image_part.media_type == "image/png"
    assert image_part.source_type == "base64"


async def test_load_view_returns_content_parts_losslessly(memory: MemoryProvider) -> None:
    await memory.ingest(_turn(_multimodal(), t=0), _ctx())
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    _assert_lossless(rec.content)


async def test_str_content_stays_str(memory: MemoryProvider) -> None:
    """契约 2 的另一半：str 进 str 出，纯文本路径不得被包装成 parts。"""
    await memory.ingest(_turn("plain text", t=0), _ctx())
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert isinstance(rec.content, str)
    assert rec.content == "plain text"


async def test_empty_string_content_survives(memory: MemoryProvider) -> None:
    """空串是合法内容（占位回合），不得被当成缺失而丢弃或转成 None。"""
    await memory.ingest(_turn("", t=0), _ctx())
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert rec.content == ""


async def test_image_byte_size_survives_roundtrip(memory: MemoryProvider) -> None:
    """Task D 的 `ImagePart.byte_size` 必须往返——丢了预算就对大图失明。"""
    await memory.ingest(_turn(_multimodal(), t=0), _ctx())
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert isinstance(rec.content, list)
    assert rec.content[1].byte_size == 64


async def test_fold_replacement_preserves_content_parts(memory: MemoryProvider) -> None:
    await memory.fold(
        [],
        [_turn(_multimodal(), t=0, kind=MemoryKind.SUMMARY, role="assistant")],
        _ctx(),
    )
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    _assert_lossless(rec.content)


async def test_recall_topic_preserves_content_parts(memory: MemoryProvider) -> None:
    await _skip_unless_topic(memory)
    await memory.ingest(_publication(_multimodal(), "A", t=0), _ctx())
    recs, _ = await memory.recall_topic("A", since=0, ctx=_ctx())
    assert len(recs) == 1
    _assert_lossless(recs[0].content)


async def test_ref_shaped_image_is_not_rewritten(memory: MemoryProvider) -> None:
    """外部化后的 ref 形态必须原样存回——provider 不得替调用方 rehydrate 或改写。"""
    await memory.ingest(
        _turn([ImagePart(data="blob:deadbeef", media_type="image/jpeg",
                         source_type="ref", byte_size=4096)], t=0),
        _ctx(),
    )
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert isinstance(rec.content, list)
    part = rec.content[0]
    assert isinstance(part, ImagePart)
    assert (part.data, part.media_type, part.source_type, part.byte_size) == (
        "blob:deadbeef", "image/jpeg", "ref", 4096)


async def test_blob_refs_roundtrip_losslessly(memory: MemoryProvider) -> None:
    """`MemoryEvent.blob_refs`（声明式 ref，L0.5 降级补偿记录用）必须无损往返。

    content 故意用纯文本占位、不含结构化 ref part——SQL provider 的 `_declared_refs`
    只回显「声明的、content 里看不见的」那部分（相减语义），若 content 里也有同一个
    结构化 ref，SQL 侧会回显空列表而 in_memory 侧回显非空，两个 provider 就没有
    一致的期望值了。这里避开那个歧义区，纯测「声明未被结构化 part 覆盖」这条主路径。
    """
    await memory.ingest(
        MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.TASK,
            address=_addr(),
            content="[image demoted]",
            timestamp=_BASE,
            role="user",
            id="mem_demoted_001",
            blob_refs=["blob:aaaa", "blob:bbbb"],
        ),
        _ctx(),
    )
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert sorted(rec.blob_refs) == ["blob:aaaa", "blob:bbbb"]


async def test_dict_shaped_parts_come_back_as_dataclasses(memory: MemoryProvider) -> None:
    """dict 形态在 `MemoryEvent.__post_init__` 即被归一（Task E2）；provider 不得
    把它「还原」成 dict——读侧拿到的必须仍是 dataclass。"""
    await memory.ingest(
        _turn([{"type": "text", "text": "look at this"},
               {"type": "image", "source_type": "base64",
                "data": _PNG_B64, "media_type": "image/png", "byte_size": 64}], t=0),
        _ctx(),
    )
    (rec,) = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    _assert_lossless(rec.content)


# ══════════════════════════════════════════════════════════════════════════════
# 跨租户隔离契约（Phase 3c Task C1b）
# ══════════════════════════════════════════════════════════════════════════════
# 已实证的泄漏：同 session_id、不同 tenant 时 tenantB 的 load_view 读到了
# ['TENANT-A-SECRET']。`start_session` 支持宿主自带 session_id，故 session_id
# 撞车不是理论问题。
#
# 本节每条**可见性**断言都正向断言「本租户读到了什么」——只断言「没读到别人的」
# 对「修过头 → 返空」完全不敏感（台账 Task D M9 / C1 M18 两次实录）。

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_SECRET = "TENANT-A-SECRET"
# 归一等价类：显式 "default" / 空串 / None 必须落同一分区（既有数据与既有调用点
# 全都走 `ProviderContext.tenant_id` 的默认值 "default"，归一取向由此确定）。
_DEFAULT_EQUIVALENTS: list[Any] = ["default", "", None]


def _tctx(
    tenant: Any,
    task_id: str | None = None,
    agent_id: str | None = None,
) -> ProviderContext:
    """同 session_id、不同 tenant 的调用上下文。

    tenant 允许传 None：`ProviderContext.tenant_id` 声明是 `str`，但宿主实际会
    塞 None / 空串——归一规则正是为这种串接不一致存在的。
    """
    return ProviderContext(
        session_id=_SESSION, tenant_id=tenant, task_id=task_id, agent_id=agent_id
    )


async def test_load_view_isolates_tenants(memory: MemoryProvider) -> None:
    """同 session_id、不同 tenant → 互相看不见。直接钉住实证的那条泄漏。"""
    await memory.ingest(_turn(_SECRET, t=0), _tctx(_TENANT_A))
    await memory.ingest(_turn("b-own", t=1), _tctx(_TENANT_B))

    view_a = await load_records(memory, _addr(), MemoryScope.TASK, _tctx(_TENANT_A))
    view_b = await load_records(memory, _addr(), MemoryScope.TASK, _tctx(_TENANT_B))
    assert [r.content for r in view_a] == [_SECRET]
    assert [r.content for r in view_b] == ["b-own"], (
        "B 必须读到**自己的**那条——只断言 B 读不到 A 的话，返空也能通过"
    )


async def test_load_view_isolates_tenants_in_agent_scope(memory: MemoryProvider) -> None:
    await memory.ingest(
        _turn(_SECRET, t=0, task_id=None, scope=MemoryScope.AGENT, role="assistant"),
        _tctx(_TENANT_A),
    )
    await memory.ingest(
        _turn("b-own", t=1, task_id=None, scope=MemoryScope.AGENT, role="assistant"),
        _tctx(_TENANT_B),
    )
    agent_addr = MemoryAddress(session_id=_SESSION, agent_id=_AGENT)

    view_a = await load_records(memory, agent_addr, MemoryScope.AGENT, _tctx(_TENANT_A))
    view_b = await load_records(memory, agent_addr, MemoryScope.AGENT, _tctx(_TENANT_B))
    assert [r.content for r in view_a] == [_SECRET]
    assert [r.content for r in view_b] == ["b-own"]


async def test_load_view_isolates_tenants_in_session_scope(memory: MemoryProvider) -> None:
    """SESSION 分区是泄漏面最大的一层：半址只有 session_id，撞车即全见。"""
    for tenant, text, t in ((_TENANT_A, _SECRET, 0), (_TENANT_B, "b-own", 1)):
        await memory.ingest(
            MemoryEvent(
                kind=MemoryKind.SUMMARY,
                scope=MemoryScope.SESSION,
                address=MemoryAddress(session_id=_SESSION),
                content=text,
                timestamp=_BASE + timedelta(seconds=t),
            ),
            _tctx(tenant),
        )
    session_addr = MemoryAddress(session_id=_SESSION)

    view_a = await load_records(memory, session_addr, MemoryScope.SESSION, _tctx(_TENANT_A))
    view_b = await load_records(memory, session_addr, MemoryScope.SESSION, _tctx(_TENANT_B))
    assert [r.content for r in view_a] == [_SECRET]
    assert [r.content for r in view_b] == ["b-own"]


async def test_load_view_same_tenant_still_sees_all_its_own_rows(
    memory: MemoryProvider,
) -> None:
    """防「修过头」：同一 tenant 字符串、不同 ProviderContext 对象 → 全量可见。

    隔离必须按 tenant **值**分区，不能退化成按调用上下文对象分区。
    """
    await memory.ingest(_turn("first", t=0), _tctx(_TENANT_A))
    await memory.ingest(_turn("second", t=1), _tctx(_TENANT_A, task_id="t1"))

    view = await load_records(memory, _addr(), MemoryScope.TASK, _tctx(_TENANT_A))
    assert [r.content for r in view] == ["first", "second"]


@pytest.mark.parametrize("read_tenant", _DEFAULT_EQUIVALENTS)
@pytest.mark.parametrize("write_tenant", _DEFAULT_EQUIVALENTS)
async def test_default_tenant_equivalents_are_one_partition(
    memory: MemoryProvider, write_tenant: Any, read_tenant: Any
) -> None:
    """归一规则：`"default"` / `""` / `None` 是同一个租户分区。

    这条比隔离本身更要紧——租户串接在写读两侧不一致时，严格比较会让 load_view
    **静默返空**（表现是「会话突然失忆」），比泄漏更难诊断。
    """
    await memory.ingest(_turn("row", t=0), _tctx(write_tenant))
    view = await load_records(memory, _addr(), MemoryScope.TASK, _tctx(read_tenant))
    assert [r.content for r in view] == ["row"]


async def test_omitted_tenant_field_matches_explicit_default(
    memory: MemoryProvider,
) -> None:
    """「缺失」= 根本不传 tenant_id（用 ProviderContext 的字段默认值）。

    仓内绝大多数构造点就是这个形态，存量数据同理——它必须与显式 "default" 等价。
    """
    await memory.ingest(_turn("legacy-row", t=0), ProviderContext(session_id=_SESSION))
    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert [r.content for r in view] == ["legacy-row"]


async def test_named_tenant_and_default_tenant_are_isolated(
    memory: MemoryProvider,
) -> None:
    """归一不得把具名租户也吞进默认分区。"""
    await memory.ingest(_turn("default-row", t=0), _ctx())
    await memory.ingest(_turn("named-row", t=1), _tctx(_TENANT_A))

    default_view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    named_view = await load_records(memory, _addr(), MemoryScope.TASK, _tctx(_TENANT_A))
    assert [r.content for r in default_view] == ["default-row"]
    assert [r.content for r in named_view] == ["named-row"]


async def test_single_tenant_path_behaviour_is_unchanged(memory: MemoryProvider) -> None:
    """既有路径（写读全程默认 tenant）行为不变：id、内容、顺序逐条对上。"""
    ids = [
        await memory.ingest(_turn(text, t=i), _ctx())
        for i, text in enumerate(("one", "two", "three"))
    ]
    view = await load_records(memory, _addr(), MemoryScope.TASK, _ctx())
    assert [r.id for r in view] == ids
    assert [r.content for r in view] == ["one", "two", "three"]


async def test_recall_topic_isolates_tenants(memory: MemoryProvider) -> None:
    """topic 名同样可能跨租户撞车（宿主自定的 long_term_* topic 是常态）。"""
    await _skip_unless_topic(memory)
    await memory.ingest(_publication(_SECRET, "A", t=0), _tctx(_TENANT_A))
    await memory.ingest(_publication("b-own", "A", t=1), _tctx(_TENANT_B))

    recs_a, _ = await memory.recall_topic("A", since=0, ctx=_tctx(_TENANT_A))
    recs_b, _ = await memory.recall_topic("A", since=0, ctx=_tctx(_TENANT_B))
    assert [r.content for r in recs_a] == [_SECRET]
    assert [r.content for r in recs_b] == ["b-own"]


async def test_publication_overwrite_does_not_reach_other_tenants(
    memory: MemoryProvider,
) -> None:
    """覆盖语义按租户分区：B 的发布不得把 A 的黑板条目标成 superseded。

    读侧隔离若只加在读上，B 一发布就把 A 的行覆盖掉——A 从此读到空，而它连
    B 的行都看不见。B 连发两条保证覆盖分支**真的执行过一次**。
    """
    await _skip_unless_topic(memory)
    await memory.ingest(_publication("a-v1", "A", t=0), _tctx(_TENANT_A))
    await memory.ingest(_publication("b-v1", "A", t=1), _tctx(_TENANT_B))
    await memory.ingest(_publication("b-v2", "A", t=2), _tctx(_TENANT_B))

    recs_a, _ = await memory.recall_topic("A", since=0, ctx=_tctx(_TENANT_A))
    recs_b, _ = await memory.recall_topic("A", since=0, ctx=_tctx(_TENANT_B))
    assert [r.content for r in recs_a] == ["a-v1"]
    assert [r.content for r in recs_b] == ["b-v2"]


async def test_recall_topic_default_tenant_equivalents_share_one_partition(
    memory: MemoryProvider,
) -> None:
    await _skip_unless_topic(memory)
    await memory.ingest(_publication("published", "A", t=0), _tctx(""))
    recs, _ = await memory.recall_topic("A", since=0, ctx=_ctx())
    assert [r.content for r in recs] == ["published"]


async def test_recall_semantic_does_not_leak_across_tenants(
    memory: MemoryProvider,
) -> None:
    await memory.ingest(_turn(_SECRET, t=0), _tctx(_TENANT_A))
    info = await _declared(memory)

    got = await memory.recall_semantic(_SECRET, _addr(), 5, _tctx(_TENANT_B))
    assert isinstance(got, list)
    assert [r for r in got if _SECRET in str(r.content)] == [], (
        "另一个 tenant 不得经语义召回读到本租户内容"
    )
    if not info.supports_semantic:
        assert got == [], "supports_semantic=False 的 provider 必须返空（core 默认行为）"
    else:
        own = await memory.recall_semantic(_SECRET, _addr(), 5, _tctx(_TENANT_A))
        assert [r for r in own if _SECRET in str(r.content)], (
            "本租户必须仍能召回自己的记录——否则是修过头把视图清空了"
        )


async def test_subscriptions_are_isolated_per_tenant(memory: MemoryProvider) -> None:
    """订阅表同样按 (tenant, session, task) 分区——topic 名与 intent 也是租户数据。"""
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(
        _SESSION, topic="a-topic", intent="subtask", ctx=_tctx(_TENANT_A), task_id="B")
    await memory.subscribe_topic(
        _SESSION, topic="b-topic", intent="subtask", ctx=_tctx(_TENANT_B), task_id="B")

    subs_a = await memory.list_subscriptions(_SESSION, ctx=_tctx(_TENANT_A), task_id="B")
    subs_b = await memory.list_subscriptions(_SESSION, ctx=_tctx(_TENANT_B), task_id="B")
    assert {s.topic for s in subs_a} == {"a-topic"}
    assert {s.topic for s in subs_b} == {"b-topic"}


async def test_same_topic_subscribed_by_two_tenants_stays_separate(
    memory: MemoryProvider,
) -> None:
    """同 (session, task, topic) 在两个 tenant 下必须是两条独立订阅。

    否则后订阅方撞上幂等分支，拿到的是**别人的**订阅（含别人的游标）。
    用 intent 当判别器：共用一条时 B 会读到 A 的 "subtask"。
    """
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(
        _SESSION, topic="shared", intent="subtask", ctx=_tctx(_TENANT_A), task_id="B")
    await memory.subscribe_topic(
        _SESSION, topic="shared", intent="long_term_background",
        ctx=_tctx(_TENANT_B), task_id="B")

    subs_a = await memory.list_subscriptions(_SESSION, ctx=_tctx(_TENANT_A), task_id="B")
    subs_b = await memory.list_subscriptions(_SESSION, ctx=_tctx(_TENANT_B), task_id="B")
    assert [(s.topic, s.intent) for s in subs_a] == [("shared", "subtask")]
    assert [(s.topic, s.intent) for s in subs_b] == [("shared", "long_term_background")]


async def test_subscriptions_default_tenant_equivalents_share_one_partition(
    memory: MemoryProvider,
) -> None:
    await _skip_unless_topic(memory)
    await memory.subscribe_topic(
        _SESSION, topic="A", intent="subtask", ctx=_tctx(None), task_id="B")
    subs = await memory.list_subscriptions(_SESSION, ctx=_ctx(), task_id="B")
    assert [s.topic for s in subs] == ["A"]
