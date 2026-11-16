"""Default memory transformations. No agent events, loop state, LLM or budgets.

Selection objects below are private values local to a call. Callers retain normal
MemoryRecords and pass their IDs back after generating a summary.
"""

from __future__ import annotations

import dataclasses
import re
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from ctx_weft.core.assembler.legacy_task_memory import (
    capsule_origin,
    normalize_legacy_tasks,
    task_record_group,
    task_record_kind,
)
from ctx_weft.core.assembler.prompts import runtime_prompt
from ctx_weft.core.media.fold import placeholder_refs
from ctx_weft.protocols.content import content_to_text
from ctx_weft.core.utils.ids import record_id
from ctx_weft.protocols import (
    MemoryAddress,
    MemoryEvent,
    MemoryKind,
    MemoryRecord,
    MemoryScope,
)
from ctx_weft.protocols.memory import MemoryOrganizeDeferred, MemoryOrganizeResult

COLLAPSE_DELIM = "\n\n---\n## Execution Summary (earlier conversation compacted)\n"
LEGACY_COLLAPSE_DELIMS = ("\n\n---\n## 执行摘要（先前对话已压缩）\n",)
Action = Literal["demote", "repair", "history", "current"]


@dataclass(frozen=True)
class _WorkingRecords:
    records: tuple[MemoryRecord, ...]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(r.id for r in self.records)

    @property
    def agent(self) -> tuple[MemoryRecord, ...]:
        return tuple(r for r in self.records if r.scope == MemoryScope.AGENT)

    @property
    def body(self) -> tuple[MemoryRecord, ...]:
        return tuple(r for r in self.records if r.scope == MemoryScope.TASK)


@dataclass(frozen=True)
class _Selection:
    action: Action
    records: tuple[MemoryRecord, ...]
    roots: tuple[str, ...]
    origins: frozenset[str]
    summary_ids: tuple[str, ...]
    anchor: datetime
    keep_last: int = 0
    closing: MemoryRecord | None = None
    parent_id: str | None = None
    original: str = ""
    lean_only: bool = True

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(r.id for r in self.records)


DIGEST_SECTIONS = ("Goal", "Constraints", "Done", "Remaining", "Evidence")
REQUIRED_DIGEST_SECTIONS = DIGEST_SECTIONS[:4]
_DIGEST_SECTION_RE = {
    name: re.compile(rf"^## {name}[ \t]*\n(.*?)(?=^## |\Z)", re.S | re.M)
    for name in DIGEST_SECTIONS
}


@dataclasses.dataclass(frozen=True)
class _Digest:
    """解析后的 digest：text 为落库形态（结构化时即原文），degraded 标降级。"""

    text: str
    degraded: bool
    sections: dict[str, str] = dataclasses.field(default_factory=dict)


def parse_structured_digest(text: str) -> _Digest:
    """宽松解析（spec: compact-fidelity）：抓 `## <Name>` 小节；Goal/Constraints/"""
    sections: dict[str, str] = {}
    for name in DIGEST_SECTIONS:
        m = _DIGEST_SECTION_RE[name].search(text)
        if m and m.group(1).strip():
            sections[name] = m.group(1).strip()
    degraded = any(name not in sections for name in REQUIRED_DIGEST_SECTIONS)
    return _Digest(text=text, degraded=degraded, sections=sections)


def _original_section(content: str) -> str:
    """取（可能已坍缩过的）USER_PROMPT 的「原始消息」节：有分隔标记取其前段，否则整体即原文。"""
    for delim in (COLLAPSE_DELIM, *LEGACY_COLLAPSE_DELIMS):
        idx = content.find(delim)
        if idx != -1:
            return content[:idx]
    return content


def _rendered_only(recs: list) -> list:
    """去掉装配时不渲染的记录（TOOL_AUDIT）。"""
    return [r for r in recs if getattr(r, "kind", None) is not MemoryKind.TOOL_AUDIT]


def _split_for_collapse(
    recs: list,
    keep_last: int,
) -> tuple[list, list]:
    """把 task 层视图切成 (折区, 保留区)：保留**至少** keep_last 条会被渲染的记录，并对齐回合边界。"""
    cut, seen = len(recs), 0
    if keep_last > 0:
        for i in range(len(recs) - 1, -1, -1):
            if getattr(recs[i], "kind", None) is MemoryKind.TOOL_AUDIT:
                continue
            seen, cut = seen + 1, i
            if seen >= keep_last:
                break
        if seen < keep_last:
            return [], list(recs)
    # Infer unfinished execution from memory itself. No caller-supplied IDs.
    pending = _pending_calls(recs)
    cut = min([cut] + [i for i, r in enumerate(recs) if _unfinished_call_record(r, pending)])
    while True:
        kept_call_ids = {
            r.metadata.get("tool_call_id")
            for r in recs[cut:]
            if r.role == "tool" and r.metadata.get("tool_call_id")
        }
        owner = next(
            (
                i
                for i in range(cut)
                if recs[i].role == "assistant"
                and any(
                    tc.get("id") in kept_call_ids
                    for tc in (recs[i].metadata.get("tool_calls") or [])
                )
            ),
            None,
        )
        groups = {task_record_group(r) for r in recs[cut:] if task_record_group(r)}
        group_owner = next((i for i in range(cut) if task_record_group(recs[i]) in groups), None)
        candidates = [i for i in (owner, group_owner) if i is not None]
        if not candidates:
            return recs[:cut], recs[cut:]
        cut = min(candidates)


def _worth_collapsing(fold: list) -> bool:
    """折区里除了**那一条最早的 user 回合**（原始请求 / 旧坍缩物）之外还有会被渲染的东西，才值得坍。"""
    rendered = _rendered_only(fold)
    first_user = next(
        (
            r
            for r in rendered
            if r.role == "user"
            and r.kind is MemoryKind.CONVERSATION_TURN
            and not task_record_kind(r)
        ),
        None,
    )
    return any(r is not first_user for r in rendered)


def _dispatch_finish_sets(recs) -> tuple[set, set]:
    """扫 conversation turn，按 origin 标记：has_dispatch（有 delegate/delegate_plan 调用回合）、"""
    has_dispatch: set = set()
    has_finish: set = set()
    for r in recs:
        if r.kind is not MemoryKind.CONVERSATION_TURN:
            continue
        oid = capsule_origin(r)
        if oid is None:
            continue
        kind = task_record_kind(r)
        if kind == "start":
            has_dispatch.add(oid)
        elif kind == "finish":
            has_finish.add(oid)
        for tc in r.metadata.get("tool_calls") or []:
            name = str(tc.get("name", ""))
            if name.endswith("finish_task"):
                has_finish.add(oid)
            elif name.endswith("delegate_task") or name.endswith("delegate_plan"):
                has_dispatch.add(oid)
    return has_dispatch, has_finish


@dataclass(frozen=True)
class _TaskGraph:
    parents: dict[str, str | None]
    first_ts: dict[str, datetime]
    roots: tuple[str, ...]

    def expand(self, roots) -> set[str]:
        found = set(roots)
        while True:
            expanded = found | {oid for oid, parent in self.parents.items() if parent in found}
            if expanded == found:
                return found
            found = expanded


def _task_graph(recs, current, body=()) -> _TaskGraph:
    """One origin/ancestry policy for history demotion and folding."""
    parents, ranks, first_ts = {}, {}, {}
    projected = normalize_legacy_tasks(recs)
    has_dispatch, has_finish = _dispatch_finish_sets(recs)
    for r in projected:
        if r.kind != MemoryKind.CONVERSATION_TURN:
            continue
        oid = capsule_origin(r)
        if oid is None:
            continue
        kind = task_record_kind(r)
        parent = r.metadata.get("parent_task_id") or None
        if parent == oid:
            parent = None
        if kind in {"start", "result"} and not r.metadata.get("legacy_task_record"):
            parent = None  # These labels describe the dispatched child.
        rank = 2 if kind == "finish" else 1 if kind in {"recap", "final_reply"} else 0
        if (
            oid not in parents
            or rank > ranks[oid]
            or (rank == ranks[oid] and parents[oid] is None and parent is not None)
        ):
            parents[oid], ranks[oid] = parent, rank
        if oid not in first_ts or r.timestamp < first_ts[oid]:
            first_ts[oid] = r.timestamp
        if kind == "finish":
            has_finish.add(oid)
    # A late segment can outlive the closure IDs folded by an earlier snapshot.
    # Keep only these residual task identities, so later compactions can still
    # select their details even when no live closure remains.
    residuals = {oid for r in recs for oid in r.metadata.get("residual_task_ids", [])}
    for r in body if residuals else ():
        oid = r.address.task_id
        if oid in residuals and oid not in parents:
            parents[oid], first_ts[oid] = None, r.timestamp
    has_result = {
        capsule_origin(r)
        for r in projected
        if (r.role == "tool" or task_record_kind(r) == "result")
        and r.metadata.get("dispatch_state") != "running"
    }
    active = has_dispatch - has_finish - has_result
    closed_children = {
        r.metadata.get("child_task_id")
        for r in projected
        if task_record_kind(r) == "result" and r.metadata.get("dispatch_state") != "running"
    }
    open_owners = {
        capsule_origin(r)
        for r in projected
        if task_record_kind(r) == "start"
        and r.metadata.get("child_task_id")
        and r.metadata["child_task_id"] not in has_finish | closed_children
    }
    # A live branch protects its ancestors, but not its completed siblings or
    # children. Select the largest completed subtrees below those ancestors.
    legacy_closed = {
        capsule_origin(r)
        for r in projected
        if r.kind == MemoryKind.CONVERSATION_TURN
        and r.role == "assistant"
        and task_record_kind(r) is None
        and not r.metadata.get("tool_calls")
    }
    closed = has_finish | has_result | legacy_closed | residuals
    protected = {current} | active | open_owners | (parents.keys() - closed)
    for oid in list(protected):
        parent = parents.get(oid)
        while parent is not None and parent not in protected:
            protected.add(parent)
            parent = parents.get(parent)
    roots = [
        oid
        for oid, parent in parents.items()
        if oid not in protected and (parent not in parents or parent in protected)
    ]
    roots.sort(key=lambda oid: first_ts[oid])
    return _TaskGraph(parents, first_ts, tuple(roots))


def _capsules(history: _WorkingRecords, current: str) -> tuple[_Selection, ...]:
    """L1: disjoint completed subtrees, oldest first; healthy lean _capsules skip."""
    turns, body = history.agent, history.body
    projected = normalize_legacy_tasks(turns)
    graph = _task_graph(turns, current, body)
    finishes = {capsule_origin(r): r for r in projected if task_record_kind(r) == "finish"}
    for row in reversed(projected):
        if (
            row.role == "assistant"
            and not task_record_kind(row)
            and not row.metadata.get("tool_calls")
        ):
            finishes.setdefault(capsule_origin(row), row)
    children, records = defaultdict(list), defaultdict(list)
    for oid, parent in graph.parents.items():
        children[parent].append(oid)
    for row in body:
        records[row.address.task_id].append(row)
    for row in turns:
        if row.kind == MemoryKind.CONVERSATION_TURN:
            records[capsule_origin(row)].append(row)
    plans = []
    for root in graph.roots:
        origins, todo = set(), [root]
        while todo:
            oid = todo.pop()
            if oid not in origins:
                origins.add(oid)
                todo.extend(children[oid])
        rows = tuple(sorted((r for oid in origins for r in records[oid]), key=_order))
        closing = finishes.get(root)
        policy = closing.metadata.get("finish_policy", {}) if closing else {}
        lean = policy.get("capsule_state") == "lean" and policy.get("has_summary") is not False
        if not rows or (lean and len(rows) == 1):
            continue
        demote = closing is not None and not lean and policy.get("has_summary") is not False
        plans.append(
            _Selection(
                "demote" if demote else "repair",
                rows,
                (root,),
                frozenset(origins),
                () if demote else tuple(r.id for r in rows),
                closing.timestamp if closing else rows[-1].timestamp,
                closing=closing,
                parent_id=graph.parents.get(root),
            )
        )
    return tuple(plans)


def _history_selection(
    history: _WorkingRecords, current: str, keep_last: int, *, lean_only: bool = True
) -> _Selection | None:
    """L2: select old lean _capsules and the existing history digest together."""
    graph = _task_graph(history.agent, current, history.body)
    roots = graph.roots
    if lean_only:
        details = {r.address.task_id for r in history.body}
        details |= {
            capsule_origin(r)
            for r in history.agent
            if r.kind == MemoryKind.CONVERSATION_TURN
            and (
                task_record_kind(r) != "finish"
                or r.metadata.get("finish_policy", {}).get("capsule_state") != "lean"
            )
        }
        roots = [oid for oid in roots if not graph.expand([oid]) & details]
    roots = roots[:-keep_last] if keep_last > 0 else roots
    origins = graph.expand(roots)
    rows = tuple(
        r
        for r in history.records
        if (r.scope == MemoryScope.TASK and r.address.task_id in origins)
        or (
            r.scope == MemoryScope.AGENT
            and (
                (r.kind == MemoryKind.CONVERSATION_TURN and capsule_origin(r) in origins)
                or (
                    r.kind == MemoryKind.SUMMARY
                    and (roots or keep_last <= 0)
                    and (not lean_only or capsule_origin(r) is None)
                )
            )
        )
    )
    if not rows:
        return None
    selected = {r.id for r in rows}
    kept = [r.timestamp for r in history.records if r.id not in selected]
    anchor = (min(kept) if kept else max(r.timestamp for r in rows)) - timedelta(microseconds=1)
    return _Selection(
        "history",
        rows,
        tuple(roots),
        frozenset(origins),
        tuple(r.id for r in rows),
        anchor,
        keep_last,
        lean_only=lean_only,
    )


def _current_selection(
    history: _WorkingRecords,
    current: str,
    keep_last: int,
) -> _Selection | None:
    """L3: retain recent complete tool rounds and the pending Act turn."""
    rows = [r for r in history.body if r.address.task_id == current]
    fold, kept = _split_for_collapse(rows, keep_last)
    if not _worth_collapsing(fold):
        return None
    original = next(
        (
            _original_section(content_to_text(r.content))
            for r in fold
            if r.kind == MemoryKind.CONVERSATION_TURN
            and r.role == "user"
            and not task_record_kind(r)
        ),
        "",
    )
    summary = tuple(r.id for r in fold)
    anchor = (kept[0] if kept else fold[0]).timestamp - timedelta(microseconds=1)
    return _Selection(
        "current", tuple(fold), (), frozenset(), summary, anchor, keep_last, original=original
    )


def _history_details_remain(history: _WorkingRecords, current: str) -> bool:
    graph = _task_graph(history.agent, current, history.body)
    origins = graph.expand(graph.roots)
    return any(r.address.task_id in origins for r in history.body) or any(
        capsule_origin(r) in origins and task_record_kind(r) in {"recap", "final_reply"}
        for r in history.agent
    )


def _order(row):
    return row.timestamp, row.metadata.get("seq_no", 0)


def _replacement(plan: _Selection, scope, digest: _Digest | None, residuals) -> MemoryEvent:
    """Materialize one validated _replacement; selection never happens here."""
    if plan.action == "demote":
        closing = plan.closing
        metadata = dict(closing.metadata)
        policy = {
            **metadata.get("finish_policy", {}),
            "capsule_state": "lean",
            "has_summary": True,
            "slot_count": 1,
            "preserve_raw": False,
        }
        policy.pop("final_reply_record_id", None)
        metadata["finish_policy"] = policy
        if not task_record_kind(closing):
            metadata["task_memory"] = {"version": 1, "kind": "finish", "group_id": closing.id}
        legacy = metadata.pop("legacy_task_record", None)
        content = (
            runtime_prompt("task_memory_record", record=legacy).strip()
            if legacy
            else closing.content
        )
        return MemoryEvent(
            id=record_id("demote_capsule", closing.id),
            kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.AGENT,
            address=MemoryAddress(session_id=scope.session_id, agent_id=scope.agent_id),
            role="user",
            content=content,
            timestamp=plan.anchor,
            metadata=metadata,
            blob_refs=closing.blob_refs,
        )
    if digest is None or not digest.text.strip():
        raise ValueError("organize requires a nonempty summary")
    metadata = {"digest_degraded": digest.degraded}
    if residuals:
        metadata["residual_task_ids"] = sorted(residuals)
    if plan.action == "current":
        content = f"{plan.original}{COLLAPSE_DELIM}{digest.text}"
        metadata.update(
            task_id=scope.task_id,
            collapsed=True,
            keep_last=plan.keep_last,
            folded_count=len(plan.ids),
        )
        return MemoryEvent(
            kind=MemoryKind.CONVERSATION_TURN,
            scope=MemoryScope.TASK,
            address=scope,
            role="user",
            content=content,
            timestamp=plan.anchor,
            metadata=metadata,
            blob_refs=placeholder_refs(content),
        )
    metadata.update(keep_last=plan.keep_last, folded_count=len(plan.roots))
    return MemoryEvent(
        kind=MemoryKind.SUMMARY,
        scope=MemoryScope.AGENT,
        address=MemoryAddress(session_id=scope.session_id, agent_id=scope.agent_id),
        role="user",
        content=digest.text,
        timestamp=plan.anchor,
        metadata=metadata,
        blob_refs=placeholder_refs(digest.text),
    )


_VIEW_KINDS = [MemoryKind.CONVERSATION_TURN, MemoryKind.SUMMARY, MemoryKind.TOOL_AUDIT]


def _address(ctx, *, current=False):
    return MemoryAddress(
        session_id=ctx.session_id, agent_id=ctx.agent_id, task_id=ctx.task_id if current else None
    )


def _cache_records(memory, ctx, rows):
    # Only immutable bodies are cached. Active membership is always rechecked.
    # Bounded per provider, tenant/session keyed; no loop-local ambient state.
    cache = getattr(memory, "_organize_record_cache", None)
    if cache is None:
        cache = memory._organize_record_cache = OrderedDict()
    for row in rows:
        key = (ctx.tenant_id or "default", ctx.session_id, row.id)
        cache[key] = deepcopy(row)
        cache.move_to_end(key)
    while len(cache) > 4096:
        cache.popitem(last=False)
    return cache


async def _read_working(memory, ctx):
    if not ctx.agent_id:
        raise ValueError("organize requires agent_id")
    ids = []
    for scope in (MemoryScope.TASK, MemoryScope.AGENT):
        ids.extend(await memory.load_view_ids(_address(ctx), scope, ctx, kinds=_VIEW_KINDS))
    cache = _cache_records(memory, ctx, ())
    prefix = (ctx.tenant_id or "default", ctx.session_id)
    missing = [rid for rid in ids if (*prefix, rid) not in cache]
    fresh = await memory.read_records(missing, ctx) if missing else []
    # Keep this call's rows even when the view exceeds the cache's bounded size.
    rows = {rid: deepcopy(cache[(*prefix, rid)]) for rid in ids if (*prefix, rid) in cache}
    rows.update((r.id, r) for r in fresh)
    _cache_records(memory, ctx, fresh)
    return _WorkingRecords(tuple(sorted((rows[rid] for rid in ids), key=_order)))


def _record_call_ids(record):
    return set(record.metadata.get("execution_call_ids", ())) | {
        call.get("id") for call in record.metadata.get("tool_calls", ()) if call.get("id")
    }


def _unfinished_call_record(record, pending):
    calls = _record_call_ids(record)
    # Historical dispatch records may retain only a count. Without identities,
    # completion cannot be proven; conservatively retain that execution record.
    unknown = record.metadata.get("tool_call_count", 0) > len(calls)
    return unknown or bool(calls & pending)


def _pending_calls(rows):
    """Infer unfinished tool rounds from durable memory, not agent state."""
    completed = {
        r.metadata.get("tool_call_id")
        for r in rows
        if r.role == "tool" or r.metadata.get("execution_completed")
    }
    return {
        call
        for r in rows
        if r.role == "assistant"
        for call in _record_call_ids(r)
        if call not in completed
    }


def _is_user(r):
    return r.kind == MemoryKind.CONVERSATION_TURN and r.role == "user" and not task_record_kind(r)


def _segment_selection(selected, live, ctx):
    scope = selected[0].scope if selected else MemoryScope.TASK
    rows = sorted(selected, key=_order)
    if not rows:
        return None
    boundary = next(
        (r for r in reversed(live.records) if _is_user(r) and _order(r) <= _order(rows[-1])), None
    )
    fold = tuple(
        r
        for r in rows
        if (boundary is None or _order(r) > _order(boundary))
        and not _is_user(r)
        and r.kind != MemoryKind.SUMMARY
    )
    if not fold:
        return None
    # Never split or consume an unfinished tool round.
    if any(_unfinished_call_record(r, _pending_calls(rows)) for r in fold):
        raise MemoryOrganizeDeferred("segment contains an unfinished tool round")
    ids = {r.id for r in fold}
    following = [
        r
        for r in live.records
        if r.scope == scope and r.id not in ids and _order(r) >= _order(fold[0])
    ]
    anchor = following[0].timestamp - timedelta(microseconds=1) if following else fold[-1].timestamp
    return _Selection("segment", fold, (), frozenset(), tuple(r.id for r in fold), anchor)


async def _commit(memory, ctx, live, selected, events, *, images=0):
    """Storage implementations check expected active IDs inside their write boundary."""
    ids = tuple(r.id for r in selected)
    added = await memory._commit_organize(list(ids), events, ctx)
    if added is None:
        return MemoryOrganizeResult(
            status="deferred", records_before=len(live.records), records_after=len(live.records)
        )
    _cache_records(memory, ctx, added)
    return MemoryOrganizeResult(
        status="changed",
        superseded_ids=ids,
        replacements=tuple(added),
        records_before=len(live.records),
        records_after=len(live.records) - len(ids) + len(added),
        demoted_images=images,
    )


async def _demote_images(memory, ctx, live, wanted, keep_recent):
    from ctx_weft.core.media.fold import rebuild_demoted_record
    from ctx_weft.core.media.policy import plan_demotions

    rows = list(live.records)
    selected, events, indices = [], [], {}
    # Timestamp tie groups belong to a view. Mixing TASK and AGENT here would
    # make an unrelated agent record prevent rewriting the task image batch.
    for scope in (MemoryScope.TASK, MemoryScope.AGENT):
        scoped = [r for r in rows if r.scope == scope]
        scoped_wanted = wanted & {r.id for r in scoped}
        if not scoped_wanted:
            continue
        plan = plan_demotions(scoped, keep_recent=keep_recent, only_ids=scoped_wanted)
        by_id = {r.id: r for r in scoped}
        for batch in plan.batches:
            if not set(batch) <= scoped_wanted:
                continue
            rebuilt = [
                rebuild_demoted_record(by_id[rid], plan.demote_indices.get(rid, ()), scope)
                for rid in batch
            ]
            if any(e is None for e in rebuilt):
                continue
            selected.extend(by_id[rid] for rid in batch)
            events.extend(rebuilt)
            indices.update(plan.demote_indices)
    if not selected:
        return MemoryOrganizeResult(records_before=len(rows), records_after=len(rows))
    images = sum(len(indices.get(r.id, ())) for r in selected)
    for row, event in zip(selected, events):
        event.id = record_id("demote_images", row.id, tuple(indices.get(row.id, ())))
    return await _commit(memory, ctx, live, selected, events, images=images)


async def recall_for_organize(memory, ctx, *, operation, keep_last=0):
    if operation not in {"fold_history", "fold_current"}:
        raise ValueError(f"unsupported summary operation: {operation!r}")
    if keep_last < 0:
        raise ValueError("keep_last must be nonnegative")
    live = await _read_working(memory, ctx)
    if operation == "fold_current":
        if not ctx.task_id:
            raise ValueError("fold_current requires task_id")
        selection = _current_selection(live, ctx.task_id, keep_last)
    else:
        # Agent has already attempted rich -> lean. Any remaining completed raw
        # history needs a supplied digest too; summarize the selected old units
        # together rather than repairing one and prematurely proceeding to L3.
        selection = _history_selection(live, ctx.task_id, keep_last, lean_only=False)
        if selection is None and keep_last == 0 and _history_details_remain(live, ctx.task_id):
            raise MemoryOrganizeDeferred("history details are not ready for folding")
    if selection is None:
        return []
    return list(selection.records)


async def organize(memory, ctx, *, operation, record_ids=None, summary=None, keep_recent_images=2):
    operations = {"demote_images", "reduce_history", "fold_history", "fold_current", "fold_segment"}
    if operation not in operations:
        raise ValueError(f"unsupported organize operation: {operation!r}")
    folding = operation.startswith("fold_")
    if folding and (record_ids is None or not isinstance(summary, str) or not summary.strip()):
        raise ValueError("fold requires fixed record_ids and a nonempty summary")
    if not folding and summary is not None:
        raise ValueError(f"{operation} does not accept a summary")
    if keep_recent_images < 0:
        raise ValueError("keep_recent_images must be nonnegative")
    live = await _read_working(memory, ctx)
    active = {r.id: r for r in live.records}
    wanted = set(active) if record_ids is None else set(record_ids)
    if not wanted:
        return MemoryOrganizeResult(records_before=len(active), records_after=len(active))
    if operation == "demote_images":
        # L0.5 retains the established task-body image policy. Explicit ranges may
        # include agent records when preparing a selected historical digest.
        if record_ids is None:
            wanted = {r.id for r in live.body}
        return await _demote_images(memory, ctx, live, wanted, keep_recent_images)
    if operation == "reduce_history":
        selections = [
            p for p in _capsules(live, ctx.task_id) if p.action == "demote" and set(p.ids) <= wanted
        ]
        if not selections:
            return MemoryOrganizeResult(records_before=len(active), records_after=len(active))
        selection = selections[0]
        event = _replacement(selection, _address(ctx), None, set())
        return await _commit(memory, ctx, live, selection.records, [event])
    # Immutable reads include superseded records: reject the old source as a whole,
    # rather than silently applying its digest to a smaller, newer source.
    selected = await memory.read_records(list(dict.fromkeys(record_ids)), ctx)
    if {r.id for r in selected} != wanted:
        raise ValueError("organize source contains inaccessible or missing records")
    if operation == "fold_segment":
        scope = MemoryScope.TASK if ctx.task_id else MemoryScope.AGENT
        selected = [
            r
            for r in selected
            if r.scope == scope and (scope == MemoryScope.AGENT or r.address.task_id == ctx.task_id)
        ]
        wanted = {r.id for r in selected}
        live = _WorkingRecords(
            tuple(
                r
                for r in live.records
                if r.scope == scope
                and (scope == MemoryScope.AGENT or r.address.task_id == ctx.task_id)
            )
        )
        active = {r.id: r for r in live.records}
    if operation != "fold_history" and not wanted <= active.keys():
        return MemoryOrganizeResult(
            status="deferred", records_before=len(active), records_after=len(active)
        )
    if operation in {"fold_current", "fold_segment"}:
        if operation == "fold_current" and not ctx.task_id:
            raise ValueError("fold_current requires task_id")
        if operation == "fold_segment":
            try:
                selection = _segment_selection(selected, live, ctx)
            except MemoryOrganizeDeferred:
                return MemoryOrganizeResult(status="deferred")
            if selection is None:
                return MemoryOrganizeResult(records_before=len(active), records_after=len(active))
            event = MemoryEvent(
                id=record_id("fold_segment", *selection.ids),
                kind=MemoryKind.SUMMARY,
                scope=selection.records[0].scope,
                address=_address(ctx, current=bool(ctx.task_id)),
                role="assistant" if ctx.task_id else "user",
                content=summary,
                timestamp=selection.anchor,
                metadata={"keep_last": 0, "archived_count": len(selection.ids)},
                blob_refs=placeholder_refs(summary),
            )
        else:
            rows = sorted(
                (
                    r
                    for r in selected
                    if r.scope == MemoryScope.TASK and r.address.task_id == ctx.task_id
                ),
                key=_order,
            )
            if len(rows) != len(selected):
                raise ValueError("fold_current source must belong to the current scope")
            all_rows = [r for r in live.body if r.address.task_id == ctx.task_id]
            if [r.id for r in all_rows[: len(rows)]] != [r.id for r in rows]:
                return MemoryOrganizeResult(status="deferred")
            kept_count = len(_rendered_only(all_rows[len(rows) :]))
            safe, _ = _split_for_collapse(all_rows, kept_count)
            if [r.id for r in safe] != [r.id for r in rows]:
                return MemoryOrganizeResult(status="deferred")
            protected = _pending_calls(all_rows)
            if any(_unfinished_call_record(r, protected) for r in rows):
                return MemoryOrganizeResult(status="deferred")
            selection = _current_selection(_WorkingRecords(tuple(rows)), ctx.task_id, 0)
            if selection is None:
                return MemoryOrganizeResult()
            kept = all_rows[len(rows) :]
            if kept:
                selection = dataclasses.replace(
                    selection, anchor=kept[0].timestamp - timedelta(microseconds=1)
                )
            event = _replacement(
                selection, _address(ctx, current=True), parse_structured_digest(summary), set()
            )
    else:
        # A history digest describes its immutable source even when an observer
        # has since replaced some closure rows. Retain every later record, fold
        # only still-active source rows, and report that history needs another pass.
        subset = _WorkingRecords(tuple(sorted(selected, key=_order)))
        selection = _history_selection(subset, ctx.task_id, 0, lean_only=False)
        if selection is None or set(selection.ids) != wanted:
            return MemoryOrganizeResult(status="deferred")
        graph = _task_graph(live.agent, ctx.task_id, live.body)
        if not set(selection.roots) <= set(graph.roots):
            return MemoryOrganizeResult(status="deferred")
        late = [
            r
            for r in live.records
            if r.id not in wanted
            and (
                (r.scope == MemoryScope.TASK and r.address.task_id in selection.origins)
                or (r.scope == MemoryScope.AGENT and capsule_origin(r) in selection.origins)
            )
        ]
        residuals = {
            r.address.task_id if r.scope == MemoryScope.TASK else capsule_origin(r) for r in late
        }
        residuals.discard(None)
        kept = [r.timestamp for r in live.records if r.id not in wanted]
        if kept and selection.action == "history":
            selection = dataclasses.replace(selection, anchor=min(kept) - timedelta(microseconds=1))
        event = _replacement(selection, _address(ctx), parse_structured_digest(summary), residuals)
        remaining = tuple(r for r in selection.records if r.id in active)
        if not remaining:
            return MemoryOrganizeResult(status="deferred")
        event.id = record_id(operation, *selection.ids)
        result = await _commit(memory, ctx, live, remaining, [event])
        if late and result.changed:
            result = dataclasses.replace(result, status="deferred")
        return result
    if event.id is None:
        event.id = record_id(operation, *selection.ids)
    return await _commit(memory, ctx, live, selection.records, [event])
