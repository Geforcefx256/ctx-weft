"""Frozen prompt-memory transfer and verification; no task scheduling or UI history."""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta

from ctx_weft.core.assembler.builder import history_records
from ctx_weft.core.assembler.context import PromptContext, record_from_json
from ctx_weft.core.control.reducers import REPLAY_BATCH
from ctx_weft.core.control.startup_view import rebuild_startup_view
from ctx_weft.core.orchestrator.model import ModelChoice
from ctx_weft.core.utils.event import new_event
from ctx_weft.protocols import MemoryAddress, MemoryEvent, MemoryKind, MemoryScope, ProviderContext
from ctx_weft.protocols.content import (
    ImagePart,
    TextPart,
    content_from_jsonable,
    content_to_jsonable,
    externalize_content,
)
from ctx_weft.protocols.events import EventOrigin, EventReadProjection, EventType
from ctx_weft.protocols.llm import deserialize_call_config
from ctx_weft.protocols.runtime import MemorySeedConflictError, SessionForkResult


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def identity(seed_id, label) -> str:
    return "seed_" + digest([seed_id, label])[:48]


async def transfer_content(content, source_store, target_store, source_ctx, target_ctx):
    """Read using the source scope, then put using the target scope, even on one store."""
    if isinstance(content, str) or content is None:
        return content
    parts = []
    for part in content:
        if isinstance(part, ImagePart) and part.source_type == "ref":
            stored = await source_store.get(part.data, source_ctx)
            if stored is None:
                raise ValueError(f"Seed image is unavailable: {part.data}")
            raw, media_type = stored
            part = replace(
                part,
                source_type="base64",
                data=base64.b64encode(raw).decode(),
                media_type=media_type or part.media_type,
            )
        parts.append(part)
    return await externalize_content(parts, blob_store=target_store, ctx=target_ctx)


def rewrite_refs(content, mappings):
    def text(value):
        for old, new in mappings.items():
            value = value.replace(old, new)
        return value

    if isinstance(content, str):
        return text(content)
    return [replace(p, text=text(p.text)) if isinstance(p, TextPart) else p for p in content]


async def content_signature(content, retained, store, ctx):
    """Compare blob bytes, not opaque references assigned by a provider's put()."""
    mappings = {}
    retained_hashes = set()
    for text_ref, store_ref in retained:
        got = await store.get(store_ref, ctx)
        if got is None:
            raise ValueError(f"Seed blob is unavailable: {store_ref}")
        raw, media_type = got
        signature = digest([hashlib.sha256(raw).hexdigest(), media_type])
        mappings[text_ref] = "seed-blob:" + signature
        retained_hashes.add(signature)
    content = rewrite_refs(content, mappings)
    if not isinstance(content, str):
        parts = []
        for part in content:
            if isinstance(part, ImagePart) and part.source_type == "ref":
                got = await store.get(part.data, ctx)
                if got is None:
                    raise ValueError(f"Seed image is unavailable: {part.data}")
                raw, media_type = got
                part = replace(
                    part,
                    data=hashlib.sha256(raw).hexdigest(),
                    media_type=media_type or part.media_type,
                    byte_size=len(raw),
                )
            parts.append(part)
        content = parts
    return content_to_jsonable(content), retained_hashes


async def prepare_seed(
    runtime,
    *,
    source_session_id,
    response_event_id,
    session_id,
    agent_id,
    fingerprint,
    template_id,
    llm_account,
    llm_model,
    llm_config,
):
    store, memory = runtime.event_store, runtime.providers.get_memory()
    head = await store.committed_head(source_session_id)

    async def find(kind, predicate, through, fields):
        matches = []
        for after in range(0, through, REPLAY_BATCH):
            rows = await store.read_range(
                source_session_id,
                after_position=after,
                through_position=min(after + REPLAY_BATCH, through),
                include_types=(kind,),
                projection=EventReadProjection({kind: fields}),
            )
            matches.extend(row for row in rows if predicate(row.event))
        return matches

    responses = await find(
        EventType.LLM_RESPONSE_FINISHED, lambda ev: ev.id == response_event_id, head, ()
    )
    response_row = responses[0] if responses else None
    if response_row is None:
        raise ValueError("Source response is not committed in this session")
    (response_row,) = await store.read_range(
        source_session_id,
        after_position=response_row.position - 1,
        through_position=response_row.position,
    )
    response, p = response_row.event, response_row.event.payload
    if (
        response.origin != EventOrigin.LOOP_ACT
        or not isinstance(p.get("content"), str)
        or not p["content"].strip()
        or p.get("tool_calls")
        or not p.get("memory_id")
    ):
        raise ValueError(
            "Fork requires a completed, nonempty actor text response without tool calls"
        )
    matches = await find(
        EventType.LLM_PROMPT_SENT,
        lambda ev: (
            ev.payload.get("request_id") == p.get("request_id")
            and ev.agent_id == response.agent_id
            and ev.task_id == response.task_id
            and ev.run_id == response.run_id
        ),
        response_row.position,
        (("request_id",),),
    )
    if len(matches) != 1 or not p.get("request_id"):
        raise ValueError("Source response does not identify one saved prompt")
    (prompt_row,) = await store.read_range(
        source_session_id,
        after_position=matches[0].position - 1,
        through_position=matches[0].position,
    )
    prompt = prompt_row.event
    inputs = PromptContext.from_payload(prompt.payload)
    if inputs.purpose != "act" or prompt.tenant_id != response.tenant_id:
        raise ValueError("Source prompt scope is inconsistent")
    view = await rebuild_startup_view(
        store, source_session_id, through_position=response_row.position
    )
    source_session = view.sessions.get(source_session_id)
    source_agent = view.agents.get(response.agent_id)
    if (
        source_session is None
        or source_agent is None
        or source_session.tenant_id != response.tenant_id
    ):
        raise ValueError("Source response has no matching persisted session and agent")
    for key, expected in (
        ("session_id", source_session_id),
        ("agent_id", response.agent_id),
        ("task_id", response.task_id),
        ("tenant_id", response.tenant_id),
    ):
        if inputs.scope.get(key) not in (None, expected):
            raise ValueError(f"Source prompt {key} does not match its response")
    source_ctx = ProviderContext(
        session_id=source_session_id,
        tenant_id=response.tenant_id,
        agent_id=response.agent_id,
        task_id=response.task_id,
    )
    target_ctx = ProviderContext(
        session_id=session_id, tenant_id=response.tenant_id, agent_id=agent_id
    )
    memory_blobs = runtime.providers.get_memory_blob_store()
    event_blobs = runtime.providers.get_event_blob_store()
    records = {r.id: r for r in await memory.read_records(inputs.memory_ids(), source_ctx)}
    record_stores = {}
    # Inline staged records in prompt events belong to the event blob namespace.
    for index, block in enumerate(inputs.memory_blocks):
        if "record" in block:
            record = record_from_json(block["record"])
            if not record.id:
                record = replace(record, id=f"inline-memory:{index}")
                block["memory_id"] = record.id
            elif block.get("memory_id") not in (None, "", record.id):
                raise ValueError("Inline source memory identity does not match its reference")
            block["memory_id"] = record.id
            records[record.id] = record
            record_stores[record.id] = event_blobs
    missing = set(inputs.memory_ids()) - records.keys()
    if missing:
        raise ValueError(f"Source prompt memories are unavailable: {sorted(missing)}")
    history_ids = list(
        dict.fromkeys(
            b["memory_id"]
            for b in inputs.memory_blocks
            if b.get("kind") == "history" and b.get("memory_id")
        )
    )
    # Shared inheritance projection: semantic normalization, with no token-budget trim.
    history = history_records([records[rid] for rid in history_ids])
    used = set(history_ids)
    items = [
        (r.content, r.role, r.metadata, r.blob_refs, record_stores.get(r.id, memory_blobs), r.id)
        for r in history
    ]
    base_ids = [b["memory_id"] for b in inputs.memory_blocks if b.get("memory_id")]
    base_ids += [
        inputs.task[k] for k in ("user_prompt_memory_id", "outputs_memory_id") if inputs.task.get(k)
    ]
    for rid in base_ids:
        if rid in used:
            continue
        record = records[rid]
        items.append(
            (
                record.content,
                record.role or "user",
                record.metadata,
                record.blob_refs,
                record_stores.get(rid, memory_blobs),
                rid,
            )
        )
        used.add(rid)
    # The task instruction precedes request-local appended messages.
    if not inputs.task.get("user_prompt_memory_id") and not inputs.task.get(
        "user_prompt_in_memory"
    ):
        content = content_from_jsonable(inputs.task.get("user_prompt"))
        if content:
            items.append((content, "user", {}, [], event_blobs, "task:user_prompt"))
    for index, message in enumerate(inputs.appended_messages):
        if message.get("memory_id"):
            rid = message["memory_id"]
            if rid not in used:
                record = records[rid]
                items.append(
                    (
                        record.content,
                        message["role"],
                        record.metadata,
                        record.blob_refs,
                        record_stores.get(rid, memory_blobs),
                        rid,
                    )
                )
                used.add(rid)
            continue
        items.append(
            (
                content_from_jsonable(message["content"]),
                message["role"],
                message,
                [],
                event_blobs,
                f"inline:{index}",
            )
        )
    items.append((p["content"], "assistant", {}, [], memory_blobs, p["memory_id"]))
    seed_id = identity(session_id, "root-memory")
    frozen = []
    call_ids = {}
    for _, role, metadata, _, _, _ in items:
        if role == "assistant":
            for call in metadata.get("tool_calls", []):
                old = call["id"]
                if old in call_ids:
                    raise ValueError("Source memories have ambiguous tool call identities")
                call_ids[old] = identity(seed_id, ["call", old])
    results = set()
    for ordinal, (content, role, metadata, refs, blob_source, source_id) in enumerate(items):
        md = {
            "memory_seed": {"seed_id": seed_id, "ordinal": ordinal, "source_memory_id": source_id}
        }
        if role == "assistant" and metadata.get("tool_calls"):
            md["tool_calls"] = deepcopy(metadata["tool_calls"])
            for call in md["tool_calls"]:
                call["id"] = call_ids[call["id"]]
        if role == "tool":
            old = metadata.get("tool_call_id")
            if old not in call_ids or old in results:
                raise ValueError("Source memories contain an unpaired tool result")
            results.add(old)
            md["tool_call_id"] = call_ids[old]
        content = await transfer_content(content, blob_source, event_blobs, source_ctx, target_ctx)
        retained = []
        for ref in refs:
            got = await memory_blobs.get(ref, source_ctx)
            if got is None:
                raise ValueError(f"Retained seed blob is unavailable: {ref}")
            raw, media_type = got
            retained.append(
                {"source_ref": ref, "event_ref": await event_blobs.put(raw, media_type, target_ctx)}
            )
        frozen.append(
            {
                "id": identity(seed_id, ordinal),
                "role": role,
                "content": content_to_jsonable(content),
                "metadata": md,
                "retained_blobs": retained,
                # A new contiguous timeline avoids sorting across unrelated source scopes.
                "timestamp": (response.timestamp + timedelta(microseconds=ordinal)).isoformat(),
            }
        )
    if set(call_ids) != results:
        raise ValueError("Source memories contain an unfinished tool call")
    choice = ModelChoice(
        account=llm_account
        if llm_account is not None
        else p.get("llm_account", source_agent.llm_account),
        model=llm_model if llm_model is not None else p.get("llm_model", source_agent.llm_model),
        config=deepcopy(llm_config)
        if llm_config is not None
        else deserialize_call_config(prompt.payload.get("llm_config")),
    )
    creation = await runtime._session_registry.prepare_container_events(
        session_id=session_id,
        agent_id=agent_id,
        tenant_id=response.tenant_id,
        template_id=template_id or inputs.template_id or source_agent.template_id,
        llm=choice,
    )
    requested = new_event(
        EventType.AGENT_MEMORY_SEED_REQUESTED,
        session_id=session_id,
        tenant_id=response.tenant_id,
        agent_id=agent_id,
        origin=EventOrigin.RUNTIME,
        payload={
            "format_version": 1,
            "seed_id": seed_id,
            "request_fingerprint": fingerprint,
            "record_count": len(frozen),
            "manifest_hash": digest(frozen),
            "records": frozen,
            "source": {
                "session_id": source_session_id,
                "prompt_event_id": prompt.id,
                "response_event_id": response.id,
                "response_position": response_row.position,
            },
        },
    )
    return (*creation, requested)


async def restore_seed(runtime, session_view):
    """Run only under the target session command guard; never from a query/reducer."""
    descriptor = session_view.context_seed
    if descriptor is None:
        return
    memory = runtime.providers.get_memory()
    key = (session_view.id, descriptor["seed_id"])
    if runtime._memory_seed_ready.get(key) is memory:
        return
    stored = await runtime.event_store.read_last_of_type(
        session_view.id, EventType.AGENT_MEMORY_SEED_REQUESTED
    )
    if stored is None or stored.event.id != descriptor["requested_event_id"]:
        raise MemorySeedConflictError("Seed request is unavailable")
    event, payload = stored.event, stored.event.payload
    if (
        event.tenant_id != session_view.tenant_id
        or event.agent_id != session_view.root_agent_id
        or payload.get("format_version") != 1
        or digest(payload["records"]) != descriptor["manifest_hash"]
        or len(payload["records"]) != descriptor["record_count"]
        or any(
            payload.get(k) != descriptor[k]
            for k in ("seed_id", "request_fingerprint", "manifest_hash", "record_count", "source")
        )
    ):
        raise MemorySeedConflictError("Seed request does not match its descriptor")
    ctx = ProviderContext(
        session_id=session_view.id,
        tenant_id=session_view.tenant_id,
        agent_id=session_view.root_agent_id,
    )
    event_blobs = runtime.providers.get_event_blob_store()
    memory_blobs = runtime.providers.get_memory_blob_store()
    address = MemoryAddress(session_id=ctx.session_id, agent_id=ctx.agent_id)

    async def verify(rows):
        by_id = {r.id: r for r in rows}
        missing = []
        for item in payload["records"]:
            actual = by_id.get(item["id"])
            if actual is None:
                missing.append(item)
                continue
            md = actual.metadata
            if (
                actual.scope != MemoryScope.AGENT
                or actual.address != address
                or actual.role != item["role"]
                or actual.kind != MemoryKind.CONVERSATION_TURN
                or actual.timestamp != datetime.fromisoformat(item["timestamp"])
                or any(md.get(k) != v for k, v in item["metadata"].items())
            ):
                raise MemorySeedConflictError(f"Seed memory differs: {item['id']}")
            expected = await content_signature(
                content_from_jsonable(item["content"]),
                [(r["source_ref"], r["event_ref"]) for r in item["retained_blobs"]],
                event_blobs,
                ctx,
            )
            observed = await content_signature(
                actual.content, [(ref, ref) for ref in actual.blob_refs], memory_blobs, ctx
            )
            if observed != expected:
                raise MemorySeedConflictError(f"Seed memory content differs: {item['id']}")
        return missing

    ids = [item["id"] for item in payload["records"]]
    missing = await verify(await memory.read_records(ids, ctx))
    if missing:
        events = []
        for item in missing:
            content = await transfer_content(
                content_from_jsonable(item["content"]), event_blobs, memory_blobs, ctx, ctx
            )
            mappings = {}
            for retained in item["retained_blobs"]:
                got = await event_blobs.get(retained["event_ref"], ctx)
                if got is None:
                    raise ValueError("Retained seed blob is unavailable")
                raw, media_type = got
                mappings[retained["source_ref"]] = await memory_blobs.put(raw, media_type, ctx)
            events.append(
                MemoryEvent(
                    id=item["id"],
                    kind=MemoryKind.CONVERSATION_TURN,
                    scope=MemoryScope.AGENT,
                    address=address,
                    role=item["role"],
                    content=rewrite_refs(content, mappings),
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    metadata=deepcopy(item["metadata"]),
                    blob_refs=list(mappings.values()),
                )
            )
        try:
            await memory.fold([], events, ctx)
        except Exception:
            if await verify(await memory.read_records(ids, ctx)):
                raise
        if await verify(await memory.read_records(ids, ctx)):
            raise RuntimeError("Seed memory write was not confirmed")
    if descriptor["applied_event_id"] is None:
        # The envelope is stable across hot retries and process restarts.
        applied = new_event(
            EventType.AGENT_MEMORY_SEED_APPLIED,
            session_id=event.session_id,
            tenant_id=event.tenant_id,
            agent_id=event.agent_id,
            origin=EventOrigin.RUNTIME,
            timestamp=event.timestamp,
            causation_id=event.id,
            payload={k: payload[k] for k in ("seed_id", "manifest_hash", "record_count")},
        )
        applied.id = identity(payload["seed_id"], "applied")
        await runtime.event_bus.emit_batch((applied,))
        descriptor["applied_event_id"] = applied.id
    runtime._memory_seed_ready[key] = memory


def fork_result(session_view):
    seed = session_view.context_seed
    return SessionForkResult(
        session_view.id,
        session_view.root_agent_id,
        seed["seed_id"],
        seed["source"]["session_id"],
        seed["source"]["response_event_id"],
        seed["source"]["prompt_event_id"],
        seed["record_count"],
    )
