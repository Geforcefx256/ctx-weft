"""Read-only prompt reconstruction behind the public runtime boundary."""

from ctx_weft.core.assembler import ContextAssembler, PromptContext
from ctx_weft.core.assembler.assembler import AssemblerDeps
from ctx_weft.core.control.reducers import rebuild_view
from ctx_weft.core.models.errors import SessionNotFound
from ctx_weft.core.orchestrator.lifecycle.template_lookup import TemplateLookup
from ctx_weft.core.utils.content import redact_content_for_event
from ctx_weft.protocols import ProviderContext
from ctx_weft.protocols.capability import CapabilityProvider
from ctx_weft.protocols.events import EventStore, EventType
from ctx_weft.protocols.memory import MemoryProvider
from ctx_weft.protocols.runtime import PromptDiagnostic


async def inspect_prompt(
    event_store: EventStore, session_id: str, event_id: str, *,
    memory: MemoryProvider, template_lookup: TemplateLookup,
    capability_provider_index: dict[str, CapabilityProvider], trim: bool = True,
) -> tuple[PromptDiagnostic, int]:
    """Return diagnostic data and its committed cut using only read dependencies.

    Runtime cache state and QueryResult metadata belong to the public runtime
    wrapper. Session identity and prompt inputs are read at the same cut here.
    """
    head = await event_store.committed_head(session_id)
    rows = await event_store.read_range(
        session_id, through_position=head, include_types=(EventType.LLM_PROMPT_SENT,))
    row = next((r for r in rows if r.event.id == event_id), None)
    if row is None:
        raise ValueError("Prompt inputs are unavailable")
    event = row.event
    if event.session_id != session_id:
        raise ValueError("Prompt event does not belong to session")
    inputs = PromptContext.from_payload(event.payload)
    view = await rebuild_view(event_store, session_id, through_position=head)
    session = view.sessions.get(session_id)
    if session is None:
        raise SessionNotFound(session_id)
    tenant_id = session.tenant_id
    if event.tenant_id != tenant_id:
        raise ValueError("Prompt event tenant does not belong to session")
    if inputs.scope.get("session_id") not in (None, session_id):
        raise ValueError("Prompt scope does not belong to session")
    if inputs.scope.get("tenant_id") not in (None, tenant_id):
        raise ValueError("Prompt scope tenant does not belong to session")
    for field in ("task_id", "agent_id"):
        owner = getattr(event, field)
        if owner and inputs.scope.get(field) not in (None, owner):
            raise ValueError(f"Prompt scope {field} does not match its event")
    ctx = ProviderContext(session_id=session_id, tenant_id=tenant_id,
                          task_id=event.task_id or inputs.scope.get("task_id"),
                          agent_id=event.agent_id or inputs.scope.get("agent_id"),
                          agent_template_id=inputs.template_id, skill_name=inputs.skill_name)
    template = (await template_lookup.get_template(inputs.template_id, None, ctx)
                if inputs.template_id else None)
    assembler = ContextAssembler([], AssemblerDeps(
        memory, [], ctx, capability_provider_index=capability_provider_index))
    prompt = await assembler.compile(inputs, template=template, trim=trim)
    result = PromptDiagnostic(
        event_id=event_id, system_prompt=prompt.system,
        messages=[{"role": m.role, "content": redact_content_for_event(m.content)}
                  for m in prompt.messages],
        tool_names=[tool.name for tool in prompt.tools],
        capability_errors=prompt.capability_errors, trim=trim)
    return result, head
