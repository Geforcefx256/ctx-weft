"""SkillExecutor resolves each operation with its context and passes the bare name."""

from __future__ import annotations

from ctx_weft.core.capabilities.skill_executor import (
    SkillExecutorCapabilityProvider,
)
from ctx_weft.protocols.capability import (
    Capability,
    CapabilityProviderInfo,
    SkillCapability,
    SkillCapabilityProvider,
    SkillDefinition,
)
from ctx_weft.protocols.context import ProviderContext


class _FakeSkillProvider(SkillCapabilityProvider):
    name = "local_skill"

    def __init__(self) -> None:
        self.list_files_called_with: str | None = None

    async def list(self, ctx) -> list[Capability]:
        return [SkillCapability(id="local_skill:pdf", name="pdf", description="d")]

    async def describe(self, ctx) -> CapabilityProviderInfo:
        return CapabilityProviderInfo(name=self.name, capability_count=1)

    async def load_definition(self, skill_name, ctx) -> SkillDefinition | None:
        return SkillDefinition(skill_id=f"local_skill:{skill_name}", skill_name=skill_name,
                               instructions="body")

    async def list_files(self, skill_name, pattern, limit, ctx) -> str:
        self.list_files_called_with = skill_name
        return "files"

    async def load_resource(self, skill_name, resource_path, ctx) -> str:
        return "res"

    async def exec_script(self, skill_name, command, ctx) -> str:
        self.exec_called_with = (skill_name, command, ctx)
        return "out"


class _Registry:
    def __init__(self, providers):
        self._p = providers

    def get_capability_providers(self):
        return list(self._p)


async def test_dispatch_routes_by_qualified_and_passes_bare_name() -> None:
    skill_provider = _FakeSkillProvider()
    executor = SkillExecutorCapabilityProvider(_Registry([skill_provider]))
    # ctx.skill_name is now the QUALIFIED skill name.
    ctx = ProviderContext(session_id="s1", tenant_id="default", skill_name="local_skill__pdf")
    events = [
        ev async for ev in executor.invoke(
            "skill_executor:list_files", {"pattern": "**/*"}, ctx,
        )
    ]
    assert any(e.kind == "result" for e in events)
    # the provider's Level3 method received the BARE skill name
    assert skill_provider.list_files_called_with == "pdf"


async def test_unknown_qualified_skill_errors() -> None:
    executor = SkillExecutorCapabilityProvider(_Registry([_FakeSkillProvider()]))
    ctx = ProviderContext(session_id="s1", tenant_id="default", skill_name="local_skill__missing")
    events = [
        ev async for ev in executor.invoke("skill_executor:list_files", {}, ctx)
    ]
    assert any(e.kind == "error" and e.payload.get("code") == "SKILL_NOT_FOUND" for e in events)


class _LongFileSkillProvider(_FakeSkillProvider):
    async def load_resource(self, skill_name, resource_path, ctx) -> str:
        return "".join(f"line{i}\n" for i in range(1, 5001))


async def test_read_file_returns_one_window_not_the_whole_file() -> None:
    executor = SkillExecutorCapabilityProvider(_Registry([_LongFileSkillProvider()]))
    ctx = ProviderContext(session_id="s1", tenant_id="default", skill_name="local_skill__pdf")
    events = [
        ev async for ev in executor.invoke(
            "skill_executor:read_file", {"path": "references/long.md"}, ctx,
        )
    ]
    content = next(e.payload["content"] for e in events if e.kind == "result")
    assert "line2000" in content and "line2001" not in content   # 默认 2000 行一屏
    assert "offset=2001" in content                              # 尾部给出续读方式


async def test_read_file_pages_forward_by_offset() -> None:
    executor = SkillExecutorCapabilityProvider(_Registry([_LongFileSkillProvider()]))
    ctx = ProviderContext(session_id="s1", tenant_id="default", skill_name="local_skill__pdf")
    events = [
        ev async for ev in executor.invoke(
            "skill_executor:read_file", {"path": "x", "offset": 2001, "limit": 2}, ctx,
        )
    ]
    content = next(e.payload["content"] for e in events if e.kind == "result")
    assert "line2001" in content and "line2002" in content
    assert "line2000" not in content and "line2003" not in content


async def test_read_file_bad_window_is_tool_error_not_crash() -> None:
    executor = SkillExecutorCapabilityProvider(_Registry([_LongFileSkillProvider()]))
    ctx = ProviderContext(session_id="s1", tenant_id="default", skill_name="local_skill__pdf")
    events = [
        ev async for ev in executor.invoke(
            "skill_executor:read_file", {"path": "x", "offset": 0}, ctx,
        )
    ]
    assert any(e.kind == "error" and e.payload.get("code") == "SKILL_EXEC_ERROR" for e in events)


async def test_exec_schema_and_dispatch_use_command_only() -> None:
    provider = _FakeSkillProvider()
    executor = SkillExecutorCapabilityProvider(_Registry([provider]))
    ctx = ProviderContext(session_id="s1", task_id="t1", skill_name="local_skill__pdf")
    cap = next(c for c in await executor.list(ctx) if c.name == "exec_script")
    assert set(cap.input_schema["properties"]) == {"command"}
    assert cap.input_schema["required"] == ["command"]
    command = 'python "$SKILL_DIR/scripts/run.py" | formatter > out.txt'
    events = [event async for event in executor.invoke(
        "skill_executor:exec_script", {"command": command}, ctx,
    )]
    assert provider.exec_called_with == ("pdf", command, ctx)
    assert events[-1].kind == "result"
    assert events[-1].payload["content"] == "out"


class _ContextualProvider(_FakeSkillProvider):
    def __init__(self):
        super().__init__()
        self.names = {"a": "pdf", "b": "sheet"}
        self.calls = []

    async def list(self, ctx):
        import asyncio
        await asyncio.sleep(0)
        name = self.names.get(ctx.session_id)
        return [SkillCapability(id=f"local_skill:{name}", name=name)] if name else []

    async def list_files(self, skill_name, pattern, limit, ctx):
        self.calls.append((skill_name, ctx))
        return skill_name


async def test_each_operation_resolves_its_own_context_concurrently():
    import asyncio

    provider = _ContextualProvider()
    executor = SkillExecutorCapabilityProvider(_Registry([provider]))
    contexts = [ProviderContext(session_id=s, skill_name=f"local_skill__{name}")
                for s, name in [("a", "pdf"), ("b", "sheet"), ("a", "pdf")]]

    async def call(ctx):
        return [e async for e in executor.invoke("skill_executor:list_files", {}, ctx)]

    results = await asyncio.gather(*(call(ctx) for ctx in contexts))
    assert all(any(e.kind == "result" for e in events) for events in results)
    assert all(actual is expected for (_, actual), expected in zip(provider.calls, contexts))
    provider.names["a"] = "new"
    changed = ProviderContext(session_id="a", skill_name="local_skill__new")
    assert any(e.kind == "result" for e in await call(changed))
    assert any(e.payload.get("code") == "SKILL_NOT_FOUND" for e in await call(contexts[0]))


async def test_duplicate_reference_is_rejected_without_invoking_either_provider():
    first, second = _FakeSkillProvider(), _FakeSkillProvider()
    executor = SkillExecutorCapabilityProvider(_Registry([first, second]))
    ctx = ProviderContext(session_id="s", skill_name="local_skill__pdf")
    events = [e async for e in executor.invoke("skill_executor:list_files", {}, ctx)]
    assert events[-1].payload["code"] == "AMBIGUOUS_SKILL"
    assert first.list_files_called_with is None and second.list_files_called_with is None
