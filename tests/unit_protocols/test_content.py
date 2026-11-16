"""Public content processing, independent of execution and blob namespaces."""

import ast
import base64
from copy import deepcopy
from pathlib import Path

import pytest

from ctx_weft import BlobStoreRequiredError, InvalidContentError
from ctx_weft.protocols.content import (
    ImagePart,
    TextPart,
    content_to_jsonable,
    externalize_content,
    normalize_content_parts,
    parse_content,
    validate_content,
)
from ctx_weft.protocols.context import ProviderContext


class Blobs:
    can_externalize = True

    def __init__(self, namespace):
        self.namespace = namespace
        self.rows = {}
        self.reads = []
        self.writes = []

    async def put(self, data, media_type, ctx):
        ref = f"blob:{self.namespace}-{len(self.rows)}"
        self.rows[ref] = (data, media_type)
        self.writes.append((data, media_type, ctx))
        return ref

    async def get(self, ref, ctx):
        self.reads.append((ref, ctx))
        return self.rows.get(ref)


def image(data=b"image"):
    return ImagePart(base64.b64encode(data).decode(), "image/png")


def test_protocol_content_has_no_core_dependency():
    import ctx_weft.protocols.content as content

    for module in (Path(content.__file__), Path(content.__file__).with_name("errors.py")):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("ctx_weft.core")
            if isinstance(node, ast.Import):
                assert not any(n.name.startswith("ctx_weft.core") for n in node.names)
    assert TextPart.__module__ == ImagePart.__module__ == "ctx_weft.protocols.content"


@pytest.mark.parametrize("raw", [42, {}, [object()], [{"type": "video"}]])
def test_new_input_rejects_unknown_shapes(raw):
    with pytest.raises(InvalidContentError):
        parse_content(raw)


def test_new_input_defaults_and_historical_unknown_parts_are_distinct():
    raw = [
        "hello",
        {"type": "text", "text": 3},
        {"type": "image", "data": "eA==", "media_type": "image/png"},
    ]
    saved = deepcopy(raw)
    parsed = parse_content(raw)
    assert parsed == [TextPart("hello"), TextPart("3"), image(b"x")]
    validate_content(parsed)
    assert raw == saved
    assert normalize_content_parts([{"type": "video"}, {"type": "text", "text": "saved"}]) == [
        TextPart("saved")
    ]


async def test_externalization_returns_detached_content_and_preserves_context():
    target = Blobs("event")
    ctx = ProviderContext(session_id="session", tenant_id="tenant")
    original = [TextPart("look"), image()]
    saved = deepcopy(original)
    result = await externalize_content(original, blob_store=target, ctx=ctx)
    assert original == saved
    assert result[0] == original[0] and result[0] is not original[0]
    assert result[1].source_type == "ref" and result[1].byte_size == 5
    assert target.writes == [(b"image", "image/png", ctx)]
    assert content_to_jsonable(result)[1]["data"] == "blob:event-0"


async def test_references_require_source_and_copy_between_separate_stores():
    source, target = Blobs("memory"), Blobs("event")
    ctx = ProviderContext("session", tenant_id="tenant")
    ref = await source.put(b"image", "image/png", ctx)
    content = [ImagePart(ref, "image/png", "ref")]
    with pytest.raises(InvalidContentError, match="explicit source"):
        await externalize_content(content, blob_store=target, ctx=ctx)
    assert not target.writes and not source.reads
    result = await externalize_content(
        content, blob_store=target, source_blob_store=source, ctx=ctx
    )
    assert result[0].data == "blob:event-0"
    assert content[0].data == ref and source.reads == [(ref, ctx)]
    assert await target.get(result[0].data, ctx) == (b"image", "image/png")
    assert await target.get(ref, ctx) is None


async def test_same_store_reference_is_verified_without_rewriting():
    store = Blobs("event")
    ctx = ProviderContext("session")
    ref = await store.put(b"image", "image/png", ctx)
    result = await externalize_content(
        [ImagePart(ref, "image/png", "ref")], blob_store=store, source_blob_store=store, ctx=ctx
    )
    assert len(store.writes) == 1 and store.reads == [(ref, ctx)]
    assert result[0].data == ref and result[0].byte_size == 5


async def test_missing_reference_does_not_search_target_namespace():
    source, target = Blobs("memory"), Blobs("event")
    ctx = ProviderContext("session")
    ref = await target.put(b"image", "image/png", ctx)
    with pytest.raises(InvalidContentError, match="unavailable"):
        await externalize_content(
            [ImagePart(ref, "image/png", "ref")],
            blob_store=target,
            source_blob_store=source,
            ctx=ctx,
        )
    assert not target.reads and len(target.writes) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        ImagePart("!", "image/png"),
        image(b"x" * (5 * 1024 * 1024 + 1)),
        ImagePart("eA==", "image/bmp"),
    ],
)
async def test_invalid_content_is_rejected_before_any_blob_write(invalid):
    target = Blobs("event")
    with pytest.raises(InvalidContentError):
        await externalize_content(
            [image(), invalid], blob_store=target, ctx=ProviderContext("session")
        )
    assert not target.writes


async def test_missing_target_store_rejects_images_but_accepts_text():
    from ctx_weft.protocols.events import NullEventBlobStore

    store = NullEventBlobStore()
    assert await externalize_content("plain", blob_store=store, ctx=ProviderContext("s")) == "plain"
    with pytest.raises(BlobStoreRequiredError):
        await externalize_content([image()], blob_store=store, ctx=ProviderContext("s"))


async def test_blob_failure_propagates_without_mutating_input():
    class Broken(Blobs):
        async def put(self, *args):
            raise OSError("storage unavailable")

    content = [image()]
    saved = deepcopy(content)
    with pytest.raises(OSError, match="storage unavailable"):
        await externalize_content(content, blob_store=Broken("event"), ctx=ProviderContext("s"))
    assert content == saved
