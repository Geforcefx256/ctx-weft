"""操作账本 conformance（spec: tool-operations；wp5-2.2）。

参数化内存 / SQLite：prepare 幂等（同 id 同内容 no-op、异绑定拒）、CAS 串行
（revision 拒后到者、状态机不倒退不跳跃）、get/全字段往返（含 parts 与 blob ref）。
"""

from __future__ import annotations

import pytest

from ctx_weft.protocols.context import ProviderContext
from ctx_weft.core.utils.ids import mint_call_id
from ctx_weft.protocols.capability import (
    OperationRecord,
    OperationStatus,
    OperationUpdate,
    RevisionConflict,
)
from ctx_weft.providers.operations import InMemoryOperationStore

_CTX = ProviderContext(session_id="s1", tenant_id="t1")


def _rec(ordinal: int = 0, **over) -> OperationRecord:
    base = dict(
        tool_call_id=mint_call_id(anchor="rec1", ordinal=ordinal,
                                  raw_id=f"call_{ordinal}", turn_seq=0),
        tenant_id="t1", session_id="s1", agent_id="a1",
        assistant_record_id="rec1", tool_ordinal=ordinal,
        tool_name="mcp:web:fetch", args_hash="hash1",
    )
    base.update(over)
    return OperationRecord(**base)


@pytest.fixture(params=["in_memory", "sqlite"])
async def store(request, tmp_path):
    if request.param == "in_memory":
        yield InMemoryOperationStore()
    else:
        from ctx_weft.providers.operations.sql import open_sqlite_operation_store
        async with open_sqlite_operation_store(tmp_path / "ops.sqlite") as s:
            yield s


async def test_prepare_idempotent_same_identity(store):
    r1 = await store.prepare(_rec(), _CTX)
    assert r1.status == OperationStatus.PREPARED and r1.revision == 1
    r2 = await store.prepare(_rec(), _CTX)          # 同 id 同身份 → no-op
    assert r2.tool_call_id == r1.tool_call_id
    with pytest.raises(ValueError):                  # 同 id 异身份 → 拒（身份被复用）
        await store.prepare(_rec(tool_name="other:tool"), _CTX)


async def test_cas_serializes_and_rejects_stale(store):
    await store.prepare(_rec(), _CTX)
    r = await store.compare_and_set(
        _rec().tool_call_id, 1,
        OperationUpdate(status=OperationStatus.STARTED, append_attempt="inv1"), _CTX)
    assert r.status == OperationStatus.STARTED and r.revision == 2
    assert r.attempts == ["inv1"]
    with pytest.raises(RevisionConflict):            # 携旧 revision 的后到者被拒
        await store.compare_and_set(
            _rec().tool_call_id, 1,
            OperationUpdate(status=OperationStatus.COMPLETED), _CTX)


async def test_state_machine_no_backward_no_skip(store):
    await store.prepare(_rec(), _CTX)
    # prepared→completed 放行（极短操作原子完结，_LEGAL 刻意允许）——先推进到 started
    await store.compare_and_set(
        _rec().tool_call_id, 1, OperationUpdate(status=OperationStatus.STARTED), _CTX)
    with pytest.raises(ValueError):                  # started → prepared 倒退
        await store.compare_and_set(
            _rec().tool_call_id, 2, OperationUpdate(status=OperationStatus.PREPARED), _CTX)
    await store.compare_and_set(                     # started → waiting_human 合法
        _rec().tool_call_id, 2, OperationUpdate(status=OperationStatus.WAITING_HUMAN), _CTX)
    with pytest.raises(ValueError):                  # waiting_human → prepared 非法
        await store.compare_and_set(
            _rec().tool_call_id, 3, OperationUpdate(status=OperationStatus.PREPARED), _CTX)


async def test_full_roundtrip_with_parts_and_blob_ref(store):
    rec = _rec(ordinal=1)
    await store.prepare(rec, _CTX)
    await store.compare_and_set(
        rec.tool_call_id, 1, OperationUpdate(status=OperationStatus.STARTED), _CTX)
    result = {"content": [{"type": "image", "ref": "blob:abc123"}]}   # blob ref 形态
    done = await store.compare_and_set(
        rec.tool_call_id, 2,
        OperationUpdate(status=OperationStatus.COMPLETED, result=result, result_set=True,
                        error=None), _CTX)
    assert done.status == OperationStatus.COMPLETED
    got = await store.get(rec.tool_call_id, _CTX)
    assert got is not None
    assert got.result == result                      # 完整结果往返（非截断文本）
    assert got.memory_result_id == ""                # gateway 接线时填
    # 终态冻结
    with pytest.raises(ValueError):
        await store.compare_and_set(
            rec.tool_call_id, got.revision,
            OperationUpdate(status=OperationStatus.STARTED), _CTX)
