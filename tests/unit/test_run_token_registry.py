"""Per-run token registry：随派发登记、随 run 注销；_pausing 闩锁下出生即 paused。"""

import pytest

from ctx_weft.core import CtxWeftRuntime
from ctx_weft.core.control import RunTokens
from ctx_weft.providers.llm.adapters.mock import MockLLMAdapter
from tests.integration.test_minimal_loop import InlineAgentTemplateProvider, make_runtime



def _rt():
    return make_runtime(llm=MockLLMAdapter(responses=[]),
                          agent_provider=InlineAgentTemplateProvider())


@pytest.mark.asyncio
async def test_register_and_deregister_run_tokens():
    rt = _rt()
    tokens = rt._register_run_tokens("s1", "t1")
    assert isinstance(tokens, RunTokens)
    assert rt._run_tokens["s1"]["t1"] is tokens
    assert not tokens.pause.is_paused and not tokens.cancel.is_cancelled
    rt._deregister_run_tokens("s1", "t1")
    assert "s1" not in rt._run_tokens          # 空桶随手回收


@pytest.mark.asyncio
async def test_born_paused_under_pausing_latch():
    rt = _rt()
    rt._pausing.add("s1")
    tokens = rt._register_run_tokens("s1", "t1")
    assert tokens.pause.is_paused is True


@pytest.mark.asyncio
async def test_root_resume_point_claimed_once_under_pausing_latch():
    # 续跑点一次性名额：闩锁窗口内第一个 root run born-pause 并认领名额，其后的 root run
    # 一律 born-cancel——root agent 同时有多个任务（后继任务/多条消息）时，防止"第一个
    # park 后、重排出的另一个 root 任务再 park"出第二个气泡。
    rt = _rt()
    rt._pausing.add("s1")
    first = rt._register_run_tokens("s1", "t1", root_run=True)
    second = rt._register_run_tokens("s1", "t2", root_run=True)
    assert first.pause.is_paused and not first.cancel.is_cancelled
    assert second.cancel.is_cancelled and not second.pause.is_paused


@pytest.mark.asyncio
async def test_pause_claim_cleared_with_latch():
    # 名额随闩锁一起回收：本轮暂停的认领不得泄漏到同 session 的下一轮暂停。
    rt = _rt()
    rt._pausing.add("s1")
    rt._register_run_tokens("s1", "t1", root_run=True)   # 认领
    rt._release_round("s1")   # 一轮跑完的轻量清理（2026-09-08 前叫 _release_session）
    assert "s1" not in rt._pausing
    assert "s1" not in rt._pause_claimed
    rt._pausing.add("s1")                                 # 下一轮暂停
    again = rt._register_run_tokens("s1", "t2", root_run=True)
    assert again.pause.is_paused is True                 # 名额可再次认领


@pytest.mark.asyncio
async def test_non_root_run_born_cancel_not_paused_under_pausing_latch():
    # M-1：闩锁窗口内重排出的非 root agent run（多级委派被 _try_resume_parent 唤醒的中间
    # 父任务）须出生即 cancel、绝不 born-pause——检查点 pause 先于 cancel，双信号会让
    # 中间 agent park 出气泡抢走续跑点。cancel 后逐级级联，直到 root agent 的任务 park。
    rt = _rt()
    rt._pausing.add("s1")
    tokens = rt._register_run_tokens("s1", "t1", root_run=False)
    assert tokens.cancel.is_cancelled is True
    assert tokens.pause.is_paused is False
