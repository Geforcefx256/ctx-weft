"""PauseToken.wait_paused resolves when paused, blocks while un-paused."""

import asyncio

import pytest

from ctx_weft.core.control.tokens import PauseToken



@pytest.mark.asyncio
async def test_wait_paused_resolves_after_pause():
    tok = PauseToken()
    waiter = asyncio.ensure_future(tok.wait_paused())
    await asyncio.sleep(0)
    assert not waiter.done()          # un-paused → blocks
    tok.pause()
    await asyncio.wait_for(waiter, timeout=1.0)   # resolves once paused


@pytest.mark.asyncio
async def test_wait_paused_returns_immediately_if_already_paused():
    tok = PauseToken()
    tok.pause()
    await asyncio.wait_for(tok.wait_paused(), timeout=1.0)
