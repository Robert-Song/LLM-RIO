from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from typing import Any
from unittest.mock import AsyncMock

import pytest

from llm_rio import process_cleanup as cleanup


@pytest.mark.parametrize("gpu_delay", [False, True])
async def test_waits_after_sigkill_for_descendants_and_gpu_release(monkeypatch, gpu_delay):
    process = AsyncMock(pid=991122, returncode=0)
    killed = False
    checks = 0

    def killpg(group, sig):
        nonlocal killed
        assert group == process.pid
        if sig == signal.SIGKILL:
            killed = True

    def members(group):
        nonlocal checks
        checks += 1
        return {process.pid + 1: "Z" if killed else "S"}

    def gpu_pids(_):
        return {process.pid + 1} if gpu_delay and checks < 5 else set()

    monkeypatch.setattr(cleanup.os, "killpg", killpg)
    monkeypatch.setattr(cleanup, "group_members", members)
    monkeypatch.setattr(cleanup, "gpu_pids", gpu_pids)
    await cleanup.terminate_engine(process, grace_seconds=0, kill_seconds=1)
    assert killed
    assert checks >= (5 if gpu_delay else 4)
    process.wait.assert_awaited_once()


async def test_unreleased_gpu_or_unavailable_telemetry_fails_closed(monkeypatch):
    process = AsyncMock(pid=991122)
    monkeypatch.setattr(cleanup.os, "killpg", lambda *_: None)
    monkeypatch.setattr(cleanup, "group_members", lambda _: {})
    for reader in (lambda _: {process.pid}, lambda _: (_ for _ in ()).throw(OSError("NVML"))):
        monkeypatch.setattr(cleanup, "gpu_pids", reader)
        with pytest.raises(cleanup.TeardownError):
            await cleanup.terminate_engine(process, grace_seconds=0, kill_seconds=0)
    process.wait.assert_not_awaited()


async def test_cancellation_waits_for_verified_cleanup(monkeypatch):
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def terminate(*args: Any, **kwargs: Any):
        entered.set()
        await finish.wait()

    monkeypatch.setattr(cleanup, "_terminate", terminate)
    task = asyncio.create_task(cleanup.terminate_engine(AsyncMock()))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.skipif(sys.platform != "linux", reason="Linux engine sessions")
async def test_real_orphan_child_ignoring_sigterm_is_killed(tmp_path):
    child_file = tmp_path / "child"
    code = """import os, signal, time, sys
pid = os.fork()
if pid:
    sys.exit(0)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], 'w').write(str(os.getpid()))
while True: time.sleep(1)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code, str(child_file), start_new_session=True
    )
    try:
        for _ in range(100):
            if child_file.exists():
                break
            await asyncio.sleep(0.01)
        child = int(child_file.read_text())
        await process.wait()
        await cleanup.terminate_engine(process, grace_seconds=0.05, kill_seconds=2)
        assert cleanup.group_members(process.pid).get(child) in (None, "Z", "X")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
