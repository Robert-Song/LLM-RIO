from __future__ import annotations

import asyncio
import os
import signal
from typing import Any, cast

import pytest

from llm_rio.validation import ProfileValidator


class FinishedProcess:
    pid = 43123
    returncode = 0

    def terminate(self) -> None:
        raise AssertionError("finished parent must not receive a direct terminate")

    def kill(self) -> None:
        raise AssertionError("finished parent must not receive a direct kill")

    async def wait(self) -> int:
        return 0


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_terminate_signals_group_after_parent_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    def killpg(process_group: int, requested_signal: int) -> None:
        signals.append((process_group, requested_signal))
        if requested_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg)

    await ProfileValidator._terminate(cast(Any, FinishedProcess()))

    assert signals == [
        (FinishedProcess.pid, signal.SIGTERM),
        (FinishedProcess.pid, 0),
    ]


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_terminate_kills_group_that_outlives_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    def killpg(process_group: int, requested_signal: int) -> None:
        signals.append((process_group, requested_signal))

    async def no_delay(_seconds: float) -> None:
        return None

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    await ProfileValidator._terminate(cast(Any, FinishedProcess()))

    assert signals[0] == (FinishedProcess.pid, signal.SIGTERM)
    assert signals[1:-1] == [(FinishedProcess.pid, 0)] * 50
