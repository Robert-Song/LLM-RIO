"""Keep unit tests independent of host credentials, configuration, and persistent state."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in os.environ:
        if name.startswith("LLMRIO_"):
            monkeypatch.delenv(name)
