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

    monkeypatch.setenv("LLMRIO_SERVING_MODE", "vllm-sleep")

    # Unit tests use a version-only executable, never a live GPU engine.
    binary_dir = tmp_path / "unit-bin"
    binary_dir.mkdir()
    for name in ("vllm", "llama-server"):
        executable = binary_dir / name
        executable.write_text("#!/bin/sh\nprintf 'unit-test-engine-1\\n'\n")
        executable.chmod(0o700)
    monkeypatch.setenv("PATH", str(binary_dir) + os.pathsep + os.environ["PATH"])
