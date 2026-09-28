"""Identify the application source actually imported by the running service."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path


def application_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return {"version": version("llm-rio"), "source_sha256": digest.hexdigest()}
