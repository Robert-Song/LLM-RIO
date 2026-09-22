"""Identity and change detection for administrator-selected local artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def local_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    files = (
        [path]
        if path.is_file()
        else sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    )
    if not files:
        raise ValueError("Local model artifact is empty")
    records = []
    for item in files:
        before = item.stat()
        with item.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        after = item.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Artifact changed while hashing; retry registration")
        records.append(
            {
                "path": item.name if path.is_file() else str(item.relative_to(path)),
                "bytes": after.st_size,
                "mtime_ns": after.st_mtime_ns,
                "sha256": digest,
            }
        )
    identity = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"revision": f"local:{identity}", "download_bytes": 0, "artifact_hashes": records}


def local_artifact_unchanged(path: Path, records: list[dict[str, Any]]) -> bool:
    try:
        root = path.resolve(strict=True)
        files = (
            [root]
            if root.is_file()
            else [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts]
        )
        names = {p.name if root.is_file() else str(p.relative_to(root)) for p in files}
        if names != {record["path"] for record in records}:
            return False
        for record in records:
            item = root if root.is_file() else root / record["path"]
            stat = item.stat()
            if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
                return False
        return True
    except (OSError, KeyError, TypeError):
        return False
