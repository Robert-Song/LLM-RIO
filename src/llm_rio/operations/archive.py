"""Archive stopped beta installations without modifying the source database."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path


def archive(source: Path, destination: Path, config: Path | None = None) -> dict[str, object]:
    source = source.resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    try:
        with (
            closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as connection,
            closing(sqlite3.connect(destination / source.name)) as backup,
        ):
            connection.backup(backup)
            if backup.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RuntimeError("Archive failed SQLite integrity check")
            backup.row_factory = sqlite3.Row
            catalog = [
                dict(row)
                for row in backup.execute(
                    "SELECT nickname, huggingface_repo, resolved_revision, "
                    "artifact_path FROM model_catalog"
                )
            ]
        vault = source.parent / f".{source.stem}-api-key-vault"
        if vault.exists():
            shutil.copy2(vault, destination / vault.name)
        if config is not None:
            shutil.copy2(config, destination / "config.toml")
        (destination / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
        with (destination / "catalog.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["nickname", "huggingface_repo", "resolved_revision", "artifact_path"],
            )
            writer.writeheader()
            writer.writerows(catalog)
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        hashes = {}
        for path in destination.iterdir():
            if path.is_file():
                with path.open("rb") as stream:
                    hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        result: dict[str, object] = {
            "source": str(source),
            "revision": revision,
            "models": len(catalog),
            "sha256": hashes,
        }
        (destination / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        for path in destination.iterdir():
            os.chmod(path, 0o600)
        return result
    except BaseException:
        # Keep incomplete output for diagnosis. Never publish it as a valid archive.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--server-stopped", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.source, args.destination, args.config), indent=2))


if __name__ == "__main__":
    main()
