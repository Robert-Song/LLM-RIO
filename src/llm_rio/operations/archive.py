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

from cryptography.fernet import Fernet

from llm_rio.operations.ownership import database_resource, owner_lock
from llm_rio.security import verify_api_key


def archive(source: Path, destination: Path, config: Path | None = None) -> dict[str, object]:
    if config is None or not config.is_file():
        raise ValueError("A complete rollback archive requires the matching --config file")
    with owner_lock(database_resource(source)):
        return _archive(source, destination, config)


def _archive(source: Path, destination: Path, config: Path) -> dict[str, object]:
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
        shutil.copy2(config, destination / "config.toml")
        with closing(sqlite3.connect(destination / source.name)) as restored:
            if restored.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='api_keys'"
            ).fetchone():
                keys = restored.execute(
                    "SELECT id, active, nickname, token_prefix, encrypted_api_key, token_hash "
                    "FROM api_keys"
                )
                rows = keys.fetchall()
                if rows:
                    if not (destination / vault.name).is_file():
                        raise RuntimeError("Credential vault is missing; archive is incomplete")
                    decryptor = Fernet((destination / vault.name).read_bytes().strip())
                    for key_id, active, nickname, prefix, encrypted, token_hash in rows:
                        plaintext = (
                            decryptor.decrypt(encrypted.encode()).decode() if encrypted else None
                        )
                        tombstone = f"deleted-{key_id}"
                        deleted = not active and all(
                            value == tombstone
                            for value in (nickname, prefix, token_hash, plaintext)
                        )
                        if not deleted and (
                            plaintext is None or not verify_api_key(token_hash, plaintext)
                        ):
                            raise RuntimeError(
                                "Archived credentials do not match the vault and hashes"
                            )
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
            "credential_restore_verified": True,
            "ownership_verified": True,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server-stopped", action="store_true", required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.source, args.destination, args.config), indent=2))


if __name__ == "__main__":
    main()
