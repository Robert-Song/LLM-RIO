"""Offline, non-overwriting rebuild that retains credentials and model metadata."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from llm_rio.database_schema import SCHEMA
from llm_rio.profiles import profile_from_dict
from llm_rio.security import default_key_vault_path, verify_api_key

_PRESERVED_TABLES = (
    "quota_accounts",
    "api_keys",
    "model_catalog",
    "model_grants",
    "model_jobs",
    "model_verification_jobs",
    "model_profiles",
    "runtime_events",
)


class RebuildError(Exception):
    """A source cannot be rebuilt without violating the preservation contract."""


def _validate_credentials(rows: list[dict[str, Any]], vault_bytes: bytes) -> int:
    try:
        cipher = Fernet(vault_bytes.strip())
    except Exception as exc:
        raise RebuildError("The source API-key vault is invalid") from exc
    active = 0
    for row in rows:
        try:
            secret = cipher.decrypt(row["encrypted_api_key"].encode()).decode()
            if row["active"]:
                if not verify_api_key(row["token_hash"], secret):
                    raise ValueError("Authentication hash mismatch")
                active += 1
        except Exception as exc:
            raise RebuildError("A stored API key failed vault/hash validation") from exc
    return active


def _validate_profiles(rows: list[dict[str, Any]], excluded: set[str]) -> list[dict[str, Any]]:
    found = set()
    retained = []
    for row in rows:
        profile_id = str(row["id"])
        valid = True
        try:
            raw = json.loads(row["profile_json"])
            # Existing runtime loaders use the row ID for legacy upserted profiles.
            raw["id"] = profile_id
            profile = profile_from_dict(raw)
            if profile.model_id != row["model_id"] or row["active"] not in (0, 1):
                valid = False
        except Exception:
            valid = False
        if profile_id in excluded:
            if valid:
                raise RebuildError(f"Refusing to exclude a valid profile: {profile_id}")
            found.add(profile_id)
        elif not valid:
            raise RebuildError(
                f"Invalid profile {profile_id}; investigate before explicitly excluding it"
            )
        else:
            retained.append(row)
    if found != excluded:
        raise RebuildError("An explicitly excluded profile was not found in the source")
    return retained


def _write_new_database(
    database: sqlite3.Connection, destination: Path, vault_bytes: bytes
) -> None:
    vault = default_key_vault_path(destination)
    outputs = [destination, vault, Path(f"{destination}-wal"), Path(f"{destination}-shm")]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise RebuildError("Destination database, vault, or sidecar already exists; use a new path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rio-rebuild-", dir=destination.parent) as temp:
        staged_database = Path(temp) / "database"
        staged_vault = Path(temp) / "vault"
        with closing(sqlite3.connect(staged_database)) as output:
            database.backup(output)
        os.chmod(staged_database, 0o600)
        staged_vault.write_bytes(vault_bytes)
        os.chmod(staged_vault, 0o600)
        for path in (staged_database, staged_vault):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        # Publish without replacing existing files, even if another process races us.
        os.link(staged_vault, vault)
        try:
            os.link(staged_database, destination)
        except BaseException:
            vault.unlink()
            raise


def rebuild_database(
    source: Path,
    *,
    destination: Path | None = None,
    exclude_profiles: set[str] | None = None,
    server_stopped: bool = False,
) -> dict[str, Any]:
    """Validate in memory; optionally create a new database and matching vault.

    This never repairs or replaces the source. An unreadable required table aborts
    the rebuild; it does not silently restore older backups or omit model metadata.
    """
    source = source.resolve(strict=True)
    if destination is not None:
        if not server_stopped:
            raise RebuildError("Stop the owning service and workers, then use --server-stopped")
        destination = destination.absolute()
        if destination.resolve() == source:
            raise RebuildError("Source and destination must be different paths")
    vault_bytes = default_key_vault_path(source).read_bytes()
    with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        records = {
            table: [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]
            for table in _PRESERVED_TABLES
        }
        state = connection.execute("SELECT * FROM service_state WHERE singleton=1").fetchone()
        fingerprint = state["machine_fingerprint"] if state is not None else None
    # Invalid optional/history JSON must not silently poison preserved model metadata.
    for table, rows in records.items():
        if table == "model_profiles":
            continue  # Explicit exclusions are validated separately below.
        for row in rows:
            for column, value in row.items():
                if column.endswith("_json") and value is not None:
                    try:
                        json.loads(value)
                    except (TypeError, ValueError) as exc:
                        raise RebuildError(
                            f"Invalid JSON in required field {table}.{column}"
                        ) from exc
    active_keys = _validate_credentials(records["api_keys"], vault_bytes)
    excluded = exclude_profiles or set()
    records["model_profiles"] = _validate_profiles(records["model_profiles"], excluded)
    now = datetime.now(UTC).isoformat()
    # Clear baselines with history so new usage is counted from the first request.
    for account in records["quota_accounts"]:
        account["balance_tokens"] = account["limit_tokens"]
        account["usage_baseline_tokens"] = 0
        account["usage_reset_at"] = now

    with closing(sqlite3.connect(":memory:")) as clean:
        clean.execute("PRAGMA foreign_keys=ON")
        clean.executescript(SCHEMA)
        clean.execute("PRAGMA defer_foreign_keys=ON")
        with clean:
            for table, rows in records.items():
                if not rows:
                    continue
                columns = list(rows[0])
                names = ",".join('"' + col.replace('"', '""') + '"' for col in columns)
                placeholders = ",".join("?" for _ in columns)
                clean.executemany(
                    f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})',
                    [[row[column] for column in columns] for row in rows],
                )
            clean.execute(
                "UPDATE service_state SET machine_fingerprint=?, mode='ACTIVE', updated_at=?",
                (fingerprint, now),
            )
        if clean.execute("PRAGMA foreign_key_check").fetchall():
            raise RebuildError("Preserved records contain broken foreign-key references")
        if clean.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise RebuildError("Rebuilt database failed its integrity check")
        size = (
            clean.execute("PRAGMA page_count").fetchone()[0]
            * clean.execute("PRAGMA page_size").fetchone()[0]
        )
        if destination is not None:
            _write_new_database(clean, destination, vault_bytes)
        return {
            "source": str(source),
            "destination": str(destination) if destination else None,
            "vault": str(default_key_vault_path(destination)) if destination else None,
            "preserved_rows": {table: len(rows) for table, rows in records.items()},
            "excluded_profiles": sorted(excluded),
            "active_keys_verified": active_keys,
            "database_bytes": size,
            "integrity_check": "ok",
            "foreign_key_check": "ok",
            "quota_reset": "balances restored to configured limits; usage baselines cleared",
            "history_reset": "requests, reservations, ledger, summaries, and workers",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--destination", type=Path)
    parser.add_argument("--exclude-profile", action="append", default=[])
    parser.add_argument("--server-stopped", action="store_true")
    args = parser.parse_args()
    try:
        report = rebuild_database(
            args.source,
            destination=args.destination,
            exclude_profiles=set(args.exclude_profile),
            server_stopped=args.server_stopped,
        )
    except (RebuildError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"Rebuild aborted: {exc}\nThe source database was not modified.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
