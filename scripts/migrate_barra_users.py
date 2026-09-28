"""Copy current release users and API keys to Barra without usage history.

The default invocation is read-only. ``--apply`` first archives the existing Barra
release database/config/vault, then copies identities, quota state, and grants from a
consistent current-release snapshot. API keys retain their values and are encrypted
with Barra's own vault. Request, reservation, worker, and usage history is not copied.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tomllib
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from llm_rio.operations.ownership import database_resource, owner_lock
from llm_rio.security import default_key_vault_path, verify_api_key

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATABASE = ROOT / "state/release/llm-rio.db"
SOURCE_VAULT = default_key_vault_path(SOURCE_DATABASE)
TARGET_CONFIG = ROOT / "config.barra.release.toml"
TARGET_DATABASE = ROOT / "state_barra/release/llm-rio.db"
TARGET_VAULT = default_key_vault_path(TARGET_DATABASE)
ARCHIVE_ROOT = ROOT / "db_backup"
KEY_COLUMNS = (
    "id",
    "nickname",
    "role",
    "quota_account_id",
    "token_prefix",
    "token_hash",
    "encrypted_api_key",
    "active",
    "created_at",
    "last_used_at",
)
ACCOUNT_COLUMNS = (
    "id",
    "nickname",
    "balance_tokens",
    "limit_tokens",
    "usage_baseline_tokens",
    "usage_reset_at",
    "unlimited",
    "created_at",
)
HISTORY_TABLES = (
    "inference_requests",
    "quota_reservations",
    "quota_ledger",
    "usage_summaries",
    "usage_summary_periods",
    "workers",
    "runtime_events",
    "model_jobs",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _immutable_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _snapshot_memory(path: Path) -> sqlite3.Connection:
    source = _read_only(path)
    snapshot = sqlite3.connect(":memory:")
    snapshot.row_factory = sqlite3.Row
    try:
        source.backup(snapshot)
    finally:
        source.close()
    if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        snapshot.close()
        raise ValueError("A source database snapshot failed its integrity check")
    return snapshot


def _model_identity(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str]:
    repo = str(row["huggingface_repo"] or "").strip()
    revision = str(row["resolved_revision"] or "").strip()
    if not repo or not revision:
        raise ValueError("A model has no pinned source identity")
    return repo, revision


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])


def _validate_source(
    source: sqlite3.Connection,
    source_vault_path: Path,
    target: sqlite3.Connection,
    target_vault_path: Path,
) -> dict[str, Any]:
    if source.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Current user source is not release schema version 1")
    if target.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Barra target is not release schema version 1")
    if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Current release source failed its integrity check")
    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Barra target failed its integrity check")
    if source.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Current release source has foreign-key errors")
    if target.execute("PRAGMA foreign_key_check").fetchall():
        raise ValueError("Barra target has foreign-key errors")
    if not source_vault_path.is_file() or not target_vault_path.is_file():
        raise ValueError("Both current-release and Barra key vaults are required")

    expected_target = {
        "api_keys": 1,
        "quota_accounts": 1,
        "model_catalog": 45,
        "model_profiles": 72,
        "model_grants": 0,
        "model_jobs": 0,
        "inference_requests": 0,
        "quota_reservations": 0,
        "quota_ledger": 0,
        "usage_summaries": 0,
        "usage_summary_periods": 0,
        "workers": 0,
    }
    actual_target = {table: _table_count(target, table) for table in expected_target}
    if actual_target != expected_target:
        raise ValueError("Barra release state changed; refusing to replace its identities")
    event_types = {
        str(row[0]): int(row[1])
        for row in target.execute(
            "SELECT event_type,count(*) FROM runtime_events GROUP BY event_type"
        )
    }
    if event_types and (
        set(event_types) != {"MACHINE_INVENTORY_DISCOVERED", "QUEUE_ENABLED"}
        or any(count < 1 for count in event_types.values())
    ):
        raise ValueError("Barra runtime history changed; refusing to replace its identities")

    pending = int(
        source.execute("SELECT count(*) FROM quota_reservations WHERE state='RESERVED'").fetchone()[
            0
        ]
    )
    if pending:
        raise ValueError("Current release has unsettled reservations; wait for requests to finish")

    source_keys = source.execute("SELECT * FROM api_keys ORDER BY created_at,id").fetchall()
    source_accounts = source.execute("SELECT * FROM quota_accounts ORDER BY id").fetchall()
    if not source_keys or not source_accounts:
        raise ValueError("Current release has no user credentials to migrate")
    key_ids = {str(row["id"]) for row in source_keys}
    account_ids = {str(row["id"]) for row in source_accounts}
    if len(key_ids) != len(source_keys) or len(account_ids) != len(source_accounts):
        raise ValueError("Current release contains duplicate identity rows")
    if any(str(row["quota_account_id"]) not in account_ids for row in source_keys):
        raise ValueError("A current release API key has no quota account")

    source_fernet = Fernet(source_vault_path.read_bytes().strip())
    plaintext_by_id: dict[str, str] = {}
    active_admins = 0
    for row in source_keys:
        key_id = str(row["id"])
        try:
            token = source_fernet.decrypt(str(row["encrypted_api_key"]).encode()).decode()
        except (ValueError, TypeError) as exc:
            raise ValueError("A current release API key cannot be decrypted") from exc
        deleted = (
            not row["active"]
            and str(row["nickname"]) == f"deleted-{key_id}"
            and str(row["token_prefix"]) == f"deleted-{key_id}"
            and str(row["token_hash"]) == f"deleted-{key_id}"
            and token == f"deleted-{key_id}"
        )
        if not deleted and (
            not token.startswith("rio_")
            or str(row["token_prefix"]) != token[:24]
            or not verify_api_key(str(row["token_hash"]), token)
        ):
            raise ValueError("A current release API key failed credential verification")
        if row["active"] and row["role"] == "admin":
            active_admins += 1
        plaintext_by_id[key_id] = token
    if not active_admins:
        raise ValueError("Current release has no active administrator key")

    source_models = {
        _model_identity(row): row
        for row in source.execute("SELECT * FROM model_catalog").fetchall()
    }
    target_models = {
        _model_identity(row): row
        for row in target.execute("SELECT * FROM model_catalog").fetchall()
    }
    if len(source_models) != _table_count(source, "model_catalog"):
        raise ValueError("Current release catalog has duplicate model identities")
    if len(target_models) != _table_count(target, "model_catalog"):
        raise ValueError("Barra catalog has duplicate model identities")
    target_id_by_identity = {identity: str(row["id"]) for identity, row in target_models.items()}
    source_creator_by_target_id: dict[str, str] = {}
    for identity, target_model in target_models.items():
        source_model = source_models.get(identity)
        if source_model is None:
            raise ValueError("A Barra model is absent from the current release catalog")
        creator_id = str(source_model["created_by_key_id"])
        if creator_id not in key_ids:
            raise ValueError("A model creator is absent from the current release key set")
        source_creator_by_target_id[str(target_model["id"])] = creator_id

    grants: list[tuple[str, str, str]] = []
    dropped_grants = 0
    for grant in source.execute(
        "SELECT key_id,model_id,created_at FROM model_grants ORDER BY key_id,model_id"
    ):
        key_id = str(grant["key_id"])
        if key_id not in key_ids:
            raise ValueError("A model grant references an API key that cannot be migrated")
        source_model = source.execute(
            "SELECT huggingface_repo,resolved_revision FROM model_catalog WHERE id=?",
            (grant["model_id"],),
        ).fetchone()
        if source_model is None:
            raise ValueError("A model grant references an unknown source model")
        identity = _model_identity(source_model)
        target_model_id = target_id_by_identity.get(identity)
        if target_model_id is None:
            dropped_grants += 1
            continue
        grants.append((key_id, target_model_id, str(grant["created_at"])))

    current_target_key_ids = {str(row[0]) for row in target.execute("SELECT id FROM api_keys")}
    if key_ids & current_target_key_ids:
        raise ValueError("Current release and Barra contain colliding API-key IDs")
    return {
        "source_keys": source_keys,
        "source_accounts": source_accounts,
        "plaintext_by_id": plaintext_by_id,
        "grants": grants,
        "source_creator_by_target_id": source_creator_by_target_id,
        "dropped_grants": dropped_grants,
        "active_admins": active_admins,
        "source_active_keys": sum(bool(row["active"]) for row in source_keys),
        "source_inactive_keys": sum(not bool(row["active"]) for row in source_keys),
        "source_key_roles": dict(Counter(str(row["role"]) for row in source_keys if row["active"])),
        "target_before": actual_target,
        "target_startup_events": sum(event_types.values()),
    }


def _archive_target(
    target_path: Path,
    config_path: Path,
    vault_path: Path,
    archive_root: Path,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    archive = archive_root / f"barra-before-api-key-migration-{stamp}"
    archive.mkdir(parents=True, mode=0o700, exist_ok=False)
    database_copy = archive / target_path.name
    source = _read_only(target_path)
    try:
        with closing(sqlite3.connect(database_copy)) as destination:
            source.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Barra pre-key-migration snapshot failed integrity check")
    finally:
        source.close()
    shutil.copyfile(config_path, archive / config_path.name)
    shutil.copyfile(vault_path, archive / vault_path.name)
    manifest = {
        "purpose": "Rollback snapshot before copying current release users to Barra",
        "created_at": _now(),
        "actor": f"unix:{getpass.getuser()}",
        "files_sha256": {
            path.name: _sha256(path)
            for path in (database_copy, archive / config_path.name, archive / vault_path.name)
        },
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    manifest["git_revision"] = (
        revision.stdout.strip() if revision.returncode == 0 else "unavailable"
    )
    (archive / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in archive.iterdir():
        os.chmod(path, 0o600)
    os.chmod(archive, 0o700)
    saved = _immutable_read_only(database_copy)
    try:
        if saved.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Barra rollback snapshot failed final integrity verification")
        archived_vault = Fernet((archive / vault_path.name).read_bytes().strip())
        for row in saved.execute("SELECT encrypted_api_key FROM api_keys"):
            archived_vault.decrypt(str(row[0]).encode())
    finally:
        saved.close()
    checked = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    if any(_sha256(archive / name) != digest for name, digest in checked["files_sha256"].items()):
        raise ValueError("Barra rollback snapshot failed its file-hash verification")
    return archive


def _apply(
    target_path: Path,
    plan: dict[str, Any],
    target_fernet: Fernet,
) -> dict[str, Any]:
    target = sqlite3.connect(target_path, isolation_level=None)
    target.row_factory = sqlite3.Row
    target.execute("PRAGMA foreign_keys=ON")
    target.execute("BEGIN IMMEDIATE")
    try:
        target.execute("PRAGMA defer_foreign_keys=ON")
        target.execute("DELETE FROM runtime_events")
        target.execute("DELETE FROM model_grants")
        target.execute("DELETE FROM api_keys")
        target.execute("DELETE FROM quota_accounts")

        for row in plan["source_accounts"]:
            target.execute(
                "INSERT INTO quota_accounts"
                "(id,nickname,balance_tokens,limit_tokens,usage_baseline_tokens,usage_reset_at,unlimited,created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                tuple(row[name] for name in ACCOUNT_COLUMNS),
            )
        for row in plan["source_keys"]:
            key_id = str(row["id"])
            values = {name: row[name] for name in KEY_COLUMNS}
            values["encrypted_api_key"] = target_fernet.encrypt(
                plan["plaintext_by_id"][key_id].encode()
            ).decode()
            target.execute(
                "INSERT INTO api_keys"
                "(id,nickname,role,quota_account_id,token_prefix,token_hash,encrypted_api_key,active,created_at,last_used_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                tuple(values[name] for name in KEY_COLUMNS),
            )
        for model_id, creator_id in plan["source_creator_by_target_id"].items():
            target.execute(
                "UPDATE model_catalog SET created_by_key_id=? WHERE id=?",
                (creator_id, model_id),
            )
        for key_id, model_id, created_at in plan["grants"]:
            target.execute(
                "INSERT INTO model_grants(key_id,model_id,created_at) VALUES (?,?,?)",
                (key_id, model_id, created_at),
            )
        if target.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Migrated Barra users violate a foreign-key constraint")
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Migrated Barra users failed database integrity check")

        target_fernet_value_by_id: dict[str, str] = {}
        for row in target.execute("SELECT id,token_hash,encrypted_api_key FROM api_keys"):
            key_id = str(row["id"])
            token = target_fernet.decrypt(str(row["encrypted_api_key"]).encode()).decode()
            if token != plan["plaintext_by_id"][key_id]:
                raise ValueError("A migrated API key value changed during vault encryption")
            if row["token_hash"] != next(
                str(source_row["token_hash"])
                for source_row in plan["source_keys"]
                if str(source_row["id"]) == key_id
            ):
                raise ValueError("A migrated API-key hash changed")
            if token.startswith("rio_") and not verify_api_key(str(row["token_hash"]), token):
                raise ValueError("A migrated API key failed its stored hash verification")
            target_fernet_value_by_id[key_id] = token
        if set(target_fernet_value_by_id) != set(plan["plaintext_by_id"]):
            raise ValueError("Migrated API-key identities differ from the source snapshot")

        counts = {
            "api_keys": int(target.execute("SELECT count(*) FROM api_keys").fetchone()[0]),
            "active_api_keys": int(
                target.execute("SELECT count(*) FROM api_keys WHERE active=1").fetchone()[0]
            ),
            "quota_accounts": int(
                target.execute("SELECT count(*) FROM quota_accounts").fetchone()[0]
            ),
            "model_grants": int(target.execute("SELECT count(*) FROM model_grants").fetchone()[0]),
            "model_catalog": int(
                target.execute("SELECT count(*) FROM model_catalog").fetchone()[0]
            ),
            "active_profiles": int(
                target.execute("SELECT count(*) FROM model_profiles WHERE active=1").fetchone()[0]
            ),
            "requests": int(
                target.execute("SELECT count(*) FROM inference_requests").fetchone()[0]
            ),
            "reservations": int(
                target.execute("SELECT count(*) FROM quota_reservations").fetchone()[0]
            ),
            "workers": int(target.execute("SELECT count(*) FROM workers").fetchone()[0]),
            "runtime_events": int(
                target.execute("SELECT count(*) FROM runtime_events").fetchone()[0]
            ),
        }
        if (
            counts["api_keys"] != len(plan["source_keys"])
            or counts["quota_accounts"] != len(plan["source_accounts"])
            or counts["model_grants"] != len(plan["grants"])
            or counts["model_catalog"] != 45
            or counts["active_profiles"] != 72
            or any(
                counts[table] for table in ("requests", "reservations", "workers", "runtime_events")
            )
        ):
            raise ValueError("Migrated Barra data does not match the expected clean-state contract")
        target.commit()
        return counts
    except BaseException:
        target.rollback()
        raise
    finally:
        target.close()


def _verify_migration(
    source: sqlite3.Connection,
    source_vault_path: Path,
    target: sqlite3.Connection,
    target_vault_path: Path,
) -> dict[str, Any]:
    for connection, label in ((source, "current release"), (target, "Barra release")):
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"{label} database failed its integrity check")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError(f"{label} database has foreign-key errors")

    source_keys = {str(row["id"]): row for row in source.execute("SELECT * FROM api_keys")}
    target_keys = {str(row["id"]): row for row in target.execute("SELECT * FROM api_keys")}
    source_vault = Fernet(source_vault_path.read_bytes().strip())
    target_vault = Fernet(target_vault_path.read_bytes().strip())
    if set(source_keys) != set(target_keys):
        raise ValueError("Barra API-key identities differ from current release")
    for key_id, source_row in source_keys.items():
        target_row = target_keys[key_id]
        for column in KEY_COLUMNS:
            if column != "encrypted_api_key" and source_row[column] != target_row[column]:
                raise ValueError("A Barra API-key setting differs from current release")
        source_token = source_vault.decrypt(str(source_row["encrypted_api_key"]).encode())
        target_token = target_vault.decrypt(str(target_row["encrypted_api_key"]).encode())
        if source_token != target_token:
            raise ValueError("A Barra API-key value differs from current release")
        if source_token.startswith(b"rio_") and not verify_api_key(
            str(target_row["token_hash"]), source_token.decode()
        ):
            raise ValueError("A Barra API key does not match its stored verifier")

    source_accounts = {
        str(row["id"]): row for row in source.execute("SELECT * FROM quota_accounts")
    }
    target_accounts = {
        str(row["id"]): row for row in target.execute("SELECT * FROM quota_accounts")
    }
    if set(source_accounts) != set(target_accounts):
        raise ValueError("Barra quota accounts differ from current release")
    for account_id, source_row in source_accounts.items():
        if any(
            source_row[column] != target_accounts[account_id][column] for column in ACCOUNT_COLUMNS
        ):
            raise ValueError("A Barra quota setting differs from current release")

    target_model_ids = {
        _model_identity(row): str(row["id"])
        for row in target.execute("SELECT * FROM model_catalog")
    }
    source_model_rows = {
        str(row["id"]): row for row in source.execute("SELECT * FROM model_catalog")
    }
    target_model_rows = {
        _model_identity(row): row for row in target.execute("SELECT * FROM model_catalog")
    }
    expected_grants: set[tuple[str, str, str]] = set()
    for row in source.execute("SELECT key_id,model_id,created_at FROM model_grants"):
        source_model = source_model_rows[str(row["model_id"])]
        identity = _model_identity(source_model)
        if identity in target_model_ids:
            expected_grants.add(
                (str(row["key_id"]), target_model_ids[identity], str(row["created_at"]))
            )
    actual_grants = {
        (str(row["key_id"]), str(row["model_id"]), str(row["created_at"]))
        for row in target.execute("SELECT key_id,model_id,created_at FROM model_grants")
    }
    if actual_grants != expected_grants:
        raise ValueError("Barra model grants differ from current release eligibility")
    for identity, target_model in target_model_rows.items():
        source_match = next(
            (row for row in source_model_rows.values() if _model_identity(row) == identity),
            None,
        )
        if (
            source_match is None
            or source_match["created_by_key_id"] != target_model["created_by_key_id"]
        ):
            raise ValueError("A Barra model creator differs from current release")

    history_counts = {table: _table_count(target, table) for table in HISTORY_TABLES}
    if any(history_counts.values()):
        raise ValueError("Barra release database contains user or runtime history")
    counts = {
        "api_keys": len(target_keys),
        "active_api_keys": sum(bool(row["active"]) for row in target_keys.values()),
        "quota_accounts": len(target_accounts),
        "model_grants": len(actual_grants),
        "catalog_models": _table_count(target, "model_catalog"),
        "active_profiles": int(
            target.execute("SELECT count(*) FROM model_profiles WHERE active=1").fetchone()[0]
        ),
        "history_rows": 0,
    }
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-database", type=Path, default=SOURCE_DATABASE)
    parser.add_argument("--source-vault", type=Path, default=SOURCE_VAULT)
    parser.add_argument("--target-database", type=Path, default=TARGET_DATABASE)
    parser.add_argument("--target-config", type=Path, default=TARGET_CONFIG)
    parser.add_argument("--target-vault", type=Path, default=None)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE_ROOT)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--apply", action="store_true", help="Archive Barra state and migrate current release users"
    )
    actions.add_argument(
        "--verify",
        action="store_true",
        help="Compare Barra users with current release without writing",
    )
    parser.add_argument(
        "--operator-confirmed-stopped",
        action="store_true",
        help="Confirm the Barra release service is stopped before replacing credentials",
    )
    args = parser.parse_args()

    source_path = args.source_database.resolve()
    source_vault = args.source_vault.resolve()
    target_path = args.target_database.resolve()
    config_path = args.target_config.resolve()
    target_vault = (args.target_vault or default_key_vault_path(target_path)).resolve()
    if source_path == target_path:
        parser.error("Current release source and Barra target must be different database files")
    if args.apply and not args.operator_confirmed_stopped:
        parser.error("--apply requires --operator-confirmed-stopped for the remote Barra service")
    if not all(
        path.is_file()
        for path in (source_path, source_vault, target_path, target_vault, config_path)
    ):
        parser.error("Current release and Barra release database/config/vault files must exist")
    try:
        target_config_data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        target_config_database = Path(str(target_config_data.get("database_path", "")))
        if not target_config_database.is_absolute():
            target_config_database = ROOT / target_config_database
        if target_config_database.resolve() != target_path:
            raise ValueError("Barra config does not select the user-migration target database")
        gpu_uuids = target_config_data.get("managed_gpu_uuids")
        if (
            target_config_data.get("serving_mode") != "queue"
            or not isinstance(gpu_uuids, list)
            or len(gpu_uuids) != 2
            or len(set(gpu_uuids)) != 2
        ):
            raise ValueError("Barra target config must select queue mode and its two GPU UUIDs")
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        parser.error(f"Barra target configuration is invalid: {exc}")
    try:
        if args.verify:
            source = _snapshot_memory(source_path)
            try:
                target = _read_only(target_path)
                try:
                    counts = _verify_migration(source, source_vault, target, target_vault)
                finally:
                    target.close()
                print(
                    json.dumps(
                        {
                            "verified": {
                                "current_release_identity_and_quota_match": True,
                                "api_key_values_match": True,
                                "grants_match_barra_catalog": True,
                                "database_integrity_and_foreign_keys": "ok",
                                "counts": counts,
                            }
                        },
                        indent=2,
                    )
                )
            finally:
                source.close()
            return
        with owner_lock(database_resource(target_path)):
            source = _snapshot_memory(source_path)
            try:
                target = _read_only(target_path)
                try:
                    plan = _validate_source(source, source_vault, target, target_vault)
                finally:
                    target.close()
                report = {
                    "source_keys": len(plan["source_keys"]),
                    "source_active_keys": plan["source_active_keys"],
                    "source_inactive_keys": plan["source_inactive_keys"],
                    "active_roles": plan["source_key_roles"],
                    "source_accounts": len(plan["source_accounts"]),
                    "barra_grants_to_copy": len(plan["grants"]),
                    "source_grants_outside_barra_catalog": plan["dropped_grants"],
                    "pending_reservations": 0,
                    "barra_bootstrap_startup_events_to_discard": plan["target_startup_events"],
                    "history_rows_copied": 0,
                    "key_values_will_be_preserved": True,
                }
                print(json.dumps({"preflight": report}, indent=2))
                if not args.apply:
                    print(
                        "Dry run only; no database or vault was changed. "
                        "Pass --apply to migrate users."
                    )
                    return

                archive = _archive_target(
                    target_path, config_path, target_vault, args.archive_root.resolve()
                )
                target_fernet = Fernet(target_vault.read_bytes().strip())
                counts = _apply(target_path, plan, target_fernet)
                print(
                    json.dumps(
                        {
                            "applied": {
                                "archive": str(archive),
                                "counts": counts,
                                "api_key_values_identical_to_current_release": True,
                                "current_release_source_modified": False,
                                "history_copied": False,
                                "barra_bootstrap_runtime_events_cleared": plan[
                                    "target_startup_events"
                                ],
                            }
                        },
                        indent=2,
                    )
                )
            finally:
                source.close()
    except (OSError, sqlite3.Error, ValueError, RuntimeError, TypeError, KeyError) as exc:
        parser.error(f"Barra user migration failed: {exc}")


if __name__ == "__main__":
    main()
