"""Migrate beta users, catalog and reusable native measurements into release state.

The source is read-only. By default this reports counts without changing either
database. ``--apply`` snapshots the existing release database and its vault before
one atomic data transaction. Inference and usage history are deliberately omitted.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shutil
import sqlite3
import tomllib
import uuid
from collections import Counter
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from llm_rio.config import ServingMode, Settings
from llm_rio.domain import Engine
from llm_rio.engines.identity import engine_identity, launch_binding
from llm_rio.inventory import discover_inventory
from llm_rio.operations.ownership import database_resource, owner_lock
from llm_rio.profiles import (
    profile_from_dict,
    profile_key,
    profile_to_dict,
    profile_verified_for_mode,
)
from llm_rio.security import ApiKeyVault, default_key_vault_path, verify_api_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BETA_CONFIG = ROOT / "config.toml"
DEFAULT_RELEASE_CONFIG = ROOT / "config.release.toml"
BACKUP_ROOT = ROOT / "db_backup"
LEGACY_PROFILE_REASON = (
    "Operator-authorized beta-to-release migration. Beta profiles had no release "
    "mode or launch-binding metadata; saved measurements, model revision, engine "
    "version, GPU placement, and matching engine settings were checked and bound "
    "to the selected release mode. No model probe was run."
)


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _row_fingerprint(connection: sqlite3.Connection, table: str) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
        digest.update(
            json.dumps(list(row), sort_keys=True, default=str, separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _audit_source_snapshot(
    source: Path, archive: Path, beta_config: Path, release_config: Path
) -> tuple[Path, dict[str, Any]]:
    manifest_path = archive / "manifest.json"
    archived_db = archive / source.name
    if not manifest_path.is_file() or not archived_db.is_file():
        raise ValueError("A verified beta archive is required before migration")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(str(manifest.get("source", ""))).resolve() != source.resolve():
        raise ValueError("The beta archive is for a different configured database")
    if (archive / "config.toml").read_bytes() != beta_config.read_bytes():
        raise ValueError("The beta archive configuration differs from the active beta config")
    source_vault = default_key_vault_path(source)
    archived_vault = archive / source_vault.name
    if not source_vault.is_file() or not archived_vault.is_file():
        raise ValueError("The matching beta API-key vault is required")
    if source_vault.read_bytes() != archived_vault.read_bytes():
        raise ValueError("The archived API-key vault differs from the configured beta vault")
    info: dict[str, Any] = {
        "path": str(source),
        "archive": str(archive),
        "archive_revision": manifest.get("revision"),
        "archive_model_count": manifest.get("models"),
        "live_bytes": source.stat().st_size,
        "archive_bytes": archived_db.stat().st_size,
        "tables_identical": {},
    }
    legacy_config = tomllib.loads(beta_config.read_text(encoding="utf-8"))
    release_config_data = tomllib.loads(release_config.read_text(encoding="utf-8"))

    def launch_engine_settings(config: dict[str, Any]) -> dict[str, Any]:
        settings = dict(config.get("engines", {}))
        # Beta accepted these unsupported extra fields without using them in native
        # vLLM launch; release rejects unknown engine settings.
        settings.pop("kvcached_mode", None)
        settings.pop("kv-cache-dtype", None)
        return settings

    info["launch_engine_settings_match"] = launch_engine_settings(
        legacy_config
    ) == launch_engine_settings(release_config_data)
    with (
        closing(
            sqlite3.connect(source.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as original,
        closing(
            sqlite3.connect(archived_db.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as saved,
    ):
        original.row_factory = sqlite3.Row
        saved.row_factory = sqlite3.Row
        if original.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Configured beta database failed its integrity check")
        if saved.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Archived beta database failed its integrity check")
        if original.execute("PRAGMA user_version").fetchone()[0] != 0:
            raise ValueError("Expected the configured source to be the unversioned beta database")
        for table in (
            "api_keys",
            "quota_accounts",
            "model_catalog",
            "model_grants",
            "model_profiles",
        ):
            if table not in _tables(original) or table not in _tables(saved):
                raise ValueError(f"Required beta table is missing: {table}")
            identical = _row_fingerprint(original, table) == _row_fingerprint(saved, table)
            info["tables_identical"][table] = identical
            if not identical:
                raise ValueError(f"The configured beta {table} changed after it was archived")

        key_rows = original.execute(
            "SELECT id, nickname, role, token_prefix, token_hash, encrypted_api_key, active "
            "FROM api_keys ORDER BY id"
        ).fetchall()
        beta_fernet = Fernet(source_vault.read_bytes().strip())
        credentials_verified = 0
        for row in key_rows:
            clear = beta_fernet.decrypt(str(row["encrypted_api_key"]).encode()).decode()
            tombstone = f"deleted-{row['id']}"
            deleted = (
                not row["active"]
                and row["nickname"] == tombstone
                and row["token_prefix"] == tombstone
                and row["token_hash"] == tombstone
                and clear == tombstone
            )
            if not deleted and (
                not clear.startswith("rio_") or not verify_api_key(str(row["token_hash"]), clear)
            ):
                raise ValueError("A beta API credential does not match its stored hash")
            credentials_verified += int(not deleted)
        info["api_credentials_verified"] = credentials_verified

        catalog_rows = original.execute("SELECT * FROM model_catalog").fetchall()
        unmappable = sum(
            not isinstance(row["huggingface_repo"], str) or not row["huggingface_repo"].strip()
            for row in catalog_rows
        )
        if unmappable:
            raise ValueError(
                "Some beta models lack a source identity; refusing to guess their source type"
            )
        info["models"] = len(catalog_rows)
        info["models_by_state"] = {
            str(row[0]): int(row[1])
            for row in original.execute("SELECT state, count(*) FROM model_catalog GROUP BY state")
        }
        info["api_keys"] = len(key_rows)
        info["api_keys_by_role_and_status"] = {
            f"{row[0]}:active={row[1]}": int(row[2])
            for row in original.execute(
                "SELECT role, active, count(*) FROM api_keys GROUP BY role, active"
            )
        }
        info["quota_accounts"] = int(
            original.execute("SELECT count(*) FROM quota_accounts").fetchone()[0]
        )
        info["model_grants"] = int(
            original.execute("SELECT count(*) FROM model_grants").fetchone()[0]
        )
        info["profile_rows"] = int(
            original.execute("SELECT count(*) FROM model_profiles").fetchone()[0]
        )
        info["request_history_rows"] = int(
            original.execute("SELECT count(*) FROM inference_requests").fetchone()[0]
        )
        info["model_registration_job_rows_omitted"] = int(
            original.execute("SELECT count(*) FROM model_jobs").fetchone()[0]
        )
        info["beta_verification_job_rows_omitted"] = int(
            original.execute("SELECT count(*) FROM model_verification_jobs").fetchone()[0]
        )
        info["available_artifacts_present"] = sum(
            Path(str(row["artifact_path"] or "")).exists()
            for row in catalog_rows
            if row["state"] == "AVAILABLE"
        )
        info["available_artifact_rows"] = info["models_by_state"].get("AVAILABLE", 0)
        profiles = original.execute(
            "SELECT id, model_id, machine_fingerprint, profile_json, active FROM model_profiles"
        ).fetchall()
        info["profile_rows_by_active"] = {
            f"active={row[0]}": int(row[1])
            for row in original.execute(
                "SELECT active, count(*) FROM model_profiles GROUP BY active"
            )
        }
        info["profile_rows_missing_explicit_mode"] = 0
        info["profile_rows_missing_launch_binding"] = 0
        info["profiles_by_saved_mode_evidence"] = Counter()
        info["profiles_reusable_in_release"] = Counter()
        info["profiles_requiring_validation_or_review"] = Counter()
        info["profile_migration_candidates"] = []
        info["profile_parse_errors"] = 0
        info["legacy_unreviewed_overrides"] = 0
        info["profiles_requiring_machine_override"] = 0

        release_settings = Settings(config_file=release_config)
        inventory = discover_inventory(
            release_settings.machine_id, release_settings.managed_gpu_uuids
        )
        hardware_uuids = {gpu.uuid for gpu in inventory.gpus}
        fingerprint_row = original.execute(
            "SELECT machine_fingerprint FROM service_state WHERE singleton=1"
        ).fetchone()
        beta_fingerprint = str(fingerprint_row[0]) if fingerprint_row and fingerprint_row[0] else ""
        info["release_machine_fingerprint_matches_beta"] = any(
            row["machine_fingerprint"] == beta_fingerprint for row in profiles
        )
        mode_identities = {
            mode: engine_identity(
                Settings(config_file=release_config, serving_mode=mode), Engine.VLLM
            )
            for mode in (ServingMode.QUEUE, ServingMode.VLLM_SLEEP)
        }
        info["release_vllm_version"] = mode_identities[ServingMode.QUEUE].get("version")
        info["beta_vllm_profile_versions"] = Counter()
        for row in profiles:
            try:
                old = json.loads(row["profile_json"])
            except (TypeError, json.JSONDecodeError):
                info["profile_parse_errors"] += 1
                continue
            if "serving_mode" not in old:
                info["profile_rows_missing_explicit_mode"] += 1
            if not old.get("launch_binding"):
                info["profile_rows_missing_launch_binding"] += 1
            raw = _normalise_profile_mode(old)
            mode_name = str(raw.get("serving_mode"))
            info["profiles_by_saved_mode_evidence"][mode_name] += 1
            info["beta_vllm_profile_versions"][str(raw.get("engine_version"))] += 1
            mode = {
                "queue": ServingMode.QUEUE,
                "vllm-sleep": ServingMode.VLLM_SLEEP,
            }.get(mode_name)
            if mode is None:
                info["profiles_requiring_validation_or_review"]["mode_not_proven"] += 1
                continue
            if not info["launch_engine_settings_match"]:
                info["profiles_requiring_validation_or_review"]["engine_settings_changed"] += 1
                continue
            raw["measurements_valid"] = bool(old.get("normal_verified")) and not bool(
                old.get("measurements_invalidated_at")
            )
            raw["launch_binding"] = ""
            try:
                profile = profile_from_dict(raw)
                mode_eligible = profile_verified_for_mode(
                    profile,
                    kvcached_required=False,
                    ram_weight_cache_required=mode is ServingMode.VLLM_SLEEP,
                    queue_mode_required=mode is ServingMode.QUEUE,
                )
            except (ValueError, KeyError, TypeError):
                info["profiles_requiring_validation_or_review"]["profile_format_incomplete"] += 1
                continue
            if not mode_eligible:
                info["profiles_requiring_validation_or_review"]["measurement_policy"] += 1
                continue
            if profile.engine_version != mode_identities[mode].get("version"):
                info["profiles_requiring_validation_or_review"]["engine_version_changed"] += 1
                continue
            model = original.execute(
                "SELECT resolved_revision, artifact_path, state FROM model_catalog WHERE id=?",
                (profile.model_id,),
            ).fetchone()
            if (
                model is None
                or profile.model_revision != model["resolved_revision"]
                or not model["artifact_path"]
                or not Path(str(model["artifact_path"])).exists()
            ):
                info["profiles_requiring_validation_or_review"]["artifact_identity_or_path"] += 1
                continue
            if not profile.eligible_gpu_sets or any(
                not set(group) <= hardware_uuids for group in profile.eligible_gpu_sets
            ):
                info["profiles_requiring_validation_or_review"]["gpu_placement_unavailable"] += 1
                continue
            audit = old.get("verification_override")
            source_profile_recorded_override = isinstance(audit, dict)
            if source_profile_recorded_override and not (
                audit.get("actor") and audit.get("reason")
            ):
                info["legacy_unreviewed_overrides"] += 1
            if row["machine_fingerprint"] != beta_fingerprint:
                info["profiles_requiring_machine_override"] += 1
            mode_settings = Settings(config_file=release_config, serving_mode=mode)
            try:
                binding = launch_binding(mode_settings, profile, Engine.VLLM)
            except (OSError, RuntimeError, ValueError):
                info["profiles_requiring_validation_or_review"]["launch_binding_unavailable"] += 1
                continue
            if not binding:
                info["profiles_requiring_validation_or_review"]["launch_binding_unavailable"] += 1
                continue
            info["profiles_reusable_in_release"][f"{mode.value}:active={bool(row['active'])}"] += 1
            info["profile_migration_candidates"].append(
                {
                    "id": str(row["id"]),
                    "model_id": str(row["model_id"]),
                    "source_fingerprint": str(row["machine_fingerprint"]),
                    "active": bool(row["active"]),
                    "mode": mode.value,
                    "profile": profile_to_dict(profile),
                    "binding": binding,
                    "legacy_override": source_profile_recorded_override,
                }
            )

    info["profiles_by_saved_mode_evidence"] = dict(info["profiles_by_saved_mode_evidence"])
    info["profiles_reusable_in_release"] = dict(info["profiles_reusable_in_release"])
    info["profiles_requiring_validation_or_review"] = dict(
        info["profiles_requiring_validation_or_review"]
    )
    info["beta_vllm_profile_versions"] = dict(info["beta_vllm_profile_versions"])
    return archived_db, info


def _normalise_profile_mode(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    sleep_evidence = (
        result.get("memory_backend") == "native"
        and result.get("vram_measurement_version") == 2
        and all(
            result.get(field) is not None
            for field in (
                "sleep_vram_mib_per_gpu",
                "wake_peak_vram_mib_per_gpu",
                "vram_baseline_mib_per_gpu",
                "host_cache_mib",
                "weight_cache_offload_seconds",
                "weight_cache_activation_seconds",
            )
        )
    )
    if result.get("memory_backend") == "kvcached":
        mode = "kv-cached"
    elif result.get("serving_mode") in {item.value for item in ServingMode}:
        mode = str(result["serving_mode"])
    elif sleep_evidence:
        mode = ServingMode.VLLM_SLEEP.value
    elif (
        result.get("memory_backend") == "native"
        and result.get("launch_args", {}).get("enable_sleep_mode") is False
    ):
        mode = ServingMode.QUEUE.value
    else:
        mode = "legacy-unclassified"
    result["serving_mode"] = mode
    result["vram_measurement_version"] = int(result.get("vram_measurement_version", 1))
    result.setdefault("launch_binding", "")
    result.setdefault("measurements_valid", False)
    return result


def _release_is_empty(connection: sqlite3.Connection) -> dict[str, Any]:
    if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Release target is not schema version 1")
    counts = {
        table: int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
        for table in (
            "api_keys",
            "quota_accounts",
            "model_catalog",
            "model_grants",
            "model_jobs",
            "model_profiles",
            "quota_reservations",
            "inference_requests",
            "workers",
        )
    }
    if counts != {
        "api_keys": 1,
        "quota_accounts": 1,
        "model_catalog": 0,
        "model_grants": 0,
        "model_jobs": 0,
        "model_profiles": 0,
        "quota_reservations": 0,
        "inference_requests": 0,
        "workers": 0,
    }:
        raise ValueError(
            "Release target has been used; refusing to merge into nonempty release state"
        )
    return counts


def _verify_release_bootstrap_credential(
    connection: sqlite3.Connection, database_path: Path
) -> int:
    vault_path = default_key_vault_path(database_path)
    if not vault_path.is_file():
        raise ValueError("Release API-key vault is missing; refusing to replace bootstrap access")
    vault = Fernet(vault_path.read_bytes().strip())
    rows = connection.execute("SELECT token_hash, encrypted_api_key FROM api_keys").fetchall()
    if not rows:
        raise ValueError("Release bootstrap credential is missing")
    for row in rows:
        try:
            clear = vault.decrypt(str(row["encrypted_api_key"]).encode()).decode()
        except (InvalidToken, ValueError, TypeError) as exc:
            raise ValueError("Release bootstrap credential cannot be decrypted") from exc
        if not verify_api_key(str(row["token_hash"]), clear):
            raise ValueError("Release bootstrap credential does not match its stored hash")
    return len(rows)


def _snapshot_release(connection: sqlite3.Connection, settings: Settings) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = BACKUP_ROOT / f"release-before-user-model-migration-{stamp}"
    destination.mkdir(parents=True, mode=0o700, exist_ok=False)
    db_target = destination / settings.database_path.name
    try:
        with closing(sqlite3.connect(db_target)) as backup:
            connection.backup(backup)
            if backup.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Release rollback snapshot failed its integrity check")
        source_vault = default_key_vault_path(settings.database_path)
        if source_vault.exists():
            shutil.copy2(source_vault, destination / source_vault.name)
        config_copy = destination / "config.release.toml"
        shutil.copy2(settings.config_file, config_copy)
        files = {p.name: _digest(p) for p in destination.iterdir() if p.is_file()}
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "database": str(settings.database_path),
            "purpose": "Rollback snapshot before beta user/model import",
            "sha256": files,
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in destination.iterdir():
            os.chmod(path, 0o600)
        return destination
    except BaseException:
        # Preserve incomplete output for diagnosis; it has no completion manifest.
        raise


def _prepare_profile_migration(
    info: dict[str, Any], settings: Settings, actor: str, reason: str
) -> list[dict[str, Any]]:
    inventory = discover_inventory(settings.machine_id, settings.managed_gpu_uuids)
    hardware_uuids = {gpu.uuid for gpu in inventory.gpus}
    with closing(sqlite3.connect(settings.database_path)) as release:
        fingerprint_row = release.execute(
            "SELECT machine_fingerprint FROM service_state WHERE singleton=1"
        ).fetchone()
    target_fingerprint = str(fingerprint_row[0]) if fingerprint_row and fingerprint_row[0] else ""
    if not target_fingerprint:
        raise ValueError("Release database has no recorded machine fingerprint")
    prepared: list[dict[str, Any]] = []
    now = datetime.now(UTC).isoformat()
    for candidate in info["profile_migration_candidates"]:
        profile = dict(candidate["profile"])
        mode = ServingMode(candidate["mode"])
        if any(not set(group) <= hardware_uuids for group in profile["eligible_gpu_sets"]):
            raise ValueError("GPU inventory changed after preflight; refusing profile migration")
        source_id = candidate["id"]
        source_fingerprint = candidate["source_fingerprint"]
        machine_changed = source_fingerprint != target_fingerprint
        profile_id = str(uuid.uuid4()) if machine_changed else source_id
        profile["id"] = profile_id
        profile["machine_fingerprint"] = target_fingerprint
        profile["launch_binding"] = launch_binding(
            settings.model_copy(update={"serving_mode": mode}),
            profile_from_dict(profile),
            Engine.VLLM,
        )
        profile["measurements_valid"] = True
        audit = {
            "source_profile_id": source_id,
            "source_fingerprint": source_fingerprint,
            "actor": actor,
            "reason": reason,
            "at": now,
            "serving_mode": mode.value,
            "migration": "beta-to-release",
            "legacy_override_preserved": bool(candidate["legacy_override"]),
        }
        profile["verification_override"] = audit
        raw = profile_to_dict(profile_from_dict(profile))
        raw["verification_override"] = audit
        key = profile_key(raw)
        prepared.append(
            {
                "id": profile_id,
                "source_id": source_id,
                "model_id": candidate["model_id"],
                "fingerprint": target_fingerprint,
                "profile_json": json.dumps(raw, sort_keys=True, separators=(",", ":")),
                "profile_key": key,
                "verified_at": now if machine_changed else None,
                "active": int(candidate["active"]),
                "machine_changed": machine_changed,
                "mode": mode.value,
                "audit": True,
            }
        )
    # Two old fingerprints can collapse onto one current-machine profile key. Keep the
    # most recent evidence, preferring an active row when otherwise equivalent.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in prepared:
        grouped.setdefault(item["profile_key"], []).append(item)
    unique: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(key=lambda item: (item["active"], item["verified_at"] or ""), reverse=True)
        selected = group[0]
        if len(group) > 1:
            raw = json.loads(selected["profile_json"])
            audit = raw.get("verification_override") or {
                "source_profile_id": selected["source_id"],
                "source_fingerprint": selected["fingerprint"],
                "actor": actor,
                "reason": reason,
                "at": now,
                "serving_mode": selected["mode"],
                "migration": "beta-to-release",
            }
            audit["collapsed_source_profile_ids"] = [item["source_id"] for item in group]
            raw["verification_override"] = audit
            selected["profile_json"] = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            selected["audit"] = True
        unique.append(selected)
    return unique


def _insert_values(
    connection: sqlite3.Connection, table: str, columns: tuple[str, ...], values: tuple[Any, ...]
) -> None:
    names = ",".join(columns)
    placeholders = ",".join("?" for _ in columns)
    connection.execute(f'INSERT INTO "{table}" ({names}) VALUES ({placeholders})', values)


def _seed_revalidation_jobs(connection: sqlite3.Connection) -> int:
    """Create retryable jobs for imported models with no active placement profile."""
    if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Revalidation jobs require a schema-version-1 release database")
    rows = connection.execute(
        "SELECT m.id FROM model_catalog m WHERE m.state != 'DISABLED' "
        "AND NOT EXISTS (SELECT 1 FROM model_profiles p "
        "WHERE p.model_id=m.id AND p.active=1) "
        "AND NOT EXISTS (SELECT 1 FROM model_jobs j WHERE j.model_id=m.id) "
        "ORDER BY m.id"
    ).fetchall()
    now = datetime.now(UTC).isoformat()
    for row in rows:
        connection.execute(
            "INSERT INTO model_jobs(id,model_id,state,stage,created_at,updated_at) "
            "VALUES (?,?,'FAILED','migration_requires_validation',?,?)",
            (str(uuid.uuid4()), str(row[0]), now, now),
        )
    return len(rows)


def _apply_migration(
    *,
    source: Path,
    source_archive: Path,
    target: Path,
    settings: Settings,
    beta_vault_path: Path,
    actor: str,
    reason: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    resources = sorted(
        (database_resource(source), database_resource(target)),
        key=lambda item: str(item),
    )
    with ExitStack() as stack:
        for resource in resources:
            stack.enter_context(owner_lock(resource))
        beta = stack.enter_context(
            closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro&immutable=1", uri=True))
        )
        beta.row_factory = sqlite3.Row
        release = stack.enter_context(closing(sqlite3.connect(target, isolation_level=None)))
        release.row_factory = sqlite3.Row
        release.execute("PRAGMA foreign_keys=ON")
        _release_is_empty(release)
        rollback_path = _snapshot_release(release, settings)

        # Recheck the evidence and target while ownership locks are held.
        archived = source_archive / source.name
        with closing(
            sqlite3.connect(archived.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        ) as saved:
            for table in (
                "api_keys",
                "quota_accounts",
                "model_catalog",
                "model_grants",
                "model_profiles",
            ):
                if _row_fingerprint(beta, table) != _row_fingerprint(saved, table):
                    raise ValueError(f"The beta {table} changed during migration preparation")
        _release_is_empty(release)

        beta_fernet = Fernet(beta_vault_path.read_bytes().strip())
        target_vault = ApiKeyVault(default_key_vault_path(target))
        keys = beta.execute("SELECT * FROM api_keys ORDER BY created_at, id").fetchall()
        accounts = beta.execute("SELECT * FROM quota_accounts ORDER BY created_at, id").fetchall()
        models = beta.execute("SELECT * FROM model_catalog ORDER BY created_at, id").fetchall()
        grants = beta.execute("SELECT * FROM model_grants ORDER BY key_id, model_id").fetchall()
        source_profiles = beta.execute(
            "SELECT id, model_id, machine_fingerprint, profile_key, profile_json, "
            "verified_at, active "
            "FROM model_profiles ORDER BY verified_at, id"
        ).fetchall()
        prepared_profiles = _prepare_profile_migration(info, settings, actor, reason)
        key_columns = (
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
        account_columns = (
            "id",
            "nickname",
            "balance_tokens",
            "limit_tokens",
            "usage_baseline_tokens",
            "usage_reset_at",
            "unlimited",
            "created_at",
        )
        model_columns = (
            "id",
            "nickname",
            "huggingface_repo",
            "source_type",
            "local_path",
            "engine",
            "requested_revision",
            "resolved_revision",
            "state",
            "artifact_path",
            "artifact_hashes_json",
            "capabilities_json",
            "request_limits_json",
            "request_defaults_json",
            "source_model_id",
            "created_by_key_id",
            "created_at",
            "updated_at",
        )
        release.execute("BEGIN IMMEDIATE")
        try:
            # The single release bootstrap identity is disposable; replace it with the
            # original beta key rows and their exact quota accounts.
            release.execute("DELETE FROM runtime_events")
            release.execute("DELETE FROM api_keys")
            release.execute("DELETE FROM quota_accounts")
            for row in accounts:
                _insert_values(
                    release,
                    "quota_accounts",
                    account_columns,
                    tuple(row[name] for name in account_columns),
                )
            imported_plaintext: dict[str, str] = {}
            for row in keys:
                clear = beta_fernet.decrypt(str(row["encrypted_api_key"]).encode()).decode()
                deleted = (
                    not row["active"]
                    and str(row["nickname"]) == f"deleted-{row['id']}"
                    and str(row["token_prefix"]) == f"deleted-{row['id']}"
                    and str(row["token_hash"]) == f"deleted-{row['id']}"
                    and clear == f"deleted-{row['id']}"
                )
                if not deleted and not verify_api_key(str(row["token_hash"]), clear):
                    raise ValueError("A beta API key failed its credential verification")
                if not deleted:
                    imported_plaintext[str(row["id"])] = clear
                values = tuple(
                    target_vault.encrypt(clear) if name == "encrypted_api_key" else row[name]
                    for name in key_columns
                )
                _insert_values(release, "api_keys", key_columns, values)
            for row in models:
                values_by_column = {
                    "id": row["id"],
                    "nickname": row["nickname"],
                    "huggingface_repo": row["huggingface_repo"],
                    "source_type": "huggingface",
                    "local_path": None,
                    "engine": "vllm",
                    "requested_revision": row["requested_revision"],
                    "resolved_revision": row["resolved_revision"],
                    "state": row["state"],
                    "artifact_path": row["artifact_path"],
                    "artifact_hashes_json": row["artifact_hashes_json"],
                    "capabilities_json": row["capabilities_json"],
                    "request_limits_json": row["request_limits_json"],
                    "request_defaults_json": row["request_defaults_json"],
                    "source_model_id": None,
                    "created_by_key_id": row["created_by_key_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                _insert_values(
                    release,
                    "model_catalog",
                    model_columns,
                    tuple(values_by_column[name] for name in model_columns),
                )
            for row in models:
                if row["source_model_id"] is not None:
                    release.execute(
                        "UPDATE model_catalog SET source_model_id=? WHERE id=?",
                        (row["source_model_id"], row["id"]),
                    )
            for row in grants:
                release.execute(
                    "INSERT INTO model_grants(key_id,model_id,created_at) VALUES (?,?,?)",
                    (row["key_id"], row["model_id"], row["created_at"]),
                )

            # Carry complete queue and sleep measurements. Old boolean trust records
            # become explicit, attributable migration overrides. Equivalent rows from
            # old machine fingerprints share one active release profile; other
            # unclassified, incomplete, or duplicate rows remain inactive.
            migrated_candidate_ids = {row["source_id"] for row in prepared_profiles}
            used_profile_keys = {item["profile_key"] for item in prepared_profiles}
            active_preserved = 0
            for row in source_profiles:
                if str(row["id"]) in migrated_candidate_ids:
                    continue
                old = json.loads(row["profile_json"])
                raw = _normalise_profile_mode(old)
                raw["id"] = str(row["id"])
                raw["model_id"] = str(row["model_id"])
                raw["machine_fingerprint"] = str(row["machine_fingerprint"])
                raw["measurements_valid"] = False
                raw["launch_binding"] = ""
                profile = profile_to_dict(profile_from_dict(raw))
                profile_key_value = profile_key(profile)
                if profile_key_value in used_profile_keys:
                    profile_key_value = hashlib.sha256(
                        f"{profile_key_value}:{row['id']}".encode()
                    ).hexdigest()
                used_profile_keys.add(profile_key_value)
                release.execute(
                    "INSERT INTO model_profiles("
                    "id,model_id,machine_fingerprint,profile_key,profile_json,verified_at,active"
                    ") "
                    "VALUES (?,?,?,?,?,?,0)",
                    (
                        str(row["id"]),
                        str(row["model_id"]),
                        str(row["machine_fingerprint"]),
                        profile_key_value,
                        json.dumps(profile, sort_keys=True, separators=(",", ":")),
                        str(row["verified_at"]),
                    ),
                )
            for item in prepared_profiles:
                profile_json = item["profile_json"]
                if item["active"]:
                    active_preserved += 1
                release.execute(
                    "INSERT INTO model_profiles("
                    "id,model_id,machine_fingerprint,profile_key,profile_json,verified_at,active"
                    ") "
                    "VALUES (?,?,?,?,?,?,?)",
                    (
                        item["id"],
                        item["model_id"],
                        item["fingerprint"],
                        item["profile_key"],
                        profile_json,
                        item["verified_at"]
                        or next(
                            str(row["verified_at"])
                            for row in source_profiles
                            if str(row["id"]) == item["source_id"]
                        ),
                        item["active"],
                    ),
                )
            new_validation_jobs = _seed_revalidation_jobs(release)
            foreign_errors = release.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_errors:
                raise ValueError(
                    "Imported records violate "
                    f"{len(foreign_errors)} release foreign-key constraints"
                )
            release.commit()
        except BaseException:
            release.rollback()
            raise
        release.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Check the result while locks are held. No secret values leave process memory.
        if release.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Migrated release database failed its integrity check")
        release_key_rows = release.execute(
            "SELECT id,token_hash,encrypted_api_key FROM api_keys"
        ).fetchall()
        target_fernet = Fernet(default_key_vault_path(target).read_bytes().strip())
        for row in release_key_rows:
            clear = target_fernet.decrypt(str(row["encrypted_api_key"]).encode()).decode()
            if str(row["id"]) in imported_plaintext and (
                clear != imported_plaintext[str(row["id"])]
                or not verify_api_key(str(row["token_hash"]), clear)
            ):
                raise RuntimeError("An imported API credential changed during encryption")
        counts = {
            table: int(release.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in (
                "api_keys",
                "quota_accounts",
                "model_catalog",
                "model_grants",
                "model_profiles",
                "model_jobs",
                "quota_reservations",
                "inference_requests",
                "runtime_events",
                "workers",
            )
        }
        return {
            "release_rollback_snapshot": str(rollback_path),
            "rows_after_migration": counts,
            "new_revalidation_jobs_created": new_validation_jobs,
            "profiles_eligible_and_imported": len(prepared_profiles),
            "active_profile_rows_preserved": active_preserved,
            "eligible_source_profile_records_collapsed_as_equivalent": len(
                info["profile_migration_candidates"]
            )
            - len(prepared_profiles),
            "active_duplicate_records_collapsed": sum(
                bool(candidate["active"]) for candidate in info["profile_migration_candidates"]
            )
            - active_preserved,
            "profiles_requiring_new_validation_or_operator_review": int(info["profile_rows"])
            - len(info["profile_migration_candidates"]),
            "inference_history_rows_migrated": 0,
            "registration_job_history_rows_migrated": 0,
            "beta_verification_job_history_rows_migrated": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta-config", type=Path, default=DEFAULT_BETA_CONFIG)
    parser.add_argument("--release-config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "db_backup/startup-repair-20260923-verified"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Back up release state and apply the migration"
    )
    parser.add_argument("--actor", default=f"unix:{getpass.getuser()}")
    parser.add_argument("--reason", default=LEGACY_PROFILE_REASON)
    args = parser.parse_args()
    if not args.beta_config.is_file() or not args.release_config.is_file():
        parser.error("both beta and release configuration files must exist")
    beta_config = tomllib.loads(args.beta_config.read_text(encoding="utf-8"))
    source = Path(str(beta_config.get("database_path", "")))
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        parser.error("the configured beta database does not exist")
    settings = Settings(config_file=args.release_config)
    target = (
        settings.database_path
        if settings.database_path.is_absolute()
        else ROOT / settings.database_path
    )
    if target.resolve() == source.resolve():
        parser.error("beta source and release target must be different files")
    settings.config_file = args.release_config
    settings.database_path = target
    archived_db, info = _audit_source_snapshot(
        source, args.archive, args.beta_config, args.release_config
    )
    info["release_config"] = str(args.release_config)
    with closing(sqlite3.connect(target)) as destination:
        destination.row_factory = sqlite3.Row
        info["release_target_before"] = _release_is_empty(destination)
        info["release_bootstrap_credentials_verified"] = _verify_release_bootstrap_credential(
            destination, target
        )
    info["selected_release_mode"] = settings.serving_mode.value
    candidates = info["profile_migration_candidates"]
    print("Preflight:")
    _safe_info = dict(info)
    _safe_info.pop("profile_migration_candidates")
    print(json.dumps(_safe_info, indent=2, default=str))
    print(json.dumps({"mode_profile_candidates": len(candidates)}, indent=2))
    if not args.apply:
        print("Dry run only; no database or vault was changed. Pass --apply to migrate.")
        return
    if not args.actor.strip() or not args.reason.strip():
        parser.error("--actor and --reason must be non-empty")
    source_vault = default_key_vault_path(source)
    counts = _apply_migration(
        source=source,
        source_archive=args.archive,
        target=target,
        settings=settings,
        beta_vault_path=source_vault,
        actor=args.actor.strip(),
        reason=args.reason.strip(),
        info=info,
    )
    print("Applied:")
    print(json.dumps(counts, indent=2, default=str))


if __name__ == "__main__":
    main()
