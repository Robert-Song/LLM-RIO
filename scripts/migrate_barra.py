"""Create an isolated Barra release database from release catalog and Barra evidence.

The beta source and current release database are read-only. The default invocation
performs a preflight only; ``--apply`` archives the configured Barra beta database,
then creates a new queue-mode database, config, vault, and local administrator.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
import tomllib
import uuid
from collections import Counter
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from cryptography.fernet import Fernet

from llm_rio.config import Settings
from llm_rio.domain import Engine
from llm_rio.engines.identity import engine_identity, launch_binding
from llm_rio.profiles import (
    profile_from_dict,
    profile_key,
    profile_to_dict,
    profile_verified_for_mode,
)
from llm_rio.security import (
    ApiKeyVault,
    default_key_vault_path,
    hash_api_key,
    issue_api_key,
    verify_api_key,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BARRA_CONFIG = ROOT / "config.barra.toml"
DEFAULT_RELEASE_CONFIG = ROOT / "config.release.toml"
DEFAULT_TARGET_CONFIG = ROOT / "config.barra.release.toml"
DEFAULT_TARGET_DATABASE = ROOT / "state_barra/release/llm-rio.db"
MIGRATION_REASON = (
    "Operator-authorized Barra beta-to-release migration. Reused only complete native "
    "queue measurements whose pinned model revision, vLLM identity, effective launch "
    "binding, and eligible GPU UUID placement were checked. No model probe was run."
)
MODEL_COLUMNS = (
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
MEASUREMENT_FIELDS = (
    "predicted_tokens_per_second",
    "load_and_warmup_seconds",
    "idle_vram_mib_per_gpu",
    "peak_vram_mib_per_gpu",
    "gpu_headroom_mib_per_gpu",
    "kv_cache_capacity_tokens",
    "max_full_length_concurrency",
    "vram_measurement_version",
    "vram_baseline_mib_per_gpu",
    "sleep_vram_mib_per_gpu",
    "wake_peak_vram_mib_per_gpu",
    "weight_cache_offload_seconds",
    "weight_cache_activation_seconds",
    "host_cache_mib",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_only(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _immutable_read_only(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _online_snapshot(source_path: Path) -> sqlite3.Connection:
    source = _read_only(source_path)
    destination = sqlite3.connect(":memory:")
    destination.row_factory = sqlite3.Row
    try:
        source.backup(destination)
    finally:
        source.close()
    if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        destination.close()
        raise ValueError("A source database snapshot failed its integrity check")
    return destination


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _serialize_toml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Configuration contains a non-finite number")
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_serialize_toml_scalar(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {type(value).__name__}")


def _serialize_toml(data: dict[str, Any]) -> str:
    lines: list[str] = []

    def emit(table: dict[str, Any], prefix: tuple[str, ...] = ()) -> None:
        scalar_items = [(key, value) for key, value in table.items() if not isinstance(value, dict)]
        child_items = [(key, value) for key, value in table.items() if isinstance(value, dict)]
        if prefix:
            if lines and lines[-1] != "":
                lines.append("")
            header = ".".join(prefix)
            lines.append(f"[{header}]")
        for key, value in scalar_items:
            lines.append(f"{key} = {_serialize_toml_scalar(value)}")
        if prefix and not scalar_items and not child_items:
            lines.append("")
        for key, value in child_items:
            emit(value, (*prefix, key))

    emit(data)
    return "\n".join(lines).rstrip() + "\n"


def _model_identity(row: sqlite3.Row | dict[str, Any]) -> tuple[str, str]:
    repo = str(row["huggingface_repo"] or "").strip()
    revision = str(row["resolved_revision"] or "").strip()
    if not repo or not revision:
        raise ValueError("A Barra model has no pinned Hugging Face identity")
    return repo, revision


def _config_data(
    barra_config: Path,
    release_config: Path,
    target_database: Path,
) -> tuple[dict[str, Any], Settings, dict[str, Any]]:
    barra = tomllib.loads(barra_config.read_text(encoding="utf-8"))
    release = tomllib.loads(release_config.read_text(encoding="utf-8"))
    engines = dict(barra.get("engines", {}))
    engines.pop("kvcached_mode", None)
    release_engines = dict(release.get("engines", {}))
    if engines != release_engines:
        raise ValueError("Barra and release engine settings differ; profiles cannot be reused")
    gpu_uuids = barra.get("managed_gpu_uuids")
    if (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != 2
        or len(set(gpu_uuids)) != 2
        or any(not isinstance(value, str) or not value.startswith("GPU-") for value in gpu_uuids)
    ):
        raise ValueError("Barra config must identify exactly two distinct GPU UUIDs")

    # Build a fresh native queue config. Secrets are supplied by the deployment
    # environment; no credential from the other server is copied into this file.
    target = dict(release)
    target.pop("hf_token", None)
    for field in ("machine_id", "api_host", "api_port", "model_store", "log_dir"):
        target[field] = barra[field]
    target["managed_gpu_uuids"] = list(gpu_uuids)
    target["database_path"] = target_database.relative_to(ROOT).as_posix()
    target["serving_mode"] = "queue"
    settings = Settings.model_validate(target)
    if settings.serving_mode.value != "queue":
        raise ValueError("Generated Barra configuration did not select queue mode")
    return target, settings, barra


def _check_artifacts(models: list[sqlite3.Row]) -> dict[str, int]:
    available = [row for row in models if row["state"] == "AVAILABLE"]
    missing_paths = 0
    missing_files = 0
    size_mismatches = 0
    for row in available:
        artifact_value = str(row["artifact_path"] or "")
        if not artifact_value:
            missing_paths += 1
            continue
        artifact = Path(artifact_value)
        if not artifact.is_absolute():
            artifact = ROOT / artifact
        if not artifact.is_dir():
            missing_paths += 1
            continue
        if artifact.name != str(row["resolved_revision"]):
            raise ValueError("An available model artifact path is not pinned to its saved revision")
        try:
            manifest = json.loads(str(row["artifact_hashes_json"] or "[]"))
        except json.JSONDecodeError as exc:
            raise ValueError("An available model has an invalid artifact manifest") from exc
        if not isinstance(manifest, list) or not manifest:
            raise ValueError("An available model has no pinned artifact manifest")
        for entry in manifest:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError("An available model has a malformed artifact manifest")
            relative = PurePosixPath(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("An available model has an unsafe artifact manifest path")
            path = artifact.joinpath(*relative.parts)
            if not path.is_file():
                missing_files += 1
            elif path.stat().st_size != int(entry.get("bytes", -1)):
                size_mismatches += 1
            elif (
                not path.is_symlink()
                or not isinstance(entry.get("digest"), str)
                or path.resolve(strict=True).name != entry["digest"]
            ):
                raise ValueError(
                    "An available model file is not linked to its saved content-addressed blob"
                )
    if missing_paths or missing_files or size_mismatches:
        raise ValueError("One or more Barra available model artifacts differ from their manifests")
    return {
        "available_models": len(available),
        "artifact_paths_missing": missing_paths,
        "artifact_files_missing": missing_files,
        "artifact_size_mismatches": size_mismatches,
    }


def _mode_from_beta(raw: dict[str, Any]) -> dict[str, Any]:
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
    elif result.get("serving_mode") in {"queue", "vllm-sleep", "kv-cached"}:
        mode = str(result["serving_mode"])
    elif sleep_evidence:
        mode = "vllm-sleep"
    elif (
        result.get("memory_backend") == "native"
        and result.get("launch_args", {}).get("enable_sleep_mode") is False
    ):
        mode = "queue"
    else:
        mode = "legacy-unclassified"
    result["serving_mode"] = mode
    result["vram_measurement_version"] = int(result.get("vram_measurement_version", 1))
    return result


def _profile_candidates(
    beta: sqlite3.Connection,
    release_models: dict[tuple[str, str], sqlite3.Row],
    settings: Settings,
    gpu_uuids: set[str],
    target_fingerprint: str,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    identity = engine_identity(settings, Engine.VLLM)
    engine_version = identity.get("version", "")
    if not engine_version:
        raise ValueError("Cannot establish the selected vLLM identity")
    source_models = {
        str(row["id"]): row for row in beta.execute("SELECT * FROM model_catalog").fetchall()
    }
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    profile_rows = beta.execute(
        "SELECT id,model_id,machine_fingerprint,profile_json,verified_at,active "
        "FROM model_profiles WHERE active=1 ORDER BY verified_at,id"
    ).fetchall()
    checked_at = _utc_now()
    for row in profile_rows:
        try:
            old = _mode_from_beta(json.loads(str(row["profile_json"])))
            old["measurements_valid"] = bool(old.get("normal_verified")) and not bool(
                old.get("measurements_invalidated_at")
            )
            profile = profile_from_dict(old)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            rejected["profile_incomplete_or_malformed"] += 1
            continue
        if old["serving_mode"] != "queue":
            rejected["not_native_queue"] += 1
            continue
        if profile.engine is not Engine.VLLM or profile.engine_version != engine_version:
            rejected["vllm_identity_mismatch"] += 1
            continue
        if not profile_verified_for_mode(
            profile,
            kvcached_required=False,
            queue_mode_required=True,
        ):
            rejected["measurement_policy_incomplete"] += 1
            continue
        if old.get("measurements_invalidated_at"):
            rejected["measurements_invalidated"] += 1
            continue
        source_model = source_models.get(str(row["model_id"]))
        if source_model is None or source_model["state"] != "AVAILABLE":
            rejected["model_not_available"] += 1
            continue
        try:
            model_identity = _model_identity(source_model)
        except ValueError:
            rejected["model_identity_missing"] += 1
            continue
        release_model = release_models.get(model_identity)
        if release_model is None or release_model["state"] != "AVAILABLE":
            rejected["release_model_unavailable"] += 1
            continue
        if profile.model_revision != release_model["resolved_revision"]:
            rejected["model_revision_mismatch"] += 1
            continue
        source_fingerprint = str(row["machine_fingerprint"] or "")
        if not source_fingerprint or profile.machine_fingerprint != source_fingerprint:
            rejected["profile_fingerprint_mismatch"] += 1
            continue
        eligible_sets = tuple(
            tuple(group)
            for group in profile.eligible_gpu_sets
            if len(group) == profile.gpu_count
            and len(set(group)) == profile.gpu_count
            and set(group) <= gpu_uuids
        )
        if not eligible_sets:
            rejected["configured_gpu_placement_mismatch"] += 1
            continue

        source_profile = profile_to_dict(profile)
        measurement_digest = _sha256_bytes(
            json.dumps(
                {key: source_profile.get(key) for key in MEASUREMENT_FIELDS},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        migrated = replace(
            profile,
            model_id=str(release_model["id"]),
            machine_fingerprint=target_fingerprint,
            eligible_gpu_sets=eligible_sets,
            serving_mode="queue",
            measurements_valid=True,
            launch_binding="",
        )
        binding = launch_binding(settings, migrated, Engine.VLLM)
        migrated = replace(migrated, launch_binding=binding)
        raw = profile_to_dict(migrated)
        audit = {
            "source_profile_id": str(row["id"]),
            "source_fingerprint": source_fingerprint,
            "source_measurement_sha256": measurement_digest,
            "actor": f"unix:{getpass.getuser()}",
            "reason": MIGRATION_REASON,
            "at": checked_at,
            "serving_mode": "queue",
            "migration": "barra-beta-to-release",
            "fingerprint_rebased": source_fingerprint != target_fingerprint,
        }
        if isinstance(old.get("verification_override"), dict):
            prior = old["verification_override"]
            audit["source_override_actor"] = str(prior.get("actor") or "unattributed-beta")
            audit["source_override_reason"] = str(
                prior.get("reason") or "unattributed beta override"
            )
        raw["verification_override"] = audit
        accepted.append(
            {
                "id": str(row["id"]),
                "model_id": str(release_model["id"]),
                "machine_fingerprint": target_fingerprint,
                "profile_key": profile_key(raw),
                "profile_json": json.dumps(raw, sort_keys=True, separators=(",", ":")),
                "verified_at": str(row["verified_at"]),
                "active": 1,
                "source_profile_id": str(row["id"]),
                "source_fingerprint": source_fingerprint,
                "mode": "queue",
                "measurement_projection": {
                    key: value
                    for key, value in raw.items()
                    if key
                    not in {
                        "id",
                        "model_id",
                        "machine_fingerprint",
                        "launch_binding",
                        "verification_override",
                    }
                },
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in accepted:
        grouped.setdefault(item["profile_key"], []).append(item)
    unique: list[dict[str, Any]] = []
    collapsed = 0
    for group in grouped.values():
        projections = {
            json.dumps(item["measurement_projection"], sort_keys=True, separators=(",", ":"))
            for item in group
        }
        if len(projections) != 1:
            raise ValueError("Equivalent Barra profiles have different saved measurements")
        group.sort(key=lambda item: item["verified_at"], reverse=True)
        chosen = group[0]
        if len(group) > 1:
            raw = json.loads(chosen["profile_json"])
            audit = raw["verification_override"]
            audit["collapsed_source_profile_ids"] = [entry["source_profile_id"] for entry in group]
            raw["verification_override"] = audit
            chosen["profile_json"] = json.dumps(raw, sort_keys=True, separators=(",", ":"))
            collapsed += len(group) - 1
        unique.append(chosen)

    if not target_fingerprint:
        raise ValueError("Barra beta service state has no current machine fingerprint")
    covered_ids = {str(item["model_id"]) for item in unique}
    available_ids = {
        str(release_models[_model_identity(model)]["id"])
        for model in source_models.values()
        if model["state"] == "AVAILABLE"
    }
    if covered_ids != available_ids:
        raise ValueError("Reusable Barra profiles do not cover every available Barra model")
    report = dict(rejected)
    report.update(
        source_active_profile_rows=len(profile_rows),
        eligible_source_profile_rows=len(accepted),
        active_profile_rows_after_deduplication=len(unique),
        duplicate_profile_rows_collapsed=collapsed,
        available_models_covered=len(covered_ids),
    )
    return unique, report, engine_version


def _prepare_models(
    beta: sqlite3.Connection,
    release: sqlite3.Connection,
) -> tuple[list[sqlite3.Row], dict[tuple[str, str], sqlite3.Row]]:
    beta_rows = beta.execute("SELECT * FROM model_catalog ORDER BY created_at,id").fetchall()
    release_rows = release.execute("SELECT * FROM model_catalog ORDER BY created_at,id").fetchall()
    beta_by_identity = {_model_identity(row): row for row in beta_rows}
    release_by_identity = {_model_identity(row): row for row in release_rows}
    if len(beta_by_identity) != len(beta_rows) or len(release_by_identity) != len(release_rows):
        raise ValueError("Catalog has duplicate model identities; refusing an ambiguous migration")
    if not set(beta_by_identity) <= set(release_by_identity):
        raise ValueError("Barra contains model revisions that are missing from the release catalog")

    common_columns = (
        "nickname",
        "huggingface_repo",
        "requested_revision",
        "resolved_revision",
        "state",
        "artifact_path",
        "artifact_hashes_json",
        "capabilities_json",
        "request_limits_json",
        "request_defaults_json",
    )
    beta_id_to_release_id: dict[str, str] = {}
    for identity, source in beta_by_identity.items():
        target = release_by_identity[identity]
        if any(source[field] != target[field] for field in common_columns):
            raise ValueError("A Barra model differs from its matching release catalog record")
        if target["source_type"] != "huggingface" or target["local_path"] is not None:
            raise ValueError("A Barra model does not map to a Hugging Face release record")
        if target["engine"] != Engine.VLLM.value:
            raise ValueError("A Barra model is not configured for vLLM in release state")
        beta_id_to_release_id[str(source["id"])] = str(target["id"])

    for identity, source in beta_by_identity.items():
        source_parent = source["source_model_id"]
        release_parent = release_by_identity[identity]["source_model_id"]
        expected_parent = (
            beta_id_to_release_id.get(str(source_parent)) if source_parent is not None else None
        )
        if expected_parent != (str(release_parent) if release_parent is not None else None):
            raise ValueError("A Barra model parent reference differs from release state")

    _check_artifacts(beta_rows)
    return beta_rows, release_by_identity


def _prepare(
    beta: sqlite3.Connection,
    release: sqlite3.Connection,
    settings: Settings,
    barra: dict[str, Any],
) -> dict[str, Any]:
    if beta.execute("PRAGMA user_version").fetchone()[0] != 0:
        raise ValueError("Configured Barra source is not the expected beta schema")
    if release.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise ValueError("Current release database is not schema version 1")
    required_beta = {
        "api_keys",
        "quota_accounts",
        "model_catalog",
        "model_profiles",
        "service_state",
    }
    required_release = {
        "api_keys",
        "quota_accounts",
        "model_catalog",
        "model_profiles",
        "service_state",
        "model_grants",
        "model_jobs",
        "quota_reservations",
        "quota_ledger",
        "inference_requests",
        "usage_summaries",
        "usage_summary_periods",
        "workers",
        "runtime_events",
    }
    if not required_beta <= _tables(beta) or not required_release <= _tables(release):
        raise ValueError("One or more required beta or release tables are missing")
    if beta.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("Configured Barra beta database failed its integrity check")

    beta_models, release_models = _prepare_models(beta, release)
    fp_row = beta.execute(
        "SELECT machine_fingerprint FROM service_state WHERE singleton=1"
    ).fetchone()
    target_fingerprint = str(fp_row[0] or "") if fp_row else ""
    if not target_fingerprint:
        raise ValueError("Barra beta service state has no current machine fingerprint")
    config_gpu_uuids = set(barra["managed_gpu_uuids"])
    profiles, profile_report, engine_version = _profile_candidates(
        beta,
        release_models,
        settings,
        config_gpu_uuids,
        target_fingerprint,
    )
    counts = {
        "barra_models": len(beta_models),
        "available_models": sum(row["state"] == "AVAILABLE" for row in beta_models),
        "disabled_models": sum(row["state"] == "DISABLED" for row in beta_models),
        "eligible_profiles": len(profiles),
        "models_with_profiles": profile_report["available_models_covered"],
        "profile_report": profile_report,
        "engine_version": engine_version,
        "mode": settings.serving_mode.value,
        "managed_gpu_count": len(config_gpu_uuids),
        "source_users_to_discard": int(beta.execute("SELECT count(*) FROM api_keys").fetchone()[0]),
        "source_grants_to_discard": int(
            beta.execute("SELECT count(*) FROM model_grants").fetchone()[0]
        ),
        "history_rows_to_discard": {
            table: int(beta.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in (
                "inference_requests",
                "quota_reservations",
                "quota_ledger",
                "usage_summaries",
                "usage_summary_periods",
                "model_jobs",
                "runtime_events",
                "workers",
            )
        },
    }
    return {
        "models": beta_models,
        "release_models": release_models,
        "profiles": profiles,
        "target_fingerprint": target_fingerprint,
        "counts": counts,
    }


def _archive_beta(
    source: Path,
    config: Path,
    vault: Path,
    archive_root: Path,
) -> Path:
    if not source.is_file() or not config.is_file() or not vault.is_file():
        raise ValueError("Configured Barra database, config, and matching vault are required")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = archive_root / f"barra-before-release-migration-{stamp}"
    archive.mkdir(parents=True, mode=0o700, exist_ok=False)
    archived_database = archive / source.name
    source_connection = _read_only(source)
    try:
        with closing(sqlite3.connect(archived_database)) as destination:
            source_connection.backup(destination)
            if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Archived Barra beta database failed its integrity check")
    finally:
        source_connection.close()
    shutil.copyfile(config, archive / "config.barra.toml")
    shutil.copyfile(vault, archive / vault.name)
    Fernet(vault.read_bytes().strip())
    archived_db = _immutable_read_only(archived_database)
    try:
        encrypted_keys = archived_db.execute("SELECT encrypted_api_key FROM api_keys").fetchall()
        fernet = Fernet((archive / vault.name).read_bytes().strip())
        for (encrypted,) in encrypted_keys:
            fernet.decrypt(str(encrypted).encode())
    finally:
        archived_db.close()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    dirty = (
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=ROOT, check=False, capture_output=True
        ).returncode
        != 0
    )
    file_hashes = {
        path.name: _sha256_file(path)
        for path in (archived_database, archive / "config.barra.toml", archive / vault.name)
    }
    manifest = {
        "purpose": "Consistent pre-release Barra beta archive; source files left unchanged",
        "created_at": _utc_now(),
        "source_database": str(source),
        "source_config": str(config),
        "source_vault": str(vault),
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "git_worktree_dirty": dirty,
        "files_sha256": file_hashes,
    }
    manifest_path = archive / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for path in archive.iterdir():
        os.chmod(path, 0o600)
    os.chmod(archive, 0o700)
    # Re-read the manifest and content hashes before using this frozen source.
    checked = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename, digest in checked["files_sha256"].items():
        if _sha256_file(archive / filename) != digest:
            raise ValueError("Barra beta archive failed its content hash verification")
    return archive


def _insert_model_rows(
    target: sqlite3.Connection,
    models: list[sqlite3.Row],
    release_models: dict[tuple[str, str], sqlite3.Row],
    admin_key_id: str,
) -> None:
    names = ",".join(MODEL_COLUMNS)
    placeholders = ",".join("?" for _ in MODEL_COLUMNS)
    for source in models:
        release_row = release_models[_model_identity(source)]
        values = {name: release_row[name] for name in MODEL_COLUMNS}
        parent = release_row["source_model_id"]
        values["source_model_id"] = str(parent) if parent is not None else None
        values["created_by_key_id"] = admin_key_id
        target.execute(
            f"INSERT INTO model_catalog ({names}) VALUES ({placeholders})",
            tuple(values[name] for name in MODEL_COLUMNS),
        )


def _clear_release_history(target: sqlite3.Connection) -> None:
    for table in (
        "runtime_events",
        "workers",
        "inference_requests",
        "quota_ledger",
        "quota_reservations",
        "usage_summaries",
        "usage_summary_periods",
        "model_jobs",
        "model_profiles",
        "model_grants",
        "model_catalog",
        "api_keys",
        "quota_accounts",
    ):
        target.execute(f'DELETE FROM "{table}"')


def _build_target_database(
    release: sqlite3.Connection,
    plan: dict[str, Any],
    staged_database: Path,
    staged_vault: Path,
    expected_models: int,
    expected_profiles: int,
) -> str:
    key_id = str(uuid.uuid4())
    account_id = str(uuid.uuid4())
    token, prefix = issue_api_key(key_id)
    token_hash = hash_api_key(token)
    key_vault = ApiKeyVault(staged_vault)
    encrypted = key_vault.encrypt(token)
    now = _utc_now()
    account_nickname = "barra-admin-account"
    key_nickname = "barra-admin"

    release.execute("PRAGMA foreign_keys=ON")
    release.execute("BEGIN IMMEDIATE")
    try:
        _clear_release_history(release)
        release.execute(
            "INSERT INTO quota_accounts"
            "(id,nickname,balance_tokens,limit_tokens,usage_baseline_tokens,usage_reset_at,unlimited,created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (account_id, account_nickname, 0, 0, 0, None, 1, now),
        )
        release.execute(
            "INSERT INTO api_keys"
            "(id,nickname,role,quota_account_id,token_prefix,token_hash,encrypted_api_key,active,created_at,last_used_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                key_id,
                key_nickname,
                "admin",
                account_id,
                prefix,
                token_hash,
                encrypted,
                1,
                now,
                None,
            ),
        )
        _insert_model_rows(
            release,
            plan["models"],
            plan["release_models"],
            key_id,
        )
        for item in plan["profiles"]:
            release.execute(
                "INSERT INTO model_profiles"
                "(id,model_id,machine_fingerprint,profile_key,profile_json,verified_at,active)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    item["id"],
                    item["model_id"],
                    item["machine_fingerprint"],
                    item["profile_key"],
                    item["profile_json"],
                    item["verified_at"],
                    1,
                ),
            )
        release.execute(
            "UPDATE service_state SET mode='ACTIVE',machine_fingerprint=?,updated_at=? "
            "WHERE singleton=1",
            (plan["target_fingerprint"], now),
        )
        if release.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Prepared Barra release records violate a foreign-key constraint")
        if release.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Prepared Barra release database failed its integrity check")
        actual_models = int(release.execute("SELECT count(*) FROM model_catalog").fetchone()[0])
        actual_profiles = int(
            release.execute("SELECT count(*) FROM model_profiles WHERE active=1").fetchone()[0]
        )
        if actual_models != expected_models or actual_profiles != expected_profiles:
            raise ValueError("Prepared Barra release database row counts differ from preflight")
        release.commit()
    except BaseException:
        release.rollback()
        key_vault.path.unlink(missing_ok=True)
        raise

    with closing(sqlite3.connect(staged_database)) as destination:
        release.backup(destination)
        destination.execute("PRAGMA journal_mode=DELETE")
        destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if destination.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Written Barra release database failed its integrity check")
        if destination.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Written Barra release database failed its foreign-key check")
        stored = destination.execute(
            "SELECT token_hash,encrypted_api_key FROM api_keys WHERE id=?", (key_id,)
        ).fetchone()
        if stored is None:
            raise ValueError("Fresh Barra administrator key was not stored")
        clear = key_vault.decrypt(str(stored[1]))
        if clear != token or not verify_api_key(str(stored[0]), clear):
            raise ValueError("Fresh Barra administrator key did not verify against its vault")
        result = {
            "api_keys": int(destination.execute("SELECT count(*) FROM api_keys").fetchone()[0]),
            "quota_accounts": int(
                destination.execute("SELECT count(*) FROM quota_accounts").fetchone()[0]
            ),
            "model_catalog": int(
                destination.execute("SELECT count(*) FROM model_catalog").fetchone()[0]
            ),
            "model_profiles": int(
                destination.execute("SELECT count(*) FROM model_profiles").fetchone()[0]
            ),
            "model_grants": int(
                destination.execute("SELECT count(*) FROM model_grants").fetchone()[0]
            ),
            "model_jobs": int(destination.execute("SELECT count(*) FROM model_jobs").fetchone()[0]),
            "requests": int(
                destination.execute("SELECT count(*) FROM inference_requests").fetchone()[0]
            ),
            "reservations": int(
                destination.execute("SELECT count(*) FROM quota_reservations").fetchone()[0]
            ),
            "workers": int(destination.execute("SELECT count(*) FROM workers").fetchone()[0]),
            "runtime_events": int(
                destination.execute("SELECT count(*) FROM runtime_events").fetchone()[0]
            ),
        }
        if result != {
            "api_keys": 1,
            "quota_accounts": 1,
            "model_catalog": expected_models,
            "model_profiles": expected_profiles,
            "model_grants": 0,
            "model_jobs": 0,
            "requests": 0,
            "reservations": 0,
            "workers": 0,
            "runtime_events": 0,
        }:
            raise ValueError(
                "Written Barra release database did not match its clean-state contract"
            )
    os.chmod(staged_database, 0o600)
    os.chmod(staged_vault, 0o600)
    # Keep the key in the vault/database only. Nothing returns to stdout or the audit.
    del token
    return json.dumps(result, sort_keys=True)


def _apply(
    *,
    beta_path: Path,
    beta_config: Path,
    beta_vault: Path,
    release_path: Path,
    target_config: Path,
    target_database: Path,
    release_config: dict[str, Any],
    settings: Settings,
    barra: dict[str, Any],
    archive_root: Path,
) -> dict[str, Any]:
    final_vault = default_key_vault_path(target_database)
    outputs = (target_database, final_vault, target_config)
    if any(path.exists() for path in outputs):
        raise ValueError("A Barra release target already exists; refusing to overwrite it")
    if not beta_vault.is_file():
        raise ValueError("The configured Barra beta key vault is missing")
    archive = _archive_beta(beta_path, beta_config, beta_vault, archive_root)
    archived_db = archive / beta_path.name
    beta = _immutable_read_only(archived_db)
    release = _online_snapshot(release_path)
    stage_id = uuid.uuid4().hex
    target_database.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if any(target_database.parent.iterdir()):
        beta.close()
        release.close()
        raise ValueError("Barra release state directory already contains files")
    os.chmod(target_database.parent, 0o700)
    staged_database = target_database.with_name(f".{target_database.name}.{stage_id}.stage")
    staged_vault = final_vault.with_name(f".{final_vault.name}.{stage_id}.stage")
    staged_config = target_config.with_name(f".{target_config.name}.{stage_id}.stage")
    moved: list[Path] = []
    try:
        plan = _prepare(beta, release, settings, barra)
        database_counts_json = _build_target_database(
            release,
            plan,
            staged_database,
            staged_vault,
            len(plan["models"]),
            len(plan["profiles"]),
        )
        staged_config.write_text(_serialize_toml(release_config), encoding="utf-8")
        os.chmod(staged_config, 0o600)
        with staged_config.open("rb") as stream:
            parsed_config = tomllib.load(stream)
        parsed_settings = Settings.model_validate(parsed_config)
        if parsed_settings.serving_mode.value != "queue":
            raise ValueError("Staged Barra config does not select queue mode")
        if parsed_settings.managed_gpu_uuids != list(barra["managed_gpu_uuids"]):
            raise ValueError("Staged Barra config changed the configured GPU UUIDs")
        staged_db_check = sqlite3.connect(f"file:{staged_database}?mode=ro", uri=True)
        try:
            if staged_db_check.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("Staged Barra database is not release schema version 1")
            if staged_db_check.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Staged Barra database failed its final integrity check")
        finally:
            staged_db_check.close()

        # Publish the private config and vault before the database. The database is
        # the final commit point, so a server never sees a partial release state.
        os.replace(staged_config, target_config)
        moved.append(target_config)
        os.replace(staged_vault, final_vault)
        moved.append(final_vault)
        os.replace(staged_database, target_database)
        moved.append(target_database)
        for path in outputs:
            os.chmod(path, 0o600)
        return {
            "archive": str(archive),
            "config": str(target_config),
            "database": str(target_database),
            "counts": json.loads(database_counts_json),
            "preflight": plan["counts"],
            "source_files_changed": False,
            "current_release_database_changed": False,
            "gpu_inference_performed": False,
        }
    except BaseException:
        for path in moved:
            path.unlink(missing_ok=True)
        for path in (staged_database, staged_vault, staged_config):
            path.unlink(missing_ok=True)
        raise
    finally:
        beta.close()
        release.close()


def _resolve_database(config_data: dict[str, Any]) -> Path:
    value = Path(str(config_data.get("database_path", "")))
    return (ROOT / value).resolve() if not value.is_absolute() else value.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--barra-config", type=Path, default=DEFAULT_BARRA_CONFIG)
    parser.add_argument("--release-config", type=Path, default=DEFAULT_RELEASE_CONFIG)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--target-database", type=Path, default=DEFAULT_TARGET_DATABASE)
    parser.add_argument("--archive-root", type=Path, default=ROOT / "db_backup")
    parser.add_argument(
        "--apply", action="store_true", help="Archive sources and create Barra release state"
    )
    args = parser.parse_args()

    config_paths = (args.barra_config, args.release_config)
    if not all(path.is_file() for path in config_paths):
        parser.error("Barra and current release TOML configurations must exist")
    barra_config = args.barra_config.resolve()
    release_config_path = args.release_config.resolve()
    target_config = args.target_config.resolve()
    target_database = args.target_database.resolve()
    if target_database == _resolve_database(
        tomllib.loads(barra_config.read_text(encoding="utf-8"))
    ):
        parser.error("Barra beta source and release target must be different database files")
    if target_database == _resolve_database(
        tomllib.loads(release_config_path.read_text(encoding="utf-8"))
    ):
        parser.error("Barra target and current release database must be different files")
    if any(
        path.exists()
        for path in (target_config, target_database, default_key_vault_path(target_database))
    ):
        parser.error("A Barra release target already exists; refusing to overwrite it")
    try:
        release_data, settings, barra = _config_data(
            barra_config,
            release_config_path,
            target_database,
        )
        beta_path = _resolve_database(barra)
        release_path = _resolve_database(
            tomllib.loads(release_config_path.read_text(encoding="utf-8"))
        )
        beta_vault = default_key_vault_path(beta_path)
        if not beta_path.is_file() or not release_path.is_file():
            raise ValueError("Configured Barra beta or current release database is missing")
        beta = _online_snapshot(beta_path)
        release = _online_snapshot(release_path)
        try:
            plan = _prepare(beta, release, settings, barra)
        finally:
            beta.close()
            release.close()
        print(json.dumps({"preflight": plan["counts"]}, indent=2))
        if not args.apply:
            print(
                "Dry run only; no database, vault, or config was changed. "
                "Pass --apply to create the Barra release state."
            )
            return
        result = _apply(
            beta_path=beta_path,
            beta_config=barra_config,
            beta_vault=beta_vault,
            release_path=release_path,
            target_config=target_config,
            target_database=target_database,
            release_config=release_data,
            settings=settings,
            barra=barra,
            archive_root=args.archive_root.resolve(),
        )
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, RuntimeError) as exc:
        parser.error(f"Barra migration preflight failed: {exc}")
    print("Applied:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
