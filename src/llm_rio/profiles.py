from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from llm_rio.domain import Engine, PlacementProfile
from llm_rio.storage import Database, _now


@dataclass(frozen=True, slots=True)
class StoredProfile:
    """A placement profile together with its catalog activation state."""

    profile: PlacementProfile
    active: bool


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def profile_from_dict(raw: dict[str, Any]) -> PlacementProfile:
    return PlacementProfile(
        id=raw["id"],
        model_id=raw["model_id"],
        model_revision=raw["model_revision"],
        engine=Engine(raw["engine"]),
        engine_version=raw["engine_version"],
        machine_fingerprint=raw["machine_fingerprint"],
        gpu_count=int(raw["gpu_count"]),
        tensor_parallel_size=int(raw["tensor_parallel_size"]),
        pipeline_parallel_size=int(raw["pipeline_parallel_size"]),
        eligible_gpu_sets=tuple(tuple(group) for group in raw["eligible_gpu_sets"]),
        dtype=raw["dtype"],
        quantization=raw["quantization"],
        max_model_len=int(raw["max_model_len"]),
        max_num_seqs=_optional_int(raw["max_num_seqs"]),
        max_num_batched_tokens=_optional_int(raw["max_num_batched_tokens"]),
        predicted_tokens_per_second=float(raw["predicted_tokens_per_second"]),
        load_and_warmup_seconds=float(raw["load_and_warmup_seconds"]),
        idle_vram_mib_per_gpu=tuple(raw["idle_vram_mib_per_gpu"]),
        peak_vram_mib_per_gpu=tuple(raw["peak_vram_mib_per_gpu"]),
        gpu_headroom_mib_per_gpu=tuple(raw["gpu_headroom_mib_per_gpu"]),
        capabilities=frozenset(raw["capabilities"]),
        launch_args=dict(raw["launch_args"]),
        gpu_memory_utilization=float(raw["gpu_memory_utilization"]),
        kv_cache_capacity_tokens=_optional_int(raw["kv_cache_capacity_tokens"]),
        max_full_length_concurrency=(
            None
            if raw["max_full_length_concurrency"] is None
            else float(raw["max_full_length_concurrency"])
        ),
    )


def profile_to_dict(profile: PlacementProfile) -> dict[str, Any]:
    result = asdict(profile)
    result["engine"] = profile.engine.value
    result["eligible_gpu_sets"] = [list(group) for group in profile.eligible_gpu_sets]
    result["capabilities"] = sorted(profile.capabilities)
    return result


def profile_key(raw: dict[str, Any]) -> str:
    identifying_fields = {
        key: raw.get(key)
        for key in (
            "model_revision",
            "artifact_hashes",
            "engine",
            "engine_version",
            "machine_fingerprint",
            "gpu_models",
            "topology_class",
            "gpu_count",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "dtype",
            "quantization",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "gpu_memory_utilization",
            "multimodal_limits",
            "tool_parser",
            "eligible_gpu_sets",
            "chat_template_hash",
        )
    }
    return hashlib.sha256(json.dumps(identifying_fields, sort_keys=True).encode()).hexdigest()


class ProfileRepository:
    def __init__(self, database: Database, machine_fingerprint: str) -> None:
        self.database = database
        self.machine_fingerprint = machine_fingerprint

    async def for_model(self, model_id: str) -> list[PlacementProfile]:
        rows = await self.database.fetchall(
            """
            SELECT id, profile_json FROM model_profiles
             WHERE model_id = ? AND machine_fingerprint = ? AND active = 1
             ORDER BY json_extract(profile_json, '$.gpu_count'),
                      json_extract(profile_json, '$.predicted_tokens_per_second') DESC
            """,
            (model_id, self.machine_fingerprint),
        )
        profiles: list[PlacementProfile] = []
        for row in rows:
            data = json.loads(row["profile_json"])
            data["id"] = row["id"]
            data["machine_fingerprint"] = self.machine_fingerprint
            profiles.append(profile_from_dict(data))
        return profiles

    async def records_for_model(self, model_id: str) -> list[StoredProfile]:
        """Return active and inactive profiles for administrator profile management."""
        rows = await self.database.fetchall(
            """
            SELECT id, profile_json, active FROM model_profiles
             WHERE model_id = ? AND machine_fingerprint = ?
             ORDER BY active DESC,
                      json_extract(profile_json, '$.gpu_count'),
                      json_extract(profile_json, '$.predicted_tokens_per_second') DESC
            """,
            (model_id, self.machine_fingerprint),
        )
        records: list[StoredProfile] = []
        for row in rows:
            data = json.loads(row["profile_json"])
            data["id"] = row["id"]
            data["machine_fingerprint"] = self.machine_fingerprint
            records.append(
                StoredProfile(profile=profile_from_dict(data), active=bool(row["active"]))
            )
        return records

    async def update(
        self,
        profile: PlacementProfile,
        *,
        make_default: bool,
    ) -> bool:
        """Persist an administrator override and keep catalog context limits in sync."""
        raw = profile_to_dict(profile)
        raw["administrator_override"] = True
        raw["administrator_override_at"] = _now()
        profile_json = json.dumps(raw)
        raw_profile_key = profile_key(raw)
        async with self.database.transaction() as connection:
            existing = await (
                await connection.execute(
                    """
                    SELECT active FROM model_profiles
                     WHERE id = ? AND model_id = ? AND machine_fingerprint = ?
                    """,
                    (profile.id, profile.model_id, self.machine_fingerprint),
                )
            ).fetchone()
            if existing is None:
                return False
            if make_default:
                await connection.execute(
                    """
                    UPDATE model_profiles SET active = 0
                     WHERE model_id = ? AND machine_fingerprint = ?
                    """,
                    (profile.model_id, self.machine_fingerprint),
                )
            active = 1 if make_default else int(existing["active"])
            await connection.execute(
                """
                UPDATE model_profiles
                   SET profile_key = ?, profile_json = ?, active = ?
                 WHERE id = ?
                """,
                (raw_profile_key, profile_json, active, profile.id),
            )
            model_row = await (
                await connection.execute(
                    "SELECT request_limits_json FROM model_catalog WHERE id = ?",
                    (profile.model_id,),
                )
            ).fetchone()
            active_rows = await (
                await connection.execute(
                    """
                    SELECT json_extract(profile_json, '$.max_model_len') AS max_model_len
                      FROM model_profiles
                     WHERE model_id = ? AND machine_fingerprint = ? AND active = 1
                    """,
                    (profile.model_id, self.machine_fingerprint),
                )
            ).fetchall()
            if model_row is not None and active_rows:
                limits = json.loads(model_row["request_limits_json"] or "{}")
                limits["max_context_tokens"] = min(
                    int(row["max_model_len"]) for row in active_rows
                )
                await connection.execute(
                    """
                    UPDATE model_catalog
                       SET request_limits_json = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (json.dumps(limits), _now(), profile.model_id),
                )
        return True

    async def save(self, profile: PlacementProfile, raw_profile_key: str) -> None:
        await self.database.execute(
            """
            INSERT INTO model_profiles
                (id, model_id, machine_fingerprint, profile_key, profile_json, verified_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key) DO UPDATE SET
                profile_json = json_set(excluded.profile_json, '$.id', model_profiles.id),
                verified_at = excluded.verified_at,
                active = 1
            """,
            (
                profile.id,
                profile.model_id,
                profile.machine_fingerprint,
                raw_profile_key,
                json.dumps(profile_to_dict(profile)),
                _now(),
            ),
        )
