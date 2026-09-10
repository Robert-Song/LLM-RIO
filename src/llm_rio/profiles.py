from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_rio.domain import (
    CURRENT_VRAM_MEASUREMENT_VERSION,
    CatalogState,
    Engine,
    PlacementProfile,
)
from llm_rio.errors import RioError
from llm_rio.storage import Database, _now


@dataclass(frozen=True, slots=True)
class StoredProfile:
    """A placement profile together with its catalog activation state."""

    profile: PlacementProfile
    active: bool


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def profile_from_dict(raw: dict[str, Any]) -> PlacementProfile:
    memory_backend = str(raw.get("memory_backend", "native"))
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
        memory_backend=memory_backend,
        sleep_vram_mib_per_gpu=(
            None
            if raw.get("sleep_vram_mib_per_gpu") is None
            else tuple(int(value) for value in raw["sleep_vram_mib_per_gpu"])
        ),
        weight_cache_offload_seconds=(
            None
            if raw.get("weight_cache_offload_seconds") is None
            else float(raw["weight_cache_offload_seconds"])
        ),
        weight_cache_activation_seconds=(
            None
            if raw.get("weight_cache_activation_seconds") is None
            else float(raw["weight_cache_activation_seconds"])
        ),
        host_cache_mib=(
            None if raw.get("host_cache_mib") is None else float(raw["host_cache_mib"])
        ),
        normal_verified=bool(raw.get("normal_verified", True)),
        kvcached_verified=bool(raw.get("kvcached_verified", memory_backend == "kvcached")),
        vram_measurement_version=int(raw.get("vram_measurement_version", 1)),
        vram_baseline_mib_per_gpu=(
            None
            if raw.get("vram_baseline_mib_per_gpu") is None
            else tuple(int(value) for value in raw["vram_baseline_mib_per_gpu"])
        ),
        wake_peak_vram_mib_per_gpu=(
            None
            if raw.get("wake_peak_vram_mib_per_gpu") is None
            else tuple(int(value) for value in raw["wake_peak_vram_mib_per_gpu"])
        ),
    )


def profile_verified_for_mode(
    profile: PlacementProfile,
    *,
    kvcached_required: bool,
    ram_weight_cache_required: bool = False,
) -> bool:
    """Return whether this exact profile was measured for the selected runtime."""
    if profile.vram_measurement_version != CURRENT_VRAM_MEASUREMENT_VERSION:
        return False
    if profile.vram_baseline_mib_per_gpu is None:
        return False
    required_vectors = (
        profile.idle_vram_mib_per_gpu,
        profile.peak_vram_mib_per_gpu,
        profile.gpu_headroom_mib_per_gpu,
        profile.vram_baseline_mib_per_gpu,
    )
    if any(len(values) != profile.gpu_count for values in required_vectors):
        return False
    if any(value < 0 for values in required_vectors for value in values):
        return False
    if any(profile.gpu_headroom_mib_per_gpu):
        return False
    if profile.wake_peak_vram_mib_per_gpu is not None and (
        len(profile.wake_peak_vram_mib_per_gpu) != profile.gpu_count
        or any(value < 0 for value in profile.wake_peak_vram_mib_per_gpu)
    ):
        return False
    if ram_weight_cache_required and (
        profile.engine is not Engine.VLLM
        or profile.sleep_vram_mib_per_gpu is None
        or len(profile.sleep_vram_mib_per_gpu) != profile.gpu_count
        or any(value < 0 for value in profile.sleep_vram_mib_per_gpu)
        or profile.wake_peak_vram_mib_per_gpu is None
    ):
        return False

    expected_backend = "kvcached" if kvcached_required else "native"
    if profile.memory_backend != expected_backend:
        return False
    if profile.engine is not Engine.VLLM:
        return not kvcached_required and profile.normal_verified
    return profile.kvcached_verified if kvcached_required else profile.normal_verified


_LAUNCH_CONFIGURATION_FIELDS = (
    "engine",
    "model_revision",
    "engine_version",
    "gpu_count",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "eligible_gpu_sets",
    "dtype",
    "quantization",
    "max_model_len",
    "max_num_seqs",
    "max_num_batched_tokens",
    "launch_args",
    "gpu_memory_utilization",
    "memory_backend",
)


def launch_configuration_changed(before: PlacementProfile, after: PlacementProfile) -> bool:
    """Return whether an edit changes anything exercised by verification."""
    return any(
        getattr(before, field) != getattr(after, field) for field in _LAUNCH_CONFIGURATION_FIELDS
    )


def invalidate_profile_measurements(profile: PlacementProfile) -> PlacementProfile:
    """Clear measurements and trust after a launch-affecting administrator edit."""
    count = profile.gpu_count
    return PlacementProfile(
        **{
            **asdict(profile),
            "predicted_tokens_per_second": 0.0,
            "load_and_warmup_seconds": 0.0,
            "idle_vram_mib_per_gpu": (0,) * count,
            "peak_vram_mib_per_gpu": (0,) * count,
            "gpu_headroom_mib_per_gpu": (0,) * count,
            "kv_cache_capacity_tokens": None,
            "max_full_length_concurrency": None,
            "sleep_vram_mib_per_gpu": None,
            "weight_cache_offload_seconds": None,
            "weight_cache_activation_seconds": None,
            "host_cache_mib": None,
            "normal_verified": False,
            "kvcached_verified": False,
            "vram_measurement_version": 0,
            "vram_baseline_mib_per_gpu": None,
            "wake_peak_vram_mib_per_gpu": None,
        }
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
            "model_id",
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
            "launch_args",
            "memory_backend",
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

    async def set_model_verified_for_both(self, model_id: str) -> int:
        """Explicitly trust every active profile for both launch backends on this machine."""
        async with self.database.transaction() as connection:
            rows = await (
                await connection.execute(
                    """
                    SELECT id, profile_json FROM model_profiles
                     WHERE model_id = ? AND machine_fingerprint = ? AND active = 1
                    """,
                    (model_id, self.machine_fingerprint),
                )
            ).fetchall()
            for row in rows:
                raw = json.loads(row["profile_json"])
                raw["normal_verified"] = True
                raw["kvcached_verified"] = True
                await connection.execute(
                    """
                    UPDATE model_profiles SET profile_json = ?, verified_at = ?
                     WHERE id = ?
                    """,
                    (json.dumps(raw), _now(), row["id"]),
                )
        return len(rows)

    async def set_profile_backend_verified(
        self,
        *,
        model_id: str,
        profile_id: str,
        backend: str,
        verified: bool,
    ) -> bool:
        """Set one backend verification flag for one local placement profile."""
        field = {
            "native": "normal_verified",
            "kvcached": "kvcached_verified",
        }.get(backend)
        if field is None:
            raise ValueError(f"Unsupported verification backend: {backend}")
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT profile_json FROM model_profiles
                     WHERE id = ? AND model_id = ? AND machine_fingerprint = ?
                    """,
                    (profile_id, model_id, self.machine_fingerprint),
                )
            ).fetchone()
            if row is None:
                return False
            raw = json.loads(row["profile_json"])
            raw[field] = verified
            await connection.execute(
                """
                UPDATE model_profiles SET profile_json = ?, verified_at = ?
                 WHERE id = ?
                """,
                (json.dumps(raw), _now(), profile_id),
            )
        return True

    @staticmethod
    def _model_rope_configuration(model: dict[str, Any]) -> tuple[bool, int, dict[str, Any]]:
        artifact_path = model.get("artifact_path")
        config_path = Path(str(artifact_path or "")) / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RioError(
                "model_config_unavailable",
                "The source model config is required to create a YaRN extension",
                status_code=409,
            ) from exc
        text_config = config.get("text_config")
        nested = isinstance(text_config, dict)
        rope_owner = text_config if nested else config
        raw_maximum = rope_owner.get("max_position_embeddings")
        if not isinstance(raw_maximum, int) or raw_maximum <= 0:
            raise RioError(
                "model_context_unknown",
                "The source model does not declare max_position_embeddings",
                status_code=409,
            )
        rope_parameters = rope_owner.get("rope_parameters")
        if not isinstance(rope_parameters, dict):
            rope_parameters = rope_owner.get("rope_scaling")
        return nested, raw_maximum, dict(rope_parameters or {})

    @staticmethod
    def _with_yarn_launch_args(
        launch_args: dict[str, Any],
        *,
        nested_text_config: bool,
        original_max_model_len: int,
        max_model_len: int,
        factor: float,
        base_rope_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        updated = dict(launch_args)
        raw_hf_overrides = updated.get("hf_overrides", {})
        if isinstance(raw_hf_overrides, str):
            try:
                raw_hf_overrides = json.loads(raw_hf_overrides)
            except json.JSONDecodeError as exc:
                raise RioError(
                    "invalid_hf_overrides",
                    "The source profile has invalid JSON in hf_overrides",
                    status_code=409,
                ) from exc
        if not isinstance(raw_hf_overrides, dict):
            raise RioError(
                "invalid_hf_overrides",
                "The source profile hf_overrides must be a JSON object",
                status_code=409,
            )
        hf_overrides = dict(raw_hf_overrides)
        if nested_text_config:
            raw_text_overrides = hf_overrides.get("text_config", {})
            if not isinstance(raw_text_overrides, dict):
                raise RioError(
                    "invalid_hf_overrides",
                    "The source profile text_config override must be a JSON object",
                    status_code=409,
                )
            owner = dict(raw_text_overrides)
        else:
            owner = hf_overrides
        existing_rope = owner.get("rope_parameters")
        if not isinstance(existing_rope, dict):
            existing_rope = owner.get("rope_scaling")
        rope_parameters = dict(base_rope_parameters)
        if isinstance(existing_rope, dict):
            rope_parameters.update(existing_rope)
        rope_parameters.update(
            {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_model_len,
            }
        )
        owner["max_position_embeddings"] = max_model_len
        owner.pop("rope_scaling", None)
        owner["rope_parameters"] = rope_parameters
        if nested_text_config:
            hf_overrides["text_config"] = owner
        updated["hf_overrides"] = hf_overrides
        return updated

    async def clone_model(
        self,
        *,
        source_model: dict[str, Any],
        nickname: str,
        creator_key_id: str,
        request_defaults: dict[str, Any],
        max_model_len: int | None,
        yarn_factor: float | None,
        yarn_original_max_model_len: int | None,
        inherit_grants: bool,
    ) -> tuple[dict[str, Any], list[PlacementProfile]]:
        """Clone active profiles into a separately scheduled model sharing one artifact."""
        if source_model.get("state") != CatalogState.AVAILABLE.value:
            raise RioError(
                "source_model_unavailable",
                "Only an available model can be cloned",
                status_code=409,
            )
        if not source_model.get("artifact_path"):
            raise RioError(
                "model_artifact_missing",
                "The source model artifact is not available on this machine",
                status_code=409,
            )
        source_records = [
            record
            for record in await self.records_for_model(str(source_model["id"]))
            if record.active
        ]
        if not source_records:
            raise RioError(
                "model_verification_required",
                "The source model has no active placement profiles on this machine",
                status_code=409,
            )

        nested_text_config = False
        native_max_model_len = 0
        base_rope_parameters: dict[str, Any] = {}
        if yarn_factor is not None:
            if any(record.profile.engine is not Engine.VLLM for record in source_records):
                raise RioError(
                    "yarn_requires_vllm",
                    "YaRN extension is supported only for vLLM placement profiles",
                    status_code=422,
                )
            nested_text_config, native_max_model_len, base_rope_parameters = (
                self._model_rope_configuration(source_model)
            )
        original_max_model_len = yarn_original_max_model_len or native_max_model_len
        if yarn_factor is not None:
            inferred_max_model_len = round(original_max_model_len * yarn_factor)
            if max_model_len is None:
                max_model_len = inferred_max_model_len
            elif not math.isclose(
                max_model_len,
                original_max_model_len * yarn_factor,
                rel_tol=0,
                abs_tol=1,
            ):
                raise RioError(
                    "invalid_yarn_context",
                    "max_model_len must equal yarn_original_max_model_len "
                    "multiplied by yarn_factor",
                    status_code=422,
                    details={"expected_max_model_len": inferred_max_model_len},
                )

        model_id = str(uuid.uuid4())
        cloned_profiles: list[PlacementProfile] = []
        cloned_profile_rows: list[tuple[str, str, str, str, str]] = []
        for record in source_records:
            source_profile = record.profile
            target_max_model_len = max_model_len or source_profile.max_model_len
            launch_args = dict(source_profile.launch_args)
            if yarn_factor is not None:
                launch_args = self._with_yarn_launch_args(
                    launch_args,
                    nested_text_config=nested_text_config,
                    original_max_model_len=original_max_model_len,
                    max_model_len=target_max_model_len,
                    factor=yarn_factor,
                    base_rope_parameters=base_rope_parameters,
                )
            profile = PlacementProfile(
                **{
                    **asdict(source_profile),
                    "id": str(uuid.uuid4()),
                    "model_id": model_id,
                    "max_model_len": target_max_model_len,
                    "launch_args": launch_args,
                }
            )
            measurements_invalidated = launch_configuration_changed(source_profile, profile)
            if measurements_invalidated:
                profile = invalidate_profile_measurements(profile)
            raw = profile_to_dict(profile)
            raw["administrator_override"] = True
            raw["administrator_override_at"] = _now()
            raw["cloned_from_profile_id"] = source_profile.id
            if measurements_invalidated:
                raw["measurements_invalidated_at"] = _now()
            cloned_profiles.append(profile)
            cloned_profile_rows.append(
                (
                    profile.id,
                    profile.machine_fingerprint,
                    profile_key(raw),
                    json.dumps(raw),
                    _now(),
                )
            )

        defaults = dict(source_model.get("request_defaults") or {})
        defaults.update(request_defaults)
        limits = dict(source_model.get("request_limits") or {})
        limits["max_context_tokens"] = min(profile.max_model_len for profile in cloned_profiles)
        now = _now()
        async with self.database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO model_catalog
                    (id, nickname, huggingface_repo, requested_revision, resolved_revision,
                     state, artifact_path, artifact_hashes_json, capabilities_json,
                     request_limits_json, request_defaults_json, source_model_id,
                     created_by_key_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    nickname,
                    source_model["huggingface_repo"],
                    source_model.get("requested_revision"),
                    source_model.get("resolved_revision"),
                    CatalogState.AVAILABLE.value,
                    source_model["artifact_path"],
                    json.dumps(source_model.get("artifact_hashes") or []),
                    json.dumps(source_model.get("capabilities") or []),
                    json.dumps(limits),
                    json.dumps(defaults),
                    source_model["id"],
                    creator_key_id,
                    now,
                    now,
                ),
            )
            for profile_id, fingerprint, raw_key, profile_json, verified_at in cloned_profile_rows:
                await connection.execute(
                    """
                    INSERT INTO model_profiles
                        (id, model_id, machine_fingerprint, profile_key, profile_json,
                         verified_at, active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (profile_id, model_id, fingerprint, raw_key, profile_json, verified_at),
                )
            if inherit_grants:
                await connection.execute(
                    """
                    INSERT INTO model_grants(key_id, model_id, created_at)
                    SELECT key_id, ?, ? FROM model_grants WHERE model_id = ?
                    """,
                    (model_id, now, source_model["id"]),
                )
        cloned_model = await self.database.model_by_id(model_id)
        if cloned_model is None:
            raise RuntimeError("cloned model was not persisted")
        return cloned_model, cloned_profiles

    async def update(
        self,
        profile: PlacementProfile,
        *,
        make_default: bool,
    ) -> bool:
        """Persist an override, invalidating measurements if launch behavior changed."""
        async with self.database.transaction() as connection:
            existing = await (
                await connection.execute(
                    """
                    SELECT active, profile_json FROM model_profiles
                     WHERE id = ? AND model_id = ? AND machine_fingerprint = ?
                    """,
                    (profile.id, profile.model_id, self.machine_fingerprint),
                )
            ).fetchone()
            if existing is None:
                return False

            before_raw = json.loads(existing["profile_json"])
            before_raw["id"] = profile.id
            before_raw["machine_fingerprint"] = self.machine_fingerprint
            before = profile_from_dict(before_raw)
            measurements_invalidated = launch_configuration_changed(before, profile)
            if measurements_invalidated:
                profile = invalidate_profile_measurements(profile)

            now = _now()
            raw = profile_to_dict(profile)
            raw["administrator_override"] = True
            raw["administrator_override_at"] = now
            if measurements_invalidated:
                raw["measurements_invalidated_at"] = now
            profile_json = json.dumps(raw)
            raw_profile_key = profile_key(raw)

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
                limits["max_context_tokens"] = min(int(row["max_model_len"]) for row in active_rows)
                await connection.execute(
                    """
                    UPDATE model_catalog
                       SET request_limits_json = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (json.dumps(limits), now, profile.model_id),
                )
        return True

    async def set_active(self, *, model_id: str, profile_id: str, active: bool) -> bool:
        """Enable or disable one stored placement profile atomically."""
        async with self.database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE model_profiles SET active = ?
                 WHERE id = ? AND model_id = ? AND machine_fingerprint = ?
                """,
                (int(active), profile_id, model_id, self.machine_fingerprint),
            )
            if cursor.rowcount == 0:
                return False
            active_rows = await (
                await connection.execute(
                    """
                    SELECT json_extract(profile_json, '$.max_model_len') AS max_model_len
                      FROM model_profiles
                     WHERE model_id = ? AND machine_fingerprint = ? AND active = 1
                    """,
                    (model_id, self.machine_fingerprint),
                )
            ).fetchall()
            if active_rows:
                model_row = await (
                    await connection.execute(
                        "SELECT request_limits_json FROM model_catalog WHERE id = ?", (model_id,)
                    )
                ).fetchone()
                if model_row is not None:
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
                        (json.dumps(limits), _now(), model_id),
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
