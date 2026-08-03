from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from llm_rio.domain import Engine, PlacementProfile
from llm_rio.storage import Database, _now


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
        quantization=raw.get("quantization"),
        max_model_len=int(raw["max_model_len"]),
        max_num_seqs=int(raw["max_num_seqs"]),
        max_num_batched_tokens=int(raw["max_num_batched_tokens"]),
        predicted_tokens_per_second=float(raw["predicted_tokens_per_second"]),
        load_and_warmup_seconds=float(raw["load_and_warmup_seconds"]),
        idle_vram_mib_per_gpu=tuple(raw["idle_vram_mib_per_gpu"]),
        peak_vram_mib_per_gpu=tuple(raw["peak_vram_mib_per_gpu"]),
        gpu_headroom_mib_per_gpu=tuple(raw["gpu_headroom_mib_per_gpu"]),
        capabilities=frozenset(raw["capabilities"]),
        launch_args=dict(raw.get("launch_args", {})),
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
            SELECT profile_json FROM model_profiles
             WHERE model_id = ? AND machine_fingerprint = ? AND active = 1
             ORDER BY json_extract(profile_json, '$.gpu_count'),
                      json_extract(profile_json, '$.predicted_tokens_per_second') DESC
            """,
            (model_id, self.machine_fingerprint),
        )
        return [profile_from_dict(json.loads(row["profile_json"])) for row in rows]

    async def save(self, profile: PlacementProfile, raw_profile_key: str) -> None:
        await self.database.execute(
            """
            INSERT INTO model_profiles
                (id, model_id, machine_fingerprint, profile_key, profile_json, verified_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_key) DO UPDATE SET
                profile_json = excluded.profile_json,
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

