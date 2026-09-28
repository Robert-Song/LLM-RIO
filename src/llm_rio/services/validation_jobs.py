"""Persist an exact selected launch before scheduling any validation work."""

from __future__ import annotations

import json
from typing import Any

from llm_rio.domain import PlacementProfile
from llm_rio.errors import RioError
from llm_rio.profiles import profile_from_dict
from llm_rio.storage import Database, _now


def profile_overrides(profile: PlacementProfile) -> dict[str, Any]:
    args = dict(profile.launch_args)
    artifact = args.pop("model", None)
    args.pop("enable_sleep_mode", None)
    args["dtype"] = profile.dtype
    if profile.quantization is not None:
        args["quantization"] = profile.quantization
    return {
        "max_model_len": profile.max_model_len,
        "max_num_seqs": profile.max_num_seqs,
        "max_num_batched_tokens": profile.max_num_batched_tokens,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "tensor_parallel_size": profile.tensor_parallel_size,
        "launch_args": args,
        "_target": {
            "profile_id": profile.id,
            "engine": profile.engine.value,
            "artifact": artifact,
            "gpu_count": profile.gpu_count,
            "tensor_parallel_size": profile.tensor_parallel_size,
            "eligible_gpu_sets": profile.eligible_gpu_sets,
        },
    }


async def queue_validation(
    database: Database,
    job_id: str,
    *,
    engines: tuple[str, ...],
    profile_id: str | None,
    overrides: dict[str, Any] | None,
) -> dict[str, object]:
    async with database.transaction() as connection:
        job = await (
            await connection.execute(
                "SELECT j.*, m.engine FROM model_jobs j JOIN model_catalog m ON m.id=j.model_id "
                "WHERE j.id=?",
                (job_id,),
            )
        ).fetchone()
        if job is None:
            raise RioError("job_not_found", "Validation job not found", status_code=404)
        if job["state"] in {"QUEUED", "RUNNING"}:
            raise RioError(
                "validation_already_running", "The model job is already running", status_code=409
            )
        settings = json.loads(job["validation_overrides_json"] or "{}")
        if profile_id:
            row = await (
                await connection.execute(
                    "SELECT profile_json FROM model_profiles WHERE id=? AND model_id=?",
                    (profile_id, job["model_id"]),
                )
            ).fetchone()
            if row is None:
                raise RioError("profile_not_found", "Placement profile not found", status_code=404)
            settings = profile_overrides(profile_from_dict(json.loads(row["profile_json"])))
        if overrides is not None:
            settings.update(overrides)
        engine = settings.get("_target", {}).get("engine", job["engine"])
        if engine not in engines:
            raise RioError(
                "unsupported_engine", "Engine is unavailable in this serving mode", status_code=422
            )
        await connection.execute(
            "UPDATE model_jobs SET state='QUEUED', stage='resolve', failure_json=NULL, "
            "validation_overrides_json=?, updated_at=? WHERE id=?",
            (json.dumps(settings), _now(), job_id),
        )
        await connection.execute(
            "UPDATE model_catalog SET state='REQUESTED', updated_at=? WHERE id=?",
            (_now(), job["model_id"]),
        )
    return {"model_id": job["model_id"], "job_id": job_id, "validation_overrides": settings}
