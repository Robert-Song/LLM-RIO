from __future__ import annotations

import json
import uuid
from pathlib import Path

from llm_rio.artifacts import local_artifact_unchanged
from llm_rio.errors import RioError
from llm_rio.modes.contracts import ModePolicy
from llm_rio.profiles import ProfileRepository, profile_from_dict, profile_key
from llm_rio.storage import _now


async def trust_measurements(
    repository: ProfileRepository,
    *,
    model_id: str,
    profile_id: str,
    mode: ModePolicy,
    gpu_uuids: set[str],
    reason: str,
    actor: str,
) -> dict[str, object]:
    if not reason.strip():
        raise RioError(
            "reason_required", "Explain why these saved measurements are trusted", status_code=422
        )
    database = repository.database
    async with database.transaction() as connection:
        row = await (
            await connection.execute(
                "SELECT * FROM model_profiles WHERE id=? AND model_id=?", (profile_id, model_id)
            )
        ).fetchone()
        model = await (
            await connection.execute("SELECT * FROM model_catalog WHERE id=?", (model_id,))
        ).fetchone()
        if row is None or model is None:
            raise RioError("profile_not_found", "Profile not found", status_code=404)
        try:
            raw = json.loads(row["profile_json"])
            profile = profile_from_dict(raw)
        except (ValueError, KeyError, TypeError) as exc:
            raise RioError(
                "measurements_incompatible",
                "Saved measurements are incomplete; run Validate/Revalidate",
                status_code=409,
            ) from exc
        artifact = Path(model["artifact_path"] or "")
        eligible = (
            bool(model["artifact_path"])
            and artifact.exists()
            and not raw.get("measurements_invalidated_at")
            and profile.model_revision == model["resolved_revision"]
            and mode.eligibility(profile).allowed
            and all(
                len(group) == len(set(group)) == profile.gpu_count and set(group) <= gpu_uuids
                for group in profile.eligible_gpu_sets
            )
            and bool(profile.eligible_gpu_sets)
        )
        if model["source_type"] == "local":
            eligible = eligible and local_artifact_unchanged(
                artifact, json.loads(model["artifact_hashes_json"]), verify_content=True
            )
        if not eligible:
            raise RioError(
                "measurements_incompatible",
                "Run real validation; these measurements cannot be trusted",
                status_code=409,
            )
        source_id = raw["id"]
        raw["id"] = str(uuid.uuid4())
        raw["machine_fingerprint"] = repository.machine_fingerprint
        audit = {
            "source_profile_id": source_id,
            "source_fingerprint": row["machine_fingerprint"],
            "actor": actor,
            "reason": reason.strip(),
            "at": _now(),
            "serving_mode": mode.capabilities.name,
        }
        raw["verification_override"] = audit
        key = profile_key(raw)
        existing = await (
            await connection.execute("SELECT id FROM model_profiles WHERE profile_key=?", (key,))
        ).fetchone()
        if existing:
            raw["id"] = existing["id"]
        await connection.execute(
            """INSERT INTO model_profiles
          (id,model_id,machine_fingerprint,profile_key,profile_json,verified_at,active)
          VALUES (?,?,?,?,?,?,1)
          ON CONFLICT(profile_key) DO UPDATE SET profile_json=excluded.profile_json, active=1""",
            (raw["id"], model_id, repository.machine_fingerprint, key, json.dumps(raw), _now()),
        )
        result = {"profile_id": raw["id"], "model_id": model_id, **audit}
        await connection.execute(
            (
                "INSERT INTO "
                "runtime_events(event_type,entity_id,payload_json,created_at) "
                "VALUES ('PROFILE_MEASUREMENTS_TRUSTED',?,?,?)"
            ),
            (raw["id"], json.dumps(result), _now()),
        )
    return result
