from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from llm_rio.domain import CatalogState
from llm_rio.errors import RioError
from llm_rio.timestamps import now as _now

if TYPE_CHECKING:
    from llm_rio.storage import Database

logger = logging.getLogger(__name__)


class CatalogRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def create_model_job(
        self,
        *,
        nickname: str,
        repo: str,
        revision: str | None,
        creator_key_id: str,
        grant_key_ids: list[str],
        local_path: str | None = None,
        engine: str = "vllm",
    ) -> tuple[str, str]:
        model_id, job_id, now = (str(uuid.uuid4()), str(uuid.uuid4()), _now())
        async with self.database.transaction() as connection:
            if grant_key_ids:
                placeholders = ",".join("?" for _ in grant_key_ids)
                rows = await (
                    await connection.execute(
                        f"SELECT id FROM api_keys WHERE active = 1 AND id IN ({placeholders})",
                        grant_key_ids,
                    )
                ).fetchall()
                existing = {row["id"] for row in rows}
                missing = sorted(set(grant_key_ids) - existing)
                if missing:
                    raise RioError(
                        "grant_key_not_found",
                        "One or more requested grant keys do not exist or are inactive",
                        status_code=404,
                        details={"missing_key_ids": missing},
                    )
            await connection.execute(
                (
                    "\n"
                    "                INSERT INTO model_catalog\n"
                    "                    (id, nickname, huggingface_repo, requested_re"
                    "vision, state,\n"
                    "                     created_by_key_id, created_at, updated_at, s"
                    "ource_type, local_path, engine)\n"
                    "                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n"
                    "                "
                ),
                (
                    model_id,
                    nickname,
                    repo,
                    revision,
                    CatalogState.REQUESTED.value,
                    creator_key_id,
                    now,
                    now,
                    "local" if local_path else "huggingface",
                    local_path,
                    engine,
                ),
            )
            await connection.execute(
                (
                    "\n"
                    "                INSERT INTO model_jobs\n"
                    "                    (id, model_id, state, stage, requested_grants"
                    "_json, created_at, updated_at)\n"
                    "                VALUES (?, ?, ?, ?, ?, ?, ?)\n"
                    "                "
                ),
                (job_id, model_id, "QUEUED", "resolve", json.dumps(grant_key_ids), now, now),
            )
        return (model_id, job_id)

    async def update_model_job(
        self,
        job_id: str,
        *,
        job_state: str,
        stage: str,
        catalog_state: CatalogState,
        progress: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
        resolved_revision: str | None = None,
        artifact_path: str | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        async with self.database.transaction() as connection:
            job = await (
                await connection.execute("SELECT model_id FROM model_jobs WHERE id = ?", (job_id,))
            ).fetchone()
            if job is None:
                raise KeyError(job_id)
            await connection.execute(
                (
                    "\n"
                    "                UPDATE model_jobs SET state = ?, stage = ?, progr"
                    "ess_json = ?, failure_json = ?,\n"
                    "                                      updated_at = ? WHERE id = ?"
                    "\n"
                    "                "
                ),
                (
                    job_state,
                    stage,
                    json.dumps(progress or {}),
                    json.dumps(failure) if failure else None,
                    _now(),
                    job_id,
                ),
            )
            updates = [
                "state = CASE WHEN state = 'DISABLED' THEN state ELSE ? END",
                "updated_at = ?",
            ]
            values: list[Any] = [catalog_state.value, _now()]
            for column, value in (
                ("resolved_revision", resolved_revision),
                ("artifact_path", artifact_path),
                (
                    "capabilities_json",
                    json.dumps(capabilities) if capabilities is not None else None,
                ),
            ):
                if value is not None:
                    updates.append(f"{column} = ?")
                    values.append(value)
            values.append(job["model_id"])
            await connection.execute(
                f"UPDATE model_catalog SET {', '.join(updates)} WHERE id = ?", values
            )

    async def get_model_job(self, job_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone(
            (
                "\n"
                "            SELECT j.*, m.nickname, m.huggingface_repo, m.request"
                "ed_revision, m.resolved_revision,\n"
                "                   m.state AS catalog_state, m.source_type, m.loc"
                "al_path, m.engine\n"
                "              FROM model_jobs j JOIN model_catalog m ON m.id = j."
                "model_id WHERE j.id = ?\n"
                "            "
            ),
            (job_id,),
        )
        if row is None:
            return None
        result = dict(row)
        for key in (
            "progress_json",
            "failure_json",
            "requested_grants_json",
            "validation_overrides_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or "null")
        return result

    async def set_model_job_validation_overrides(
        self, job_id: str, overrides: dict[str, Any]
    ) -> bool:
        cursor = await self.database.execute(
            (
                "\n"
                "            UPDATE model_jobs\n"
                "               SET validation_overrides_json = ?, updated_at = ?\n"
                "             WHERE id = ?\n"
                "            "
            ),
            (json.dumps(overrides), _now(), job_id),
        )
        return cursor.rowcount > 0

    async def model_by_nickname(self, nickname: str) -> dict[str, Any] | None:
        row = await self.database.fetchone(
            "SELECT * FROM model_catalog WHERE nickname = ?", (nickname,)
        )
        if row is None:
            return None
        return self.database._decode_model(row)

    async def model_by_id(self, model_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone("SELECT * FROM model_catalog WHERE id = ?", (model_id,))
        return self.database._decode_model(row) if row else None

    async def update_model_request_defaults(
        self, model_id: str, updates: dict[str, Any | None]
    ) -> dict[str, Any] | None:
        """Apply request-default changes, removing values explicitly set to null."""
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT request_defaults_json FROM model_catalog WHERE id = ?", (model_id,)
                )
            ).fetchone()
            if row is None:
                return None
            defaults = json.loads(row["request_defaults_json"] or "{}")
            for key, value in updates.items():
                if value is None:
                    defaults.pop(key, None)
                else:
                    defaults[key] = value
            await connection.execute(
                (
                    "\n"
                    "                UPDATE model_catalog\n"
                    "                   SET request_defaults_json = ?, updated_at = ?\n"
                    "                 WHERE id = ?\n"
                    "                "
                ),
                (json.dumps(defaults), _now(), model_id),
            )
        return await self.database.model_by_id(model_id)

    async def list_models(
        self, key_id: str | None = None, *, include_registration_jobs: bool = False
    ) -> list[dict[str, Any]]:
        if key_id is None:
            rows = await self.database.fetchall("SELECT * FROM model_catalog ORDER BY nickname")
        else:
            rows = await self.database.fetchall(
                (
                    "\n"
                    "                SELECT m.* FROM model_catalog m\n"
                    "                  JOIN model_grants g ON g.model_id = m.id\n"
                    "                 WHERE g.key_id = ? AND m.state = ? ORDER BY m.ni"
                    "ckname\n"
                    "                "
                ),
                (key_id, CatalogState.AVAILABLE.value),
            )
        result = [self.database._decode_model(row) for row in rows]
        if not include_registration_jobs or key_id is not None or (not result):
            return result
        job_rows = await self.database.fetchall(
            "\n"
            "            SELECT j.id, j.model_id, j.state, j.stage, j.failure_"
            "json, j.created_at, j.updated_at\n"
            "              FROM model_jobs j\n"
            "             WHERE j.id = (\n"
            "                 SELECT newer.id\n"
            "                   FROM model_jobs newer\n"
            "                  WHERE newer.model_id = j.model_id\n"
            "                  ORDER BY newer.created_at DESC, newer.id DESC\n"
            "                  LIMIT 1\n"
            "             )\n"
            "            "
        )
        jobs_by_model: dict[str, dict[str, Any]] = {}
        for row in job_rows:
            job = dict(row)
            job["failure"] = json.loads(job.pop("failure_json") or "null")
            jobs_by_model[str(job.pop("model_id"))] = job
        for model in result:
            model["registration_job"] = jobs_by_model.get(str(model["id"]))
        return result

    async def has_model_grant(self, key_id: str, model_id: str) -> bool:
        row = await self.database.fetchone(
            "SELECT 1 FROM model_grants WHERE key_id = ? AND model_id = ?", (key_id, model_id)
        )
        return row is not None

    async def replace_model_grants(self, key_id: str, model_ids: list[str]) -> None:
        async with self.database.transaction() as connection:
            key = await (
                await connection.execute("SELECT 1 FROM api_keys WHERE id = ?", (key_id,))
            ).fetchone()
            if key is None:
                raise KeyError(key_id)
            await connection.execute("DELETE FROM model_grants WHERE key_id = ?", (key_id,))
            await connection.executemany(
                "INSERT INTO model_grants(key_id, model_id, created_at) VALUES (?, ?, ?)",
                [(key_id, model_id, _now()) for model_id in model_ids],
            )

    async def update_model_access(
        self, *, key_id: str, model_nicknames: list[str], mode: str
    ) -> list[str]:
        requested_names = set(model_nicknames)
        async with self.database.transaction() as connection:
            key = await (
                await connection.execute(
                    "SELECT 1 FROM api_keys WHERE id = ? AND active = 1", (key_id,)
                )
            ).fetchone()
            if key is None:
                raise KeyError(key_id)
            models_by_name: dict[str, str] = {}
            if requested_names:
                placeholders = ",".join("?" for _ in requested_names)
                rows = await (
                    await connection.execute(
                        (
                            "SELECT id, nickname FROM model_catalog WHERE nickname IN ("
                            f"{placeholders}"
                            ")"
                        ),
                        sorted(requested_names),
                    )
                ).fetchall()
                models_by_name = {str(row["nickname"]): str(row["id"]) for row in rows}
            missing = sorted(requested_names - set(models_by_name))
            if missing:
                raise RioError(
                    "model_not_found",
                    "One or more model nicknames do not exist",
                    status_code=404,
                    details={"missing_models": missing},
                )
            existing_rows = await (
                await connection.execute(
                    (
                        "\n"
                        "                SELECT m.id, m.nickname FROM model_grants g\n"
                        "                  JOIN model_catalog m ON m.id = g.model_id\n"
                        "                 WHERE g.key_id = ?\n"
                        "                "
                    ),
                    (key_id,),
                )
            ).fetchall()
            existing = {str(row["nickname"]): str(row["id"]) for row in existing_rows}
            if mode == "add":
                desired = {**existing, **models_by_name}
            elif mode == "remove":
                desired = {
                    nickname: model_id
                    for nickname, model_id in existing.items()
                    if nickname not in requested_names
                }
            else:
                desired = models_by_name
            await connection.execute("DELETE FROM model_grants WHERE key_id = ?", (key_id,))
            if desired:
                now = _now()
                await connection.executemany(
                    "INSERT INTO model_grants(key_id, model_id, created_at) VALUES (?, ?, ?)",
                    [(key_id, model_id, now) for model_id in desired.values()],
                )
            return sorted(desired)

    async def disable_model(self, model_id: str) -> bool:
        cursor = await self.database.execute(
            "UPDATE model_catalog SET state = ?, updated_at = ? WHERE id = ?",
            (CatalogState.DISABLED.value, _now(), model_id),
        )
        return cursor.rowcount > 0
