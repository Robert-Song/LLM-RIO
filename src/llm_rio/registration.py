from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from llm_rio.config import Settings
from llm_rio.domain import CatalogState, Engine, MachineInventory, ServiceMode
from llm_rio.profiles import ProfileRepository, profile_key, profile_to_dict
from llm_rio.storage import Database, _now
from llm_rio.validation import (
    CandidateShape,
    ProfileValidator,
    ValidationError,
    ValidationPreempted,
    build_candidate_shapes,
)


class RegistrationManager:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        inventory: MachineInventory,
        profile_repository: ProfileRepository,
        validator: ProfileValidator,
    ) -> None:
        self.settings = settings
        self.database = database
        self.inventory = inventory
        self.profile_repository = profile_repository
        self.validator = validator
        self._verification_tasks: dict[str, asyncio.Task[None]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._validation_lock = asyncio.Lock()

    async def resume(self) -> None:
        rows = await self.database.fetchall(
            "SELECT id FROM model_jobs WHERE state IN ('QUEUED', 'RUNNING')"
        )
        for row in rows:
            self.start(row["id"])
        verification_rows = await self.database.fetchall(
            """
            SELECT id FROM model_verification_jobs
             WHERE state IN ('QUEUED', 'RUNNING')
            """
        )
        for row in verification_rows:
            self.start_kvcached_verification(row["id"])

    def start(self, job_id: str) -> None:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            self._tasks[job_id] = asyncio.create_task(
                self._run(job_id), name=f"model-registration-{job_id}"
            )

    def start_kvcached_verification(self, job_id: str) -> None:
        task = self._verification_tasks.get(job_id)
        if task is None or task.done():
            self._verification_tasks[job_id] = asyncio.create_task(
                self._run_kvcached_verification(job_id),
                name=f"model-kvcached-verification-{job_id}",
            )

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        for task in self._verification_tasks.values():
            task.cancel()
        await asyncio.gather(
            *self._tasks.values(),
            *self._verification_tasks.values(),
            return_exceptions=True,
        )

    async def create_kvcached_verification_job(self, model_id: str) -> dict[str, Any]:
        pending = await self.database.fetchone(
            """
            SELECT id FROM model_verification_jobs
             WHERE model_id = ? AND backend = 'kvcached'
               AND state IN ('QUEUED', 'RUNNING')
             ORDER BY created_at DESC LIMIT 1
            """,
            (model_id,),
        )
        if pending is not None:
            job = await self.kvcached_verification_job(str(pending["id"]))
            if job is not None:
                return job
        job_id = str(uuid.uuid4())
        now = _now()
        await self.database.execute(
            """
            INSERT INTO model_verification_jobs
                (id, model_id, backend, state, stage, created_at, updated_at)
            VALUES (?, ?, 'kvcached', 'QUEUED', 'queued', ?, ?)
            """,
            (job_id, model_id, now, now),
        )
        self.start_kvcached_verification(job_id)
        job = await self.kvcached_verification_job(job_id)
        if job is None:
            raise RuntimeError("verification job disappeared after creation")
        return job

    async def kvcached_verification_job(self, job_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone(
            """
            SELECT j.*, m.nickname
              FROM model_verification_jobs j
              JOIN model_catalog m ON m.id = j.model_id
             WHERE j.id = ?
            """,
            (job_id,),
        )
        if row is None:
            return None
        result = dict(row)
        result["progress"] = json.loads(result.pop("progress_json"))
        failure = result.pop("failure_json")
        result["failure"] = json.loads(failure) if failure else None
        return result

    async def latest_kvcached_verification_job(self, model_id: str) -> dict[str, Any] | None:
        row = await self.database.fetchone(
            """
            SELECT id FROM model_verification_jobs
             WHERE model_id = ? AND backend = 'kvcached'
             ORDER BY created_at DESC LIMIT 1
            """,
            (model_id,),
        )
        return None if row is None else await self.kvcached_verification_job(str(row["id"]))

    async def _update_kvcached_verification_job(
        self,
        job_id: str,
        *,
        state: str,
        stage: str,
        progress: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> None:
        await self.database.execute(
            """
            UPDATE model_verification_jobs
               SET state = ?, stage = ?, progress_json = ?, failure_json = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                state,
                stage,
                json.dumps(progress or {}),
                json.dumps(failure) if failure is not None else None,
                _now(),
                job_id,
            ),
        )

    @staticmethod
    def _candidate_from_profile(profile: Any) -> CandidateShape:
        return CandidateShape(
            gpu_count=profile.gpu_count,
            tensor_parallel_size=profile.tensor_parallel_size,
            launch_args=dict(profile.launch_args),
            max_model_len=profile.max_model_len,
            max_num_seqs=profile.max_num_seqs,
            max_num_batched_tokens=profile.max_num_batched_tokens,
            gpu_memory_utilization=profile.gpu_memory_utilization,
            dtype=profile.dtype,
            quantization=profile.quantization,
            eligible_gpu_sets=profile.eligible_gpu_sets,
        )

    async def _wait_for_validation_window(self, job_id: str, *, verification: bool = False) -> None:
        """Downloads may run during serving; native-mode GPU probes require maintenance."""
        if not self.validator.scheduler.validation_requires_maintenance:
            return
        reported = False
        while await self.database.service_mode() is not ServiceMode.MAINTENANCE_READY:
            if not reported:
                progress = {
                    "message": (
                        "Waiting for maintenance to finish draining; use llmctl maintenance drain"
                    )
                }
                if verification:
                    await self._update_kvcached_verification_job(
                        job_id, state="QUEUED", stage="waiting_for_maintenance", progress=progress
                    )
                else:
                    await self.database.update_model_job(
                        job_id,
                        job_state="QUEUED",
                        stage="waiting_for_maintenance",
                        catalog_state=CatalogState.VALIDATION_PENDING,
                        progress=progress,
                    )
                reported = True
            await asyncio.sleep(1.0)

    async def _run_kvcached_verification(self, job_id: str) -> None:
        try:
            job = await self.kvcached_verification_job(job_id)
            if job is None:
                return
            model = await self.database.model_by_id(str(job["model_id"]))
            if model is None or not model.get("artifact_path"):
                raise ValidationError(
                    "verification_preflight",
                    "model artifact is unavailable on this machine",
                )
            records = [
                record
                for record in await self.profile_repository.records_for_model(str(job["model_id"]))
                if record.active and record.profile.engine is Engine.VLLM
            ]
            if not records:
                raise ValidationError(
                    "verification_preflight",
                    "no active vLLM placement profile is available to verify",
                )

            accepted: list[Any] = []
            last_error: ValidationError | None = None
            for index, record in enumerate(records, 1):
                while True:
                    await self._wait_for_validation_window(job_id, verification=True)
                    await self._update_kvcached_verification_job(
                        job_id,
                        state="RUNNING",
                        stage="validating",
                        progress={
                            "profile": index,
                            "profiles_total": len(records),
                            "tensor_parallel_size": record.profile.tensor_parallel_size,
                        },
                    )
                    try:
                        async with self._validation_lock:
                            verified = await self.validator.validate_vllm(
                                model_id=str(model["id"]),
                                model_revision=str(model.get("resolved_revision") or ""),
                                model_path=Path(str(model["artifact_path"])),
                                nickname=str(model["nickname"]),
                                candidate=self._candidate_from_profile(record.profile),
                                backend="kvcached",
                            )
                        accepted.extend(verified)
                        break
                    except ValidationPreempted:
                        await self._update_kvcached_verification_job(
                            job_id,
                            state="QUEUED",
                            stage="validation_requeued",
                            progress={"profile": index, "profiles_total": len(records)},
                        )
                        await asyncio.sleep(5.0)
                    except ValidationError as exc:
                        last_error = exc
                        break
            if not accepted:
                if last_error is not None:
                    raise last_error
                raise ValidationError("validation", "no kvcached placement passed validation")

            async with self.database.transaction() as connection:
                for profile in accepted:
                    raw = profile_to_dict(profile)
                    await connection.execute(
                        """
                        INSERT INTO model_profiles
                            (id, model_id, machine_fingerprint, profile_key, profile_json,
                             verified_at, active)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(profile_key) DO UPDATE SET
                            profile_json = json_set(
                                excluded.profile_json, '$.id', model_profiles.id
                            ),
                            verified_at = excluded.verified_at,
                            active = 1
                        """,
                        (
                            profile.id,
                            profile.model_id,
                            profile.machine_fingerprint,
                            profile_key(raw),
                            json.dumps(raw),
                            _now(),
                        ),
                    )
            await self._update_kvcached_verification_job(
                job_id,
                state="COMPLETED",
                stage="complete",
                progress={"profiles_verified": len(accepted)},
            )
            await self.database.record_event(
                "MODEL_KVCACHED_VERIFIED",
                str(model["id"]),
                {"job_id": job_id, "profiles_verified": len(accepted)},
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            stage = exc.stage if isinstance(exc, ValidationError) else "unexpected"
            details = exc.details if isinstance(exc, ValidationError) else {}
            failure = {
                "stage": stage,
                "message": str(exc),
                "details": details,
                "traceback": "".join(traceback.format_exception(exc))[-8000:],
            }
            await self._update_kvcached_verification_job(
                job_id,
                state="FAILED",
                stage=stage,
                failure=failure,
            )
            await self.database.record_event(
                "MODEL_KVCACHED_VERIFICATION_FAILED",
                payload={"job_id": job_id, "failure": failure},
            )

    async def _run(self, job_id: str) -> None:
        try:
            job = await self.database.get_model_job(job_id)
            if job is None:
                return
            # A retry revalidates the cataloged artifact rather than following a
            # moving ref such as ``main``. New jobs have no resolved revision.
            resolved_revision = job.get("resolved_revision")
            if resolved_revision:
                job = {**job, "requested_revision": resolved_revision}
            await self.database.update_model_job(
                job_id,
                job_state="RUNNING",
                stage="resolve",
                catalog_state=CatalogState.DOWNLOADING,
            )
            resolved = await asyncio.to_thread(self._resolve, job)
            self._check_disk(resolved["download_bytes"])
            await self.database.update_model_job(
                job_id,
                job_state="RUNNING",
                stage="download",
                catalog_state=CatalogState.DOWNLOADING,
                resolved_revision=resolved["revision"],
                progress={"download_bytes": resolved["download_bytes"]},
            )
            artifact_path = await asyncio.to_thread(
                snapshot_download,
                repo_id=job["huggingface_repo"],
                revision=resolved["revision"],
                cache_dir=self.settings.model_store / "huggingface",
                token=self.settings.hf_token,
            )
            inspection = await asyncio.to_thread(self._inspect, Path(artifact_path), resolved)
            await self.database.update_model_job(
                job_id,
                job_state="RUNNING",
                stage="validation_pending",
                catalog_state=CatalogState.VALIDATION_PENDING,
                artifact_path=str(artifact_path),
                capabilities=inspection["capabilities"],
                progress={"inspection": inspection},
            )
            profiles = await self._validate_with_requeue(
                job_id=job_id,
                job=job,
                artifact_path=Path(artifact_path),
                resolved_revision=resolved["revision"],
                inspection=inspection,
            )
            if not profiles:
                raise ValidationError("validation", "no candidate placement passed validation")
            async with self.database.transaction() as connection:
                await connection.execute(
                    """
                    UPDATE model_profiles
                       SET active = 0
                     WHERE model_id = ? AND machine_fingerprint = ?
                    """,
                    (job["model_id"], self.inventory.fingerprint),
                )
                for profile in profiles:
                    raw = profile_to_dict(profile)
                    await connection.execute(
                        """
                        INSERT INTO model_profiles
                            (id, model_id, machine_fingerprint, profile_key, profile_json,
                             verified_at, active)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(profile_key) DO UPDATE SET
                            profile_json = json_set(
                                excluded.profile_json, '$.id', model_profiles.id
                            ),
                            verified_at = excluded.verified_at,
                            active = 1
                        """,
                        (
                            profile.id,
                            profile.model_id,
                            profile.machine_fingerprint,
                            profile_key(raw),
                            json.dumps(raw),
                            _now(),
                        ),
                    )
                await connection.execute(
                    """
                    UPDATE model_catalog
                       SET state = ?, resolved_revision = ?, artifact_path = ?,
                           artifact_hashes_json = ?, capabilities_json = ?,
                           request_limits_json = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (
                        CatalogState.AVAILABLE.value,
                        resolved["revision"],
                        str(artifact_path),
                        json.dumps(resolved["artifact_hashes"]),
                        json.dumps(
                            sorted(
                                set.intersection(
                                    *(set(profile.capabilities) for profile in profiles)
                                )
                            )
                        ),
                        json.dumps(
                            {
                                "max_context_tokens": min(
                                    profile.max_model_len for profile in profiles
                                ),
                            }
                        ),
                        _now(),
                        job["model_id"],
                    ),
                )
                for key_id in job["requested_grants"]:
                    await connection.execute(
                        """
                        INSERT OR IGNORE INTO model_grants(key_id, model_id, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (key_id, job["model_id"], _now()),
                    )
                await connection.execute(
                    """
                    UPDATE model_jobs SET state = 'COMPLETED', stage = 'complete',
                           progress_json = ?, updated_at = ? WHERE id = ?
                    """,
                    (json.dumps({"profiles": len(profiles)}), _now(), job_id),
                )
            await self.validator.scheduler.warm_model_once(job["model_id"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._record_failure(job_id, exc)

    def _resolve(self, job: dict[str, Any]) -> dict[str, Any]:
        info = HfApi(token=self.settings.hf_token).model_info(
            repo_id=job["huggingface_repo"],
            revision=job.get("requested_revision"),
            files_metadata=True,
        )
        artifacts = []
        total = 0
        for sibling in info.siblings or []:
            size = int(getattr(sibling, "size", 0) or 0)
            total += size
            blob_id = getattr(sibling, "blob_id", None)
            lfs = getattr(sibling, "lfs", None)
            digest = getattr(lfs, "sha256", None) if lfs else blob_id
            artifacts.append({"path": sibling.rfilename, "bytes": size, "digest": digest})
        return {
            "revision": info.sha,
            "download_bytes": total,
            "artifact_hashes": artifacts,
        }

    def _check_disk(self, download_bytes: int) -> None:
        free = shutil.disk_usage(self.settings.model_store).free
        required = int(download_bytes * 1.1)
        if free < required:
            raise ValidationError(
                "disk_capacity",
                "insufficient free space for pinned model snapshot",
                {"free_bytes": free, "required_bytes": required},
            )

    @staticmethod
    def _inspect(path: Path, resolved: dict[str, Any]) -> dict[str, Any]:
        config_path = path / "config.json"
        if not config_path.exists():
            raise ValidationError("inspection", "config.json is missing")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer_config_path = path / "tokenizer_config.json"
        tokenizer_config = (
            json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
            if tokenizer_config_path.exists()
            else {}
        )
        weight_files = [
            item
            for item in resolved["artifact_hashes"]
            if item["path"].endswith((".safetensors", ".bin", ".gguf"))
        ]
        if not weight_files:
            raise ValidationError("inspection", "no supported weight artifact was found")
        architectures = config.get("architectures") or []
        text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        tok_max_len = tokenizer_config.get("model_max_length")
        if not isinstance(tok_max_len, int) or tok_max_len >= 10_000_000:
            tok_max_len = None
        max_model_len = int(
            config.get("max_position_embeddings")
            or text_cfg.get("max_position_embeddings")
            or config.get("model_max_length")
            or text_cfg.get("model_max_length")
            or tok_max_len
            or 4096
        )
        max_model_len = max(1, max_model_len)
        quantization = config.get("quantization_config", {}).get("quant_method")
        dtype_value = str(config.get("torch_dtype") or "auto").lower()
        dtype = {
            "float16": "half",
            "fp16": "half",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float",
            "fp32": "float",
        }.get(dtype_value, dtype_value)
        if dtype not in {"auto", "half", "bfloat16", "float"}:
            dtype = "auto"
        has_chat_template = bool(tokenizer_config.get("chat_template"))
        capabilities = ["chat", "streaming"] if has_chat_template else ["completions"]
        return {
            "architectures": architectures,
            "weight_bytes": sum(int(item["bytes"]) for item in weight_files),
            "weight_files": [item["path"] for item in weight_files],
            "max_model_len": max_model_len,
            "dtype": dtype,
            "quantization": quantization,
            "capabilities": capabilities,
            "chat_template_hash": hashlib.sha256(
                str(tokenizer_config.get("chat_template", "")).encode()
            ).hexdigest(),
            "multimodal": any(key in config for key in ("vision_config", "audio_config")),
        }

    async def _validate_with_requeue(
        self,
        *,
        job_id: str,
        job: dict[str, Any],
        artifact_path: Path,
        resolved_revision: str,
        inspection: dict[str, Any],
    ) -> list[Any]:
        raw_overrides = job.get("validation_overrides")
        overrides = raw_overrides if isinstance(raw_overrides, dict) else {}
        requested_max_model_len = overrides.get("max_model_len")
        source_max_model_len = inspection["max_model_len"]
        max_model_len = source_max_model_len
        if isinstance(requested_max_model_len, int):
            max_model_len = (
                requested_max_model_len
                if source_max_model_len is None
                else min(source_max_model_len, requested_max_model_len)
            )
        candidates = build_candidate_shapes(
            inventory=self.inventory,
            weight_bytes=inspection["weight_bytes"],
            max_model_len=max_model_len,
            reserved_vram_mib=self.settings.reserved_vram_mib,
            dtype=inspection["dtype"],
            quantization=inspection["quantization"],
            gpu_memory_utilization=overrides.get(
                "gpu_memory_utilization", self.settings.engines.gpu_memory_utilization
            ),
            max_model_len_limit=overrides.get("max_model_len", self.settings.engines.max_model_len),
            max_num_seqs=overrides.get("max_num_seqs", self.settings.engines.max_num_seqs),
            max_num_batched_tokens=overrides.get(
                "max_num_batched_tokens", self.settings.engines.max_num_batched_tokens
            ),
        )
        if not candidates:
            raise ValidationError("candidate_shapes", "model cannot fit any homogeneous GPU set")
        accepted = []
        last_validation_error: ValidationError | None = None
        for candidate in candidates:
            while True:
                await self._wait_for_validation_window(job_id)
                await self.database.update_model_job(
                    job_id,
                    job_state="RUNNING",
                    stage="validating",
                    catalog_state=CatalogState.VALIDATING,
                    progress={
                        "gpu_count": candidate.gpu_count,
                        "validation_overrides": overrides,
                    },
                )
                try:
                    async with self._validation_lock:
                        candidate_profiles = await self.validator.validate_vllm(
                            model_id=job["model_id"],
                            model_revision=resolved_revision,
                            model_path=artifact_path,
                            nickname=job["nickname"],
                            candidate=candidate,
                        )
                    accepted.extend(candidate_profiles)
                    break
                except ValidationPreempted as exc:
                    await self.database.update_model_job(
                        job_id,
                        job_state="QUEUED",
                        stage="gpu_memory_wait"
                        if exc.stage == "gpu_memory_wait"
                        else "validation_requeued",
                        catalog_state=CatalogState.VALIDATION_PENDING,
                        progress={"message": str(exc), **exc.details},
                    )
                    await asyncio.sleep(5.0)
                    continue
                except ValidationError as exc:
                    last_validation_error = exc
                    break
            # Automatic registration only needs one-GPU placements. Larger tensor-parallel
            # profiles remain an explicit administrator choice in the TUI.
            if accepted and candidate.tensor_parallel_size == 1:
                break
        if not accepted and last_validation_error is not None:
            raise last_validation_error
        return accepted

    async def _record_failure(self, job_id: str, exc: Exception) -> None:
        stage = exc.stage if isinstance(exc, ValidationError) else "unexpected"
        details = exc.details if isinstance(exc, ValidationError) else {}
        failure = {
            "stage": stage,
            "message": str(exc),
            "details": details,
            "environment": {
                "machine_fingerprint": self.inventory.fingerprint,
                "driver": self.inventory.driver_version,
                "cuda": self.inventory.cuda_driver_version,
                "gpu_models": [device.name for device in self.inventory.gpus],
            },
            "traceback": "".join(traceback.format_exception(exc))[-8000:],
        }
        await self.database.update_model_job(
            job_id,
            job_state="FAILED",
            stage=stage,
            catalog_state=CatalogState.NEEDS_ADMIN_REVIEW,
            failure=failure,
        )
