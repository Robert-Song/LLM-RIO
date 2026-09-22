from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from llm_rio.artifacts import local_manifest
from llm_rio.config import Settings
from llm_rio.domain import CatalogState, MachineInventory, ServiceMode
from llm_rio.profiles import ProfileRepository, profile_key, profile_to_dict
from llm_rio.storage import Database, _now
from llm_rio.validation import (
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
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._validation_lock = asyncio.Lock()

    async def resume(self) -> None:
        rows = await self.database.fetchall(
            "SELECT id FROM model_jobs WHERE state IN ('QUEUED', 'RUNNING')"
        )
        for row in rows:
            self.start(row["id"])

    def start(self, job_id: str) -> None:
        task = self._tasks.get(job_id)
        if task is None or task.done():
            self._tasks[job_id] = asyncio.create_task(
                self._run(job_id), name=f"model-registration-{job_id}"
            )

    async def close(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(
            *self._tasks.values(),
            return_exceptions=True,
        )

    async def _wait_for_validation_window(self, job_id: str) -> None:
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
                await self.database.update_model_job(
                    job_id,
                    job_state="QUEUED",
                    stage="waiting_for_maintenance",
                    catalog_state=CatalogState.VALIDATION_PENDING,
                    progress=progress,
                )
                reported = True
            await asyncio.sleep(1.0)

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
            if job.get("local_path"):
                artifact_path = str(Path(job["local_path"]).resolve(strict=True))
                resolved = await asyncio.to_thread(local_manifest, Path(artifact_path))
            else:
                resolved = await asyncio.to_thread(self._resolve, job)
                self._check_disk(resolved["download_bytes"])
                artifact_path = await asyncio.to_thread(
                    snapshot_download,
                    repo_id=job["huggingface_repo"],
                    revision=resolved["revision"],
                    cache_dir=self.settings.model_store / "huggingface",
                    token=self.settings.hf_token,
                )
                if job.get("engine") == "llama.cpp":
                    ggufs = sorted(Path(artifact_path).glob("*.gguf"))
                    if len(ggufs) != 1:
                        raise ValidationError(
                            "inspection",
                            (
                                "GGUF repository must contain exactly one GGUF; otherwise register"
                                " an explicit local GGUF file"
                            ),
                        )
                    artifact_path = str(ggufs[0])
            inspection: dict[str, Any]
            if job.get("engine") == "llama.cpp":
                inspection = {
                    "weight_bytes": Path(artifact_path).stat().st_size,
                    "max_model_len": self.settings.engines.max_model_len or 4096,
                    "max_model_len_is_fallback": True,
                    "dtype": "auto",
                    "quantization": None,
                    "capabilities": ["chat", "streaming"],
                }
            else:
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
        declared_max_model_len = (
            config.get("max_position_embeddings")
            or text_cfg.get("max_position_embeddings")
            or config.get("model_max_length")
            or text_cfg.get("model_max_length")
            or tok_max_len
        )
        max_model_len = max(1, int(declared_max_model_len or 4096))
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
            "max_model_len_is_fallback": declared_max_model_len is None,
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
        requested_max_model_len = overrides.get(
            "max_model_len", self.settings.engines.max_model_len
        )
        source_max_model_len = inspection["max_model_len"]
        max_model_len = source_max_model_len
        if isinstance(requested_max_model_len, int):
            max_model_len = (
                requested_max_model_len
                if source_max_model_len is None or inspection.get("max_model_len_is_fallback")
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
        requested_tp = overrides.get("tensor_parallel_size")
        if requested_tp is not None:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.tensor_parallel_size == requested_tp
            ]
        launch_args = dict(overrides.get("launch_args") or {})
        dtype = launch_args.pop("dtype", inspection["dtype"])
        quantization = launch_args.pop("quantization", inspection["quantization"])
        candidates = [
            replace(candidate, dtype=dtype, quantization=quantization, launch_args=launch_args)
            for candidate in candidates
        ]
        if not candidates:
            message = "model cannot fit any homogeneous GPU set"
            if requested_tp is not None:
                message = f"No eligible GPU placement for requested TP={requested_tp}"
            raise ValidationError("candidate_shapes", message)
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
                        if job.get("engine") == "llama.cpp":
                            from llm_rio.llama_validation import validate_llama_cpp

                            candidate_profiles = await validate_llama_cpp(
                                settings=self.settings,
                                inventory=self.inventory,
                                scheduler=self.validator.scheduler,
                                probes=self.validator,
                                model_id=job["model_id"],
                                model_revision=resolved_revision,
                                gguf_path=artifact_path,
                                nickname=job["nickname"],
                                candidate=candidate,
                            )
                        else:
                            candidate_profiles = await self.validator.validate_vllm(
                                model_id=job["model_id"],
                                model_revision=resolved_revision,
                                model_path=artifact_path,
                                nickname=job["nickname"],
                                candidate=candidate,
                                backend="kvcached"
                                if self.settings.effective_kvcached_mode == "required"
                                else "native",
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
