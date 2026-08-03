from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import traceback
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from llm_rio.config import Settings
from llm_rio.domain import CatalogState, MachineInventory
from llm_rio.llama_validation import validate_llama_cpp
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
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)

    async def _run(self, job_id: str) -> None:
        try:
            job = await self.database.get_model_job(job_id)
            if job is None:
                return
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
                for profile in profiles:
                    raw = profile_to_dict(profile)
                    await connection.execute(
                        """
                        INSERT INTO model_profiles
                            (id, model_id, machine_fingerprint, profile_key, profile_json,
                             verified_at, active)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                        ON CONFLICT(profile_key) DO UPDATE SET
                            profile_json = excluded.profile_json,
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
                           artifact_hashes_json = ?, capabilities_json = ?, request_limits_json = ?, updated_at = ?
                     WHERE id = ?
                    """,
                    (
                        CatalogState.AVAILABLE.value,
                        resolved["revision"],
                        str(artifact_path),
                        json.dumps(resolved["artifact_hashes"]),
                        json.dumps(sorted(set.intersection(*(
                            set(profile.capabilities) for profile in profiles
                        )))),
                        json.dumps({
                            "max_context_tokens": min(
                                profile.max_model_len for profile in profiles
                            ),
                            "max_output_tokens": self.settings.max_output_tokens,
                            "max_n": self.settings.max_n,
                        }),
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
        max_model_len = int(
            config.get("max_position_embeddings")
            or config.get("model_max_length")
            or tokenizer_config.get("model_max_length")
            or 4096
        )
        max_model_len = min(max_model_len, 1_000_000)
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
        candidates = build_candidate_shapes(
            inventory=self.inventory,
            weight_bytes=inspection["weight_bytes"],
            max_model_len=inspection["max_model_len"],
            reserved_vram_mib=self.settings.reserved_vram_mib,
            dtype=inspection["dtype"],
            quantization=inspection["quantization"],
        )
        if not candidates:
            raise ValidationError("candidate_shapes", "model cannot fit any homogeneous GPU set")
        accepted = []
        last_validation_error: ValidationError | None = None
        for candidate in candidates:
            while True:
                await self.database.update_model_job(
                    job_id,
                    job_state="RUNNING",
                    stage="validating",
                    catalog_state=CatalogState.VALIDATING,
                    progress={"gpu_count": candidate.gpu_count},
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
                except ValidationPreempted:
                    await self.database.update_model_job(
                        job_id,
                        job_state="QUEUED",
                        stage="validation_requeued",
                        catalog_state=CatalogState.VALIDATION_PENDING,
                    )
                    await asyncio.sleep(5.0)
                    continue
                except ValidationError as exc:
                    last_validation_error = exc
                    break
            # Smallest viable placement is mandatory. Larger shapes are useful only when measured.
            if accepted and candidate.gpu_count >= min(item.gpu_count for item in accepted) + 1:
                break
        gguf_files = [
            artifact_path / name
            for name in inspection["weight_files"]
            if name.lower().endswith(".gguf")
        ]
        while (
            not accepted
            and self.settings.engines.enable_llama_cpp
            and gguf_files
        ):
            try:
                async with self._validation_lock:
                    accepted.extend(await validate_llama_cpp(
                        settings=self.settings,
                        inventory=self.inventory,
                        scheduler=self.validator.scheduler,
                        probes=self.validator,
                        model_id=job["model_id"],
                        model_revision=resolved_revision,
                        gguf_path=gguf_files[0],
                        nickname=job["nickname"],
                        candidate=candidates[0],
                    ))
            except ValidationPreempted:
                await self.database.update_model_job(
                    job_id,
                    job_state="QUEUED",
                    stage="validation_requeued",
                    catalog_state=CatalogState.VALIDATION_PENDING,
                )
                await asyncio.sleep(5.0)
                continue
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

