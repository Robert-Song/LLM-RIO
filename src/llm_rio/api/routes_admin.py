from __future__ import annotations

import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status

from llm_rio.api.dependencies import AdminPrincipal
from llm_rio.api.schemas import (
    CreateKeyRequest,
    KeySecretResponse,
    MaintenanceRequest,
    ModelProfileCloneRequest,
    ModelRequestDefaultsUpdate,
    ProfileEditRequest,
    QuotaUpdate,
)
from llm_rio.domain import Engine, PlacementProfile, RuntimeState, ServiceMode
from llm_rio.errors import RioError
from llm_rio.inventory import candidate_gpu_sets
from llm_rio.profiles import StoredProfile, profile_to_dict
from llm_rio.security import issue_api_key, token_prefix

router = APIRouter()


async def _create_key(request: Request, body: CreateKeyRequest) -> KeySecretResponse:
    database = request.app.state.database

    if body.models:
        available_models = {str(model["nickname"]) for model in await database.list_models()}
        missing_models = sorted(set(body.models) - available_models)
        if missing_models:
            raise RioError(
                "model_not_found",
                "One or more model nicknames do not exist",
                status_code=404,
                details={"missing_models": missing_models},
            )
    key_id = str(uuid.uuid4())
    if body.quota_account_id:
        account = await database.fetchone(
            "SELECT id, nickname FROM quota_accounts WHERE id = ?", (body.quota_account_id,)
        )
        if account is None:
            raise RioError("account_not_found", "Quota account was not found", status_code=404)
        account_id = account["id"]
        account_nickname = account["nickname"]
    else:
        account_id = str(uuid.uuid4())
        account_nickname = body.quota_account_nickname or body.nickname
    if body.api_key is None:
        token, prefix = issue_api_key(key_id)
    else:
        token = body.api_key
        prefix = token_prefix(token)
    limit_tokens = body.limit_tokens if body.limit_tokens is not None else 0
    unlimited = body.limit_tokens is None
    await database.create_key(
        key_id=key_id,
        nickname=body.nickname,
        role=body.role,
        account_id=account_id,
        account_nickname=account_nickname,
        prefix=prefix,
        api_key=token,
        limit_tokens=limit_tokens,
        unlimited=unlimited,
    )
    if body.models:
        await database.update_model_access(
            key_id=key_id, model_nicknames=body.models, mode="replace"
        )
    return KeySecretResponse(id=key_id, nickname=body.nickname, api_key=token)


@router.post(
    "/admin/keys",
    response_model=KeySecretResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_key(
    request: Request, body: CreateKeyRequest, _: AdminPrincipal
) -> KeySecretResponse:
    return await _create_key(request, body)


@router.get("/admin/keys")
async def list_keys(request: Request, _: AdminPrincipal) -> dict[str, object]:
    return {"data": await request.app.state.database.list_keys()}


@router.post("/admin/keys/{key_id}/rotate", response_model=KeySecretResponse)
async def rotate_key(key_id: str, request: Request, _: AdminPrincipal) -> KeySecretResponse:
    row = await request.app.state.database.fetchone(
        "SELECT nickname FROM api_keys WHERE id = ? AND token_prefix NOT LIKE 'deleted-%'",
        (key_id,),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Key not found")
    token, prefix = issue_api_key(key_id)
    await request.app.state.database.replace_key_secret(key_id, prefix, token)
    return KeySecretResponse(id=key_id, nickname=row["nickname"], api_key=token)


@router.post("/admin/keys/{key_id}/revoke", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_key(key_id: str, request: Request, _: AdminPrincipal) -> Response:
    if not await request.app.state.database.set_key_active(key_id, False):
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/admin/keys/{key_id}/restore", status_code=status.HTTP_204_NO_CONTENT)
async def restore_key(key_id: str, request: Request, _: AdminPrincipal) -> Response:
    if not await request.app.state.database.set_key_active(key_id, True):
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/admin/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_key(key_id: str, request: Request, _: AdminPrincipal) -> Response:
    if not await request.app.state.database.delete_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/admin/keys/{key_id}/quota", status_code=status.HTTP_204_NO_CONTENT)
async def update_quota(
    key_id: str, body: QuotaUpdate, request: Request, _: AdminPrincipal
) -> Response:
    if not await request.app.state.database.update_quota(key_id, body.limit_tokens, body.unlimited):
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/admin/keys/{key_id}/usage/reset")
async def reset_usage(key_id: str, request: Request, _: AdminPrincipal) -> dict[str, object]:
    result = await request.app.state.database.reset_usage(key_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return result


def _profile_payload(record: StoredProfile) -> dict[str, object]:
    payload: dict[str, object] = profile_to_dict(record.profile)
    payload["active"] = record.active
    return payload


def _gguf_files(model: dict[str, object]) -> list[str]:
    artifact_path = model.get("artifact_path")
    if not artifact_path:
        return []
    root = Path(str(artifact_path))
    if not root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".gguf"
    )


def _resolve_gguf_file(model: dict[str, object], relative_path: str) -> Path:
    artifact_path = model.get("artifact_path")
    if not artifact_path:
        raise RioError(
            "model_artifact_missing",
            "The model artifact is not available on this machine",
            status_code=409,
        )
    root = Path(str(artifact_path)).resolve()
    candidate = (root / relative_path).resolve()
    if candidate.suffix.lower() != ".gguf" or not candidate.is_relative_to(root):
        raise RioError(
            "invalid_gguf_file",
            "GGUF files must be paths inside this model's downloaded artifact",
            status_code=422,
        )
    if not candidate.is_file():
        raise RioError(
            "gguf_file_not_found",
            f"GGUF file '{relative_path}' was not found in this model artifact",
            status_code=404,
            details={"available_gguf_files": _gguf_files(model)},
        )
    return candidate


def _resize_per_gpu_measurement(values: tuple[int, ...], gpu_count: int) -> tuple[int, ...]:
    """Keep stored per-GPU diagnostics structurally valid after an admin TP override."""
    if not values:
        return (0,) * gpu_count
    return tuple(values[min(index, len(values) - 1)] for index in range(gpu_count))


def _apply_profile_edit(
    *,
    profile: PlacementProfile,
    model: dict[str, object],
    request: ProfileEditRequest,
    managed_gpu_count: int,
    eligible_gpu_sets: tuple[tuple[str, ...], ...],
    llama_cpp_enabled: bool,
) -> PlacementProfile:
    fields = request.model_fields_set
    updated = profile
    if "tensor_parallel_size" in fields:
        target_gpu_count = request.tensor_parallel_size
        if (
            target_gpu_count is None
            or target_gpu_count > managed_gpu_count
            or not eligible_gpu_sets
        ):
            raise RioError(
                "invalid_tensor_parallel_size",
                f"Tensor parallelism must be between 1 and {managed_gpu_count} on this machine",
                status_code=422,
            )
        updated = replace(
            updated,
            gpu_count=target_gpu_count,
            tensor_parallel_size=target_gpu_count,
            eligible_gpu_sets=eligible_gpu_sets,
            idle_vram_mib_per_gpu=_resize_per_gpu_measurement(
                updated.idle_vram_mib_per_gpu, target_gpu_count
            ),
            peak_vram_mib_per_gpu=_resize_per_gpu_measurement(
                updated.peak_vram_mib_per_gpu, target_gpu_count
            ),
            gpu_headroom_mib_per_gpu=_resize_per_gpu_measurement(
                updated.gpu_headroom_mib_per_gpu, target_gpu_count
            ),
        )
    if "max_model_len" in fields:
        if request.max_model_len is None:
            raise RioError("invalid_max_model_len", "max_model_len cannot be null", status_code=422)
        updated = replace(updated, max_model_len=request.max_model_len)
    if "max_num_seqs" in fields:
        updated = replace(updated, max_num_seqs=request.max_num_seqs)
    if "max_num_batched_tokens" in fields:
        updated = replace(updated, max_num_batched_tokens=request.max_num_batched_tokens)
    if "gpu_memory_utilization" in fields:
        if request.gpu_memory_utilization is None:
            raise RioError(
                "invalid_gpu_memory_utilization",
                "gpu_memory_utilization cannot be null",
                status_code=422,
            )
        updated = replace(updated, gpu_memory_utilization=request.gpu_memory_utilization)
    if "engine" in fields:
        updated = replace(updated, engine=request.engine or updated.engine)

    launch_args = dict(updated.launch_args)
    if "gguf_file" in fields:
        if updated.engine is not Engine.LLAMA_CPP:
            raise RioError(
                "gguf_requires_llama_cpp",
                "Select llama.cpp before choosing a GGUF model file",
                status_code=422,
            )
        if request.gguf_file is None:
            raise RioError("invalid_gguf_file", "gguf_file cannot be null", status_code=422)
        launch_args["model"] = str(_resolve_gguf_file(model, request.gguf_file))
    if updated.engine is Engine.LLAMA_CPP:
        if not llama_cpp_enabled:
            raise RioError(
                "llama_cpp_disabled",
                "Set engines.enable_llama_cpp = true before selecting llama.cpp manually",
                status_code=409,
            )
        if "model" not in launch_args:
            raise RioError(
                "gguf_file_required",
                "Choose a GGUF file when switching this profile to llama.cpp",
                status_code=422,
                details={"available_gguf_files": _gguf_files(model)},
            )
        if "n_gpu_layers" in fields:
            if request.n_gpu_layers is None:
                raise RioError(
                    "invalid_n_gpu_layers", "n_gpu_layers cannot be null", status_code=422
                )
            launch_args["n_gpu_layers"] = request.n_gpu_layers
        else:
            launch_args.setdefault("n_gpu_layers", 99)
        updated = replace(
            updated,
            dtype="gguf",
            quantization=updated.quantization or "gguf",
            launch_args=launch_args,
        )
    else:
        if "gguf_file" in fields or "n_gpu_layers" in fields:
            raise RioError(
                "llama_cpp_option_requires_llama_cpp",
                "GGUF and n_gpu_layers are llama.cpp-only settings",
                status_code=422,
            )
        for key in ("model", "n_gpu_layers", "tensor_split", "split_mode"):
            launch_args.pop(key, None)
        updated = replace(
            updated,
            dtype="auto",
            quantization=None,
            launch_args=launch_args,
        )
    return updated


@router.get("/admin/models/{model_id}/profiles")
async def list_model_profiles(
    model_id: str, request: Request, _: AdminPrincipal
) -> dict[str, object]:
    model = await request.app.state.database.model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    records = await request.app.state.profiles.records_for_model(model_id)
    return {
        "data": [_profile_payload(record) for record in records],
        "available_gguf_files": _gguf_files(model),
    }


@router.patch("/admin/models/{model_id}")
async def update_model_request_defaults(
    model_id: str,
    body: ModelRequestDefaultsUpdate,
    request: Request,
    _: AdminPrincipal,
) -> dict[str, object]:
    if not body.model_fields_set:
        raise RioError(
            "model_default_update_empty",
            "Provide at least one request default to change.",
            status_code=422,
        )
    updates = {field: getattr(body, field) for field in body.model_fields_set}
    database = request.app.state.database
    model = await database.update_model_request_defaults(model_id, updates)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    await database.record_event(
        "MODEL_REQUEST_DEFAULTS_UPDATED",
        model_id,
        {"request_defaults": model["request_defaults"]},
    )
    return {"model": model}


@router.post(
    "/admin/models/{model_id}/clone",
    status_code=status.HTTP_201_CREATED,
)
async def clone_model_profile(
    model_id: str,
    body: ModelProfileCloneRequest,
    request: Request,
    principal: AdminPrincipal,
) -> dict[str, object]:
    database = request.app.state.database
    source_model = await database.model_by_id(model_id)
    if source_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    cloned_model, profiles = await request.app.state.profiles.clone_model(
        source_model=source_model,
        nickname=body.nickname,
        creator_key_id=principal.key_id,
        request_defaults=body.request_defaults,
        max_model_len=body.max_model_len,
        yarn_factor=body.yarn_factor,
        yarn_original_max_model_len=body.yarn_original_max_model_len,
        inherit_grants=body.inherit_grants,
    )
    await database.record_event(
        "MODEL_PROFILE_CLONED",
        str(cloned_model["id"]),
        {
            "source_model_id": model_id,
            "nickname": body.nickname,
            "profile_count": len(profiles),
            "request_defaults": cloned_model["request_defaults"],
            "max_context_tokens": cloned_model["request_limits"]["max_context_tokens"],
            "yarn_factor": body.yarn_factor,
            "shared_artifact_path": cloned_model["artifact_path"],
        },
    )
    return {
        "model": cloned_model,
        "profiles": [
            _profile_payload(StoredProfile(profile=profile, active=True)) for profile in profiles
        ],
        "shared_artifact": True,
    }


@router.patch("/admin/models/{model_id}/profiles/{profile_id}")
async def update_model_profile(
    model_id: str,
    profile_id: str,
    body: ProfileEditRequest,
    request: Request,
    _: AdminPrincipal,
) -> dict[str, object]:
    editable_fields = body.model_fields_set - {"make_default", "restart_workers"}
    if not editable_fields:
        raise RioError(
            "profile_update_empty",
            "Provide at least one profile setting to change",
            status_code=422,
        )
    database = request.app.state.database
    model = await database.model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    records = await request.app.state.profiles.records_for_model(model_id)
    selected = next((record for record in records if record.profile.id == profile_id), None)
    if selected is None:
        raise HTTPException(status_code=404, detail="Placement profile not found")
    target_tp = selected.profile.tensor_parallel_size
    if "tensor_parallel_size" in body.model_fields_set:
        if body.tensor_parallel_size is None:
            raise RioError(
                "invalid_tensor_parallel_size",
                "tensor_parallel_size cannot be null",
                status_code=422,
            )
        target_tp = body.tensor_parallel_size
    gpu_sets = candidate_gpu_sets(request.app.state.inventory, target_tp)
    updated = _apply_profile_edit(
        profile=selected.profile,
        model=model,
        request=body,
        managed_gpu_count=len(request.app.state.inventory.gpus),
        eligible_gpu_sets=gpu_sets,
        llama_cpp_enabled=request.app.state.settings.engines.enable_llama_cpp,
    )
    try:
        saved = await request.app.state.profiles.update(updated, make_default=body.make_default)
    except sqlite3.IntegrityError as exc:
        raise RioError(
            "duplicate_profile",
            "Another placement profile already has these identifying settings",
            status_code=409,
        ) from exc
    if not saved:
        raise HTTPException(status_code=404, detail="Placement profile not found")
    await database.record_event(
        "MODEL_PROFILE_OVERRIDDEN",
        updated.id,
        {
            "model_id": model_id,
            "engine": updated.engine.value,
            "tensor_parallel_size": updated.tensor_parallel_size,
            "max_model_len": updated.max_model_len,
            "make_default": body.make_default,
        },
    )
    drained_worker_ids: list[str] = []
    if body.restart_workers:
        for worker in list(request.app.state.supervisor.workers.values()):
            if worker.model_id == model_id:
                drained_worker_ids.append(worker.id)
                await request.app.state.supervisor.drain(worker.id)
    return {
        "profile": _profile_payload(
            StoredProfile(profile=updated, active=body.make_default or selected.active)
        ),
        "drained_worker_ids": drained_worker_ids,
        "restart_required": not body.restart_workers,
    }


@router.post("/admin/models/{model_id}/profiles/{profile_id}/{action}")
async def set_model_profile_active(
    model_id: str,
    profile_id: str,
    action: str,
    request: Request,
    _: AdminPrincipal,
) -> dict[str, object]:
    if action not in {"enable", "disable"}:
        raise HTTPException(status_code=404, detail="Profile action not found")
    active = action == "enable"
    database = request.app.state.database
    model = await database.model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    saved = await request.app.state.profiles.set_active(
        model_id=model_id, profile_id=profile_id, active=active
    )
    if not saved:
        raise HTTPException(status_code=404, detail="Placement profile not found")
    drained_worker_ids: list[str] = []
    if not active:
        for worker in list(request.app.state.supervisor.workers.values()):
            if worker.profile.id == profile_id:
                drained_worker_ids.append(worker.id)
                await request.app.state.supervisor.drain(worker.id)
    await database.record_event(
        f"MODEL_PROFILE_{action.upper()}D",
        profile_id,
        {"model_id": model_id, "active": active},
    )
    return {
        "profile_id": profile_id,
        "active": active,
        "drained_worker_ids": drained_worker_ids,
    }


async def _scheduler_status(request: Request) -> dict[str, object]:
    database = request.app.state.database
    models = {model["id"]: model["nickname"] for model in await database.list_models()}
    scheduler = request.app.state.scheduler
    workers = []
    for worker in request.app.state.supervisor.workers.values():
        workers.append(
            {
                "worker_id": worker.id,
                "model": models.get(worker.model_id, worker.model_id),
                "state": ("STOPPED" if worker.state is RuntimeState.COLD else worker.state.value),
                "gpu_uuids": worker.gpu_uuids,
                "profile_id": worker.profile.id,
                "ready_at": worker.ready_at.isoformat() if worker.ready_at else None,
                "tensor_parallel_size": worker.profile.tensor_parallel_size,
                "active_requests": len(worker.admitted_request_ids),
                "queued_requests": len(scheduler.queues.for_model(worker.model_id)),
                "accepted_requests": worker.accepted_requests,
            }
        )
    mode: ServiceMode = await database.service_mode()
    return {
        "mode": mode.value,
        "workers": workers,
        "queued_models": {
            models.get(model_id, model_id): len(scheduler.queues.for_model(model_id))
            for model_id in scheduler.queues.pending_models()
        },
    }


@router.get("/admin/status")
async def scheduler_status(request: Request, _: AdminPrincipal) -> dict[str, object]:
    return await _scheduler_status(request)


@router.get("/admin/requests")
async def inference_request_logs(
    test_run_id: str, request: Request, _: AdminPrincipal
) -> dict[str, object]:
    if not test_run_id or len(test_run_id) > 128:
        raise RioError("invalid_test_run_id", "A valid test_run_id is required")
    return {
        "test_run_id": test_run_id,
        "requests": await request.app.state.database.inference_requests_for_test_run(test_run_id),
    }


@router.post("/admin/maintenance", status_code=status.HTTP_202_ACCEPTED)
async def change_maintenance(
    body: MaintenanceRequest, request: Request, _: AdminPrincipal
) -> dict[str, str]:
    scheduler = request.app.state.scheduler
    if body.mode == "drain":
        await scheduler.enter_maintenance()
    else:
        await scheduler.resume()
    return {"mode": (await request.app.state.database.service_mode()).value}


@router.get("/admin/maintenance")
async def maintenance_status(request: Request, _: AdminPrincipal) -> dict[str, object]:
    return await _scheduler_status(request)
