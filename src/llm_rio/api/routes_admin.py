from __future__ import annotations

import sqlite3
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status

from llm_rio.api.dependencies import AdminPrincipal
from llm_rio.api.schemas import (
    CreateKeyRequest,
    KeySecretResponse,
    MaintenanceRequest,
    ModelProfileCloneRequest,
    ModelRequestDefaultsUpdate,
    ModelVerificationTrustRequest,
    ProfileEditRequest,
    QuotaUpdate,
    UsageSummarizeRequest,
)
from llm_rio.errors import RioError
from llm_rio.inventory import candidate_gpu_sets
from llm_rio.profiles import (
    StoredProfile,
    launch_configuration_changed,
    profile_key,
    profile_to_dict,
)
from llm_rio.security import issue_api_key
from llm_rio.services.diagnostics import DiagnosticsService
from llm_rio.services.profile_edit import _apply_profile_edit, _gguf_files, _profile_payload

router = APIRouter()


@router.post(
    "/admin/keys",
    response_model=KeySecretResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_key(
    request: Request, body: CreateKeyRequest, _: AdminPrincipal
) -> KeySecretResponse:
    from llm_rio.services.access import create_key as create

    return await create(request.app.state.database, body)


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
    result: dict[str, object] | None = await request.app.state.database.reset_usage(key_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return result


@router.post("/admin/usage/summarize")
async def summarize_usage(
    request: Request,
    _: AdminPrincipal,
    body: UsageSummarizeRequest | None = None,
) -> dict[str, object]:
    database = request.app.state.database
    result: dict[str, object] = await database.summarize_usage(body.through if body else None)
    await database.record_event(
        "USAGE_SUMMARIZED",
        payload={key: result[key] for key in ("period_start", "period_end", "deleted")},
    )
    return result


@router.get("/admin/models/{model_id}/profiles")
async def list_model_profiles(
    model_id: str, request: Request, _: AdminPrincipal
) -> dict[str, object]:
    model = await request.app.state.database.model_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    records = await request.app.state.profiles.records_for_model(model_id)
    mode = request.app.state.scheduler.mode
    data = []
    for record in records:
        payload = _profile_payload(record)
        payload["eligibility"] = asdict(mode.eligibility(record.profile))
        data.append(payload)
    return {
        "data": data,
        "saved_measurements": [
            _profile_payload(record)
            for record in await request.app.state.profiles.records_for_model(
                model_id, include_other_machines=True
            )
        ],
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
    target_engine = body.engine or selected.profile.engine
    if target_engine.value not in request.app.state.scheduler.mode.capabilities.engines:
        raise RioError(
            "unsupported_engine", "Engine is unavailable in the running mode", status_code=422
        )
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
        conflicting = next(
            (
                record
                for record in await request.app.state.profiles.records_for_model(model_id)
                if record.profile.id != profile_id
                and profile_key(profile_to_dict(record.profile))
                == profile_key(profile_to_dict(updated))
            ),
            None,
        )
        if conflicting is None:
            raise
        raise RioError(
            "duplicate_profile",
            f"Placement profile {conflicting.profile.id} already has these settings "
            f"({'active' if conflicting.active else 'inactive'}). "
            "Select that existing profile and enable it if needed.",
            status_code=409,
            details={
                "existing_profile_id": conflicting.profile.id,
                "existing_profile_active": conflicting.active,
            },
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
    launch_changed = launch_configuration_changed(selected.profile, updated)
    if body.restart_workers or launch_changed:
        for worker in list(request.app.state.supervisor.workers.values()):
            if worker.model_id == model_id:
                drained_worker_ids.append(worker.id)
                await request.app.state.supervisor.drain(worker.id)
    return {
        "profile": _profile_payload(
            StoredProfile(profile=updated, active=body.make_default or selected.active)
        ),
        "drained_worker_ids": drained_worker_ids,
        "restart_required": not (body.restart_workers or launch_changed),
        "eligibility": asdict(request.app.state.scheduler.mode.eligibility(updated)),
        "verification_required": (
            []
            if request.app.state.scheduler.mode.eligibility(updated).allowed
            else [request.app.state.scheduler.mode.eligibility(updated).reason]
        ),
    }


@router.post("/admin/models/{model_id}/profiles/{profile_id}/trust")
async def trust_measurements(
    model_id: str,
    profile_id: str,
    body: ModelVerificationTrustRequest,
    request: Request,
    principal: AdminPrincipal,
) -> dict[str, object]:
    from llm_rio.services.profile_trust import trust_measurements as trust

    return await trust(
        request.app.state.profiles,
        model_id=model_id,
        profile_id=profile_id,
        mode=request.app.state.scheduler.mode,
        gpu_uuids={gpu.uuid for gpu in request.app.state.inventory.gpus},
        reason=body.reason,
        actor=principal.key_id,
    )


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
    if active:
        records = await request.app.state.profiles.records_for_model(model_id)
        selected = next((record for record in records if record.profile.id == profile_id), None)
        if selected is None:
            raise HTTPException(status_code=404, detail="Placement profile not found")
        eligibility = request.app.state.scheduler.mode.eligibility(selected.profile)
        if selected.profile.model_revision != model.get("resolved_revision"):
            raise RioError(
                "profile_ineligible",
                "Artifact revision changed; run Validate/Revalidate",
                status_code=409,
                details={"reason": "artifact_revision_changed"},
            )
        if not eligibility.allowed:
            raise RioError(
                "profile_ineligible",
                "Run Validate/Revalidate before enabling",
                status_code=409,
                details={"reason": eligibility.reason},
            )
        if model.get("source_type") == "local":
            from llm_rio.artifacts import local_artifact_unchanged

            if not local_artifact_unchanged(
                Path(str(model["artifact_path"])), model.get("artifact_hashes") or []
            ):
                raise RioError(
                    "profile_ineligible",
                    "Local artifact changed; run Validate/Revalidate",
                    status_code=409,
                    details={"reason": "artifact_changed"},
                )
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


def _diagnostics(request: Request) -> DiagnosticsService:
    return DiagnosticsService(
        request.app.state.database, request.app.state.scheduler, request.app.state.inventory
    )


@router.get("/admin/dashboard")
async def dashboard(request: Request, _: AdminPrincipal) -> dict[str, object]:
    return await _diagnostics(request).dashboard()


@router.get("/admin/status")
async def scheduler_status(request: Request, _: AdminPrincipal) -> dict[str, object]:
    return await _diagnostics(request).status()


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
    return await _diagnostics(request).status()


@router.get("/admin/capabilities")
async def capabilities(request: Request, _: AdminPrincipal) -> dict[str, object]:
    from dataclasses import asdict

    from llm_rio.operations.build_identity import application_identity

    settings = request.app.state.settings
    return {
        "application": application_identity(),
        "configuration": {
            "database_path": str(request.app.state.settings.database_path.resolve()),
            "managed_gpu_uuids": [gpu.uuid for gpu in request.app.state.inventory.gpus],
            "quota_charge_requested_maximum": settings.quota_charge_requested_maximum,
        },
        **asdict(request.app.state.scheduler.mode.capabilities),
        "actions": [
            "register",
            "validate",
            "profile_edit",
            "profile_activation",
            "trust_measurements",
            "maintenance",
            "usage_summary",
        ],
        "model_sources": ["huggingface", "local_directory"]
        + (
            ["local_gguf"]
            if "llama.cpp" in request.app.state.scheduler.mode.capabilities.engines
            else []
        ),
    }
