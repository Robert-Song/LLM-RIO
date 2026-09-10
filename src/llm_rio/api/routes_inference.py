from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from llm_rio.api.dependencies import CurrentPrincipal
from llm_rio.api.inference_proxy import (
    _nonstream_backend,
    _open_worker_stream,
    _stream_backend,
    _StreamLeaseFinalizer,
)
from llm_rio.api.inference_validation import _apply_model_defaults, _validate_request
from llm_rio.api.schemas import ChatCompletionRequest
from llm_rio.domain import CatalogState, PlacementProfile, Role, ServiceMode
from llm_rio.errors import MaintenanceError, RioError
from llm_rio.profiles import profile_verified_for_mode
from llm_rio.queueing import QueuedRequest
from llm_rio.security import Principal, hash_idempotency_key

router = APIRouter()


async def _available_models(request: Request, principal: Principal) -> list[dict[str, Any]]:
    database = request.app.state.database
    if principal.role is Role.USER:
        return cast(list[dict[str, Any]], await database.list_models(principal.key_id))
    return [
        model
        for model in await database.list_models()
        if model["state"] == CatalogState.AVAILABLE.value
    ]


def _routable_profiles(
    request: Request, profiles: list[PlacementProfile]
) -> list[PlacementProfile]:
    scheduler = getattr(request.app.state, "scheduler", None)
    runtime = getattr(scheduler, "kvcached", None)
    kvcached_required = bool(getattr(runtime, "enabled", False))
    ram_weight_cache_required = bool(
        getattr(getattr(scheduler, "planner", None), "prism_weight_cache_enabled", False)
    )
    return [
        profile
        for profile in profiles
        if profile_verified_for_mode(
            profile,
            kvcached_required=kvcached_required,
            ram_weight_cache_required=ram_weight_cache_required,
            queue_mode_required=bool(
                getattr(getattr(scheduler, "settings", None), "queue_mode_enabled", False)
            ),
        )
    ]


async def _resolve_model(request: Request, principal: Principal, nickname: str) -> dict[str, Any]:
    database = request.app.state.database
    available = [model["nickname"] for model in await _available_models(request, principal)]
    model = await database.model_by_nickname(nickname)
    if model is None:
        raise RioError(
            "model_not_found",
            f"Model '{nickname}' is not in this machine's catalog",
            status_code=404,
            details={"available_models": available},
        )
    if principal.role is Role.USER and not await database.has_model_grant(
        principal.key_id, model["id"]
    ):
        raise RioError(
            "model_not_allowed",
            f"This key is not granted access to model '{nickname}'",
            status_code=403,
            details={"available_models": available},
        )
    if model["state"] != CatalogState.AVAILABLE.value:
        raise RioError(
            "model_unavailable",
            f"Model '{nickname}' is not available",
            status_code=409,
            details={"catalog_state": model["state"], "available_models": available},
        )
    profiles = await request.app.state.profiles.for_model(model["id"])
    if not profiles:
        raise RioError(
            "model_verification_required",
            "This model needs verification on the current machine fingerprint",
            status_code=503,
        )
    if not _routable_profiles(request, profiles):
        runtime = getattr(getattr(request.app.state, "scheduler", None), "kvcached", None)
        backend = "kvcached" if getattr(runtime, "enabled", False) else "native"
        error_code = (
            "prism_profile_revalidation_required"
            if backend == "kvcached"
            else "model_backend_verification_required"
        )
        raise RioError(
            error_code,
            f"This model needs a placement profile verified with {backend}",
            status_code=503,
            details={"required_memory_backend": backend},
        )
    return cast(dict[str, Any], model)


@router.get("/v1/models")
async def list_models(request: Request, principal: CurrentPrincipal) -> dict[str, object]:
    models = await _available_models(request, principal)
    data: list[dict[str, Any]] = []
    for model in models:
        profiles = await request.app.state.profiles.for_model(model["id"])
        routable_profiles = _routable_profiles(request, profiles)
        state = (
            "available"
            if routable_profiles
            else "prism_profile_revalidation_required"
            if profiles
            else "verification_required"
        )
        data.append(
            {
                "id": model["nickname"],
                "object": "model",
                "owned_by": "llm-rio",
                "state": state,
                "callable": bool(routable_profiles),
                "revision": model["resolved_revision"],
                "profile_ids": [profile.id for profile in profiles],
                "placement_profiles": [
                    {
                        "id": profile.id,
                        "engine": profile.engine.value,
                        "memory_backend": profile.memory_backend,
                        "gpu_count": profile.gpu_count,
                        "tensor_parallel_size": profile.tensor_parallel_size,
                        "eligible_gpu_sets": profile.eligible_gpu_sets,
                    }
                    for profile in profiles
                ],
                "capabilities": model["capabilities"],
                "request_defaults": model.get("request_defaults", {}),
                "source_model_id": model.get("source_model_id"),
            }
        )
    return {
        "object": "list",
        "data": data,
    }


@router.get("/v1/me/usage")
async def usage(request: Request, principal: CurrentPrincipal) -> dict[str, Any]:
    return cast(dict[str, Any], await request.app.state.database.usage(principal))


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request, body: ChatCompletionRequest, principal: CurrentPrincipal
) -> JSONResponse | StreamingResponse:
    request.state.model = body.model
    if await request.app.state.database.service_mode() is not ServiceMode.ACTIVE:
        raise MaintenanceError()
    model = await _resolve_model(request, principal, body.model)
    body = _apply_model_defaults(body, model)
    prompt_estimate, reservation_estimate, enforced_output_limit = _validate_request(
        request, body, model
    )
    payload = body.model_dump(exclude_none=True)
    if enforced_output_limit is not None:
        payload["max_tokens"] = enforced_output_limit
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    request.state.admission_time = datetime.now(UTC)
    test_run_id = request.headers.get("X-Test-Run-ID")
    client_worker = request.headers.get("X-Client-Worker")
    if test_run_id is not None and len(test_run_id) > 128:
        raise RioError("invalid_test_run_id", "X-Test-Run-ID is too long")
    if client_worker is not None and len(client_worker) > 128:
        raise RioError("invalid_client_worker", "X-Client-Worker is too long")
    request.state.test_run_id = test_run_id
    idempotency_value = request.headers.get("Idempotency-Key") or request_id
    reservation_id = await request.app.state.database.reserve_quota(
        request_id=request_id,
        idempotency_hash=hash_idempotency_key(idempotency_value),
        principal=principal,
        model_id=model["id"],
        estimated_tokens=reservation_estimate,
        estimated_prompt_tokens=prompt_estimate,
        test_run_id=test_run_id,
        client_worker=client_worker,
    )
    queued = QueuedRequest(
        id=request_id,
        model_id=model["id"],
        tenant_id=principal.quota_account_id,
        estimated_tokens=reservation_estimate,
        payload=payload,
        reservation_id=reservation_id,
        is_stream=body.stream,
    )
    try:
        lease = await request.app.state.scheduler.enqueue(queued)
    except BaseException:
        await request.app.state.database.release_reservation(reservation_id, "queue_rejected")
        raise
    queue_wait_milliseconds = max(
        0, int((datetime.now(UTC) - queued.enqueued_at).total_seconds() * 1000)
    )
    request.state.queue_wait_ms = queue_wait_milliseconds

    payload = dict(queued.payload)
    payload["model"] = model["nickname"]
    if body.stream:
        payload["stream_options"] = {**(body.stream_options or {}), "include_usage": True}
        backend_response = await _open_worker_stream(
            request=request,
            payload=payload,
            lease=lease,
            reservation_id=reservation_id,
        )
        finalizer = _StreamLeaseFinalizer(
            request=request,
            payload=payload,
            lease=lease,
            reservation_id=reservation_id,
            response=backend_response,
        )
        return StreamingResponse(
            _stream_backend(
                request=request,
                payload=payload,
                lease=lease,
                reservation_id=reservation_id,
                prompt_estimate=prompt_estimate,
                response=backend_response,
                finalizer=finalizer,
            ),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "X-Worker-ID": lease.worker_id,
                "X-Queue-Wait-Ms": str(queue_wait_milliseconds),
                "Cache-Control": "no-cache",
            },
            background=BackgroundTask(finalizer.abandon),
        )
    return await _nonstream_backend(
        request=request,
        payload=payload,
        lease=lease,
        reservation_id=reservation_id,
        prompt_estimate=prompt_estimate,
        queue_wait_milliseconds=queue_wait_milliseconds,
    )
