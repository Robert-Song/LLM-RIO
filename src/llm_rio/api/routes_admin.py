from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status

from llm_rio.api.dependencies import AdminPrincipal
from llm_rio.api.schemas import (
    CreateKeyRequest,
    KeySecretResponse,
    MaintenanceRequest,
    QuotaUpdate,
)
from llm_rio.domain import RuntimeState, ServiceMode
from llm_rio.errors import RioError
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
        "SELECT nickname FROM api_keys WHERE id = ?", (key_id,)
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
