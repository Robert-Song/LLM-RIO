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
        available_models = {
            str(model["nickname"]) for model in await database.list_models()
        }
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
    await database.create_key(
        key_id=key_id,
        nickname=body.nickname,
        role=body.role,
        account_id=account_id,
        account_nickname=account_nickname,
        prefix=prefix,
        api_key=token,
        limit_tokens=body.limit_tokens,
        unlimited=body.unlimited,
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
async def rotate_key(
    key_id: str, request: Request, _: AdminPrincipal
) -> KeySecretResponse:
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
async def delete_key(
    key_id: str, request: Request, _: AdminPrincipal
) -> Response:
    if not await request.app.state.database.delete_key(key_id):
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.put("/admin/keys/{key_id}/quota", status_code=status.HTTP_204_NO_CONTENT)
async def update_quota(
    key_id: str, body: QuotaUpdate, request: Request, _: AdminPrincipal
) -> Response:
    if not await request.app.state.database.update_quota(
        key_id, body.limit_tokens, body.unlimited
    ):
        raise HTTPException(status_code=404, detail="Key not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/admin/keys/{key_id}/usage/reset")
async def reset_usage(
    key_id: str, request: Request, _: AdminPrincipal
) -> dict[str, object]:
    result = await request.app.state.database.reset_usage(key_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return result


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
    workers = [
        {
            "id": worker.id,
            "model_id": worker.model_id,
            "gpu_uuids": worker.gpu_uuids,
            "state": worker.state.value,
            "admitted_requests": len(worker.admitted_request_ids),
        }
        for worker in request.app.state.supervisor.workers.values()
        if worker.state is not RuntimeState.COLD
    ]
    mode: ServiceMode = await request.app.state.database.service_mode()
    return {"mode": mode.value, "workers": workers}

