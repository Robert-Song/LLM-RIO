from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request, Response, status

from llm_rio.api.dependencies import StaffPrincipal
from llm_rio.api.schemas import (
    GrantUpdate,
    ModelAccessUpdate,
    ModelJobRetryRequest,
    RegisterModelRequest,
)
from llm_rio.errors import RioError
from llm_rio.services.validation_jobs import queue_validation

router = APIRouter()


@router.post("/staff/models", status_code=status.HTTP_202_ACCEPTED)
async def register_model(
    body: RegisterModelRequest, request: Request, principal: StaffPrincipal
) -> dict[str, str]:
    database = request.app.state.database
    if body.engine.value not in request.app.state.scheduler.mode.capabilities.engines:
        raise RioError(
            "unsupported_engine", "Engine is unavailable in this serving mode", status_code=422
        )
    if body.local_path:
        from pathlib import Path

        artifact = Path(body.local_path)
        if not artifact.is_absolute() or not artifact.exists():
            raise RioError(
                "invalid_local_path",
                "Use an existing absolute server-local artifact path",
                status_code=422,
            )
        if body.engine.value == "llama.cpp" and (
            not artifact.is_file() or artifact.suffix.lower() != ".gguf"
        ):
            raise RioError(
                "invalid_local_path", "llama.cpp registration requires a GGUF file", status_code=422
            )
        if body.engine.value == "vllm" and not artifact.is_dir():
            raise RioError(
                "invalid_local_path",
                "vLLM registration requires a model directory",
                status_code=422,
            )
    grant_key_ids: list[str] = []
    for selector in body.grant_to_keys:
        key = await database.key_by_selector(selector)
        if key is None:
            raise RioError(
                "grant_key_not_found",
                f"API key '{selector}' does not exist or is inactive",
                status_code=404,
            )
        grant_key_ids.append(str(key["id"]))
    model_id, job_id = await database.create_model_job(
        nickname=body.nickname,
        repo=body.huggingface_repo or "local",
        local_path=body.local_path,
        engine=body.engine.value,
        revision=body.revision,
        creator_key_id=principal.key_id,
        grant_key_ids=grant_key_ids,
    )
    request.app.state.registration.start(job_id)
    return {"model_id": model_id, "job_id": job_id}


@router.get("/staff/model-jobs/{job_id}")
async def model_job(job_id: str, request: Request, _: StaffPrincipal) -> dict[str, object]:
    job = await request.app.state.database.get_model_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return cast(dict[str, object], job)


@router.post("/staff/model-jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_model_job(
    job_id: str,
    request: Request,
    _: StaffPrincipal,
    body: ModelJobRetryRequest | None = None,
) -> dict[str, object]:
    result = await queue_validation(
        request.app.state.database,
        job_id,
        engines=request.app.state.scheduler.mode.capabilities.engines,
        profile_id=body.profile_id if body else None,
        overrides=(
            body.validation_overrides.model_dump(exclude_none=True, exclude_unset=True)
            if body and body.validation_overrides
            else None
        ),
    )
    await request.app.state.database.record_event(
        "MODEL_VALIDATION_QUEUED",
        str(result["model_id"]),
        result,
    )
    request.app.state.registration.start(job_id)
    return result


@router.get("/staff/models")
async def staff_models(request: Request, _: StaffPrincipal) -> dict[str, object]:
    return {"data": await request.app.state.database.list_models(include_registration_jobs=True)}


@router.post("/staff/models/{model_id}/disable", status_code=status.HTTP_202_ACCEPTED)
async def disable_model(model_id: str, request: Request, _: StaffPrincipal) -> dict[str, str]:
    await request.app.state.registration.cancel_model(model_id)
    if not await request.app.state.database.disable_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    for worker in request.app.state.supervisor.workers.values():
        if worker.model_id == model_id:
            await request.app.state.supervisor.drain(worker.id)
    return {"state": "DISABLED"}


@router.post("/staff/model-access")
async def update_model_access(
    body: ModelAccessUpdate, request: Request, _: StaffPrincipal
) -> dict[str, object]:
    database = request.app.state.database
    key = await database.key_by_selector(body.key)
    if key is None:
        raise RioError(
            "grant_key_not_found",
            f"API key '{body.key}' does not exist or is inactive",
            status_code=404,
        )
    models = await database.update_model_access(
        key_id=str(key["id"]),
        model_nicknames=body.models,
        mode=body.mode,
    )
    return {"key": key["nickname"], "models": models}


@router.put("/staff/keys/{key_id}/model-grants", status_code=status.HTTP_204_NO_CONTENT)
async def update_grants(
    key_id: str, body: GrantUpdate, request: Request, _: StaffPrincipal
) -> Response:
    try:
        await request.app.state.database.replace_model_grants(key_id, body.model_ids)
    except KeyError:
        raise HTTPException(status_code=404, detail="Key not found") from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
