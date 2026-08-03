from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response, status

from llm_rio.api.dependencies import StaffPrincipal
from llm_rio.api.schemas import GrantUpdate, RegisterModelRequest

router = APIRouter()


@router.post("/staff/models", status_code=status.HTTP_202_ACCEPTED)
async def register_model(
    body: RegisterModelRequest, request: Request, principal: StaffPrincipal
) -> dict[str, str]:
    model_id, job_id = await request.app.state.database.create_model_job(
        nickname=body.nickname,
        repo=body.huggingface_repo,
        revision=body.revision,
        creator_key_id=principal.key_id,
        grant_key_ids=body.grant_to_key_ids,
    )
    request.app.state.registration.start(job_id)
    return {"model_id": model_id, "job_id": job_id}


@router.get("/staff/model-jobs/{job_id}")
async def model_job(
    job_id: str, request: Request, _: StaffPrincipal
) -> dict[str, object]:
    job = await request.app.state.database.get_model_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/staff/models")
async def staff_models(request: Request, _: StaffPrincipal) -> dict[str, object]:
    return {"data": await request.app.state.database.list_models()}


@router.post("/staff/models/{model_id}/disable", status_code=status.HTTP_202_ACCEPTED)
async def disable_model(
    model_id: str, request: Request, _: StaffPrincipal
) -> dict[str, str]:
    if not await request.app.state.database.disable_model(model_id):
        raise HTTPException(status_code=404, detail="Model not found")
    for worker in request.app.state.supervisor.workers.values():
        if worker.model_id == model_id:
            await request.app.state.supervisor.drain(worker.id)
    return {"state": "DISABLED"}


@router.put("/staff/keys/{key_id}/model-grants", status_code=status.HTTP_204_NO_CONTENT)
async def update_grants(
    key_id: str, body: GrantUpdate, request: Request, _: StaffPrincipal
) -> Response:
    try:
        await request.app.state.database.replace_model_grants(key_id, body.model_ids)
    except KeyError:
        raise HTTPException(status_code=404, detail="Key not found") from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)

