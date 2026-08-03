from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_rio.api.dependencies import CurrentPrincipal
from llm_rio.api.schemas import ChatCompletionRequest
from llm_rio.domain import CatalogState, ServiceMode
from llm_rio.errors import MaintenanceError, RioError
from llm_rio.queueing import QueuedRequest
from llm_rio.security import hash_idempotency_key

router = APIRouter()


def _rough_tokens(value: Any) -> int:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if not encoded or encoded == '""':
        return 0
    return max(1, (len(encoded) + 3) // 4)

def _conservative_prompt_tokens(value: Any) -> int:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    # Byte fallback tokenizers cannot produce more text tokens than UTF-8 input bytes.
    return max(1, len(encoded.encode("utf-8")))


def _contains_multimodal(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") not in {None, "text"}
            for part in content
        ):
            return True
    return False


async def _resolve_model(request: Request, principal: CurrentPrincipal, nickname: str) -> dict[str, Any]:
    database = request.app.state.database
    available = [model["nickname"] for model in await database.list_models(principal.key_id)]
    model = await database.model_by_nickname(nickname)
    if model is None:
        raise RioError(
            "model_not_found",
            f"Model '{nickname}' is not in this machine's catalog",
            status_code=404,
            details={"available_models": available},
        )
    if not await database.has_model_grant(principal.key_id, model["id"]):
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
    return model


def _validate_request(
    request: Request, body: ChatCompletionRequest, model: dict[str, Any]
) -> tuple[int, int]:
    settings = request.app.state.settings
    model_limits = model["request_limits"]
    max_context_tokens = int(model_limits["max_context_tokens"])
    max_prompt_tokens = (
        min(settings.max_prompt_tokens, max_context_tokens)
        if settings.max_prompt_tokens is not None
        else max_context_tokens
    )
    limits = {**{
        "max_prompt_tokens": max_prompt_tokens,
        "max_output_tokens": settings.max_output_tokens,
        "max_n": settings.max_n,
    }, **model_limits}
    prompt_tokens = _rough_tokens(body.messages)
    reservation_prompt_tokens = _conservative_prompt_tokens(body.messages)
    if prompt_tokens > int(limits["max_prompt_tokens"]):
        raise RioError("context_length_exceeded", "The prompt exceeds the configured limit")
    max_context_tokens = int(limits["max_context_tokens"])
    if prompt_tokens + body.output_limit > max_context_tokens:
        raise RioError("context_length_exceeded", "Prompt plus output exceeds the model context")
    if body.output_limit > int(limits["max_output_tokens"]):
        raise RioError("max_tokens_exceeded", "The requested output limit is too high")
    if body.n > int(limits["max_n"]):
        raise RioError("n_exceeded", "The requested number of choices is too high")
    capabilities = set(model["capabilities"])
    if body.tools and "tools" not in capabilities:
        raise RioError("tools_not_supported", "Tool use was not validated for this model")
    if body.response_format and "structured_output" not in capabilities:
        raise RioError(
            "structured_output_not_supported",
            "Structured output was not validated for this model profile",
        )
    if _contains_multimodal(body.messages) and "vision" not in capabilities:
        raise RioError("multimodal_not_supported", "Multimodal input was not validated for this model")
    return prompt_tokens, reservation_prompt_tokens + body.output_limit * body.n


@router.get("/v1/models")
async def list_models(request: Request, principal: CurrentPrincipal) -> dict[str, object]:
    models = await request.app.state.database.list_models(principal.key_id)
    return {
        "object": "list",
        "data": [
            {
                "id": model["nickname"],
                "object": "model",
                "owned_by": "llm-rio",
                "capabilities": model["capabilities"],
            }
            for model in models
        ],
    }


@router.get("/v1/me/usage")
async def usage(request: Request, principal: CurrentPrincipal) -> dict[str, Any]:
    return await request.app.state.database.usage(principal)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request: Request, body: ChatCompletionRequest, principal: CurrentPrincipal
) -> JSONResponse | StreamingResponse:
    request.state.model = body.model
    if await request.app.state.database.service_mode() is not ServiceMode.ACTIVE:
        raise MaintenanceError()
    model = await _resolve_model(request, principal, body.model)
    prompt_estimate, reservation_estimate = _validate_request(request, body, model)
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    idempotency_value = request.headers.get("Idempotency-Key") or request_id
    reservation_id = await request.app.state.database.reserve_quota(
        request_id=request_id,
        idempotency_hash=hash_idempotency_key(idempotency_value),
        principal=principal,
        model_id=model["id"],
        estimated_tokens=reservation_estimate,
    )
    await request.app.state.database.create_inference_request(
        request_id=request_id,
        principal=principal,
        model_id=model["id"],
        reservation_id=reservation_id,
        estimated_tokens=reservation_estimate,
    )
    queued = QueuedRequest(
        id=request_id,
        model_id=model["id"],
        tenant_id=principal.quota_account_id,
        estimated_tokens=reservation_estimate,
        payload=body.model_dump(exclude_none=True),
        reservation_id=reservation_id,
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

    payload = body.model_dump(exclude_none=True)
    payload["model"] = model["nickname"]
    if body.stream:
        payload["stream_options"] = {**(body.stream_options or {}), "include_usage": True}
        return StreamingResponse(
            _stream_backend(
                request=request,
                payload=payload,
                lease=lease,
                reservation_id=reservation_id,
                prompt_estimate=prompt_estimate,
            ),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "X-Queue-Wait-Ms": str(queue_wait_milliseconds),
                "Cache-Control": "no-cache",
            },
        )
    return await _nonstream_backend(
        request=request,
        payload=payload,
        lease=lease,
        reservation_id=reservation_id,
        prompt_estimate=prompt_estimate,
        queue_wait_milliseconds=queue_wait_milliseconds,
    )


async def _nonstream_backend(
    *, request: Request, payload: dict[str, Any], lease: Any, reservation_id: str,
    prompt_estimate: int, queue_wait_milliseconds: int
) -> JSONResponse:
    error_code: str | None = None
    actual = prompt = completion = 0
    try:
        response = await request.app.state.worker_client.post(
                f"{lease.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {lease.internal_api_key}"},
                json=payload,
            )
        content_type = response.headers.get("content-type", "application/json")
        try:
            response_body = response.json()
        except ValueError:
            response_body = {"error": {"message": response.text, "type": "worker_error"}}
        if response.is_success:
            usage = response_body.get("usage") or {}
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            actual = int(usage.get("total_tokens", prompt + completion))
            if actual == 0:
                prompt = prompt_estimate
                completion = sum(
                    _rough_tokens(choice.get("message", {}).get("content", ""))
                    for choice in response_body.get("choices") or []
                )
                actual = prompt + completion
        else:
            error_code = "worker_rejected"
        return JSONResponse(
            response_body,
            status_code=response.status_code,
            headers={
                "X-Request-ID": lease.request_id,
                "X-Queue-Wait-Ms": str(queue_wait_milliseconds),
                "Content-Type": content_type,
            },
        )
    except (httpx.HTTPError, asyncio.CancelledError):
        error_code = "worker_transport_error"
        raise
    finally:
        if request.app.state.settings.quota_charge_requested_maximum and error_code is None:
            actual = lease.estimated_tokens
        await request.app.state.database.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=actual,
            prompt_tokens=prompt,
            completion_tokens=completion,
            error_code=error_code,
        )
        await request.app.state.scheduler.release(lease)


async def _stream_backend(
    *,
    request: Request,
    payload: dict[str, Any],
    lease: Any,
    reservation_id: str,
    prompt_estimate: int,
):
    prompt = completion = 0
    observed_completion = 0
    error_code: str | None = None
    pending_text = ""
    try:
        async with request.app.state.worker_client.stream(
                "POST",
                f"{lease.base_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {lease.internal_api_key}"},
                json=payload,
            ) as response:
                if not response.is_success:
                    error_code = "worker_rejected"
                async for chunk in response.aiter_bytes():
                    text = pending_text + chunk.decode("utf-8", errors="ignore")
                    lines = text.split("\n")
                    pending_text = lines.pop()
                    for line in lines:
                        line = line.rstrip("\r")
                        if not line.startswith("data: ") or line[6:] == "[DONE]":
                            continue
                        try:
                            event = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        usage = event.get("usage")
                        if usage:
                            prompt = int(usage.get("prompt_tokens", 0))
                            completion = int(usage.get("completion_tokens", 0))
                        for choice in event.get("choices") or []:
                            observed_completion += _rough_tokens(
                                choice.get("delta", {}).get("content", "")
                            )
                    yield chunk
    except asyncio.CancelledError:
        error_code = "client_disconnected"
        raise
    except httpx.HTTPError:
        error_code = "worker_transport_error"
        raise
    finally:
        actual = prompt + completion if prompt or completion else prompt_estimate + observed_completion
        if request.app.state.settings.quota_charge_requested_maximum and error_code is None:
            actual = lease.estimated_tokens
        await request.app.state.database.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=actual,
            prompt_tokens=prompt or prompt_estimate,
            completion_tokens=completion or observed_completion,
            error_code=error_code,
        )
        await request.app.state.scheduler.release(lease)

