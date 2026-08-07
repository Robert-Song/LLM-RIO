from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_rio.api.dependencies import CurrentPrincipal
from llm_rio.api.schemas import ChatCompletionRequest
from llm_rio.domain import CatalogState, Role, ServiceMode
from llm_rio.errors import MaintenanceError, RioError
from llm_rio.queueing import QueuedRequest
from llm_rio.security import Principal, hash_idempotency_key

router = APIRouter()

request_logger = logging.getLogger("llm_rio.requests")

_IMAGE_CONTENT_TYPES = frozenset({"image", "image_url", "input_image"})


def _messages_for_accounting(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace image payloads with placeholders without changing the forwarded request."""
    accounting_messages: list[dict[str, Any]] = []
    for message in messages:
        accounting_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            accounting_content: list[Any] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in _IMAGE_CONTENT_TYPES:
                    accounting_content.append(part)
                    continue
                image_part: dict[str, Any] = {
                    "type": part.get("type"),
                    "image": "<image>",
                }
                detail = part.get("detail")
                image_url = part.get("image_url")
                if detail is None and isinstance(image_url, dict):
                    detail = image_url.get("detail")
                if detail is not None:
                    image_part["detail"] = detail
                accounting_content.append(image_part)
            accounting_message["content"] = accounting_content
        accounting_messages.append(accounting_message)
    return accounting_messages


def _rough_tokens(value: Any) -> int:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if not encoded or encoded == '""':
        return 0
    return max(1, (len(encoded) + 3) // 4)


def _conservative_prompt_tokens(value: Any) -> int:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    # Byte fallback tokenizers cannot produce more text tokens than UTF-8 input bytes.
    return max(1, len(encoded.encode("utf-8")))


async def _available_models(request: Request, principal: Principal) -> list[dict[str, Any]]:
    database = request.app.state.database
    if principal.role is Role.USER:
        return await database.list_models(principal.key_id)
    return [
        model
        for model in await database.list_models()
        if model["state"] == CatalogState.AVAILABLE.value
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
    return model


def _validate_request(
    request: Request, body: ChatCompletionRequest, model: dict[str, Any]
) -> tuple[int, int, int | None]:
    settings = request.app.state.settings
    model_limits = model["request_limits"]
    max_context_tokens = int(model_limits["max_context_tokens"])

    raw_prompt_cap = model_limits.get("max_prompt_tokens")
    if raw_prompt_cap is not None:
        max_prompt_tokens = int(raw_prompt_cap)
        if settings.max_prompt_tokens is not None:
            max_prompt_tokens = min(max_prompt_tokens, settings.max_prompt_tokens)
    elif settings.max_prompt_tokens is not None:
        max_prompt_tokens = min(settings.max_prompt_tokens, max_context_tokens)
    else:
        max_prompt_tokens = max_context_tokens

    max_output_tokens = (
        min(settings.max_output_tokens, max_context_tokens)
        if settings.max_output_tokens is not None
        else None
    )
    max_n = settings.max_n

    accounting_messages = _messages_for_accounting(body.messages)
    prompt_tokens = _rough_tokens(accounting_messages)
    reservation_prompt_tokens = _conservative_prompt_tokens(accounting_messages)
    requested_output_tokens = body.output_limit
    remaining_context_tokens = max(0, max_context_tokens - prompt_tokens)
    if prompt_tokens > max_prompt_tokens:
        raise RioError("context_length_exceeded", "The prompt exceeds the configured limit")
    if requested_output_tokens is not None:
        if prompt_tokens + requested_output_tokens > max_context_tokens:
            raise RioError(
                "context_length_exceeded", "Prompt plus output exceeds the model context"
            )
        if max_output_tokens is not None and requested_output_tokens > max_output_tokens:
            raise RioError("max_tokens_exceeded", "The requested output limit is too high")
    enforced_output_limit = (
        min(max_output_tokens, remaining_context_tokens)
        if requested_output_tokens is None and max_output_tokens is not None
        else None
    )
    if max_n is not None and body.n > max_n:
        raise RioError("n_exceeded", "The requested number of choices is too high")
    capabilities = set(model["capabilities"])
    if body.response_format and "structured_output" not in capabilities:
        raise RioError(
            "structured_output_not_supported",
            "Structured output was not validated for this model profile",
        )
    reservation_output_tokens = (
        requested_output_tokens
        if requested_output_tokens is not None
        else enforced_output_limit
        if enforced_output_limit is not None
        else remaining_context_tokens
    )
    reservation_tokens = reservation_prompt_tokens + reservation_output_tokens * body.n
    return prompt_tokens, reservation_tokens, enforced_output_limit


@router.get("/v1/models")
async def list_models(request: Request, principal: CurrentPrincipal) -> dict[str, object]:
    models = await _available_models(request, principal)
    data: list[dict[str, Any]] = []
    for model in models:
        profiles = await request.app.state.profiles.for_model(model["id"])
        data.append(
            {
                "id": model["nickname"],
                "object": "model",
                "owned_by": "llm-rio",
                "state": "available" if profiles else "verification_required",
                "callable": bool(profiles),
                "revision": model["resolved_revision"],
                "profile_ids": [profile.id for profile in profiles],
                "placement_profiles": [
                    {
                        "id": profile.id,
                        "gpu_count": profile.gpu_count,
                        "tensor_parallel_size": profile.tensor_parallel_size,
                        "eligible_gpu_sets": profile.eligible_gpu_sets,
                    }
                    for profile in profiles
                ],
                "capabilities": model["capabilities"],
            }
        )
    return {
        "object": "list",
        "data": data,
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
    )
    await request.app.state.database.create_inference_request(
        request_id=request_id,
        principal=principal,
        model_id=model["id"],
        reservation_id=reservation_id,
        estimated_tokens=reservation_estimate,
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
        return StreamingResponse(
            _stream_backend(
                request=request,
                payload=payload,
                lease=lease,
                reservation_id=reservation_id,
                prompt_estimate=prompt_estimate,
                response=backend_response,
            ),
            media_type="text/event-stream",
            headers={
                "X-Request-ID": request_id,
                "X-Worker-ID": lease.worker_id,
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


def _log_request_completion(
    *,
    request: Request,
    payload: dict[str, Any],
    lease: Any,
    completion_status: str,
    prompt_tokens: int,
    completion_tokens: int,
    error_code: str | None,
) -> None:
    admission_time: datetime = request.state.admission_time
    request_logger.info(
        "%s",
        json.dumps(
            {
                "event": "inference_request_completed",
                "test_run_id": getattr(request.state, "test_run_id", None),
                "request_id": lease.request_id,
                "logical_model": payload["model"],
                "selected_worker_id": lease.worker_id,
                "admission_time": admission_time.isoformat(),
                "worker_accepted_time": lease.admitted_at.isoformat(),
                "completion_time": datetime.now(UTC).isoformat(),
                "completion_status": completion_status,
                "token_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                },
                "error_code": error_code,
            },
            sort_keys=True,
        ),
    )


async def _nonstream_backend(
    *,
    request: Request,
    payload: dict[str, Any],
    lease: Any,
    reservation_id: str,
    prompt_estimate: int,
    queue_wait_milliseconds: int,
) -> JSONResponse:
    error_code: str | None = None
    actual = prompt = completion = 0
    try:
        response = await request.app.state.worker_client.post(
            f"{lease.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {lease.internal_api_key}"},
            json=payload,
        )
        if not response.is_success:
            error_code = "worker_rejected"
            raise RioError(
                "worker_rejected",
                "The inference worker rejected the request",
                status_code=502,
            )
        try:
            response_body = response.json()
        except ValueError as exc:
            error_code = "worker_protocol_error"
            raise RioError(
                "worker_protocol_error",
                "The inference worker returned an invalid response",
                status_code=502,
            ) from exc
        response_body["model"] = payload["model"]
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
        return JSONResponse(
            response_body,
            status_code=response.status_code,
            headers={
                "X-Request-ID": lease.request_id,
                "X-Worker-ID": lease.worker_id,
                "X-Queue-Wait-Ms": str(queue_wait_milliseconds),
            },
        )
    except asyncio.CancelledError:
        error_code = "client_disconnected"
        raise
    except httpx.HTTPError as exc:
        error_code = "worker_transport_error"
        raise RioError(
            "worker_unavailable",
            "The selected inference worker became unavailable",
            status_code=503,
        ) from exc
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
        _log_request_completion(
            request=request,
            payload=payload,
            lease=lease,
            completion_status="FAILED" if error_code else "COMPLETED",
            prompt_tokens=prompt,
            completion_tokens=completion,
            error_code=error_code,
        )
        await request.app.state.scheduler.release(lease)


async def _open_worker_stream(
    *,
    request: Request,
    payload: dict[str, Any],
    lease: Any,
    reservation_id: str,
) -> httpx.Response:
    worker_request = request.app.state.worker_client.build_request(
        "POST",
        f"{lease.base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {lease.internal_api_key}"},
        json=payload,
    )
    error_code: str
    try:
        response = await request.app.state.worker_client.send(worker_request, stream=True)
    except asyncio.CancelledError:
        error_code = "client_disconnected"
        await request.app.state.database.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            error_code=error_code,
        )
        _log_request_completion(
            request=request,
            payload=payload,
            lease=lease,
            completion_status="FAILED",
            prompt_tokens=0,
            completion_tokens=0,
            error_code=error_code,
        )
        await request.app.state.scheduler.release(lease)
        raise
    except httpx.HTTPError as exc:
        error_code = "worker_transport_error"
        status_code = 503
        cause: Exception | None = exc
    else:
        if response.is_success:
            return response
        await response.aclose()
        error_code = "worker_rejected"
        status_code = 502
        cause = None
    await request.app.state.database.settle_quota(
        reservation_id=reservation_id,
        actual_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        error_code=error_code,
    )
    _log_request_completion(
        request=request,
        payload=payload,
        lease=lease,
        completion_status="FAILED",
        prompt_tokens=0,
        completion_tokens=0,
        error_code=error_code,
    )
    await request.app.state.scheduler.release(lease)
    error = RioError(
        "worker_unavailable",
        "The selected inference worker could not start the request",
        status_code=status_code,
    )
    if cause is not None:
        raise error from cause
    raise error


async def _stream_backend(
    *,
    request: Request,
    payload: dict[str, Any],
    lease: Any,
    reservation_id: str,
    prompt_estimate: int,
    response: httpx.Response,
):
    prompt = completion = 0
    observed_completion = 0
    error_code: str | None = None
    pending_text = ""
    saw_done = False
    try:
        async for chunk in response.aiter_bytes():
            text = pending_text + chunk.decode("utf-8", errors="ignore")
            lines = text.split("\n")
            pending_text = lines.pop()
            for line in lines:
                line = line.rstrip("\r")
                if not line.startswith("data: "):
                    continue
                if line[6:] == "[DONE]":
                    saw_done = True
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                if event.get("error"):
                    error_code = "worker_stream_error"
                usage = event.get("usage")
                if usage:
                    prompt = int(usage.get("prompt_tokens", 0))
                    completion = int(usage.get("completion_tokens", 0))
                for choice in event.get("choices") or []:
                    observed_completion += _rough_tokens(choice.get("delta", {}).get("content", ""))
            yield chunk
        if not saw_done:
            error_code = "worker_stream_incomplete"
            event = {
                "error": {
                    "message": "The inference stream ended before its terminal event",
                    "type": "worker_stream_error",
                    "code": error_code,
                }
            }
            yield f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
    except asyncio.CancelledError:
        error_code = "client_disconnected"
        raise
    except httpx.HTTPError:
        error_code = "worker_transport_error"
        event = {
            "error": {
                "message": "The inference worker stream became unavailable",
                "type": "worker_stream_error",
                "code": error_code,
            }
        }
        yield f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
    finally:
        await response.aclose()
        actual = (
            prompt + completion if prompt or completion else prompt_estimate + observed_completion
        )
        if request.app.state.settings.quota_charge_requested_maximum and error_code is None:
            actual = lease.estimated_tokens
        settled_prompt = prompt or prompt_estimate
        settled_completion = completion or observed_completion
        await request.app.state.database.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=actual,
            prompt_tokens=settled_prompt,
            completion_tokens=settled_completion,
            error_code=error_code,
        )
        _log_request_completion(
            request=request,
            payload=payload,
            lease=lease,
            completion_status="FAILED" if error_code else "COMPLETED",
            prompt_tokens=settled_prompt,
            completion_tokens=settled_completion,
            error_code=error_code,
        )
        await request.app.state.scheduler.release(lease)
