"""Worker HTTP forwarding and cancellation-safe lease finalization."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse

from llm_rio.api.inference_validation import _rough_tokens
from llm_rio.errors import RioError
from llm_rio.worker_protocol import SSEDecoder, completion_choices, token_usage

request_logger = logging.getLogger("llm_rio.requests")


def _completion_tokens(message: dict[str, Any]) -> int:
    return sum(
        _rough_tokens(message.get(field))
        for field in ("content", "reasoning_content", "reasoning", "tool_calls")
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


async def _finish_lease(
    *,
    request: Request,
    payload: dict[str, Any],
    lease: Any,
    reservation_id: str,
    actual_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    error_code: str | None,
) -> None:
    """Persist a terminal request state before freeing the worker admission."""
    try:
        await request.app.state.database.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=actual_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code=error_code,
        )
    except Exception:
        request_logger.exception("Could not settle inference request %s", lease.request_id)
    finally:
        try:
            await request.app.state.scheduler.release(lease)
        except Exception:
            request_logger.exception("Could not release worker admission for %s", lease.request_id)
    _log_request_completion(
        request=request,
        payload=payload,
        lease=lease,
        completion_status="FAILED" if error_code else "COMPLETED",
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error_code=error_code,
    )


async def _complete_lease(**kwargs: Any) -> None:
    """Run terminal state transfer in its own task so disconnects cannot strand a lease."""
    cleanup = asyncio.create_task(_finish_lease(**kwargs), name="inference-lease-cleanup")
    try:
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        cleanup.add_done_callback(_consume_cleanup_result)
        raise


def _consume_cleanup_result(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        request_logger.exception("Inference lease cleanup task failed")


async def _close_worker_response(response: httpx.Response) -> None:
    try:
        await asyncio.wait_for(response.aclose(), timeout=5.0)
    except TimeoutError:
        request_logger.warning("Timed out closing worker response")
    except Exception:
        request_logger.warning("Could not close worker response", exc_info=True)


class _StreamLeaseFinalizer:
    """Finalize an opened worker stream exactly once, even if its body never starts."""

    def __init__(
        self,
        *,
        request: Request,
        payload: dict[str, Any],
        lease: Any,
        reservation_id: str,
        response: httpx.Response,
    ) -> None:
        self.request = request
        self.payload = payload
        self.lease = lease
        self.reservation_id = reservation_id
        self.response = response
        self._finished = False
        self._lock = asyncio.Lock()

    async def finish(
        self,
        *,
        actual_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        error_code: str | None,
    ) -> None:
        async with self._lock:
            if self._finished:
                return
            self._finished = True
        try:
            await _complete_lease(
                request=self.request,
                payload=self.payload,
                lease=self.lease,
                reservation_id=self.reservation_id,
                actual_tokens=actual_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error_code=error_code,
            )
        finally:
            await _close_worker_response(self.response)

    async def abandon(self) -> None:
        await self.finish(
            actual_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            error_code="client_disconnected",
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
        request_timeout = request.app.state.settings.worker_request_timeout_seconds
        worker_request = request.app.state.worker_client.post(
            f"{lease.base_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {lease.internal_api_key}"},
            json=payload,
        )
        response = (
            await asyncio.wait_for(worker_request, timeout=request_timeout)
            if request_timeout is not None
            else await worker_request
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
            choices = completion_choices(response_body, streaming=False)
            usage = token_usage(response_body)
        except (TypeError, ValueError) as exc:
            error_code = "worker_protocol_error"
            raise RioError(
                "worker_protocol_error",
                "The inference worker returned an invalid response",
                status_code=502,
            ) from exc
        response_body["model"] = payload["model"]
        if usage is not None:
            prompt, completion, actual = usage
        else:
            prompt = prompt_estimate
            completion = sum(_completion_tokens(choice["message"]) for choice in choices)
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
    except TimeoutError as exc:
        error_code = "worker_request_timeout"
        raise RioError(
            "worker_request_timeout",
            "The selected inference worker did not respond before the request timeout",
            status_code=504,
        ) from exc
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
        await _complete_lease(
            request=request,
            payload=payload,
            lease=lease,
            reservation_id=reservation_id,
            actual_tokens=actual,
            prompt_tokens=prompt,
            completion_tokens=completion,
            error_code=error_code,
        )


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
    try:
        request_timeout = request.app.state.settings.worker_request_timeout_seconds
        pending_response = request.app.state.worker_client.send(worker_request, stream=True)
        response = (
            await asyncio.wait_for(pending_response, timeout=request_timeout)
            if request_timeout is not None
            else await pending_response
        )
    except asyncio.CancelledError:
        error_code = "client_disconnected"
        await _complete_lease(
            request=request,
            payload=payload,
            lease=lease,
            reservation_id=reservation_id,
            actual_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            error_code=error_code,
        )
        raise
    except TimeoutError as exc:
        error_code = "worker_request_timeout"
        status_code = 504
        cause: Exception | None = exc
    except httpx.HTTPError as exc:
        error_code = "worker_transport_error"
        status_code = 503
        cause = exc
    else:
        if response.is_success:
            return cast(httpx.Response, response)
        error_code = "worker_rejected"
        status_code = 502
        cause = None
        await _complete_lease(
            request=request,
            payload=payload,
            lease=lease,
            reservation_id=reservation_id,
            actual_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            error_code=error_code,
        )
        await _close_worker_response(response)
        raise RioError(
            "worker_unavailable",
            "The selected inference worker could not start the request",
            status_code=status_code,
        )
    await _complete_lease(
        request=request,
        payload=payload,
        lease=lease,
        reservation_id=reservation_id,
        actual_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        error_code=error_code,
    )
    error = RioError(
        "worker_request_timeout"
        if error_code == "worker_request_timeout"
        else "worker_unavailable",
        "The selected inference worker did not respond before the request timeout"
        if error_code == "worker_request_timeout"
        else "The selected inference worker could not start the request",
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
    finalizer: _StreamLeaseFinalizer | None = None,
) -> AsyncIterator[bytes]:
    prompt = completion = 0
    observed_completion = 0
    error_code: str | None = None
    decoder = SSEDecoder()
    usage_seen = False
    saw_done = False
    finalizer = finalizer or _StreamLeaseFinalizer(
        request=request,
        payload=payload,
        lease=lease,
        reservation_id=reservation_id,
        response=response,
    )
    try:
        chunks = response.aiter_bytes()
        stream_idle_timeout = request.app.state.settings.worker_stream_idle_timeout_seconds
        while True:
            try:
                next_chunk = anext(chunks)
                chunk = (
                    await asyncio.wait_for(next_chunk, timeout=stream_idle_timeout)
                    if stream_idle_timeout is not None
                    else await next_chunk
                )
            except StopAsyncIteration:
                break
            await request.app.state.scheduler.touch(lease)
            for data in decoder.feed(chunk):
                if data.strip() == "[DONE]":
                    saw_done = True
                    continue
                event = json.loads(data)
                choices = completion_choices(event, streaming=True)
                usage = token_usage(event)
                if event.get("error"):
                    error_code = "worker_stream_error"
                if usage is not None:
                    prompt, completion, _ = usage
                    usage_seen = True
                for choice in choices:
                    observed_completion += _completion_tokens(choice.get("delta") or {})
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
    except TimeoutError:
        error_code = "worker_stream_idle_timeout"
        event = {
            "error": {
                "message": "The inference worker stream was idle for too long",
                "type": "worker_stream_error",
                "code": error_code,
            }
        }
        yield f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
    except (TypeError, ValueError):
        error_code = "worker_protocol_error"
        event = {
            "error": {
                "message": "The inference worker returned an invalid stream event",
                "type": "worker_stream_error",
                "code": error_code,
            }
        }
        yield f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
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
        actual = prompt + completion if usage_seen else prompt_estimate + observed_completion
        if request.app.state.settings.quota_charge_requested_maximum and error_code is None:
            actual = lease.estimated_tokens
        settled_prompt = prompt if usage_seen else prompt_estimate
        settled_completion = completion if usage_seen else observed_completion
        await finalizer.finish(
            actual_tokens=actual,
            prompt_tokens=settled_prompt,
            completion_tokens=settled_completion,
            error_code=error_code,
        )
