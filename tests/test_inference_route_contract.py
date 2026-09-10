from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_scheduler_contract import GPU_0, make_profile

from llm_rio.api.inference_proxy import (
    _open_worker_stream,
    _stream_backend,
    _StreamLeaseFinalizer,
)
from llm_rio.api.inference_validation import _apply_model_defaults, _validate_request
from llm_rio.api.routes_inference import _resolve_model, _routable_profiles
from llm_rio.api.schemas import ChatCompletionRequest
from llm_rio.config import Settings
from llm_rio.domain import CatalogState, Role
from llm_rio.errors import RioError
from llm_rio.security import Principal


def prism_request(*profiles: object) -> SimpleNamespace:
    class Database:
        async def list_models(self) -> list[dict[str, Any]]:
            return [
                {
                    "id": "model-id",
                    "nickname": "model",
                    "state": CatalogState.AVAILABLE.value,
                }
            ]

        async def model_by_nickname(self, nickname: str) -> dict[str, Any] | None:
            assert nickname == "model"
            return (await self.list_models())[0]

    class Profiles:
        async def for_model(self, model_id: str) -> list[object]:
            assert model_id == "model-id"
            return list(profiles)

    state = SimpleNamespace(
        database=Database(),
        profiles=Profiles(),
        scheduler=SimpleNamespace(kvcached=SimpleNamespace(enabled=True)),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


def test_prism_routes_only_profiles_validated_with_kvcached() -> None:
    native = make_profile("native", "model-id", (GPU_0,))
    cached = make_profile("cached", "model-id", (GPU_0,), memory_backend="kvcached")
    request = prism_request(native, cached)

    assert _routable_profiles(request, [native, cached]) == [cached]  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_prism_rejects_native_profile_before_queueing() -> None:
    native = make_profile("native", "model-id", (GPU_0,))
    principal = Principal("key", "admin", Role.ADMIN, "account", True)

    with pytest.raises(RioError) as captured:
        await _resolve_model(prism_request(native), principal, "model")

    assert captured.value.code == "prism_profile_revalidation_required"
    assert captured.value.status_code == 503


def test_model_defaults_fill_omitted_fields_and_explicit_request_wins() -> None:
    model = {
        "request_defaults": {
            "temperature": 0.7,
            "top_p": 0.85,
            "top_k": 40,
            "reasoning_effort": "medium",
        }
    }
    omitted = ChatCompletionRequest(
        model="qwen",
        messages=[{"role": "user", "content": "test"}],
    )
    explicit = ChatCompletionRequest(
        model="qwen",
        messages=[{"role": "user", "content": "test"}],
        temperature=0,
        top_p=0.4,
        top_k=0,
        reasoning_effort="none",
    )

    effective = _apply_model_defaults(omitted, model)
    unchanged = _apply_model_defaults(explicit, model)

    assert effective.temperature == 0.7
    assert effective.top_p == 0.85
    assert effective.top_k == 40
    assert effective.reasoning_effort == "medium"
    assert unchanged.temperature == 0
    assert unchanged.top_p == 0.4
    assert unchanged.top_k == 0
    assert unchanged.reasoning_effort == "none"


class CancellingWorkerClient:
    def build_request(self, *args: Any, **kwargs: Any) -> object:
        return object()

    async def send(self, request: object, *, stream: bool) -> None:
        assert stream
        raise asyncio.CancelledError


class RecordingDatabase:
    def __init__(self) -> None:
        self.settlements: list[dict[str, Any]] = []

    async def settle_quota(self, **kwargs: Any) -> None:
        self.settlements.append(kwargs)


class RecordingScheduler:
    def __init__(self) -> None:
        self.released: list[object] = []
        self.touched: list[object] = []

    async def release(self, lease: object) -> None:
        self.released.append(lease)

    async def touch(self, lease: object) -> None:
        self.touched.append(lease)


def test_tool_declarations_are_not_blocked_by_stale_catalog_capabilities(
    tmp_path: Path,
) -> None:
    settings = Settings(config_file=tmp_path / "missing.toml")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    body = ChatCompletionRequest(
        model="gemma",
        messages=[{"role": "user", "content": "test"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a test value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    model = {
        "request_limits": {"max_context_tokens": 4096},
        "capabilities": ["chat", "streaming"],
    }

    _, _, enforced_output_limit = _validate_request(request, body, model)

    assert enforced_output_limit is None
    assert body.tools and body.tools[0]["function"]["name"] == "lookup"


def test_image_input_is_forwardable_without_catalog_vision_capability(tmp_path: Path) -> None:
    settings = Settings(config_file=tmp_path / "missing.toml")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    image_url = "data:image/png;base64," + "A" * 100_000
    body = ChatCompletionRequest(
        model="gemma",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=32,
    )
    model = {
        "request_limits": {"max_context_tokens": 4096},
        "capabilities": ["chat", "streaming"],
    }

    prompt_tokens, reservation_tokens, enforced_output_limit = _validate_request(
        request, body, model
    )

    assert prompt_tokens < 100
    assert reservation_tokens < 256
    assert enforced_output_limit is None
    assert body.model_dump(exclude_none=True)["messages"][0]["content"][1]["image_url"] == {
        "url": image_url
    }


def test_image_bytes_do_not_change_prompt_or_quota_estimates(tmp_path: Path) -> None:
    settings = Settings(config_file=tmp_path / "missing.toml")
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))
    model = {
        "request_limits": {"max_context_tokens": 4096},
        "capabilities": ["chat", "streaming"],
    }

    def estimates(encoded_image: str) -> tuple[int, int, int | None]:
        body = ChatCompletionRequest(
            model="qwen",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is shown?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                        },
                    ],
                }
            ],
            max_tokens=16,
        )
        return _validate_request(request, body, model)

    assert estimates("AAAA") == estimates("A" * 100_000)


@pytest.mark.asyncio
async def test_disconnect_before_stream_headers_settles_and_releases_lease() -> None:
    database = RecordingDatabase()
    scheduler = RecordingScheduler()
    app_state = SimpleNamespace(
        worker_client=CancellingWorkerClient(),
        database=database,
        scheduler=scheduler,
        settings=SimpleNamespace(worker_request_timeout_seconds=None),
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=app_state),
        state=SimpleNamespace(
            admission_time=datetime.now(UTC),
            test_run_id="cancel-test",
        ),
    )
    lease = SimpleNamespace(
        request_id="request-id",
        worker_id="worker-id",
        admitted_at=datetime.now(UTC),
        internal_api_key="internal-key",
        base_url="http://worker.invalid",
    )

    with pytest.raises(asyncio.CancelledError):
        await _open_worker_stream(
            request=request,
            payload={"model": "logical-model"},
            lease=lease,
            reservation_id="reservation-id",
        )

    assert database.settlements == [
        {
            "reservation_id": "reservation-id",
            "actual_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error_code": "client_disconnected",
        }
    ]
    assert scheduler.released == [lease]


class HangingStreamResponse:
    def __init__(self) -> None:
        self.closed = False

    async def aiter_bytes(self):
        await asyncio.Event().wait()
        yield b""

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_idle_worker_stream_settles_and_releases_lease(tmp_path: Path) -> None:
    database = RecordingDatabase()
    scheduler = RecordingScheduler()
    settings = Settings(
        config_file=tmp_path / "missing.toml",
        worker_stream_idle_timeout_seconds=0.01,
    )
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(database=database, scheduler=scheduler, settings=settings)
        ),
        state=SimpleNamespace(admission_time=datetime.now(UTC), test_run_id="timeout-test"),
    )
    lease = SimpleNamespace(
        request_id="request-id",
        worker_id="worker-id",
        admitted_at=datetime.now(UTC),
        estimated_tokens=32,
    )
    response = HangingStreamResponse()

    chunks = [
        chunk
        async for chunk in _stream_backend(
            request=request,
            payload={"model": "logical-model"},
            lease=lease,
            reservation_id="reservation-id",
            prompt_estimate=3,
            response=response,  # type: ignore[arg-type]
        )
    ]

    assert b"worker_stream_idle_timeout" in b"".join(chunks)
    assert database.settlements == [
        {
            "reservation_id": "reservation-id",
            "actual_tokens": 3,
            "prompt_tokens": 3,
            "completion_tokens": 0,
            "error_code": "worker_stream_idle_timeout",
        }
    ]
    assert scheduler.released == [lease]
    assert response.closed


@pytest.mark.asyncio
async def test_unstarted_worker_stream_is_released_by_response_background_cleanup() -> None:
    database = RecordingDatabase()
    scheduler = RecordingScheduler()
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                database=database,
                scheduler=scheduler,
                settings=SimpleNamespace(quota_charge_requested_maximum=False),
            )
        ),
        state=SimpleNamespace(admission_time=datetime.now(UTC), test_run_id="abandoned-stream"),
    )
    lease = SimpleNamespace(
        request_id="request-id",
        worker_id="worker-id",
        admitted_at=datetime.now(UTC),
        estimated_tokens=32,
    )
    response = HangingStreamResponse()
    finalizer = _StreamLeaseFinalizer(
        request=request,
        payload={"model": "logical-model"},
        lease=lease,
        reservation_id="reservation-id",
        response=response,  # type: ignore[arg-type]
    )

    await finalizer.abandon()
    await finalizer.finish(
        actual_tokens=5,
        prompt_tokens=2,
        completion_tokens=3,
        error_code=None,
    )

    assert database.settlements == [
        {
            "reservation_id": "reservation-id",
            "actual_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error_code": "client_disconnected",
        }
    ]
    assert scheduler.released == [lease]
    assert response.closed
