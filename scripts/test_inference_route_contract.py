from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rio.api.routes_inference import _open_worker_stream, _validate_request
from llm_rio.api.schemas import ChatCompletionRequest
from llm_rio.config import Settings


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

    async def release(self, lease: object) -> None:
        self.released.append(lease)


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
