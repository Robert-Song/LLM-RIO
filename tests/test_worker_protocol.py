from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from llm_rio.api.inference_proxy import _nonstream_backend, _stream_backend
from llm_rio.errors import RioError
from llm_rio.worker_protocol import SSEDecoder, completion_choices, token_usage


def test_sse_decoder_preserves_split_unicode_multiline_and_optional_space() -> None:
    raw = 'data:{"choices":[],\r\ndata: "text":"你好🌎"}\r\n\r\ndata:[DONE]\n\n'.encode()
    decoder = SSEDecoder()
    events = []
    for byte in raw:
        events.extend(decoder.feed(bytes([byte])))
    assert json.loads(events[0])["text"] == "你好🌎"
    assert events[1] == "[DONE]"


@pytest.mark.parametrize(
    "usage",
    [[], "bad", {"prompt_tokens": -1}, {"completion_tokens": True}, {"total_tokens": "bad"}],
)
def test_invalid_usage_is_rejected(usage: Any) -> None:
    with pytest.raises(ValueError):
        token_usage({"usage": usage})


@pytest.mark.parametrize(
    "event", [[], None, 1, {"choices": None}, {"choices": [None]}, {"choices": [{"message": []}]}]
)
def test_invalid_completion_shapes_are_rejected(event: Any) -> None:
    with pytest.raises(ValueError):
        completion_choices(event, streaming=False)


class ResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def context():
    settlements = []
    released = []

    async def settle(**kwargs):
        settlements.append(kwargs)

    async def release(lease):
        released.append(lease)

    async def touch(lease):
        pass

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                database=SimpleNamespace(settle_quota=settle),
                scheduler=SimpleNamespace(release=release, touch=touch),
                settings=SimpleNamespace(
                    worker_request_timeout_seconds=None,
                    worker_stream_idle_timeout_seconds=None,
                    quota_charge_requested_maximum=True,
                ),
            )
        ),
        state=SimpleNamespace(admission_time=datetime.now(UTC)),
    )
    lease = SimpleNamespace(
        request_id="request",
        worker_id="worker",
        admitted_at=datetime.now(UTC),
        estimated_tokens=100,
        base_url="http://worker.invalid",
        internal_api_key="test",
    )
    return request, lease, settlements, released


@pytest.mark.parametrize(
    "body", [[], {"choices": None}, {"choices": [{"message": {}}], "usage": {"prompt_tokens": -1}}]
)
async def test_bad_nonstream_response_fails_without_charging_maximum(context, body) -> None:
    request, lease, settlements, released = context
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        request.app.state.worker_client = client
        with pytest.raises(RioError) as error:
            await _nonstream_backend(
                request=request,
                payload={"model": "model"},
                lease=lease,
                reservation_id="reservation",
                prompt_estimate=10,
                queue_wait_milliseconds=0,
            )
    assert error.value.code == "worker_protocol_error"
    assert settlements[0]["error_code"] == "worker_protocol_error"
    assert settlements[0]["actual_tokens"] == 0
    assert released == [lease]


async def test_stream_preserves_bytes_and_authoritative_zero_usage(context) -> None:
    request, lease, settlements, released = context
    request.app.state.settings.quota_charge_requested_maximum = False
    event = {"choices": [{"delta": {"content": "你好🌎"}}]}
    usage = {
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    raw = (
        "data:"
        + json.dumps(event, ensure_ascii=False)
        + "\n\ndata:"
        + json.dumps(usage)
        + "\n\ndata:[DONE]\n\n"
    ).encode()
    response = httpx.Response(200, stream=ResponseStream([bytes([b]) for b in raw]))
    output = [
        chunk
        async for chunk in _stream_backend(
            request=request,
            payload={"model": "model"},
            lease=lease,
            reservation_id="reservation",
            prompt_estimate=10,
            response=response,
        )
    ]
    assert b"".join(output) == raw
    assert settlements[0]["actual_tokens"] == 0
    assert settlements[0]["prompt_tokens"] == 0
    assert settlements[0]["completion_tokens"] == 0
    assert settlements[0]["error_code"] is None
    assert response.is_closed
    assert released == [lease]


@pytest.mark.parametrize(
    "raw",
    [b"data: []\n\n", b"data: {invalid}\n\n", b'data: {"usage":{"completion_tokens":-1}}\n\n'],
)
async def test_malformed_stream_is_failed_and_finalized(context, raw) -> None:
    request, lease, settlements, released = context
    response = httpx.Response(200, stream=ResponseStream([raw]))
    output = [
        chunk
        async for chunk in _stream_backend(
            request=request,
            payload={"model": "model"},
            lease=lease,
            reservation_id="reservation",
            prompt_estimate=10,
            response=response,
        )
    ]
    assert b"worker_protocol_error" in b"".join(output)
    assert settlements[0]["error_code"] == "worker_protocol_error"
    assert settlements[0]["actual_tokens"] != lease.estimated_tokens
    assert response.is_closed
    assert released == [lease]
