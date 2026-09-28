"""Synthetic tests of the physical runner's oracles; these are not GPU results."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import AsyncMock

import httpx
import pytest

from scripts.native_acceptance import Gate, authoritative_usage, reconcile


def sample_and_row():
    sample = {
        "request_id": "request-1",
        "model": "model",
        "ok": True,
        "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3},
    }
    row = {
        "request_id": "request-1",
        "model": "model",
        "completion_status": "COMPLETED",
        "error_code": None,
        "accepted_count": 1,
        "completion_count": 1,
        "reservation_state": "SETTLED",
        "reserved_tokens": 20,
        "charged_tokens": 3,
        "ledger_delta_tokens": -3,
        "account_unlimited": 0,
        "token_usage": {"prompt_tokens": 3, "completion_tokens": 0},
    }
    return sample, row


@pytest.mark.parametrize(
    "change",
    [
        {"request_id": "different"},
        {"token_usage": {"prompt_tokens": 4, "completion_tokens": 0}},
        {"completion_count": 2},
        {"charged_tokens": 2},
        {"ledger_delta_tokens": -4},
        {"reservation_state": "RESERVED"},
        {"accepted_count": 0},
    ],
)
def test_accounting_oracle_catches_wrong_identity_usage_and_ledger(change):
    sample, row = sample_and_row()
    assert not reconcile([sample], [row])
    assert reconcile([sample], [{**row, **change}])
    assert reconcile([sample], [row, row])


def test_requested_maximum_charging_requires_matching_ledger():
    sample, row = sample_and_row()
    row.update(charged_tokens=20, ledger_delta_tokens=-20)
    assert not reconcile([sample], [row], charge_maximum=True)
    assert reconcile([sample], [row])


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": True, "completion_tokens": 0, "total_tokens": 1},
        {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 3},
    ],
)
def test_incomplete_usage_is_never_authoritative(usage):
    with pytest.raises(ValueError):
        authoritative_usage({"usage": usage})


@pytest.mark.parametrize("stream,broken", [(False, False), (True, False), (True, True)])
async def test_real_http_request_protocol_records_ids_ttft_and_incomplete_stream(
    tmp_path, stream, broken
):
    def backend(request: httpx.Request):
        rid = request.headers["X-Request-ID"]
        assert request.headers["Idempotency-Key"] == rid
        body = {
            "choices": [{"index": 0, "message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
        headers = {"X-Request-ID": rid, "X-Worker-ID": "worker", "X-Queue-Wait-Ms": "100"}
        if not stream:
            return httpx.Response(200, json=body, headers=headers)
        body["choices"] = [{"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}]
        content = f"data: {json.dumps(body)}\n\n" + ("" if broken else "data: [DONE]\n\n")
        return httpx.Response(200, text=content, headers=headers)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(backend), base_url="http://test"
    ) as client:
        gate = Gate(
            {"mode": "queue", "workload": {"operation_timeout_seconds": 1}}, tmp_path, client
        )
        result = await gate.traffic(client, {"model": "model", "stream": stream}, "contract")
    assert result["ok"] is not broken
    assert result["queue_wait_ms"] == 100
    if stream:
        assert result["ttft_seconds"] >= 0
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


async def test_runner_refuses_wrong_running_application_before_draining(tmp_path):
    import zipfile

    wheel = tmp_path / "app.whl"
    with zipfile.ZipFile(wheel, "w") as target:
        target.writestr("llm_rio/__init__.py", "package = 1\n")
    async with httpx.AsyncClient(base_url="http://test") as client:
        gate = Gate(
            {
                "mode": "queue",
                "workload": {"operation_timeout_seconds": 1},
                "build_files": {"wheel": str(wheel)},
            },
            tmp_path,
            client,
        )
        gate.api = AsyncMock(
            return_value={
                "name": "queue",
                "experimental": False,
                "application": {"source_sha256": "wrong"},
            }
        )
        gate.drain = AsyncMock()
        with pytest.raises(ValueError, match="differs from the recorded wheel"):
            await gate.prepare()
        gate.drain.assert_not_awaited()


def test_usage_results_remain_independent_when_models_and_usage_match():
    sample, row = sample_and_row()
    second = deepcopy(sample)
    second["request_id"] = "request-2"
    swapped = {**row, "request_id": "request-3"}
    assert reconcile([sample, second], [row, swapped])
