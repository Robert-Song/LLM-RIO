from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from llm_rio import cli as cli_api


def test_models_add_wait_prints_automatic_validation_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str, dict[str, Any] | None]] = []
    jobs = iter(
        [
            {
                "id": "job-1",
                "nickname": "qwen3-8b",
                "state": "RUNNING",
                "catalog_state": "DOWNLOADING",
                "stage": "download",
            },
            {
                "id": "job-1",
                "nickname": "qwen3-8b",
                "state": "RUNNING",
                "catalog_state": "VALIDATING",
                "stage": "validating",
            },
            {
                "id": "job-1",
                "nickname": "qwen3-8b",
                "state": "COMPLETED",
                "catalog_state": "AVAILABLE",
                "stage": "complete",
            },
        ]
    )

    def request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        requests.append((method, path, json_body))
        if (method, path) == ("POST", "/staff/models"):
            return {"model_id": "model-1", "job_id": "job-1"}
        if (method, path) == ("GET", "/staff/model-jobs/job-1"):
            return next(jobs)
        raise AssertionError((method, path, json_body))

    monkeypatch.setattr(cli_api, "_request", request)
    monkeypatch.setattr(cli_api.time, "sleep", lambda _: None)

    result = CliRunner().invoke(
        cli_api.app,
        ["models", "add", "qwen3-8b", "Qwen/Qwen3-8B", "--wait"],
    )

    assert result.exit_code == 0, result.output
    assert "RUNNING / DOWNLOADING / download" in result.output
    assert "RUNNING / VALIDATING / validating" in result.output
    assert "COMPLETED / AVAILABLE / complete" in result.output
    assert "validated and available" in result.output
    assert requests[0][2] == {
        "nickname": "qwen3-8b",
        "huggingface_repo": "Qwen/Qwen3-8B",
        "revision": None,
        "grant_to_keys": [],
    }
