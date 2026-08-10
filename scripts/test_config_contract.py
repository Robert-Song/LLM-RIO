from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import llm_rio.cli as cli_module
from llm_rio.api.routes_admin import _create_key
from llm_rio.api.routes_inference import _rough_tokens, _validate_request
from llm_rio.api.schemas import ChatCompletionRequest, CreateKeyRequest
from llm_rio.config import EngineSettings, Settings
from llm_rio.domain import Role
from llm_rio.errors import QueueFullError
from llm_rio.queueing import DeficitRoundRobinQueue, QueuedRequest


def permissive_settings(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(config_file=tmp_path / "missing.toml", **overrides)


def request_for(settings: Settings) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))


def model_with_context(
    max_context_tokens: int,
    *,
    stored_output_policy: int | None = None,
    stored_n_policy: int | None = None,
) -> dict[str, Any]:
    return {
        "request_limits": {
            "max_context_tokens": max_context_tokens,
            "max_output_tokens": stored_output_policy,
            "max_n": stored_n_policy,
        },
        "capabilities": ["chat", "streaming"],
    }


def queued_request(index: int, tenant: str = "tenant") -> QueuedRequest:
    return QueuedRequest(
        id=f"request-{index}",
        model_id="model",
        tenant_id=tenant,
        estimated_tokens=1,
        payload={},
        reservation_id=f"reservation-{index}",
    )


def test_builtin_settings_leave_policy_limits_unset(tmp_path: Path) -> None:
    settings = permissive_settings(tmp_path)

    assert settings.queue_capacity_per_model is None
    assert settings.queue_capacity_per_tenant is None
    assert settings.worker_startup_timeout_seconds is None
    assert settings.worker_drain_watchdog_seconds is None
    assert settings.validation_idle_window_seconds == 0
    assert settings.minimum_residency_seconds == 0
    assert settings.max_prompt_tokens is None
    assert settings.max_output_tokens is None
    assert settings.max_n is None
    assert settings.engines.gpu_memory_utilization is None
    assert settings.engines.max_model_len is None
    assert settings.engines.max_num_seqs is None
    assert settings.engines.max_num_batched_tokens is None


def test_single_example_mentions_every_public_setting(tmp_path: Path) -> None:
    example_path = Path("config.example.toml")
    text = example_path.read_text(encoding="utf-8")
    settings = Settings(config_file=example_path)

    assert example_path.exists()
    assert not Path("config.required.toml").exists()
    assert settings.queue_capacity_per_model is None
    assert settings.queue_capacity_per_tenant is None
    assert settings.worker_startup_timeout_seconds is None
    assert settings.worker_drain_watchdog_seconds is None
    for name in Settings.model_fields:
        if name not in {"config_file", "hf_token", "engines"}:
            assert name in text
    for name in EngineSettings.model_fields:
        assert name in text


def test_unbounded_queue_accepts_work_beyond_old_defaults() -> None:
    queue = DeficitRoundRobinQueue(total_capacity=None, tenant_capacity=None)

    for index in range(2_000):
        queue.put(queued_request(index))

    assert len(queue) == 2_000


def test_configured_queue_capacity_still_rejects_excess_work() -> None:
    queue = DeficitRoundRobinQueue(total_capacity=1, tenant_capacity=1)
    queue.put(queued_request(0))

    with pytest.raises(QueueFullError):
        queue.put(queued_request(1))


def test_cli_key_creation_is_unlimited_unless_limit_is_supplied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads: list[dict[str, Any]] = []

    def fake_request(method: str, path: str, *, json_body: dict[str, Any]) -> dict[str, str]:
        assert (method, path) == ("POST", "/admin/keys")
        payloads.append(json_body)
        return {"nickname": "researcher", "api_key": "rio_test"}

    monkeypatch.setattr(cli_module, "_request", fake_request)
    runner = CliRunner()

    unlimited_result = runner.invoke(cli_module.app, ["keys", "create", "researcher"])
    limited_result = runner.invoke(
        cli_module.app, ["keys", "create", "student", "--limit", "100000"]
    )

    assert unlimited_result.exit_code == 0
    assert limited_result.exit_code == 0
    assert payloads[0]["limit_tokens"] is None
    assert "unlimited" not in payloads[0]
    assert payloads[1]["limit_tokens"] == 100_000


def test_omitted_output_limit_has_no_implicit_1024_token_cap(tmp_path: Path) -> None:
    settings = permissive_settings(tmp_path)
    body = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "x" * 800}],
    )
    prompt_tokens = _rough_tokens(body.messages)
    context_tokens = prompt_tokens + 16

    prompt, reservation, enforced = _validate_request(
        request_for(settings),
        body,
        model_with_context(context_tokens),
    )

    assert body.output_limit is None
    assert prompt == prompt_tokens
    assert reservation > prompt
    assert enforced is None


def test_registration_time_policy_snapshots_do_not_restrict_runtime(
    tmp_path: Path,
) -> None:
    settings = permissive_settings(tmp_path)
    body = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=128,
        n=2,
    )

    _, _, enforced = _validate_request(
        request_for(settings),
        body,
        model_with_context(
            4096,
            stored_output_policy=64,
            stored_n_policy=1,
        ),
    )

    assert enforced is None


def test_configured_output_limit_restricts_omitted_client_limit(tmp_path: Path) -> None:
    settings = permissive_settings(tmp_path, max_output_tokens=64)
    body = ChatCompletionRequest(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
    )

    _, _, enforced = _validate_request(
        request_for(settings),
        body,
        model_with_context(4096),
    )

    assert enforced == 64


class RecordingKeyDatabase:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create_key(self, **kwargs: Any) -> None:
        self.created.append(kwargs)


@pytest.mark.asyncio
async def test_new_key_without_quota_policy_is_unlimited() -> None:
    database = RecordingKeyDatabase()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(database=database)))

    await _create_key(
        request,
        CreateKeyRequest(nickname="researcher", role=Role.USER),
    )

    assert database.created[0]["limit_tokens"] == 0
    assert database.created[0]["unlimited"] is True


@pytest.mark.asyncio
async def test_explicit_key_limit_enables_quota_restriction() -> None:
    database = RecordingKeyDatabase()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(database=database)))

    await _create_key(
        request,
        CreateKeyRequest(nickname="student", role=Role.USER, limit_tokens=100_000),
    )

    assert database.created[0]["limit_tokens"] == 100_000
    assert database.created[0]["unlimited"] is False
