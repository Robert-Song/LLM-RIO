from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import llm_rio.cli as cli_module
from llm_rio import admin_client as client_api
from llm_rio.api.inference_validation import _rough_tokens, _validate_request
from llm_rio.api.schemas import ChatCompletionRequest, CreateKeyRequest
from llm_rio.domain import Role
from llm_rio.errors import QueueFullError
from llm_rio.queueing import DeficitRoundRobinQueue, QueuedRequest
from llm_rio.services.access import create_key as _create_key
from tests.release_fixtures import Settings


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
    assert settings.worker_request_timeout_seconds is None
    assert settings.worker_stream_idle_timeout_seconds is None
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
    example_path = Path(__file__).resolve().parents[1] / "config.example.toml"
    text = example_path.read_text(encoding="utf-8")
    settings = Settings(config_file=example_path)

    assert example_path.exists()
    assert not Path("config.required.toml").exists()
    assert settings.queue_capacity_per_model is None
    assert settings.queue_capacity_per_tenant is None
    assert settings.worker_startup_timeout_seconds is None
    assert settings.worker_drain_watchdog_seconds is None
    assert settings.worker_request_timeout_seconds is None
    assert settings.worker_stream_idle_timeout_seconds is None
    for name in __import__("llm_rio.config", fromlist=["Settings"]).Settings.model_fields:
        if name not in {"config_file", "hf_token", "engines"}:
            assert name in text
    for name in __import__(
        "llm_rio.config", fromlist=["EngineSettings"]
    ).EngineSettings.model_fields:
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

    monkeypatch.setattr(client_api, "request", fake_request)
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

    await _create_key(
        database,
        CreateKeyRequest(nickname="researcher", role=Role.USER),
    )

    assert database.created[0]["limit_tokens"] == 0
    assert database.created[0]["unlimited"] is True


@pytest.mark.asyncio
async def test_explicit_key_limit_enables_quota_restriction() -> None:
    database = RecordingKeyDatabase()

    await _create_key(
        database,
        CreateKeyRequest(nickname="student", role=Role.USER, limit_tokens=100_000),
    )

    assert database.created[0]["limit_tokens"] == 100_000
    assert database.created[0]["unlimited"] is False


def test_environment_selects_configuration_file(tmp_path, monkeypatch) -> None:
    selected = tmp_path / "selected.toml"
    selected.write_text('machine_id = "selected"\napi_port = 8123\n')
    monkeypatch.setenv("LLMRIO_CONFIG_FILE", str(selected))
    monkeypatch.setenv("LLMRIO_API_PORT", "8124")
    settings = Settings()
    assert settings.machine_id == "selected"
    assert settings.api_port == 8124
    assert settings.config_file == selected


def test_explicit_config_selector_overrides_environment(tmp_path, monkeypatch) -> None:
    environment = tmp_path / "environment.toml"
    environment.write_text('machine_id = "environment"\n')
    explicit = tmp_path / "explicit.toml"
    explicit.write_text('machine_id = "explicit"\n')
    monkeypatch.setenv("LLMRIO_CONFIG_FILE", str(environment))
    assert Settings(config_file=explicit).machine_id == "explicit"


def test_dotenv_selects_configuration_file(tmp_path) -> None:
    selected = tmp_path / "selected.toml"
    selected.write_text('machine_id = "dotenv"\n')
    (tmp_path / ".env").write_text(f"LLMRIO_CONFIG_FILE={selected}\n")
    assert Settings().machine_id == "dotenv"


@pytest.mark.parametrize("command", [["serve"], ["doctor"], ["status"]])
def test_cli_reports_beta_configuration_without_traceback_or_secrets(
    tmp_path, monkeypatch, command
) -> None:
    monkeypatch.delenv("LLMRIO_SERVING_MODE")
    (tmp_path / "config.toml").write_text(
        'prism_weight_cache_mode = "ram"\nhf_token = "never-display-this-token"\n'
    )
    result = CliRunner().invoke(cli_module.app, command)
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output
    assert "serving_mode" in result.output
    assert "Beta settings are unsupported" in result.output
    assert "Traceback" not in result.output
    assert "never-display-this-token" not in result.output
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("selector", ["cli", "environment", "dotenv"])
def test_cli_rejects_missing_explicit_configuration(tmp_path, monkeypatch, selector) -> None:
    path = tmp_path / "typo.toml"
    args = ["serve"]
    if selector == "cli":
        args += ["--config", str(path)]
    elif selector == "environment":
        monkeypatch.setenv("LLMRIO_CONFIG_FILE", str(path))
    else:
        (tmp_path / ".env").write_text(f"LLMRIO_CONFIG_FILE={path}\n")
    result = CliRunner().invoke(cli_module.app, args)
    assert result.exit_code == 1
    assert "Configuration file not found" in result.output
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("selector", ["cli", "environment", "dotenv"])
def test_serve_resolves_release_configuration_over_beta_default(
    tmp_path, monkeypatch, selector
) -> None:
    monkeypatch.delenv("LLMRIO_SERVING_MODE")
    (tmp_path / "config.toml").write_text('prism_weight_cache_mode = "ram"\n')
    path = tmp_path / "release.toml"
    path.write_text('serving_mode = "queue"\napi_port = 8123\n')
    served = []
    monkeypatch.setattr(cli_module, "create_app", lambda settings: settings)
    monkeypatch.setattr(cli_module.uvicorn, "run", lambda app, **kwargs: served.append(app))
    args = ["serve"]
    if selector == "cli":
        args += ["--config", str(path)]
    elif selector == "environment":
        monkeypatch.setenv("LLMRIO_CONFIG_FILE", str(path))
    else:
        (tmp_path / ".env").write_text(f"LLMRIO_CONFIG_FILE={path}\n")
    result = CliRunner().invoke(cli_module.app, args)
    assert result.exit_code == 0, result.output
    assert len(served) == 1
    assert served[0].config_file == path
    assert served[0].serving_mode.value == "queue"
    assert served[0].api_port == 8123


@pytest.mark.parametrize(
    "content, env",
    [
        ('hf_token = "private"\n', {}),
        ('serving_mode = "queue"\n[broken', {}),
        ('serving_mode = "queue"\n', {"LLMRIO_MANAGED_GPU_UUIDS": "invalid-json"}),
    ],
)
def test_cli_reports_missing_mode_malformed_toml_and_environment(
    tmp_path, monkeypatch, content, env
) -> None:
    monkeypatch.delenv("LLMRIO_SERVING_MODE")
    (tmp_path / "config.toml").write_text(content)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    result = CliRunner().invoke(cli_module.app, ["serve"])
    assert result.exit_code == 1
    assert "configuration" in result.output.lower()
    assert "Traceback" not in result.output
    assert "private" not in result.output


@pytest.mark.parametrize("selector", ["file", "cli", "environment", "dotenv"])
@pytest.mark.parametrize("mode", ["queue", "vllm-sleep"])
def test_mode_selection_with_both_native_configuration_sections(
    tmp_path, monkeypatch, selector, mode
) -> None:
    monkeypatch.delenv("LLMRIO_SERVING_MODE")
    other = "vllm-sleep" if mode == "queue" else "queue"
    config = tmp_path / "config.toml"
    config.write_text(
        f'serving_mode = "{mode if selector == "file" else other}"\n'
        "[modes.queue]\nscale_window_seconds = 12\n"
        "[modes.vllm_sleep]\nidle_sleep_seconds = 17\n"
        'preload_models = ["saved-sleep-model"]\nhost_cache_max_gib = 100\n'
    )
    original = config.read_bytes()
    if selector == "environment":
        monkeypatch.setenv("LLMRIO_SERVING_MODE", mode)
    elif selector == "dotenv":
        (tmp_path / ".env").write_text(f"LLMRIO_SERVING_MODE={mode}\n")
    args = ["serve", "--mode", mode] if selector == "cli" else ["serve"]
    served = []
    monkeypatch.setattr(cli_module, "create_app", lambda settings: settings)
    monkeypatch.setattr(cli_module.uvicorn, "run", lambda app, **kwargs: served.append(app))
    result = CliRunner().invoke(cli_module.app, args)
    assert result.exit_code == 0, result.output
    assert len(served) == 1
    settings = served[0]
    assert settings.serving_mode.value == mode
    assert settings.modes.queue.scale_window_seconds == 12
    assert settings.modes.vllm_sleep.idle_sleep_seconds == 17
    if mode == "queue":
        assert settings.residency.preload_models == []
        assert settings.residency.host_cache_max_gib is None
        assert not settings.ram_weight_cache_enabled
    else:
        assert settings.residency.preload_models == ["saved-sleep-model"]
        assert settings.residency.host_cache_max_gib == 100
    assert config.read_bytes() == original


@pytest.mark.parametrize(
    "extra",
    [
        "[modes.vllm_sleep]\nmisspelled_option = 5\n",
        "[modes.vllm_sleep]\nidle_sleep_seconds = -1\n",
        "[modes.unknown]\nsetting = 1\n",
    ],
)
def test_dormant_mode_sections_still_reject_unknown_and_invalid_settings(
    tmp_path, monkeypatch, extra
) -> None:
    monkeypatch.delenv("LLMRIO_SERVING_MODE")
    (tmp_path / "config.toml").write_text('serving_mode = "queue"\n' + extra)
    result = CliRunner().invoke(cli_module.app, ["serve"])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output
    assert not (tmp_path / "state").exists()
