"""Tests for settings loading, validation, and error types."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_rio.config import Settings
from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.errors import (
    AuthenticationError,
    AuthorizationError,
    MaintenanceError,
    QueueFullError,
    QuotaExceededError,
    RioError,
)
from llm_rio.profiles import profile_from_dict, profile_to_dict


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings()
        assert settings.machine_id == "local"
        assert settings.api_host == "127.0.0.1"
        assert settings.api_port == 8000
        assert settings.queue_capacity_per_model == 1024
        assert settings.worker_port_start == 18000
        assert settings.worker_port_end == 18999

    def test_toml_loading(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            'machine_id = "hpc-01"\napi_port = 9000\nreserved_vram_mib = 4096\n'
        )
        settings = Settings(config_file=config)
        assert settings.machine_id == "hpc-01"
        assert settings.api_port == 9000
        assert settings.reserved_vram_mib == 4096

    def test_environment_overrides_toml(self, tmp_path: Path, monkeypatch) -> None:
        config = tmp_path / "config.toml"
        config.write_text('api_port = 9000\n')
        monkeypatch.setenv("LLMRIO_API_PORT", "7777")
        settings = Settings(config_file=config)
        assert settings.api_port == 7777

    def test_invalid_port_range_raises(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("worker_port_start = 20000\nworker_port_end = 10000\n")
        with pytest.raises(ValidationError, match="worker_port_start"):
            Settings(config_file=config)

    def test_negative_durations_raise(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("fair_share_seconds = -5\n")
        with pytest.raises(ValidationError):
            Settings(config_file=config)

    def test_negative_reserved_vram_raises(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text("reserved_vram_mib = -1\n")
        with pytest.raises(ValidationError, match="reserved_vram_mib"):
            Settings(config_file=config)

    def test_engine_settings_defaults(self) -> None:
        settings = Settings()
        assert settings.engines.vllm_executable == "vllm"
        assert settings.engines.llama_cpp_executable == "llama-server"
        assert settings.engines.enable_llama_cpp is False

    def test_ensure_directories_creates_paths(self, tmp_path: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            f'database_path = "{tmp_path / "nested" / "state" / "db.sqlite"}"\n'
            f'model_store = "{tmp_path / "nested" / "models"}"\n'
            f'log_dir = "{tmp_path / "nested" / "logs"}"\n'
        )
        settings = Settings(config_file=config)
        settings.ensure_directories()
        assert (tmp_path / "nested" / "state").is_dir()
        assert (tmp_path / "nested" / "models").is_dir()
        assert (tmp_path / "nested" / "logs").is_dir()


class TestRioErrors:
    def test_openai_body(self) -> None:
        error = RioError(
            "custom_code", "Something failed", status_code=418, details={"x": 1}
        )
        assert error.openai_body() == {
            "error": {
                "message": "Something failed",
                "type": "custom_code",
                "code": "custom_code",
                "x": 1,
            }
        }

    def test_authentication_error(self) -> None:
        error = AuthenticationError()
        assert error.status_code == 401
        assert error.code == "invalid_api_key"

    def test_authorization_error(self) -> None:
        error = AuthorizationError()
        assert error.status_code == 403
        assert error.code == "permission_denied"

    def test_quota_exceeded_error(self) -> None:
        error = QuotaExceededError(available=10, requested=100)
        assert error.status_code == 429
        assert error.code == "quota_exceeded"
        assert error.details == {"available_tokens": 10, "requested_tokens": 100}

    def test_queue_full_error(self) -> None:
        error = QueueFullError()
        assert error.status_code == 429
        assert error.code == "queue_full"

    def test_maintenance_error(self) -> None:
        error = MaintenanceError()
        assert error.status_code == 503
        assert error.code == "service_maintenance"


class TestWorkerPlacement:
    def test_defaults(self) -> None:
        profile = PlacementProfile(
            id="p1",
            model_id="m1",
            model_revision="rev1",
            engine=Engine.VLLM,
            engine_version="0.26.0",
            machine_fingerprint="fp",
            gpu_count=1,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            eligible_gpu_sets=(("GPU-a",),),
            dtype="half",
            quantization=None,
            max_model_len=4096,
            max_num_seqs=None,
            max_num_batched_tokens=None,
            predicted_tokens_per_second=100.0,
            load_and_warmup_seconds=30.0,
            idle_vram_mib_per_gpu=(1000,),
            peak_vram_mib_per_gpu=(8000,),
            gpu_headroom_mib_per_gpu=(2000,),
            capabilities=frozenset({"chat"}),
            launch_args={},
            gpu_memory_utilization=0.9,
            kv_cache_capacity_tokens=None,
            max_full_length_concurrency=None,
        )
        worker = WorkerPlacement(id="w1", profile=profile, gpu_uuids=("GPU-a",), port=18000)
        assert worker.state is RuntimeState.LOADING
        assert worker.model_id == "m1"
        assert worker.is_routable is False
        worker.state = RuntimeState.READY
        assert worker.is_routable is True


class TestProfileSerialization:
    def _profile(self) -> PlacementProfile:
        return PlacementProfile(
            id="p1",
            model_id="m1",
            model_revision="rev1",
            engine=Engine.VLLM,
            engine_version="0.26.0",
            machine_fingerprint="fp",
            gpu_count=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            eligible_gpu_sets=(("GPU-a", "GPU-b"),),
            dtype="bfloat16",
            quantization="awq",
            max_model_len=8192,
            max_num_seqs=32,
            max_num_batched_tokens=None,
            predicted_tokens_per_second=250.5,
            load_and_warmup_seconds=45.0,
            idle_vram_mib_per_gpu=(2048, 2048),
            peak_vram_mib_per_gpu=(9000, 9000),
            gpu_headroom_mib_per_gpu=(2048, 2048),
            capabilities=frozenset({"chat", "streaming"}),
            launch_args={"max_model_len": 8192},
            gpu_memory_utilization=0.9,
            kv_cache_capacity_tokens=50000,
            max_full_length_concurrency=4.0,
        )

    def test_round_trip(self) -> None:
        profile = self._profile()
        restored = profile_from_dict(profile_to_dict(profile))
        assert restored == profile

    def test_optional_fields_survive_round_trip(self) -> None:
        profile = self._profile()
        restored = profile_from_dict(profile_to_dict(profile))
        assert restored.max_num_batched_tokens is None
        assert restored.max_full_length_concurrency == 4.0
        assert restored.quantization == "awq"
        assert restored.eligible_gpu_sets == (("GPU-a", "GPU-b"),)
