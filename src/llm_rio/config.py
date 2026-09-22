from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal
from typing import Literal as _Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from llm_rio.modes.settings import CacheSettings, ModeSettings

KVCachedMode = _Literal["none", "required"]


class ServingMode(str, Enum):
    QUEUE = "queue"
    VLLM_SLEEP = "vllm-sleep"
    KV_CACHED = "kv-cached"


class EngineSettings(BaseModel):
    model_config = {"extra": "forbid"}
    vllm_executable: str = "vllm"
    llama_cpp_executable: str = "llama-server"
    # Allows an administrator to select llama.cpp for an existing profile.
    # It never enables automatic vLLM-to-llama.cpp fallback.
    enable_llama_cpp: bool = False
    environment: dict[str, str] = Field(default_factory=dict)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_model_len: int | None = Field(default=None, gt=0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMRIO_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="forbid",
    )

    config_file: Path = Field(default=Path("config.toml"), exclude=True)
    serving_mode: ServingMode = Field(...)
    machine_id: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8002, ge=1, le=65535)
    database_path: Path = Path("state/release/llm-rio.db")
    model_store: Path = Path("models")
    log_dir: Path = Path("logs")
    managed_gpu_uuids: list[str] = Field(default_factory=list)
    reserved_vram_mib: int = Field(default=2048, ge=0)
    queue_capacity_per_model: int | None = Field(default=None, gt=0)
    queue_capacity_per_tenant: int | None = Field(default=None, gt=0)
    wait_duration_seconds: float = Field(default=5.0, gt=0)
    minimum_residency_seconds: float = Field(default=0.0, ge=0)
    fair_share_seconds: float = Field(default=7200.0, gt=0)
    validation_idle_window_seconds: float = Field(default=0.0, ge=0)
    modes: ModeSettings = Field(default_factory=ModeSettings)
    worker_startup_timeout_seconds: float | None = Field(default=None, gt=0)
    worker_drain_watchdog_seconds: float | None = Field(default=None, gt=0)
    worker_request_timeout_seconds: float | None = Field(default=None, gt=0)
    worker_stream_idle_timeout_seconds: float | None = Field(default=None, gt=0)
    worker_port_start: int = Field(default=18000, ge=1, le=65535)
    worker_port_end: int = Field(default=18999, ge=1, le=65535)
    scheduler_tick_seconds: float = Field(default=1.0, gt=0)
    max_prompt_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_n: int | None = Field(default=None, gt=0)
    quota_charge_requested_maximum: bool = False
    capture_worker_engine_logs: bool = True
    hf_token: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "ERROR"
    engines: EngineSettings = Field(default_factory=EngineSettings)

    @property
    def residency(self) -> CacheSettings:
        if self.serving_mode is ServingMode.KV_CACHED:
            return self.modes.kv_cached
        return self.modes.vllm_sleep

    @property
    def effective_kvcached_mode(self) -> KVCachedMode:
        return "required" if self.serving_mode is ServingMode.KV_CACHED else "none"

    @property
    def ram_weight_cache_enabled(self) -> bool:
        return self.serving_mode is not ServingMode.QUEUE

    @property
    def queue_mode_enabled(self) -> bool:
        return self.serving_mode is ServingMode.QUEUE

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Resolve the selector with the same precedence as other settings.
        config_path = Path("config.toml")
        for source in (init_settings, env_settings, dotenv_settings):
            selected = source().get("config_file")
            if selected is not None:
                config_path = Path(selected)
                break
        toml_source = TomlConfigSettingsSource(settings_cls, toml_file=config_path)
        return init_settings, env_settings, dotenv_settings, toml_source, file_secret_settings

    @model_validator(mode="after")
    def validate_settings(self) -> Settings:
        if self.worker_port_start > self.worker_port_end:
            raise ValueError("worker_port_start must not exceed worker_port_end")
        return self

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_store.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
