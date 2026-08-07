from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


class EngineSettings(BaseModel):
    vllm_executable: str = "vllm"
    llama_cpp_executable: str = "llama-server"
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
        extra="ignore",
    )

    config_file: Path = Field(default=Path("config.toml"), exclude=True)
    machine_id: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8002, ge=1, le=65535)
    database_path: Path = Path("state/llm-rio.db")
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
    worker_startup_timeout_seconds: float | None = Field(default=None, gt=0)
    worker_drain_watchdog_seconds: float | None = Field(default=None, gt=0)
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        config_path = Path("config.toml")
        if hasattr(init_settings, "init_kwargs"):
            config_path = Path(init_settings.init_kwargs.get("config_file", config_path))
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
