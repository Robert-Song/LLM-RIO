from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


class EngineSettings(BaseModel):
    vllm_executable: str = "vllm"
    llama_cpp_executable: str = "llama-server"
    enable_llama_cpp: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLMRIO_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    config_file: Path = Field(default=Path("config.toml"), exclude=True)
    machine_id: str = "local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    database_path: Path = Path("state/llm-rio.db")
    model_store: Path = Path("models")
    log_dir: Path = Path("logs")
    managed_gpu_uuids: list[str] = Field(default_factory=list)
    reserved_vram_mib: int = 2048
    queue_capacity_per_model: int = 1024
    queue_capacity_per_tenant: int = 128
    wait_duration_seconds: float = 900.0
    minimum_residency_seconds: float = 300.0
    placement_cooldown_seconds: float = 60.0
    fair_share_seconds: float = 7200.0
    validation_idle_window_seconds: float = 600.0
    worker_startup_timeout_seconds: float = 900.0
    worker_drain_watchdog_seconds: float = 600.0
    worker_port_start: int = 18000
    worker_port_end: int = 18999
    scheduler_tick_seconds: float = 1.0
    max_prompt_tokens: int = 32768
    max_output_tokens: int = 8192
    max_n: int = 4
    default_quota_tokens: int = 1_000_000
    quota_charge_requested_maximum: bool = False
    capture_worker_engine_logs: bool = True
    hf_token: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    engines: EngineSettings = Field(default_factory=EngineSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: object,
        env_settings: object,
        dotenv_settings: object,
        file_secret_settings: object,
    ) -> tuple[object, ...]:
        config_path = Path("config.toml")
        if hasattr(init_settings, "init_kwargs"):
            config_path = Path(init_settings.init_kwargs.get("config_file", config_path))
        toml_source = TomlConfigSettingsSource(settings_cls, toml_file=config_path)
        return init_settings, env_settings, dotenv_settings, toml_source, file_secret_settings

    @model_validator(mode="after")
    def validate_settings(self) -> Settings:
        if self.worker_port_start > self.worker_port_end:
            raise ValueError("worker_port_start must not exceed worker_port_end")
        if self.fair_share_seconds <= 0 or self.wait_duration_seconds <= 0:
            raise ValueError("scheduler durations must be positive")
        if self.reserved_vram_mib < 0:
            raise ValueError("reserved_vram_mib cannot be negative")
        return self

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_store.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

