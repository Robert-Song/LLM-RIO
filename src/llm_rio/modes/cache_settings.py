from pydantic import BaseModel, Field, model_validator


class CacheSettings(BaseModel):
    model_config = {"extra": "forbid"}
    preload_models: list[str] = Field(default_factory=list)
    idle_sleep_seconds: float = Field(default=45, ge=0)
    host_cache_max_gib: float | None = Field(default=None, gt=0)
    host_cache_min_available_gib: float = Field(default=4, gt=0)
    swap_max_used_gib: float = Field(default=0, ge=0)
    transition_timeout_seconds: float = Field(default=180, gt=0)

    @model_validator(mode="after")
    def validate_preload(self) -> "CacheSettings":
        self.preload_models = [name.strip() for name in self.preload_models]
        if any(not name for name in self.preload_models) or (
            "*" in self.preload_models and self.preload_models != ["*"]
        ):
            raise ValueError("preload_models must contain names or a single '*'")
        return self
