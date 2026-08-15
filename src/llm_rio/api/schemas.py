from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator

from llm_rio.domain import Engine, Role


class CreateKeyRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=100)
    role: Role
    quota_account_id: str | None = None
    quota_account_nickname: str | None = None
    limit_tokens: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("limit_tokens", "balance_tokens"),
    )
    models: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("models", "model_ids"),
    )
    api_key: str | None = Field(default=None, min_length=24)

    @field_validator("api_key")
    @classmethod
    def custom_api_key_uses_rio_format(cls, value: str | None) -> str | None:
        if value is not None and not value.startswith("rio_"):
            raise ValueError("custom API keys must start with rio_")
        return value


class KeySecretResponse(BaseModel):
    id: str
    nickname: str
    api_key: str
    warning: str = "Administrators can retrieve the full key later with the key-list command."


class QuotaUpdate(BaseModel):
    limit_tokens: int = Field(
        ge=0,
        validation_alias=AliasChoices("limit_tokens", "balance_tokens"),
    )
    unlimited: bool = False


class GrantUpdate(BaseModel):
    model_ids: list[str]


class ModelAccessUpdate(BaseModel):
    key: str = Field(min_length=1)
    models: list[str]
    mode: Literal["add", "remove", "replace"] = "add"


class RegisterModelRequest(BaseModel):
    nickname: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    huggingface_repo: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str | None = None
    grant_to_keys: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("grant_to_keys", "grant_to_key_ids"),
    )


class MaintenanceRequest(BaseModel):
    mode: Literal["drain", "active"]


class ProfileEditRequest(BaseModel):
    """Administrator override for a validated placement profile."""

    engine: Engine | None = None
    tensor_parallel_size: int | None = Field(default=None, gt=0)
    max_model_len: int | None = Field(default=None, gt=0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    max_num_batched_tokens: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    gguf_file: str | None = Field(default=None, min_length=1)
    n_gpu_layers: int | None = Field(default=None, ge=0)
    make_default: bool = False
    restart_workers: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None

    @field_validator("messages")
    @classmethod
    def messages_are_not_empty(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("messages cannot be empty")
        return value

    @property
    def output_limit(self) -> int | None:
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens


class MaintenanceStatus(BaseModel):
    mode: str
    workers: list[dict[str, Any]]
