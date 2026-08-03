from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from llm_rio.domain import Role


class CreateKeyRequest(BaseModel):
    nickname: str = Field(min_length=1, max_length=100)
    role: Role
    quota_account_id: str | None = None
    quota_account_nickname: str | None = None
    balance_tokens: int = Field(default=1_000_000, ge=0)
    unlimited: bool = False
    model_ids: list[str] = Field(default_factory=list)


class KeySecretResponse(BaseModel):
    id: str
    nickname: str
    api_key: str
    warning: str = "Administrators can retrieve the full key later with the key-list command."


class QuotaUpdate(BaseModel):
    balance_tokens: int = Field(ge=0)
    unlimited: bool = False


class GrantUpdate(BaseModel):
    model_ids: list[str]


class RegisterModelRequest(BaseModel):
    nickname: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
    huggingface_repo: str = Field(pattern=r"^[^/\s]+/[^/\s]+$")
    revision: str | None = None
    grant_to_key_ids: list[str] = Field(default_factory=list)


class MaintenanceRequest(BaseModel):
    mode: Literal["drain", "active"]


class ChatCompletionRequest(BaseModel):
    model_config = {"extra": "allow"}

    model: str
    messages: list[dict[str, Any]]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
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
    def output_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 1024


class MaintenanceStatus(BaseModel):
    mode: str
    workers: list[dict[str, Any]]

