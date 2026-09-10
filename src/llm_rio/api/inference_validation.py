"""Request defaults, limits, and conservative quota estimates."""

from __future__ import annotations

import json
from typing import Any

from fastapi import Request

from llm_rio.api.schemas import ChatCompletionRequest
from llm_rio.errors import RioError

_IMAGE_CONTENT_TYPES = frozenset({"image", "image_url", "input_image"})


def _messages_for_accounting(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace image payloads with placeholders without changing the forwarded request."""
    accounting_messages: list[dict[str, Any]] = []
    for message in messages:
        accounting_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            accounting_content: list[Any] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") not in _IMAGE_CONTENT_TYPES:
                    accounting_content.append(part)
                    continue
                image_part: dict[str, Any] = {
                    "type": part.get("type"),
                    "image": "<image>",
                }
                detail = part.get("detail")
                image_url = part.get("image_url")
                if detail is None and isinstance(image_url, dict):
                    detail = image_url.get("detail")
                if detail is not None:
                    image_part["detail"] = detail
                accounting_content.append(image_part)
            accounting_message["content"] = accounting_content
        accounting_messages.append(accounting_message)
    return accounting_messages


def _rough_tokens(value: Any) -> int:
    if value is None:
        return 0
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    if not encoded or encoded == '""':
        return 0
    return max(1, (len(encoded) + 3) // 4)


def _conservative_prompt_tokens(value: Any) -> int:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    # Byte fallback tokenizers cannot produce more text tokens than UTF-8 input bytes.
    return max(1, len(encoded.encode("utf-8")))


_MODEL_DEFAULT_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
    "reasoning_effort",
)


def _apply_model_defaults(
    body: ChatCompletionRequest, model: dict[str, Any]
) -> ChatCompletionRequest:
    defaults = model.get("request_defaults")
    if not isinstance(defaults, dict):
        return body
    updates = {
        field: defaults[field]
        for field in _MODEL_DEFAULT_FIELDS
        if field in defaults and getattr(body, field) is None
    }
    return body.model_copy(update=updates) if updates else body


def _validate_request(
    request: Request, body: ChatCompletionRequest, model: dict[str, Any]
) -> tuple[int, int, int | None]:
    settings = request.app.state.settings
    model_limits = model["request_limits"]
    max_context_tokens = int(model_limits["max_context_tokens"])

    raw_prompt_cap = model_limits.get("max_prompt_tokens")
    if raw_prompt_cap is not None:
        max_prompt_tokens = int(raw_prompt_cap)
        if settings.max_prompt_tokens is not None:
            max_prompt_tokens = min(max_prompt_tokens, settings.max_prompt_tokens)
    elif settings.max_prompt_tokens is not None:
        max_prompt_tokens = min(settings.max_prompt_tokens, max_context_tokens)
    else:
        max_prompt_tokens = max_context_tokens

    max_output_tokens = (
        min(settings.max_output_tokens, max_context_tokens)
        if settings.max_output_tokens is not None
        else None
    )
    max_n = settings.max_n

    accounting_messages = _messages_for_accounting(body.messages)
    prompt_tokens = _rough_tokens(accounting_messages)
    reservation_prompt_tokens = _conservative_prompt_tokens(accounting_messages)
    requested_output_tokens = body.output_limit
    remaining_context_tokens = max(0, max_context_tokens - prompt_tokens)
    if prompt_tokens > max_prompt_tokens:
        raise RioError("context_length_exceeded", "The prompt exceeds the configured limit")
    if requested_output_tokens is not None:
        if prompt_tokens + requested_output_tokens > max_context_tokens:
            raise RioError(
                "context_length_exceeded", "Prompt plus output exceeds the model context"
            )
        if max_output_tokens is not None and requested_output_tokens > max_output_tokens:
            raise RioError("max_tokens_exceeded", "The requested output limit is too high")
    enforced_output_limit = (
        min(max_output_tokens, remaining_context_tokens)
        if requested_output_tokens is None and max_output_tokens is not None
        else None
    )
    if max_n is not None and body.n > max_n:
        raise RioError("n_exceeded", "The requested number of choices is too high")
    capabilities = set(model["capabilities"])
    if body.response_format and "structured_output" not in capabilities:
        raise RioError(
            "structured_output_not_supported",
            "Structured output was not validated for this model profile",
        )
    reservation_output_tokens = (
        requested_output_tokens
        if requested_output_tokens is not None
        else enforced_output_limit
        if enforced_output_limit is not None
        else remaining_context_tokens
    )
    reservation_tokens = reservation_prompt_tokens + reservation_output_tokens * body.n
    return prompt_tokens, reservation_tokens, enforced_output_limit
