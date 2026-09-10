"""Decode worker responses without coupling protocol validation to HTTP or accounting."""

from __future__ import annotations

import codecs
from typing import Any


class SSEDecoder:
    """Incrementally decode UTF-8 SSE frames, including multiline data fields."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._pending = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes) -> list[str]:
        self._pending += self._decoder.decode(chunk)
        events = []
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self._data:
                    events.append("\n".join(self._data))
                    self._data.clear()
                continue
            field, _, value = line.partition(":")
            if field == "data":
                self._data.append(value.removeprefix(" "))
        return events


def token_usage(event: dict[str, Any]) -> tuple[int, int, int] | None:
    """Return reported usage, preserving explicit zeros and rejecting invalid counters."""
    usage = event.get("usage")
    if usage is None or usage == {}:
        return None
    if not isinstance(usage, dict):
        raise ValueError("worker usage must be an object")
    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    if any(type(value) is not int or value < 0 for value in (prompt, completion)):
        raise ValueError("worker token counts must be nonnegative integers")
    total = usage.get("total_tokens", prompt + completion)
    if type(total) is not int or total < 0:
        raise ValueError("worker total token count must be a nonnegative integer")
    return prompt, completion, total


def completion_choices(event: Any, *, streaming: bool) -> list[dict[str, Any]]:
    """Validate shapes before the proxy uses worker-controlled JSON values."""
    if not isinstance(event, dict):
        raise ValueError("worker response must be an object")
    choices = event.get("choices", [] if streaming else None)
    if not isinstance(choices, list):
        raise ValueError("worker choices must be an array")
    field = "delta" if streaming else "message"
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get(field), dict):
            raise ValueError(f"worker choice must contain a {field} object")
    return choices
