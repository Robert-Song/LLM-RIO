#!/usr/bin/env python3
"""Standalone remote acceptance suite for the four LLM-RIO production models.

Set LLMRIO_API_KEY before running. The endpoint defaults to isolated localhost
port 3737 and can be changed with LLMRIO_BASE_URL.

Safe, non-maintenance run:

    python3 four_model_prism_acceptance.py

Complete transition run (disruptive: drains every model twice):

    python3 four_model_prism_acceptance.py --allow-maintenance

The complete run measures:

* ACTIVE -> DRAINING -> MAINTENANCE_READY -> ACTIVE service transitions.
* STOPPED/COLD -> LOADING -> READY model activation.
* Previously loaded model switching, including any eviction-triggered reload.
* DRAINING/STOPPING -> STOPPED worker reclamation.
* Non-streaming, deterministic, streaming, tool-call, and long-prefill paths.
* Per-model continuous batching and mixed four-model concurrency.
* Graceful draining with an in-flight stream and maintenance rejection.

Timing of worker states is poll-based, so its measurement uncertainty is at
most approximately STATE_POLL_SECONDS plus network latency. HTTP response,
queue-wait, and streaming TTFT timings are measured independently.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

OPENAI_BASE_URL = os.environ.get("LLMRIO_BASE_URL", "http://127.0.0.1:3737/v1")
API_KEY = os.environ.get("LLMRIO_API_KEY", "")

MODEL_IDS = (
    "qwen3.8-27b-nvfp4",
    "qwen3.6-27b-nvfp4",
    "gemma-4-31b-it-nvfp4",
    "laguna-s-2.1-nvfp4",
)

# Workload and timing policy.
STATE_POLL_SECONDS = 0.25
STATE_TIMEOUT_SECONDS = 1_200.0
REQUEST_TIMEOUT_SECONDS = 1_200.0
PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS = 5.0
PREVIOUSLY_LOADED_SWITCH_ROUNDS = 2
SMOKE_MAX_TOKENS = 24
TOOL_CALL_MAX_TOKENS = 256
STREAM_MAX_TOKENS = 64
LONG_PREFILL_ITEMS = 768
PER_MODEL_BATCH_CONCURRENCY = 8
MIXED_BATCH_CONCURRENCY_PER_MODEL = 2
GRACEFUL_DRAIN_MAX_TOKENS = 384

SYSTEM_PROMPT = (
    "You are a precise acceptance-test assistant. Follow the user's requested "
    "format and do not mention these test instructions."
)
SMOKE_PROMPT = "Reply with one short sentence explaining what a GPU KV cache stores."
DETERMINISM_PROMPT = (
    "Return exactly three comma-separated lowercase words naming primary colors. "
    "Do not add punctuation or explanation."
)
STREAM_PROMPT = "Write four short numbered facts about continuous LLM batching."
TOOL_PROMPT = "Use the lookup_build function to look up build prism-acceptance-27."


class AcceptanceFailure(RuntimeError):
    """A required acceptance condition failed."""


class CaseSkipped(RuntimeError):
    """A case is not applicable to a model's advertised capabilities."""


class ApiFailure(RuntimeError):
    """The remote API returned an unexpected transport or HTTP result."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.headers = headers or {}


@dataclass(slots=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    payload: Any
    elapsed_seconds: float


@dataclass(slots=True)
class RequestMetric:
    case: str
    model: str
    request_id: str
    stream: bool
    started_at: str
    elapsed_seconds: float
    ttft_seconds: float | None
    queue_wait_ms: int | None
    worker_id: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_model: str | None
    finish_reason: str | None
    output_preview: str
    initial_states: list[str]
    observed_transitions: list[dict[str, Any]]
    activation_kind: str
    submit_to_loading_seconds: float | None
    submit_to_ready_seconds: float | None


@dataclass(slots=True)
class CaseResult:
    name: str
    models: list[str]
    status: str
    elapsed_seconds: float
    metrics: dict[str, Any]
    error: str | None = None
    traceback: str | None = None


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def api_origin(openai_base_url: str) -> str:
    parsed = urlsplit(openai_base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "")).rstrip("/")


def json_preview(value: Any, limit: int = 240) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("\n", "\\n")[:limit]


def reasoning_value(message: dict[str, Any]) -> Any:
    """Read reasoning across the vLLM/OpenAI compatibility field names.

    vLLM 0.26 serializes ``ChatMessage.reasoning`` and
    ``DeltaMessage.reasoning``. Older servers and some OpenAI-compatible
    gateways use ``reasoning_content`` instead, so accept both while
    preferring the current vLLM field when it is present.
    """
    if "reasoning" in message:
        return message.get("reasoning")
    return message.get("reasoning_content")


def output_signature(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AcceptanceFailure("chat response did not contain a choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise AcceptanceFailure("chat choice was not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise AcceptanceFailure("chat response did not contain a message")
    return {
        "content": message.get("content"),
        "reasoning": reasoning_value(message),
        "tool_calls": message.get("tool_calls"),
        "finish_reason": choice.get("finish_reason"),
    }


def usage_values(payload: dict[str, Any]) -> tuple[int, int, int]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or prompt + completion)
    return prompt, completion, total


def gpu_memory_reclamation(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_rows = {
        str(gpu.get("uuid")): gpu
        for gpu in before.get("gpus") or []
        if isinstance(gpu, dict) and gpu.get("uuid")
    }
    after_rows = {
        str(gpu.get("uuid")): gpu
        for gpu in after.get("gpus") or []
        if isinstance(gpu, dict) and gpu.get("uuid")
    }
    common = sorted(set(before_rows) & set(after_rows))
    if not common:
        raise AcceptanceFailure("GPU dashboard snapshots had no common GPU UUIDs")
    uncleared = {
        gpu_uuid: list(after_rows[gpu_uuid].get("placement_models") or [])
        for gpu_uuid in common
        if after_rows[gpu_uuid].get("placement_models")
    }
    if uncleared:
        raise AcceptanceFailure(
            f"maintenance-ready dashboard retained live placements: {uncleared}"
        )
    per_gpu: list[dict[str, Any]] = []
    comparable_deltas: list[float] = []
    for gpu_uuid in common:
        before_used = before_rows[gpu_uuid].get("used_vram_mib")
        after_used = after_rows[gpu_uuid].get("used_vram_mib")
        delta = (
            float(before_used) - float(after_used)
            if isinstance(before_used, int | float) and isinstance(after_used, int | float)
            else None
        )
        if delta is not None:
            comparable_deltas.append(delta)
        per_gpu.append(
            {
                "uuid": gpu_uuid,
                "before_used_vram_mib": before_used,
                "after_used_vram_mib": after_used,
                "reclaimed_vram_mib": delta,
            }
        )
    return {
        "all_dashboard_placements_released": True,
        "total_reclaimed_vram_mib": sum(comparable_deltas),
        "per_gpu": per_gpu,
    }


class RemoteApi:
    def __init__(self) -> None:
        self.openai_base_url = OPENAI_BASE_URL.rstrip("/")
        self.origin = api_origin(self.openai_base_url)
        self.default_headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

    def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
        headers: dict[str, str] | None = None,
        accepted_statuses: set[int] | None = None,
    ) -> HttpResult:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        merged = dict(self.default_headers)
        if headers:
            merged.update(headers)
        request = Request(url, data=body, headers=merged, method=method)
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
                status = int(response.status)
                response_headers = {key.lower(): value for key, value in response.headers.items()}
        except HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
            response_headers = {key.lower(): value for key, value in exc.headers.items()}
        except (TimeoutError, URLError, OSError) as exc:
            raise ApiFailure(f"{method} {url} failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        text = raw.decode("utf-8", errors="replace")
        try:
            decoded: Any = json.loads(text) if text else {}
        except json.JSONDecodeError:
            decoded = {"raw": text}
        allowed = accepted_statuses or set(range(200, 300))
        if status not in allowed:
            raise ApiFailure(
                f"{method} {url} returned HTTP {status}: {text[-2000:]}",
                status=status,
                body=text,
                headers=response_headers,
            )
        return HttpResult(status, response_headers, decoded, elapsed)

    def health(self) -> HttpResult:
        return self.request_json(
            "GET",
            f"{self.origin}/health",
            headers={"Authorization": ""},
        )

    def models(self) -> HttpResult:
        return self.request_json("GET", f"{self.openai_base_url}/models")

    def admin_status(self, timeout: float = 20.0) -> dict[str, Any]:
        result = self.request_json("GET", f"{self.origin}/admin/status", timeout=timeout)
        if not isinstance(result.payload, dict):
            raise ApiFailure("admin status response was not an object")
        return result.payload

    def dashboard(self) -> dict[str, Any]:
        result = self.request_json("GET", f"{self.origin}/admin/dashboard", timeout=60.0)
        if not isinstance(result.payload, dict):
            raise ApiFailure("dashboard response was not an object")
        return result.payload

    def maintenance(self, mode: str) -> HttpResult:
        return self.request_json(
            "POST",
            f"{self.origin}/admin/maintenance",
            payload={"mode": mode},
        )

    def request_records(self, run_id: str) -> list[dict[str, Any]]:
        query = urlencode({"test_run_id": run_id})
        result = self.request_json("GET", f"{self.origin}/admin/requests?{query}")
        payload = result.payload
        records = payload.get("requests") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            raise ApiFailure("admin request-record response did not contain a list")
        return [record for record in records if isinstance(record, dict)]

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        run_id: str,
        client_worker: str,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[HttpResult, str]:
        request_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": "none",
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 27,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        headers = {
            "X-Test-Run-ID": run_id,
            "X-Request-ID": request_id,
            "X-Client-Worker": client_worker,
        }
        return (
            self.request_json(
                "POST",
                f"{self.openai_base_url}/chat/completions",
                payload=payload,
                headers=headers,
            ),
            request_id,
        )

    def chat_stream(
        self,
        *,
        model: str,
        prompt: str,
        max_tokens: int,
        run_id: str,
        client_worker: str,
    ) -> tuple[HttpResult, str, float | None]:
        request_id = str(uuid.uuid4())
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": "none",
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 27,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        headers = {
            **self.default_headers,
            "X-Test-Run-ID": run_id,
            "X-Request-ID": request_id,
            "X-Client-Worker": client_worker,
        }
        request = Request(
            f"{self.openai_base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        chunks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        first_token_seconds: float | None = None
        finish_reason: str | None = None
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = int(response.status)
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ApiFailure(f"invalid SSE JSON: {data[:500]}") from exc
                    if not isinstance(event, dict):
                        continue
                    chunks.append(event)
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    choices = event.get("choices")
                    if not isinstance(choices, list):
                        continue
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            continue
                        content = delta.get("content")
                        reasoning = reasoning_value(delta)
                        tool_calls = delta.get("tool_calls")
                        produced = bool(content or reasoning or tool_calls)
                        if produced and first_token_seconds is None:
                            first_token_seconds = time.perf_counter() - started
                        if isinstance(content, str):
                            text_parts.append(content)
                        if isinstance(reasoning, str):
                            reasoning_parts.append(reasoning)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiFailure(
                f"stream returned HTTP {exc.code}: {body[-2000:]}",
                status=int(exc.code),
                body=body,
            ) from exc
        except (TimeoutError, URLError, OSError) as exc:
            raise ApiFailure(f"stream request failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        if status != 200:
            raise ApiFailure(f"stream returned HTTP {status}", status=status)
        if first_token_seconds is None:
            raise AcceptanceFailure(f"{model} stream produced no content, reasoning, or tool delta")
        payload_out = {
            "id": chunks[0].get("id") if chunks else None,
            "model": chunks[0].get("model") if chunks else model,
            "choices": [
                {
                    "message": {
                        "content": "".join(text_parts),
                        "reasoning": "".join(reasoning_parts),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        return (
            HttpResult(status, response_headers, payload_out, elapsed),
            request_id,
            first_token_seconds,
        )


class TransitionMonitor:
    def __init__(self, api: RemoteApi) -> None:
        self.api = api
        self.started = time.perf_counter()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: dict[str, Any] = {}
        self._mode: str | None = None
        self._worker_states: dict[str, str] = {}
        self._worker_models: dict[str, str] = {}
        self._events: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self.max_active_requests: dict[str, int] = {}
        self.max_queued_requests: dict[str, int] = {}

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run,
                name="llm-rio-transition-monitor",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.observe(self.api.admin_status())
            except Exception as exc:
                with self._condition:
                    self._errors.append(f"{utc_now()} {type(exc).__name__}: {exc}")
                    self._condition.notify_all()
            self._stop.wait(STATE_POLL_SECONDS)

    def observe(self, status: dict[str, Any]) -> None:
        observed_at = self.elapsed()
        mode = str(status.get("mode") or "UNKNOWN")
        workers = status.get("workers")
        worker_rows = workers if isinstance(workers, list) else []
        with self._condition:
            if mode != self._mode:
                self._events.append(
                    {
                        "kind": "service_mode",
                        "at_seconds": observed_at,
                        "from": self._mode,
                        "to": mode,
                    }
                )
                self._mode = mode
            for raw in worker_rows:
                if not isinstance(raw, dict):
                    continue
                worker_id = str(raw.get("worker_id") or raw.get("id") or "")
                model = str(raw.get("model") or raw.get("model_id") or "")
                state = str(raw.get("state") or "UNKNOWN")
                if not worker_id:
                    continue
                previous = self._worker_states.get(worker_id)
                if previous != state:
                    self._events.append(
                        {
                            "kind": "worker_state",
                            "at_seconds": observed_at,
                            "worker_id": worker_id,
                            "model": model,
                            "from": previous,
                            "to": state,
                            "gpu_uuids": list(raw.get("gpu_uuids") or []),
                            "tensor_parallel_size": raw.get("tensor_parallel_size"),
                        }
                    )
                    self._worker_states[worker_id] = state
                self._worker_models[worker_id] = model
                active = int(raw.get("active_requests") or 0)
                queued = int(raw.get("queued_requests") or 0)
                self.max_active_requests[model] = max(
                    active,
                    self.max_active_requests.get(model, 0),
                )
                self.max_queued_requests[model] = max(
                    queued,
                    self.max_queued_requests.get(model, 0),
                )
            self._latest = status
            self._condition.notify_all()

    def mark(self, name: str, **details: Any) -> None:
        with self._condition:
            self._events.append(
                {
                    "kind": "phase",
                    "at_seconds": self.elapsed(),
                    "name": name,
                    **details,
                }
            )

    def event_index(self) -> int:
        with self._condition:
            return len(self._events)

    def events_since(self, index: int, model: str | None = None) -> list[dict[str, Any]]:
        with self._condition:
            events = [dict(event) for event in self._events[index:]]
        if model is None:
            return events
        return [event for event in events if event.get("model") == model]

    def model_states(self, model: str) -> list[str]:
        with self._condition:
            return sorted(
                state
                for worker_id, state in self._worker_states.items()
                if self._worker_models.get(worker_id) == model
            )

    def wait_for(
        self,
        predicate: Callable[[dict[str, Any]], bool],
        *,
        description: str,
        timeout: float = STATE_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._latest and predicate(self._latest):
                    return dict(self._latest)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AcceptanceFailure(f"timed out waiting for {description}")
                self._condition.wait(timeout=min(remaining, 1.0))

    def report(self) -> dict[str, Any]:
        with self._condition:
            events = [dict(event) for event in self._events]
            errors = list(self._errors)
        return {
            "poll_interval_seconds": STATE_POLL_SECONDS,
            "events": events,
            "durations": transition_durations(events),
            "max_active_requests_by_model": dict(sorted(self.max_active_requests.items())),
            "max_queued_requests_by_model": dict(sorted(self.max_queued_requests.items())),
            "monitor_errors": errors,
        }


def transition_durations(events: list[dict[str, Any]]) -> dict[str, Any]:
    workers: dict[str, list[dict[str, Any]]] = {}
    modes: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") == "worker_state":
            workers.setdefault(str(event.get("worker_id")), []).append(event)
        elif event.get("kind") == "service_mode":
            modes.append(event)
    worker_durations: list[dict[str, Any]] = []
    for worker_id, records in workers.items():
        pending: dict[str, float] = {}
        for event in records:
            state = str(event.get("to"))
            at = float(event.get("at_seconds") or 0)
            if state in {"LOADING", "DRAINING", "STOPPING"}:
                pending[state] = at
            if state == "READY" and "LOADING" in pending:
                worker_durations.append(
                    {
                        "worker_id": worker_id,
                        "model": event.get("model"),
                        "transition": "LOADING_TO_READY",
                        "seconds": at - pending.pop("LOADING"),
                    }
                )
            if state == "STOPPED":
                for source in ("DRAINING", "STOPPING"):
                    if source in pending:
                        worker_durations.append(
                            {
                                "worker_id": worker_id,
                                "model": event.get("model"),
                                "transition": f"{source}_TO_STOPPED",
                                "seconds": at - pending.pop(source),
                            }
                        )
    mode_durations: list[dict[str, Any]] = []
    draining_at: float | None = None
    for event in modes:
        state = str(event.get("to"))
        at = float(event.get("at_seconds") or 0)
        if state == "DRAINING":
            draining_at = at
        elif state == "MAINTENANCE_READY" and draining_at is not None:
            mode_durations.append(
                {
                    "transition": "DRAINING_TO_MAINTENANCE_READY",
                    "seconds": at - draining_at,
                }
            )
            draining_at = None
    return {"workers": worker_durations, "service": mode_durations}


class AcceptanceSuite:
    def __init__(self, *, allow_maintenance: bool, quick: bool) -> None:
        self.api = RemoteApi()
        self.allow_maintenance = allow_maintenance
        self.quick = quick
        self.run_id = f"four-model-prism-{uuid.uuid4()}"
        self.started_at = utc_now()
        self.started = time.perf_counter()
        self.monitor = TransitionMonitor(self.api)
        self.cases: list[CaseResult] = []
        self.request_metrics: list[RequestMetric] = []
        self._metrics_lock = threading.Lock()
        self.capabilities: dict[str, set[str]] = {}
        self.gpu_snapshots: list[dict[str, Any]] = []
        self.initial_ready_models: set[str] = set()
        self._maintenance_touched = False

    def run_case(
        self,
        name: str,
        models: list[str],
        action: Callable[[], dict[str, Any]],
    ) -> bool:
        print(f"[RUN ] {name}", flush=True)
        started = time.perf_counter()
        try:
            metrics = action()
        except CaseSkipped as exc:
            result = CaseResult(
                name,
                models,
                "SKIP",
                time.perf_counter() - started,
                {},
                str(exc),
            )
            print(f"[SKIP] {name}: {exc}", flush=True)
        except Exception as exc:
            result = CaseResult(
                name,
                models,
                "FAIL",
                time.perf_counter() - started,
                {},
                str(exc),
                traceback.format_exc(),
            )
            print(f"[FAIL] {name}: {exc}", flush=True)
        else:
            result = CaseResult(
                name,
                models,
                "PASS",
                time.perf_counter() - started,
                metrics,
            )
            print(f"[PASS] {name} ({result.elapsed_seconds:.3f}s)", flush=True)
        self.cases.append(result)
        return result.status == "PASS"

    def append_metric(self, metric: RequestMetric) -> None:
        with self._metrics_lock:
            self.request_metrics.append(metric)

    def preflight(self) -> dict[str, Any]:
        health = self.api.health()
        models_result = self.api.models()
        status = self.api.admin_status()
        if status.get("mode") != "ACTIVE":
            raise AcceptanceFailure(
                f"service must begin ACTIVE; current mode is {status.get('mode')}"
            )
        raw_models = (
            models_result.payload.get("data") if isinstance(models_result.payload, dict) else None
        )
        records = raw_models if isinstance(raw_models, list) else []
        by_id = {
            str(record.get("id")): record
            for record in records
            if isinstance(record, dict) and record.get("id")
        }
        missing = [model for model in MODEL_IDS if model not in by_id]
        if missing:
            raise AcceptanceFailure(f"API key cannot see required models: {missing}")
        uncallable = [model for model in MODEL_IDS if not by_id[model].get("callable")]
        if uncallable:
            states = {model: by_id[model].get("state") for model in uncallable}
            raise AcceptanceFailure(
                "required models are not callable with the active Prism runtime: "
                f"{states}. Revalidate them with kvcached enabled."
            )
        placement_profiles: dict[str, list[dict[str, Any]]] = {}
        for model in MODEL_IDS:
            raw_capabilities = by_id[model].get("capabilities")
            self.capabilities[model] = {
                str(value)
                for value in (raw_capabilities if isinstance(raw_capabilities, list) else [])
            }

            raw_profiles = by_id[model].get("placement_profiles")
            profiles = [
                profile
                for profile in (raw_profiles if isinstance(raw_profiles, list) else [])
                if isinstance(profile, dict)
            ]
            invalid = [
                profile
                for profile in profiles
                if int(profile.get("gpu_count") or 0) <= 0
                or int(profile.get("tensor_parallel_size") or 0) <= 0
                or not profile.get("eligible_gpu_sets")
            ]
            compatible = [
                profile
                for profile in profiles
                if profile.get("engine") == "vllm" and profile.get("memory_backend") == "kvcached"
            ]
            if not profiles or invalid:
                raise AcceptanceFailure(
                    f"{model} has missing or invalid placement profiles: {profiles}"
                )
            if not compatible:
                raise AcceptanceFailure(
                    f"{model} has no vLLM profile validated with kvcached. "
                    "Run model retry/revalidation before this suite."
                )
            placement_profiles[model] = profiles
        self.initial_ready_models = {
            str(worker.get("model"))
            for worker in status.get("workers") or []
            if isinstance(worker, dict) and worker.get("state") == "READY"
        }
        self.monitor.observe(status)
        self.snapshot_gpus("initial")
        return {
            "health_status": health.status,
            "models_status": models_result.status,
            "mode": status.get("mode"),
            "visible_models": sorted(by_id),
            "required_models": list(MODEL_IDS),
            "capabilities": {model: sorted(values) for model, values in self.capabilities.items()},
            "placement_profiles": placement_profiles,
            "initial_ready_models": sorted(self.initial_ready_models),
        }

    def snapshot_gpus(self, label: str) -> dict[str, Any]:
        dashboard = self.api.dashboard()
        raw_gpus = dashboard.get("gpus")
        gpus = raw_gpus if isinstance(raw_gpus, list) else []
        snapshot = {
            "label": label,
            "at_seconds": self.monitor.elapsed(),
            "mode": dashboard.get("mode"),
            "gpus": [
                {
                    "index": gpu.get("index"),
                    "uuid": gpu.get("uuid"),
                    "name": gpu.get("name"),
                    "used_vram_mib": gpu.get("used_vram_mib"),
                    "total_vram_mib": gpu.get("total_vram_mib"),
                    "gpu_utilization_percent": gpu.get("gpu_utilization_percent"),
                    "placement_count": len(gpu.get("placements") or []),
                    "placement_models": sorted(
                        {
                            str(placement.get("model"))
                            for placement in gpu.get("placements") or []
                            if isinstance(placement, dict)
                        }
                    ),
                }
                for gpu in gpus
                if isinstance(gpu, dict)
            ],
        }
        self.gpu_snapshots.append(snapshot)
        self.monitor.mark(f"gpu_snapshot:{label}")
        return snapshot

    def reset_to_cold(self) -> dict[str, Any]:
        self._maintenance_touched = True
        before_snapshot = (
            self.gpu_snapshots[-1]
            if self.gpu_snapshots
            else self.snapshot_gpus("before_initial_drain")
        )
        started = time.perf_counter()
        response = self.api.maintenance("drain")
        ack_seconds = time.perf_counter() - started
        if response.payload.get("mode") != "DRAINING":
            raise AcceptanceFailure(f"drain did not enter DRAINING: {response.payload}")
        settled = self.monitor.wait_for(
            lambda status: status.get("mode") == "MAINTENANCE_READY",
            description="initial MAINTENANCE_READY",
        )
        settle_seconds = time.perf_counter() - started
        non_stopped = [
            worker
            for worker in settled.get("workers") or []
            if isinstance(worker, dict)
            and (worker.get("state") != "STOPPED" or int(worker.get("active_requests") or 0))
        ]
        if non_stopped:
            raise AcceptanceFailure(f"maintenance-ready still has live workers: {non_stopped}")
        snapshot = self.snapshot_gpus("initial_maintenance_ready")
        resume_started = time.perf_counter()
        resume = self.api.maintenance("active")
        if resume.payload.get("mode") != "ACTIVE":
            raise AcceptanceFailure(f"resume did not enter ACTIVE: {resume.payload}")
        self.monitor.wait_for(
            lambda status: status.get("mode") == "ACTIVE",
            description="ACTIVE after initial reset",
        )
        resume_seconds = time.perf_counter() - resume_started
        return {
            "drain_ack_seconds": ack_seconds,
            "drain_to_maintenance_ready_seconds": settle_seconds,
            "resume_to_active_seconds": resume_seconds,
            "gpu_memory_reclamation": gpu_memory_reclamation(before_snapshot, snapshot),
            "maintenance_ready_gpu_snapshot": snapshot,
        }

    def timed_chat(
        self,
        *,
        case: str,
        model: str,
        prompt: str,
        max_tokens: int,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
        client_worker: str | None = None,
    ) -> tuple[RequestMetric, dict[str, Any]]:
        initial_states = self.monitor.model_states(model)
        event_index = self.monitor.event_index()
        request_started_elapsed = self.monitor.elapsed()
        started_at = utc_now()
        worker_label = client_worker or case
        if stream:
            result, request_id, ttft = self.api.chat_stream(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                run_id=self.run_id,
                client_worker=worker_label,
            )
        else:
            result, request_id = self.api.chat(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                run_id=self.run_id,
                client_worker=worker_label,
                stream=False,
                extra=extra,
            )
            ttft = None
        if not isinstance(result.payload, dict):
            raise AcceptanceFailure(f"{model} returned a non-object response")
        signature = output_signature(result.payload)
        meaningful = (
            signature.get("content") or signature.get("reasoning") or signature.get("tool_calls")
        )
        if not meaningful:
            raise AcceptanceFailure(f"{model} returned an empty assistant message")
        prompt_tokens, completion_tokens, total_tokens = usage_values(result.payload)
        events = self.monitor.events_since(event_index, model)
        loading_times = [
            float(event["at_seconds"])
            for event in events
            if event.get("kind") == "worker_state" and event.get("to") == "LOADING"
        ]
        ready_times = [
            float(event["at_seconds"])
            for event in events
            if event.get("kind") == "worker_state" and event.get("to") == "READY"
        ]
        was_ready = "READY" in initial_states
        activation_kind = "resident" if was_ready and not loading_times else "cold_or_evicted"
        queue_wait_raw = result.headers.get("x-queue-wait-ms")
        queue_wait_ms = int(queue_wait_raw) if queue_wait_raw and queue_wait_raw.isdigit() else None
        metric = RequestMetric(
            case=case,
            model=model,
            request_id=request_id,
            stream=stream,
            started_at=started_at,
            elapsed_seconds=result.elapsed_seconds,
            ttft_seconds=ttft,
            queue_wait_ms=queue_wait_ms,
            worker_id=result.headers.get("x-worker-id"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_model=(
                str(result.payload.get("model"))
                if result.payload.get("model") is not None
                else None
            ),
            finish_reason=(
                str(signature.get("finish_reason"))
                if signature.get("finish_reason") is not None
                else None
            ),
            output_preview=json_preview(signature),
            initial_states=initial_states,
            observed_transitions=events,
            activation_kind=activation_kind,
            submit_to_loading_seconds=(
                min(loading_times) - request_started_elapsed if loading_times else None
            ),
            submit_to_ready_seconds=min(ready_times) - request_started_elapsed
            if ready_times
            else None,
        )
        self.append_metric(metric)
        return metric, result.payload

    def smoke_case(self, model: str) -> dict[str, Any]:
        metric, _ = self.timed_chat(
            case=f"{model}:activation-smoke",
            model=model,
            prompt=SMOKE_PROMPT,
            max_tokens=SMOKE_MAX_TOKENS,
        )
        return asdict(metric)

    def determinism_case(self, model: str) -> dict[str, Any]:
        first_metric, first = self.timed_chat(
            case=f"{model}:determinism-a",
            model=model,
            prompt=DETERMINISM_PROMPT,
            max_tokens=20,
        )
        second_metric, second = self.timed_chat(
            case=f"{model}:determinism-b",
            model=model,
            prompt=DETERMINISM_PROMPT,
            max_tokens=20,
        )
        first_signature = output_signature(first)
        second_signature = output_signature(second)
        if first_signature != second_signature:
            raise AcceptanceFailure(
                f"{model} temperature=0/seed=27 outputs differed: "
                f"{json_preview(first_signature)} != {json_preview(second_signature)}"
            )
        return {
            "first": asdict(first_metric),
            "second": asdict(second_metric),
            "signature": first_signature,
        }

    def streaming_case(self, model: str) -> dict[str, Any]:
        metric, _ = self.timed_chat(
            case=f"{model}:stream",
            model=model,
            prompt=STREAM_PROMPT,
            max_tokens=STREAM_MAX_TOKENS,
            stream=True,
        )
        if metric.ttft_seconds is None:
            raise AcceptanceFailure(f"{model} stream did not expose TTFT")
        return asdict(metric)

    def tool_case(self, model: str) -> dict[str, Any]:
        if "tools" not in self.capabilities.get(model, set()):
            raise CaseSkipped(f"{model} does not advertise tool capability")
        tool = {
            "type": "function",
            "function": {
                "name": "lookup_build",
                "description": "Look up one build identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"build_id": {"type": "string"}},
                    "required": ["build_id"],
                    "additionalProperties": False,
                },
            },
        }
        metric, payload = self.timed_chat(
            case=f"{model}:tool",
            model=model,
            prompt=TOOL_PROMPT,
            max_tokens=TOOL_CALL_MAX_TOKENS,
            extra={
                "tools": [tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "lookup_build"},
                },
            },
        )
        signature = output_signature(payload)
        tool_calls = signature.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            raise AcceptanceFailure(f"{model} did not return the forced tool call")
        encoded = json.dumps(tool_calls, ensure_ascii=False)
        if "lookup_build" not in encoded or "prism-acceptance-27" not in encoded:
            raise AcceptanceFailure(f"{model} returned an incorrect tool call: {encoded[:1000]}")
        return {"request": asdict(metric), "tool_calls": tool_calls}

    def long_prefill_case(self, model: str) -> dict[str, Any]:
        items = 128 if self.quick else LONG_PREFILL_ITEMS
        prompt = (
            "Read the numbered records and report only the final record number. "
            "Records: " + " ".join(f"[record-{index:04d}: stable]" for index in range(items))
        )
        metric, _ = self.timed_chat(
            case=f"{model}:long-prefill",
            model=model,
            prompt=prompt,
            max_tokens=24,
        )
        if metric.prompt_tokens <= 0:
            raise AcceptanceFailure(f"{model} long-prefill response omitted usage")
        return {"records": items, "request": asdict(metric)}

    def batch_case(
        self,
        *,
        name: str,
        models: list[str],
        concurrency_per_model: int,
    ) -> dict[str, Any]:
        assignments = [(model, index) for model in models for index in range(concurrency_per_model)]
        barrier = threading.Barrier(len(assignments))
        metrics: list[RequestMetric] = []
        failures: list[str] = []
        lock = threading.Lock()

        def task(model: str, index: int) -> None:
            try:
                barrier.wait(timeout=30.0)
                metric, _ = self.timed_chat(
                    case=name,
                    model=model,
                    prompt=(
                        f"Request {index}: give two concise facts about GPU inference "
                        "batch scheduling."
                    ),
                    max_tokens=48,
                    client_worker=f"{name}:{model}:{index}",
                )
                with lock:
                    metrics.append(metric)
            except Exception as exc:
                with lock:
                    failures.append(f"{model}[{index}]: {type(exc).__name__}: {exc}")

        wall_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            futures = [executor.submit(task, model, index) for model, index in assignments]
            for future in futures:
                future.result(timeout=REQUEST_TIMEOUT_SECONDS + 60.0)
        wall_seconds = time.perf_counter() - wall_started
        if failures:
            raise AcceptanceFailure(f"{name} had {len(failures)} failures: {failures[:8]}")
        if len(metrics) != len(assignments):
            raise AcceptanceFailure(
                f"{name} completed {len(metrics)} of {len(assignments)} requests"
            )
        latencies = [metric.elapsed_seconds for metric in metrics]
        queue_waits = [
            float(metric.queue_wait_ms) for metric in metrics if metric.queue_wait_ms is not None
        ]
        completion_tokens = sum(metric.completion_tokens for metric in metrics)
        return {
            "attempted": len(assignments),
            "succeeded": len(metrics),
            "wall_seconds": wall_seconds,
            "completion_tokens": completion_tokens,
            "completion_tokens_per_second": completion_tokens / max(wall_seconds, 1e-9),
            "latency_seconds": {
                "min": min(latencies),
                "mean": statistics.fmean(latencies),
                "p50": percentile(latencies, 0.50),
                "p95": percentile(latencies, 0.95),
                "max": max(latencies),
            },
            "queue_wait_ms": {
                "p50": percentile(queue_waits, 0.50),
                "p95": percentile(queue_waits, 0.95),
                "max": max(queue_waits) if queue_waits else 0,
            },
            "worker_ids": sorted(
                {metric.worker_id for metric in metrics if metric.worker_id is not None}
            ),
            "by_model": {
                model: sum(1 for metric in metrics if metric.model == model) for model in models
            },
        }

    def previously_loaded_switch_case(self) -> dict[str, Any]:
        prior_by_model = {
            model: [metric for metric in self.request_metrics if metric.model == model]
            for model in MODEL_IDS
        }
        missing_prior = [model for model, metrics in prior_by_model.items() if not metrics]
        if missing_prior:
            raise AcceptanceFailure(
                f"models were not successfully exercised before switch test: {missing_prior}"
            )
        rounds = 1 if self.quick else PREVIOUSLY_LOADED_SWITCH_ROUNDS
        results: list[dict[str, Any]] = []
        violations: list[str] = []
        for round_index in range(1, rounds + 1):
            for model in MODEL_IDS:
                previous_metrics = [
                    metric for metric in self.request_metrics if metric.model == model
                ]
                previous_worker_ids = sorted(
                    {
                        metric.worker_id
                        for metric in previous_metrics
                        if metric.worker_id is not None
                    }
                )
                initial_states = self.monitor.model_states(model)
                print(
                    "[STEP] previously-loaded-switch "
                    f"round={round_index}/{rounds} model={model} "
                    f"states={initial_states or ['UNOBSERVED']}",
                    flush=True,
                )
                metric, _ = self.timed_chat(
                    case=f"previously-loaded-switch:round-{round_index}",
                    model=model,
                    prompt="Reply with exactly: ready",
                    max_tokens=4,
                )
                switch_path = (
                    "resident" if metric.activation_kind == "resident" else "evicted_reload"
                )
                results.append(
                    {
                        **asdict(metric),
                        "round": round_index,
                        "switch_path": switch_path,
                        "previous_request_count": len(previous_metrics),
                        "previous_worker_ids": previous_worker_ids,
                        "worker_reused": metric.worker_id in previous_worker_ids,
                        "target_seconds": PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS,
                        "target_met": (
                            metric.elapsed_seconds <= PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS
                        ),
                    }
                )
                if metric.elapsed_seconds > PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS:
                    violations.append(
                        f"{model} round {round_index}: {metric.elapsed_seconds:.3f}s "
                        f"via {switch_path}"
                    )
        if violations:
            raise AcceptanceFailure(
                "previously loaded model switch exceeded "
                f"{PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS:.3f}s target: " + "; ".join(violations)
            )
        return {
            "switch_target_seconds": PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS,
            "rounds": rounds,
            "requests": results,
        }

    def graceful_drain_case(self) -> dict[str, Any]:
        self._maintenance_touched = True
        before_snapshot = self.snapshot_gpus("before_graceful_drain")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        stream_future = executor.submit(
            self.timed_chat,
            case="graceful-drain-inflight",
            model=MODEL_IDS[0],
            prompt=(
                "Write a long numbered checklist for validating an inference server. "
                "Continue until the output limit."
            ),
            max_tokens=GRACEFUL_DRAIN_MAX_TOKENS,
            stream=True,
        )
        try:
            self.monitor.wait_for(
                lambda status: any(
                    isinstance(worker, dict)
                    and worker.get("model") == MODEL_IDS[0]
                    and worker.get("state") == "READY"
                    and int(worker.get("active_requests") or 0) > 0
                    for worker in status.get("workers") or []
                ),
                description="active stream before graceful drain",
            )
            drain_started = time.perf_counter()
            response = self.api.maintenance("drain")
            drain_ack_seconds = time.perf_counter() - drain_started
            if response.payload.get("mode") != "DRAINING":
                raise AcceptanceFailure(
                    f"graceful drain did not enter DRAINING: {response.payload}"
                )
            rejected, _ = self.api.chat(
                model=MODEL_IDS[2],
                prompt="Reply with OK.",
                max_tokens=4,
                run_id=self.run_id,
                client_worker="maintenance-rejection",
                extra=None,
            )
            raise AcceptanceFailure(
                f"maintenance request unexpectedly succeeded with HTTP {rejected.status}"
            )
        except ApiFailure as exc:
            if exc.status != 503:
                raise
            retry_after = exc.headers.get("retry-after")
            if not retry_after:
                raise AcceptanceFailure(
                    "maintenance rejection omitted the required Retry-After header"
                ) from exc
            rejection = {
                "status": exc.status,
                "retry_after": retry_after,
                "body_preview": exc.body[-500:],
            }
        finally:
            try:
                stream_metric, _ = stream_future.result(timeout=REQUEST_TIMEOUT_SECONDS)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        settled = self.monitor.wait_for(
            lambda status: status.get("mode") == "MAINTENANCE_READY",
            description="graceful MAINTENANCE_READY",
        )
        drain_settle_seconds = time.perf_counter() - drain_started
        non_stopped = [
            worker
            for worker in settled.get("workers") or []
            if isinstance(worker, dict)
            and (worker.get("state") != "STOPPED" or int(worker.get("active_requests") or 0))
        ]
        if non_stopped:
            raise AcceptanceFailure(f"graceful drain left live workers: {non_stopped}")
        snapshot = self.snapshot_gpus("graceful_maintenance_ready")
        resume_started = time.perf_counter()
        resume = self.api.maintenance("active")
        if resume.payload.get("mode") != "ACTIVE":
            raise AcceptanceFailure(f"graceful-drain resume failed: {resume.payload}")
        self.monitor.wait_for(
            lambda status: status.get("mode") == "ACTIVE",
            description="ACTIVE after graceful drain",
        )
        return {
            "drain_ack_seconds": drain_ack_seconds,
            "drain_to_maintenance_ready_seconds": drain_settle_seconds,
            "resume_to_active_seconds": time.perf_counter() - resume_started,
            "gpu_memory_reclamation": gpu_memory_reclamation(before_snapshot, snapshot),
            "inflight_stream": asdict(stream_metric),
            "maintenance_rejection": rejection,
            "maintenance_ready_gpu_snapshot": snapshot,
        }

    def transition_coverage_case(self) -> dict[str, Any]:
        report = self.monitor.report()
        events = report["events"]
        durations = report["durations"]
        ready_models = {
            str(event.get("model"))
            for event in events
            if event.get("kind") == "worker_state" and event.get("to") == "READY"
        }
        missing_ready = sorted(set(MODEL_IDS) - ready_models)
        if missing_ready:
            raise AcceptanceFailure(f"state monitor never observed READY for: {missing_ready}")
        result = {
            "ready_models": sorted(ready_models),
            "loading_to_ready": durations["workers"],
            "draining_to_maintenance_ready": durations["service"],
        }
        if not self.allow_maintenance:
            return result
        loading_models = {
            str(item.get("model"))
            for item in durations["workers"]
            if item.get("transition") == "LOADING_TO_READY"
        }
        stopped_models = {
            str(event.get("model"))
            for event in events
            if event.get("kind") == "worker_state" and event.get("to") == "STOPPED"
        }
        missing_loading = sorted(set(MODEL_IDS) - loading_models)
        missing_stopped = sorted(set(MODEL_IDS) - stopped_models)
        if missing_loading or missing_stopped or not durations["service"]:
            raise AcceptanceFailure(
                "incomplete disruptive transition timing coverage: "
                f"missing_loading={missing_loading}, "
                f"missing_stopped={missing_stopped}, "
                f"service_durations={len(durations['service'])}"
            )
        result["stopped_models"] = sorted(stopped_models)
        return result

    def restore_active(self) -> None:
        try:
            status = self.api.admin_status()
            if status.get("mode") != "ACTIVE":
                self.api.maintenance("active")
        except Exception as exc:
            print(f"[WARN] could not restore ACTIVE service mode: {exc}", file=sys.stderr)

    def fetch_records(self) -> dict[str, Any]:
        expected_ids = {metric.request_id for metric in self.request_metrics}
        deadline = time.monotonic() + 30.0
        records: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            records = self.api.request_records(self.run_id)
            if len(records) >= len(expected_ids):
                break
            time.sleep(0.5)
        record_ids = {str(record.get("request_id")) for record in records}
        missing = sorted(expected_ids - record_ids)
        unexpected = sorted(record_ids - expected_ids)
        non_completed = [
            record for record in records if record.get("completion_status") != "COMPLETED"
        ]
        invalid_counts = [
            record
            for record in records
            if int(record.get("accepted_count") or 0) != 1
            or int(record.get("completion_count") or 0) != 1
        ]
        missing_tp = [
            record for record in records if int(record.get("tensor_parallel_size") or 0) <= 0
        ]
        if missing or unexpected or non_completed or invalid_counts or missing_tp:
            raise AcceptanceFailure(
                "server request-record correlation failed: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"non_completed={len(non_completed)}, "
                f"invalid_counts={len(invalid_counts)}, missing_tp={len(missing_tp)}"
            )
        return {
            "expected_successful_requests": len(expected_ids),
            "record_count": len(records),
            "completed": len(records),
            "failed": 0,
            "tensor_parallel_sizes_by_model": {
                model: sorted(
                    {
                        int(record["tensor_parallel_size"])
                        for record in records
                        if record.get("model") == model
                    }
                )
                for model in MODEL_IDS
            },
            "worker_ids_by_model": {
                model: sorted(
                    {
                        str(record["worker_id"])
                        for record in records
                        if record.get("model") == model and record.get("worker_id")
                    }
                )
                for model in MODEL_IDS
            },
            "records": records,
        }

    def run(self) -> dict[str, Any]:
        self.monitor.start()
        try:
            if not self.run_case("preflight", list(MODEL_IDS), self.preflight):
                return self.report()
            if self.allow_maintenance:
                self.run_case("initial-cold-reset", list(MODEL_IDS), self.reset_to_cold)

            for model in MODEL_IDS:
                self.monitor.mark(f"model_phase:{model}", model=model)
                self.run_case(
                    f"{model}:activation-smoke",
                    [model],
                    partial(self.smoke_case, model),
                )
                self.run_case(
                    f"{model}:determinism",
                    [model],
                    partial(self.determinism_case, model),
                )
                self.run_case(
                    f"{model}:streaming",
                    [model],
                    partial(self.streaming_case, model),
                )
                self.run_case(
                    f"{model}:tool-call",
                    [model],
                    partial(self.tool_case, model),
                )
                self.run_case(
                    f"{model}:long-prefill",
                    [model],
                    partial(self.long_prefill_case, model),
                )
                concurrency = 2 if self.quick else PER_MODEL_BATCH_CONCURRENCY
                self.run_case(
                    f"{model}:continuous-batch-{concurrency}",
                    [model],
                    partial(
                        self.batch_case,
                        name=f"{model}:continuous-batch-{concurrency}",
                        models=[model],
                        concurrency_per_model=concurrency,
                    ),
                )

            self.run_case(
                "gpu-snapshot-after-per-model",
                list(MODEL_IDS),
                partial(self.snapshot_gpus, "after_per_model_cases"),
            )
            self.run_case(
                "previously-loaded-switch-round-robin",
                list(MODEL_IDS),
                self.previously_loaded_switch_case,
            )
            mixed_concurrency = 1 if self.quick else MIXED_BATCH_CONCURRENCY_PER_MODEL
            self.run_case(
                f"mixed-four-model-batch-{mixed_concurrency}-each",
                list(MODEL_IDS),
                partial(
                    self.batch_case,
                    name=f"mixed-four-model-batch-{mixed_concurrency}-each",
                    models=list(MODEL_IDS),
                    concurrency_per_model=mixed_concurrency,
                ),
            )
            self.run_case(
                "gpu-snapshot-after-mixed-batch",
                list(MODEL_IDS),
                partial(self.snapshot_gpus, "after_mixed_cases"),
            )
            if self.allow_maintenance:
                self.run_case(
                    "graceful-drain-with-inflight-stream",
                    list(MODEL_IDS),
                    self.graceful_drain_case,
                )
            self.run_case(
                "state-transition-coverage",
                list(MODEL_IDS),
                self.transition_coverage_case,
            )
            self.run_case("server-request-records", list(MODEL_IDS), self.fetch_records)
        finally:
            if self._maintenance_touched:
                self.restore_active()
            self.monitor.stop()
        return self.report()

    def report(self) -> dict[str, Any]:
        transition_report = self.monitor.report()
        status_counts = {
            state: sum(1 for case in self.cases if case.status == state)
            for state in ("PASS", "FAIL", "SKIP")
        }
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "configuration": {
                "openai_base_url": OPENAI_BASE_URL,
                "model_ids": list(MODEL_IDS),
                "api_key_fingerprint": hashlib.sha256(API_KEY.encode()).hexdigest()[:12],
                "allow_maintenance": self.allow_maintenance,
                "quick": self.quick,
                "state_poll_seconds": STATE_POLL_SECONDS,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "tool_call_max_tokens": TOOL_CALL_MAX_TOKENS,
                "previously_loaded_switch_target_seconds": (
                    PREVIOUSLY_LOADED_SWITCH_TARGET_SECONDS
                ),
                "previously_loaded_switch_rounds": (
                    1 if self.quick else PREVIOUSLY_LOADED_SWITCH_ROUNDS
                ),
                "per_model_batch_concurrency": (2 if self.quick else PER_MODEL_BATCH_CONCURRENCY),
                "mixed_batch_concurrency_per_model": (
                    1 if self.quick else MIXED_BATCH_CONCURRENCY_PER_MODEL
                ),
            },
            "passed": status_counts["FAIL"] == 0 and status_counts["PASS"] > 0,
            "case_counts": status_counts,
            "cases": [asdict(case) for case in self.cases],
            "requests": [asdict(metric) for metric in self.request_metrics],
            "state_transitions": transition_report,
            "gpu_snapshots": self.gpu_snapshots,
        }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Run standalone functional, batching, Prism residency, and state-transition "
            "tests against the four configured LLM-RIO models."
        )
    )
    result.add_argument(
        "--allow-maintenance",
        action="store_true",
        help=(
            "Authorize disruptive drain/resume tests. This rejects new production requests "
            "while draining and stops every resident model."
        ),
    )
    result.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller long prompts and concurrency while retaining every case type.",
    )
    result.add_argument(
        "--output",
        type=Path,
        help="JSON report path; defaults to a timestamped file in the current directory.",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if not API_KEY:
        print(
            "Set LLMRIO_API_KEY to a dedicated admin test key before running. "
            "LLMRIO_BASE_URL defaults to http://127.0.0.1:3737/v1.",
            file=sys.stderr,
        )
        return 2
    if not args.allow_maintenance:
        print(
            "[INFO] Maintenance transitions are disabled. Pass --allow-maintenance only "
            "during an authorized test window to measure forced cold/load/drain transitions.",
            flush=True,
        )
    suite = AcceptanceSuite(
        allow_maintenance=bool(args.allow_maintenance),
        quick=bool(args.quick),
    )
    report = suite.run()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"four-model-prism-acceptance-{timestamp}.json")
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "case_counts")}, indent=2))
    print(f"Report: {output.resolve()}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
