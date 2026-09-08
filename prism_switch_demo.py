#!/usr/bin/env python3
"""Run the presentation's sequential warm model-switch demonstration.

The default sequence is Qwen -> Gemma -> Laguna, repeated twice. The script
uses the public LLM-RIO endpoint on port 3737, polls the GPU dashboard while
each request runs, enforces the ten-second end-to-end target, and writes a JSON
evidence report under ``diagnostics/``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

OPENAI_BASE_URL = os.environ.get("LLMRIO_BASE_URL", "http://127.0.0.1:3737/v1")
MODELS = (
    "qwen3.8-27b-nvfp4",
    "gemma-4-31b-it-nvfp4",
    "laguna-s-2.1-nvfp4",
)
REQUEST_TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.5


class DemoFailure(RuntimeError):
    """A required warm-switch demonstration gate failed."""


@dataclass(slots=True)
class SwitchResult:
    step: int
    round: int
    model: str
    request_id: str
    worker_id: str | None
    elapsed_seconds: float
    queue_wait_ms: int | None
    prompt_tokens: int
    completion_tokens: int
    output_preview: str
    before_states: list[str]
    after_workers: list[dict[str, Any]]


def api_origin() -> str:
    parsed = urlsplit(OPENAI_BASE_URL.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "")).rstrip("/")


def require_safe_demo_endpoint() -> None:
    parsed = urlsplit(OPENAI_BASE_URL)
    if parsed.port == 8002:
        raise DemoFailure("refusing to run the demo against protected port 8002")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise DemoFailure(
            f"refusing non-loopback demo endpoint {OPENAI_BASE_URL!r}; "
            "use the isolated port-3737 service"
        )


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str], float]:
    encoded = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {"Accept": "application/json"}
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    request_headers.update(headers or {})
    started = time.perf_counter()
    try:
        with urlopen(
            Request(url, data=encoded, headers=request_headers, method=method),
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            raw = response.read()
            payload = json.loads(raw) if raw else None
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return payload, response_headers, time.perf_counter() - started
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise DemoFailure(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise DemoFailure(f"{method} {url} failed: {exc}") from exc


def admin_status(admin_key: str) -> dict[str, Any]:
    payload, _, _ = request_json(
        "GET", f"{api_origin()}/admin/status", api_key=admin_key
    )
    if not isinstance(payload, dict):
        raise DemoFailure("GET /admin/status returned invalid data")
    return payload


def model_workers(status: dict[str, Any], model: str) -> list[dict[str, Any]]:
    return [
        worker
        for worker in status.get("workers") or []
        if isinstance(worker, dict) and worker.get("model") == model
    ]


def preflight(admin_key: str) -> dict[str, Any]:
    health, _, _ = request_json("GET", f"{api_origin()}/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise DemoFailure(f"LLM-RIO health check failed: {health!r}")
    models, _, _ = request_json("GET", f"{OPENAI_BASE_URL}/models", api_key=admin_key)
    rows = models.get("data") if isinstance(models, dict) else None
    by_id = {
        str(row.get("id")): row
        for row in rows or []
        if isinstance(row, dict) and row.get("id")
    }
    unavailable = {
        model: by_id.get(model)
        for model in MODELS
        if model not in by_id or not bool(by_id[model].get("callable"))
    }
    if unavailable:
        raise DemoFailure(f"switch models are not callable: {unavailable}")
    status = admin_status(admin_key)
    cold = {
        model: [worker.get("state") for worker in model_workers(status, model)]
        for model in MODELS
        if not any(
            worker.get("state") in {"READY", "SLEEPING"}
            for worker in model_workers(status, model)
        )
    }
    if cold:
        raise DemoFailure(f"all switch models must be RAM-warmed first: {cold}")
    return status


def dashboard(admin_key: str) -> dict[str, Any]:
    payload, _, _ = request_json(
        "GET", f"{api_origin()}/admin/dashboard", api_key=admin_key
    )
    if not isinstance(payload, dict):
        raise DemoFailure("GET /admin/dashboard returned invalid data")
    return payload


def compact_dashboard(snapshot: dict[str, Any], elapsed: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for gpu in snapshot.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        placements = [
            {
                "model": placement.get("model"),
                "state": placement.get("state"),
                "weight_storage": placement.get("weight_storage"),
                "active": (placement.get("continuous_batching_slots") or {}).get(
                    "active"
                ),
            }
            for placement in gpu.get("placements") or []
            if isinstance(placement, dict)
        ]
        rows.append(
            {
                "index": gpu.get("index"),
                "uuid": gpu.get("uuid"),
                "utilization_percent": gpu.get("gpu_utilization_percent"),
                "used_vram_mib": gpu.get("used_vram_mib"),
                "total_vram_mib": gpu.get("total_vram_mib"),
                "placements": placements,
            }
        )
    return {"at_seconds": elapsed, "gpus": rows}


def print_dashboard(sample: dict[str, Any]) -> None:
    parts: list[str] = []
    for gpu in sample["gpus"]:
        placements = ",".join(
            f"{row.get('model')}:{row.get('state')}[{row.get('weight_storage')}]"
            for row in gpu["placements"]
        )
        parts.append(
            f"GPU{gpu.get('index')} util={gpu.get('utilization_percent')}% "
            f"vram={gpu.get('used_vram_mib')}/{gpu.get('total_vram_mib')}MiB "
            f"{placements or '-'}"
        )
    print(f"[GPU ] +{sample['at_seconds']:6.1f}s " + " | ".join(parts), flush=True)


def chat(
    admin_key: str,
    model: str,
    test_run_id: str,
) -> tuple[dict[str, Any], dict[str, str], float, str]:
    request_id = str(uuid.uuid4())
    payload, headers, elapsed = request_json(
        "POST",
        f"{OPENAI_BASE_URL}/chat/completions",
        api_key=admin_key,
        headers={"X-Test-Run-ID": test_run_id, "X-Request-ID": request_id},
        body={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer directly in one sentence without hidden reasoning.",
                },
                {
                    "role": "user",
                    "content": (
                        "In one sentence, explain why a sleeping model can wake faster "
                        "than a cold model."
                    ),
                },
            ],
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": 64,
        },
    )
    if not isinstance(payload, dict):
        raise DemoFailure(f"{model} returned non-object data")
    return payload, headers, elapsed, request_id


def parse_result(
    *,
    step: int,
    round_index: int,
    model: str,
    request_id: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    elapsed: float,
    before_states: list[str],
    after_workers: list[dict[str, Any]],
) -> SwitchResult:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise DemoFailure(f"{model} returned no completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise DemoFailure(f"{model} returned no assistant message")
    output = message.get("content") or message.get("reasoning") or message.get(
        "reasoning_content"
    )
    if not isinstance(output, str) or not output.strip():
        raise DemoFailure(f"{model} returned empty output")
    usage_value = payload.get("usage")
    usage = usage_value if isinstance(usage_value, dict) else {}
    queue_wait = headers.get("x-queue-wait-ms")
    return SwitchResult(
        step=step,
        round=round_index,
        model=model,
        request_id=request_id,
        worker_id=headers.get("x-worker-id"),
        elapsed_seconds=elapsed,
        queue_wait_ms=int(queue_wait) if queue_wait and queue_wait.isdigit() else None,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        output_preview=output.strip().replace("\n", " ")[:240],
        before_states=before_states,
        after_workers=after_workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--target-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.target_seconds <= 0:
        parser.error("--target-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    admin_key = os.environ.get("LLMRIO_API_KEY", "")
    if not admin_key:
        print("[FAIL] Set LLMRIO_API_KEY to an administrator key.", file=sys.stderr)
        return 2
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    test_run_id = f"prism-switch-{timestamp}-{uuid.uuid4().hex[:8]}"
    run_started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    results: list[SwitchResult] = []
    failures: list[str] = []
    try:
        require_safe_demo_endpoint()
        status = preflight(admin_key)
        print(
            "[PASS] preflight: "
            + ", ".join(
                f"{model}={','.join(str(w.get('state')) for w in model_workers(status, model))}"
                for model in MODELS
            ),
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            step = 0
            for round_index in range(1, args.rounds + 1):
                for model in MODELS:
                    step += 1
                    before = admin_status(admin_key)
                    before_states = [
                        str(worker.get("state"))
                        for worker in model_workers(before, model)
                    ]
                    print(
                        f"[SWITCH] step={step} round={round_index} model={model} "
                        f"before={before_states}",
                        flush=True,
                    )
                    future = executor.submit(chat, admin_key, model, test_run_id)
                    while not future.done():
                        sample = compact_dashboard(
                            dashboard(admin_key), time.perf_counter() - run_started
                        )
                        samples.append(sample)
                        print_dashboard(sample)
                        time.sleep(POLL_SECONDS)
                    payload, headers, elapsed, request_id = future.result()
                    after = admin_status(admin_key)
                    result = parse_result(
                        step=step,
                        round_index=round_index,
                        model=model,
                        request_id=request_id,
                        payload=payload,
                        headers=headers,
                        elapsed=elapsed,
                        before_states=before_states,
                        after_workers=model_workers(after, model),
                    )
                    results.append(result)
                    print(
                        f"[PASS] step={step} model={model} elapsed={elapsed:.3f}s "
                        f"queue={result.queue_wait_ms}ms worker={str(result.worker_id or '-')[:8]} "
                        f"output={result.output_preview!r}",
                        flush=True,
                    )
                    if elapsed >= args.target_seconds:
                        failures.append(
                            f"step {step} {model}: {elapsed:.3f}s exceeded "
                            f"{args.target_seconds:.3f}s"
                        )
    except Exception as exc:
        failures.append(str(exc))
        print(f"[FAIL] {exc}", file=sys.stderr)

    elapsed = time.perf_counter() - run_started
    report = {
        "test_run_id": test_run_id,
        "endpoint": OPENAI_BASE_URL,
        "models": list(MODELS),
        "rounds": args.rounds,
        "target_seconds": args.target_seconds,
        "elapsed_seconds": elapsed,
        "status": (
            "passed"
            if not failures and len(results) == args.rounds * len(MODELS)
            else "failed"
        ),
        "failures": failures,
        "results": [asdict(result) for result in results],
        "dashboard_samples": samples,
    }
    output = Path("diagnostics") / f"{test_run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[SUMMARY] status={report['status']} calls={len(results)}/"
        f"{args.rounds * len(MODELS)} elapsed={elapsed:.2f}s report={output.resolve()}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
