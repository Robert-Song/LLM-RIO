#!/usr/bin/env python3
"""Run three simultaneous users against three RAM-warmed LLM-RIO models.

The endpoint defaults to isolated localhost port 3737; override it with
LLMRIO_BASE_URL. Set LLMRIO_API_KEY to an administrator credential. The script
creates three temporary user keys, shows the live GPU dashboard during
inference, deletes only the keys it created, and writes a JSON evidence report.
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
MODEL_IDS = (
    "qwen3.8-27b-nvfp4",
    "gemma-4-31b-it-nvfp4",
    "laguna-s-2.1-nvfp4",
)
REQUEST_TIMEOUT_SECONDS = 300.0
DASHBOARD_POLL_SECONDS = 0.5


class DemoFailure(RuntimeError):
    """A required precondition or request failed."""


@dataclass(slots=True)
class DemoUser:
    nickname: str
    key_id: str
    api_key: str
    model: str


@dataclass(slots=True)
class RequestResult:
    user: str
    model: str
    request_id: str
    elapsed_seconds: float
    queue_wait_ms: int | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    output_preview: str


def api_origin() -> str:
    parsed = urlsplit(OPENAI_BASE_URL.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "")).rstrip("/")


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str], float]:
    encoded = json.dumps(body).encode() if body is not None else None
    request_headers = {"Accept": "application/json"}
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    if api_key is not None:
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
        raw = exc.read().decode("utf-8", errors="replace")
        raise DemoFailure(f"{method} {url} returned HTTP {exc.code}: {raw}") from exc
    except URLError as exc:
        raise DemoFailure(f"{method} {url} failed: {exc}") from exc


def require_ready_models(admin_key: str) -> None:
    health, _, _ = request_json("GET", f"{api_origin()}/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise DemoFailure(f"LLM-RIO health check failed: {health!r}")

    models, _, _ = request_json("GET", f"{OPENAI_BASE_URL}/models", api_key=admin_key)
    rows = models.get("data") if isinstance(models, dict) else None
    if not isinstance(rows, list):
        raise DemoFailure("GET /v1/models returned an invalid response")
    by_id = {
        str(row.get("id")): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    unavailable = {
        model: by_id.get(model)
        for model in MODEL_IDS
        if model not in by_id or not bool(by_id[model].get("callable"))
    }
    if unavailable:
        raise DemoFailure(f"Demo models are not callable: {unavailable}")

    status, _, _ = request_json("GET", f"{api_origin()}/admin/status", api_key=admin_key)
    workers = status.get("workers") if isinstance(status, dict) else None
    warmed = {
        str(worker.get("model"))
        for worker in workers or []
        if isinstance(worker, dict) and worker.get("state") in {"READY", "SLEEPING"}
    }
    missing = sorted(set(MODEL_IDS) - warmed)
    if missing:
        raise DemoFailure(
            "All models must be resident before measuring warm service; "
            f"missing READY/SLEEPING workers: {missing}"
        )


def create_demo_users(admin_key: str, run_suffix: str) -> list[DemoUser]:
    users: list[DemoUser] = []
    try:
        for index, model in enumerate(MODEL_IDS, 1):
            nickname = f"mixed-demo-{run_suffix}-user-{index}"
            payload, _, _ = request_json(
                "POST",
                f"{api_origin()}/admin/keys",
                api_key=admin_key,
                body={"nickname": nickname, "role": "user", "models": [model]},
            )
            if not isinstance(payload, dict):
                raise DemoFailure(f"Key creation returned invalid data for {nickname}")
            users.append(
                DemoUser(
                    nickname=nickname,
                    key_id=str(payload["id"]),
                    api_key=str(payload["api_key"]),
                    model=model,
                )
            )
    except BaseException:
        delete_demo_users(admin_key, users)
        raise
    return users


def delete_demo_users(admin_key: str, users: list[DemoUser]) -> list[str]:
    failures: list[str] = []
    for user in users:
        try:
            request_json(
                "DELETE",
                f"{api_origin()}/admin/keys/{user.key_id}",
                api_key=admin_key,
            )
        except DemoFailure as exc:
            failures.append(f"{user.nickname}: {exc}")
    return failures


def run_user_request(user: DemoUser, test_run_id: str) -> RequestResult:
    request_id = str(uuid.uuid4())
    payload, headers, elapsed = request_json(
        "POST",
        f"{OPENAI_BASE_URL}/chat/completions",
        api_key=user.api_key,
        headers={"X-Test-Run-ID": test_run_id, "X-Request-ID": request_id},
        body={
            "model": user.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer directly and concisely without hidden reasoning.",
                },
                {
                    "role": "user",
                    "content": (
                        "Give eight numbered, one-sentence facts about how continuous "
                        "batching improves an LLM service."
                    ),
                },
            ],
            "reasoning_effort": "none",
            "temperature": 0,
            "max_tokens": 192,
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise DemoFailure(f"{user.model} returned no completion choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise DemoFailure(f"{user.model} returned no assistant message")
    output = message.get("content") or message.get("reasoning") or message.get("reasoning_content")
    if not isinstance(output, str) or not output.strip():
        raise DemoFailure(f"{user.model} returned an empty assistant message")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    queue_header = headers.get("x-queue-wait-ms")
    return RequestResult(
        user=user.nickname,
        model=user.model,
        request_id=request_id,
        elapsed_seconds=elapsed,
        queue_wait_ms=int(queue_header) if queue_header is not None else None,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        finish_reason=str(choice.get("finish_reason")) if choice.get("finish_reason") else None,
        output_preview=output.strip().replace("\n", " ")[:200],
    )


def dashboard_snapshot(admin_key: str) -> dict[str, Any]:
    payload, _, _ = request_json("GET", f"{api_origin()}/admin/dashboard", api_key=admin_key)
    if not isinstance(payload, dict):
        raise DemoFailure("GET /admin/dashboard returned an invalid response")
    return payload


def print_dashboard(snapshot: dict[str, Any], elapsed_seconds: float) -> None:
    for gpu in snapshot.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        placement_text: list[str] = []
        for placement in gpu.get("placements") or []:
            if not isinstance(placement, dict):
                continue
            slots = placement.get("continuous_batching_slots") or {}
            active = slots.get("active")
            capacity = slots.get("capacity")
            slot_text = f"{active}/{capacity}" if capacity is not None else f"active={active}"
            placement_text.append(f"{placement.get('model')}:{placement.get('state')}[{slot_text}]")
        print(
            f"[GPU ] +{elapsed_seconds:5.1f}s gpu={gpu.get('index')} "
            f"util={gpu.get('gpu_utilization_percent')}% "
            f"vram={gpu.get('used_vram_mib')}/{gpu.get('total_vram_mib')} MiB "
            f"placements={','.join(placement_text) or '-'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-users",
        action="store_true",
        help="Keep generated user keys instead of deleting them after the run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    admin_key = os.environ.get("LLMRIO_API_KEY", "")
    if not admin_key:
        print("[FAIL] Set LLMRIO_API_KEY to an administrator key.", file=sys.stderr)
        return 2

    run_started = time.perf_counter()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    test_run_id = f"three-user-mixed-{timestamp}-{uuid.uuid4().hex[:8]}"
    users: list[DemoUser] = []
    dashboard_samples: list[dict[str, Any]] = []
    results: list[RequestResult] = []
    cleanup_failures: list[str] = []
    return_code = 1
    try:
        print("[RUN ] preflight: require all three models callable and RAM-warmed")
        require_ready_models(admin_key)
        print("[PASS] preflight")

        print("[RUN ] provision three isolated user keys")
        users = create_demo_users(admin_key, test_run_id[-8:])
        for user in users:
            print(f"[USER] {user.nickname} -> {user.model}")
        print("[PASS] provision")

        print("[RUN ] simultaneous three-user / three-model inference")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(users)) as executor:
            futures = {executor.submit(run_user_request, user, test_run_id): user for user in users}
            while not all(future.done() for future in futures):
                snapshot = dashboard_snapshot(admin_key)
                dashboard_samples.append(snapshot)
                print_dashboard(snapshot, time.perf_counter() - run_started)
                time.sleep(DASHBOARD_POLL_SECONDS)
            for future, user in futures.items():
                try:
                    result = future.result()
                except Exception as exc:
                    raise DemoFailure(f"{user.nickname} / {user.model}: {exc}") from exc
                results.append(result)
                print(
                    f"[PASS] {result.user} / {result.model}: "
                    f"{result.elapsed_seconds:.3f}s, queue={result.queue_wait_ms}ms, "
                    f"tokens={result.prompt_tokens}+{result.completion_tokens}"
                )
        print("[PASS] mixed inference")
        return_code = 0
    except DemoFailure as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
    finally:
        if users and not args.keep_users:
            cleanup_failures = delete_demo_users(admin_key, users)
            if cleanup_failures:
                print(f"[WARN] key cleanup failed: {cleanup_failures}", file=sys.stderr)
                return_code = 1

        report = {
            "test_run_id": test_run_id,
            "endpoint": OPENAI_BASE_URL,
            "models": list(MODEL_IDS),
            "started_at": timestamp,
            "elapsed_seconds": time.perf_counter() - run_started,
            "status": "passed" if return_code == 0 else "failed",
            "users_kept": bool(args.keep_users),
            "results": [asdict(result) for result in results],
            "dashboard_samples": dashboard_samples,
            "cleanup_failures": cleanup_failures,
        }
        output = Path("diagnostics") / f"{test_run_id}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[INFO] report: {output}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
