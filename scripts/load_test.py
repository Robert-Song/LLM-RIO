#!/usr/bin/env python3
"""Bounded concurrent load tester for LLM-RIO's chat-completions endpoint.

The API key is read from an environment variable so it does not appear in the
process list. Generated model text is never included in metrics or result files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(slots=True)
class RequestResult:
    request_number: int
    ok: bool
    status_code: int | None
    latency_seconds: float
    time_to_first_token_seconds: float | None
    queue_wait_milliseconds: int | None
    prompt_tokens: int
    completion_tokens: int
    error: str | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return parsed


def concurrency_ramp(value: str) -> list[int]:
    try:
        stages = [positive_int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be comma-separated positive integers") from exc
    if not stages:
        raise argparse.ArgumentTypeError("must contain at least one concurrency level")
    if stages != sorted(set(stages)):
        raise argparse.ArgumentTypeError("levels must be unique and increasing")
    return stages


def normalize_base_url(value: str) -> str:
    base_url = value.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    if not base_url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    return base_url


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def approximate_prompt(target_tokens: int) -> str:
    words = (
        "research evidence method system measurement analysis inference optimization "
        "concurrency scheduler latency throughput memory batching experiment validation "
    ).split()
    repetitions = math.ceil(target_tokens / len(words))
    body = (words * repetitions)[:target_tokens]
    return " ".join(body)


def parse_queue_wait(headers: httpx.Headers) -> int | None:
    value = headers.get("X-Queue-Wait-Ms")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def update_usage(event: dict[str, Any], current: tuple[int, int]) -> tuple[int, int]:
    usage = event.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", current[0]) or current[0])
    completion = int(usage.get("completion_tokens", current[1]) or current[1])
    return prompt, completion


def observed_text(event: dict[str, Any]) -> str:
    pieces: list[str] = []
    for choice in event.get("choices") or []:
        content = choice.get("delta") or choice.get("message") or {}
        for field in ("content", "reasoning_content", "reasoning"):
            value = content.get(field)
            if isinstance(value, str):
                pieces.append(value)
    return "".join(pieces)


async def request_once(
    client: httpx.AsyncClient,
    *,
    url: str,
    api_key: str,
    model: str,
    prompt_body: str,
    request_number: int,
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> RequestResult:
    request_id = str(uuid.uuid4())
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are participating in a controlled inference load test.",
            },
            {
                "role": "user",
                "content": (
                    f"Load-test request {request_number}. Analyze the following terms in detail "
                    f"and continue until the output limit: {prompt_body}"
                ),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": request_number,
        "stream": stream,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Request-ID": request_id,
        "Idempotency-Key": request_id,
    }
    started = time.perf_counter()
    status_code: int | None = None
    queue_wait: int | None = None
    prompt_tokens = 0
    completion_tokens = 0
    first_token_at: float | None = None
    observed_characters = 0
    saw_stream_event = False
    saw_done = False
    stream_error: str | None = None

    try:
        if stream:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                status_code = response.status_code
                queue_wait = parse_queue_wait(response.headers)
                if not response.is_success:
                    body = (await response.aread()).decode("utf-8", errors="replace")[:500]
                    return RequestResult(
                        request_number,
                        False,
                        status_code,
                        time.perf_counter() - started,
                        None,
                        queue_wait,
                        0,
                        0,
                        f"HTTP {status_code}: {body}",
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    saw_stream_event = True
                    if event.get("error"):
                        stream_error = f"stream error: {str(event['error'])[:500]}"
                        continue
                    prompt_tokens, completion_tokens = update_usage(
                        event, (prompt_tokens, completion_tokens)
                    )
                    text = observed_text(event)
                    if text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    observed_characters += len(text)
            if stream_error or not saw_stream_event or not saw_done:
                if stream_error:
                    error = stream_error
                elif not saw_stream_event:
                    error = "stream returned no SSE data events"
                else:
                    error = "stream ended without [DONE]"
                return RequestResult(
                    request_number,
                    False,
                    status_code,
                    time.perf_counter() - started,
                    first_token_at - started if first_token_at is not None else None,
                    queue_wait,
                    prompt_tokens,
                    completion_tokens,
                    error,
                )
        else:
            response = await client.post(url, headers=headers, json=payload)
            status_code = response.status_code
            queue_wait = parse_queue_wait(response.headers)
            if not response.is_success:
                return RequestResult(
                    request_number,
                    False,
                    status_code,
                    time.perf_counter() - started,
                    None,
                    queue_wait,
                    0,
                    0,
                    f"HTTP {status_code}: {response.text[:500]}",
                )
            event = response.json()
            prompt_tokens, completion_tokens = update_usage(event, (0, 0))
            observed_characters = len(observed_text(event))
            first_token_at = time.perf_counter()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(
            request_number,
            False,
            status_code,
            time.perf_counter() - started,
            None,
            queue_wait,
            prompt_tokens,
            completion_tokens,
            f"{type(exc).__name__}: {exc}",
        )

    if completion_tokens == 0 and observed_characters:
        completion_tokens = max(1, math.ceil(observed_characters / 4))
    finished = time.perf_counter()
    return RequestResult(
        request_number,
        True,
        status_code,
        finished - started,
        first_token_at - started if first_token_at is not None else None,
        queue_wait,
        prompt_tokens,
        completion_tokens,
        None,
    )


def summarize(
    results: list[RequestResult], *, concurrency: int, wall_seconds: float
) -> dict[str, Any]:
    successful = [result for result in results if result.ok]
    latencies = [result.latency_seconds for result in successful]
    first_tokens = [
        result.time_to_first_token_seconds
        for result in successful
        if result.time_to_first_token_seconds is not None
    ]
    queue_waits = [
        float(result.queue_wait_milliseconds)
        for result in successful
        if result.queue_wait_milliseconds is not None
    ]
    completion_tokens = sum(result.completion_tokens for result in successful)
    prompt_tokens = sum(result.prompt_tokens for result in successful)
    errors = Counter(
        (result.error or "unknown error").split(":", maxsplit=1)[0]
        for result in results
        if not result.ok
    )
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "error_rate": (len(results) - len(successful)) / len(results),
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(len(successful) / wall_seconds, 3),
        "completion_tokens_per_second": round(completion_tokens / wall_seconds, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_seconds": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "time_to_first_token_seconds": {
            "p50": percentile(first_tokens, 0.50),
            "p95": percentile(first_tokens, 0.95),
            "p99": percentile(first_tokens, 0.99),
        },
        "queue_wait_milliseconds": {
            "p50": percentile(queue_waits, 0.50),
            "p95": percentile(queue_waits, 0.95),
            "p99": percentile(queue_waits, 0.99),
            "max": max(queue_waits) if queue_waits else None,
        },
        "errors": dict(errors),
    }


async def run_stage(
    client: httpx.AsyncClient,
    *,
    concurrency: int,
    requests_per_worker: int,
    request_offset: int,
    request_options: dict[str, Any],
) -> tuple[list[RequestResult], dict[str, Any]]:
    request_count = concurrency * requests_per_worker
    queue: asyncio.Queue[int] = asyncio.Queue()
    for number in range(request_offset, request_offset + request_count):
        queue.put_nowait(number)

    results: list[RequestResult] = []
    progress_interval = max(1, request_count // 10)

    async def worker() -> None:
        while True:
            try:
                number = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await request_once(client, request_number=number, **request_options)
                results.append(result)
                completed = len(results)
                if completed % progress_interval == 0 or completed == request_count:
                    failures = sum(not item.ok for item in results)
                    print(
                        f"  progress {completed}/{request_count}; failures={failures}",
                        flush=True,
                    )
            finally:
                queue.task_done()

    started = time.perf_counter()
    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(*tasks)
    wall_seconds = time.perf_counter() - started
    results.sort(key=lambda item: item.request_number)
    return results, summarize(results, concurrency=concurrency, wall_seconds=wall_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send bounded concurrent OpenAI-compatible requests to LLM-RIO."
    )
    parser.add_argument(
        "--preset",
        choices=["0", "1", "2", "3", "test0", "test1", "test2", "test3", "check-all", "heavy-qwen", "heavy-laguna", "multi-model"],
        help="Run a preset load test mode (0: check all models, 1: heavy 128-parallel Qwen call, 2: simultaneous dual-model call, 3: heavy 128-parallel laguna call).",
    )
    parser.add_argument("--base-url", default=os.getenv("LLMRIO_API_URL"))
    parser.add_argument("--model", default=os.getenv("LLMRIO_MODEL"))
    parser.add_argument("--api-key-env", default="LLMRIO_API_KEY")
    parser.add_argument("--concurrency", type=positive_int, default=8)
    parser.add_argument(
        "--ramp",
        type=concurrency_ramp,
        help="Increasing comma-separated stages, for example 1,8,16,32,64,128.",
    )
    parser.add_argument("--requests-per-worker", type=positive_int, default=2)
    parser.add_argument("--prompt-tokens", type=positive_int, default=64)
    parser.add_argument("--max-tokens", type=positive_int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use streaming responses so time to first token can be measured.",
    )
    parser.add_argument(
        "--max-error-rate",
        type=probability,
        default=0.05,
        help="Do not advance to the next ramp stage if this rate is exceeded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write metrics and truncated errors as JSON; generated text is never saved.",
    )
    return parser


async def run_preset_0(base_url: str, api_key: str, timeout: httpx.Timeout) -> int:
    """Preset 0: Load check calling all models in catalog once."""
    print("=== PRESET 0: Single Load Check Across All Models ===", flush=True)
    models_to_test = ["laguna-s-2.1-nvfp4", "qwen3.6-27b-nvfp4", "gemma-4-31b-it-nvfp4"]
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"})
            if resp.is_success:
                data = resp.json()
                discovered = [m["id"] for m in data.get("data", []) if "id" in m]
                if discovered:
                    models_to_test = discovered
        except Exception:
            pass

        print(f"Testing {len(models_to_test)} models: {models_to_test}\n", flush=True)
        any_failed = False
        for idx, model_name in enumerate(models_to_test):
            print(f"[{idx+1}/{len(models_to_test)}] Requesting model '{model_name}'...", flush=True)
            result = await request_once(
                client,
                request_number=idx + 1,
                url=f"{base_url}/v1/chat/completions",
                api_key=api_key,
                model=model_name,
                prompt_body="Reply with a short greeting.",
                max_tokens=16,
                temperature=0.0,
                stream=True,
            )
            status = "OK" if result.ok else "FAILED"
            ttft = f"{result.time_to_first_token_seconds:.3f}s" if result.time_to_first_token_seconds else "N/A"
            print(f"  Result: {status} | Status Code: {result.status_code} | Latency: {result.latency_seconds:.3f}s | TTFT: {ttft}")
            if not result.ok:
                print(f"  Error: {result.error}", file=sys.stderr)
                any_failed = True
        return 1 if any_failed else 0


async def run_preset_1(
    base_url: str, api_key: str, args: argparse.Namespace, timeout: httpx.Timeout
) -> int:
    """Preset 1: Heavy model call on Qwen (128 parallel requests)."""
    print("=== PRESET 1: Heavy Concurrency Load Test on Qwen (128 parallel) ===", flush=True)
    model_name = args.model or "qwen3.6-27b-nvfp4"
    concurrency = 128
    requests_per_worker = args.requests_per_worker if args.requests_per_worker > 1 else 10
    total_requests = concurrency * requests_per_worker
    prompt_body = approximate_prompt(args.prompt_tokens)

    url = f"{base_url}/v1/chat/completions"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    request_options = {
        "url": url,
        "api_key": api_key,
        "model": model_name,
        "prompt_body": prompt_body,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        print(f"Launching {total_requests} requests ({concurrency} workers x {requests_per_worker} reqs) on '{model_name}'...", flush=True)
        results, summary = await run_stage(
            client,
            concurrency=concurrency,
            requests_per_worker=requests_per_worker,
            request_offset=0,
            request_options=request_options,
        )
        print("\nSummary for Heavy Qwen Test:")
        print(json.dumps(summary, indent=2), flush=True)
        return 1 if summary["failed"] > 0 else 0


async def run_preset_2(
    base_url: str, api_key: str, args: argparse.Namespace, timeout: httpx.Timeout
) -> int:
    """Preset 2: Multi-model concurrent load test (16 concurrent per model, 32 total)."""
    print("=== PRESET 2: Dual Model Concurrent Load Test (32 total concurrent) ===", flush=True)
    concurrency_per_model = 16
    requests_per_worker = args.requests_per_worker if args.requests_per_worker > 1 else 2

    models_to_run = ["qwen3.6-27b-nvfp4", "gemma-4-31b-it-nvfp4"]
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(f"{base_url}/v1/models", headers={"Authorization": f"Bearer {api_key}"})
            if resp.is_success:
                available = [m["id"] for m in resp.json().get("data", []) if "id" in m]
                if "gemma-4-31b-it-nvfp4" not in available and "laguna-s-2.1-nvfp4" in available:
                    models_to_run = ["qwen3.6-27b-nvfp4", "laguna-s-2.1-nvfp4"]
                elif len(available) >= 2:
                    models_to_run = available[:2]
        except Exception:
            pass

    total_concurrency = concurrency_per_model * len(models_to_run)
    print(f"Running models concurrently: {models_to_run} ({concurrency_per_model} workers each = {total_concurrency} total concurrent)", flush=True)

    url = f"{base_url}/v1/chat/completions"
    prompt_body = approximate_prompt(args.prompt_tokens)
    limits = httpx.Limits(max_connections=total_concurrency, max_keepalive_connections=total_concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        async def run_model_test(model_name: str, offset: int):
            request_options = {
                "url": url,
                "api_key": api_key,
                "model": model_name,
                "prompt_body": prompt_body,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "stream": args.stream,
            }
            results, summary = await run_stage(
                client,
                concurrency=concurrency_per_model,
                requests_per_worker=requests_per_worker,
                request_offset=offset,
                request_options=request_options,
            )
            return model_name, results, summary

        tasks = [
            run_model_test(model_name, idx * 1000)
            for idx, model_name in enumerate(models_to_run)
        ]
        outcomes = await asyncio.gather(*tasks)

        all_results = []
        any_failed = False
        for model_name, results, summary in outcomes:
            all_results.extend(results)
            print(f"\n--- Summary for Model '{model_name}' ---")
            print(json.dumps(summary, indent=2), flush=True)
            if summary["failed"] > 0:
                any_failed = True

        combined_summary = summarize(all_results, concurrency=total_concurrency, wall_seconds=max(s["wall_seconds"] for _, _, s in outcomes))
        print("\n=== COMBINED DUAL-MODEL SUMMARY ===")
        print(json.dumps(combined_summary, indent=2), flush=True)

        return 1 if any_failed else 0


async def run_preset_3(
    base_url: str, api_key: str, args: argparse.Namespace, timeout: httpx.Timeout
) -> int:
    """Preset 3: Heavy model call on laguna (128 parallel requests)."""
    print("=== PRESET 3: Heavy Concurrency Load Test on laguna (128 parallel) ===", flush=True)
    model_name = args.model or "laguna-s-2.1-nvfp4"
    concurrency = 128
    requests_per_worker = args.requests_per_worker if args.requests_per_worker > 1 else 10
    total_requests = concurrency * requests_per_worker
    prompt_body = approximate_prompt(args.prompt_tokens)

    url = f"{base_url}/v1/chat/completions"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    request_options = {
        "url": url,
        "api_key": api_key,
        "model": model_name,
        "prompt_body": prompt_body,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        print(f"Launching {total_requests} requests ({concurrency} workers x {requests_per_worker} reqs) on '{model_name}'...", flush=True)
        results, summary = await run_stage(
            client,
            concurrency=concurrency,
            requests_per_worker=requests_per_worker,
            request_offset=0,
            request_options=request_options,
        )
        print("\nSummary for Heavy laguna Test:")
        print(json.dumps(summary, indent=2), flush=True)
        return 1 if summary["failed"] > 0 else 0


async def async_main(args: argparse.Namespace) -> int:
    if not args.base_url:
        raise ValueError("set LLMRIO_API_URL or pass --base-url")
    if not args.preset and not args.model:
        raise ValueError("set LLMRIO_MODEL or pass --model or pass --preset (0, 1, 2)")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.timeout <= 0:
        raise ValueError("--timeout must be greater than zero")

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"environment variable {args.api_key_env} is not set")
    base_url = normalize_base_url(args.base_url)
    timeout = httpx.Timeout(args.timeout, connect=30.0, pool=args.timeout)

    if args.preset in ("0", "test0", "check-all"):
        return await run_preset_0(base_url, api_key, timeout)
    elif args.preset in ("1", "test1", "heavy-qwen"):
        return await run_preset_1(base_url, api_key, args, timeout)
    elif args.preset in ("2", "test2", "multi-model"):
        return await run_preset_2(base_url, api_key, args, timeout)
    elif args.preset in ("3", "test3", "heavy-laguna"):
        return await run_preset_3(base_url, api_key, args, timeout)
    url = f"{base_url}/v1/chat/completions"
    stages = args.ramp or [args.concurrency]
    prompt_body = approximate_prompt(args.prompt_tokens)
    timeout = httpx.Timeout(args.timeout, connect=30.0, pool=args.timeout)
    limits = httpx.Limits(
        max_connections=max(stages), max_keepalive_connections=max(stages)
    )
    request_options = {
        "url": url,
        "api_key": api_key,
        "model": args.model,
        "prompt_body": prompt_body,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": args.stream,
    }
    report: dict[str, Any] = {
        "base_url": base_url,
        "model": args.model,
        "stream": args.stream,
        "prompt_tokens_requested_approximately": args.prompt_tokens,
        "max_completion_tokens": args.max_tokens,
        "requests_per_worker": args.requests_per_worker,
        "stages": [],
    }
    any_failures = False
    request_offset = 0

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for warmup_number in range(args.warmup):
            print(f"Warm-up request {warmup_number + 1}/{args.warmup}...", flush=True)
            warmup_options = {
                **request_options,
                "prompt_body": "Reply with exactly: ready",
                "max_tokens": min(args.max_tokens, 16),
            }
            result = await request_once(
                client,
                request_number=-(warmup_number + 1),
                **warmup_options,
            )
            if not result.ok:
                print(f"Warm-up failed: {result.error}", file=sys.stderr)
                return 2

        for concurrency in stages:
            request_count = concurrency * args.requests_per_worker
            print(
                f"Stage concurrency={concurrency}, requests={request_count}, "
                f"max_tokens={args.max_tokens}...",
                flush=True,
            )
            results, summary = await run_stage(
                client,
                concurrency=concurrency,
                requests_per_worker=args.requests_per_worker,
                request_offset=request_offset,
                request_options=request_options,
            )
            request_offset += request_count
            report["stages"].append(
                {"summary": summary, "requests": [asdict(item) for item in results]}
            )
            print(json.dumps(summary, indent=2), flush=True)
            any_failures = any_failures or bool(summary["failed"])
            if summary["error_rate"] > args.max_error_rate:
                print(
                    "Stopping ramp: stage error rate exceeded --max-error-rate.",
                    file=sys.stderr,
                )
                break

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Metrics written to {args.output}")
    return 1 if any_failures else 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(async_main(args))
    except (ValueError, KeyboardInterrupt) as exc:
        print(f"load test stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
