#!/usr/bin/env python3
"""Deterministic external acceptance runner for the LLM-RIO test contract."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_now() -> str:
    return utc_now().isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def normalize_base_url(value: str) -> str:
    result = value.rstrip("/")
    if result.endswith("/v1"):
        result = result[:-3]
    if not result.startswith(("http://", "https://")):
        raise ValueError("BASE_URL/--base-url must begin with http:// or https://")
    return result


def event_text(event: dict[str, Any]) -> str:
    pieces: list[str] = []
    for choice in event.get("choices") or []:
        message = choice.get("delta") or choice.get("message") or {}
        if not isinstance(message, dict):
            continue
        for key in ("content", "reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str):
                pieces.append(value)
    return "".join(pieces)


@dataclass(slots=True)
class RequestResult:
    request_id: str
    client_worker: str
    sequence: int
    model: str
    stream: bool
    ok: bool
    status_code: int | None
    worker_id: str | None
    response_model: str | None
    started_at: str
    completed_at: str
    started_monotonic: float
    completed_monotonic: float
    latency_ms: float
    ttft_ms: float | None
    queue_wait_ms: int | None
    prompt_tokens: int
    completion_tokens: int
    output_characters: int
    saw_terminal_event: bool
    chunk_worker_ids: list[str]
    error: str | None


class JsonlSink:
    def __init__(self, path: Path, *, echo: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.echo = echo
        self._lines: list[str] = []

    async def emit(self, result: RequestResult) -> None:
        line = json.dumps(asdict(result), sort_keys=True)
        self._lines.append(line)
        if self.echo:
            print(line, flush=True)

    def close(self) -> None:
        content = "\n".join(self._lines)
        if content:
            content += "\n"
        self.path.write_text(content, encoding="utf-8")


@dataclass(slots=True)
class Checks:
    failures: list[str] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.failures.append(message)


@dataclass(slots=True)
class Outcome:
    results: list[RequestResult] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    skip_reason: str | None = None


@dataclass(slots=True)
class Context:
    args: argparse.Namespace
    client: httpx.AsyncClient
    sink: JsonlSink
    checks: Checks
    run_id: str
    base_url: str
    admin_key: str
    team_a_key: str
    team_b_key: str
    qwen: str
    gemma: str
    laguna: str
    initial_status: dict[str, Any]
    catalog: dict[str, dict[str, Any]]
    managed_gpu_count: int


def auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def queue_wait(headers: httpx.Headers) -> int | None:
    raw = headers.get("X-Queue-Wait-Ms")
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


async def request_once(
    context: Context,
    *,
    api_key: str,
    model: str,
    client_worker: str,
    sequence: int,
    stream: bool = True,
    max_tokens: int | None = None,
    prompt: str = "Reply with exactly: OK",
    force_output_limit: bool = False,
) -> RequestResult:
    request_id = str(uuid.uuid4())
    started_at = iso_now()
    started = time.perf_counter()
    status_code: int | None = None
    worker_id: str | None = None
    response_model: str | None = None
    wait_ms: int | None = None
    prompt_tokens = completion_tokens = output_characters = 0
    first_token: float | None = None
    saw_terminal = not stream
    chunk_worker_ids: set[str] = set()
    error: str | None = None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens or context.args.max_tokens,
        "temperature": 0,
        "seed": sequence,
        "stream": stream,
    }
    if force_output_limit:
        payload["ignore_eos"] = True
    headers = {
        **auth(api_key),
        "X-Request-ID": request_id,
        "Idempotency-Key": request_id,
        "X-Test-Run-ID": context.run_id,
        "X-Client-Worker": client_worker,
    }

    try:
        if stream:
            async with context.client.stream(
                "POST",
                f"{context.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                status_code = response.status_code
                worker_id = response.headers.get("X-Worker-ID")
                wait_ms = queue_wait(response.headers)
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")[:500]
                    error = f"HTTP {response.status_code}: {body}"
                else:
                    saw_event = False
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            saw_terminal = True
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            error = error or "stream contained invalid JSON"
                            continue
                        saw_event = True
                        if event.get("error"):
                            error = f"stream error: {event['error']}"
                        raw_model = event.get("model")
                        if isinstance(raw_model, str):
                            response_model = raw_model
                        raw_worker = event.get("worker_id")
                        if isinstance(raw_worker, str):
                            chunk_worker_ids.add(raw_worker)
                        usage = event.get("usage") or {}
                        prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens) or 0)
                        completion_tokens = int(
                            usage.get("completion_tokens", completion_tokens) or 0
                        )
                        text = event_text(event)
                        if text and first_token is None:
                            first_token = time.perf_counter()
                        output_characters += len(text)
                    if not saw_event:
                        error = error or "stream returned no data events"
                    if not saw_terminal:
                        error = error or "stream ended without [DONE]"
        else:
            response = await context.client.post(
                f"{context.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            status_code = response.status_code
            worker_id = response.headers.get("X-Worker-ID")
            wait_ms = queue_wait(response.headers)
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:500]}"
            else:
                try:
                    event = response.json()
                except ValueError:
                    event = {}
                    error = "response was not valid JSON"
                response_model = event.get("model")
                usage = event.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                output_characters = len(event_text(event))
                first_token = time.perf_counter()
    except (httpx.HTTPError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"

    completed = time.perf_counter()
    if completion_tokens == 0 and output_characters:
        completion_tokens = max(1, math.ceil(output_characters / 4))
    if status_code == 200 and not worker_id:
        error = error or "gateway omitted X-Worker-ID"
    if status_code == 200 and response_model != model:
        error = error or f"response model mismatch: expected {model}, got {response_model}"
    if status_code == 200 and output_characters <= 0:
        error = error or "successful response had no non-empty output choice"
    if len(chunk_worker_ids) > 1:
        error = error or f"stream mixed worker IDs: {sorted(chunk_worker_ids)}"
    if worker_id and chunk_worker_ids and chunk_worker_ids != {worker_id}:
        error = error or "stream worker ID did not match the gateway-selected worker"

    result = RequestResult(
        request_id=request_id,
        client_worker=client_worker,
        sequence=sequence,
        model=model,
        stream=stream,
        ok=error is None and status_code == 200,
        status_code=status_code,
        worker_id=worker_id,
        response_model=response_model,
        started_at=started_at,
        completed_at=iso_now(),
        started_monotonic=started,
        completed_monotonic=completed,
        latency_ms=(completed - started) * 1000,
        ttft_ms=(first_token - started) * 1000 if first_token is not None else None,
        queue_wait_ms=wait_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        output_characters=output_characters,
        saw_terminal_event=saw_terminal,
        chunk_worker_ids=sorted(chunk_worker_ids),
        error=error,
    )
    await context.sink.emit(result)
    return result


async def get_status(context: Context) -> dict[str, Any]:
    response = await context.client.get(
        f"{context.base_url}/admin/status", headers=auth(context.admin_key)
    )
    response.raise_for_status()
    status = response.json()
    status["observed_at"] = iso_now()
    return status


async def monitor_status(
    context: Context,
    stop: asyncio.Event,
    snapshots: list[dict[str, Any]],
) -> None:
    while True:
        try:
            snapshots.append(await get_status(context))
        except httpx.HTTPError as exc:
            context.checks.failures.append(f"admin status polling failed: {exc}")
            return
        if stop.is_set():
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=context.args.poll_interval)


async def wait_for_status(
    context: Context,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    description: str,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any] | None:
    deadline = time.monotonic() + context.args.deadline
    while time.monotonic() < deadline:
        status = await get_status(context)
        snapshots.append(status)
        if predicate(status):
            return status
        try:
            await asyncio.sleep(context.args.poll_interval)
        except asyncio.CancelledError:
            raise
    context.checks.failures.append(f"deadline waiting for scheduler event: {description}")
    return None


async def fetch_request_records(context: Context, expected: int) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 30
    latest: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        response = await context.client.get(
            f"{context.base_url}/admin/requests",
            params={"test_run_id": context.run_id},
            headers=auth(context.admin_key),
        )
        response.raise_for_status()
        latest = response.json().get("requests") or []
        terminal = [
            record
            for record in latest
            if record.get("completion_status") in {"COMPLETED", "FAILED"}
        ]
        if len(latest) >= expected and len(terminal) >= expected:
            return latest
        await asyncio.sleep(0.2)
    context.checks.failures.append(
        f"request ledger did not reach {expected} terminal records; observed {len(latest)}"
    )
    return latest


def validate_results_and_records(
    checks: Checks,
    results: list[RequestResult],
    records: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> None:
    records_by_id = {record["request_id"]: record for record in records}
    ready_at_by_worker: dict[str, datetime] = {}
    for snapshot in snapshots:
        for worker in snapshot.get("workers") or []:
            ready_at = parse_time(worker.get("ready_at"))
            worker_id = worker.get("worker_id")
            if worker_id and ready_at:
                ready_at_by_worker[worker_id] = ready_at
    checks.require(
        len(records_by_id) == len(records),
        "request ledger returned duplicate request IDs",
    )
    for result in results:
        checks.require(result.ok, f"request {result.request_id} failed: {result.error}")
        record = records_by_id.get(result.request_id)
        checks.require(record is not None, f"request {result.request_id} is absent from admin logs")
        if record is None:
            continue
        checks.require(
            record.get("model") == result.model,
            f"request {result.request_id} logged the wrong logical model",
        )
        checks.require(
            record.get("worker_id") == result.worker_id,
            f"request {result.request_id} worker header/log mismatch",
        )
        checks.require(
            record.get("accepted_count") == 1,
            f"request {result.request_id} was accepted {record.get('accepted_count')} times",
        )
        checks.require(
            record.get("completion_count") == 1,
            f"request {result.request_id} completed {record.get('completion_count')} times",
        )
        checks.require(
            record.get("completion_status") == "COMPLETED",
            f"request {result.request_id} terminal state is {record.get('completion_status')}",
        )
        admitted = parse_time(record.get("admission_time"))
        accepted = parse_time(record.get("worker_accepted_time"))
        completed = parse_time(record.get("completion_time"))
        checks.require(
            admitted is not None
            and accepted is not None
            and completed is not None
            and admitted <= accepted <= completed,
            f"request {result.request_id} has invalid lifecycle timestamps",
        )
        worker_ready_at = ready_at_by_worker.get(str(record.get("worker_id")))
        checks.require(
            worker_ready_at is not None and accepted is not None and worker_ready_at <= accepted,
            f"request {result.request_id} lacks proof its worker was READY before acceptance",
        )
        if result.error and any(
            marker in result.error.lower()
            for marker in ("traceback", "127.0.0.1:18", "connection refused")
        ):
            checks.failures.append(f"request {result.request_id} leaked a private engine failure")


def ready_workers(status: dict[str, Any], model: str | None = None) -> list[dict[str, Any]]:
    return [
        worker
        for worker in status.get("workers") or []
        if worker.get("state") == "READY" and (model is None or worker.get("model") == model)
    ]


def placement_gpu_sets(model: dict[str, Any], gpu_count: int) -> set[tuple[str, ...]]:
    return {
        tuple(gpu_set)
        for profile in model.get("placement_profiles") or []
        if profile.get("gpu_count") == gpu_count
        for gpu_set in profile.get("eligible_gpu_sets") or []
    }


def catalog_checks(context: Context) -> None:
    expected = (context.qwen, context.gemma, context.laguna)
    for alias in expected:
        model = context.catalog.get(alias)
        context.checks.require(model is not None, f"catalog is missing {alias}")
        if model is None:
            continue
        context.checks.require(model.get("callable") is True, f"catalog marks {alias} non-callable")
        context.checks.require(
            bool(model.get("revision")), f"catalog omits immutable revision for {alias}"
        )
        context.checks.require(
            bool(model.get("profile_ids")), f"catalog omits profile IDs for {alias}"
        )

    for alias in (context.qwen, context.gemma):
        model = context.catalog.get(alias) or {}
        single_sets = placement_gpu_sets(model, 1)
        context.checks.require(
            len(single_sets) >= context.managed_gpu_count,
            f"{alias} lacks a validated one-GPU placement on every managed GPU",
        )
    laguna = context.catalog.get(context.laguna) or {}
    multi = [
        profile
        for profile in laguna.get("placement_profiles") or []
        if profile.get("gpu_count") == context.managed_gpu_count
        and profile.get("tensor_parallel_size") == context.managed_gpu_count
    ]
    context.checks.require(
        bool(multi),
        f"{context.laguna} lacks a validated all-GPU tensor-parallel profile",
    )


def make_group(
    context: Context,
    *,
    api_key: str,
    model: str,
    workers: int,
    repeats: int,
    start: asyncio.Event,
    prefix: str,
    results: list[RequestResult],
    max_tokens: int | None = None,
    prompt: str = "Reply with exactly: OK",
    force_output_limit: bool = False,
) -> tuple[list[asyncio.Task[None]], asyncio.Event]:
    ready = asyncio.Event()
    ready_count = 0
    ready_lock = asyncio.Lock()

    async def client_worker(index: int) -> None:
        nonlocal ready_count
        worker_name = f"{prefix}-{index:03d}"
        async with ready_lock:
            ready_count += 1
            if ready_count == workers:
                ready.set()
        await start.wait()
        for sequence in range(repeats):
            result = await request_once(
                context,
                api_key=api_key,
                model=model,
                client_worker=worker_name,
                sequence=index * repeats + sequence,
                stream=True,
                max_tokens=max_tokens,
                prompt=prompt,
                force_output_limit=force_output_limit,
            )
            results.append(result)

    return (
        [
            asyncio.create_task(client_worker(index), name=f"{prefix}-{index:03d}")
            for index in range(workers)
        ],
        ready,
    )


async def gather_with_deadline(
    context: Context, tasks: list[asyncio.Task[Any]], description: str
) -> None:
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=context.args.deadline)
    except TimeoutError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        context.checks.failures.append(f"{description} exceeded the test deadline")


async def run_preset_0(context: Context) -> Outcome:
    outcome = Outcome()
    for index, model in enumerate((context.qwen, context.gemma, context.laguna)):
        outcome.results.append(
            await request_once(
                context,
                api_key=context.admin_key,
                model=model,
                client_worker=f"smoke-{model}",
                sequence=index,
                stream=False,
                max_tokens=16,
            )
        )
        expected_replicas = (
            context.managed_gpu_count if model in {context.qwen, context.gemma} else 1
        )
        await wait_for_status(
            context,
            lambda status, alias=model, count=expected_replicas: len(ready_workers(status, alias))
            >= count,
            description=f"{expected_replicas} ready {model} placement(s)",
            snapshots=outcome.snapshots,
        )

    outcome.records = await fetch_request_records(context, len(outcome.results))
    validate_results_and_records(
        context.checks, outcome.results, outcome.records, outcome.snapshots
    )
    for record in outcome.records:
        expected_tp = context.managed_gpu_count if record["model"] == context.laguna else 1
        context.checks.require(
            record.get("tensor_parallel_size") == expected_tp,
            (
                f"{record['model']} used tp={record.get('tensor_parallel_size')}, "
                f"expected {expected_tp}"
            ),
        )
        context.checks.require(
            len(record.get("gpu_uuids") or []) == expected_tp,
            f"{record['model']} logged the wrong GPU-set size",
        )
    return outcome


async def run_preset_1(context: Context) -> Outcome:
    outcome = Outcome()
    start = asyncio.Event()
    tasks, ready = make_group(
        context,
        api_key=context.team_a_key,
        model=context.qwen,
        workers=128,
        repeats=context.args.requests_per_worker,
        start=start,
        prefix="qwen",
        results=outcome.results,
    )
    await ready.wait()
    monitor_stop = asyncio.Event()
    monitor = asyncio.create_task(monitor_status(context, monitor_stop, outcome.snapshots))
    start.set()
    await gather_with_deadline(context, tasks, "preset 1 workload")
    monitor_stop.set()
    await monitor
    expected = 128 * context.args.requests_per_worker
    context.checks.require(
        len(outcome.results) == expected,
        f"preset 1 attempted {len(outcome.results)} terminal calls, expected {expected}",
    )
    outcome.records = await fetch_request_records(context, len(outcome.results))
    validate_results_and_records(
        context.checks, outcome.results, outcome.records, outcome.snapshots
    )

    counts = Counter(
        record.get("worker_id") for record in outcome.records if record.get("model") == context.qwen
    )
    counts.pop(None, None)
    context.checks.require(
        len(counts) >= 2,
        f"Qwen traffic reached {len(counts)} replica(s), expected at least 2",
    )
    context.checks.require(
        all(count > 0 for count in counts.values()),
        "a ready Qwen replica accepted no requests",
    )
    placements = {
        record["worker_id"]: tuple(record.get("gpu_uuids") or [])
        for record in outcome.records
        if record.get("worker_id") in counts
    }
    all_owned_gpus = [gpu for gpu_set in placements.values() for gpu in gpu_set]
    context.checks.require(
        len(all_owned_gpus) == len(set(all_owned_gpus))
        and all(len(gpu_set) == 1 for gpu_set in placements.values()),
        "Qwen replicas did not own disjoint one-GPU placements",
    )
    context.checks.require(
        any(len(ready_workers(snapshot, context.qwen)) >= 2 for snapshot in outcome.snapshots),
        "status polling never observed two ready Qwen replicas",
    )
    outcome.details["requests_per_replica"] = dict(counts)
    return outcome


def dual_profiles_compatible(context: Context) -> bool:
    qwen_sets = placement_gpu_sets(context.catalog.get(context.qwen) or {}, 1)
    gemma_sets = placement_gpu_sets(context.catalog.get(context.gemma) or {}, 1)
    return any(set(qwen).isdisjoint(gemma) for qwen in qwen_sets for gemma in gemma_sets)


async def run_preset_2(context: Context) -> Outcome:
    outcome = Outcome()
    if not dual_profiles_compatible(context):
        outcome.skip_reason = "profile incompatibility: no disjoint Qwen/Gemma one-GPU sets"
        return outcome

    qwen_start = asyncio.Event()
    gemma_start = asyncio.Event()
    qwen_tasks, qwen_ready = make_group(
        context,
        api_key=context.team_a_key,
        model=context.qwen,
        workers=16,
        repeats=context.args.requests_per_worker,
        start=qwen_start,
        prefix="qwen",
        results=outcome.results,
        max_tokens=context.args.coexistence_qwen_max_tokens,
        prompt=(
            "Write a long numbered technical checklist. Continue producing distinct "
            "items until the output limit is reached; do not stop early."
        ),
        force_output_limit=True,
    )
    gemma_tasks, gemma_ready = make_group(
        context,
        api_key=context.team_b_key,
        model=context.gemma,
        workers=16,
        repeats=context.args.requests_per_worker,
        start=gemma_start,
        prefix="gemma",
        results=outcome.results,
        max_tokens=context.args.coexistence_gemma_max_tokens,
        prompt=(
            "Write a long numbered technical checklist. Continue producing distinct "
            "items until the output limit is reached; do not stop early."
        ),
    )
    await asyncio.gather(qwen_ready.wait(), gemma_ready.wait())
    monitor_stop = asyncio.Event()
    monitor = asyncio.create_task(monitor_status(context, monitor_stop, outcome.snapshots))
    qwen_start.set()
    await asyncio.sleep(2)
    gemma_start.set()
    await gather_with_deadline(context, qwen_tasks + gemma_tasks, "preset 2 workload")
    monitor_stop.set()
    await monitor

    outcome.details["qwen_max_tokens_per_request"] = context.args.coexistence_qwen_max_tokens
    outcome.details["gemma_max_tokens_per_request"] = context.args.coexistence_gemma_max_tokens

    expected = 32 * context.args.requests_per_worker
    context.checks.require(
        len(outcome.results) == expected,
        f"preset 2 completed {len(outcome.results)} calls, expected {expected}",
    )
    outcome.records = await fetch_request_records(context, len(outcome.results))
    validate_results_and_records(
        context.checks, outcome.results, outcome.records, outcome.snapshots
    )

    workers: dict[str, dict[str, Any]] = {}
    for record in outcome.records:
        workers[record["worker_id"]] = record
    qwen_workers = [row for row in workers.values() if row["model"] == context.qwen]
    gemma_workers = [row for row in workers.values() if row["model"] == context.gemma]
    context.checks.require(bool(qwen_workers), "no Qwen worker accepted preset 2 traffic")
    context.checks.require(bool(gemma_workers), "no Gemma worker accepted preset 2 traffic")
    context.checks.require(
        all(
            set(qwen.get("gpu_uuids") or []).isdisjoint(gemma.get("gpu_uuids") or [])
            for qwen in qwen_workers
            for gemma in gemma_workers
        ),
        "Qwen and Gemma active placements overlapped",
    )
    context.checks.require(
        any(
            ready_workers(snapshot, context.qwen) and ready_workers(snapshot, context.gemma)
            for snapshot in outcome.snapshots
        ),
        "status polling never observed ready Qwen/Gemma coexistence",
    )

    by_id = {record["request_id"]: record for record in outcome.records}
    first_wave: dict[str, list[dict[str, Any]]] = {context.qwen: [], context.gemma: []}
    for result in outcome.results:
        if result.sequence % context.args.requests_per_worker == 0:
            record = by_id.get(result.request_id)
            if record:
                first_wave[result.model].append(record)
    for model, other in ((context.qwen, context.gemma), (context.gemma, context.qwen)):
        accepted = [parse_time(record.get("worker_accepted_time")) for record in first_wave[model]]
        other_completed = [
            parse_time(record.get("completion_time")) for record in first_wave[other]
        ]
        accepted = [value for value in accepted if value is not None]
        other_completed = [value for value in other_completed if value is not None]
        context.checks.require(
            bool(accepted) and bool(other_completed) and min(accepted) < max(other_completed),
            f"{model} was starved until {other} completed its initial wave",
        )
    return outcome


async def run_preset_3(context: Context) -> Outcome:
    outcome = Outcome()
    qwen_start = asyncio.Event()
    qwen_tasks, qwen_ready = make_group(
        context,
        api_key=context.team_a_key,
        model=context.qwen,
        workers=32,
        repeats=context.args.requests_per_worker,
        start=qwen_start,
        prefix="qwen",
        results=outcome.results,
    )
    await qwen_ready.wait()
    monitor_stop = asyncio.Event()
    monitor = asyncio.create_task(monitor_status(context, monitor_stop, outcome.snapshots))
    qwen_start.set()
    await wait_for_status(
        context,
        lambda status: len(ready_workers(status, context.qwen)) >= 2
        and sum(worker.get("active_requests", 0) for worker in ready_workers(status, context.qwen))
        > 0,
        description="two active Qwen replicas before Laguna admission",
        snapshots=outcome.snapshots,
    )
    laguna_task = asyncio.create_task(
        request_once(
            context,
            api_key=context.team_b_key,
            model=context.laguna,
            client_worker="laguna-000",
            sequence=0,
            stream=True,
        )
    )
    await gather_with_deadline(context, qwen_tasks + [laguna_task], "preset 3 workload")
    if laguna_task.done() and not laguna_task.cancelled():
        laguna_result = laguna_task.result()
        if laguna_result not in outcome.results:
            outcome.results.append(laguna_result)
    monitor_stop.set()
    await monitor

    expected = 32 * context.args.requests_per_worker + 1
    context.checks.require(
        len(outcome.results) == expected,
        f"preset 3 completed {len(outcome.results)} calls, expected {expected}",
    )
    outcome.records = await fetch_request_records(context, len(outcome.results))
    validate_results_and_records(
        context.checks, outcome.results, outcome.records, outcome.snapshots
    )

    queued_snapshots = [
        snapshot
        for snapshot in outcome.snapshots
        if (snapshot.get("queued_models") or {}).get(context.laguna, 0) > 0
        and sum(
            worker.get("active_requests", 0)
            for worker in snapshot.get("workers") or []
            if worker.get("model") == context.qwen
        )
        > 0
    ]
    context.checks.require(
        bool(queued_snapshots),
        "status never observed Laguna queued behind active Qwen",
    )
    for snapshot in queued_snapshots:
        qwen = [
            worker
            for worker in snapshot.get("workers") or []
            if worker.get("model") == context.qwen and worker.get("state") != "STOPPED"
        ]
        context.checks.require(
            len(qwen) >= 2 and all(worker.get("state") in {"READY", "LOADING"} for worker in qwen),
            "Qwen was partially drained while one reclaimed GPU could not fit Laguna",
        )
    laguna_records = [record for record in outcome.records if record.get("model") == context.laguna]
    context.checks.require(
        len(laguna_records) == 1,
        f"expected one Laguna request log, got {len(laguna_records)}",
    )
    if laguna_records:
        record = laguna_records[0]
        context.checks.require(
            record.get("tensor_parallel_size") == context.managed_gpu_count
            and len(record.get("gpu_uuids") or []) == context.managed_gpu_count,
            "Laguna did not run as one all-GPU tensor-parallel worker",
        )
    return outcome


def gpu_used_vram_mib() -> dict[str, int]:
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    try:
        result: dict[str, int] = {}
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuid_value = raw_uuid.decode() if isinstance(raw_uuid, bytes) else str(raw_uuid)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result[uuid_value] = int(memory.used // (1024 * 1024))
        return result
    finally:
        pynvml.nvmlShutdown()


async def run_preset_4(context: Context) -> Outcome:
    outcome = Outcome()
    slow_task = asyncio.create_task(
        request_once(
            context,
            api_key=context.team_a_key,
            model=context.qwen,
            client_worker="maintenance-inflight",
            sequence=0,
            stream=True,
            max_tokens=context.args.maintenance_max_tokens,
            prompt=(
                "Write a long numbered technical checklist. Continue producing distinct "
                "items until the output limit is reached."
            ),
        )
    )
    await wait_for_status(
        context,
        lambda status: any(
            worker.get("model") == context.qwen
            and worker.get("state") == "READY"
            and worker.get("active_requests", 0) > 0
            for worker in status.get("workers") or []
        ),
        description="an active bounded Qwen stream",
        snapshots=outcome.snapshots,
    )
    drain_response = await context.client.post(
        f"{context.base_url}/admin/maintenance",
        headers=auth(context.admin_key),
        json={"mode": "drain"},
    )
    context.checks.require(
        drain_response.status_code == 202 and drain_response.json().get("mode") == "DRAINING",
        "maintenance drain was not durable before its response",
    )
    rejected = await context.client.post(
        f"{context.base_url}/v1/chat/completions",
        headers={
            **auth(context.team_b_key),
            "X-Test-Run-ID": context.run_id,
            "X-Request-ID": str(uuid.uuid4()),
        },
        json={
            "model": context.gemma,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 16,
        },
    )
    context.checks.require(
        rejected.status_code == 503 and bool(rejected.headers.get("Retry-After")),
        "new maintenance inference was not immediately rejected with 503 and Retry-After",
    )
    await gather_with_deadline(context, [slow_task], "maintenance in-flight request")
    if slow_task.done() and not slow_task.cancelled():
        outcome.results.append(slow_task.result())

    ready_status = await wait_for_status(
        context,
        lambda status: status.get("mode") == "MAINTENANCE_READY",
        description="MAINTENANCE_READY",
        snapshots=outcome.snapshots,
    )
    if ready_status is not None:
        context.checks.require(
            all(
                worker.get("state") == "STOPPED" and worker.get("active_requests") == 0
                for worker in ready_status.get("workers") or []
            ),
            "maintenance-ready status still had a live model worker",
        )
        used = gpu_used_vram_mib()
        outcome.details["maintenance_ready_gpu_used_vram_mib"] = used
        context.checks.require(
            all(value <= context.args.idle_vram_mib for value in used.values()),
            f"GPU memory did not return below {context.args.idle_vram_mib} MiB: {used}",
        )

    resume = await context.client.post(
        f"{context.base_url}/admin/maintenance",
        headers=auth(context.admin_key),
        json={"mode": "active"},
    )
    context.checks.require(
        resume.status_code == 202 and resume.json().get("mode") == "ACTIVE",
        "maintenance resume did not restore ACTIVE mode",
    )
    outcome.results.append(
        await request_once(
            context,
            api_key=context.team_a_key,
            model=context.qwen,
            client_worker="maintenance-resumed",
            sequence=1,
            stream=False,
            max_tokens=16,
        )
    )
    outcome.snapshots.append(await get_status(context))
    outcome.records = await fetch_request_records(context, len(outcome.results))
    validate_results_and_records(
        context.checks, outcome.results, outcome.records, outcome.snapshots
    )
    return outcome


def model_metrics(
    results: list[RequestResult], records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    record_by_id = {record["request_id"]: record for record in records}
    metrics: dict[str, dict[str, Any]] = {}
    for model in sorted({result.model for result in results}):
        selected = [result for result in results if result.model == model]
        completed = [result for result in selected if result.ok]
        accepted_times = [
            parse_time(record_by_id[result.request_id].get("worker_accepted_time"))
            for result in completed
            if result.request_id in record_by_id
        ]
        completion_times = [
            parse_time(record_by_id[result.request_id].get("completion_time"))
            for result in completed
            if result.request_id in record_by_id
        ]
        accepted_times = [value for value in accepted_times if value is not None]
        completion_times = [value for value in completion_times if value is not None]
        wall = (
            (max(completion_times) - min(accepted_times)).total_seconds()
            if accepted_times and completion_times
            else 0
        )
        tokens = sum(result.completion_tokens for result in completed)
        metrics[model] = {
            "attempted": len(selected),
            "succeeded": len(completed),
            "failed": len(selected) - len(completed),
            "completion_tokens": tokens,
            "requests_per_second": len(completed) / wall if wall > 0 else 0,
            "tokens_per_second": tokens / wall if wall > 0 else 0,
            "p95_ttft_ms": percentile(
                [result.ttft_ms for result in completed if result.ttft_ms is not None],
                0.95,
            ),
            "p95_latency_ms": percentile([result.latency_ms for result in completed], 0.95),
        }
    return metrics


def build_summary(
    context: Context,
    outcome: Outcome,
    *,
    starting_balance: dict[str, Any],
    snapshot_path: Path,
) -> dict[str, Any]:
    successful = [result for result in outcome.results if result.ok]
    accepted = [
        parse_time(record.get("worker_accepted_time"))
        for record in outcome.records
        if record.get("completion_status") == "COMPLETED"
    ]
    completed = [
        parse_time(record.get("completion_time"))
        for record in outcome.records
        if record.get("completion_status") == "COMPLETED"
    ]
    accepted = [value for value in accepted if value is not None]
    completed = [value for value in completed if value is not None]
    common_wall = (max(completed) - min(accepted)).total_seconds() if accepted and completed else 0
    tokens = sum(result.completion_tokens for result in successful)
    by_worker = Counter(
        record.get("worker_id")
        for record in outcome.records
        if record.get("completion_status") == "COMPLETED"
    )
    by_worker.pop(None, None)
    return {
        "preset": int(context.args.preset),
        "run_id": context.run_id,
        "status": (
            "SKIPPED" if outcome.skip_reason else "PASS" if not context.checks.failures else "FAIL"
        ),
        "skip_reason": outcome.skip_reason,
        "attempted": len(outcome.results),
        "succeeded": len(successful),
        "failed": len(outcome.results) - len(successful),
        "by_model": model_metrics(outcome.results, outcome.records),
        "by_worker": dict(by_worker),
        "combined_tokens_per_second": tokens / common_wall if common_wall > 0 else 0,
        "p95_ttft_ms": percentile(
            [result.ttft_ms for result in successful if result.ttft_ms is not None],
            0.95,
        ),
        "p95_latency_ms": percentile([result.latency_ms for result in successful], 0.95),
        "failures": context.checks.failures,
        "starting_balance": starting_balance,
        "initial_status": context.initial_status,
        "catalog": list(context.catalog.values()),
        "details": outcome.details,
        "artifacts": {
            "requests_jsonl": str(context.sink.path),
            "scheduler_snapshots": str(snapshot_path),
        },
    }


async def ensure_active(
    client: httpx.AsyncClient, base_url: str, admin_key: str, deadline: float
) -> dict[str, Any]:
    limit = time.monotonic() + deadline
    while True:
        response = await client.get(f"{base_url}/admin/status", headers=auth(admin_key))
        response.raise_for_status()
        status = response.json()
        mode = status.get("mode")
        if mode == "ACTIVE":
            status["observed_at"] = iso_now()
            return status
        if mode == "MAINTENANCE_READY":
            resumed = await client.post(
                f"{base_url}/admin/maintenance",
                headers=auth(admin_key),
                json={"mode": "active"},
            )
            resumed.raise_for_status()
        elif mode != "DRAINING":
            raise RuntimeError(f"unknown service mode: {mode}")
        if time.monotonic() >= limit:
            raise TimeoutError("service did not become ACTIVE before the deadline")
        await asyncio.sleep(0.2)


async def read_usage(
    client: httpx.AsyncClient, base_url: str, keys: dict[str, str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for nickname, api_key in keys.items():
        response = await client.get(f"{base_url}/v1/me/usage", headers=auth(api_key))
        response.raise_for_status()
        result[nickname] = response.json()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, choices=["0", "1", "2", "3", "4"])
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"))
    parser.add_argument("--admin-key", default=os.getenv("ADMIN_KEY"))
    parser.add_argument("--team-a-key", default=os.getenv("TEAM_A_KEY"))
    parser.add_argument("--team-b-key", default=os.getenv("TEAM_B_KEY"))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN_MODEL", "qwen3.6-27b-nvfp4"))
    parser.add_argument("--gemma-model", default=os.getenv("GEMMA_MODEL", "gemma-4-31b-it-nvfp4"))
    parser.add_argument("--laguna-model", default=os.getenv("LAGUNA_MODEL", "laguna-s-2.1-nvfp4"))
    parser.add_argument("--requests-per-worker", type=positive_int, default=10)
    parser.add_argument("--max-tokens", type=positive_int, default=16)
    parser.add_argument("--coexistence-qwen-max-tokens", type=positive_int, default=4096)
    parser.add_argument("--coexistence-gemma-max-tokens", type=positive_int, default=512)
    parser.add_argument("--maintenance-max-tokens", type=positive_int, default=512)
    parser.add_argument("--timeout", type=positive_float, default=1800)
    parser.add_argument("--deadline", type=positive_float, default=7200)
    parser.add_argument("--poll-interval", type=positive_float, default=0.5)
    parser.add_argument("--idle-vram-mib", type=positive_int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics/acceptance"))
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write request JSONL without echoing every line to the terminal.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    for name in ("base_url", "admin_key", "team_a_key", "team_b_key"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} or its environment variable is required")
    base_url = normalize_base_url(args.base_url)
    run_id = str(uuid.uuid4())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"preset-{args.preset}-{run_id}"
    jsonl_path = args.output_dir / f"{stem}.jsonl"
    summary_path = args.output_dir / f"{stem}.summary.json"
    snapshot_path = args.output_dir / f"{stem}.scheduler.json"
    sink = JsonlSink(jsonl_path, echo=not args.quiet)
    checks = Checks()
    timeout = httpx.Timeout(args.timeout, connect=30, pool=args.timeout)
    limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            trust_env=False,
        ) as client:
            initial_status = await ensure_active(client, base_url, args.admin_key, args.deadline)
            health_response = await client.get(f"{base_url}/health")
            health_response.raise_for_status()
            managed_gpu_count = int(health_response.json()["managed_gpus"])
            catalog_response = await client.get(
                f"{base_url}/v1/models", headers=auth(args.admin_key)
            )
            catalog_response.raise_for_status()
            catalog_rows = catalog_response.json().get("data") or []
            catalog = {row["id"]: row for row in catalog_rows}
            starting_balance = await read_usage(
                client,
                base_url,
                {"teamA": args.team_a_key, "teamB": args.team_b_key},
            )
            context = Context(
                args=args,
                client=client,
                sink=sink,
                checks=checks,
                run_id=run_id,
                base_url=base_url,
                admin_key=args.admin_key,
                team_a_key=args.team_a_key,
                team_b_key=args.team_b_key,
                qwen=args.qwen_model,
                gemma=args.gemma_model,
                laguna=args.laguna_model,
                initial_status=initial_status,
                catalog=catalog,
                managed_gpu_count=managed_gpu_count,
            )
            catalog_checks(context)
            runners = {
                "0": run_preset_0,
                "1": run_preset_1,
                "2": run_preset_2,
                "3": run_preset_3,
                "4": run_preset_4,
            }
            outcome = await runners[args.preset](context)
            snapshot_path.write_text(
                json.dumps(outcome.snapshots, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            summary = build_summary(
                context,
                outcome,
                starting_balance=starting_balance,
                snapshot_path=snapshot_path,
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
            print(f"summary: {summary_path}", flush=True)
            if outcome.skip_reason:
                return 0
            return 1 if checks.failures else 0
    finally:
        sink.close()


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(async_main(args))
    except (ValueError, httpx.HTTPError, OSError, TimeoutError) as exc:
        print(f"acceptance test failed before completion: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("acceptance test interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
