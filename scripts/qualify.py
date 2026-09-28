"""Collect model-independent native traffic, residency and accounting evidence.

Use on a dedicated qualification service. This tool generates inference traffic and
enters maintenance at the end. It does not qualify every deployment workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from collections import Counter
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def completion_usage(payload: dict[str, Any]) -> int:
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("completion_tokens"), int):
        raise ValueError("Missing authoritative completion-token usage")
    if usage["completion_tokens"] < 0:
        raise ValueError("Negative completion-token usage")
    return int(usage["completion_tokens"])


def accounting_matches(requests: list[dict[str, Any]], rows: list[dict[str, Any]]) -> bool:
    """Compare authoritative response usage with one settled row per request."""
    if len(rows) != len(requests) or any(
        not isinstance(item.get("completion_tokens"), int) for item in requests
    ):
        return False
    expected = Counter((item["model"], item["completion_tokens"]) for item in requests)
    observed = Counter(
        (row.get("model"), row.get("token_usage", {}).get("completion_tokens")) for row in rows
    )
    return expected == observed and all(
        row.get("completion_status") == "COMPLETED"
        and row.get("error_code") is None
        and row.get("accepted_count") == 1
        and row.get("completion_count") == 1
        and isinstance(row.get("token_usage", {}).get("prompt_tokens"), int)
        for row in rows
    )


async def request_sample(
    client: httpx.AsyncClient, model: str, stream: bool, max_tokens: int, run_id: str
) -> dict[str, Any]:
    started = time.monotonic()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello briefly."}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    result: dict[str, Any] = {"model": model, "stream": stream}
    try:
        if stream:
            last_usage = None
            completed = False
            chunks = 0
            async with client.stream(
                "POST", "/v1/chat/completions", json=payload, headers={"X-Test-Run-ID": run_id}
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        completed = True
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise ValueError(f"Stream error: {chunk['error']}")
                    chunks += 1
                    if chunk.get("usage") is not None:
                        last_usage = completion_usage(chunk)
            if not completed or not chunks or last_usage is None:
                raise ValueError("Incomplete stream or missing usage")
            result["completion_tokens"] = last_usage
        else:
            response = await client.post(
                "/v1/chat/completions", json=payload, headers={"X-Test-Run-ID": run_id}
            )
            response.raise_for_status()
            body = response.json()
            if not body.get("choices"):
                raise ValueError("Completion has no choices")
            result["completion_tokens"] = completion_usage(body)
        result["ok"] = True
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        result.update(ok=False, error=str(exc))
    result["latency_seconds"] = time.monotonic() - started
    return result


async def qualify(args: argparse.Namespace) -> bool:
    token = os.environ.get("LLMRIO_API_KEY")
    if not token:
        raise SystemExit("Set LLMRIO_API_KEY; reports never store the credential")
    run_id = f"qualification-{uuid.uuid4()}"
    args.output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "request_results": [],
        "observed_transitions": 0,
    }
    async with httpx.AsyncClient(
        base_url=args.url, timeout=args.timeout, headers={"Authorization": f"Bearer {token}"}
    ) as client:

        async def get(path: str) -> dict[str, Any]:
            response = await client.get(path)
            response.raise_for_status()
            return dict(response.json())

        caps = await get("/admin/capabilities")
        if caps.get("experimental"):
            raise SystemExit("Run experimental qualification separately")
        catalog = await get("/v1/models")
        callable_names = [item["id"] for item in catalog["data"] if item.get("callable")]
        models = args.models or callable_names
        if not models or set(models) - set(callable_names):
            raise SystemExit("Every selected model must have a callable validated profile")
        if len(set(models)) < 2:
            raise SystemExit("Mixed-load qualification requires at least two callable models")
        initial_status = await get("/admin/status")
        if initial_status["mode"] != "ACTIVE":
            raise SystemExit("Resume the dedicated qualification service before running traffic")
        staff_catalog = await get("/staff/models")
        # Select explicit safe catalog fields; never serialize key or vault responses.
        artifacts = [
            {
                key: model.get(key)
                for key in (
                    "nickname",
                    "source_type",
                    "resolved_revision",
                    "artifact_path",
                    "engine",
                )
            }
            for model in staff_catalog["data"]
            if model["nickname"] in models
        ]
        manifest = {
            "capabilities": caps,
            "models": artifacts,
            "profiles": {},
            "initial_status": initial_status,
            "dashboard": await get("/admin/dashboard"),
            "limits": {"concurrency": args.concurrency, "max_tokens": args.max_tokens},
        }
        for model in staff_catalog["data"]:
            if model["nickname"] in models:
                manifest["profiles"][model["nickname"]] = await get(
                    f"/admin/models/{model['id']}/profiles"
                )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        stopping = asyncio.Event()
        telemetry_errors: list[str] = []
        states: dict[str, str] = {}

        async def sample() -> None:
            with (args.output / "telemetry.jsonl").open("w") as output:
                while not stopping.is_set():
                    try:
                        status = await get("/admin/status")
                        dashboard = await get("/admin/dashboard")
                        for worker in status["workers"]:
                            worker_id, state = worker["worker_id"], worker["state"]
                            previous = states.get(worker_id)
                            if (
                                previous is not None
                                and state != previous
                                and state in {"READY", "SLEEPING", "STOPPED"}
                            ):
                                report["observed_transitions"] += 1
                            states[worker_id] = state
                        output.write(
                            json.dumps(
                                {
                                    "at": datetime.now(UTC).isoformat(),
                                    "status": status,
                                    "dashboard": dashboard,
                                }
                            )
                            + "\n"
                        )
                        output.flush()
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        telemetry_errors.append(str(exc))
                    with suppress(TimeoutError):
                        await asyncio.wait_for(stopping.wait(), timeout=args.sample_interval)

        task = asyncio.create_task(sample())
        started = time.monotonic()
        index = 0
        try:
            while time.monotonic() - started < args.duration:
                # Rotate the model per burst, exercising both saturation and switching.
                model = models[index % len(models)]
                results = await asyncio.gather(
                    *(
                        request_sample(
                            client, model, (index + offset) % 2 == 0, args.max_tokens, run_id
                        )
                        for offset in range(args.concurrency)
                    )
                )
                report["request_results"].extend(results)
                index += 1
            report["traffic_seconds"] = time.monotonic() - started
            response = await client.post("/admin/maintenance", json={"mode": "drain"})
            response.raise_for_status()
            deadline = time.monotonic() + args.timeout
            while True:
                final = await get("/admin/status")
                if final["mode"] == "MAINTENANCE_READY":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Maintenance did not finish draining")
                await asyncio.sleep(1)
            report["final_status"] = final
            report["accounting"] = await get(f"/admin/requests?test_run_id={run_id}")
        finally:
            stopping.set()
            await task
            report["telemetry_errors"] = telemetry_errors
            report["traffic_gate_passed"] = False
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        requests = report["request_results"]
        rows = report["accounting"]["requests"]
        ownership = final.get("resource_ownership", {})
        report["gate_checks"] = {
            "all_requests_succeeded": bool(requests) and all(item["ok"] for item in requests),
            "telemetry_complete": not telemetry_errors,
            "one_hour_load": report["traffic_seconds"] >= 3600,
            "hundred_transitions": report["observed_transitions"]
            >= args.minimum_transitions
            >= 100,
            "workers_stopped": all(
                worker["state"] == "STOPPED" and worker["pid"] is None
                for worker in final["workers"]
            ),
            "queues_drained": not final.get("queued_models"),
            "resources_released": ownership.get("request_leases") == 0
            and ownership.get("reserved_ports") == []
            and ownership.get("validation_gpu_uuids") == [],
            "accounting_matches": accounting_matches(requests, rows),
        }
        report["traffic_gate_passed"] = all(report["gate_checks"].values())
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return bool(report["traffic_gate_passed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("LLMRIO_API_URL", "http://127.0.0.1:8002"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--minimum-transitions", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.duration, args.concurrency, args.max_tokens, args.timeout, args.sample_interval)
        <= 0
    ):
        parser.error("duration, concurrency, limits and timeouts must be positive")
    passed = asyncio.run(qualify(args))
    print(
        f"Traffic evidence: {args.output}; traffic gate {'passed' if passed else 'pending/failed'}"
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
