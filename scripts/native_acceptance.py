"""Real-engine final traffic gate. Uses public APIs and independent nvidia-smi telemetry.

Run only against an isolated acceptance service. Credentials come from LLMRIO_API_KEY.
The driver performs real registration probes, creates disposable users, calls models,
cycles residency, runs mixed load, reconciles accounting, and drains the service.
Manual acceptance cases remain separate, explicitly unexecuted gates.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import subprocess
import time
import tomllib
import uuid
import zipfile
from contextlib import AsyncExitStack, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wheel_source_identity(wheel: Path) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(wheel) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.startswith("llm_rio/") and name.endswith(".py")
        )
        if not names:
            raise ValueError("No LLM-RIO application source in wheel")
        for name in names:
            digest.update(name.removeprefix("llm_rio/").encode() + b"\0")
            digest.update(archive.read(name))
    return digest.hexdigest()


def authoritative_usage(body: dict[str, Any]) -> dict[str, int]:
    value = body.get("usage")
    if not isinstance(value, dict):
        raise ValueError("Missing authoritative usage")
    result = {
        name: value.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if any(type(item) is not int or item < 0 for item in result.values()):
        raise ValueError("Usage must contain nonnegative integer prompt/completion/total tokens")
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise ValueError("Usage total does not equal prompt plus completion tokens")
    return result


def reconcile(
    requests: list[dict[str, Any]], rows: list[dict[str, Any]], *, charge_maximum: bool = False
) -> list[str]:
    """Join by unique request ID, never by model/token coincidences."""
    errors = []
    expected = {item["request_id"]: item for item in requests}
    observed = {row["request_id"]: row for row in rows}
    if len(expected) != len(requests) or len(observed) != len(rows):
        errors.append("Duplicate request identity")
    if expected.keys() != observed.keys():
        errors.append("Client/server request ID sets differ")
    for rid in expected.keys() & observed.keys():
        sample, row = expected[rid], observed[rid]
        if not sample.get("ok"):
            errors.append(f"{rid}: unsuccessful healthy request")
            continue
        usage = sample["usage"]
        if any(
            row.get("token_usage", {}).get(key) != usage[key]
            for key in ("prompt_tokens", "completion_tokens")
        ):
            errors.append(f"{rid}: authoritative usage mismatch")
        if (
            row.get("model") != sample["model"]
            or row.get("completion_status") != "COMPLETED"
            or row.get("error_code")
            or row.get("accepted_count") != 1
            or row.get("completion_count") != 1
            or row.get("reservation_state") != "SETTLED"
        ):
            errors.append(f"{rid}: request did not settle exactly once")
        # Product charging contract caps usage at the original reservation.
        charged = (
            row.get("reserved_tokens", -1)
            if charge_maximum
            else min(usage["total_tokens"], row.get("reserved_tokens", -1))
        )
        if row.get("charged_tokens") != charged:
            errors.append(f"{rid}: charged usage mismatch")
        delta = 0 if row.get("account_unlimited") else -charged
        if row.get("ledger_delta_tokens") != delta:
            errors.append(f"{rid}: reservation/refund ledger mismatch")
    return errors


def gpu_snapshot(uuids: list[str]) -> dict[str, Any]:
    def query(kind: str, fields: str) -> list[list[str]]:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return [
            [value.strip() for value in line.split(",")]
            for line in completed.stdout.splitlines()
            if line.strip()
        ]

    devices = query("gpu", "uuid,name,driver_version,memory.total,memory.used,utilization.gpu")
    selected = [row for row in devices if row[0] in uuids]
    if {row[0] for row in selected} != set(uuids):
        raise ValueError("Managed GPU UUIDs do not match independent nvidia-smi inventory")
    processes = [
        row for row in query("compute-apps", "gpu_uuid,pid,used_gpu_memory") if row[0] in uuids
    ]
    return {
        "devices": selected,
        "processes": processes,
        "host_meminfo": Path("/proc/meminfo").read_text(),
    }


class Gate:
    def __init__(self, manifest: dict[str, Any], output: Path, admin: httpx.AsyncClient) -> None:
        self.manifest, self.output, self.admin = manifest, output, admin
        self.run_id = f"native-{uuid.uuid4()}"
        self.report: dict[str, Any] = {
            "run_id": self.run_id,
            "mode": manifest["mode"],
            "started_at": datetime.now(UTC).isoformat(),
            "scope": "real GPU traffic and registration; manual cases are separate",
            "requests": [],
            "cycles": [],
            "checks": {},
            "telemetry_errors": [],
            "traffic_gate_passed": False,
            "production_release_approved": False,
        }
        self.deadline = float(manifest["workload"]["operation_timeout_seconds"])
        self.stopping = asyncio.Event()
        self.personas: list[httpx.AsyncClient] = []

    async def api(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self.admin.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    async def wait_status(self, predicate: Any, description: str) -> dict[str, Any]:
        until = time.monotonic() + self.deadline
        while time.monotonic() < until:
            status = await self.api("GET", "/admin/status")
            if predicate(status):
                return status
            await asyncio.sleep(0.5)
        raise TimeoutError(f"Timed out waiting for {description}")

    async def drain(self) -> dict[str, Any]:
        await self.api("POST", "/admin/maintenance", json={"mode": "drain"})
        return await self.wait_status(
            lambda s: s["mode"] == "MAINTENANCE_READY"
            and all(w["state"] == "STOPPED" and w["pid"] is None for w in s["workers"])
            and not s["queued_models"]
            and s["resource_ownership"]
            == {
                "request_leases": 0,
                "reserved_ports": [],
                "validation_gpu_uuids": [],
            },
            "verified maintenance teardown",
        )

    async def prepare(self) -> None:
        caps = await self.api("GET", "/admin/capabilities")
        if caps["name"] != self.manifest["mode"] or caps["experimental"]:
            raise ValueError("Service mode does not match the native manifest")
        expected_source = wheel_source_identity(Path(self.manifest["build_files"]["wheel"]))
        if caps["application"]["source_sha256"] != expected_source:
            raise ValueError(
                "Running application differs from the recorded wheel; reinstall/restart"
            )
        config = tomllib.loads(Path(self.manifest["config_file"]).read_text())
        running = caps["configuration"]
        if (
            running["database_path"] != str(Path(config["database_path"]).resolve())
            or set(running["managed_gpu_uuids"]) != set(self.manifest["managed_gpu_uuids"])
            or running["quota_charge_requested_maximum"]
            != config.get("quota_charge_requested_maximum", False)
        ):
            raise ValueError("Running database/GPU/quota configuration differs from test inputs")
        self.report["capabilities"] = caps
        await self.drain()
        baseline = await asyncio.to_thread(gpu_snapshot, self.manifest["managed_gpu_uuids"])
        if baseline["processes"]:
            raise ValueError("Selected acceptance GPUs have compute processes after drain")
        self.report["initial_gpu"] = baseline
        catalog = (await self.api("GET", "/staff/models"))["data"]
        by_name = {model["nickname"]: model for model in catalog}
        for fixture in self.manifest["models"]:
            nickname = fixture["nickname"]
            if nickname not in by_name:
                created = await self.api(
                    "POST",
                    "/staff/models",
                    json={
                        "nickname": nickname,
                        **fixture["registration"],
                    },
                )
                job_id = created["job_id"]
                await self.wait_job(job_id)
            else:
                current = by_name[nickname]
                source = fixture["registration"]
                if source.get("local_path"):
                    if (
                        Path(current.get("local_path") or "").resolve()
                        != Path(source["local_path"]).resolve()
                    ):
                        raise ValueError(f"{nickname}: existing local source differs from manifest")
                elif (
                    current["huggingface_repo"] != source["huggingface_repo"]
                    or current["resolved_revision"] != source["revision"]
                ):
                    raise ValueError(f"{nickname}: existing HF source differs from manifest")
                job_id = current["registration_job"]["id"]
                if current["registration_job"]["state"] in ("QUEUED", "RUNNING"):
                    await self.wait_job(job_id)
            body: dict[str, Any] = {"validation_overrides": fixture.get("validation_overrides", {})}
            if fixture.get("profile_id"):
                body["profile_id"] = fixture["profile_id"]
            await self.api("POST", f"/staff/model-jobs/{job_id}/retry", json=body)
            await self.wait_job(job_id)
        catalog = (await self.api("GET", "/staff/models"))["data"]
        selected = [m for m in catalog if m["nickname"] in self.names]
        self.report["catalog"] = [
            {
                k: model.get(k)
                for k in (
                    "id",
                    "nickname",
                    "engine",
                    "source_type",
                    "resolved_revision",
                    "artifact_path",
                    "artifact_hashes",
                    "request_limits",
                )
            }
            for model in selected
        ]
        if len({(m["artifact_path"], m["resolved_revision"]) for m in selected}) < 2:
            raise ValueError("Mixed-artifact acceptance requires at least two distinct artifacts")
        self.report["profiles"] = {
            model["nickname"]: await self.api("GET", f"/admin/models/{model['id']}/profiles")
            for model in selected
        }
        await self.api("POST", "/admin/maintenance", json={"mode": "active"})
        listed = (await self.api("GET", "/v1/models"))["data"]
        if set(self.names) - {m["id"] for m in listed if m["callable"]}:
            raise ValueError("Every fixture must be callable after actual validation")
        self.report["checks"]["real_validation_and_registration"] = True

    @property
    def names(self) -> list[str]:
        return [model["nickname"] for model in self.manifest["models"]]

    async def wait_job(self, job_id: str) -> None:
        until = time.monotonic() + float(self.manifest["workload"]["validation_timeout_seconds"])
        while time.monotonic() < until:
            job = await self.api("GET", f"/staff/model-jobs/{job_id}")
            if job["state"] == "COMPLETED":
                return
            if job["state"] not in ("QUEUED", "RUNNING"):
                raise ValueError(f"Validation {job_id} failed: {job.get('failure')}")
            await asyncio.sleep(2)
        raise TimeoutError(f"Validation {job_id} exceeded the declared deadline")

    async def traffic(
        self, client: httpx.AsyncClient, payload: dict[str, Any], case: str
    ) -> dict[str, Any]:
        rid = str(uuid.uuid4())
        sample: dict[str, Any] = {
            "request_id": rid,
            "case": case,
            "model": payload["model"],
            "stream": bool(payload.get("stream")),
            "ok": False,
        }
        start = time.monotonic()
        headers = {
            "X-Request-ID": rid,
            "Idempotency-Key": rid,
            "X-Test-Run-ID": self.run_id,
            "X-Client-Worker": case,
        }
        content = ""
        try:
            if payload.get("stream"):
                payload = {**payload, "stream_options": {"include_usage": True}}
                done, finished, usage = False, set(), None
                async with client.stream(
                    "POST", "/v1/chat/completions", json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    sample["worker_id"] = response.headers.get("X-Worker-ID")
                    sample["queue_wait_ms"] = float(response.headers["X-Queue-Wait-Ms"])
                    if response.headers.get("X-Request-ID") != rid:
                        raise ValueError("Response request ID mismatch")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            done = True
                            break
                        chunk = json.loads(raw)
                        if "error" in chunk:
                            raise ValueError(f"Stream error: {chunk['error']}")
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            if any(
                                delta.get(k)
                                for k in ("content", "reasoning", "reasoning_content", "tool_calls")
                            ):
                                sample.setdefault("ttft_seconds", time.monotonic() - start)
                            content += delta.get("content") or ""
                            if choice.get("finish_reason") is not None:
                                finished.add(choice["index"])
                        if chunk.get("usage") is not None:
                            usage = authoritative_usage(chunk)
                if not done or finished != set(range(payload.get("n", 1))) or usage is None:
                    raise ValueError("Stream missing finish, DONE or authoritative usage")
                sample["usage"] = usage
            else:
                response = await client.post("/v1/chat/completions", json=payload, headers=headers)
                response.raise_for_status()
                sample["worker_id"] = response.headers.get("X-Worker-ID")
                sample["queue_wait_ms"] = float(response.headers["X-Queue-Wait-Ms"])
                if response.headers.get("X-Request-ID") != rid:
                    raise ValueError("Response request ID mismatch")
                body = response.json()
                if len(body.get("choices") or []) != payload.get("n", 1) or any(
                    choice.get("finish_reason") is None for choice in body["choices"]
                ):
                    raise ValueError("Completion missing choices or finish reason")
                sample["usage"] = authoritative_usage(body)
                sample["response"] = body
                content = body["choices"][0]["message"].get("content") or ""
            sample["ok"] = True
            sample["content"] = content
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            sample["error"] = str(exc)
        sample["latency_seconds"] = time.monotonic() - start
        self.report["requests"].append(sample)
        with (self.output / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(sample, allow_nan=False) + "\n")
        return sample

    def payload(self, model: str, index: int, *, stream: bool | None = None) -> dict[str, Any]:
        prompts = [
            [{"role": "user", "content": "Explain why the sky appears blue in three sentences."}],
            [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": "Remember the word maple."},
                {"role": "assistant", "content": "I will remember maple."},
                {"role": "user", "content": "What word did I ask you to remember?"},
            ],
            [{"role": "user", "content": "Translate Hello into Korean, Japanese, and French. 🌍"}],
            [
                {
                    "role": "user",
                    "content": "Describe a safe plan for organizing a large library. " * 20,
                }
            ],
        ]
        tokens = self.manifest["workload"]["output_tokens"]
        return {
            "model": model,
            "messages": prompts[index % len(prompts)],
            "max_tokens": tokens[index % len(tokens)],
            "temperature": 0.2,
            "stream": (index % 2 == 0) if stream is None else stream,
        }

    async def interface_checks(self) -> None:
        checks = []

        async def expect(
            client: httpx.AsyncClient, method: str, path: str, status: int, **kwargs: Any
        ) -> None:
            response = await client.request(method, path, **kwargs)
            checks.append(
                {
                    "method": method,
                    "path": path,
                    "expected": status,
                    "observed": response.status_code,
                }
            )
            if response.status_code != status:
                raise ValueError(
                    f"Interface check {method} {path}: expected {status}, "
                    f"got {response.status_code}"
                )

        self.report["interface_checks"] = checks
        await expect(self.personas[0], "GET", "/admin/status", 403)
        await expect(self.personas[0], "GET", "/v1/models", 200)
        await expect(
            self.personas[0], "GET", "/v1/models", 401, headers={"Authorization": "Bearer invalid"}
        )
        body = self.payload(self.names[0], 1)
        await expect(
            self.personas[0], "POST", "/v1/chat/completions", 422, json={**body, "max_tokens": 0}
        )
        await expect(
            self.personas[0], "POST", "/v1/chat/completions", 422, json={**body, "messages": []}
        )
        await expect(
            self.personas[0],
            "POST",
            "/v1/chat/completions",
            404,
            json={**body, "model": f"absent-{self.run_id}"},
        )
        self.report["checks"]["role_and_invalid_request_contracts"] = True

    async def residency_cycles(self) -> None:
        sleeping = self.manifest["mode"] == "vllm-sleep"
        for index in range(self.manifest["workload"]["controlled_cycles"]):
            model = self.names[index % len(self.names)]
            before = await self.api("GET", "/admin/status")
            cached = [
                w for w in before["workers"] if w["model"] == model and w["state"] == "SLEEPING"
            ]
            # Warm the selected artifact once before checking a cached wake cycle.
            if sleeping and not cached:
                first = await self.traffic(
                    self.personas[0], self.payload(model, index), "cycle-warm"
                )
                if not first["ok"]:
                    raise ValueError("Residency warmup failed")
                before = await self.wait_status(
                    lambda s, model=model: any(
                        w["model"] == model and w["state"] == "SLEEPING" for w in s["workers"]
                    ),
                    f"{model} sleeping",
                )
                cached = [
                    w for w in before["workers"] if w["model"] == model and w["state"] == "SLEEPING"
                ]
            result = await self.traffic(
                self.personas[index % 2], self.payload(model, index), "controlled-cycle"
            )
            if not result["ok"]:
                raise ValueError("Controlled residency request failed")
            after = await self.api("GET", "/admin/status")
            if sleeping:
                retained = {w["worker_id"]: w["pid"] for w in cached}
                awake = [
                    w
                    for w in after["workers"]
                    if w["model"] == model
                    and w["state"] == "READY"
                    and w["worker_id"] == result["worker_id"]
                    and retained.get(w["worker_id"]) == w["pid"]
                    and w["pid"]
                ]
                if not awake:
                    raise ValueError("Cached wake did not retain a worker identity and PID")
                terminal = await self.wait_status(
                    lambda s, worker_id=awake[0]["worker_id"]: any(
                        w["worker_id"] == worker_id and w["state"] == "SLEEPING"
                        for w in s["workers"]
                    ),
                    "same worker sleeping after wake",
                )
            else:
                active = [
                    w for w in after["workers"] if w["model"] == model and w["state"] != "STOPPED"
                ]
                if not active:
                    raise ValueError(
                        "No live queue worker observed after inference; increase idle timer"
                    )
                terminal = await self.wait_status(
                    lambda s: all(
                        w["state"] == "STOPPED" and w["pid"] is None for w in s["workers"]
                    ),
                    "queue workers fully stopped",
                )
                external = await asyncio.to_thread(gpu_snapshot, self.manifest["managed_gpu_uuids"])
                if external["processes"]:
                    raise ValueError("GPU compute processes remain after queue teardown")
            self.report["cycles"].append(
                {
                    "index": index,
                    "model": model,
                    "before": before,
                    "after": after,
                    "terminal": terminal,
                }
            )
            print(f"{self.manifest['mode']}: cycle {index + 1} complete", flush=True)

    async def sample(self) -> None:
        with (self.output / "telemetry.jsonl").open("w") as stream:
            while not self.stopping.is_set():
                try:
                    status, dashboard, gpu = await asyncio.gather(
                        self.api("GET", "/admin/status"),
                        self.api("GET", "/admin/dashboard"),
                        asyncio.to_thread(gpu_snapshot, self.manifest["managed_gpu_uuids"]),
                    )
                    stream.write(
                        json.dumps(
                            {
                                "at": datetime.now(UTC).isoformat(),
                                "status": status,
                                "dashboard": dashboard,
                                "independent_gpu": gpu,
                            }
                        )
                        + "\n"
                    )
                    stream.flush()
                except Exception as exc:
                    self.report["telemetry_errors"].append(str(exc))
                with suppress(TimeoutError):
                    await asyncio.wait_for(self.stopping.wait(), timeout=1)

    async def run(self) -> bool:
        await self.prepare()
        async with AsyncExitStack() as stack:
            for persona in ("alice", "bob"):
                secret = await self.api(
                    "POST",
                    "/admin/keys",
                    json={
                        "nickname": f"{self.run_id}-{persona}",
                        "role": "user",
                        "limit_tokens": 1_000_000_000,
                        "models": self.names,
                    },
                )
                self.personas.append(
                    await stack.enter_async_context(
                        httpx.AsyncClient(
                            base_url=str(self.admin.base_url),
                            timeout=self.deadline,
                            headers={"Authorization": f"Bearer {secret['api_key']}"},
                        )
                    )
                )
            initial_usage = [(await client.get("/v1/me/usage")).json() for client in self.personas]
            self.report["initial_usage"] = initial_usage
            await self.interface_checks()
            telemetry = asyncio.create_task(self.sample())
            try:
                for model in self.names:
                    for index in range(4):
                        sample = await self.traffic(
                            self.personas[index % 2], self.payload(model, index), "request-shapes"
                        )
                        if not sample["ok"]:
                            raise ValueError("Basic request contract failed")
                for feature in self.manifest.get("feature_requests", []):
                    sample = await self.traffic(
                        self.personas[0], feature["payload"], feature["case_id"]
                    )
                    if not sample["ok"]:
                        raise ValueError(f"Feature request failed: {feature['case_id']}")
                    if feature.get("expect_json"):
                        json.loads(sample["content"])
                    if feature.get("expect_tool_call"):
                        calls = sample["response"]["choices"][0]["message"].get("tool_calls")
                        if not calls:
                            raise ValueError("Expected a tool call from engine-supported fixture")
                        followup = dict(feature["payload"])
                        followup["messages"] = [
                            *followup["messages"],
                            sample["response"]["choices"][0]["message"],
                            *[
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "content": feature["tool_result"],
                                }
                                for call in calls
                            ],
                        ]
                        followup.pop("tool_choice", None)
                        await self.traffic(self.personas[0], followup, "tool-result-continuation")
                if self.manifest["mode"] == "queue":
                    await self.drain()
                    await self.api("POST", "/admin/maintenance", json={"mode": "active"})
                await self.residency_cycles()
                start = time.monotonic()
                finish = start + self.manifest["workload"]["soak_seconds"]

                async def user(index: int) -> None:
                    rng = random.Random(self.manifest["seed"] + index)
                    iteration = index
                    while time.monotonic() < finish:
                        await self.traffic(
                            self.personas[index % 2],
                            self.payload(rng.choice(self.names), iteration),
                            f"soak-user-{index}",
                        )
                        iteration += 1
                        await asyncio.sleep(
                            rng.uniform(0, self.manifest["workload"]["think_seconds"])
                        )

                await asyncio.gather(
                    *[user(i) for i in range(self.manifest["workload"]["concurrency"])]
                )
                self.report["healthy_soak_seconds"] = time.monotonic() - start
            finally:
                try:
                    self.report["final_status"] = await self.drain()
                    self.report["final_gpu"] = await asyncio.to_thread(
                        gpu_snapshot, self.manifest["managed_gpu_uuids"]
                    )
                finally:
                    self.stopping.set()
                    await telemetry
            rows = (await self.api("GET", "/admin/requests", params={"test_run_id": self.run_id}))[
                "requests"
            ]
            self.report["accounting"] = rows
            errors = reconcile(
                self.report["requests"],
                rows,
                charge_maximum=tomllib.loads(Path(self.manifest["config_file"]).read_text()).get(
                    "quota_charge_requested_maximum", False
                ),
            )
            final_usage = [(await client.get("/v1/me/usage")).json() for client in self.personas]
            self.report["final_usage"] = final_usage
            for before, after in zip(initial_usage, final_usage, strict=True):
                charged = sum(
                    r["charged_tokens"] for r in rows if r["account_id"] == before["account_id"]
                )
                if before["balance_tokens"] - after["balance_tokens"] != charged:
                    errors.append(f"{before['account_id']}: account balance mismatch")
                if after["lifetime_charged_tokens"] - before["lifetime_charged_tokens"] != charged:
                    errors.append(f"{before['account_id']}: lifetime accounting mismatch")
            self.report["accounting_errors"] = errors
            metrics = {}
            for field in ("latency_seconds", "ttft_seconds", "queue_wait_ms"):
                values = sorted(r[field] for r in self.report["requests"] if field in r)
                metrics[field] = {
                    "count": len(values),
                    "p50": values[len(values) // 2] if values else None,
                    "p95": values[min(len(values) - 1, int(len(values) * 0.95))]
                    if values
                    else None,
                    "max": max(values) if values else None,
                }
            metrics["soak_output_tokens_per_second"] = (
                sum(
                    r.get("usage", {}).get("completion_tokens", 0)
                    for r in self.report["requests"]
                    if r["case"].startswith("soak-user-")
                )
                / self.report["healthy_soak_seconds"]
            )
            self.report["metrics"] = metrics
            self.report["checks"].update(
                {
                    "all_requests_succeeded": all(r["ok"] for r in self.report["requests"]),
                    "accounting_exact": not errors,
                    "telemetry_complete": not self.report["telemetry_errors"],
                    "hundred_controlled_cycles": len(self.report["cycles"]) >= 100,
                    "one_hour_mixed_load": self.report["healthy_soak_seconds"] >= 3600,
                    "independent_gpu_cleanup": not self.report["final_gpu"]["processes"],
                    "maintenance_cleanup": True,
                }
            )
            self.report["traffic_gate_passed"] = all(self.report["checks"].values())
            return bool(self.report["traffic_gate_passed"])


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest["mode"] not in ("queue", "vllm-sleep"):
        raise ValueError("Only native modes are accepted")
    if len(manifest["models"]) < 2 or len({m["nickname"] for m in manifest["models"]}) != len(
        manifest["models"]
    ):
        raise ValueError("At least two unique model names are required")
    if not manifest["managed_gpu_uuids"] or any(
        not gpu.startswith("GPU-") for gpu in manifest["managed_gpu_uuids"]
    ):
        raise ValueError("Provide exact physical GPU UUIDs")
    work = manifest["workload"]
    for field in (
        "operation_timeout_seconds",
        "validation_timeout_seconds",
        "controlled_cycles",
        "soak_seconds",
        "concurrency",
    ):
        if (
            type(work[field]) not in (int, float)
            or not math.isfinite(work[field])
            or work[field] <= 0
        ):
            raise ValueError(f"Invalid workload field: {field}")
    if not work["output_tokens"] or any(type(n) is not int or n < 1 for n in work["output_tokens"]):
        raise ValueError("Output token limits must be positive integers")
    if work["think_seconds"] < 0:
        raise ValueError("think_seconds cannot be negative")
    for field in ("controlled_cycles", "concurrency"):
        if type(work[field]) is not int:
            raise ValueError(f"{field} must be an integer")
    if not {"wheel", "lock"} <= manifest["build_files"].keys():
        raise ValueError("Record the exact installed wheel and lock file")
    for value in manifest["build_files"].values():
        if not Path(value).is_file():
            raise ValueError(f"Build evidence file missing: {value}")
    config = tomllib.loads(Path(manifest["config_file"]).read_text())
    if config.get("serving_mode") != manifest["mode"] or set(config["managed_gpu_uuids"]) != set(
        manifest["managed_gpu_uuids"]
    ):
        raise ValueError("Config mode/GPU UUIDs must match the manifest")
    if not Path(config["database_path"]).is_absolute():
        raise ValueError("Use an absolute isolated acceptance database path")
    for model in manifest["models"]:
        registration = model["registration"]
        if not registration.get("local_path") and not registration.get("revision"):
            raise ValueError("Hugging Face sources need a pinned revision")


async def main_async(args: argparse.Namespace) -> bool:
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    token = os.environ.get("LLMRIO_API_KEY")
    if not token:
        raise ValueError("Set LLMRIO_API_KEY to the isolated service administrator credential")
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    async with httpx.AsyncClient(
        base_url=manifest["url"], timeout=60, headers={"Authorization": f"Bearer {token}"}
    ) as admin:
        gate = Gate(manifest, args.output, admin)
        gate.report["input_sha256"] = sha256(args.manifest)
        gate.report["config_sha256"] = sha256(Path(manifest["config_file"]))
        gate.report["build_sha256"] = {
            key: sha256(Path(value)) for key, value in manifest["build_files"].items()
        }
        try:
            return await gate.run()
        except Exception as exc:
            gate.report["fatal_error"] = str(exc)
            return False
        finally:
            gate.report["ended_at"] = datetime.now(UTC).isoformat()
            write_json(args.output / "report.json", gate.report)
            # Raw credentials/config are deliberately absent from evidence files.
            write_json(
                args.output / "evidence-hashes.json",
                {path.name: sha256(path) for path in args.output.iterdir() if path.is_file()},
            )
            print(
                json.dumps(
                    {
                        "report": str((args.output / "report.json").resolve()),
                        "traffic_gate_passed": gate.report["traffic_gate_passed"],
                        "production_release_approved": False,
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check", action="store_true", help="Validate local inputs only; sends no traffic"
    )
    args = parser.parse_args()
    if args.check:
        validate_manifest(json.loads(args.manifest.read_text()))
        print("Inputs valid; no hardware acceptance was executed.")
    else:
        if args.output is None:
            parser.error("--output is required for a GPU run")
        raise SystemExit(0 if asyncio.run(main_async(args)) else 1)


if __name__ == "__main__":
    main()
