from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import secrets
import signal
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import httpx

from llm_rio.inventory import gpu_environment
from llm_rio.prism import add_kvcached_vllm_flags, detect_kvcached


class CompatibilityFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ServerSpec:
    label: str
    model: str
    served_name: str
    port: int


@dataclass(slots=True)
class ServerHandle:
    spec: ServerSpec
    process: asyncio.subprocess.Process
    log_path: Path
    log_handle: BinaryIO
    load_seconds: float


def _gpu_snapshot(gpu_uuids: tuple[str, ...]) -> dict[str, int]:
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    try:
        return {
            gpu_uuid: int(
                pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)).used
                // (1024 * 1024)
            )
            for gpu_uuid in gpu_uuids
        }
    finally:
        pynvml.nvmlShutdown()


def _active_gpu_processes(gpu_uuids: tuple[str, ...]) -> dict[str, list[int]]:
    import pynvml

    pynvml.nvmlInit()
    try:
        result: dict[str, list[int]] = {}
        for gpu_uuid in gpu_uuids:
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                result[gpu_uuid] = sorted({int(process.pid) for process in processes})
        return result
    finally:
        pynvml.nvmlShutdown()


async def _wait_for_health(
    process: asyncio.subprocess.Process,
    *,
    port: int,
    api_key: str,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if process.returncode is not None:
                raise CompatibilityFailure(
                    f"vLLM exited during startup with status {process.returncode}"
                )
            try:
                response = await client.get(f"http://127.0.0.1:{port}/health", headers=headers)
                if response.is_success:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1.0)
    raise CompatibilityFailure(f"vLLM on port {port} did not become healthy")


async def _launch(
    spec: ServerSpec,
    *,
    gpu_uuids: tuple[str, ...],
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    executable: str,
    api_key: str,
    startup_timeout_seconds: float,
    diagnostics_dir: Path,
) -> ServerHandle:
    runtime = detect_kvcached("required")
    command = [
        executable,
        "serve",
        spec.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.port),
        "--api-key",
        api_key,
        "--served-model-name",
        spec.served_name,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    add_kvcached_vllm_flags(command, runtime)
    environment = gpu_environment(gpu_uuids)
    environment.update(runtime.environment(pythonpath=environment.get("PYTHONPATH")))
    environment["VLLM_API_KEY"] = api_key
    log_path = diagnostics_dir / f"{spec.label}.log"
    log_handle = log_path.open("ab", buffering=0)
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    except BaseException:
        log_handle.close()
        raise
    try:
        await _wait_for_health(
            process,
            port=spec.port,
            api_key=api_key,
            timeout_seconds=startup_timeout_seconds,
        )
    except BaseException:
        handle = ServerHandle(spec, process, log_path, log_handle, 0.0)
        await _terminate(handle)
        raise
    return ServerHandle(
        spec=spec,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        load_seconds=time.monotonic() - started,
    )


async def _completion(
    handle: ServerHandle,
    *,
    api_key: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float = 300.0,
) -> tuple[float, dict[str, Any]]:
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"http://127.0.0.1:{handle.spec.port}/v1/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": handle.spec.served_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        )
    if not response.is_success:
        raise CompatibilityFailure(
            f"{handle.spec.label} inference returned HTTP {response.status_code}: "
            f"{response.text[-2000:]}"
        )
    payload = response.json()
    if not payload.get("choices"):
        raise CompatibilityFailure(f"{handle.spec.label} returned no completion choices")
    return time.monotonic() - started, payload


async def _memory_probe(
    handle: ServerHandle,
    *,
    api_key: str,
    gpu_uuids: tuple[str, ...],
    prompt_tokens: int,
    output_tokens: int,
    reclaim_wait_seconds: float,
) -> dict[str, Any]:
    resident = _gpu_snapshot(gpu_uuids)
    prompt = "token " * prompt_tokens
    task = asyncio.create_task(
        _completion(
            handle,
            api_key=api_key,
            prompt=prompt,
            max_tokens=output_tokens,
        )
    )
    peak = dict(resident)
    while not task.done():
        current = _gpu_snapshot(gpu_uuids)
        peak = {gpu: max(peak[gpu], current[gpu]) for gpu in gpu_uuids}
        await asyncio.sleep(0.1)
    latency, payload = await task
    deadline = time.monotonic() + reclaim_wait_seconds
    post = _gpu_snapshot(gpu_uuids)
    while time.monotonic() < deadline:
        current = _gpu_snapshot(gpu_uuids)
        post = {gpu: min(post[gpu], current[gpu]) for gpu in gpu_uuids}
        await asyncio.sleep(0.25)
    growth = {gpu: max(0, peak[gpu] - resident[gpu]) for gpu in gpu_uuids}
    reclaimed = {gpu: max(0, peak[gpu] - post[gpu]) for gpu in gpu_uuids}
    return {
        "resident_mib": resident,
        "peak_mib": peak,
        "post_request_mib": post,
        "growth_mib": growth,
        "reclaimed_mib": reclaimed,
        "request_seconds": latency,
        "completion_tokens": int(payload.get("usage", {}).get("completion_tokens", 0)),
    }


async def _terminate(handle: ServerHandle) -> None:
    process_group = handle.process.pid
    try:
        if handle.process.returncode is None:
            try:
                os.killpg(process_group, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                handle.process.terminate()
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=30.0)
            except TimeoutError:
                try:
                    os.killpg(os.getpgid(handle.process.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    handle.process.kill()
                await handle.process.wait()
        # The API parent can exit before its TP workers. Since launch uses
        # start_new_session=True, PID is the stable process-group ID.
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(process_group, signal.SIGKILL)
    finally:
        handle.log_handle.close()


async def _wait_for_process_reclamation(
    gpu_uuids: tuple[str, ...], baseline: dict[str, int], timeout_seconds: float = 30.0
) -> dict[str, int]:
    deadline = time.monotonic() + timeout_seconds
    best = _gpu_snapshot(gpu_uuids)
    while time.monotonic() < deadline:
        current = _gpu_snapshot(gpu_uuids)
        best = {gpu: min(best[gpu], current[gpu]) for gpu in gpu_uuids}
        if all(best[gpu] <= baseline[gpu] + 512 for gpu in gpu_uuids):
            break
        await asyncio.sleep(0.5)
    return best


async def run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    gpu_uuids = tuple(args.gpu_uuid)
    if len(gpu_uuids) != args.tensor_parallel_size:
        raise CompatibilityFailure("provide exactly one --gpu-uuid per tensor-parallel rank")
    active = _active_gpu_processes(gpu_uuids)
    if active:
        raise CompatibilityFailure(
            f"compatibility test refuses GPUs with active compute processes: {active}"
        )
    runtime = detect_kvcached("required")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    diagnostics_dir = args.diagnostics_dir / f"prism-compat-{timestamp}"
    diagnostics_dir.mkdir(parents=True, exist_ok=False)
    baseline = _gpu_snapshot(gpu_uuids)
    api_key = f"rio_prism_compat_{secrets.token_urlsafe(24)}"
    handles: list[ServerHandle] = []
    report: dict[str, Any] = {
        "runtime": {
            "kvcached_version": runtime.package_version,
            "kvcached_revision": runtime.source_revision,
            "vllm_version": runtime.vllm_version,
            "officially_tested": runtime.officially_tested,
        },
        "gpu_uuids": gpu_uuids,
        "tensor_parallel_size": args.tensor_parallel_size,
        "baseline_mib": baseline,
        "checks": {},
        "diagnostics_dir": str(diagnostics_dir),
    }
    specs = (
        ServerSpec("model-a", args.model_a, "prism-compat-a", args.port_a),
        ServerSpec("model-b", args.model_b or args.model_a, "prism-compat-b", args.port_b),
    )
    try:
        first = await _launch(
            specs[0],
            gpu_uuids=gpu_uuids,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            executable=args.vllm_executable,
            api_key=api_key,
            startup_timeout_seconds=args.startup_timeout_seconds,
            diagnostics_dir=diagnostics_dir,
        )
        handles.append(first)
        first_latency, first_payload = await _completion(
            first, api_key=api_key, prompt="Hello", max_tokens=8
        )
        first_text = str(first_payload["choices"][0].get("text", ""))
        second = await _launch(
            specs[1],
            gpu_uuids=gpu_uuids,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            executable=args.vllm_executable,
            api_key=api_key,
            startup_timeout_seconds=args.startup_timeout_seconds,
            diagnostics_dir=diagnostics_dir,
        )
        handles.append(second)
        second_latency, second_payload = await _completion(
            second, api_key=api_key, prompt="Hello", max_tokens=8
        )
        second_text = str(second_payload["choices"][0].get("text", ""))
        resident_activation_seconds, _ = await _completion(
            first, api_key=api_key, prompt="Switch back", max_tokens=1
        )
        memory = await _memory_probe(
            second,
            api_key=api_key,
            gpu_uuids=gpu_uuids,
            prompt_tokens=args.probe_prompt_tokens,
            output_tokens=args.probe_output_tokens,
            reclaim_wait_seconds=args.reclaim_wait_seconds,
        )
        total_growth = sum(memory["growth_mib"].values())
        total_reclaimed = sum(memory["reclaimed_mib"].values())
        required_reclaimed = math.ceil(total_growth * args.minimum_reclaim_fraction)
        memory["required_reclaimed_mib"] = required_reclaimed
        elastic_ok = (
            total_growth >= args.minimum_probe_growth_mib and total_reclaimed >= required_reclaimed
        )
        autopatch_ok = all(
            "Successfully patched vllm"
            in handle.log_path.read_text(encoding="utf-8", errors="replace")
            for handle in handles
        )
        packed_kv_shim_ok = all(
            "LLM-RIO packed "
            in handle.log_path.read_text(encoding="utf-8", errors="replace")
            and " KV shim active:"
            in handle.log_path.read_text(encoding="utf-8", errors="replace")
            for handle in handles
        )
        report["servers"] = [
            {
                "label": first.spec.label,
                "model": first.spec.model,
                "load_seconds": first.load_seconds,
                "inference_seconds": first_latency,
                "log_path": str(first.log_path),
                "completion_text": first_text,
            },
            {
                "label": second.spec.label,
                "model": second.spec.model,
                "load_seconds": second.load_seconds,
                "inference_seconds": second_latency,
                "log_path": str(second.log_path),
                "completion_text": second_text,
            },
        ]
        report["resident_activation_seconds"] = resident_activation_seconds
        report["memory_probe"] = memory
        report["checks"].update(
            {
                "model_load": True,
                "kvcached_autopatch": autopatch_ok,
                "packed_kv_vllm026_shim": packed_kv_shim_ok,
                "inference": True,
                "same_model_determinism": (
                    specs[0].model != specs[1].model or first_text == second_text
                ),
                "co_resident_vllm": True,
                "tensor_parallel": True,
                "resident_activation": (
                    resident_activation_seconds <= args.activation_target_seconds
                ),
                "elastic_kv_reclamation": elastic_ok,
            }
        )
    finally:
        for handle in reversed(handles):
            await _terminate(handle)
        final_memory = await _wait_for_process_reclamation(gpu_uuids, baseline)
        report["final_mib"] = final_memory
        report["checks"]["process_memory_reclamation"] = all(
            final_memory[gpu] <= baseline[gpu] + 512 for gpu in gpu_uuids
        )
        report["passed"] = bool(report["checks"]) and all(report["checks"].values())
        report_path = diagnostics_dir / "report.json"
        report["report_path"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return (0 if report["passed"] else 1), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate kvcached co-residency with vLLM; never uses sleep mode."
    )
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b")
    parser.add_argument("--gpu-uuid", action="append", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-executable", default="vllm")
    parser.add_argument("--port-a", type=int, default=19100)
    parser.add_argument("--port-b", type=int, default=19101)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--activation-target-seconds", type=float, default=5.0)
    parser.add_argument("--probe-prompt-tokens", type=int, default=8192)
    parser.add_argument("--probe-output-tokens", type=int, default=256)
    parser.add_argument("--minimum-probe-growth-mib", type=int, default=64)
    parser.add_argument("--minimum-reclaim-fraction", type=float, default=0.5)
    parser.add_argument("--reclaim-wait-seconds", type=float, default=10.0)
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("diagnostics"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        code, report = asyncio.run(run(args))
    except (CompatibilityFailure, OSError, RuntimeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        raise SystemExit(2) from exc
    report["passed"] = code == 0
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
