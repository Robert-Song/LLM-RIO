from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import re
import secrets
import signal
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_rio.config import Settings
from llm_rio.domain import Engine, MachineInventory, PlacementProfile
from llm_rio.inventory import candidate_gpu_sets, gpu_environment
from llm_rio.prism import add_kvcached_vllm_flags, detect_kvcached
from llm_rio.runtime import ResidencyScheduler
from llm_rio.tool_support import detect_vllm_parser_configuration


class ValidationError(RuntimeError):
    def __init__(self, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class ValidationPreempted(ValidationError):
    def __init__(self) -> None:
        super().__init__("validation", "validation yielded to production inference")


def validation_log_path(
    *,
    log_dir: Path,
    nickname: str,
    engine: str,
    tensor_parallel_size: int,
    gpu_indices: tuple[int, ...],
) -> Path:
    """Build a readable, sortable, and collision-resistant validation log path."""
    safe_nickname = re.sub(r"[^a-zA-Z0-9._-]+", "-", nickname).strip("._-").lower()
    safe_nickname = safe_nickname[:80] or "model"
    safe_engine = re.sub(r"[^a-zA-Z0-9._-]+", "-", engine).strip("._-").lower()
    safe_engine = safe_engine[:32] or "engine"
    gpu_label = "-".join(str(index) for index in gpu_indices) or "unknown"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    unique_suffix = uuid.uuid4().hex[:8]
    filename = (
        f"{timestamp}-validation-{safe_nickname}-{safe_engine}-tp{tensor_parallel_size}-"
        f"gpus{gpu_label}-{unique_suffix}.log"
    )
    return log_dir / filename


@dataclass(frozen=True, slots=True)
class CandidateShape:
    gpu_count: int
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    gpu_memory_utilization: float
    dtype: str
    quantization: str | None
    eligible_gpu_sets: tuple[tuple[str, ...], ...]


def build_candidate_shapes(
    *,
    inventory: MachineInventory,
    weight_bytes: int,
    max_model_len: int,
    reserved_vram_mib: int,
    dtype: str,
    quantization: str | None,
    gpu_memory_utilization: float | None = None,
    max_model_len_limit: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
) -> list[CandidateShape]:
    required_mib = max(1, int(weight_bytes / (1024 * 1024) * 1.15))
    candidates: list[CandidateShape] = []
    for gpu_count in range(1, len(inventory.gpus) + 1):
        gpu_sets = candidate_gpu_sets(inventory, gpu_count)
        viable_sets: list[tuple[str, ...]] = []
        for gpu_set in gpu_sets:
            devices = [device for device in inventory.gpus if device.uuid in gpu_set]
            usable = sum(max(0, device.total_vram_mib - reserved_vram_mib) for device in devices)
            if usable >= required_mib:
                viable_sets.append(gpu_set)
        if viable_sets:
            smallest_vram_mib = min(
                device.total_vram_mib
                for device in inventory.gpus
                if any(device.uuid in gpu_set for gpu_set in viable_sets)
            )
            automatic_utilization = max(
                0.01,
                min(1.0, (smallest_vram_mib - reserved_vram_mib) / smallest_vram_mib),
            )
            candidates.append(
                CandidateShape(
                    gpu_count=gpu_count,
                    tensor_parallel_size=gpu_count,
                    max_model_len=(
                        min(max_model_len, max_model_len_limit)
                        if max_model_len_limit is not None
                        else max_model_len
                    ),
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    gpu_memory_utilization=(
                        gpu_memory_utilization
                        if gpu_memory_utilization is not None
                        else automatic_utilization
                    ),
                    dtype=dtype,
                    quantization=quantization,
                    eligible_gpu_sets=tuple(viable_sets),
                )
            )
    return candidates


class ProfileValidator:
    """Runs preemptible, idle-only engine contract and capacity probes."""

    def __init__(
        self,
        settings: Settings,
        inventory: MachineInventory,
        scheduler: ResidencyScheduler,
    ) -> None:
        self.settings = settings
        self.inventory = inventory
        self.scheduler = scheduler
        self.kvcached = detect_kvcached(settings.engines.kvcached_mode)

    async def validate_vllm(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: Path,
        nickname: str,
        candidate: CandidateShape,
    ) -> list[PlacementProfile]:
        profiles: list[PlacementProfile] = []
        failures: list[ValidationError] = []
        for gpu_set in candidate.eligible_gpu_sets:
            while not await self.scheduler.acquire_validation_gpus(gpu_set):  # noqa: ASYNC110
                await asyncio.sleep(5.0)
            try:
                profile = await self._probe_vllm(
                    model_id=model_id,
                    model_revision=model_revision,
                    model_path=model_path,
                    nickname=nickname,
                    candidate=candidate,
                    gpu_set=gpu_set,
                )
                profiles.append(profile)
            except ValidationPreempted:
                raise
            except ValidationError as exc:
                failures.append(exc)
            finally:
                await self.scheduler.release_validation_gpus(gpu_set)
        if not profiles and failures:
            raise failures[-1]
        return profiles

    async def _probe_vllm(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: Path,
        nickname: str,
        candidate: CandidateShape,
        gpu_set: tuple[str, ...],
    ) -> PlacementProfile:
        port = self.settings.worker_port_end + 1
        api_key = f"rio_validation_{secrets.token_urlsafe(32)}"
        command = [
            self.settings.engines.vllm_executable,
            "serve",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            api_key,
            "--served-model-name",
            nickname,
            "--tensor-parallel-size",
            str(candidate.tensor_parallel_size),
            "--dtype",
            candidate.dtype,
            "--max-model-len",
            str(candidate.max_model_len),
            "--gpu-memory-utilization",
            str(candidate.gpu_memory_utilization),
        ]
        if candidate.max_num_seqs is not None:
            command.extend(["--max-num-seqs", str(candidate.max_num_seqs)])
        if candidate.max_num_batched_tokens is not None:
            command.extend(["--max-num-batched-tokens", str(candidate.max_num_batched_tokens)])
        if candidate.quantization:
            command.extend(["--quantization", candidate.quantization])
        parsers = detect_vllm_parser_configuration(model_path)
        if parsers.tool_parser is not None:
            command.extend(["--enable-auto-tool-choice", "--tool-call-parser", parsers.tool_parser])
        if parsers.reasoning_parser is not None:
            command.extend(["--reasoning-parser", parsers.reasoning_parser])
        add_kvcached_vllm_flags(command, self.kvcached)
        gpu_indices = tuple(
            device.index for device in self.inventory.gpus if device.uuid in gpu_set
        )
        log_path = validation_log_path(
            log_dir=self.settings.log_dir,
            nickname=nickname,
            engine="vllm",
            tensor_parallel_size=candidate.tensor_parallel_size,
            gpu_indices=gpu_indices,
        )
        environment = gpu_environment(gpu_set, self.settings.engines.environment)
        environment["VLLM_API_KEY"] = api_key
        environment.update(self.kvcached.environment(pythonpath=environment.get("PYTHONPATH")))
        started = time.monotonic()
        with log_path.open("ab", buffering=0) as log_handle:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ValidationError(
                    "engine_launch", str(exc), {"log_path": str(log_path)}
                ) from exc
            try:
                await self._wait_for_health(process, port, api_key)
                load_seconds = time.monotonic() - started
                kv_cache_capacity, max_concurrency = self._capacity_from_log(log_path)
                idle_memory = self._used_vram(gpu_set)
                throughput, peak_memory = await self._generation_contract(
                    process=process,
                    port=port,
                    api_key=api_key,
                    nickname=nickname,
                    gpu_set=gpu_set,
                )
            except ValidationError as exc:
                await self._terminate(process)
                exc.details.setdefault("log_path", str(log_path))
                raise
            except BaseException:
                await self._terminate(process)
                raise
            await self._terminate(process)
        try:
            version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            version = "executable-managed"
        return PlacementProfile(
            id=str(uuid.uuid4()),
            model_id=model_id,
            model_revision=model_revision,
            engine=Engine.VLLM,
            engine_version=version,
            machine_fingerprint=self.inventory.fingerprint,
            gpu_count=candidate.gpu_count,
            tensor_parallel_size=candidate.tensor_parallel_size,
            pipeline_parallel_size=1,
            eligible_gpu_sets=(gpu_set,),
            dtype=candidate.dtype,
            quantization=candidate.quantization,
            max_model_len=candidate.max_model_len,
            max_num_seqs=candidate.max_num_seqs,
            max_num_batched_tokens=candidate.max_num_batched_tokens,
            predicted_tokens_per_second=throughput,
            load_and_warmup_seconds=load_seconds,
            idle_vram_mib_per_gpu=idle_memory,
            peak_vram_mib_per_gpu=peak_memory,
            gpu_headroom_mib_per_gpu=tuple(self.settings.reserved_vram_mib for _ in gpu_set),
            capabilities=frozenset(
                {"chat", "streaming", "tools"}
                if parsers.tool_parser is not None
                else {"chat", "streaming"}
            ),
            launch_args={},
            gpu_memory_utilization=candidate.gpu_memory_utilization,
            kv_cache_capacity_tokens=kv_cache_capacity,
            max_full_length_concurrency=max_concurrency,
            memory_backend=self.kvcached.memory_backend,
        )

    @staticmethod
    def _capacity_from_log(log_path: Path) -> tuple[int | None, float | None]:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        capacities = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", text)
        concurrencies = re.findall(
            r"Maximum concurrency for [\d,]+ tokens per request:\s*([\d.]+)x", text
        )
        capacity = int(capacities[-1].replace(",", "")) if capacities else None
        concurrency = float(concurrencies[-1]) if concurrencies else None
        return capacity, concurrency

    async def _wait_for_health(
        self, process: asyncio.subprocess.Process, port: int, api_key: str
    ) -> None:
        startup_timeout = self.settings.worker_startup_timeout_seconds
        deadline = time.monotonic() + startup_timeout if startup_timeout is not None else None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while deadline is None or time.monotonic() < deadline:
                if self.scheduler.validation_should_yield():
                    raise ValidationPreempted()
                if process.returncode is not None:
                    raise ValidationError(
                        "engine_startup", f"engine exited with status {process.returncode}"
                    )
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{port}/health",
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    if response.is_success:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        raise ValidationError("engine_startup", "engine health check timed out")

    async def _generation_contract(
        self,
        *,
        process: asyncio.subprocess.Process,
        port: int,
        api_key: str,
        nickname: str,
        gpu_set: tuple[str, ...],
    ) -> tuple[float, tuple[int, ...]]:
        payload = {
            "model": nickname,
            "messages": [{"role": "user", "content": "Reply with a short greeting."}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        completion_tokens = 0
        saw_done = False
        peak = list(self._used_vram(gpu_set))
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=120.0) as client:  # noqa: SIM117
            async with client.stream(
                "POST",
                f"http://127.0.0.1:{port}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            ) as response:
                if not response.is_success:
                    body = (await response.aread()).decode(errors="replace")[-2000:]
                    raise ValidationError(
                        "generation",
                        f"generation returned HTTP {response.status_code}",
                        {"body": body},
                    )
                async for line in response.aiter_lines():
                    if self.scheduler.validation_should_yield():
                        raise ValidationPreempted()
                    current = self._used_vram(gpu_set)
                    peak = [max(old, new) for old, new in zip(peak, current, strict=True)]
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        saw_done = True
                        continue
                    chunk = json.loads(data)
                    usage = chunk.get("usage")
                    if usage:
                        completion_tokens = int(usage.get("completion_tokens", 0))
        elapsed = max(time.monotonic() - started, 0.001)
        if process.returncode is not None or not saw_done or completion_tokens <= 0:
            raise ValidationError("streaming_contract", "stream or usage contract failed")
        return completion_tokens / elapsed, tuple(peak)

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, OSError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except TimeoutError:
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                process.kill()
            await process.wait()

    @staticmethod
    def _used_vram(gpu_set: tuple[str, ...]) -> tuple[int, ...]:
        import pynvml

        pynvml.nvmlInit()
        try:
            result = []
            for uuid_value in gpu_set:
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid_value)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result.append(int(memory.used // (1024 * 1024)))
            return tuple(result)
        finally:
            pynvml.nvmlShutdown()
