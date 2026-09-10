from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import math
import re
import secrets
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from llm_rio.config import Settings
from llm_rio.domain import (
    CURRENT_VRAM_MEASUREMENT_VERSION,
    Engine,
    MachineInventory,
    PlacementProfile,
)
from llm_rio.gpu_memory import read_gpu_memory
from llm_rio.host_memory import sample_process_group_memory
from llm_rio.inventory import candidate_gpu_sets, gpu_environment
from llm_rio.prism import add_kvcached_vllm_flags, detect_kvcached
from llm_rio.process_cleanup import TeardownError, terminate_engine
from llm_rio.runtime import ResidencyScheduler
from llm_rio.tool_support import detect_vllm_parser_configuration


class ValidationError(RuntimeError):
    def __init__(self, stage: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class ValidationPreempted(ValidationError):
    def __init__(
        self,
        message: str = "validation yielded to production inference",
        *,
        stage: str = "validation",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(stage, message, details)


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


def _model_launch_args(model_path: Path) -> dict[str, Any]:
    """Derive mandatory vLLM arguments from immutable model metadata."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config.get("architectures")
    architecture_names = {str(value) for value in architectures if isinstance(architectures, list)}
    if config.get("model_type") == "deepseek_v4" or "DeepseekV4ForCausalLM" in architecture_names:
        return {"kv_cache_dtype": "fp8"}
    return {}


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
    launch_args: dict[str, Any] = field(default_factory=dict)


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


class _VramSampler:
    """Continuously measure one worker above a stable pre-launch GPU baseline."""

    def __init__(
        self,
        read_vram: Callable[[tuple[str, ...]], tuple[int, ...]],
        gpu_set: tuple[str, ...],
        *,
        interval_seconds: float = 0.1,
    ) -> None:
        self._read_vram = read_vram
        self.gpu_set = gpu_set
        self.interval_seconds = interval_seconds
        self.baseline_mib = read_vram(gpu_set)
        self._latest_mib = self.baseline_mib
        self._peak_delta_mib = [0 for _ in gpu_set]
        self._minimum_absolute_mib = list(self.baseline_mib)
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    def _delta(current: tuple[int, ...], baseline: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            max(0, observed - initial) for observed, initial in zip(current, baseline, strict=True)
        )

    def _observe(self, current: tuple[int, ...]) -> tuple[int, ...]:
        if len(current) != len(self.baseline_mib):
            raise RuntimeError("GPU VRAM sample shape changed during validation")
        self._latest_mib = current
        delta = self._delta(current, self.baseline_mib)
        self._peak_delta_mib = [
            max(peak, observed) for peak, observed in zip(self._peak_delta_mib, delta, strict=True)
        ]
        self._minimum_absolute_mib = [
            min(minimum, observed)
            for minimum, observed in zip(self._minimum_absolute_mib, current, strict=True)
        ]
        return delta

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="validation-vram-sampler")

    async def _run(self) -> None:
        while not self._stop.is_set():
            current = await asyncio.to_thread(self._read_vram, self.gpu_set)
            self._observe(current)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)

    def sample_now(self) -> tuple[int, ...]:
        return self._observe(self._read_vram(self.gpu_set))

    def peak(self) -> tuple[int, ...]:
        self.sample_now()
        return tuple(self._peak_delta_mib)

    def reset_peak(self) -> None:
        current = self.sample_now()
        self._peak_delta_mib = list(current)

    @property
    def baseline_drop_mib(self) -> tuple[int, ...]:
        return tuple(
            max(0, initial - minimum)
            for initial, minimum in zip(self.baseline_mib, self._minimum_absolute_mib, strict=True)
        )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        self.sample_now()


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
        self._validation_ports: set[int] = set()
        self._validation_port_lock = asyncio.Lock()

    @staticmethod
    def _local_port_available(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    async def _reserve_validation_port(self) -> int:
        first_port = self.settings.worker_port_end + 1
        async with self._validation_port_lock:
            for port in range(first_port, first_port + 128):
                if port in self._validation_ports or not self._local_port_available(port):
                    continue
                self._validation_ports.add(port)
                return port
        raise ValidationError(
            "engine_launch",
            f"no free validation port is available in {first_port}-{first_port + 127}",
        )

    async def _release_validation_port(self, port: int) -> None:
        async with self._validation_port_lock:
            self._validation_ports.discard(port)

    async def validate_vllm(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: Path,
        nickname: str,
        candidate: CandidateShape,
        backend: Literal["native", "kvcached"] = "native",
    ) -> list[PlacementProfile]:
        async def validate_gpu_set(
            gpu_set: tuple[str, ...],
        ) -> PlacementProfile | ValidationError:
            while not await self.scheduler.acquire_validation_gpus(gpu_set):  # noqa: ASYNC110
                await asyncio.sleep(5.0)
            teardown_verified = True
            try:
                return await self._probe_vllm(
                    model_id=model_id,
                    model_revision=model_revision,
                    model_path=model_path,
                    nickname=nickname,
                    candidate=candidate,
                    gpu_set=gpu_set,
                    backend=backend,
                )
            except TeardownError:
                teardown_verified = False
                raise
            except ValidationPreempted:
                raise
            except ValidationError as exc:
                return exc
            finally:
                if teardown_verified:
                    await self.scheduler.release_validation_gpus(gpu_set)

        tasks = [
            asyncio.create_task(validate_gpu_set(gpu_set), name=f"vllm-validation-{index}")
            for index, gpu_set in enumerate(candidate.eligible_gpu_sets, 1)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        profiles: list[PlacementProfile] = []
        failures: list[ValidationError] = []
        for result in results:
            if isinstance(result, ValidationError):
                failures.append(result)
            else:
                profiles.append(result)
        if not profiles and failures:
            raise failures[-1]
        return profiles

    async def _check_native_headroom(
        self, gpu_set: tuple[str, ...], candidate: CandidateShape
    ) -> None:
        """Wait for driver/context memory to disappear instead of launching into OOM."""
        try:
            samples = await asyncio.to_thread(read_gpu_memory, gpu_set)
            deficits = {
                gpu: {"free_vram_mib": samples[gpu].free_mib, "required_free_vram_mib": required}
                for gpu in gpu_set
                if samples[gpu].free_mib
                < (required := math.ceil(samples[gpu].total_mib * candidate.gpu_memory_utilization))
            }
        except Exception as exc:
            raise ValidationPreempted(
                "Cannot read GPU memory; validation will retry",
                stage="gpu_memory_wait",
                details={"error": str(exc)},
            ) from exc
        if deficits:
            raise ValidationPreempted(
                "Waiting for GPU memory to be released before validation",
                stage="gpu_memory_wait",
                details={"gpus": deficits},
            )

    async def _probe_vllm(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: Path,
        nickname: str,
        candidate: CandidateShape,
        gpu_set: tuple[str, ...],
        backend: Literal["native", "kvcached"],
    ) -> PlacementProfile:
        conservative = backend == "native" and self.scheduler.validation_requires_maintenance
        queue_mode = conservative and self.settings.queue_mode_enabled
        initial = (
            min(candidate.gpu_memory_utilization, 0.80)
            if conservative and not queue_mode
            else candidate.gpu_memory_utilization
        )
        budgets = [initial]
        if conservative:
            deltas = (0.02, 0.04, 0.06, 0.08, 0.10) if queue_mode else (0.10, 0.20)
            budgets.extend(round(initial - delta, 4) for delta in deltas if initial - delta >= 0.40)
        for index, budget in enumerate(budgets):
            attempt = replace(candidate, gpu_memory_utilization=budget)
            if conservative:
                await self._check_native_headroom(gpu_set, attempt)
            port = await self._reserve_validation_port()
            teardown_verified = True
            try:
                return await self._probe_vllm_on_port(
                    model_id=model_id,
                    model_revision=model_revision,
                    model_path=model_path,
                    nickname=nickname,
                    candidate=attempt,
                    gpu_set=gpu_set,
                    backend=backend,
                    port=port,
                )
            except TeardownError:
                teardown_verified = False
                raise
            except ValidationPreempted:
                raise
            except ValidationError as exc:
                log_path = exc.details.get("log_path")
                log = Path(log_path).read_text(errors="replace").lower() if log_path else ""
                memory_failure = exc.stage == "gpu_capacity" or (
                    exc.stage == "engine_startup"
                    and any(
                        marker in log
                        for marker in (
                            "cuda out of memory",
                            "cuda error: out of memory",
                            "outofmemoryerror",
                        )
                    )
                )
                if not memory_failure or index == len(budgets) - 1:
                    raise
                await self.scheduler.database.record_event(
                    "VALIDATION_MEMORY_RETRY",
                    model_id,
                    {
                        "gpu_uuids": gpu_set,
                        "failed_utilization": budget,
                        "next_utilization": budgets[index + 1],
                        "log_path": log_path,
                    },
                )
            finally:
                if teardown_verified:
                    await self._release_validation_port(port)
        raise AssertionError("validation budget attempts exhausted")

    async def _probe_vllm_on_port(
        self,
        *,
        model_id: str,
        model_revision: str,
        model_path: Path,
        nickname: str,
        candidate: CandidateShape,
        gpu_set: tuple[str, ...],
        backend: Literal["native", "kvcached"],
        port: int,
    ) -> PlacementProfile:
        gpu_indices = tuple(
            device.index for device in self.inventory.gpus if device.uuid in gpu_set
        )
        api_key = f"rio_validation_{secrets.token_urlsafe(32)}"
        launch_args = {
            **_model_launch_args(model_path),
            **candidate.launch_args,
        }
        kvcached = detect_kvcached("required") if backend == "kvcached" else detect_kvcached("none")
        ram_weight_cache_enabled = self.settings.ram_weight_cache_enabled
        launch_args["enable_sleep_mode"] = ram_weight_cache_enabled
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
        for key, value in launch_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, list):
                for item in value:
                    command.extend([flag, str(item)])
            elif isinstance(value, dict):
                command.extend([flag, json.dumps(value, separators=(",", ":"), sort_keys=True)])
            elif value is not None:
                command.extend([flag, str(value)])
        add_kvcached_vllm_flags(command, kvcached)
        log_path = validation_log_path(
            log_dir=self.settings.log_dir,
            nickname=nickname,
            engine=f"vllm-{backend}",
            tensor_parallel_size=candidate.tensor_parallel_size,
            gpu_indices=gpu_indices,
        )
        environment = gpu_environment(
            gpu_set,
            self.settings.engines.environment,
            executable=self.settings.engines.vllm_executable,
        )
        if self.settings.queue_mode_enabled:
            environment.update(
                ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0", VLLM_SERVER_DEV_MODE="0"
            )
            environment.pop("LLM_RIO_KVCACHED_VLLM026_SHIM", None)
        environment["VLLM_API_KEY"] = api_key
        environment.update(kvcached.environment(pythonpath=environment.get("PYTHONPATH")))
        if ram_weight_cache_enabled:
            environment["VLLM_SERVER_DEV_MODE"] = "1"
        started = time.monotonic()
        sleep_memory: tuple[int, ...] | None = None
        wake_peak_memory: tuple[int, ...] | None = None
        offload_seconds: float | None = None
        activation_seconds: float | None = None
        host_cache_mib: float | None = None
        sampler = _VramSampler(self._used_vram, gpu_set)
        sampler.start()
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
                await sampler.stop()
                raise ValidationError(
                    "engine_launch", str(exc), {"log_path": str(log_path)}
                ) from exc
            try:
                await self._wait_for_health(process, port, api_key)
                load_seconds = time.monotonic() - started
                kv_cache_capacity, max_concurrency = self._capacity_from_log(log_path)
                idle_memory = sampler.sample_now()
                throughput = await self._generation_contract(
                    process=process,
                    port=port,
                    api_key=api_key,
                    nickname=nickname,
                )
                peak_memory = sampler.peak()
                if ram_weight_cache_enabled:
                    (
                        offload_seconds,
                        activation_seconds,
                        sleep_memory,
                        host_cache_mib,
                    ) = await self._sleep_wake_contract(
                        port=port,
                        api_key=api_key,
                        process_pid=process.pid,
                        sampler=sampler,
                    )
                    post_wake_throughput = await self._generation_contract(
                        process=process,
                        port=port,
                        api_key=api_key,
                        nickname=nickname,
                    )
                    throughput = min(throughput, post_wake_throughput)
                    wake_peak_memory = sampler.peak()
                    peak_memory = tuple(
                        max(before, after)
                        for before, after in zip(peak_memory, wake_peak_memory, strict=True)
                    )
                await sampler.stop()
                self._validate_vram_measurements(
                    gpu_set=gpu_set,
                    peak_memory=peak_memory,
                    baseline_drop_mib=sampler.baseline_drop_mib,
                )
            except ValidationError as exc:
                exc.details.setdefault("log_path", str(log_path))
                raise
            finally:
                try:
                    await sampler.stop()
                finally:
                    await self._terminate(process, gpu_uuids=gpu_set)
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
            gpu_headroom_mib_per_gpu=(0,) * len(gpu_set),
            capabilities=frozenset(
                {"chat", "streaming", "tools"}
                if parsers.tool_parser is not None
                else {"chat", "streaming"}
            ),
            launch_args=launch_args,
            gpu_memory_utilization=candidate.gpu_memory_utilization,
            kv_cache_capacity_tokens=kv_cache_capacity,
            max_full_length_concurrency=max_concurrency,
            memory_backend=backend,
            sleep_vram_mib_per_gpu=sleep_memory,
            weight_cache_offload_seconds=offload_seconds,
            weight_cache_activation_seconds=activation_seconds,
            host_cache_mib=host_cache_mib,
            normal_verified=backend == "native",
            kvcached_verified=backend == "kvcached",
            vram_measurement_version=CURRENT_VRAM_MEASUREMENT_VERSION,
            vram_baseline_mib_per_gpu=sampler.baseline_mib,
            wake_peak_vram_mib_per_gpu=wake_peak_memory,
        )

    async def _sleep_wake_contract(
        self,
        *,
        port: int,
        api_key: str,
        process_pid: int,
        sampler: _VramSampler,
    ) -> tuple[float, float, tuple[int, ...], float]:
        if self.scheduler.validation_should_yield():
            raise ValidationPreempted()
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = self.settings.prism_transition_timeout_seconds
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                started = time.monotonic()
                sleep_response = await client.post(
                    f"http://127.0.0.1:{port}/sleep",
                    headers=headers,
                    params={"level": "1"},
                )
                if not sleep_response.is_success:
                    raise ValidationError(
                        "weight_cache",
                        f"sleep returned HTTP {sleep_response.status_code}",
                        {"body": sleep_response.text[-2000:]},
                    )
                offload_seconds = time.monotonic() - started
                sleeping_response = await client.get(
                    f"http://127.0.0.1:{port}/is_sleeping",
                    headers=headers,
                )
                sleeping_response.raise_for_status()
                if sleeping_response.json().get("is_sleeping") is not True:
                    raise ValidationError("weight_cache", "engine did not enter level-1 sleep")
                sleep_memory = sampler.sample_now()
                process_memory = await asyncio.to_thread(sample_process_group_memory, process_pid)
                sampler.reset_peak()
                started = time.monotonic()
                wake_response = await client.post(
                    f"http://127.0.0.1:{port}/wake_up",
                    headers=headers,
                )
                if not wake_response.is_success:
                    raise ValidationError(
                        "weight_cache",
                        f"wake returned HTTP {wake_response.status_code}",
                        {"body": wake_response.text[-2000:]},
                    )
                activation_seconds = time.monotonic() - started
                awake_response = await client.get(
                    f"http://127.0.0.1:{port}/is_sleeping",
                    headers=headers,
                )
                awake_response.raise_for_status()
                if awake_response.json().get("is_sleeping") is not False:
                    raise ValidationError("weight_cache", "engine remained asleep after wake")
        except ValidationError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise ValidationError("weight_cache", str(exc)) from exc
        if self.scheduler.validation_should_yield():
            raise ValidationPreempted()
        return offload_seconds, activation_seconds, sleep_memory, process_memory.accounted_mib

    def _validate_vram_measurements(
        self,
        *,
        gpu_set: tuple[str, ...],
        peak_memory: tuple[int, ...],
        baseline_drop_mib: tuple[int, ...],
    ) -> None:
        if len(peak_memory) != len(gpu_set) or len(baseline_drop_mib) != len(gpu_set):
            raise ValidationError(
                "gpu_measurement",
                "GPU VRAM measurement shape changed during validation",
            )
        unstable = {
            gpu_uuid: drop
            for gpu_uuid, drop in zip(gpu_set, baseline_drop_mib, strict=True)
            if drop > 0
        }
        if unstable:
            raise ValidationError(
                "gpu_measurement",
                "pre-launch GPU baseline changed during validation",
                {"baseline_drop_mib_per_gpu": unstable},
            )
        totals = {device.uuid: device.total_vram_mib for device in self.inventory.gpus}
        violations: dict[str, dict[str, int]] = {}
        for gpu_uuid, observed_mib in zip(gpu_set, peak_memory, strict=True):
            total_mib = totals.get(gpu_uuid)
            if total_mib is None:
                raise ValidationError(
                    "gpu_measurement", f"GPU {gpu_uuid} is missing from machine inventory"
                )
            budget_mib = max(0, total_mib - self.settings.reserved_vram_mib)
            if observed_mib > budget_mib:
                violations[gpu_uuid] = {
                    "measured_peak_mib": observed_mib,
                    "schedulable_budget_mib": budget_mib,
                    "total_vram_mib": total_mib,
                    "reserved_vram_mib": self.settings.reserved_vram_mib,
                }
        if violations:
            raise ValidationError(
                "gpu_capacity",
                "measured worker peak exceeds the schedulable GPU budget",
                {"violations": violations},
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
    ) -> float:
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
        return completion_tokens / elapsed

    @staticmethod
    async def _terminate(
        process: asyncio.subprocess.Process, *, gpu_uuids: tuple[str, ...] = ()
    ) -> None:
        await terminate_engine(process, gpu_uuids=gpu_uuids)

    @staticmethod
    def _used_vram(gpu_set: tuple[str, ...]) -> tuple[int, ...]:
        import pynvml  # type: ignore[import-untyped]

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
