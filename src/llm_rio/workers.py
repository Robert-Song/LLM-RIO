from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_rio.config import Settings
from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.gpu_memory import read_gpu_memory, required_free_vram
from llm_rio.host_memory import (
    HostMemorySample,
    gib_to_mib,
    sample_host_memory,
    sample_process_group_memory,
)
from llm_rio.inventory import gpu_environment
from llm_rio.prism import add_kvcached_vllm_flags, detect_kvcached
from llm_rio.process_cleanup import terminate_engine
from llm_rio.profiles import profile_verified_for_mode
from llm_rio.storage import Database, _now
from llm_rio.tool_support import detect_vllm_parser_configuration

logger = logging.getLogger(__name__)

WorkerEventCallback = Callable[[str, str], Awaitable[None]]

_GPU_RESIDENT_STATES = frozenset(
    {
        RuntimeState.LOADING,
        RuntimeState.READY,
        RuntimeState.DRAINING,
        RuntimeState.OFFLOADING,
        RuntimeState.WAKING,
        RuntimeState.STOPPING,
    }
)


class WorkerLaunchError(RuntimeError):
    pass


def worker_log_path(*, log_dir: Path, served_model_name: str, worker_id: str) -> Path:
    """Build a sortable log name that identifies the worker's model."""
    safe_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", served_model_name)
    safe_model_name = safe_model_name.strip("._-").lower()[:80] or "model"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{timestamp}-worker-{safe_model_name}-{worker_id}.log"


class WorkerSupervisor:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self.workers: dict[str, WorkerPlacement] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._log_handles: dict[str, Any] = {}
        self._log_paths: dict[str, Path] = {}
        self._lock = asyncio.Lock()
        self._transition_locks: dict[str, asyncio.Lock] = {}
        self._host_cache_lock = asyncio.Lock()
        self._capacity_deferrals: set[tuple[str, tuple[str, ...]]] = set()
        self._drain_to_sleep: set[str] = set()
        self._event_callback: WorkerEventCallback | None = None
        self.internal_api_key = f"rio_internal_{secrets.token_urlsafe(32)}"
        self.kvcached = detect_kvcached(settings.effective_kvcached_mode)

    def set_event_callback(self, callback: WorkerEventCallback) -> None:
        self._event_callback = callback

    @property
    def ram_weight_cache_enabled(self) -> bool:
        return self.settings.ram_weight_cache_enabled

    @property
    def persistent_host_weight_cache_enabled(self) -> bool:
        return self.kvcached.environment().get("LLM_RIO_KVCACHED_VLLM026_SHIM") == "1"

    async def _refresh_cached_worker_samples(self) -> None:
        async with self._lock:
            targets = [
                (worker.id, worker.process_pid)
                for worker in self.workers.values()
                if worker.host_weights_cached and worker.process_pid is not None
            ]
        samples = await asyncio.gather(
            *(
                asyncio.to_thread(sample_process_group_memory, process_pid)
                for _, process_pid in targets
            )
        )
        async with self._lock:
            for (worker_id, process_pid), sample in zip(targets, samples, strict=True):
                worker = self.workers.get(worker_id)
                if worker is None or worker.process_pid != process_pid:
                    continue
                worker.process_rss_mib = sample.rss_mib
                worker.process_pss_mib = sample.pss_mib
                worker.process_swap_mib = sample.swap_pss_mib or sample.swap_mib
                worker.host_cache_accounted_mib = sample.accounted_mib
                worker.host_cache_accounting_source = sample.source

    def _host_cache_limit_mib(self, host: HostMemorySample) -> float:
        configured = gib_to_mib(getattr(self.settings, "prism_host_cache_max_gib", None))
        if configured is not None:
            return configured
        reserve = (
            gib_to_mib(getattr(self.settings, "prism_host_cache_min_available_gib", 4.0)) or 0.0
        )
        return max(0.0, host.effective_total_mib - reserve)

    def _host_cache_pressure_reason(
        self,
        host: HostMemorySample,
        cache_accounted_mib: float,
        cache_swap_mib: float = 0.0,
    ) -> str | None:
        swap_limit = gib_to_mib(getattr(self.settings, "prism_swap_max_used_gib", 0.0)) or 0.0
        if cache_swap_mib > swap_limit:
            return "swap_pressure"
        minimum_available = (
            gib_to_mib(getattr(self.settings, "prism_host_cache_min_available_gib", 4.0)) or 0.0
        )
        if host.available_mib < minimum_available:
            return "ram_headroom"
        if cache_accounted_mib > self._host_cache_limit_mib(host):
            return "ram_budget"
        return None

    async def host_cache_status(self) -> dict[str, float | str | None]:
        await self._refresh_cached_worker_samples()
        host = await asyncio.to_thread(sample_host_memory)
        async with self._lock:
            cached_workers = [
                worker for worker in self.workers.values() if worker.host_weights_cached
            ]
            cache_accounted_mib = sum(worker.host_cache_accounted_mib for worker in cached_workers)
            cache_swap_mib = sum(worker.process_swap_mib for worker in cached_workers)
        return {
            "source": host.source,
            "effective_total_mib": host.effective_total_mib,
            "available_mib": host.available_mib,
            "swap_used_mib": host.swap_used_mib,
            "swap_total_mib": host.swap_total_mib,
            "cache_accounted_mib": cache_accounted_mib,
            "cache_swap_mib": cache_swap_mib,
            "cache_limit_mib": self._host_cache_limit_mib(host),
            "pressure_reason": self._host_cache_pressure_reason(
                host, cache_accounted_mib, cache_swap_mib
            ),
        }

    async def enforce_host_cache_budget(self, *, incoming_worker_id: str | None = None) -> bool:
        """Evict least-recently-demanded sleeping workers until host pressure clears."""
        if not self.ram_weight_cache_enabled:
            return True
        host_cache_lock = getattr(self, "_host_cache_lock", None)
        if host_cache_lock is None:
            host_cache_lock = self._host_cache_lock = asyncio.Lock()
        async with host_cache_lock:
            while True:
                status = await self.host_cache_status()
                reason = status["pressure_reason"]
                if reason is None:
                    return True
                async with self._lock:
                    candidates = sorted(
                        (
                            worker
                            for worker in self.workers.values()
                            if worker.id != incoming_worker_id
                            and worker.state is RuntimeState.SLEEPING
                            and worker.host_weights_cached
                            and not worker.admitted_request_ids
                        ),
                        key=lambda worker: worker.last_demand_at,
                    )
                    if not candidates:
                        incoming = (
                            self.workers.get(incoming_worker_id)
                            if incoming_worker_id is not None
                            else None
                        )
                        if incoming is not None:
                            incoming.last_cache_eviction_reason = str(reason)
                        return False
                    victim = candidates[0]
                    victim.last_cache_eviction_reason = str(reason)
                    victim.state = RuntimeState.STOPPING
                await self.database.record_event(
                    "WORKER_CACHE_EVICTED",
                    victim.id,
                    {
                        "reason": reason,
                        "cache_accounted_mib": status["cache_accounted_mib"],
                        "cache_limit_mib": status["cache_limit_mib"],
                        "host_available_mib": status["available_mib"],
                        "host_swap_used_mib": status["swap_used_mib"],
                        "cache_swap_mib": status["cache_swap_mib"],
                    },
                )
                await self._stop(victim.id, force=False)

    async def ensure_gpu_capacity(
        self,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        *,
        waking_worker_id: str | None = None,
    ) -> bool:
        """Evict idle sleeping workers in LRU order, re-reading VRAM before admission.

        The scheduler holds its state lock across this check and launch/wake.
        Never infer that stopping a process has already released its GPU memory.
        """
        key = (profile.id, gpu_uuids)
        if not profile_verified_for_mode(
            profile,
            kvcached_required=False,
            ram_weight_cache_required=self.ram_weight_cache_enabled,
            queue_mode_required=self.settings.queue_mode_enabled,
        ):
            return False
        while True:
            waking_worker = self.workers.get(waking_worker_id) if waking_worker_id else None
            if waking_worker_id and (
                waking_worker is None or waking_worker.state is not RuntimeState.SLEEPING
            ):
                return False
            try:
                samples = await asyncio.to_thread(read_gpu_memory, gpu_uuids)
                deficits = {
                    gpu: {
                        "free_vram_mib": samples[gpu].free_mib,
                        "required_free_vram_mib": required,
                    }
                    for index, gpu in enumerate(gpu_uuids)
                    if samples[gpu].free_mib
                    < (
                        required := required_free_vram(
                            profile,
                            index,
                            samples[gpu],
                            reserve_mib=self.settings.reserved_vram_mib,
                            waking_pid=waking_worker.process_pid if waking_worker else None,
                            waking=waking_worker is not None,
                        )
                    )
                }
            except Exception as exc:
                if key not in self._capacity_deferrals:
                    await self.database.record_event(
                        "GPU_CAPACITY_DEFERRED",
                        profile.model_id,
                        {"reason": "vram_unavailable", "error": str(exc), "gpu_uuids": gpu_uuids},
                    )
                    self._capacity_deferrals.add(key)
                return False
            if not deficits:
                self._capacity_deferrals.discard(key)
                return True
            async with self._lock:
                candidates = sorted(
                    (
                        worker
                        for worker in self.workers.values()
                        if worker.id != waking_worker_id
                        and worker.state is RuntimeState.SLEEPING
                        and not worker.admitted_request_ids
                        and set(worker.gpu_uuids) & deficits.keys()
                    ),
                    key=lambda worker: (worker.last_demand_at, worker.id),
                )
            if not candidates:
                if waking_worker is not None and not waking_worker.admitted_request_ids:
                    # If its residual cannot be safely credited (or foreign allocations
                    # leave too little room), release the target's context too. A later
                    # scheduler tick can cold-start it after observing reclaimed VRAM.
                    waking_worker.last_cache_eviction_reason = "gpu_vram_pressure_cold_restart"
                    await self.database.record_event(
                        "WORKER_CACHE_EVICTED",
                        waking_worker.id,
                        {"reason": "gpu_vram_pressure_cold_restart", "gpus": deficits},
                    )
                    await self.stop(waking_worker.id, force=False)
                    return False
                if key not in self._capacity_deferrals:
                    await self.database.record_event(
                        "GPU_CAPACITY_DEFERRED",
                        profile.model_id,
                        {"reason": "insufficient_free_vram", "gpus": deficits},
                    )
                    self._capacity_deferrals.add(key)
                return False
            victim = candidates[0]
            victim.last_cache_eviction_reason = "gpu_vram_pressure"
            await self.database.record_event(
                "WORKER_CACHE_EVICTED",
                victim.id,
                {
                    "reason": "gpu_vram_pressure",
                    "incoming_model_id": profile.model_id,
                    "gpus": deficits,
                },
            )
            await self.stop(victim.id, force=False)
            if victim.state is not RuntimeState.COLD:
                return False
            await asyncio.sleep(0.1)

    @property
    def occupied_gpu_uuids(self) -> set[str]:
        return {
            gpu
            for worker in self.workers.values()
            if worker.state in _GPU_RESIDENT_STATES
            for gpu in worker.gpu_uuids
        }

    @property
    def cached_gpu_uuids(self) -> set[str]:
        return {
            gpu
            for worker in self.workers.values()
            if worker.state is RuntimeState.SLEEPING
            for gpu in worker.gpu_uuids
        }

    async def launch(
        self,
        *,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        model_path: str,
        served_model_name: str,
    ) -> WorkerPlacement:
        if len(gpu_uuids) != profile.gpu_count or gpu_uuids not in profile.eligible_gpu_sets:
            raise WorkerLaunchError("placement does not match a validated GPU set")
        if not profile_verified_for_mode(
            profile,
            kvcached_required=self.kvcached.enabled,
            ram_weight_cache_required=self.ram_weight_cache_enabled,
            queue_mode_required=self.settings.queue_mode_enabled,
        ):
            raise WorkerLaunchError(
                "placement profile is not verified for the configured vLLM memory backend"
            )
        async with self._lock:
            requested_gpus = set(gpu_uuids)
            overlapping_workers = [
                worker
                for worker in self.workers.values()
                if worker.state is not RuntimeState.COLD
                and bool(set(worker.gpu_uuids) & requested_gpus)
            ]
            overlap = {
                gpu for worker in overlapping_workers for gpu in worker.gpu_uuids
            } & requested_gpus
            if overlap and not self._can_share_gpus(profile, gpu_uuids, overlapping_workers):
                raise WorkerLaunchError(f"GPU UUIDs already owned: {sorted(overlap)}")
            worker_id = str(uuid.uuid4())
            port = self._allocate_port()
            worker = WorkerPlacement(
                id=worker_id,
                profile=profile,
                gpu_uuids=gpu_uuids,
                port=port,
            )
            self.workers[worker_id] = worker
            try:
                command = self._command(worker, model_path, served_model_name)
                environment = self._environment(worker)
            except BaseException:
                self.workers.pop(worker_id, None)
                raise
            log_path: Path | None = None
            log_handle: Any | None = None
            worker_output: Any = asyncio.subprocess.DEVNULL
            if self.settings.capture_worker_engine_logs:
                log_path = worker_log_path(
                    log_dir=self.settings.log_dir,
                    served_model_name=served_model_name,
                    worker_id=worker_id,
                )
                log_handle = log_path.open("ab", buffering=0)
                worker_output = log_handle
                self._log_handles[worker_id] = log_handle
                self._log_paths[worker_id] = log_path
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=worker_output,
                    stderr=asyncio.subprocess.STDOUT,
                    env=environment,
                    # Make the engine its own group so a forced shutdown cannot signal
                    # the API server, while killpg still reaches all engine descendants.
                    start_new_session=True,
                )
            except BaseException:
                await self._cleanup(worker_id)
                self.workers.pop(worker_id, None)
                raise
            worker.process_pid = process.pid
            self._processes[worker_id] = process
            try:
                await self._persist(worker)
                await self.database.record_event(
                    "WORKER_LOADING",
                    worker_id,
                    {
                        "gpu_uuids": gpu_uuids,
                        "command": self._redact_command(command),
                    },
                )
            except BaseException:
                worker.state = RuntimeState.STOPPING
                await terminate_engine(process, gpu_uuids=worker.gpu_uuids, force=True)
                await self._cleanup(worker_id)
                self.workers.pop(worker_id, None)
                raise
            asyncio.create_task(self._await_ready(worker), name=f"worker-ready-{worker_id}")
            asyncio.create_task(self._monitor(worker), name=f"worker-monitor-{worker_id}")
            return worker

    def _allocate_port(self) -> int:
        used = {
            worker.port for worker in self.workers.values() if worker.state is not RuntimeState.COLD
        }
        for port in range(self.settings.worker_port_start, self.settings.worker_port_end + 1):
            if port not in used:
                return port
        raise WorkerLaunchError("private worker port range is exhausted")

    def _can_share_gpus(
        self,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        overlapping_workers: list[WorkerPlacement],
    ) -> bool:
        if self.settings.queue_mode_enabled:
            return False
        kvcached_required = self.kvcached.enabled
        if profile.engine is not Engine.VLLM or not profile_verified_for_mode(
            profile,
            kvcached_required=kvcached_required,
            ram_weight_cache_required=self.ram_weight_cache_enabled,
            queue_mode_required=self.settings.queue_mode_enabled,
        ):
            return False
        if any(
            worker.profile.engine is not Engine.VLLM
            or not profile_verified_for_mode(
                worker.profile,
                kvcached_required=kvcached_required,
                ram_weight_cache_required=self.ram_weight_cache_enabled,
                queue_mode_required=self.settings.queue_mode_enabled,
            )
            for worker in overlapping_workers
        ):
            return False
        for gpu_uuid in gpu_uuids:
            colocated = [worker for worker in overlapping_workers if gpu_uuid in worker.gpu_uuids]
            active = sum(worker.state in _GPU_RESIDENT_STATES for worker in colocated)
            max_active = self.settings.prism_max_workers_per_gpu if kvcached_required else 1
            if active >= max_active:
                return False
        return True

    def _environment(self, worker: WorkerPlacement) -> dict[str, str]:
        executable = (
            self.settings.engines.vllm_executable
            if worker.profile.engine is Engine.VLLM
            else self.settings.engines.llama_cpp_executable
        )
        environment = gpu_environment(
            worker.gpu_uuids,
            self.settings.engines.environment,
            executable=executable,
        )
        if self.settings.queue_mode_enabled:
            environment.update(
                ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0", VLLM_SERVER_DEV_MODE="0"
            )
            environment.pop("LLM_RIO_KVCACHED_VLLM026_SHIM", None)
        if worker.profile.engine is Engine.VLLM:
            environment["VLLM_API_KEY"] = self.internal_api_key
            if self.kvcached.enabled:
                environment.update(
                    self.kvcached.environment(pythonpath=environment.get("PYTHONPATH"))
                )
            if self.ram_weight_cache_enabled:
                # vLLM gates its authenticated sleep/wake routes behind this
                # opt-in. Workers listen only on loopback private ports.
                environment["VLLM_SERVER_DEV_MODE"] = "1"
        return environment

    def _command(
        self, worker: WorkerPlacement, model_path: str, served_model_name: str
    ) -> list[str]:
        profile = worker.profile
        if profile.engine is Engine.VLLM:
            command = [
                self.settings.engines.vllm_executable,
                "serve",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(worker.port),
                "--api-key",
                self.internal_api_key,
                "--served-model-name",
                served_model_name,
                "--tensor-parallel-size",
                str(profile.tensor_parallel_size),
                "--pipeline-parallel-size",
                str(profile.pipeline_parallel_size),
                "--dtype",
                profile.dtype,
                "--max-model-len",
                str(profile.max_model_len),
                "--gpu-memory-utilization",
                str(profile.gpu_memory_utilization),
            ]
            if profile.max_num_seqs is not None:
                command.extend(["--max-num-seqs", str(profile.max_num_seqs)])
            if profile.max_num_batched_tokens is not None:
                command.extend(["--max-num-batched-tokens", str(profile.max_num_batched_tokens)])
            if profile.quantization:
                command.extend(["--quantization", profile.quantization])
            parsers = detect_vllm_parser_configuration(model_path)
            if parsers.tool_parser is not None:
                command.extend(
                    ["--enable-auto-tool-choice", "--tool-call-parser", parsers.tool_parser]
                )
            if parsers.reasoning_parser is not None:
                command.extend(["--reasoning-parser", parsers.reasoning_parser])
            if self.ram_weight_cache_enabled and not profile.launch_args.get("enable_sleep_mode"):
                command.append("--enable-sleep-mode")
        elif profile.engine is Engine.LLAMA_CPP and self.settings.engines.enable_llama_cpp:
            command = [
                self.settings.engines.llama_cpp_executable,
                "--model",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(worker.port),
                "--ctx-size",
                str(profile.max_model_len),
                "--api-key",
                self.internal_api_key,
            ]
            if profile.max_num_seqs is not None:
                command.extend(["--parallel", str(profile.max_num_seqs)])
        else:
            raise WorkerLaunchError(f"engine is disabled or unsupported: {profile.engine}")
        for key, value in profile.launch_args.items():
            if self.settings.queue_mode_enabled and key == "enable_sleep_mode":
                continue
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
        if profile.engine is Engine.VLLM and self.kvcached.enabled:
            add_kvcached_vllm_flags(command, self.kvcached)
        return command

    @staticmethod
    def _redact_command(command: list[str]) -> list[str]:
        redacted = list(command)
        for index, value in enumerate(redacted[:-1]):
            if value == "--api-key":
                redacted[index + 1] = "[REDACTED]"
        return redacted

    async def _await_ready(self, worker: WorkerPlacement) -> None:
        startup_timeout = self.settings.worker_startup_timeout_seconds
        deadline = (
            asyncio.get_running_loop().time() + startup_timeout
            if startup_timeout is not None
            else None
        )
        headers = {"Authorization": f"Bearer {self.internal_api_key}"}
        async with httpx.AsyncClient(timeout=5.0) as client:
            while deadline is None or asyncio.get_running_loop().time() < deadline:
                process = self._processes.get(worker.id)
                if process is None or process.returncode is not None:
                    await self._fail(worker, "process_exited_during_startup")
                    return
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{worker.port}/health", headers=headers
                    )
                    if response.is_success:
                        async with self._lock:
                            if worker.state is not RuntimeState.LOADING:
                                return
                            worker.state = RuntimeState.READY
                            worker.ready_at = datetime.now(UTC)
                            worker.last_demand_at = worker.ready_at
                        await self._persist(worker)
                        await self.database.record_event("WORKER_READY", worker.id)
                        await self._emit(worker.id, "ready")
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        await self._fail(worker, "startup_timeout")

    async def _monitor(self, worker: WorkerPlacement) -> None:
        process = self._processes[worker.id]
        return_code = await process.wait()
        if worker.state not in {RuntimeState.STOPPING, RuntimeState.COLD}:
            await self._fail(worker, f"unexpected_exit_{return_code}")

    async def _fail(self, worker: WorkerPlacement, reason: str) -> None:
        admitted: list[str]
        async with self._lock:
            if worker.state is RuntimeState.COLD:
                return
            admitted = list(worker.admitted_request_ids)
            worker.admitted_request_ids.clear()
            worker.outstanding_token_work = 0
            worker.state = RuntimeState.STOPPING
            self._drain_to_sleep.discard(worker.id)
            process = self._processes.get(worker.id)
        if process:
            await terminate_engine(process, gpu_uuids=worker.gpu_uuids, force=True)
        async with self._lock:
            worker.state = RuntimeState.COLD
            worker.process_pid = None
            worker.host_weights_cached = False
            worker.host_cache_accounted_mib = 0.0
            worker.host_cache_accounting_source = None
            worker.process_rss_mib = 0.0
            worker.process_pss_mib = 0.0
            worker.process_swap_mib = 0.0
        await self._persist(worker)
        log_path = self._log_paths.get(worker.id)
        await self.database.record_event(
            "WORKER_FAILED",
            worker.id,
            {
                "reason": reason,
                "request_ids": admitted,
                "log_path": str(log_path) if log_path else None,
            },
        )
        await self._cleanup(worker.id, retain_log=True)
        await self._emit(worker.id, f"failed:{reason}")

    async def admit(self, worker_id: str, request_id: str, estimated_tokens: int) -> None:
        async with self._lock:
            worker = self.workers[worker_id]
            if worker.state is not RuntimeState.READY:
                raise RuntimeError("worker is not routable")
            worker.admitted_request_ids.add(request_id)
            worker.accepted_requests += 1
            worker.outstanding_token_work += estimated_tokens
            worker.last_demand_at = datetime.now(UTC)

    async def release(self, worker_id: str, request_id: str, estimated_tokens: int) -> None:
        should_stop = False
        should_offload = False
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None:
                return
            worker.admitted_request_ids.discard(request_id)
            worker.last_demand_at = datetime.now(UTC)
            worker.outstanding_token_work = max(0, worker.outstanding_token_work - estimated_tokens)
            drained = worker.state is RuntimeState.DRAINING and not worker.admitted_request_ids
            should_offload = drained and worker_id in self._drain_to_sleep
            should_stop = drained and not should_offload
        if should_offload:
            await self._offload(worker_id)
        elif should_stop:
            await self.stop(worker_id, force=False)
        await self._emit(worker_id, "released")

    async def sleep(self, worker_id: str) -> None:
        """Drain a routable worker and retain its weights in host RAM."""
        if not self.ram_weight_cache_enabled:
            await self.drain(worker_id)
            return
        offload_now = False
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None or worker.state in {
                RuntimeState.COLD,
                RuntimeState.OFFLOADING,
                RuntimeState.SLEEPING,
                RuntimeState.WAKING,
                RuntimeState.STOPPING,
            }:
                return
            if worker.state is RuntimeState.LOADING:
                return
            self._drain_to_sleep.add(worker_id)
            if worker.admitted_request_ids:
                worker.state = RuntimeState.DRAINING
                worker.drain_started_at = datetime.now(UTC)
            else:
                offload_now = True
        if offload_now:
            await self._offload(worker_id)
            return
        await self._persist(worker)
        await self.database.record_event(
            "WORKER_DRAINING_TO_RAM",
            worker_id,
            {"active_requests": len(worker.admitted_request_ids)},
        )

    async def _offload(self, worker_id: str) -> None:
        transition_lock = self._transition_locks.setdefault(worker_id, asyncio.Lock())
        async with transition_lock:
            async with self._lock:
                worker = self.workers.get(worker_id)
                if worker is None or worker.state in {
                    RuntimeState.COLD,
                    RuntimeState.SLEEPING,
                    RuntimeState.STOPPING,
                }:
                    return
                if worker.admitted_request_ids:
                    self._drain_to_sleep.add(worker_id)
                    worker.state = RuntimeState.DRAINING
                    worker.drain_started_at = datetime.now(UTC)
                    return
                worker.state = RuntimeState.OFFLOADING
                worker.drain_started_at = None
                self._drain_to_sleep.discard(worker_id)
            await self._persist(worker)
            await self.database.record_event("WORKER_OFFLOADING", worker_id)
            started = asyncio.get_running_loop().time()
            try:
                await self._post_engine(worker, "/sleep", params={"level": "1"})
            except Exception as exc:
                await self._fail(worker, f"weight_offload_failed:{type(exc).__name__}")
                return
            elapsed = asyncio.get_running_loop().time() - started
            async with self._lock:
                if worker.state is not RuntimeState.OFFLOADING:
                    return
                worker.state = RuntimeState.SLEEPING
                worker.sleeping_at = datetime.now(UTC)
                worker.last_offload_seconds = elapsed
                worker.host_weights_cached = True
            retained = await self.enforce_host_cache_budget(incoming_worker_id=worker_id)
            if not retained:
                reason = worker.last_cache_eviction_reason or "ram_budget"
                await self.database.record_event(
                    "WORKER_CACHE_REJECTED",
                    worker_id,
                    {
                        "reason": reason,
                        "host_cache_accounted_mib": worker.host_cache_accounted_mib,
                        "process_swap_mib": worker.process_swap_mib,
                    },
                )
                await self._stop(worker_id, force=False)
                return
            async with self._lock:
                if worker.state is not RuntimeState.SLEEPING:
                    return
            await self._persist(worker)
            await self.database.record_event(
                "WORKER_WEIGHTS_CACHED",
                worker_id,
                {
                    "offload_seconds": elapsed,
                    "storage": "host_ram",
                    "host_cache_accounted_mib": worker.host_cache_accounted_mib,
                    "accounting_source": worker.host_cache_accounting_source,
                },
            )
            await self._emit(worker_id, "sleeping")

    async def wake(self, worker_id: str) -> None:
        """Restore a SLEEPING worker's retained weights from host RAM."""
        if not self.ram_weight_cache_enabled:
            raise WorkerLaunchError("host-RAM weight caching is disabled")
        transition_lock = self._transition_locks.setdefault(worker_id, asyncio.Lock())
        async with transition_lock:
            async with self._lock:
                worker = self.workers.get(worker_id)
                if worker is None or worker.state is not RuntimeState.SLEEPING:
                    return
                worker.state = RuntimeState.WAKING
            await self._persist(worker)
            await self.database.record_event(
                "WORKER_WAKING",
                worker_id,
                {"storage": "host_ram"},
            )
            started = asyncio.get_running_loop().time()
            try:
                await self._post_engine(worker, "/wake_up")
            except Exception as exc:
                await self._fail(worker, f"weight_restore_failed:{type(exc).__name__}")
                return
            elapsed = asyncio.get_running_loop().time() - started
            async with self._lock:
                if worker.state is not RuntimeState.WAKING:
                    return
                now = datetime.now(UTC)
                worker.state = RuntimeState.READY
                worker.ready_at = now
                worker.last_demand_at = now
                worker.sleeping_at = None
                worker.last_activation_seconds = elapsed
                if not self.persistent_host_weight_cache_enabled:
                    worker.host_weights_cached = False
            await self._persist(worker)
            await self.database.record_event(
                "WORKER_WEIGHTS_RESTORED",
                worker_id,
                {"activation_seconds": elapsed, "storage": "host_ram"},
            )
            await self._emit(worker_id, "ready")

    async def _post_engine(
        self,
        worker: WorkerPlacement,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {self.internal_api_key}"}
        timeout = self.settings.prism_transition_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"http://127.0.0.1:{worker.port}{path}",
                headers=headers,
                params=params,
            )
            response.raise_for_status()

    async def drain(self, worker_id: str) -> None:
        stop_now = False
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None or worker.state in {
                RuntimeState.STOPPING,
                RuntimeState.COLD,
            }:
                return
            self._drain_to_sleep.discard(worker_id)
            worker.state = RuntimeState.DRAINING
            worker.drain_started_at = datetime.now(UTC)
            stop_now = not worker.admitted_request_ids
        await self._persist(worker)
        await self.database.record_event("WORKER_DRAINING", worker_id)
        if stop_now:
            await self.stop(worker_id, force=False)

    async def enforce_drain_watchdogs(self) -> list[tuple[str, list[str]]]:
        now = datetime.now(UTC)
        watchdog = self.settings.worker_drain_watchdog_seconds
        overdue: list[tuple[str, list[str]]] = []
        for worker in list(self.workers.values()):
            if (
                watchdog is not None
                and worker.state is RuntimeState.DRAINING
                and worker.drain_started_at
                and (now - worker.drain_started_at).total_seconds() > watchdog
            ):
                overdue.append((worker.id, list(worker.admitted_request_ids)))
                await self.stop(worker.id, force=True)
        return overdue

    async def stop(self, worker_id: str, *, force: bool) -> None:
        transition_lock = self._transition_locks.setdefault(worker_id, asyncio.Lock())
        async with transition_lock:
            await self._stop(worker_id, force=force)

    async def _stop(self, worker_id: str, *, force: bool) -> None:
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None or worker.state is RuntimeState.COLD:
                return
            if worker.admitted_request_ids and not force:
                return
            worker.state = RuntimeState.STOPPING
            process = self._processes.get(worker_id)
        try:
            await self._persist(worker)
        except Exception:
            # A shutdown must never leave a running engine behind because a best-effort
            # STOPPING record could not be written.  The final COLD persistence follows
            # after the process is gone.
            logger.exception("could not persist worker %s before stopping it", worker_id)
        if process:
            await terminate_engine(process, gpu_uuids=worker.gpu_uuids, force=force)
        async with self._lock:
            worker.state = RuntimeState.COLD
            worker.process_pid = None
            worker.admitted_request_ids.clear()
            worker.outstanding_token_work = 0
            worker.sleeping_at = None
            worker.host_weights_cached = False
            worker.host_cache_accounted_mib = 0.0
            worker.host_cache_accounting_source = None
            worker.process_rss_mib = 0.0
            worker.process_pss_mib = 0.0
            worker.process_swap_mib = 0.0
            self._drain_to_sleep.discard(worker_id)
        try:
            await self._persist(worker)
            await self.database.record_event("WORKER_COLD", worker_id, {"forced": force})
            await self._emit(worker_id, "cold")
        finally:
            await self._cleanup(worker_id)

    async def stop_all(self, *, force: bool = False) -> None:
        worker_ids = list(self.workers)
        if not force:
            for worker_id in worker_ids:
                await self.drain(worker_id)
        tasks = [self.stop(worker_id, force=force) for worker_id in worker_ids]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for worker_id, result in zip(worker_ids, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error("could not stop worker %s during shutdown: %r", worker_id, result)

    async def _cleanup(self, worker_id: str, *, retain_log: bool = False) -> None:
        self._drain_to_sleep.discard(worker_id)
        self._processes.pop(worker_id, None)
        handle = self._log_handles.pop(worker_id, None)
        if handle:
            handle.close()
        log_path = self._log_paths.pop(worker_id, None)
        if log_path and not retain_log:
            with contextlib.suppress(FileNotFoundError):
                log_path.unlink()

    async def _persist(self, worker: WorkerPlacement) -> None:
        await self.database.execute(
            """
            INSERT INTO workers
                (id, model_id, profile_id, gpu_uuids_json, port, pid, state,
                 host_cache_accounted_mib, host_cache_accounting_source,
                 process_rss_mib, process_pss_mib, process_swap_mib,
                 last_cache_eviction_reason, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET pid = excluded.pid, state = excluded.state,
                host_cache_accounted_mib = excluded.host_cache_accounted_mib,
                host_cache_accounting_source = excluded.host_cache_accounting_source,
                process_rss_mib = excluded.process_rss_mib,
                process_pss_mib = excluded.process_pss_mib,
                process_swap_mib = excluded.process_swap_mib,
                last_cache_eviction_reason = excluded.last_cache_eviction_reason,
                updated_at = excluded.updated_at
            """,
            (
                worker.id,
                worker.model_id,
                worker.profile.id,
                json.dumps(worker.gpu_uuids),
                worker.port,
                worker.process_pid,
                worker.state.value,
                worker.host_cache_accounted_mib,
                worker.host_cache_accounting_source,
                worker.process_rss_mib,
                worker.process_pss_mib,
                worker.process_swap_mib,
                worker.last_cache_eviction_reason,
                _now(),
                _now(),
            ),
        )

    async def _emit(self, worker_id: str, event: str) -> None:
        if self._event_callback:
            await self._event_callback(worker_id, event)
