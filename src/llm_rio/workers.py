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
from llm_rio.domain import PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.engine_runtime import detect_kvcached
from llm_rio.engines.identity import launch_binding
from llm_rio.engines.launch import adapter
from llm_rio.gpu_memory import read_gpu_memory, required_free_vram
from llm_rio.host_memory import (
    HostMemorySample,
)
from llm_rio.host_memory import sample_host_memory as sample_host_memory
from llm_rio.host_memory import sample_process_group_memory as sample_process_group_memory
from llm_rio.modes.lifecycle import WorkerLifecycle
from llm_rio.ports import PortAllocator
from llm_rio.process_cleanup import terminate_engine
from llm_rio.profiles import profile_verified_for_mode
from llm_rio.storage import Database, _now
from llm_rio.worker_types import WorkerLaunchError as WorkerLaunchError

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
        self.ports = PortAllocator(settings.worker_port_start, settings.worker_port_end)
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

    @property
    def lifecycle(self) -> WorkerLifecycle:
        if not hasattr(self, "_lifecycle"):
            if self.settings.queue_mode_enabled:
                from llm_rio.modes.queue.lifecycle import QueueLifecycle

                self._lifecycle: WorkerLifecycle = QueueLifecycle(self)
            elif self.settings.effective_kvcached_mode == "required":
                from llm_rio.modes.kv_cached.lifecycle import KVCachedLifecycle

                self._lifecycle = KVCachedLifecycle(self)
            else:
                from llm_rio.modes.vllm_sleep.lifecycle import SleepLifecycle

                self._lifecycle = SleepLifecycle(self)
        return self._lifecycle

    def set_event_callback(self, callback: WorkerEventCallback) -> None:
        self._event_callback = callback

    @property
    def ram_weight_cache_enabled(self) -> bool:
        return self.settings.ram_weight_cache_enabled

    @property
    def persistent_host_weight_cache_enabled(self) -> bool:
        return self.kvcached.environment().get("LLM_RIO_KVCACHED_VLLM026_SHIM") == "1"

    async def _refresh_cached_worker_samples(self) -> None:
        return await self.lifecycle._refresh_cached_worker_samples()

    def _host_cache_limit_mib(self, host: HostMemorySample) -> float:
        return self.lifecycle._host_cache_limit_mib(host)

    def _host_cache_pressure_reason(
        self, host: HostMemorySample, cache_accounted_mib: float, cache_swap_mib: float = 0.0
    ) -> str | None:
        return self.lifecycle._host_cache_pressure_reason(host, cache_accounted_mib, cache_swap_mib)

    async def host_cache_status(self) -> dict[str, float | str | None]:
        return await self.lifecycle.host_cache_status()

    async def enforce_host_cache_budget(self, *, incoming_worker_id: str | None = None) -> bool:
        return await self.lifecycle.enforce_host_cache_budget(incoming_worker_id=incoming_worker_id)

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
        if profile.launch_binding != launch_binding(self.settings, profile, profile.engine):
            raise WorkerLaunchError("Launch configuration changed; run Validate/Revalidate")
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
            port = self.ports.reserve(worker_id)
            worker = WorkerPlacement(
                id=worker_id,
                profile=profile,
                gpu_uuids=gpu_uuids,
                port=port,
            )
            self.workers[worker_id] = worker
            try:
                spec = adapter(profile.engine).launch(
                    settings=self.settings,
                    shape=profile,
                    artifact=Path(str(profile.launch_args.get("model", model_path))),
                    nickname=served_model_name,
                    gpu_uuids=gpu_uuids,
                    port=port,
                    api_key=self.internal_api_key,
                )
                command = list(spec.command)
                environment = spec.environment
            except BaseException:
                self.ports.release(port, worker_id)
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

    def _can_share_gpus(
        self,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        overlapping_workers: list[WorkerPlacement],
    ) -> bool:
        return self.lifecycle._can_share_gpus(profile, gpu_uuids, overlapping_workers)

    def _environment(self, worker: WorkerPlacement) -> dict[str, str]:
        return (
            adapter(worker.profile.engine)
            .launch(
                settings=self.settings,
                shape=worker.profile,
                artifact=Path(str(worker.profile.launch_args.get("model", "."))),
                nickname=worker.model_id,
                gpu_uuids=worker.gpu_uuids,
                port=worker.port,
                api_key=self.internal_api_key,
            )
            .environment
        )

    def _command(
        self, worker: WorkerPlacement, model_path: str, served_model_name: str
    ) -> list[str]:
        artifact = Path(str(worker.profile.launch_args.get("model", model_path)))
        return list(
            adapter(worker.profile.engine)
            .launch(
                settings=self.settings,
                shape=worker.profile,
                artifact=artifact,
                nickname=served_model_name,
                gpu_uuids=worker.gpu_uuids,
                port=worker.port,
                api_key=self.internal_api_key,
            )
            .command
        )

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
        return await self.lifecycle.sleep(worker_id)

    async def _offload(self, worker_id: str) -> None:
        return await self.lifecycle._offload(worker_id)

    async def wake(self, worker_id: str) -> None:
        return await self.lifecycle.wake(worker_id)

    async def _post_engine(
        self, worker: WorkerPlacement, path: str, *, params: dict[str, str] | None = None
    ) -> None:
        return await self.lifecycle._post_engine(worker, path, params=params)

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
        worker = self.workers.get(worker_id)
        if worker is not None and self.ports.snapshot().get(worker.port) == worker_id:
            self.ports.release(worker.port, worker_id)
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
