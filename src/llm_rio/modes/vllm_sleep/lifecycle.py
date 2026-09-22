from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx

import llm_rio.workers as resources
from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.host_memory import (
    HostMemorySample,
    gib_to_mib,
)
from llm_rio.profiles import profile_verified_for_mode

if TYPE_CHECKING:
    from llm_rio.workers import WorkerSupervisor
from llm_rio.worker_types import WorkerLaunchError

logger = logging.getLogger(__name__)


class SleepLifecycle:
    def __init__(self, supervisor: WorkerSupervisor) -> None:
        self.supervisor = supervisor
        self.settings = supervisor.settings.modes.vllm_sleep

    async def _refresh_cached_worker_samples(self) -> None:
        async with self.supervisor._lock:
            targets = [
                (worker.id, worker.process_pid)
                for worker in self.supervisor.workers.values()
                if worker.host_weights_cached and worker.process_pid is not None
            ]
        samples = await asyncio.gather(
            *(
                asyncio.to_thread(resources.sample_process_group_memory, process_pid)
                for _, process_pid in targets
            )
        )
        async with self.supervisor._lock:
            for (worker_id, process_pid), sample in zip(targets, samples, strict=True):
                worker = self.supervisor.workers.get(worker_id)
                if worker is None or worker.process_pid != process_pid:
                    continue
                worker.process_rss_mib = sample.rss_mib
                worker.process_pss_mib = sample.pss_mib
                worker.process_swap_mib = sample.swap_pss_mib or sample.swap_mib
                worker.host_cache_accounted_mib = sample.accounted_mib
                worker.host_cache_accounting_source = sample.source

    def _host_cache_limit_mib(self, host: HostMemorySample) -> float:
        configured = gib_to_mib(getattr(self.settings, "host_cache_max_gib", None))
        if configured is not None:
            return configured
        reserve = gib_to_mib(getattr(self.settings, "host_cache_min_available_gib", 4.0)) or 0.0
        return max(0.0, host.effective_total_mib - reserve)

    def _host_cache_pressure_reason(
        self, host: HostMemorySample, cache_accounted_mib: float, cache_swap_mib: float = 0.0
    ) -> str | None:
        swap_limit = gib_to_mib(getattr(self.settings, "swap_max_used_gib", 0.0)) or 0.0
        if cache_swap_mib > swap_limit:
            return "swap_pressure"
        minimum_available = (
            gib_to_mib(getattr(self.settings, "host_cache_min_available_gib", 4.0)) or 0.0
        )
        if host.available_mib < minimum_available:
            return "ram_headroom"
        if cache_accounted_mib > self.supervisor._host_cache_limit_mib(host):
            return "ram_budget"
        return None

    async def host_cache_status(self) -> dict[str, float | str | None]:
        await self.supervisor._refresh_cached_worker_samples()
        host = await asyncio.to_thread(resources.sample_host_memory)
        async with self.supervisor._lock:
            cached_workers = [
                worker for worker in self.supervisor.workers.values() if worker.host_weights_cached
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
            "cache_limit_mib": self.supervisor._host_cache_limit_mib(host),
            "pressure_reason": self.supervisor._host_cache_pressure_reason(
                host, cache_accounted_mib, cache_swap_mib
            ),
        }

    async def enforce_host_cache_budget(self, *, incoming_worker_id: str | None = None) -> bool:
        """Evict least-recently-demanded sleeping workers until host pressure clears."""
        if not self.supervisor.ram_weight_cache_enabled:
            return True
        host_cache_lock = getattr(self, "_host_cache_lock", None)
        if host_cache_lock is None:
            host_cache_lock = self.supervisor._host_cache_lock = asyncio.Lock()
        async with host_cache_lock:
            while True:
                status = await self.supervisor.host_cache_status()
                reason = status["pressure_reason"]
                if reason is None:
                    return True
                async with self.supervisor._lock:
                    candidates = sorted(
                        (
                            worker
                            for worker in self.supervisor.workers.values()
                            if worker.id != incoming_worker_id
                            and worker.state is RuntimeState.SLEEPING
                            and worker.host_weights_cached
                            and (not worker.admitted_request_ids)
                        ),
                        key=lambda worker: worker.last_demand_at,
                    )
                    if not candidates:
                        incoming = (
                            self.supervisor.workers.get(incoming_worker_id)
                            if incoming_worker_id is not None
                            else None
                        )
                        if incoming is not None:
                            incoming.last_cache_eviction_reason = str(reason)
                        return False
                    victim = candidates[0]
                    victim.last_cache_eviction_reason = str(reason)
                    victim.state = RuntimeState.STOPPING
                await self.supervisor.database.record_event(
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
                await self.supervisor._stop(victim.id, force=False)

    def _can_share_gpus(
        self,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        overlapping_workers: list[WorkerPlacement],
    ) -> bool:
        kvcached_required = False
        if profile.engine is not Engine.VLLM or not profile_verified_for_mode(
            profile,
            kvcached_required=kvcached_required,
            ram_weight_cache_required=True,
            queue_mode_required=False,
        ):
            return False
        if any(
            worker.profile.engine is not Engine.VLLM
            or not profile_verified_for_mode(
                worker.profile,
                kvcached_required=kvcached_required,
                ram_weight_cache_required=True,
                queue_mode_required=False,
            )
            for worker in overlapping_workers
        ):
            return False
        for gpu_uuid in gpu_uuids:
            colocated = [worker for worker in overlapping_workers if gpu_uuid in worker.gpu_uuids]
            active = sum(worker.state in resources._GPU_RESIDENT_STATES for worker in colocated)
            max_active = 1
            if active >= max_active:
                return False
        return True

    async def sleep(self, worker_id: str) -> None:
        """Drain a routable worker and retain its weights in host RAM."""
        if not self.supervisor.ram_weight_cache_enabled:
            await self.supervisor.drain(worker_id)
            return
        offload_now = False
        async with self.supervisor._lock:
            worker = self.supervisor.workers.get(worker_id)
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
            self.supervisor._drain_to_sleep.add(worker_id)
            if worker.admitted_request_ids:
                worker.state = RuntimeState.DRAINING
                worker.drain_started_at = datetime.now(UTC)
            else:
                offload_now = True
        if offload_now:
            await self.supervisor._offload(worker_id)
            return
        await self.supervisor._persist(worker)
        await self.supervisor.database.record_event(
            "WORKER_DRAINING_TO_RAM",
            worker_id,
            {"active_requests": len(worker.admitted_request_ids)},
        )

    async def _offload(self, worker_id: str) -> None:
        transition_lock = self.supervisor._transition_locks.setdefault(worker_id, asyncio.Lock())
        async with transition_lock:
            async with self.supervisor._lock:
                worker = self.supervisor.workers.get(worker_id)
                if worker is None or worker.state in {
                    RuntimeState.COLD,
                    RuntimeState.SLEEPING,
                    RuntimeState.STOPPING,
                }:
                    return
                if worker.admitted_request_ids:
                    self.supervisor._drain_to_sleep.add(worker_id)
                    worker.state = RuntimeState.DRAINING
                    worker.drain_started_at = datetime.now(UTC)
                    return
                worker.state = RuntimeState.OFFLOADING
                worker.drain_started_at = None
                self.supervisor._drain_to_sleep.discard(worker_id)
            await self.supervisor._persist(worker)
            await self.supervisor.database.record_event("WORKER_OFFLOADING", worker_id)
            started = asyncio.get_running_loop().time()
            try:
                await self.supervisor._post_engine(worker, "/sleep", params={"level": "1"})
            except Exception as exc:
                await self.supervisor._fail(worker, f"weight_offload_failed:{type(exc).__name__}")
                return
            elapsed = asyncio.get_running_loop().time() - started
            async with self.supervisor._lock:
                if worker.state is not RuntimeState.OFFLOADING:
                    return
                worker.state = RuntimeState.SLEEPING
                worker.sleeping_at = datetime.now(UTC)
                worker.last_offload_seconds = elapsed
                worker.host_weights_cached = True
            retained = await self.supervisor.enforce_host_cache_budget(incoming_worker_id=worker_id)
            if not retained:
                reason = worker.last_cache_eviction_reason or "ram_budget"
                await self.supervisor.database.record_event(
                    "WORKER_CACHE_REJECTED",
                    worker_id,
                    {
                        "reason": reason,
                        "host_cache_accounted_mib": worker.host_cache_accounted_mib,
                        "process_swap_mib": worker.process_swap_mib,
                    },
                )
                await self.supervisor._stop(worker_id, force=False)
                return
            async with self.supervisor._lock:
                if worker.state is not RuntimeState.SLEEPING:
                    return
            await self.supervisor._persist(worker)
            await self.supervisor.database.record_event(
                "WORKER_WEIGHTS_CACHED",
                worker_id,
                {
                    "offload_seconds": elapsed,
                    "storage": "host_ram",
                    "host_cache_accounted_mib": worker.host_cache_accounted_mib,
                    "accounting_source": worker.host_cache_accounting_source,
                },
            )
            await self.supervisor._emit(worker_id, "sleeping")

    async def wake(self, worker_id: str) -> None:
        """Restore a SLEEPING worker's retained weights from host RAM."""
        if not self.supervisor.ram_weight_cache_enabled:
            raise WorkerLaunchError("host-RAM weight caching is disabled")
        transition_lock = self.supervisor._transition_locks.setdefault(worker_id, asyncio.Lock())
        async with transition_lock:
            async with self.supervisor._lock:
                worker = self.supervisor.workers.get(worker_id)
                if worker is None or worker.state is not RuntimeState.SLEEPING:
                    return
                worker.state = RuntimeState.WAKING
            await self.supervisor._persist(worker)
            await self.supervisor.database.record_event(
                "WORKER_WAKING", worker_id, {"storage": "host_ram"}
            )
            started = asyncio.get_running_loop().time()
            try:
                await self.supervisor._post_engine(worker, "/wake_up")
            except Exception as exc:
                await self.supervisor._fail(worker, f"weight_restore_failed:{type(exc).__name__}")
                return
            elapsed = asyncio.get_running_loop().time() - started
            async with self.supervisor._lock:
                if worker.state is not RuntimeState.WAKING:
                    return
                now = datetime.now(UTC)
                worker.state = RuntimeState.READY
                worker.ready_at = now
                worker.last_demand_at = now
                worker.sleeping_at = None
                worker.last_activation_seconds = elapsed
                if not self.supervisor.persistent_host_weight_cache_enabled:
                    worker.host_weights_cached = False
            await self.supervisor._persist(worker)
            await self.supervisor.database.record_event(
                "WORKER_WEIGHTS_RESTORED",
                worker_id,
                {"activation_seconds": elapsed, "storage": "host_ram"},
            )
            await self.supervisor._emit(worker_id, "ready")

    async def _post_engine(
        self, worker: WorkerPlacement, path: str, *, params: dict[str, str] | None = None
    ) -> None:
        headers = {"Authorization": f"Bearer {self.supervisor.internal_api_key}"}
        timeout = self.settings.transition_timeout_seconds
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"http://127.0.0.1:{worker.port}{path}", headers=headers, params=params
            )
            response.raise_for_status()
