from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import cast

from llm_rio.config import Settings
from llm_rio.domain import CatalogState, MachineInventory, RuntimeState, ServiceMode
from llm_rio.errors import MaintenanceError, RioError
from llm_rio.planner import (
    DrainPlacement,
    GreedyPlacementPlanner,
    QueuePressure,
    SleepPlacement,
    StartPlacement,
    WakePlacement,
)
from llm_rio.prism import detect_kvcached
from llm_rio.profiles import ProfileRepository, profile_verified_for_mode
from llm_rio.queueing import ModelQueues, QueuedRequest
from llm_rio.storage import Database
from llm_rio.workers import WorkerSupervisor

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerLease:
    worker_id: str
    request_id: str
    reservation_id: str
    base_url: str
    internal_api_key: str
    estimated_tokens: int
    admitted_at: datetime
    is_stream: bool = False
    last_activity_at: datetime | None = None


class ResidencyScheduler:
    """Serialized authority for routing, placement, draining, and maintenance."""

    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        inventory: MachineInventory,
        profiles: ProfileRepository,
        supervisor: WorkerSupervisor,
    ) -> None:
        self.settings = settings
        self.database = database
        self.inventory = inventory
        self.profiles = profiles
        self.supervisor = supervisor
        self.kvcached = detect_kvcached(settings.effective_kvcached_mode)
        self.queues = ModelQueues(
            settings.queue_capacity_per_model, settings.queue_capacity_per_tenant
        )
        self.planner = GreedyPlacementPlanner(
            wait_duration_seconds=settings.wait_duration_seconds,
            minimum_residency_seconds=settings.minimum_residency_seconds,
            fair_share_seconds=settings.fair_share_seconds,
            prism_enabled=self.kvcached.enabled or supervisor.ram_weight_cache_enabled,
            queue_mode=settings.queue_mode_enabled,
            kvcached_required=self.kvcached.enabled,
            gpu_vram_mib={device.uuid: device.total_vram_mib for device in inventory.gpus},
            reserved_vram_mib=settings.reserved_vram_mib,
            prism_max_workers_per_gpu=settings.prism_max_workers_per_gpu,
            prism_sleep_gpu_reserve_mib=settings.prism_sleep_gpu_reserve_mib,
            prism_idle_sleep_seconds=settings.prism_idle_sleep_seconds,
            prism_weight_cache_enabled=getattr(
                supervisor,
                "ram_weight_cache_enabled",
                self.kvcached.enabled and settings.prism_weight_cache_mode == "ram",
            ),
        )
        self._state_lock = asyncio.Lock()
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._request_leases: dict[str, WorkerLease] = {}
        self._validation_gpu_uuids: set[str] = set()
        self._last_arrival_at = datetime.now(UTC)
        self._maintenance_requested = False
        self._prism_preload_model_ids: dict[str, int] = {}
        self._prism_one_time_warm_model_ids: set[str] = set()
        self._reported_incompatible_profiles: set[str] = set()
        self._prism_configured = False
        supervisor.set_event_callback(self.worker_event)

    @property
    def serving_mode(self) -> str:
        if self.kvcached.enabled:
            return "kv-cached"
        return "vllm-sleep" if self.supervisor.ram_weight_cache_enabled else "queue"

    @property
    def validation_requires_maintenance(self) -> bool:
        return not self.kvcached.enabled

    @property
    def validation_gpu_uuids(self) -> tuple[str, ...]:
        return tuple(sorted(self._validation_gpu_uuids))

    async def start(self) -> None:
        if self._task is None:
            self._maintenance_requested = (
                await self.database.service_mode() is not ServiceMode.ACTIVE
            )
            await self._configure_prism()
            self._task = asyncio.create_task(self._run(), name="residency-scheduler")

    async def _configure_prism(self) -> None:
        if self._prism_configured:
            return
        self._prism_configured = True
        runtime = self.kvcached
        if not runtime.enabled:
            if self.settings.effective_kvcached_mode == "auto":
                await self.database.record_event(
                    "PRISM_UNAVAILABLE",
                    payload={"reason": runtime.reason},
                )
            event_type = (
                "PRISM_NATIVE_SLEEP_ENABLED"
                if self.supervisor.ram_weight_cache_enabled
                else "QUEUE_ENABLED"
                if self.settings.queue_mode_enabled
                else "PRISM_DISABLED"
            )
        else:
            event_type = "PRISM_ENABLED"
        await self.database.record_event(
            event_type,
            payload={
                "serving_mode": self.serving_mode,
                "kvcached_version": runtime.package_version,
                "kvcached_revision": runtime.source_revision,
                "vllm_version": runtime.vllm_version,
                "officially_tested": runtime.officially_tested,
                "max_workers_per_gpu": (
                    self.settings.prism_max_workers_per_gpu if runtime.enabled else 1
                ),
                "weight_cache": (
                    "vllm_sleep_level_1" if self.supervisor.ram_weight_cache_enabled else "disabled"
                ),
                "host_cache_max_gib": self.settings.prism_host_cache_max_gib,
                "host_cache_min_available_gib": (self.settings.prism_host_cache_min_available_gib),
                "swap_max_used_gib": self.settings.prism_swap_max_used_gib,
                "idle_sleep_seconds": self.settings.prism_idle_sleep_seconds,
            },
        )
        if self.settings.queue_mode_enabled:
            return  # Queue mode loads only in response to real queued requests.
        selectors = self.settings.prism_preload_models
        if not selectors:
            return
        models = await self.database.list_models()
        by_nickname = {str(model["nickname"]): model for model in models}
        selected = (
            models
            if selectors == ["*"]
            else [by_nickname[name] for name in selectors if name in by_nickname]
        )
        selected_nicknames = {str(model["nickname"]) for model in selected}
        for nickname in selectors:
            if nickname != "*" and nickname not in selected_nicknames:
                await self.database.record_event(
                    "PRISM_PRELOAD_SKIPPED",
                    payload={"nickname": nickname, "reason": "model_not_found"},
                )
        for model in selected:
            if model.get("state") == CatalogState.AVAILABLE.value and model.get("artifact_path"):
                model_id = str(model["id"])
                self._prism_preload_model_ids[model_id] = (
                    self._prism_preload_model_ids.get(model_id, 0) + 1
                )
            else:
                await self.database.record_event(
                    "PRISM_PRELOAD_SKIPPED",
                    str(model["id"]),
                    {
                        "nickname": model["nickname"],
                        "reason": "model_not_available",
                    },
                )

    async def close(self) -> None:
        self._closed = True
        self._event.set()
        try:
            if self._task:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
        finally:
            try:
                for lease in list(self._request_leases.values()):
                    await self.database.release_reservation(
                        lease.reservation_id, "service_shutdown"
                    )
            finally:
                # Worker teardown cannot depend on scheduler or quota cleanup completing.
                await self.supervisor.stop_all(force=True)

    async def enqueue(self, request: QueuedRequest) -> WorkerLease:
        request.assignment = asyncio.get_running_loop().create_future()
        async with self._state_lock:
            if await self.database.service_mode() is not ServiceMode.ACTIVE:
                raise MaintenanceError()
            self.queues.for_model(request.model_id).put(request)
            self._last_arrival_at = datetime.now(UTC)
        self._event.set()
        try:
            return cast(WorkerLease, await request.assignment)
        except asyncio.CancelledError:
            async with self._state_lock:
                removed = self.queues.remove(request.model_id, request.id)
            if removed:
                await self.database.release_reservation(request.reservation_id, "client_cancelled")
            raise

    async def warm_model_once(self, model_id: str) -> None:
        """Populate one newly validated model into the host-RAM weight cache."""
        if self.validation_requires_maintenance or not self.planner.prism_weight_cache_enabled:
            return
        async with self._state_lock:
            self._prism_one_time_warm_model_ids.add(model_id)
        await self.database.record_event(
            "PRISM_MODEL_WARM_REQUESTED",
            model_id,
            {"target": "host_ram"},
        )
        self._event.set()

    async def release(self, lease: WorkerLease) -> None:
        if self._request_leases.pop(lease.request_id, None) is None:
            return
        await self.supervisor.release(lease.worker_id, lease.request_id, lease.estimated_tokens)
        self._event.set()

    async def touch(self, lease: WorkerLease) -> None:
        async with self._state_lock:
            current = self._request_leases.get(lease.request_id)
            if current is not None:
                self._request_leases[lease.request_id] = replace(
                    current, last_activity_at=datetime.now(UTC)
                )

    async def worker_event(self, worker_id: str, event: str) -> None:
        if event.startswith("failed:"):
            failed = [
                lease for lease in self._request_leases.values() if lease.worker_id == worker_id
            ]
            for lease in failed:
                self._request_leases.pop(lease.request_id, None)
                await self.database.release_reservation(lease.reservation_id, "worker_failed")
        self._event.set()

    async def acquire_validation_gpus(self, gpu_uuids: tuple[str, ...]) -> bool:
        worker_ids: list[str] = []
        maintenance_only = self.validation_requires_maintenance
        preserve_weight_cache = not maintenance_only and self.supervisor.ram_weight_cache_enabled
        async with self._state_lock:
            mode = await self.database.service_mode()
            window_open = (
                mode is ServiceMode.MAINTENANCE_READY
                if maintenance_only
                else mode is ServiceMode.ACTIVE and not self._maintenance_requested
            )
            idle_for = (datetime.now(UTC) - self._last_arrival_at).total_seconds()
            overlapping = [
                worker
                for worker in self.supervisor.workers.values()
                if worker.state is not RuntimeState.COLD
                and bool(set(worker.gpu_uuids) & set(gpu_uuids))
            ]
            if (
                not window_open
                or self._closed
                or self.queues.pending_models()
                or (
                    not maintenance_only and idle_for < self.settings.validation_idle_window_seconds
                )
                or set(gpu_uuids) & self._validation_gpu_uuids
                or any(worker.admitted_request_ids for worker in overlapping)
                or any(
                    worker.state not in {RuntimeState.READY, RuntimeState.SLEEPING}
                    for worker in overlapping
                )
            ):
                return False
            self._validation_gpu_uuids.update(gpu_uuids)
            worker_ids = [worker.id for worker in overlapping]
        try:
            if preserve_weight_cache:
                results = await asyncio.gather(
                    *(
                        self.supervisor.sleep(worker.id)
                        for worker in overlapping
                        if worker.state is RuntimeState.READY
                    ),
                    return_exceptions=True,
                )
            else:
                results = await asyncio.gather(
                    *(self.supervisor.stop(worker_id, force=False) for worker_id in worker_ids),
                    return_exceptions=True,
                )
            failures = [result for result in results if isinstance(result, BaseException)]
            async with self._state_lock:
                retained_worker_ids = [
                    worker_id
                    for worker_id in worker_ids
                    if (worker := self.supervisor.workers.get(worker_id)) is not None
                    and worker.state is RuntimeState.SLEEPING
                ]
                unresolved_worker_ids = [
                    worker_id
                    for worker_id in worker_ids
                    if (worker := self.supervisor.workers.get(worker_id)) is not None
                    and (
                        worker.state is not RuntimeState.COLD
                        if maintenance_only
                        else worker.state not in {RuntimeState.COLD, RuntimeState.SLEEPING}
                    )
                ]
                demand_arrived = bool(self.queues.pending_models())
                if failures or unresolved_worker_ids or demand_arrived:
                    self._validation_gpu_uuids.difference_update(gpu_uuids)
            if failures or unresolved_worker_ids or demand_arrived:
                await self.database.record_event(
                    (
                        "VALIDATION_GPUS_DEFERRED"
                        if demand_arrived and not failures and not unresolved_worker_ids
                        else "VALIDATION_GPUS_ACQUIRE_FAILED"
                    ),
                    payload={
                        "gpu_uuids": gpu_uuids,
                        "transition_failures": len(failures),
                        "unresolved_worker_ids": unresolved_worker_ids,
                        "production_demand": demand_arrived,
                        "preserved_cached_worker_ids": retained_worker_ids,
                    },
                )
                self._event.set()
                return False
            await self.database.record_event(
                "VALIDATION_GPUS_ACQUIRED",
                payload={
                    "gpu_uuids": gpu_uuids,
                    "preserved_cached_worker_ids": retained_worker_ids,
                    "evicted_cached_worker_ids": ([] if preserve_weight_cache else worker_ids),
                },
            )
            return True
        except BaseException:
            async with self._state_lock:
                self._validation_gpu_uuids.difference_update(gpu_uuids)
            self._event.set()
            raise

    async def release_validation_gpus(self, gpu_uuids: tuple[str, ...]) -> None:
        async with self._state_lock:
            self._validation_gpu_uuids.difference_update(gpu_uuids)
        await self.database.record_event(
            "VALIDATION_GPUS_RELEASED", payload={"gpu_uuids": gpu_uuids}
        )
        self._event.set()

    def validation_should_yield(self) -> bool:
        wrong_mode = (
            not self._maintenance_requested
            if self.validation_requires_maintenance
            else self._maintenance_requested
        )
        return bool(self.queues.pending_models()) or self._closed or wrong_mode

    async def enter_maintenance(self) -> None:
        async with self._state_lock:
            if await self.database.service_mode() is not ServiceMode.ACTIVE:
                return
            self._maintenance_requested = True
            await self.database.set_service_mode(ServiceMode.DRAINING)
            rejected = self.queues.drain_all()
        for request in rejected:
            if request.assignment and not request.assignment.done():
                request.assignment.set_exception(MaintenanceError())
            await self.database.release_reservation(request.reservation_id, "maintenance")
        for worker in list(self.supervisor.workers.values()):
            await self.supervisor.drain(worker.id)
        self._event.set()

    async def resume(self) -> None:
        async with self._state_lock:
            if self._validation_gpu_uuids:
                raise RioError(
                    "validation_in_progress",
                    "Wait for model validation to finish before resuming service",
                    status_code=409,
                    details={"gpu_uuids": sorted(self._validation_gpu_uuids)},
                )
            if any(
                worker.state is not RuntimeState.COLD for worker in self.supervisor.workers.values()
            ):
                raise RioError(
                    "maintenance_not_ready",
                    "Workers are still draining or stopping",
                    status_code=409,
                )
            await self.database.set_service_mode(ServiceMode.ACTIVE)
            self._maintenance_requested = False
        self._event.set()

    async def _run(self) -> None:
        while not self._closed:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._event.wait(), timeout=self.settings.scheduler_tick_seconds
                )
            self._event.clear()
            try:
                await self._reconcile()
            except Exception:
                logger.exception("scheduler reconciliation failed")

    async def _reconcile(self) -> None:
        await self._expire_inactive_streams()
        await self._release_settled_leases()
        overdue = await self.supervisor.enforce_drain_watchdogs()
        for _, request_ids in overdue:
            for request_id in request_ids:
                lease = self._request_leases.pop(request_id, None)
                if lease:
                    await self.database.release_reservation(lease.reservation_id, "drain_watchdog")

        await self.supervisor.enforce_host_cache_budget()
        mode = await self.database.service_mode()
        if mode is ServiceMode.DRAINING:
            async with self._state_lock:
                if (
                    await self.database.service_mode() is ServiceMode.DRAINING
                    and not self._validation_gpu_uuids
                    and all(
                        worker.state is RuntimeState.COLD
                        for worker in self.supervisor.workers.values()
                    )
                ):
                    await self.database.set_service_mode(ServiceMode.MAINTENANCE_READY)
            return
        if mode is ServiceMode.MAINTENANCE_READY:
            return

        await self._route_ready_work()
        pressures = self._pressures()
        now = datetime.now(UTC)
        warm_model_ids = {
            worker.model_id
            for worker in self.supervisor.workers.values()
            if worker.state in {RuntimeState.READY, RuntimeState.SLEEPING}
        }
        self._prism_one_time_warm_model_ids.difference_update(warm_model_ids)
        pressured_model_ids = {pressure.model_id for pressure in pressures}
        preload_requirements = dict(self._prism_preload_model_ids)
        for model_id in self._prism_one_time_warm_model_ids:
            preload_requirements.setdefault(model_id, 1)
        pressures.extend(
            QueuePressure(
                model_id=model_id,
                requests=0,
                estimated_tokens=1,
                oldest_enqueued_at=now,
                preload=True,
                desired_workers=desired_workers,
            )
            for model_id, desired_workers in sorted(preload_requirements.items())
            if model_id not in pressured_model_ids
            and sum(
                worker.model_id == model_id
                and worker.state not in {RuntimeState.COLD, RuntimeState.STOPPING}
                for worker in self.supervisor.workers.values()
            )
            < desired_workers
        )
        profile_map = {
            pressure.model_id: await self.profiles.for_model(pressure.model_id)
            for pressure in pressures
        }
        if self.settings.queue_mode_enabled:
            profile_map = {
                model_id: [
                    profile
                    for profile in model_profiles
                    if profile_verified_for_mode(
                        profile, kvcached_required=False, queue_mode_required=True
                    )
                ]
                for model_id, model_profiles in profile_map.items()
            }
        if self.kvcached.enabled or self.planner.prism_weight_cache_enabled:
            for model_id, model_profiles in profile_map.items():
                compatible = any(
                    profile_verified_for_mode(
                        profile,
                        kvcached_required=self.kvcached.enabled,
                        ram_weight_cache_required=(self.planner.prism_weight_cache_enabled),
                    )
                    for profile in model_profiles
                )
                if (
                    model_profiles
                    and not compatible
                    and model_id not in self._reported_incompatible_profiles
                ):
                    self._reported_incompatible_profiles.add(model_id)
                    await self.database.record_event(
                        "PRISM_PROFILE_REVALIDATION_REQUIRED",
                        model_id,
                    )
        actions = self.planner.plan(
            now=now,
            all_gpu_uuids={device.uuid for device in self.inventory.gpus}
            - self._validation_gpu_uuids,
            workers=list(self.supervisor.workers.values()),
            pressures=pressures,
            profiles=profile_map,
        )
        sleep_actions = [action for action in actions if isinstance(action, SleepPlacement)]
        if sleep_actions:
            await asyncio.gather(
                *(self.supervisor.sleep(action.worker_id) for action in sleep_actions)
            )
        for action in actions:
            if isinstance(action, SleepPlacement):
                continue
            if isinstance(action, DrainPlacement):
                await self.supervisor.drain(action.worker_id)
            elif isinstance(action, WakePlacement):
                await self._wake_if_active(action)
            elif isinstance(action, StartPlacement):
                await self._launch_if_active(action)

    async def _expire_inactive_streams(self) -> None:
        timeout = self.settings.worker_stream_idle_timeout_seconds
        if timeout is None:
            return
        now = datetime.now(UTC)
        async with self._state_lock:
            expired = [
                lease
                for lease in self._request_leases.values()
                if lease.is_stream
                and lease.last_activity_at is not None
                and (now - lease.last_activity_at).total_seconds() >= timeout
            ]
        for lease in expired:
            await self.database.release_reservation(
                lease.reservation_id, "worker_stream_idle_timeout"
            )
            await self.release(lease)
            await self.database.record_event(
                "WORKER_STREAM_IDLE_TIMEOUT",
                lease.worker_id,
                {"request_id": lease.request_id, "timeout_seconds": timeout},
            )

    async def _release_settled_leases(self) -> None:
        admitted_request_ids = await self.database.admitted_request_ids()
        stale_leases = [
            lease
            for request_id, lease in self._request_leases.items()
            if request_id not in admitted_request_ids
        ]
        for lease in stale_leases:
            logger.warning(
                "Releasing stale worker admission for completed request %s", lease.request_id
            )
            await self.release(lease)

    async def _launch_if_active(self, action: StartPlacement) -> None:
        async with self._state_lock:
            if (
                self._maintenance_requested
                or await self.database.service_mode() is not ServiceMode.ACTIVE
                or bool(set(action.gpu_uuids) & self._validation_gpu_uuids)
            ):
                return
            model = await self.database.model_by_id(action.profile.model_id)
            if model is None or not model.get("artifact_path"):
                await self.database.record_event(
                    "PLACEMENT_REJECTED",
                    action.profile.model_id,
                    {"reason": "artifact_path_missing"},
                )
                return
            if not self.kvcached.enabled and not await self.supervisor.ensure_gpu_capacity(
                action.profile, action.gpu_uuids
            ):
                return
            await self.supervisor.launch(
                profile=action.profile,
                gpu_uuids=action.gpu_uuids,
                model_path=model["artifact_path"],
                served_model_name=model["nickname"],
            )

    async def _wake_if_active(self, action: WakePlacement) -> None:
        async with self._state_lock:
            worker = self.supervisor.workers.get(action.worker_id)
            if (
                worker is None
                or worker.state is not RuntimeState.SLEEPING
                or self._maintenance_requested
                or await self.database.service_mode() is not ServiceMode.ACTIVE
                or bool(set(worker.gpu_uuids) & self._validation_gpu_uuids)
            ):
                return
            if not self.kvcached.enabled and not await self.supervisor.ensure_gpu_capacity(
                worker.profile, worker.gpu_uuids, waking_worker_id=worker.id
            ):
                return
            await self.supervisor.wake(action.worker_id)

    async def _route_ready_work(self) -> None:
        async with self._state_lock:
            for model_id in self.queues.pending_models():
                queue = self.queues.for_model(model_id)
                while len(queue):
                    ready = [
                        worker
                        for worker in self.supervisor.workers.values()
                        if worker.model_id == model_id and worker.is_routable
                    ]
                    if not ready:
                        break
                    worker = min(ready, key=lambda item: item.outstanding_token_work)
                    request = queue.pop()
                    if request is None:
                        break
                    admitted = False
                    try:
                        await self.supervisor.admit(worker.id, request.id, request.estimated_tokens)
                        admitted = True
                        if not await self.database.mark_request_admitted(request.id, worker.id):
                            raise RuntimeError("request is no longer queued")
                    except Exception as exc:
                        if admitted:
                            with contextlib.suppress(Exception):
                                await self.supervisor.release(
                                    worker.id, request.id, request.estimated_tokens
                                )
                        with contextlib.suppress(Exception):
                            await self.database.release_reservation(
                                request.reservation_id, "admission_failed"
                            )
                        if request.assignment and not request.assignment.done():
                            request.assignment.set_exception(exc)
                        continue
                    admitted_at = datetime.now(UTC)
                    lease = WorkerLease(
                        worker_id=worker.id,
                        request_id=request.id,
                        reservation_id=request.reservation_id,
                        base_url=f"http://127.0.0.1:{worker.port}",
                        internal_api_key=self.supervisor.internal_api_key,
                        estimated_tokens=request.estimated_tokens,
                        admitted_at=admitted_at,
                        is_stream=request.is_stream,
                        last_activity_at=admitted_at if request.is_stream else None,
                    )
                    self._request_leases[request.id] = lease
                    if request.assignment and not request.assignment.done():
                        request.assignment.set_result(lease)
                    else:
                        await self.database.release_reservation(
                            request.reservation_id, "client_cancelled"
                        )
                        await self.release(lease)

    def _pressures(self) -> list[QueuePressure]:
        pressures: list[QueuePressure] = []
        model_ids = set(self.queues.pending_models())
        model_ids.update(
            worker.model_id
            for worker in self.supervisor.workers.values()
            if worker.admitted_request_ids
        )
        for model_id in model_ids:
            queue = self.queues.for_model(model_id)
            active_workers = [
                worker
                for worker in self.supervisor.workers.values()
                if worker.model_id == model_id and worker.admitted_request_ids
            ]
            oldest_candidates = [
                value
                for value in (
                    queue.oldest_enqueued_at,
                    *(worker.last_demand_at for worker in active_workers),
                )
                if value is not None
            ]
            if not oldest_candidates:
                continue
            pressures.append(
                QueuePressure(
                    model_id=model_id,
                    requests=len(queue)
                    + sum(len(worker.admitted_request_ids) for worker in active_workers),
                    estimated_tokens=queue.estimated_token_work
                    + sum(worker.outstanding_token_work for worker in active_workers),
                    oldest_enqueued_at=min(oldest_candidates),
                )
            )
        return pressures
