from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from llm_rio.config import Settings
from llm_rio.domain import MachineInventory, RuntimeState, ServiceMode
from llm_rio.errors import MaintenanceError, RioError
from llm_rio.planner import DrainPlacement, GreedyPlacementPlanner, QueuePressure, StartPlacement
from llm_rio.profiles import ProfileRepository
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
        self.queues = ModelQueues(
            settings.queue_capacity_per_model, settings.queue_capacity_per_tenant
        )
        self.planner = GreedyPlacementPlanner(
            wait_duration_seconds=settings.wait_duration_seconds,
            minimum_residency_seconds=settings.minimum_residency_seconds,
            fair_share_seconds=settings.fair_share_seconds,
        )
        self._state_lock = asyncio.Lock()
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._request_leases: dict[str, WorkerLease] = {}
        self._validation_gpu_uuids: set[str] = set()
        self._last_arrival_at = datetime.now(UTC)
        self._maintenance_requested = False
        supervisor.set_event_callback(self.worker_event)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="residency-scheduler")

    async def close(self) -> None:
        self._closed = True
        self._event.set()
        if self._task:
            await self._task
        for lease in list(self._request_leases.values()):
            await self.database.release_reservation(lease.reservation_id, "service_shutdown")
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
            return await request.assignment
        except asyncio.CancelledError:
            async with self._state_lock:
                removed = self.queues.remove(request.model_id, request.id)
            if removed:
                await self.database.release_reservation(request.reservation_id, "client_cancelled")
            raise

    async def release(self, lease: WorkerLease) -> None:
        if self._request_leases.pop(lease.request_id, None) is None:
            return
        await self.supervisor.release(lease.worker_id, lease.request_id, lease.estimated_tokens)
        self._event.set()

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
        async with self._state_lock:
            idle_for = (datetime.now(UTC) - self._last_arrival_at).total_seconds()
            if (
                await self.database.service_mode() is not ServiceMode.ACTIVE
                or self._maintenance_requested
                or self.queues.pending_models()
                or idle_for < self.settings.validation_idle_window_seconds
                or set(gpu_uuids) & self.supervisor.occupied_gpu_uuids
                or set(gpu_uuids) & self._validation_gpu_uuids
            ):
                return False
            self._validation_gpu_uuids.update(gpu_uuids)
            await self.database.record_event(
                "VALIDATION_GPUS_ACQUIRED", payload={"gpu_uuids": gpu_uuids}
            )
            return True

    async def release_validation_gpus(self, gpu_uuids: tuple[str, ...]) -> None:
        async with self._state_lock:
            self._validation_gpu_uuids.difference_update(gpu_uuids)
        await self.database.record_event(
            "VALIDATION_GPUS_RELEASED", payload={"gpu_uuids": gpu_uuids}
        )
        self._event.set()

    def validation_should_yield(self) -> bool:
        return bool(self.queues.pending_models()) or self._closed or self._maintenance_requested

    async def enter_maintenance(self) -> None:
        async with self._state_lock:
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
        await self._release_settled_leases()
        overdue = await self.supervisor.enforce_drain_watchdogs()
        for _, request_ids in overdue:
            for request_id in request_ids:
                lease = self._request_leases.pop(request_id, None)
                if lease:
                    await self.database.release_reservation(lease.reservation_id, "drain_watchdog")

        mode = await self.database.service_mode()
        if mode is ServiceMode.DRAINING:
            if not self._validation_gpu_uuids and all(
                worker.state is RuntimeState.COLD for worker in self.supervisor.workers.values()
            ):
                await self.database.set_service_mode(ServiceMode.MAINTENANCE_READY)
            return
        if mode is ServiceMode.MAINTENANCE_READY:
            return

        await self._route_ready_work()
        pressures = self._pressures()
        profile_map = {
            pressure.model_id: await self.profiles.for_model(pressure.model_id)
            for pressure in pressures
        }
        actions = self.planner.plan(
            now=datetime.now(UTC),
            all_gpu_uuids={device.uuid for device in self.inventory.gpus}
            - self._validation_gpu_uuids,
            workers=list(self.supervisor.workers.values()),
            pressures=pressures,
            profiles=profile_map,
        )
        for action in actions:
            if isinstance(action, DrainPlacement):
                await self.supervisor.drain(action.worker_id)
            elif isinstance(action, StartPlacement):
                await self._launch_if_active(action)

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
            await self.supervisor.launch(
                profile=action.profile,
                gpu_uuids=action.gpu_uuids,
                model_path=model["artifact_path"],
                served_model_name=model["nickname"],
            )

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
                    try:
                        await self.supervisor.admit(worker.id, request.id, request.estimated_tokens)
                        await self.database.mark_request_admitted(request.id, worker.id)
                    except Exception as exc:
                        if request.assignment and not request.assignment.done():
                            request.assignment.set_exception(exc)
                        continue
                    lease = WorkerLease(
                        worker_id=worker.id,
                        request_id=request.id,
                        reservation_id=request.reservation_id,
                        base_url=f"http://127.0.0.1:{worker.port}",
                        internal_api_key=self.supervisor.internal_api_key,
                        estimated_tokens=request.estimated_tokens,
                        admitted_at=datetime.now(UTC),
                    )
                    self._request_leases[request.id] = lease
                    if request.assignment and not request.assignment.done():
                        request.assignment.set_result(lease)

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
