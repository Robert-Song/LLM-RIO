from __future__ import annotations

import asyncio

from llm_rio.domain import MachineInventory, RuntimeState, ServiceMode
from llm_rio.inventory import read_live_gpu_status
from llm_rio.runtime import ResidencyScheduler
from llm_rio.storage import Database


class DiagnosticsService:
    def __init__(
        self, database: Database, scheduler: ResidencyScheduler, inventory: MachineInventory
    ) -> None:
        self.database = database
        self.scheduler = scheduler
        self.inventory = inventory

    async def status(self) -> dict[str, object]:
        database = self.database
        models = {model["id"]: model["nickname"] for model in await database.list_models()}
        scheduler = self.scheduler
        host_cache = await self.scheduler.supervisor.host_cache_status()
        workers = []
        for worker in self.scheduler.supervisor.workers.values():
            workers.append(
                {
                    "worker_id": worker.id,
                    "pid": worker.process_pid,
                    "model": models.get(worker.model_id, worker.model_id),
                    "state": (
                        "STOPPED" if worker.state is RuntimeState.COLD else worker.state.value
                    ),
                    "weight_storage": (
                        "host_ram"
                        if worker.state is RuntimeState.SLEEPING
                        else "transitioning"
                        if worker.state in {RuntimeState.OFFLOADING, RuntimeState.WAKING}
                        else "gpu+host_ram"
                        if worker.host_weights_cached
                        else "gpu"
                        if worker.state is not RuntimeState.COLD
                        else "none"
                    ),
                    "gpu_uuids": worker.gpu_uuids,
                    "profile_id": worker.profile.id,
                    "ready_at": worker.ready_at.isoformat() if worker.ready_at else None,
                    "sleeping_at": (worker.sleeping_at.isoformat() if worker.sleeping_at else None),
                    "last_activation_seconds": worker.last_activation_seconds,
                    "last_offload_seconds": worker.last_offload_seconds,
                    "tensor_parallel_size": worker.profile.tensor_parallel_size,
                    "active_requests": len(worker.admitted_request_ids),
                    "queued_requests": len(scheduler.queues.for_model(worker.model_id)),
                    "accepted_requests": worker.accepted_requests,
                    "host_cache_accounted_mib": worker.host_cache_accounted_mib,
                    "host_cache_accounting_source": worker.host_cache_accounting_source,
                    "process_rss_mib": worker.process_rss_mib,
                    "process_pss_mib": worker.process_pss_mib,
                    "process_swap_mib": worker.process_swap_mib,
                    "last_cache_eviction_reason": worker.last_cache_eviction_reason,
                }
            )
        mode: ServiceMode = await database.service_mode()
        return {
            "mode": mode.value,
            "serving_mode": scheduler.serving_mode,
            "validation": {
                "requires_maintenance": scheduler.validation_requires_maintenance,
                "gpu_uuids": scheduler.validation_gpu_uuids,
            },
            "residency": {
                "kvcached": scheduler.kvcached.enabled,
                "weight_cache": (
                    "host_ram" if self.scheduler.supervisor.ram_weight_cache_enabled else "disabled"
                ),
                "cached_workers": sum(
                    worker.state is RuntimeState.SLEEPING
                    for worker in self.scheduler.supervisor.workers.values()
                ),
                "host_memory": host_cache,
            },
            "workers": workers,
            "resource_ownership": scheduler.resource_ownership_snapshot(),
            "queued_models": {
                models.get(model_id, model_id): len(scheduler.queues.for_model(model_id))
                for model_id in scheduler.queues.pending_models()
            },
        }

    async def dashboard(self) -> dict[str, object]:
        database = self.database
        usage, gpu_samples, live_requests = await asyncio.gather(
            database.dashboard_usage(),
            asyncio.to_thread(read_live_gpu_status, self.inventory),
            database.live_requests(),
        )
        models = {
            str(model["id"]): str(model["nickname"]) for model in await database.list_models()
        }
        scheduler = self.scheduler
        placements_by_gpu: dict[str, list[dict[str, object]]] = {}
        for worker in self.scheduler.supervisor.workers.values():
            if worker.state is RuntimeState.COLD:
                continue
            active_slots = len(worker.admitted_request_ids)
            capacity = worker.profile.max_num_seqs
            placement: dict[str, object] = {
                "worker_id": worker.id,
                "pid": worker.process_pid,
                "model_id": worker.model_id,
                "model": models.get(worker.model_id, worker.model_id),
                "engine": worker.profile.engine.value,
                "state": worker.state.value,
                "weight_storage": (
                    "host_ram"
                    if worker.state is RuntimeState.SLEEPING
                    else "transitioning"
                    if worker.state in {RuntimeState.OFFLOADING, RuntimeState.WAKING}
                    else "gpu+host_ram"
                    if worker.host_weights_cached
                    else "gpu"
                ),
                "last_activation_seconds": worker.last_activation_seconds,
                "last_offload_seconds": worker.last_offload_seconds,
                "continuous_batching_slots": {
                    "active": active_slots,
                    "capacity": capacity,
                    "available": max(0, capacity - active_slots) if capacity is not None else None,
                },
                "max_batched_tokens": worker.profile.max_num_batched_tokens,
                "outstanding_token_work": worker.outstanding_token_work,
                "queued_requests": len(scheduler.queues.for_model(worker.model_id)),
            }
            for gpu_uuid in worker.gpu_uuids:
                placements_by_gpu.setdefault(gpu_uuid, []).append(placement)

        gpus: list[dict[str, object]] = []
        for sample in gpu_samples:
            item = dict(sample)
            item["placements"] = placements_by_gpu.get(str(sample["uuid"]), [])
            gpus.append(item)

        return {
            "generated_at": usage["generated_at"],
            "mode": (await database.service_mode()).value,
            "usage": usage,
            "gpus": gpus,
            "requests": live_requests,
        }
