from __future__ import annotations

from typing import TYPE_CHECKING

from llm_rio.domain import PlacementProfile, WorkerPlacement
from llm_rio.host_memory import (
    HostMemorySample,
)
from llm_rio.worker_types import WorkerLaunchError

if TYPE_CHECKING:
    from llm_rio.workers import WorkerSupervisor


class QueueLifecycle:
    def __init__(self, supervisor: WorkerSupervisor) -> None:
        self.supervisor = supervisor

    async def _refresh_cached_worker_samples(self) -> None:
        return None

    def _host_cache_limit_mib(self, host: HostMemorySample) -> float:
        return 0.0

    def _host_cache_pressure_reason(
        self, host: HostMemorySample, cache_accounted_mib: float, cache_swap_mib: float = 0.0
    ) -> str | None:
        return None

    async def host_cache_status(self) -> dict[str, float | str | None]:
        return {}

    async def enforce_host_cache_budget(self, *, incoming_worker_id: str | None = None) -> bool:
        return True

    def _can_share_gpus(
        self,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        overlapping_workers: list[WorkerPlacement],
    ) -> bool:
        return False

    async def sleep(self, worker_id: str) -> None:
        raise WorkerLaunchError("Queue mode does not support sleep or wake")

    async def _offload(self, worker_id: str) -> None:
        raise WorkerLaunchError("Queue mode does not support sleep or wake")

    async def wake(self, worker_id: str) -> None:
        raise WorkerLaunchError("Queue mode does not support sleep or wake")

    async def _post_engine(
        self, worker: WorkerPlacement, path: str, *, params: dict[str, str] | None = None
    ) -> None:
        raise WorkerLaunchError("Queue mode does not support sleep or wake")
