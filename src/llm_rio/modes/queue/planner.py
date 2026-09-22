from __future__ import annotations

import math
from datetime import datetime

from llm_rio.domain import PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.modes.actions import (
    DrainPlacement,
    PlannerAction,
    QueuePressure,
    StartPlacement,
)


class QueuePlanner:
    """Small-host enumerating planner; all GPU groups came from measured profiles."""

    def __init__(
        self,
        *,
        wait_duration_seconds: float,
        minimum_residency_seconds: float,
        fair_share_seconds: float,
        scale_window_seconds: float = 30.0,
        minimum_marginal_efficiency: float = 0.05,
        gpu_vram_mib: dict[str, int] | None = None,
        reserved_vram_mib: int = 0,
        cache_max_workers_per_gpu: int = 1,
        cache_sleep_gpu_reserve_mib: int = 1536,
        cache_idle_sleep_seconds: float = 45.0,
    ) -> None:
        self.wait_duration_seconds = wait_duration_seconds
        self.minimum_residency_seconds = minimum_residency_seconds
        self.fair_share_seconds = fair_share_seconds
        self.scale_window_seconds = scale_window_seconds
        self.minimum_marginal_efficiency = minimum_marginal_efficiency
        self.gpu_vram_mib = gpu_vram_mib or {}
        self.reserved_vram_mib = reserved_vram_mib
        self.cache_max_workers_per_gpu = cache_max_workers_per_gpu
        self.cache_sleep_gpu_reserve_mib = cache_sleep_gpu_reserve_mib
        self.cache_idle_sleep_seconds = cache_idle_sleep_seconds

    def plan(
        self,
        *,
        now: datetime,
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
        pressures: list[QueuePressure],
        profiles: dict[str, list[PlacementProfile]],
    ) -> list[PlannerAction]:
        actions: list[PlannerAction] = []
        active = [worker for worker in workers if worker.state is not RuntimeState.COLD]
        used = {gpu for worker in active for gpu in worker.gpu_uuids}
        free = all_gpu_uuids - used
        pressure_by_model = {pressure.model_id: pressure for pressure in pressures}
        idle_workers = [
            worker
            for worker in active
            if worker.state is RuntimeState.READY
            and worker.model_id not in pressure_by_model
            and (not worker.admitted_request_ids)
            and ((now - worker.last_demand_at).total_seconds() >= self.wait_duration_seconds)
            and self._residency_satisfied(worker, now)
        ]
        for worker in idle_workers:
            actions.append(DrainPlacement(worker.id, "idle_timeout"))
        if actions:
            return actions
        for pressure in sorted(pressures, key=lambda item: item.oldest_enqueued_at):
            model_workers = [
                worker
                for worker in active
                if worker.model_id == pressure.model_id
                and worker.state in {RuntimeState.LOADING, RuntimeState.READY}
            ]
            candidates = profiles.get(pressure.model_id, [])
            if not candidates:
                continue
            if not model_workers:
                starts = self._maximum_smallest_placements(candidates, free)
                if starts:
                    return [
                        StartPlacement(profile, gpu_set, "cold_backlog")
                        for profile, gpu_set in starts
                    ]
                drain = self._choose_preemption(
                    now=now,
                    candidates=candidates,
                    workers=active,
                    pressure=pressure,
                    pressure_by_model=pressure_by_model,
                )
                if drain:
                    return [DrainPlacement(worker.id, "incompatible_backlog") for worker in drain]
                free.difference_update(
                    gpu
                    for profile in candidates
                    for gpu_set in profile.eligible_gpu_sets
                    for gpu in gpu_set
                )
                continue
            ready_workers = [
                worker for worker in model_workers if worker.state is RuntimeState.READY
            ]
            one_gpu_profiles = [profile for profile in candidates if profile.gpu_count == 1]
            if not ready_workers or not one_gpu_profiles:
                continue
            capacity = sum(
                worker.profile.predicted_tokens_per_second * self.scale_window_seconds
                for worker in ready_workers
            )
            desired = min(
                max(pressure.requests, 1),
                max(1, math.ceil(pressure.estimated_tokens / max(capacity, 1))),
            )
            if desired <= len(model_workers):
                continue
            starts = self._maximum_smallest_placements(one_gpu_profiles, free)
            useful = [
                start
                for start in starts[: max(0, desired - len(model_workers))]
                if self._replica_has_useful_margin(start[0], ready_workers)
            ]
            if useful:
                return [
                    StartPlacement(profile, gpu_set, "replica_backlog")
                    for profile, gpu_set in useful
                ]
            if not starts:
                drain = self._choose_preemption(
                    now=now,
                    candidates=candidates,
                    workers=active,
                    pressure=pressure,
                    pressure_by_model=pressure_by_model,
                )
                if drain:
                    return [DrainPlacement(worker.id, "replica_capacity") for worker in drain]
        return actions

    @classmethod
    def _maximum_smallest_placements(
        cls, profiles: list[PlacementProfile], free: set[str]
    ) -> list[tuple[PlacementProfile, tuple[str, ...]]]:
        """Fill free GPUs with independent instances of the smallest validated shape."""
        if not profiles:
            return []
        smallest_gpu_count = min(profile.gpu_count for profile in profiles)
        candidates = [profile for profile in profiles if profile.gpu_count == smallest_gpu_count]
        remaining = set(free)
        result: list[tuple[PlacementProfile, tuple[str, ...]]] = []
        while True:
            start = cls._smallest_fitting(candidates, remaining)
            if start is None:
                return result
            result.append(start)
            remaining.difference_update(start[1])

    def _replica_has_useful_margin(
        self, profile: PlacementProfile, ready_workers: list[WorkerPlacement]
    ) -> bool:
        current = sum(worker.profile.predicted_tokens_per_second for worker in ready_workers)
        if current <= 0:
            return True
        marginal = profile.predicted_tokens_per_second / current
        return marginal >= self.minimum_marginal_efficiency

    def _residency_satisfied(self, worker: WorkerPlacement, now: datetime) -> bool:
        if self.minimum_residency_seconds == 0:
            return True
        if worker.ready_at is None:
            return False
        return (now - worker.ready_at).total_seconds() >= self.minimum_residency_seconds

    def _choose_queue_preemption(
        self,
        *,
        now: datetime,
        candidates: list[PlacementProfile],
        workers: list[WorkerPlacement],
        pressure: QueuePressure,
        pressure_by_model: dict[str, QueuePressure],
    ) -> list[WorkerPlacement] | None:
        options: list[list[WorkerPlacement]] = []
        for profile in sorted(candidates, key=lambda item: item.gpu_count):
            for gpu_set in profile.eligible_gpu_sets:
                blockers = [w for w in workers if set(w.gpu_uuids) & set(gpu_set)]
                if not blockers or any(
                    w.state is not RuntimeState.READY
                    or w.model_id == pressure.model_id
                    or (not self._residency_satisfied(w, now))
                    for w in blockers
                ):
                    continue
                if any(
                    w.model_id in pressure_by_model
                    and pressure_by_model[w.model_id].oldest_enqueued_at
                    < pressure.oldest_enqueued_at
                    and (
                        not any(
                            other.model_id == w.model_id
                            and other not in blockers
                            and (other.state is RuntimeState.READY)
                            for other in workers
                        )
                    )
                    for w in blockers
                ):
                    continue
                options.append(blockers)
        return (
            min(
                options,
                key=lambda group: (
                    len(group),
                    sum(w.profile.predicted_tokens_per_second for w in group),
                ),
            )
            if options
            else None
        )

    def _choose_preemption(
        self,
        *,
        now: datetime,
        candidates: list[PlacementProfile],
        workers: list[WorkerPlacement],
        pressure: QueuePressure,
        pressure_by_model: dict[str, QueuePressure],
    ) -> list[WorkerPlacement] | None:
        return self._choose_queue_preemption(
            now=now,
            candidates=candidates,
            workers=workers,
            pressure=pressure,
            pressure_by_model=pressure_by_model,
        )
        smallest_gpu_count = min(profile.gpu_count for profile in candidates)
        compatible_gpus = {
            gpu
            for profile in candidates
            if profile.gpu_count == smallest_gpu_count
            for gpu_set in profile.eligible_gpu_sets
            for gpu in gpu_set
        }
        idle_blockers = [
            worker
            for worker in workers
            if worker.model_id not in pressure_by_model
            and (not worker.admitted_request_ids)
            and self._residency_satisfied(worker, now)
            and bool(set(worker.gpu_uuids) & compatible_gpus)
        ]
        if idle_blockers:
            return sorted(
                idle_blockers, key=lambda worker: worker.profile.predicted_tokens_per_second
            )
        resident_counts: dict[str, int] = {}
        for worker in workers:
            if worker.state in {RuntimeState.LOADING, RuntimeState.READY}:
                resident_counts[worker.model_id] = resident_counts.get(worker.model_id, 0) + 1
        for profile in sorted(candidates, key=lambda item: item.gpu_count):
            for required_set in profile.eligible_gpu_sets:
                blockers = [
                    worker for worker in workers if set(worker.gpu_uuids) & set(required_set)
                ]
                if any(worker.model_id == pressure.model_id for worker in blockers):
                    continue
                if any(worker.model_id not in pressure_by_model for worker in blockers):
                    continue
                if not blockers or any(
                    worker.state is RuntimeState.DRAINING for worker in blockers
                ):
                    continue
                selected_counts: dict[str, int] = {}
                drainable = True
                for worker in blockers:
                    if worker.model_id not in pressure_by_model:
                        continue
                    ready_at = worker.ready_at or now
                    fair_wait_started = max(pressure.oldest_enqueued_at, ready_at)
                    starvation_override = (
                        now - fair_wait_started
                    ).total_seconds() >= self.fair_share_seconds
                    if starvation_override:
                        continue
                    if not self._residency_satisfied(worker, now):
                        drainable = False
                        break
                    already_selected = selected_counts.get(worker.model_id, 0)
                    remaining = resident_counts.get(worker.model_id, 0) - already_selected - 1
                    if remaining < 1:
                        drainable = False
                        break
                    selected_counts[worker.model_id] = already_selected + 1
                if drainable:
                    return sorted(
                        blockers,
                        key=lambda worker: (
                            worker.model_id in pressure_by_model,
                            worker.profile.predicted_tokens_per_second,
                        ),
                    )
        return None

    @staticmethod
    def _smallest_fitting(
        profiles: list[PlacementProfile], free: set[str]
    ) -> tuple[PlacementProfile, tuple[str, ...]] | None:
        for profile in sorted(
            profiles, key=lambda item: (item.gpu_count, -item.predicted_tokens_per_second)
        ):
            for gpu_set in profile.eligible_gpu_sets:
                if set(gpu_set) <= free:
                    return (profile, gpu_set)
        return None
