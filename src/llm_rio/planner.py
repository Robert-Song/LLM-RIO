from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations

from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement


@dataclass(frozen=True, slots=True)
class QueuePressure:
    model_id: str
    requests: int
    estimated_tokens: int
    oldest_enqueued_at: datetime
    preload: bool = False


@dataclass(frozen=True, slots=True)
class StartPlacement:
    profile: PlacementProfile
    gpu_uuids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class DrainPlacement:
    worker_id: str
    reason: str


PlannerAction = StartPlacement | DrainPlacement


class GreedyPlacementPlanner:
    """Small-host enumerating planner; all GPU groups came from measured profiles."""

    def __init__(
        self,
        *,
        wait_duration_seconds: float,
        minimum_residency_seconds: float,
        fair_share_seconds: float,
        scale_window_seconds: float = 30.0,
        minimum_marginal_efficiency: float = 0.05,
        prism_enabled: bool = False,
        gpu_vram_mib: dict[str, int] | None = None,
        reserved_vram_mib: int = 0,
        prism_max_workers_per_gpu: int = 2,
    ) -> None:
        self.wait_duration_seconds = wait_duration_seconds
        self.minimum_residency_seconds = minimum_residency_seconds
        self.fair_share_seconds = fair_share_seconds
        self.scale_window_seconds = scale_window_seconds
        self.minimum_marginal_efficiency = minimum_marginal_efficiency
        self.prism_enabled = prism_enabled
        self.gpu_vram_mib = gpu_vram_mib or {}
        self.reserved_vram_mib = reserved_vram_mib
        self.prism_max_workers_per_gpu = prism_max_workers_per_gpu

    def plan(
        self,
        *,
        now: datetime,
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
        pressures: list[QueuePressure],
        profiles: dict[str, list[PlacementProfile]],
    ) -> list[PlannerAction]:
        if self.prism_enabled:
            return self._plan_prism(
                now=now,
                all_gpu_uuids=all_gpu_uuids,
                workers=workers,
                pressures=pressures,
                profiles=profiles,
            )

        actions: list[PlannerAction] = []
        active = [
            worker
            for worker in workers
            if worker.state in {RuntimeState.LOADING, RuntimeState.READY, RuntimeState.DRAINING}
        ]
        used = {gpu for worker in active for gpu in worker.gpu_uuids}
        free = all_gpu_uuids - used
        pressure_by_model = {pressure.model_id: pressure for pressure in pressures}

        idle_workers = [
            worker
            for worker in active
            if worker.state is RuntimeState.READY
            and worker.model_id not in pressure_by_model
            and not worker.admitted_request_ids
            and (now - worker.last_demand_at).total_seconds() >= self.wait_duration_seconds
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
            desired = max(1, math.ceil(pressure.estimated_tokens / max(capacity, 1)))
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

    def _plan_prism(
        self,
        *,
        now: datetime,
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
        pressures: list[QueuePressure],
        profiles: dict[str, list[PlacementProfile]],
    ) -> list[PlannerAction]:
        """Keep validated vLLM engines resident and share their elastic KV pool."""
        active = [
            worker
            for worker in workers
            if worker.state in {RuntimeState.LOADING, RuntimeState.READY, RuntimeState.DRAINING}
        ]
        real_pressure = {
            pressure.model_id: pressure for pressure in pressures if not pressure.preload
        }
        for pressure in sorted(
            pressures,
            key=lambda item: (item.preload, item.oldest_enqueued_at),
        ):
            model_workers = [
                worker
                for worker in active
                if worker.model_id == pressure.model_id
                and worker.state in {RuntimeState.LOADING, RuntimeState.READY}
            ]
            candidates = [
                profile
                for profile in profiles.get(pressure.model_id, [])
                if profile.engine is Engine.VLLM and profile.memory_backend == "kvcached"
            ]
            if not candidates:
                continue

            if not model_workers:
                start = self._smallest_prism_fitting(candidates, all_gpu_uuids, active)
                if start is not None:
                    reason = "prism_preload" if pressure.preload else "prism_cold_backlog"
                    return [StartPlacement(start[0], start[1], reason)]
                if pressure.preload:
                    continue
                drain = self._choose_prism_eviction(
                    now=now,
                    candidates=candidates,
                    all_gpu_uuids=all_gpu_uuids,
                    active=active,
                    pressure=pressure,
                    real_pressure=real_pressure,
                )
                if drain:
                    return [DrainPlacement(worker.id, "prism_weight_capacity") for worker in drain]
                continue

            if pressure.preload or any(
                worker.state is RuntimeState.LOADING for worker in model_workers
            ):
                continue
            ready_workers = [
                worker for worker in model_workers if worker.state is RuntimeState.READY
            ]
            capacity = sum(
                worker.profile.predicted_tokens_per_second * self.scale_window_seconds
                for worker in ready_workers
            )
            desired = max(1, math.ceil(pressure.estimated_tokens / max(capacity, 1)))
            if desired <= len(model_workers):
                continue
            start = self._smallest_prism_fitting(candidates, all_gpu_uuids, active)
            if start is not None and self._replica_has_useful_margin(start[0], ready_workers):
                return [StartPlacement(start[0], start[1], "prism_replica_backlog")]
        return []

    def _smallest_prism_fitting(
        self,
        profiles: list[PlacementProfile],
        all_gpu_uuids: set[str],
        active: list[WorkerPlacement],
    ) -> tuple[PlacementProfile, tuple[str, ...]] | None:
        for profile in sorted(
            profiles,
            key=lambda item: (
                item.gpu_count,
                sum(item.idle_vram_mib_per_gpu),
                -item.predicted_tokens_per_second,
            ),
        ):
            for gpu_set in profile.eligible_gpu_sets:
                if set(gpu_set) <= all_gpu_uuids and self._prism_fits(profile, gpu_set, active):
                    return profile, gpu_set
        return None

    def _prism_fits(
        self,
        profile: PlacementProfile,
        gpu_set: tuple[str, ...],
        active: list[WorkerPlacement],
    ) -> bool:
        if (
            len(profile.idle_vram_mib_per_gpu) != len(gpu_set)
            or len(profile.peak_vram_mib_per_gpu) != len(gpu_set)
            or len(profile.gpu_headroom_mib_per_gpu) != len(gpu_set)
        ):
            return False
        if any(
            worker.model_id == profile.model_id and worker.gpu_uuids == gpu_set for worker in active
        ):
            return False
        for gpu_uuid, requested_mib in zip(gpu_set, profile.idle_vram_mib_per_gpu, strict=True):
            colocated = [worker for worker in active if gpu_uuid in worker.gpu_uuids]
            if len(colocated) >= self.prism_max_workers_per_gpu:
                return False
            if any(
                worker.profile.engine is not Engine.VLLM
                or worker.profile.memory_backend != "kvcached"
                for worker in colocated
            ):
                return False
            used_mib = 0
            for worker in colocated:
                index = worker.gpu_uuids.index(gpu_uuid)
                if (
                    index >= len(worker.profile.idle_vram_mib_per_gpu)
                    or index >= len(worker.profile.peak_vram_mib_per_gpu)
                    or index >= len(worker.profile.gpu_headroom_mib_per_gpu)
                ):
                    return False
                used_mib += (
                    max(
                        worker.profile.idle_vram_mib_per_gpu[index],
                        worker.profile.peak_vram_mib_per_gpu[index],
                    )
                    + worker.profile.gpu_headroom_mib_per_gpu[index]
                )
            total_mib = self.gpu_vram_mib.get(gpu_uuid, 0)
            utilization = min(
                [profile.gpu_memory_utilization]
                + [worker.profile.gpu_memory_utilization for worker in colocated]
            )
            budget_mib = min(
                total_mib - self.reserved_vram_mib,
                math.floor(total_mib * utilization),
            )
            requested_footprint_mib = (
                max(requested_mib, profile.peak_vram_mib_per_gpu[gpu_set.index(gpu_uuid)])
                + profile.gpu_headroom_mib_per_gpu[gpu_set.index(gpu_uuid)]
            )
            if requested_footprint_mib + used_mib > budget_mib:
                return False
        return True

    def _choose_prism_eviction(
        self,
        *,
        now: datetime,
        candidates: list[PlacementProfile],
        all_gpu_uuids: set[str],
        active: list[WorkerPlacement],
        pressure: QueuePressure,
        real_pressure: dict[str, QueuePressure],
    ) -> list[WorkerPlacement] | None:
        resident_counts: dict[str, int] = {}
        for worker in active:
            if worker.state in {RuntimeState.LOADING, RuntimeState.READY}:
                resident_counts[worker.model_id] = resident_counts.get(worker.model_id, 0) + 1
        incoming_starved = (
            now - pressure.oldest_enqueued_at
        ).total_seconds() >= self.fair_share_seconds

        options: list[tuple[int, float, list[WorkerPlacement]]] = []
        for profile in candidates:
            for gpu_set in profile.eligible_gpu_sets:
                if not set(gpu_set) <= all_gpu_uuids:
                    continue
                blockers = [
                    worker
                    for worker in active
                    if worker.model_id != pressure.model_id
                    and worker.state is RuntimeState.READY
                    and bool(set(worker.gpu_uuids) & set(gpu_set))
                    and not worker.admitted_request_ids
                    and self._residency_satisfied(worker, now)
                    and (
                        worker.model_id not in real_pressure
                        or resident_counts.get(worker.model_id, 0) > 1
                        or incoming_starved
                    )
                ]
                found_for_gpu_set = False
                for count in range(1, len(blockers) + 1):
                    for selected_tuple in combinations(blockers, count):
                        selected = list(selected_tuple)
                        remaining = [worker for worker in active if worker not in selected]
                        if self._prism_fits(profile, gpu_set, remaining):
                            lost_throughput = sum(
                                worker.profile.predicted_tokens_per_second for worker in selected
                            )
                            options.append((count, lost_throughput, selected))
                            found_for_gpu_set = True
                    if found_for_gpu_set:
                        break
        if not options:
            return None
        return min(options, key=lambda item: (item[0], item[1]))[2]

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

    @staticmethod
    def _smallest_fitting(
        profiles: list[PlacementProfile], free: set[str]
    ) -> tuple[PlacementProfile, tuple[str, ...]] | None:
        for profile in sorted(
            profiles,
            key=lambda item: (item.gpu_count, -item.predicted_tokens_per_second),
        ):
            for gpu_set in profile.eligible_gpu_sets:
                if set(gpu_set) <= free:
                    return profile, gpu_set
        return None

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

    def _choose_preemption(
        self,
        *,
        now: datetime,
        candidates: list[PlacementProfile],
        workers: list[WorkerPlacement],
        pressure: QueuePressure,
        pressure_by_model: dict[str, QueuePressure],
    ) -> list[WorkerPlacement] | None:
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
            and not worker.admitted_request_ids
            and self._residency_satisfied(worker, now)
            and bool(set(worker.gpu_uuids) & compatible_gpus)
        ]
        if idle_blockers:
            return sorted(
                idle_blockers,
                key=lambda worker: worker.profile.predicted_tokens_per_second,
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
                    fair_wait_started = max(
                        pressure.oldest_enqueued_at,
                        ready_at,
                    )
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

                # Only drain when the entire validated incoming GPU set becomes free.
                # This prevents useless partial unloads for tensor-parallel models.
                if drainable:
                    return sorted(
                        blockers,
                        key=lambda worker: (
                            worker.model_id in pressure_by_model,
                            worker.profile.predicted_tokens_per_second,
                        ),
                    )
        return None


def utc_now() -> datetime:
    return datetime.now(UTC)
