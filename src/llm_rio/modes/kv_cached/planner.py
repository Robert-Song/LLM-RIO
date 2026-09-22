from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime
from itertools import combinations

from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.modes.actions import (
    DrainPlacement,
    PlannerAction,
    QueuePressure,
    SleepPlacement,
    StartPlacement,
    WakePlacement,
)
from llm_rio.profiles import profile_verified_for_mode


class KVCachedPlanner:
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
        """Combine backend-appropriate GPU placement with host-RAM weights."""
        live_workers = [
            worker
            for worker in workers
            if worker.state not in {RuntimeState.COLD, RuntimeState.STOPPING}
        ]
        real_pressure = {
            pressure.model_id: pressure for pressure in pressures if not pressure.preload
        }
        idle_workers = [
            worker
            for worker in live_workers
            if worker.state is RuntimeState.READY
            and worker.model_id not in real_pressure
            and (not worker.admitted_request_ids)
            and ((now - worker.last_demand_at).total_seconds() >= self.cache_idle_sleep_seconds)
            and self._residency_satisfied(worker, now)
        ]
        if idle_workers:
            return [
                SleepPlacement(worker.id, "residency_idle_weight_cache") for worker in idle_workers
            ]
        for pressure in sorted(pressures, key=lambda item: (item.preload, item.oldest_enqueued_at)):
            candidates = [
                profile
                for profile in profiles.get(pressure.model_id, [])
                if profile.engine is Engine.VLLM
                and profile_verified_for_mode(
                    profile, kvcached_required=True, ram_weight_cache_required=True
                )
            ]
            if not candidates:
                continue
            model_workers = [
                worker for worker in live_workers if worker.model_id == pressure.model_id
            ]
            transitioning = [
                worker
                for worker in model_workers
                if worker.state
                in {
                    RuntimeState.LOADING,
                    RuntimeState.DRAINING,
                    RuntimeState.OFFLOADING,
                    RuntimeState.WAKING,
                }
            ]
            ready_workers = [
                worker for worker in model_workers if worker.state is RuntimeState.READY
            ]
            sleeping_workers = [
                worker
                for worker in model_workers
                if worker.state is RuntimeState.SLEEPING
                and worker.profile.id in {profile.id for profile in candidates}
            ]
            if pressure.preload:
                if len(model_workers) >= pressure.desired_workers:
                    continue
                if transitioning:
                    continue
                preload_candidates = (
                    [profile for profile in candidates if profile.gpu_count == 1]
                    if pressure.desired_workers > 1
                    else candidates
                )
                if not preload_candidates:
                    continue
                start = self._smallest_cached_cache_fitting(
                    preload_candidates, all_gpu_uuids, live_workers
                )
                if start is not None:
                    return [StartPlacement(start[0], start[1], "residency_preload")]
                blockers = self._choose_cache_sleep(
                    now=now,
                    candidates=preload_candidates,
                    all_gpu_uuids=all_gpu_uuids,
                    workers=live_workers,
                    pressure=pressure,
                    real_pressure=real_pressure,
                )
                if blockers:
                    return [
                        SleepPlacement(worker.id, "residency_weight_capacity")
                        for worker in blockers
                    ]
                continue
            if transitioning:
                continue
            if not ready_workers:
                if sleeping_workers:
                    per_worker_capacity = max(
                        worker.profile.predicted_tokens_per_second * self.scale_window_seconds
                        for worker in sleeping_workers
                    )
                    desired = min(
                        max(pressure.requests, 1),
                        max(1, math.ceil(pressure.estimated_tokens / max(per_worker_capacity, 1))),
                    )
                    wakes = self._cached_wakes(
                        sleeping_workers=sleeping_workers,
                        all_gpu_uuids=all_gpu_uuids,
                        live_workers=live_workers,
                        limit=desired,
                    )
                    if wakes:
                        return [
                            WakePlacement(
                                worker.id,
                                "residency_ram_cache_hit"
                                if index == 0
                                else "residency_replica_ram_cache_hit",
                            )
                            for index, worker in enumerate(wakes)
                        ]
                    blockers = self._choose_cache_sleep(
                        now=now,
                        candidates=[worker.profile for worker in sleeping_workers],
                        all_gpu_uuids=all_gpu_uuids,
                        workers=live_workers,
                        pressure=pressure,
                        real_pressure=real_pressure,
                        existing_workers=sleeping_workers,
                    )
                    if blockers:
                        return [
                            SleepPlacement(worker.id, "residency_weight_capacity")
                            for worker in blockers
                        ]
                    continue
                start = self._smallest_cached_cache_fitting(candidates, all_gpu_uuids, live_workers)
                if start is not None:
                    reason = "residency_preload" if pressure.preload else "residency_cold_backlog"
                    return [StartPlacement(start[0], start[1], reason)]
                blockers = self._choose_cache_sleep(
                    now=now,
                    candidates=candidates,
                    all_gpu_uuids=all_gpu_uuids,
                    workers=live_workers,
                    pressure=pressure,
                    real_pressure=real_pressure,
                )
                if blockers:
                    return [
                        SleepPlacement(worker.id, "residency_weight_capacity")
                        for worker in blockers
                    ]
                if not pressure.preload:
                    evictions = self._choose_cache_cache_eviction(
                        candidates=candidates,
                        all_gpu_uuids=all_gpu_uuids,
                        workers=live_workers,
                        real_pressure=real_pressure,
                    )
                    if evictions:
                        return [
                            DrainPlacement(worker.id, "residency_ram_cache_lru")
                            for worker in evictions
                        ]
                continue
            capacity = sum(
                worker.profile.predicted_tokens_per_second * self.scale_window_seconds
                for worker in ready_workers
            )
            desired = min(
                max(pressure.requests, 1),
                max(1, math.ceil(pressure.estimated_tokens / max(capacity, 1))),
            )
            if desired <= len(ready_workers):
                continue
            one_gpu_candidates = [profile for profile in candidates if profile.gpu_count == 1]
            one_gpu_sleeping = [
                worker for worker in sleeping_workers if worker.profile.gpu_count == 1
            ]
            if not one_gpu_candidates:
                continue
            wake = self._first_cached_wake(
                sleeping_workers=one_gpu_sleeping,
                all_gpu_uuids=all_gpu_uuids,
                live_workers=live_workers,
            )
            if wake is not None:
                return [WakePlacement(wake.id, "residency_replica_ram_cache_hit")]
            start = self._smallest_cached_cache_fitting(
                one_gpu_candidates, all_gpu_uuids, live_workers
            )
            if start is not None and self._replica_has_useful_margin(start[0], ready_workers):
                return [StartPlacement(start[0], start[1], "residency_replica_backlog")]
            blockers = self._choose_cache_sleep(
                now=now,
                candidates=one_gpu_candidates,
                all_gpu_uuids=all_gpu_uuids,
                workers=live_workers,
                pressure=pressure,
                real_pressure=real_pressure,
            )
            if blockers:
                return [
                    SleepPlacement(worker.id, "residency_replica_capacity") for worker in blockers
                ]
        return []

    def _cached_wakes(
        self,
        *,
        sleeping_workers: list[WorkerPlacement],
        all_gpu_uuids: set[str],
        live_workers: list[WorkerPlacement],
        limit: int,
    ) -> list[WorkerPlacement]:
        selected: list[WorkerPlacement] = []
        remaining = list(sleeping_workers)
        projected = list(live_workers)
        while remaining and len(selected) < limit:
            wake = self._first_cached_wake(
                sleeping_workers=remaining, all_gpu_uuids=all_gpu_uuids, live_workers=projected
            )
            if wake is None:
                break
            selected.append(wake)
            remaining = [worker for worker in remaining if worker.id != wake.id]
            projected = [
                replace(worker, state=RuntimeState.WAKING) if worker.id == wake.id else worker
                for worker in projected
            ]
        return selected

    def _first_cached_wake(
        self,
        *,
        sleeping_workers: list[WorkerPlacement],
        all_gpu_uuids: set[str],
        live_workers: list[WorkerPlacement],
    ) -> WorkerPlacement | None:
        for worker in sorted(
            sleeping_workers,
            key=lambda item: (
                item.last_activation_seconds
                if item.last_activation_seconds is not None
                else item.profile.load_and_warmup_seconds,
                item.last_demand_at,
            ),
        ):
            if set(worker.gpu_uuids) <= all_gpu_uuids and self._cached_cache_fits(
                worker.profile, worker.gpu_uuids, live_workers, existing_worker_id=worker.id
            ):
                return worker
        return None

    def _smallest_cached_cache_fitting(
        self,
        profiles: list[PlacementProfile],
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
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
                if set(gpu_set) <= all_gpu_uuids and self._cached_cache_fits(
                    profile, gpu_set, workers
                ):
                    return (profile, gpu_set)
        return None

    @staticmethod
    def _active_vram_mib(profile: PlacementProfile, index: int) -> int | None:
        if index >= len(profile.idle_vram_mib_per_gpu) or index >= len(
            profile.peak_vram_mib_per_gpu
        ):
            return None
        values = [profile.idle_vram_mib_per_gpu[index], profile.peak_vram_mib_per_gpu[index]]
        wake_peak = profile.wake_peak_vram_mib_per_gpu
        if wake_peak is not None:
            if index >= len(wake_peak):
                return None
            values.append(wake_peak[index])
        return max(values)

    def _cached_cache_fits(
        self,
        profile: PlacementProfile,
        gpu_set: tuple[str, ...],
        workers: list[WorkerPlacement],
        *,
        existing_worker_id: str | None = None,
        prospective_sleep: frozenset[str] = frozenset(),
    ) -> bool:
        if profile.engine is not Engine.VLLM or not profile_verified_for_mode(
            profile, kvcached_required=True, ram_weight_cache_required=True
        ):
            return False
        live = [worker for worker in workers if worker.state is not RuntimeState.COLD]
        if any(
            worker.id != existing_worker_id
            and worker.model_id == profile.model_id
            and (worker.gpu_uuids == gpu_set)
            for worker in live
        ):
            return False
        gpu_resident_states = {
            RuntimeState.LOADING,
            RuntimeState.READY,
            RuntimeState.DRAINING,
            RuntimeState.OFFLOADING,
            RuntimeState.WAKING,
            RuntimeState.STOPPING,
        }
        for requested_index, gpu_uuid in enumerate(gpu_set):
            colocated = [worker for worker in live if gpu_uuid in worker.gpu_uuids]
            if any(
                worker.profile.engine is not Engine.VLLM
                or not profile_verified_for_mode(
                    worker.profile, kvcached_required=True, ram_weight_cache_required=True
                )
                for worker in colocated
            ):
                return False
            active = [
                worker
                for worker in colocated
                if worker.id != existing_worker_id
                and worker.id not in prospective_sleep
                and (worker.state in gpu_resident_states)
            ]
            active_limit = self.cache_max_workers_per_gpu
            if len(active) + 1 > active_limit:
                return False
            used_mib = 0
            for worker in active:
                worker_index = worker.gpu_uuids.index(gpu_uuid)
                footprint = self._active_vram_mib(worker.profile, worker_index)
                if footprint is None:
                    return False
                used_mib += footprint
            sleeping = [
                worker
                for worker in colocated
                if worker.id != existing_worker_id
                and (worker.state is RuntimeState.SLEEPING or worker.id in prospective_sleep)
            ]
            for sleeping_worker in sleeping:
                worker_index = sleeping_worker.gpu_uuids.index(gpu_uuid)
                measured = sleeping_worker.profile.sleep_vram_mib_per_gpu
                if measured is None or worker_index >= len(measured):
                    return False
                used_mib += measured[worker_index]
            requested_mib = self._active_vram_mib(profile, requested_index)
            if requested_mib is None:
                return False
            budget_mib = self.gpu_vram_mib.get(gpu_uuid, 0) - self.reserved_vram_mib
            if used_mib + requested_mib > budget_mib:
                return False
        return True

    def _choose_cache_sleep(
        self,
        *,
        now: datetime,
        candidates: list[PlacementProfile],
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
        pressure: QueuePressure,
        real_pressure: dict[str, QueuePressure],
        existing_workers: list[WorkerPlacement] | None = None,
    ) -> list[WorkerPlacement] | None:
        resident_counts: dict[str, int] = {}
        for worker in workers:
            if worker.state is RuntimeState.READY:
                resident_counts[worker.model_id] = resident_counts.get(worker.model_id, 0) + 1
        incoming_starved = (
            now - pressure.oldest_enqueued_at
        ).total_seconds() >= self.fair_share_seconds
        placements: list[tuple[PlacementProfile, tuple[str, ...], str | None]]
        if existing_workers is None:
            placements = [
                (profile, gpu_set, None)
                for profile in candidates
                for gpu_set in profile.eligible_gpu_sets
            ]
        else:
            candidate_ids = {profile.id for profile in candidates}
            placements = [
                (worker.profile, worker.gpu_uuids, worker.id)
                for worker in existing_workers
                if worker.profile.id in candidate_ids
            ]
        options: list[tuple[int, float, float, list[WorkerPlacement]]] = []
        for profile, gpu_set, existing_worker_id in placements:
            if not set(gpu_set) <= all_gpu_uuids:
                continue
            blockers = [
                worker
                for worker in workers
                if worker.model_id != pressure.model_id
                and worker.state is RuntimeState.READY
                and bool(set(worker.gpu_uuids) & set(gpu_set))
                and (not worker.admitted_request_ids)
                and self._residency_satisfied(worker, now)
                and (
                    worker.model_id not in real_pressure
                    or resident_counts.get(worker.model_id, 0) > 1
                    or incoming_starved
                )
            ]
            for count in range(1, len(blockers) + 1):
                found = False
                for selected_tuple in combinations(blockers, count):
                    selected = list(selected_tuple)
                    selected_ids = frozenset(worker.id for worker in selected)
                    if not self._cached_cache_fits(
                        profile,
                        gpu_set,
                        workers,
                        existing_worker_id=existing_worker_id,
                        prospective_sleep=selected_ids,
                    ):
                        continue
                    lost_throughput = sum(
                        worker.profile.predicted_tokens_per_second for worker in selected
                    )
                    last_demand_score = sum(
                        worker.last_demand_at.timestamp() for worker in selected
                    )
                    options.append((count, lost_throughput, last_demand_score, selected))
                    found = True
                if found:
                    break
        if not options:
            return None
        return min(options, key=lambda item: (item[0], item[1], item[2]))[3]

    def _choose_cache_cache_eviction(
        self,
        *,
        candidates: list[PlacementProfile],
        all_gpu_uuids: set[str],
        workers: list[WorkerPlacement],
        real_pressure: dict[str, QueuePressure],
    ) -> list[WorkerPlacement] | None:
        """Evict least-recently-used sleeping processes when the RAM cache is full."""
        options: list[tuple[int, float, tuple[str, ...], list[WorkerPlacement]]] = []
        for profile in candidates:
            for gpu_set in profile.eligible_gpu_sets:
                if not set(gpu_set) <= all_gpu_uuids:
                    continue
                evictable = [
                    worker
                    for worker in workers
                    if worker.state is RuntimeState.SLEEPING
                    and worker.model_id not in real_pressure
                    and (worker.model_id != profile.model_id)
                    and bool(set(worker.gpu_uuids) & set(gpu_set))
                ]
                for count in range(1, len(evictable) + 1):
                    found = False
                    for selected_tuple in combinations(evictable, count):
                        selected = list(selected_tuple)
                        selected_ids = {worker.id for worker in selected}
                        remaining = [worker for worker in workers if worker.id not in selected_ids]
                        if not self._cached_cache_fits(profile, gpu_set, remaining):
                            continue
                        most_recent_demand = max(
                            worker.last_demand_at.timestamp() for worker in selected
                        )
                        options.append(
                            (count, most_recent_demand, tuple(sorted(selected_ids)), selected)
                        )
                        found = True
                    if found:
                        break
        if not options:
            return None
        return min(options, key=lambda item: (item[0], item[1], item[2]))[3]

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
