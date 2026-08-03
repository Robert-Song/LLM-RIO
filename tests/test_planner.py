"""Planner tests across 1, 2, 4, and 8 GPU inventories (as promised by the README)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.planner import (
    DrainPlacement,
    GreedyPlacementPlanner,
    QueuePressure,
    StartPlacement,
)

NOW = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)


def make_profile(
    model_id: str,
    gpu_count: int,
    *,
    tokens_per_second: float = 100.0,
    eligible: tuple[tuple[str, ...], ...] | None = None,
) -> PlacementProfile:
    return PlacementProfile(
        id=f"profile-{model_id}-{gpu_count}",
        model_id=model_id,
        model_revision="rev1",
        engine=Engine.VLLM,
        engine_version="0.26.0",
        machine_fingerprint="fingerprint",
        gpu_count=gpu_count,
        tensor_parallel_size=gpu_count,
        pipeline_parallel_size=1,
        eligible_gpu_sets=eligible or (tuple(f"gpu-{i}" for i in range(gpu_count)),),
        dtype="bfloat16",
        quantization=None,
        max_model_len=8192,
        max_num_seqs=32,
        max_num_batched_tokens=None,
        predicted_tokens_per_second=tokens_per_second,
        load_and_warmup_seconds=60.0,
        idle_vram_mib_per_gpu=(2048,) * gpu_count,
        peak_vram_mib_per_gpu=(8192,) * gpu_count,
        gpu_headroom_mib_per_gpu=(2048,) * gpu_count,
        capabilities=frozenset({"chat", "streaming"}),
        launch_args={},
        gpu_memory_utilization=0.9,
        kv_cache_capacity_tokens=100_000,
        max_full_length_concurrency=8.0,
    )


def make_worker(
    model_id: str,
    gpu_uuids: tuple[str, ...],
    *,
    state: RuntimeState = RuntimeState.READY,
    ready_at: datetime | None = None,
    last_demand_at: datetime | None = None,
) -> WorkerPlacement:
    return WorkerPlacement(
        id=f"worker-{model_id}-{gpu_uuids[0]}",
        profile=make_profile(model_id, len(gpu_uuids), eligible=(gpu_uuids,)),
        gpu_uuids=gpu_uuids,
        port=18000,
        state=state,
        ready_at=ready_at,
        last_demand_at=last_demand_at or NOW,
    )


def make_pressure(
    model_id: str, requests: int, tokens: int, oldest: datetime | None = None
) -> QueuePressure:
    return QueuePressure(
        model_id=model_id,
        requests=requests,
        estimated_tokens=tokens,
        oldest_enqueued_at=oldest or NOW,
    )


def planner(**overrides) -> GreedyPlacementPlanner:
    kwargs = {
        "wait_duration_seconds": 900.0,
        "minimum_residency_seconds": 300.0,
        "fair_share_seconds": 7200.0,
    }
    kwargs.update(overrides)
    return GreedyPlacementPlanner(**kwargs)


def inventory(gpu_count: int) -> set[str]:
    return {f"gpu-{i}" for i in range(gpu_count)}


@pytest.mark.parametrize("gpu_count", [1, 2, 4, 8])
class TestColdStart:
    def test_cold_start_places_smallest_fitting(self, gpu_count: int) -> None:
        plan = planner()
        profiles = {
            "model-a": [make_profile("model-a", 1), make_profile("model-a", 2)],
        }
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(gpu_count),
            workers=[],
            pressures=[make_pressure("model-a", 5, 10_000)],
            profiles=profiles,
        )
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, StartPlacement)
        assert action.profile.gpu_count == 1
        assert action.reason == "cold_backlog"
        assert set(action.gpu_uuids) <= inventory(gpu_count)

    def test_no_action_without_pressure(self, gpu_count: int) -> None:
        actions = planner().plan(
            now=NOW,
            all_gpu_uuids=inventory(gpu_count),
            workers=[],
            pressures=[],
            profiles={"model-a": [make_profile("model-a", 1)]},
        )
        assert actions == []

    def test_no_action_without_profiles(self, gpu_count: int) -> None:
        actions = planner().plan(
            now=NOW,
            all_gpu_uuids=inventory(gpu_count),
            workers=[],
            pressures=[make_pressure("model-a", 5, 10_000)],
            profiles={},
        )
        assert actions == []

    def test_no_action_when_gpus_are_occupied(self, gpu_count: int) -> None:
        plan = planner()
        # Recently READY workers are protected by minimum residency, so a cold
        # backlog cannot preempt them.
        worker = make_worker(
            "model-other",
            tuple(inventory(gpu_count)),
            ready_at=NOW - timedelta(seconds=30),
        )
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(gpu_count),
            workers=[worker],
            pressures=[make_pressure("model-a", 5, 10_000)],
            profiles={"model-a": [make_profile("model-a", 1)]},
        )
        assert actions == []


class TestIdleDrain:
    def test_idle_ready_worker_drains_after_wait(self) -> None:
        plan = planner(wait_duration_seconds=60.0)
        worker = make_worker(
            "model-a",
            ("gpu-0",),
            last_demand_at=NOW - timedelta(seconds=120),
        )
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[],
            profiles={},
        )
        assert len(actions) == 1
        assert isinstance(actions[0], DrainPlacement)
        assert actions[0].reason == "idle_timeout"

    def test_recently_used_worker_is_not_drained(self) -> None:
        plan = planner(wait_duration_seconds=60.0)
        worker = make_worker("model-a", ("gpu-0",), last_demand_at=NOW)
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[],
            profiles={},
        )
        assert actions == []

    def test_worker_with_admitted_requests_is_not_drained(self) -> None:
        plan = planner(wait_duration_seconds=60.0)
        worker = make_worker(
            "model-a",
            ("gpu-0",),
            last_demand_at=NOW - timedelta(seconds=120),
        )
        worker.admitted_request_ids.add("req-1")
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[],
            profiles={},
        )
        assert actions == []

    def test_worker_with_pressure_is_not_drained(self) -> None:
        plan = planner(wait_duration_seconds=60.0)
        worker = make_worker(
            "model-a",
            ("gpu-0",),
            last_demand_at=NOW - timedelta(seconds=120),
        )
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[make_pressure("model-a", 2, 1000)],
            profiles={"model-a": [make_profile("model-a", 1)]},
        )
        assert actions == []


class TestReplicaScaling:
    def test_scales_out_when_backlog_exceeds_capacity(self) -> None:
        plan = planner(scale_window_seconds=30.0)
        worker = make_worker("model-a", ("gpu-0",))
        # The candidate replica must be eligible on the free GPU (gpu-1).
        profiles = {
            "model-a": [
                make_profile("model-a", 1, tokens_per_second=100.0, eligible=(("gpu-1",),))
            ]
        }
        # One worker with 100 t/s can absorb 3000 tokens in the 30s window.
        pressure = make_pressure("model-a", 50, 100_000)
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[pressure],
            profiles=profiles,
        )
        assert len(actions) == 1
        action = actions[0]
        assert isinstance(action, StartPlacement)
        assert action.reason == "replica_backlog"
        assert action.gpu_uuids == ("gpu-1",)

    def test_no_scale_out_within_capacity(self) -> None:
        plan = planner(scale_window_seconds=30.0)
        worker = make_worker("model-a", ("gpu-0",))
        profiles = {"model-a": [make_profile("model-a", 1, tokens_per_second=100.0)]}
        pressure = make_pressure("model-a", 2, 1000)
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[pressure],
            profiles=profiles,
        )
        assert actions == []

    def test_no_scale_out_when_marginal_efficiency_too_low(self) -> None:
        plan = planner(scale_window_seconds=30.0, minimum_marginal_efficiency=0.5)
        worker = make_worker("model-a", ("gpu-0",), state=RuntimeState.READY)
        worker.profile = make_profile(
            "model-a", 1, tokens_per_second=100.0, eligible=(("gpu-0",),)
        )
        # A second replica at 10 t/s is a 10% margin — below the 50% threshold.
        profiles = {"model-a": [make_profile("model-a", 1, tokens_per_second=10.0)]}
        pressure = make_pressure("model-a", 50, 100_000)
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[pressure],
            profiles=profiles,
        )
        assert actions == []

    def test_no_scale_out_when_no_free_gpu(self) -> None:
        plan = planner(scale_window_seconds=30.0)
        worker = make_worker("model-a", ("gpu-0", "gpu-1"))
        profiles = {"model-a": [make_profile("model-a", 1, tokens_per_second=10.0)]}
        pressure = make_pressure("model-a", 50, 100_000)
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[worker],
            pressures=[pressure],
            profiles=profiles,
        )
        assert actions == []


class TestPreemption:
    def test_cold_backlog_preempts_resident_worker_when_starved(self) -> None:
        plan = planner(fair_share_seconds=30.0)
        blocker = make_worker(
            "model-other",
            ("gpu-0",),
            ready_at=NOW - timedelta(minutes=10),
        )
        profiles = {"model-a": [make_profile("model-a", 1)]}
        pressure = make_pressure(
            "model-a", 5, 10_000, oldest=NOW - timedelta(minutes=10)
        )
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[blocker],
            pressures=[pressure],
            profiles=profiles,
        )
        assert len(actions) == 1
        assert isinstance(actions[0], DrainPlacement)
        assert actions[0].reason == "incompatible_backlog"

    def test_recently_started_worker_is_protected(self) -> None:
        plan = planner(
            fair_share_seconds=30.0,
            minimum_residency_seconds=300.0,
        )
        blocker = make_worker("model-other", ("gpu-0",), ready_at=NOW)
        profiles = {"model-a": [make_profile("model-a", 1)]}
        pressure = make_pressure(
            "model-a", 5, 10_000, oldest=NOW - timedelta(minutes=10)
        )
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[blocker],
            pressures=[pressure],
            profiles=profiles,
        )
        assert actions == []

    def test_no_preemption_when_incoming_is_not_faster(self) -> None:
        plan = planner()
        blocker = make_worker(
            "model-other",
            ("gpu-0",),
            ready_at=NOW - timedelta(minutes=10),
        )
        blocker.profile = make_profile(
            "model-other", 1, tokens_per_second=500.0, eligible=(("gpu-0",),)
        )
        profiles = {"model-a": [make_profile("model-a", 1, tokens_per_second=100.0)]}
        # Both models are under pressure: the resident worker is only replaced on a
        # clear score win, and 100 t/s incoming is not faster than 500 t/s outgoing.
        pressure = make_pressure(
            "model-a", 5, 10_000, oldest=NOW - timedelta(minutes=10)
        )
        other_pressure = make_pressure(
            "model-other", 2, 5_000, oldest=NOW - timedelta(minutes=5)
        )
        actions = plan.plan(
            now=NOW,
            all_gpu_uuids=inventory(2),
            workers=[blocker],
            pressures=[pressure, other_pressure],
            profiles=profiles,
        )
        assert actions == []
