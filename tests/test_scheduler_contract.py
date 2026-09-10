from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from llm_rio.config import Settings
from llm_rio.domain import (
    Engine,
    MachineInventory,
    PlacementProfile,
    RuntimeState,
    ServiceMode,
    WorkerPlacement,
)
from llm_rio.errors import MaintenanceError
from llm_rio.planner import (
    DrainPlacement,
    GreedyPlacementPlanner,
    QueuePressure,
    SleepPlacement,
    StartPlacement,
    WakePlacement,
)
from llm_rio.queueing import QueuedRequest
from llm_rio.runtime import ResidencyScheduler, WorkerLease

GPU_0 = "GPU-0"
GPU_1 = "GPU-1"


def make_profile(
    profile_id: str,
    model_id: str,
    gpu_set: tuple[str, ...],
    *,
    tokens_per_second: float = 10.0,
    idle_vram_mib: int = 1,
    memory_backend: str = "native",
) -> PlacementProfile:
    gpu_count = len(gpu_set)
    return PlacementProfile(
        id=profile_id,
        model_id=model_id,
        model_revision="immutable-revision",
        engine=Engine.VLLM,
        engine_version="test",
        machine_fingerprint="machine",
        gpu_count=gpu_count,
        tensor_parallel_size=gpu_count,
        pipeline_parallel_size=1,
        eligible_gpu_sets=(gpu_set,),
        dtype="auto",
        quantization=None,
        max_model_len=4096,
        max_num_seqs=128,
        max_num_batched_tokens=None,
        predicted_tokens_per_second=tokens_per_second,
        load_and_warmup_seconds=1.0,
        idle_vram_mib_per_gpu=(idle_vram_mib,) * gpu_count,
        peak_vram_mib_per_gpu=(idle_vram_mib,) * gpu_count,
        gpu_headroom_mib_per_gpu=(0,) * gpu_count,
        capabilities=frozenset({"chat", "streaming"}),
        launch_args={},
        gpu_memory_utilization=0.9,
        kv_cache_capacity_tokens=4096,
        max_full_length_concurrency=1.0,
        memory_backend=memory_backend,
        kvcached_verified=memory_backend == "kvcached",
        vram_measurement_version=2,
        vram_baseline_mib_per_gpu=(0,) * gpu_count,
        wake_peak_vram_mib_per_gpu=(idle_vram_mib,) * gpu_count,
        sleep_vram_mib_per_gpu=(1,) * gpu_count,
    )


def make_worker(worker_id: str, profile: PlacementProfile) -> WorkerPlacement:
    now = datetime.now(UTC)
    return WorkerPlacement(
        id=worker_id,
        profile=profile,
        gpu_uuids=profile.eligible_gpu_sets[0],
        port=18000,
        state=RuntimeState.READY,
        ready_at=now,
        last_demand_at=now,
    )


def pressure(model_id: str, age_seconds: float, tokens: int = 1000) -> QueuePressure:
    return QueuePressure(
        model_id=model_id,
        requests=32,
        estimated_tokens=tokens,
        oldest_enqueued_at=datetime.now(UTC) - timedelta(seconds=age_seconds),
    )


def planner() -> GreedyPlacementPlanner:
    return GreedyPlacementPlanner(
        wait_duration_seconds=5,
        minimum_residency_seconds=0,
        fair_share_seconds=7200,
    )


def prism_planner(*, max_workers: int = 2) -> GreedyPlacementPlanner:
    return GreedyPlacementPlanner(
        wait_duration_seconds=5,
        minimum_residency_seconds=0,
        fair_share_seconds=7200,
        prism_enabled=True,
        kvcached_required=True,
        gpu_vram_mib={GPU_0: 100, GPU_1: 100},
        reserved_vram_mib=10,
        prism_max_workers_per_gpu=max_workers,
        prism_sleep_gpu_reserve_mib=1,
    )


def test_prism_starts_a_second_model_on_an_occupied_gpu() -> None:
    qwen = make_profile("qwen", "qwen", (GPU_0,), idle_vram_mib=30, memory_backend="kvcached")
    gemma = make_profile("gemma", "gemma", (GPU_0,), idle_vram_mib=40, memory_backend="kvcached")

    actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[make_worker("qwen-worker", qwen)],
        pressures=[pressure("gemma", 0)],
        profiles={"gemma": [gemma]},
    )

    assert actions == [StartPlacement(gemma, (GPU_0,), "prism_cold_backlog")]


def test_prism_proactively_caches_idle_weights_in_ram() -> None:
    qwen = make_profile("qwen", "qwen", (GPU_0,), idle_vram_mib=30, memory_backend="kvcached")
    worker = make_worker("qwen-worker", qwen)
    worker.last_demand_at = datetime.now(UTC) - timedelta(hours=1)

    actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[worker],
        pressures=[],
        profiles={},
    )

    assert actions == [SleepPlacement(worker.id, "prism_idle_weight_cache")]


def test_prism_evicts_an_idle_resident_only_for_real_demand() -> None:
    qwen = make_profile("qwen", "qwen", (GPU_0,), idle_vram_mib=60, memory_backend="kvcached")
    gemma = make_profile("gemma", "gemma", (GPU_0,), idle_vram_mib=60, memory_backend="kvcached")
    worker = make_worker("qwen-worker", qwen)

    demand_actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[worker],
        pressures=[pressure("gemma", 0)],
        profiles={"gemma": [gemma]},
    )
    preload_actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[worker],
        pressures=[
            QueuePressure(
                model_id="gemma",
                requests=0,
                estimated_tokens=1,
                oldest_enqueued_at=datetime.now(UTC),
                preload=True,
            )
        ],
        profiles={"gemma": [gemma]},
    )

    assert demand_actions == [SleepPlacement(worker.id, "prism_weight_capacity")]
    assert preload_actions == [SleepPlacement(worker.id, "prism_weight_capacity")]


def test_prism_wakes_a_cached_model_instead_of_cold_starting() -> None:
    qwen = make_profile("qwen", "qwen", (GPU_0,), idle_vram_mib=30, memory_backend="kvcached")
    worker = make_worker("qwen-worker", qwen)
    worker.state = RuntimeState.SLEEPING

    actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[worker],
        pressures=[pressure("qwen", 0)],
        profiles={"qwen": [qwen]},
    )

    assert actions == [WakePlacement(worker.id, "prism_ram_cache_hit")]


def test_prism_sleeping_count_does_not_force_lru_eviction() -> None:
    old = make_profile("old", "old", (GPU_0,), idle_vram_mib=20, memory_backend="kvcached")
    recent = make_profile("recent", "recent", (GPU_0,), idle_vram_mib=20, memory_backend="kvcached")
    incoming = make_profile(
        "incoming", "incoming", (GPU_0,), idle_vram_mib=20, memory_backend="kvcached"
    )
    old_worker = make_worker("old-worker", old)
    old_worker.state = RuntimeState.SLEEPING
    old_worker.last_demand_at = datetime.now(UTC) - timedelta(hours=2)
    recent_worker = make_worker("recent-worker", recent)
    recent_worker.state = RuntimeState.SLEEPING
    recent_worker.last_demand_at = datetime.now(UTC) - timedelta(hours=1)

    actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[recent_worker, old_worker],
        pressures=[pressure("incoming", 0)],
        profiles={"incoming": [incoming]},
    )

    assert actions == [StartPlacement(incoming, (GPU_0,), "prism_cold_backlog")]


def test_prism_preload_uses_memory_budget_not_sleeping_count() -> None:
    cached = make_profile("cached", "cached", (GPU_0,), idle_vram_mib=20, memory_backend="kvcached")
    incoming = make_profile(
        "incoming", "incoming", (GPU_0,), idle_vram_mib=20, memory_backend="kvcached"
    )
    cached_worker = make_worker("cached-worker", cached)
    cached_worker.state = RuntimeState.SLEEPING
    preload = QueuePressure(
        model_id="incoming",
        requests=0,
        estimated_tokens=1,
        oldest_enqueued_at=datetime.now(UTC),
        preload=True,
    )

    actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[cached_worker],
        pressures=[preload],
        profiles={"incoming": [incoming]},
    )

    assert actions == [StartPlacement(incoming, (GPU_0,), "prism_preload")]


def test_prism_never_offloads_a_worker_with_an_active_request() -> None:
    qwen = make_profile("qwen", "qwen", (GPU_0,), idle_vram_mib=60, memory_backend="kvcached")
    gemma = make_profile("gemma", "gemma", (GPU_0,), idle_vram_mib=60, memory_backend="kvcached")
    worker = make_worker("qwen-worker", qwen)
    worker.admitted_request_ids.add("active")

    actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[worker],
        pressures=[pressure("gemma", 0)],
        profiles={"gemma": [gemma]},
    )

    assert actions == []


def test_prism_rejects_native_profiles_and_supports_tp_colocation() -> None:
    native = make_profile("native", "native", (GPU_0,), idle_vram_mib=10)
    qwen_tp = make_profile(
        "qwen-tp",
        "qwen",
        (GPU_0, GPU_1),
        idle_vram_mib=30,
        memory_backend="kvcached",
    )
    gemma_tp = make_profile(
        "gemma-tp",
        "gemma",
        (GPU_0, GPU_1),
        idle_vram_mib=40,
        memory_backend="kvcached",
    )

    native_actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0},
        workers=[],
        pressures=[pressure("native", 0)],
        profiles={"native": [native]},
    )
    tp_actions = prism_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=[make_worker("qwen-worker", qwen_tp)],
        pressures=[pressure("gemma", 0)],
        profiles={"gemma": [gemma_tp]},
    )

    assert native_actions == []
    assert tp_actions == [StartPlacement(gemma_tp, (GPU_0, GPU_1), "prism_cold_backlog")]


def test_cold_single_gpu_model_fills_both_validated_gpu_slots() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_1 = make_profile("qwen-1", "qwen", (GPU_1,))
    qwen_tp2 = make_profile("qwen-tp2", "qwen", (GPU_0, GPU_1), tokens_per_second=20)

    actions = planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=[],
        pressures=[pressure("qwen", 1)],
        profiles={"qwen": [qwen_0, qwen_1, qwen_tp2]},
    )

    starts = [action for action in actions if isinstance(action, StartPlacement)]
    assert len(starts) == 2
    assert {action.gpu_uuids for action in starts} == {(GPU_0,), (GPU_1,)}
    assert all(action.profile.tensor_parallel_size == 1 for action in starts)


def test_dual_model_demand_downscales_only_one_redundant_replica() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_1 = make_profile("qwen-1", "qwen", (GPU_1,))
    gemma_1 = make_profile("gemma-1", "gemma", (GPU_1,))
    workers = [make_worker("qwen-worker-0", qwen_0), make_worker("qwen-worker-1", qwen_1)]

    actions = planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=workers,
        pressures=[pressure("qwen", 2), pressure("gemma", 0)],
        profiles={"qwen": [qwen_0, qwen_1], "gemma": [gemma_1]},
    )

    assert actions == [DrainPlacement("qwen-worker-1", "incompatible_backlog")]


def test_configured_minimum_residency_delays_replacement() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_1 = make_profile("qwen-1", "qwen", (GPU_1,))
    gemma_1 = make_profile("gemma-1", "gemma", (GPU_1,))
    workers = [
        make_worker("qwen-worker-0", qwen_0),
        make_worker("qwen-worker-1", qwen_1),
    ]
    now = datetime.now(UTC)
    for worker in workers:
        worker.ready_at = now
    restrictive_planner = GreedyPlacementPlanner(
        wait_duration_seconds=5,
        minimum_residency_seconds=300,
        fair_share_seconds=7200,
    )
    pressures = [pressure("qwen", 2), pressure("gemma", 0)]
    profiles = {"qwen": [qwen_0, qwen_1], "gemma": [gemma_1]}

    assert (
        restrictive_planner.plan(
            now=now,
            all_gpu_uuids={GPU_0, GPU_1},
            workers=workers,
            pressures=pressures,
            profiles=profiles,
        )
        == []
    )
    assert restrictive_planner.plan(
        now=now + timedelta(seconds=301),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=workers,
        pressures=pressures,
        profiles=profiles,
    ) == [DrainPlacement("qwen-worker-1", "incompatible_backlog")]


def test_multi_gpu_request_never_causes_useless_partial_preemption() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_1 = make_profile("qwen-1", "qwen", (GPU_1,))
    laguna = make_profile("laguna-tp2", "laguna", (GPU_0, GPU_1))
    workers = [make_worker("qwen-worker-0", qwen_0), make_worker("qwen-worker-1", qwen_1)]
    now = datetime.now(UTC)

    protected_actions = planner().plan(
        now=now,
        all_gpu_uuids={GPU_0, GPU_1},
        workers=workers,
        pressures=[pressure("qwen", 2), pressure("laguna", 0)],
        profiles={"qwen": [qwen_0, qwen_1], "laguna": [laguna]},
    )
    assert protected_actions == []

    idle_reclaim_actions = planner().plan(
        now=now,
        all_gpu_uuids={GPU_0, GPU_1},
        workers=workers,
        pressures=[pressure("laguna", 0)],
        profiles={"laguna": [laguna]},
    )
    assert {
        action.worker_id for action in idle_reclaim_actions if isinstance(action, DrainPlacement)
    } == {"qwen-worker-0", "qwen-worker-1"}


def test_sustained_backlog_never_preempts_its_own_ready_replicas() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_1 = make_profile("qwen-1", "qwen", (GPU_1,))
    workers = [
        make_worker("qwen-worker-0", qwen_0),
        make_worker("qwen-worker-1", qwen_1),
    ]

    actions = planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=workers,
        pressures=[pressure("qwen", 0, tokens=100_000)],
        profiles={"qwen": [qwen_0, qwen_1]},
    )

    assert actions == []


def test_wait_duration_does_not_delay_reclaim_for_a_new_model() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_1 = make_profile("qwen-1", "qwen", (GPU_1,))
    gemma_0 = make_profile("gemma-0", "gemma", (GPU_0,))
    gemma_1 = make_profile("gemma-1", "gemma", (GPU_1,))
    qwen_workers = [
        make_worker("qwen-worker-0", qwen_0),
        make_worker("qwen-worker-1", qwen_1),
    ]
    long_keep_alive_planner = GreedyPlacementPlanner(
        wait_duration_seconds=3600,
        minimum_residency_seconds=0,
        fair_share_seconds=7200,
    )

    actions = long_keep_alive_planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=qwen_workers,
        pressures=[pressure("gemma", 0)],
        profiles={"gemma": [gemma_0, gemma_1]},
    )

    assert {action.worker_id for action in actions if isinstance(action, DrainPlacement)} == {
        "qwen-worker-0",
        "qwen-worker-1",
    }


def test_wait_duration_does_not_delay_idle_reclaim_for_replication() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    gemma_0 = make_profile("gemma-0", "gemma", (GPU_0,))
    gemma_1 = make_profile("gemma-1", "gemma", (GPU_1,))
    qwen_worker = make_worker("qwen-worker-0", qwen_0)
    gemma_worker = make_worker("gemma-worker-1", gemma_1)
    long_keep_alive_planner = GreedyPlacementPlanner(
        wait_duration_seconds=3600,
        minimum_residency_seconds=0,
        fair_share_seconds=7200,
    )

    actions = long_keep_alive_planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=[qwen_worker, gemma_worker],
        pressures=[pressure("gemma", 0, tokens=10_000)],
        profiles={"gemma": [gemma_0, gemma_1]},
    )

    assert actions == [DrainPlacement("qwen-worker-0", "replica_capacity")]


def test_wait_duration_only_controls_automatic_idle_unload() -> None:
    qwen_0 = make_profile("qwen-0", "qwen", (GPU_0,))
    qwen_worker = make_worker("qwen-worker-0", qwen_0)
    now = datetime.now(UTC)
    long_keep_alive_planner = GreedyPlacementPlanner(
        wait_duration_seconds=3600,
        minimum_residency_seconds=0,
        fair_share_seconds=7200,
    )

    assert (
        long_keep_alive_planner.plan(
            now=now,
            all_gpu_uuids={GPU_0, GPU_1},
            workers=[qwen_worker],
            pressures=[],
            profiles={},
        )
        == []
    )

    assert long_keep_alive_planner.plan(
        now=now + timedelta(seconds=3601),
        all_gpu_uuids={GPU_0, GPU_1},
        workers=[qwen_worker],
        pressures=[],
        profiles={},
    ) == [DrainPlacement("qwen-worker-0", "idle_timeout")]


class FakeDatabase:
    def __init__(self) -> None:
        self.mode = ServiceMode.ACTIVE
        self.released: list[tuple[str, str]] = []
        self.admitted: list[tuple[str, str]] = []
        self.active_request_ids: set[str] = set()
        self.admission_result = True
        self.events: list[tuple[str, str | None, dict[str, Any]]] = []

    async def service_mode(self) -> ServiceMode:
        return self.mode

    async def set_service_mode(self, mode: ServiceMode) -> None:
        self.mode = mode

    async def release_reservation(self, reservation_id: str, reason: str) -> None:
        self.released.append((reservation_id, reason))

    async def mark_request_admitted(self, request_id: str, worker_id: str) -> bool:
        self.admitted.append((request_id, worker_id))
        if self.admission_result:
            self.active_request_ids.add(request_id)
        return self.admission_result

    async def admitted_request_ids(self) -> set[str]:
        return self.active_request_ids

    async def record_event(
        self, event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> None:
        self.events.append((event_type, entity_id, payload or {}))


class FakeProfiles:
    async def for_model(self, model_id: str) -> list[PlacementProfile]:
        return []


class FakeSupervisor:
    ram_weight_cache_enabled = False

    def __init__(self) -> None:
        self.internal_api_key = "test-internal-key"
        self.workers: dict[str, WorkerPlacement] = {}
        self.callback: Any = None
        self.admitted: list[tuple[str, str, int]] = []
        self.released: list[tuple[str, str, int]] = []

    def set_event_callback(self, callback: Any) -> None:
        self.callback = callback

    @property
    def occupied_gpu_uuids(self) -> set[str]:
        return set()

    async def drain(self, worker_id: str) -> None:
        raise AssertionError("no workers should exist in this test")

    async def stop_all(self, *, force: bool = False) -> None:
        return None

    async def admit(self, worker_id: str, request_id: str, tokens: int) -> None:
        worker = self.workers[worker_id]
        worker.admitted_request_ids.add(request_id)
        worker.outstanding_token_work += tokens
        self.admitted.append((worker_id, request_id, tokens))

    async def release(self, worker_id: str, request_id: str, tokens: int) -> None:
        worker = self.workers[worker_id]
        worker.admitted_request_ids.discard(request_id)
        worker.outstanding_token_work = max(0, worker.outstanding_token_work - tokens)
        self.released.append((worker_id, request_id, tokens))


@pytest.mark.asyncio
async def test_newly_validated_model_requests_one_time_ram_warm() -> None:
    database = FakeDatabase()
    supervisor = FakeSupervisor()
    supervisor.ram_weight_cache_enabled = True
    scheduler = ResidencyScheduler(
        settings=Settings(),
        database=database,  # type: ignore[arg-type]
        inventory=MachineInventory(
            machine_id="test",
            driver_version="test",
            cuda_driver_version=None,
            gpus=(),
            topology_hash="test",
            fingerprint="test",
        ),
        profiles=FakeProfiles(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
    )

    scheduler.kvcached = SimpleNamespace(enabled=True)
    await scheduler.warm_model_once("new-model")

    assert scheduler._prism_one_time_warm_model_ids == {"new-model"}
    assert database.events[-1] == (
        "PRISM_MODEL_WARM_REQUESTED",
        "new-model",
        {"target": "host_ram"},
    )


@pytest.mark.asyncio
async def test_settled_request_is_released_from_scheduler_memory() -> None:
    database = FakeDatabase()
    supervisor = FakeSupervisor()
    worker = make_worker("gemma-worker", make_profile("gemma-profile", "gemma", (GPU_0,)))
    worker.admitted_request_ids.add("completed-request")
    worker.outstanding_token_work = 128
    supervisor.workers[worker.id] = worker
    scheduler = ResidencyScheduler(
        settings=Settings(wait_duration_seconds=600, minimum_residency_seconds=10),
        database=database,  # type: ignore[arg-type]
        inventory=MachineInventory(
            machine_id="test",
            driver_version="test",
            cuda_driver_version=None,
            gpus=(),
            topology_hash="test",
            fingerprint="test",
        ),
        profiles=FakeProfiles(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    lease = WorkerLease(
        worker_id=worker.id,
        request_id="completed-request",
        reservation_id="reservation",
        base_url="http://127.0.0.1:18000",
        internal_api_key="internal-key",
        estimated_tokens=128,
        admitted_at=datetime.now(UTC),
    )
    scheduler._request_leases[lease.request_id] = lease

    await scheduler._release_settled_leases()

    assert worker.admitted_request_ids == set()
    assert worker.outstanding_token_work == 0
    assert supervisor.released == [(worker.id, lease.request_id, 128)]
    assert scheduler._request_leases == {}


@pytest.mark.asyncio
async def test_maintenance_wins_atomic_race_against_new_admission() -> None:
    database = FakeDatabase()
    supervisor = FakeSupervisor()
    scheduler = ResidencyScheduler(
        settings=Settings(
            queue_capacity_per_model=8,
            queue_capacity_per_tenant=8,
            wait_duration_seconds=5,
            minimum_residency_seconds=1,
            fair_share_seconds=60,
        ),
        database=database,  # type: ignore[arg-type]
        inventory=MachineInventory(
            machine_id="test",
            driver_version="test",
            cuda_driver_version=None,
            gpus=(),
            topology_hash="test",
            fingerprint="test",
        ),
        profiles=FakeProfiles(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    request = QueuedRequest(
        id="request",
        model_id="qwen",
        tenant_id="tenant",
        estimated_tokens=16,
        payload={},
        reservation_id="reservation",
    )

    await scheduler._state_lock.acquire()
    maintenance_task = asyncio.create_task(scheduler.enter_maintenance())
    await asyncio.sleep(0)
    enqueue_task = asyncio.create_task(scheduler.enqueue(request))
    await asyncio.sleep(0)
    scheduler._state_lock.release()

    await maintenance_task
    with pytest.raises(MaintenanceError):
        await asyncio.wait_for(enqueue_task, timeout=0.5)
    assert database.mode is ServiceMode.DRAINING
    assert scheduler.queues.pending_models() == []


@pytest.mark.asyncio
async def test_ready_model_is_not_gated_by_another_models_loading_worker() -> None:
    database = FakeDatabase()
    supervisor = FakeSupervisor()
    scheduler = ResidencyScheduler(
        settings=Settings(
            queue_capacity_per_model=8,
            queue_capacity_per_tenant=8,
            wait_duration_seconds=5,
            minimum_residency_seconds=1,
            fair_share_seconds=60,
        ),
        database=database,  # type: ignore[arg-type]
        inventory=MachineInventory(
            machine_id="test",
            driver_version="test",
            cuda_driver_version=None,
            gpus=(),
            topology_hash="test",
            fingerprint="test",
        ),
        profiles=FakeProfiles(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    qwen = make_worker("qwen-worker", make_profile("qwen-profile", "qwen", (GPU_0,)))
    gemma = make_worker(
        "gemma-worker",
        make_profile("gemma-profile", "gemma", (GPU_1,)),
    )
    gemma.state = RuntimeState.LOADING
    gemma.ready_at = None
    supervisor.workers = {qwen.id: qwen, gemma.id: gemma}
    for request in (
        QueuedRequest(
            id="qwen-request",
            model_id="qwen",
            tenant_id="team-a",
            estimated_tokens=16,
            payload={},
            reservation_id="qwen-reservation",
        ),
        QueuedRequest(
            id="gemma-request",
            model_id="gemma",
            tenant_id="team-b",
            estimated_tokens=16,
            payload={},
            reservation_id="gemma-reservation",
        ),
    ):
        scheduler.queues.for_model(request.model_id).put(request)

    await scheduler._route_ready_work()

    assert database.admitted == [("qwen-request", "qwen-worker")]
    assert len(scheduler.queues.for_model("qwen")) == 0
    assert len(scheduler.queues.for_model("gemma")) == 1

    gemma.state = RuntimeState.READY
    gemma.ready_at = datetime.now(UTC)
    await scheduler._route_ready_work()

    assert set(database.admitted) == {
        ("qwen-request", "qwen-worker"),
        ("gemma-request", "gemma-worker"),
    }
    assert scheduler.queues.pending_models() == []


@pytest.mark.asyncio
async def test_stream_idle_watchdog_clears_stale_worker_admission() -> None:
    database = FakeDatabase()
    supervisor = FakeSupervisor()
    worker = make_worker("gemma-worker", make_profile("gemma-profile", "gemma", (GPU_0,)))
    worker.admitted_request_ids.add("stalled-request")
    worker.outstanding_token_work = 64
    supervisor.workers[worker.id] = worker
    scheduler = ResidencyScheduler(
        settings=Settings(
            wait_duration_seconds=600,
            minimum_residency_seconds=10,
            worker_stream_idle_timeout_seconds=1,
        ),
        database=database,  # type: ignore[arg-type]
        inventory=MachineInventory(
            machine_id="test",
            driver_version="test",
            cuda_driver_version=None,
            gpus=(),
            topology_hash="test",
            fingerprint="test",
        ),
        profiles=FakeProfiles(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    lease = WorkerLease(
        worker_id=worker.id,
        request_id="stalled-request",
        reservation_id="stalled-reservation",
        base_url="http://127.0.0.1:18000",
        internal_api_key="internal-key",
        estimated_tokens=64,
        admitted_at=datetime.now(UTC) - timedelta(seconds=2),
        is_stream=True,
        last_activity_at=datetime.now(UTC) - timedelta(seconds=2),
    )
    database.active_request_ids.add(lease.request_id)
    scheduler._request_leases[lease.request_id] = lease

    await scheduler._expire_inactive_streams()

    assert database.released == [(lease.reservation_id, "worker_stream_idle_timeout")]
    assert worker.admitted_request_ids == set()
    assert worker.outstanding_token_work == 0
    assert supervisor.released == [(worker.id, lease.request_id, lease.estimated_tokens)]
    assert scheduler._request_leases == {}
    assert database.events == [
        (
            "WORKER_STREAM_IDLE_TIMEOUT",
            worker.id,
            {"request_id": lease.request_id, "timeout_seconds": 1},
        )
    ]


@pytest.mark.asyncio
async def test_failed_admission_rolls_back_worker_state_and_reservation() -> None:
    database = FakeDatabase()
    database.admission_result = False
    supervisor = FakeSupervisor()
    worker = make_worker("gemma-worker", make_profile("gemma-profile", "gemma", (GPU_0,)))
    supervisor.workers[worker.id] = worker
    scheduler = ResidencyScheduler(
        settings=Settings(wait_duration_seconds=600, minimum_residency_seconds=10),
        database=database,  # type: ignore[arg-type]
        inventory=MachineInventory(
            machine_id="test",
            driver_version="test",
            cuda_driver_version=None,
            gpus=(),
            topology_hash="test",
            fingerprint="test",
        ),
        profiles=FakeProfiles(),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
    )
    queued = QueuedRequest(
        id="request",
        model_id="gemma",
        tenant_id="tenant",
        estimated_tokens=64,
        payload={},
        reservation_id="reservation",
    )
    queued.assignment = asyncio.get_running_loop().create_future()
    scheduler.queues.for_model(queued.model_id).put(queued)

    await scheduler._route_ready_work()

    with pytest.raises(RuntimeError, match="no longer queued"):
        queued.assignment.result()
    assert worker.admitted_request_ids == set()
    assert worker.outstanding_token_work == 0
    assert supervisor.released == [(worker.id, queued.id, queued.estimated_tokens)]
    assert database.released == [(queued.reservation_id, "admission_failed")]
    assert scheduler._request_leases == {}
