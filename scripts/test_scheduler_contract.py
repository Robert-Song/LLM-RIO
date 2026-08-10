from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
    StartPlacement,
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
        idle_vram_mib_per_gpu=(1,) * gpu_count,
        peak_vram_mib_per_gpu=(2,) * gpu_count,
        gpu_headroom_mib_per_gpu=(1,) * gpu_count,
        capabilities=frozenset({"chat", "streaming"}),
        launch_args={},
        gpu_memory_utilization=0.9,
        kv_cache_capacity_tokens=4096,
        max_full_length_concurrency=1.0,
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
        action.worker_id
        for action in idle_reclaim_actions
        if isinstance(action, DrainPlacement)
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

    assert long_keep_alive_planner.plan(
        now=now,
        all_gpu_uuids={GPU_0, GPU_1},
        workers=[qwen_worker],
        pressures=[],
        profiles={},
    ) == []

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

    async def service_mode(self) -> ServiceMode:
        return self.mode

    async def set_service_mode(self, mode: ServiceMode) -> None:
        self.mode = mode

    async def release_reservation(self, reservation_id: str, reason: str) -> None:
        self.released.append((reservation_id, reason))

    async def mark_request_admitted(self, request_id: str, worker_id: str) -> None:
        self.admitted.append((request_id, worker_id))

    async def admitted_request_ids(self) -> set[str]:
        return self.active_request_ids


class FakeProfiles:
    async def for_model(self, model_id: str) -> list[PlacementProfile]:
        return []


class FakeSupervisor:
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
