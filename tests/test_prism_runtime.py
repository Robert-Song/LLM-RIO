from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.kvcached_vllm_compat import (
    _install_allocate_slots_rollback_shim,
    _install_hybrid_sleep_wake_shim,
    _install_persistent_weight_backup_shim,
)
from llm_rio.planner import (
    GreedyPlacementPlanner,
    QueuePressure,
    StartPlacement,
    WakePlacement,
)


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)

    def warning(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)


def _install_fake_allocate_patch() -> tuple[type[Any], _Logger]:
    class FakeKVCacheManager:
        def allocate_slots(self, request: Any) -> None:
            first, _second = self.coordinator.single_type_managers
            first.req_to_blocks.setdefault(request.request_id, []).append(
                SimpleNamespace(block_id=7)
            )
            first.new_block_ids.append(7)
            return None

    class FakePatch:
        def patch_allocate_slots(self, manager_module: Any) -> bool:
            # Stand in for kvcached's exception-to-None wrapper.
            original = manager_module.KVCacheManager.allocate_slots

            def patched(manager: Any, *args: Any, **kwargs: Any) -> Any:
                return original(manager, *args, **kwargs)

            manager_module.KVCacheManager.allocate_slots = patched
            return True

    fake_kvp = SimpleNamespace(KVCacheManagerAllocateSlotsPatch=FakePatch)
    logger = _Logger()
    _install_allocate_slots_rollback_shim(fake_kvp, logger)
    FakePatch().patch_allocate_slots(SimpleNamespace(KVCacheManager=FakeKVCacheManager))
    return FakeKVCacheManager, logger


def test_waiting_partial_shared_pool_allocation_is_rolled_back() -> None:
    manager_class, logger = _install_fake_allocate_patch()
    groups = [
        SimpleNamespace(
            req_to_blocks={},
            new_block_ids=[],
            _pending_cow_copies=[],
        )
        for _ in range(2)
    ]

    class Manager(manager_class):
        coordinator = SimpleNamespace(single_type_managers=groups)

        def __init__(self) -> None:
            self.freed: list[str] = []

        def free(self, request: Any) -> None:
            self.freed.append(request.request_id)
            for group in groups:
                group.req_to_blocks.pop(request.request_id, None)

    manager = Manager()
    request = SimpleNamespace(
        request_id="waiting-request",
        status=SimpleNamespace(name="WAITING"),
    )

    assert manager.allocate_slots(request) is None
    assert manager.freed == ["waiting-request"]
    assert all(request.request_id not in group.req_to_blocks for group in groups)
    assert groups[0].new_block_ids == []
    assert any("Rolled back partial shared KV allocation" in item for item in logger.messages)


def test_hybrid_fp8_sleep_wake_zeroes_nested_cache_groups() -> None:
    class FakeTensor:
        def __init__(self) -> None:
            self.zero_calls = 0

        def zero_(self) -> None:
            self.zero_calls += 1

    class FakeGPUModelRunner:
        def __init__(self) -> None:
            self.kv_caches: list[Any] = [[FakeTensor(), FakeTensor()], FakeTensor()]
            self.scales_reset = False

        def init_fp8_kv_scales(self) -> None:
            for cache_tensor in self.kv_caches:
                cache_tensor.zero_()
            self.scales_reset = True

    logger = _Logger()
    module = SimpleNamespace(GPUModelRunner=FakeGPUModelRunner)
    _install_hybrid_sleep_wake_shim(module, logger)
    _install_hybrid_sleep_wake_shim(module, logger)
    runner = FakeGPUModelRunner()
    original_nested_group = runner.kv_caches[0]

    runner.init_fp8_kv_scales()

    assert runner.kv_caches[0] is original_nested_group
    assert [tensor.zero_calls for tensor in original_nested_group] == [1, 1]
    assert runner.kv_caches[1].zero_calls == 1
    assert runner.scales_reset
    assert logger.messages == ["Installed LLM-RIO hybrid FP8 KV-cache wake compatibility shim"]


def test_host_weight_backup_survives_wake_and_is_reused_on_next_sleep() -> None:
    original_backup = object()

    class FakeAllocator:
        default_tag = "default"

        def __init__(self) -> None:
            self.pointer_to_data = {
                1: SimpleNamespace(tag="weights", cpu_backup_tensor=original_backup),
                2: SimpleNamespace(tag="default", cpu_backup_tensor=None),
            }

        def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
            selected = (
                (self.default_tag,)
                if offload_tags is None
                else (offload_tags,)
                if isinstance(offload_tags, str)
                else offload_tags
            )
            for data in self.pointer_to_data.values():
                if data.tag in selected:
                    data.cpu_backup_tensor = object()

        def wake_up(self, tags: list[str] | None = None) -> None:
            for data in self.pointer_to_data.values():
                if tags is None or data.tag in tags:
                    data.cpu_backup_tensor = None

    logger = _Logger()
    module = SimpleNamespace(CuMemAllocator=FakeAllocator)
    _install_persistent_weight_backup_shim(module, logger)
    allocator = FakeAllocator()

    allocator.wake_up()
    assert allocator.pointer_to_data[1].cpu_backup_tensor is original_backup
    allocator.sleep(("weights",))
    assert allocator.pointer_to_data[1].cpu_backup_tensor is original_backup
    assert allocator.pointer_to_data[1].tag == "weights"


def _profile(
    model_id: str,
    gpu_set: tuple[str, ...],
    idle_mib: tuple[int, ...],
    *,
    utilization: float = 0.92,
    peak_mib: tuple[int, ...] | None = None,
    sleep_mib: tuple[int, ...] | None = None,
    wake_peak_mib: tuple[int, ...] | None = None,
    backend: str = "native",
) -> PlacementProfile:
    measured_peak = peak_mib or idle_mib
    return PlacementProfile(
        id=f"profile-{model_id}-{len(gpu_set)}",
        model_id=model_id,
        model_revision="revision",
        engine=Engine.VLLM,
        engine_version="0.26.0",
        machine_fingerprint="machine",
        gpu_count=len(gpu_set),
        tensor_parallel_size=len(gpu_set),
        pipeline_parallel_size=1,
        eligible_gpu_sets=(gpu_set,),
        dtype="auto",
        quantization=None,
        max_model_len=4096,
        max_num_seqs=128,
        max_num_batched_tokens=None,
        predicted_tokens_per_second=10.0,
        load_and_warmup_seconds=1.0,
        idle_vram_mib_per_gpu=idle_mib,
        peak_vram_mib_per_gpu=measured_peak,
        gpu_headroom_mib_per_gpu=(0,) * len(gpu_set),
        capabilities=frozenset({"chat"}),
        launch_args={},
        gpu_memory_utilization=utilization,
        kv_cache_capacity_tokens=4096,
        max_full_length_concurrency=1.0,
        memory_backend=backend,
        sleep_vram_mib_per_gpu=(1536,) * len(gpu_set) if sleep_mib is None else sleep_mib,
        normal_verified=backend == "native",
        kvcached_verified=backend == "kvcached",
        vram_measurement_version=2,
        vram_baseline_mib_per_gpu=(0,) * len(gpu_set),
        wake_peak_vram_mib_per_gpu=wake_peak_mib or measured_peak,
    )


def test_prism_has_no_fixed_sleeping_worker_count_limit() -> None:
    now = datetime.now(UTC)
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    older = WorkerPlacement(
        id="older-cache-entry",
        profile=_profile("model-a", ("GPU-0",), (30_000,)),
        gpu_uuids=("GPU-0",),
        port=18000,
        state=RuntimeState.SLEEPING,
    )
    older.last_demand_at = now - timedelta(minutes=2)
    newer = WorkerPlacement(
        id="newer-cache-entry",
        profile=_profile("model-b", ("GPU-0",), (30_000,)),
        gpu_uuids=("GPU-0",),
        port=18001,
        state=RuntimeState.SLEEPING,
    )
    newer.last_demand_at = now - timedelta(minutes=1)
    incoming = _profile("model-c", ("GPU-0",), (30_000,))

    actions = planner.plan(
        now=now,
        all_gpu_uuids={"GPU-0"},
        workers=[newer, older],
        pressures=[
            QueuePressure(
                model_id="model-c",
                requests=1,
                estimated_tokens=128,
                oldest_enqueued_at=now - timedelta(seconds=2),
            )
        ],
        profiles={"model-c": [incoming]},
    )

    assert len(actions) == 1
    assert isinstance(actions[0], StartPlacement)
    assert actions[0].reason == "prism_cold_backlog"


def test_native_sleep_uses_incremental_peak_and_applies_global_reserve_once() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        kvcached_required=False,
        gpu_vram_mib={"GPU-0": 97_249},
        reserved_vram_mib=2048,
    )
    # The per-device budget is physical VRAM minus the global reserve exactly once.
    # Neither gpu_memory_utilization nor the legacy per-profile headroom is reapplied.
    fitting = _profile(
        "fitting",
        ("GPU-0",),
        (95_201,),
        peak_mib=(95_201,),
        utilization=0.92,
    )
    assert planner._cached_prism_fits(fitting, ("GPU-0",), [])

    too_large = _profile(
        "too-large",
        ("GPU-0",),
        (95_202,),
        peak_mib=(95_202,),
        utilization=0.92,
    )

    assert not planner._cached_prism_fits(too_large, ("GPU-0",), [])


def test_cached_fit_composes_active_peak_and_sleep_residual() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        kvcached_required=False,
        gpu_vram_mib={"GPU-0": 97_249},
        reserved_vram_mib=2048,
    )
    sleeping_profile = _profile("sleeping", ("GPU-0",), (30_000,), sleep_mib=(5_201,))
    sleeping = WorkerPlacement(
        id="sleeping-worker",
        profile=sleeping_profile,
        gpu_uuids=("GPU-0",),
        port=18000,
        state=RuntimeState.SLEEPING,
    )
    fitting = _profile("incoming", ("GPU-0",), (90_000,))
    assert planner._cached_prism_fits(fitting, ("GPU-0",), [sleeping])

    sleeping.profile = replace(sleeping_profile, sleep_vram_mib_per_gpu=(5_202,))
    assert not planner._cached_prism_fits(fitting, ("GPU-0",), [sleeping])


def test_cached_fit_rejects_legacy_absolute_vram_profile() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_249},
        reserved_vram_mib=2048,
    )
    legacy = replace(
        _profile("legacy", ("GPU-0",), (30_000,)),
        vram_measurement_version=1,
        vram_baseline_mib_per_gpu=None,
    )
    assert not planner._cached_prism_fits(legacy, ("GPU-0",), [])


def test_prism_falls_back_to_tp_profile_that_uses_composed_measured_peaks() -> None:
    gpu_set = ("GPU-0", "GPU-1")
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    resident_profile = _profile("resident", gpu_set, (65_000, 65_000), backend="kvcached")
    resident = WorkerPlacement(
        id="resident-worker",
        profile=resident_profile,
        gpu_uuids=gpu_set,
        port=18000,
        state=RuntimeState.READY,
    )
    # TP1 exceeds the physical-minus-reserve budget on GPU-1 once its measured
    # peak is composed with the resident peak; the measured TP2 shape fits.
    tp1 = _profile("incoming", ("GPU-1",), (30_181,), peak_mib=(31_019,), backend="kvcached")
    tp2 = _profile(
        "incoming",
        gpu_set,
        (19_977, 19_421),
        peak_mib=(20_107, 19_551),
        backend="kvcached",
    )

    assert not planner._prism_fits(tp1, ("GPU-1",), [resident])
    assert planner._prism_fits(tp2, gpu_set, [resident])


def test_prism_preload_warms_requested_replica_count_without_waking_cache() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (30_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (30_000,))
    cached = WorkerPlacement(
        id="cached-hot-model",
        profile=gpu0_profile,
        gpu_uuids=("GPU-0",),
        port=18000,
        state=RuntimeState.SLEEPING,
    )

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=[cached],
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=0,
                oldest_enqueued_at=datetime.now(UTC),
                estimated_tokens=1,
                preload=True,
                desired_workers=2,
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile]},
    )

    assert len(actions) == 1
    assert isinstance(actions[0], StartPlacement)
    assert actions[0].gpu_uuids == ("GPU-1",)
    assert cached.state is RuntimeState.SLEEPING


def test_prism_replica_scaling_does_not_add_tp2_after_both_tp1_copies() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (30_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (30_000,))
    tp2_profile = _profile("hot-model", ("GPU-0", "GPU-1"), (20_000, 20_000))
    workers = [
        WorkerPlacement(
            id="hot-gpu-0",
            profile=gpu0_profile,
            gpu_uuids=("GPU-0",),
            port=18000,
            state=RuntimeState.READY,
        ),
        WorkerPlacement(
            id="hot-gpu-1",
            profile=gpu1_profile,
            gpu_uuids=("GPU-1",),
            port=18001,
            state=RuntimeState.READY,
        ),
    ]

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=workers,
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=100,
                estimated_tokens=100_000,
                oldest_enqueued_at=datetime.now(UTC),
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile, tp2_profile]},
    )

    assert actions == []


def test_prism_single_large_request_does_not_start_unsplittable_replica() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (30_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (30_000,))
    resident = WorkerPlacement(
        id="hot-gpu-0",
        profile=gpu0_profile,
        gpu_uuids=("GPU-0",),
        port=18000,
        state=RuntimeState.READY,
    )

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=[resident],
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=1,
                estimated_tokens=100_000,
                oldest_enqueued_at=datetime.now(UTC),
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile]},
    )

    assert actions == []


def test_prism_concurrent_large_requests_can_start_replica() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (30_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (30_000,))
    resident = WorkerPlacement(
        id="hot-gpu-0",
        profile=gpu0_profile,
        gpu_uuids=("GPU-0",),
        port=18000,
        state=RuntimeState.READY,
    )

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=[resident],
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=2,
                estimated_tokens=100_000,
                oldest_enqueued_at=datetime.now(UTC),
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile]},
    )

    assert len(actions) == 1
    assert isinstance(actions[0], StartPlacement)
    assert actions[0].gpu_uuids == ("GPU-1",)


def test_prism_burst_wakes_two_cached_replicas_before_admission() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (30_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (30_000,))
    workers = [
        WorkerPlacement(
            id="cached-gpu-0",
            profile=gpu0_profile,
            gpu_uuids=("GPU-0",),
            port=18000,
            state=RuntimeState.SLEEPING,
        ),
        WorkerPlacement(
            id="cached-gpu-1",
            profile=gpu1_profile,
            gpu_uuids=("GPU-1",),
            port=18001,
            state=RuntimeState.SLEEPING,
        ),
    ]

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=workers,
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=40,
                estimated_tokens=100_000,
                oldest_enqueued_at=datetime.now(UTC),
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile]},
    )

    assert len(actions) == 2
    assert all(isinstance(action, WakePlacement) for action in actions)
    assert {action.worker_id for action in actions if isinstance(action, WakePlacement)} == {
        "cached-gpu-0",
        "cached-gpu-1",
    }


def test_prism_single_request_wakes_only_one_cached_replica() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (30_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (30_000,))
    workers = [
        WorkerPlacement(
            id="cached-gpu-0",
            profile=gpu0_profile,
            gpu_uuids=("GPU-0",),
            port=18000,
            state=RuntimeState.SLEEPING,
        ),
        WorkerPlacement(
            id="cached-gpu-1",
            profile=gpu1_profile,
            gpu_uuids=("GPU-1",),
            port=18001,
            state=RuntimeState.SLEEPING,
        ),
    ]

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=workers,
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=1,
                estimated_tokens=100_000,
                oldest_enqueued_at=datetime.now(UTC),
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile]},
    )

    assert len(actions) == 1
    assert isinstance(actions[0], WakePlacement)


def test_duplicate_preload_never_substitutes_one_tp2_worker() -> None:
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        gpu_vram_mib={"GPU-0": 97_887, "GPU-1": 97_887},
        reserved_vram_mib=2048,
        prism_max_workers_per_gpu=2,
    )
    resident_profile = _profile("resident", ("GPU-0", "GPU-1"), (60_000, 60_000))
    resident = WorkerPlacement(
        id="resident",
        profile=resident_profile,
        gpu_uuids=("GPU-0", "GPU-1"),
        port=18000,
        state=RuntimeState.READY,
    )
    gpu0_profile = _profile("hot-model", ("GPU-0",), (32_000,))
    gpu1_profile = _profile("hot-model", ("GPU-1",), (32_000,))
    tp2_profile = _profile("hot-model", ("GPU-0", "GPU-1"), (20_000, 20_000))

    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=[resident],
        pressures=[
            QueuePressure(
                model_id="hot-model",
                requests=0,
                estimated_tokens=1,
                oldest_enqueued_at=datetime.now(UTC),
                preload=True,
                desired_workers=2,
            )
        ],
        profiles={"hot-model": [gpu0_profile, gpu1_profile, tp2_profile]},
    )

    assert actions
    assert not any(isinstance(action, StartPlacement) for action in actions)


def test_required_kvcached_mode_filters_unverified_profiles() -> None:
    now = datetime.now(UTC)
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        kvcached_required=True,
        gpu_vram_mib={"GPU-0": 97_887},
        reserved_vram_mib=2048,
    )
    unverified = replace(
        _profile("model-a", ("GPU-0",), (30_000,), backend="kvcached"),
        kvcached_verified=False,
    )
    pressure = QueuePressure(
        model_id="model-a",
        requests=1,
        estimated_tokens=128,
        oldest_enqueued_at=now - timedelta(seconds=2),
    )

    assert (
        planner.plan(
            now=now,
            all_gpu_uuids={"GPU-0"},
            workers=[],
            pressures=[pressure],
            profiles={"model-a": [unverified]},
        )
        == []
    )

    verified = replace(unverified, kvcached_verified=True)
    actions = planner.plan(
        now=now,
        all_gpu_uuids={"GPU-0"},
        workers=[],
        pressures=[pressure],
        profiles={"model-a": [verified]},
    )
    assert len(actions) == 1
    assert isinstance(actions[0], StartPlacement)


def test_native_sleep_planner_rejects_llama_cpp_profiles() -> None:
    now = datetime.now(UTC)
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        kvcached_required=False,
        gpu_vram_mib={"GPU-0": 97_887},
        reserved_vram_mib=2048,
    )
    llama_profile = replace(
        _profile("model-a", ("GPU-0",), (30_000,)),
        engine=Engine.LLAMA_CPP,
        memory_backend="native",
        kvcached_verified=False,
    )

    actions = planner.plan(
        now=now,
        all_gpu_uuids={"GPU-0"},
        workers=[],
        pressures=[
            QueuePressure(
                model_id="model-a",
                requests=1,
                estimated_tokens=128,
                oldest_enqueued_at=now - timedelta(seconds=2),
            )
        ],
        profiles={"model-a": [llama_profile]},
    )

    assert actions == []
