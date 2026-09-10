from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm_rio import registration
from llm_rio.domain import GpuDevice, MachineInventory, RuntimeState, ServiceMode
from llm_rio.registration import RegistrationManager
from llm_rio.runtime import ResidencyScheduler
from llm_rio.validation import (
    CandidateShape,
    ProfileValidator,
    ValidationError,
    _model_launch_args,
    _VramSampler,
)


def test_deepseek_v4_uses_mandatory_fp8_kv_cache(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}),
        encoding="utf-8",
    )

    assert _model_launch_args(tmp_path) == {"kv_cache_dtype": "fp8"}


async def test_vram_sampler_captures_incremental_peak_above_one_baseline() -> None:
    current = [1000, 2000]

    def read_vram(_gpu_set: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(current)

    sampler = _VramSampler(read_vram, ("GPU-0", "GPU-1"), interval_seconds=0.001)
    sampler.start()
    current[:] = [1400, 2600]
    await asyncio.sleep(0.01)
    current[:] = [1100, 2200]
    await sampler.stop()

    assert sampler.baseline_mib == (1000, 2000)
    assert sampler.peak() == (400, 600)
    assert sampler.baseline_drop_mib == (0, 0)


def test_vram_capacity_uses_physical_minus_global_reserve_once() -> None:
    validator = ProfileValidator.__new__(ProfileValidator)
    validator.settings = cast(Any, SimpleNamespace(reserved_vram_mib=2048))
    validator.inventory = MachineInventory(
        machine_id="test",
        driver_version="test",
        cuda_driver_version="test",
        gpus=(GpuDevice("GPU-0", 0, "GPU", 97_249),),
        topology_hash="test",
        fingerprint="test",
    )

    validator._validate_vram_measurements(
        gpu_set=("GPU-0",),
        peak_memory=(95_201,),
        baseline_drop_mib=(0,),
    )
    with pytest.raises(ValidationError) as capacity_error:
        validator._validate_vram_measurements(
            gpu_set=("GPU-0",),
            peak_memory=(95_202,),
            baseline_drop_mib=(0,),
        )
    assert capacity_error.value.stage == "gpu_capacity"

    with pytest.raises(ValidationError) as baseline_error:
        validator._validate_vram_measurements(
            gpu_set=("GPU-0",),
            peak_memory=(1,),
            baseline_drop_mib=(1,),
        )
    assert baseline_error.value.stage == "gpu_measurement"


class ValidationDatabase:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def service_mode(self) -> ServiceMode:
        return ServiceMode.ACTIVE

    async def record_event(
        self,
        event_type: str,
        _entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((event_type, payload or {}))


class ValidationQueues:
    def __init__(self) -> None:
        self.has_demand = False

    def pending_models(self) -> list[str]:
        return ["production-model"] if self.has_demand else []


class ValidationSupervisor:
    ram_weight_cache_enabled = True

    def __init__(self, workers: list[Any], queues: ValidationQueues) -> None:
        self.workers = {worker.id: worker for worker in workers}
        self.queues = queues
        self.sleep_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.arrive_during_sleep = False

    async def sleep(self, worker_id: str) -> None:
        self.sleep_calls.append(worker_id)
        self.workers[worker_id].state = RuntimeState.SLEEPING
        if self.arrive_during_sleep:
            self.queues.has_demand = True

    async def stop(self, worker_id: str, *, force: bool) -> None:
        self.stop_calls.append(worker_id)
        self.workers[worker_id].state = RuntimeState.COLD


def validation_scheduler(*workers: Any) -> tuple[ResidencyScheduler, ValidationSupervisor]:
    queues = ValidationQueues()
    supervisor = ValidationSupervisor(list(workers), queues)
    scheduler = ResidencyScheduler.__new__(ResidencyScheduler)
    scheduler.settings = SimpleNamespace(validation_idle_window_seconds=0)
    scheduler.database = ValidationDatabase()
    scheduler.supervisor = supervisor
    scheduler.queues = queues
    scheduler._state_lock = asyncio.Lock()
    scheduler._validation_gpu_uuids = set()
    scheduler._last_arrival_at = datetime.now(UTC) - timedelta(seconds=60)
    scheduler._maintenance_requested = False
    scheduler._closed = False
    scheduler.kvcached = SimpleNamespace(enabled=True)
    scheduler._event = asyncio.Event()
    return scheduler, supervisor


async def test_validation_preserves_ram_cached_workers() -> None:
    ready = SimpleNamespace(
        id="ready",
        state=RuntimeState.READY,
        gpu_uuids=("GPU-0",),
        admitted_request_ids=set(),
    )
    sleeping = SimpleNamespace(
        id="sleeping",
        state=RuntimeState.SLEEPING,
        gpu_uuids=("GPU-0",),
        admitted_request_ids=set(),
    )
    scheduler, supervisor = validation_scheduler(ready, sleeping)

    acquired = await scheduler.acquire_validation_gpus(("GPU-0",))

    assert acquired
    assert supervisor.sleep_calls == ["ready"]
    assert supervisor.stop_calls == []
    assert ready.state is RuntimeState.SLEEPING
    assert sleeping.state is RuntimeState.SLEEPING
    assert scheduler._validation_gpu_uuids == {"GPU-0"}
    assert scheduler.database.events[-1] == (
        "VALIDATION_GPUS_ACQUIRED",
        {
            "gpu_uuids": ("GPU-0",),
            "preserved_cached_worker_ids": ["ready", "sleeping"],
            "evicted_cached_worker_ids": [],
        },
    )


async def test_validation_defers_if_production_arrives_while_worker_sleeps() -> None:
    ready = SimpleNamespace(
        id="ready",
        state=RuntimeState.READY,
        gpu_uuids=("GPU-0",),
        admitted_request_ids=set(),
    )
    scheduler, supervisor = validation_scheduler(ready)
    supervisor.arrive_during_sleep = True

    acquired = await scheduler.acquire_validation_gpus(("GPU-0",))

    assert not acquired
    assert supervisor.sleep_calls == ["ready"]
    assert supervisor.stop_calls == []
    assert scheduler._validation_gpu_uuids == set()
    assert scheduler.database.events[-1][0] == "VALIDATION_GPUS_DEFERRED"
    assert scheduler.database.events[-1][1]["production_demand"] is True


def candidate_shape(
    tensor_parallel_size: int,
    gpu_sets: tuple[tuple[str, ...], ...],
) -> CandidateShape:
    return CandidateShape(
        gpu_count=tensor_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=4096,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        gpu_memory_utilization=0.9,
        dtype="auto",
        quantization=None,
        eligible_gpu_sets=gpu_sets,
    )


class ConcurrentProbeScheduler:
    def __init__(self) -> None:
        self.acquired: set[tuple[str, ...]] = set()
        self.released: set[tuple[str, ...]] = set()

    async def acquire_validation_gpus(self, gpu_set: tuple[str, ...]) -> bool:
        self.acquired.add(gpu_set)
        return True

    async def release_validation_gpus(self, gpu_set: tuple[str, ...]) -> None:
        self.released.add(gpu_set)


async def test_vllm_validation_starts_disjoint_gpu_probes_concurrently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gpu_sets = (("GPU-0",), ("GPU-1",))
    scheduler = ConcurrentProbeScheduler()
    validator = ProfileValidator(
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        cast(Any, scheduler),
    )
    both_started = asyncio.Event()
    allow_completion = asyncio.Event()
    started: set[tuple[str, ...]] = set()

    async def probe(_validator: ProfileValidator, **kwargs: Any) -> Any:
        gpu_set = cast(tuple[str, ...], kwargs["gpu_set"])
        started.add(gpu_set)
        if len(started) == len(gpu_sets):
            both_started.set()
        await allow_completion.wait()
        return gpu_set

    monkeypatch.setattr(ProfileValidator, "_probe_vllm", probe)
    validation = asyncio.create_task(
        validator.validate_vllm(
            model_id="model",
            model_revision="revision",
            model_path=tmp_path,
            nickname="model",
            candidate=candidate_shape(1, gpu_sets),
        )
    )
    try:
        await asyncio.wait_for(both_started.wait(), timeout=0.5)
        assert started == set(gpu_sets)
        assert scheduler.acquired == set(gpu_sets)
    finally:
        allow_completion.set()

    assert await validation == list(gpu_sets)
    assert scheduler.released == set(gpu_sets)


async def test_validation_port_reservations_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = ProfileValidator.__new__(ProfileValidator)
    validator.settings = cast(Any, SimpleNamespace(worker_port_end=19_000))
    validator._validation_ports = set()
    validator._validation_port_lock = asyncio.Lock()
    monkeypatch.setattr(
        ProfileValidator,
        "_local_port_available",
        staticmethod(lambda _port: True),
    )

    first, second = await asyncio.gather(
        validator._reserve_validation_port(),
        validator._reserve_validation_port(),
    )

    assert (first, second) == (19_001, 19_002)
    await validator._release_validation_port(first)
    await validator._release_validation_port(second)
    assert validator._validation_ports == set()


class RegistrationDatabaseStub:
    def __init__(self) -> None:
        self.progress: list[dict[str, Any] | None] = []

    async def update_model_job(self, _job_id: str, **kwargs: Any) -> None:
        self.progress.append(cast(dict[str, Any] | None, kwargs.get("progress")))


class RegistrationValidatorStub:
    scheduler = SimpleNamespace(validation_requires_maintenance=False)

    def __init__(self, *, fail_tp1: bool = False) -> None:
        self.fail_tp1 = fail_tp1
        self.validated_tensor_parallel_sizes: list[int] = []

    async def validate_vllm(self, **kwargs: Any) -> list[str]:
        shape = cast(CandidateShape, kwargs["candidate"])
        self.validated_tensor_parallel_sizes.append(shape.tensor_parallel_size)
        if self.fail_tp1 and shape.tensor_parallel_size == 1:
            raise ValidationError("validation", "TP=1 does not fit")
        return [f"tp{shape.tensor_parallel_size}"]


def registration_manager(validator: RegistrationValidatorStub) -> RegistrationManager:
    manager = RegistrationManager.__new__(RegistrationManager)
    manager.database = RegistrationDatabaseStub()
    manager.inventory = cast(Any, SimpleNamespace())
    manager.settings = cast(
        Any,
        SimpleNamespace(
            reserved_vram_mib=2048,
            engines=SimpleNamespace(
                gpu_memory_utilization=None,
                max_model_len=None,
                max_num_seqs=None,
                max_num_batched_tokens=None,
            ),
        ),
    )
    manager.validator = cast(Any, validator)
    manager._validation_lock = asyncio.Lock()
    return manager


async def validate_registration(manager: RegistrationManager) -> list[Any]:
    return await manager._validate_with_requeue(
        job_id="job",
        job={"model_id": "model", "nickname": "model"},
        artifact_path=Path("/tmp/model"),
        resolved_revision="revision",
        inspection={
            "weight_bytes": 1,
            "max_model_len": 4096,
            "dtype": "auto",
            "quantization": None,
        },
    )


async def test_automatic_registration_stops_after_successful_tp1_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = RegistrationValidatorStub()
    manager = registration_manager(validator)
    shapes = [candidate_shape(1, (("GPU-0",),)), candidate_shape(2, (("GPU-0", "GPU-1"),))]
    monkeypatch.setattr(registration, "build_candidate_shapes", lambda **_kwargs: shapes)

    assert await validate_registration(manager) == ["tp1"]
    assert validator.validated_tensor_parallel_sizes == [1]


async def test_automatic_registration_uses_tp2_when_tp1_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = RegistrationValidatorStub(fail_tp1=True)
    manager = registration_manager(validator)
    shapes = [candidate_shape(1, (("GPU-0",),)), candidate_shape(2, (("GPU-0", "GPU-1"),))]
    monkeypatch.setattr(registration, "build_candidate_shapes", lambda **_kwargs: shapes)

    assert await validate_registration(manager) == ["tp2"]
    assert validator.validated_tensor_parallel_sizes == [1, 2]


async def test_registration_uses_persisted_validation_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = RegistrationValidatorStub()
    manager = registration_manager(validator)
    captured: dict[str, Any] = {}

    def build_candidates(**kwargs: Any) -> list[CandidateShape]:
        captured.update(kwargs)
        return [candidate_shape(2, (("GPU-0", "GPU-1"),))]

    monkeypatch.setattr(registration, "build_candidate_shapes", build_candidates)

    await manager._validate_with_requeue(
        job_id="job",
        job={
            "model_id": "model",
            "nickname": "model",
            "validation_overrides": {
                "max_model_len": 262_144,
                "max_num_seqs": 256,
                "gpu_memory_utilization": 0.84,
            },
        },
        artifact_path=Path("/tmp/model"),
        resolved_revision="revision",
        inspection={
            "weight_bytes": 1,
            "max_model_len": 1_048_576,
            "dtype": "auto",
            "quantization": None,
        },
    )

    assert captured["max_model_len"] == 262_144
    assert captured["max_model_len_limit"] == 262_144
    assert captured["max_num_seqs"] == 256
    assert captured["gpu_memory_utilization"] == 0.84
