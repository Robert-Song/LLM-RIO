from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_scheduler_contract import FakeDatabase, FakeProfiles, make_profile

from llm_rio.config import Settings
from llm_rio.domain import MachineInventory, RuntimeState, WorkerPlacement
from llm_rio.gpu_memory import GpuMemory, read_gpu_memory, required_free_vram
from llm_rio.planner import StartPlacement, WakePlacement
from llm_rio.runtime import ResidencyScheduler
from llm_rio.workers import WorkerSupervisor


def profile(gpus=("GPU-0",), peak=80, sleep=5, utilization=0.8):
    base = make_profile("incoming", "incoming", gpus)
    return replace(
        base,
        peak_vram_mib_per_gpu=(peak,) * len(gpus),
        wake_peak_vram_mib_per_gpu=(peak,) * len(gpus),
        sleep_vram_mib_per_gpu=(sleep,) * len(gpus),
        gpu_memory_utilization=utilization,
    )


def worker(name, *, gpus=("GPU-0",), age=0, state=RuntimeState.SLEEPING):
    return WorkerPlacement(
        name,
        replace(profile(gpus), id=name, model_id=name),
        gpus,
        18000,
        state=state,
        process_pid=123,
        last_demand_at=datetime.now(UTC) - timedelta(seconds=age),
    )


@pytest.fixture
def supervisor():
    return WorkerSupervisor(Settings(reserved_vram_mib=5), FakeDatabase())


def test_native_startup_budget_is_enforced_even_when_measured_peak_is_small() -> None:
    assert (
        required_free_vram(profile(peak=30, utilization=0.9), 0, GpuMemory(100, 85), reserve_mib=5)
        == 90
    )
    assert (
        required_free_vram(profile(peak=90, utilization=0.9), 0, GpuMemory(100, 95), reserve_mib=5)
        == 95
    )


def test_wake_only_credits_target_residual_verified_in_live_process_memory() -> None:
    p = profile(peak=90, sleep=5)
    assert (
        required_free_vram(
            p, 0, GpuMemory(100, 80, {123: 5, 456: 15}), reserve_mib=5, waking_pid=123, waking=True
        )
        == 90
    )
    # Historical residual must not over-credit less memory in the current process.
    assert (
        required_free_vram(
            p, 0, GpuMemory(100, 80, {123: 2}), reserve_mib=5, waking_pid=123, waking=True
        )
        == 93
    )


async def test_sleeping_workers_are_evicted_in_lru_order_until_live_memory_fits(
    supervisor, monkeypatch
) -> None:
    oldest = worker("oldest", age=30)
    newer = worker("newer", age=20)
    newest = worker("newest", age=10)
    supervisor.workers = {w.id: w for w in (newer, newest, oldest)}
    free = 50
    reads = []
    stopped = []

    def sample(_):
        reads.append(free)
        return {"GPU-0": GpuMemory(100, free)}

    async def stop(worker_id, *, force):
        nonlocal free
        assert not force
        stopped.append(worker_id)
        supervisor.workers[worker_id].state = RuntimeState.COLD
        free += 20

    monkeypatch.setattr("llm_rio.workers.read_gpu_memory", sample)
    monkeypatch.setattr(supervisor, "stop", stop)
    assert await supervisor.ensure_gpu_capacity(profile(), ("GPU-0",))
    assert stopped == ["oldest", "newer"]
    assert reads == [50, 70, 90]
    assert newest.state is RuntimeState.SLEEPING
    assert oldest.last_cache_eviction_reason == "gpu_vram_pressure"


async def test_eviction_never_targets_busy_workers_or_unrelated_gpus(
    supervisor, monkeypatch
) -> None:
    busy = worker("busy")
    busy.admitted_request_ids.add("request")
    ready = worker("ready", state=RuntimeState.READY)
    unrelated = worker("unrelated", gpus=("GPU-1",))
    supervisor.workers = {w.id: w for w in (busy, ready, unrelated)}
    supervisor.stop = AsyncMock()
    monkeypatch.setattr("llm_rio.workers.read_gpu_memory", lambda _: {"GPU-0": GpuMemory(100, 10)})
    assert not await supervisor.ensure_gpu_capacity(profile(), ("GPU-0",))
    supervisor.stop.assert_not_awaited()


async def test_process_stop_is_not_assumed_to_release_vram(supervisor, monkeypatch) -> None:
    sleeper = worker("sleeper")
    supervisor.workers = {sleeper.id: sleeper}
    monkeypatch.setattr("llm_rio.workers.read_gpu_memory", lambda _: {"GPU-0": GpuMemory(100, 70)})

    async def stop(*args, **kwargs):
        sleeper.state = RuntimeState.COLD

    monkeypatch.setattr(supervisor, "stop", stop)
    assert not await supervisor.ensure_gpu_capacity(profile(), ("GPU-0",))
    assert supervisor.database.events[-1][0] == "GPU_CAPACITY_DEFERRED"


async def test_tensor_parallel_admission_checks_each_gpu_and_evicts_on_deficient_gpu(
    supervisor, monkeypatch
) -> None:
    enough = worker("enough", gpus=("GPU-0",), age=30)
    blocked = worker("blocked", gpus=("GPU-1",), age=10)
    supervisor.workers = {w.id: w for w in (enough, blocked)}
    free = {"GPU-0": 95, "GPU-1": 70}
    monkeypatch.setattr(
        "llm_rio.workers.read_gpu_memory",
        lambda _: {gpu: GpuMemory(100, value) for gpu, value in free.items()},
    )
    stopped = []

    async def stop(worker_id, **kwargs):
        stopped.append(worker_id)
        supervisor.workers[worker_id].state = RuntimeState.COLD
        free["GPU-1"] = 90

    monkeypatch.setattr(supervisor, "stop", stop)
    assert await supervisor.ensure_gpu_capacity(profile(("GPU-0", "GPU-1")), ("GPU-0", "GPU-1"))
    assert stopped == ["blocked"]


async def test_wake_preserves_target_and_evicts_other_sleepers(supervisor, monkeypatch) -> None:
    target, other = worker("target", age=30), worker("other", age=10)
    target.profile = profile(peak=90, sleep=5)
    supervisor.workers = {w.id: w for w in (target, other)}
    free = 80
    monkeypatch.setattr(
        "llm_rio.workers.read_gpu_memory", lambda _: {"GPU-0": GpuMemory(100, free, {123: 5})}
    )
    stopped = []

    async def stop(worker_id, **kwargs):
        nonlocal free
        stopped.append(worker_id)
        supervisor.workers[worker_id].state = RuntimeState.COLD
        free = 95

    monkeypatch.setattr(supervisor, "stop", stop)
    assert await supervisor.ensure_gpu_capacity(
        target.profile, ("GPU-0",), waking_worker_id=target.id
    )
    assert stopped == ["other"]
    assert target.state is RuntimeState.SLEEPING


async def test_unaccountable_wake_residual_can_fall_back_to_cold_start(
    supervisor, monkeypatch
) -> None:
    target = worker("target")
    target.profile = profile(peak=90, sleep=10)
    supervisor.workers = {target.id: target}
    monkeypatch.setattr("llm_rio.workers.read_gpu_memory", lambda _: {"GPU-0": GpuMemory(100, 90)})
    supervisor.stop = AsyncMock()
    assert not await supervisor.ensure_gpu_capacity(
        target.profile, ("GPU-0",), waking_worker_id=target.id
    )
    supervisor.stop.assert_awaited_once_with(target.id, force=False)
    assert target.last_cache_eviction_reason == "gpu_vram_pressure_cold_restart"


async def test_missing_nvml_defers_without_eviction_and_does_not_spam_events(
    supervisor, monkeypatch
) -> None:
    def fail(_):
        raise RuntimeError("NVML unavailable")

    monkeypatch.setattr("llm_rio.workers.read_gpu_memory", fail)
    supervisor.stop = AsyncMock()
    assert not await supervisor.ensure_gpu_capacity(profile(), ("GPU-0",))
    assert not await supervisor.ensure_gpu_capacity(profile(), ("GPU-0",))
    supervisor.stop.assert_not_awaited()
    assert len(supervisor.database.events) == 1


@pytest.mark.parametrize("experimental", [False, True])
async def test_scheduler_requires_headroom_before_launch_and_wake(supervisor, experimental) -> None:
    database = supervisor.database
    database.model_by_id = AsyncMock(
        return_value={"artifact_path": "/fake/model", "nickname": "incoming"}
    )
    scheduler = ResidencyScheduler(
        settings=supervisor.settings,
        database=database,
        inventory=MachineInventory("test", "driver", None, (), "topology", "fingerprint"),
        profiles=FakeProfiles(),
        supervisor=supervisor,
    )
    scheduler.kvcached = SimpleNamespace(enabled=experimental)
    supervisor.ensure_gpu_capacity = AsyncMock(return_value=False)
    supervisor.launch = AsyncMock()
    supervisor.wake = AsyncMock()
    target = worker("target")
    supervisor.workers = {target.id: target}
    await scheduler._launch_if_active(StartPlacement(profile(), ("GPU-0",), "test"))
    await scheduler._wake_if_active(WakePlacement(target.id, "test"))
    assert supervisor.launch.await_count == int(experimental)
    assert supervisor.wake.await_count == int(experimental)
    assert supervisor.ensure_gpu_capacity.await_count == (0 if experimental else 2)


def test_nvml_reader_counts_free_vram_and_tensor_parallel_child_contexts(monkeypatch) -> None:
    import sys

    mib = 1024 * 1024
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetMemoryInfo=lambda _: SimpleNamespace(
            total=100 * mib, free=65 * mib, used=35 * mib
        ),
        nvmlDeviceGetComputeRunningProcesses=lambda _: [
            SimpleNamespace(pid=11, usedGpuMemory=10 * mib),
            SimpleNamespace(pid=12, usedGpuMemory=5 * mib),
            SimpleNamespace(pid=21, usedGpuMemory=20 * mib),
        ],
        NVMLError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.setattr("llm_rio.gpu_memory.os.getpgid", lambda pid: 10 if pid in (11, 12) else 20)
    assert read_gpu_memory(("GPU-0",)) == {"GPU-0": GpuMemory(100, 65, {10: 15, 20: 20})}
