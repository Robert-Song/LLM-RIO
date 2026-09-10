from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_scheduler_contract import FakeDatabase, FakeProfiles, FakeSupervisor, make_profile
from test_validation_cleanup import (
    candidate_shape,
    validation_scheduler,
)

from llm_rio.config import Settings
from llm_rio.domain import MachineInventory, RuntimeState, ServiceMode, WorkerPlacement
from llm_rio.errors import RioError
from llm_rio.gpu_memory import GpuMemory
from llm_rio.planner import GreedyPlacementPlanner, QueuePressure, WakePlacement
from llm_rio.registration import RegistrationManager
from llm_rio.runtime import ResidencyScheduler
from llm_rio.validation import ProfileValidator, ValidationPreempted


def normal_scheduler(*workers):
    scheduler, supervisor = validation_scheduler(*workers)
    scheduler.kvcached.enabled = False
    return scheduler, supervisor


@pytest.mark.parametrize("mode", [ServiceMode.ACTIVE, ServiceMode.DRAINING])
async def test_normal_validation_cannot_acquire_before_maintenance_ready(mode) -> None:
    sleeper = SimpleNamespace(
        id="sleeper", state=RuntimeState.SLEEPING, gpu_uuids=("GPU-0",), admitted_request_ids=set()
    )
    scheduler, supervisor = normal_scheduler(sleeper)
    scheduler.database.service_mode = AsyncMock(return_value=mode)
    assert not await scheduler.acquire_validation_gpus(("GPU-0",))
    assert supervisor.stop_calls == []
    assert supervisor.sleep_calls == []
    assert scheduler._validation_gpu_uuids == set()


async def test_maintenance_validation_fully_unloads_retained_contexts() -> None:
    sleepers = [
        SimpleNamespace(
            id=f"sleeper-{i}",
            state=RuntimeState.SLEEPING,
            gpu_uuids=("GPU-0",),
            admitted_request_ids=set(),
        )
        for i in range(2)
    ]
    scheduler, supervisor = normal_scheduler(*sleepers)
    scheduler.database.service_mode = AsyncMock(return_value=ServiceMode.MAINTENANCE_READY)
    scheduler._maintenance_requested = True
    assert await scheduler.acquire_validation_gpus(("GPU-0",))
    assert supervisor.stop_calls == ["sleeper-0", "sleeper-1"]
    assert supervisor.sleep_calls == []
    assert all(worker.state is RuntimeState.COLD for worker in sleepers)
    assert not scheduler.validation_should_yield()
    with pytest.raises(RioError) as error:
        await scheduler.resume()
    assert error.value.code == "validation_in_progress"
    await scheduler.release_validation_gpus(("GPU-0",))
    scheduler.database.set_service_mode = AsyncMock()
    await scheduler.resume()
    scheduler.database.set_service_mode.assert_awaited_once_with(ServiceMode.ACTIVE)
    assert scheduler.validation_should_yield()


async def test_failed_unload_does_not_grant_validation_gpu() -> None:
    sleeper = SimpleNamespace(
        id="sleeper", state=RuntimeState.SLEEPING, gpu_uuids=("GPU-0",), admitted_request_ids=set()
    )
    scheduler, supervisor = normal_scheduler(sleeper)
    scheduler.database.service_mode = AsyncMock(return_value=ServiceMode.MAINTENANCE_READY)
    supervisor.stop = AsyncMock()  # Process remains alive despite stop returning.
    assert not await scheduler.acquire_validation_gpus(("GPU-0",))
    assert scheduler._validation_gpu_uuids == set()


async def test_registration_waits_and_exposes_maintenance_stage(monkeypatch) -> None:
    database = SimpleNamespace(
        service_mode=AsyncMock(return_value=ServiceMode.ACTIVE), update_model_job=AsyncMock()
    )
    manager = RegistrationManager.__new__(RegistrationManager)
    manager.database = database
    manager.validator = SimpleNamespace(
        scheduler=SimpleNamespace(validation_requires_maintenance=True)
    )
    sleeping = asyncio.Event()
    proceed = asyncio.Event()

    async def wait(_):
        sleeping.set()
        await proceed.wait()

    monkeypatch.setattr("llm_rio.registration.asyncio.sleep", wait)
    task = asyncio.create_task(manager._wait_for_validation_window("job"))
    try:
        await asyncio.wait_for(sleeping.wait(), 1)
        assert not task.done()
        update = database.update_model_job.call_args.kwargs
        assert update["stage"] == "waiting_for_maintenance"
        assert update["job_state"] == "QUEUED"
        database.service_mode.return_value = ServiceMode.MAINTENANCE_READY
    finally:
        proceed.set()
    await asyncio.wait_for(task, 1)


async def test_normal_validation_does_not_request_a_post_verification_warm() -> None:
    scheduler = ResidencyScheduler(
        settings=Settings(),
        database=FakeDatabase(),
        inventory=MachineInventory("test", "driver", None, (), "topology", "fingerprint"),
        profiles=FakeProfiles(),
        supervisor=FakeSupervisor(),
    )
    scheduler.planner.prism_weight_cache_enabled = True
    await scheduler.warm_model_once("verified-model")
    assert scheduler._prism_one_time_warm_model_ids == set()


async def test_validation_preflight_defers_until_live_context_memory_is_released(
    monkeypatch,
) -> None:
    validator = ProfileValidator.__new__(ProfileValidator)
    monkeypatch.setattr(
        "llm_rio.validation.read_gpu_memory", lambda _: {"GPU-0": GpuMemory(100, 80)}
    )
    with pytest.raises(ValidationPreempted) as error:
        await validator._check_native_headroom(("GPU-0",), candidate_shape(1, (("GPU-0",),)))
    assert error.value.stage == "gpu_memory_wait"
    assert error.value.details["gpus"]["GPU-0"]["required_free_vram_mib"] == 90
    monkeypatch.setattr(
        "llm_rio.validation.read_gpu_memory", lambda _: {"GPU-0": GpuMemory(100, 95)}
    )
    await validator._check_native_headroom(("GPU-0",), candidate_shape(1, (("GPU-0",),)))


def test_native_planner_reaches_live_admission_when_sleeping_residuals_block_wake() -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    target_profile = replace(
        make_profile("target", "target", ("GPU-0",)),
        peak_vram_mib_per_gpu=(90,),
        wake_peak_vram_mib_per_gpu=(90,),
    )
    target = WorkerPlacement(
        "target", target_profile, ("GPU-0",), 18000, state=RuntimeState.SLEEPING
    )
    other = WorkerPlacement(
        "other",
        replace(target_profile, id="other", model_id="other", sleep_vram_mib_per_gpu=(20,)),
        ("GPU-0",),
        18001,
        state=RuntimeState.SLEEPING,
    )
    planner = GreedyPlacementPlanner(
        wait_duration_seconds=1,
        minimum_residency_seconds=0,
        fair_share_seconds=60,
        prism_enabled=True,
        kvcached_required=False,
        gpu_vram_mib={"GPU-0": 100},
        reserved_vram_mib=5,
    )
    actions = planner.plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0"},
        workers=[target, other],
        pressures=[QueuePressure("target", 1, 10, datetime.now(UTC))],
        profiles={"target": [target_profile]},
    )
    assert actions == [WakePlacement("target", "prism_ram_cache_hit")]


async def test_cancelled_gpu_acquisition_releases_validation_ownership() -> None:
    sleeper = SimpleNamespace(
        id="sleeper", state=RuntimeState.SLEEPING, gpu_uuids=("GPU-0",), admitted_request_ids=set()
    )
    scheduler, supervisor = normal_scheduler(sleeper)
    scheduler.database.service_mode = AsyncMock(return_value=ServiceMode.MAINTENANCE_READY)
    stopping = asyncio.Event()

    async def stop(*args, **kwargs):
        stopping.set()
        await asyncio.Event().wait()

    supervisor.stop = stop
    task = asyncio.create_task(scheduler.acquire_validation_gpus(("GPU-0",)))
    await asyncio.wait_for(stopping.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert scheduler.validation_gpu_uuids == ()


async def test_maintenance_changes_pending_sleep_to_full_unload() -> None:
    from llm_rio.workers import WorkerSupervisor

    database = FakeDatabase()
    database.execute = AsyncMock()
    settings = Settings(reserved_vram_mib=5)
    supervisor = WorkerSupervisor(settings, database)
    resident = WorkerPlacement(
        "resident",
        make_profile("resident", "model", ("GPU-0",)),
        ("GPU-0",),
        18000,
        state=RuntimeState.DRAINING,
    )
    resident.admitted_request_ids.add("inflight")
    supervisor.workers[resident.id] = resident
    supervisor._drain_to_sleep.add(resident.id)
    supervisor.enforce_host_cache_budget = AsyncMock(return_value=True)
    scheduler = ResidencyScheduler(
        settings=settings,
        database=database,
        inventory=MachineInventory("test", "driver", None, (), "topology", "fingerprint"),
        profiles=FakeProfiles(),
        supervisor=supervisor,
    )
    await scheduler.enter_maintenance()
    assert resident.id not in supervisor._drain_to_sleep
    assert resident.state is RuntimeState.DRAINING
    await scheduler._reconcile()
    assert database.mode is ServiceMode.DRAINING
    await supervisor.release(resident.id, "inflight", 10)
    assert resident.state is RuntimeState.COLD
    await scheduler._reconcile()
    assert database.mode is ServiceMode.MAINTENANCE_READY


@pytest.mark.parametrize("queue_mode", [False, True])
@pytest.mark.parametrize("failure", [False, True, "sampler"])
async def test_validation_probe_unloads_and_preserves_vram_measurements(
    tmp_path, monkeypatch, failure, queue_mode
) -> None:
    from llm_rio.domain import GpuDevice
    from llm_rio.profiles import profile_from_dict, profile_to_dict

    settings = Settings(
        log_dir=tmp_path, reserved_vram_mib=5, serving_mode="queue" if queue_mode else "vllm-sleep"
    )
    inventory = MachineInventory(
        "test", "driver", None, (GpuDevice("GPU-0", 0, "fake", 100),), "topology", "fingerprint"
    )
    validator = ProfileValidator(settings, inventory, SimpleNamespace())
    process = SimpleNamespace(pid=123)
    sampler = SimpleNamespace(
        start=lambda: None,
        stop=AsyncMock(side_effect=RuntimeError("probe failed") if failure == "sampler" else None),
        sample_now=lambda: (80,),
        peak=lambda: (90,),
        baseline_mib=(1,),
        baseline_drop_mib=(0,),
    )
    monkeypatch.setattr("llm_rio.validation._VramSampler", lambda *_: sampler)
    monkeypatch.setattr(
        "llm_rio.validation.asyncio.create_subprocess_exec", AsyncMock(return_value=process)
    )
    validator._wait_for_health = AsyncMock()
    validator._generation_contract = AsyncMock(
        return_value=100, side_effect=RuntimeError("probe failed") if failure is True else None
    )
    validator._sleep_wake_contract = AsyncMock(return_value=(0.1, 0.2, (5,), 10.0))
    validator._terminate = AsyncMock()
    kwargs = dict(
        model_id="model",
        model_revision="revision",
        model_path=tmp_path,
        nickname="model",
        candidate=candidate_shape(1, (("GPU-0",),)),
        gpu_set=("GPU-0",),
        backend="native",
        port=19000,
    )
    if failure:
        with pytest.raises(RuntimeError, match="probe failed"):
            await validator._probe_vllm_on_port(**kwargs)
    else:
        measured = await validator._probe_vllm_on_port(**kwargs)
        restored = profile_from_dict(profile_to_dict(measured))
        assert restored.peak_vram_mib_per_gpu == (90,)
        if queue_mode:
            assert restored.launch_args["enable_sleep_mode"] is False
            assert restored.wake_peak_vram_mib_per_gpu is None
            assert restored.sleep_vram_mib_per_gpu is None
            validator._sleep_wake_contract.assert_not_awaited()
        else:
            assert restored.wake_peak_vram_mib_per_gpu == (90,)
            assert restored.sleep_vram_mib_per_gpu == (5,)
        assert restored.vram_baseline_mib_per_gpu == (1,)
    validator._terminate.assert_awaited_once_with(process, gpu_uuids=("GPU-0",))


async def test_maintenance_api_reports_probes_and_blocks_resume(tmp_path) -> None:
    import httpx

    from llm_rio.api.app import create_app
    from llm_rio.api.dependencies import current_principal
    from llm_rio.domain import Role
    from llm_rio.security import Principal
    from llm_rio.workers import WorkerSupervisor

    database = FakeDatabase()
    database.mode = ServiceMode.MAINTENANCE_READY
    database.list_models = AsyncMock(return_value=[])
    settings = Settings()
    supervisor = WorkerSupervisor(settings, database)
    supervisor.host_cache_status = AsyncMock(return_value={})
    scheduler = ResidencyScheduler(
        settings=settings,
        database=database,
        inventory=MachineInventory("test", "driver", None, (), "topology", "fingerprint"),
        profiles=FakeProfiles(),
        supervisor=supervisor,
    )
    scheduler._validation_gpu_uuids.add("GPU-0")
    app = create_app(settings)
    app.state.database, app.state.scheduler, app.state.supervisor = database, scheduler, supervisor
    app.dependency_overrides[current_principal] = lambda: Principal(
        "key", "admin", Role.ADMIN, "account", True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        status = await client.get("/admin/maintenance")
        assert status.json()["validation"] == {"requires_maintenance": True, "gpu_uuids": ["GPU-0"]}
        response = await client.post("/admin/maintenance", json={"mode": "active"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "validation_in_progress"
    assert database.mode is ServiceMode.MAINTENANCE_READY
