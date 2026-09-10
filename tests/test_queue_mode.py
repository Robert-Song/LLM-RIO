import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_scheduler_contract import FakeDatabase, make_profile, make_worker, pressure
from test_validation_cleanup import candidate_shape

from llm_rio.config import EngineSettings, ServingMode, Settings
from llm_rio.domain import RuntimeState
from llm_rio.planner import DrainPlacement, GreedyPlacementPlanner, StartPlacement
from llm_rio.profiles import profile_from_dict, profile_to_dict, profile_verified_for_mode
from llm_rio.validation import ProfileValidator, ValidationError
from llm_rio.workers import WorkerSupervisor


def queue_profile(name="model", gpus=("GPU-0",)):
    return replace(
        make_profile(name, name, gpus),
        launch_args={"enable_sleep_mode": False},
        sleep_vram_mib_per_gpu=None,
        wake_peak_vram_mib_per_gpu=None,
    )


def queue_planner():
    return GreedyPlacementPlanner(
        wait_duration_seconds=5,
        minimum_residency_seconds=0,
        fair_share_seconds=7200,
        queue_mode=True,
    )


@pytest.mark.parametrize(
    "mode,backend,cache",
    [
        ("queue", "none", False),
        ("vllm-sleep", "none", True),
        ("kv-cached", "required", True),
    ],
)
def test_explicit_modes_override_legacy_flags(mode, backend, cache):
    settings = Settings(serving_mode=mode, engines=EngineSettings(kvcached_mode="required"))
    assert settings.effective_kvcached_mode == backend
    assert settings.ram_weight_cache_enabled is cache
    assert settings.queue_mode_enabled is (mode == "queue")


def test_default_and_legacy_experimental_mode_preserved(monkeypatch):
    assert Settings().effective_kvcached_mode == "none"
    assert Settings().ram_weight_cache_enabled
    assert (
        Settings(engines=EngineSettings(kvcached_mode="required")).effective_kvcached_mode
        == "required"
    )
    monkeypatch.setenv("LLMRIO_SERVING_MODE", "queue")
    assert Settings().serving_mode is ServingMode.QUEUE


def test_queue_profile_requires_measurement_without_sleep():
    profile = queue_profile()
    assert profile_verified_for_mode(
        profile_from_dict(profile_to_dict(profile)),
        kvcached_required=False,
        queue_mode_required=True,
    )
    assert not profile_verified_for_mode(
        make_profile("old", "old", ("GPU-0",)), kvcached_required=False, queue_mode_required=True
    )
    assert not profile_verified_for_mode(
        profile, kvcached_required=False, ram_weight_cache_required=True
    )


def test_queue_commands_do_not_enable_sleep_or_kvcached(monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")
    monkeypatch.setenv("ENABLE_KVCACHED", "true")
    supervisor = WorkerSupervisor(Settings(serving_mode="queue"), FakeDatabase())
    # Even direct command generation cannot retain a stray sleep flag.
    worker = make_worker("w", replace(queue_profile(), launch_args={"enable_sleep_mode": True}))
    command = supervisor._command(worker, "/model", "model")
    assert "--enable-sleep-mode" not in command
    assert not supervisor.ram_weight_cache_enabled
    assert not supervisor.kvcached.enabled
    environment = supervisor._environment(worker)
    assert environment["VLLM_SERVER_DEV_MODE"] == "0"
    assert environment["ENABLE_KVCACHED"] == "false"


@pytest.mark.parametrize("gpu_count", [1, 2])
async def test_queue_validation_tries_high_utilization_and_small_memory_backoff(
    tmp_path, gpu_count
):
    validator = ProfileValidator.__new__(ProfileValidator)
    validator.settings = Settings(serving_mode="queue")
    validator.scheduler = SimpleNamespace(
        validation_requires_maintenance=True, database=SimpleNamespace(record_event=AsyncMock())
    )
    validator._check_native_headroom = AsyncMock()
    validator._reserve_validation_port = AsyncMock(return_value=19000)
    validator._release_validation_port = AsyncMock()
    gpus = tuple(f"GPU-{i}" for i in range(gpu_count))
    candidate = replace(candidate_shape(gpu_count, (gpus,)), gpu_memory_utilization=0.96)
    log = tmp_path / "engine.log"
    log.write_text("CUDA out of memory")
    attempts = []

    async def probe(**kwargs):
        attempt = kwargs["candidate"]
        attempts.append(attempt.gpu_memory_utilization)
        if len(attempts) == 1:
            raise ValidationError("engine_startup", "OOM", {"log_path": str(log)})
        assert attempt.gpu_memory_utilization > 0.8
        return replace(
            queue_profile(gpus=gpus), gpu_memory_utilization=attempt.gpu_memory_utilization
        )

    validator._probe_vllm_on_port = probe
    profile = await validator._probe_vllm(
        model_id="model",
        model_revision="r",
        model_path=tmp_path,
        nickname="model",
        candidate=candidate,
        gpu_set=gpus,
        backend="native",
    )
    assert attempts == [0.96, 0.94]
    assert profile.gpu_memory_utilization == 0.94
    assert validator._release_validation_port.await_count == 2


@pytest.mark.parametrize(
    "state", [RuntimeState.LOADING, RuntimeState.DRAINING, RuntimeState.STOPPING]
)
def test_queue_keeps_transitioning_gpus_owned(state):
    worker = make_worker("old", queue_profile("old"))
    worker.state = state
    assert (
        queue_planner().plan(
            now=datetime.now(UTC),
            all_gpu_uuids={"GPU-0"},
            workers=[worker],
            pressures=[pressure("new", 10)],
            profiles={"new": [queue_profile("new")]},
        )
        == []
    )


def test_queue_drains_busy_model_for_older_backlog_without_interrupting_requests():
    worker = make_worker("old", queue_profile("old"))
    worker.admitted_request_ids.add("inflight")
    actions = queue_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0"},
        workers=[worker],
        pressures=[pressure("old", 1), pressure("new", 10)],
        profiles={"old": [worker.profile], "new": [queue_profile("new")]},
    )
    assert actions == [DrainPlacement("old", "incompatible_backlog")]
    assert worker.admitted_request_ids == {"inflight"}


def test_queue_keeps_older_resident_backlog_before_newer_model():
    worker = make_worker("old", queue_profile("old"))
    actions = queue_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0"},
        workers=[worker],
        pressures=[pressure("old", 20), pressure("new", 10)],
        profiles={"old": [worker.profile], "new": [queue_profile("new")]},
    )
    assert actions == []


def test_queue_tp2_waits_for_both_workers_to_fully_unload():
    workers = [make_worker(f"w{i}", queue_profile(f"old{i}", (f"GPU-{i}",))) for i in range(2)]
    profile = queue_profile("new", ("GPU-0", "GPU-1"))
    args = dict(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=workers,
        pressures=[pressure("new", 10)],
        profiles={"new": [profile]},
    )
    assert {a.worker_id for a in queue_planner().plan(**args)} == {"w0", "w1"}
    workers[0].state = RuntimeState.COLD
    workers[1].state = RuntimeState.STOPPING
    assert queue_planner().plan(**args) == []
    workers[1].state = RuntimeState.COLD
    assert queue_planner().plan(**args) == [
        StartPlacement(profile, ("GPU-0", "GPU-1"), "cold_backlog")
    ]


def test_queue_retains_duplicate_placement_and_replica_scaling():
    profiles = [queue_profile("new", (f"GPU-{i}",)) for i in range(2)]
    args = dict(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=[],
        pressures=[pressure("new", 10, tokens=100000)],
        profiles={"new": profiles},
    )
    actions = queue_planner().plan(**args)
    assert len(actions) == 2
    assert all(isinstance(a, StartPlacement) for a in actions)
    worker = make_worker("w", profiles[0])
    args["workers"] = [worker]
    actions = queue_planner().plan(**args)
    assert actions == [StartPlacement(profiles[1], ("GPU-1",), "replica_backlog")]


async def test_queue_drain_finishes_admitted_work_before_cold_stop(monkeypatch):
    supervisor = WorkerSupervisor(Settings(serving_mode="queue"), FakeDatabase())
    worker = make_worker("w", queue_profile())
    worker.admitted_request_ids.add("r")
    supervisor.workers[worker.id] = worker
    supervisor._persist = AsyncMock()
    supervisor.stop = AsyncMock()
    supervisor._offload = AsyncMock()
    await supervisor.drain(worker.id)
    assert worker.state is RuntimeState.DRAINING
    supervisor.stop.assert_not_awaited()
    await supervisor.release(worker.id, "r", 1)
    supervisor.stop.assert_awaited_once_with(worker.id, force=False)
    supervisor._offload.assert_not_awaited()


def test_queue_does_not_fill_partial_tp_group_with_younger_work():
    stopped = make_worker("old", queue_profile("old", ("GPU-1",)))
    stopped.state = RuntimeState.STOPPING
    actions = queue_planner().plan(
        now=datetime.now(UTC),
        all_gpu_uuids={"GPU-0", "GPU-1"},
        workers=[stopped],
        pressures=[pressure("large", 20), pressure("small", 1)],
        profiles={
            "large": [queue_profile("large", ("GPU-0", "GPU-1"))],
            "small": [queue_profile("small", ("GPU-0",))],
        },
    )
    assert actions == []


def test_queue_routability_and_revalidation_script_agree():
    from fire_all_native_revalidations import is_valid_native_v2
    from llm_rio.api.routes_inference import _routable_profiles

    profile = queue_profile()
    raw = json.loads(json.dumps({**profile_to_dict(profile), "active": True}))
    assert is_valid_native_v2(raw, queue_mode=True)
    assert not is_valid_native_v2(raw)
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                scheduler=SimpleNamespace(
                    settings=Settings(serving_mode="queue"),
                    kvcached=SimpleNamespace(enabled=False),
                    planner=SimpleNamespace(prism_weight_cache_enabled=False),
                )
            )
        )
    )
    assert _routable_profiles(request, [profile, make_profile("sleep", "sleep", ("GPU-0",))]) == [
        profile
    ]


def test_cli_mode_override_reaches_server_without_config_changes(monkeypatch):
    from unittest.mock import Mock

    from typer.testing import CliRunner

    import llm_rio.cli as cli

    run = Mock()
    monkeypatch.setattr(cli, "create_app", lambda settings: settings)
    monkeypatch.setattr(cli.uvicorn, "run", run)
    result = CliRunner().invoke(cli.app, ["serve", "--mode", "queue"])
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].serving_mode is ServingMode.QUEUE


async def test_queue_registration_advances_to_tp2_after_tp1_failure(tmp_path, monkeypatch):
    import asyncio

    from llm_rio.registration import RegistrationManager

    candidates = [
        replace(
            candidate_shape(n, (tuple(f"GPU-{i}" for i in range(n)),)), gpu_memory_utilization=0.96
        )
        for n in (1, 2)
    ]
    monkeypatch.setattr("llm_rio.registration.build_candidate_shapes", lambda **_: candidates)
    manager = RegistrationManager.__new__(RegistrationManager)
    manager.settings = Settings(serving_mode="queue")
    manager.inventory = SimpleNamespace()
    manager.database = SimpleNamespace(update_model_job=AsyncMock())
    manager._validation_lock = asyncio.Lock()
    manager._wait_for_validation_window = AsyncMock()
    profile = queue_profile(gpus=("GPU-0", "GPU-1"))
    manager.validator = SimpleNamespace(
        validate_vllm=AsyncMock(
            side_effect=[ValidationError("engine_startup", "TP1 OOM"), [profile]]
        )
    )
    profiles = await manager._validate_with_requeue(
        job_id="job",
        job={"model_id": "model", "nickname": "model"},
        artifact_path=tmp_path,
        resolved_revision="r",
        inspection={
            "max_model_len": 4096,
            "weight_bytes": 1,
            "dtype": "auto",
            "quantization": None,
        },
    )
    assert profiles == [profile]
    attempts = manager.validator.validate_vllm.call_args_list
    assert [a.kwargs["candidate"].tensor_parallel_size for a in attempts] == [1, 2]
    assert all(a.kwargs["candidate"].gpu_memory_utilization == 0.96 for a in attempts)
