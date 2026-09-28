"""Release contracts that cross persistence, capability, and HTTP boundaries."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from llm_rio.api.app import create_app
from llm_rio.api.dependencies import current_principal
from llm_rio.artifacts import local_manifest
from llm_rio.config import Settings
from llm_rio.domain import Role
from llm_rio.engines.identity import launch_binding
from llm_rio.modes.factory import create_mode
from llm_rio.ports import PortAllocator
from llm_rio.profiles import ProfileRepository
from llm_rio.security import Principal
from llm_rio.workers import WorkerSupervisor
from scripts.migrate_release_data import _seed_revalidation_jobs
from tests.test_profile_admin_contract import inventory

pytest_plugins = ["tests.test_verification_recovery"]


async def test_imported_model_uses_existing_full_parameter_retry(saved_models, tmp_path):
    with sqlite3.connect(saved_models.path) as connection:
        assert _seed_revalidation_jobs(connection) == 2
        assert _seed_revalidation_jobs(connection) == 0
    row = await saved_models.fetchone("SELECT id FROM model_jobs WHERE model_id='model'")
    assert row is not None
    assert (
        await saved_models.fetchone("SELECT id FROM model_jobs WHERE model_id='disabled'") is None
    )
    settings = Settings(serving_mode="vllm-sleep", config_file=tmp_path / "absent.toml")
    app = create_app(settings)
    started: list[str] = []
    app.state.database = saved_models
    app.state.scheduler = SimpleNamespace(mode=create_mode(settings, inventory()))
    app.state.registration = SimpleNamespace(start=started.append)
    app.dependency_overrides[current_principal] = lambda: Principal(
        "admin-id", "admin", Role.ADMIN, "account", True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            f"/staff/model-jobs/{row['id']}/retry",
            json={
                "validation_overrides": {
                    "tensor_parallel_size": 1,
                    "max_model_len": 131072,
                    "max_num_seqs": 4,
                    "max_num_batched_tokens": 8192,
                    "gpu_memory_utilization": 0.83,
                }
            },
        )
    assert response.status_code == 202, response.text
    job = await saved_models.get_model_job(response.json()["job_id"])
    assert job is not None
    assert job["model_id"] == "model"
    assert job["state"] == "QUEUED"
    assert job["validation_overrides"] == {
        "tensor_parallel_size": 1,
        "max_model_len": 131072,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.83,
    }
    assert started == [job["id"]]
    assert await saved_models.key_count() == 1
    assert len(await saved_models.list_models()) == 3


async def test_saved_profile_is_reachable_and_trust_route_is_audited(saved_models, tmp_path):
    settings = Settings(serving_mode="queue", config_file=tmp_path / "absent.toml")
    repository = ProfileRepository(saved_models, "current")
    app = create_app(settings)
    app.state.database = saved_models
    app.state.profiles = repository
    app.state.inventory = inventory()
    app.state.scheduler = SimpleNamespace(mode=create_mode(settings, inventory()))
    app.dependency_overrides[current_principal] = lambda: Principal(
        "admin-id", "admin", Role.ADMIN, "account", True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        listed = await client.get("/admin/models/model/profiles")
        assert listed.status_code == 200, listed.text
        assert [p["id"] for p in listed.json()["saved_measurements"]] == ["old-profile"]
        assert listed.json()["data"] == []
        trusted = await client.post(
            "/admin/models/model/profiles/old-profile/trust",
            json={"reason": "Same artifact and GPU UUID"},
        )
        assert trusted.status_code == 200, trusted.text
        assert trusted.json()["actor"] == "admin-id"
    assert len(await repository.for_model("model")) == 1
    events = await saved_models.fetchall(
        "SELECT payload_json FROM runtime_events WHERE event_type='PROFILE_MEASUREMENTS_TRUSTED'"
    )
    assert json.loads(events[0]["payload_json"])["source_profile_id"] == "old-profile"


async def test_trust_route_requires_admin(saved_models, tmp_path):
    settings = Settings(serving_mode="queue", config_file=tmp_path / "absent.toml")
    app = create_app(settings)
    app.state.database = saved_models
    app.dependency_overrides[current_principal] = lambda: Principal(
        "user-id", "user", Role.USER, "account", True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/admin/models/model/profiles/old-profile/trust", json={"reason": "unauthorized"}
        )
    assert response.status_code == 403


async def test_cloned_local_model_keeps_source_manifest(saved_models, tmp_path):
    source = tmp_path / "local-model"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"test"}')
    manifest = local_manifest(source)
    await saved_models.execute(
        "UPDATE model_catalog SET source_type='local', local_path=?, artifact_path=?, "
        "resolved_revision=?, artifact_hashes_json=?, engine='vllm' WHERE id='model'",
        (str(source), str(source), manifest["revision"], json.dumps(manifest["artifact_hashes"])),
    )
    await saved_models.execute(
        "UPDATE model_profiles SET active=1, "
        "profile_json=json_set(profile_json, '$.model_revision', ?) WHERE id='old-profile'",
        (manifest["revision"],),
    )
    repository = ProfileRepository(saved_models, "old")
    model = await saved_models.model_by_id("model")
    assert model is not None
    cloned, _ = await repository.clone_model(
        source_model=model,
        nickname="clone",
        creator_key_id="admin",
        request_defaults={},
        max_model_len=None,
        yarn_factor=None,
        yarn_original_max_model_len=None,
        inherit_grants=False,
    )
    assert cloned["source_type"] == "local"
    assert cloned["local_path"] == str(source)
    assert cloned["engine"] == "vllm"
    assert cloned["artifact_hashes"] == manifest["artifact_hashes"]


def test_launch_binding_detects_engine_and_environment_changes(tmp_path):
    from tests.release_fixtures import replace as measured_replace
    from tests.test_profile_admin_contract import make_profile

    executable = tmp_path / "vllm"
    executable.write_bytes(b"#!/bin/sh\necho engine-one\n")
    executable.chmod(0o755)
    settings = Settings(serving_mode="queue", engines={"vllm_executable": str(executable)})
    profile = measured_replace(
        make_profile("profile", "model", ("GPU-0",)),
        memory_backend="native",
        serving_mode="queue",
        launch_args={"enable_sleep_mode": False},
    )
    profile = replace(profile, launch_binding=launch_binding(settings, profile, profile.engine))
    mode = create_mode(settings, inventory())
    assert mode.eligibility(profile).allowed
    executable.write_bytes(b"#!/bin/sh\necho engine-two\n")
    assert mode.eligibility(profile).reason == "launch_configuration_changed"
    executable.write_bytes(b"#!/bin/sh\necho engine-one\n")
    executable.chmod(0o755)
    settings.engines.environment["VLLM_ATTENTION_BACKEND"] = "other"
    assert mode.eligibility(profile).reason == "launch_configuration_changed"


def test_port_allocator_retains_uncertain_owner():
    ports = PortAllocator(19000, 19000)
    port = ports.reserve("validation")
    with pytest.raises(RuntimeError, match="exhausted"):
        ports.reserve("serving")
    with pytest.raises(RuntimeError, match="owner"):
        ports.release(port, "serving")
    assert ports.snapshot() == {19000: "validation"}
    ports.release(port, "validation")
    assert ports.reserve("serving") == port


def test_worker_admission_uses_the_bound_mode_policy():
    from llm_rio.modes.contracts import ProfileEligibility
    from tests.test_profile_admin_contract import make_profile

    supervisor = WorkerSupervisor(Settings(serving_mode="queue"), SimpleNamespace())
    denied = SimpleNamespace(
        eligibility=lambda _profile: ProfileEligibility(False, "mode_mismatch")
    )
    supervisor.bind_mode(denied)
    assert not supervisor._profile_allowed(make_profile("profile", "model", ("GPU-0",)))
    with pytest.raises(RuntimeError, match="already bound"):
        supervisor.bind_mode(denied)


def test_native_selection_does_not_activate_experimental_modules(tmp_path):
    import subprocess
    import sys

    script = (
        "import sys; from llm_rio.config import Settings; "
        "from llm_rio.modes.factory import create_mode; "
        "from llm_rio.domain import MachineInventory; "
        "create_mode(Settings(serving_mode='queue', config_file='absent.toml'), "
        "MachineInventory('x','x',None,(),'x','x')); "
        "assert 'llm_rio.modes.kv_cached.runtime' not in sys.modules; "
        "assert 'llm_rio.modes.kv_cached.kvcached_vllm_compat' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


@pytest.mark.parametrize("gpu_count", [1, 2, 4, 8])
def test_queue_replica_and_tp_reservations_scale_with_inventory(gpu_count):
    from datetime import UTC, datetime, timedelta

    from llm_rio.modes.actions import QueuePressure, StartPlacement
    from llm_rio.modes.queue.planner import QueuePlanner
    from tests.test_scheduler_contract import make_profile

    uuids = tuple(f"GPU-{index}" for index in range(gpu_count))
    profile = replace(
        make_profile("replica-profile", "model", (uuids[0],)),
        eligible_gpu_sets=tuple((gpu,) for gpu in uuids),
    )
    planner = QueuePlanner(
        wait_duration_seconds=5,
        minimum_residency_seconds=0,
    )
    now = datetime.now(UTC)
    demand = QueuePressure("model", 32, 1000, now - timedelta(seconds=10))
    actions = planner.plan(
        now=now,
        all_gpu_uuids=set(uuids),
        workers=[],
        pressures=[demand],
        profiles={"model": [profile]},
    )
    assert len(actions) == gpu_count
    assert all(isinstance(action, StartPlacement) for action in actions)
    assert {action.gpu_uuids for action in actions if isinstance(action, StartPlacement)} == {
        (gpu,) for gpu in uuids
    }

    tp_profile = replace(
        profile,
        id="tp-profile",
        gpu_count=gpu_count,
        tensor_parallel_size=gpu_count,
        eligible_gpu_sets=(uuids,),
        idle_vram_mib_per_gpu=(1,) * gpu_count,
        peak_vram_mib_per_gpu=(2,) * gpu_count,
    )
    tp_actions = planner.plan(
        now=now,
        all_gpu_uuids=set(uuids),
        workers=[],
        pressures=[demand],
        profiles={"model": [tp_profile]},
    )
    assert len(tp_actions) == 1
    assert isinstance(tp_actions[0], StartPlacement)
    assert tp_actions[0].gpu_uuids == uuids
