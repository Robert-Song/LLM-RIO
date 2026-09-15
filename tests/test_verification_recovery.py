from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from test_api_lifecycle import settings
from test_profile_admin_contract import inventory, make_profile

from llm_rio.api.app import create_app
from llm_rio.api.dependencies import current_principal
from llm_rio.domain import Role
from llm_rio.inventory import discover_inventory
from llm_rio.profiles import (
    ProfileRepository,
    profile_key,
    profile_to_dict,
    profile_verified_for_mode,
)
from llm_rio.security import Principal
from llm_rio.storage import Database, _now


@pytest.fixture
async def saved_models(tmp_path):
    database = Database(tmp_path / "test.db")
    await database.open()
    await database.create_key(
        key_id="admin",
        nickname="admin",
        role=Role.ADMIN,
        account_id="account",
        account_nickname="account",
        prefix="rio_admin_prefix_12345678",
        api_key="rio_admin_prefix_12345678_secret",
        limit_tokens=0,
        unlimited=True,
    )
    for name, state in [("model", "AVAILABLE"), ("missing", "AVAILABLE"), ("disabled", "DISABLED")]:
        await database.execute(
            """INSERT INTO model_catalog
               (id, nickname, huggingface_repo, resolved_revision, state, artifact_path,
                created_by_key_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name,
                name,
                "org/model",
                "immutable-revision",
                state,
                str(tmp_path),
                "admin",
                _now(),
                _now(),
            ),
        )
    profile = replace(
        make_profile("old-profile", "model", ("GPU-0",)),
        machine_fingerprint="old",
        memory_backend="native",
        normal_verified=False,
        kvcached_verified=False,
    )
    raw = profile_to_dict(profile)
    await database.execute(
        """INSERT INTO model_profiles
           (id, model_id, machine_fingerprint, profile_key, profile_json, verified_at, active)
           VALUES (?, ?, ?, ?, ?, ?, 0)""",
        (profile.id, profile.model_id, "old", profile_key(raw), json.dumps(raw), _now()),
    )
    try:
        yield database
    finally:
        await database.close()


async def test_trust_recovers_inference_profiles_and_is_idempotent(saved_models):
    repo = ProfileRepository(saved_models, "current")
    assert await repo.for_model("model") == []
    result = await repo.trust_available_models(gpu_uuids={"GPU-0"}, backend="native")
    assert [item["model_id"] for item in result["data"]] == ["model"]
    assert [item["model_id"] for item in result["skipped"]] == ["missing"]
    profiles = await repo.for_model("model")
    assert len(profiles) == 1
    assert profile_verified_for_mode(profiles[0], kvcached_required=False)
    assert not profiles[0].kvcached_verified
    assert profiles[0].peak_vram_mib_per_gpu == (2,)
    assert profiles[0].machine_fingerprint == "current"
    original = await saved_models.fetchone(
        "SELECT * FROM model_profiles WHERE id = ?", ("old-profile",)
    )
    assert original["active"] == 0
    assert original["machine_fingerprint"] == "old"
    assert not json.loads(original["profile_json"])["normal_verified"]
    await repo.trust_available_models(gpu_uuids={"GPU-0"}, backend="native")
    assert [p.id for p in await repo.for_model("model")] == [p.id for p in profiles]
    events = await saved_models.fetchall(
        "SELECT * FROM runtime_events WHERE event_type = 'MODEL_VERIFICATION_TRUSTED_BY_ADMIN'"
    )
    assert len(events) == 2
    assert json.loads(events[0]["payload_json"])["source_fingerprint"] == "old"


@pytest.mark.parametrize(
    "change", ["gpu", "revision", "artifact", "invalidated", "malformed", "legacy", "backend"]
)
async def test_incompatible_saved_profiles_are_reported_without_changes(saved_models, change):
    gpu_uuids = {"different-gpu"} if change == "gpu" else {"GPU-0"}
    if change == "revision":
        await saved_models.execute("UPDATE model_catalog SET resolved_revision = 'new'")
    elif change == "artifact":
        await saved_models.execute(
            "UPDATE model_catalog SET artifact_path = '/missing-rio-artifacts'"
        )
    elif change == "invalidated":
        await saved_models.execute(
            "UPDATE model_profiles SET profile_json = "
            "json_set(profile_json, '$.measurements_invalidated_at', 'today')"
        )
    elif change == "legacy":
        await saved_models.execute(
            "UPDATE model_profiles SET profile_json = "
            "json_set(profile_json, '$.vram_measurement_version', 1)"
        )
    elif change == "backend":
        await saved_models.execute(
            "UPDATE model_profiles SET profile_json = "
            "json_set(profile_json, '$.memory_backend', 'unknown')"
        )
    elif change == "malformed":
        await saved_models.execute("UPDATE model_profiles SET profile_json = '{}'")
    repo = ProfileRepository(saved_models, "current")
    result = await repo.trust_available_models(gpu_uuids=gpu_uuids, backend="both")
    assert result["data"] == []
    assert len(result["skipped"]) == 2
    assert await repo.for_model("model") == []


async def test_fingerprint_round_trip_preserves_activation(saved_models):
    await saved_models.execute("UPDATE model_profiles SET active = 1")
    await saved_models.set_machine_fingerprint("old")
    await saved_models.set_machine_fingerprint("temporary")
    assert await ProfileRepository(saved_models, "temporary").for_model("model") == []
    await saved_models.set_machine_fingerprint("old")
    assert len(await ProfileRepository(saved_models, "old").for_model("model")) == 1


@pytest.mark.parametrize("bulk", [False, True])
async def test_trust_endpoints_enforce_admin_and_recover_profiles(saved_models, tmp_path, bulk):
    app = create_app(settings(tmp_path))
    app.state.database = saved_models
    app.state.profiles = ProfileRepository(saved_models, "current")
    app.state.inventory = inventory()
    app.state.scheduler = SimpleNamespace(kvcached=SimpleNamespace(enabled=False))
    role = Role.USER
    app.dependency_overrides[current_principal] = lambda: Principal(
        "admin", "admin", role, "account", True
    )
    path = "/admin/models/trust-available" if bulk else "/admin/models/model/trust-verification"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.post(path, json={})).status_code == 403
        role = Role.ADMIN
        assert (await client.post(path, json={"backend": "invalid"})).status_code == 422
        response = await client.post(path, json={})
        assert response.status_code == 200, response.text
        assert len(await app.state.profiles.for_model("model")) == 1


def test_inventory_ignores_order_indices_topology_and_platform(monkeypatch):
    devices = ["GPU-A", "GPU-B"]
    driver = ["570.0"]
    nvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        NVMLError=RuntimeError,
        nvmlSystemGetDriverVersion=lambda: driver[0],
        nvmlSystemGetCudaDriverVersion_v2=lambda: 12080,
        nvmlDeviceGetCount=lambda: len(devices),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUUID=lambda i: devices[i],
        nvmlDeviceGetMemoryInfo=lambda i: SimpleNamespace(total=1024**3),
        nvmlDeviceGetCudaComputeCapability=lambda i: (9, 0),
        nvmlDeviceGetPciInfo=lambda i: SimpleNamespace(busId=f"bus-{i}"),
        nvmlDeviceGetName=lambda i: "test-gpu",
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("llm_rio.inventory._read_topology", lambda: ({}, "initial"))
    first = discover_inventory("machine", [])
    devices.reverse()
    monkeypatch.setattr("llm_rio.inventory._read_topology", lambda: ({}, "unavailable"))
    monkeypatch.setattr("platform.platform", lambda: "new-kernel")
    monkeypatch.setattr("platform.processor", lambda: "changed-cpu-string")
    assert discover_inventory("machine", []).fingerprint == first.fingerprint
    driver[0] = "580.0"
    assert discover_inventory("machine", []).fingerprint != first.fingerprint
    driver[0] = "570.0"
    devices[0] = "GPU-C"
    assert discover_inventory("machine", []).fingerprint != first.fingerprint
    devices[:] = ["GPU-A", "GPU-B"]
    assert discover_inventory("machine", ["GPU-A"]).fingerprint != first.fingerprint


@pytest.mark.parametrize("bulk", [False, True])
async def test_trust_respects_running_queue_mode(saved_models, tmp_path, bulk):
    app = create_app(settings(tmp_path))
    app.state.profiles = ProfileRepository(saved_models, "current")
    app.state.inventory = inventory()
    app.state.scheduler = SimpleNamespace(
        kvcached=SimpleNamespace(enabled=False), settings=SimpleNamespace(queue_mode_enabled=True)
    )
    app.dependency_overrides[current_principal] = lambda: Principal(
        "admin", "admin", Role.ADMIN, "account", True
    )
    path = "/admin/models/trust-available" if bulk else "/admin/models/model/trust-verification"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(path, json={})
        if bulk:
            assert response.status_code == 200
            assert response.json()["data"] == []
            assert len(response.json()["skipped"]) == 2
        else:
            assert response.status_code == 409
        assert await app.state.profiles.for_model("model") == []
        await saved_models.execute(
            "UPDATE model_profiles SET profile_json = "
            "json_set(profile_json, '$.launch_args.enable_sleep_mode', json('false'))"
        )
        assert (await client.post(path, json={})).status_code == 200
        assert len(await app.state.profiles.for_model("model")) == 1
