from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from llm_rio.domain import Role
from llm_rio.inventory import discover_inventory
from llm_rio.profiles import (
    ProfileRepository,
    profile_key,
    profile_to_dict,
)
from llm_rio.storage import Database, _now
from tests.release_fixtures import replace
from tests.test_profile_admin_contract import make_profile


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
        measurements_valid=True,
        serving_mode="queue",
        launch_args={"enable_sleep_mode": False},
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


async def test_fingerprint_round_trip_preserves_activation(saved_models):
    await saved_models.execute("UPDATE model_profiles SET active = 1")
    await saved_models.set_machine_fingerprint("old")
    await saved_models.set_machine_fingerprint("temporary")
    assert await ProfileRepository(saved_models, "temporary").for_model("model") == []
    await saved_models.set_machine_fingerprint("old")
    assert len(await ProfileRepository(saved_models, "old").for_model("model")) == 1


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
