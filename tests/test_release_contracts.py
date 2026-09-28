from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_rio.artifacts import local_artifact_unchanged, local_manifest
from llm_rio.config import Settings
from llm_rio.domain import Engine
from llm_rio.engines.launch import adapter
from llm_rio.errors import RioError
from llm_rio.modes.factory import create_mode
from llm_rio.operations.archive import archive
from llm_rio.profiles import ProfileRepository
from llm_rio.services.profile_trust import trust_measurements
from tests.test_profile_admin_contract import inventory
from tests.test_validation_cleanup import candidate_shape

pytest_plugins = ["tests.test_verification_recovery"]


def test_explicit_mode_required_and_legacy_settings_rejected(monkeypatch):
    monkeypatch.delenv("LLMRIO_SERVING_MODE")
    with pytest.raises(ValidationError, match="serving_mode"):
        Settings()
    with pytest.raises(ValidationError, match="Extra inputs"):
        Settings(serving_mode="queue", prism_weight_cache_mode="ram")


def test_inactive_mode_settings_do_not_affect_queue(monkeypatch):
    monkeypatch.delenv("LLMRIO_SERVING_MODE", raising=False)
    settings = Settings(
        serving_mode="queue",
        modes={"vllm_sleep": {"idle_sleep_seconds": 10, "preload_models": ["*"]}},
    )
    assert settings.modes.vllm_sleep.idle_sleep_seconds == 10
    assert settings.residency.preload_models == []
    assert settings.residency.idle_sleep_seconds != 10
    assert not settings.ram_weight_cache_enabled


def test_incompatible_active_engine_settings_are_rejected(monkeypatch):
    monkeypatch.delenv("LLMRIO_SERVING_MODE", raising=False)
    with pytest.raises(ValidationError, match="queue-only"):
        Settings(serving_mode="vllm-sleep", engines={"enable_llama_cpp": True})


def test_local_artifact_content_identity_and_change_detection(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")
    resolved = local_manifest(model)
    assert resolved["revision"].startswith("local:")
    assert local_artifact_unchanged(model, resolved["artifact_hashes"])
    weights.write_bytes(b"changed weights")
    assert not local_artifact_unchanged(model, resolved["artifact_hashes"])
    assert local_manifest(model)["revision"] != resolved["revision"]


def test_local_gguf_launch_is_shared_and_queue_only(tmp_path):
    shape = candidate_shape(1, (("GPU-0",),))
    settings = Settings(serving_mode="queue", engines={"enable_llama_cpp": True})
    args = dict(
        settings=settings,
        shape=shape,
        artifact=tmp_path / "model.gguf",
        nickname="model",
        gpu_uuids=("GPU-0",),
        port=18000,
        api_key="private",
    )
    spec = adapter(Engine.LLAMA_CPP).launch(**args)
    assert spec.command.count("--model") == 1
    assert spec.command.count("--n-gpu-layers") == 1
    assert "--enable-sleep-mode" not in spec.command
    with pytest.raises(ValueError, match="queue-only"):
        Settings(serving_mode="vllm-sleep", engines={"enable_llama_cpp": True})


async def test_advanced_trust_preserves_evidence_and_audits_actor(saved_models):
    repository = ProfileRepository(saved_models, "current")
    mode = create_mode(Settings(serving_mode="queue"), inventory())
    result = await trust_measurements(
        repository,
        model_id="model",
        profile_id="old-profile",
        mode=mode,
        gpu_uuids={"GPU-0"},
        reason="Driver qualification completed",
        actor="admin",
    )
    assert result["source_profile_id"] == "old-profile"
    assert result["actor"] == "admin"
    original = await saved_models.fetchone(
        "SELECT machine_fingerprint FROM model_profiles WHERE id='old-profile'"
    )
    assert original[0] == "old"
    profiles = await repository.for_model("model")
    assert len(profiles) == 1 and profiles[0].peak_vram_mib_per_gpu == (2,)


@pytest.mark.parametrize(
    "change", ["gpu", "mode", "revision", "invalidated", "measurements", "reason"]
)
async def test_advanced_trust_cannot_manufacture_eligibility(saved_models, change):
    repository = ProfileRepository(saved_models, "current")
    mode = create_mode(
        Settings(serving_mode="vllm-sleep" if change == "mode" else "queue"), inventory()
    )
    if change == "revision":
        await saved_models.execute("UPDATE model_catalog SET resolved_revision='changed'")
    if change in ("invalidated", "measurements"):
        field = (
            "measurements_invalidated_at" if change == "invalidated" else "vram_measurement_version"
        )
        await saved_models.execute(
            "UPDATE model_profiles SET profile_json=json_set(profile_json, ?, ?)",
            (f"$.{field}", "changed" if change == "invalidated" else 0),
        )
    with pytest.raises(RioError):
        await trust_measurements(
            repository,
            model_id="model",
            profile_id="old-profile",
            mode=mode,
            gpu_uuids={"different"} if change == "gpu" else {"GPU-0"},
            reason=" " if change == "reason" else "qualified",
            actor="admin",
        )
    assert await repository.for_model("model") == []


def test_archive_preserves_source_and_exports_only_catalog(tmp_path, monkeypatch):
    source = tmp_path / "beta.db"
    with sqlite3.connect(source) as db:
        db.execute(
            "CREATE TABLE model_catalog(nickname, huggingface_repo, "
            "resolved_revision, artifact_path)"
        )
        db.execute("INSERT INTO model_catalog VALUES ('model','org/model','rev','/models/model')")
    vault = tmp_path / ".beta-api-key-vault"
    vault.write_bytes(b"vault")
    original = source.read_bytes()
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    destination = tmp_path / "archive"
    config = tmp_path / "config.toml"
    config.write_text('serving_mode = "queue"\n')
    result = archive(source, destination, config)
    assert result["models"] == 1
    assert source.read_bytes() == original
    assert (destination / vault.name).read_bytes() == b"vault"
    assert (
        json.loads((destination / "catalog.json").read_text())[0]["artifact_path"]
        == "/models/model"
    )
    with pytest.raises(FileExistsError):
        archive(source, destination, config)
