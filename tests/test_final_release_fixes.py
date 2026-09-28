"""Audited failures: real persistence/process locks, synthetic inference engines only."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.fernet import Fernet, InvalidToken

from llm_rio.operations.archive import archive
from llm_rio.operations.ownership import database_resource, owner_lock
from llm_rio.profiles import profile_verified_for_mode
from llm_rio.services.validation_jobs import queue_validation
from scripts.audit_native_release import profile, run, settings

pytest_plugins = ["tests.test_verification_recovery"]


async def test_all_six_original_audit_failures_are_closed(tmp_path):
    findings = await run(tmp_path)
    assert {item["id"] for item in findings} == {f"AUD-0{i}" for i in range(1, 7)}
    assert not [item for item in findings if item["confirmed"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("host_cache_mib", None),
        ("host_cache_mib", float("nan")),
        ("weight_cache_offload_seconds", None),
        ("weight_cache_activation_seconds", -1),
        ("idle_vram_mib_per_gpu", (float("inf"),)),
        ("peak_vram_mib_per_gpu", (False,)),
        ("vram_baseline_mib_per_gpu", ("bad",)),
        ("sleep_vram_mib_per_gpu", (float("nan"),)),
        ("wake_peak_vram_mib_per_gpu", (-1,)),
        ("predicted_tokens_per_second", float("inf")),
        ("eligible_gpu_sets", ()),
    ],
)
def test_sleep_evidence_rejects_incomplete_nonfinite_and_inconsistent_values(
    tmp_path, field, value
):
    cfg = settings(tmp_path, "vllm-sleep")
    cfg.engines.vllm_executable = "vllm"
    good = profile(cfg)
    assert profile_verified_for_mode(good, kvcached_required=False, ram_weight_cache_required=True)
    assert not profile_verified_for_mode(
        replace(good, **{field: value}),
        kvcached_required=False,
        ram_weight_cache_required=True,
    )


def test_owner_lock_refuses_second_owner_and_releases_after_error(tmp_path):
    resource = database_resource(tmp_path / "state.db")
    root = tmp_path / "locks"
    with pytest.raises(ValueError), owner_lock(resource, root=root):
        with (
            pytest.raises(RuntimeError, match="Another service owns"),
            owner_lock(resource, root=root),
        ):
            pytest.fail("second owner entered")
        raise ValueError("startup interrupted")
    with owner_lock(resource, root=root):
        pass


async def test_selected_profile_revalidation_is_atomic_and_keeps_launch(saved_models):
    from llm_rio.domain import CatalogState

    model_id, job_id = await saved_models.create_model_job(
        nickname="target",
        repo="org/model",
        revision="pinned",
        creator_key_id="admin",
        grant_key_ids=[],
    )
    await saved_models.update_model_job(
        job_id,
        job_state="COMPLETED",
        stage="complete",
        catalog_state=CatalogState.AVAILABLE,
    )
    await saved_models.execute(
        "UPDATE model_profiles SET model_id=? WHERE id=?", (model_id, "old-profile")
    )
    calls = await asyncio.gather(
        *[
            queue_validation(
                saved_models,
                job_id,
                engines=("vllm",),
                profile_id="old-profile",
                overrides={"max_model_len": 8192},
            )
            for _ in range(2)
        ],
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in calls) == 1
    job = await saved_models.get_model_job(job_id)
    assert job["validation_overrides"]["_target"]["profile_id"] == "old-profile"
    assert job["validation_overrides"]["max_model_len"] == 8192
    assert job["validation_overrides"]["_target"]["engine"] == "vllm"


async def test_archive_checks_owner_config_and_matching_vault(saved_models, tmp_path, monkeypatch):
    source = saved_models.path if hasattr(saved_models, "path") else saved_models.database_path
    await saved_models.close()
    config = tmp_path / "config.toml"
    config.write_text('serving_mode="queue"\n')
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    with (
        owner_lock(database_resource(source)),
        pytest.raises(RuntimeError, match="Another service owns"),
    ):
        archive(source, tmp_path / "live", config)
    assert not (tmp_path / "live").exists()
    with pytest.raises(ValueError, match="config"):
        archive(source, tmp_path / "no-config")
    result = archive(source, tmp_path / "good", config)
    assert result["credential_restore_verified"]
    vault = source.with_name(f".{source.stem}-api-key-vault")
    vault.write_bytes(Fernet.generate_key())
    with pytest.raises(InvalidToken):
        archive(source, tmp_path / "wrong-vault", config)
    assert not (tmp_path / "wrong-vault" / "manifest.json").exists()
    assert json.loads((tmp_path / "good" / "manifest.json").read_text())["ownership_verified"]


@pytest.mark.parametrize("tamper", [False, True])
async def test_archive_verifies_deleted_credential_tombstones(
    saved_models, tmp_path, monkeypatch, tamper
):
    from llm_rio.domain import Role

    await saved_models.create_key(
        key_id="removed",
        nickname="removed",
        role=Role.USER,
        account_id="removed-account",
        account_nickname="removed-account",
        prefix="rio_removed_prefix_123456",
        api_key="rio_removed_prefix_123456_secret",
        limit_tokens=0,
        unlimited=True,
    )
    assert await saved_models.delete_key("removed")
    if tamper:
        await saved_models.execute("UPDATE api_keys SET token_hash='invalid' WHERE id='removed'")
    source = saved_models.path if hasattr(saved_models, "path") else saved_models.database_path
    await saved_models.close()
    config = tmp_path / "config.toml"
    config.write_text('serving_mode="queue"\n')
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    destination = tmp_path / "deleted-credentials"
    if tamper:
        with pytest.raises(RuntimeError, match="credentials do not match"):
            archive(source, destination, config)
        assert not (destination / "manifest.json").exists()
    else:
        assert archive(source, destination, config)["credential_restore_verified"]


async def test_retry_partial_overrides_keep_saved_launch_settings(saved_models):
    from llm_rio.domain import CatalogState

    _, job_id = await saved_models.create_model_job(
        nickname="partial",
        repo="org/model",
        revision="pinned",
        creator_key_id="admin",
        grant_key_ids=[],
    )
    await saved_models.update_model_job(
        job_id,
        job_state="COMPLETED",
        stage="complete",
        catalog_state=CatalogState.AVAILABLE,
    )
    await saved_models.set_model_job_validation_overrides(
        job_id,
        {
            "max_num_seqs": 8,
            "launch_args": {"hf_overrides": {"rope_theta": 10000}},
        },
    )
    await queue_validation(
        saved_models, job_id, engines=("vllm",), profile_id=None, overrides={"max_model_len": 8192}
    )
    job = await saved_models.get_model_job(job_id)
    assert job["validation_overrides"] == {
        "max_model_len": 8192,
        "max_num_seqs": 8,
        "launch_args": {"hf_overrides": {"rope_theta": 10000}},
    }
