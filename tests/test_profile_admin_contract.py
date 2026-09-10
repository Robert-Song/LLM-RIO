from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_rio.api.inference_validation import _apply_model_defaults
from llm_rio.api.routes_admin import _apply_profile_edit
from llm_rio.api.schemas import (
    ChatCompletionRequest,
    ModelProfileCloneRequest,
    ProfileEditRequest,
)
from llm_rio.domain import (
    Engine,
    GpuDevice,
    MachineInventory,
    PlacementProfile,
    Role,
)
from llm_rio.errors import RioError
from llm_rio.inventory import candidate_gpu_sets
from llm_rio.profiles import ProfileRepository, profile_key, profile_to_dict
from llm_rio.storage import Database, _now

GPU_0 = "GPU-0"
GPU_1 = "GPU-1"


def make_profile(profile_id: str, model_id: str, gpu_set: tuple[str, ...]) -> PlacementProfile:
    gpu_count = len(gpu_set)
    return PlacementProfile(
        id=profile_id,
        model_id=model_id,
        model_revision="immutable-revision",
        engine=Engine.VLLM,
        engine_version="test",
        machine_fingerprint="machine",
        gpu_count=gpu_count,
        tensor_parallel_size=gpu_count,
        pipeline_parallel_size=1,
        eligible_gpu_sets=(gpu_set,),
        dtype="auto",
        quantization=None,
        max_model_len=4096,
        max_num_seqs=128,
        max_num_batched_tokens=None,
        predicted_tokens_per_second=10.0,
        load_and_warmup_seconds=1.0,
        idle_vram_mib_per_gpu=(1,) * gpu_count,
        peak_vram_mib_per_gpu=(2,) * gpu_count,
        gpu_headroom_mib_per_gpu=(0,) * gpu_count,
        capabilities=frozenset({"chat", "streaming"}),
        launch_args={},
        gpu_memory_utilization=0.9,
        kv_cache_capacity_tokens=4096,
        max_full_length_concurrency=1.0,
        memory_backend="kvcached",
        sleep_vram_mib_per_gpu=(1,) * gpu_count,
        host_cache_mib=128.0,
        normal_verified=True,
        kvcached_verified=True,
        vram_measurement_version=2,
        vram_baseline_mib_per_gpu=(0,) * gpu_count,
        wake_peak_vram_mib_per_gpu=(2,) * gpu_count,
    )


def test_generation_defaults_clone_and_apply_to_requests() -> None:
    clone = ModelProfileCloneRequest(
        nickname="model-copy",
        temperature=0.3,
        top_p=0.9,
        top_k=40,
        min_p=0.1,
        presence_penalty=-0.5,
        repetition_penalty=1.05,
        reasoning_effort="high",
    )

    assert clone.request_defaults == {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.1,
        "presence_penalty": -0.5,
        "repetition_penalty": 1.05,
        "reasoning_effort": "high",
    }
    request = ChatCompletionRequest(model="model-copy", messages=[{"role": "user"}])
    defaulted = _apply_model_defaults(request, {"request_defaults": clone.request_defaults})
    assert defaulted.min_p == 0.1
    assert defaulted.presence_penalty == -0.5
    assert defaulted.repetition_penalty == 1.05


@pytest.mark.asyncio
async def test_deleted_key_is_hidden_but_audit_tombstone_remains(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="deleted-key",
            nickname="former-user",
            role=Role.USER,
            account_id="former-account",
            account_nickname="former-user",
            prefix="rio_former_user_12345678",
            api_key="rio_former_user_12345678_secret",
            limit_tokens=100,
            unlimited=False,
        )
        assert await database.delete_key("deleted-key")
        assert await database.list_keys() == []
        row = await database.fetchone(
            "SELECT nickname FROM api_keys WHERE id = ?", ("deleted-key",)
        )
        assert row is not None and row["nickname"] == "deleted-deleted-key"
        assert not await database.set_key_active("deleted-key", True)
    finally:
        await database.close()


def inventory() -> MachineInventory:
    return MachineInventory(
        machine_id="test",
        driver_version="test",
        cuda_driver_version="test",
        gpus=(
            GpuDevice(uuid=GPU_0, index=0, name="GPU", total_vram_mib=48_000),
            GpuDevice(uuid=GPU_1, index=1, name="GPU", total_vram_mib=48_000),
        ),
        topology_hash="test",
        fingerprint="machine",
    )


def test_profile_override_rebuilds_placement_for_new_tensor_parallel_size() -> None:
    base = make_profile("profile", "model", (GPU_0,))
    request = ProfileEditRequest(tensor_parallel_size=2, max_model_len=8192)

    updated = _apply_profile_edit(
        profile=base,
        model={},
        request=request,
        managed_gpu_count=2,
        eligible_gpu_sets=candidate_gpu_sets(inventory(), 2),
        llama_cpp_enabled=False,
    )

    assert updated.engine is Engine.VLLM
    assert updated.gpu_count == 2
    assert updated.tensor_parallel_size == 2
    assert updated.eligible_gpu_sets == ((GPU_0, GPU_1),)
    assert updated.max_model_len == 8192
    assert len(updated.idle_vram_mib_per_gpu) == 2
    assert updated.idle_vram_mib_per_gpu == (0, 0)
    assert updated.peak_vram_mib_per_gpu == (0, 0)
    assert updated.sleep_vram_mib_per_gpu is None
    assert updated.vram_measurement_version == 0
    assert updated.vram_baseline_mib_per_gpu is None
    assert updated.wake_peak_vram_mib_per_gpu is None
    assert not updated.normal_verified
    assert not updated.kvcached_verified


def test_vllm_limit_edit_preserves_dtype_and_quantization() -> None:
    base = replace(
        make_profile("profile", "model", (GPU_0,)),
        dtype="bfloat16",
        quantization="fp8",
    )
    updated = _apply_profile_edit(
        profile=base,
        model={},
        request=ProfileEditRequest(max_num_seqs=16),
        managed_gpu_count=2,
        eligible_gpu_sets=candidate_gpu_sets(inventory(), 1),
        llama_cpp_enabled=False,
    )

    assert updated.dtype == "bfloat16"
    assert updated.quantization == "fp8"
    assert updated.max_num_seqs == 16
    assert updated.vram_measurement_version == 0


def test_llama_cpp_override_requires_explicit_gguf_and_persists_ngl(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    gguf_file = model_root / "model-q4.gguf"
    gguf_file.write_bytes(b"GGUF")
    base = make_profile("profile", "model", (GPU_0,))
    request = ProfileEditRequest(
        engine=Engine.LLAMA_CPP,
        gguf_file="model-q4.gguf",
        n_gpu_layers=99,
    )

    updated = _apply_profile_edit(
        profile=base,
        model={"artifact_path": str(model_root)},
        request=request,
        managed_gpu_count=2,
        eligible_gpu_sets=candidate_gpu_sets(inventory(), 1),
        llama_cpp_enabled=True,
    )

    assert updated.engine is Engine.LLAMA_CPP
    assert updated.dtype == "gguf"
    assert updated.launch_args["model"] == str(gguf_file)
    assert updated.launch_args["n_gpu_layers"] == 99


def test_llama_cpp_override_stays_disabled_until_explicitly_enabled(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model-q4.gguf").write_bytes(b"GGUF")
    request = ProfileEditRequest(engine=Engine.LLAMA_CPP, gguf_file="model-q4.gguf")

    with pytest.raises(RioError, match="enable_llama_cpp"):
        _apply_profile_edit(
            profile=make_profile("profile", "model", (GPU_0,)),
            model={"artifact_path": str(model_root)},
            request=request,
            managed_gpu_count=2,
            eligible_gpu_sets=candidate_gpu_sets(inventory(), 1),
            llama_cpp_enabled=False,
        )


@pytest.mark.asyncio
async def test_profile_override_updates_default_and_catalog_context_limit(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="admin-key",
            nickname="admin",
            role=Role.ADMIN,
            account_id="admin-account",
            account_nickname="admin-account",
            prefix="rio_admin_prefix_12345678",
            api_key="rio_admin_prefix_12345678_secret",
            limit_tokens=0,
            unlimited=True,
        )
        await database.execute(
            """
            INSERT INTO model_catalog
                (id, nickname, huggingface_repo, state, created_by_key_id, created_at, updated_at,
                 request_limits_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "model",
                "model",
                "org/model",
                "AVAILABLE",
                "admin-key",
                _now(),
                _now(),
                '{"max_context_tokens": 4096}',
            ),
        )
        profile = make_profile("profile", "model", (GPU_0,))
        raw = profile_to_dict(profile)
        await database.execute(
            """
            INSERT INTO model_profiles
                (id, model_id, machine_fingerprint, profile_key, profile_json, verified_at, active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                profile.id,
                profile.model_id,
                profile.machine_fingerprint,
                profile_key(raw),
                json.dumps(raw),
                _now(),
            ),
        )
        repository = ProfileRepository(database, "machine")
        assert await repository.update(replace(profile, max_model_len=8192), make_default=True)

        records = await repository.records_for_model("model")
        assert len(records) == 1
        assert records[0].active
        assert records[0].profile.max_model_len == 8192
        assert records[0].profile.vram_measurement_version == 0
        assert records[0].profile.vram_baseline_mib_per_gpu is None
        assert records[0].profile.peak_vram_mib_per_gpu == (0,)
        assert not records[0].profile.normal_verified
        assert not records[0].profile.kvcached_verified
        model = await database.model_by_id("model")
        assert model is not None
        assert model["request_limits"]["max_context_tokens"] == 8192
        updated = await database.update_model_request_defaults(
            "model", {"min_p": 0.1, "presence_penalty": -0.5, "repetition_penalty": 1.05}
        )
        assert updated is not None
        assert updated["request_defaults"] == {
            "min_p": 0.1,
            "presence_penalty": -0.5,
            "repetition_penalty": 1.05,
        }
        cleared = await database.update_model_request_defaults("model", {"min_p": None})
        assert cleared is not None and "min_p" not in cleared["request_defaults"]
        assert await repository.set_active(model_id="model", profile_id="profile", active=False)
        records = await repository.records_for_model("model")
        assert not records[0].active
        assert await repository.set_active(model_id="model", profile_id="profile", active=True)
        records = await repository.records_for_model("model")
        assert records[0].active

    finally:
        await database.close()


def test_profile_key_separates_logical_models_and_launch_configuration() -> None:
    raw = profile_to_dict(make_profile("profile", "model-a", (GPU_0,)))
    other_model = {**raw, "model_id": "model-b"}
    with_overrides = {
        **raw,
        "launch_args": {"hf_overrides": {"max_position_embeddings": 8192}},
    }

    assert profile_key(raw) != profile_key(other_model)
    assert profile_key(raw) != profile_key(with_overrides)


@pytest.mark.asyncio
async def test_clone_model_shares_weights_but_has_separate_yarn_profiles_and_defaults(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="admin-key",
            nickname="admin",
            role=Role.ADMIN,
            account_id="admin-account",
            account_nickname="admin-account",
            prefix="rio_admin_prefix_12345678",
            api_key="rio_admin_prefix_12345678_secret",
            limit_tokens=0,
            unlimited=True,
        )
        artifact_path = tmp_path / "qwen3.8-27b-nvfp4"
        artifact_path.mkdir()
        (artifact_path / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5",
                    "text_config": {
                        "max_position_embeddings": 262_144,
                        "rope_parameters": {
                            "rope_type": "default",
                            "rope_theta": 10_000_000.0,
                            "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10],
                            "mrope_interleaved": True,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        await database.execute(
            """
            INSERT INTO model_catalog
                (id, nickname, huggingface_repo, requested_revision, resolved_revision,
                 state, artifact_path, artifact_hashes_json, capabilities_json,
                 request_limits_json, request_defaults_json, created_by_key_id,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "source-model",
                "qwen3.8-27b-nvfp4",
                "org/qwen",
                "main",
                "immutable-revision",
                "AVAILABLE",
                str(artifact_path),
                '["weight-hash"]',
                '["chat", "streaming"]',
                '{"max_context_tokens": 262144}',
                '{"temperature": 0.6}',
                "admin-key",
                _now(),
                _now(),
            ),
        )
        await database.execute(
            """
            INSERT INTO model_grants(key_id, model_id, created_at)
            VALUES (?, ?, ?)
            """,
            ("admin-key", "source-model", _now()),
        )
        source_profile = replace(
            make_profile("source-profile", "source-model", (GPU_0,)),
            max_model_len=262_144,
            kv_cache_capacity_tokens=262_144,
        )
        raw_profile = profile_to_dict(source_profile)
        await database.execute(
            """
            INSERT INTO model_profiles
                (id, model_id, machine_fingerprint, profile_key, profile_json,
                 verified_at, active)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            """,
            (
                source_profile.id,
                source_profile.model_id,
                source_profile.machine_fingerprint,
                profile_key(raw_profile),
                json.dumps(raw_profile),
                _now(),
            ),
        )
        repository = ProfileRepository(database, "machine")
        source_model = await database.model_by_id("source-model")
        assert source_model is not None

        clone, profiles = await repository.clone_model(
            source_model=source_model,
            nickname="qwen3.8-27b-nvfp4-ext",
            creator_key_id="admin-key",
            request_defaults={"reasoning_effort": "medium"},
            max_model_len=1_048_576,
            yarn_factor=4,
            yarn_original_max_model_len=None,
            inherit_grants=True,
        )

        assert clone["id"] != source_model["id"]
        assert clone["source_model_id"] == source_model["id"]
        assert clone["artifact_path"] == source_model["artifact_path"]
        assert clone["artifact_hashes"] == source_model["artifact_hashes"]
        assert clone["request_defaults"] == {
            "temperature": 0.6,
            "reasoning_effort": "medium",
        }
        assert clone["request_limits"]["max_context_tokens"] == 1_048_576
        assert len(profiles) == 1
        cloned_profile = profiles[0]
        assert cloned_profile.id != source_profile.id
        assert cloned_profile.model_id == clone["id"]
        assert cloned_profile.max_model_len == 1_048_576
        assert cloned_profile.vram_measurement_version == 0
        assert cloned_profile.vram_baseline_mib_per_gpu is None
        assert cloned_profile.wake_peak_vram_mib_per_gpu is None
        assert not cloned_profile.normal_verified
        assert not cloned_profile.kvcached_verified
        text_overrides = cloned_profile.launch_args["hf_overrides"]["text_config"]
        assert text_overrides["max_position_embeddings"] == 1_048_576
        assert text_overrides["rope_parameters"] == {
            "rope_type": "yarn",
            "rope_theta": 10_000_000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
            "factor": 4,
            "original_max_position_embeddings": 262_144,
        }
        assert await database.has_model_grant("admin-key", clone["id"])

        persisted_source = await database.model_by_id("source-model")
        assert persisted_source is not None
        assert persisted_source["request_defaults"] == {"temperature": 0.6}
        assert persisted_source["request_limits"]["max_context_tokens"] == 262_144
        source_profiles = await repository.for_model("source-model")
        assert source_profiles[0].launch_args == {}
        assert source_profiles[0].max_model_len == 262_144
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_trust_both_is_scoped_to_active_profiles_on_current_machine(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="admin-key",
            nickname="admin",
            role=Role.ADMIN,
            account_id="admin-account",
            account_nickname="admin-account",
            prefix="rio_admin_prefix_12345678",
            api_key="rio_admin_prefix_12345678_secret",
            limit_tokens=0,
            unlimited=True,
        )
        await database.execute(
            """
            INSERT INTO model_catalog
                (id, nickname, huggingface_repo, state, created_by_key_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("model", "model", "org/model", "AVAILABLE", "admin-key", _now(), _now()),
        )
        local_profile = replace(
            make_profile("local-profile", "model", (GPU_0,)),
            normal_verified=False,
            kvcached_verified=False,
        )
        foreign_profile = replace(
            local_profile,
            id="foreign-profile",
            machine_fingerprint="other-machine",
        )
        for profile in (local_profile, foreign_profile):
            raw = profile_to_dict(profile)
            await database.execute(
                """
                INSERT INTO model_profiles
                    (id, model_id, machine_fingerprint, profile_key, profile_json,
                     verified_at, active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    profile.id,
                    profile.model_id,
                    profile.machine_fingerprint,
                    profile_key(raw),
                    json.dumps(raw),
                    _now(),
                ),
            )

        repository = ProfileRepository(database, "machine")
        assert await repository.set_model_verified_for_both("model") == 1
        local = (await repository.records_for_model("model"))[0].profile
        assert local.normal_verified
        assert local.kvcached_verified
        foreign_row = await database.fetchone(
            "SELECT profile_json FROM model_profiles WHERE id = ?",
            ("foreign-profile",),
        )
        assert foreign_row is not None
        foreign = json.loads(foreign_row["profile_json"])
        assert not foreign["normal_verified"]
        assert not foreign["kvcached_verified"]
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_profile_backend_verification_is_scoped_to_one_profile_and_backend(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="admin-key",
            nickname="admin",
            role=Role.ADMIN,
            account_id="admin-account",
            account_nickname="admin-account",
            prefix="rio_admin_prefix_12345678",
            api_key="rio_admin_prefix_12345678_secret",
            limit_tokens=0,
            unlimited=True,
        )
        await database.execute(
            """
            INSERT INTO model_catalog
                (id, nickname, huggingface_repo, state, created_by_key_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("model", "model", "org/model", "AVAILABLE", "admin-key", _now(), _now()),
        )
        local = replace(
            make_profile("local-profile", "model", (GPU_0,)),
            normal_verified=False,
            kvcached_verified=False,
        )
        other = replace(local, id="other-profile", max_num_seqs=64)
        for profile in (local, other):
            raw = profile_to_dict(profile)
            await database.execute(
                """
                INSERT INTO model_profiles
                    (id, model_id, machine_fingerprint, profile_key, profile_json,
                     verified_at, active)
                VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    profile.id,
                    profile.model_id,
                    profile.machine_fingerprint,
                    profile_key(raw),
                    json.dumps(raw),
                    _now(),
                ),
            )
        repository = ProfileRepository(database, "machine")
        assert await repository.set_profile_backend_verified(
            model_id="model",
            profile_id="local-profile",
            backend="native",
            verified=True,
        )
        records = {
            record.profile.id: record.profile
            for record in await repository.records_for_model("model")
        }
        assert records["local-profile"].normal_verified
        assert not records["local-profile"].kvcached_verified
        assert not records["other-profile"].normal_verified
        assert not records["other-profile"].kvcached_verified
    finally:
        await database.close()
