from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError as SchemaError

from llm_rio.api.schemas import ModelValidationOverrides
from llm_rio.ports import PortAllocator
from llm_rio.profiles import (
    ProfileRepository,
    profile_key,
    profile_to_dict,
    profile_verified_for_mode,
)
from llm_rio.ui.components import _revalidation_overrides
from llm_rio.validation import CandidateShape, ProfileValidator, ValidationError
from llm_rio.workers import WorkerSupervisor
from tests.release_fixtures import Settings
from tests.test_profile_admin_contract import inventory
from tests.test_scheduler_contract import make_worker
from tests.test_validation_cleanup import RegistrationValidatorStub, registration_manager
from tests.test_verification_recovery import saved_models as saved_models


@pytest.mark.parametrize(
    "args",
    [
        {"host": "0.0.0.0"},
        {"max-model-len": 100},
        {"no-enable-sleep-mode": True},
        {"tensor_parallel_size": 4},
        {"config": "/tmp/overrides"},
        {"--dtype": "half"},
        {"kv-cache-dtype": "fp8", "kv_cache_dtype": "auto"},
    ],
)
def test_extra_arguments_cannot_override_managed_settings(args):
    with pytest.raises(SchemaError):
        ModelValidationOverrides(launch_args=args)


def test_form_parses_typed_overrides_and_normalizes_engine_names():
    result = _revalidation_overrides(
        {
            "max_model_len": "131072",
            "gpu_memory_utilization": "0.85",
            "tensor_parallel_size": "2",
            "max_num_seqs": "",
            "launch_args": '{"kv-cache-dtype":"fp8","hf_overrides":{"rope_theta":10000}}',
        }
    )
    assert result == {
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 2,
        "launch_args": {"kv_cache_dtype": "fp8", "hf_overrides": {"rope_theta": 10000}},
    }


@pytest.mark.parametrize(
    "values",
    [
        {"launch_args": "[]"},
        {"launch_args": "{oops"},
        {"tensor_parallel_size": "0"},
        {"gpu_memory_utilization": "1.1"},
    ],
)
def test_invalid_form_values_do_not_generate_a_request(values):
    with pytest.raises(ValueError):
        _revalidation_overrides(values)


@pytest.mark.parametrize("tp", [2, 3])
async def test_explicit_tp_is_never_replaced_by_automatic_tp(tmp_path, tp):
    validator = RegistrationValidatorStub()
    manager = registration_manager(validator)
    manager.inventory = inventory()
    manager._wait_for_validation_window = AsyncMock()
    validator.validate_vllm = AsyncMock(return_value=["validated"])
    request = dict(
        job_id="job",
        job={
            "model_id": "model",
            "nickname": "model",
            "validation_overrides": {
                "tensor_parallel_size": tp,
                "gpu_memory_utilization": 0.85,
                "max_model_len": 131072,
                "launch_args": {
                    "dtype": "bfloat16",
                    "quantization": "awq",
                    "kv_cache_dtype": "fp8",
                },
            },
        },
        artifact_path=tmp_path,
        resolved_revision="revision",
        inspection={
            "max_model_len": 131072,
            "weight_bytes": 1,
            "dtype": "auto",
            "quantization": None,
        },
    )
    if tp == 3:
        with pytest.raises(ValidationError, match="TP=3"):
            await manager._validate_with_requeue(**request)
        validator.validate_vllm.assert_not_awaited()
    else:
        assert await manager._validate_with_requeue(**request) == ["validated"]
        candidate = validator.validate_vllm.call_args.kwargs["candidate"]
        assert candidate.tensor_parallel_size == 2
        assert candidate.max_model_len == 131072
        assert candidate.gpu_memory_utilization == 0.85
        assert candidate.dtype == "bfloat16"
        assert candidate.quantization == "awq"
        assert candidate.launch_args == {"kv_cache_dtype": "fp8"}


async def test_probe_persists_parameters_reused_by_serving(
    saved_models,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    settings = Settings(
        config_file=tmp_path / "missing.toml",
        serving_mode="queue",
        log_dir=tmp_path,
    )
    validator = ProfileValidator(
        settings,
        inventory(),
        SimpleNamespace(supervisor=SimpleNamespace(ports=PortAllocator(18000, 18999))),
    )
    probe = AsyncMock(return_value=SimpleNamespace(pid=1234))
    monkeypatch.setattr("llm_rio.validation.asyncio.create_subprocess_exec", probe)
    monkeypatch.setattr("llm_rio.engines.launch.gpu_environment", lambda *a, **kw: {})
    monkeypatch.setattr(validator, "_used_vram", lambda gpus: (1000,) * len(gpus))
    monkeypatch.setattr(validator, "_wait_for_health", AsyncMock())
    monkeypatch.setattr(validator, "_generation_contract", AsyncMock(return_value=15.0))
    monkeypatch.setattr(validator, "_terminate", AsyncMock())
    candidate = CandidateShape(
        gpu_count=2,
        tensor_parallel_size=2,
        max_model_len=131072,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        quantization="awq",
        eligible_gpu_sets=(("GPU-0", "GPU-1"),),
        launch_args={
            "kv_cache_dtype": "fp8",
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "hf_overrides": {"rope_theta": 10000},
        },
    )
    profile = await validator._probe_vllm_on_port(
        model_id="model",
        model_revision="immutable-revision",
        model_path=tmp_path,
        nickname="model",
        candidate=candidate,
        gpu_set=("GPU-0", "GPU-1"),
        backend="native",
        port=19000,
    )
    repository = ProfileRepository(saved_models, "machine")
    await repository.save(profile, profile_key(profile_to_dict(profile)))
    stored = (await repository.for_model("model"))[0]
    assert profile_verified_for_mode(stored, kvcached_required=False, queue_mode_required=True)
    assert stored.launch_args == {**candidate.launch_args, "enable_sleep_mode": False}
    supervisor = WorkerSupervisor(settings, saved_models)
    worker = make_worker("worker", stored)
    command = supervisor._command(worker, str(tmp_path), "model")
    probe_command = list(probe.call_args.args)
    for flag, value in {
        "--tensor-parallel-size": "2",
        "--max-model-len": "131072",
        "--gpu-memory-utilization": "0.85",
        "--max-num-seqs": "8",
        "--max-num-batched-tokens": "2048",
        "--dtype": "bfloat16",
        "--quantization": "awq",
        "--kv-cache-dtype": "fp8",
    }.items():
        assert command[command.index(flag) + 1] == value
        assert probe_command[probe_command.index(flag) + 1] == value
    for flag in ["--enforce-eager", "--no-enable-prefix-caching"]:
        assert flag in command and flag in probe_command
    for argv in [command, probe_command]:
        assert json.loads(argv[argv.index("--hf-overrides") + 1]) == {"rope_theta": 10000}


async def test_retry_api_persists_tp_and_engine_arguments(
    saved_models,  # noqa: F811
    tmp_path,
):
    from unittest.mock import Mock

    import httpx

    from llm_rio.api.app import create_app
    from llm_rio.api.dependencies import current_principal
    from llm_rio.domain import CatalogState, Role
    from llm_rio.security import Principal
    from tests.test_api_lifecycle import settings

    _, job_id = await saved_models.create_model_job(
        nickname="revalidate",
        repo="org/model",
        revision=None,
        creator_key_id="admin",
        grant_key_ids=[],
    )
    await saved_models.update_model_job(
        job_id, job_state="COMPLETED", stage="complete", catalog_state=CatalogState.AVAILABLE
    )
    app = create_app(settings(tmp_path))
    app.state.database = saved_models
    app.state.scheduler = SimpleNamespace(
        mode=SimpleNamespace(capabilities=SimpleNamespace(engines=("vllm",)))
    )
    app.state.registration = SimpleNamespace(start=Mock())
    app.dependency_overrides[current_principal] = lambda: Principal(
        "admin", "admin", Role.ADMIN, "account", True
    )
    overrides = {
        "max_model_len": 131072,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 2,
        "launch_args": {"kv-cache-dtype": "fp8"},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        invalid = await client.post(
            f"/staff/model-jobs/{job_id}/retry",
            json={"validation_overrides": {"launch_args": {"max_model_len": 131072}}},
        )
        assert invalid.status_code == 422
        app.state.registration.start.assert_not_called()
        response = await client.post(
            f"/staff/model-jobs/{job_id}/retry", json={"validation_overrides": overrides}
        )
    assert response.status_code == 202, response.text
    job = await saved_models.get_model_job(job_id)
    assert job["validation_overrides"] == {**overrides, "launch_args": {"kv_cache_dtype": "fp8"}}
    app.state.registration.start.assert_called_once_with(job_id)
