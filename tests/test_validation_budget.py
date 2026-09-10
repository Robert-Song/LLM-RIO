from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_validation_cleanup import candidate_shape

from llm_rio.config import Settings
from llm_rio.process_cleanup import TeardownError
from llm_rio.validation import ProfileValidator, ValidationError


@pytest.mark.parametrize("normal,expected", [(True, [0.8, 0.7]), (False, [0.95, 0.85])])
async def test_normal_memory_retry_persists_smaller_budget(tmp_path, normal, expected):
    validator = ProfileValidator.__new__(ProfileValidator)
    validator.settings = Settings()
    validator.scheduler = SimpleNamespace(
        validation_requires_maintenance=normal,
        database=SimpleNamespace(record_event=AsyncMock()),
    )
    validator._check_native_headroom = AsyncMock()
    validator._reserve_validation_port = AsyncMock(return_value=19000)
    validator._release_validation_port = AsyncMock()
    log = tmp_path / "engine.log"
    log.write_text("torch.OutOfMemoryError: CUDA out of memory")
    seen = []

    async def probe(**kwargs):
        seen.append(kwargs["candidate"].gpu_memory_utilization)
        if len(seen) == 1:
            raise ValidationError("engine_startup", "exit", {"log_path": str(log)})
        return kwargs["candidate"]

    validator._probe_vllm_on_port = probe
    args = dict(
        model_id="m",
        model_revision="r",
        model_path=tmp_path,
        nickname="m",
        candidate=replace(candidate_shape(1, (("GPU-0",),)), gpu_memory_utilization=0.95),
        gpu_set=("GPU-0",),
        backend="native",
    )
    if normal:
        profile = await validator._probe_vllm(**args)
        assert profile.gpu_memory_utilization == 0.7
        assert seen == expected
        assert validator._release_validation_port.await_count == 2
    else:
        with pytest.raises(ValidationError):
            await validator._probe_vllm(**args)
        assert seen == [0.95]  # Prism validation does not inherit normal policy.


@pytest.mark.parametrize("failure", ["configuration", "oom", "teardown"])
async def test_retries_bounded_and_teardown_failure_keeps_reservations(tmp_path, failure):
    validator = ProfileValidator.__new__(ProfileValidator)
    validator.settings = Settings()
    released = AsyncMock()
    validator.scheduler = SimpleNamespace(
        validation_requires_maintenance=True,
        database=SimpleNamespace(record_event=AsyncMock()),
        acquire_validation_gpus=AsyncMock(return_value=True),
        release_validation_gpus=released,
    )
    validator._check_native_headroom = AsyncMock()
    validator._reserve_validation_port = AsyncMock(return_value=19000)
    validator._release_validation_port = AsyncMock()
    log = tmp_path / "engine.log"
    log.write_text("CUDA out of memory" if failure == "oom" else "unsupported architecture")
    error = (
        TeardownError("GPU still owned")
        if failure == "teardown"
        else ValidationError("engine_startup", "exit", {"log_path": str(log)})
    )
    validator._probe_vllm_on_port = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        await validator.validate_vllm(
            model_id="m",
            model_revision="r",
            model_path=tmp_path,
            nickname="m",
            candidate=replace(candidate_shape(1, (("GPU-0",),)), gpu_memory_utilization=0.95),
        )
    assert validator._probe_vllm_on_port.await_count == (3 if failure == "oom" else 1)
    assert released.await_count == (0 if failure == "teardown" else 1)
    assert validator._release_validation_port.await_count == (
        0 if failure == "teardown" else 3 if failure == "oom" else 1
    )
