from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from typer.testing import CliRunner

from llm_rio import admin_client as client_api
from llm_rio import cli
from llm_rio.api.app import create_app
from llm_rio.api.dependencies import current_principal
from llm_rio.api.schemas import ProfileEditRequest
from llm_rio.domain import Role
from llm_rio.inventory import candidate_gpu_sets
from llm_rio.modes.factory import create_mode
from llm_rio.profiles import StoredProfile, invalidate_profile_measurements
from llm_rio.registration import RegistrationManager
from llm_rio.security import Principal
from llm_rio.services.profile_edit import _apply_profile_edit
from tests.release_fixtures import replace
from tests.test_api_lifecycle import settings
from tests.test_profile_admin_contract import inventory, make_profile
from tests.test_validation_cleanup import (
    RegistrationValidatorStub,
    candidate_shape,
    registration_manager,
)


@pytest.mark.parametrize("context", [4096, 131072])
def test_unchanged_tp_preserves_probed_gpu_placement(context):
    base = make_profile("profile", "model", ("GPU-0",))
    updated = _apply_profile_edit(
        profile=base,
        model={},
        request=ProfileEditRequest(tensor_parallel_size=1, max_model_len=context),
        managed_gpu_count=2,
        eligible_gpu_sets=candidate_gpu_sets(inventory(), 1),
        llama_cpp_enabled=False,
    )
    assert updated.eligible_gpu_sets == base.eligible_gpu_sets
    assert updated.measurements_valid is (context == 4096)
    assert updated.vram_measurement_version == (2 if context == 4096 else 0)


@pytest.mark.parametrize("duplicate", [False, True])
async def test_profile_edit_reports_recovery_action(tmp_path, duplicate):
    app = create_app(settings(tmp_path))
    app.dependency_overrides[current_principal] = lambda: Principal(
        "key", "admin", Role.ADMIN, "account", True
    )
    base = replace(make_profile("profile", "model", ("GPU-0",)), memory_backend="native")
    conflict = invalidate_profile_measurements(
        replace(base, id="existing-profile", max_model_len=131072)
    )
    records = [StoredProfile(base, True), StoredProfile(conflict, False)]
    app.state.settings = settings(tmp_path)
    app.state.inventory = inventory()
    app.state.scheduler = SimpleNamespace(mode=create_mode(app.state.settings, app.state.inventory))
    app.state.database = SimpleNamespace(
        model_by_id=AsyncMock(return_value={"id": "model"}), record_event=AsyncMock()
    )
    app.state.supervisor = SimpleNamespace(workers={}, drain=AsyncMock())
    app.state.profiles = SimpleNamespace(
        records_for_model=AsyncMock(return_value=records),
        update=AsyncMock(
            return_value=True,
            side_effect=sqlite3.IntegrityError("UNIQUE constraint failed") if duplicate else None,
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/admin/models/model/profiles/profile", json={"max_model_len": 131072}
        )
    if duplicate:
        assert response.status_code == 409
        error = response.json()["error"]
        assert error["existing_profile_id"] == "existing-profile"
        assert error["existing_profile_active"] is False
        assert "inactive" in error["message"]
        app.state.database.record_event.assert_not_awaited()
    else:
        assert response.status_code == 200
        assert response.json()["verification_required"] == ["measurements_invalid"]
        assert response.json()["profile"]["vram_measurement_version"] == 0


@pytest.mark.parametrize("declared_limit", [None, 8192])
@pytest.mark.parametrize("override_source", ["job", "settings"])
async def test_real_validation_override_can_exceed_only_fallback(
    tmp_path, monkeypatch, declared_limit, override_source
):
    text_config = {"model_type": "gemma3_text"}
    if declared_limit is not None:
        text_config["max_position_embeddings"] = declared_limit
    (tmp_path / "config.json").write_text(json.dumps({"text_config": text_config}))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 10**30}))
    inspection = RegistrationManager._inspect(
        tmp_path, {"artifact_hashes": [{"path": "model.safetensors", "bytes": 1}]}
    )
    assert inspection["max_model_len"] == (declared_limit or 4096)
    assert inspection["max_model_len_is_fallback"] is (declared_limit is None)
    manager = registration_manager(RegistrationValidatorStub())
    captured = []

    def shapes(**kwargs):
        captured.append(kwargs)
        return [candidate_shape(1, (("GPU-0",),))]

    monkeypatch.setattr("llm_rio.registration.build_candidate_shapes", shapes)
    job = {"model_id": "model", "nickname": "model"}
    if override_source == "job":
        job["validation_overrides"] = {"max_model_len": 131072}
    else:
        manager.settings.engines.max_model_len = 131072
    await manager._validate_with_requeue(
        job_id="job",
        job=job,
        artifact_path=tmp_path,
        resolved_revision="revision",
        inspection=inspection,
    )
    assert captured[0]["max_model_len"] == (declared_limit or 131072)


def test_retry_context_override_sends_only_changed_limit(monkeypatch):
    calls = []
    from llm_rio.commands import models

    monkeypatch.setattr(models, "_job_id_from_selector", lambda _: "job")

    def request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if method == "GET":
            return {"validation_overrides": {"max_num_seqs": 8, "max_model_len": 4096}}
        return {"job_id": "job"}

    monkeypatch.setattr(client_api, "request", request)
    result = CliRunner().invoke(
        cli.app, ["models", "validate", "gemma3-27b-awq", "--max-model-len", "131072"]
    )
    assert result.exit_code == 0, result.output
    assert calls[-1] == (
        "POST",
        "/staff/model-jobs/job/retry",
        {"json_body": {"validation_overrides": {"max_model_len": 131072}}},
    )
