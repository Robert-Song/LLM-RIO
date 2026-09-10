from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_profile_admin_contract import inventory, make_profile

import llm_rio.api.app as app_module
from llm_rio.api.dependencies import current_principal
from llm_rio.api.schemas import ProfileEditRequest
from llm_rio.config import Settings
from llm_rio.domain import Role
from llm_rio.profiles import StoredProfile
from llm_rio.security import Principal


def settings(tmp_path: Path) -> Settings:
    return Settings(
        config_file=tmp_path / "missing.toml",
        database_path=tmp_path / "state/test.db",
        model_store=tmp_path / "models",
        log_dir=tmp_path / "logs",
    )


async def test_inventory_failure_closes_database_and_http_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = SimpleNamespace(open=AsyncMock(), close=AsyncMock())
    client = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app_module, "Database", lambda _: database)
    monkeypatch.setattr(app_module.httpx, "AsyncClient", lambda **_: client)

    def fail_inventory(*args):
        raise RuntimeError("inventory failed")

    app = app_module.create_app(settings(tmp_path), inventory_provider=fail_inventory)
    with pytest.raises(RuntimeError, match="inventory failed"):
        async with app.router.lifespan_context(app):
            pytest.fail("startup unexpectedly succeeded")
    database.close.assert_awaited_once()
    client.aclose.assert_awaited_once()


@pytest.mark.parametrize("failure", ["scheduler", "registration", "shutdown"])
async def test_partial_startup_and_shutdown_always_close_all_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    closed = []
    real_database = app_module.Database
    opened = []

    def database_factory(path):
        database = real_database(path)
        opened.append(database)
        return database

    async def close_scheduler():
        closed.append("scheduler")
        if failure == "shutdown":
            raise RuntimeError("shutdown failed")

    async def close_registration():
        closed.append("registration")

    scheduler = SimpleNamespace(
        start=AsyncMock(
            side_effect=RuntimeError("scheduler failed") if failure == "scheduler" else None
        ),
        close=close_scheduler,
    )
    registration = SimpleNamespace(
        resume=AsyncMock(
            side_effect=RuntimeError("registration failed") if failure == "registration" else None
        ),
        close=close_registration,
    )
    monkeypatch.setattr(app_module, "Database", database_factory)
    monkeypatch.setattr(app_module, "ResidencyScheduler", lambda **_: scheduler)
    monkeypatch.setattr(app_module, "RegistrationManager", lambda **_: registration)
    monkeypatch.setattr(app_module, "ProfileValidator", lambda *_: object())
    app = app_module.create_app(settings(tmp_path), inventory_provider=lambda *_: inventory())
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        async with app.router.lifespan_context(app):
            pass
    assert closed == ["scheduler", "registration"]
    assert app.state.worker_client.is_closed
    with pytest.raises(RuntimeError, match="not open"):
        _ = opened[0].connection


async def test_custom_validation_errors_return_json_422(tmp_path: Path) -> None:
    app = app_module.create_app(settings(tmp_path))
    app.dependency_overrides[current_principal] = lambda: Principal(
        "key", "admin", Role.ADMIN, "account", True
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions", json={"model": "model", "messages": []}
        )
    assert response.status_code == 422
    details = response.json()["error"]["details"]
    assert details[0]["ctx"]["error"] == "messages cannot be empty"


async def test_profile_edit_endpoint_supplies_managed_gpu_count(tmp_path: Path) -> None:
    app = app_module.create_app(settings(tmp_path))
    app.dependency_overrides[current_principal] = lambda: Principal(
        "key", "admin", Role.ADMIN, "account", True
    )
    profile = make_profile("profile", "model", ("GPU-0",))
    app.state.settings = settings(tmp_path)
    app.state.inventory = inventory()
    app.state.database = SimpleNamespace(
        model_by_id=AsyncMock(return_value={"id": "model"}), record_event=AsyncMock()
    )
    app.state.profiles = SimpleNamespace(
        records_for_model=AsyncMock(return_value=[StoredProfile(profile=profile, active=True)]),
        update=AsyncMock(return_value=True),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.patch(
            "/admin/models/model/profiles/profile",
            json=ProfileEditRequest(tensor_parallel_size=2).model_dump(exclude_unset=True),
        )
    assert response.status_code == 200, response.text
    updated = app.state.profiles.update.call_args.args[0]
    assert updated.tensor_parallel_size == 2
    assert updated.eligible_gpu_sets == (("GPU-0", "GPU-1"),)
    assert not updated.normal_verified
