"""End-to-end API tests against the FastAPI app with a fake GPU inventory."""

from __future__ import annotations

from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from llm_rio.api.app import create_app
from llm_rio.config import Settings
from llm_rio.domain import GpuDevice, MachineInventory


def fake_inventory(machine_id: str, managed_gpu_uuids: list[str]) -> MachineInventory:
    devices = (
        GpuDevice(
            uuid="GPU-fake-1",
            index=0,
            name="FakeGPU",
            total_vram_mib=24000,
            compute_capability="9.0",
            pci_bus_id="0000:00:00.0",
        ),
    )
    return MachineInventory(
        machine_id=machine_id,
        driver_version="1.0",
        cuda_driver_version="13.3",
        gpus=devices,
        topology_hash="topo-hash",
        fingerprint="fingerprint-hash",
        topology={"GPU0": {"GPU0": "X"}},
    )


@pytest_asyncio.fixture
async def client(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
machine_id = "test-host"
database_path = "{tmp_path / 'state' / 'llm-rio.db'}"
model_store = "{tmp_path / 'models'}"
log_dir = "{tmp_path / 'logs'}"
api_port = 8000
"""
    )
    settings = Settings(config_file=config)
    app = create_app(settings=settings, inventory_provider=fake_inventory)
    transport = ASGITransport(app=app)
    async with (
        AsyncClient(transport=transport, base_url="http://test") as client,
        app.router.lifespan_context(app),
    ):
        yield client


class TestHealth:
    async def test_health(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["mode"] == "ACTIVE"
        assert body["machine_id"] == "test-host"
        assert body["managed_gpus"] == 1

    async def test_health_without_auth(self, client: AsyncClient) -> None:
        # /health is unauthenticated by design.
        response = await client.get("/health")
        assert response.status_code == 200


class TestAuthentication:
    async def test_missing_bearer_rejected(self, client: AsyncClient) -> None:
        response = await client.get("/v1/models")
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == "invalid_api_key"

    async def test_invalid_token_rejected(self, client: AsyncClient) -> None:
        response = await client.get(
            "/v1/models", headers={"Authorization": "Bearer rio_bogus_token_value_123"}
        )
        assert response.status_code == 401

    async def test_admin_created_on_first_startup(self, client: AsyncClient) -> None:
        # The lifespan creates an initial admin; authenticate by recovering
        # the key from the database through the CLI credential path.
        from llm_rio.security import ApiKeyVault, default_key_vault_path

        settings = client._transport.app.state.settings
        vault_path = default_key_vault_path(settings.database_path)
        database = client._transport.app.state.database
        row = await database.fetchone(
            """
            SELECT encrypted_api_key FROM api_keys
             WHERE role = 'admin' AND active = 1
             ORDER BY created_at, id LIMIT 1
            """
        )
        assert row is not None
        token = ApiKeyVault(vault_path).decrypt(str(row["encrypted_api_key"]))
        response = await client.get(
            "/v1/models", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["data"] == []


class TestChatCompletions:
    async def test_unknown_model_rejected(self, client: AsyncClient) -> None:
        from llm_rio.security import ApiKeyVault, default_key_vault_path

        settings = client._transport.app.state.settings
        vault_path = default_key_vault_path(settings.database_path)
        database = client._transport.app.state.database
        row = await database.fetchone(
            """
            SELECT encrypted_api_key FROM api_keys
             WHERE role = 'admin' AND active = 1
             ORDER BY created_at, id LIMIT 1
            """
        )
        token = ApiKeyVault(vault_path).decrypt(str(row["encrypted_api_key"]))
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "model": "does-not-exist",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 16,
            },
        )
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "model_not_found"

    async def test_empty_messages_rejected(self, client: AsyncClient) -> None:
        from llm_rio.security import ApiKeyVault, default_key_vault_path

        settings = client._transport.app.state.settings
        vault_path = default_key_vault_path(settings.database_path)
        database = client._transport.app.state.database
        row = await database.fetchone(
            """
            SELECT encrypted_api_key FROM api_keys
             WHERE role = 'admin' AND active = 1
             ORDER BY created_at, id LIMIT 1
            """
        )
        token = ApiKeyVault(vault_path).decrypt(str(row["encrypted_api_key"]))
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "x", "messages": [], "max_tokens": 16},
        )
        assert response.status_code == 422

    async def test_maintenance_mode_rejects_requests(
        self, client: AsyncClient, tmp_path: Path
    ) -> None:
        from llm_rio.domain import ServiceMode
        from llm_rio.security import ApiKeyVault, default_key_vault_path

        settings = client._transport.app.state.settings
        vault_path = default_key_vault_path(settings.database_path)
        database = client._transport.app.state.database
        row = await database.fetchone(
            """
            SELECT encrypted_api_key FROM api_keys
             WHERE role = 'admin' AND active = 1
             ORDER BY created_at, id LIMIT 1
            """
        )
        token = ApiKeyVault(vault_path).decrypt(str(row["encrypted_api_key"]))
        await database.set_service_mode(ServiceMode.DRAINING)
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": "does-not-exist",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "service_maintenance"
        finally:
            await database.set_service_mode(ServiceMode.ACTIVE)


class TestAdminRoutes:
    async def test_keys_require_admin(self, client: AsyncClient) -> None:
        # No key yet -> the admin key works; but an ordinary route check:
        # creating a key requires the admin role, which we exercise via DB-backed key.
        from llm_rio.security import ApiKeyVault, default_key_vault_path

        settings = client._transport.app.state.settings
        vault_path = default_key_vault_path(settings.database_path)
        database = client._transport.app.state.database
        row = await database.fetchone(
            """
            SELECT encrypted_api_key FROM api_keys
             WHERE role = 'admin' AND active = 1
             ORDER BY created_at, id LIMIT 1
            """
        )
        admin_token = ApiKeyVault(vault_path).decrypt(str(row["encrypted_api_key"]))

        # Create a plain USER key through the API.
        response = await client.post(
            "/admin/keys",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"nickname": "dev-user", "role": "user", "limit_tokens": 5000},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["nickname"] == "dev-user"
        assert body["api_key"].startswith("rio_")

        # USER key cannot list admin keys.
        response = await client.get(
            "/admin/keys", headers={"Authorization": f"Bearer {body['api_key']}"}
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "permission_denied"

    async def test_usage_endpoint(self, client: AsyncClient) -> None:
        from llm_rio.security import ApiKeyVault, default_key_vault_path

        settings = client._transport.app.state.settings
        vault_path = default_key_vault_path(settings.database_path)
        database = client._transport.app.state.database
        row = await database.fetchone(
            """
            SELECT encrypted_api_key FROM api_keys
             WHERE role = 'admin' AND active = 1
             ORDER BY created_at, id LIMIT 1
            """
        )
        token = ApiKeyVault(vault_path).decrypt(str(row["encrypted_api_key"]))
        response = await client.get(
            "/v1/me/usage", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["account_nickname"] == "admin-account"
        assert body["unlimited"] is True
        assert body["settled_requests"] == 0
