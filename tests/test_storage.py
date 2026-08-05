"""Database integration tests against a temporary SQLite file (no GPU required)."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import pytest_asyncio

from llm_rio.domain import CatalogState, Role, ServiceMode
from llm_rio.errors import QuotaExceededError, RioError
from llm_rio.security import Principal, issue_api_key
from llm_rio.storage import Database


@pytest_asyncio.fixture
async def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.db")
    await db.open()
    yield db
    await db.close()


async def create_user_key(
    database: Database,
    *,
    nickname: str = "alice",
    limit_tokens: int = 1000,
    unlimited: bool = False,
    role: Role = Role.USER,
) -> tuple[Principal, str]:
    key_id, account_id = str(uuid.uuid4()), str(uuid.uuid4())
    token, prefix = issue_api_key(key_id)
    await database.create_key(
        key_id=key_id,
        nickname=nickname,
        role=role,
        account_id=account_id,
        account_nickname=f"{nickname}-account",
        prefix=prefix,
        api_key=token,
        limit_tokens=limit_tokens,
        unlimited=unlimited,
    )
    principal = await database.authenticate(prefix, token)
    assert principal is not None
    return principal, token


async def create_model(database: Database, key_id: str, nickname: str = "model-a") -> str:
    model_id = str(uuid.uuid4())
    await database.execute(
        """
        INSERT INTO model_catalog
            (id, nickname, huggingface_repo, state, created_by_key_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (model_id, nickname, "org/model", CatalogState.AVAILABLE.value, key_id, "now", "now"),
    )
    return model_id


class TestDatabaseLifecycle:
    async def test_open_creates_schema(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "new.db")
        await db.open()
        try:
            tables = await db.fetchall(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
            names = {row["name"] for row in tables}
            assert {
                "api_keys",
                "quota_accounts",
                "model_catalog",
                "model_jobs",
                "model_profiles",
                "quota_reservations",
                "quota_ledger",
                "inference_requests",
                "workers",
                "runtime_events",
                "service_state",
            } <= names
        finally:
            await db.close()

    async def test_default_service_mode_is_active(self, database: Database) -> None:
        assert await database.service_mode() is ServiceMode.ACTIVE

    async def test_closed_database_raises(self, tmp_path: Path) -> None:
        db = Database(tmp_path / "closed.db")
        with pytest.raises(RuntimeError, match="not open"):
            _ = db.connection
        await db.close()


class TestKeys:
    async def test_create_and_authenticate(self, database: Database) -> None:
        principal, token = await create_user_key(database)
        assert principal.nickname == "alice"
        assert principal.role is Role.USER
        assert principal.unlimited is False

    async def test_authenticate_rejects_wrong_token(self, database: Database) -> None:
        _, token = await create_user_key(database)
        prefix = token[:24]
        assert await database.authenticate(prefix, "rio_wrong_token") is None

    async def test_authenticate_ignores_inactive_key(self, database: Database) -> None:
        principal, token = await create_user_key(database, nickname="bob")
        await database.set_key_active(principal.key_id, False)
        assert await database.authenticate(token[:24], token) is None

    async def test_key_count(self, database: Database) -> None:
        assert await database.key_count() == 0
        await create_user_key(database)
        await create_user_key(database, nickname="bob")
        assert await database.key_count() == 2

    async def test_key_by_selector_nickname(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        record = await database.key_by_selector("alice")
        assert record is not None and record["id"] == principal.key_id

    async def test_key_by_selector_full_token(self, database: Database) -> None:
        _, token = await create_user_key(database)
        record = await database.key_by_selector(token)
        assert record is not None and record["nickname"] == "alice"

    async def test_list_keys_decrypts_api_key(self, database: Database) -> None:
        _, token = await create_user_key(database)
        records = await database.list_keys()
        assert len(records) == 1
        assert records[0]["api_key"] == token

    async def test_delete_key_tombstones(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        assert await database.delete_key(principal.key_id) is True
        # Tombstoned row remains; the key is inactive and unusable.
        assert await database.key_count() == 1
        row = await database.fetchone(
            "SELECT active, nickname FROM api_keys WHERE id = ?", (principal.key_id,)
        )
        assert row is not None
        assert row["active"] == 0
        assert row["nickname"].startswith("deleted-")

    async def test_last_active_admin_cannot_be_revoked(self, database: Database) -> None:
        principal, _ = await create_user_key(database, role=Role.ADMIN, unlimited=True)
        with pytest.raises(RioError, match="Create another administrator"):
            await database.set_key_active(principal.key_id, False)

    async def test_second_admin_can_be_revoked(self, database: Database) -> None:
        admin_one, _ = await create_user_key(
            database, nickname="admin1", role=Role.ADMIN, unlimited=True
        )
        admin_two, _ = await create_user_key(
            database, nickname="admin2", role=Role.ADMIN, unlimited=True
        )
        assert await database.set_key_active(admin_two.key_id, False) is True
        # admin_one is now the last active admin and must be protected.
        with pytest.raises(RioError, match="Create another administrator"):
            await database.set_key_active(admin_one.key_id, False)


class TestQuota:
    async def test_reserve_settle_refund(self, database: Database) -> None:
        principal, _ = await create_user_key(database, limit_tokens=1000)
        model_id = await create_model(database, principal.key_id)
        reservation_id = await database.reserve_quota(
            request_id="req-1",
            idempotency_hash="hash-1",
            principal=principal,
            model_id=model_id,
            estimated_tokens=100,
        )
        usage = await database.usage(principal)
        assert usage["balance_tokens"] == 900
        # Settlement with fewer actual tokens refunds the difference.
        await database.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=40,
            prompt_tokens=10,
            completion_tokens=30,
        )
        usage = await database.usage(principal)
        assert usage["balance_tokens"] == 960
        assert usage["used_tokens"] == 40
        assert usage["settled_requests"] == 1

    async def test_idempotent_reservation_returns_same_id(
        self, database: Database
    ) -> None:
        principal, _ = await create_user_key(database)
        model_id = await create_model(database, principal.key_id)
        first = await database.reserve_quota(
            request_id="req-1",
            idempotency_hash="hash-1",
            principal=principal,
            model_id=model_id,
            estimated_tokens=100,
        )
        second = await database.reserve_quota(
            request_id="req-1",
            idempotency_hash="hash-1",
            principal=principal,
            model_id=model_id,
            estimated_tokens=100,
        )
        assert first == second

    async def test_idempotency_key_conflict(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        model_id = await create_model(database, principal.key_id)
        await database.reserve_quota(
            request_id="req-1",
            idempotency_hash="hash-1",
            principal=principal,
            model_id=model_id,
            estimated_tokens=100,
        )
        with pytest.raises(RioError, match="already used for another request"):
            await database.reserve_quota(
                request_id="req-2",
                idempotency_hash="hash-1",
                principal=principal,
                model_id=model_id,
                estimated_tokens=100,
            )

    async def test_quota_exceeded(self, database: Database) -> None:
        principal, _ = await create_user_key(database, limit_tokens=100)
        model_id = await create_model(database, principal.key_id)
        with pytest.raises(QuotaExceededError):
            await database.reserve_quota(
                request_id="req-1",
                idempotency_hash="hash-1",
                principal=principal,
                model_id=model_id,
                estimated_tokens=500,
            )

    async def test_unlimited_key_never_exceeds(self, database: Database) -> None:
        principal, _ = await create_user_key(database, unlimited=True, limit_tokens=100)
        model_id = await create_model(database, principal.key_id)
        reservation_id = await database.reserve_quota(
            request_id="req-1",
            idempotency_hash="hash-1",
            principal=principal,
            model_id=model_id,
            estimated_tokens=10_000_000,
        )
        assert reservation_id
        usage = await database.usage(principal)
        assert usage["unlimited"] is True

    async def test_release_reservation_refunds(self, database: Database) -> None:
        principal, _ = await create_user_key(database, limit_tokens=1000)
        model_id = await create_model(database, principal.key_id)
        reservation_id = await database.reserve_quota(
            request_id="req-1",
            idempotency_hash="hash-1",
            principal=principal,
            model_id=model_id,
            estimated_tokens=100,
        )
        await database.release_reservation(reservation_id, "client_cancelled")
        usage = await database.usage(principal)
        assert usage["balance_tokens"] == 1000
        assert usage["settled_requests"] == 1


class TestModels:
    async def test_model_lifecycle(self, database: Database) -> None:
        principal, _ = await create_user_key(database, role=Role.ADMIN, unlimited=True)
        model_id, job_id = await database.create_model_job(
            nickname="my-model",
            repo="org/model",
            revision=None,
            creator_key_id=principal.key_id,
            grant_key_ids=[principal.key_id],
        )
        job = await database.get_model_job(job_id)
        assert job is not None
        assert job["state"] == "QUEUED"
        assert job["model_id"] == model_id

        await database.update_model_job(
            job_id,
            job_state="RUNNING",
            stage="download",
            catalog_state=CatalogState.DOWNLOADING,
        )
        job = await database.get_model_job(job_id)
        assert job["state"] == "RUNNING"
        assert job["catalog_state"] == CatalogState.DOWNLOADING.value

    async def test_model_by_nickname_and_id(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        model_id = await create_model(database, principal.key_id, nickname="model-a")
        by_nickname = await database.model_by_nickname("model-a")
        by_id = await database.model_by_id(model_id)
        assert by_nickname is not None and by_nickname["id"] == model_id
        assert by_id is not None and by_id["nickname"] == "model-a"
        assert await database.model_by_nickname("missing") is None

    async def test_grants_filter_available_models(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        model_id = await create_model(database, principal.key_id, nickname="model-a")
        # Without a grant, USER keys see nothing.
        assert await database.list_models(principal.key_id) == []
        await database.replace_model_grants(principal.key_id, [model_id])
        models = await database.list_models(principal.key_id)
        assert len(models) == 1
        assert models[0]["nickname"] == "model-a"

    async def test_update_model_access_modes(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        await create_model(database, principal.key_id, nickname="model-a")
        await create_model(database, principal.key_id, nickname="model-b")
        await database.update_model_access(
            key_id=principal.key_id, model_nicknames=["model-a"], mode="add"
        )
        granted = await database.update_model_access(
            key_id=principal.key_id, model_nicknames=["model-b"], mode="add"
        )
        assert granted == ["model-a", "model-b"]
        removed = await database.update_model_access(
            key_id=principal.key_id, model_nicknames=["model-a"], mode="remove"
        )
        assert removed == ["model-b"]
        replaced = await database.update_model_access(
            key_id=principal.key_id, model_nicknames=["model-a"], mode="replace"
        )
        assert replaced == ["model-a"]

    async def test_update_model_access_missing_model(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        with pytest.raises(RioError, match="do not exist"):
            await database.update_model_access(
                key_id=principal.key_id, model_nicknames=["ghost"], mode="add"
            )

    async def test_disable_model(self, database: Database) -> None:
        principal, _ = await create_user_key(database)
        model_id = await create_model(database, principal.key_id)
        assert await database.disable_model(model_id) is True
        assert await database.disable_model("missing") is False
        model = await database.model_by_id(model_id)
        assert model is not None and model["state"] == CatalogState.DISABLED.value


class TestEvents:
    async def test_record_event_round_trip(self, database: Database) -> None:
        await database.record_event("WORKER_READY", "worker-1", {"gpus": 2})
        rows = await database.fetchall(
            "SELECT event_type, entity_id, payload_json FROM runtime_events"
        )
        assert len(rows) == 1
        assert rows[0]["event_type"] == "WORKER_READY"
        assert rows[0]["entity_id"] == "worker-1"
        assert '"gpus": 2' in rows[0]["payload_json"]
