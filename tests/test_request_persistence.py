from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from llm_rio.domain import Role
from llm_rio.errors import RioError
from llm_rio.security import Principal
from llm_rio.storage import Database

PRINCIPAL = Principal("key", "user", Role.USER, "account", False)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "test.db")
    await database.open()
    try:
        await database.create_key(
            key_id="key",
            nickname="user",
            role=Role.USER,
            account_id="account",
            account_nickname="account",
            prefix="rio_test_prefix_123456789",
            api_key="rio_test_secret_123456789",
            limit_tokens=1000,
            unlimited=False,
        )
        await database.create_model_job(
            nickname="model",
            repo="org/model",
            revision=None,
            creator_key_id="key",
            grant_key_ids=[],
        )
        yield database
    finally:
        await database.close()


async def reserve(database: Database, request_id: str = "request", key: str = "hash") -> str:
    model = await database.model_by_nickname("model")
    assert model is not None
    return await database.reserve_quota(
        request_id=request_id,
        idempotency_hash=key,
        principal=PRINCIPAL,
        model_id=model["id"],
        estimated_tokens=100,
        estimated_prompt_tokens=12,
        test_run_id="regression",
        client_worker="client",
    )


async def test_reservation_creates_queued_request_atomically(database: Database) -> None:
    reservation_id = await reserve(database)
    row = await database.fetchone("SELECT * FROM inference_requests")
    assert row is not None
    assert row["reservation_id"] == reservation_id
    assert row["state"] == "QUEUED"
    assert row["estimated_prompt_tokens"] == 12
    assert row["test_run_id"] == "regression"
    assert (await database.usage(PRINCIPAL))["balance_tokens"] == 900


async def test_failed_request_insert_rolls_back_quota_and_ledger(database: Database) -> None:
    await database.executescript("""
        CREATE TRIGGER fail_request BEFORE INSERT ON inference_requests
        BEGIN SELECT RAISE(ABORT, 'request insert failed'); END;
    """)
    with pytest.raises(sqlite3.IntegrityError, match="request insert failed"):
        await reserve(database)
    assert await database.fetchall("SELECT * FROM quota_reservations") == []
    assert await database.fetchall("SELECT * FROM quota_ledger") == []
    assert (await database.usage(PRINCIPAL))["balance_tokens"] == 1000


@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize(
    "request_id,key", [("request", "hash"), ("new", "hash"), ("request", "new")]
)
async def test_duplicate_request_cannot_reuse_or_charge_reservation(
    database: Database,
    settled: bool,
    request_id: str,
    key: str,
) -> None:
    reservation_id = await reserve(database)
    if settled:
        await database.settle_quota(reservation_id=reservation_id, actual_tokens=20)
    before = await database.usage(PRINCIPAL)
    with pytest.raises(RioError) as error:
        await reserve(database, request_id, key)
    assert error.value.status_code == 409
    assert await database.usage(PRINCIPAL) == before
    assert len(await database.fetchall("SELECT * FROM inference_requests")) == 1


async def test_concurrent_duplicates_only_reserve_once(database: Database) -> None:
    results = await asyncio.gather(reserve(database), reserve(database), return_exceptions=True)
    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, RioError) for result in results) == 1
    assert (await database.usage(PRINCIPAL))["balance_tokens"] == 900


async def test_cancelled_transaction_rolls_back_and_connection_remains_usable(
    database: Database,
) -> None:
    entered = asyncio.Event()

    async def write() -> None:
        async with database.transaction() as connection:
            await connection.execute("UPDATE quota_accounts SET balance_tokens = 0")
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(write())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await database.usage(PRINCIPAL))["balance_tokens"] == 1000
    await reserve(database)


async def test_failed_open_closes_connection_and_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database(tmp_path / "failed.db")
    migrate = database._migrate_schema

    async def fail() -> None:
        raise RuntimeError("migration failed")

    monkeypatch.setattr(database, "_migrate_schema", fail)
    with pytest.raises(RuntimeError, match="migration failed"):
        await database.open()
    with pytest.raises(RuntimeError, match="database is not open"):
        _ = database.connection
    monkeypatch.setattr(database, "_migrate_schema", migrate)
    await database.open()
    try:
        assert await database.key_count() == 0
        with pytest.raises(RuntimeError, match="already open"):
            await database.open()
    finally:
        await database.close()
