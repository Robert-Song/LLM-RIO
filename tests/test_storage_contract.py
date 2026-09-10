from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from llm_rio.storage import Database


class AuthenticationDatabase(Database):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.updates: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchone(self, sql: str, parameters: Iterable[Any] = ()) -> dict[str, Any]:
        return {
            "id": "key-id",
            "nickname": "team",
            "role": "user",
            "quota_account_id": "account-id",
            "token_hash": "stored-hash",
            "unlimited": 1,
        }

    async def execute(self, sql: str, parameters: Iterable[Any] = ()) -> None:
        self.updates.append((sql, tuple(parameters)))


@pytest.mark.asyncio
async def test_authentication_hash_verification_runs_off_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    caller_thread = threading.get_ident()
    verifier_threads: list[int] = []

    def verify(stored_hash: str, token: str) -> bool:
        assert stored_hash == "stored-hash"
        assert token == "presented-token"
        verifier_threads.append(threading.get_ident())
        return True

    monkeypatch.setattr("llm_rio.storage.verify_api_key", verify)
    database = AuthenticationDatabase(tmp_path / "unused.db")

    principal = await database.authenticate("rio_prefix", "presented-token")

    assert principal is not None
    assert principal.key_id == "key-id"
    assert verifier_threads and verifier_threads[0] != caller_thread
    assert len(database.updates) == 1


@pytest.mark.asyncio
async def test_existing_model_catalog_gains_profile_default_columns(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE model_catalog (
                id TEXT PRIMARY KEY,
                nickname TEXT NOT NULL UNIQUE,
                huggingface_repo TEXT NOT NULL,
                requested_revision TEXT,
                resolved_revision TEXT,
                state TEXT NOT NULL,
                artifact_path TEXT,
                artifact_hashes_json TEXT NOT NULL DEFAULT '[]',
                capabilities_json TEXT NOT NULL DEFAULT '[]',
                request_limits_json TEXT NOT NULL DEFAULT '{}',
                created_by_key_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

    database = Database(database_path)
    await database.open()
    try:
        rows = await database.fetchall("PRAGMA table_info(model_catalog)")
        columns = {str(row["name"]): row for row in rows}

        assert columns["request_defaults_json"]["dflt_value"] == "'{}'"
        assert "source_model_id" in columns
    finally:
        await database.close()
