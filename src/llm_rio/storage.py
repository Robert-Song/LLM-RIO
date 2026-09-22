from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from llm_rio.database_schema import SCHEMA
from llm_rio.domain import CatalogState, Role, ServiceMode
from llm_rio.repositories.accounting import AccountingRepository
from llm_rio.repositories.catalog import CatalogRepository
from llm_rio.repositories.events import EventsRepository
from llm_rio.repositories.identity import IdentityRepository
from llm_rio.security import (
    ApiKeyVault,
    Principal,
    default_key_vault_path,
)

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path, key_vault_path: Path | None = None) -> None:
        self.path = path
        self.key_vault_path = key_vault_path or default_key_vault_path(path)
        self._key_vault: ApiKeyVault | None = None
        self._connection: aiosqlite.Connection | None = None
        self.identity = IdentityRepository(self)
        self.catalog = CatalogRepository(self)
        self.accounting = AccountingRepository(self)
        self.events = EventsRepository(self)
        self._transaction_lock = asyncio.Lock()

    @property
    def key_vault(self) -> ApiKeyVault:
        if self._key_vault is None:
            self._key_vault = ApiKeyVault(self.key_vault_path)
        return self._key_vault

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("database is not open")
        return self._connection

    async def open(self) -> None:
        if self._connection is not None:
            raise RuntimeError("database is already open")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.path, isolation_level=None)
        try:
            self._connection.row_factory = aiosqlite.Row
            version = await self.fetchone("PRAGMA user_version")
            tables = await self.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
            if tables and (version is None or version[0] != 1):
                raise RuntimeError(
                    "Beta or unsupported database: archive it and use a new release database path"
                )
            await self.execute("PRAGMA foreign_keys=ON")
            await self.execute("PRAGMA journal_mode=WAL")
            await self.execute("PRAGMA synchronous=NORMAL")
            await self.execute("PRAGMA busy_timeout=5000")
            await self.executescript(SCHEMA)
            await self.execute("PRAGMA user_version=1")
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @asynccontextmanager
    async def transaction(self, *, immediate: bool = True) -> AsyncIterator[aiosqlite.Connection]:
        async with self._transaction_lock:
            connection = self.connection
            if connection.in_transaction:
                logger.error("Rolling back an orphaned SQLite transaction before starting new work")
                await connection.rollback()
            try:
                await connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            except BaseException:
                await connection.rollback()
                raise
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            try:
                await connection.commit()
            except BaseException:
                await connection.rollback()
                raise

    async def execute(self, sql: str, parameters: Iterable[Any] = ()) -> aiosqlite.Cursor:
        async with self._transaction_lock:
            return await self.connection.execute(sql, tuple(parameters))

    async def executescript(self, sql: str) -> aiosqlite.Cursor:
        async with self._transaction_lock:
            return await self.connection.executescript(sql)

    async def fetchone(self, sql: str, parameters: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with (
            self._transaction_lock,
            self.connection.execute(sql, tuple(parameters)) as cursor,
        ):
            return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with (
            self._transaction_lock,
            self.connection.execute(sql, tuple(parameters)) as cursor,
        ):
            return list(await cursor.fetchall())

    async def authenticate(self, prefix: str, token: str) -> Principal | None:
        return await self.identity.authenticate(prefix, token)

    async def key_by_selector(
        self, selector: str, *, active_only: bool = True
    ) -> dict[str, Any] | None:
        return await self.identity.key_by_selector(selector, active_only=active_only)

    async def key_count(self) -> int:
        return await self.identity.key_count()

    async def create_key(
        self,
        *,
        key_id: str,
        nickname: str,
        role: Role,
        account_id: str,
        account_nickname: str,
        prefix: str,
        api_key: str,
        limit_tokens: int,
        unlimited: bool,
    ) -> None:
        return await self.identity.create_key(
            key_id=key_id,
            nickname=nickname,
            role=role,
            account_id=account_id,
            account_nickname=account_nickname,
            prefix=prefix,
            api_key=api_key,
            limit_tokens=limit_tokens,
            unlimited=unlimited,
        )

    async def list_keys(self) -> list[dict[str, Any]]:
        return await self.identity.list_keys()

    async def set_key_active(self, key_id: str, active: bool) -> bool:
        return await self.identity.set_key_active(key_id, active)

    async def replace_key_secret(self, key_id: str, prefix: str, api_key: str) -> bool:
        return await self.identity.replace_key_secret(key_id, prefix, api_key)

    async def delete_key(self, key_id: str) -> bool:
        return await self.identity.delete_key(key_id)

    async def update_quota(self, key_id: str, limit_tokens: int, unlimited: bool) -> bool:
        return await self.identity.update_quota(key_id, limit_tokens, unlimited)

    async def reset_usage(self, key_id: str) -> dict[str, Any] | None:
        return await self.identity.reset_usage(key_id)

    async def create_model_job(
        self,
        *,
        nickname: str,
        repo: str,
        revision: str | None,
        creator_key_id: str,
        grant_key_ids: list[str],
        local_path: str | None = None,
        engine: str = "vllm",
    ) -> tuple[str, str]:
        return await self.catalog.create_model_job(
            nickname=nickname,
            repo=repo,
            revision=revision,
            creator_key_id=creator_key_id,
            grant_key_ids=grant_key_ids,
            local_path=local_path,
            engine=engine,
        )

    async def update_model_job(
        self,
        job_id: str,
        *,
        job_state: str,
        stage: str,
        catalog_state: CatalogState,
        progress: dict[str, Any] | None = None,
        failure: dict[str, Any] | None = None,
        resolved_revision: str | None = None,
        artifact_path: str | None = None,
        capabilities: list[str] | None = None,
    ) -> None:
        return await self.catalog.update_model_job(
            job_id,
            job_state=job_state,
            stage=stage,
            catalog_state=catalog_state,
            progress=progress,
            failure=failure,
            resolved_revision=resolved_revision,
            artifact_path=artifact_path,
            capabilities=capabilities,
        )

    async def get_model_job(self, job_id: str) -> dict[str, Any] | None:
        return await self.catalog.get_model_job(job_id)

    async def set_model_job_validation_overrides(
        self, job_id: str, overrides: dict[str, Any]
    ) -> bool:
        return await self.catalog.set_model_job_validation_overrides(job_id, overrides)

    async def model_by_nickname(self, nickname: str) -> dict[str, Any] | None:
        return await self.catalog.model_by_nickname(nickname)

    async def model_by_id(self, model_id: str) -> dict[str, Any] | None:
        return await self.catalog.model_by_id(model_id)

    async def update_model_request_defaults(
        self, model_id: str, updates: dict[str, Any | None]
    ) -> dict[str, Any] | None:
        return await self.catalog.update_model_request_defaults(model_id, updates)

    @staticmethod
    def _decode_model(row: aiosqlite.Row) -> dict[str, Any]:
        result = dict(row)
        for key in (
            "artifact_hashes_json",
            "capabilities_json",
            "request_limits_json",
            "request_defaults_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    async def list_models(
        self, key_id: str | None = None, *, include_registration_jobs: bool = False
    ) -> list[dict[str, Any]]:
        return await self.catalog.list_models(
            key_id, include_registration_jobs=include_registration_jobs
        )

    async def has_model_grant(self, key_id: str, model_id: str) -> bool:
        return await self.catalog.has_model_grant(key_id, model_id)

    async def replace_model_grants(self, key_id: str, model_ids: list[str]) -> None:
        return await self.catalog.replace_model_grants(key_id, model_ids)

    async def update_model_access(
        self, *, key_id: str, model_nicknames: list[str], mode: str
    ) -> list[str]:
        return await self.catalog.update_model_access(
            key_id=key_id, model_nicknames=model_nicknames, mode=mode
        )

    async def disable_model(self, model_id: str) -> bool:
        return await self.catalog.disable_model(model_id)

    async def reserve_quota(
        self,
        *,
        request_id: str,
        idempotency_hash: str,
        principal: Principal,
        model_id: str,
        estimated_tokens: int,
        estimated_prompt_tokens: int | None = None,
        test_run_id: str | None = None,
        client_worker: str | None = None,
    ) -> str:
        return await self.accounting.reserve_quota(
            request_id=request_id,
            idempotency_hash=idempotency_hash,
            principal=principal,
            model_id=model_id,
            estimated_tokens=estimated_tokens,
            estimated_prompt_tokens=estimated_prompt_tokens,
            test_run_id=test_run_id,
            client_worker=client_worker,
        )

    async def summarize_usage(self, through: datetime | None = None) -> dict[str, Any]:
        return await self.accounting.summarize_usage(through)

    async def dashboard_usage(self) -> dict[str, Any]:
        return await self.accounting.dashboard_usage()

    async def live_requests(self) -> list[dict[str, Any]]:
        return await self.accounting.live_requests()

    async def mark_request_admitted(self, request_id: str, worker_id: str) -> bool:
        return await self.accounting.mark_request_admitted(request_id, worker_id)

    async def admitted_request_ids(self) -> set[str]:
        return await self.accounting.admitted_request_ids()

    async def settle_quota(
        self,
        *,
        reservation_id: str,
        actual_tokens: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        error_code: str | None = None,
    ) -> None:
        return await self.accounting.settle_quota(
            reservation_id=reservation_id,
            actual_tokens=actual_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            error_code=error_code,
        )

    async def release_reservation(self, reservation_id: str, error_code: str) -> None:
        return await self.accounting.release_reservation(reservation_id, error_code)

    async def inference_requests_for_test_run(self, test_run_id: str) -> list[dict[str, Any]]:
        return await self.accounting.inference_requests_for_test_run(test_run_id)

    async def usage(self, principal: Principal) -> dict[str, Any]:
        return await self.accounting.usage(principal)

    async def service_mode(self) -> ServiceMode:
        return await self.events.service_mode()

    async def set_service_mode(self, mode: ServiceMode) -> None:
        return await self.events.set_service_mode(mode)

    async def set_machine_fingerprint(self, fingerprint: str) -> str | None:
        return await self.events.set_machine_fingerprint(fingerprint)

    async def recover_orphaned_state(self) -> None:
        return await self.events.recover_orphaned_state()

    async def record_event(
        self, event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> None:
        return await self.events.record_event(event_type, entity_id, payload)
