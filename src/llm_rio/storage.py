from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from llm_rio.domain import CatalogState, Role, ServiceMode
from llm_rio.errors import QuotaExceededError, RioError
from llm_rio.security import (
    ApiKeyVault,
    Principal,
    default_key_vault_path,
    hash_api_key,
    verify_api_key,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_accounts (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL UNIQUE,
    balance_tokens INTEGER NOT NULL CHECK (balance_tokens >= 0),
    unlimited INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('user', 'ta', 'admin')),
    quota_account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    token_prefix TEXT NOT NULL UNIQUE,
    token_hash TEXT NOT NULL,
    encrypted_api_key TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_used_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(token_prefix) WHERE active = 1;

CREATE TABLE IF NOT EXISTS model_catalog (
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
    created_by_key_id TEXT NOT NULL REFERENCES api_keys(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_grants (
    key_id TEXT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    model_id TEXT NOT NULL REFERENCES model_catalog(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (key_id, model_id)
);

CREATE TABLE IF NOT EXISTS model_jobs (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    state TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress_json TEXT NOT NULL DEFAULT '{}',
    failure_json TEXT,
    requested_grants_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_profiles (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    machine_fingerprint TEXT NOT NULL,
    profile_key TEXT NOT NULL UNIQUE,
    profile_json TEXT NOT NULL,
    verified_at TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_profiles_model_machine
ON model_profiles(model_id, machine_fingerprint) WHERE active = 1;

CREATE TABLE IF NOT EXISTS quota_reservations (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    idempotency_hash TEXT NOT NULL,
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    key_id TEXT NOT NULL REFERENCES api_keys(id),
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens >= 0),
    actual_tokens INTEGER,
    state TEXT NOT NULL CHECK (state IN ('RESERVED', 'SETTLED', 'RELEASED')),
    created_at TEXT NOT NULL,
    settled_at TEXT,
    UNIQUE (key_id, idempotency_hash)
);

CREATE TABLE IF NOT EXISTS quota_ledger (
    id TEXT PRIMARY KEY,
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    reservation_id TEXT REFERENCES quota_reservations(id),
    delta_tokens INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (reservation_id, reason)
);

CREATE TABLE IF NOT EXISTS inference_requests (
    id TEXT PRIMARY KEY,
    key_id TEXT NOT NULL REFERENCES api_keys(id),
    account_id TEXT NOT NULL REFERENCES quota_accounts(id),
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    reservation_id TEXT NOT NULL REFERENCES quota_reservations(id),
    worker_id TEXT,
    state TEXT NOT NULL,
    estimated_tokens INTEGER NOT NULL,
    actual_prompt_tokens INTEGER,
    actual_completion_tokens INTEGER,
    error_code TEXT,
    created_at TEXT NOT NULL,
    admitted_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL REFERENCES model_catalog(id),
    profile_id TEXT NOT NULL REFERENCES model_profiles(id),
    gpu_uuids_json TEXT NOT NULL,
    port INTEGER NOT NULL,
    pid INTEGER,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runtime_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    entity_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS service_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    mode TEXT NOT NULL,
    machine_fingerprint TEXT,
    updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO service_state(singleton, mode, updated_at) VALUES (1, 'ACTIVE', CURRENT_TIMESTAMP);
"""


class Database:
    def __init__(self, path: Path, key_vault_path: Path | None = None) -> None:
        self.path = path
        self.key_vault_path = key_vault_path or default_key_vault_path(path)
        self.key_vault = ApiKeyVault(self.key_vault_path)
        self._connection: aiosqlite.Connection | None = None
        self._transaction_lock = asyncio.Lock()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("database is not open")
        return self._connection

    async def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.path, isolation_level=None)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA foreign_keys=ON")
        await self._connection.execute("PRAGMA journal_mode=WAL")
        await self._connection.execute("PRAGMA synchronous=NORMAL")
        await self._connection.execute("PRAGMA busy_timeout=5000")
        await self._connection.executescript(SCHEMA)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    @asynccontextmanager
    async def transaction(self, *, immediate: bool = True) -> AsyncIterator[aiosqlite.Connection]:
        async with self._transaction_lock:
            connection = self.connection
            await connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield connection
            except BaseException:
                await connection.rollback()
                raise
            else:
                await connection.commit()

    async def execute(self, sql: str, parameters: Iterable[Any] = ()) -> None:
        async with self._transaction_lock:
            await self.connection.execute(sql, tuple(parameters))

    async def fetchone(self, sql: str, parameters: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with self._transaction_lock:
            cursor = await self.connection.execute(sql, tuple(parameters))
            return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with self._transaction_lock:
            cursor = await self.connection.execute(sql, tuple(parameters))
            return await cursor.fetchall()

    async def authenticate(self, prefix: str, token: str) -> Principal | None:
        row = await self.fetchone(
            """
            SELECT k.id, k.nickname, k.role, k.quota_account_id, k.token_hash, a.unlimited
              FROM api_keys k JOIN quota_accounts a ON a.id = k.quota_account_id
             WHERE k.token_prefix = ? AND k.active = 1
            """,
            (prefix,),
        )
        if row is None or not verify_api_key(row["token_hash"], token):
            return None
        await self.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), row["id"]))
        return Principal(
            key_id=row["id"],
            nickname=row["nickname"],
            role=Role(row["role"]),
            quota_account_id=row["quota_account_id"],
            unlimited=bool(row["unlimited"]),
        )

    async def key_count(self) -> int:
        row = await self.fetchone("SELECT COUNT(*) AS count FROM api_keys")
        return int(row["count"]) if row else 0

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
        balance_tokens: int,
        unlimited: bool,
    ) -> None:
        if role is Role.ADMIN:
            unlimited = True
        async with self.transaction() as connection:
            await connection.execute(
                """
                INSERT OR IGNORE INTO quota_accounts
                    (id, nickname, balance_tokens, unlimited, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (account_id, account_nickname, balance_tokens, int(unlimited), _now()),
            )
            if role is Role.ADMIN:
                await connection.execute(
                    "UPDATE quota_accounts SET unlimited = 1 WHERE id = ?",
                    (account_id,),
                )
            await connection.execute(
                """
                INSERT INTO api_keys
                    (id, nickname, role, quota_account_id, token_prefix, token_hash,
                     encrypted_api_key, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key_id,
                    nickname,
                    role.value,
                    account_id,
                    prefix,
                    hash_api_key(api_key),
                    self.key_vault.encrypt(api_key),
                    _now(),
                ),
            )

    async def list_keys(self) -> list[dict[str, Any]]:
        rows = await self.fetchall(
            """
            SELECT k.id, k.nickname, k.role, k.quota_account_id, k.encrypted_api_key, k.active,
                   k.created_at, k.last_used_at, a.nickname AS account_nickname,
                   a.balance_tokens, a.unlimited
              FROM api_keys k JOIN quota_accounts a ON a.id = k.quota_account_id
             ORDER BY k.nickname
            """
        )
        result = []
        for row in rows:
            item = dict(row)
            item["api_key"] = self.key_vault.decrypt(item.pop("encrypted_api_key"))
            result.append(item)
        return result

    async def set_key_active(self, key_id: str, active: bool) -> bool:
        async with self.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT role, active FROM api_keys WHERE id = ?", (key_id,)
                )
            ).fetchone()
            if row is None:
                return False
            if not active and row["role"] == Role.ADMIN.value and bool(row["active"]):
                count = await (
                    await connection.execute(
                        "SELECT COUNT(*) AS count FROM api_keys WHERE role = ? AND active = 1",
                        (Role.ADMIN.value,),
                    )
                ).fetchone()
                if count is not None and int(count["count"]) <= 1:
                    raise RioError(
                        "last_admin_key",
                        "Create another administrator before revoking the last active admin key",
                        status_code=409,
                    )
            cursor = await connection.execute(
                "UPDATE api_keys SET active = ? WHERE id = ?", (int(active), key_id)
            )
            return cursor.rowcount > 0

    async def replace_key_secret(self, key_id: str, prefix: str, api_key: str) -> bool:
        cursor = await self.connection.execute(
            """
            UPDATE api_keys
               SET token_prefix = ?, token_hash = ?, encrypted_api_key = ?, active = 1
             WHERE id = ?
            """,
            (prefix, hash_api_key(api_key), self.key_vault.encrypt(api_key), key_id),
        )
        return cursor.rowcount > 0

    async def delete_key(self, key_id: str) -> bool:
        async with self.transaction() as connection:
            row = await (await connection.execute(
                "SELECT quota_account_id, role, active FROM api_keys WHERE id = ?", (key_id,)
            )).fetchone()
            if row is None:
                return False
            if row["role"] == Role.ADMIN.value and bool(row["active"]):
                count = await (
                    await connection.execute(
                        "SELECT COUNT(*) AS count FROM api_keys WHERE role = ? AND active = 1",
                        (Role.ADMIN.value,),
                    )
                ).fetchone()
                if count is not None and int(count["count"]) <= 1:
                    raise RioError(
                        "last_admin_key",
                        "Create another administrator before deleting the last active admin key",
                        status_code=409,
                    )
            # Preserve referential/audit history while deleting all credential utility.
            tombstone = f"deleted-{key_id}"
            await connection.execute(
                """
                UPDATE api_keys
                   SET active = 0, nickname = ?, token_prefix = ?, token_hash = ?,
                       encrypted_api_key = ?
                 WHERE id = ?
                """,
                (tombstone, tombstone, tombstone, self.key_vault.encrypt(tombstone), key_id),
            )
            await connection.execute("DELETE FROM model_grants WHERE key_id = ?", (key_id,))
        return True

    async def update_quota(self, key_id: str, balance_tokens: int, unlimited: bool) -> bool:
        async with self.transaction() as connection:
            row = await (await connection.execute(
                "SELECT role, quota_account_id FROM api_keys WHERE id = ?", (key_id,)
            )).fetchone()
            if row is None:
                return False
            if row["role"] == Role.ADMIN.value and not unlimited:
                raise RioError("invalid_quota", "Admin keys must remain unlimited", status_code=409)
            await connection.execute(
                "UPDATE quota_accounts SET balance_tokens = ?, unlimited = ? WHERE id = ?",
                (balance_tokens, int(unlimited), row["quota_account_id"]),
            )
        return True

    async def create_model_job(
        self,
        *,
        nickname: str,
        repo: str,
        revision: str | None,
        creator_key_id: str,
        grant_key_ids: list[str],
    ) -> tuple[str, str]:
        model_id, job_id, now = str(uuid.uuid4()), str(uuid.uuid4()), _now()
        async with self.transaction() as connection:
            if grant_key_ids:
                placeholders = ",".join("?" for _ in grant_key_ids)
                rows = await (await connection.execute(
                    f"SELECT id FROM api_keys WHERE active = 1 AND id IN ({placeholders})",
                    grant_key_ids,
                )).fetchall()
                existing = {row["id"] for row in rows}
                missing = sorted(set(grant_key_ids) - existing)
                if missing:
                    raise RioError(
                        "grant_key_not_found",
                        "One or more requested grant keys do not exist or are inactive",
                        status_code=404,
                        details={"missing_key_ids": missing},
                    )
            await connection.execute(
                """
                INSERT INTO model_catalog
                    (id, nickname, huggingface_repo, requested_revision, state,
                     created_by_key_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    nickname,
                    repo,
                    revision,
                    CatalogState.REQUESTED.value,
                    creator_key_id,
                    now,
                    now,
                ),
            )
            await connection.execute(
                """
                INSERT INTO model_jobs
                    (id, model_id, state, stage, requested_grants_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (job_id, model_id, "QUEUED", "resolve", json.dumps(grant_key_ids), now, now),
            )
        return model_id, job_id

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
        async with self.transaction() as connection:
            job = await (await connection.execute(
                "SELECT model_id FROM model_jobs WHERE id = ?", (job_id,)
            )).fetchone()
            if job is None:
                raise KeyError(job_id)
            await connection.execute(
                """
                UPDATE model_jobs SET state = ?, stage = ?, progress_json = ?, failure_json = ?,
                                      updated_at = ? WHERE id = ?
                """,
                (
                    job_state,
                    stage,
                    json.dumps(progress or {}),
                    json.dumps(failure) if failure else None,
                    _now(),
                    job_id,
                ),
            )
            updates = ["state = ?", "updated_at = ?"]
            values: list[Any] = [catalog_state.value, _now()]
            for column, value in (
                ("resolved_revision", resolved_revision),
                ("artifact_path", artifact_path),
                ("capabilities_json", json.dumps(capabilities) if capabilities is not None else None),
            ):
                if value is not None:
                    updates.append(f"{column} = ?")
                    values.append(value)
            values.append(job["model_id"])
            await connection.execute(
                f"UPDATE model_catalog SET {', '.join(updates)} WHERE id = ?", values
            )

    async def get_model_job(self, job_id: str) -> dict[str, Any] | None:
        row = await self.fetchone(
            """
            SELECT j.*, m.nickname, m.huggingface_repo, m.requested_revision, m.resolved_revision,
                   m.state AS catalog_state
              FROM model_jobs j JOIN model_catalog m ON m.id = j.model_id WHERE j.id = ?
            """,
            (job_id,),
        )
        if row is None:
            return None
        result = dict(row)
        for key in ("progress_json", "failure_json", "requested_grants_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or "null")
        return result

    async def model_by_nickname(self, nickname: str) -> dict[str, Any] | None:
        row = await self.fetchone("SELECT * FROM model_catalog WHERE nickname = ?", (nickname,))
        if row is None:
            return None
        return self._decode_model(row)

    async def model_by_id(self, model_id: str) -> dict[str, Any] | None:
        row = await self.fetchone("SELECT * FROM model_catalog WHERE id = ?", (model_id,))
        return self._decode_model(row) if row else None

    @staticmethod
    def _decode_model(row: aiosqlite.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("artifact_hashes_json", "capabilities_json", "request_limits_json"):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        return result

    async def list_models(self, key_id: str | None = None) -> list[dict[str, Any]]:
        if key_id is None:
            rows = await self.fetchall("SELECT * FROM model_catalog ORDER BY nickname")
        else:
            rows = await self.fetchall(
                """
                SELECT m.* FROM model_catalog m
                  JOIN model_grants g ON g.model_id = m.id
                 WHERE g.key_id = ? AND m.state = ? ORDER BY m.nickname
                """,
                (key_id, CatalogState.AVAILABLE.value),
            )
        return [self._decode_model(row) for row in rows]

    async def has_model_grant(self, key_id: str, model_id: str) -> bool:
        row = await self.fetchone(
            "SELECT 1 FROM model_grants WHERE key_id = ? AND model_id = ?", (key_id, model_id)
        )
        return row is not None

    async def replace_model_grants(self, key_id: str, model_ids: list[str]) -> None:
        async with self.transaction() as connection:
            key = await (await connection.execute(
                "SELECT 1 FROM api_keys WHERE id = ?", (key_id,)
            )).fetchone()
            if key is None:
                raise KeyError(key_id)
            await connection.execute("DELETE FROM model_grants WHERE key_id = ?", (key_id,))
            await connection.executemany(
                "INSERT INTO model_grants(key_id, model_id, created_at) VALUES (?, ?, ?)",
                [(key_id, model_id, _now()) for model_id in model_ids],
            )

    async def disable_model(self, model_id: str) -> bool:
        cursor = await self.connection.execute(
            "UPDATE model_catalog SET state = ?, updated_at = ? WHERE id = ?",
            (CatalogState.DISABLED.value, _now(), model_id),
        )
        return cursor.rowcount > 0

    async def reserve_quota(
        self,
        *,
        request_id: str,
        idempotency_hash: str,
        principal: Principal,
        model_id: str,
        estimated_tokens: int,
    ) -> str:
        async with self.transaction() as connection:
            existing = await (await connection.execute(
                """
                SELECT id, request_id FROM quota_reservations
                 WHERE key_id = ? AND idempotency_hash = ?
                """,
                (principal.key_id, idempotency_hash),
            )).fetchone()
            if existing:
                if existing["request_id"] != request_id:
                    raise RioError(
                        "idempotency_conflict",
                        "The idempotency key was already used for another request",
                        status_code=409,
                    )
                return str(existing["id"])
            account = await (await connection.execute(
                "SELECT balance_tokens, unlimited FROM quota_accounts WHERE id = ?",
                (principal.quota_account_id,),
            )).fetchone()
            if account is None:
                raise RuntimeError("quota account is missing")
            unlimited = bool(account["unlimited"])
            balance = int(account["balance_tokens"])
            if not unlimited and balance < estimated_tokens:
                raise QuotaExceededError(balance, estimated_tokens)
            reservation_id = str(uuid.uuid4())
            await connection.execute(
                """
                INSERT INTO quota_reservations
                    (id, request_id, idempotency_hash, account_id, key_id, model_id,
                     reserved_tokens, state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'RESERVED', ?)
                """,
                (
                    reservation_id,
                    request_id,
                    idempotency_hash,
                    principal.quota_account_id,
                    principal.key_id,
                    model_id,
                    estimated_tokens,
                    _now(),
                ),
            )
            if not unlimited:
                await connection.execute(
                    "UPDATE quota_accounts SET balance_tokens = balance_tokens - ? WHERE id = ?",
                    (estimated_tokens, principal.quota_account_id),
                )
                await connection.execute(
                    """
                    INSERT INTO quota_ledger
                        (id, account_id, reservation_id, delta_tokens, reason, created_at)
                    VALUES (?, ?, ?, ?, 'reservation', ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        principal.quota_account_id,
                        reservation_id,
                        -estimated_tokens,
                        _now(),
                    ),
                )
        return reservation_id

    async def create_inference_request(
        self,
        *,
        request_id: str,
        principal: Principal,
        model_id: str,
        reservation_id: str,
        estimated_tokens: int,
    ) -> None:
        await self.execute(
            """
            INSERT OR IGNORE INTO inference_requests
                (id, key_id, account_id, model_id, reservation_id, state,
                 estimated_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, 'QUEUED', ?, ?)
            """,
            (
                request_id,
                principal.key_id,
                principal.quota_account_id,
                model_id,
                reservation_id,
                estimated_tokens,
                _now(),
            ),
        )

    async def mark_request_admitted(self, request_id: str, worker_id: str) -> None:
        await self.execute(
            """
            UPDATE inference_requests SET state = 'ADMITTED', worker_id = ?, admitted_at = ?
             WHERE id = ? AND state = 'QUEUED'
            """,
            (worker_id, _now(), request_id),
        )

    async def settle_quota(
        self,
        *,
        reservation_id: str,
        actual_tokens: int,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        error_code: str | None = None,
    ) -> None:
        actual_tokens = max(0, actual_tokens)
        async with self.transaction() as connection:
            reservation = await (await connection.execute(
                "SELECT * FROM quota_reservations WHERE id = ?", (reservation_id,)
            )).fetchone()
            if reservation is None or reservation["state"] != "RESERVED":
                return
            account = await (await connection.execute(
                "SELECT unlimited FROM quota_accounts WHERE id = ?", (reservation["account_id"],)
            )).fetchone()
            reserved = int(reservation["reserved_tokens"])
            charged = min(actual_tokens, reserved)
            refund = reserved - charged
            if account is not None and not bool(account["unlimited"]) and refund:
                await connection.execute(
                    "UPDATE quota_accounts SET balance_tokens = balance_tokens + ? WHERE id = ?",
                    (refund, reservation["account_id"]),
                )
                await connection.execute(
                    """
                    INSERT OR IGNORE INTO quota_ledger
                        (id, account_id, reservation_id, delta_tokens, reason, created_at)
                    VALUES (?, ?, ?, ?, 'settlement_refund', ?)
                    """,
                    (str(uuid.uuid4()), reservation["account_id"], reservation_id, refund, _now()),
                )
            await connection.execute(
                """
                UPDATE quota_reservations SET actual_tokens = ?, state = 'SETTLED', settled_at = ?
                 WHERE id = ? AND state = 'RESERVED'
                """,
                (charged, _now(), reservation_id),
            )
            state = "FAILED" if error_code else "COMPLETED"
            await connection.execute(
                """
                UPDATE inference_requests
                   SET state = ?, actual_prompt_tokens = ?, actual_completion_tokens = ?,
                       error_code = ?, completed_at = ?
                 WHERE reservation_id = ?
                """,
                (state, prompt_tokens, completion_tokens, error_code, _now(), reservation_id),
            )

    async def release_reservation(self, reservation_id: str, error_code: str) -> None:
        await self.settle_quota(reservation_id=reservation_id, actual_tokens=0, error_code=error_code)

    async def usage(self, principal: Principal) -> dict[str, Any]:
        account = await self.fetchone(
            "SELECT nickname, balance_tokens, unlimited FROM quota_accounts WHERE id = ?",
            (principal.quota_account_id,),
        )
        totals = await self.fetchone(
            """
            SELECT COALESCE(SUM(actual_tokens), 0) AS charged_tokens,
                   COUNT(*) AS settled_requests
              FROM quota_reservations WHERE account_id = ? AND state = 'SETTLED'
            """,
            (principal.quota_account_id,),
        )
        return {
            "account_id": principal.quota_account_id,
            "account_nickname": account["nickname"] if account else None,
            "balance_tokens": account["balance_tokens"] if account else 0,
            "unlimited": bool(account["unlimited"]) if account else False,
            "charged_tokens": int(totals["charged_tokens"]) if totals else 0,
            "settled_requests": int(totals["settled_requests"]) if totals else 0,
        }

    async def service_mode(self) -> ServiceMode:
        row = await self.fetchone("SELECT mode FROM service_state WHERE singleton = 1")
        return ServiceMode(row["mode"] if row else ServiceMode.ACTIVE.value)

    async def set_service_mode(self, mode: ServiceMode) -> None:
        await self.execute(
            "UPDATE service_state SET mode = ?, updated_at = ? WHERE singleton = 1",
            (mode.value, _now()),
        )
        await self.record_event("SERVICE_MODE_CHANGED", payload={"mode": mode.value})

    async def set_machine_fingerprint(self, fingerprint: str) -> str | None:
        async with self.transaction() as connection:
            row = await (await connection.execute(
                "SELECT machine_fingerprint FROM service_state WHERE singleton = 1"
            )).fetchone()
            previous = row["machine_fingerprint"] if row else None
            if previous and previous != fingerprint:
                await connection.execute(
                    "UPDATE model_profiles SET active = 0 WHERE machine_fingerprint != ?",
                    (fingerprint,),
                )
            await connection.execute(
                "UPDATE service_state SET machine_fingerprint = ?, updated_at = ? WHERE singleton = 1",
                (fingerprint, _now()),
            )
        return previous

    async def recover_orphaned_state(self) -> None:
        """Begin cold and refund reservations that cannot have a live local request."""
        async with self.transaction() as connection:
            reservations = await (await connection.execute(
                "SELECT id FROM quota_reservations WHERE state = 'RESERVED'"
            )).fetchall()
        for reservation in reservations:
            await self.release_reservation(reservation["id"], "service_restarted")
        await self.execute(
            "UPDATE workers SET state = 'COLD', pid = NULL, updated_at = ? WHERE state != 'COLD'",
            (_now(),),
        )
        await self.execute(
            """
            UPDATE model_jobs SET state = 'QUEUED', stage = 'validation_requeued', updated_at = ?
             WHERE state = 'RUNNING' AND stage LIKE 'validat%'
            """,
            (_now(),),
        )

    async def record_event(
        self, event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> None:
        await self.execute(
            """
            INSERT INTO runtime_events(event_type, entity_id, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (event_type, entity_id, json.dumps(payload or {}), _now()),
        )

