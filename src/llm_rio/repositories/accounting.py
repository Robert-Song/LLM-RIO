from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from llm_rio.errors import QuotaExceededError, RioError
from llm_rio.security import (
    Principal,
)
from llm_rio.timestamps import now as _now
from llm_rio.usage_summary import summarize_usage_records
from llm_rio.usage_summary import usage_dashboard as build_usage_dashboard

if TYPE_CHECKING:
    from llm_rio.storage import Database

logger = logging.getLogger(__name__)


class AccountingRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

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
        """Reserve tokens and create the queued request atomically; never replay inference."""
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens must be nonnegative")
        async with self.database.transaction() as connection:
            existing = await (
                await connection.execute(
                    (
                        "\n"
                        "                SELECT id, request_id FROM quota_reservations\n"
                        "                 WHERE key_id = ? AND idempotency_hash = ?\n"
                        "                "
                    ),
                    (principal.key_id, idempotency_hash),
                )
            ).fetchone()
            if existing:
                raise RioError(
                    "idempotency_conflict",
                    "The idempotency key was already used for a request",
                    status_code=409,
                )
            duplicate = await (
                await connection.execute(
                    (
                        "SELECT 1 FROM quota_reservations WHERE request_id = ? UNION ALL "
                        "SELECT 1 FROM inference_requests WHERE id = ?"
                    ),
                    (request_id, request_id),
                )
            ).fetchone()
            if duplicate:
                raise RioError(
                    "request_id_conflict",
                    "The request ID was already used for a request",
                    status_code=409,
                )
            account = await (
                await connection.execute(
                    "SELECT balance_tokens, unlimited FROM quota_accounts WHERE id = ?",
                    (principal.quota_account_id,),
                )
            ).fetchone()
            if account is None:
                raise RuntimeError("quota account is missing")
            unlimited = bool(account["unlimited"])
            balance = int(account["balance_tokens"])
            if not unlimited and balance < estimated_tokens:
                raise QuotaExceededError(balance, estimated_tokens)
            reservation_id = str(uuid.uuid4())
            await connection.execute(
                (
                    "\n"
                    "                INSERT INTO quota_reservations\n"
                    "                    (id, request_id, idempotency_hash, account_id"
                    ", key_id, model_id,\n"
                    "                     reserved_tokens, state, created_at)\n"
                    "                VALUES (?, ?, ?, ?, ?, ?, ?, 'RESERVED', ?)\n"
                    "                "
                ),
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
                    (
                        "\n"
                        "                    INSERT INTO quota_ledger\n"
                        "                        (id, account_id, reservation_id, delta_to"
                        "kens, reason, created_at)\n"
                        "                    VALUES (?, ?, ?, ?, 'reservation', ?)\n"
                        "                    "
                    ),
                    (
                        str(uuid.uuid4()),
                        principal.quota_account_id,
                        reservation_id,
                        -estimated_tokens,
                        _now(),
                    ),
                )
            await connection.execute(
                (
                    "\n"
                    "                INSERT INTO inference_requests\n"
                    "                    (id, key_id, account_id, model_id, reservatio"
                    "n_id, state,\n"
                    "                     estimated_tokens, estimated_prompt_tokens, t"
                    "est_run_id, client_worker,\n"
                    "                     created_at)\n"
                    "                VALUES (?, ?, ?, ?, ?, 'QUEUED', ?, ?, ?, ?, ?)\n"
                    "                "
                ),
                (
                    request_id,
                    principal.key_id,
                    principal.quota_account_id,
                    model_id,
                    reservation_id,
                    estimated_tokens,
                    estimated_prompt_tokens,
                    test_run_id,
                    client_worker,
                    _now(),
                ),
            )
        return reservation_id

    async def summarize_usage(self, through: datetime | None = None) -> dict[str, Any]:
        cutoff = through or datetime.now(UTC)
        async with self.database.transaction() as connection:
            return await summarize_usage_records(connection, through=cutoff)

    async def dashboard_usage(self) -> dict[str, Any]:
        async with self.database._transaction_lock:
            return await build_usage_dashboard(self.database.connection, now=datetime.now(UTC))

    async def live_requests(self) -> list[dict[str, Any]]:
        """Return requests that are still waiting for or using a worker."""
        rows = await self.database.fetchall(
            "\n"
            "            SELECT r.id AS request_id, r.state, k.nickname AS api"
            "_key,\n"
            "                   m.nickname AS model, r.estimated_prompt_tokens"
            ",\n"
            "                   r.estimated_tokens, r.created_at, r.admitted_a"
            "t, r.worker_id\n"
            "              FROM inference_requests r\n"
            "              JOIN api_keys k ON k.id = r.key_id\n"
            "              JOIN model_catalog m ON m.id = r.model_id\n"
            "             WHERE r.state IN ('QUEUED', 'ADMITTED')\n"
            "             ORDER BY CASE r.state WHEN 'QUEUED' THEN 0 ELSE 1 EN"
            "D,\n"
            "                      r.created_at, r.id\n"
            "            "
        )
        return [dict(row) for row in rows]

    async def mark_request_admitted(self, request_id: str, worker_id: str) -> bool:
        async with (
            self.database._transaction_lock,
            self.database.connection.execute(
                (
                    "\n"
                    "                UPDATE inference_requests\n"
                    "                   SET state = 'ADMITTED', worker_id = ?, admitte"
                    "d_at = ?,\n"
                    "                       accepted_count = accepted_count + 1\n"
                    "                 WHERE id = ? AND state = 'QUEUED'\n"
                    "                "
                ),
                (worker_id, _now(), request_id),
            ) as cursor,
        ):
            return cursor.rowcount == 1

    async def admitted_request_ids(self) -> set[str]:
        rows = await self.database.fetchall(
            "SELECT id FROM inference_requests WHERE state = 'ADMITTED'"
        )
        return {str(row["id"]) for row in rows}

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
        async with self.database.transaction() as connection:
            reservation = await (
                await connection.execute(
                    "SELECT * FROM quota_reservations WHERE id = ?", (reservation_id,)
                )
            ).fetchone()
            if reservation is None or reservation["state"] != "RESERVED":
                return
            account = await (
                await connection.execute(
                    "SELECT unlimited FROM quota_accounts WHERE id = ?",
                    (reservation["account_id"],),
                )
            ).fetchone()
            reserved = int(reservation["reserved_tokens"])
            charged = min(actual_tokens, reserved)
            refund = reserved - charged
            if account is not None and (not bool(account["unlimited"])) and refund:
                await connection.execute(
                    "UPDATE quota_accounts SET balance_tokens = balance_tokens + ? WHERE id = ?",
                    (refund, reservation["account_id"]),
                )
                await connection.execute(
                    (
                        "\n"
                        "                    INSERT OR IGNORE INTO quota_ledger\n"
                        "                        (id, account_id, reservation_id, delta_to"
                        "kens, reason, created_at)\n"
                        "                    VALUES (?, ?, ?, ?, 'settlement_refund', ?)\n"
                        "                    "
                    ),
                    (str(uuid.uuid4()), reservation["account_id"], reservation_id, refund, _now()),
                )
            await connection.execute(
                (
                    "\n"
                    "                UPDATE quota_reservations SET actual_tokens = ?, "
                    "state = 'SETTLED', settled_at = ?\n"
                    "                 WHERE id = ? AND state = 'RESERVED'\n"
                    "                "
                ),
                (charged, _now(), reservation_id),
            )
            state = "FAILED" if error_code else "COMPLETED"
            await connection.execute(
                (
                    "\n"
                    "                UPDATE inference_requests\n"
                    "                   SET state = ?, actual_prompt_tokens = ?, actua"
                    "l_completion_tokens = ?,\n"
                    "                       error_code = ?, completed_at = ?,\n"
                    "                       completion_count = completion_count + 1\n"
                    "                 WHERE reservation_id = ?\n"
                    "                "
                ),
                (state, prompt_tokens, completion_tokens, error_code, _now(), reservation_id),
            )

    async def release_reservation(self, reservation_id: str, error_code: str) -> None:
        await self.database.settle_quota(
            reservation_id=reservation_id, actual_tokens=0, error_code=error_code
        )

    async def inference_requests_for_test_run(self, test_run_id: str) -> list[dict[str, Any]]:
        rows = await self.database.fetchall(
            """
            SELECT r.id AS request_id, r.test_run_id, r.client_worker,
                   r.account_id, r.key_id, m.nickname AS model, r.worker_id,
                   r.state AS completion_status, r.error_code,
                   r.estimated_tokens, r.actual_prompt_tokens, r.actual_completion_tokens,
                   r.accepted_count, r.completion_count,
                   r.created_at AS admission_time, r.admitted_at AS worker_accepted_time,
                   r.completed_at AS completion_time, w.gpu_uuids_json,
                   json_extract(p.profile_json, '$.tensor_parallel_size') AS tensor_parallel_size,
                   q.id AS reservation_id, q.state AS reservation_state,
                   q.reserved_tokens, q.actual_tokens AS charged_tokens,
                   a.unlimited AS account_unlimited,
                   COALESCE((SELECT SUM(delta_tokens) FROM quota_ledger l
                             WHERE l.reservation_id=q.id), 0) AS ledger_delta_tokens
              FROM inference_requests r
              JOIN model_catalog m ON m.id=r.model_id
              JOIN quota_reservations q ON q.id=r.reservation_id
              JOIN quota_accounts a ON a.id=r.account_id
              LEFT JOIN workers w ON w.id=r.worker_id
              LEFT JOIN model_profiles p ON p.id=w.profile_id
             WHERE r.test_run_id=? ORDER BY r.created_at, r.id
            """,
            (test_run_id,),
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            raw_gpu_uuids = item.pop("gpu_uuids_json")
            item["gpu_uuids"] = json.loads(raw_gpu_uuids) if raw_gpu_uuids else []
            item["token_usage"] = {
                "prompt_tokens": item.pop("actual_prompt_tokens"),
                "completion_tokens": item.pop("actual_completion_tokens"),
            }
            result.append(item)
        return result

    async def usage(self, principal: Principal) -> dict[str, Any]:
        account = await self.database.fetchone(
            (
                "\n"
                "            SELECT nickname, balance_tokens, limit_tokens, usage_"
                "baseline_tokens,\n"
                "                   usage_reset_at, unlimited FROM quota_accounts "
                "WHERE id = ?\n"
                "            "
            ),
            (principal.quota_account_id,),
        )
        totals = await self.database.fetchone(
            (
                "\n"
                "            SELECT charged_tokens, settled_requests\n"
                "              FROM account_lifetime_usage WHERE account_id = ?\n"
                "            "
            ),
            (principal.quota_account_id,),
        )
        lifetime_charged = int(totals["charged_tokens"]) if totals else 0
        baseline = int(account["usage_baseline_tokens"]) if account else 0
        used_tokens = max(0, lifetime_charged - baseline)
        return {
            "account_id": principal.quota_account_id,
            "account_nickname": account["nickname"] if account else None,
            "balance_tokens": account["balance_tokens"] if account else 0,
            "limit_tokens": account["limit_tokens"] if account else 0,
            "unlimited": bool(account["unlimited"]) if account else False,
            "used_tokens": used_tokens,
            "charged_tokens": used_tokens,
            "lifetime_charged_tokens": lifetime_charged,
            "settled_requests": int(totals["settled_requests"]) if totals else 0,
            "usage_reset_at": account["usage_reset_at"] if account else None,
        }
