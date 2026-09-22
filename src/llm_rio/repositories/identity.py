from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from llm_rio.domain import Role
from llm_rio.errors import RioError
from llm_rio.security import (
    Principal,
    hash_api_key,
    verify_api_key,
)
from llm_rio.timestamps import now as _now

if TYPE_CHECKING:
    from llm_rio.storage import Database

logger = logging.getLogger(__name__)


class IdentityRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def authenticate(self, prefix: str, token: str) -> Principal | None:
        row = await self.database.fetchone(
            (
                "\n"
                "            SELECT k.id, k.nickname, k.role, k.quota_account_id, "
                "k.token_hash, a.unlimited\n"
                "              FROM api_keys k JOIN quota_accounts a ON a.id = k.q"
                "uota_account_id\n"
                "             WHERE k.token_prefix = ? AND k.active = 1\n"
                "            "
            ),
            (prefix,),
        )
        if row is None or not await asyncio.to_thread(verify_api_key, row["token_hash"], token):
            return None
        await self.database.execute(
            "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), row["id"])
        )
        return Principal(
            key_id=row["id"],
            nickname=row["nickname"],
            role=Role(row["role"]),
            quota_account_id=row["quota_account_id"],
            unlimited=bool(row["unlimited"]),
        )

    async def key_by_selector(
        self, selector: str, *, active_only: bool = True
    ) -> dict[str, Any] | None:
        active_clause = " AND active = 1" if active_only else ""
        row = await self.database.fetchone(
            f"SELECT id, nickname, token_hash FROM api_keys WHERE nickname = ?{active_clause}",
            (selector,),
        )
        if row is not None:
            return {"id": str(row["id"]), "nickname": str(row["nickname"])}
        if not selector.startswith("rio_") or len(selector) < 24:
            return None
        row = await self.database.fetchone(
            f"SELECT id, nickname, token_hash FROM api_keys WHERE token_prefix = ?{active_clause}",
            (selector[:24],),
        )
        if row is None or not await asyncio.to_thread(
            verify_api_key, str(row["token_hash"]), selector
        ):
            return None
        return {"id": str(row["id"]), "nickname": str(row["nickname"])}

    async def key_count(self) -> int:
        row = await self.database.fetchone("SELECT COUNT(*) AS count FROM api_keys")
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
        limit_tokens: int,
        unlimited: bool,
    ) -> None:
        if role is Role.ADMIN:
            unlimited = True
        token_hash = await asyncio.to_thread(hash_api_key, api_key)
        encrypted_api_key = self.database.key_vault.encrypt(api_key)
        async with self.database.transaction() as connection:
            await connection.execute(
                (
                    "\n"
                    "                INSERT OR IGNORE INTO quota_accounts\n"
                    "                    (id, nickname, balance_tokens, limit_tokens, "
                    "unlimited, created_at)\n"
                    "                VALUES (?, ?, ?, ?, ?, ?)\n"
                    "                "
                ),
                (account_id, account_nickname, limit_tokens, limit_tokens, int(unlimited), _now()),
            )
            if role is Role.ADMIN:
                await connection.execute(
                    "UPDATE quota_accounts SET unlimited = 1 WHERE id = ?", (account_id,)
                )
            await connection.execute(
                (
                    "\n"
                    "                INSERT INTO api_keys\n"
                    "                    (id, nickname, role, quota_account_id, token_"
                    "prefix, token_hash,\n"
                    "                     encrypted_api_key, created_at)\n"
                    "                VALUES (?, ?, ?, ?, ?, ?, ?, ?)\n"
                    "                "
                ),
                (
                    key_id,
                    nickname,
                    role.value,
                    account_id,
                    prefix,
                    token_hash,
                    encrypted_api_key,
                    _now(),
                ),
            )

    async def list_keys(self) -> list[dict[str, Any]]:
        rows = await self.database.fetchall(
            "\n"
            "            SELECT k.id, k.nickname, k.role, k.quota_account_id, "
            "k.encrypted_api_key, k.active,\n"
            "                   k.created_at, k.last_used_at, a.nickname AS ac"
            "count_nickname,\n"
            "                   a.balance_tokens, a.limit_tokens, a.usage_base"
            "line_tokens,\n"
            "                   a.usage_reset_at, a.unlimited,\n"
            "                   COALESCE((\n"
            "                       SELECT charged_tokens FROM account_lifetim"
            "e_usage\n"
            "                        WHERE account_id = a.id\n"
            "                   ), 0) AS lifetime_charged_tokens,\n"
            "                   COALESCE((\n"
            "                       SELECT charged_tokens FROM key_lifetime_us"
            "age WHERE key_id = k.id\n"
            "                   ), 0) AS key_lifetime_charged_tokens,\n"
            "                   COALESCE((\n"
            "                       SELECT settled_requests FROM account_lifet"
            "ime_usage WHERE account_id = a.id\n"
            "                   ), 0) AS settled_requests\n"
            "              FROM api_keys k JOIN quota_accounts a ON a.id = k.q"
            "uota_account_id\n"
            "             WHERE k.token_prefix NOT LIKE 'deleted-%'\n"
            "             ORDER BY k.nickname\n"
            "            "
        )
        grant_rows = await self.database.fetchall(
            "\n"
            "            SELECT g.key_id, m.nickname FROM model_grants g\n"
            "              JOIN model_catalog m ON m.id = g.model_id\n"
            "             ORDER BY g.key_id, m.nickname\n"
            "            "
        )
        grants: dict[str, list[str]] = {}
        for grant in grant_rows:
            grants.setdefault(str(grant["key_id"]), []).append(str(grant["nickname"]))
        result = []
        for row in rows:
            item = dict(row)
            item["api_key"] = self.database.key_vault.decrypt(item.pop("encrypted_api_key"))
            item["active"] = bool(item["active"])
            item["unlimited"] = bool(item["unlimited"])
            item["used_tokens"] = max(
                0, int(item["lifetime_charged_tokens"]) - int(item["usage_baseline_tokens"])
            )
            item["granted_models"] = grants.get(str(item["id"]), [])
            result.append(item)
        return result

    async def set_key_active(self, key_id: str, active: bool) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT role, active, token_prefix FROM api_keys WHERE id = ?", (key_id,)
                )
            ).fetchone()
            if row is None:
                return False
            if active and str(row["token_prefix"]).startswith("deleted-"):
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
        token_hash = await asyncio.to_thread(hash_api_key, api_key)
        cursor = await self.database.execute(
            (
                "\n"
                "            UPDATE api_keys\n"
                "               SET token_prefix = ?, token_hash = ?, encrypted_ap"
                "i_key = ?, active = 1\n"
                "             WHERE id = ? AND token_prefix NOT LIKE 'deleted-%'\n"
                "            "
            ),
            (prefix, token_hash, self.database.key_vault.encrypt(api_key), key_id),
        )
        return cursor.rowcount > 0

    async def delete_key(self, key_id: str) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT quota_account_id, role, active FROM api_keys WHERE id = ?", (key_id,)
                )
            ).fetchone()
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
            tombstone = f"deleted-{key_id}"
            await connection.execute(
                (
                    "\n"
                    "                UPDATE api_keys\n"
                    "                   SET active = 0, nickname = ?, token_prefix = ?"
                    ", token_hash = ?,\n"
                    "                       encrypted_api_key = ?\n"
                    "                 WHERE id = ?\n"
                    "                "
                ),
                (
                    tombstone,
                    tombstone,
                    tombstone,
                    self.database.key_vault.encrypt(tombstone),
                    key_id,
                ),
            )
            await connection.execute("DELETE FROM model_grants WHERE key_id = ?", (key_id,))
        return True

    async def update_quota(self, key_id: str, limit_tokens: int, unlimited: bool) -> bool:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT role, quota_account_id FROM api_keys WHERE id = ?", (key_id,)
                )
            ).fetchone()
            if row is None:
                return False
            if row["role"] == Role.ADMIN.value and (not unlimited):
                raise RioError("invalid_quota", "Admin keys must remain unlimited", status_code=409)
            account = await (
                await connection.execute(
                    "SELECT usage_baseline_tokens FROM quota_accounts WHERE id = ?",
                    (row["quota_account_id"],),
                )
            ).fetchone()
            totals = await (
                await connection.execute(
                    (
                        "\n"
                        "                SELECT charged_tokens FROM account_lifetime_usage"
                        " WHERE account_id = ?\n"
                        "                "
                    ),
                    (row["quota_account_id"],),
                )
            ).fetchone()
            baseline = int(account["usage_baseline_tokens"]) if account else 0
            charged = int(totals["charged_tokens"]) if totals else 0
            used_tokens = max(0, charged - baseline)
            remaining_tokens = max(0, limit_tokens - used_tokens)
            await connection.execute(
                (
                    "\n"
                    "                UPDATE quota_accounts\n"
                    "                   SET balance_tokens = ?, limit_tokens = ?, unli"
                    "mited = ?\n"
                    "                 WHERE id = ?\n"
                    "                "
                ),
                (remaining_tokens, limit_tokens, int(unlimited), row["quota_account_id"]),
            )
        return True

    async def reset_usage(self, key_id: str) -> dict[str, Any] | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    (
                        "\n"
                        "                SELECT k.quota_account_id, a.limit_tokens, a.unli"
                        "mited\n"
                        "                  FROM api_keys k JOIN quota_accounts a ON a.id ="
                        " k.quota_account_id\n"
                        "                 WHERE k.id = ?\n"
                        "                "
                    ),
                    (key_id,),
                )
            ).fetchone()
            if row is None:
                return None
            totals = await (
                await connection.execute(
                    (
                        "\n"
                        "                SELECT charged_tokens FROM account_lifetime_usage"
                        " WHERE account_id = ?\n"
                        "                "
                    ),
                    (row["quota_account_id"],),
                )
            ).fetchone()
            charged = int(totals["charged_tokens"]) if totals else 0
            reset_at = _now()
            await connection.execute(
                (
                    "\n"
                    "                UPDATE quota_accounts\n"
                    "                   SET balance_tokens = limit_tokens, usage_basel"
                    "ine_tokens = ?, usage_reset_at = ?\n"
                    "                 WHERE id = ?\n"
                    "                "
                ),
                (charged, reset_at, row["quota_account_id"]),
            )
            return {
                "quota_account_id": row["quota_account_id"],
                "limit_tokens": int(row["limit_tokens"]),
                "balance_tokens": int(row["limit_tokens"]),
                "unlimited": bool(row["unlimited"]),
                "used_tokens": 0,
                "usage_reset_at": reset_at,
            }
