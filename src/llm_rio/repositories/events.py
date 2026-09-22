from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from llm_rio.domain import ServiceMode
from llm_rio.timestamps import now as _now

if TYPE_CHECKING:
    from llm_rio.storage import Database

logger = logging.getLogger(__name__)


class EventsRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def service_mode(self) -> ServiceMode:
        row = await self.database.fetchone("SELECT mode FROM service_state WHERE singleton = 1")
        return ServiceMode(row["mode"] if row else ServiceMode.ACTIVE.value)

    async def set_service_mode(self, mode: ServiceMode) -> None:
        await self.database.execute(
            "UPDATE service_state SET mode = ?, updated_at = ? WHERE singleton = 1",
            (mode.value, _now()),
        )
        await self.database.record_event("SERVICE_MODE_CHANGED", payload={"mode": mode.value})

    async def set_machine_fingerprint(self, fingerprint: str) -> str | None:
        async with self.database.transaction() as connection:
            row = await (
                await connection.execute(
                    "SELECT machine_fingerprint FROM service_state WHERE singleton = 1"
                )
            ).fetchone()
            previous = row["machine_fingerprint"] if row else None
            await connection.execute(
                (
                    "UPDATE service_state SET machine_fingerprint = ?, updated_at = ? "
                    "WHERE singleton = 1"
                ),
                (fingerprint, _now()),
            )
        return previous

    async def recover_orphaned_state(self) -> None:
        """Begin cold and refund reservations that cannot have a live local request."""
        async with self.database.transaction() as connection:
            reservations = await (
                await connection.execute(
                    "SELECT id FROM quota_reservations WHERE state = 'RESERVED'"
                )
            ).fetchall()
        for reservation in reservations:
            await self.database.release_reservation(reservation["id"], "service_restarted")
        await self.database.execute(
            (
                "\n"
                "            UPDATE workers SET state = 'COLD', pid = NULL,\n"
                "                host_cache_accounted_mib = 0, host_cache_accounti"
                "ng_source = NULL,\n"
                "                process_rss_mib = 0, process_pss_mib = 0, process"
                "_swap_mib = 0,\n"
                "                updated_at = ?\n"
                "             WHERE state != 'COLD' OR pid IS NOT NULL\n"
                "            "
            ),
            (_now(),),
        )
        await self.database.execute(
            (
                "\n"
                "            UPDATE model_jobs SET state = 'QUEUED', stage = 'vali"
                "dation_requeued', updated_at = ?\n"
                "             WHERE state = 'RUNNING' AND stage LIKE 'validat%'\n"
                "            "
            ),
            (_now(),),
        )

    async def record_event(
        self, event_type: str, entity_id: str | None = None, payload: dict[str, Any] | None = None
    ) -> None:
        await self.database.execute(
            (
                "\n"
                "            INSERT INTO runtime_events(event_type, entity_id, pay"
                "load_json, created_at)\n"
                "            VALUES (?, ?, ?, ?)\n"
                "            "
            ),
            (event_type, entity_id, json.dumps(payload or {}), _now()),
        )
