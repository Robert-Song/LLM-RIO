from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from llm_rio.process_cleanup import TeardownError, group_members, terminate_engine
from llm_rio.storage import Database


def _matching_managed_process(pid: int, port: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command = handle.read().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return False
    return ("vllm" in command or "llama-server" in command) and f"--port {port}" in command


@dataclass
class RecordedProcess:
    pid: int

    async def wait(self) -> int:
        # This is not our child to reap. terminate_engine already verified that
        # all group members exited and all owned GPU process records vanished.
        return 0


async def terminate_recorded_workers(database: Database) -> list[dict[str, object]]:
    """Reconcile surviving engines without acting on a reused, unrelated PID."""
    rows = await database.fetchall(
        """
        SELECT id, pid, port, state, gpu_uuids_json FROM workers
         WHERE state != 'COLD' AND pid IS NOT NULL
        """
    )
    results: list[dict[str, object]] = []
    if os.name != "posix":
        return [
            {"worker_id": row["id"], "action": "manual_review", "reason": "non_posix_host"}
            for row in rows
        ]
    for row in rows:
        pid, port = int(row["pid"]), int(row["port"])
        if not _matching_managed_process(pid, port):
            members = await asyncio.to_thread(group_members, pid)
            if any(state not in {"Z", "X"} for state in members.values()):
                raise TeardownError(
                    f"Recorded worker {row['id']} has surviving group {pid} but its parent "
                    "identity cannot be verified; refusing to clear recovery state"
                )
            action = (
                "already_gone" if not os.path.exists(f"/proc/{pid}") else "pid_identity_mismatch"
            )
            results.append({"worker_id": row["id"], "action": action})
            continue
        await terminate_engine(
            RecordedProcess(pid), gpu_uuids=tuple(json.loads(row["gpu_uuids_json"]))
        )
        results.append(
            {
                "worker_id": row["id"],
                "action": "terminated",
                "pid": pid,
                "gpu_uuids_json": row["gpu_uuids_json"],
            }
        )
    return results
