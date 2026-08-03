from __future__ import annotations

import asyncio
import os
import signal

from llm_rio.storage import Database


def _matching_managed_process(pid: int, port: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command = handle.read().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return False
    return ("vllm" in command or "llama-server" in command) and f"--port {port}" in command


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
            action = "already_gone" if not os.path.exists(f"/proc/{pid}") else "pid_identity_mismatch"
            results.append({"worker_id": row["id"], "action": action})
            continue
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            for _ in range(100):
                if not os.path.exists(f"/proc/{pid}"):
                    break
                await asyncio.sleep(0.1)
            if os.path.exists(f"/proc/{pid}") and _matching_managed_process(pid, port):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            released = not os.path.exists(f"/proc/{pid}")
            results.append(
                {
                    "worker_id": row["id"],
                    "action": "terminated" if released else "termination_unverified",
                    "pid": pid,
                    "gpu_uuids_json": row["gpu_uuids_json"],
                }
            )
        except (ProcessLookupError, PermissionError) as exc:
            results.append(
                {"worker_id": row["id"], "action": "failed", "reason": type(exc).__name__}
            )
    return results

