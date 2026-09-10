"""Terminate isolated engine sessions and verify process and GPU ownership release."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from pathlib import Path
from typing import Protocol


class EngineProcess(Protocol):
    @property
    def pid(self) -> int: ...

    async def wait(self) -> int: ...


class TeardownError(RuntimeError):
    """The engine must remain reserved because its teardown could not be verified."""


def group_members(group: int) -> dict[int, str]:
    # Zombies have exited and own no CUDA context; init may reap them much later.
    members = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group:
                members[int(entry.name)] = fields[0]
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def gpu_pids(gpu_uuids: tuple[str, ...]) -> set[int]:
    if not gpu_uuids:
        return set()
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    try:
        return {
            int(process.pid)
            for gpu_uuid in gpu_uuids
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(
                pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
            )
        }
    finally:
        pynvml.nvmlShutdown()


async def _terminate(
    process: EngineProcess,
    *,
    gpu_uuids: tuple[str, ...],
    force: bool,
    grace_seconds: float,
    kill_seconds: float,
) -> None:
    group = process.pid  # start_new_session=True at every engine launch
    known_pids = {group}
    last_error: Exception | None = None
    for requested_signal, timeout in (
        (signal.SIGKILL if force else signal.SIGTERM, grace_seconds),
        (signal.SIGKILL, kill_seconds),
    ):
        try:
            known_pids.update(await asyncio.to_thread(group_members, group))
        except Exception as exc:
            raise TeardownError(f"Cannot inspect engine group {group}: {exc}") from exc
        with contextlib.suppress(ProcessLookupError):
            try:
                os.killpg(group, requested_signal)
            except OSError as exc:
                if not isinstance(exc, ProcessLookupError):
                    raise TeardownError(f"Cannot signal engine group {group}: {exc}") from exc
                raise
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                members = await asyncio.to_thread(group_members, group)
                known_pids.update(members)
                live = {pid for pid, state in members.items() if state not in {"Z", "X"}}
                resident = await asyncio.to_thread(gpu_pids, gpu_uuids)
                if not live and not known_pids.intersection(resident):
                    await asyncio.wait_for(process.wait(), timeout=max(kill_seconds, 0.1))
                    return
                last_error = None
            except Exception as exc:
                last_error = exc
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(0.1)
    raise TeardownError(
        f"Could not verify teardown of engine group {group}; GPUs remain reserved"
        + (f": {last_error}" if last_error else "")
    )


async def terminate_engine(
    process: EngineProcess,
    *,
    gpu_uuids: tuple[str, ...] = (),
    force: bool = False,
    grace_seconds: float = 10.0,
    kill_seconds: float = 10.0,
) -> None:
    """Complete teardown even if the caller is cancelled; fail closed on timeout."""
    task = asyncio.create_task(
        _terminate(
            process,
            gpu_uuids=gpu_uuids,
            force=force,
            grace_seconds=grace_seconds,
            kill_seconds=kill_seconds,
        )
    )
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()  # TeardownError takes precedence over cancellation.
    if cancelled:
        raise asyncio.CancelledError
