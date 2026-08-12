from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import signal
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from llm_rio.config import Settings
from llm_rio.domain import Engine, PlacementProfile, RuntimeState, WorkerPlacement
from llm_rio.inventory import gpu_environment
from llm_rio.storage import Database, _now
from llm_rio.tool_support import detect_vllm_parser_configuration

logger = logging.getLogger(__name__)

WorkerEventCallback = Callable[[str, str], Awaitable[None]]


async def _terminate_worker_process_tree(
    process: asyncio.subprocess.Process, *, force: bool
) -> None:
    """Request termination of a worker and every engine process it owns."""
    if process.returncode is not None:
        return

    if os.name == "posix":
        try:
            killpg = getattr(os, "killpg", None)
            getpgid = getattr(os, "getpgid", None)
            if killpg is not None and getpgid is not None:
                termination_signal = (
                    getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM
                )
                killpg(getpgid(process.pid), termination_signal)
                return
        except (AttributeError, ProcessLookupError, OSError):
            # Preserve the direct-child fallback for platforms or launchers without a group.
            pass
    with contextlib.suppress(ProcessLookupError, OSError):
        process.kill() if force else process.terminate()


class WorkerLaunchError(RuntimeError):
    pass


def worker_log_path(*, log_dir: Path, served_model_name: str, worker_id: str) -> Path:
    """Build a sortable log name that identifies the worker's model."""
    safe_model_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", served_model_name)
    safe_model_name = safe_model_name.strip("._-").lower()[:80] or "model"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return log_dir / f"{timestamp}-worker-{safe_model_name}-{worker_id}.log"


class WorkerSupervisor:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self.workers: dict[str, WorkerPlacement] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._log_handles: dict[str, Any] = {}
        self._log_paths: dict[str, Path] = {}
        self._lock = asyncio.Lock()
        self._event_callback: WorkerEventCallback | None = None
        self.internal_api_key = f"rio_internal_{secrets.token_urlsafe(32)}"

    def set_event_callback(self, callback: WorkerEventCallback) -> None:
        self._event_callback = callback

    @property
    def occupied_gpu_uuids(self) -> set[str]:
        return {
            gpu
            for worker in self.workers.values()
            if worker.state is not RuntimeState.COLD
            for gpu in worker.gpu_uuids
        }

    async def launch(
        self,
        *,
        profile: PlacementProfile,
        gpu_uuids: tuple[str, ...],
        model_path: str,
        served_model_name: str,
    ) -> WorkerPlacement:
        if len(gpu_uuids) != profile.gpu_count or gpu_uuids not in profile.eligible_gpu_sets:
            raise WorkerLaunchError("placement does not match a validated GPU set")
        async with self._lock:
            overlap = set(gpu_uuids) & self.occupied_gpu_uuids
            if overlap:
                raise WorkerLaunchError(f"GPU UUIDs already owned: {sorted(overlap)}")
            worker_id = str(uuid.uuid4())
            port = self._allocate_port()
            worker = WorkerPlacement(
                id=worker_id,
                profile=profile,
                gpu_uuids=gpu_uuids,
                port=port,
            )
            self.workers[worker_id] = worker
            try:
                command = self._command(worker, model_path, served_model_name)
            except BaseException:
                self.workers.pop(worker_id, None)
                raise
            environment = gpu_environment(gpu_uuids, self.settings.engines.environment)
            if worker.profile.engine is Engine.VLLM:
                environment["VLLM_API_KEY"] = self.internal_api_key
            log_path: Path | None = None
            log_handle: Any | None = None
            worker_output: Any = asyncio.subprocess.DEVNULL
            if self.settings.capture_worker_engine_logs:
                log_path = worker_log_path(
                    log_dir=self.settings.log_dir,
                    served_model_name=served_model_name,
                    worker_id=worker_id,
                )
                log_handle = log_path.open("ab", buffering=0)
                worker_output = log_handle
                self._log_handles[worker_id] = log_handle
                self._log_paths[worker_id] = log_path
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=worker_output,
                    stderr=asyncio.subprocess.STDOUT,
                    env=environment,
                    # Make the engine its own group so a forced shutdown cannot signal
                    # the API server, while killpg still reaches all engine descendants.
                    start_new_session=True,
                )
            except BaseException:
                await self._cleanup(worker_id)
                self.workers.pop(worker_id, None)
                raise
            worker.process_pid = process.pid
            self._processes[worker_id] = process
            try:
                await self._persist(worker)
                await self.database.record_event(
                    "WORKER_LOADING",
                    worker_id,
                    {
                        "gpu_uuids": gpu_uuids,
                        "command": self._redact_command(command),
                    },
                )
            except BaseException:
                with contextlib.suppress(Exception):
                    await _terminate_worker_process_tree(process, force=True)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), timeout=10)
                await self._cleanup(worker_id)
                self.workers.pop(worker_id, None)
                raise
            asyncio.create_task(self._await_ready(worker), name=f"worker-ready-{worker_id}")
            asyncio.create_task(self._monitor(worker), name=f"worker-monitor-{worker_id}")
            return worker

    def _allocate_port(self) -> int:
        used = {
            worker.port for worker in self.workers.values() if worker.state is not RuntimeState.COLD
        }
        for port in range(self.settings.worker_port_start, self.settings.worker_port_end + 1):
            if port not in used:
                return port
        raise WorkerLaunchError("private worker port range is exhausted")

    def _command(
        self, worker: WorkerPlacement, model_path: str, served_model_name: str
    ) -> list[str]:
        profile = worker.profile
        if profile.engine is Engine.VLLM:
            command = [
                self.settings.engines.vllm_executable,
                "serve",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(worker.port),
                "--api-key",
                self.internal_api_key,
                "--served-model-name",
                served_model_name,
                "--tensor-parallel-size",
                str(profile.tensor_parallel_size),
                "--pipeline-parallel-size",
                str(profile.pipeline_parallel_size),
                "--dtype",
                profile.dtype,
                "--max-model-len",
                str(profile.max_model_len),
                "--gpu-memory-utilization",
                str(profile.gpu_memory_utilization),
            ]
            if profile.max_num_seqs is not None:
                command.extend(["--max-num-seqs", str(profile.max_num_seqs)])
            if profile.max_num_batched_tokens is not None:
                command.extend(["--max-num-batched-tokens", str(profile.max_num_batched_tokens)])
            if profile.quantization:
                command.extend(["--quantization", profile.quantization])
            parsers = detect_vllm_parser_configuration(model_path)
            if parsers.tool_parser is not None:
                command.extend(
                    ["--enable-auto-tool-choice", "--tool-call-parser", parsers.tool_parser]
                )
            if parsers.reasoning_parser is not None:
                command.extend(["--reasoning-parser", parsers.reasoning_parser])
        elif profile.engine is Engine.LLAMA_CPP and self.settings.engines.enable_llama_cpp:
            command = [
                self.settings.engines.llama_cpp_executable,
                "--model",
                model_path,
                "--host",
                "127.0.0.1",
                "--port",
                str(worker.port),
                "--ctx-size",
                str(profile.max_model_len),
                "--api-key",
                self.internal_api_key,
            ]
            if profile.max_num_seqs is not None:
                command.extend(["--parallel", str(profile.max_num_seqs)])
        else:
            raise WorkerLaunchError(f"engine is disabled or unsupported: {profile.engine}")
        for key, value in profile.launch_args.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif isinstance(value, list):
                for item in value:
                    command.extend([flag, str(item)])
            elif value is not None:
                command.extend([flag, str(value)])
        return command

    @staticmethod
    def _redact_command(command: list[str]) -> list[str]:
        redacted = list(command)
        for index, value in enumerate(redacted[:-1]):
            if value == "--api-key":
                redacted[index + 1] = "[REDACTED]"
        return redacted

    async def _await_ready(self, worker: WorkerPlacement) -> None:
        startup_timeout = self.settings.worker_startup_timeout_seconds
        deadline = (
            asyncio.get_running_loop().time() + startup_timeout
            if startup_timeout is not None
            else None
        )
        headers = {"Authorization": f"Bearer {self.internal_api_key}"}
        async with httpx.AsyncClient(timeout=5.0) as client:
            while deadline is None or asyncio.get_running_loop().time() < deadline:
                process = self._processes.get(worker.id)
                if process is None or process.returncode is not None:
                    await self._fail(worker, "process_exited_during_startup")
                    return
                try:
                    response = await client.get(
                        f"http://127.0.0.1:{worker.port}/health", headers=headers
                    )
                    if response.is_success:
                        async with self._lock:
                            if worker.state is not RuntimeState.LOADING:
                                return
                            worker.state = RuntimeState.READY
                            worker.ready_at = datetime.now(UTC)
                            worker.last_demand_at = worker.ready_at
                        await self._persist(worker)
                        await self.database.record_event("WORKER_READY", worker.id)
                        await self._emit(worker.id, "ready")
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        await self._fail(worker, "startup_timeout")

    async def _monitor(self, worker: WorkerPlacement) -> None:
        process = self._processes[worker.id]
        return_code = await process.wait()
        if worker.state not in {RuntimeState.STOPPING, RuntimeState.COLD}:
            await self._fail(worker, f"unexpected_exit_{return_code}")

    async def _fail(self, worker: WorkerPlacement, reason: str) -> None:
        admitted: list[str]
        async with self._lock:
            if worker.state is RuntimeState.COLD:
                return
            admitted = list(worker.admitted_request_ids)
            worker.admitted_request_ids.clear()
            worker.outstanding_token_work = 0
            worker.state = RuntimeState.COLD
            process = self._processes.get(worker.id)
        if process and process.returncode is None:
            await _terminate_worker_process_tree(process, force=True)
            await process.wait()
        await self._persist(worker)
        log_path = self._log_paths.get(worker.id)
        await self.database.record_event(
            "WORKER_FAILED",
            worker.id,
            {
                "reason": reason,
                "request_ids": admitted,
                "log_path": str(log_path) if log_path else None,
            },
        )
        await self._cleanup(worker.id, retain_log=True)
        await self._emit(worker.id, f"failed:{reason}")

    async def admit(self, worker_id: str, request_id: str, estimated_tokens: int) -> None:
        async with self._lock:
            worker = self.workers[worker_id]
            if worker.state is not RuntimeState.READY:
                raise RuntimeError("worker is not routable")
            worker.admitted_request_ids.add(request_id)
            worker.accepted_requests += 1
            worker.outstanding_token_work += estimated_tokens
            worker.last_demand_at = datetime.now(UTC)

    async def release(self, worker_id: str, request_id: str, estimated_tokens: int) -> None:
        should_stop = False
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None:
                return
            worker.admitted_request_ids.discard(request_id)
            worker.last_demand_at = datetime.now(UTC)
            worker.outstanding_token_work = max(0, worker.outstanding_token_work - estimated_tokens)
            should_stop = worker.state is RuntimeState.DRAINING and not worker.admitted_request_ids
        if should_stop:
            await self.stop(worker_id, force=False)
        await self._emit(worker_id, "released")

    async def drain(self, worker_id: str) -> None:
        stop_now = False
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None or worker.state in {
                RuntimeState.DRAINING,
                RuntimeState.STOPPING,
                RuntimeState.COLD,
            }:
                return
            worker.state = RuntimeState.DRAINING
            worker.drain_started_at = datetime.now(UTC)
            stop_now = not worker.admitted_request_ids
        await self._persist(worker)
        await self.database.record_event("WORKER_DRAINING", worker_id)
        if stop_now:
            await self.stop(worker_id, force=False)

    async def enforce_drain_watchdogs(self) -> list[tuple[str, list[str]]]:
        now = datetime.now(UTC)
        watchdog = self.settings.worker_drain_watchdog_seconds
        overdue: list[tuple[str, list[str]]] = []
        for worker in list(self.workers.values()):
            if (
                watchdog is not None
                and worker.state is RuntimeState.DRAINING
                and worker.drain_started_at
                and (now - worker.drain_started_at).total_seconds() > watchdog
            ):
                overdue.append((worker.id, list(worker.admitted_request_ids)))
                await self.stop(worker.id, force=True)
        return overdue

    async def stop(self, worker_id: str, *, force: bool) -> None:
        async with self._lock:
            worker = self.workers.get(worker_id)
            if worker is None or worker.state is RuntimeState.COLD:
                return
            if worker.admitted_request_ids and not force:
                return
            worker.state = RuntimeState.STOPPING
            process = self._processes.get(worker_id)
        try:
            await self._persist(worker)
        except Exception:
            # A shutdown must never leave a running engine behind because a best-effort
            # STOPPING record could not be written.  The final COLD persistence follows
            # after the process is gone.
            logger.exception("could not persist worker %s before stopping it", worker_id)
        try:
            if process and process.returncode is None:
                await _terminate_worker_process_tree(process, force=force)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0 if force else 30.0)
                except TimeoutError:
                    await _terminate_worker_process_tree(process, force=True)
                    await process.wait()
        finally:
            async with self._lock:
                worker.state = RuntimeState.COLD
                worker.process_pid = None
                worker.admitted_request_ids.clear()
                worker.outstanding_token_work = 0
            try:
                await self._persist(worker)
                await self.database.record_event("WORKER_COLD", worker_id, {"forced": force})
                await self._emit(worker_id, "cold")
            finally:
                await self._cleanup(worker_id)

    async def stop_all(self, *, force: bool = False) -> None:
        worker_ids = list(self.workers)
        if not force:
            for worker_id in worker_ids:
                await self.drain(worker_id)
        tasks = [self.stop(worker_id, force=force) for worker_id in worker_ids]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for worker_id, result in zip(worker_ids, results, strict=True):
                if isinstance(result, BaseException):
                    logger.error("could not stop worker %s during shutdown: %r", worker_id, result)

    async def _cleanup(self, worker_id: str, *, retain_log: bool = False) -> None:
        self._processes.pop(worker_id, None)
        handle = self._log_handles.pop(worker_id, None)
        if handle:
            handle.close()
        log_path = self._log_paths.pop(worker_id, None)
        if log_path and not retain_log:
            with contextlib.suppress(FileNotFoundError):
                log_path.unlink()

    async def _persist(self, worker: WorkerPlacement) -> None:
        await self.database.execute(
            """
            INSERT INTO workers
                (id, model_id, profile_id, gpu_uuids_json, port, pid, state, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET pid = excluded.pid, state = excluded.state,
                updated_at = excluded.updated_at
            """,
            (
                worker.id,
                worker.model_id,
                worker.profile.id,
                json.dumps(worker.gpu_uuids),
                worker.port,
                worker.process_pid,
                worker.state.value,
                _now(),
                _now(),
            ),
        )

    async def _emit(self, worker_id: str, event: str) -> None:
        if self._event_callback:
            await self._event_callback(worker_id, event)
