from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from test_prism_runtime import _profile

from llm_rio.domain import RuntimeState, WorkerPlacement
from llm_rio.workers import WorkerSupervisor


class _TransitionDatabase:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict[str, Any]]] = []
        self.persisted_states: list[str] = []

    async def execute(self, _query: str, parameters: tuple[Any, ...]) -> None:
        self.persisted_states.append(str(parameters[6]))

    async def record_event(
        self,
        event_type: str,
        entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((event_type, entity_id, payload or {}))


def _supervisor(worker: WorkerPlacement) -> tuple[WorkerSupervisor, _TransitionDatabase]:
    database = _TransitionDatabase()
    supervisor = WorkerSupervisor.__new__(WorkerSupervisor)
    supervisor.settings = SimpleNamespace(
        prism_weight_cache_mode="ram",
        prism_transition_timeout_seconds=10.0,
        engines=SimpleNamespace(
            vllm_executable="/opt/llm-rio/bin/vllm",
            llama_cpp_executable="llama-server",
            environment={"PATH": "/usr/local/bin:/usr/bin"},
        ),
    )
    supervisor.database = database
    supervisor.workers = {worker.id: worker}
    supervisor._processes = {}
    supervisor._log_handles = {}
    supervisor._log_paths = {}
    supervisor._lock = asyncio.Lock()
    supervisor._transition_locks = {}
    supervisor._drain_to_sleep = set()
    supervisor._event_callback = None
    supervisor.internal_api_key = "internal-test-key"
    supervisor.kvcached = SimpleNamespace(
        enabled=True,
        environment=lambda **_kwargs: {"LLM_RIO_KVCACHED_VLLM026_SHIM": "1"},
    )
    return supervisor, database


def _ready_worker(*, active_request: bool = False) -> WorkerPlacement:
    worker = WorkerPlacement(
        id="worker-1",
        profile=_profile("model-a", ("GPU-0",), (30_000,)),
        gpu_uuids=("GPU-0",),
        port=19370,
        state=RuntimeState.READY,
    )
    if active_request:
        worker.admitted_request_ids.add("request-1")
        worker.outstanding_token_work = 128
    return worker


def test_worker_environment_finds_helpers_beside_absolute_vllm() -> None:
    worker = _ready_worker()
    supervisor, _database = _supervisor(worker)

    environment = supervisor._environment(worker)

    assert environment["PATH"].split(os.pathsep) == [
        str(Path("/opt/llm-rio/bin").resolve()),
        "/usr/local/bin",
        "/usr/bin",
    ]
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-0"
    assert environment["VLLM_API_KEY"] == "internal-test-key"


async def test_worker_sleep_wake_round_trip_preserves_host_backup() -> None:
    worker = _ready_worker()
    supervisor, database = _supervisor(worker)
    calls: list[tuple[str, dict[str, str] | None]] = []

    async def post_engine(
        _worker: WorkerPlacement,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> None:
        calls.append((path, params))

    supervisor._post_engine = post_engine  # type: ignore[method-assign]

    await supervisor.sleep(worker.id)

    assert worker.state is RuntimeState.SLEEPING
    assert worker.host_weights_cached
    assert supervisor.occupied_gpu_uuids == set()
    assert supervisor.cached_gpu_uuids == {"GPU-0"}

    await supervisor.wake(worker.id)

    assert worker.state is RuntimeState.READY
    assert worker.host_weights_cached
    assert supervisor.occupied_gpu_uuids == {"GPU-0"}
    assert calls == [("/sleep", {"level": "1"}), ("/wake_up", None)]
    assert database.persisted_states == ["OFFLOADING", "SLEEPING", "WAKING", "READY"]
    assert [event[0] for event in database.events] == [
        "WORKER_OFFLOADING",
        "WORKER_WEIGHTS_CACHED",
        "WORKER_WAKING",
        "WORKER_WEIGHTS_RESTORED",
    ]


async def test_worker_waits_for_active_request_before_sleeping() -> None:
    worker = _ready_worker(active_request=True)
    supervisor, _database = _supervisor(worker)
    calls: list[str] = []

    async def post_engine(
        _worker: WorkerPlacement,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> None:
        calls.append(path)

    supervisor._post_engine = post_engine  # type: ignore[method-assign]

    await supervisor.sleep(worker.id)

    assert worker.state is RuntimeState.DRAINING
    assert calls == []

    await supervisor.release(worker.id, "request-1", 128)

    assert worker.state is RuntimeState.SLEEPING
    assert calls == ["/sleep"]


async def test_worker_transition_failure_fails_closed_without_deadlock() -> None:
    worker = _ready_worker()
    supervisor, database = _supervisor(worker)

    async def failed_post(
        _worker: WorkerPlacement,
        _path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> None:
        raise RuntimeError("simulated transition failure")

    supervisor._post_engine = failed_post  # type: ignore[method-assign]

    await asyncio.wait_for(supervisor.sleep(worker.id), timeout=1.0)

    assert worker.state is RuntimeState.COLD
    assert not worker.host_weights_cached
    failure = next(event for event in database.events if event[0] == "WORKER_FAILED")
    assert failure[2]["reason"] == "weight_offload_failed:RuntimeError"
