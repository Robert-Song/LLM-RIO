from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm_rio.domain import RuntimeState, ServiceMode
from llm_rio.runtime import ResidencyScheduler
from llm_rio.validation import ProfileValidator, _model_launch_args


def test_deepseek_v4_uses_mandatory_fp8_kv_cache(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"}
        ),
        encoding="utf-8",
    )

    assert _model_launch_args(tmp_path) == {"kv_cache_dtype": "fp8"}


class FinishedProcess:
    pid = 43123
    returncode = 0

    def terminate(self) -> None:
        raise AssertionError("finished parent must not receive a direct terminate")

    def kill(self) -> None:
        raise AssertionError("finished parent must not receive a direct kill")

    async def wait(self) -> int:
        return 0


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_terminate_signals_group_after_parent_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    def killpg(process_group: int, requested_signal: int) -> None:
        signals.append((process_group, requested_signal))
        if requested_signal == 0:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg)

    await ProfileValidator._terminate(cast(Any, FinishedProcess()))

    assert signals == [
        (FinishedProcess.pid, signal.SIGTERM),
        (FinishedProcess.pid, 0),
    ]


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_terminate_kills_group_that_outlives_grace_period(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    def killpg(process_group: int, requested_signal: int) -> None:
        signals.append((process_group, requested_signal))

    async def no_delay(_seconds: float) -> None:
        return None

    monkeypatch.setattr(os, "killpg", killpg)
    monkeypatch.setattr(asyncio, "sleep", no_delay)

    await ProfileValidator._terminate(cast(Any, FinishedProcess()))

    assert signals[0] == (FinishedProcess.pid, signal.SIGTERM)
    assert signals[1:-1] == [(FinishedProcess.pid, 0)] * 50


class ValidationDatabase:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def service_mode(self) -> ServiceMode:
        return ServiceMode.ACTIVE

    async def record_event(
        self,
        event_type: str,
        _entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((event_type, payload or {}))


class ValidationQueues:
    def __init__(self) -> None:
        self.has_demand = False

    def pending_models(self) -> list[str]:
        return ["production-model"] if self.has_demand else []


class ValidationSupervisor:
    ram_weight_cache_enabled = True

    def __init__(self, workers: list[Any], queues: ValidationQueues) -> None:
        self.workers = {worker.id: worker for worker in workers}
        self.queues = queues
        self.sleep_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.arrive_during_sleep = False

    async def sleep(self, worker_id: str) -> None:
        self.sleep_calls.append(worker_id)
        self.workers[worker_id].state = RuntimeState.SLEEPING
        if self.arrive_during_sleep:
            self.queues.has_demand = True

    async def stop(self, worker_id: str, *, force: bool) -> None:
        self.stop_calls.append(worker_id)
        self.workers[worker_id].state = RuntimeState.COLD


def validation_scheduler(*workers: Any) -> tuple[ResidencyScheduler, ValidationSupervisor]:
    queues = ValidationQueues()
    supervisor = ValidationSupervisor(list(workers), queues)
    scheduler = ResidencyScheduler.__new__(ResidencyScheduler)
    scheduler.settings = SimpleNamespace(validation_idle_window_seconds=0)
    scheduler.database = ValidationDatabase()
    scheduler.supervisor = supervisor
    scheduler.queues = queues
    scheduler._state_lock = asyncio.Lock()
    scheduler._validation_gpu_uuids = set()
    scheduler._last_arrival_at = datetime.now(UTC) - timedelta(seconds=60)
    scheduler._maintenance_requested = False
    scheduler._event = asyncio.Event()
    return scheduler, supervisor


async def test_validation_preserves_ram_cached_workers() -> None:
    ready = SimpleNamespace(
        id="ready",
        state=RuntimeState.READY,
        gpu_uuids=("GPU-0",),
        admitted_request_ids=set(),
    )
    sleeping = SimpleNamespace(
        id="sleeping",
        state=RuntimeState.SLEEPING,
        gpu_uuids=("GPU-0",),
        admitted_request_ids=set(),
    )
    scheduler, supervisor = validation_scheduler(ready, sleeping)

    acquired = await scheduler.acquire_validation_gpus(("GPU-0",))

    assert acquired
    assert supervisor.sleep_calls == ["ready"]
    assert supervisor.stop_calls == []
    assert ready.state is RuntimeState.SLEEPING
    assert sleeping.state is RuntimeState.SLEEPING
    assert scheduler._validation_gpu_uuids == {"GPU-0"}
    assert scheduler.database.events[-1] == (
        "VALIDATION_GPUS_ACQUIRED",
        {
            "gpu_uuids": ("GPU-0",),
            "preserved_cached_worker_ids": ["ready", "sleeping"],
            "evicted_cached_worker_ids": [],
        },
    )


async def test_validation_defers_if_production_arrives_while_worker_sleeps() -> None:
    ready = SimpleNamespace(
        id="ready",
        state=RuntimeState.READY,
        gpu_uuids=("GPU-0",),
        admitted_request_ids=set(),
    )
    scheduler, supervisor = validation_scheduler(ready)
    supervisor.arrive_during_sleep = True

    acquired = await scheduler.acquire_validation_gpus(("GPU-0",))

    assert not acquired
    assert supervisor.sleep_calls == ["ready"]
    assert supervisor.stop_calls == []
    assert scheduler._validation_gpu_uuids == set()
    assert scheduler.database.events[-1][0] == "VALIDATION_GPUS_DEFERRED"
    assert scheduler.database.events[-1][1]["production_demand"] is True
