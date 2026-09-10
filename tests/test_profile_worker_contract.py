from __future__ import annotations

import json
import re
import signal
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from test_scheduler_contract import GPU_0, make_profile, make_worker

from llm_rio.config import EngineSettings, Settings
from llm_rio.domain import RuntimeState
from llm_rio.prism import KVCachedRuntime
from llm_rio.profiles import ProfileRepository, profile_to_dict
from llm_rio.tool_support import (
    detect_vllm_reasoning_parser,
    detect_vllm_tool_parser,
)
from llm_rio.workers import WorkerSupervisor, worker_log_path


class ProfileRows:
    async def fetchall(self, sql: str, parameters: tuple[str, str]) -> list[dict[str, Any]]:
        assert "active = 1" in sql
        profile = make_profile("json-generated-id", "model", (GPU_0,))
        raw = profile_to_dict(profile)
        return [{"id": "database-row-id", "profile_json": json.dumps(raw)}]


@pytest.mark.asyncio
async def test_profile_row_id_is_authoritative_after_conflict_update() -> None:
    repository = ProfileRepository(
        ProfileRows(),  # type: ignore[arg-type]
        "machine",
    )

    profiles = await repository.for_model("model")

    assert len(profiles) == 1
    assert profiles[0].id == "database-row-id"


class FailingDatabase:
    async def execute(self, sql: str, parameters: tuple[Any, ...]) -> None:
        raise sqlite3.IntegrityError("profile foreign key")

    async def record_event(
        self,
        event_type: str,
        entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        raise AssertionError("loading event must not be recorded after persistence failure")


class RecordingDatabase:
    def __init__(self) -> None:
        self.events: list[tuple[str, str | None, dict[str, Any] | None]] = []

    async def execute(self, sql: str, parameters: tuple[Any, ...]) -> None:
        pass

    async def record_event(
        self,
        event_type: str,
        entity_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.events.append((event_type, entity_id, payload))


@pytest.mark.parametrize(
    ("template", "parser"),
    [
        ("{% if tools %}<|tool_call>call:name{}<tool_call|>{% endif %}", "gemma4"),
        ("{% if tools %}<tool_call><function=name></function>{% endif %}", "qwen3_xml"),
        (
            "{% if tools %}<tool_call>x<arg_key>k</arg_key>"
            "<arg_value>v</arg_value></tool_call>{% endif %}",
            "poolside_v1",
        ),
    ],
)
def test_tool_parser_is_inferred_from_chat_template(
    tmp_path: Path, template: str, parser: str
) -> None:
    model_path = tmp_path / parser
    model_path.mkdir()
    (model_path / "chat_template.jinja").write_text(template, encoding="utf-8")

    assert detect_vllm_tool_parser(model_path) == parser


def test_vllm_worker_enables_detected_tool_parser(tmp_path: Path) -> None:
    model_path = tmp_path / "gemma"
    model_path.mkdir()
    (model_path / "chat_template.jinja").write_text(
        "{% if tools %}<|tool_call>call:name{}<tool_call|>{% endif %}",
        encoding="utf-8",
    )
    supervisor = WorkerSupervisor(
        Settings(config_file=tmp_path / "missing.toml"),
        FailingDatabase(),  # type: ignore[arg-type]
    )
    worker = make_worker("worker", make_profile("profile", "model", (GPU_0,)))

    command = supervisor._command(worker, str(model_path), "gemma")

    parser_index = command.index("--tool-call-parser")
    assert command[parser_index + 1] == "gemma4"
    assert "--enable-auto-tool-choice" in command


def test_kvcached_worker_launch_is_explicit_and_shareable(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(
        Settings(config_file=tmp_path / "missing.toml", prism_max_workers_per_gpu=2),
        FailingDatabase(),  # type: ignore[arg-type]
    )
    supervisor.kvcached = KVCachedRuntime(
        enabled=True,
        package_version="0.1.5",
        vllm_version="0.26.0",
        officially_tested=False,
        reason="untested_vllm_version",
    )
    profile = replace(
        make_profile("profile", "model", (GPU_0,)),
        memory_backend="kvcached",
        kvcached_verified=True,
    )
    worker = make_worker("worker", profile)
    other = make_worker(
        "other",
        replace(
            make_profile("other", "other", (GPU_0,)),
            memory_backend="kvcached",
            kvcached_verified=True,
        ),
    )

    command = supervisor._command(worker, str(tmp_path), "model")
    environment = supervisor._environment(worker)

    assert "--no-enable-prefix-caching" in command
    assert "--enable-sleep-mode" in command
    assert environment["ENABLE_KVCACHED"] == "true"
    assert environment["KVCACHED_AUTOPATCH"] == "1"
    assert environment["VLLM_SERVER_DEV_MODE"] == "1"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert "VLLM_USE_V1" not in environment
    assert supervisor._can_share_gpus(profile, (GPU_0,), [other])


@pytest.mark.asyncio
async def test_ram_cached_worker_drains_then_sleeps_and_wakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = RecordingDatabase()
    supervisor = WorkerSupervisor(
        Settings(config_file=tmp_path / "missing.toml"),
        database,  # type: ignore[arg-type]
    )
    supervisor.kvcached = KVCachedRuntime(
        enabled=True,
        package_version="0.1.5",
        vllm_version="0.26.0",
        officially_tested=False,
        reason="untested_vllm_version",
    )
    profile = replace(
        make_profile("profile", "model", (GPU_0,)),
        memory_backend="kvcached",
        kvcached_verified=True,
    )
    worker = make_worker("worker", profile)
    worker.process_pid = 42
    worker.admitted_request_ids.add("request")
    worker.outstanding_token_work = 64
    supervisor.workers[worker.id] = worker
    posts: list[tuple[str, dict[str, str] | None]] = []

    async def fake_post(
        _worker: Any,
        path: str,
        *,
        params: dict[str, str] | None = None,
    ) -> None:
        posts.append((path, params))

    monkeypatch.setattr(supervisor, "_post_engine", fake_post)

    await supervisor.sleep(worker.id)
    assert worker.state is RuntimeState.DRAINING
    assert posts == []

    await supervisor.release(worker.id, "request", 64)
    assert cast(RuntimeState, worker.state) is RuntimeState.SLEEPING
    assert worker.host_weights_cached
    assert worker.process_pid == 42
    assert posts == [("/sleep", {"level": "1"})]

    await supervisor.wake(worker.id)
    assert cast(RuntimeState, worker.state) is RuntimeState.READY
    assert worker.host_weights_cached
    assert worker.process_pid == 42
    assert posts == [("/sleep", {"level": "1"}), ("/wake_up", None)]
    assert {event[0] for event in database.events} >= {
        "WORKER_DRAINING_TO_RAM",
        "WORKER_WEIGHTS_CACHED",
        "WORKER_WEIGHTS_RESTORED",
    }


def test_vllm_worker_serializes_hf_overrides_as_json(tmp_path: Path) -> None:
    model_path = tmp_path / "qwen"
    model_path.mkdir()
    hf_overrides = {
        "text_config": {
            "max_position_embeddings": 1_048_576,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 262_144,
            },
        }
    }
    profile = replace(
        make_profile("profile", "model", (GPU_0,)),
        max_model_len=1_048_576,
        launch_args={"hf_overrides": hf_overrides},
    )
    supervisor = WorkerSupervisor(
        Settings(config_file=tmp_path / "missing.toml"),
        FailingDatabase(),  # type: ignore[arg-type]
    )
    worker = make_worker("worker", profile)

    command = supervisor._command(worker, str(model_path), "qwen-ext")

    override_index = command.index("--hf-overrides")
    assert json.loads(command[override_index + 1]) == hf_overrides
    max_length_index = command.index("--max-model-len")
    assert command[max_length_index + 1] == "1048576"


def test_vllm_worker_enables_qwen_reasoning_parser(tmp_path: Path) -> None:
    model_path = tmp_path / "qwen"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "qwen3_5"}', encoding="utf-8")
    (model_path / "chat_template.jinja").write_text(
        "{% if tools %}<tool_call><function=name></function>{% endif %}"
        "{% if enable_thinking %}<think>{% endif %}</think>",
        encoding="utf-8",
    )
    supervisor = WorkerSupervisor(
        Settings(config_file=tmp_path / "missing.toml"),
        FailingDatabase(),  # type: ignore[arg-type]
    )
    worker = make_worker("worker", make_profile("profile", "model", (GPU_0,)))

    command = supervisor._command(worker, str(model_path), "qwen")

    parser_index = command.index("--reasoning-parser")
    assert command[parser_index + 1] == "qwen3"
    assert detect_vllm_reasoning_parser(model_path) == "qwen3"


def test_worker_log_name_starts_with_utc_datetime_and_identifies_model(tmp_path: Path) -> None:
    path = worker_log_path(
        log_dir=tmp_path,
        served_model_name="Research/Model v2",
        worker_id="f5e50bb7-0b2b-4e84-aeee-f06e7c8c8a65",
    )

    assert re.match(
        r"^\d{8}T\d{6}Z-worker-research-model-v2-f5e50bb7-0b2b-4e84-aeee-f06e7c8c8a65\.log$",
        path.name,
    )


@pytest.mark.asyncio
async def test_worker_log_is_discarded_after_clean_shutdown(tmp_path: Path) -> None:
    supervisor = WorkerSupervisor(
        Settings(log_dir=tmp_path),
        RecordingDatabase(),  # type: ignore[arg-type]
    )
    log_path = tmp_path / "worker.log"
    log_path.write_text("normal engine output", encoding="utf-8")
    supervisor._log_paths["worker"] = log_path

    await supervisor._cleanup("worker")

    assert not log_path.exists()


@pytest.mark.asyncio
async def test_worker_failure_retains_log_and_records_its_path(tmp_path: Path) -> None:
    database = RecordingDatabase()
    supervisor = WorkerSupervisor(
        Settings(log_dir=tmp_path),
        database,  # type: ignore[arg-type]
    )
    worker = make_worker("worker", make_profile("profile", "model", (GPU_0,)))
    log_path = tmp_path / "failed-worker.log"
    log_path.write_text("engine traceback", encoding="utf-8")
    supervisor.workers[worker.id] = worker
    supervisor._log_paths[worker.id] = log_path

    await supervisor._fail(worker, "unexpected_exit_1")

    assert log_path.exists()
    assert database.events[-1] == (
        "WORKER_FAILED",
        worker.id,
        {
            "reason": "unexpected_exit_1",
            "request_ids": [],
            "log_path": str(log_path),
        },
    )


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 987654
        self.returncode: int | None = None
        self.waited = False

    async def wait(self) -> int:
        self.waited = True
        self.returncode = -signal.SIGKILL
        return self.returncode


@pytest.mark.asyncio
async def test_post_spawn_persistence_failure_kills_and_forgets_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = FakeProcess()
    monkeypatch.setattr("llm_rio.process_cleanup.group_members", lambda _: {})
    monkeypatch.setattr("llm_rio.process_cleanup.gpu_pids", lambda _: set())
    killed: list[tuple[int, signal.Signals]] = []

    async def create_subprocess(*args: str, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("llm_rio.workers.asyncio.create_subprocess_exec", create_subprocess)
    monkeypatch.setattr("llm_rio.process_cleanup.os.getpgid", lambda pid: 24680)
    monkeypatch.setattr(
        "llm_rio.process_cleanup.os.killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    supervisor = WorkerSupervisor(
        Settings(
            capture_worker_engine_logs=False,
            engines=EngineSettings(kvcached_mode="disabled"),
        ),
        FailingDatabase(),  # type: ignore[arg-type]
    )
    profile = make_profile("missing-profile", "model", (GPU_0,))

    with pytest.raises(sqlite3.IntegrityError):
        await supervisor.launch(
            profile=profile,
            gpu_uuids=(GPU_0,),
            model_path="/immutable/model",
            served_model_name="model",
        )

    assert killed == [(process.pid, signal.SIGKILL)]
    assert process.waited
    assert supervisor.workers == {}
    assert supervisor._processes == {}


@pytest.mark.asyncio
async def test_release_starts_idle_grace_from_request_completion() -> None:
    supervisor = WorkerSupervisor(
        Settings(),
        FailingDatabase(),  # type: ignore[arg-type]
    )
    worker = make_worker("worker", make_profile("profile", "model", (GPU_0,)))
    worker.admitted_request_ids.add("request")
    worker.outstanding_token_work = 128
    worker.last_demand_at = datetime.now(UTC) - timedelta(minutes=5)
    supervisor.workers[worker.id] = worker
    before_release = datetime.now(UTC)

    await supervisor.release(worker.id, "request", 128)

    assert worker.admitted_request_ids == set()
    assert worker.outstanding_token_work == 0
    assert worker.last_demand_at >= before_release


def test_internal_worker_key_cannot_be_parsed_as_a_cli_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("llm_rio.workers.secrets.token_urlsafe", lambda _: "-leading-hyphen")

    supervisor = WorkerSupervisor(
        Settings(),
        FailingDatabase(),  # type: ignore[arg-type]
    )

    assert supervisor.internal_api_key == "rio_internal_-leading-hyphen"
    assert not supervisor.internal_api_key.startswith("-")


@pytest.mark.parametrize("failed_engine", [False, True])
async def test_unverified_teardown_keeps_worker_stopping_and_tracked(monkeypatch, failed_engine):
    from unittest.mock import AsyncMock

    from llm_rio.process_cleanup import TeardownError

    supervisor = WorkerSupervisor(Settings(), RecordingDatabase())
    worker = make_worker("worker", make_profile("profile", "model", (GPU_0,)))
    process = FakeProcess()
    worker.process_pid = process.pid
    supervisor.workers[worker.id] = worker
    supervisor._processes[worker.id] = process
    monkeypatch.setattr(
        "llm_rio.workers.terminate_engine", AsyncMock(side_effect=TeardownError("still resident"))
    )
    with pytest.raises(TeardownError):
        if failed_engine:
            await supervisor._fail(worker, "unexpected_exit_1")
        else:
            await supervisor.stop(worker.id, force=True)
    assert worker.state is RuntimeState.STOPPING
    assert worker.process_pid == process.pid
    assert supervisor._processes[worker.id] is process
