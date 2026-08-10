from __future__ import annotations

import json
import signal
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from test_scheduler_contract import GPU_0, make_profile, make_worker

from llm_rio.config import Settings
from llm_rio.profiles import ProfileRepository, profile_to_dict
from llm_rio.tool_support import detect_vllm_tool_parser
from llm_rio.workers import WorkerSupervisor


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
    killed: list[tuple[int, signal.Signals]] = []

    async def create_subprocess(*args: str, **kwargs: Any) -> FakeProcess:
        return process

    monkeypatch.setattr("llm_rio.workers.asyncio.create_subprocess_exec", create_subprocess)
    monkeypatch.setattr("llm_rio.workers.os.getpgid", lambda pid: 24680)
    monkeypatch.setattr(
        "llm_rio.workers.os.killpg",
        lambda pgid, sig: killed.append((pgid, sig)),
    )
    supervisor = WorkerSupervisor(
        Settings(capture_worker_engine_logs=False),
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

    assert killed == [(24680, signal.SIGKILL)]
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
