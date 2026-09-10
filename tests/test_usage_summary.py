from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llm_rio import cli
from llm_rio.domain import Role
from llm_rio.security import Principal
from llm_rio.storage import Database


async def _add_model(database: Database, model_id: str) -> None:
    now = datetime.now(UTC).isoformat()
    await database.execute(
        """
        INSERT INTO model_catalog
            (id, nickname, huggingface_repo, state, created_by_key_id, created_at, updated_at)
        VALUES (?, ?, ?, 'AVAILABLE', 'user-key', ?, ?)
        """,
        (model_id, model_id, f"org/{model_id}", now, now),
    )


async def _complete_request(
    database: Database,
    principal: Principal,
    *,
    request_id: str,
    model_id: str,
    charged_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    duration_seconds: int,
) -> None:
    reservation_id = await database.reserve_quota(
        request_id=request_id,
        idempotency_hash=f"hash-{request_id}",
        principal=principal,
        model_id=model_id,
        estimated_tokens=charged_tokens,
    )
    await database.mark_request_admitted(request_id, "worker")
    await database.settle_quota(
        reservation_id=reservation_id,
        actual_tokens=charged_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    completed_at = datetime.now(UTC)
    admitted_at = completed_at - timedelta(seconds=duration_seconds)
    await database.execute(
        """
        UPDATE inference_requests SET admitted_at = ?, completed_at = ? WHERE id = ?
        """,
        (admitted_at.isoformat(), completed_at.isoformat(), request_id),
    )


@pytest.mark.asyncio
async def test_summarize_replaces_current_and_extends_lifetime_without_breaking_usage(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="user-key",
            nickname="user",
            role=Role.USER,
            account_id="account",
            account_nickname="account",
            prefix="rio_user_prefix_123456789",
            api_key="rio_user_prefix_123456789_secret",
            limit_tokens=1_000,
            unlimited=False,
        )
        await _add_model(database, "model-a")
        await _add_model(database, "model-b")
        principal = Principal("user-key", "user", Role.USER, "account", False)

        await _complete_request(
            database,
            principal,
            request_id="request-a",
            model_id="model-a",
            charged_tokens=75,
            prompt_tokens=50,
            completion_tokens=25,
            duration_seconds=10,
        )
        await database.reserve_quota(
            request_id="active-request",
            idempotency_hash="hash-active",
            principal=principal,
            model_id="model-a",
            estimated_tokens=10,
        )

        before_first = await database.dashboard_usage()
        assert before_first["current"]["token_usage"] == 75
        assert before_first["current"]["average_output_tokens_per_second"] == pytest.approx(
            2.5, abs=0.01
        )
        assert before_first["model_popularity"]["current"][0]["model"] == "model-a"

        first = await database.summarize_usage(datetime.now(UTC))
        assert first["summarized_requests"] == 1
        assert first["deleted"]["inference_requests"] == 1
        assert first["raw_requests_remaining"] == 1
        assert first["summarized"]["output_tokens_for_rate"] == 25
        assert first["summarized"]["active_output_seconds"] == pytest.approx(10, abs=0.01)
        assert first["current"]["request_count"] == 0
        after_first = await database.dashboard_usage()
        assert after_first["current"]["token_usage"] == 0
        assert after_first["total"]["token_usage"] == 75

        usage = await database.usage(principal)
        assert usage["lifetime_charged_tokens"] == 75
        assert usage["settled_requests"] == 1
        keys = await database.list_keys()
        assert keys[0]["lifetime_charged_tokens"] == 75
        assert keys[0]["key_lifetime_charged_tokens"] == 75

        reset = await database.reset_usage("user-key")
        assert reset is not None
        usage_after_reset = await database.usage(principal)
        assert usage_after_reset["lifetime_charged_tokens"] == 75
        assert usage_after_reset["used_tokens"] == 0

        await _complete_request(
            database,
            principal,
            request_id="request-b",
            model_id="model-b",
            charged_tokens=30,
            prompt_tokens=20,
            completion_tokens=10,
            duration_seconds=5,
        )
        live_second = await database.dashboard_usage()
        assert live_second["current"]["token_usage"] == 30
        assert live_second["total"]["token_usage"] == 105
        assert live_second["current"]["average_output_tokens_per_second"] == pytest.approx(
            2.0, abs=0.01
        )
        assert live_second["model_popularity"]["current"][0]["model"] == "model-b"
        assert [item["model"] for item in live_second["model_popularity"]["total"]] == [
            "model-a",
            "model-b",
        ]
        second = await database.summarize_usage(datetime.now(UTC))
        assert second["current"]["request_count"] == 0
        assert second["summarized"]["charged_tokens"] == 30
        assert second["lifetime"]["request_count"] == 2
        assert second["lifetime"]["charged_tokens"] == 105

        summaries = await database.fetchall(
            """
            SELECT window, model_id, charged_tokens, completion_tokens
              FROM usage_summaries ORDER BY window, model_id
            """
        )
        assert [tuple(row) for row in summaries] == [
            ("total", "model-a", 75, 25),
            ("total", "model-b", 30, 10),
        ]
        usage = await database.usage(principal)
        assert usage["lifetime_charged_tokens"] == 105
        assert usage["used_tokens"] == 30
        assert usage["settled_requests"] == 2
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_live_requests_include_queue_metadata(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    await database.open()
    try:
        await database.create_key(
            key_id="user-key",
            nickname="researcher",
            role=Role.USER,
            account_id="account",
            account_nickname="account",
            prefix="rio_user_prefix_123456789",
            api_key="rio_user_prefix_123456789_secret",
            limit_tokens=1_000,
            unlimited=False,
        )
        await _add_model(database, "model-a")
        principal = Principal("user-key", "researcher", Role.USER, "account", False)
        await database.reserve_quota(
            request_id="request-1",
            idempotency_hash="hash-request-1",
            principal=principal,
            model_id="model-a",
            estimated_tokens=512,
            estimated_prompt_tokens=24,
        )

        queued = await database.live_requests()
        assert queued[0]["state"] == "QUEUED"
        assert queued[0]["api_key"] == "researcher"
        assert queued[0]["model"] == "model-a"
        assert queued[0]["estimated_prompt_tokens"] == 24
        assert queued[0]["estimated_tokens"] == 512

        assert await database.mark_request_admitted("request-1", "worker-1")
        admitted = await database.live_requests()
        assert admitted[0]["state"] == "ADMITTED"
    finally:
        await database.close()


def test_llmctl_summarize_calls_admin_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def request(
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        calls.append((method, path, json_body))
        return {
            "period_end": "2026-08-21T00:00:00+00:00",
            "summarized_requests": 4,
            "deleted": {
                "inference_requests": 4,
                "quota_reservations": 4,
                "quota_ledger": 8,
            },
            "raw_requests_remaining": 1,
            "current": {},
            "lifetime": {},
        }

    monkeypatch.setattr(cli, "_request", request)
    result = CliRunner().invoke(
        cli.app,
        ["summarize", "--through", "2026-08-21T00:00:00+00:00", "--json"],
    )

    assert result.exit_code == 0
    assert calls == [
        (
            "POST",
            "/admin/usage/summarize",
            {"through": "2026-08-21T00:00:00+00:00"},
        )
    ]
