from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import aiosqlite

from llm_rio.errors import RioError

_METRICS = (
    "request_count",
    "successful_requests",
    "failed_requests",
    "reserved_tokens",
    "charged_tokens",
    "prompt_tokens",
    "completion_tokens",
    "output_tokens_for_rate",
    "active_output_seconds",
    "timed_requests",
)


def _metric_values(row: aiosqlite.Row | None) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    for name in _METRICS:
        raw = row[name] if row is not None else 0
        values[name] = float(raw or 0) if name == "active_output_seconds" else int(raw or 0)
    return values


async def summarize_usage_records(
    connection: aiosqlite.Connection,
    *,
    through: datetime,
) -> dict[str, Any]:
    """Atomically replace the current window, extend lifetime totals, and remove raw rows."""
    if through.tzinfo is None or through.utcoffset() is None:
        raise RioError(
            "invalid_summary_cutoff",
            "The usage summary cutoff must include a timezone",
            status_code=422,
        )
    cutoff = through.astimezone(UTC).isoformat()
    summarized_at = datetime.now(UTC).isoformat()

    previous_current = await (
        await connection.execute(
            "SELECT period_end FROM usage_summary_periods WHERE window = 'current'"
        )
    ).fetchone()
    if previous_current is not None and cutoff <= str(previous_current["period_end"]):
        raise RioError(
            "summary_cutoff_not_advanced",
            "The usage summary cutoff must be later than the previous cutoff",
            status_code=409,
            details={"previous_cutoff": previous_current["period_end"]},
        )

    rows = list(
        await (
            await connection.execute(
                """
            SELECT q.account_id, q.key_id, q.model_id,
                   COUNT(*) AS request_count,
                   SUM(CASE WHEN r.state = 'COMPLETED' THEN 1 ELSE 0 END)
                       AS successful_requests,
                   SUM(CASE WHEN r.state = 'COMPLETED' THEN 0 ELSE 1 END)
                       AS failed_requests,
                   SUM(q.reserved_tokens) AS reserved_tokens,
                   SUM(COALESCE(q.actual_tokens, 0)) AS charged_tokens,
                   SUM(COALESCE(r.actual_prompt_tokens, 0)) AS prompt_tokens,
                   SUM(COALESCE(r.actual_completion_tokens, 0)) AS completion_tokens,
                   SUM(CASE
                         WHEN COALESCE(r.actual_completion_tokens, 0) > 0
                          AND r.admitted_at IS NOT NULL
                          AND r.completed_at IS NOT NULL
                          AND julianday(r.completed_at) > julianday(r.admitted_at)
                         THEN r.actual_completion_tokens ELSE 0
                       END) AS output_tokens_for_rate,
                   SUM(CASE
                         WHEN COALESCE(r.actual_completion_tokens, 0) > 0
                          AND r.admitted_at IS NOT NULL
                          AND r.completed_at IS NOT NULL
                          AND julianday(r.completed_at) > julianday(r.admitted_at)
                         THEN (julianday(r.completed_at) - julianday(r.admitted_at)) * 86400.0
                         ELSE 0
                       END) AS active_output_seconds,
                   SUM(CASE
                         WHEN COALESCE(r.actual_completion_tokens, 0) > 0
                          AND r.admitted_at IS NOT NULL
                          AND r.completed_at IS NOT NULL
                          AND julianday(r.completed_at) > julianday(r.admitted_at)
                         THEN 1 ELSE 0
                       END) AS timed_requests,
                   MIN(COALESCE(r.completed_at, q.settled_at)) AS first_completed_at,
                   MAX(COALESCE(r.completed_at, q.settled_at)) AS last_completed_at
              FROM quota_reservations q
              LEFT JOIN inference_requests r ON r.reservation_id = q.id
             WHERE q.state = 'SETTLED' AND q.settled_at IS NOT NULL AND q.settled_at <= ?
             GROUP BY q.account_id, q.key_id, q.model_id
            """,
                (cutoff,),
            )
        ).fetchall()
    )

    batch_metrics: dict[str, int | float] = {
        name: 0.0 if name == "active_output_seconds" else 0 for name in _METRICS
    }
    for row in rows:
        for name, value in _metric_values(row).items():
            batch_metrics[name] += value

    first_completed = min(
        (str(row["first_completed_at"]) for row in rows if row["first_completed_at"]),
        default=cutoff,
    )
    period_start = (
        str(previous_current["period_end"]) if previous_current is not None else first_completed
    )

    await connection.execute("DELETE FROM usage_summaries WHERE window = 'current'")
    insert_columns = ", ".join(_METRICS)
    placeholders = ", ".join("?" for _ in _METRICS)

    update_metrics = ", ".join(
        f"{name} = usage_summaries.{name} + excluded.{name}" for name in _METRICS
    )
    await connection.executemany(
        f"""
        INSERT INTO usage_summaries
            (window, account_id, key_id, model_id, {insert_columns},
             first_completed_at, last_completed_at)
        VALUES ('total', ?, ?, ?, {placeholders}, ?, ?)
        ON CONFLICT(window, account_id, key_id, model_id) DO UPDATE SET
            {update_metrics},
            first_completed_at = MIN(usage_summaries.first_completed_at,
                                     excluded.first_completed_at),
            last_completed_at = MAX(usage_summaries.last_completed_at,
                                    excluded.last_completed_at)
        """,
        [
            (
                row["account_id"],
                row["key_id"],
                row["model_id"],
                *(_metric_values(row)[name] for name in _METRICS),
                row["first_completed_at"],
                row["last_completed_at"],
            )
            for row in rows
        ],
    )

    previous_total = await (
        await connection.execute("SELECT * FROM usage_summary_periods WHERE window = 'total'")
    ).fetchone()
    lifetime_metrics = _metric_values(previous_total)
    for name, value in batch_metrics.items():
        lifetime_metrics[name] += value
    current_metrics: dict[str, int | float] = {
        name: 0.0 if name == "active_output_seconds" else 0 for name in _METRICS
    }
    lifetime_start = (
        str(previous_total["period_start"]) if previous_total is not None else period_start
    )

    period_columns = ", ".join(_METRICS)
    period_placeholders = ", ".join("?" for _ in _METRICS)
    period_updates = ", ".join(f"{name} = excluded.{name}" for name in _METRICS)
    for window, start, metrics in (
        ("current", cutoff, current_metrics),
        ("total", lifetime_start, lifetime_metrics),
    ):
        await connection.execute(
            f"""
            INSERT INTO usage_summary_periods
                (window, period_start, period_end, summarized_at, {period_columns})
            VALUES (?, ?, ?, ?, {period_placeholders})
            ON CONFLICT(window) DO UPDATE SET
                period_start = excluded.period_start,
                period_end = excluded.period_end,
                summarized_at = excluded.summarized_at,
                {period_updates}
            """,
            (
                window,
                start,
                cutoff,
                summarized_at,
                *(metrics[name] for name in _METRICS),
            ),
        )

    eligible = "state = 'SETTLED' AND settled_at IS NOT NULL AND settled_at <= ?"
    ledger_cursor = await connection.execute(
        f"""
        DELETE FROM quota_ledger
         WHERE reservation_id IN (SELECT id FROM quota_reservations WHERE {eligible})
        """,
        (cutoff,),
    )
    request_cursor = await connection.execute(
        f"""
        DELETE FROM inference_requests
         WHERE reservation_id IN (SELECT id FROM quota_reservations WHERE {eligible})
        """,
        (cutoff,),
    )
    reservation_cursor = await connection.execute(
        f"DELETE FROM quota_reservations WHERE {eligible}",
        (cutoff,),
    )
    remaining = await (
        await connection.execute("SELECT COUNT(*) AS count FROM inference_requests")
    ).fetchone()

    return {
        "period_start": period_start,
        "period_end": cutoff,
        "summarized_at": summarized_at,
        "summarized_requests": int(batch_metrics["request_count"]),
        "summary_rows": len(rows),
        "summarized": batch_metrics,
        "deleted": {
            "inference_requests": max(0, request_cursor.rowcount),
            "quota_reservations": max(0, reservation_cursor.rowcount),
            "quota_ledger": max(0, ledger_cursor.rowcount),
        },
        "raw_requests_remaining": int(remaining["count"]) if remaining else 0,
        "current": current_metrics,
        "lifetime": lifetime_metrics,
    }


def _window_payload(
    metrics: dict[str, int | float],
    *,
    period_start: str,
    period_end: str,
) -> dict[str, Any]:
    start = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
    end = datetime.fromisoformat(period_end.replace("Z", "+00:00"))
    elapsed_seconds = max(0.0, (end - start).total_seconds())
    token_usage = int(metrics["prompt_tokens"]) + int(metrics["completion_tokens"])
    active_seconds = float(metrics["active_output_seconds"])
    output_tokens = int(metrics["output_tokens_for_rate"])
    return {
        "period_start": period_start,
        "period_end": period_end,
        **metrics,
        "token_usage": token_usage,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": token_usage / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "tokens_per_minute": token_usage * 60.0 / elapsed_seconds if elapsed_seconds > 0 else 0.0,
        "average_output_tokens_per_second": output_tokens / active_seconds
        if active_seconds > 0
        else None,
    }


async def usage_dashboard(
    connection: aiosqlite.Connection,
    *,
    now: datetime,
) -> dict[str, Any]:
    """Return live current/lifetime usage and per-model popularity."""
    now_iso = now.astimezone(UTC).isoformat()
    current_period = await (
        await connection.execute(
            "SELECT period_start FROM usage_summary_periods WHERE window = 'current'"
        )
    ).fetchone()
    total_period = await (
        await connection.execute(
            "SELECT period_start FROM usage_summary_periods WHERE window = 'total'"
        )
    ).fetchone()

    raw_rows = list(
        await (
            await connection.execute(
                """
                SELECT q.model_id, m.nickname AS model,
                       COUNT(*) AS request_count,
                       SUM(CASE WHEN r.state = 'COMPLETED' THEN 1 ELSE 0 END)
                           AS successful_requests,
                       SUM(CASE WHEN r.state = 'COMPLETED' THEN 0 ELSE 1 END)
                           AS failed_requests,
                       SUM(q.reserved_tokens) AS reserved_tokens,
                       SUM(COALESCE(q.actual_tokens, 0)) AS charged_tokens,
                       SUM(COALESCE(r.actual_prompt_tokens, 0)) AS prompt_tokens,
                       SUM(COALESCE(r.actual_completion_tokens, 0)) AS completion_tokens,
                       SUM(CASE
                             WHEN COALESCE(r.actual_completion_tokens, 0) > 0
                              AND r.admitted_at IS NOT NULL
                              AND r.completed_at IS NOT NULL
                              AND julianday(r.completed_at) > julianday(r.admitted_at)
                             THEN r.actual_completion_tokens ELSE 0
                           END) AS output_tokens_for_rate,
                       SUM(CASE
                             WHEN COALESCE(r.actual_completion_tokens, 0) > 0
                              AND r.admitted_at IS NOT NULL
                              AND r.completed_at IS NOT NULL
                              AND julianday(r.completed_at) > julianday(r.admitted_at)
                             THEN (julianday(r.completed_at) - julianday(r.admitted_at)) * 86400.0
                             ELSE 0
                           END) AS active_output_seconds,
                       SUM(CASE
                             WHEN COALESCE(r.actual_completion_tokens, 0) > 0
                              AND r.admitted_at IS NOT NULL
                              AND r.completed_at IS NOT NULL
                              AND julianday(r.completed_at) > julianday(r.admitted_at)
                             THEN 1 ELSE 0
                           END) AS timed_requests,
                       MIN(q.settled_at) AS first_completed_at
                  FROM quota_reservations q
                  JOIN model_catalog m ON m.id = q.model_id
                  LEFT JOIN inference_requests r ON r.reservation_id = q.id
                 WHERE q.state = 'SETTLED'
                 GROUP BY q.model_id, m.nickname
                """
            )
        ).fetchall()
    )
    raw_metrics: dict[str, int | float] = {
        name: 0.0 if name == "active_output_seconds" else 0 for name in _METRICS
    }
    for row in raw_rows:
        for name, value in _metric_values(row).items():
            raw_metrics[name] += value

    summary_metrics_row = await (
        await connection.execute(
            f"""
            SELECT {", ".join(f"COALESCE(SUM({name}), 0) AS {name}" for name in _METRICS)}
              FROM usage_summaries WHERE window = 'total'
            """
        )
    ).fetchone()
    total_metrics = _metric_values(summary_metrics_row)
    for name, value in raw_metrics.items():
        total_metrics[name] += value

    earliest_raw = min(
        (str(row["first_completed_at"]) for row in raw_rows if row["first_completed_at"]),
        default=now_iso,
    )
    current_start = (
        str(current_period["period_start"]) if current_period is not None else earliest_raw
    )
    total_start = str(total_period["period_start"]) if total_period is not None else earliest_raw

    summary_models = list(
        await (
            await connection.execute(
                """
                SELECT s.model_id, m.nickname AS model,
                       SUM(s.request_count) AS request_count,
                       SUM(s.charged_tokens) AS charged_tokens,
                       SUM(s.prompt_tokens) AS prompt_tokens,
                       SUM(s.completion_tokens) AS completion_tokens
                  FROM usage_summaries s
                  JOIN model_catalog m ON m.id = s.model_id
                 WHERE s.window = 'total'
                 GROUP BY s.model_id, m.nickname
                """
            )
        ).fetchall()
    )

    def popularity(
        base_rows: list[aiosqlite.Row],
        extra_rows: list[aiosqlite.Row] | None = None,
    ) -> list[dict[str, Any]]:
        by_model: dict[str, dict[str, Any]] = {}
        all_rows = [*base_rows, *(extra_rows or [])]
        for row in all_rows:
            model_id = str(row["model_id"])
            item = by_model.setdefault(
                model_id,
                {
                    "model_id": model_id,
                    "model": str(row["model"]),
                    "request_count": 0,
                    "charged_tokens": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )
            for field in (
                "request_count",
                "charged_tokens",
                "prompt_tokens",
                "completion_tokens",
            ):
                item[field] += int(row[field] or 0)
        ranked = sorted(
            by_model.values(),
            key=lambda item: (
                -(int(item["prompt_tokens"]) + int(item["completion_tokens"])),
                str(item["model"]),
            ),
        )
        grand_total = sum(
            int(item["prompt_tokens"]) + int(item["completion_tokens"]) for item in ranked
        )
        for rank, item in enumerate(ranked, 1):
            item["rank"] = rank
            item["token_usage"] = int(item["prompt_tokens"]) + int(item["completion_tokens"])
            item["share"] = item["token_usage"] / grand_total if grand_total else 0.0
        return ranked

    return {
        "generated_at": now_iso,
        "current": _window_payload(
            raw_metrics,
            period_start=current_start,
            period_end=now_iso,
        ),
        "total": _window_payload(
            total_metrics,
            period_start=total_start,
            period_end=now_iso,
        ),
        "model_popularity": {
            "current": popularity(raw_rows),
            "total": popularity(summary_models, raw_rows),
        },
    }
