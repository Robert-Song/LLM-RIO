from __future__ import annotations

from typing import Any

import click
import httpx

from llm_rio import connection


def request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {connection.api_key()}"}
    try:
        response = httpx.request(
            method,
            f"{connection.base_url()}{path}",
            headers=headers,
            json=json_body,
            timeout=(
                httpx.Timeout(60.0, read=600.0)
                if method.upper() == "POST" and path == "/admin/usage/summarize"
                else 60.0
            ),
        )
    except httpx.HTTPError as exc:
        raise click.ClickException(f"Request failed: {exc}") from exc
    if not response.is_success:
        raise click.ClickException(f"HTTP {response.status_code}: {response.text}")
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def key_records() -> list[dict[str, Any]]:
    payload = request("GET", "/admin/keys")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("The server returned an invalid API-key list.")
    return [record for record in data if isinstance(record, dict)]


def key_record(selector: str) -> dict[str, Any]:
    matches = [
        record
        for record in key_records()
        if selector == record.get("nickname") or selector == record.get("api_key")
    ]
    if not matches:
        raise click.ClickException(
            f"API key '{selector}' was not found. Use its nickname or full API key."
        )
    if len(matches) > 1:
        raise click.ClickException(f"API key selector '{selector}' is ambiguous.")
    return matches[0]


def model_records() -> list[dict[str, Any]]:
    payload = request("GET", "/staff/models")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("The server returned an invalid model list.")
    return [record for record in data if isinstance(record, dict)]


def model_record(nickname: str) -> dict[str, Any]:
    matches = [record for record in model_records() if nickname == record.get("nickname")]
    if not matches:
        raise click.ClickException(f"Model nickname '{nickname}' was not found.")
    return matches[0]


def model_profiles(nickname: str) -> dict[str, Any]:
    model = model_record(nickname)
    payload = request("GET", f"/admin/models/{model['id']}/profiles")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise click.ClickException("The server returned an invalid placement-profile list.")
    return payload


def profile_record(nickname: str, selector: str) -> dict[str, Any]:
    records = model_profiles(nickname).get("data")
    if not isinstance(records, list):
        raise click.ClickException("The server returned an invalid placement-profile list.")
    profiles = [record for record in records if isinstance(record, dict)]
    if selector.isdecimal():
        position = int(selector) - 1
        if 0 <= position < len(profiles):
            return profiles[position]
    matches = [record for record in profiles if selector == record.get("id")]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise click.ClickException(
            "Placement profile was not found. Use its full ID or its number from `models profiles`."
        )
    raise click.ClickException("Placement profile selector is ambiguous.")


class AdminClient:
    """Shared typed HTTP operations; presentation belongs to CLI/TUI."""

    request = staticmethod(request)
    key_records = staticmethod(key_records)
    key_record = staticmethod(key_record)
    model_records = staticmethod(model_records)
    model_record = staticmethod(model_record)
    model_profiles = staticmethod(model_profiles)
    profile_record = staticmethod(profile_record)
