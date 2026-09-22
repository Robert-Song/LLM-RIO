from __future__ import annotations

import typer

from llm_rio import admin_client as client_api
from llm_rio.commands.common import (
    _print,
    _print_key_record,
    _print_key_records,
    _set_key_limit,
    keys_app,
)
from llm_rio.commands.common import app as app
from llm_rio.domain import Role


@keys_app.command("list")
def list_keys(json_output: bool = typer.Option(False, "--json")) -> None:
    """List every API key, including full recoverable key values."""
    records = client_api.key_records()
    if json_output:
        internal_fields = {"id", "quota_account_id", "usage_baseline_tokens"}
        public_records = [
            {key: value for key, value in record.items() if key not in internal_fields}
            for record in records
        ]
        _print({"data": public_records})
    else:
        _print_key_records(records)


@keys_app.command("show")
def show_key(key: str) -> None:
    """Show one API key selected by nickname or full key."""
    _print_key_record(client_api.key_record(key))


@keys_app.command("usage")
def show_key_usage(key: str) -> None:
    """Show quota and usage for one API key selected by nickname or full key."""
    _print_key_record(client_api.key_record(key))


@keys_app.command("create")
def create_key(
    nickname: str,
    role: Role = typer.Option(Role.USER, "--role"),
    limit_tokens: int | None = typer.Option(None, "--limit", "--balance"),
    account_id: str | None = typer.Option(None, "--account-id"),
    grant: list[str] | None = typer.Option(None, "--grant", help="Model nickname to grant."),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Optional custom rio_ API key; generated when omitted."
    ),
) -> None:
    """Create an API key and print its complete value."""
    result = client_api.request(
        "POST",
        "/admin/keys",
        json_body={
            "nickname": nickname,
            "role": role.value,
            "limit_tokens": limit_tokens,
            "quota_account_id": account_id,
            "models": grant or [],
            "api_key": api_key,
        },
    )
    typer.echo(f"API key created for '{result['nickname']}'.")
    typer.echo(f"API key: {result['api_key']}")


@keys_app.command("rotate")
def rotate_key(key: str) -> None:
    """Rotate an API key selected by nickname or full key."""
    record = client_api.key_record(key)
    result = client_api.request("POST", f"/admin/keys/{record['id']}/rotate")
    typer.echo(f"API key rotated for '{result['nickname']}'.")
    typer.echo(f"API key: {result['api_key']}")


@keys_app.command("revoke")
def revoke_key(key: str) -> None:
    """Deactivate an API key selected by nickname or full key."""
    record = client_api.key_record(key)
    client_api.request("POST", f"/admin/keys/{record['id']}/revoke")
    typer.echo(f"API key '{record['nickname']}' revoked.")


@keys_app.command("restore")
def restore_key(key: str) -> None:
    """Reactivate a revoked API key selected by nickname or full key."""
    record = client_api.key_record(key)
    client_api.request("POST", f"/admin/keys/{record['id']}/restore")
    typer.echo(f"API key '{record['nickname']}' restored.")


@keys_app.command("delete")
def delete_key(key: str) -> None:
    """Remove credential utility while retaining audit history."""
    record = client_api.key_record(key)
    client_api.request("DELETE", f"/admin/keys/{record['id']}")
    typer.echo(f"API key '{record['nickname']}' deleted.")


@keys_app.command("quota")
def update_quota(
    key: str,
    limit_tokens: int = typer.Option(..., "--limit", "--balance"),
    unlimited: bool | None = typer.Option(None, "--unlimited/--limited"),
) -> None:
    """Set the token quota for this usage period."""
    _set_key_limit(key, limit_tokens, unlimited)


@keys_app.command("reset-usage")
def reset_usage(key: str) -> None:
    """Reset current-period usage while preserving lifetime audit totals."""
    record = client_api.key_record(key)
    result = client_api.request("POST", f"/admin/keys/{record['id']}/usage/reset")
    typer.echo(f"Usage reset for '{record['nickname']}'.")
    _print(result)
