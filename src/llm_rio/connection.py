from __future__ import annotations

import os
import sqlite3
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import click
from pydantic import ValidationError
from pydantic_settings import SettingsError

from llm_rio.config import ServingMode, Settings
from llm_rio.security import ApiKeyVault, default_key_vault_path


def settings(config: Path | None = None, *, mode: ServingMode | None = None) -> Settings:
    options: dict[str, Any] = {} if config is None else {"config_file": config}
    if config is not None and not config.is_file():
        raise click.ClickException(f"Configuration file not found: {config}")
    if mode is not None:
        options["serving_mode"] = mode
    try:
        resolved = Settings(**options)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        details = "\n".join(
            f"  {'.'.join(map(str, error['loc'])) or 'settings'}: {error['msg']}"
            for error in errors
        )
        hint = (
            "Select serving_mode = 'queue', 'vllm-sleep', or 'kv-cached' in TOML, "
            "LLMRIO_SERVING_MODE, or serve --mode."
        )
        if any(
            str(part).startswith("prism_") or part == "kvcached_mode"
            for error in errors
            for part in error["loc"]
        ):
            hint += (
                " Beta settings are unsupported: preserve the old configuration and database, "
                "then use a release configuration with a new database "
                "(see docs/CONFIGURATION.md and examples/config/)."
            )
        raise click.ClickException(f"Invalid configuration:\n{details}\n{hint}") from None
    except (OSError, tomllib.TOMLDecodeError, SettingsError) as exc:
        raise click.ClickException(f"Cannot load configuration: {exc}") from None
    if not resolved.config_file.is_file() and ("config_file" in resolved.model_fields_set):
        raise click.ClickException(f"Configuration file not found: {resolved.config_file}")
    return resolved


def base_url() -> str:
    configured = os.environ.get("LLMRIO_API_URL")
    if configured:
        return configured.rstrip("/")
    resolved = settings()
    return f"http://127.0.0.1:{resolved.api_port}"


def api_key() -> str:
    value = os.environ.get("LLMRIO_API_KEY")
    if value:
        return value
    hostname = urlsplit(base_url()).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise click.ClickException(
            "Remote administration requires LLMRIO_API_KEY; local commands recover an admin "
            "credential automatically from the protected host database."
        )
    resolved = settings()
    database_path = resolved.database_path.resolve()
    vault_path = default_key_vault_path(database_path)
    if not database_path.exists():
        raise click.ClickException(
            f"Local database not found at {database_path}. Start LLM-RIO once before using "
            "management commands."
        )
    if not vault_path.exists():
        raise click.ClickException(f"Local API-key vault not found at {vault_path}.")
    try:
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                """
                SELECT encrypted_api_key FROM api_keys
                 WHERE role = 'admin' AND active = 1
                 ORDER BY created_at, id LIMIT 1
                """
            ).fetchone()
    except sqlite3.Error as exc:
        raise click.ClickException(f"Cannot read local administrator data: {exc}") from exc
    if row is None:
        raise click.ClickException("No active local administrator key exists.")
    try:
        return ApiKeyVault(vault_path).decrypt(str(row[0]))
    except (OSError, RuntimeError) as exc:
        raise click.ClickException(f"Cannot recover a local administrator key: {exc}") from exc
