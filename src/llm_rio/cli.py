from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import typer
import uvicorn
import click

from llm_rio.api.app import create_app
from llm_rio.config import Settings
from llm_rio.domain import Role
from llm_rio.inventory import InventoryError, discover_inventory
from llm_rio.security import ApiKeyVault, default_key_vault_path

app = typer.Typer(help="Local administration and lifecycle commands for LLM-RIO.")
keys_app = typer.Typer(help="Manage API keys.")
models_app = typer.Typer(help="Manage the model catalog.")
maintenance_app = typer.Typer(help="Drain or resume this machine.")
app.add_typer(keys_app, name="keys")
app.add_typer(models_app, name="models")
app.add_typer(maintenance_app, name="maintenance")


def _settings(config: Path) -> Settings:
    return Settings(config_file=config)


def _base_url() -> str:
    configured = os.environ.get("LLMRIO_API_URL")
    if configured:
        return configured.rstrip("/")
    settings = _settings(Path(os.environ.get("LLMRIO_CONFIG", "config.toml")))
    return f"http://127.0.0.1:{settings.api_port}"


def _api_key() -> str:
    value = os.environ.get("LLMRIO_API_KEY")
    if value:
        return value
    hostname = urlsplit(_base_url()).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise click.ClickException(
            "Remote administration requires LLMRIO_API_KEY; local commands recover an admin "
            "credential automatically from the protected host database."
        )
    config_path = Path(os.environ.get("LLMRIO_CONFIG", "config.toml"))
    settings = _settings(config_path)
    database_path = settings.database_path.resolve()
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


def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {_api_key()}"}
    try:
        response = httpx.request(
            method,
            f"{_base_url()}{path}",
            headers=headers,
            json=json_body,
            timeout=60.0,
        )
    except httpx.HTTPError as exc:
        typer.echo(f"Request failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    if not response.is_success:
        typer.echo(f"HTTP {response.status_code}: {response.text}", err=True)
        raise typer.Exit(1)
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def _print(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _key_records() -> list[dict[str, Any]]:
    payload = _request("GET", "/admin/keys")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("The server returned an invalid API-key list.")
    return [record for record in data if isinstance(record, dict)]


def _key_record(selector: str) -> dict[str, Any]:
    matches = [
        record
        for record in _key_records()
        if selector == record.get("nickname") or selector == record.get("api_key")
    ]
    if not matches:
        raise click.ClickException(
            f"API key '{selector}' was not found. Use its nickname or full API key."
        )
    if len(matches) > 1:
        raise click.ClickException(f"API key selector '{selector}' is ambiguous.")
    return matches[0]


def _model_records() -> list[dict[str, Any]]:
    payload = _request("GET", "/staff/models")
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("The server returned an invalid model list.")
    return [record for record in data if isinstance(record, dict)]


def _model_record(nickname: str) -> dict[str, Any]:
    matches = [record for record in _model_records() if nickname == record.get("nickname")]
    if not matches:
        raise click.ClickException(f"Model nickname '{nickname}' was not found.")
    return matches[0]


def _job_id_for_model(nickname: str) -> str:
    model = _model_record(nickname)
    job = model.get("registration_job")
    if not isinstance(job, dict) or not isinstance(job.get("id"), str):
        raise click.ClickException(f"Model '{nickname}' has no registration job.")
    return job["id"]


def _job_id_from_selector(job_or_model: str) -> str:
    """Accept a job UUID or a model nickname so review does not require internal IDs."""
    try:
        return _job_id_for_model(job_or_model)
    except click.ClickException:
        return job_or_model


def _review_guidance(stage: object) -> str:
    guidance = {
        "resolve": (
            "Verify the Hugging Face repository, requested revision, and HF token access."
        ),
        "disk_capacity": "Free enough model-store space, then retry the registration.",
        "inspection": "Confirm the snapshot contains a supported config and model-weight files.",
        "candidate_shapes": (
            "Use a smaller or quantized model, or make enough homogeneous GPU VRAM available."
        ),
        "engine_launch": (
            "Check the validation log and confirm the configured vLLM executable can start."
        ),
        "engine_startup": (
            "Read the validation log, then correct the engine, CUDA, or model compatibility issue."
        ),
        "generation": (
            "Read the validation log and correct the model or engine compatibility issue."
        ),
        "streaming_contract": (
            "Read the validation log and correct the model or engine streaming compatibility issue."
        ),
        "llama_cpp_launch": (
            "Check the validation log and confirm the configured llama.cpp executable can start."
        ),
    }
    return guidance.get(
        str(stage),
        "Review the failure details and traceback, correct the host or model issue, then retry.",
    )


def _print_model_job(job: dict[str, Any]) -> None:
    typer.echo(f"Model: {job.get('nickname', '(unknown)')}")
    typer.echo(f"Registration job: {job.get('id')}")
    typer.echo(f"Status: {job.get('state')} / {job.get('catalog_state')}")
    typer.echo(f"Stage: {job.get('stage')}")
    failure = job.get("failure")
    if not isinstance(failure, dict):
        return
    typer.echo(f"Failure: {failure.get('message', '(no message recorded)')}")
    details = failure.get("details")
    if isinstance(details, dict):
        log_path = details.get("log_path")
        if log_path:
            typer.echo(f"Diagnostic log: {log_path}")
    typer.echo("\nAdministrator action:")
    typer.echo(f"  1. {_review_guidance(failure.get('stage', job.get('stage')))}")
    typer.echo(f"  2. Retry: ./llmctl models retry {job.get('id')}")
    typer.echo(
        f"  3. If this model will not be fixed, disable it: "
        f"./llmctl models disable {job.get('nickname')}"
    )


def _print_model_records(records: list[dict[str, Any]]) -> None:
    if not records:
        typer.echo("No models found.")
        return
    typer.echo("Models\n")
    for model in records:
        typer.echo(str(model.get("nickname")))
        typer.echo(f"   State: {model.get('state')}")
        typer.echo(f"   Repository: {model.get('huggingface_repo')}")
        job = model.get("registration_job")
        if isinstance(job, dict):
            typer.echo(f"   Registration job: {job.get('id')}")
            typer.echo(f"   Job: {job.get('state')} at {job.get('stage')}")
            failure = job.get("failure")
            if isinstance(failure, dict):
                typer.echo(f"   Failure: {failure.get('message', '(no message recorded)')}")
            if model.get("state") == "NEEDS_ADMIN_REVIEW":
                typer.echo(f"   Next: ./llmctl models review {model.get('nickname')}")
        typer.echo()


def _print_key_record(record: dict[str, Any], number: int | None = None) -> None:
    heading = f"{number}. {record.get('nickname')}" if number is not None else str(
        record.get("nickname")
    )
    typer.echo(heading)
    typer.echo(f"   API key: {record.get('api_key')}")
    typer.echo(f"   Role: {record.get('role')}")
    typer.echo(f"   Quota account: {record.get('account_nickname')}")
    typer.echo(f"   Active: {'yes' if record.get('active') else 'no'}")
    if record.get("unlimited"):
        typer.echo("   Token limit: unlimited")
    else:
        typer.echo(f"   Token limit: {int(record.get('limit_tokens') or 0):,}")
        typer.echo(f"   Used since reset: {int(record.get('used_tokens') or 0):,}")
        typer.echo(f"   Remaining: {int(record.get('balance_tokens') or 0):,}")
    typer.echo(
        f"   Lifetime charged: {int(record.get('key_lifetime_charged_tokens') or 0):,}"
    )
    granted_models = record.get("granted_models") or []
    typer.echo(f"   Models: {', '.join(granted_models) if granted_models else '(none)'}")
    typer.echo(f"   Created: {record.get('created_at')}")
    typer.echo(f"   Last used: {record.get('last_used_at') or 'never'}")


def _print_key_records(records: list[dict[str, Any]]) -> None:
    if not records:
        typer.echo("No API keys found.")
        return
    typer.echo("API keys\n")
    for index, record in enumerate(records, 1):
        _print_key_record(record, index)
        typer.echo()


@app.command()
def serve(
    config: Path = typer.Option(Path("config.toml"), "--config", help="TOML configuration file"),
) -> None:
    """Run the machine-local API and scheduler."""
    settings = _settings(config)
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = (
        "%(asctime)s | %(levelprefix)s %(message)s"
    )
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=False,
        log_config=log_config,
        workers=1,
    )


@app.command()
def doctor(
    config: Path = typer.Option(Path("config.toml"), "--config"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect host prerequisites without loading a model."""
    settings = _settings(config)
    report: dict[str, Any] = {
        "machine_id": settings.machine_id,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executables": {
            "nvidia-smi": shutil.which("nvidia-smi"),
            "vllm": shutil.which(settings.engines.vllm_executable),
            "llama.cpp": shutil.which(settings.engines.llama_cpp_executable),
        },
        "paths": {},
        "errors": [],
    }
    for name, path in {
        "database_parent": settings.database_path.parent,
        "model_store": settings.model_store,
        "log_dir": settings.log_dir,
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        report["paths"][name] = {"path": str(path.resolve()), "writable": os.access(path, os.W_OK)}
    try:
        inventory = discover_inventory(settings.machine_id, settings.managed_gpu_uuids)
        report["inventory"] = {
            "driver_version": inventory.driver_version,
            "cuda_driver_version": inventory.cuda_driver_version,
            "fingerprint": inventory.fingerprint,
            "topology_hash": inventory.topology_hash,
            "gpus": [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "vram_mib": gpu.total_vram_mib,
                    "compute_capability": gpu.compute_capability,
                    "pci_bus_id": gpu.pci_bus_id,
                }
                for gpu in inventory.gpus
            ],
        }
    except InventoryError as exc:
        report["errors"].append({"stage": "inventory", "message": str(exc)})
    if not report["executables"]["vllm"]:
        report["errors"].append({"stage": "engine", "message": "vllm executable not found"})
    if json_output:
        _print(report)
    else:
        typer.echo(json.dumps(report, indent=2))
    if report["errors"]:
        raise typer.Exit(1)


@keys_app.command("list")
def list_keys(json_output: bool = typer.Option(False, "--json")) -> None:
    """List every API key, including full recoverable key values."""
    records = _key_records()
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
    _print_key_record(_key_record(key))


@keys_app.command("usage")
def show_key_usage(key: str) -> None:
    """Show quota and usage for one API key selected by nickname or full key."""
    _print_key_record(_key_record(key))


@keys_app.command("create")
def create_key(
    nickname: str,
    role: Role = typer.Option(Role.USER, "--role"),
    limit_tokens: int = typer.Option(1_000_000, "--limit", "--balance"),
    unlimited: bool = typer.Option(False, "--unlimited"),
    account_id: str | None = typer.Option(None, "--account-id"),
    grant: list[str] | None = typer.Option(None, "--grant", help="Model nickname to grant."),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Optional custom rio_ API key; generated when omitted."
    ),
) -> None:
    """Create an API key and print its complete value."""
    result = _request(
        "POST",
        "/admin/keys",
        json_body={
            "nickname": nickname,
            "role": role.value,
            "limit_tokens": limit_tokens,
            "unlimited": unlimited,
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
    record = _key_record(key)
    result = _request("POST", f"/admin/keys/{record['id']}/rotate")
    typer.echo(f"API key rotated for '{result['nickname']}'.")
    typer.echo(f"API key: {result['api_key']}")


@keys_app.command("revoke")
def revoke_key(key: str) -> None:
    """Deactivate an API key selected by nickname or full key."""
    record = _key_record(key)
    _request("POST", f"/admin/keys/{record['id']}/revoke")
    typer.echo(f"API key '{record['nickname']}' revoked.")


@keys_app.command("delete")
def delete_key(key: str) -> None:
    """Remove credential utility while retaining audit history."""
    record = _key_record(key)
    _request("DELETE", f"/admin/keys/{record['id']}")
    typer.echo(f"API key '{record['nickname']}' deleted.")


@keys_app.command("remove")
def remove_key(key: str) -> None:
    """Alias for keys delete."""
    delete_key(key)


def _set_key_limit(key: str, limit_tokens: int, unlimited: bool | None) -> None:
    record = _key_record(key)
    resolved_unlimited = bool(record.get("unlimited")) if unlimited is None else unlimited
    _request(
        "PUT",
        f"/admin/keys/{record['id']}/quota",
        json_body={"limit_tokens": limit_tokens, "unlimited": resolved_unlimited},
    )
    typer.echo(f"Token limit updated for '{record['nickname']}'.")


@keys_app.command("limit")
def update_limit(
    key: str,
    limit_tokens: int = typer.Option(..., "--limit", "--balance"),
    unlimited: bool | None = typer.Option(None, "--unlimited/--limited"),
) -> None:
    """Set the lifetime token limit for the current usage period."""
    _set_key_limit(key, limit_tokens, unlimited)


@keys_app.command("quota")
def update_quota(
    key: str,
    limit_tokens: int = typer.Option(..., "--limit", "--balance"),
    unlimited: bool | None = typer.Option(None, "--unlimited/--limited"),
) -> None:
    """Alias for keys limit."""
    _set_key_limit(key, limit_tokens, unlimited)


@keys_app.command("reset-usage")
def reset_usage(key: str) -> None:
    """Reset current-period usage while preserving lifetime audit totals."""
    record = _key_record(key)
    result = _request("POST", f"/admin/keys/{record['id']}/usage/reset")
    typer.echo(f"Usage reset for '{record['nickname']}'.")
    _print(result)


@models_app.command("add")
def add_model(
    nickname: str,
    huggingface_repo: str,
    revision: str | None = typer.Option(None, "--revision"),
    grant_to: list[str] | None = typer.Option(
        None, "--grant-to", help="API-key nickname or full API key."
    ),
) -> None:
    """Register a model and optionally grant it to named API keys."""
    result = _request(
        "POST",
        "/staff/models",
        json_body={
            "nickname": nickname,
            "huggingface_repo": huggingface_repo,
            "revision": revision,
            "grant_to_keys": grant_to or [],
        },
    )
    typer.echo(f"Registration started for '{nickname}'.")
    typer.echo(f"Registration job: {result['job_id']}")
    typer.echo(f"Check progress: ./llmctl models review {nickname}")


@models_app.command("list")
def list_models(json_output: bool = typer.Option(False, "--json")) -> None:
    """List models with registration jobs and next steps for failed registrations."""
    payload = _request("GET", "/staff/models")
    if json_output:
        _print(payload)
        return
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("The server returned an invalid model list.")
    _print_model_records([record for record in data if isinstance(record, dict)])


@models_app.command("review")
@models_app.command("job")
def model_job(
    job_or_model: str,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Explain a registration job; accepts its ID or the model nickname."""
    job = _request("GET", f"/staff/model-jobs/{_job_id_from_selector(job_or_model)}")
    if json_output:
        _print(job)
        return
    _print_model_job(job)


@models_app.command("retry")
def retry_model_job(job_or_model: str) -> None:
    """Retry a failed registration; accepts its ID or the model nickname."""
    job_id = _job_id_from_selector(job_or_model)
    result = _request("POST", f"/staff/model-jobs/{job_id}/retry")
    typer.echo(f"Registration job {result['job_id']} was queued for retry.")


@models_app.command("disable")
def disable_model(nickname: str) -> None:
    """Disable a model selected by nickname."""
    model = _model_record(nickname)
    _print(_request("POST", f"/staff/models/{model['id']}/disable"))


@models_app.command("access")
def model_access(key: str) -> None:
    """List model access for an API-key nickname or full API key."""
    record = _key_record(key)
    models = record.get("granted_models") or []
    typer.echo(f"{record['nickname']}: {', '.join(models) if models else '(none)'}")


@models_app.command("grant")
def grant_models(key: str, model: list[str]) -> None:
    """Add model nicknames without removing the key's existing access."""
    result = _request(
        "POST",
        "/staff/model-access",
        json_body={"key": key, "models": model, "mode": "add"},
    )
    typer.echo(f"Granted {', '.join(model)} to '{result['key']}'.")


@models_app.command("revoke")
def revoke_models(key: str, model: list[str]) -> None:
    """Remove model nicknames without changing the key's other access."""
    result = _request(
        "POST",
        "/staff/model-access",
        json_body={"key": key, "models": model, "mode": "remove"},
    )
    typer.echo(f"Revoked {', '.join(model)} from '{result['key']}'.")


@maintenance_app.command("drain")
def maintenance_drain() -> None:
    _print(_request("POST", "/admin/maintenance", json_body={"mode": "drain"}))


@maintenance_app.command("status")
def maintenance_status() -> None:
    _print(_request("GET", "/admin/maintenance"))


@maintenance_app.command("resume")
def maintenance_resume() -> None:
    _print(_request("POST", "/admin/maintenance", json_body={"mode": "active"}))


if __name__ == "__main__":
    app()

