from __future__ import annotations

import copy
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import httpx
import typer
import uvicorn

from llm_rio.api.app import create_app
from llm_rio.config import Settings
from llm_rio.domain import Role
from llm_rio.inventory import InventoryError, discover_inventory

app = typer.Typer(help="Admin client and local lifecycle commands for LLM-RIO.")
keys_app = typer.Typer(help="Manage API keys.")
models_app = typer.Typer(help="Manage the model catalog.")
maintenance_app = typer.Typer(help="Drain or resume this machine.")
app.add_typer(keys_app, name="keys")
app.add_typer(models_app, name="models")
app.add_typer(maintenance_app, name="maintenance")


def _settings(config: Path) -> Settings:
    return Settings(config_file=config)


def _base_url() -> str:
    return os.environ.get("LLMRIO_API_URL", "http://127.0.0.1:8000").rstrip("/")


def _api_key() -> str:
    value = os.environ.get("LLMRIO_API_KEY")
    if not value:
        raise typer.BadParameter("Set LLMRIO_API_KEY for authenticated management commands")
    return value


def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> Any:
    headers = {"Authorization": f"Bearer {_api_key()}"} if authenticated else {}
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
def list_keys() -> None:
    _print(_request("GET", "/admin/keys"))


@keys_app.command("create")
def create_key(
    nickname: str,
    role: Role = typer.Option(Role.USER, "--role"),
    balance_tokens: int = typer.Option(1_000_000, "--balance"),
    unlimited: bool = typer.Option(False, "--unlimited"),
    account_id: str | None = typer.Option(None, "--account-id"),
    grant: list[str] | None = typer.Option(None, "--grant"),
) -> None:
    _print(
        _request(
            "POST",
            "/admin/keys",
            json_body={
                "nickname": nickname,
                "role": role.value,
                "balance_tokens": balance_tokens,
                "unlimited": unlimited,
                "quota_account_id": account_id,
                "model_ids": grant or [],
            },
        )
    )


@keys_app.command("rotate")
def rotate_key(key_id: str) -> None:
    _print(_request("POST", f"/admin/keys/{key_id}/rotate"))


@keys_app.command("revoke")
def revoke_key(key_id: str) -> None:
    _request("POST", f"/admin/keys/{key_id}/revoke")
    typer.echo("Key revoked.")


@keys_app.command("delete")
def delete_key(key_id: str) -> None:
    _request("DELETE", f"/admin/keys/{key_id}")
    typer.echo("Key deleted.")


@keys_app.command("quota")
def update_quota(
    key_id: str,
    balance_tokens: int = typer.Option(..., "--balance"),
    unlimited: bool = typer.Option(False, "--unlimited"),
) -> None:
    _request(
        "PUT",
        f"/admin/keys/{key_id}/quota",
        json_body={"balance_tokens": balance_tokens, "unlimited": unlimited},
    )
    typer.echo("Quota updated.")


@models_app.command("add")
def add_model(
    nickname: str,
    huggingface_repo: str,
    revision: str | None = typer.Option(None, "--revision"),
    grant_to: list[str] | None = typer.Option(None, "--grant-to"),
) -> None:
    _print(
        _request(
            "POST",
            "/staff/models",
            json_body={
                "nickname": nickname,
                "huggingface_repo": huggingface_repo,
                "revision": revision,
                "grant_to_key_ids": grant_to or [],
            },
        )
    )


@models_app.command("list")
def list_models() -> None:
    _print(_request("GET", "/staff/models"))


@models_app.command("job")
def model_job(job_id: str) -> None:
    _print(_request("GET", f"/staff/model-jobs/{job_id}"))


@models_app.command("disable")
def disable_model(model_id: str) -> None:
    _print(_request("POST", f"/staff/models/{model_id}/disable"))


@models_app.command("grant")
def grant_models(key_id: str, model_id: list[str]) -> None:
    _request(
        "PUT", f"/staff/keys/{key_id}/model-grants", json_body={"model_ids": model_id}
    )
    typer.echo("Model grants replaced.")


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

