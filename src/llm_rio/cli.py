from __future__ import annotations

import copy
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any

import typer
import uvicorn

from llm_rio import admin_client as client_api
from llm_rio.api.app import create_app
from llm_rio.commands import keys as keys
from llm_rio.commands import maintenance as maintenance
from llm_rio.commands import models as models
from llm_rio.commands.common import (
    _print,
)
from llm_rio.commands.common import app as app
from llm_rio.config import ServingMode, Settings
from llm_rio.connection import settings as _settings
from llm_rio.inventory import InventoryError, discover_inventory


@app.command()
def serve(
    config: Path | None = typer.Option(None, "--config", help="TOML configuration file"),
    mode: ServingMode | None = typer.Option(None, "--mode", help="Model residency mode"),
) -> None:
    """Run the machine-local API and scheduler."""
    options: dict[str, Any] = {}
    if config is not None:
        options["config_file"] = config
    if mode is not None:
        options["serving_mode"] = mode
    settings = Settings(**options)
    log_config = copy.deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s | %(levelprefix)s %(message)s"
    uvicorn.run(
        create_app(settings),
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        access_log=False,
        log_config=log_config,
        workers=1,
    )


@app.command("capabilities")
def capabilities() -> None:
    """Show the running mode, engines, and supported administrative actions."""
    _print(client_api.request("GET", "/admin/capabilities"))


@app.command("status")
def status(dashboard: bool = typer.Option(False, "--dashboard")) -> None:
    """Show worker/resource state or live dashboard metrics."""
    _print(client_api.request("GET", "/admin/dashboard" if dashboard else "/admin/status"))


@app.command()
def doctor(
    config: Path | None = typer.Option(None, "--config"),
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


@app.command("summarize")
def summarize_usage(
    through: str | None = typer.Option(None, "--through", help="Timezone-aware ISO cutoff."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Compact settled per-call usage into current-period and lifetime summaries."""
    body = {"through": through} if through is not None else None
    result = client_api.request("POST", "/admin/usage/summarize", json_body=body)
    if json_output:
        _print(result)
        return
    deleted = result.get("deleted", {}) if isinstance(result, dict) else {}
    typer.echo(
        f"Summarized {int(result.get('summarized_requests', 0)):,} settled request(s) "
        f"through {result.get('period_end')}."
    )
    typer.echo(
        f"Deleted {int(deleted.get('inference_requests', 0)):,} request row(s), "
        f"{int(deleted.get('quota_reservations', 0)):,} reservation row(s), and "
        f"{int(deleted.get('quota_ledger', 0)):,} ledger row(s)."
    )
    typer.echo(f"Raw request rows remaining: {int(result.get('raw_requests_remaining', 0)):,}")


def interactive_menu() -> None:
    """Launch the full-screen administration console for LLM-RIO."""
    # Import lazily so scriptable subcommands don't pay the TUI import/startup cost.
    from llm_rio.tui import run_tui

    launch = run_tui()
    if launch is not None:
        serve(config=launch.config, mode=launch.mode)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        interactive_menu()


@app.command("interactive")
def interactive_cmd() -> None:
    """Launch the full-screen terminal administration interface."""
    interactive_menu()


if __name__ == "__main__":
    app()
