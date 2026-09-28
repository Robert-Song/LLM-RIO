from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import typer

from llm_rio import admin_client as client_api
from llm_rio.commands.common import (
    _job_id_from_selector,
    _print,
    _print_model_job,
    _print_model_records,
    _print_profile_records,
    _wait_for_model_job,
    models_app,
)
from llm_rio.commands.common import app as app


@models_app.command("add")
def add_model(
    nickname: str,
    huggingface_repo: str | None = typer.Argument(None),
    local_path: Path | None = typer.Option(None, "--local-path"),
    engine: str = typer.Option("vllm", "--engine"),
    revision: str | None = typer.Option(None, "--revision"),
    grant_to: list[str] | None = typer.Option(
        None, "--grant-to", help="API-key nickname or full API key."
    ),
    wait: bool = typer.Option(
        False,
        "--wait/--no-wait",
        help="Follow download and validation; normal-mode validation requires maintenance.",
    ),
    poll_seconds: float = typer.Option(2.0, "--poll-seconds", min=0.1),
    timeout_seconds: float | None = typer.Option(
        None,
        "--timeout-seconds",
        min=1.0,
        help="Stop following after this many seconds; the server-side job continues.",
    ),
) -> None:
    """Register a model; the server downloads and validates it automatically."""
    result = client_api.request(
        "POST",
        "/staff/models",
        json_body={
            "nickname": nickname,
            "huggingface_repo": huggingface_repo,
            "local_path": str(local_path) if local_path else None,
            "engine": engine,
            "revision": revision,
            "grant_to_keys": grant_to or [],
        },
    )
    typer.echo(f"Registration started for '{nickname}'.")
    typer.echo(f"Registration job: {result['job_id']}")
    if wait:
        _wait_for_model_job(
            str(result["job_id"]), poll_seconds=poll_seconds, timeout_seconds=timeout_seconds
        )
    else:
        typer.echo(f"Check progress: ./llmctl models review {nickname}")


@models_app.command("clone-profile")
def clone_model_profile(
    source_nickname: str,
    nickname: str,
    temperature: float | None = typer.Option(None, "--temperature"),
    top_p: float | None = typer.Option(None, "--top-p"),
    top_k: int | None = typer.Option(None, "--top-k"),
    min_p: float | None = typer.Option(None, "--min-p"),
    presence_penalty: float | None = typer.Option(None, "--presence-penalty"),
    repetition_penalty: float | None = typer.Option(None, "--repetition-penalty"),
    reasoning_effort: str | None = typer.Option(None, "--reasoning-effort"),
    max_model_len: int | None = typer.Option(None, "--max-model-len"),
    yarn_factor: float | None = typer.Option(None, "--yarn-factor"),
    yarn_original_max_model_len: int | None = typer.Option(None, "--yarn-original-max-model-len"),
    inherit_grants: bool = typer.Option(True, "--inherit-grants/--no-inherit-grants"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Clone active profiles into a new logical model that shares the source weights."""
    source = client_api.model_record(source_nickname)
    payload: dict[str, Any] = {
        key: value
        for key, value in {
            "nickname": nickname,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "reasoning_effort": reasoning_effort,
            "max_model_len": max_model_len,
            "yarn_factor": yarn_factor,
            "yarn_original_max_model_len": yarn_original_max_model_len,
            "inherit_grants": inherit_grants,
        }.items()
        if value is not None
    }
    result = client_api.request("POST", f"/admin/models/{source['id']}/clone", json_body=payload)
    if json_output:
        _print(result)
        return
    model = result.get("model") if isinstance(result, dict) else None
    profiles = result.get("profiles") if isinstance(result, dict) else None
    if not isinstance(model, dict):
        raise click.ClickException("The server returned an invalid cloned-model record.")
    profile_count = len(profiles) if isinstance(profiles, list) else 0
    typer.echo(
        "Created logical model '"
        f"{model.get('nickname')}"
        "' with "
        f"{profile_count}"
        " cloned placement profile(s)."
    )
    typer.echo(f"Shared artifact: {model.get('artifact_path')}")
    defaults = model.get("request_defaults")
    if isinstance(defaults, dict) and defaults:
        typer.echo(f"Request defaults: {json.dumps(defaults, sort_keys=True)}")


@models_app.command("profiles")
def list_model_profiles(
    nickname: str,
    json_output: bool = typer.Option(False, "--json"),
    saved: bool = typer.Option(
        False, "--saved", help="Include evidence from previous machine fingerprints"
    ),
) -> None:
    """List active and inactive placement profiles for one model (admin only)."""
    payload = client_api.model_profiles(nickname)
    if json_output:
        _print(payload)
        return
    records = [
        record
        for record in payload.get("saved_measurements" if saved else "data", [])
        if isinstance(record, dict)
    ]
    _print_profile_records(records)
    gguf_files = payload.get("available_gguf_files")
    if isinstance(gguf_files, list) and gguf_files:
        typer.echo("Available GGUF files:")
        for filename in gguf_files:
            typer.echo(f"  {filename}")


@models_app.command("profile-edit")
def edit_model_profile(
    nickname: str,
    profile: str,
    engine: str | None = typer.Option(None, "--engine", help="vllm or llama.cpp"),
    tensor_parallel_size: int | None = typer.Option(None, "--tp"),
    max_model_len: int | None = typer.Option(None, "--max-model-len"),
    max_num_seqs: int | None = typer.Option(None, "--max-num-seqs"),
    max_num_batched_tokens: int | None = typer.Option(None, "--max-num-batched-tokens"),
    gpu_memory_utilization: float | None = typer.Option(None, "--gpu-memory-utilization"),
    gguf_file: str | None = typer.Option(None, "--gguf-file"),
    n_gpu_layers: int | None = typer.Option(None, "--n-gpu-layers"),
    make_default: bool = typer.Option(False, "--make-default"),
    restart_workers: bool = typer.Option(False, "--restart-workers"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Override one stored placement profile (admin only)."""
    model = client_api.model_record(nickname)
    payload: dict[str, Any] = {
        key: value
        for key, value in {
            "engine": engine,
            "tensor_parallel_size": tensor_parallel_size,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "gpu_memory_utilization": gpu_memory_utilization,
            "gguf_file": gguf_file,
            "n_gpu_layers": n_gpu_layers,
        }.items()
        if value is not None
    }
    if make_default:
        payload["make_default"] = True
    if restart_workers:
        payload["restart_workers"] = True
    if not payload or set(payload) <= {"make_default", "restart_workers"}:
        raise click.ClickException("Specify at least one profile setting to change.")
    result = client_api.request(
        "PATCH", f"/admin/models/{model['id']}/profiles/{profile}", json_body=payload
    )
    if json_output:
        _print(result)
        return
    updated = result.get("profile") if isinstance(result, dict) else None
    if isinstance(updated, dict):
        _print_profile_records([updated])
    if isinstance(result, dict) and result.get("verification_required"):
        typer.echo(
            "Verification required for: "
            + ", ".join(result["verification_required"])
            + (
                ". Editing launch settings invalidates measurements. "
                "Run models validate with the desired limits to measure a new profile."
            )
        )
    if isinstance(result, dict) and result.get("drained_worker_ids"):
        typer.echo("Draining workers: " + ", ".join(result["drained_worker_ids"]))
    elif isinstance(result, dict) and result.get("restart_required"):
        typer.echo(
            "The profile will be used on the next worker launch; existing workers were not changed."
        )


@models_app.command("list")
def list_models(json_output: bool = typer.Option(False, "--json")) -> None:
    """List models with registration jobs and next steps for failed registrations."""
    payload = client_api.request("GET", "/staff/models")
    if json_output:
        _print(payload)
        return
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("The server returned an invalid model list.")
    _print_model_records([record for record in data if isinstance(record, dict)])


@models_app.command("review")
def model_job(job_or_model: str, json_output: bool = typer.Option(False, "--json")) -> None:
    """Explain a registration job; accepts its ID or the model nickname."""
    job = client_api.request("GET", f"/staff/model-jobs/{_job_id_from_selector(job_or_model)}")
    if json_output:
        _print(job)
        return
    _print_model_job(job)


@models_app.command("validate")
def retry_model_job(
    job_or_model: str,
    profile: str | None = typer.Option(
        None, "--profile", help="Profile ID or position to probe exactly."
    ),
    max_model_len: int | None = typer.Option(
        None, "--max-model-len", min=1, help="Context length to probe during real validation."
    ),
) -> None:
    """Re-run registration; accepts its ID or the model nickname."""
    job_id = _job_id_from_selector(job_or_model)
    if profile is not None:
        job = client_api.request("GET", f"/staff/model-jobs/{job_id}")
        selected = client_api.profile_record(str(job["nickname"]), profile)
        result = client_api.validate_job(
            job_id,
            profile_id=str(selected["id"]),
            overrides={"max_model_len": max_model_len} if max_model_len is not None else None,
        )
    elif max_model_len is None:
        result = client_api.validate_job(job_id)
    else:
        result = client_api.validate_job(job_id, overrides={"max_model_len": max_model_len})
    typer.echo(f"Registration job {result['job_id']} was queued for retry.")


@models_app.command("disable")
def disable_model(nickname: str) -> None:
    """Disable a model selected by nickname."""
    model = client_api.model_record(nickname)
    _print(client_api.request("POST", f"/staff/models/{model['id']}/disable"))


@models_app.command("access")
def model_access(key: str) -> None:
    """List model access for an API-key nickname or full API key."""
    record = client_api.key_record(key)
    models = record.get("granted_models") or []
    typer.echo(f"{record['nickname']}: {(', '.join(models) if models else '(none)')}")


@models_app.command("grant")
def grant_models(key: str, model: list[str]) -> None:
    """Add model nicknames without removing the key's existing access."""
    result = client_api.request(
        "POST", "/staff/model-access", json_body={"key": key, "models": model, "mode": "add"}
    )
    typer.echo(f"Granted {', '.join(model)} to '{result['key']}'.")


@models_app.command("revoke")
def revoke_models(key: str, model: list[str]) -> None:
    """Remove model nicknames without changing the key's other access."""
    result = client_api.request(
        "POST", "/staff/model-access", json_body={"key": key, "models": model, "mode": "remove"}
    )
    typer.echo(f"Revoked {', '.join(model)} from '{result['key']}'.")


@models_app.command("trust-measurements")
def trust_measurements(
    nickname: str, profile_id: str, reason: str = typer.Option(..., "--reason")
) -> None:
    """Advanced: trust one compatible saved profile without running probes."""
    model = client_api.model_record(nickname)
    _print(
        client_api.request(
            "POST",
            f"/admin/models/{model['id']}/profiles/{profile_id}/trust",
            json_body={"reason": reason},
        )
    )


@models_app.command("profile-state")
def profile_state(
    nickname: str, profile: str, enabled: bool = typer.Option(..., "--enable/--disable")
) -> None:
    """Enable or disable one measured placement profile."""
    model = client_api.model_record(nickname)
    selected = client_api.profile_record(nickname, profile)
    action = "enable" if enabled else "disable"
    _print(
        client_api.request(
            "POST", f"/admin/models/{model['id']}/profiles/{selected['id']}/{action}"
        )
    )


@models_app.command("defaults")
def update_defaults(nickname: str, values: str = typer.Option(..., "--values")) -> None:
    """Set request defaults from a JSON object; null clears a setting."""
    from llm_rio.api.schemas import ModelRequestDefaultsUpdate

    try:
        body = ModelRequestDefaultsUpdate.model_validate_json(values).model_dump(exclude_unset=True)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    model = client_api.model_record(nickname)
    _print(client_api.request("PATCH", f"/admin/models/{model['id']}", json_body=body))
