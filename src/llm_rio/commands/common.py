from __future__ import annotations

import json
import time
from typing import Any

import click
import typer

from llm_rio import admin_client as client_api

app = typer.Typer(help="Local administration and lifecycle commands for LLM-RIO.")
keys_app = typer.Typer(help="Manage API keys.")
models_app = typer.Typer(help="Manage the model catalog.")
maintenance_app = typer.Typer(help="Drain or resume this machine.")
app.add_typer(keys_app, name="keys")
app.add_typer(models_app, name="models")
app.add_typer(maintenance_app, name="maintenance")


def _print(value: Any) -> None:
    typer.echo(json.dumps(value, indent=2, sort_keys=True))


def _print_profile_records(records: list[dict[str, Any]]) -> None:
    if not records:
        typer.echo("No placement profiles are recorded for this model on this machine.")
        return
    typer.echo("Placement profiles\n")
    for index, profile in enumerate(records, 1):
        active = "active/default" if profile.get("active") else "inactive"
        typer.echo(f"{index}. {profile.get('id')} ({active})")
        typer.echo(
            "   Engine: "
            f"{profile.get('engine')}"
            " | GPUs / TP: "
            f"{profile.get('gpu_count')}"
            " / "
            f"{profile.get('tensor_parallel_size')}"
        )
        typer.echo(
            f"   Context: {int(profile.get('max_model_len') or 0):,}"
            f" | Max sequences: {profile.get('max_num_seqs') or 'engine default'}"
            f" | Max batched tokens: {profile.get('max_num_batched_tokens') or 'engine default'}"
        )
        typer.echo(f"   GPU memory utilization: {profile.get('gpu_memory_utilization')}")
        launch_args = profile.get("launch_args")
        if isinstance(launch_args, dict) and launch_args:
            typer.echo(f"   Launch overrides: {json.dumps(launch_args, sort_keys=True)}")
        typer.echo()


def _job_id_for_model(nickname: str) -> str:
    model = client_api.model_record(nickname)
    job = model.get("registration_job")
    if not isinstance(job, dict) or not isinstance(job.get("id"), str):
        raise click.ClickException(f"Model '{nickname}' has no registration job.")
    return str(job["id"])


def _job_id_from_selector(job_or_model: str) -> str:
    """Accept a job UUID or a model nickname so review does not require internal IDs."""
    try:
        return _job_id_for_model(job_or_model)
    except click.ClickException:
        return job_or_model


def _review_guidance(stage: object) -> str:
    guidance = {
        "resolve": "Verify the Hugging Face repository, requested revision, and HF token access.",
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
    if job.get("stage") == "waiting_for_maintenance":
        typer.echo("Next step: run llmctl maintenance drain, or use Drain in the TUI.")
    if job.get("stage") == "gpu_memory_wait":
        typer.echo("Waiting for free GPU memory; validation will retry automatically.")
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
    typer.echo(f"  2. Retry: ./llmctl models validate {job.get('nickname')}")
    typer.echo(
        "  3. If this model will not be fixed, disable it: ./llmctl models disable "
        f"{job.get('nickname')}"
    )


def _wait_for_model_job(
    job_id: str, *, poll_seconds: float, timeout_seconds: float | None
) -> dict[str, Any]:
    """Poll a registration job while printing each newly observed stage."""
    started = time.monotonic()
    previous: tuple[object, object, object] | None = None
    while True:
        job = client_api.request("GET", f"/staff/model-jobs/{job_id}")
        if not isinstance(job, dict):
            raise click.ClickException("The server returned an invalid registration job.")
        state = job.get("state")
        marker = (state, job.get("catalog_state"), job.get("stage"))
        if marker != previous:
            elapsed = time.monotonic() - started
            typer.echo(f"[{elapsed:7.1f}s] {marker[0]} / {marker[1]} / {marker[2]}")
            previous = marker
        if state == "COMPLETED":
            typer.echo(f"Model '{job.get('nickname')}' is validated and available.")
            return job
        if state == "FAILED":
            _print_model_job(job)
            raise click.ClickException(
                f"Registration job {job_id} failed at stage {job.get('stage')}."
            )
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise click.ClickException(
                f"Timed out waiting for registration job {job_id}; it is still running."
            )
        time.sleep(poll_seconds)


def _print_model_records(records: list[dict[str, Any]]) -> None:
    if not records:
        typer.echo("No models found.")
        return
    typer.echo("Models\n")
    for model in records:
        typer.echo(str(model.get("nickname")))
        typer.echo(f"   State: {model.get('state')}")
        typer.echo(f"   Repository: {model.get('huggingface_repo')}")
        if model.get("source_model_id"):
            typer.echo(f"   Shared weights from model ID: {model.get('source_model_id')}")
        defaults = model.get("request_defaults")
        if isinstance(defaults, dict) and defaults:
            typer.echo(f"   Request defaults: {json.dumps(defaults, sort_keys=True)}")
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
    heading = (
        f"{number}. {record.get('nickname')}" if number is not None else str(record.get("nickname"))
    )
    typer.echo(heading)
    typer.echo(f"   API key: {record.get('api_key')}")
    typer.echo(f"   Role: {record.get('role')}")
    typer.echo(f"   Quota account: {record.get('account_nickname')}")
    typer.echo(f"   Active: {('yes' if record.get('active') else 'no')}")
    if record.get("unlimited"):
        typer.echo("   Token limit: unlimited")
    else:
        typer.echo(f"   Token limit: {int(record.get('limit_tokens') or 0):,}")
        typer.echo(f"   Used since reset: {int(record.get('used_tokens') or 0):,}")
        typer.echo(f"   Remaining: {int(record.get('balance_tokens') or 0):,}")
    typer.echo(f"   Lifetime charged: {int(record.get('key_lifetime_charged_tokens') or 0):,}")
    granted_models = record.get("granted_models") or []
    typer.echo(f"   Models: {(', '.join(granted_models) if granted_models else '(none)')}")
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


def _set_key_limit(key: str, limit_tokens: int, unlimited: bool | None) -> None:
    record = client_api.key_record(key)
    resolved_unlimited = bool(record.get("unlimited")) if unlimited is None else unlimited
    client_api.request(
        "PUT",
        f"/admin/keys/{record['id']}/quota",
        json_body={"limit_tokens": limit_tokens, "unlimited": resolved_unlimited},
    )
    typer.echo(f"Token limit updated for '{record['nickname']}'.")
