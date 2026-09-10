#!/usr/bin/env python3
"""Drain the normal-mode server and requeue all enabled catalog registrations.

Run with the server's Python environment after restarting the updated server.
This submits work; maintenance remains enabled until validation completes and
final_deployment_test.py --continue explicitly resumes serving.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
LLMCTL = PROJECT_DIR / "llmctl"
CURRENT_VRAM_MEASUREMENT_VERSION = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only-invalid",
        action="store_true",
        help="Skip models with an existing valid native measurement.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the models that would be requeued without changing any jobs.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write the submitted, skipped, and failed records as JSON.",
    )
    return parser.parse_args()


def llmctl_json(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        [str(LLMCTL), *args],
        cwd=PROJECT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"llmctl {' '.join(args)} failed: {message}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llmctl {' '.join(args)} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"llmctl {' '.join(args)} returned a non-object JSON response")
    return payload


def is_valid_native_v2(profile: dict[str, Any], *, queue_mode: bool = False) -> bool:
    """Mirror the native portion of the routability invariant without kvcached."""
    if not profile.get("active"):
        return False
    if profile.get("memory_backend", "native") != "native":
        return False
    if not profile.get("normal_verified"):
        return False
    if profile.get("vram_measurement_version") != CURRENT_VRAM_MEASUREMENT_VERSION:
        return False

    gpu_count = profile.get("gpu_count")
    if not isinstance(gpu_count, int) or gpu_count < 1:
        return False
    vector_names = (
        "idle_vram_mib_per_gpu",
        "peak_vram_mib_per_gpu",
        "gpu_headroom_mib_per_gpu",
        "vram_baseline_mib_per_gpu",
    )
    vectors: list[list[Any]] = []
    for name in vector_names:
        vector = profile.get(name)
        if not isinstance(vector, list) or len(vector) != gpu_count:
            return False
        if any(not isinstance(value, int) or value < 0 for value in vector):
            return False
        vectors.append(vector)
    if any(vectors[2]):  # v2 profiles never retain legacy per-profile headroom.
        return False

    if queue_mode:
        return (
            profile.get("engine") == "vllm"
            and profile.get("launch_args", {}).get("enable_sleep_mode") is False
        )

    # Sleep mode additionally requires measured residual and wake peak.
    if profile.get("engine") != "vllm":
        return False
    for name in ("sleep_vram_mib_per_gpu", "wake_peak_vram_mib_per_gpu"):
        vector = profile.get(name)
        if (
            not isinstance(vector, list)
            or len(vector) != gpu_count
            or any(not isinstance(value, int) or value < 0 for value in vector)
        ):
            return False
    return True


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report written to {path}")


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "mode": "dry-run" if args.dry_run else "submit",
        "submitted": [],
        "already_valid": [],
        "skipped": [],
        "failed": [],
    }
    try:
        models_payload = llmctl_json("models", "list", "--json")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    models = models_payload.get("data")
    if not isinstance(models, list):
        print("llmctl models list returned no model data", file=sys.stderr)
        return 2

    try:
        status = llmctl_json("maintenance", "status")
        if not status.get("validation", {}).get("requires_maintenance"):
            raise RuntimeError("This script requires normal mode; Prism is enabled.")
        report["serving_mode"] = status.get("serving_mode", "vllm-sleep")
        print(f"Server mode: {report['serving_mode']}")
        if not args.dry_run:
            llmctl_json("maintenance", "drain")
            print("Maintenance requested; validation waits for draining to finish.")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    for model in models:
        if not isinstance(model, dict) or model.get("state") == "DISABLED":
            continue
        nickname = model.get("nickname")
        if not isinstance(nickname, str):
            continue
        if args.only_invalid:
            try:
                profiles_payload = llmctl_json("models", "profiles", nickname, "--json")
            except RuntimeError as exc:
                report["failed"].append({"model": nickname, "reason": str(exc)})
                continue
            profiles = profiles_payload.get("data")
            if isinstance(profiles, list) and any(
                isinstance(profile, dict)
                and is_valid_native_v2(profile, queue_mode=report["serving_mode"] == "queue")
                for profile in profiles
            ):
                report["already_valid"].append(nickname)
                continue

        job = model.get("registration_job")
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            report["skipped"].append(
                {"model": nickname, "reason": "no registration job available to retry"}
            )
            continue
        if job.get("state") in {"QUEUED", "RUNNING"}:
            report["skipped"].append(
                {"model": nickname, "reason": f"job already {job['state'].lower()}"}
            )
            continue

        record = {"model": nickname, "job_id": job["id"]}
        if args.dry_run:
            report["submitted"].append(record)
            print(f"WOULD REQUEUE {nickname} ({job['id']})")
            continue
        completed = subprocess.run(
            [str(LLMCTL), "models", "retry", job["id"]],
            cwd=PROJECT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            reason = completed.stderr.strip() or completed.stdout.strip()
            report["failed"].append({**record, "reason": reason})
            print(f"FAILED {nickname}: {reason}", file=sys.stderr)
            continue
        report["submitted"].append(record)
        print(f"REQUEUED {nickname} ({job['id']})")

    write_report(args.report, report)
    print(
        "Summary: "
        f"requeued={len(report['submitted'])} "
        f"already-valid={len(report['already_valid'])} "
        f"skipped={len(report['skipped'])} "
        f"failed={len(report['failed'])}"
    )
    if not args.dry_run:
        print("After jobs finish, run: .venv/bin/python final_deployment_test.py --continue")
    return (
        1
        if report["failed"]
        or any(
            item["reason"] == "no registration job available to retry" for item in report["skipped"]
        )
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
