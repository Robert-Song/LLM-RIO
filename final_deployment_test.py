#!/usr/bin/env python3
"""Test the normal-mode catalog after native revalidation.

Use --continue to check all selected registration jobs, resume the server, and
run inference. Pending or failed registrations block resumption. Without this
flag, the server must already be ACTIVE. Set LLMRIO_API_KEY for inference;
administrative checks require an admin LLMRIO_API_KEY, or use --api-key for
inference while leaving LLMRIO_API_KEY unset to recover local admin credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from fire_all_native_revalidations import llmctl_json

DEFAULT_BASE_URL = "http://127.0.0.1:8003/v1"

SYSTEM_PROMPT = "You are a deployment smoke-test assistant. Follow the user exactly."
USER_PROMPT = "Reply with exactly: LLM-RIO deployment test passed."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LLMRIO_API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("LLMRIO_API_KEY"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model nickname to test; repeat this option for multiple models.",
    )
    selection.add_argument(
        "--all-available",
        action="store_true",
        help="Test only routable models; excludes failed registrations from coverage.",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume from maintenance after checking registration jobs.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--report", type=Path, help="Optional path for a JSON result report.")
    return parser.parse_args()


def write_report(report_path: Path | None, report: dict[str, Any]) -> None:
    if report_path is None:
        return
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Set LLMRIO_API_KEY or pass --api-key.", file=sys.stderr)
        return 2
    if args.max_tokens <= 0:
        print("--max-tokens must be positive.", file=sys.stderr)
        return 2

    root_url = args.base_url.rstrip("/").removesuffix("/v1")
    os.environ["LLMRIO_API_URL"] = root_url
    # The supplied inference key may belong to a user. llmctl independently
    # recovers local admin credentials unless an admin key is supplied via env.
    client = OpenAI(
        base_url=root_url + "/v1", api_key=args.api_key, timeout=args.timeout, max_retries=0
    )
    report: dict[str, Any] = {"base_url": args.base_url, "results": []}

    try:
        status = llmctl_json("maintenance", "status")
        if not status.get("validation", {}).get("requires_maintenance"):
            raise RuntimeError("Deployment test requires normal mode.")
        report["serving_mode"] = status.get("serving_mode", "vllm-sleep")
        print(f"Server mode: {report['serving_mode']}")
        catalog = llmctl_json("models", "list", "--json")["data"]
        enabled = {m["nickname"]: m for m in catalog if m.get("state") != "DISABLED"}
        targets = list(dict.fromkeys(args.model)) if args.model else sorted(enabled)
        if args.all_available:
            targets = sorted(model.id for model in client.models.list().data)
        if not targets:
            raise RuntimeError("No models selected; refusing an empty deployment pass.")
        invalid = []
        for name in targets:
            model = enabled.get(name, {})
            job = model.get("registration_job") or {}
            if model.get("state") != "AVAILABLE" or job.get("state") != "COMPLETED":
                invalid.append(f"{name}: catalog={model.get('state')}, job={job.get('state')}")
        if args.resume:
            invalid.extend(
                f"{name}: registration still pending"
                for name, model in enabled.items()
                if (model.get("registration_job") or {}).get("state") in {"QUEUED", "RUNNING"}
                and name not in targets
            )
        if invalid:
            raise RuntimeError("Registrations are not ready: " + "; ".join(invalid))
        if status.get("validation", {}).get("gpu_uuids"):
            raise RuntimeError("Validation or unverified teardown still owns GPUs.")
        available = sorted(model.id for model in client.models.list().data)
        missing = [name for name in targets if name not in available]
        if missing:
            raise RuntimeError(
                "Selected models are not routable for this key: " + ", ".join(missing)
            )
        if args.resume and status["mode"] != "ACTIVE":
            llmctl_json("maintenance", "resume")
        elif status["mode"] != "ACTIVE":
            raise RuntimeError(
                "Server is in maintenance; use --continue after validation completes."
            )
    except Exception as exc:
        report["preflight_error"] = f"{type(exc).__name__}: {exc}"
        write_report(args.report, report)
        print(report["preflight_error"], file=sys.stderr)
        return 2

    missing = [model for model in targets if model not in available]
    report["available_models"] = available
    report["targets"] = targets
    report["missing"] = missing
    if missing:
        write_report(args.report, report)
        print("NOT READY: expected models are not AVAILABLE:", ", ".join(missing), file=sys.stderr)
        return 1

    for model in targets:
        started = time.monotonic()
        result: dict[str, Any] = {"model": model}
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ],
                max_tokens=args.max_tokens,
                temperature=0,
            )
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("Response contains no assistant text")
            if response.choices[0].finish_reason != "stop":
                raise RuntimeError(f"Incomplete response: {response.choices[0].finish_reason}")
            if not response.usage or response.usage.completion_tokens <= 0:
                raise RuntimeError("Response is missing positive token usage")
            if response.choices[0].message.content.strip() != "LLM-RIO deployment test passed.":
                raise RuntimeError("Response did not follow the smoke-test prompt")
            result.update(
                status="PASS",
                response_model=response.model,
                elapsed_seconds=round(time.monotonic() - started, 3),
                finish_reason=response.choices[0].finish_reason,
                completion_tokens=(response.usage.completion_tokens if response.usage else None),
                text=response.choices[0].message.content,
            )
            print(f"PASS {model} ({result['elapsed_seconds']:.3f}s)")
        except Exception as exc:  # noqa: BLE001
            result.update(
                status="FAIL",
                elapsed_seconds=round(time.monotonic() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
            )
            print(f"FAIL {model}: {result['error']}", file=sys.stderr)
        report["results"].append(result)

    write_report(args.report, report)
    failures = [result for result in report["results"] if result["status"] != "PASS"]
    print(f"Completed {len(targets)} models; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
