#!/usr/bin/env python3
"""Final OpenAI-compatible smoke test for the non-NVFP4 LLM-RIO catalog.

Run after every expected profile has reached AVAILABLE:

    LLMRIO_API_KEY='...' .venv/bin/python final_deployment_test.py

Use --model to test one or more explicit names, or --all-available only when
you intentionally want to include profiles outside this registration batch.
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


DEFAULT_BASE_URL = "http://127.0.0.1:8003/v1"

# These are the non-gated, non-NVFP4 profiles submitted on 2026-09-09.  Do
# not add the legacy *-nvfp4 profiles here: this is the final experiment gate.
EXPECTED_MODELS = (
    "qwen3.8-27b-fp8",
    #"qwen3.8-27b-int4",
    #"qwen3.6-35b-a3b-fp8",
    "qwen3.6-35b-a3b-int4",
    "qwen3.6-27b-fp8",
    "qwen3.6-27b-int4",
    "qwen3.5-122b-a10b-fp8",
    "qwen3.5-122b-a10b-int4",
    #"qwen3.5-35b-a3b-fp8",
    "qwen3.5-35b-a3b-int4",
    "qwen3.5-27b-fp8",
    "qwen3.5-27b-int4",
    "qwen3.5-9b-int4",
    "qwen3-235b-a22b-int4",
    "qwen3-next-80b-a3b-fp8",
    "qwen3-32b-fp8",
    "qwen3-32b-awq",
    "qwen3-30b-a3b-fp8",
    "qwen3-30b-a3b-int4",
    "qwen3-14b-fp8",
    "qwen3-14b-awq",
    "qwen3-8b-fp8",
    "qwen3-8b-awq",
    "qwen3-4b-fp8",
    "qwen3-4b-awq",
    "qwen3-1.7b-fp8",
    "qwen3-1.7b-int4",
    "laguna-s-2.1-fp8",
    "laguna-s-2.1-int4",
    "gemma-4-31b-it-int4",
    "llama-3.3-70b-instruct-awq",
    "olmo-3-32b-think-8bit",
    "olmo-3-32b-think-4bit",
    "glm-4.5-air-fp8",
    "glm-4.5-air-awq",
)

SYSTEM_PROMPT = "You are a deployment smoke-test assistant. Follow the user exactly."
USER_PROMPT = "Reply with exactly: LLM-RIO deployment test passed."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.getenv("LLMRIO_API_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("LLMRIO_API_KEY"))
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model nickname to test; repeat this option for multiple models.",
    )
    parser.add_argument(
        "--all-available",
        action="store_true",
        help="Test every model returned by /v1/models instead of the expected batch.",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
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

    client = OpenAI(base_url=args.base_url.rstrip("/"), api_key=args.api_key, timeout=args.timeout)
    report: dict[str, Any] = {"base_url": args.base_url, "results": []}

    try:
        available = sorted(model.id for model in client.models.list().data)
    except Exception as exc:  # noqa: BLE001
        report["model_list_error"] = f"{type(exc).__name__}: {exc}"
        write_report(args.report, report)
        print(report["model_list_error"], file=sys.stderr)
        return 2

    if args.model:
        targets = args.model
    elif args.all_available:
        targets = available
    else:
        targets = list(EXPECTED_MODELS)
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
