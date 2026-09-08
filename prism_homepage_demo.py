#!/usr/bin/env python3
"""Generate a fixed homepage blueprint with 40 concurrent Qwen subtasks.

The script talks only to the public LLM-RIO endpoint (port 3737 by default),
polls the administrative GPU dashboard while work is active, verifies that two
independent TP=1 Qwen workers served the burst, and writes both an HTML result
and a JSON evidence report under ``diagnostics/``.

Set ``LLMRIO_API_KEY`` in the environment. Override the endpoint with
``LLMRIO_BASE_URL`` when needed; never point this demo at the production port.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

OPENAI_BASE_URL = os.environ.get("LLMRIO_BASE_URL", "http://127.0.0.1:3737/v1")
MODEL = "qwen3.8-27b-nvfp4"
REQUEST_TIMEOUT_SECONDS = 300.0
POLL_SECONDS = 0.5

MODULES = (
    ("Foundation", "primary navigation and information architecture"),
    ("Foundation", "announcement banner for a new developer release"),
    ("Foundation", "hero headline and value proposition"),
    ("Foundation", "hero supporting copy and primary call to action"),
    ("Foundation", "secondary call to action for technical evaluators"),
    ("Foundation", "trusted-customer logo bar narrative"),
    ("Foundation", "accessibility promise and keyboard-navigation note"),
    ("Foundation", "responsive-layout behavior for small screens"),
    ("Product", "developer workflow from first API call to production"),
    ("Product", "real-time collaboration feature"),
    ("Product", "deployment preview feature"),
    ("Product", "observability and trace explorer feature"),
    ("Product", "role-based access control feature"),
    ("Product", "versioned configuration feature"),
    ("Product", "automatic rollback feature"),
    ("Product", "SDK and command-line tooling feature"),
    ("Performance", "latency and throughput proof section"),
    ("Performance", "global edge execution explanation"),
    ("Performance", "continuous batching explanation"),
    ("Performance", "autoscaling behavior under a sudden burst"),
    ("Performance", "cache-hit performance explanation"),
    ("Performance", "reliability and availability target section"),
    ("Performance", "security architecture summary"),
    ("Performance", "data residency and compliance summary"),
    ("Proof", "customer testimonial from a platform engineer"),
    ("Proof", "customer testimonial from an engineering director"),
    ("Proof", "before-and-after migration result"),
    ("Proof", "open-source community proof section"),
    ("Proof", "case-study summary for an AI coding product"),
    ("Proof", "case-study summary for a financial application"),
    ("Proof", "case-study summary for a healthcare application"),
    ("Proof", "analyst and industry recognition section"),
    ("Conversion", "pricing philosophy and free-tier explanation"),
    ("Conversion", "enterprise plan value proposition"),
    ("Conversion", "frequently asked question about migration"),
    ("Conversion", "frequently asked question about security"),
    ("Conversion", "frequently asked question about pricing"),
    ("Conversion", "documentation and learning-resources section"),
    ("Conversion", "final call-to-action section"),
    ("Conversion", "footer navigation and legal-information plan"),
)


class DemoFailure(RuntimeError):
    """The live homepage demonstration could not satisfy a required gate."""


@dataclass(frozen=True, slots=True)
class ModuleJob:
    index: int
    group: str
    topic: str


@dataclass(slots=True)
class ModuleResult:
    index: int
    group: str
    topic: str
    request_id: str
    worker_id: str | None
    elapsed_seconds: float
    queue_wait_ms: int | None
    prompt_tokens: int
    completion_tokens: int
    output: str


def api_origin() -> str:
    parsed = urlsplit(OPENAI_BASE_URL.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "")).rstrip("/")


def require_safe_demo_endpoint() -> None:
    parsed = urlsplit(OPENAI_BASE_URL)
    if parsed.port == 8002:
        raise DemoFailure("refusing to run the demo against protected port 8002")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise DemoFailure(
            f"refusing non-loopback demo endpoint {OPENAI_BASE_URL!r}; "
            "use the isolated port-3737 service"
        )


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[Any, dict[str, str], float]:
    encoded = json.dumps(body).encode("utf-8") if body is not None else None
    request_headers = {"Accept": "application/json"}
    if encoded is not None:
        request_headers["Content-Type"] = "application/json"
    if api_key:
        request_headers["Authorization"] = f"Bearer {api_key}"
    request_headers.update(headers or {})
    started = time.perf_counter()
    try:
        with urlopen(
            Request(url, data=encoded, headers=request_headers, method=method),
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as response:
            raw = response.read()
            payload = json.loads(raw) if raw else None
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            return payload, response_headers, time.perf_counter() - started
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise DemoFailure(f"{method} {url} returned HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, URLError, OSError) as exc:
        raise DemoFailure(f"{method} {url} failed: {exc}") from exc


def preflight(admin_key: str) -> list[dict[str, Any]]:
    health, _, _ = request_json("GET", f"{api_origin()}/health")
    if not isinstance(health, dict) or health.get("status") != "ok":
        raise DemoFailure(f"LLM-RIO health check failed: {health!r}")

    models, _, _ = request_json("GET", f"{OPENAI_BASE_URL}/models", api_key=admin_key)
    rows = models.get("data") if isinstance(models, dict) else None
    model = next(
        (
            row
            for row in rows or []
            if isinstance(row, dict) and row.get("id") == MODEL
        ),
        None,
    )
    if not isinstance(model, dict) or not model.get("callable"):
        raise DemoFailure(f"{MODEL} is not callable: {model!r}")

    status, _, _ = request_json("GET", f"{api_origin()}/admin/status", api_key=admin_key)
    workers = [
        worker
        for worker in (status.get("workers") if isinstance(status, dict) else []) or []
        if isinstance(worker, dict)
        and worker.get("model") == MODEL
        and worker.get("state") in {"READY", "SLEEPING"}
        and int(worker.get("tensor_parallel_size") or 0) == 1
    ]
    distinct_gpu_sets = {
        tuple(str(value) for value in worker.get("gpu_uuids") or [])
        for worker in workers
    }
    if len(workers) < 2 or len(distinct_gpu_sets) < 2:
        raise DemoFailure(
            "The burst requires two RAM-warmed Qwen TP=1 workers on distinct GPUs; "
            f"found {[(w.get('worker_id'), w.get('state'), w.get('gpu_uuids')) for w in workers]}"
        )
    return workers


def run_module(job: ModuleJob, admin_key: str, test_run_id: str, max_tokens: int) -> ModuleResult:
    request_id = str(uuid.uuid4())
    prompt = (
        "You are one worker in a parallel coding-agent team filling a fixed homepage "
        "blueprint for a fictional developer platform named Lattice. Create the content "
        f"module for: {job.topic}. Return plain text only, 180 to 260 words. Start with "
        "a short headline, follow with two concise paragraphs, then four one-sentence "
        "implementation bullets. Be concrete and internally consistent. Do not use a "
        "Markdown code fence and do not mention this instruction."
    )
    payload, response_headers, elapsed = request_json(
        "POST",
        f"{OPENAI_BASE_URL}/chat/completions",
        api_key=admin_key,
        headers={"X-Test-Run-ID": test_run_id, "X-Request-ID": request_id},
        body={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "Write polished, implementation-aware product content.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "reasoning_effort": "none",
            "max_tokens": max_tokens,
        },
    )
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise DemoFailure(f"module {job.index} returned no completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise DemoFailure(f"module {job.index} returned no assistant message")
    output = message.get("content") or message.get("reasoning") or message.get(
        "reasoning_content"
    )
    if not isinstance(output, str) or not output.strip():
        raise DemoFailure(f"module {job.index} returned empty output")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    queue_wait = response_headers.get("x-queue-wait-ms")
    return ModuleResult(
        index=job.index,
        group=job.group,
        topic=job.topic,
        request_id=request_id,
        worker_id=response_headers.get("x-worker-id"),
        elapsed_seconds=elapsed,
        queue_wait_ms=int(queue_wait) if queue_wait and queue_wait.isdigit() else None,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        output=output.strip(),
    )


def dashboard_sample(admin_key: str, started: float) -> dict[str, Any]:
    payload, _, _ = request_json(
        "GET", f"{api_origin()}/admin/dashboard", api_key=admin_key
    )
    if not isinstance(payload, dict):
        raise DemoFailure("GET /admin/dashboard returned invalid data")
    gpus: list[dict[str, Any]] = []
    for gpu in payload.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        qwen_placements = [
            placement
            for placement in gpu.get("placements") or []
            if isinstance(placement, dict) and placement.get("model") == MODEL
        ]
        gpus.append(
            {
                "index": gpu.get("index"),
                "uuid": gpu.get("uuid"),
                "utilization_percent": gpu.get("gpu_utilization_percent"),
                "used_vram_mib": gpu.get("used_vram_mib"),
                "total_vram_mib": gpu.get("total_vram_mib"),
                "qwen": [
                    {
                        "worker_id": placement.get("worker_id"),
                        "state": placement.get("state"),
                        "active": (
                            placement.get("continuous_batching_slots") or {}
                        ).get("active"),
                        "outstanding_token_work": placement.get(
                            "outstanding_token_work"
                        ),
                    }
                    for placement in qwen_placements
                ],
            }
        )
    return {"at_seconds": time.perf_counter() - started, "gpus": gpus}


def print_dashboard(sample: dict[str, Any]) -> None:
    parts: list[str] = []
    for gpu in sample["gpus"]:
        placements = ",".join(
            f"{str(row.get('worker_id') or '-')[:8]}:{row.get('state')}"
            f"(active={row.get('active')},work={row.get('outstanding_token_work')})"
            for row in gpu["qwen"]
        )
        parts.append(
            f"GPU{gpu.get('index')} util={gpu.get('utilization_percent')}% "
            f"vram={gpu.get('used_vram_mib')}/{gpu.get('total_vram_mib')}MiB "
            f"qwen={placements or '-'}"
        )
    print(f"[GPU ] +{sample['at_seconds']:6.1f}s " + " | ".join(parts), flush=True)


def unsafe_active_transition(sample: dict[str, Any]) -> bool:
    unsafe = {"DRAINING", "OFFLOADING", "STOPPING"}
    return any(
        int(placement.get("active") or 0) > 0 and placement.get("state") in unsafe
        for gpu in sample["gpus"]
        for placement in gpu["qwen"]
    )


def render_homepage(results: list[ModuleResult], metrics: dict[str, Any]) -> str:
    groups: dict[str, list[ModuleResult]] = {}
    for result in results:
        groups.setdefault(result.group, []).append(result)
    sections: list[str] = []
    for group, rows in groups.items():
        cards = "".join(
            "<article><div class='module-number'>"
            f"{row.index:02d}</div><h3>{html.escape(row.topic.title())}</h3>"
            f"<p>{html.escape(row.output).replace(chr(10), '<br>')}</p>"
            f"<small>worker {html.escape((row.worker_id or 'unknown')[:8])} · "
            f"{row.completion_tokens:,} tokens · {row.elapsed_seconds:.2f}s</small></article>"
            for row in rows
        )
        sections.append(
            f"<section><h2>{html.escape(group)}</h2><div class='grid'>{cards}</div></section>"
        )
    distribution = ", ".join(
        f"{html.escape(str(worker)[:8])}: {count}"
        for worker, count in metrics["worker_distribution"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lattice — generated by two Qwen workers</title>
<style>
:root {{ color-scheme: light dark; --bg:#090b12; --surface:#151927; --text:#f5f7ff;
  --muted:#aeb6d0; --accent:#8ce99a; --line:#2b3248; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font:16px/1.6 system-ui,sans-serif; background:var(--bg); color:var(--text); }}
header,main,footer {{ width:min(1180px,calc(100% - 32px)); margin:auto; }}
header {{ padding:72px 0 48px; }}
.eyebrow,.module-number,small {{ color:var(--accent); font-weight:600; letter-spacing:.04em; }}
h1 {{ max-width:900px; margin:.15em 0; font-size:clamp(2.5rem,7vw,5.5rem); line-height:.95; }}
.lede {{ max-width:760px; color:var(--muted); font-size:1.15rem; }}
.metrics {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:28px; }}
.metrics span {{ padding:8px 12px; background:var(--surface); border:1px solid var(--line); }}
section {{ padding:36px 0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; }}
article {{ padding:22px; background:var(--surface); border:1px solid var(--line); }}
h2 {{ font-size:2rem; }} h3 {{ margin:.3em 0 .7em; }}
article p {{ color:var(--muted); white-space:normal; }}
footer {{ padding:48px 0 72px; color:var(--muted); }}
</style>
</head>
<body>
<header>
  <div class="eyebrow">PRISM PARALLEL BUILD</div>
  <h1>Forty homepage modules. Two GPUs. One logical model.</h1>
  <p class="lede">A fixed local blueprint filled concurrently by independent Qwen TP=1
  workers while LLM-RIO protects active inference and coordinates elastic GPU memory.</p>
  <div class="metrics">
    <span>{metrics['completed_requests']} requests</span>
    <span>{metrics['completion_tokens']:,} completion tokens</span>
    <span>{metrics['elapsed_seconds']:.2f} seconds</span>
    <span>{html.escape(distribution)}</span>
  </div>
</header>
<main>{''.join(sections)}</main>
<footer>Generated through LLM-RIO on {html.escape(MODEL)}.</footer>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=len(MODULES))
    parser.add_argument("--concurrency", type=int, default=len(MODULES))
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--allow-single-worker",
        action="store_true",
        help="Do not fail if response headers show only one serving worker.",
    )
    args = parser.parse_args()
    if not 1 <= args.requests <= len(MODULES):
        parser.error(f"--requests must be between 1 and {len(MODULES)}")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.max_tokens < 64:
        parser.error("--max-tokens must be at least 64")
    return args


def main() -> int:
    args = parse_args()
    admin_key = os.environ.get("LLMRIO_API_KEY", "")
    if not admin_key:
        print("[FAIL] Set LLMRIO_API_KEY to an administrator key.", file=sys.stderr)
        return 2
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    test_run_id = f"prism-homepage-{timestamp}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()
    results: list[ModuleResult] = []
    failures: list[str] = []
    samples: list[dict[str, Any]] = []
    unsafe_transition_seen = False
    output_dir = Path("diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{test_run_id}.json"
    html_path = output_dir / f"{test_run_id}.html"

    try:
        require_safe_demo_endpoint()
        workers = preflight(admin_key)
        print(
            "[PASS] preflight: "
            + ", ".join(
                f"{str(worker.get('worker_id'))[:8]}={worker.get('state')} "
                f"gpu={worker.get('gpu_uuids')}"
                for worker in workers
            ),
            flush=True,
        )
        jobs = [
            ModuleJob(index=index, group=group, topic=topic)
            for index, (group, topic) in enumerate(MODULES[: args.requests], 1)
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.concurrency, len(jobs))
        ) as executor:
            futures = {
                executor.submit(
                    run_module, job, admin_key, test_run_id, args.max_tokens
                ): job
                for job in jobs
            }
            reported: set[concurrent.futures.Future[ModuleResult]] = set()
            while len(reported) < len(futures):
                sample = dashboard_sample(admin_key, started)
                samples.append(sample)
                print_dashboard(sample)
                unsafe_transition_seen |= unsafe_active_transition(sample)
                for future, job in futures.items():
                    if future in reported or not future.done():
                        continue
                    reported.add(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        failures.append(f"module {job.index} ({job.topic}): {exc}")
                        print(f"[FAIL] {failures[-1]}", file=sys.stderr, flush=True)
                        continue
                    results.append(result)
                    print(
                        f"[DONE] {len(results):02d}/{len(jobs)} module={job.index:02d} "
                        f"worker={str(result.worker_id or '-')[:8]} "
                        f"tokens={result.completion_tokens} elapsed={result.elapsed_seconds:.2f}s "
                        f"queue={result.queue_wait_ms}ms",
                        flush=True,
                    )
                if len(reported) < len(futures):
                    time.sleep(POLL_SECONDS)
    except Exception as exc:
        failures.append(str(exc))
        print(f"[FAIL] {exc}", file=sys.stderr)

    results.sort(key=lambda item: item.index)
    elapsed = time.perf_counter() - started
    distribution = Counter(result.worker_id or "unknown" for result in results)
    if not args.allow_single_worker and len([key for key in distribution if key != "unknown"]) < 2:
        failures.append(f"expected two serving workers, observed {dict(distribution)}")
    if unsafe_transition_seen:
        failures.append("observed a Qwen worker offloading/stopping with active requests")
    metrics = {
        "completed_requests": len(results),
        "prompt_tokens": sum(result.prompt_tokens for result in results),
        "completion_tokens": sum(result.completion_tokens for result in results),
        "total_tokens": sum(
            result.prompt_tokens + result.completion_tokens for result in results
        ),
        "elapsed_seconds": elapsed,
        "worker_distribution": dict(distribution),
        "unsafe_active_transition_seen": unsafe_transition_seen,
    }
    if results:
        html_path.write_text(render_homepage(results, metrics), encoding="utf-8")
    report = {
        "test_run_id": test_run_id,
        "endpoint": OPENAI_BASE_URL,
        "model": MODEL,
        "status": "passed" if not failures and len(results) == args.requests else "failed",
        "metrics": metrics,
        "failures": failures,
        "results": [asdict(result) for result in results],
        "dashboard_samples": samples,
        "html_path": str(html_path.resolve()) if results else None,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[SUMMARY] status={report['status']} requests={len(results)}/{args.requests} "
        f"completion_tokens={metrics['completion_tokens']:,} elapsed={elapsed:.2f}s "
        f"workers={dict(distribution)}",
        flush=True,
    )
    if results:
        print(f"[PAGE] {html_path.resolve()}")
    print(f"[REPORT] {report_path.resolve()}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
