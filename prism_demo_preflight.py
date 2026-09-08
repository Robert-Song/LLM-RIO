#!/usr/bin/env python3
"""Fail-fast checks for the isolated single-machine Prism demonstration."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEMO_PORT = 3737
PROTECTED_PORT = 8002
MIN_AVAILABLE_RAM_GIB = 220.0
TARGET_NICKNAME = "qwen3-8b-q8"
TARGET_REPOSITORY = "Qwen/Qwen3-8B-FP8"
REQUIRED_PRELOADS = Counter(
    {
        "qwen3.8-27b-nvfp4": 2,
        "gemma-4-31b-it-nvfp4": 1,
        "laguna-s-2.1-nvfp4": 1,
    }
)


@dataclass(slots=True)
class Checks:
    failures: list[str] = field(default_factory=list)

    def passed(self, message: str) -> None:
        print(f"[PASS] {message}")

    def failed(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")

    def warning(self, message: str) -> None:
        print(f"[WARN] {message}")

    def information(self, message: str) -> None:
        print(f"[INFO] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/tmp/llm-rio-prism-rehearsal/config.toml"),
    )
    parser.add_argument(
        "--phase",
        choices=("before-start", "ready"),
        default="before-start",
        help="Use ready after all preload workers have entered SLEEPING.",
    )
    return parser.parse_args()


def resolve_path(config_path: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def gpu_rows() -> list[tuple[str, float, float]]:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[str, float, float]] = []
    for line in process.stdout.splitlines():
        uuid, memory, utilization = (part.strip() for part in line.split(","))
        rows.append((uuid, float(memory), float(utilization)))
    return rows


def get_json(url: str, api_key: str | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with urlopen(Request(url, headers=headers), timeout=10) as response:
            payload = json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"{url} returned HTTP {exc.code}") from exc
    except (OSError, TimeoutError, URLError) as exc:
        raise RuntimeError(f"{url} is unavailable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return payload


def check_repository(checks: Checks) -> None:
    root = Path(__file__).resolve().parent
    process = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    branch = process.stdout.strip()
    if process.returncode == 0 and branch == "prism":
        checks.passed(f"restored source is branch prism at {root}")
    else:
        checks.failed(f"expected branch prism at {root}, found {branch or 'unknown'}")


def check_configuration(
    checks: Checks, config_path: Path, config: dict[str, Any], phase: str
) -> tuple[Path, Path]:
    api_host = str(config.get("api_host") or "")
    api_port = int(config.get("api_port") or 0)
    if api_host in {"127.0.0.1", "localhost", "::1"} and api_port == DEMO_PORT:
        checks.passed("public API is isolated on loopback port 3737")
    else:
        checks.failed(f"unsafe demo API endpoint {api_host}:{api_port}")

    worker_start = int(config.get("worker_port_start") or 0)
    worker_end = int(config.get("worker_port_end") or 0)
    if worker_start <= worker_end and not worker_start <= PROTECTED_PORT <= worker_end:
        checks.passed(f"private worker range {worker_start}-{worker_end} excludes 8002")
    else:
        checks.failed(f"private worker range {worker_start}-{worker_end} is unsafe")

    engines = config.get("engines") or {}
    if (
        config.get("prism_weight_cache_mode") == "ram"
        and engines.get("kvcached_mode") == "required"
    ):
        checks.passed("RAM weight cache and required kvcached mode are enabled")
    else:
        checks.failed("Prism requires prism_weight_cache_mode=ram and kvcached_mode=required")

    actual_preloads = Counter(config.get("prism_preload_models") or [])
    missing = REQUIRED_PRELOADS - actual_preloads
    if actual_preloads == REQUIRED_PRELOADS:
        checks.passed(f"preload multiplicities are demo-ready: {dict(actual_preloads)}")
    else:
        checks.failed(
            f"preload list is not demo-ready; missing={dict(missing)} "
            f"deepseek_enabled={'deepseek-v4-flash-0731' in actual_preloads}"
        )

    if float(config.get("minimum_residency_seconds") or 0) == 0:
        checks.passed("minimum_residency_seconds is zero for switching")
    else:
        checks.failed("minimum_residency_seconds must be zero for the latency demo")
    idle_sleep = float(config.get("prism_idle_sleep_seconds") or 0)
    if 0 < idle_sleep <= 45:
        checks.passed(f"idle workers enter the RAM cache after {idle_sleep:g}s")
    else:
        checks.failed("prism_idle_sleep_seconds must be between 0 and 45 for rehearsal")

    if listening(PROTECTED_PORT):
        checks.information("protected port 8002 is active and will not be touched")
    else:
        checks.information("protected port 8002 is not listening; it remains out of scope")
    demo_listening = listening(DEMO_PORT)
    if phase == "before-start" and not demo_listening:
        checks.passed("port 3737 is free for a fresh service")
    elif phase == "ready" and demo_listening:
        checks.passed("port 3737 has the expected running service")
    else:
        expected = "free" if phase == "before-start" else "listening"
        checks.failed(f"port 3737 must be {expected} during phase {phase}")

    database_path = resolve_path(config_path, str(config.get("database_path") or ""))
    model_store = resolve_path(config_path, str(config.get("model_store") or ""))
    return database_path, model_store


def valid_cache_profile(raw: dict[str, Any]) -> bool:
    return bool(
        raw.get("memory_backend") == "kvcached"
        and raw.get("sleep_vram_mib_per_gpu")
        and raw.get("weight_cache_offload_seconds") is not None
        and raw.get("weight_cache_activation_seconds") is not None
    )


def check_database(
    checks: Checks, database_path: Path, model_store: Path, phase: str
) -> None:
    if not database_path.is_file():
        checks.failed(f"isolated database is missing: {database_path}")
        return
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        mode_row = connection.execute(
            "SELECT mode, machine_fingerprint FROM service_state WHERE singleton=1"
        ).fetchone()
        mode = str(mode_row["mode"]) if mode_row else "missing"
        fingerprint = str(mode_row["machine_fingerprint"] or "") if mode_row else ""
        if mode == "ACTIVE" and fingerprint:
            checks.passed(f"isolated database is ACTIVE with fingerprint {fingerprint[:12]}")
        else:
            checks.failed(f"isolated database state is mode={mode}, fingerprint={fingerprint!r}")

        catalog_rows = connection.execute("SELECT * FROM model_catalog").fetchall()
        catalog = {str(row["nickname"]): row for row in catalog_rows}
        for nickname in REQUIRED_PRELOADS:
            row = catalog.get(nickname)
            artifact = Path(str(row["artifact_path"])) if row and row["artifact_path"] else None
            if row and row["state"] == "AVAILABLE" and artifact and artifact.is_dir():
                checks.passed(f"{nickname} is AVAILABLE with local weights")
            else:
                state = row["state"] if row else "missing"
                checks.failed(f"{nickname} catalog/artifact is not ready: {state}")

        conflicts = [
            row
            for row in catalog_rows
            if row["nickname"] == TARGET_NICKNAME
            or row["huggingface_repo"] == TARGET_REPOSITORY
        ]
        if conflicts:
            checks.failed("temporary qwen3-8b-q8 catalog/repository row already exists")
        else:
            checks.passed("temporary qwen3-8b-q8 catalog row is absent")

        profile_rows = connection.execute(
            """
            SELECT c.nickname, p.machine_fingerprint, p.profile_json
              FROM model_profiles p
              JOIN model_catalog c ON c.id=p.model_id
             WHERE p.active=1
            """
        ).fetchall()
        profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in profile_rows:
            raw = json.loads(row["profile_json"])
            if row["machine_fingerprint"] == fingerprint:
                profiles[str(row["nickname"])].append(raw)

        qwen_tp1 = [
            raw
            for raw in profiles["qwen3.8-27b-nvfp4"]
            if raw.get("tensor_parallel_size") == 1 and valid_cache_profile(raw)
        ]
        qwen_gpu_sets = {
            tuple(gpu_set)
            for raw in qwen_tp1
            for gpu_set in raw.get("eligible_gpu_sets") or []
        }
        if len(qwen_tp1) >= 2 and len(qwen_gpu_sets) >= 2:
            checks.passed("Qwen has two measured TP=1 cache profiles on distinct GPUs")
        else:
            checks.failed("Qwen lacks two distinct measured TP=1 cache profiles")

        gemma = [raw for raw in profiles["gemma-4-31b-it-nvfp4"] if valid_cache_profile(raw)]
        if gemma:
            checks.passed("Gemma has a measured kvcached sleep/wake profile")
        else:
            checks.failed("Gemma lacks a measured kvcached sleep/wake profile")

        laguna = [
            raw
            for raw in profiles["laguna-s-2.1-nvfp4"]
            if raw.get("tensor_parallel_size") == 2 and valid_cache_profile(raw)
        ]
        if laguna:
            checks.passed("Laguna has a measured TP=2 kvcached sleep/wake profile")
        else:
            checks.failed("Laguna lacks a measured TP=2 kvcached sleep/wake profile")

        live_workers = connection.execute(
            "SELECT COUNT(*) FROM workers WHERE state != 'COLD' OR pid IS NOT NULL"
        ).fetchone()[0]
        cold_workers = connection.execute(
            "SELECT COUNT(*) FROM workers WHERE state = 'COLD' AND pid IS NULL"
        ).fetchone()[0]
        if phase == "before-start":
            if live_workers == 0:
                checks.passed("isolated database has no recorded live worker")
            else:
                checks.failed(f"isolated database has {live_workers} stale live worker records")
        else:
            checks.information(
                f"isolated database has {live_workers} running-phase worker record(s)"
            )
        if cold_workers:
            checks.information(
                f"{cold_workers} historical COLD worker rows are ignored by live scheduling"
            )

    target_cache = model_store / "huggingface" / "models--Qwen--Qwen3-8B-FP8"
    target_matches = list(target_cache.parent.glob(f"{target_cache.name}*"))
    if target_matches:
        checks.failed(
            "Qwen3-8B-FP8 cache already exists: "
            + ", ".join(str(path) for path in target_matches)
        )
    else:
        checks.passed("Qwen3-8B-FP8 cache is absent, so onboarding will download")

    old_cache = model_store / "huggingface" / "models--Qwen--Qwen3-8B"
    if old_cache.exists():
        checks.warning(
            f"unrelated BF16 Qwen3-8B cache remains at {old_cache}; it cannot satisfy the FP8 repo"
        )


def check_host(checks: Checks, phase: str) -> None:
    ram = available_ram_gib()
    if ram >= MIN_AVAILABLE_RAM_GIB:
        checks.passed(f"host has {ram:.1f} GiB available RAM")
    else:
        checks.failed(
            f"host has only {ram:.1f} GiB available RAM; need {MIN_AVAILABLE_RAM_GIB:.0f} GiB"
        )

    rows = gpu_rows()
    if len(rows) >= 2:
        checks.passed(f"nvidia-smi reports {len(rows)} GPUs")
    else:
        checks.failed(f"the demo needs two GPUs; nvidia-smi reports {len(rows)}")
    if phase == "before-start":
        busy = [row for row in rows if row[1] > 1024 or row[2] > 5]
        if busy:
            checks.failed(f"GPUs are not idle: {[(row[0], row[1], row[2]) for row in busy]}")
        else:
            checks.passed("all GPUs are idle before worker startup")


def check_ready_service(checks: Checks, config: dict[str, Any]) -> None:
    api_key = os.environ.get("LLMRIO_API_KEY")
    if not api_key:
        checks.failed("set LLMRIO_API_KEY before running --phase ready")
        return
    origin = f"http://127.0.0.1:{int(config['api_port'])}"
    try:
        health = get_json(f"{origin}/health")
        status = get_json(f"{origin}/admin/status", api_key)
    except RuntimeError as exc:
        checks.failed(str(exc))
        return
    if health.get("status") == "ok":
        checks.passed("port-3737 health endpoint is OK")
    else:
        checks.failed(f"health response is not OK: {health!r}")

    prism = status.get("prism") or {}
    if prism.get("kvcached") is True and prism.get("weight_cache") == "host_ram":
        checks.passed("running scheduler reports kvcached plus host-RAM weight cache")
    else:
        checks.failed(f"running scheduler is not in Prism RAM-cache mode: {prism!r}")

    workers = [row for row in status.get("workers") or [] if isinstance(row, dict)]
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for worker in workers:
        by_model[str(worker.get("model"))].append(worker)

    expected = {
        "qwen3.8-27b-nvfp4": 2,
        "gemma-4-31b-it-nvfp4": 1,
        "laguna-s-2.1-nvfp4": 1,
    }
    for model, count in expected.items():
        cached = [
            worker
            for worker in by_model[model]
            if worker.get("state") == "SLEEPING"
            and worker.get("weight_storage") == "host_ram"
            and int(worker.get("active_requests") or 0) == 0
        ]
        if len(cached) >= count:
            checks.passed(f"{model} has {len(cached)} sleeping host-RAM worker(s)")
        else:
            states = [
                (worker.get("state"), worker.get("weight_storage"))
                for worker in by_model[model]
            ]
            checks.failed(f"{model} needs {count} sleeping cache worker(s), found {states}")

    qwen_cached = [
        worker
        for worker in by_model["qwen3.8-27b-nvfp4"]
        if worker.get("state") == "SLEEPING"
        and int(worker.get("tensor_parallel_size") or 0) == 1
    ]
    qwen_gpu_sets = {tuple(worker.get("gpu_uuids") or []) for worker in qwen_cached}
    if len(qwen_cached) >= 2 and len(qwen_gpu_sets) >= 2:
        checks.passed("two Qwen TP=1 replicas are cached on distinct GPUs")
    else:
        checks.failed("running service lacks two distinct sleeping Qwen TP=1 replicas")

    if status.get("queued_models"):
        checks.failed(f"queues are not empty: {status['queued_models']!r}")
    else:
        checks.passed("all model queues are empty before the demo")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    checks = Checks()
    if not config_path.is_file():
        print(f"[FAIL] config file is missing: {config_path}")
        return 1
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    check_repository(checks)
    database_path, model_store = check_configuration(
        checks, config_path, config, args.phase
    )
    check_database(checks, database_path, model_store, args.phase)
    check_host(checks, args.phase)
    if args.phase == "ready":
        check_ready_service(checks, config)

    if checks.failures:
        print(f"[SUMMARY] FAILED with {len(checks.failures)} blocking check(s)")
        return 1
    print(f"[SUMMARY] PASS: Prism demo phase {args.phase} is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
