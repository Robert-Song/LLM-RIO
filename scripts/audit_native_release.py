"""Reproduce native-release audit findings without GPUs or an existing database.

These synthetic checks are audit evidence, not hardware acceptance. Exit 1 means
one or more reviewed defects is still reproducible. All state lives in a temporary
directory; no server is started and no engine or model is downloaded.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from llm_rio.api.routes_inference import _resolve_model
from llm_rio.config import Settings
from llm_rio.domain import Engine, GpuDevice, MachineInventory, PlacementProfile, Role
from llm_rio.engines.identity import launch_binding
from llm_rio.modes.factory import create_mode
from llm_rio.modes.queue.planner import QueuePlanner
from llm_rio.modes.vllm_sleep.lifecycle import SleepLifecycle
from llm_rio.profiles import ProfileRepository, profile_key, profile_to_dict
from llm_rio.security import Principal, issue_api_key
from llm_rio.storage import Database, _now


def settings(root: Path, mode: str = "queue") -> Settings:
    return Settings(
        _env_file=None,
        config_file=root / "absent.toml",
        serving_mode=mode,
        engines={"vllm_executable": str(root / "vllm")},
    )


def profile(config: Settings) -> PlacementProfile:
    sleeping = config.serving_mode.value == "vllm-sleep"
    value = PlacementProfile(
        id="profile",
        model_id="source",
        model_revision="revision",
        engine=Engine.VLLM,
        engine_version="synthetic",
        machine_fingerprint="machine",
        gpu_count=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        eligible_gpu_sets=(("GPU-0",),),
        dtype="auto",
        quantization=None,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        predicted_tokens_per_second=10.0,
        load_and_warmup_seconds=1.0,
        idle_vram_mib_per_gpu=(100,),
        peak_vram_mib_per_gpu=(200,),
        gpu_headroom_mib_per_gpu=(0,),
        capabilities=frozenset({"chat", "streaming"}),
        launch_args={"enable_sleep_mode": sleeping},
        gpu_memory_utilization=0.8,
        kv_cache_capacity_tokens=4096,
        max_full_length_concurrency=1.0,
        serving_mode=config.serving_mode.value,
        vram_measurement_version=2,
        vram_baseline_mib_per_gpu=(0,),
        sleep_vram_mib_per_gpu=(10,) if sleeping else None,
        wake_peak_vram_mib_per_gpu=(200,) if sleeping else None,
        host_cache_mib=100.0 if sleeping else None,
        weight_cache_offload_seconds=0.1 if sleeping else None,
        weight_cache_activation_seconds=0.1 if sleeping else None,
    )
    return replace(value, launch_binding=launch_binding(config, value, value.engine))


async def run(root: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []

    def result(identifier: str, confirmed: bool, details: object) -> None:
        findings.append({"id": identifier, "confirmed": confirmed, "observed": details})

    pending_model = {
        "id": "pending",
        "nickname": "pending",
        "state": "REQUESTED",
        "source_type": "local",
        "artifact_path": None,
        "artifact_hashes": [],
    }
    database = SimpleNamespace(
        list_models=AsyncMock(return_value=[pending_model]),
        model_by_nickname=AsyncMock(return_value=pending_model),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(database=database)))
    try:
        await _resolve_model(
            request, Principal("admin", "admin", Role.ADMIN, "account", True), "pending"
        )
    except Exception as exc:
        result("AUD-01", isinstance(exc, TypeError), type(exc).__name__ + ": " + str(exc))
    else:
        result("AUD-01", False, "No internal error")

    executable = root / "vllm"
    executable.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        "print(Path(__file__).with_suffix('.version').read_text())\n"
    )
    executable.chmod(0o700)
    version_file = executable.with_suffix(".version")
    version_file.write_text("external-engine-version-one")

    config = settings(root)
    base = profile(config)
    single = replace(base, eligible_gpu_sets=(("GPU-0",),))
    tp = replace(
        base, id="tp", gpu_count=2, tensor_parallel_size=2, eligible_gpu_sets=(("GPU-1", "GPU-2"),)
    )
    placements = QueuePlanner._maximum_smallest_placements([single, tp], {"GPU-1", "GPU-2"})
    result(
        "AUD-02", not placements, {"eligible_free_tp2": True, "chosen_placements": len(placements)}
    )

    entered = 0
    release = asyncio.Event()

    async def host_status() -> dict[str, object]:
        nonlocal entered
        entered += 1
        await release.wait()
        return {"pressure_reason": None}

    supervisor = SimpleNamespace(
        settings=settings(root, "vllm-sleep"),
        ram_weight_cache_enabled=True,
        _host_cache_lock=asyncio.Lock(),
        host_cache_status=host_status,
    )
    lifecycle = SleepLifecycle(supervisor)
    tasks = [asyncio.create_task(lifecycle.enforce_host_cache_budget()) for _ in range(2)]
    for _ in range(10):
        await asyncio.sleep(0)
    concurrent = entered
    release.set()
    await asyncio.gather(*tasks)
    result("AUD-03", concurrent == 2, {"concurrent_budget_checks": concurrent, "expected": 1})

    inventory = MachineInventory(
        "test", "test", None, (GpuDevice("GPU-0", 0, "test", 10000),), "test", "machine"
    )
    sleep_config = settings(root, "vllm-sleep")
    sleep_profile = profile(sleep_config)
    incomplete = replace(
        sleep_profile,
        host_cache_mib=None,
        weight_cache_offload_seconds=None,
        weight_cache_activation_seconds=None,
    )
    nonfinite = replace(
        sleep_profile, idle_vram_mib_per_gpu=(float("nan"),), peak_vram_mib_per_gpu=(float("nan"),)
    )
    mode = create_mode(sleep_config, inventory)
    missing_allowed = mode.eligibility(incomplete).allowed
    nonfinite_allowed = mode.eligibility(nonfinite).allowed
    result(
        "AUD-04",
        missing_allowed or nonfinite_allowed,
        {
            "incomplete_sleep_evidence_allowed": missing_allowed,
            "nonfinite_vram_allowed": nonfinite_allowed,
        },
    )

    before = launch_binding(config, base, Engine.VLLM)
    old_version = (
        await asyncio.to_thread(subprocess.check_output, [str(executable), "--version"], text=True)
    ).strip()
    version_file.write_text("external-engine-version-two")
    after = launch_binding(config, base, Engine.VLLM)
    new_version = (
        await asyncio.to_thread(subprocess.check_output, [str(executable), "--version"], text=True)
    ).strip()
    result(
        "AUD-05",
        old_version != new_version and before == after,
        {
            "external_engine_version_changed": old_version != new_version,
            "binding_changed": before != after,
        },
    )

    store = Database(root / "audit.db")
    await store.open()
    try:
        token, prefix = issue_api_key("admin")
        await store.create_key(
            key_id="admin",
            nickname="admin",
            role=Role.ADMIN,
            account_id="account",
            account_nickname="account",
            prefix=prefix,
            api_key=token,
            limit_tokens=0,
            unlimited=True,
        )
        model_id, _ = await store.create_model_job(
            nickname="source",
            repo="audit/model",
            revision="revision",
            creator_key_id="admin",
            grant_key_ids=[],
        )
        await store.execute(
            "UPDATE model_catalog SET state='AVAILABLE', artifact_path=?, "
            "resolved_revision='revision' WHERE id=?",
            (str(root), model_id),
        )
        measured = replace(base, model_id=model_id)
        raw = profile_to_dict(measured)
        await store.execute(
            "INSERT INTO model_profiles "
            "(id,model_id,machine_fingerprint,profile_key,profile_json,verified_at,active) "
            "VALUES (?,?,?,?,?,?,1)",
            (measured.id, model_id, "machine", profile_key(raw), json.dumps(raw), _now()),
        )
        repository = ProfileRepository(store, "machine")
        source = await store.model_by_id(model_id)
        assert source is not None
        clone, profiles = await repository.clone_model(
            source_model=source,
            nickname="extended",
            creator_key_id="admin",
            request_defaults={},
            max_model_len=8192,
            yarn_factor=None,
            yarn_original_max_model_len=None,
            inherit_grants=False,
        )
        jobs = await store.fetchall("SELECT id FROM model_jobs WHERE model_id=?", (clone["id"],))
        result(
            "AUD-06",
            not jobs and not profiles[0].measurements_valid,
            {
                "clone_needs_revalidation": not profiles[0].measurements_valid,
                "clone_validation_jobs": len(jobs),
            },
        )
    finally:
        await store.close()
    return findings


def main() -> None:
    # Disregard operator environment settings; this audit is entirely synthetic.
    for name in list(os.environ):
        if name.startswith("LLMRIO_"):
            del os.environ[name]
    with tempfile.TemporaryDirectory(prefix="rio-native-audit-") as directory:
        findings = asyncio.run(run(Path(directory)))
    print(json.dumps({"kind": "offline-audit", "findings": findings}, indent=2))
    raise SystemExit(1 if any(item["confirmed"] for item in findings) else 0)


if __name__ == "__main__":
    main()
