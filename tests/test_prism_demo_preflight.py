from __future__ import annotations

from typing import Any

import prism_demo_preflight as preflight


def sleeping_worker(model: str, gpu: str, *, tp: int = 1) -> dict[str, Any]:
    return {
        "model": model,
        "state": "SLEEPING",
        "weight_storage": "host_ram",
        "active_requests": 0,
        "tensor_parallel_size": tp,
        "gpu_uuids": [gpu] if tp == 1 else ["gpu-0", "gpu-1"],
    }


def test_valid_cache_profile_requires_measured_ram_transition() -> None:
    profile = {
        "memory_backend": "kvcached",
        "sleep_vram_mib_per_gpu": [1024],
        "weight_cache_offload_seconds": 2.0,
        "weight_cache_activation_seconds": 0.5,
    }
    assert preflight.valid_cache_profile(profile)
    profile["weight_cache_activation_seconds"] = None
    assert not preflight.valid_cache_profile(profile)


def test_host_ram_threshold_accounts_for_cached_ready_workers(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "available_ram_gib", lambda: 100.0)
    monkeypatch.setattr(preflight, "gpu_rows", lambda: [("GPU-0", 0.0, 0.0), ("GPU-1", 0.0, 0.0)])

    before_start = preflight.Checks()
    preflight.check_host(before_start, "before-start")
    assert before_start.failures

    ready = preflight.Checks()
    preflight.check_host(ready, "ready")
    assert ready.failures == []


def test_ready_service_accepts_complete_sleeping_cache(monkeypatch) -> None:
    status = {
        "prism": {"kvcached": True, "weight_cache": "host_ram"},
        "queued_models": {},
        "workers": [
            sleeping_worker("qwen3.8-27b-nvfp4", "gpu-0"),
            sleeping_worker("qwen3.8-27b-nvfp4", "gpu-1"),
            sleeping_worker("gemma-4-31b-it-nvfp4", "gpu-0"),
            sleeping_worker("laguna-s-2.1-nvfp4", "gpu-0", tp=2),
        ],
    }

    def fake_get_json(url: str, api_key: str | None = None) -> dict[str, Any]:
        return {"status": "ok"} if url.endswith("/health") else status

    monkeypatch.setenv("LLMRIO_API_KEY", "test-key")
    monkeypatch.setattr(preflight, "get_json", fake_get_json)
    checks = preflight.Checks()
    preflight.check_ready_service(checks, {"api_port": 3737})
    assert checks.failures == []


def test_ready_service_rejects_missing_replica_and_gpu_resident_weights(monkeypatch) -> None:
    status = {
        "prism": {"kvcached": True, "weight_cache": "host_ram"},
        "queued_models": {},
        "workers": [
            {
                **sleeping_worker("qwen3.8-27b-nvfp4", "gpu-0"),
                "state": "READY",
                "weight_storage": "gpu",
            },
            sleeping_worker("gemma-4-31b-it-nvfp4", "gpu-0"),
            sleeping_worker("laguna-s-2.1-nvfp4", "gpu-0", tp=2),
        ],
    }

    def fake_get_json(url: str, api_key: str | None = None) -> dict[str, Any]:
        return {"status": "ok"} if url.endswith("/health") else status

    monkeypatch.setenv("LLMRIO_API_KEY", "test-key")
    monkeypatch.setattr(preflight, "get_json", fake_get_json)
    checks = preflight.Checks()
    preflight.check_ready_service(checks, {"api_port": 3737})
    assert any("Qwen" in failure or "qwen" in failure for failure in checks.failures)
