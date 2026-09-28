"""Translate historical regression fixture inputs to explicit release configurations.

This module is test-only. Production rejects every legacy option.
"""

from dataclasses import replace as dataclass_replace
from typing import Any

from llm_rio.config import Settings as ReleaseSettings
from llm_rio.domain import PlacementProfile as ReleaseProfile
from llm_rio.engines.identity import launch_binding
from llm_rio.modes.kv_cached.planner import KVCachedPlanner
from llm_rio.modes.queue.planner import QueuePlanner
from llm_rio.modes.vllm_sleep.planner import SleepPlanner


def EngineSettings(**kwargs: Any) -> dict[str, Any]:
    return kwargs


def Settings(**kwargs: Any) -> ReleaseSettings:
    engines = kwargs.get("engines", {})
    if isinstance(engines, dict):
        engines = dict(engines)
        backend = engines.pop("kvcached_mode", "none")
        kwargs["engines"] = engines
    else:
        backend = "none"
    weights = kwargs.pop("prism_weight_cache_mode", "ram")
    mode = kwargs.get("serving_mode") or (
        "kv-cached" if backend == "required" else "vllm-sleep" if weights == "ram" else "queue"
    )
    kwargs["serving_mode"] = mode
    cache = {
        key.removeprefix("prism_"): kwargs.pop(key)
        for key in list(kwargs)
        if key.startswith("prism_")
    }
    cache.pop("sleep_gpu_reserve_mib", None)
    native = {k: v for k, v in cache.items() if k != "max_workers_per_gpu"}
    kwargs.setdefault("modes", {mode.replace("-", "_"): cache if mode == "kv-cached" else native})
    return ReleaseSettings(**kwargs)


def GreedyPlacementPlanner(**kwargs: Any) -> Any:
    cached = kwargs.pop("prism_enabled", False)
    kwargs.pop("queue_mode", None)
    elastic = kwargs.pop("kvcached_required", False)
    kwargs.pop("prism_weight_cache_enabled", None)
    kwargs = {key.replace("prism_", "cache_"): value for key, value in kwargs.items()}
    cls = KVCachedPlanner if elastic else SleepPlanner if cached else QueuePlanner
    if cls is QueuePlanner:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "wait_duration_seconds",
                "minimum_residency_seconds",
                "scale_window_seconds",
                "minimum_marginal_efficiency",
            }
        }
    return cls(**kwargs)


def profile_fields(kwargs: dict[str, Any]) -> dict[str, Any]:
    fields = dict(kwargs)
    normal = fields.pop("normal_verified", True)
    cached = fields.pop("kvcached_verified", True)
    if "normal_verified" in kwargs or "kvcached_verified" in kwargs:
        fields["measurements_valid"] = (
            normal if fields.get("memory_backend", "native") == "native" else cached
        )
    return fields


def PlacementProfile(**kwargs: Any) -> ReleaseProfile:
    fields = profile_fields(kwargs)
    fields.setdefault(
        "serving_mode",
        "kv-cached"
        if fields.get("memory_backend") == "kvcached"
        else "vllm-sleep"
        if fields.get("sleep_vram_mib_per_gpu") is not None
        and fields.get("launch_args", {}).get("enable_sleep_mode") is not False
        else "queue",
    )
    return measured(ReleaseProfile(**fields))


def replace(obj: Any, **kwargs: Any) -> Any:
    fields = profile_fields(kwargs)
    if (
        isinstance(obj, ReleaseProfile)
        and "kvcached_verified" in kwargs
        and obj.memory_backend == "kvcached"
    ):
        fields["measurements_valid"] = kwargs["kvcached_verified"]
    if "memory_backend" in fields:
        fields["serving_mode"] = (
            "kv-cached" if fields["memory_backend"] == "kvcached" else "vllm-sleep"
        )
    if "launch_args" in fields and fields["launch_args"].get("enable_sleep_mode") is False:
        fields["serving_mode"] = "queue"
    result = dataclass_replace(obj, **fields)
    return measured(result) if isinstance(result, ReleaseProfile) else result


def measured(profile: ReleaseProfile) -> ReleaseProfile:
    settings = ReleaseSettings(serving_mode=profile.serving_mode)
    return dataclass_replace(
        profile, launch_binding=launch_binding(settings, profile, profile.engine)
    )
