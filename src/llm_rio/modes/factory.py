from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_rio.config import ServingMode, Settings
from llm_rio.domain import CURRENT_VRAM_MEASUREMENT_VERSION, MachineInventory, PlacementProfile
from llm_rio.engines.identity import launch_binding
from llm_rio.modes.contracts import ModeCapabilities, PlacementPlanner, ProfileEligibility
from llm_rio.profiles import profile_verified_for_mode


@dataclass(frozen=True)
class SelectedMode:
    capabilities: ModeCapabilities
    planner: PlacementPlanner
    settings: Settings

    def eligibility(self, profile: PlacementProfile) -> ProfileEligibility:
        if profile.engine.value not in self.capabilities.engines:
            return ProfileEligibility(False, "unsupported_engine")
        if not profile.measurements_valid:
            return ProfileEligibility(False, "measurements_invalid")
        if profile.serving_mode != self.capabilities.name:
            return ProfileEligibility(False, "mode_mismatch")
        if profile.vram_measurement_version != CURRENT_VRAM_MEASUREMENT_VERSION:
            return ProfileEligibility(False, "measurement_format_unsupported")
        try:
            current_binding = launch_binding(self.settings, profile, profile.engine)
        except (OSError, RuntimeError, ValueError):
            return ProfileEligibility(False, "engine_identity_unavailable")
        if profile.launch_binding != current_binding:
            return ProfileEligibility(False, "launch_configuration_changed")
        allowed = profile_verified_for_mode(
            profile,
            kvcached_required=self.capabilities.name == "kv-cached",
            ram_weight_cache_required=self.capabilities.sleep,
            queue_mode_required=self.capabilities.name == "queue",
        )
        return ProfileEligibility(allowed, "eligible" if allowed else "incompatible_measurements")


def create_mode(settings: Settings, inventory: MachineInventory) -> SelectedMode:
    kwargs: dict[str, Any] = dict(
        wait_duration_seconds=settings.wait_duration_seconds,
        minimum_residency_seconds=settings.minimum_residency_seconds,
        fair_share_seconds=settings.fair_share_seconds,
        gpu_vram_mib={gpu.uuid: gpu.total_vram_mib for gpu in inventory.gpus},
        reserved_vram_mib=settings.reserved_vram_mib,
    )
    planner: PlacementPlanner
    if settings.serving_mode is ServingMode.QUEUE:
        from llm_rio.modes.queue.planner import QueuePlanner

        planner = QueuePlanner(
            wait_duration_seconds=settings.wait_duration_seconds,
            minimum_residency_seconds=settings.minimum_residency_seconds,
            **settings.modes.queue.model_dump(),
        )
        engines = ("vllm", "llama.cpp") if settings.engines.enable_llama_cpp else ("vllm",)
        capabilities = ModeCapabilities("queue", False, engines, False, True)
    elif settings.serving_mode is ServingMode.VLLM_SLEEP:
        from llm_rio.modes.vllm_sleep.planner import SleepPlanner

        planner = SleepPlanner(
            **kwargs, cache_idle_sleep_seconds=settings.modes.vllm_sleep.idle_sleep_seconds
        )
        capabilities = ModeCapabilities("vllm-sleep", False, ("vllm",), True, True)
    else:
        from llm_rio.modes.kv_cached.planner import KVCachedPlanner

        planner = KVCachedPlanner(
            **kwargs,
            cache_idle_sleep_seconds=settings.modes.kv_cached.idle_sleep_seconds,
            cache_max_workers_per_gpu=settings.modes.kv_cached.max_workers_per_gpu,
        )
        capabilities = ModeCapabilities("kv-cached", True, ("vllm",), True, False)
    return SelectedMode(capabilities, planner, settings)
