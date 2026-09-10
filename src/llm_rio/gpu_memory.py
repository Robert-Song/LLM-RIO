"""Live GPU headroom and measured native worker admission requirements."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from llm_rio.domain import Engine, PlacementProfile

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class GpuMemory:
    total_mib: int
    free_mib: int
    process_group_mib: dict[int, int] = field(default_factory=dict)


def read_gpu_memory(gpu_uuids: tuple[str, ...]) -> dict[str, GpuMemory]:
    """Read global free VRAM, including allocations from sleeping and foreign processes."""
    import pynvml  # type: ignore[import-untyped]

    pynvml.nvmlInit()
    try:
        samples = {}
        for gpu_uuid in gpu_uuids:
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            groups: dict[int, int] = {}
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except pynvml.NVMLError:
                processes = []
            for process in processes:
                used = process.usedGpuMemory
                if not isinstance(used, int) or not 0 <= used <= memory.used:
                    continue
                try:
                    group = os.getpgid(process.pid)
                except (OSError, ProcessLookupError):
                    continue
                groups[group] = groups.get(group, 0) + used // _MIB
            samples[gpu_uuid] = GpuMemory(
                total_mib=int(memory.total // _MIB),
                free_mib=int(memory.free // _MIB),
                process_group_mib=groups,
            )
        return samples
    finally:
        pynvml.nvmlShutdown()


def required_free_vram(
    profile: PlacementProfile,
    index: int,
    sample: GpuMemory,
    *,
    reserve_mib: int,
    waking_pid: int | None = None,
    waking: bool = False,
) -> int:
    """Use measured peaks; on wake credit only demonstrably resident target memory."""
    peak = max(
        profile.peak_vram_mib_per_gpu[index],
        (profile.wake_peak_vram_mib_per_gpu or profile.peak_vram_mib_per_gpu)[index],
    )
    if waking:
        measured_sleep = (profile.sleep_vram_mib_per_gpu or (0,) * profile.gpu_count)[index]
        resident = sample.process_group_mib.get(waking_pid, 0) if waking_pid is not None else 0
        # The target's context is already included in global used VRAM. Other
        # sleepers' contexts receive no credit and must fit or be evicted.
        credit = min(measured_sleep, resident)
        return max(0, peak - credit) + reserve_mib
    required = peak + reserve_mib
    if profile.engine is Engine.VLLM:
        # Native vLLM also requires its configured fraction of *physical* VRAM
        # to be free at startup, even when measured generation used less.
        required = max(required, math.ceil(sample.total_mib * profile.gpu_memory_utilization))
    return required
