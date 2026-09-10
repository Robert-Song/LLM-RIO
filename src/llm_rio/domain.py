from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# Version 1 profiles stored whole-device NVML usage and therefore could not be
# composed safely with other active or sleeping workers. Version 2 stores the
# validation worker's incremental usage above a captured per-GPU baseline.
CURRENT_VRAM_MEASUREMENT_VERSION = 2


def utc_now() -> datetime:
    return datetime.now(UTC)


class Role(StrEnum):
    USER = "user"
    TA = "ta"
    ADMIN = "admin"


class CatalogState(StrEnum):
    REQUESTED = "REQUESTED"
    DOWNLOADING = "DOWNLOADING"
    VALIDATION_PENDING = "VALIDATION_PENDING"
    VALIDATING = "VALIDATING"
    AVAILABLE = "AVAILABLE"
    NEEDS_ADMIN_REVIEW = "NEEDS_ADMIN_REVIEW"
    DISABLED = "DISABLED"


class RuntimeState(StrEnum):
    COLD = "COLD"
    LOADING = "LOADING"
    READY = "READY"
    DRAINING = "DRAINING"
    OFFLOADING = "OFFLOADING"
    SLEEPING = "SLEEPING"
    WAKING = "WAKING"
    STOPPING = "STOPPING"


class ServiceMode(StrEnum):
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    MAINTENANCE_READY = "MAINTENANCE_READY"


class Engine(StrEnum):
    VLLM = "vllm"
    LLAMA_CPP = "llama.cpp"


@dataclass(frozen=True, slots=True)
class GpuDevice:
    uuid: str
    index: int
    name: str
    total_vram_mib: int
    compute_capability: str | None = None
    pci_bus_id: str | None = None
    numa_node: int | None = None


@dataclass(frozen=True, slots=True)
class MachineInventory:
    machine_id: str
    driver_version: str
    cuda_driver_version: str | None
    gpus: tuple[GpuDevice, ...]
    topology_hash: str
    fingerprint: str
    topology: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlacementProfile:
    id: str
    model_id: str
    model_revision: str
    engine: Engine
    engine_version: str
    machine_fingerprint: str
    gpu_count: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    eligible_gpu_sets: tuple[tuple[str, ...], ...]
    dtype: str
    quantization: str | None
    max_model_len: int
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    predicted_tokens_per_second: float
    load_and_warmup_seconds: float
    idle_vram_mib_per_gpu: tuple[int, ...]
    peak_vram_mib_per_gpu: tuple[int, ...]
    gpu_headroom_mib_per_gpu: tuple[int, ...]
    capabilities: frozenset[str]
    launch_args: dict[str, Any]
    gpu_memory_utilization: float
    kv_cache_capacity_tokens: int | None
    max_full_length_concurrency: float | None
    memory_backend: str = "native"
    sleep_vram_mib_per_gpu: tuple[int, ...] | None = None
    weight_cache_offload_seconds: float | None = None
    weight_cache_activation_seconds: float | None = None
    host_cache_mib: float | None = None
    normal_verified: bool = True
    kvcached_verified: bool = False
    vram_measurement_version: int = 1
    vram_baseline_mib_per_gpu: tuple[int, ...] | None = None
    wake_peak_vram_mib_per_gpu: tuple[int, ...] | None = None


@dataclass(slots=True)
class WorkerPlacement:
    id: str
    profile: PlacementProfile
    gpu_uuids: tuple[str, ...]
    port: int
    state: RuntimeState = RuntimeState.LOADING
    admitted_request_ids: set[str] = field(default_factory=set)
    accepted_requests: int = 0
    outstanding_token_work: int = 0
    ready_at: datetime | None = None
    last_demand_at: datetime = field(default_factory=utc_now)
    drain_started_at: datetime | None = None
    sleeping_at: datetime | None = None
    last_activation_seconds: float | None = None
    last_offload_seconds: float | None = None
    host_weights_cached: bool = False
    process_pid: int | None = None
    host_cache_accounted_mib: float = 0.0
    host_cache_accounting_source: str | None = None
    process_rss_mib: float = 0.0
    process_pss_mib: float = 0.0
    process_swap_mib: float = 0.0
    last_cache_eviction_reason: str | None = None

    @property
    def model_id(self) -> str:
        return self.profile.model_id

    @property
    def is_routable(self) -> bool:
        return self.state is RuntimeState.READY
