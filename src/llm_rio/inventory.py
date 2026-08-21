from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from itertools import combinations
from typing import Any

from llm_rio.domain import GpuDevice, MachineInventory


class InventoryError(RuntimeError):
    pass


def _cuda_version_from_driver(raw_version: int) -> str:
    major = raw_version // 1000
    minor = (raw_version % 1000) // 10
    return f"{major}.{minor}"


def _read_topology() -> tuple[dict[str, dict[str, str]], str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return {}, "unavailable"
    lines = [line.rstrip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {}, "unavailable"
    headers = lines[0].split()
    gpu_headers = [header for header in headers if header.startswith("GPU")]
    topology: dict[str, dict[str, str]] = {}
    for line in lines[1:]:
        cells = line.split()
        if not cells or not cells[0].startswith("GPU"):
            continue
        topology[cells[0]] = {
            peer: cells[index + 1]
            for index, peer in enumerate(gpu_headers)
            if index + 1 < len(cells)
        }
    digest = hashlib.sha256(completed.stdout.encode()).hexdigest()
    return topology, digest


def discover_inventory(machine_id: str, managed_gpu_uuids: list[str]) -> MachineInventory:
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
    except Exception as exc:
        raise InventoryError(f"NVML initialization failed: {exc}") from exc

    try:
        driver_version = str(pynvml.nvmlSystemGetDriverVersion())
        try:
            cuda_version = _cuda_version_from_driver(pynvml.nvmlSystemGetCudaDriverVersion_v2())
        except pynvml.NVMLError:
            cuda_version = None
        discovered_devices: list[GpuDevice] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = str(pynvml.nvmlDeviceGetUUID(handle))
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
            except pynvml.NVMLError:
                compute_capability = None
            try:
                pci_bus_id = str(pynvml.nvmlDeviceGetPciInfo(handle).busId)
            except pynvml.NVMLError:
                pci_bus_id = None
            discovered_devices.append(
                GpuDevice(
                    uuid=uuid,
                    index=index,
                    name=str(pynvml.nvmlDeviceGetName(handle)),
                    total_vram_mib=int(memory.total // (1024 * 1024)),
                    compute_capability=compute_capability,
                    pci_bus_id=pci_bus_id,
                )
            )

        by_uuid = {device.uuid: device for device in discovered_devices}
        by_index = {device.index: device for device in discovered_devices}
        visible_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        scheduler_visible: set[str]
        if visible_value is None:
            scheduler_visible = set(by_uuid)
        else:
            tokens = [token.strip() for token in visible_value.split(",") if token.strip()]
            if visible_value.strip() in {"", "-1", "NoDevFiles", "none", "void"}:
                tokens = []
            scheduler_visible = set()
            for token in tokens:
                if token.isdecimal():
                    device = by_index.get(int(token))
                    if device is None:
                        raise InventoryError(
                            f"CUDA_VISIBLE_DEVICES references unknown NVML index {token}"
                        )
                    scheduler_visible.add(device.uuid)
                    continue
                if token.startswith("MIG-"):
                    raise InventoryError(
                        "MIG CUDA_VISIBLE_DEVICES entries are not supported by this release"
                    )
                matches = [uuid for uuid in by_uuid if uuid == token or uuid.startswith(token)]
                if len(matches) != 1:
                    raise InventoryError(
                        f"CUDA_VISIBLE_DEVICES entry {token!r} did not resolve to one GPU UUID"
                    )
                scheduler_visible.add(matches[0])

        configured = set(managed_gpu_uuids)
        missing = configured - set(by_uuid)
        if missing:
            raise InventoryError(f"Configured managed GPU UUIDs were not found: {sorted(missing)}")
        outside_allocation = configured - scheduler_visible
        if visible_value is not None and outside_allocation:
            raise InventoryError(
                "Configured managed GPU UUIDs are outside CUDA_VISIBLE_DEVICES: "
                f"{sorted(outside_allocation)}"
            )
        selected = configured or scheduler_visible
        devices = [device for device in discovered_devices if device.uuid in selected]
    finally:
        pynvml.nvmlShutdown()

    if not devices:
        raise InventoryError("No managed NVIDIA GPUs were discovered")
    topology, topology_hash = _read_topology()
    fingerprint_payload = {
        "driver": driver_version,
        "cuda": cuda_version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "gpus": [asdict(device) for device in devices],
        "topology_hash": topology_hash,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    return MachineInventory(
        machine_id=machine_id,
        driver_version=driver_version,
        cuda_driver_version=cuda_version,
        gpus=tuple(devices),
        topology_hash=topology_hash,
        fingerprint=fingerprint,
        topology=topology,
    )


def gpu_environment(
    gpu_uuids: tuple[str, ...], overrides: dict[str, str] | None = None
) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(overrides or {})
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
    return environment


def candidate_gpu_sets(inventory: MachineInventory, gpu_count: int) -> tuple[tuple[str, ...], ...]:
    """Enumerate deterministic homogeneous subsets; exact validation decides eligibility."""
    by_model: dict[tuple[str, int], list[GpuDevice]] = {}
    for device in inventory.gpus:
        by_model.setdefault((device.name, device.total_vram_mib), []).append(device)
    candidates: list[tuple[str, ...]] = []
    for devices in by_model.values():
        for group in combinations(devices, gpu_count):
            candidates.append(tuple(device.uuid for device in group))
    device_by_uuid = {device.uuid: device for device in inventory.gpus}
    link_cost = {
        "X": 0,
        "NV18": 1,
        "NV12": 2,
        "NV8": 3,
        "NV4": 4,
        "NV2": 5,
        "NV1": 6,
        "PIX": 10,
        "PXB": 20,
        "PHB": 30,
        "NODE": 40,
        "SYS": 50,
    }

    def topology_cost(group: tuple[str, ...]) -> tuple[int, tuple[str, ...]]:
        cost = 0
        for left, right in combinations(group, 2):
            left_name = f"GPU{device_by_uuid[left].index}"
            right_name = f"GPU{device_by_uuid[right].index}"
            link = inventory.topology.get(left_name, {}).get(right_name, "SYS")
            if link.startswith("NV") and link not in link_cost:
                link = "NV1"
            cost += link_cost.get(link, 50)
        return cost, group

    candidates.sort(key=topology_cost)
    return tuple(candidates)


def read_live_gpu_status(inventory: MachineInventory) -> list[dict[str, object]]:
    """Sample nvidia-smi/NVML-style health metrics for the managed GPUs."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as exc:
        return [
            {
                "index": device.index,
                "uuid": device.uuid,
                "name": device.name,
                "total_vram_mib": device.total_vram_mib,
                "available": False,
                "error": str(exc),
            }
            for device in inventory.gpus
        ]

    def optional(call: Callable[..., Any], *args: object) -> Any:
        try:
            return call(*args)
        except Exception:
            return None

    samples: list[dict[str, object]] = []
    try:
        for device in inventory.gpus:
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(device.uuid)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = optional(
                    pynvml.nvmlDeviceGetTemperature,
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
                power_draw = optional(pynvml.nvmlDeviceGetPowerUsage, handle)
                power_limit = optional(pynvml.nvmlDeviceGetEnforcedPowerLimit, handle)
                fan_speed = optional(pynvml.nvmlDeviceGetFanSpeed, handle)
                graphics_clock = optional(
                    pynvml.nvmlDeviceGetClockInfo,
                    handle,
                    pynvml.NVML_CLOCK_GRAPHICS,
                )
                memory_clock = optional(
                    pynvml.nvmlDeviceGetClockInfo,
                    handle,
                    pynvml.NVML_CLOCK_MEM,
                )
                compute_processes = optional(
                    pynvml.nvmlDeviceGetComputeRunningProcesses,
                    handle,
                )
                samples.append(
                    {
                        "index": device.index,
                        "uuid": device.uuid,
                        "name": device.name,
                        "available": True,
                        "gpu_utilization_percent": int(utilization.gpu),
                        "memory_utilization_percent": int(utilization.memory),
                        "used_vram_mib": int(memory.used // (1024 * 1024)),
                        "free_vram_mib": int(memory.free // (1024 * 1024)),
                        "total_vram_mib": int(memory.total // (1024 * 1024)),
                        "temperature_c": int(temperature) if temperature is not None else None,
                        "power_draw_w": round(float(power_draw) / 1000.0, 1)
                        if power_draw is not None
                        else None,
                        "power_limit_w": round(float(power_limit) / 1000.0, 1)
                        if power_limit is not None
                        else None,
                        "fan_speed_percent": int(fan_speed) if fan_speed is not None else None,
                        "graphics_clock_mhz": int(graphics_clock)
                        if graphics_clock is not None
                        else None,
                        "memory_clock_mhz": int(memory_clock) if memory_clock is not None else None,
                        "compute_process_count": len(compute_processes)
                        if isinstance(compute_processes, list)
                        else None,
                    }
                )
            except Exception as exc:
                samples.append(
                    {
                        "index": device.index,
                        "uuid": device.uuid,
                        "name": device.name,
                        "total_vram_mib": device.total_vram_mib,
                        "available": False,
                        "error": str(exc),
                    }
                )
    finally:
        pynvml.nvmlShutdown()
    return samples
