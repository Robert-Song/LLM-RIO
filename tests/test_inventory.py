"""Tests for GPU inventory discovery helpers (pure logic, no NVML required)."""

from __future__ import annotations

import pytest

from llm_rio.domain import GpuDevice, MachineInventory
from llm_rio.inventory import (
    InventoryError,
    _cuda_version_from_driver,
    candidate_gpu_sets,
    discover_inventory,
    gpu_environment,
)


class TestCudaVersionFromDriver:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (12060, "12.6"),
            (13030, "13.3"),
            (11040, "11.4"),
            (12000, "12.0"),
            (11040, "11.4"),
        ],
    )
    def test_parsing(self, raw: int, expected: str) -> None:
        assert _cuda_version_from_driver(raw) == expected


class TestGpuEnvironment:
    def test_sets_cuda_visible_devices(self) -> None:
        env = gpu_environment(("GPU-a", "GPU-b"))
        assert env["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"

    def test_inherits_os_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLMRIO_TEST_VAR", "inherited")
        env = gpu_environment(("GPU-a",))
        assert env["LLMRIO_TEST_VAR"] == "inherited"

    def test_overrides_win(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NCCL_DEBUG", "INFO")
        env = gpu_environment(("GPU-a",), {"NCCL_DEBUG": "WARN"})
        assert env["NCCL_DEBUG"] == "WARN"


def make_inventory(
    gpu_count: int,
    *,
    topology: dict[str, dict[str, str]] | None = None,
) -> MachineInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{index}",
            index=index,
            name="TestGPU",
            total_vram_mib=24000,
            compute_capability="9.0",
            pci_bus_id=f"0000:{index:02d}:00.0",
        )
        for index in range(gpu_count)
    )
    return MachineInventory(
        machine_id="test-machine",
        driver_version="1.0",
        cuda_driver_version="13.3",
        gpus=devices,
        topology_hash="hash",
        fingerprint="fingerprint",
        topology=topology or {},
    )


class TestCandidateGpuSets:
    def test_single_gpu_sets(self) -> None:
        inventory = make_inventory(3)
        sets = candidate_gpu_sets(inventory, 1)
        assert len(sets) == 3
        assert all(len(group) == 1 for group in sets)

    def test_pair_sets_are_homogeneous(self) -> None:
        inventory = make_inventory(3)
        sets = candidate_gpu_sets(inventory, 2)
        assert len(sets) == 3
        assert all(len(group) == 2 for group in sets)
        # Combinations of 3 GPUs taken 2 at a time: 3 pairs.
        flat = {gpu for group in sets for gpu in group}
        assert flat == {"GPU-0", "GPU-1", "GPU-2"}

    def test_gpu_count_exceeding_inventory_returns_empty(self) -> None:
        inventory = make_inventory(2)
        assert candidate_gpu_sets(inventory, 4) == ()

    def test_topology_orders_sets_by_link_cost(self) -> None:
        inventory = make_inventory(3, topology={
            "GPU0": {"GPU0": "X", "GPU1": "NV1", "GPU2": "SYS"},
            "GPU1": {"GPU0": "NV1", "GPU1": "X", "GPU2": "NV8"},
            "GPU2": {"GPU0": "SYS", "GPU1": "NV8", "GPU2": "X"},
        })
        sets = candidate_gpu_sets(inventory, 2)
        # (GPU-1, GPU-2) = NV8 (cost 3), (GPU-0, GPU-1) = NV1 (cost 6),
        # (GPU-0, GPU-2) = SYS (cost 50). Sorted ascending by cost.
        assert sets[0] == ("GPU-1", "GPU-2")
        assert sets[1] == ("GPU-0", "GPU-1")
        assert sets[2] == ("GPU-0", "GPU-2")

    def test_unknown_topology_link_defaults_to_sys(self) -> None:
        inventory = make_inventory(2)
        sets = candidate_gpu_sets(inventory, 2)
        assert len(sets) == 1
        assert sets[0] == ("GPU-0", "GPU-1")

    def test_mixed_vram_groups_are_homogeneous_by_model(self) -> None:
        devices = (
            GpuDevice(uuid="GPU-big-1", index=0, name="Big", total_vram_mib=80000),
            GpuDevice(uuid="GPU-big-2", index=1, name="Big", total_vram_mib=80000),
            GpuDevice(uuid="GPU-small", index=2, name="Small", total_vram_mib=8000),
        )
        inventory = MachineInventory(
            machine_id="m",
            driver_version="1.0",
            cuda_driver_version="13.3",
            gpus=devices,
            topology_hash="h",
            fingerprint="f",
        )
        sets = candidate_gpu_sets(inventory, 2)
        # Only the two "Big" GPUs form a homogeneous pair.
        assert sets == (("GPU-big-1", "GPU-big-2"),)


class TestDiscoverInventory:
    def test_raises_when_pynvml_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ModuleNotFoundError("No module named 'pynvml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(InventoryError, match="NVML initialization failed"):
            discover_inventory("machine", [])
