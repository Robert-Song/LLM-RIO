"""Tests for candidate-shape generation and validation helpers."""

from __future__ import annotations

from pathlib import Path

from llm_rio.domain import GpuDevice, MachineInventory
from llm_rio.validation import (
    build_candidate_shapes,
    validation_log_path,
)


def make_inventory(*vram_mib: int) -> MachineInventory:
    devices = tuple(
        GpuDevice(
            uuid=f"GPU-{index}",
            index=index,
            name="TestGPU",
            total_vram_mib=size,
            compute_capability="9.0",
        )
        for index, size in enumerate(vram_mib)
    )
    return MachineInventory(
        machine_id="m",
        driver_version="1.0",
        cuda_driver_version="13.3",
        gpus=devices,
        topology_hash="h",
        fingerprint="f",
    )


class TestBuildCandidateShapes:
    def test_single_shape_for_single_gpu(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000),
            weight_bytes=8_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
        )
        assert len(shapes) == 1
        shape = shapes[0]
        assert shape.gpu_count == 1
        assert shape.tensor_parallel_size == 1
        assert shape.max_model_len == 4096
        assert shape.eligible_gpu_sets == (("GPU-0",),)

    def test_model_too_big_for_single_gpu_uses_multiple(self) -> None:
        # 40 GB weights + 15% overhead = 46 GB > 24 GB single GPU.
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000, 24000),
            weight_bytes=40_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
        )
        assert len(shapes) == 1
        assert shapes[0].gpu_count == 2
        assert shapes[0].tensor_parallel_size == 2

    def test_multiple_gpu_counts_are_candidates(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000, 24000),
            weight_bytes=8_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
        )
        gpu_counts = {shape.gpu_count for shape in shapes}
        assert gpu_counts == {1, 2}

    def test_max_model_len_limit_is_applied(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000),
            weight_bytes=8_000_000_000,
            max_model_len=8192,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
            max_model_len_limit=2048,
        )
        assert all(shape.max_model_len == 2048 for shape in shapes)

    def test_explicit_gpu_memory_utilization_wins(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000),
            weight_bytes=8_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
            gpu_memory_utilization=0.5,
        )
        assert all(shape.gpu_memory_utilization == 0.5 for shape in shapes)

    def test_automatic_utilization_is_sane(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000),
            weight_bytes=8_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
        )
        assert all(0.0 < shape.gpu_memory_utilization <= 1.0 for shape in shapes)

    def test_no_candidates_when_nothing_fits(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000),
            weight_bytes=200_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="half",
            quantization=None,
        )
        assert shapes == []

    def test_requested_limits_are_preserved(self) -> None:
        shapes = build_candidate_shapes(
            inventory=make_inventory(24000),
            weight_bytes=8_000_000_000,
            max_model_len=4096,
            reserved_vram_mib=2048,
            dtype="bfloat16",
            quantization="awq",
            max_num_seqs=64,
            max_num_batched_tokens=4096,
        )
        assert all(shape.dtype == "bfloat16" for shape in shapes)
        assert all(shape.quantization == "awq" for shape in shapes)
        assert all(shape.max_num_seqs == 64 for shape in shapes)
        assert all(shape.max_num_batched_tokens == 4096 for shape in shapes)


class TestValidationLogPath:
    def test_readable_sortable_name(self) -> None:
        path = validation_log_path(
            log_dir=Path("/tmp/logs"),
            nickname="My Model!",
            engine="vllm",
            tensor_parallel_size=2,
            gpu_indices=(0, 1),
        )
        assert path.parent == Path("/tmp/logs")
        assert path.name.startswith("validation-my-model-vllm-tp2-gpus0-1-")
        assert path.name.endswith(".log")

    def test_safe_nickname_sanitized(self) -> None:
        path = validation_log_path(
            log_dir=Path("/tmp/logs"),
            nickname="../etc/passwd",
            engine="vllm",
            tensor_parallel_size=1,
            gpu_indices=(0,),
        )
        assert ".." not in path.name
        assert "/" not in path.name
        assert path.name.startswith("validation-etc-passwd-vllm")

    def test_unique_suffix_prevents_collision(self) -> None:
        first = validation_log_path(
            log_dir=Path("/tmp/logs"),
            nickname="model",
            engine="vllm",
            tensor_parallel_size=1,
            gpu_indices=(0,),
        )
        second = validation_log_path(
            log_dir=Path("/tmp/logs"),
            nickname="model",
            engine="vllm",
            tensor_parallel_size=1,
            gpu_indices=(0,),
        )
        assert first != second

    def test_empty_gpu_indices_label(self) -> None:
        path = validation_log_path(
            log_dir=Path("/tmp/logs"),
            nickname="model",
            engine="vllm",
            tensor_parallel_size=1,
            gpu_indices=(),
        )
        assert "gpusunknown" in path.name
