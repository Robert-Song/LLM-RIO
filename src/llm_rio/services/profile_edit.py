from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llm_rio.api.schemas import ProfileEditRequest
from llm_rio.domain import Engine, PlacementProfile
from llm_rio.errors import RioError
from llm_rio.profiles import (
    StoredProfile,
    invalidate_profile_measurements,
    launch_configuration_changed,
    profile_to_dict,
)


def _profile_payload(record: StoredProfile) -> dict[str, object]:
    payload: dict[str, object] = profile_to_dict(record.profile)
    payload["active"] = record.active
    return payload


def _gguf_files(model: dict[str, object]) -> list[str]:
    artifact_path = model.get("artifact_path")
    if not artifact_path:
        return []
    root = Path(str(artifact_path))
    if root.is_file() and root.suffix.lower() == ".gguf":
        return [root.name]
    if not root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".gguf"
    )


def _resolve_gguf_file(model: dict[str, object], relative_path: str) -> Path:
    artifact_path = model.get("artifact_path")
    if not artifact_path:
        raise RioError(
            "model_artifact_missing",
            "The model artifact is not available on this machine",
            status_code=409,
        )
    root = Path(str(artifact_path)).resolve()
    if root.is_file():
        if relative_path in {root.name, str(root)}:
            return root
        raise RioError(
            "invalid_gguf_file", "Re-register to select a different local artifact", status_code=422
        )
    candidate = (root / relative_path).resolve()
    if candidate.suffix.lower() != ".gguf" or not candidate.is_relative_to(root):
        raise RioError(
            "invalid_gguf_file",
            "GGUF files must be paths inside this model's downloaded artifact",
            status_code=422,
        )
    if not candidate.is_file():
        raise RioError(
            "gguf_file_not_found",
            f"GGUF file '{relative_path}' was not found in this model artifact",
            status_code=404,
            details={"available_gguf_files": _gguf_files(model)},
        )
    return candidate


def _resize_per_gpu_measurement(values: tuple[int, ...], gpu_count: int) -> tuple[int, ...]:
    """Keep stored per-GPU diagnostics structurally valid after an admin TP override."""
    if not values:
        return (0,) * gpu_count
    return tuple(values[min(index, len(values) - 1)] for index in range(gpu_count))


def _apply_profile_edit(
    *,
    profile: PlacementProfile,
    model: dict[str, object],
    request: ProfileEditRequest,
    managed_gpu_count: int,
    eligible_gpu_sets: tuple[tuple[str, ...], ...],
    llama_cpp_enabled: bool,
) -> PlacementProfile:
    fields = request.model_fields_set
    updated = profile
    if "tensor_parallel_size" in fields:
        target_gpu_count = request.tensor_parallel_size
        if (
            target_gpu_count is None
            or target_gpu_count > managed_gpu_count
            or not eligible_gpu_sets
        ):
            raise RioError(
                "invalid_tensor_parallel_size",
                f"Tensor parallelism must be between 1 and {managed_gpu_count} on this machine",
                status_code=422,
            )
        updated = replace(
            updated,
            gpu_count=target_gpu_count,
            tensor_parallel_size=target_gpu_count,
            eligible_gpu_sets=(
                eligible_gpu_sets
                if target_gpu_count != profile.tensor_parallel_size
                else profile.eligible_gpu_sets
            ),
            idle_vram_mib_per_gpu=_resize_per_gpu_measurement(
                updated.idle_vram_mib_per_gpu, target_gpu_count
            ),
            peak_vram_mib_per_gpu=_resize_per_gpu_measurement(
                updated.peak_vram_mib_per_gpu, target_gpu_count
            ),
            gpu_headroom_mib_per_gpu=_resize_per_gpu_measurement(
                updated.gpu_headroom_mib_per_gpu, target_gpu_count
            ),
        )
    if "max_model_len" in fields:
        if request.max_model_len is None:
            raise RioError("invalid_max_model_len", "max_model_len cannot be null", status_code=422)
        updated = replace(updated, max_model_len=request.max_model_len)
    if "max_num_seqs" in fields:
        updated = replace(updated, max_num_seqs=request.max_num_seqs)
    if "max_num_batched_tokens" in fields:
        updated = replace(updated, max_num_batched_tokens=request.max_num_batched_tokens)
    if "gpu_memory_utilization" in fields:
        if request.gpu_memory_utilization is None:
            raise RioError(
                "invalid_gpu_memory_utilization",
                "gpu_memory_utilization cannot be null",
                status_code=422,
            )
        updated = replace(updated, gpu_memory_utilization=request.gpu_memory_utilization)
    if "engine" in fields:
        updated = replace(updated, engine=request.engine or updated.engine)

    launch_args = dict(updated.launch_args)
    if "gguf_file" in fields:
        if updated.engine is not Engine.LLAMA_CPP:
            raise RioError(
                "gguf_requires_llama_cpp",
                "Select llama.cpp before choosing a GGUF model file",
                status_code=422,
            )
        if request.gguf_file is None:
            raise RioError("invalid_gguf_file", "gguf_file cannot be null", status_code=422)
        launch_args["model"] = str(_resolve_gguf_file(model, request.gguf_file))
    if updated.engine is Engine.LLAMA_CPP:
        if not llama_cpp_enabled:
            raise RioError(
                "llama_cpp_disabled",
                "Set engines.enable_llama_cpp = true before selecting llama.cpp manually",
                status_code=409,
            )
        if "model" not in launch_args:
            raise RioError(
                "gguf_file_required",
                "Choose a GGUF file when switching this profile to llama.cpp",
                status_code=422,
                details={"available_gguf_files": _gguf_files(model)},
            )
        if "n_gpu_layers" in fields:
            if request.n_gpu_layers is None:
                raise RioError(
                    "invalid_n_gpu_layers", "n_gpu_layers cannot be null", status_code=422
                )
            launch_args["n_gpu_layers"] = request.n_gpu_layers
        else:
            launch_args.setdefault("n_gpu_layers", 99)
        updated = replace(
            updated,
            dtype="gguf",
            quantization=updated.quantization or "gguf",
            launch_args=launch_args,
        )
    else:
        if "gguf_file" in fields or "n_gpu_layers" in fields:
            raise RioError(
                "llama_cpp_option_requires_llama_cpp",
                "GGUF and n_gpu_layers are llama.cpp-only settings",
                status_code=422,
            )
        for key in ("model", "n_gpu_layers", "tensor_split", "split_mode"):
            launch_args.pop(key, None)
        if profile.engine is Engine.LLAMA_CPP:
            updated = replace(
                updated,
                dtype="auto",
                quantization=None,
                launch_args=launch_args,
            )
        else:
            updated = replace(updated, launch_args=launch_args)
    return (
        invalidate_profile_measurements(updated)
        if launch_configuration_changed(profile, updated)
        else updated
    )
