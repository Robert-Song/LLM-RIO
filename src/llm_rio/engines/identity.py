"""Bind measured profiles to effective launch settings and the installed engine."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from llm_rio.config import Settings
from llm_rio.domain import Engine
from llm_rio.engines.launch import LaunchShape


@lru_cache(maxsize=32)
def _binary_digest(path: str, size: int, modified_ns: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def launch_binding(settings: Settings, shape: LaunchShape, engine: Engine) -> str:
    executable = (
        settings.engines.vllm_executable
        if engine is Engine.VLLM
        else settings.engines.llama_cpp_executable
    )
    path = shutil.which(executable)
    binary: dict[str, Any] = {"configured": executable, "resolved": path}
    if path:
        stat = Path(path).stat()
        binary["sha256"] = _binary_digest(path, stat.st_size, stat.st_mtime_ns)
    if engine is Engine.VLLM:
        try:
            binary["distribution"] = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            binary["distribution"] = None
    # These variables change kernels, loading or memory behavior. Capture configured
    # overrides too; neither API credentials nor unrelated shell state are persisted.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("VLLM_", "CUDA_", "NCCL_", "OMP_", "KVCACHED_"))
        or key in {"PYTHONPATH", "LD_LIBRARY_PATH"}
    }
    environment.update(settings.engines.environment)
    for key in ("CUDA_VISIBLE_DEVICES", "VLLM_API_KEY", "VLLM_SERVER_DEV_MODE"):
        environment.pop(key, None)
    if settings.serving_mode.value != "kv-cached":
        environment = {k: v for k, v in environment.items() if not k.startswith("KVCACHED_")}
    args = dict(shape.launch_args)
    args.pop("model", None)  # Artifact identity is recorded independently.
    args["enable_sleep_mode"] = settings.ram_weight_cache_enabled
    if engine is Engine.LLAMA_CPP:
        args.pop("enable_sleep_mode", None)
        args.setdefault("n_gpu_layers", 999)
    payload = {
        "format": 1,
        "engine": engine.value,
        "binary": binary,
        "mode": settings.serving_mode.value,
        "environment": environment,
        "shape": {
            name: getattr(shape, name)
            for name in (
                "tensor_parallel_size",
                "dtype",
                "quantization",
                "max_model_len",
                "max_num_seqs",
                "max_num_batched_tokens",
                "gpu_memory_utilization",
            )
        },
        "pipeline_parallel_size": getattr(shape, "pipeline_parallel_size", 1),
        "arguments": args,
    }
    if engine is Engine.LLAMA_CPP:
        payload["shape"] = {
            name: getattr(shape, name) for name in ("max_model_len", "max_num_seqs")
        }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
