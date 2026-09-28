"""Bind measured profiles to effective launch settings and the installed engine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
from functools import lru_cache
from pathlib import Path

from llm_rio.config import Settings
from llm_rio.domain import Engine
from llm_rio.engines.launch import LaunchShape


@lru_cache(maxsize=32)
def _binary_digest(path: str, size: int, modified_ns: int) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _capture(command: list[str], environment: dict[str, str]) -> str:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError("Engine identity probe timed out") from None
    if process.returncode:
        raise RuntimeError("Engine identity probe failed")
    return stdout.strip() or stderr.strip()


@lru_cache(maxsize=32)
def _distribution_path(
    interpreter: str,
    wrapper_digest: str,
    environment: tuple[tuple[str, str], ...],
) -> str:
    # Query the engine interpreter, never the router's installed distributions.
    return _capture(
        [
            interpreter,
            "-c",
            "import importlib.metadata as m; print(m.distribution('vllm')._path)",
        ],
        dict(environment),
    )


def engine_identity(settings: Settings, engine: Engine) -> dict[str, str]:
    executable = (
        settings.engines.vllm_executable
        if engine is Engine.VLLM
        else settings.engines.llama_cpp_executable
    )
    environment = {**os.environ, **settings.engines.environment}
    if settings.serving_mode.value != "kv-cached":
        for key in list(environment):
            if key.startswith("KVCACHED_") or key == "LLM_RIO_KVCACHED_VLLM026_SHIM":
                environment.pop(key)
        environment.update(ENABLE_KVCACHED="false", KVCACHED_AUTOPATCH="0")
    path = shutil.which(executable, path=environment.get("PATH"))
    if not path:
        raise RuntimeError(f"Cannot establish engine identity: executable {executable!r} not found")
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    digest = _binary_digest(str(resolved), stat.st_size, stat.st_mtime_ns)
    result = {"configured": executable, "resolved": str(resolved), "sha256": digest}
    header = resolved.read_bytes()[:8192] if stat.st_size < 65536 else b""
    if engine is Engine.VLLM and b"from vllm.entrypoints.cli.main import main" in header:
        first = header.splitlines()[0].decode()
        interpreter = first.removeprefix("#!").strip()
        if not Path(interpreter).is_file():
            raise RuntimeError("Use a vLLM console script with an absolute interpreter shebang")
        distribution = Path(
            _distribution_path(
                interpreter,
                digest,
                tuple(sorted(environment.items())),
            )
        )
        metadata = (distribution / "METADATA").read_bytes()
        record = (distribution / "RECORD").read_bytes()
        versions = [
            line[9:] for line in metadata.decode().splitlines() if line.startswith("Version: ")
        ]
        if len(versions) != 1:
            raise RuntimeError("Engine distribution has no unambiguous version")
        result.update(
            version=versions[0],
            distribution=str(distribution),
            distribution_sha256=hashlib.sha256(metadata + record).hexdigest(),
            interpreter=str(Path(interpreter).resolve()),
        )
    else:
        version = _capture([str(resolved), "--version"], environment)
        if not version:
            raise RuntimeError("Engine --version returned no identity")
        result["version"] = version
    return result


def launch_binding(settings: Settings, shape: LaunchShape, engine: Engine) -> str:
    binary = engine_identity(settings, engine)
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
        "format": 2,
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
