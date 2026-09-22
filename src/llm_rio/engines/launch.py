from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_rio.config import Settings
from llm_rio.domain import Engine
from llm_rio.engine_args import extra_engine_arguments
from llm_rio.engine_runtime import add_kvcached_vllm_flags, detect_kvcached
from llm_rio.inventory import gpu_environment
from llm_rio.tool_support import detect_vllm_parser_configuration


class LaunchShape(Protocol):
    @property
    def tensor_parallel_size(self) -> int: ...
    @property
    def dtype(self) -> str: ...
    @property
    def quantization(self) -> str | None: ...
    @property
    def max_model_len(self) -> int: ...
    @property
    def max_num_seqs(self) -> int | None: ...
    @property
    def max_num_batched_tokens(self) -> int | None: ...
    @property
    def gpu_memory_utilization(self) -> float: ...
    @property
    def launch_args(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    command: tuple[str, ...]
    environment: dict[str, str]
    artifact: Path
    gpu_uuids: tuple[str, ...]
    mode: str
    engine: Engine


class EngineAdapter(Protocol):
    def launch(
        self,
        *,
        settings: Settings,
        shape: LaunchShape,
        artifact: Path,
        nickname: str,
        gpu_uuids: tuple[str, ...],
        port: int,
        api_key: str,
    ) -> LaunchSpec: ...


class VLLMAdapter:
    def launch(
        self,
        *,
        settings: Settings,
        shape: LaunchShape,
        artifact: Path,
        nickname: str,
        gpu_uuids: tuple[str, ...],
        port: int,
        api_key: str,
    ) -> LaunchSpec:
        command = [
            settings.engines.vllm_executable,
            "serve",
            str(artifact),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key",
            api_key,
            "--served-model-name",
            nickname,
            "--tensor-parallel-size",
            str(shape.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(getattr(shape, "pipeline_parallel_size", 1)),
            "--dtype",
            shape.dtype,
            "--max-model-len",
            str(shape.max_model_len),
            "--gpu-memory-utilization",
            str(shape.gpu_memory_utilization),
        ]
        for flag, value in (
            ("max-num-seqs", shape.max_num_seqs),
            ("max-num-batched-tokens", shape.max_num_batched_tokens),
            ("quantization", shape.quantization),
        ):
            if value is not None:
                command.extend([f"--{flag}", str(value)])
        parsers = detect_vllm_parser_configuration(artifact)
        if parsers.tool_parser:
            command.extend(["--enable-auto-tool-choice", "--tool-call-parser", parsers.tool_parser])
        if parsers.reasoning_parser:
            command.extend(["--reasoning-parser", parsers.reasoning_parser])
        args = dict(shape.launch_args)
        args["enable_sleep_mode"] = settings.ram_weight_cache_enabled
        command.extend(extra_engine_arguments(args, explicit_false=True))
        runtime = detect_kvcached(settings.effective_kvcached_mode)
        add_kvcached_vllm_flags(command, runtime)
        environment = gpu_environment(
            gpu_uuids, settings.engines.environment, executable=settings.engines.vllm_executable
        )
        # The native modes never inherit activation flags from an experimental shell.
        for name in list(environment):
            if name.startswith("KVCACHED_") or name == "LLM_RIO_KVCACHED_VLLM026_SHIM":
                environment.pop(name)
        environment.update(
            ENABLE_KVCACHED="false",
            KVCACHED_AUTOPATCH="0",
            VLLM_SERVER_DEV_MODE="1" if settings.ram_weight_cache_enabled else "0",
            VLLM_API_KEY=api_key,
        )
        environment.update(runtime.environment(pythonpath=environment.get("PYTHONPATH")))
        return LaunchSpec(
            tuple(command),
            environment,
            artifact,
            gpu_uuids,
            settings.serving_mode.value,
            Engine.VLLM,
        )


class LlamaCppAdapter:
    def launch(
        self,
        *,
        settings: Settings,
        shape: LaunchShape,
        artifact: Path,
        nickname: str,
        gpu_uuids: tuple[str, ...],
        port: int,
        api_key: str,
    ) -> LaunchSpec:
        if not settings.queue_mode_enabled or not settings.engines.enable_llama_cpp:
            raise ValueError("llama.cpp requires queue mode and engines.enable_llama_cpp=true")
        command = [
            settings.engines.llama_cpp_executable,
            "--model",
            str(artifact),
            "--alias",
            nickname,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(shape.max_model_len),
            "--api-key",
            api_key,
        ]
        if shape.max_num_seqs is not None:
            command.extend(["--parallel", str(shape.max_num_seqs)])
        args = {
            key: value
            for key, value in shape.launch_args.items()
            if key not in {"model", "enable_sleep_mode"}
        }
        args.setdefault("n_gpu_layers", 999)
        command.extend(extra_engine_arguments(args))
        environment = gpu_environment(
            gpu_uuids,
            settings.engines.environment,
            executable=settings.engines.llama_cpp_executable,
        )
        return LaunchSpec(
            tuple(command),
            environment,
            artifact,
            gpu_uuids,
            settings.serving_mode.value,
            Engine.LLAMA_CPP,
        )


def adapter(engine: Engine) -> EngineAdapter:
    return VLLMAdapter() if engine is Engine.VLLM else LlamaCppAdapter()
