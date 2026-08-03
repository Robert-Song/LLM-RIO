from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from llm_rio.config import Settings
from llm_rio.domain import Engine, MachineInventory, PlacementProfile
from llm_rio.inventory import gpu_environment
from llm_rio.runtime import ResidencyScheduler
from llm_rio.validation import CandidateShape, ProfileValidator, ValidationError, ValidationPreempted


async def validate_llama_cpp(
    *,
    settings: Settings,
    inventory: MachineInventory,
    scheduler: ResidencyScheduler,
    probes: ProfileValidator,
    model_id: str,
    model_revision: str,
    gguf_path: Path,
    nickname: str,
    candidate: CandidateShape,
) -> list[PlacementProfile]:
    """Validate the pinned llama.cpp fallback against the same streaming contract."""
    profiles: list[PlacementProfile] = []
    failures: list[ValidationError] = []
    for gpu_set in candidate.eligible_gpu_sets:
        while not await scheduler.acquire_validation_gpus(gpu_set):
            await asyncio.sleep(5.0)
        try:
            profiles.append(
                await _probe_llama_cpp(
                    settings=settings,
                    inventory=inventory,
                    probes=probes,
                    model_id=model_id,
                    model_revision=model_revision,
                    gguf_path=gguf_path,
                    nickname=nickname,
                    candidate=candidate,
                    gpu_set=gpu_set,
                )
            )
        except ValidationPreempted:
            raise
        except ValidationError as exc:
            failures.append(exc)
        finally:
            await scheduler.release_validation_gpus(gpu_set)
    if not profiles and failures:
        raise failures[-1]
    return profiles


async def _probe_llama_cpp(
    *,
    settings: Settings,
    inventory: MachineInventory,
    probes: ProfileValidator,
    model_id: str,
    model_revision: str,
    gguf_path: Path,
    nickname: str,
    candidate: CandidateShape,
    gpu_set: tuple[str, ...],
) -> PlacementProfile:
    port = settings.worker_port_end + 2
    api_key = uuid.uuid4().hex
    command = [
        settings.engines.llama_cpp_executable,
        "--model",
        str(gguf_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(candidate.max_model_len),
        "--parallel",
        str(candidate.max_num_seqs),
        "--n-gpu-layers",
        "999",
        "--api-key",
        api_key,
    ]
    validation_id = str(uuid.uuid4())
    log_path = settings.log_dir / f"validation-llama-{validation_id}.log"
    started = time.monotonic()
    with log_path.open("ab", buffering=0) as log_handle:
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log_handle,
                stderr=asyncio.subprocess.STDOUT,
                env=gpu_environment(gpu_set, settings.engines.environment),
                start_new_session=True,
            )
        except OSError as exc:
            raise ValidationError(
                "llama_cpp_launch", str(exc), {"log_path": str(log_path)}
            ) from exc
        try:
            await probes._wait_for_health(process, port, api_key)
            load_seconds = time.monotonic() - started
            idle_memory = probes._used_vram(gpu_set)
            throughput, peak_memory = await probes._generation_contract(
                process=process,
                port=port,
                api_key=api_key,
                nickname=nickname,
                gpu_set=gpu_set,
            )
        except BaseException as exc:
            await probes._terminate(process)
            if isinstance(exc, ValidationError):
                exc.details.setdefault("log_path", str(log_path))
            raise
        await probes._terminate(process)
    engine_version = await _llama_cpp_version(settings.engines.llama_cpp_executable)
    return PlacementProfile(
        id=str(uuid.uuid4()),
        model_id=model_id,
        model_revision=model_revision,
        engine=Engine.LLAMA_CPP,
        engine_version=engine_version,
        machine_fingerprint=inventory.fingerprint,
        gpu_count=candidate.gpu_count,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        eligible_gpu_sets=(gpu_set,),
        dtype="gguf",
        quantization=candidate.quantization or "gguf",
        max_model_len=candidate.max_model_len,
        max_num_seqs=candidate.max_num_seqs,
        max_num_batched_tokens=candidate.max_num_batched_tokens,
        predicted_tokens_per_second=throughput,
        load_and_warmup_seconds=load_seconds,
        idle_vram_mib_per_gpu=idle_memory,
        peak_vram_mib_per_gpu=peak_memory,
        gpu_headroom_mib_per_gpu=tuple(settings.reserved_vram_mib for _ in gpu_set),
        capabilities=frozenset({"chat", "streaming"}),
        launch_args={"model": str(gguf_path), "n_gpu_layers": 999},
    )


async def _llama_cpp_version(executable: str) -> str:
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        output, _ = await asyncio.wait_for(process.communicate(), timeout=10.0)
    except (OSError, TimeoutError):
        return "version-unavailable"
    first_line = output.decode(errors="replace").splitlines()
    return first_line[0][:200] if first_line else "version-unavailable"
