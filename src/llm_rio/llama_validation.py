from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from llm_rio.config import Settings
from llm_rio.domain import (
    CURRENT_VRAM_MEASUREMENT_VERSION,
    Engine,
    MachineInventory,
    PlacementProfile,
)
from llm_rio.engines.identity import launch_binding
from llm_rio.engines.launch import adapter
from llm_rio.process_cleanup import TeardownError
from llm_rio.runtime import ResidencyScheduler
from llm_rio.validation import (
    CandidateShape,
    ProfileValidator,
    ValidationError,
    ValidationPreempted,
    _VramSampler,
    validation_log_path,
)


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
        while not await scheduler.acquire_validation_gpus(gpu_set):  # noqa: ASYNC110
            await asyncio.sleep(5.0)
        teardown_verified = True
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
        except TeardownError:
            teardown_verified = False
            raise
        except ValidationPreempted:
            raise
        except ValidationError as exc:
            failures.append(exc)
        finally:
            if teardown_verified:
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
    port = await probes._reserve_validation_port()
    teardown_verified = True
    try:
        api_key = uuid.uuid4().hex
        spec = adapter(Engine.LLAMA_CPP).launch(
            settings=settings,
            shape=candidate,
            artifact=gguf_path,
            nickname=nickname,
            gpu_uuids=gpu_set,
            port=port,
            api_key=api_key,
        )
        command = spec.command
        gpu_indices = tuple(device.index for device in inventory.gpus if device.uuid in gpu_set)
        log_path = validation_log_path(
            log_dir=settings.log_dir,
            nickname=nickname,
            engine="llama-cpp",
            tensor_parallel_size=1,
            gpu_indices=gpu_indices,
        )
        started = time.monotonic()
        sampler = _VramSampler(probes._used_vram, gpu_set)
        sampler.start()
        with log_path.open("ab", buffering=0) as log_handle:
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=log_handle,
                    stderr=asyncio.subprocess.STDOUT,
                    env=spec.environment,
                    start_new_session=True,
                )
            except OSError as exc:
                await sampler.stop()
                raise ValidationError(
                    "llama_cpp_launch", str(exc), {"log_path": str(log_path)}
                ) from exc
            try:
                await probes._wait_for_health(process, port, api_key)
                load_seconds = time.monotonic() - started
                idle_memory = sampler.sample_now()
                throughput = await probes._generation_contract(
                    process=process,
                    port=port,
                    api_key=api_key,
                    nickname=nickname,
                )
                peak_memory = sampler.peak()
                await sampler.stop()
                probes._validate_vram_measurements(
                    gpu_set=gpu_set,
                    peak_memory=peak_memory,
                    baseline_drop_mib=sampler.baseline_drop_mib,
                )
            except BaseException as exc:
                await sampler.stop()
                await probes._terminate(process, gpu_uuids=gpu_set)
                if isinstance(exc, ValidationError):
                    exc.details.setdefault("log_path", str(log_path))
                raise
            await probes._terminate(process, gpu_uuids=gpu_set)
        engine_version = await _llama_cpp_version(settings.engines.llama_cpp_executable)
        return PlacementProfile(
            id=str(uuid.uuid4()),
            model_id=model_id,
            model_revision=model_revision,
            engine=Engine.LLAMA_CPP,
            serving_mode="queue",
            engine_version=engine_version,
            launch_binding=launch_binding(settings, candidate, Engine.LLAMA_CPP),
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
            gpu_headroom_mib_per_gpu=(0,) * len(gpu_set),
            capabilities=frozenset({"chat", "streaming"}),
            launch_args={
                **candidate.launch_args,
                "model": str(gguf_path),
                "n_gpu_layers": candidate.launch_args.get("n_gpu_layers", 999),
                "enable_sleep_mode": False,
            },
            gpu_memory_utilization=candidate.gpu_memory_utilization,
            kv_cache_capacity_tokens=None,
            max_full_length_concurrency=None,
            vram_measurement_version=CURRENT_VRAM_MEASUREMENT_VERSION,
            vram_baseline_mib_per_gpu=sampler.baseline_mib,
        )
    except TeardownError:
        teardown_verified = False
        raise
    finally:
        if teardown_verified:
            await probes._release_validation_port(port)


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
