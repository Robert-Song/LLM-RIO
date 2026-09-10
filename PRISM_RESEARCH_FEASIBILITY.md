# Full Prism feasibility record

## Decision

LLM-RIO now implements the requested operational Prism behavior while retaining
vLLM as its only inference engine. kvcached supplies elastic physical KV pages;
the residency scheduler dynamically sleeps idle blockers, wakes demanded models,
replicates hot models, and bounds both GPU-resident and RAM-cached processes.
vLLM level-1 sleep preserves each initialized engine and its model weights in
host RAM, and the vLLM 0.26 adapter retains that immutable host backup after wake
so later offloads do not recopy the weights from GPU.

This is not a source transplant of
[Multi-LLM/prism-research](https://github.com/Multi-LLM/prism-research). That
artifact is an old SGLang 0.3.4.post2 fork, pins vLLM 0.6.3.post1, and changes
SGLang's model runner, request handler, GPU scheduler, global scheduler, and
server lifecycle. It does not support the current vLLM 0.26, CUDA 13, Blackwell,
hybrid-attention, and ModelOpt/NVFP4 stack used here.

The implementation backend therefore differs: the paper uses generic SGLang
workers and shared CPU model objects, while LLM-RIO keeps a bounded
model-specific vLLM process per cached placement. Both avoid checkpoint I/O and
engine construction on a cache hit. The tradeoff is higher host-RAM and process
overhead in exchange for current-model compatibility and preservation of the
existing OpenAI-serving path.

## What full Prism adds beyond kvcached

The upstream design has four cooperating pieces:

1. **Preinitialized elastic workers.** A generic SGLang worker can change its
   active model configuration without paying process and engine initialization
   on every switch.
2. **CPU-shared model objects.** Models are loaded into host memory once and
   shared with worker processes.
3. **Parallel CPU-to-GPU transfer.** Empty GPU model objects are populated from
   shared CPU tensors using multiple threads, CUDA streams, and optionally
   other GPUs as copy brokers.
4. **Two-level scheduling.** A global scheduler selects placement and migration;
   per-GPU schedulers activate, resize, preempt, and deactivate models.

kvcached supplies elastic KV memory to this design. It does not implement the
first three pieces. The relevant upstream entry points are:

- [launch_multi_model_server.py](https://github.com/Multi-LLM/prism-research/blob/main/python/sglang/launch_multi_model_server.py)
- [worker_pool_model_runner.py](https://github.com/Multi-LLM/prism-research/blob/main/python/sglang/srt/model_executor/worker_pool_model_runner.py)
- [model_sevice.py](https://github.com/Multi-LLM/prism-research/blob/main/python/sglang/multi_model/model_sevice.py)
- [simple_global.py](https://github.com/Multi-LLM/prism-research/blob/main/python/sglang/multi_model/scheduling/policy/simple_global.py)
- [pyproject.toml](https://github.com/Multi-LLM/prism-research/blob/main/python/pyproject.toml)

## Hardware measurements

These automatic-validation measurements came from the two RTX PRO 6000
Blackwell GPUs on 2026-09-02. Cold time includes process and model construction,
compilation, KV profiling, graph capture, and API readiness. First offload
populates the persistent host backup; wake is the user-visible RAM-to-GPU model
activation.

| Model and placement | Cold | First offload/cache fill | Wake | Sleeping VRAM |
|---|---:|---:|---:|---:|
| Qwen3-8B-FP8, TP=1 | 112-139 s | 4.94-8.15 s | 0.20-0.22 s | 2,325 MiB |
| Qwen3-8B-FP8, TP=2 | 134 s | 2.70 s | 0.16 s | 3,087 / 2,531 MiB |
| Qwen3.8 27B NVFP4, TP=1 | 182-187 s | 12.70-13.13 s | 0.50-0.51 s | 5,057 MiB |
| Qwen3.8 27B NVFP4, TP=2 | 216 s | 6.73 s | 0.36 s | 4,409 / 3,853 MiB |
| Gemma 4 31B NVFP4, TP=1 | 161-162 s | 16.67-17.49 s | 0.70-0.73 s | 2,975 MiB |
| Gemma 4 31B NVFP4, TP=2 | 193 s | 9.20 s | 0.55 s | 3,667 / 3,109 MiB |
| Laguna-S-2.1 NVFP4, TP=2 | 140 s | 25.78 s | 1.54 s | 6,863 / 6,307 MiB |

The large gap between cold and wake is the result the RAM weight cache is
designed to preserve. Startup preloads and one-time post-registration warming
pay engine construction and the first cache fill ahead of presentation-time
requests. The routed Qwen3-8B cache-hit smoke request completed in 0.887 seconds
end to end while retaining the same PID.
A routed Qwen3.8 request later woke in 0.522 seconds and completed in 1.413
seconds end to end; because its host backup survived wake, the following idle
offload took 0.352 seconds instead of repeating the 12.814-second cache fill.

## Why a small vLLM patch is insufficient

In vLLM 0.26, an EngineCore is constructed for one model configuration.
Scheduler capacity, KV cache groups, attention metadata, tokenizer/parser
behavior, compilation artifacts, CUDA graphs, tensor-parallel state, and the
model runner all depend on that configuration. Replacing only the model object
would leave the rest of the engine describing the old model.

A vLLM-only port would need, at minimum:

- a reusable worker-pool protocol and activation RPC;
- model-specific CPU shared-memory weight preparation for every TP rank;
- an empty-device model and parallel copy path for each quantization;
- complete scheduler and KV-layout reconfiguration on activation;
- safe invalidation or recapture of compiled graphs and communication buffers;
- request admission, preemption, rollback, migration, and failure recovery;
- architecture qualification for current and future models;
- a versioned vLLM compatibility surface rather than private monkey patches.

That is a serving-engine fork. It is larger and riskier than LLM-RIO's control
plane and cannot be presented as a clean implementation of the upstream
artifact.

LLM-RIO deliberately uses vLLM sleep mode as the model-specific worker-cache
mechanism. Combined with kvcached KV elasticity, admission-aware residency
scheduling, persistent CPU weight backups, and bounded cache eviction, it is
the operational Prism backend used by this service.

## Compatibility boundary

The upstream artifact's published environment uses SGLang 0.3.4.post2, vLLM
0.6.3.post1, the prism/shm kvcached branch, a CUDA 12.1 SGLang container,
Redis, and examples dominated by Llama 3.2 1B/3B instances.

The current LLM-RIO environment uses vLLM 0.26.0, Torch 2.11.0+cu130, CUDA
13.0, RTX PRO 6000 Blackwell GPUs, current Qwen/Gemma/Laguna architectures,
NVFP4 quantization, no SGLang engine, and no Redis dependency.

The research code can still be evaluated in an isolated container, but a pass
there would not validate the production models or satisfy the requirement that
vLLM remain the only inference engine.

## Live Prism demonstration

### Startup warming

The demo configuration prepares one Gemma worker, one TP=2 Laguna worker, and
two distinct Qwen TP=1 workers. Repeating a nickname means “warm another
validated placement”; it does not create a TP=2 substitute:

~~~toml
prism_preload_models = [
  "qwen3.8-27b-nvfp4",
  "qwen3.8-27b-nvfp4",
  "gemma-4-31b-it-nvfp4",
  "laguna-s-2.1-nvfp4",
]
prism_weight_cache_mode = "ram"
~~~

Start the isolated service and dashboard, then wait until each desired copy is
either `READY` or `SLEEPING`:

~~~bash
./llmctl serve
./llmctl
~~~

Cold construction and the first GPU-to-RAM cache fill happen during this
startup phase. Once idle, the processes stay alive with `weight_storage =
host_ram`. The dashboard and `GET /admin/status` show the worker state,
placement, TP size, cache tier, accepted requests, and last wake/offload time.

### Automatic model addition

The presentation-time sample is the official Qwen FP8 checkpoint:

~~~bash
./llmctl models add qwen3-8b-q8 Qwen/Qwen3-8B-FP8 --wait
~~~

The same operation is available from the TUI. Registration resolves an
immutable revision, checks disk, downloads to the configured model store,
inspects the snapshot, derives candidate placements, and validates every
profile with baseline generation, sleep, VRAM reclamation, wake, and
post-wake generation. Only then does the model become `AVAILABLE`. The
scheduler requests a one-time warm so a later chat uses RAM rather than
rebuilding the engine.

The 2026-09-02 run resolved revision
`220b46e3b2180893580a4454f21f22d3ebb187d3` and produced three passing
kvcached profiles. TP=1 cold construction was 112-139 seconds, first cache fill
was 4.94-8.15 seconds, and wake was 0.20-0.22 seconds. TP=2 cold construction
was 134 seconds, cache fill 2.70 seconds, and wake 0.16 seconds.

### Repeated switch and persistent host backup

A sleeping Qwen3.8 TP=1 worker was called through LLM-RIO, not its private vLLM
port. It woke in 0.522 seconds and returned HTTP 200 in 1.413 seconds end to
end. The worker ID and process stayed unchanged. Its initial cache fill had
taken 12.814 seconds; after the request, the next idle offload took only 0.352
seconds because the immutable CPU tensors survived wake.

Laguna-S-2.1-NVFP4 passed TP=2 Prism validation on the same host: 140.16 seconds
cold, 25.78 seconds for the first cache fill, 1.542 seconds to wake, and
6,863/6,307 MiB residual sleeping VRAM. It replaces DeepSeek in this host's
switching set. Together with Qwen3.8 (about 0.5-second wake) and Gemma 4
(about 0.6-0.7-second wake), it keeps model activation below the 10-second demo
target after startup warming.

The final public-API round robin produced:

| Model | First call | Second call |
|---|---:|---:|
| Qwen | 5.63 s | 8.61 s |
| Gemma | 2.33 s | 1.10 s |
| Laguna | 7.30 s | 8.37 s |

All six calls returned HTTP 200 and stayed below ten seconds end to end. Final
worker status recorded wake times of 0.648 seconds for Qwen, 0.707 seconds for
Gemma, and 1.661 seconds for Laguna; repeated offloads took 0.146-0.218
seconds.

### Same-model burst

The first uncached burst kept the serving Qwen worker at 100% GPU utilization
while the scheduler initialized another TP=1 placement on the other GPU; the
active worker was never offloaded. After both placements were RAM-warm, a
second run woke them in 0.533 and 0.610 seconds.

A two-wave, 40-request homepage workload completed without errors and returned
17,425 completion tokens (18,665 total). The follow-up wave was assigned 8/8
across the two TP=1 workers. The planner now refuses to add a redundant TP=2
copy after both one-GPU placements exist.

For the three-model mixed demonstration:

~~~bash
export LLMRIO_API_KEY='an-active-admin-key'
python3 three_user_mixed_demo.py
~~~

The script defaults to `http://127.0.0.1:3737/v1`, accepts both `READY` and
`SLEEPING` warmed workers, provisions temporary disjoint user keys, prints
live GPU/dashboard samples, and removes the keys it created.

### DeepSeek-V4-Flash-0731 boundary

The exact DeepSeek checkpoint downloaded successfully but is not enabled in the
demo preload set. vLLM 0.26 on SM120 selects
`FLASHINFER_MLA_SPARSE_DSV4`, while the installed FlashInfer lacks its sparse
MLA decode API. Forcing `FLASHMLA_SPARSE_DSV4` is not a safe workaround:
upstream reports a separate warmup failure for this same checkpoint and
software line. Keep the catalog item in `NEEDS_ADMIN_REVIEW` and use Laguna
until the upstream SM120 correctness and backend issues are resolved.

- [vLLM SM120 FlashInfer issue #50720](https://github.com/vllm-project/vllm/issues/50720)
- [vLLM FlashMLA fallback issue #50774](https://github.com/vllm-project/vllm/issues/50774)
- [Official vLLM DeepSeek-V4 recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml)

## Research worker-pool revisit criteria

The current backend satisfies LLM-RIO's operational Prism contract without
adding another inference engine. Revisit a source-level generic worker pool if
vLLM exposes a supported atomic multi-model EngineCore API, Prism publishes a
maintained current-vLLM backend, or the project explicitly accepts and budgets
a serving-engine fork. Any replacement must pass the same cold/warm,
correctness, streaming, TP, concurrency, memory reclamation, rollback, orphan
cleanup, and repeated-switching gates on production hardware.
