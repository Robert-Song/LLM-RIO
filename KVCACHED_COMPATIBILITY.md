# kvcached compatibility and upgrade runbook

This runbook covers two jobs:

1. Removing LLM-RIO's temporary vLLM 0.26 packed-KV, legacy-runner, and
   shared-pool retry adapters after upstream kvcached provides equivalent support.
2. Qualifying a new model architecture, vLLM release, kvcached release, CUDA
   stack, or GPU architecture without putting the production service at risk.

It is intentionally fail-closed. A model loading once is not proof of
compatibility. The release gate includes output correctness, tensor
parallelism, concurrent residency, dynamic KV growth and reclamation, process
reclamation, and sustained concurrent load.

LLM-RIO continues to use vLLM as its only inference engine. Prism mode combines
kvcached's shared elastic KV pool with vLLM level-1 sleep and a local vLLM 0.26
adapter that retains immutable CPU weight backups across repeated wake/sleep cycles.

This is a behavior-compatible implementation of Prism's temporal weight-cache
semantics using separate, model-specific initialized vLLM workers. It does not
transplant the research artifact's generic SGLang worker pool or its shared model
object format. See `PRISM_RESEARCH_FEASIBILITY.md` for that implementation boundary
and the target-host measurements.
## Current qualified baseline

As of 2026-09-02, the qualified experimental tuple is:

- vLLM: `0.26.0`
- kvcached: revision
  `60cad949389af6bbf1d65c4eddf325113df5a9eb`
- Local adapter: `src/llm_rio/kvcached_vllm_compat.py`
- Layout: non-contiguous
- kvcached page size: 4 MiB
- Prefix-cache retention and page preallocation: disabled
- Tested target: `unsloth/Qwen3.8-27B-NVFP4`
- Tested placement shapes: TP=1 and TP=2
- Final TP=2 report:
  `diagnostics/prism-compat-20260826T223904Z/report.json`

Upstream's support table must be read as a starting point, not as local
certification. The pinned revision advertises vLLM testing through 0.24.0 and
lists sliding-window and hybrid support. Its hybrid guide specifically
discusses Qwen3.5/3.6 GDN and Gemma 4, but that does not prove compatibility
with vLLM 0.26.0, these exact checkpoints, their quantization, or this host.

References:

- [Pinned kvcached revision](https://github.com/ovg-project/kvcached/tree/60cad949389af6bbf1d65c4eddf325113df5a9eb)
- [Upstream model compatibility matrix](https://github.com/ovg-project/kvcached/issues/425)
- [Upstream hybrid-model example](https://github.com/ovg-project/kvcached/tree/60cad949389af6bbf1d65c4eddf325113df5a9eb/examples/08_hybrid_attention_models)
- [vLLM hybrid KV-cache design](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)
- [kvcached cross-version roadmap](https://github.com/ovg-project/kvcached/issues/273)

## Local model status

“Upstream family support” and “qualified on this host” are different states.
The latter requires a passing LLM-RIO hardware report for the exact model
revision, vLLM/kvcached tuple, quantization, GPU type, and TP shape.

| Catalog model | Snapshot architecture | KV structure | Current conclusion |
|---|---|---|---|
| `qwen3.8-27b-nvfp4` | `Qwen3_5ForConditionalGeneration` | 48 linear-attention plus 16 full-attention layers | TP=1/TP=2 generation, sleep, wake, and post-wake generation pass. TP=1 wakes in about 0.5 seconds; persistent-backup re-offload measured 0.352 seconds. Two cached TP=1 copies served the live burst. |
| `qwen3.6-27b-nvfp4` | `Qwen3_5ForConditionalGeneration` | 48 linear-attention plus 16 full-attention layers | TP=1 inference, streaming, long prefill, and continuous batch-8 pass with the packed-hybrid adapter. Forced tools require enough output budget for reasoning before the XML call. |
| `gemma-4-31b-it-nvfp4` | `Gemma4ForConditionalGeneration` | 50 sliding-window plus 10 full-attention layers | TP=1/TP=2 generation, sleep, wake, and post-wake generation pass; wake measured 0.55-0.73 seconds. |
| `laguna-s-2.1-nvfp4` | `LagunaForCausalLM` | 36 sliding-window plus 12 full-attention layers; TP=2 native profile | TP=2 full Prism validation passes: 140.16-second cold construction, 25.78-second first cache fill, 1.542-second wake, and 6,863/6,307 MiB sleeping VRAM. |
| Temporary Qwen3-8B and Qwen3-8B-FP8 samples | `Qwen3ForCausalLM` | Ordinary grouped-query attention | TP=1/TP=2 automatic registration, sleep/wake, and post-wake probes passed. Their profiles are removed and their catalog entries are disabled tombstones with artifact metadata cleared; hard deletion and external cache removal await explicit approval. |
| `deepseek-v4-flash-0731` | `DeepseekV4ForCausalLM` | Sparse hybrid MLA | Not qualified on SM120: the vLLM 0.26 default requires a FlashInfer sparse MLA API absent from the installed build, and the alternate FlashMLA backend has a separate upstream warmup failure. The catalog row remains `NEEDS_ADMIN_REVIEW`; it is not preloaded. |

The local adapter is not keyed to a model nickname. It activates when vLLM
0.26 supplies the four-dimensional symmetric packed K/V shape for ordinary
MHA/GQA or `HYBRID_LINEAR` allocation. Qwen 3.6 is expected to use the
hybrid path; Gemma and Laguna use the packed MHA path. Explicit-K/V and MLA
layouts still pass through unchanged. LLM-RIO's kvcached mode also forces
`VLLM_USE_V2_MODEL_RUNNER=0` on vLLM 0.26 because the pinned kvcached
revision patches only the legacy runner class.

The adapter also normalizes an allocator race that is independent of model
architecture. A colocated engine can consume the last shared physical page
between vLLM's free-block check and `ElasticBlockPool.get_new_blocks()`.
The pinned revision raises an uncaught `ValueError` in that early path and
kills EngineCore; LLM-RIO translates it to kvcached's existing
`KVCachePoolExhausted` scheduling-retry signal.

The previous native vLLM placement profiles do not count as kvcached
qualification. They were created without `memory_backend = "kvcached"`.

Consequently, do **not** remove the adapter after testing only Qwen 3.8.
Qwen 3.6, Gemma, and Laguna must all pass without it at their production TP
shapes. Familiar architecture or cache-group names are not proof.

## What is temporary and what should remain

### Candidate for removal after upstream support lands

- The immutable experimental kvcached revision in `pyproject.toml`.
- `KVCACHED_COMPAT_REVISION` and the vLLM 0.26-only compatibility rule in
  `src/llm_rio/prism.py`.
- `src/llm_rio/kvcached_vllm_compat.py`.
- `src/llm_rio/kvcached_bootstrap/`.
- `LLM_RIO_KVCACHED_VLLM026_SHIM` and the bootstrap `PYTHONPATH` injection.
- The vLLM 0.26 `VLLM_USE_V2_MODEL_RUNNER=0` override.
- The compatibility-runner assertion requiring an
  `LLM-RIO packed ... KV shim active` allocation marker.
- README wording that describes the local packed-KV adapter.

The local compatibility layer handles five vLLM 0.26 mismatches:

1. vLLM may select its V2 model runner, while the pinned kvcached revision
   patches only the legacy runner import path. LLM-RIO forces the legacy
   runner so every worker calls `init_kvcached(is_worker=True)`.
2. vLLM passes symmetric MHA/GQA and hybrid K/V tensors as
   `(blocks, heads, tokens, 2 * head_size)`, while the pinned allocator
   expects an explicit K/V dimension. The adapter expands the allocation
   shape, manages one combined physical K/V page, and reconstructs the exact
   NHD-strided view expected by vLLM.
3. The shared physical KV free count is a snapshot, but the pinned
   `get_new_blocks()` treats a changed count as a local invariant violation.
   LLM-RIO converts only its exact `Cannot get N free blocks from the pool`
   error to the scheduler's normal capacity-miss path. Unrelated `ValueError`
   exceptions remain fatal.
4. A capacity race after one hybrid cache group has allocated pages can leave
   partial request state behind. The adapter rolls back those group ledgers
   before vLLM retries the waiting request.
5. vLLM 0.26's FP8 wake initializer assumes every top-level KV-cache entry is
   one tensor, but hybrid TP placements can expose a nested tensor list. The
   adapter zeros that tree safely. It also retains the immutable CPU weight
   tensors after wake and reuses them on later sleeps; upstream otherwise drops
   the backup and recopies all weights from GPU on every offload.



### Re-test separately; do not remove with the adapter by default

- `KVCACHED_PAGE_SIZE_MB=4`: selected because the target hybrid block is
  larger than 2 MiB. An upstream update may calculate this automatically, but
  that must be demonstrated for every cache group and TP rank.
- `KVCACHED_CONTIGUOUS_LAYOUT=false`: the pinned upstream guide recommends
  the non-contiguous path for linear-attention hybrids. Compare correctness
  and performance before changing it.
- `NCCL_NET_PLUGIN=none` and `NCCL_NET=Socket`: host-specific TP stability
  settings. Re-evaluate on a new interconnect, driver, container, or GPU, but
  do not conflate them with the Qwen tensor adapter.

### Product policy and safety behavior; normally keep

- Demand-driven allocation and immediate return of freed request pages.
- Zero minimum/maximum reserved pages and disabled page preallocation.
- Prefix caching disabled until bounded prefix retention is deliberately
  qualified.
- Active and sleeping admission using measured peak and residual VRAM.
- FCFS pressure, preload priority, starvation protection, idle-to-RAM offload,
  RAM-cache-hit wake, and bounded LRU process eviction.
- Exact process-group teardown, including orphan TP worker cleanup.
- The compatibility runner's refusal to use GPUs with active compute
  processes.
- Level-1 sleep validation, post-wake output validation, and persistent host
  backups for immutable weights.

## Clean removal procedure

### 1. Define the candidate tuple

Record all of the following before editing dependencies:

```text
LLM-RIO commit
vLLM version and commit/build
kvcached version and immutable commit
Python, Torch, CUDA runtime, NVIDIA driver
GPU product, compute capability, VRAM, and count
NCCL version and transport
FlashInfer/FlashAttention versions
model repository and immutable snapshot
quantization and KV-cache dtype
max model length, max sequences, max batched tokens
TP and PP sizes
```

Do not test an unpinned branch name such as `main`. Resolve it to a commit.

### 2. Preserve the last-known-good environment

Use a separate git branch/worktree and Python environment. Do not upgrade the
production environment in place. Preserve:

- The current lock file and exact direct-reference metadata.
- The last passing JSON report and engine logs.
- A deterministic prompt/output corpus from native vLLM and from the current
  adapter.
- `nvidia-smi -q`, `nvidia-smi topo -m`, and package-version output.

Drain the service before any test that uses its managed GPUs:

```bash
./llmctl maintenance drain
./llmctl maintenance status
```

The compatibility runner will refuse to start if an active compute process is
still attached to a selected GPU.

### 3. Make upstream-native behavior selectable before deleting code

Do this in the candidate branch:

1. Update the kvcached pin in `pyproject.toml`.
2. Update the accepted revision/version logic in `src/llm_rio/prism.py`.
3. Separate “accepted experimental tuple” from “needs local vLLM 0.26
   compatibility layer.” Do not pretend a release is officially tested merely
   to suppress the shim or legacy-runner override.
4. Launch the candidate with bootstrap injection disabled.
5. Keep the adapter files present but inactive until native validation is
   complete. This makes rollback a one-line selection rather than a code
   reconstruction.
6. Change `prism_compat.py` so it records
   `allocator_implementation = "upstream"` and fails if the local shim marker
   appears. The current runner expects that marker and therefore cannot, by
   itself, certify removal.

A clean implementation should eventually expose an explicit runtime field
such as `needs_vllm026_compatibility`; version acceptance, packed-layout
adaptation, and model-runner selection are separate decisions.

### 4. Verify the installed source

After syncing the isolated environment, confirm both package versions and the
resolved kvcached commit:

```bash
uv run python -c 'import importlib.metadata as m; print(m.version("vllm")); print(m.version("kvcached")); print(m.distribution("kvcached").read_text("direct_url.json"))'
```

Fail if the commit is not the reviewed commit. Do not rely only on
kvcached's package version because source revisions can share a version.

### 5. Run the native-upstream qualification matrix

For every exact checkpoint, first run vanilla vLLM without kvcached and save
deterministic responses. Then run kvcached without the local adapter.

At minimum, execute:

| Model | Required shape |
|---|---|
| Qwen 3.8 27B NVFP4 | TP=1 and TP=2 |
| Qwen 3.6 27B NVFP4 | TP=1 and TP=2 |
| Gemma 4 31B NVFP4 | every intended production TP shape |
| Laguna S 2.1 NVFP4 | TP=2 |

Example for one TP=2 shape:

```bash
uv run python -m llm_rio.prism_compat \
  --model-a /absolute/path/to/model-a-snapshot \
  --model-b /absolute/path/to/model-b-snapshot \
  --gpu-uuid GPU-... \
  --gpu-uuid GPU-... \
  --tensor-parallel-size 2
```

Omit `--model-b` to test two residents of the same checkpoint. Provide it to
test two different checkpoints that use the same TP shape.

The runner must prove all of these:

- Both servers load without the local adapter.
- kvcached autopatch is active.
- Deterministic output matches the vanilla baseline, not merely “returns
  text.”
- Streaming and non-streaming inference both work.
- Every TP rank enters the expected allocator path.
- Both engines are genuinely resident at once.
- Switching to an already-resident engine remains under the activation SLO.
- A long request produces measurable physical KV growth.
- Most of that growth is returned after request completion.
- Termination returns GPU memory to baseline.
- No API, EngineCore, or TP worker process survives.

Do not use the current Qwen report as proof for Qwen 3.6, Gemma, or Laguna.

### 6. Add simultaneous saturation tests

The basic runner sends bounded requests and does not prove worst-case shared
KV behavior. Before production, run:

- Concurrent prefill-heavy requests against both residents.
- Concurrent decode-heavy requests against both residents.
- The configured `max_num_seqs` and `max_num_batched_tokens`.
- Long contexts near each intended operational limit.
- Cancellation and client disconnect during allocation.
- One server terminating while the other is busy.
- Repeated load/serve/reclaim cycles.
- At least one run long enough to expose fragmentation or page-retention
  drift.

Record combined throughput, per-model TTFT/ITL, errors, peak memory, and
reclaimed memory. Stop immediately on output divergence, CUDA illegal memory
access, NCCL failure, allocator corruption, or unexplained monotonic memory
growth.

A passing run must never contain `Cannot get N free blocks from the pool` or
an EngineCore fatal error. Temporary pressure may emit the translated
`Shared physical KV pool is exhausted` scheduling warning, but every admitted
request must eventually complete or reach its configured timeout cleanly.

kvcached virtualizes KV pages; it does not make the sum of two unbounded peak
workloads fit in finite physical VRAM. Establish a tested concurrency envelope
for each co-resident model set.

### 7. Remove the adapter only after every gate passes

In one reviewable change:

1. Delete `src/llm_rio/kvcached_vllm_compat.py`.
2. Delete `src/llm_rio/kvcached_bootstrap/`.
3. Delete `LLM_RIO_KVCACHED_VLLM026_SHIM` and bootstrap `PYTHONPATH` logic.
4. Remove the exact old compatibility revision.
5. Update the supported version/revision policy to the newly qualified tuple.
6. Replace the shim-log assertion with an upstream-native allocator assertion.
7. Re-evaluate, but do not automatically delete, the page/layout/NCCL policy
   variables listed above.
8. Update README and this model matrix.
9. Run unit tests, Ruff, mypy, compile checks, and the entire hardware matrix
   again from the final post-deletion tree.

The final hardware test must be run after deletion. A pass from the
“adapter inactive but still present” stage is necessary but not sufficient.

### 8. Revalidate LLM-RIO profiles

Profiles are versioned by engine and memory backend. Do not edit an old native
profile to claim kvcached validation.

With kvcached required in the candidate service:

```bash
./llmctl models retry qwen3.8-27b-nvfp4
./llmctl models retry qwen3.6-27b-nvfp4
./llmctl models retry gemma-4-31b-it-nvfp4
./llmctl models retry laguna-s-2.1-nvfp4
```

Review every registration job and validation log:

```bash
./llmctl models review MODEL_NICKNAME
./llmctl models profiles MODEL_NICKNAME
```

Run the standalone remote suite after all four profiles validate. Put a dedicated
admin test key in the hard-coded `API_KEY` constant first:

```bash
python3 four_model_prism_acceptance.py --quick
python3 four_model_prism_acceptance.py --allow-maintenance
```

The second command is disruptive: it drains all resident models for the initial
cold baseline and again while an inference is active. Its JSON report includes
per-request latency and TTFT, observed load/ready/stop transition durations,
effective TP values, batching statistics, per-GPU VRAM reclamation, and an
explicit previously-loaded resident-or-evicted switch latency test.


The CLI does not currently print the backend field. Verify it read-only in the
local catalog:

```bash
sqlite3 -readonly state/llm-rio.db "SELECT c.nickname,json_extract(p.profile_json,'$.memory_backend') FROM model_profiles p JOIN model_catalog c ON c.id=p.model_id WHERE p.active=1 ORDER BY c.nickname;"
```

Confirm newly active profiles were produced by the candidate engine and have
`memory_backend = "kvcached"`. Do not use `profile-edit` as a substitute for
benchmark validation.

### 9. Roll out gradually

1. Start with `kvcached_mode = "required"` and one explicit preload model.
2. Run smoke, load, and reclamation checks through the real gateway.
3. Add the next model only after the first remains stable.
4. Exercise at least one mixed-model workload.
5. Expand to `prism_preload_models = ["*"]` only after the complete resident
   set and its concurrency envelope are known.

Keep the previous code, lock file, environment, and reports available for
rollback. Rollback means draining, restoring the last-known-good tuple,
syncing its isolated environment, revalidating profiles if their identity
changed, and then resuming service.

## Procedure for a new model, vLLM version, or GPU architecture

### Phase A: inventory before execution

Inspect the checkpoint's immutable `config.json`, including nested
`text_config`:

```bash
jq '{architectures,model_type,text_config:(.text_config // null | {architectures,model_type,num_hidden_layers,layer_types,sliding_window,attention_chunk_size,full_attention_interval,head_dim,num_attention_heads,num_key_value_heads,linear_num_key_heads,linear_num_value_heads,linear_key_head_dim,linear_value_head_dim})}' /path/to/snapshot/config.json
```

Classify:

- Full attention: usually MHA/GQA.
- Full plus sliding/chunked/local attention: attention-only hybrid.
- Full plus linear attention, Mamba, GDN, or another recurrent state:
  linear/SSM hybrid.
- MLA or mixed MLA.
- More than two KV cache spec types.
- Different KV geometry between groups.
- Cross-layer KV sharing or another novel layout.

Names are hints only. The vLLM startup log's actual KV cache spec classes,
tensor shapes, block sizes, strides, group count, and page bytes are the
runtime source of truth.

Also record:

```bash
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
nvidia-smi topo -m
uv run python -c 'import torch,vllm; print(torch.__version__, torch.version.cuda, vllm.__version__); print(torch.cuda.get_device_capability())'
```

For a different GPU generation, do not inherit NCCL, attention-backend, CUDA
Graph, quantization-kernel, or page-layout assumptions without measurement.

### Phase B: establish a vanilla vLLM oracle

Before enabling kvcached:

- Prove model load, non-streaming inference, streaming inference, TP, and
  intended context length on vanilla vLLM.
- Save deterministic outputs for short prompts, long prefills, multi-turn
  prompts, tool calls, reasoning output, and image input where applicable.
- Save startup logs and memory/latency measurements.

If vanilla vLLM fails, the problem is not a kvcached compatibility problem.
Resolve it in vLLM/model/quantization first.

### Phase C: integrate in increasing-risk order

Run these stages in order and stop at the first failure:

1. Import kvcached and confirm autopatch discovery.
2. Start one model at TP=1.
3. Compare deterministic output with vanilla.
4. Measure request-level KV growth and reclamation.
5. Run every intended TP/PP shape.
6. Start two copies of the same model.
7. Start mixed models with the same placement shape.
8. Apply concurrent prefill and decode pressure.
9. Terminate and verify complete process/memory reclamation.

Never debug several new variables together. Change only one of model,
vLLM, kvcached, Torch/CUDA, GPU architecture, quantization, layout, or TP
between comparisons.

### Phase D: classify the first failure

| First failure | Likely boundary | Investigation |
|---|---|---|
| Import or autopatch failure | Packaging/API-version drift | Inspect installed `direct_url.json`, import paths, and kvcached patch discovery |
| Unsupported KV spec such as a new Mamba/linear/local class | Architecture support gap | Inspect vLLM's emitted KV spec groups and add the smallest upstream unit test |
| Shape/stride error during KV allocation | vLLM–kvcached allocator interface drift | Capture shape, dtype, layout, block size, kernel block size, TP rank, and expected native stride |
| Physical block larger than kvcached page | Page geometry | Determine maximum block bytes across every group/rank; increase page size only as much as required |
| TP-only crash or NCCL segmentation fault | Transport/process topology | Reproduce TP=1 versus TP>1; inspect rank logs, NCCL version/plugin, topology, and process groups |
| Server responds but tokens differ from vanilla | Correctness defect | Stop; compare tensors/strides and deterministic token IDs before any performance work |
| Memory grows but does not return | Retention/reclamation policy | Check prefix caching, cached-token limits, reserved pages, preallocation, and live requests |
| Parent exits but GPU memory remains | Orphan worker lifecycle | Inspect PID/PPID/PGID and kill/test the exact process group |
| Works alone but fails co-resident | Physical capacity or global-control gap | Measure static weights plus simultaneous dynamic KV peaks; reduce concurrency or add resource control |

### Phase E: patch with a strict boundary

Prefer fixing and testing upstream. If a local adapter is temporarily
unavoidable:

- Pin the exact upstream source revision it patches.
- Guard on exact vLLM range, kvcached revision, attention/spec type, shape
  rank, layout, and any required stride.
- Pass every other architecture/layout directly to upstream unchanged.
- Make incompatible conditions fail loudly.
- Emit a unique log marker whenever the adapter executes.
- Add a CPU shape/stride regression test, including TP rank geometries where
  head counts differ.
- Add the real-GPU compatibility check.
- Document its deletion conditions in this file.

Do not use broad monkeypatches that silently reinterpret unknown tensor
layouts. A new architecture should fail closed rather than produce plausible
but incorrect tokens.

### Phase F: acceptance and evidence

Each qualification report should contain:

- Exact software, model, and hardware tuple.
- All commands and environment overrides.
- Per-rank logs.
- Vanilla and kvcached deterministic token comparison.
- Load/warmup time and resident activation time.
- Idle, peak, post-request, and final GPU memory.
- Request KV growth and reclaimed bytes.
- TP and co-residency results.
- Concurrent-load throughput and latency.
- Process table before and after teardown.
- Known limitations and the tested concurrency envelope.

Store reports under a timestamped diagnostics directory. Never overwrite the
last-known-good report.

## Decision rule

Use these labels consistently:

- **Upstream-listed**: the family appears in upstream documentation.
- **Structurally likely**: emitted KV specs match a supported family.
- **Basic pass**: one model loads and returns correct output.
- **Locally qualified**: the complete model/TP/memory/reclamation matrix passes
  on the exact deployment tuple.
- **Production approved**: locally qualified plus sustained mixed-model load
  passes within an explicit concurrency envelope.

Only the last two labels authorize enabling a model in kvcached production.
