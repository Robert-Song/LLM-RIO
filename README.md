# LLM-RIO

LLM-RIO is a machine-local, multi-tenant LLM service for one Linux/HPC GPU host. It exposes an
OpenAI-compatible chat endpoint while automatically managing fair queues, measured GPU-set
placements, independent replicas, safe draining, catalog registration, and token quotas.

Staff control model availability, access, and policy. The application alone controls loading,
GPU UUID placement, replication, draining, and unloading. Separate installations do not form a
cluster and never share queues.

## Status

This repository contains the control-plane implementation and external-HPC acceptance suite.
The kvcached elastic-KV path and host-RAM model-weight cache have been exercised on the target
two-GPU host; see [PRISM_RESEARCH_FEASIBILITY.md](PRISM_RESEARCH_FEASIBILITY.md) for measured
offload, activation, and compatibility results.

## Manual start on the target Linux host

Installation is intentionally command-by-command; running `setup.sh` is not required. Review the
full sequence in [docs/OPERATIONS.md](docs/OPERATIONS.md). The core production commands are:

```bash
git clone <repository>
cd LLM-RIO
python3 -m pip install --user uv==0.12.1
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12
uv sync --extra engine --extra prism
cp config.example.toml config.toml
mkdir -p state models logs diagnostics
chmod +x llmctl
./llmctl serve
```


`config.example.toml` is the single configuration template. It explains every setting; optional
restrictions remain commented out so copying it preserves permissive research defaults.
Load site-specific Python or CUDA modules before these commands when required by the HPC. The first
startup prints the initial admin key before accepting requests; administrators can create any
number of additional admin keys. Local `llmctl` management commands automatically recover an
active admin credential from the protected host database, so they do not require
`LLMRIO_API_KEY`. Remote management still requires an explicitly supplied admin key.

### Serving modes

Select a mode with `./llmctl serve --mode queue`, `LLMRIO_SERVING_MODE=queue` for other
launchers, or top-level `serving_mode = "queue"` in TOML. Explicit mode selection overrides
the legacy kvcached and weight-cache switches. Omit it to preserve existing launch behavior;
the built-in default remains `vllm-sleep`.

| Mode | Switching | GPU ownership | Validation |
| --- | --- | --- | --- |
| `queue` | Drain admitted work, fully unload, then cold-start | One worker per GPU; independent replicas on free GPUs | Maintenance; no sleep; configured/hardware-derived GPU utilization |
| `vllm-sleep` | Level-1 sleep in RAM, then wake | One active worker per GPU; sleeping contexts may remain | Maintenance; verifies sleep/wake; initial utilization capped at 0.80 |
| `kv-cached` | Experimental elastic KV and optional RAM weight caching | Measured co-residency | Experimental validation policy |

Queue mode prioritizes older model backlogs while retaining tenant fairness, smallest viable
GPU placements, TP fallback, and replica scaling. Admitted requests finish before a blocking
worker unloads. New workers cannot reuse its GPUs until process and GPU teardown is verified.
Queued TP models keep priority over younger work on a partially released GPU group.
`minimum_residency_seconds` still applies. Queue mode ignores `prism_preload_models` and
sleep/cache settings; it loads on demand and fully unloads idle workers after the usual idle wait.

Queue mode requires fresh measurements with sleep explicitly disabled. Old sleep-mode profiles
are not routable in queue mode; run the revalidation script after switching. Both scripts report
the server's actual serving mode, also visible through maintenance status in the API, CLI, and TUI.

### Prism scheduling: elastic KV plus a host-RAM weight cache

LLM-RIO can keep multiple vLLM engines ready on the same GPU set while kvcached allocates physical
KV memory on demand. When an idle engine blocks another model, the scheduler uses vLLM level-1
sleep to release its model and KV allocations from GPU memory while preserving its initialized
process and model weights in host RAM. A later request wakes that same PID and copies the weights
back instead of rebuilding the engine or rereading the checkpoint.

Set `prism_weight_cache_mode = "ram"` to retain warm level-1 sleeping workers.
With `engines.kvcached_mode = "none"` (also selected by an empty value), workers use native vLLM
sleep/wake and at most one engine is GPU-resident on each GPU. Set
`engines.kvcached_mode = "required"` only after installing the legacy-named `prism` extra and
manually verifying the model with kvcached on this machine; required mode enables measured
multi-worker GPU co-residency.

`prism_preload_models` is only the startup warming policy; it is not an eligibility list. Any
verified model requested later can cold-start, displace an idle resident to RAM, and remain cached
after its request drains. Repeat a nickname to prepare distinct placements before a burst.

Normal registration performs fail-closed native validation only after maintenance has fully
drained the host (`MAINTENANCE_READY`). Downloads and inspection can proceed while serving;
GPU validation jobs then report `waiting_for_maintenance`. Enter maintenance through the TUI's
Drain button or `./llmctl maintenance drain`. All production workers, including sleeping cached
processes, are fully unloaded before normal-mode probes run. Independent GPU placements for a
candidate shape are probed concurrently. When any TP=1 placement passes, its verified one-GPU
profiles are registered and larger TP shapes are not probed automatically; create and validate a
larger profile explicitly in the TUI if needed. In RAM weight-cache mode, every probe must pass
generation, level-1 sleep, wake, and post-wake generation. Each validation worker is terminated after the probe,
and successful normal-mode registration does not request an automatic warm load. Resume is
blocked while validation workers own GPUs. Experimental kvcached mode retains idle-time validation.

VRAM measurement format v2 captures one per-GPU baseline immediately before launch, samples
continuously through post-wake generation, and stores incremental idle, active-peak, sleeping
residual, and wake-peak usage relative to that baseline. Legacy absolute-VRAM profiles are not
routable and cannot be mixed with v2 profiles. kvcached validation remains explicit through the
Profiles page's **Verify kvcached** action. The independent administrator verification flags cannot
make a legacy or invalidated profile routable without compatible v2 measurements.

The scheduler never offloads a worker with an admitted request. It protects minimum residency and
fair-share rules, sleeps idle blockers under GPU pressure, and prefers a matching sleeping worker
over a cold start. Warm capacity is memory-based: `prism_host_cache_max_gib` limits PSS-accounted
worker process groups, `prism_host_cache_min_available_gib` preserves host/cgroup headroom, and
`prism_swap_max_used_gib` sets their process-group swap-pressure ceiling. The OS must configure swap
before startup; virtual address size is not counted as cache capacity. Pressure evicts the
least-recently-used eligible sleeping process.

For every GPU, the planner enforces `sum(active peaks + sleeping residuals) <= physical VRAM -
reserved_vram_mib`. Active footprint is the maximum measured initial/wake peak; sleeping footprint
is the measured level-1 residual. The global reserve is subtracted exactly once, and
`gpu_memory_utilization` is not applied again as a scheduler ceiling. `prism_max_workers_per_gpu`
applies only to simultaneously GPU-resident kvcached engines; native mode permits one.

In normal mode, the planner can propose a launch or wake that requires reclaiming sleeping
contexts. Before executing it, the service compares the stored per-GPU active/wake peak plus the
global reserve against **live NVML free VRAM**. Cold launches also respect native vLLM's configured
startup memory fraction. Sleeping workers on deficient GPUs are fully stopped in LRU order;
VRAM is sampled again after each stop. Active requests are never evicted. A wake credits only
its own measured residual that is also visible in the process group's current NVML allocation.
If a safe wake remains impossible, its cached process can be unloaded for a later cold start.
Missing telemetry or remaining external allocations defer admission rather than launch into OOM.
External processes can still allocate memory after a sample, so this is not an exclusive GPU lock.

On the target RTX PRO 6000 host, the first Qwen3-8B validation measured 4.25-8.07 seconds to
offload, 0.25-0.34 seconds to restore, and roughly 2.3-3.0 GiB sleeping VRAM. A routed cache-hit
request completed in 0.887 seconds total and retained the worker PID. Treat these numbers as
checkpoint-, placement-, and host-specific; registration records the current measurements rather
than assuming a paper benchmark. A later Qwen3.8 27B live run woke in 0.522 seconds, served a short
request in 1.413 seconds end to end, and re-entered sleep in 0.352 seconds while keeping the host
backup. Two cached Qwen3.8 TP=1 workers then woke in 0.533 and 0.610 seconds. A 40-request,
two-wave homepage workload returned 17,425 completion tokens (18,665 total) with no errors, and
the second wave split evenly across the two workers. Scale-out is deliberately limited to the
model's one-GPU profiles once one TP=1 copy is serving; it does not add a redundant TP=2 worker.
In a six-call Qwen/Gemma/Laguna round robin, every public-API response completed in 1.10-8.61
seconds. Measured worker wakes were 0.648, 0.707, and 1.661 seconds, while repeated offloads were
0.146-0.218 seconds.


kvcached has not officially validated vLLM 0.26.0. The legacy-named `prism` extra pins upstream
revision `60cad949` for its vLLM 0.26 and hybrid-cache fixes. LLM-RIO also supplies a narrowly
scoped packed-KV tensor adapter for vLLM 0.26 MHA/GQA and linear-attention hybrids, and selects the
legacy vLLM model runner patched by this kvcached revision. It uses 4 MiB kvcached pages for large
KV blocks, immediately returns freed request pages instead of retaining prefix-cache pages, and
normalizes a shared-pool free-count race to vLLM's ordinary scheduling-retry path. If that race
occurs midway through a hybrid allocation, the adapter also rolls back the waiting request's
partial group state before retrying. The adapter additionally handles vLLM 0.26's nested hybrid
FP8 cache groups during wake. Run the isolated compatibility check before enabling it for the
service:

```bash
uv run python -m llm_rio.prism_compat \
  --model-a /path/to/model-a --model-b /path/to/model-b \
  --gpu-uuid GPU-... --tensor-parallel-size 1
```

Use [KVCACHED_COMPATIBILITY.md](KVCACHED_COMPATIBILITY.md) to qualify new models, vLLM releases,
kvcached revisions, and GPU architectures, or to remove the temporary vLLM 0.26 compatibility
layer after upstream support is proven.

## Terminal administration

Run `llmctl` without arguments to open the full-screen administration interface:

```bash
./llmctl
```

The TUI provides dashboards and forms for API keys, quotas, model registration and access,
registration jobs, placement profiles, maintenance mode, host diagnostics, and service startup.
Its dashboard refreshes every two seconds while visible and shows current/total token throughput,
non-empty output throughput, model popularity, and live NVIDIA GPU and worker-placement status.
Use the mouse or keyboard to navigate; `R` refreshes the current page and `Q` exits. Destructive
operations require confirmation.

All command-oriented workflows remain available for scripts and runbooks. For example,
`./llmctl keys list`, `./llmctl models review MODEL`, `./llmctl maintenance drain`,
`./llmctl doctor`, and `./llmctl serve` behave as before. `./llmctl interactive` is an explicit
alias for opening the TUI.

New model registration automatically resolves, downloads, and inspects the model. In normal
mode, hardware validation waits for maintenance. Add `--wait` to print each state transition
until the model is available or fails (enter maintenance from another terminal if needed):

```bash
./llmctl models add qwen3-8b Qwen/Qwen3-8B --grant-to teamA --wait
```

The same asynchronous workflow starts with `POST /staff/models` and is observed with
`GET /staff/model-jobs/{job_id}`.

For normal-mode deployment:

1. Queue new registrations or retry failed registrations using the API, CLI, or TUI.
2. Run `./llmctl maintenance drain` (or choose **Maintenance → Drain** in the TUI).
3. Use `./llmctl maintenance status` and `./llmctl models review MODEL` to follow draining and
   validation. Waiting jobs start automatically when the service reaches `MAINTENANCE_READY`.
   A previous failed job needs `./llmctl models retry MODEL` to be queued again.
4. Once validation completes, run `./llmctl maintenance resume` or choose **Resume** in the TUI.
   Models load on demand or under an explicitly configured preload policy.

The equivalent API operations are `POST /admin/maintenance` with `{"mode":"drain"}` or
`{"mode":"active"}`, and `GET /admin/maintenance` for status. The status includes whether
validation requires maintenance and the GPUs currently held by probes. `gpu_memory_wait`
means validation is waiting for live memory availability; it retries automatically.

To revalidate the entire enabled catalog in queue mode, restart the updated server with
`./llmctl serve --mode queue` (or set `LLMRIO_SERVING_MODE=queue` in your launcher), then run:

```bash
.venv/bin/python fire_all_native_revalidations.py --report revalidation-submit.json
# Follow jobs with llmctl / TUI; leave the server in maintenance until they finish.
.venv/bin/python final_deployment_test.py --continue --report deployment-results.json
```

The first script requests maintenance and retries all enabled models, including failed models
and models with existing measurements. `--dry-run` makes no changes; `--only-invalid` restores
selective revalidation. Already queued/running jobs are left running. Disabled models are skipped.
The second script requires an admin `LLMRIO_API_KEY`, or an inference `--api-key` with the
environment key unset so local `llmctl` can recover admin credentials. Its default target is the
entire enabled catalog, including NVFP4. `--continue` checks registration completion, routability,
and GPU reservations before resuming. Pending/failed registrations block the full-catalog test;
it does not wait for them automatically. `--model NAME` tests an explicit subset;
`--all-available` tests only routable models and therefore provides partial coverage.
A pass requires completed assistant text matching the test prompt and positive token usage.
This is a sequential inference smoke test, not a concurrency or long-context load test.

`vllm-sleep` validation caps initial `gpu_memory_utilization` at **0.80**, including higher
configured overrides. Confirmed CUDA OOM or measured-capacity failures retry at 0.10 and 0.20
below that initial fraction (never below 0.40). Lower explicit fractions remain lower. Each
attempt must fully tear down before retrying; the passing fraction is stored in the profile
and used for serving. Unsupported-model and configuration errors are not retried as memory
failures. Queue mode removes the 0.80 cap: the first attempt uses the configured fraction,
or the hardware-derived fraction if none is configured. Confirmed memory failures retry in
0.02 decrements, up to five retries (never below 0.40), before trying a larger TP shape.
An explicit utilization remains an upper bound; adjust it if a model needs a higher fraction.
Both native modes retain live VRAM admission and the global reserve. Experimental Prism
validation keeps its existing allocation policy.

Settled per-call usage can be compacted from the TUI Maintenance page or with:

```bash
./llmctl summarize
```

The same admin-only operation is available as `POST /admin/usage/summarize`; an optional JSON
body such as `{"through":"2026-08-21T00:00:00Z"}` sets an exact timezone-aware cutoff. Each run
adds the completed current window to `total`, resets `current` to begin at the cutoff, and deletes
the settled raw request, reservation, and ledger rows included in that summary. Active or queued
requests are left untouched. A weekly cron job can therefore invoke `./llmctl summarize`; use
`LLMRIO_API_URL` and `LLMRIO_API_KEY` when the command runs away from the service host.

The admin-only `GET /admin/dashboard` endpoint exposes the same live data used by the TUI:
current and total usage/throughput, output tokens per active generation second, ranked per-model
token usage, NVML-backed GPU health, loaded model placements, and continuous-batching slot use.

## API surface

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/me/usage`
- `POST /staff/models`, `GET /staff/model-jobs/{job_id}`, and
  `POST /staff/model-jobs/{job_id}/retry`. `GET /staff/models` includes each
  model's registration job so failed registrations can be reviewed by model nickname.
- `POST /staff/model-access` using API-key selectors and model nicknames
- admin key, quota, atomic maintenance routes, admin-only placement-profile overrides, and
  `POST /admin/models/{model_id}/clone` for shared-weight logical model profiles

The admin CLI uses the authenticated management routes but automatically recovers a local admin
credential from the protected database/vault. Use `./llmctl models profiles MODEL` to inspect
placement profiles and `./llmctl models profile-edit MODEL PROFILE_ID` to override a stored profile.
Any launch-affecting override clears throughput, VRAM, sleep/wake, and verification measurements;
the exact edited configuration must pass real validation before inference can route to it. Use
`--restart-workers` to drain workers still using an older profile. Key and model access commands
accept human-readable nicknames (or a complete API key for key selection).

### Shared-weight model profiles and request defaults

A cloned model receives its own catalog model ID but references the same downloaded artifact
directory and hashes. An exact launch-configuration clone may reuse the source measurement; a clone
that changes any launch setting is deliberately unroutable until revalidated. Access grants are
inherited by default; pass `--no-inherit-grants` to start without them.

A profile can store defaults for `temperature`, `top_p`, `top_k`, and `reasoning_effort`. The gateway
fills only omitted request fields, so any value explicitly supplied in a chat-completions request
wins. Blank clone options inherit any defaults already stored on the source model.

For example, this creates the extended Qwen profile described above. The YaRN factor and context
length are stored in the cloned vLLM placement profiles as Hugging Face config overrides:

```bash
./llmctl models profile-clone \
  qwen3.8-27b-nvfp4 \
  qwen3.8-27b-nvfp4-ext \
  --reasoning-effort medium \
  --max-model-len 1048576 \
  --yarn-factor 4 \
  --yarn-original-max-model-len 262144
```

The same operation is available through the TUI's Models page with **Clone profile**. A clone that
changes context length, YaRN, or another launch setting has all inherited measurements invalidated
and is not routable until that exact configuration passes real validation.

### Image inputs

The chat-completions endpoint forwards OpenAI-compatible image content directly to the selected
model worker without applying a catalog capability gate. The caller is responsible for choosing a
model that supports the supplied image format; an incompatible model returns its worker error.
Both remote URLs and base64 data URLs can be supplied using an `image_url` content part:

```json
{
  "model": "gemma-4-31b-it-nvfp4",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,<encoded image>"}
        }
      ]
    }
  ]
}
```

## Development

The repository is managed with `uv`. The automated suite uses temporary databases and mocked
workers and can run without starting the service or loading models:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/llm_rio
```

Install `--extra engine` to include the optional Torch compatibility tests. Hardware demos and
acceptance scripts are separate, explicit operations. See [CONTRIBUTING.md](CONTRIBUTING.md)
for module responsibilities, test isolation, configuration selection, and request accounting.

No scheduler branch or acceptance test assumes a two-GPU host; simulated planner tests cover
1, 2, 4, and 8 GPU inventories, while real capacity and performance are always established by
machine-specific validation profiles.
