# Working on LLM-RIO

## Development checks

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src/llm_rio
```

All automated tests live in `tests/`. Its autouse fixture changes into a temporary directory
and clears inherited `LLMRIO_` environment variables for each test. Tests can set their own
variables and create their own TOML files. Never point fixtures at `state/`, `state_barra/`,
`config.toml`, or `config.barra.toml`. Tests using checked-in assets should resolve paths from
`__file__`, rather than assuming the repository is the current directory.

The core suite uses SQLite files under pytest's temporary directory, HTTP mocks, simulated GPU
inventories, and mocked worker processes. The packed-KV compatibility tests skip when Torch is
not installed; `uv sync --extra dev --extra engine` includes that optional dependency. Passing
unit tests does not establish real GPU capacity, model compatibility, or throughput. Those
require the separate hardware acceptance procedures described in the existing Prism runbooks.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `api/app.py` | Application startup, teardown, middleware, and error responses |
| `api/routes_inference.py` | Model access, quota admission, queue submission, HTTP responses |
| `api/inference_validation.py` | Model defaults, request limits, and token estimates |
| `api/inference_proxy.py` | Worker HTTP requests, streaming, and lease finalization |
| `worker_protocol.py` | Incremental SSE decoding and worker JSON/usage validation |
| `database_schema.py` | Backward-compatible SQLite table, index, and view definitions |
| `storage.py` | Database initialization/migrations, transactions, credentials, catalog, quotas |
| `queueing.py` | Per-model, per-tenant deficit-round-robin admission |
| `gpu_memory.py` | Live NVML headroom and native launch/wake memory requirements |
| `planner.py` | Placement decisions from measured profiles and queue pressure |
| `runtime.py` | Scheduler execution and worker lease ownership |
| `workers.py` | Worker processes, launch configuration, and sleep/wake transitions |
| `registration.py`, `validation.py`, `profiles.py` | Registration jobs and measured profiles |
| `cli.py`, `tui.py` | Command-line and terminal administration |

Keep worker parsing independent of FastAPI and persistence. Keep policy decisions in the planner
and process operations in the supervisor. Existing CLI commands and HTTP routes remain the
entry points for operators; application helpers inside the package are internal APIs.

## Inference invariants

`Database.reserve_quota` creates the quota reservation, ledger debit, and `QUEUED` request in
one transaction. Pass request metadata to this operation; there is no separate
`create_inference_request` step. A failed insert must roll everything back.

A request ID is unique across requests, and an idempotency key is unique per API key while its
reservation record is retained. Reusing either returns HTTP 409 rather than launching another
inference against an existing reservation. This service does not cache or replay responses.
Usage compaction removes settled raw records, so these checks are not permanent deduplication.

Settlement is idempotent and must run before worker admission is released. Worker streams must
close even on disconnect, timeout, invalid JSON, or an unconsumed response body. Explicit zero
usage is authoritative; absent usage uses an estimate. Missing terminal events and malformed
worker responses must mark requests failed rather than charge the configured requested maximum.

The admission queue spends a tenant's remaining token credit before rotating to another tenant.
It resets credit when a tenant drains and skips rounds where no request can fit. Do not replace
this with one-request-per-tenant rotation: equal request counts are not equal token work.

## State and configuration

Schema changes must preserve existing databases. Exercise migrations against temporary legacy
schemas. A cleanup or refactor must never reset host databases, rotate host vaults, or rewrite
site configuration as a side effect of tests.

Configuration precedence is explicit constructor arguments, environment variables, `.env`, TOML,
and file secrets. The TOML selector follows the same precedence: `Settings(config_file=...)`,
`LLMRIO_CONFIG_FILE`, then the default `config.toml`. `llmctl serve --config PATH` remains available.

Application resources are registered for cleanup before background services start. Cleanup runs
in this order: scheduler/workers, registration probes, worker HTTP client, database. A failed
startup or a cleanup exception must not skip the remaining resources.

## Manual clients

`client_code_snippet.py` and `scripts/yarn_context_demo.py` are manual network clients, not unit
tests. Supply `LLMRIO_API_KEY` and optionally `LLMRIO_API_URL`; neither embeds host credentials.
The former also has historical embedding comparisons for servers offering an embedding endpoint;
LLM-RIO's public inference endpoint is chat completions. These examples require their own optional
client dependencies (`openai`, and `numpy`/`requests` for embedding comparisons).

## Native validation and serving memory

Native modes (`queue` and `vllm-sleep`) mean kvcached is not enabled. GPU validation in this mode requires
`MAINTENANCE_READY`; resolving and downloading artifacts may happen earlier. The drain operation
must fully stop sleeping workers and convert any pending drain-to-sleep into drain-to-stop.
A probe holds its GPU reservation until its process is terminated, including on failure or
cancellation. Resume must reject active probe reservations. Completed normal validation must
not schedule an automatic warm load.

The existing profile JSON persists `peak_vram_mib_per_gpu`, `wake_peak_vram_mib_per_gpu`,
`sleep_vram_mib_per_gpu`, and `vram_baseline_mib_per_gpu`. No database reset or schema migration
is needed for this policy. Legacy or invalidated profiles still need real revalidation.

The native planner treats sleeping workers as reclaimable; the scheduler's live capacity gate
is mandatory before executing a launch or wake. For each GPU, cold-start free memory must be at
least `max(measured_peak + reserve, physical_total * gpu_memory_utilization)`. The measured peak
includes the larger initial/wake peak. Wake free memory must cover `measured_peak - credited_residual
+ reserve`, with credit capped by both the measured sleeping residual and the target process
group's live allocation. This prevents double-counting the target while accounting for all other
sleepers and foreign allocations through NVML's global free memory.

Evict only idle sleeping workers on deficient GPUs, oldest demand first. Re-read memory after
stopping each process. Never evict an admitted request, assume a process exit already reclaimed
VRAM, or proceed without telemetry. Keep the experimental kvcached planner and admission policy
separate. Test both branches with mocked NVML/processes before attempting hardware acceptance.

Engine teardown uses `process_cleanup.terminate_engine`: signal the isolated process group,
wait for all live descendants to exit, and confirm NVML no longer lists their PIDs on the
assigned GPUs. Linux zombies count as exited but must also disappear from NVML. Cancellation
waits for cleanup; timeout or unavailable telemetry raises `TeardownError`. Validation retains
its GPU/port reservations, and production workers retain STOPPING state and process tracking,
until teardown can be verified. Never mark a worker COLD in a teardown-failure finally block.
Sleep-mode validation attempts start at min(requested fraction, 0.80) with at most two smaller
memory-only retries. Queue mode starts at the requested or hardware-derived fraction with up
to five 0.02 decrements on memory failures, and performs no sleep/wake probe. Its profiles must
record `launch_args.enable_sleep_mode = false`; sleep-mode measurements cannot be reused. Persist the fraction of the successful attempt in the placement profile.
