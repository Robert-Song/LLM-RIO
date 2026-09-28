# Installation and configuration

The router requires Python 3.11–3.13. `uv sync --locked` installs the router and TUI;
`--extra dev` adds checks; `--extra engine` installs the pinned native vLLM version.
llama.cpp is an operator-installed executable. `--extra kvcached` is optional and
experimental; see [its compatibility instructions](EXPERIMENTAL.md).

Use `uv run --no-sync` after installation so a command does not silently resynchronize
an environment with different extras. The repository's `llmctl` wrapper also uses the
installed environment. Separate experimental and native environments.

Configuration precedence is explicit constructor/CLI options, `LLMRIO_` environment,
`.env`, TOML, then defaults. `LLMRIO_CONFIG_FILE` selects the TOML file unless a CLI
`--config` is provided. Paths are relative to the process working directory. Nested
settings use double underscores, such as `LLMRIO_ENGINES__VLLM_EXECUTABLE`.

`serving_mode` is required: `queue`, `vllm-sleep`, or `kv-cached`. Unknown TOML and
nested settings are rejected. Beta `prism_*` switches and engine fallback controls
are not accepted. Changing mode requires restart. Use a new release config; do not
point it at the old lab database.

A file may retain separate `[modes.queue]`, `[modes.vllm_sleep]` and
`[modes.kv_cached]` sections. Only the selected `serving_mode` is active. Changing
that selection in TOML, the environment, CLI `--mode`, or the TUI start form does
not require deleting the other sections. The selected mode uses its own section
or defaults when absent; queue never inherits sleep preloads or memory policy.
All sections remain schema-checked for unknown names and invalid values. Shared
engine options must still be compatible with the active mode; for example,
`engines.enable_llama_cpp = true` requires queue mode. Switching requires a restart
and profiles validated for the selected mode.

For a checkout that still has beta `config.toml`, keep that file and prepare a
separate release file from [the sleep example](../examples/config/vllm-sleep.toml)
or [the queue example](../examples/config/queue.toml). Set
`LLMRIO_CONFIG_FILE=config.release.toml` in the shell or the checkout's `.env`
after editing the new file. Preserve other existing `.env` entries. A shell selector
takes precedence over `.env`; `serve --config PATH` takes precedence over both.

`./llmctl` opens the administration console. Choose **Diagnostics → Start service**
to start the server in that terminal, or run `./llmctl serve` directly. Both use
the same configuration resolver. The start form checks configuration before closing
and retains entered values on failure. Missing explicitly selected files, beta fields,
invalid TOML and missing modes produce configuration errors without printing values
such as the Hugging Face token. The release catalog starts empty: register and validate
existing artifacts against the new mode instead of importing beta verification flags.

| Settings | Purpose |
| --- | --- |
| `database_path` | Fresh release schema, default `state/release/llm-rio.db` |
| `api_host`, `api_port` | Public interface; default `0.0.0.0:8002` |
| `model_store`, `log_dir` | Download cache and engine logs; local sources stay in place |
| `managed_gpu_uuids`, `reserved_vram_mib` | GPU ownership scope and safety reserve |
| `worker_port_start`, `worker_port_end` | Shared private serving/validation port pool |
| `queue_capacity_per_model`, `queue_capacity_per_tenant` | Optional admission bounds |
| `wait_duration_seconds`, `minimum_residency_seconds`, `fair_share_seconds` | Scheduling latency/fairness controls |
| `worker_*timeout_seconds`, `worker_drain_watchdog_seconds` | Optional startup, request, stream-idle and drain watchdog limits |
| `max_prompt_tokens`, `max_output_tokens`, `max_n` | Optional router-wide request limits |
| `quota_charge_requested_maximum` | Conservative quota policy instead of measured output usage |
| `engines.vllm_executable`, `engines.llama_cpp_executable` | Explicit launcher paths |
| `engines.enable_llama_cpp` | Makes llama.cpp available in queue mode only |
| `engines.environment` | Engine environment; native launch strips experimental activation |
| `engines.max_model_len`, `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization` | Initial validation limits; measured effective values are saved in profiles |
| `modes.queue.scale_window_seconds`, `minimum_marginal_efficiency` | Queue replica demand horizon and minimum throughput gain |
| `modes.vllm_sleep` | Native sleep preload, idle grace, RAM/swap limits and transition timeout |
| `modes.kv_cached` | Independent experimental cache settings and per-GPU worker cap |

All shipped defaults appear in [config.example.toml](../config.example.toml).
Unset optional numeric values are omitted from TOML; zero is not an unlimited sentinel.
In `vllm-sleep`, validation starts at 0.80 GPU memory utilization when the value
is omitted. An explicit engine or per-validation value is honored, including values
above 0.80; confirmed memory failures use the mode's lower retry budgets.
Supply `LLMRIO_HF_TOKEN` for gated repositories. Keep credentials out of tracked TOML.
For remote administration use `LLMRIO_API_URL` and `LLMRIO_API_KEY`; these are client
connection settings, distinct from server binding settings.

Engine upgrades, environment changes and launch edits require revalidation. A separately
managed executable must be kept reproducible by the operator, including its interpreter,
CUDA libraries, and dependency lock. The router's lock file cannot pin another environment.

Engine identity is obtained from the executed environment: standard vLLM console
scripts use their interpreter's distribution metadata/RECORD, and other executable
wrappers must implement a deterministic, bounded `--version`. Unavailable identity
blocks eligibility. Changed engine identity invalidates launch bindings, including
an external environment update with an unchanged wrapper. Identity must remain stable
across validation. Do not update an engine environment while its service is running.

All cooperating instances on a host run as the same service account. Startup acquires
process-lifetime database and managed-GPU UUID locks before recovery. These local
locks do not coordinate different Unix accounts or remote hosts. The host operator
must assign disjoint GPUs to independent accounts. `fair_share_seconds` is used by
sleep/experimental rotation policy; queue uses oldest-backlog ordering and tenant DRR.

When installing a built wheel for qualification, constrain its transitive dependencies
with a native export of the same `uv.lock` (`uv export --locked --no-dev
--no-emit-project --no-hashes --format requirements-txt`). An unconstrained wheel
install alone does not reproduce the lockfile environment. CI and the manual GPU
kit both install the wheel with this exported constraint file.
