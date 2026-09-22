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
| `modes.vllm_sleep` | Native sleep preload, idle grace, RAM/swap limits and transition timeout |
| `modes.kv_cached` | Independent experimental cache settings and per-GPU worker cap |

All shipped defaults appear in [config.example.toml](../config.example.toml).
Unset optional numeric values are omitted from TOML; zero is not an unlimited sentinel.
Supply `LLMRIO_HF_TOKEN` for gated repositories. Keep credentials out of tracked TOML.
For remote administration use `LLMRIO_API_URL` and `LLMRIO_API_KEY`; these are client
connection settings, distinct from server binding settings.

Engine upgrades, environment changes and launch edits require revalidation. A separately
managed executable must be kept reproducible by the operator, including its interpreter,
CUDA libraries, and dependency lock. The router's lock file cannot pin another environment.
