# LLM-RIO

LLM-RIO is a machine-local, multi-tenant LLM service for one Linux/HPC GPU host. It exposes an
OpenAI-compatible chat endpoint while automatically managing fair queues, measured GPU-set
placements, independent replicas, safe draining, catalog registration, and token quotas.

Staff control model availability, access, and policy. The application alone controls loading,
GPU UUID placement, replication, draining, and unloading. Separate installations do not form a
cluster and never share queues.

## Status

This repository contains the greenfield control-plane implementation and an external-HPC
acceptance suite. Hardware execution and endpoint testing are intentionally deferred until the
target HPC environment and a test API key are available.

## Manual start on the target Linux host

Installation is intentionally command-by-command; running `setup.sh` is not required. Review the
full sequence in [docs/OPERATIONS.md](docs/OPERATIONS.md). The core production commands are:

```bash
git clone <repository>
cd LLM-RIO
python3 -m pip install --user uv==0.12.1
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12
uv sync --extra engine
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
An override is not revalidated; use `--make-default` to make it the only active profile and
`--restart-workers` to drain current workers. Key and model access commands accept human-readable
nicknames (or a complete API key for key selection), so internal database IDs are not required.

### Shared-weight model profiles and request defaults

A cloned model profile receives its own catalog model ID and copies the source model's active
placement profiles, but references the same downloaded artifact directory and hashes. Consequently,
it is independently routable and may be resident alongside the source model without downloading or
copying the weights. Access grants are inherited by default; pass `--no-inherit-grants` to start
without them.

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

The same operation is available through the TUI's Models page with **Clone profile**. Cloned
placement profiles are administrator overrides and are not benchmark-revalidated; their context
size must fit the selected GPU placement at worker startup.

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

The repository is managed with `uv`. On a non-GPU development machine, edit and inspect without
starting the service or running the hardware acceptance suite. On the target host:

```bash
uv sync --extra dev --extra engine
uv run pytest
```

No scheduler branch or acceptance test assumes a two-GPU host; simulated planner tests cover
1, 2, 4, and 8 GPU inventories, while real capacity and performance are always established by
machine-specific validation profiles.
