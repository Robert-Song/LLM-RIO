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
cp config.required.toml config.toml
mkdir -p state models logs diagnostics
chmod +x llmctl
./llmctl serve
```


`config.required.toml` is the minimal starting file. `config.example.toml` is the complete,
sectioned reference for every customizable setting.
Load site-specific Python or CUDA modules before these commands when required by the HPC. The first
startup prints the initial admin key before accepting requests; that admin can create any number of
additional admin keys.


## API surface

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/me/usage`
- `POST /staff/models`, `GET /staff/model-jobs/{job_id}`, and
  `POST /staff/model-jobs/{job_id}/retry`
- staff model-disable and grant-management routes
- admin key, quota, and atomic maintenance routes

The admin CLI is a thin HTTP client for the same management routes. Only `serve` and `doctor` are
local lifecycle/diagnostic commands.

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
