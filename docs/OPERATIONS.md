# LLM-RIO Operations

This guide covers the manual, command-by-command startup sequence for a Linux/HPC
GPU host. `setup.sh` performs the same steps automatically; running it is not
required. Review this page once before the first production start.

## Prerequisites

- A Linux host with an NVIDIA driver installed and `nvidia-smi` on `PATH`
- Python 3.12 (or a version allowed by `pyproject.toml`)
- `uv` 0.12.x (`python3 -m pip install --user uv==0.12.1`)
- Access to Hugging Face for model snapshots (a token via `LLMRIO_HF_TOKEN` in
  `.env` when the catalog contains gated repositories)
- The vLLM engine extra: `uv sync --extra engine`
- Load site-specific Python or CUDA modules **before** these commands when the
  HPC requires them

## Manual start

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

- `config.required.toml` is the minimal starting file. `config.example.toml` is
  the complete, sectioned reference for every customizable setting.
- `config.toml` is git-ignored; keep host-specific values out of version control.
- The first startup creates and visibly prints the initial administrator API key
  before accepting requests. Store it somewhere safe.

## Verifying the installation

```bash
./llmctl doctor --json
```

`doctor` checks that `nvidia-smi`, vLLM, and the configured directories exist
and are writable, then reports the discovered GPU inventory (driver, CUDA
driver, per-GPU UUID, VRAM, compute capability, and PCI bus ID). It exits
non-zero when a prerequisite is missing.

## Managing keys

Local `llmctl` management commands automatically recover an active admin
credential from the protected host database/vault, so they do **not** require
`LLMRIO_API_KEY`. Remote management still requires an explicitly supplied admin
key (`export LLMRIO_API_KEY=...`).

```bash
./llmctl keys list                 # every key, including recoverable values
./llmctl keys create alice --role user --limit 1000000
./llmctl keys create bob --role ta --unlimited
./llmctl keys revoke alice         # deactivate
./llmctl keys delete alice         # remove credential utility, keep audit trail
./llmctl keys limit alice --limit 2000000
./llmctl keys reset-usage alice    # reset current-period usage, keep lifetime totals
```

## Registering a model

```bash
./llmctl models add my-model org/repo --grant-to alice
./llmctl models review my-model    # follow the registration job
./llmctl models retry my-model     # after fixing a reported failure
./llmctl models disable my-model   # if it will not be fixed
./llmctl models grant alice my-model
```

Registration downloads the pinned snapshot, inspects `config.json` and the
weight artifacts, then runs preemptible idle-only engine probes on candidate
GPU sets. Only placements that pass the streaming contract are stored as
profiles; the smallest viable placement is mandatory.

## Model access

```bash
./llmctl models grant alice my-model     # add nicknames
./llmctl models revoke alice my-model    # remove nicknames
./llmctl models access alice             # list current grants
```

## Maintenance

Draining stops admission, lets admitted requests finish, and stops workers once
they are idle. `resume` returns the machine to `ACTIVE` only after every worker
is cold.

```bash
./llmctl maintenance drain
./llmctl maintenance status
./llmctl maintenance resume
```

## Health and monitoring

```bash
curl http://127.0.0.1:8000/health
```

Returns `{"status": "ok", "mode": "...", "machine_id": "...", "managed_gpus": N}`.
The access log emits one line per request with the model, key nickname, request
ID, queue wait, and elapsed time. Engine stdout/stderr is captured under
`logs/worker-<id>.log` while `capture_worker_engine_logs` is enabled.

## Troubleshooting

| Symptom | Action |
|---|---|
| `doctor` fails at inventory | Confirm the NVIDIA driver and `nvidia-smi`; load HPC modules |
| Initial admin key was lost | `./llmctl keys list` recovers it from the local vault |
| Registration fails at `engine_startup` | Read `logs/validation-*.log`, fix engine/CUDA/model compatibility, retry |
| Registration fails at `disk_capacity` | Free model-store space, then `./llmctl models retry <job>` |
| Workers never reach READY | Check `logs/worker-<id>.log`; confirm GPU UUIDs and VRAM headroom |
| Service reports `service_maintenance` | `./llmctl maintenance status`, then `resume` once workers are cold |

For anything not covered here, open an issue with the output of
`./llmctl doctor --json` and the relevant `logs/validation-*.log` file.
