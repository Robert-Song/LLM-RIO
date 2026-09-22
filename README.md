# LLM-RIO

LLM-RIO is a machine-local, multi-tenant inference router. It runs one serving mode
per service, schedules measured GPU placements, and exposes OpenAI-compatible chat
and model-list endpoints. Model compatibility belongs to the selected engine:
LLM-RIO has no model or GPU allowlist. Registration validates the actual artifact,
engine, launch settings, and placement before the router serves it.

| Mode | Intended use | Engines | Release status |
| --- | --- | --- | --- |
| `queue` | Cold starts, large individual models, minimum resident overhead | vLLM; optional llama.cpp | Production candidate; qualification required |
| `vllm-sleep` | Frequent switching with enough host RAM and residual VRAM for cached workers | vLLM native level-1 sleep | Production candidate; qualification required |
| `kv-cached` | Experimental elastic KV placement | vLLM with pinned optional compatibility layer | Experimental |

Modes are selected explicitly and require a restart to change. There is no automatic
engine fallback. Local directories and GGUF files are reused in place.

## Install and start

Python 3.11–3.13 is supported by the router. GPU engine/platform constraints are
separate. Use the committed lock file:

```sh
uv sync --locked --extra engine
cp examples/config/queue.toml config.release.toml
export LLMRIO_CONFIG_FILE="$PWD/config.release.toml"
uv run --no-sync llm-rio doctor
uv run --no-sync llm-rio serve
```

Native installation does not install kv-cached. For a separate externally managed
vLLM or llama.cpp executable, install the router with `uv sync --locked` and set its
path in `[engines]`. Keep the engine environment with your deployment manifest.
Experimental installation has [separate instructions](docs/EXPERIMENTAL.md).

Use a **new** database at `state/release/llm-rio.db`. Beta schemas are rejected;
there are no serving-path migrations. Preserve your old installation with the
[backup and rollback procedure](docs/OPERATIONS.md) before creating release state.
The initial administrator credential is generated on first startup and stored in
the protected local credential vault. Local administration can recover it;
remote administration requires `LLMRIO_API_URL` and `LLMRIO_API_KEY`.

In another terminal with the same configuration selector:

```sh
uv run --no-sync llm-rio maintenance drain
uv run --no-sync llm-rio models add example organization/model --revision COMMIT_SHA
# Or reuse a directory on the server:
uv run --no-sync llm-rio models add local-example --local-path /absolute/model/directory
uv run --no-sync llm-rio models review example
uv run --no-sync llm-rio maintenance resume
```

Use your own repository and revision in place of the placeholders. Native validation
waits until maintenance has drained admitted work. Resume after validation completes.
`models validate NAME` always probes; it also retries failed registration jobs.
Run `uv run --no-sync llm-rio` for the TUI, or append `--help` for CLI help.

## Documentation and verification

The [documentation index](docs/README.md) covers architecture, mode selection,
configuration, actions, inference, operations, and release acceptance.
[Release progress and evidence](docs/release/CLEANUP.md) distinguish automated
checks from unfinished hardware qualification. Passing unit tests is not a release
qualification claim.

```sh
uv sync --locked --extra dev
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests scripts/check_docs.py scripts/qualify.py
uv run --no-sync ruff format --check src tests scripts/check_docs.py scripts/qualify.py
uv run --no-sync mypy src/llm_rio
uv run --no-sync python scripts/check_docs.py
```
