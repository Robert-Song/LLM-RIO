# Experimental kv-cached boundary

`kv-cached` is experimental and does not define native release eligibility. Native queue
and sleep installation/startup do not install, import, or activate its compatibility patches.
Dependency detection, bootstrap `sitecustomize`, packed-KV adaptations and elastic placement
live under `src/llm_rio/modes/kv_cached`.

| Component | Pinned candidate | Qualification status |
| --- | --- | --- |
| Native vLLM extra | `0.26.0` | Per-installation native validation required |
| Experimental kv-cached | `60cad949389af6bbf1d65c4eddf325113df5a9eb` | Optional; compatibility tests and separate hardware report required |
| Local compatibility patches | Repository revision | Covered by isolated optional tests; historical GPU results are not current release evidence |
| Other engine/model/GPU combinations | Operator-selected | No router allowlist; validate the actual combination; experimental rejection does not block native release |

Use a separate environment. kv-cached's source build imports Torch; install the engine
first, then build the optional dependency with its declared isolation exception:

```sh
uv sync --locked --extra engine
uv sync --locked --extra engine --extra kvcached
export LLMRIO_CONFIG_FILE="$PWD/examples/config/kv-cached.toml"
uv run --no-sync llm-rio doctor
```

This command sequence belongs in an experimental checkout/environment, not a native
production environment. Do not mix both services on managed GPUs. Missing/incompatible
experimental dependencies must fail explicit `kv-cached` selection; there is no fallback.
The isolated compatibility harness is available as
`python -m llm_rio.modes.kv_cached.compatibility --help`.

[Historical research](historical/README.md) records earlier tuples and local observations.
They are not promises of support for a model family or a current qualification result.
