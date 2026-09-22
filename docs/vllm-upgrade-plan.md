# vLLM Upgrade and Canary Plan

## Purpose

Upgrade vLLM without disrupting the currently verified production models. The
upgrade canary must prove that the four currently excluded profiles are safe to
serve before any production promotion:

- Qwen 3.5 35B-A3B FP8
- Qwen 3.6 35B-A3B FP8
- Qwen 3.8 27B INT4
- DeepSeek-V4-Flash

This is a release procedure, not an assumption that a version bump fixes every
model issue.

## 1. Keep production stable

Until a canary passes, leave the four profiles above disabled/unavailable.

- Continue serving only models with a completed validation profile.
- Keep the current production vLLM environment intact.
- Record the exact current vLLM, CUDA, driver, PyTorch, and LLM-RIO versions.
- Do not modify the production profile merely because a canary profile passes.

## 2. Enter drain mode deliberately

Only begin this step after an explicit maintenance/idle decision.

1. Stop accepting new inference requests.
2. Let existing requests and streaming responses finish.
3. Confirm no workers are generating and no requests remain queued.
4. Stop production workers cleanly.
5. Preserve the production environment and model-profile database for rollback.

Do not infer safety from a momentary lack of requests; a client may hold a
long-running stream or reconnect.

## 3. Create an isolated canary

Use an isolated runtime so it cannot collide with production:

- A new API port (for example, `8004`; do not reuse production `8003`).
- A dedicated virtual environment, pinned to the exact candidate vLLM version.
- Separate state/database, logs, model-worker ports, and cache directories.
- The same GPU driver and model files as production.

Before installing, record the candidate version and its dependency lock or
installation command. Do not use an unpinned `latest` build for promotion.

> DeepSeek-V4 support predates vLLM 0.27.0, but the selected candidate version
> still needs validation against the exact DeepSeek checkpoint and hardware.

## 4. Canary validation matrix

Register the four excluded profiles in the canary only. For every profile,
retain its exact successful launch parameters as the candidate production
profile.

| Test | Required result |
| --- | --- |
| Engine startup | Model loads with the planned tensor parallelism and context window. |
| Normal profile | Health check and a real client generation succeed. |
| KV-cached profile | Cache-enabled startup and a real client generation succeed. |
| Sleep and wake | Sleep, wake, then a post-wake generation all succeed. |
| Reload/eviction | Model can be unloaded and loaded again without orphan workers or port conflicts. |
| Concurrency | Use the intended production concurrency; confirm no startup or generation OOM. |
| Regression set | Smoke-test every currently verified production model in the new environment. |

For the Qwen 35B FP8 models, test the production target of TP=2 and 256k
context. Do not count a lower-context or eager-only result as a pass unless
that exact restriction is acceptable in production.

For DeepSeek-V4-Flash, do not count a startup-only result as a pass: run normal
and KV-cached generation, plus sleep/wake if that is enabled for the profile.

## 5. Acceptance gate

Promote only when all of the following are true:

- Every required test above passed on the canary.
- The client smoke test passed against the canary API.
- No validation log contains an engine crash, OOM, wake failure, or
  authentication/port collision.
- The selected configuration is reproducible after a clean worker restart.
- The candidate vLLM version is pinned.

If any test fails, keep production unchanged, retain the redacted canary logs,
and mark the affected profile unavailable. A failed canary is not a reason to
weaken production validation.

## 6. Promotion and rollback

1. Keep production drained.
2. Apply the same pinned environment and validated profile settings to
   production.
3. Start production on its normal port and verify health plus the client smoke
   test.
4. Re-enable only the profiles that passed the full canary matrix.
5. Monitor initial requests for startup, cache, and wake failures.

Rollback immediately by restoring the preserved production environment and
profile database, then keep newly failing profiles disabled. Do not attempt to
roll back by changing live worker arguments in place.

## Operational notes

- Protect logs: validation workers may emit temporary API credentials in their
  command/configuration output. Do not commit or broadly share raw logs; rotate
  any credential that was exposed.
- Keep the canary API and worker ports distinct from production to prevent a
  health check from reaching the wrong temporary-authenticated worker.
- Record the final model profile overrides, including context length, GPU memory
  utilization, tensor parallelism, and concurrency limits.
